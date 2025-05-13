from icu_benchmarks.models.wrappers import ImputationWrapper
import gin
import math
import torch
from torch import nn
import torch.nn.functional as F


@gin.configurable("Simple_Diffusion")
class SimpleDiffusionModel(ImputationWrapper):
    """Imputation model based on a Simple Diffusion Model.
    Adapted from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL."""

    requires_backprop = True

    input_size = []

    def __init__(self, *args, input_size, **kwargs):
        super().__init__(*args, input_size=input_size, **kwargs)

        factor = 2
        num_layers = 4
        down_channels = [32 * factor**i for i in range(num_layers)]  # Start with 32 channels
        up_channels = down_channels[::-1]
        time_emb_dim = 32

        self.input_size = input_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv1d(input_size[2], down_channels[0], 2)

        # Downsample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )

        # Upsample
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )

        # Final Output
        self.output = nn.ConvTranspose1d(up_channels[-1], input_size[2], 2)


    def forward(self, amputated, timestep):
        amputated = torch.nan_to_num(amputated, nan=0.0)
        amputated = amputated.permute(0, 2, 1)  # Now shape is (batch_size, channels, length)
        # model_input = torch.cat((amputated, amputation_mask), dim=1)

        # output = self.model(model_input)
        # output = output.reshape(amputated.shape)

        # Embedd time
        t = self.time_mlp(timestep)

        # Initial Convolution
        x = self.conv0(amputated)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        # Output Layer
        output = self.output(x)
        output = output.permute(0, 2, 1)  # Convert back to [batch_size, seq_len, num_features]
        return output

    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals[t].to(t.device)  # Direct indexing instead of gather
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # mean + variance
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    # Define beta schedule
    T = 300
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        super().on_fit_start()

    def training_step(self, batch):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)
        batch_size = amputated.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        loss = self.get_loss(target, t)

        self.log("train/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["train"].values():
            metric.update((torch.flatten(target, start_dim=1), torch.flatten(target, start_dim=1)))

        return loss

    def validation_step(self, batch, batch_index):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)
        batch_size = amputated.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

        betas_t = self.get_index_from_list(self.betas, t, amputated.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, amputated.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, amputated.shape)

        model_output = self(amputated, t)
        model_mean = sqrt_recip_alphas_t * (amputated - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, amputated.shape)

        # Compute imputated without an if statement
        noise = torch.randn_like(amputated)
        sqrt_posterior_variance_t = torch.sqrt(posterior_variance_t)
        imputated = model_mean + sqrt_posterior_variance_t * noise

        # When t == 0, sqrt_posterior_variance_t is zero, so imputated == model_mean

        # Update the amputated tensor with imputated values where needed
        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log("val/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["val"].values():
            metric.update((torch.flatten(amputated, start_dim=1), torch.flatten(target, start_dim=1)))


    def test_step(self, batch, batch_index):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)
        batch_size = amputated.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

        betas_t = self.get_index_from_list(self.betas, t, amputated.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, amputated.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, amputated.shape)

        model_output = self(amputated, t)
        model_mean = sqrt_recip_alphas_t * (amputated - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, amputated.shape)

        # Compute imputated without an if statement
        noise = torch.randn_like(amputated)
        sqrt_posterior_variance_t = torch.sqrt(posterior_variance_t)
        imputated = model_mean + sqrt_posterior_variance_t * noise

        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log("test/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["test"].values():
            metric.update((torch.flatten(amputated, start_dim=1), torch.flatten(target, start_dim=1)))



class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            in_ch *= 2
            self.conv1 = nn.ConvTranspose1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 2, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # Transformer Encoder for Feature Self-Attention
        self.feature_layer = nn.TransformerEncoderLayer(
            d_model=in_ch, nhead=1, dim_feedforward=64, activation="gelu", batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(self.feature_layer, num_layers=1)

        # Transformer Encoder for Time Self-Attention
        self.time_layer = nn.TransformerEncoderLayer(
            d_model=out_ch, nhead=1, dim_feedforward=64, activation="gelu", batch_first=True
        )
        self.time_transformer = nn.TransformerEncoder(self.time_layer, num_layers=1)

    def forward(self, x, t):
        # Apply Feature Self-Attention
        h = self.feature_transformer(x.permute(0, 2, 1))
        h = h.permute(0, 2, 1)
        # First Convolution
        h = self.bnorm1(self.relu(self.conv1(h)))
        # Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1)
        # Add time
        h = h + time_emb
        # Apply Time Self-Attention
        h = self.time_transformer(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1)
        # Second Convolution
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 0.05)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

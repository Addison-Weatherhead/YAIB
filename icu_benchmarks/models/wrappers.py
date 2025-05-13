import logging
from abc import ABC
from typing import Dict, Any, List, Optional, Union
import os
import json
import torchmetrics
from sklearn.metrics import log_loss, mean_squared_error

import torch
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.optim import Optimizer, Adam
import matplotlib.pyplot as plt
import inspect
import gin
import numpy as np
from ignite.exceptions import NotComputableError
from icu_benchmarks.models.constants import ImputationInit
from icu_benchmarks.models.utils import create_optimizer, create_scheduler
from joblib import dump
from pytorch_lightning import LightningModule
import pandas as pd

from icu_benchmarks.models.constants import MLMetrics, DLMetrics
from icu_benchmarks.contants import RunMode
from icu_benchmarks.global_config import GlobalConfig

gin.config.external_configurable(nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.mse_loss, module="torch.nn.functional")

gin.config.external_configurable(mean_squared_error, module="sklearn.metrics")
gin.config.external_configurable(log_loss, module="sklearn.metrics")

def plot_imputation_uncertainty_by_block(imputed_blocks_stds, save_path):
    """
    Plots the uncertainty of imputations over time for each block size in subplots within a single PNG file.

    Args:
        imputed_blocks_stds: List of tensors, each representing the standard deviations
                             of imputations for a single imputed block.
        save_path: Path to save the output plot.
    """
    #block_sizes = [2, 4, 8, 16, 32]
    block_sizes = [2, 3, 4, 5, 6]
    # Create subplots: one for each block size
    n_plots = len(block_sizes)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 5 * n_plots), sharex=False, sharey=False)

    if n_plots == 1:  # If there's only one block size, axes will not be a list
        axes = [axes]

    for ax, block_size in zip(axes, block_sizes):
        # Filter blocks for the current block size
        filtered_blocks = [tensor for tensor in imputed_blocks_stds if tensor.shape[0] == block_size]
        if len(filtered_blocks) == 0:
            continue
        # Stack the filtered blocks into a tensor
        stacked_blocks = torch.stack(filtered_blocks).cpu().detach()
        
        # Compute mean and standard deviation along the batch (0th) dimension
        block_std_means = stacked_blocks.mean(dim=0).numpy()
        block_std_stds = stacked_blocks.std(dim=0).numpy()

        # Time steps for the block
        time_steps = list(range(block_size))

        # Plot for this block size
        ax.plot(
            time_steps,
            block_std_means,
            marker='o',
            label=f'Block size {block_size}'
        )
        ax.fill_between(
            time_steps,
            block_std_means - block_std_stds,
            block_std_means + block_std_stds,
            alpha=0.2,
            label='±1 std dev'
        )

        # Set titles and labels for each subplot
        ax.set_title(f"Uncertainty Over Time (Block Size = {block_size}, n={len(filtered_blocks)})")
        ax.set_xlabel("Time Steps from Start of Imputed Block")
        ax.set_ylabel("Average Imputation Standard Deviation")
        ax.legend()
        ax.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def extract_imputed_blocks(mask, imputed_stds):
    # Result list to store the imputed blocks
    imputed_blocks = []
    # Iterate over each sample in the batch
    for i in range(mask.shape[0]):
        mask_row = mask[i]
        data_row = imputed_stds[i]
        start_idx = None
        for t in range(mask.shape[1]):
            if mask_row[t] == 1:
                if start_idx is None:
                    start_idx = t
            else:
                if start_idx is not None:
                    block = data_row[start_idx:t]
                    imputed_blocks.append(block)
                    start_idx = None
        
        # Handle the case where the block reaches the end of the time series
        if start_idx is not None:
            block = data_row[start_idx:]
            imputed_blocks.append(block)

    print('Distribution of detected block sizes:')
    size_dist = {}
    for block in imputed_blocks:
        if block.shape[0] not in size_dist:
            size_dist[block.shape[0]] = 1
        else:
            size_dist[block.shape[0]] += 1
    for k in sorted(size_dist.keys()):
        print('%d: %d'%(k, size_dist[k]))
    block_sizes = [block.shape[0] for block in imputed_blocks]
    return imputed_blocks

@gin.configurable("BaseModule")
class BaseModule(LightningModule):
    # DL type models, requires backpropagation
    requires_backprop = False
    # Loss function weight initialization type
    weight = None
    # Metrics to be logged
    metrics = {}
    trained_columns = None
    # Type of run mode
    run_mode = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def step_fn(self, batch, step_prefix=""):
        raise NotImplementedError()

    def finalize_step(self, step_prefix=""):
        pass

    def set_metrics(self, *args, **kwargs):
        self.metrics = {}

    def set_trained_columns(self, columns: List[str]):
        self.trained_columns = columns

    def set_weight(self, weight, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        return self.step_fn(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step_fn(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step_fn(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.finalize_step("train")

    def on_validation_epoch_end(self) -> None:
        self.finalize_step("val")

    def on_test_epoch_end(self) -> None:
        self.finalize_step("test")
        if GlobalConfig.missingness_type == 'blockBO' and isinstance(self, UncertaintyImputationWrapper):
            plot_imputation_uncertainty_by_block(self.imputed_blocks_stds, 'results_tables/block_imputation_uncertainty/%s_%s.png'%(GlobalConfig.dataset_name, GlobalConfig.model_name))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["class"] = self.__class__
        checkpoint["trained_columns"] = self.trained_columns
        return super().on_save_checkpoint(checkpoint)

    def save_model(self, save_path, file_name, file_extension):
        raise NotImplementedError()

    def check_supported_runmode(self, runmode: RunMode):
        if runmode not in self._supported_run_modes:
            raise ValueError(f"Runmode {runmode} not supported for {self.__class__.__name__}")
        return True

@gin.configurable("DLWrapper")
class DLWrapper(BaseModule, ABC):
    requires_backprop = True
    _metrics_warning_printed = set()
    _supported_run_modes = [RunMode.classification, RunMode.regression, RunMode.imputation]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        """General interface for Deep Learning (DL) models."""
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.input_shape = input_shape
        self.lr = lr
        print('LR chosen for optimizer: ', self.lr)
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_steps = lr_steps
        self.epochs = epochs
        self.input_size = input_size
        self.initialization_method = initialization_method
        self.scaler = None

    def on_fit_start(self):
        print('DLWrapper on_fit_start called')
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_fit_start()

    def on_train_start(self):
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_train_start()

    def finalize_step(self, step_prefix=""):
        try:
            self.log_dict(
                {
                    f"{step_prefix}/{name}": (
                        np.float32(metric.compute()) if isinstance(metric.compute(), np.float64) else metric.compute()
                    )
                    for name, metric in self.metrics[step_prefix].items()
                    if "_Curve" not in name
                },
                sync_dist=True,
            )
            for metric in self.metrics[step_prefix].values():
                metric.reset()
        except (NotComputableError, ValueError):
            if step_prefix not in self._metrics_warning_printed:
                self._metrics_warning_printed.add(step_prefix)
                logging.warning(f"Metrics for {step_prefix} not computable")
            pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""

        if isinstance(self.optimizer, str):
            optimizer = create_optimizer(self.optimizer, self.lr, self.hparams.momentum)
        elif isinstance(self.optimizer, Optimizer):
            # Already set
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self.parameters())

        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == "":
            return optimizer
        scheduler = create_scheduler(
            self.hparams.lr_scheduler, optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs
        )
        optimizers = {"optimizer": optimizer, "lr_scheduler": scheduler}
        logging.info(f"Using: {optimizers}")
        return optimizers
    
    def on_validation_epoch_start(self) -> None:
        self.metrics = {
            step_name: {metric_name: metric() for metric_name, metric in self.set_metrics().items()}
            for step_name in ["train", "val", "test"]
        }
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.metrics = {
            step_name: {metric_name: metric() for metric_name, metric in self.set_metrics().items()}
            for step_name in ["train", "val", "test"]
        }
        return super().on_test_epoch_start()

    def save_model(self, save_path, file_name, file_extension=".ckpt"):
        path = save_path / (file_name + file_extension)
        try:
            torch.save(self, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")


@gin.configurable("DLPredictionWrapper")
class DLPredictionWrapper(DLWrapper):
    """Interface for Deep Learning models."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            input_shape=input_shape,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.output_transform = None
        self.loss_weights = None

    def set_weight(self, weight, dataset):
        """Set the weight for the loss function."""

        if isinstance(weight, list):
            weight = FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = FloatTensor(dataset.get_balance()).to(self.device)
        self.loss_weights = weight

    def set_metrics(self, *args):
        """Set the evaluation metrics for the prediction model."""

        def softmax_binary_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred[:, -1], y

        def softmax_multi_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred, y

        # Output transform is not applied for contrib metrics, so we do our own.
        if self.run_mode == RunMode.classification:
            # Binary classification
            if self.logit.out_features == 2:
                self.output_transform = softmax_binary_output_transform
                metrics = DLMetrics.BINARY_CLASSIFICATION
            else:
                # Multiclass classification
                self.output_transform = softmax_multi_output_transform
                metrics = DLMetrics.MULTICLASS_CLASSIFICATION
        # Regression
        elif self.run_mode == RunMode.regression:
            self.output_transform = lambda x: x
            metrics = DLMetrics.REGRESSION
        else:
            raise ValueError(f"Run mode {self.run_mode} not supported.")
        for key, value in metrics.items():
            # Torchmetrics metrics are not moved to the device by default
            if isinstance(value, torchmetrics.Metric):
                value.to(self.device)
        return metrics

    def step_fn(self, element, step_prefix=""):
        """Perform a step in the DL prediction model training loop.

        Args:
            element (object):
            step_prefix (str): Step type, by default: test, train, val.
        """

        if len(element) == 2:
            data, labels = element[0], element[1].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
            mask = torch.ones_like(labels).bool()

        elif len(element) == 3:
            data, labels, mask = element[0], element[1].to(self.device), element[2].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
        else:
            raise Exception("Loader should return either (data, label) or (data, label, mask)")
        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target
        prediction = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1]).to(self.device)
        target = torch.masked_select(labels, mask).to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device)) + aux_loss
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(f"Run mode {self.run_mode} not yet supported. Please implement it.")
        transformed_output = self.output_transform((prediction, target))

        for key, value in self.metrics[step_prefix].items():
            if isinstance(value, torchmetrics.Metric):
                if key == "Binary_Fairness":
                    feature_names = key.feature_helper(self.trainer)
                    value.update(transformed_output[0], transformed_output[1], data, feature_names)
                else:
                    value.update(transformed_output[0], transformed_output[1])
            else:
                value.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@gin.configurable("MLWrapper")
class MLWrapper(BaseModule, ABC):
    """Interface for prediction with traditional Scikit-learn-like Machine Learning models."""

    requires_backprop = False
    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, *args, run_mode=RunMode.classification, loss=log_loss, patience=10, mps=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = None
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.loss = loss
        self.patience = patience
        self.mps = mps
        self.log_dir = None

    def set_metrics(self, labels):
        if self.run_mode == RunMode.classification:
            # Binary classification
            if len(np.unique(labels)) == 2:
                print('Set metrics set for Binary Classification')
                # if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x[:, 1]
                self.label_transform = lambda x: x

                self.metrics = MLMetrics.BINARY_CLASSIFICATION
            # Multiclass classification
            else:
                # Todo: verify multiclass classification
                self.output_transform = lambda x: np.argmax(x, axis=-1)
                self.label_transform = lambda x: x
                self.metrics = MLMetrics.MULTICLASS_CLASSIFICATION

        # Regression
        else:
            if self.scaler is not None:  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
                self.label_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
            else:
                self.output_transform = lambda x: x
                self.label_transform = lambda x: x
            self.metrics = MLMetrics.REGRESSION

    def fit(self, train_dataset, val_dataset):
        """Fit the model to the training data."""
        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()

        self.set_metrics(train_label)

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=self.weight)

        val_loss = self.fit_model(train_rep, train_label, val_rep, val_label)

        train_pred = self.predict(train_rep)

        logging.debug(f"Model:{self.model}")
        self.log("train/loss", self.loss(train_label, train_pred), sync_dist=True)
        logging.debug(f"Train loss: {self.loss(train_label, train_pred)}")
        self.log("val/loss", val_loss, sync_dist=True)
        logging.debug(f"Val loss: {val_loss}")
        self.log_metrics(train_label, train_pred, "train")

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fit the model to the training data (default SKlearn syntax)"""
        self.model.fit(train_data, train_labels)
        val_loss = 0.0
        return val_loss

    def validation_step(self, val_dataset, _):
        val_rep, val_label = val_dataset.get_data_and_labels()
        val_rep, val_label = torch.from_numpy(val_rep).to(self.device), torch.from_numpy(val_label).to(self.device)
        self.set_metrics(val_label)

        val_pred = self.predict(val_rep)

        self.log("val/loss", self.loss(val_label, val_pred), sync_dist=True)
        logging.info(f"Val loss: {self.loss(val_label, val_pred)}")
        self.log_metrics(val_label, val_pred, "val")

    def test_step(self, dataset, _):
        print('Test step run for MLWrapper')
        test_rep, test_label = dataset
        test_rep, test_label = test_rep.squeeze(), test_label.squeeze()
        self.set_metrics(test_label)
        test_pred = self.predict(test_rep)

        if self.mps:
            self.log("test/loss", np.float32(self.loss(test_label, test_pred)), sync_dist=True)
            self.log_metrics(np.float32(test_label), np.float32(test_pred), "test")
        else:
            self.log("test/loss", self.loss(test_label, test_pred), sync_dist=True)
            self.log_metrics(test_label, test_pred, "test")
        logging.debug(f"Test loss: {self.loss(test_label, test_pred)}")
        return self.loss(test_label, test_pred)
    def predict(self, features):
        if self.run_mode == RunMode.regression:
            return self.model.predict(features)
        else:  # Classification: return probabilities
            return self.model.predict_proba(features)

    def log_metrics(self, label, pred, metric_type):
        """Log metrics to the PL logs."""

        log_dict = {
                # MPS dependent type casting
                f"{metric_type}/{name}": metric(self.label_transform(label), self.output_transform(pred))
                if not self.mps
                else metric(self.label_transform(label), self.output_transform(pred))
                # Fore very metric
                for name, metric in self.metrics.items()
                # Filter out metrics that return a tuple (e.g. precision_recall_curve)
                if not isinstance(metric(self.label_transform(label), self.output_transform(pred)), tuple)
            }
        print('log_dict: ', log_dict)
        self.log_dict(
            log_dict,
            sync_dist=True,
        )

        with (self.log_dir / f"{metric_type}_metrics.json").open("w") as f:
            json.dump({k: v for k,v in log_dict.items() if metric_type in k}, f, indent=4)

    def configure_optimizers(self):
        return None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["label_transform"]
        del state["output_transform"]
        return state

    def save_model(self, save_path, file_name, file_extension=".joblib"):
        path = save_path / (file_name + file_extension)
        try:
            dump(self.model, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")

    def set_model_args(self, model, *args, **kwargs):
        """Set hyperparameters of the model if they are supported by the model."""
        signature = inspect.signature(model.__init__).parameters
        possible_hps = list(signature.keys())
        # Get passed keyword arguments
        arguments = locals()["kwargs"]
        # Get valid hyperparameters
        hyperparams = {key: value for key, value in arguments.items() if key in possible_hps}
        logging.debug(f"Creating model with: {hyperparams}.")
        return model(**hyperparams)

@gin.configurable("ImputationWrapper")
class ImputationWrapper(DLWrapper):
    """Interface for imputation models."""

    requires_backprop = True
    _supported_run_modes = [RunMode.imputation]

    def __init__(
        self,
        loss: nn.modules.loss._Loss = MSELoss(),
        optimizer: Union[str, Optimizer] = "adam",
        run_mode: RunMode = RunMode.imputation,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        input_size: Tensor = None,
        initialization_method: ImputationInit = ImputationInit.NORMAL,
        epochs=100,
        **kwargs: str,
    ) -> None:
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer

    def set_metrics(self):
        return DLMetrics.IMPUTATION

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == ImputationInit.NORMAL:
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == ImputationInit.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == ImputationInit.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == ImputationInit.ORTHOGONAL:
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def on_fit_start(self) -> None:
        self.init_weights(self.hparams.initialization_method)
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
        return super().on_fit_start()

    def step_fn(self, batch, step_prefix=""):
        amputated, amputation_mask, target, target_missingness = batch
        imputated = self(amputated, amputation_mask)
        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)

        for metric in self.metrics[step_prefix].values():
            metric.update(
                (torch.flatten(amputated.detach(), start_dim=1).clone(), torch.flatten(target.detach(), start_dim=1).clone())
            )
        return loss

    def fit(self, train_dataset, val_dataset):
        raise NotImplementedError()

    def predict_step(self, data, amputation_mask=None):
        return self(data, amputation_mask)

    def predict(self, data):
        self.eval()
        data = data.to(self.device)
        data_missingness = torch.isnan(data).to(torch.float32)
        prediction = self.predict_step(data, data_missingness)
        data[data_missingness.bool()] = prediction[data_missingness.bool()]
        return data


@gin.configurable("UncertaintyImputationWrapper")
class UncertaintyImputationWrapper(ImputationWrapper):
    """Interface for uncertainty imputation models."""

    requires_backprop = True
    _supported_run_modes = [RunMode.imputation]

    def __init__(
        self,
        loss: nn.modules.loss._Loss = MSELoss(),
        optimizer: Union[str, Optimizer] = "adam",
        run_mode: RunMode = RunMode.imputation,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        input_size: Tensor = None,
        initialization_method: ImputationInit = ImputationInit.NORMAL,
        epochs=100,
        **kwargs: str,
    ) -> None:
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer

        
        self.num_val_uncertainty_steps = 10 # Number of times to compute uncertainty measure during validation process. This allows us to have error bars on MAE vs Uncertainty Plots.
        self.val_uncertainties = [[] for _ in range(self.num_val_uncertainty_steps)]
        self.val_abs_errors = [[] for _ in range(self.num_val_uncertainty_steps)]
        self.thresholds = []
        self._threshold = float('inf') # The threshold of uncertainty. When the model is more uncertain than this, it doesn't impute. By default, float(inf) so model imputes everything

        self.imputed_blocks_stds = []
    def get_threshold(self):
        return self._threshold
    def set_threshold(self, threshold):
        self._threshold = threshold
        print('Imputation Model threshold updated to ', self._threshold)

    def on_fit_start(self) -> None:
        print('UncertaintyImputationWrapper on_fit_start called')
        self.init_weights(self.hparams.initialization_method)
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
        return super().on_fit_start()


    def predict_with_uncertainty(self, amputated, amputation_mask, n_forward_passes=16): # Was 32
        """Perform Monte Carlo Dropout during inference to estimate uncertainty """
        print('predict with uncertainty called', flush=True)
        self.train()
        with torch.no_grad():
            preds = []
            for _ in range(n_forward_passes):
                preds.append(self.forward(amputated, amputation_mask, mc_mode=True))
            preds = torch.stack(preds, dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        assert (std>0).any()
        return mean, std

    
    def step_fn(self, batch, step_prefix=""):
        # of shape (1, bs, T, D)
        amputated, amputation_mask, target, target_missingness = batch
        print('step_fn called with step_prefix: ', step_prefix, flush=True)
        if step_prefix == "val":
            for i in range(self.num_val_uncertainty_steps):
                #original_amputated = amputated.clone()
                # We do MC dropout to gather uncertainty
                imputated_mean, imputated_std = self.predict_with_uncertainty(
                    amputated, amputation_mask
                )
                
                # amputation_mask is True if the value was originally missing or is artifically missing
                # ~target_missingness is True if the value was not originally missing
                # So artificial_missing_mask is True only for values that were artificially missing and not originally missing
                artificial_missing_mask = amputation_mask.int() & ~(target_missingness.int())
                
                # store uncertainties and errors *ONLY for the amputed positions*
                masked = (artificial_missing_mask > 0)
                print('torch.sum(masked)', torch.sum(masked), ' masked shape: ', masked.shape)
                selected_unc = imputated_std[masked]         # shape (#missing_positions,)
                selected_err = (imputated_mean - target)[masked].abs()
                print('sum(selected_unc==0): ', sum(selected_unc==0), 'selected_unc.shape: ', selected_unc.shape)
                # Save for dynamic thresholding
                self.val_uncertainties[i].append(selected_unc.detach().cpu())
                self.val_abs_errors[i].append(selected_err.detach().cpu())
            amputated[artificial_missing_mask > 0] = imputated_mean[artificial_missing_mask > 0]
        elif step_prefix == "test":
            # of shape (1, bs, T, d)
            imputated_mean, imputated_std = self.predict_with_uncertainty(amputated, amputation_mask)
            if GlobalConfig.missingness_type == 'blockBO':
                self.imputed_blocks_stds.extend(
                    extract_imputed_blocks(amputation_mask.squeeze()[:, :, 0], 
                                        imputated_std.squeeze().mean(dim=2)) # Each block is of shape (T,), we avg across dimensions
                    )

            # Update imputated values to use the mean imputations
            amputated[amputation_mask > 0] = imputated_mean[amputation_mask > 0]
            amputated[target_missingness > 0] = target[target_missingness > 0]


            # Log the uncertainty
            self.log(f"{step_prefix}/imputation_std", imputated_std.mean().item(), prog_bar=True)

        else:
            # Standard imputation (no uncertainty estimation)
            imputated = self(amputated, amputation_mask)
            amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
            amputated[target_missingness > 0] = target[target_missingness > 0]

        # Compute loss
        loss = self.loss(amputated, target)
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)

        # Update metrics
        for metric in self.metrics[step_prefix].values():
            metric.update(
                (torch.flatten(amputated.detach(), start_dim=1).clone(), torch.flatten(target.detach(), start_dim=1).clone())
            )
        
        return loss

    def predict_step(self, data, amputation_mask=None):
        print('predict_step called!', flush=True)
        return self(data, amputation_mask)

    def predict(self, data):
        print('predict called with threshold cutoff of : ', self.get_threshold(), flush=True)
        print('data shape in predict: ', data.shape, flush=True)
        print('data contains na values: ', torch.isnan(data).any())
        self.eval()
        data = data.to(self.device)
        data_missingness = torch.isnan(data).to(torch.float32)
        prediction, uncertainty = self.predict_with_uncertainty(data, data_missingness)
        mask = data_missingness.bool() & (uncertainty<self._threshold).bool()
        data[mask] = prediction[mask]
        return data
    
    def on_validation_epoch_start(self) -> None:
        print('on_validation_epoch_start called!')
        super().on_validation_epoch_start()
        # Clear out any existing lists
        self.val_uncertainties = [[] for _ in range(self.num_val_uncertainty_steps)]
        self.val_abs_errors = [[] for _ in range(self.num_val_uncertainty_steps)]

    def on_validation_epoch_end(self) -> None:
        print('on_validation_epoch_end called!')
        super().on_validation_epoch_end()

        # If we gathered uncertainties, compute dynamic thresholds
        if len(self.val_uncertainties[0]) > 0:
            quantiles = [0.1 * i for i in range(1, 11)]
            all_mae = []
            all_thresholds = []
            for ind in range(self.num_val_uncertainty_steps):
                # Sort for torch.quantile, it requires sorted values
                uncertainties, sort_inds = torch.sort(torch.cat(self.val_uncertainties[ind], dim=0))  # shape: (#all_missing_in_val, )
                abs_errors = torch.cat(self.val_abs_errors[ind], dim=0)[sort_inds]         # shape: (#all_missing_in_val, )

                thresholds = torch.quantile(uncertainties, torch.tensor(quantiles, device=uncertainties.device))
                self.thresholds = thresholds
                all_thresholds.append(thresholds)
                mae_values = []
                
                #n_samples = [] # Num data points imputed per threshold cutoff
                for i, thr in enumerate(thresholds):
                    # points with uncertainty < thr
                    mask = (uncertainties < thr)
                    print('mask.sum(): ', mask.sum())
                    if mask.sum() == 0:
                        mae = float('nan')
                    else:
                        mae = abs_errors[mask].mean().item()
                    mae_values.append(mae)
                    #n_samples.append(mask.sum())
                    self.log(f"val/mae_uncertainty<percentile_{int(quantiles[i]*100)}", mae, prog_bar=False)
                all_mae.append(np.array(mae_values))
            
            
            all_mae = np.array(all_mae)
            os.makedirs(f"results_tables/uncertainty_threshold_MAE/{GlobalConfig.dataset_name}", exist_ok=True)
            outpath = (
                f"results_tables/uncertainty_threshold_MAE/{GlobalConfig.dataset_name}/"
                f"{GlobalConfig.model_name}_{GlobalConfig.missingness_type}.png"
            )
            
            
            plt.figure(figsize=(8, 6))
            mean_mae = np.mean(all_mae, axis=0)
            std_mae = np.std(all_mae, axis=0)
            data_df = pd.DataFrame({
                "Uncertainty Quantiles": quantiles,
                "Mean MAE": mean_mae,
                "Std MAE": std_mae
            })
            data_df.to_csv(f'table_data/imputation/uncertainty_mae_{GlobalConfig.dataset_name}_{GlobalConfig.model_name}_{GlobalConfig.missingness_type}.csv')
            plt.plot(quantiles, mean_mae, '-o', label="Mean MAE", color="steelblue")
            plt.fill_between(quantiles, mean_mae - std_mae, mean_mae + std_mae, color="steelblue", alpha=0.3, label="±1 Std Dev")
            plt.title("Validation MAE vs. Uncertainty Thresholds (Dynamic Percentiles)")
            plt.xlabel("Uncertainty Quantiles")
            plt.ylabel("MAE (on imputations with uncertainty<threshold)")
            plt.grid(True)
            plt.savefig(outpath)
            plt.close()


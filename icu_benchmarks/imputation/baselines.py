"""Baseline imputation methods. These methods imported from other frameworks and are used as baselines for comparison."""
import torch
import pandas as pd
from hyperimpute.plugins.imputers import Imputers as HyperImpute
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from typing import Type, Union, Optional
import numpy as np
from types import MethodType
from icu_benchmarks.models.wrappers import ImputationWrapper, UncertaintyImputationWrapper
from pypots.imputation import BRITS, SAITS, Transformer
from pypots.imputation.saits.data import DatasetForSAITS
from pypots.imputation.transformer.data import DatasetForTransformer
from pypots.data.checking import key_in_data_set
import gin
from torch.utils.data import DataLoader
from pypots.data.dataset.base import BaseDataset


class Transformer_mc(Transformer):
    # Updated Transformer predict method to allow for mc dropout
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        eval_mode=True
    ) -> dict:
        if eval_mode:
            self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict
    
    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        eval_mode=True
    ) -> np.ndarray:

        result_dict = self.predict(test_set, file_type=file_type, eval_mode=eval_mode)
        return result_dict["imputation"]
    
    # Update Transformer fit method so return_X_ori is true, so we can use our own amputation
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForTransformer(train_set, return_X_ori=True, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForTransformer(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
    

class SAITS_mc(SAITS):
    # Updated SAITS impute method to allow for mc dropout
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        diagonal_attention_mask: bool = True,
        return_latent_vars: bool = False,
        eval_mode=True
    ) -> dict:
        
        # Step 1: wrap the input data with classes Dataset and DataLoader
        if eval_mode:
            self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []
        first_DMSA_attn_weights_collector = []
        second_DMSA_attn_weights_collector = []
        combining_weights_collector = []

        # Step 2: process the data with the model
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, diagonal_attention_mask, training=False)
                imputation_collector.append(results["imputed_data"])

                if return_latent_vars:
                    first_DMSA_attn_weights_collector.append(results["first_DMSA_attn_weights"].cpu().numpy())
                    second_DMSA_attn_weights_collector.append(results["second_DMSA_attn_weights"].cpu().numpy())
                    combining_weights_collector.append(results["combining_weights"].cpu().numpy())

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }

        if return_latent_vars:
            latent_var_collector = {
                "first_DMSA_attn_weights": np.concatenate(first_DMSA_attn_weights_collector),
                "second_DMSA_attn_weights": np.concatenate(second_DMSA_attn_weights_collector),
                "combining_weights": np.concatenate(combining_weights_collector),
            }
            result_dict["latent_vars"] = latent_var_collector

        return result_dict
    
    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        eval_mode=True,
    ) -> np.ndarray:

        result_dict = self.predict(test_set, file_type=file_type, eval_mode=eval_mode)
        return result_dict["imputation"]
    
    # Update SAITS fit method so return_X_ori is true, so we can use our own amputation
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForSAITS(train_set, return_X_ori=True, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")


@gin.configurable("KNN")
class KNNImputation(ImputationWrapper):
    """Imputation using Scikit-Learn K-Nearest Neighbour."""

    requires_backprop = False

    def __init__(self, *args, n_neighbors=2, **kwargs) -> None:
        super().__init__(*args, n_neighbors=n_neighbors, **kwargs)
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("MICE")
class MICEImputation(ImputationWrapper):
    """Imputation using Scikit-Learn MICE."""

    requires_backprop = False

    def __init__(self, *args, max_iter=100, verbose=2, imputation_order="random", random_state=0, **kwargs) -> None:
        super().__init__(
            *args, max_iter=max_iter, verbose=verbose, imputation_order=imputation_order, random_state=random_state, **kwargs
        )
        self.imputer = IterativeImputer(
            estimator=LinearRegression(),
            max_iter=max_iter,
            verbose=verbose,
            imputation_order=imputation_order,
            random_state=random_state,
        )

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("Mean")
class MeanImputation(ImputationWrapper):
    """Mean imputation using Scikit-Learn SimpleImputer."""

    requires_backprop = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="mean")

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output

        

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        print('debatched_values shape: ', debatched_values.shape)
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        output = output.reshape(amputated_values.shape)
        return output
        print('output shape: ', output.shape)
        time_steps = range(200)
        var1, var2 = debatched_values[:200, 0], debatched_values[:200, 1]
        print('Var 1 nan inds: ', torch.nonzero(torch.isnan(var1)))
        print('var1: ', var1)
        pred1, pred2 = output[0][0][:200, 0],output[0][0][:200, 1]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(time_steps, var1, marker='o')
        #plt.plot(time_steps, var2)
        pred1_onlymissing = pred1
        pred1_onlymissing[~torch.isnan(var1)] = float('nan')
        print('pred1_onlymissing: ', pred1_onlymissing)
        plt.plot(time_steps, pred1_onlymissing, '--', marker='o')
        #plt.plot(time_steps, pred2, '--')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()

        # Show the plot
        plt.savefig('./MeanPred.png')
        assert False
        return output


@gin.configurable("Forward")
class ForwardImputation(ImputationWrapper):
    """Forward imputation using last observed value."""

    requires_backprop = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, train_dataset, val_dataset):
        # Forward imputation does not require fitting, so this method can be left empty
        pass

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.squeeze(0).to("cpu").numpy()  # Shape: (bs, T, num_feat)

        # Apply forward fill for each patient separately
        imputated_values = []
        for patient_data in debatched_values:
            # Forward fill, then backfill
            filled_patient_data = (
                pd.DataFrame(patient_data)
                .fillna(method="ffill")
                .fillna(method="bfill") # After forward filling, any remaining NaNs are from the start of the time series. Back fill in this case.
                .values
            )
            imputated_values.append(filled_patient_data)

        # Convert back to a torch tensor and restore the original shape
        output = torch.Tensor(imputated_values).to(amputated_values.device)  # Shape: (bs, T, num_feat)
        
        return output



@gin.configurable("Median")
class MedianImputation(ImputationWrapper):
    """Median imputation using Scikit-Learn SimpleImputer."""

    requires_backprop = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("Zero")
class ZeroImputation(ImputationWrapper):
    """Zero imputation using Scikit-Learn SimpleImputer."""

    requires_backprop = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("MostFrequent")
class MostFrequentImputation(ImputationWrapper):
    """Most frequent imputation using Scikit-Learn SimpleImputer."""

    requires_backprop = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(train_dataset.amputated_values.values)

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)

        output = output.reshape(amputated_values.shape)
        return output


def wrap_hyperimpute_model(methodName: str, configName: str) -> Type:
    class HyperImputeImputation(ImputationWrapper):
        """Imputation using HyperImpute package."""

        requires_backprop = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.imputer = HyperImpute().get(methodName, **self.model_params())

        @gin.configurable(module=configName)
        def model_params(self, **kwargs):
            return kwargs

        def fit(self, train_dataset, val_dataset):
            self.imputer.fit(train_dataset.amputated_values.values)

        def forward(self, amputated_values, amputation_mask):
            debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
            debatched_values = debatched_values.to(float).to("cpu").numpy()
            with torch.inference_mode(mode=False):
                output = torch.Tensor(self.imputer.transform(debatched_values).values).to(amputated_values.device)
            output = output.reshape(amputated_values.shape)
            return output

    return gin.configurable(configName)(HyperImputeImputation)


GAINImputation = wrap_hyperimpute_model("gain", "GAIN")
MissForestImputation = wrap_hyperimpute_model("sklearn_missforest", "MissForest")
ICEImputation = wrap_hyperimpute_model("ice", "ICE")
SoftImputeImputation = wrap_hyperimpute_model("softimpute", "SoftImpute")
SinkhornImputation = wrap_hyperimpute_model("sinkhorn", "Sinkhorn")
MiracleImputation = wrap_hyperimpute_model("miracle", "Miracle")
MiwaeImputation = wrap_hyperimpute_model("miwae", "Miwae")
HyperImputation = wrap_hyperimpute_model("hyperimpute", "HyperImpute")


@gin.configurable("BRITS")
class BRITSImputation(ImputationWrapper):
    """Bidirectional Recurrent Imputation for Time Series (BRITS) imputation using PyPots package."""

    requires_backprop = False

    def __init__(self, *args, input_size, epochs=1, rnn_hidden_size=64, batch_size=256, **kwargs) -> None:
        super().__init__(
            *args, input_size=input_size, epochs=epochs, rnn_hidden_size=rnn_hidden_size, batch_size=batch_size, **kwargs
        )
        self.imputer = BRITS(
            n_steps=input_size[1],
            n_features=input_size[2],
            rnn_hidden_size=rnn_hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(
            torch.Tensor(
                train_dataset.amputated_values.values.reshape(-1, train_dataset.maxlen, train_dataset.features_df.shape[1])
            )
        )

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.to(self.imputer.device).squeeze()
        self.imputer.model = self.imputer.model.to(self.imputer.device)
        output = torch.Tensor(self.imputer.impute(debatched_values)).to(self.device)

        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("SAITS")
class SAITSImputation(UncertaintyImputationWrapper):
    """Self-Attention based Imputation for Time Series (SAITS) imputation using PyPots package."""

    requires_backprop = False # Handled within the library
    is_mc_dropout_capable = True # For mc dropout during inference

    def __init__(self, *args, input_size, epochs, n_layers, d_model, d_inner, n_head, d_k, d_v, dropout, **kwargs) -> None:
        super().__init__(
            *args,
            input_size=input_size,
            epochs=epochs,
            n_layers=n_layers,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            **kwargs
        )
        self.imputer = SAITS_mc(
            n_steps=input_size[1],
            n_features=input_size[2],
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_inner,
            n_heads=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            epochs=epochs
        )

        print('SAITSImputation model initialized with %d steps, %d features'%(input_size[1], input_size[2]))

    def fit(self, train_dataset, val_dataset):
        print('train_dataset.amputated_values.values[0]: ', train_dataset.amputated_values.values[0])
        X_ori_train = []
        for amputated_window, amputation_mask, window, window_missingness_mask in train_dataset:
            # Each are of shape (seq_len, num_vars)
            # Amputation mask is 1 where either amputation occurred or genuine missingness occured
            # window_missingness_mask is 1 where genuine missingness is, 0 otherwise
            # window is the original data. Missing values are 0 filled.
            window[window_missingness_mask.bool()] = np.nan
            X_ori_train.append(window)
        
        # Do same for val set
        X_ori_val = []
        for amputated_window, amputation_mask, window, window_missingness_mask in val_dataset:
            window[window_missingness_mask.bool()] = np.nan
            X_ori_val.append(window)
       
        self.imputer.fit(
            # Train set
            {"X": torch.Tensor(
                train_dataset.amputated_values.values.reshape(-1, train_dataset.maxlen, train_dataset.features_df.shape[1])
            ),
            "X_ori": torch.stack(X_ori_train)},
            
            # Val set
            {"X": torch.Tensor(
                val_dataset.amputated_values.values.reshape(-1, val_dataset.maxlen, val_dataset.features_df.shape[1])
            ),
            "X_ori": torch.stack(X_ori_val)}
        )
    '''
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.to(self.imputer.device).squeeze()
        self.imputer.model = self.imputer.model.to(self.imputer.device)
        #output = torch.Tensor(self.imputer.impute(debatched_values)).to(self.device)
        debatched_values_np = debatched_values.cpu().numpy() # I added
        output = torch.Tensor(self.imputer.impute({"X": debatched_values_np})).to(self.device) # I added
        output = output.reshape(amputated_values.shape)
        return output
    '''

    def enable_mc_dropout(self):
        """Enable dropout during inference by recursively setting all dropout layers to train mode."""
        def set_dropout_train(module):
            if isinstance(module, torch.nn.Dropout):
                module.train()
            for child in module.children():
                set_dropout_train(child)
                
        set_dropout_train(self.imputer.model)

    def forward(self, amputated_values, amputation_mask, mc_mode=False):
        debatched_values = amputated_values.to(self.imputer.device).squeeze()
        self.imputer.model = self.imputer.model.to(self.imputer.device)
        debatched_values_np = debatched_values.cpu().numpy()
        if mc_mode:
            # Enable dropout for MC sampling
            self.enable_mc_dropout()
            output = torch.Tensor(self.imputer.impute({"X": debatched_values_np}, eval_mode=False)).to(self.device)
        else:
            output = torch.Tensor(self.imputer.impute({"X": debatched_values_np}, eval_mode=True)).to(self.device)
        
        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("Attention")
class AttentionImputation(UncertaintyImputationWrapper):
    """Attention based Imputation (Transformer) imputation using PyPots package."""

    requires_backprop = False # Handled within the library
    is_mc_dropout_capable = True # For mc dropout during inference

    def __init__(self, *args, input_size, epochs, n_layers, d_model, d_inner, n_head, d_k, d_v, dropout, **kwargs) -> None:
        super().__init__(
            *args,
            input_size=input_size,
            epochs=epochs,
            n_layers=n_layers,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            **kwargs
        )
        self.imputer = Transformer_mc(
            n_steps=input_size[1],
            n_features=input_size[2],
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_inner,
            n_heads=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            epochs=epochs
        )
        print('AttentionImputation model initialized with %d steps, %d features'%(input_size[1], input_size[2]))

    def fit(self, train_dataset, val_dataset):
        print('train_dataset.amputated_values.values[0]: ', train_dataset.amputated_values.values[0])
        X_ori_train = []
        for amputated_window, amputation_mask, window, window_missingness_mask in train_dataset:
            # Each are of shape (seq_len, num_vars)
            # Amputation mask is 1 where either amputation occurred or genuine missingness occured
            # window_missingness_mask is 1 where genuine missingness is, 0 otherwise
            # window is the original data. Missing values are 0 filled.
            window[window_missingness_mask.bool()] = np.nan
            X_ori_train.append(window)
        
        # Do same for val set
        X_ori_val = []
        for amputated_window, amputation_mask, window, window_missingness_mask in val_dataset:
            window[window_missingness_mask.bool()] = np.nan
            X_ori_val.append(window)
       
        self.imputer.fit(
            # Train set
            {"X": torch.Tensor(
                train_dataset.amputated_values.values.reshape(-1, train_dataset.maxlen, train_dataset.features_df.shape[1])
            ),
            "X_ori": torch.stack(X_ori_train)},
            
            # Val set
            {"X": torch.Tensor(
                val_dataset.amputated_values.values.reshape(-1, val_dataset.maxlen, val_dataset.features_df.shape[1])
            ),
            "X_ori": torch.stack(X_ori_val)}
        )

    def enable_mc_dropout(self):
        """Enable dropout during inference by recursively setting all dropout layers to train mode."""
        def set_dropout_train(module):
            if isinstance(module, torch.nn.Dropout):
                module.train()
            for child in module.children():
                set_dropout_train(child)
                
        set_dropout_train(self.imputer.model)

    def forward(self, amputated_values, amputation_mask, mc_mode=False):
        amputated_values_torch = amputated_values.to(self.imputer.device).squeeze()
        self.imputer.model = self.imputer.model.to(self.imputer.device)
        debatched_values_np = amputated_values_torch.cpu().numpy()
        if mc_mode:
            # Enable dropout for MC sampling
            self.enable_mc_dropout()
            output = torch.Tensor(self.imputer.impute({"X": debatched_values_np}, eval_mode=False)).to(self.device)
        else:
            output = torch.Tensor(self.imputer.impute({"X": debatched_values_np}, eval_mode=True)).to(self.device)
        
        output = output.reshape(amputated_values.shape)
        return output

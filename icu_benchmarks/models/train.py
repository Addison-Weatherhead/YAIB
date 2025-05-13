import os
import gin
import torch
import logging
import pandas as pd
from joblib import load
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pathlib import Path
from icu_benchmarks.data.loader import PredictionDataset, ImputationDataset
from icu_benchmarks.models.utils import save_config_file, JSONMetricsLogger
from icu_benchmarks.contants import RunMode
from icu_benchmarks.data.constants import DataSplit as Split
import matplotlib.pyplot as plt
from icu_benchmarks.global_config import GlobalConfig
import re
import numpy as np

cpu_core_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

def plot_train_samples(train_dataset, n_samples: int = 5):
    counter = 0
    save_path = f'results_tables/train_samples/{GlobalConfig.dataset_name}/{GlobalConfig.missingness_type}/'
    os.makedirs(save_path, exist_ok=True)
    
    for amputated_window, amputation_mask, window, window_missing_values in train_dataset:
        if counter >= n_samples:
            break
        
        # Set missing values to np.nan for proper plotting
        window = window.numpy()
        window_missing_values = window_missing_values.numpy()
        amputation_mask = amputation_mask.numpy()
        
        window[window_missing_values == 1] = np.nan  # Replace missing values with np.nan
        synthetic_mask = amputation_mask - window_missing_values  # Highlight synthetic missingness
        
        seq_len, num_vars = window.shape
        time_axis = np.arange(seq_len)  # X-axis is time
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(12, 6))
        for var_idx in range(num_vars):
            ax.plot(time_axis, window[:, var_idx], label=train_dataset.vars['DYNAMIC'][var_idx], alpha=0.7)
            
            # Highlight synthetic missing values in red
            synthetic_indices = np.where(synthetic_mask[:, var_idx] == 1)[0]
            ax.scatter(synthetic_indices, window[synthetic_indices, var_idx], color='red', label='Synthetic Missing' if var_idx == 0 else "", zorder=3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Variables")
        ax.set_title(f"Train Sample {counter}")
        ax.legend()
        
        plt.savefig(os.path.join(save_path, f'sample_{counter}.png'))
        plt.close()
        
        counter += 1


def read_training_loss(file_path):
    training_loss = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"epoch \d+: training loss ([0-9\.]+)", line)
                if match:
                    training_loss.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return training_loss

def assure_minimum_length(dataset):
    if len(dataset) < 2:
        return [dataset[0], dataset[0]]
    return dataset

def pad_collate_fn(batch):
    """Collate function to handle variable-length padding."""
    data, amputation_masks, original, missingness_masks = zip(*batch)
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    amputation_masks = torch.nn.utils.rnn.pad_sequence(amputation_masks, batch_first=True, padding_value=0)
    original = torch.nn.utils.rnn.pad_sequence(original, batch_first=True, padding_value=0)
    missingness_masks = torch.nn.utils.rnn.pad_sequence(missingness_masks, batch_first=True, padding_value=0)
    return data, amputation_masks, original, missingness_masks


@gin.configurable("train_common")
def train_common(
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    max_seq_len: int,
    eval_only: bool = False,
    load_weights: bool = False,
    source_dir: Path = None,
    reproducible: bool = False,
    mode: str = RunMode.classification,
    mask_proportion: float = None,
    mask_method: str=None,
    mask_observation_proportion=0.3,
    model: object = gin.REQUIRED,
    weight: str = None,
    optimizer: type = Adam,
    precision=32,
    batch_size=64,
    epochs=200, # I changed this
    patience=5,
    min_delta=1e-5,
    test_on: str = Split.test,
    dataset_names=None,
    use_wandb: bool = False,
    cpu: bool = False,
    verbose=False,
    ram_cache=False,
    pl_model=True,
    train_only=False,
    num_workers: int = min(cpu_core_count, torch.cuda.device_count() * 8 * int(torch.cuda.is_available()), 32),
    save_ckpt=False,
):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
        max_seq_len: Max allowable stay length. Stays longer will be truncated, shorter will be padded.
        eval_only: If set to true, skip training and only evaluate the model.
        load_weights: If set to true, skip training and load weights from source_dir instead.
        source_dir: If set to load weights, path to directory containing trained weights.
        reproducible: If set to true, set torch to run reproducibly.
        mode: Mode of the model. Can be one of the values of RunMode.
        mask_proportion: The proportion of the dataset to mask for amputation (for Imputation task only),
        mask_method: The mask method to use for dataset amputation (for Imputation task only),
        mask_observation_proportion: The proportion of the observed values to ampute (for Imputation task only),
        model: Model to be trained.
        weight: Weight to be used for the loss function.
        optimizer: Optimizer to be used for training.
        precision: Pytorch precision to be used for training. Can be 16 or 32.
        batch_size: Batch size to be used for training.
        epochs: Number of epochs to train for.
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in loss to be considered an improvement.
        test_on: If set to "test", evaluate the model on the test set. If set to "val", evaluate on the validation set.
        use_wandb: If set to true, log to wandb.
        cpu: If set to true, run on cpu.
        verbose: Enable detailed logging.
        ram_cache: Whether to cache the data in RAM.
        pl_model: Loading a pytorch lightning model.
        num_workers: Number of workers to use for data loading.
        save_ckpt: True iff we want to save the model checkpoint after training
    """

    logging.info(f"Training model: {model.__name__}.")

    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(log_dir)  # We save the operative config before and also after training
    print('Dataset keys: ', data.keys(), flush=True)
    print('val data: ', data['val'],flush=True)
    print('train data num unique stay_ids: ', data['train']['FEATURES']['stay_id'].nunique(), flush=True)
    print('val data num unique stay_ids: ', data['val']['FEATURES']['stay_id'].nunique(), flush=True)
    print('test data num unique stay_ids: ', data['test']['FEATURES']['stay_id'].nunique(), flush=True)
    if mode == RunMode.imputation:
        train_dataset = ImputationDataset(data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"], mask_proportion=mask_proportion, mask_method=mask_method, mask_observation_proportion=mask_observation_proportion, max_seq_len=max_seq_len)
        val_dataset = ImputationDataset(data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"], mask_proportion=mask_proportion, mask_method=mask_method, mask_observation_proportion=mask_observation_proportion, max_seq_len=max_seq_len)
        plot_train_samples(train_dataset)
    else:
        train_dataset = PredictionDataset(data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"])
        val_dataset = PredictionDataset(data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"])
    print('len(train_dataset): ', len(train_dataset), flush=True)
    print('len(val_dataset): ', len(val_dataset), flush=True)

    train_dataset, val_dataset = assure_minimum_length(train_dataset), assure_minimum_length(val_dataset)
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))

    if not eval_only:
        logging.info(
            f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
            f" {len(val_dataset)} samples."
        )
    logging.info(f"Using {num_workers} workers for data loading.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    data_shape = next(iter(train_loader))[0].shape
    print('data_shape: ', data_shape)

    if load_weights:
        model = load_model(model, source_dir, pl_model=pl_model)
    else:
        model = model(optimizer=optimizer, input_size=data_shape, epochs=epochs, run_mode=mode)
    model.log_dir = log_dir

    model.set_weight(weight, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())
    loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(save_dir=log_dir))
    callbacks = [
        EarlyStopping(monitor="val/loss", min_delta=min_delta, patience=patience, strict=False, verbose=verbose),
        ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=min(100, len(train_loader) // 2)))
    if precision == 16 or "16-mixed":
        torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        max_epochs=epochs if model.requires_backprop else 1,
        callbacks=callbacks,
        precision=precision,
        accelerator="auto" if not cpu else "cpu",
        devices=max(torch.cuda.device_count(), 1),
        deterministic="warn" if reproducible else False,
        benchmark=not reproducible,
        enable_progress_bar=verbose,
        logger=loggers,
        num_sanity_val_steps=-1,
        log_every_n_steps=5,
    )
    if not eval_only:
        print('model.requires_backprop: ', model.requires_backprop)
        if model.requires_backprop:
            logging.info("Training DL model.")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            logging.info("Training complete.")
        else:
            logging.info("Training ML model.")
            model.fit(train_dataset, val_dataset)
            logging.info("Training complete.")
    if train_only:
        logging.info("Finished training full model.")
        save_config_file(log_dir)
        return 0
    if mode == RunMode.imputation:
        print('Run mode is imputation, creating ImputationDataset for test set')
        test_dataset = ImputationDataset(data, split=test_on, name=dataset_names["test"], mask_proportion=mask_proportion, mask_method=mask_method, mask_observation_proportion=mask_observation_proportion, max_seq_len=max_seq_len)
    else:
        test_dataset = PredictionDataset(data, split=test_on, name=dataset_names["test"])
    test_dataset = assure_minimum_length(test_dataset)

    print('len(test_dataset): ', len(test_dataset), flush=True)
    print('test set:')
    print(test_dataset, flush=True)
    logging.info(f"Testing on {test_dataset.name}  with {len(test_dataset)} samples.")
    if model.requires_backprop:
        test_loader = DataLoader(
                test_dataset,
                batch_size=min(batch_size * 4, len(test_dataset)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            ) 
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    

    model.set_weight("balanced", train_dataset)
    if model.requires_backprop or model.run_mode == RunMode.imputation: # Standard / old way
        trainer.validate(model, dataloaders=val_loader, verbose=verbose)
        test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
    else:
        model.validation_step(val_dataset, None)
        test_loss = model.test_step((test_dataset.get_data_and_labels()), None)
    

    save_config_file(log_dir)

    # Plotting train loss
    train_losses = read_training_loss(f"slurm_outputs/{GlobalConfig.dataset_name}/{mode.lower()}/{GlobalConfig.dataset_name}_{mode.lower()}_{GlobalConfig.model_name}_{GlobalConfig.missingness_type}.out")
    os.makedirs(f"results_tables/loss_plots/{GlobalConfig.dataset_name}", exist_ok=True)
    plt.plot(train_losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.savefig(f"results_tables/loss_plots/{GlobalConfig.dataset_name}/{GlobalConfig.model_name}_{GlobalConfig.missingness_type}_train_loss.png")

    if save_ckpt:
        model.metrics = {} # Clear metrics, there are issues with saving them
        model.save_model(log_dir, "last")
    return test_loss


def load_model(model, source_dir, pl_model=True):
    if source_dir.exists():
        if model.requires_backprop:
            if (source_dir / "model.ckpt").exists():
                model_path = source_dir / "model.ckpt"
            elif (source_dir / "model-v1.ckpt").exists():
                model_path = source_dir / "model-v1.ckpt"
            elif (source_dir / "last.ckpt").exists():
                model_path = source_dir / "last.ckpt"
            else:
                return Exception(f"No weights to load at path : {source_dir}")
            if pl_model:
                model = model.load_from_checkpoint(model_path)
            else:
                checkpoint = torch.load(model_path)
                model.load_from_checkpoint(checkpoint)
        else:
            model_path = source_dir / "model.joblib"
            model = load(model_path)
    else:
        raise Exception(f"No weights to load at path : {source_dir}")
    logging.info(f"Loaded {type(model)} model from {model_path}")
    return model

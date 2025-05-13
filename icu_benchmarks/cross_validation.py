import json
from datetime import datetime
import logging
import gin
from pathlib import Path
from pytorch_lightning import seed_everything
from icu_benchmarks.global_config import GlobalConfig
from icu_benchmarks.wandb_utils import wandb_log
from icu_benchmarks.run_utils import aggregate_results
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import log_full_line
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import UncertaintyImputationWrapper
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
@gin.configurable
def execute_repeated_cv(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    eval_only: bool = False,
    train_size: int = None,
    load_weights: bool = False,
    source_dir: Path = None,
    cv_repetitions: int = 2, #5,
    cv_repetitions_to_train: int = 1,#None,
    cv_folds: int = 5,
    cv_folds_to_train: int = 1,#None,
    reproducible: bool = False,
    debug: bool = False,
    generate_cache: bool = False,
    load_cache: bool = False,
    test_on: str = "test",
    mode: str = RunMode.classification,
    mask_method: str = None,
    mask_proportion: float = None,
    mask_observation_proportion: float = None,
    pretrained_imputation_model: object = None,
    cpu: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    complete_train: bool = False,
    epochs: int = 100,
    save_ckpt: bool = False,
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:

        complete_train: Use the full data for training instead of held out test splits.
        wandb: Use wandb for logging.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        eval_only: Whether to only evaluate the model.
        train_size: Fixed size of train split (including validation data).
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        cv_folds: Number of folds for cross validation.
        cv_folds_to_train: Number of folds to use during training. If None, all folds are trained on.
        cv_repetitions: Amount of cross validation repetitions.
        cv_repetitions_to_train: Amount of training repetitions. If None, all repetitions are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        generate_cache: Whether to generate and save cache.
        load_cache: Whether to load previously cached data.
        test_on: Dataset to test on. Can be "test" or "val" (e.g. for hyperparameter tuning).
        mode: Run mode. Can be one of the values of RunMode
        pretrained_imputation_model: Use a pretrained imputation model.
        cpu: Whether to run on CPU.
        verbose: Enable detailed logging.
    Returns:
        The average loss of all folds.
    """
    logging.info("Training for %d epochs"%epochs)
    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds
    agg_loss = 0
    seed_everything(seed, reproducible)
    if complete_train:
        logging.info("Will train full model without cross validation.")
        cv_repetitions_to_train = 1
        cv_folds_to_train = 1

    else:
        logging.info(f"Starting nested CV with {cv_repetitions_to_train} repetitions of {cv_folds_to_train} folds.")

    for repetition in range(cv_repetitions_to_train):
        for fold_index in range(cv_folds_to_train):
            repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
            repetition_fold_dir.mkdir(parents=True, exist_ok=True)

            start_time = datetime.now()
            print('train_size: ', train_size)
            # We have an initial threshold of -inf in order to have no selective imputation (ie raw data)
            thresholds = [-np.inf] + list(pretrained_imputation_model.thresholds) if isinstance(pretrained_imputation_model, UncertaintyImputationWrapper) else [None]
            
            for ind, threshold in enumerate(thresholds):
                if isinstance(pretrained_imputation_model, UncertaintyImputationWrapper):
                    repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}" / f"threshold_{round((ind)*0.1, 1)}"
                    repetition_fold_dir.mkdir(parents=True, exist_ok=True)
                    pretrained_imputation_model.set_threshold(threshold) 
                data, min_seq_len, max_seq_len = preprocess_data(
                    data_dir,
                    seed=seed,
                    debug=debug,
                    load_cache=load_cache,
                    generate_cache=generate_cache,
                    cv_repetitions=cv_repetitions,
                    repetition_index=repetition,
                    train_size=train_size,
                    cv_folds=cv_folds,
                    fold_index=fold_index,
                    pretrained_imputation_model=pretrained_imputation_model,
                    runmode=mode,
                    complete_train=complete_train
                )
                if 'OUTCOME' in data['val']:
                    print("Validation set imbalance ratio: ", data['val']['OUTCOME']['label'].value_counts()[1]/data['val']['OUTCOME']['label'].value_counts()[0])
                    print("Train set imbalance ratio: ", data['train']['OUTCOME']['label'].value_counts()[1]/data['train']['OUTCOME']['label'].value_counts()[0])
                    print("Test set imbalance ratio: ", data['test']['OUTCOME']['label'].value_counts()[1]/data['test']['OUTCOME']['label'].value_counts()[0])
                
                preprocess_time = datetime.now() - start_time
                start_time = datetime.now()
                agg_loss += train_common(
                    data,
                    log_dir=repetition_fold_dir,
                    max_seq_len=max_seq_len,
                    eval_only=eval_only,
                    load_weights=load_weights,
                    source_dir=source_dir,
                    reproducible=reproducible,
                    test_on=test_on,
                    mode=mode,
                    mask_method=mask_method,
                    mask_proportion=mask_proportion,
                    mask_observation_proportion=mask_observation_proportion,
                    cpu=cpu,
                    verbose=verbose,
                    use_wandb=wandb,
                    train_only=complete_train,
                    epochs=epochs, # I added this
                    save_ckpt=save_ckpt
                )
                train_time = datetime.now() - start_time

                log_full_line(
                    f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| PROCEDURE DURATION {train_time}",
                    level=logging.INFO,
                )
                durations = {"preprocessing_duration": preprocess_time, "train_duration": train_time}

                with open(repetition_fold_dir / "durations.json", "w") as f:
                    json.dump(durations, f, cls=JsonResultLoggingEncoder)
                if wandb:
                    wandb_log({"Iteration": repetition * cv_folds_to_train + fold_index})
                if repetition * cv_folds_to_train + fold_index > 1:
                    aggregate_results(log_dir)
            
            if isinstance(pretrained_imputation_model, UncertaintyImputationWrapper):
                for metric_type in ['train', 'val', 'test']:
                    pth = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
                    metrics = {}
                    threshold_quantiles = [round(0.1*val, 1) for val in range(0, 11)]
                    for thresh_quant in threshold_quantiles:
                        file_name = pth / f"threshold_{thresh_quant}" / f"{metric_type}_metrics.json"
                        with open(file_name, 'r') as f:
                            data = json.load(f)
                            for k in data.keys():
                                if k not in metrics: 
                                    metrics[k] = [data[k]]
                                else:
                                    metrics[k].append(data[k])
                    save_fldr = f"results_tables/classification_uncertainty_thresholds/{GlobalConfig.dataset_name}/{GlobalConfig.pretrained_imputation_missingness_type}/{pretrained_imputation_model.__class__.__name__}/{metric_type}/"
                    os.makedirs(save_fldr, exist_ok=True)
                    for metric, values in metrics.items():
                        plt.figure(figsize=(8, 6))
                        plt.plot(threshold_quantiles, values, marker='o', linestyle='-')

                        plt.xlabel("Threshold Quantiles")
                        metric_name = metric.replace(f'{metric_type}/', '')
                        plt.ylabel(metric_name)
                        plt.title(f"{metric_name} vs. Threshold Quantiles")
                        plt.grid(True)

                        plot_path = os.path.join(save_fldr, f"{metric_name}.png")
                        plt.savefig(plot_path)
                        plt.close()


                        


        log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)

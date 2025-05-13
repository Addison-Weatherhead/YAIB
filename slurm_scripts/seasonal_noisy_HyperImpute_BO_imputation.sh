#!/bin/bash
#SBATCH --job-name=seasonal_noisy_HyperImpute_BO_imputation
#SBATCH --output=./slurm_outputs/seasonal_noisy/imputation/seasonal_noisy_imputation_HyperImpute_BO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "seasonal_noisy HyperImpute BO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/seasonal_noisy" \
    --name "seasonal_noisy" \
    --experiment "configs/imputation_models/HyperImpute" \
    --task SimulatedDatasetImputation \
    --model "HyperImpute" \
    --mask-method "BO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "seasonal_noisy HyperImpute BO Imputation End"

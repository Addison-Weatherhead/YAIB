#!/bin/bash
#SBATCH --job-name=seasonal_HyperImpute_MCAR_imputation
#SBATCH --output=./slurm_outputs/seasonal/imputation/seasonal_imputation_HyperImpute_MCAR.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "seasonal HyperImpute MCAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/seasonal" \
    --name "seasonal" \
    --experiment "configs/imputation_models/HyperImpute" \
    --task SimulatedDatasetImputation \
    --model "HyperImpute" \
    --mask-method "MCAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "seasonal HyperImpute MCAR Imputation End"

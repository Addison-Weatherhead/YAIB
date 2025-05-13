#!/bin/bash
#SBATCH --job-name=trend_sinusoids_HyperImpute_blockBO_imputation
#SBATCH --output=./slurm_outputs/trend_sinusoids/imputation/trend_sinusoids_imputation_HyperImpute_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "trend_sinusoids HyperImpute blockBO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/trend_sinusoids" \
    --name "trend_sinusoids" \
    --experiment "configs/imputation_models/HyperImpute" \
    --task SimulatedDatasetImputation \
    --model "HyperImpute" \
    --mask-method "blockBO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "trend_sinusoids HyperImpute blockBO Imputation End"

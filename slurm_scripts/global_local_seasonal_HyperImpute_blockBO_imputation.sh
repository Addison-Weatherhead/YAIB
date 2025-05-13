#!/bin/bash
#SBATCH --job-name=global_local_seasonal_HyperImpute_blockBO_imputation
#SBATCH --output=./slurm_outputs/global_local_seasonal/imputation/global_local_seasonal_imputation_HyperImpute_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "global_local_seasonal HyperImpute blockBO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/global_local_seasonal" \
    --name "global_local_seasonal" \
    --experiment "configs/imputation_models/HyperImpute" \
    --task SimulatedDatasetImputation \
    --model "HyperImpute" \
    --mask-method "blockBO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "global_local_seasonal HyperImpute blockBO Imputation End"

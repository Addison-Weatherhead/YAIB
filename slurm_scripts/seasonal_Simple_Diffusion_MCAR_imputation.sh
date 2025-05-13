#!/bin/bash
#SBATCH --job-name=seasonal_Simple_Diffusion_MCAR_imputation
#SBATCH --output=./slurm_outputs/seasonal/imputation/seasonal_imputation_Simple_Diffusion_MCAR.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "seasonal Simple_Diffusion MCAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/seasonal" \
    --name "seasonal" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task SimulatedDatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "MCAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "seasonal Simple_Diffusion MCAR Imputation End"

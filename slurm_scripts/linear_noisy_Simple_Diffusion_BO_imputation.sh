#!/bin/bash
#SBATCH --job-name=linear_noisy_Simple_Diffusion_BO_imputation
#SBATCH --output=./slurm_outputs/linear_noisy/imputation/linear_noisy_imputation_Simple_Diffusion_BO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "linear_noisy Simple_Diffusion BO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/linear_noisy" \
    --name "linear_noisy" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task SimulatedDatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "BO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "linear_noisy Simple_Diffusion BO Imputation End"

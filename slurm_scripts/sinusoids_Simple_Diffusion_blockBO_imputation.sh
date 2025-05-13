#!/bin/bash
#SBATCH --job-name=sinusoids_Simple_Diffusion_blockBO_imputation
#SBATCH --output=./slurm_outputs/sinusoids/imputation/sinusoids_imputation_Simple_Diffusion_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "sinusoids Simple_Diffusion blockBO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/sinusoids" \
    --name "sinusoids" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task SimulatedDatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "blockBO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "sinusoids Simple_Diffusion blockBO Imputation End"

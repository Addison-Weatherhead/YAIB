#!/bin/bash
#SBATCH --job-name=exponential_decay_Simple_Diffusion_BO_imputation
#SBATCH --output=./slurm_outputs/exponential_decay/imputation/exponential_decay_imputation_Simple_Diffusion_BO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=2:00:00

echo "exponential_decay Simple_Diffusion BO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "demo_data/simulated/exponential_decay" \
    --name "exponential_decay" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task SimulatedDatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "BO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "exponential_decay Simple_Diffusion BO Imputation End"

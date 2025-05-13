#!/bin/bash
#SBATCH --job-name=eicu_Simple_Diffusion_blockBO_imputation
#SBATCH --output=./slurm_outputs/eicu/imputation/eicu_imputation_Simple_Diffusion_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "eicu Simple_Diffusion blockBO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/eicu" \
    --name "eicu" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task DatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "blockBO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "eicu Simple_Diffusion blockBO Imputation End"

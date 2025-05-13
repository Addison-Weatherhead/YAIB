#!/bin/bash
#SBATCH --job-name=hirid_Simple_Diffusion_MCAR_imputation
#SBATCH --output=./slurm_outputs/hirid/imputation/hirid_imputation_Simple_Diffusion_MCAR.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=4:00:00

echo "hirid Simple_Diffusion MCAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/hirid" \
    --name "hirid" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task DatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "MCAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "hirid Simple_Diffusion MCAR Imputation End"

#!/bin/bash
#SBATCH --job-name=miiv_Simple_Diffusion_MAR_imputation
#SBATCH --output=./slurm_outputs/miiv/imputation/miiv_imputation_Simple_Diffusion_MAR.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "miiv Simple_Diffusion MAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/miiv" \
    --name "miiv" \
    --experiment "configs/imputation_models/Simple_Diffusion" \
    --task DatasetImputation \
    --model "Simple_Diffusion" \
    --mask-method "MAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "miiv Simple_Diffusion MAR Imputation End"

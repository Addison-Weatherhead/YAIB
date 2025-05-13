#!/bin/bash
#SBATCH --job-name=miiv_HyperImpute_BO_imputation
#SBATCH --output=./slurm_outputs/miiv/imputation/miiv_imputation_HyperImpute_BO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "miiv HyperImpute BO Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/miiv" \
    --name "miiv" \
    --experiment "configs/imputation_models/HyperImpute" \
    --task DatasetImputation \
    --model "HyperImpute" \
    --mask-method "BO" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "miiv HyperImpute BO Imputation End"

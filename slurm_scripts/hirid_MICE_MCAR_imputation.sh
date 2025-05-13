#!/bin/bash
#SBATCH --job-name=hirid_MICE_MCAR_imputation
#SBATCH --output=./slurm_outputs/hirid/imputation/hirid_imputation_MICE_MCAR.out
#SBATCH --qos=cpu_qos
#SBATCH --partition='cpu'
#
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "hirid MICE MCAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/hirid" \
    --name "hirid" \
    --experiment "configs/imputation_models/MICE" \
    --task DatasetImputation \
    --model "MICE" \
    --mask-method "MCAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "hirid MICE MCAR Imputation End"

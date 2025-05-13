#!/bin/bash
#SBATCH --job-name=eicu_Forward_MCAR_imputation
#SBATCH --output=./slurm_outputs/eicu/imputation/eicu_imputation_Forward_MCAR.out
#SBATCH --qos=cpu_qos
#SBATCH --partition='cpu'
#
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "eicu Forward MCAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/eicu" \
    --name "eicu" \
    --experiment "configs/imputation_models/Forward" \
    --task DatasetImputation \
    --model "Forward" \
    --mask-method "MCAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "eicu Forward MCAR Imputation End"

#!/bin/bash
#SBATCH --job-name=miiv_MICE_MNAR_imputation
#SBATCH --output=./slurm_outputs/miiv/imputation/miiv_imputation_MICE_MNAR.out
#SBATCH --qos=cpu_qos
#SBATCH --partition='cpu'
#
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00

echo "miiv MICE MNAR Imputation Start"

# Imputation task
icu-benchmarks train \
    --data-dir "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/miiv" \
    --name "miiv" \
    --experiment "configs/imputation_models/MICE" \
    --task DatasetImputation \
    --model "MICE" \
    --mask-method "MNAR" \
    --mask-proportion 0.3 \
    --mask-observation-proportion 0.3 \
    --log-dir ../yaib_logs/ \
    --save-checkpoint \
    #--seed 2222 \



echo "miiv MICE MNAR Imputation End"

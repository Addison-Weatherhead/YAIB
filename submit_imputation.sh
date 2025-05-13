#!/bin/bash
#SBATCH --job-name=eicu_Attention_blockBO_imputation
#SBATCH --output=./slurm_outputs/eicu/eicu_imputation_Attention_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,t4v1,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=8:00:00
#SBATCH --exclude=gpu154

echo "eicu Attention blockBO Imputation Start"
icu-benchmarks train -d "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/eicu" -n "eicu" -t DatasetImputation -m "Attention" -mm "blockBO" -mp 0.3 -mop 0.3 -lc -gc -s 2222 -l ../yaib_logs/
echo "eicu Attention blockBO Imputation End"

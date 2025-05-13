#!/bin/bash
#SBATCH --job-name=hirid_Attention_blockBO_classification
#SBATCH --output=./slurm_outputs/hirid/classification/hirid_classification_Attention_blockBO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=3:00:00

echo "hirid Attention blockBO Classification Start"

# Classification task
icu-benchmarks train \
    --data-dir /projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/hirid \
    --name hirid \
    --task BinaryClassification \
    --task-name Mortality24 \
    --model LGBMClassifier \
    --hyperparams LGBMClassifier.min_child_samples=10 \
    --log-dir ../yaib_logs/ \
    --pretrained-imputation /fs01/home/addisonw/yaib_logs/hirid/configs/imputation_models/optimal_hyper_params/Attention/blockBO/hirid/config/2025-02-25T00-55-07/repetition_0/fold_0/last.ckpt \
    #--tune \ # Forego for now
    #--seed 2222 \

echo "hirid Attention blockBO Classification End"

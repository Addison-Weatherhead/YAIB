#!/bin/bash
#SBATCH --job-name=eicu_Attention_MNAR_classification
#SBATCH --output=./slurm_outputs/eicu/classification/eicu_classification_Attention_MNAR.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=3:00:00

echo "eicu Attention MNAR Classification Start"

# Classification task
icu-benchmarks train \
    --data-dir /projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/eicu \
    --name eicu \
    --task BinaryClassification \
    --task-name Mortality24 \
    --model LGBMClassifier \
    --hyperparams LGBMClassifier.min_child_samples=10 \
    --log-dir ../yaib_logs/ \
    --pretrained-imputation /fs01/home/addisonw/yaib_logs/eicu/configs/imputation_models/optimal_hyper_params/Attention/MNAR/eicu/config/2025-02-24T23-05-31/repetition_0/fold_0/last.ckpt \
    #--tune \ # Forego for now
    #--seed 2222 \

echo "eicu Attention MNAR Classification End"

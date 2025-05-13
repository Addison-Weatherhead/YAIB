#!/bin/bash
#SBATCH --job-name=miiv_Attention_BO_classification
#SBATCH --output=./slurm_outputs/miiv/classification/miiv_classification_Attention_BO.out
#SBATCH --qos=normal
#SBATCH --partition='t4v2,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time=3:00:00

echo "miiv Attention BO Classification Start"

# Classification task
icu-benchmarks train \
    --data-dir /projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/miiv \
    --name miiv \
    --task BinaryClassification \
    --task-name Mortality24 \
    --model LGBMClassifier \
    --hyperparams LGBMClassifier.min_child_samples=10 \
    --log-dir ../yaib_logs/ \
    --pretrained-imputation /fs01/home/addisonw/yaib_logs/miiv/configs/imputation_models/optimal_hyper_params/Attention/BO/miiv/config/2025-02-24T22-05-09/repetition_0/fold_0/last.ckpt \
    #--tune \ # Forego for now
    #--seed 2222 \

echo "miiv Attention BO Classification End"

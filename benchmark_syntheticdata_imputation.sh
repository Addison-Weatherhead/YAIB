#!/bin/bash
#SBATCH --job-name=benchmark_syntheticdata_imputation
#SBATCH --output=synthdata_imputation_MCAR.out
#SBATCH --qos=scavenger
#SBATCH --partition=t4v1,t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --open-mode=truncate
#SBATCH --time=4:00:00


datasets=("exponential_decay" "linear" "seasonal" "linear_noisy" "seasonal_noisy")
methods=("Mean" "Attention" "MICE" "SAITS" "MissForest")

for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    echo "$dataset $method Imputation Start"
    icu-benchmarks train -d "demo_data/simulated/$dataset" -n "$dataset" -t SimulatedDatasetImputationMCAR -m "$method" -lc -gc -s 2222 -l ../yaib_logs/
    echo "$dataset $method Imputation End"
  done
done



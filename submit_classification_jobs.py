import os
import subprocess
import yaml

# DONT CHANGE
datasets = ["miiv", "eicu", "hirid"]
methods = ["Attention", "SAITS"]
missingness_types = ["MCAR", "MAR", "MNAR", "BO", "blockBO"]

# For running
datasets = ["miiv"]
methods = ["Attention"]
missingness_types = ["MNAR"]

method_gpu_usage = {
    "Attention": True,
    "SAITS": True,
}

CPU_PARTITIONS = "cpu"
GPU_PARTITIONS = "t4v2,rtx6000,a40"
EHR_TIME_LIMIT = "3:00:00"

base_sh_script = """#!/bin/bash
#SBATCH --job-name={dataset}_{method}_{missingness_type}_classification
#SBATCH --output={output_path}
#SBATCH --qos={qos}
#SBATCH --partition='{partition}'
{gpu_resources}
#SBATCH --mem=32G
#SBATCH --open-mode=truncate
#SBATCH --time={time_limit}

echo "{dataset} {method} {missingness_type} Classification Start"

# Classification task
icu-benchmarks train \\
    --data-dir {data_dir} \\
    --name {dataset} \\
    --task BinaryClassification \\
    --task-name Mortality24 \\
    --model LGBMClassifier \\
    --hyperparams LGBMClassifier.min_child_samples=10 \\
    --log-dir ../yaib_logs/ \\
    --pretrained-imputation {path_to_ckpt} \\
    #--tune \\ # Forego for now
    #--seed 2222 \\

echo "{dataset} {method} {missingness_type} Classification End"
"""

output_dir = "./slurm_outputs"
scripts_dir = "./slurm_scripts"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)

for dataset in datasets:
    for method in methods:
        for missingness_type in missingness_types:

            data_dir = (
                f"/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/{dataset}"
            )

            out_file_name = f"{dataset}/classification/{dataset}_classification_{method}_{missingness_type}.out"
            out_file_path = os.path.join(output_dir, out_file_name)
            with open(f'slurm_outputs/{dataset}/imputation/{dataset}_imputation_{method}_{missingness_type}.out', 'r') as f:
                imputation_file = f.read()
            path_to_ckpt = '/' + imputation_file.split('INFO - root : Logging to /')[1].split('\n')[0] + '/repetition_0/fold_0/last.ckpt'
            script_content = base_sh_script.format(
                output_path=out_file_path,
                qos='normal' if method_gpu_usage[method] else 'cpu_qos',
                partition=GPU_PARTITIONS if method_gpu_usage[method] else CPU_PARTITIONS,
                gpu_resources="#SBATCH --gres=gpu:1" if method_gpu_usage[method] else "#",
                dataset=dataset,
                method=method,
                missingness_type=missingness_type,
                data_dir=data_dir,
                time_limit=EHR_TIME_LIMIT,
                path_to_ckpt=path_to_ckpt
            )

            script_path = os.path.join(scripts_dir, f"{dataset}_{method}_{missingness_type}_classification.sh")

            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            subprocess.run(["sbatch", script_path])

            print(f"Submitted job for {dataset} with {method} and {missingness_type}")

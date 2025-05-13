import os
import subprocess
import yaml
import time

# DONT CHANGE
synthetic_datasets = [
    "exponential_decay", "linear", "seasonal", "linear_noisy", "seasonal_noisy",
    "sinusoids", "trend_sinusoids", "global_local_seasonal"
]
datasets = ["miiv", "eicu", "hirid", *synthetic_datasets]
methods = ["Attention", "SAITS", "Simple_Diffusion", "Forward", "Mean", "MICE", "MissForest"]
missingness_types = ["MCAR", "MAR", "MNAR", "BO", "blockBO"]

# For running
datasets = ["miiv", "eicu", "hirid"]
methods = ["Mean"]
missingness_types = ["blockBO"]


method_gpu_usage = {
    "Attention": True,
    "SAITS": True,
    "HyperImpute": True,
    "Simple_Diffusion": True,
    "Mean": False,
    "Forward": False,
    "MICE": False,
    "MissForest": False
}

CPU_PARTITIONS = "cpu"
GPU_PARTITIONS = "t4v2,rtx6000,a40"
SIM_TIME_LIMIT = "2:00:00"
EHR_TIME_LIMIT = "8:00:00"

base_sh_script = """#!/bin/bash
#SBATCH --job-name={dataset}_{method}_{missingness_type}_imputation
#SBATCH --output={output_path}
#SBATCH --qos={qos}
#SBATCH --partition='{partition}'
{gpu_resources}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate
#SBATCH --time={time_limit}

echo "{dataset} {method} {missingness_type} Imputation Start"

# Imputation task
icu-benchmarks train \\
    --data-dir "{data_dir}" \\
    --name "{dataset}" \\
    --experiment "{config_path}" \\
    --task {task} \\
    --model "{method}" \\
    --mask-method "{missingness_type}" \\
    --mask-proportion 0.3 \\
    --mask-observation-proportion 0.3 \\
    --log-dir ../yaib_logs/ \\
    --save-checkpoint \\
    #--seed 2222 \\



echo "{dataset} {method} {missingness_type} Imputation End"
"""

output_dir = "./slurm_outputs"
scripts_dir = "./slurm_scripts"
config_base_dir = "configs/imputation_models/optimal_hyper_params"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)

for dataset in datasets:
    for method in methods:
        for missingness_type in missingness_types:
            time.sleep(1)
            config_path = os.path.join(config_base_dir, method, missingness_type, dataset, "config.gin")

            if not os.path.exists(config_path):
                config_path = os.path.join("configs/imputation_models/", f"{method}.gin")
                print(f"Using default gin for {method}")


            data_dir = (
                f"demo_data/simulated/{dataset}"
                if dataset in synthetic_datasets
                else f"/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/{dataset}"
                #else f"/projects/YAIB-cohorts-processed-data/COHORT_data/base/{dataset}"
            )

            out_file_name = f"{dataset}/imputation/{dataset}_imputation_{method}_{missingness_type}.out"
            out_file_path = os.path.join(output_dir, out_file_name)

            script_content = base_sh_script.format(
                output_path=out_file_path,
                qos='normal' if method_gpu_usage[method] else 'cpu_qos',
                partition=GPU_PARTITIONS if method_gpu_usage[method] else CPU_PARTITIONS,
                gpu_resources="#SBATCH --gres=gpu:1" if method_gpu_usage[method] else "#",
                dataset=dataset,
                method=method,
                task='SimulatedDatasetImputation' if dataset in synthetic_datasets else 'DatasetImputation',
                missingness_type=missingness_type,
                config_path=config_path.replace('.gin', ''),
                data_dir=data_dir,
                output_dir=output_dir,
                time_limit=SIM_TIME_LIMIT if dataset in synthetic_datasets else EHR_TIME_LIMIT
            )

            script_path = os.path.join(scripts_dir, f"{dataset}_{method}_{missingness_type}_imputation.sh")

            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            subprocess.run(["sbatch", script_path])

            print(f"Submitted job for {dataset} with {method} and {missingness_type}")

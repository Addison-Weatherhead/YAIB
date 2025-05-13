import os
import subprocess
import yaml
import re

models = ['Attention', 'SAITS']
missingness_types = ['MCAR', 'MAR', 'BO', 'MNAR', 'blockBO']
datasets = ['eicu', 'exponential_decay', 'global_local_seasonal', 'linear', 'linear_noisy', 
            'seasonal', 'seasonal_noisy', 'sinusoids', 'trend_sinusoids']
agents_per_sweep = 4


models = ['SAITS']
missingness_types = ['BO']
datasets = ['seasonal', 'seasonal_noisy', 'sinusoids']

# Base directories for config files
config_files = {
    'Attention': 'scripts/sweep_configs/hyperparameter_sweeps/attention_sweep.yml',
    'SAITS': 'scripts/sweep_configs/hyperparameter_sweeps/saits_sweep.yml'
}

output_base_path = "hyper_param_tuning"

# Path for the eicu dataset
eicu_data_dir = "/projects/YAIB-cohorts-processed-data/COHORT_data/mortality24/eicu/"
eicu_task = "DatasetImputation"

# Path for simulated datasets
simulated_data_dir = "demo_data/simulated/"

# Function to update and write the configuration file
def update_config(model, missingness, dataset):
    # Load the base config file depending on the model
    config_path = config_files.get(model)
    
    if not config_path:
        print(f"No config file found for model: {model}, skipping...")
        return None

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the mask method (missingness type)
    config['parameters']['ImputationDataset.mask_method']['values'] = [missingness]

    # Update the data directory and task
    if dataset == "eicu":
        config['parameters']['data_dir']['values'] = [eicu_data_dir]
        config['command'][4] = eicu_task  # Set the task to DatasetImputation for eicu
        config['command'][6] = eicu_data_dir
    else:
        config['parameters']['data_dir']['values'] = [f"{simulated_data_dir}{dataset}"]
        config['command'][4] = "SimulatedDatasetImputation"  # Set task to SimulatedDatasetImputation
        config['command'][6] = f"{simulated_data_dir}{dataset}"


    # Update the name with model, dataset, and missingness
    config['name'] = f"{model} hyperparameter sweep - {dataset} - {missingness}"

    # Write the updated config to a temporary file
    temp_config_path = "temp_sweep_config.yml"
    with open(temp_config_path, 'w') as file:
        yaml.dump(config, file)

    return temp_config_path

# Function to create and submit SLURM job for each agent
def submit_sweep_job(model, missingness, dataset, agent_id, full_sweep_id):
    # Create the output path for the agent
    output_path = f"{output_base_path}/{model}/{missingness}/{dataset}_agent{agent_id}.out"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directories if they don't exist
    
    # Create the SLURM sbatch script content
    sbatch_script = f"""#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition='t4v2,t4v1,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --output={output_path}

# Run the wandb agent
wandb agent {full_sweep_id}
"""
    # Write the sbatch script to a single file that will be overwritten each time
    sbatch_filename = "run_sweep.sh"  # Overwrite each time
    with open(sbatch_filename, 'w') as f:
        f.write(sbatch_script)

    # Submit the job
    subprocess.run(["sbatch", sbatch_filename])

# Function to initiate sweep and create agents
def run_sweep_for_combination(model, missingness, dataset):
    # Update the YAML config file
    temp_config_path = update_config(model, missingness, dataset)
    
    if not temp_config_path:
        return  # Skip if no config path could be generated

    # Initialize the sweep and capture the sweep ID using wandb command
    result = subprocess.run(['wandb', 'sweep', temp_config_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to initialize sweep for {temp_config_path}: {result.stderr}")
        return
    
    # Print the entire stdout and stderr to debug
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    
    # Use regex to extract the full sweep path (entity/project/sweep_id) from stderr
    full_sweep_id_match = re.search(r"wandb agent ([\w\-]+\/[\w\-]+\/[\w\-]+)", result.stderr)
    
    if full_sweep_id_match:
        full_sweep_id = full_sweep_id_match.group(1)
        print(f"Extracted full sweep ID: {full_sweep_id}")
    else:
        print(f"Failed to extract full sweep ID for {temp_config_path}")
        print(f"wandb output: {result.stderr}")
        return

    for agent_id in range(agents_per_sweep):
        submit_sweep_job(model, missingness, dataset, agent_id, full_sweep_id)

# Main loop to iterate through all combinations
for model in models:
    for missingness in missingness_types:
        for dataset in datasets:
            print(f"Running sweep for Model: {model}, Missingness: {missingness}, Dataset: {dataset}")
            run_sweep_for_combination(model, missingness, dataset)

import os
import re
import wandb
from pathlib import Path

wandb.login()

# Define parameters
models = ['Attention', 'SAITS']
missingness_types = ['blockBO', 'BO', 'MAR', 'MCAR', 'MNAR']
datasets = ['eicu', 'exponential_decay', 'global_local_seasonal', 'linear', 'linear_noisy', 'seasonal', 'seasonal_noisy', 'sinusoids', 'trend_sinusoids']
agent_file_template = "hyper_param_tuning/{model}/{missingness_type}/{dataset}_agent0.out"
save_dir = "configs/imputation_models/optimal_hyper_params"

os.makedirs(save_dir, exist_ok=True)

sweep_link_regex = r"View sweep at (https://wandb\\.ai/.+?/sweeps/\\w+)"

def extract_sweep_link(agent_file_path):
    """Extracts the sweep link from the agent output file."""
    with open(agent_file_path, 'r') as file:
        for line in file:
            if 'View sweep at ' in line:
                # Extract the portion after 'View sweep at ' up to the end of the line
                return line.split('View sweep at ')[1].strip()
    return None


def get_best_run_config(sweep_url, save_path):
    """Extracts the best run config from the sweep and saves it to a file."""
    # Parse sweep details from URL
    entity, project, sweep_id = sweep_url.split('/')[-4], sweep_url.split('/')[-3], sweep_url.split('/')[-1]

    # Initialize W&B API and fetch the sweep
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    # Find the best run based on `val/mae`
    best_run = min(sweep.runs, key=lambda run: run.summary.get('val/mae', float('inf')))

    # Save the config to the specified path
    config = best_run.config
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    return best_run.url

# Iterate over all combinations
for model in models:
    for missingness_type in missingness_types:
        for dataset in datasets:
            # Construct the path to the agent output file
            agent_file_path = agent_file_template.format(
                model=model, 
                missingness_type=missingness_type, 
                dataset=dataset
            )

            # Check if the agent file exists
            if not os.path.exists(agent_file_path):
                print(f"Agent file not found: {agent_file_path}")
                continue

            # Extract the sweep link
            sweep_url = extract_sweep_link(agent_file_path)
            if not sweep_url:
                print(f"Sweep link not found in: {agent_file_path}")
                continue

            # Determine the save path for the config file
            config_save_path = f"{save_dir}/{model}/{missingness_type}/{dataset}/config.yaml"

            # Extract the best run and save the config
            try:
                best_run_url = get_best_run_config(sweep_url, config_save_path)
            except Exception as e:
                print(f"Failed to process sweep {sweep_url}: {e}")
                continue

            # Print the required information
            print(f"{model} {missingness_type} {dataset}")
            print(f"Sweep: {sweep_url}")
            print(f"Best Run: {best_run_url}\n")

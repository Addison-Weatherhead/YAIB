import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

EHR_DATASETS = ['eicu', 'miiv', 'hirid']
MISSINGNESS_TYPES = ['BO', 'MAR', 'MCAR', 'MNAR', 'blockBO']
MODEL_NAMES = ['Attention', 'SAITS', 'MICE', 'Mean', 'MissForest', 'Simple_Diffusion']

COLORS = {
    'BO': '#1f77b4',    # Blue
    'MAR': '#ff7f0e',   # Orange
    'MCAR': '#2ca02c',  # Green
    'MNAR': '#d62728',   # Red
    'blockBO': '#FFD700' # Gold
}

def process_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        parts = contents.split("Accumulated results: ")
        if len(parts) < 2:
            print('No Accumulated Results found')
            return None
        last_part = parts[-1]
        results_dict = eval(last_part.split('\n')[0].replace("nan", "'nan'"))
        mae = 'test_mae' if 'test_mae' in results_dict['avg'] else 'mae'
        avg_mae = round(results_dict['avg'][mae], 3)
        return avg_mae
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return None
    except Exception as e:
        return None

def collect_results(directory):
    results = {dataset: {} for dataset in EHR_DATASETS}

    for dataset in EHR_DATASETS:
        dataset_path = os.path.join(directory, dataset, 'imputation')
        if not os.path.exists(dataset_path):
            continue
        for file in os.listdir(dataset_path):
            if file.endswith('.out'):
                try:
                    file_path = os.path.join(dataset_path, file)
                    print('Processing file: ', file_path)
                    parts = file.replace('.out', '').split('_imputation_')[1].rsplit('_', 1)
                    model, missingness = parts
                    if model in MODEL_NAMES:
                        if missingness not in MISSINGNESS_TYPES:
                            continue
                        mae = process_file(file_path)
                        if mae:
                            if model not in results[dataset]:
                                results[dataset][model] = {}
                            results[dataset][model][missingness] = mae
                except Exception as e:
                    continue
    return results

def plot_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset, model_data in results.items():
        models = [model for model in MODEL_NAMES if model in model_data]
        group_width = 1.0  # controls space between groups
        bar_width = 0.15   # narrower bars if you want room within group

        x = np.arange(len(models)) * group_width
        offsets = np.linspace(-bar_width * (len(MISSINGNESS_TYPES)-1)/2,
                            bar_width * (len(MISSINGNESS_TYPES)-1)/2,
                            len(MISSINGNESS_TYPES))

        
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, miss_type in enumerate(MISSINGNESS_TYPES):
            mae_vals = [model_data[model].get(miss_type, np.nan) for model in models]
            bar_pos = x + offsets[i]
            ax.bar(bar_pos, mae_vals, bar_width, label=miss_type, color=COLORS[miss_type])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title(f'{dataset.upper()} - Imputation MAE by Missingness Type')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Missingness type')
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.85))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_mae_barplot.png'))
        plt.close()

# Set paths
slurm_outputs_dir = 'slurm_outputs'
output_plots_dir = 'results_plots'

# Run process
results = collect_results(slurm_outputs_dir)
plot_results(results, output_plots_dir)

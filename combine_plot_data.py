import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
COMBINE_IMPUTATION = False
COMBINE_CLASSIFICATION = False

synthetic_datasets = [
    "exponential_decay", "linear", "seasonal", "linear_noisy", "seasonal_noisy",
    "sinusoids", "trend_sinusoids", "global_local_seasonal"
]
datasets = ["miiv", "eicu", "hirid", *synthetic_datasets]
methods = ["Attention", "SAITS"]
missingness_types = ["MCAR", "MAR", "MNAR", "BO", "blockBO"]





datasets = ["miiv", "eicu", "hirid", "exponential_decay", "linear", "seasonal", "seasonal_noisy"]
methods = ["Attention", "SAITS"]
missingness_types = ["MCAR", "MAR", "MNAR", "BO", "blockBO"]

dataset_formatted_names = {'miiv': 'MIMIC IV', 'eicu': 'eICU', 'hirid': 'HiRID', 'exponential_decay': 'Exponential Decay', 'linear': 'Linear', 'seasonal': 'Seasonal', 'seasonal_noisy': 'Seasonal Noisy'}
colors = ['steelblue', 'crimson', 'darkorange', 'forestgreen', 'purple']
if COMBINE_IMPUTATION:
    for dataset in datasets:
        fig, axes = plt.subplots(len(methods), 1, figsize=(8, 6*len(methods)))
        for i, model in enumerate(methods):
            for j, missingness_type in enumerate(missingness_types):
                df = pd.read_csv(f'table_data/imputation/uncertainty_mae_{dataset}_{model}_{missingness_type}.csv')
                quantiles = np.array(df['Uncertainty Quantiles'].values)
                mean_mae = np.array(df['Mean MAE'].values)
                std_mae = np.array(df['Std MAE'].values)
                ax = axes[i]
                ax.fill_between(quantiles, mean_mae - std_mae, mean_mae + std_mae, color=colors[j], alpha=0.3, label=f"{missingness_type} Mean MAE ±1 Std Dev")
                ax.set_title(f"{dataset_formatted_names[dataset]} - {model}")
                ax.set_xlabel("Uncertainty Quantiles")
                ax.set_ylabel("MAE (on imputations with uncertainty<threshold)")
                ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"results_tables/combined_uncertainty_threshold_MAE/{dataset}.png", dpi=200, bbox_inches="tight")





datasets = ["miiv", "eicu", "hirid"]
methods = ["Attention", "SAITS"]
missingness_types = ["MCAR", "MAR", "MNAR", "BO", "blockBO"]

if COMBINE_CLASSIFICATION:
    for dataset in datasets:
        for i, model in enumerate(methods):
            for j, missingness_type in enumerate(missingness_types):
                output_file = f'slurm_outputs/{dataset}/classification/{dataset}_classification_{model}_{missingness_type}.out'
                with open(output_file, 'r') as f:
                    content = f.read()
                print(dataset, model, missingness_type)
                log_dir = content.split('Logging to directory: ')[1].split('threshold')[0]
                val_ratio = float(content.split('Validation set imbalance ratio:  ')[1].split('\n')[0])
                val_AUPRCs = []
                test_AUPRCs = []
                thresholds = [round(val*0.1, 1) for val in range(0, 11)]
                for threshold in thresholds:
                    with open(log_dir + f'threshold_{threshold}/val_metrics.json', 'r') as f:
                        val_data = json.load(f)
                    with open(log_dir + f'threshold_{threshold}/test_metrics.json', 'r') as f:
                        test_data = json.load(f)
                    val_AUPRCs.append(val_data['val/AUPRC'])
                    test_AUPRCs.append(test_data['test/AUPRC'])
                plt.figure(figsize=(8, 6))
                plt.plot(thresholds, val_AUPRCs)
                plt.grid()
                #plt.plot(thresholds, test_AUPRCs, label=f'Test AUPRC', color=colors[1])
                plt.title(f'Validation AUPRC for {model} on {dataset} with {missingness_type}')
                plt.xlabel('Uncertainty Quantiles')
                plt.ylabel('AUPRC')
                #plt.legend()
                #plt.axhline(y=val_ratio, color='black', linestyle=':', label='Random Performance')
                plt.tight_layout()
                plt.savefig(f"results_tables/combined_classification_uncertainty_thresholds/{dataset}_{model}_{missingness_type}.png", dpi=200, bbox_inches="tight")




s = r"""\begin{figure}[h]
   \centering    
   \includegraphics[width=3in]{images/classification_plots/IMAGE_NAME.png} 
   \caption{Impact of Uncertainty on AUPRC for a classifier trained for mortality prediction, using a pretrained imputation model trained for MISSINGNESS_TYPE missingness on DATASET}
   \label{fig:FIG_LABEL} 
 \end{figure}
"""

for dataset in datasets:
    for missingness_type in missingness_types:
        for model in methods:
            print(s.replace('IMAGE_NAME', dataset.lower() + '_' + model + '_' + missingness_type).replace('MISSINGNESS_TYPE', missingness_type).replace('DATASET', dataset).replace('FIG_LABEL', 'fig:classfcn_' + dataset.lower() + '_' + model + '_' + missingness_type))


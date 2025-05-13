
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
    

for dataset in EHR_DATASETS:
    for model_name in MODEL_NAMES:
        for missingness_type in MISSINGNESS_TYPES: 
            print(f'Dataset: {dataset}, Missingness Type: {missingness_type}, Model Name: {model_name}')
            file_name = f'slurm_outputs/{dataset}/imputation/{dataset}_imputation_{model_name}_{missingness_type}.out'
            avg_mae = process_file(file_name)
            if not avg_mae:
                print('MAE IS NONE')
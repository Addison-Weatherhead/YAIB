import os
import yaml

# Base path where YAML files are stored
base_path = "configs/imputation_models/optimal_hyper_params"

# Function to convert YAML content to a .gin string
def convert_yaml_to_gin(yaml_content, model_name, dataset_name):
    gin_lines = [
        "import gin.torch.external_configurables",
        "import icu_benchmarks.models.wrappers",
        "import icu_benchmarks.models.dl_models",
        "import icu_benchmarks.models.utils",
        "import icu_benchmarks.imputation.baselines",
        "import icu_benchmarks.data.preprocessor",
        "",
        "# Train params",
        f"train_common.model = @{model_name}",
        "preprocess.min_seq_len=169" if dataset_name=='eicu' else "",
        "",
        "# Model params"
    ]

    # Map YAML keys to gin parameter names
    param_prefix = model_name
    model_specific_params = [
        f"{param_prefix}.n_layers",
        f"{param_prefix}.d_model",
        f"{param_prefix}.d_inner",
        f"{param_prefix}.n_head",
        f"{param_prefix}.d_k",
        f"{param_prefix}.d_v",
        f"{param_prefix}.dropout",
        f"{param_prefix}.epochs",
    ]

    for key, value in yaml_content.items():
        # Handle ImputationWrapper.lr separately
        if key == "Adam.lr":
            gin_lines.append(f"ImputationWrapper.lr = {value}")
        elif key == 'epochs':
            gin_lines.append(f"{model_name}.epochs = {value}")
        # Map model-specific parameters
        elif key in model_specific_params:
            gin_key = key.replace(f"{param_prefix}.", f"{param_prefix}.")
            gin_lines.append(f"{gin_key} = {value}")

    gin_lines.append("")
    return "\n".join(gin_lines)

# Function to process YAML files and generate corresponding .gin files
def process_yaml_files():
    for model_name in ["Attention", "SAITS"]:
        for missingness_type in ["MCAR", "MAR", "BO", "MNAR", "blockBO"]:
            for dataset_name in [
                "eicu", "exponential_decay", "global_local_seasonal", "linear", "linear_noisy",
                "seasonal", "seasonal_noisy", "sinusoids", "trend_sinusoids"
            ]:
                # Construct YAML file path
                yaml_path = os.path.join(base_path, model_name, missingness_type, dataset_name, "config.yaml")
                if not os.path.exists(yaml_path):
                    continue

                # Read YAML file
                with open(yaml_path, "r") as yaml_file:
                    yaml_content = yaml.safe_load(yaml_file)

                # Convert to .gin format
                gin_content = convert_yaml_to_gin(yaml_content, model_name, dataset_name)

                # Construct .gin file path
                gin_dir = os.path.join(base_path, model_name, missingness_type, dataset_name)
                os.makedirs(gin_dir, exist_ok=True)
                gin_path = os.path.join(gin_dir, "config.gin")

                # Write .gin file
                with open(gin_path, "w") as gin_file:
                    gin_file.write(gin_content)

                print(f"Converted {yaml_path} to {gin_path}")

# Execute the script
if __name__ == "__main__":
    process_yaml_files()

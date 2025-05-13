#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition='t4v2,t4v1,rtx6000,a40'
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --open-mode=truncate
#SBATCH --ntasks=1
#SBATCH --output=hyper_param_tuning/SAITS/BO/sinusoids_agent3.out

# Run the wandb agent
wandb agent addisonweatherhead/YAIB/m5lh5bet

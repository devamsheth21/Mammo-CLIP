#!/bin/bash
#SBATCH -p dgx
#SBATCH --account=dgxgrp2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --qos=high
#SBATCH --time=2:00:00

# Change the working directory
cd /mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/Mammo-CLIP/src/codebase/finetune/scripts

# Load the environment
conda activate mammoalbef

# Launch the finetuning script with accelerate
accelerate launch finetune-accelerate.py
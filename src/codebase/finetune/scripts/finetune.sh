#!/bin/bash
#SBATCH -p dgx
#SBATCH --account=dgxgrp2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --qos=high
#SBATCH --time=2:00:00
#SBATCH --pty bash
# Change the working directory
cd /mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/Mammo-CLIP/src/codebase/finetune/

# Load the environment
conda activate mammoalbef

# To specify cuda devices, SELECT devices which have full memory free : 
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune-accelerate.py

# Launch the finetuning script with accelerate
accelerate launch finetune-accelerate.py
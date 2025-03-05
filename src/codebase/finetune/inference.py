import h5py
import torch
from tqdm import tqdm
from accelerate import Accelerator
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
import yaml
import argparse
from model_utils import load_model
from datasets.data_utils import load_dataloader
from loss import contrastive_loss
from accelerate import Accelerator
import random
import numpy as np
from accelerate import init_empty_weights


def save_embeddings(model, dataloader, output_path, accelerator, exp_num):
    """Save embeddings in HDF5 format with proper distributed handling"""
    model.eval()
    embeddings = {"text_embeddings": [], "image_embeddings": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings", 
                         disable=not accelerator.is_local_main_process):
            outputs = model(batch)
            
            # Gather embeddings across all GPUs
            image_embeds = accelerator.gather(outputs["image_embeddings"].detach())
            text_embeds = accelerator.gather(outputs["text_embeddings"].detach())

            if accelerator.is_local_main_process:
                embeddings["image_embeddings"].append(image_embeds.cpu().float().numpy())
                embeddings["text_embeddings"].append(text_embeds.cpu().float().numpy())
        # Critical synchronization point before saving
        accelerator.wait_for_everyone()


    # Only main process saves the final file
    if accelerator.is_local_main_process:
        print("Last batch shape :" , len(batch['acc']))
        # Concatenate all batches
        image_features = np.concatenate(embeddings["image_embeddings"])
        text_features = np.concatenate(embeddings["text_embeddings"])
        print("Image embeddings shape:", image_features.shape)
        print("Text embeddings shape:", text_features.shape)
        # Save to HDF5 with metadata
        output_path = f"embeddings/embeddings_exp{exp_num}.h5"
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=image_features,
                            chunks=True, compression="gzip")
            f.create_dataset("txt_features", data=text_features,
                            chunks=True, compression="gzip")
            f.attrs["num_samples"] = len(image_features)
            f.attrs["embedding_dim"] = image_features.shape[1]

        print(f"Saved embeddings to {output_path} with shape {image_features.shape}")

def inference_pipeline(checkpoint_path, data_config, exp_num):
    accelerator = Accelerator()
    device = accelerator.device
    
    with open("configs/finetune-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Load checkpoint
    config['device'] = device
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    with init_empty_weights(): # For initializing skeleton Model
        model, tokenizer = load_model(config)
    model.load_state_dict(checkpoint["model"], assign=True)
    
    # Prepare model and dataloader
    
    # config['batch_size'] = 12
    dataloader = load_dataloader(config, tokenizer=tokenizer, split=None) # Implement your data loading
    model, dataloader = accelerator.prepare(model, dataloader)
    # Run embedding generation
    save_embeddings(model, dataloader, "embeddings.h5", accelerator, exp_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_num", type=int, default=1, help="Experiment number")
    parser.add_argument("--data_config", type=str, default="finetune-config.yaml", help="Path to the data configuration file")

    args = parser.parse_args()

    checkpoint_path = f"checkpoints/exp{args.exp_num}_best.pth"

    inference_pipeline(checkpoint_path, args.data_config, args.exp_num)

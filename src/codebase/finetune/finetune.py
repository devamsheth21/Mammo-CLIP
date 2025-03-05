import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from model_utils import load_model
from datasets.data_utils import load_dataloader
from loss import contrastive_loss
from accelerate import Accelerator
import random
import numpy as np

# Validation Function
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            loss = contrastive_loss(output["image_embeddings"], output["text_embeddings"], output["logit_scale"])
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training Function
def train(model, dataloader, val_dataloader, epochs, device, config, accelerator, writer, logger, optimizer, scheduler):
    model.train()
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        if config['selective_sampling']:
            # at every epoch, shuffle data with custom sampling function for medical data
            print(f"Shuffling training data for epoch {epoch}")
            dataloader.dataset.shuffle(bs=config['batch_size'],
                                       rare_grp_ratio=config['rare_grp_ratio'],
                                       batch_shuffle=config['batch_shuffle']
                                       )
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False,
                            disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            optimizer.zero_grad()
            with accelerator.accumulate(model):
                output = model(batch)
                loss = contrastive_loss(output["image_embeddings"], output["text_embeddings"], output["logit_scale"])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        loss_history.append(avg_loss)

        # Validation step
        avg_val_loss = validate(model, val_dataloader, device)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        val_loss_history.append(avg_val_loss)

        # Save checkpoint if val loss is less than previous best
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            checkpoint_path = f"checkpoints/exp{config['experiment']}_best_model.pth"
            accelerator.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, checkpoint_path)

    # Save final model
    accelerator.save(model.state_dict(), f"checkpoints/exp{config['experiment']}_last_model.pth")

    # Plot Loss Curve
    save_loss_curve(loss_history, val_loss_history, config['experiment'])

    print("Training Complete!")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_loss_curve(loss_history, val_loss_history, experiment):
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")  # Add validation loss curve
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/loss_curve_{experiment}.png")
    plt.close()

def main():
    # Load Config
    with open("configs/finetune-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Set Random Seed
    set_random_seed(config["seed"])
    # Setup Paths
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Logging
    logging.basicConfig(
        filename="logs/run.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # TensorBoard Writer
    writer = SummaryWriter(f"logs/tensorboard/exp{config['experiment']}")

    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16")
    device = accelerator.device

    #Cuda
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # Load Model & Data
    model, tokenizer = load_model(config)
    train_dataloader = load_dataloader(config, tokenizer, split="train")
    val_dataloader = load_dataloader(config, tokenizer, split="val")

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.image_model.parameters(), lr=float(config["learning_rate"]), weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Prepare everything with `accelerate`
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )


    train(model, train_dataloader, val_dataloader, config["epochs"], device, config, accelerator, writer, logger, optimizer, scheduler)


if __name__ == "__main__":
    main()

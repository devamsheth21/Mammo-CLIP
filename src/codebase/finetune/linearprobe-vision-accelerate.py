import torch
from torch.utils.data import DataLoader
from datasets.data_utils import load_dataloader
from torchvision import datasets, transforms
import os
import yaml
from accelerate import Accelerator
from model_utils import load_model
from mv_utils import set_random_seed, save_loss_curve
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from accelerate import init_empty_weights
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from accelerate import load_checkpoint_and_dispatch
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd


class VisionLinearProbe(nn.Module):
    def __init__(self, vision_model, num_classes):
        super(VisionLinearProbe, self).__init__()
        self.vision_model = vision_model
        self.linear = nn.Linear(vision_model.aggregator.projection_layer.out_features, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            images , view_seq = x['images'] , x['view_seqs']
            features = self.vision_model(images , view_seq)
        logits = self.linear(features)
        return logits

def save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, accelerator, is_best=False):
    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint = {
        "epoch": epoch,
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    filename = f"checkpoints/lp_exp{config['experiment']}/lp_exp{config['experiment']}_best.pth" if is_best else f"checkpoints/lp_exp{config['experiment']}/lp_exp{config['experiment']}_epoch{epoch}.pth"
    accelerator.save(checkpoint, filename)

def validate_and_calculate_accuracy(model, dataloader, accelerator, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            outputs = model(batch)
            loss = criterion(outputs, batch["birads"])
            total_loss += loss.detach().item()
            
            _, predicted = torch.max(outputs, 1)
            predicted, labels = accelerator.gather_for_metrics((predicted, batch["birads"]))
            total += labels.cpu().shape[0]
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_linear_probe(model, dataloader, val_dataloader, epochs, config, accelerator, optimizer, criterion, writer, logger):
    model.train()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    loss_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["birads"])
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.detach().item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        # Validation
        avg_val_loss , avg_val_acc = validate_and_calculate_accuracy(model, val_dataloader, accelerator , criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
            accelerator.print(avg_loss,avg_val_loss,avg_val_acc)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
            loss_history.append(avg_loss)
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(avg_val_acc)
            # writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

            if avg_val_loss < best_val_loss or avg_val_acc > best_val_acc:
                best_val_loss = min(avg_val_loss, best_val_loss)
                best_val_acc = max(avg_val_acc, best_val_acc)
                save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, accelerator, is_best=True)
                logger.info(f"New best model saved at epoch {epoch+1}")

        accelerator.wait_for_everyone()
        
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_checkpoint = {
            "model": unwrapped_model.state_dict(),
            "config": config,
            "epoch": epochs
        }
        accelerator.save(final_checkpoint, f"checkpoints/lp_exp{config['experiment']}/lp_exp{config['experiment']}_final.pth")

        save_loss_curve(loss_history, val_loss_history, f"lp_exp{config['experiment']}", acc=val_acc_history)

    accelerator.print("Training Complete!")
    
def main():
    # Load config
    with open("configs/linear-probe-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    config['device'] = device
    
    # Seed setup
    set_random_seed(config["seed"])
    
    if accelerator.is_local_main_process:
        os.makedirs(f"checkpoints/lp_exp{config['experiment']}", exist_ok=True)
        os.makedirs(f"results/lp_exp{config['experiment']}", exist_ok=True)
        writer = SummaryWriter(f"logs/tensorboard/lp_exp{config['experiment']}")
        logging.basicConfig(
            filename="logs/run.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger()
    else:
        writer = None
        logger = None

    checkpoint = torch.load(config["finetuned_weights_path"], map_location="cpu")
    with init_empty_weights(): # For initializing skeleton Model
        model, _ = load_model(config)
    model.load_state_dict(checkpoint["model"], assign=True)# Load finetuned model
    # model = load_checkpoint_and_dispatch(model, config["finetuned_weights_path"], device_map="auto")
    
    # Create VisionLinearProbe model
    vision_model = model.image_model
    for param in vision_model.parameters():
        param.requires_grad = False  # Freeze vision model weights
    linear_probe_model = VisionLinearProbe(vision_model, config["num_classes"])

    # linear_probe_model.to(device)
    
    # Prepare data loaders
    train_dataloader = load_dataloader(config, split="train")
    val_dataloader = load_dataloader(config, split="val")
    test_dataloader = load_dataloader(config, split="test")
    
    # Prepare optimizer and loss function
    optimizer = optim.Adam(linear_probe_model.linear.parameters(), lr=config["learning_rate"])

    # Instead of reading counts from config, get them from the dataset:
    if config["weighted_loss"]:
        class_counts = torch.tensor(train_dataloader.dataset.class_counts, dtype=torch.float)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # normalization (optional)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Prepare with Accelerator
    linear_probe_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        linear_probe_model, optimizer, train_dataloader, val_dataloader
    )
    
    # Train linear probe
    train_linear_probe(
        model=linear_probe_model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config["epochs"],
        config=config,
        accelerator=accelerator,
        optimizer=optimizer,
        criterion=criterion,
        logger=logger,
        writer=writer,
    )
    
    print("Linear probing complete!")

if __name__ == "__main__":
    main()






import torch
from torch.utils.data import DataLoader
from datasets.data_utils import load_dataloader
from torchvision import datasets, transforms
import os
import yaml
from model_utils import load_model
from mv_utils import set_random_seed, save_loss_curve
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import argparse

class VisionLinearProbe(nn.Module):
    def __init__(self, vision_model, num_classes):
        super(VisionLinearProbe, self).__init__()
        self.vision_model = vision_model
        self.linear = nn.Linear(vision_model.aggregator.projection_layer.out_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, device=None):
        with torch.no_grad():
            images, view_seq = x['images'], x['view_seqs']
            features = self.vision_model(images, view_seq)
        logits = self.linear(features)
        # probabilities = self.softmax(logits)
        return logits

def save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    filename = f"checkpoints/lp{config['classname']}_exp{config['experiment']}/lp{config['classname']}_exp{config['experiment']}_best.pth" if is_best else f"checkpoints/lp{config['classname']}_exp{config['experiment']}/lp{config['classname']}_exp{config['experiment']}_epoch{epoch}.pth"
    torch.save(checkpoint, filename)

def validate_and_calculate_accuracy(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(batch)
            loss = criterion(outputs, batch[config['classname']])
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += batch[config['classname']].size(0)
            correct += (predicted == batch[config['classname']]).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_linear_probe(model, dataloader, val_dataloader, epochs, config, optimizer, scheduler, criterion, writer, logger, device):
    model.train()
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    print("Trainable Layers:")
    for layer in trainable_layers:
        print(layer)
    print(f"Total trainable layers: {len(trainable_layers)}")
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
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(batch)
            loss = criterion(outputs, batch[config['classname']])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        # Validation
        avg_val_loss , avg_val_acc = validate_and_calculate_accuracy(model, val_dataloader, criterion, device, config)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
        loss_history.append(avg_loss)
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_acc)

        if avg_val_loss < best_val_loss: # or avg_val_acc > best_val_acc: # Save model only based on validation loss
            best_val_loss = min(avg_val_loss, best_val_loss)
            best_val_acc = max(avg_val_acc, best_val_acc)
            save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, is_best=True)
            logger.info(f"New best model saved at epoch {epoch+1}")
        # scheduler.step()
    final_checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "epoch": epochs
    }
    torch.save(final_checkpoint, f"checkpoints/lp{config['classname']}_exp{config['experiment']}/lp{config['classname']}_exp{config['experiment']}_final.pth")

    save_loss_curve(loss_history, val_loss_history, f"lp{config['classname']}_exp{config['experiment']}")

    print("Training Complete!")




def main(classname):
    # Load config
    with open("configs/linear-probe-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    config['classname'] = classname
    print(f"Device: {device}")
    # Seed setup
    set_random_seed(config["seed"])
    
    os.makedirs(f"checkpoints/lp{config['classname']}_exp{config['experiment']}", exist_ok=True)
    os.makedirs(f"results/lp{config['classname']}_exp{config['experiment']}", exist_ok=True)
    writer = SummaryWriter(f"logs/tensorboard/lp{config['classname']}_exp{config['experiment']}")
    logging.basicConfig(
        filename="logs/run.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    checkpoint = torch.load(config["finetuned_weights_path"], map_location="cpu")
    model, _ = load_model(config)
    model.load_state_dict(checkpoint["model"], strict=False)
        

    # Create VisionLinearProbe model
    vision_model = model.image_model
    for param in vision_model.parameters():
        param.requires_grad = False  # Freeze vision model weights
    vision_model.eval()
    linear_probe_model = VisionLinearProbe(vision_model, config["num_classes"]).to(device)
    
    # Prepare data loaders
    train_dataloader = load_dataloader(config, split="train")
    val_dataloader = load_dataloader(config, split="val")
    test_dataloader = load_dataloader(config, split="test")
    
    # Prepare optimizer and loss function
    optimizer = optim.Adam(linear_probe_model.linear.parameters(), lr=config["learning_rate"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Instead of reading counts from config, get them from the dataset:
    if config["weighted_loss"]:
        print(train_dataloader.dataset.class_counts, classname , train_dataloader.dataset.class_counts[classname])
        class_counts = torch.tensor(train_dataloader.dataset.class_counts[classname], dtype=torch.float)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # normalization (optional)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Train linear probe
    train_linear_probe(
        model=linear_probe_model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config["epochs"],
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        logger=logger,
        writer=writer,
        device=device,
    )

    
    print("Linear probing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="birads", help="Classification for Birads or density")
    args = parser.parse_args()
    main(args.class_name)

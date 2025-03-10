import torch
from torch.utils.data import DataLoader
from datasets.data_utils import load_dataloader
from torchvision import datasets, transforms
import os
import yaml
from model_utils import load_model
from mv_utils import set_random_seed, save_loss_curve
import argparse
from typing import Union
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

def save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, loss_history, val_loss_history, val_acc_history, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
    }
    filename = f"checkpoints/lp{config['classname']}_exp{config['experiment']}/lp{config['classname']}_exp{config['experiment']}_best.pth" if is_best else f"checkpoints/lp{config['classname']}_exp{config['experiment']}/lp{config['classname']}_exp{config['experiment']}_epoch{epoch}.pth"
    print(f"Saving checkpoint at epoch {epoch}... at : {filename}")
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

def train_linear_probe(model, dataloader, val_dataloader, epochs, config, optimizer, scheduler, criterion, writer, logger, device,  start_epoch=0, best_val_loss=float('inf'), best_val_acc=0.0):
    model.train()
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    print("Trainable Layers:")
    for layer in trainable_layers:
        print(layer)
    print(f"Total trainable layers: {len(trainable_layers)}")
    print(f"Linear layer output features: {model.linear.out_features}")
    loss_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(start_epoch, epochs):
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
            save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, loss_history, val_loss_history, val_acc_history, is_best=True)
            logger.info(f"New best model saved at epoch {epoch+1}")
        # scheduler.step()
    save_checkpoint(model, optimizer, epoch, best_val_loss, best_val_acc, config, loss_history, val_loss_history, val_acc_history, is_best=False)
    save_loss_curve(loss_history, val_loss_history, f"lp{config['classname']}_exp{config['experiment']}")

    print("Training Complete!")




def main(classname, checkpoint_path=None, args=None):
    # Load config
    with open("configs/linear-probe-config.yaml", "r") as f:
        config = yaml.safe_load(f)
     # Update config with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            print(f"{key} chnaging to {value}")
            config[key] = value
    
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
    # Log the configuration file as a text summary
    writer.add_text('config', str(config))

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
    print("Loading data loaders...")
    train_dataloader = load_dataloader(config, split="train")
    print("----Train dataloader loaded----")
    val_dataloader = load_dataloader(config, split="val")
    print("----Val dataloader loaded----")
    test_dataloader = load_dataloader(config, split="test")
    print("----Test dataloader loaded----")
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
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        linear_probe_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resuming training from epoch {start_epoch}")
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
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )

    
    print("Linear probing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to resume training")
    parser.add_argument("--classname", type=str, default="birads", help="Classification for Birads or density")
    parser.add_argument("--config_path", type=str, default="configs/linear-probe-config.yaml", help="Path to the config file")
    parser.add_argument("--experiment", type=str, help="Experiment number")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes")
    # parser.add_argument("--finetuned_weights_path", type=str, help="Path to the finetuned weights")
    parser.add_argument("--epochs", type=int,default=None,help="Total epochs")
    parser.add_argument("--test", action="store_true", help="Flag for testing")


    args = parser.parse_args()
    if not args.test:
        args.test = None
    main(args.classname, args.checkpoint_path, args)

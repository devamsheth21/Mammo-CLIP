import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import yaml
from accelerate import Accelerator
from model_utils import load_model
from utils import set_random_seed

class VisionLinearProbe(nn.Module):
    def __init__(self, vision_model, num_classes):
        super(VisionLinearProbe, self).__init__()
        self.vision_model = vision_model
        self.linear = nn.Linear(vision_model.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.vision_model(x)
        logits = self.linear(features)
        return logits

def load_dataloader(config, split="train"):
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"])
    ])
    dataset = datasets.ImageFolder(root=config[f"{split}_data_path"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    return dataloader

def train_linear_probe(model, dataloader, val_dataloader, epochs, config, accelerator, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress_bar:
            images, labels = batch
            images, labels = images.to(accelerator.device), labels.to(accelerator.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images, labels = batch
                images, labels = images.to(accelerator.device), labels.to(accelerator.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        model.train()

def main():
    # Load config
    with open("linear-probe-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    config['device'] = device
    
    # Seed setup
    set_random_seed(config["seed"])
    
    # Load finetuned model
    model, _ = load_model(config)
    checkpoint = torch.load(config["finetuned_weights_path"], map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # Create VisionLinearProbe model
    vision_model = model.image_model
    for param in vision_model.parameters():
        param.requires_grad = False  # Freeze vision model weights
    linear_probe_model = VisionLinearProbe(vision_model, config["num_classes"])
    linear_probe_model.to(device)
    
    # Prepare data loaders
    train_dataloader = load_dataloader(config, split="train")
    val_dataloader = load_dataloader(config, split="val")
    
    # Prepare optimizer and loss function
    optimizer = optim.Adam(linear_probe_model.linear.parameters(), lr=config["learning_rate"])
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
        criterion=criterion
    )
    
    print("Linear probing complete!")

if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import h5py
import json
from eval.recallevalv2 import Evaluation
import sys
sys.path.append('..')
from breastclip.scheduler import build_scheduler

def evaluate(model, dataloader, accelerator, exp_num):
    """ Evaluate and Save embeddings for experiment in HDF5 format with proper distributed handling"""
    model.eval()
    dataset_length = len(dataloader.dataset) 
    embeddings = {"text_embeddings": [], "image_embeddings": [], "accession_numbers": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings", 
                         disable=not accelerator.is_local_main_process):
            outputs = model(batch)
            
            # Gather embeddings across all GPUs
            image_embeds = accelerator.gather(outputs["image_embeddings"].detach())
            text_embeds = accelerator.gather(outputs["text_embeddings"].detach())
            acc_numbers = batch["acc"]  # List of strings
            if accelerator.use_distributed:
                # Gather strings across processes
                acc_numbers = accelerator.gather_for_metrics(acc_numbers)

            if accelerator.is_local_main_process:
                embeddings["image_embeddings"].append(image_embeds.cpu().float().numpy())
                embeddings["text_embeddings"].append(text_embeds.cpu().float().numpy())
                embeddings["accession_numbers"].extend(acc_numbers)  # Append strings directly


    # Only main process saves the final file
    if accelerator.is_local_main_process:
        # Concatenate all batches
        image_features = np.concatenate(embeddings["image_embeddings"])[0:dataset_length]
        text_features = np.concatenate(embeddings["text_embeddings"])[0:dataset_length]
        accession_numbers = embeddings["accession_numbers"]
        
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Text Features Shape: {text_features.shape}")
        print(f"Accession Numbers Shape: {len(accession_numbers)}")

        # Save to HDF5 with metadata
        output_path = f"embeddings/embeddings_exp{exp_num}.h5"
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=image_features,
                            chunks=True, compression="gzip")
            f.create_dataset("txt_features", data=text_features,
                            chunks=True, compression="gzip")
            f.create_dataset("accession_numbers", data=accession_numbers,
                            chunks=True, compression="gzip")
            f.attrs["num_samples"] = len(image_features)
            f.attrs["embedding_dim"] = image_features.shape[1]
        
        print(f"Saved embeddings to {output_path} with shape {image_features.shape}")
        # output_path = os.path.abspath(output_path)
        # # Evaluate
        # evaluator = Evaluation(modelname='multiview', config=config, exp_num=exp_num, embeddings_path=output_path)
        # results_df = evaluator.run_eval()
        # print(results_df)


def validate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validating", leave=False, 
                     disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            output = model(batch)
            loss = contrastive_loss(output["image_embeddings"], 
                                  output["text_embeddings"], 
                                  output["logit_scale"])
        total_loss += accelerator.gather(loss).sum().item()
    avg_loss = total_loss / (len(dataloader) * accelerator.num_processes)
    return avg_loss

def train(model, dataloader, val_dataloader, epochs, config, accelerator, writer, logger, optimizer, scheduler):
    model.train()
    best_val_loss = float('inf')
    loss_history = []
    val_loss_history = []
    # # For gradient clipping
    max_grad_norm = config.get('max_grad_norm', 1.0)

    for epoch in range(epochs):
        # if config['selective_sampling'] and accelerator.is_local_main_process:
        #     dataloader.dataset.shuffle(
        #         bs=config['batch_size'],
        #         rare_grp_ratio=config['rare_grp_ratio'],
        #         batch_shuffle=config['batch_shuffle']
        #     )
            # accelerator.wait_for_everyone()

        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", 
                           leave=False, disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(batch)
                loss = contrastive_loss(output["image_embeddings"], 
                                       output["text_embeddings"], 
                                       output["logit_scale"])
                accelerator.backward(loss)
                
                # if max_grad_norm > 0:
                #     if accelerator.sync_gradients:  # Critical check
                #         # Required for AMP compatibility
                #         accelerator.unscale_gradients(optimizer)
                #         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
                
                optimizer.step()

            gathered_loss = accelerator.gather(loss.detach())
            total_loss += gathered_loss.sum().item()
            # Progress bar shows mean loss across devices
            progress_bar.set_postfix(loss=gathered_loss.mean().item())



        # Step scheduler after epoch
        scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / (len(dataloader) * accelerator.num_processes)
        avg_val_loss = validate(model, val_dataloader, accelerator)

        # Logging (only on main process)
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            loss_history.append(avg_loss)
            val_loss_history.append(avg_val_loss)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

            # Save checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    "epoch": epoch,
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                }
                accelerator.save(checkpoint, f"checkpoints/exp{config['experiment']}_best.pth")
                logger.info(f"New best model saved at epoch {epoch+1}")

        accelerator.wait_for_everyone()

    # Final save (main process only)
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_checkpoint = {
            "model": unwrapped_model.state_dict(),
            "config": config,
            "epoch": epochs
        }
        accelerator.save(final_checkpoint, f"checkpoints/exp{config['experiment']}_final.pth")
        save_loss_curve(loss_history, val_loss_history, config['experiment'])

    accelerator.print("Training Complete!")


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
    plt.savefig(f"results/exp{config['experiment']}/loss_curve_{experiment}.png")
    plt.close()


def main():
    # Config loading
    with open("configs/finetune-config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize accelerator first
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("grad_accum_steps", 2),
        mixed_precision=config.get("mixed_precision", "fp16")
    )
    device = accelerator.device
    config['device'] = device
    # Seed setup
    set_random_seed(config["seed"])
    
    # Setup logging (main process only)
    if accelerator.is_local_main_process:
        os.makedirs(f"checkpoints/exp{config['experiment']}", exist_ok=True)
        os.makedirs(f"results/exp{config['experiment']}", exist_ok=True)
        writer = SummaryWriter(f"logs/tensorboard/exp{config['experiment']}")
        logging.basicConfig(
            filename="logs/run.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger()
    else:
        writer = None
        logger = None
    
    # Model and Data
    with accelerator.main_process_first():
        #only load once
        model, tokenizer = load_model(config)
    train_dataloader = load_dataloader(config, tokenizer=tokenizer, split="train")
    val_dataloader = load_dataloader(config, tokenizer=tokenizer, split="val")
    accelerator.print(f"====Data Loader length {len(train_dataloader.dataset)}=====")
    # Optimizer and Scheduler
    optimizer = AdamW(model.image_model.parameters(),  # Train image model
                     lr=float(config["learning_rate"]), 
                     weight_decay=config["weight_decay"])
    lr_config = config["lr_config"] 
    warmup_epochs = lr_config['config']['warmup_epochs']
    total_epochs = lr_config['config']['total_epochs']
    warmup_steps = warmup_epochs
    total_steps = total_epochs

    # Update lr_config with the calculated steps
    lr_config['config'].update({
        'total_steps': total_steps,
        'warmup_steps': warmup_steps
    })
    print("--Lr config-- : ", lr_config)
    scheduler = build_scheduler(optimizer, lr_config)
    # scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Prepare with Accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    train(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config["lr_config"]["config"]["total_epochs"],
        config=config,
        accelerator=accelerator,
        writer=writer,
        logger=logger,
        optimizer=optimizer,
        scheduler=scheduler
    )

    test_dataloader = load_dataloader(config, tokenizer, split=None)
    accelerator.print(f"====Data Loader length {len(test_dataloader.dataset)}=====")
    test_dataloader = accelerator.prepare(test_dataloader)
    evaluate(model=model, dataloader=test_dataloader, accelerator=accelerator, exp_num=config['experiment'])
    if accelerator.is_local_main_process:
        writer.close()
if __name__ == "__main__":
    main()
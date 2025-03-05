import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mv_utils import evaluate_model
from datasets.data_utils import load_dataloader
from model_utils import load_model
import yaml
from mv_models import VisionLinearProbe
import argparse

def test_linear_probe(model, dataloader, accelerator = None):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing", leave=False)
        for batch in progress_bar:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            if accelerator is not None:
                predicted, labels = accelerator.gather_for_metrics((predicted, batch["birads"]))
            else:
                labels = batch["birads"]
            total += labels.cpu().shape[0]
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            progress_bar.set_postfix(acc=correct / total)
    accuracy = 100 * correct / total

    return accuracy, predictions, targets

def main(classname):
    # Load config
    with open("configs/linear-probe-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print(f"Device: {device}")
    if classname :
        config['classname'] = classname
    # Load model
    checkpoint = torch.load(config["finetuned_weights_path"], map_location="cpu")
    model, _ = load_model(config)
    model.load_state_dict(checkpoint["model"], strict=False)
    
    # Create VisionLinearProbe model
    vision_model = model.image_model
    for param in vision_model.parameters():
        param.requires_grad = False  # Freeze vision model weights
    linear_probe_model = VisionLinearProbe(vision_model, config["num_classes"]).to(device)

    # Load the best checkpoint
    best_checkpoint_path = f"checkpoints/lp{config.get('classname','')}_exp{config['experiment']}/lp{config.get('classname','')}_exp{config['experiment']}_best.pth"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    linear_probe_model.load_state_dict(best_checkpoint["model"])
    
    # Prepare test dataloader
    test_dataloader = load_dataloader(config, split="test")
    experiment=config['experiment']
    # Evaluate the model
    accuracy, predictions, targets = test_linear_probe(linear_probe_model, test_dataloader, accelerator=None)
    evaluate_model(predictions, targets, f"lp_exp{experiment}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(classification_report(targets, predictions))
    print(f"ROC AUC Score: {roc_auc_score(targets, predictions):.4f}")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Mammo-CLIP Linear Probe Evaluation")
    parser.add_argument("--classname", type=str, default=None, help="Class name for the linear probe")
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    classname = args.classname
    main(classname)
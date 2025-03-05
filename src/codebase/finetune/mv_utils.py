import os
import numpy as np
import pandas as pd
import random
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os
import pandas as pd
import matplotlib.pyplot as plt
def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_loss_curve(loss_history, val_loss_history, experiment, **kwargs):
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")  # Add validation loss curve
    if 'acc' in kwargs:
        acc = kwargs['acc']
        if isinstance(acc, list):
            plt.plot(acc, label="Accuracy")  # Add accuracy curve
        else:
            print("Invalid format for 'acc'. Please pass a list of values.")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/{experiment}/loss_curve_{experiment}.png")
    plt.close()


def plot_roc_curve(targets, predictions, experiment):
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{experiment}/auroc_{experiment}.png")
    plt.close()




def evaluate_model(predictions, targets, experiment):
    """
    Evaluate the performance of a classification model.

    Args:
    - predictions (list): Model predictions.
    - targets (list): True labels.
    - experiment (str): Name of the experiment.

    Returns:
    - None
    """

    # Create the results directory if it doesn't exist
    results_dir = f"results/{experiment}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Calculate the classification report
    report = classification_report(targets, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{results_dir}/classification_report.csv", index=True)

    # Plot the confusion matrix
    matrix = confusion_matrix(targets, predictions)
    plt.imshow(matrix, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.clf()

    # Calculate the AuROC
    try:
        auc = roc_auc_score(targets, predictions)
        print(f"AuROC for {experiment}: {auc}")
    except ValueError:
        print(f"Cannot calculate AUROC for {experiment} due to single class prediction.")

    # Plot the AuROC curve
    try:
        fpr, tpr, _ = roc_curve(targets, predictions)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("AuROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(f"{results_dir}/auroc.png")
        plt.clf()
    except ValueError:
        print(f"Cannot plot AUROC curve for {experiment} due to single class prediction.")

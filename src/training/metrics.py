import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config.config import CONFIG, LABELS, logger
from src.training.logits import extract_last_token_logits


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Calculates the classification accuracy across a dataset.

    Args:
        data_loader (DataLoader): The data loader providing the batches.
        model (nn.Module): The GPT-2 classification model.
        device (torch.device): The hardware device.
        num_batches (int, optional): The maximum number of batches to evaluate. Defaults to all batches.

    Returns:
        float: The accuracy as a ratio of correct predictions to total examples.
    """
    was_training = model.training
    model.eval()
    correct_predictions, num_examples = 0, 0

    if len(data_loader) == 0:
        logger.warning("calc_accuracy_loader called with an empty DataLoader; returning nan")
        if was_training:
            model.train()
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch, attention_mask) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(input_batch, attention_mask=attention_mask)
                last_logits = extract_last_token_logits(logits, attention_mask)
                predicted_labels = torch.argmax(last_logits, dim=-1)

                correct_predictions += (predicted_labels == target_batch).sum().item()
                num_examples += predicted_labels.shape[0]
            else:
                break

    if num_examples == 0:
        if was_training:
            model.train()
        return float("nan")

    if was_training:
        model.train()
    return correct_predictions / num_examples


def save_loss_plot(train_losses, val_losses, examples_seen, save_path=None):
    """Save training and validation loss curves from intermediate evaluation steps.

    Args:
        train_losses (list[float]): Training loss at each evaluation step.
        val_losses (list[float]): Validation loss at each evaluation step.
        examples_seen (int): Total training examples processed when training finished.
        save_path (str, optional): Output path. Defaults to CONFIG loss plot path.
    """
    if not train_losses:
        logger.warning("No intermediate loss values recorded; skipping loss plot.")
        return

    if save_path is None:
        save_path = CONFIG["paths"]["loss_plot_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(train_losses)
    steps_tensor = list(range(1, n + 1))
    examples_seen_tensor = [
        examples_seen * (step - 1) / max(n - 1, 1) for step in steps_tensor
    ]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps_tensor, train_losses, label="Training loss")
    ax1.plot(steps_tensor, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Evaluation step")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen_tensor, train_losses, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss plot saved to {save_path}")


def save_accuracy_plot(train_accs, val_accs, save_path=None):
    """Save a line plot of per-epoch training and validation accuracy.

    Args:
        train_accs (list[float]): Training accuracy values per epoch.
        val_accs (list[float]): Validation accuracy values per epoch.
        save_path (str, optional): Output path. Defaults to CONFIG accuracy plot path.
    """
    if save_path is None:
        save_path = CONFIG["paths"]["accuracy_plot_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_accs, marker="o", label="Training accuracy (est.)")
    plt.plot(epochs, val_accs, marker="o", label="Validation accuracy (full)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Accuracy plot saved to {save_path}")


def generate_classification_metrics(data_loader, model, device):
    """Generates and logs the classification report and confusion matrix.

    Evaluates the full dataset to compute detailed precision, recall, and F1 scores
    for each class, and saves a visual confusion matrix to the plots directory.

    Args:
        data_loader (DataLoader): The data loader containing the evaluation dataset.
        model (nn.Module): The trained classification model.
        device (torch.device): The hardware device.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input_batch, target_batch, attention_mask in data_loader:
            input_batch = input_batch.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_batch, attention_mask=attention_mask)
            last_logits = extract_last_token_logits(logits, attention_mask)
            preds = torch.argmax(last_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_batch.cpu().numpy())

    target_names = [LABELS[i] for i in range(len(LABELS))]

    logger.info("Classification Report:")
    report = classification_report(all_targets, all_preds, target_names=target_names)
    logger.info("\n%s", report)

    cm = confusion_matrix(all_targets, all_preds)
    save_path = CONFIG["paths"]["confusion_matrix_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

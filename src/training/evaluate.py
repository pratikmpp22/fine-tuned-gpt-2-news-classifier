import torch

from src.config.config import logger
from src.training.logits import extract_last_token_logits


def calc_loss_batch(input_batch, target_batch, attention_mask, model, device):
    """Calculates the cross-entropy loss for a single batch.

    Args:
        input_batch (torch.Tensor): The padded sequence of token IDs.
        target_batch (torch.Tensor): The target classification labels.
        attention_mask (torch.Tensor): The mask identifying real tokens vs padding.
        model (nn.Module): The GPT-2 classification model.
        device (torch.device): The hardware device.

    Returns:
        tuple: The scalar loss tensor and the batch logits.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_batch, attention_mask=attention_mask)
    last_logits = extract_last_token_logits(logits, attention_mask)

    loss = torch.nn.functional.cross_entropy(last_logits, target_batch)
    return loss, last_logits


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculates the average cross-entropy loss across multiple batches.

    Args:
        data_loader (DataLoader): The data loader providing the batches.
        model (nn.Module): The GPT-2 classification model.
        device (torch.device): The hardware device.
        num_batches (int, optional): The maximum number of batches to evaluate. Defaults to all batches.

    Returns:
        float: The average loss across the evaluated batches.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        logger.warning("calc_loss_loader called with an empty DataLoader; returning nan")
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch, attention_mask) in enumerate(data_loader):
        if i < num_batches:
            loss, _ = calc_loss_batch(
                input_batch, target_batch, attention_mask, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluates the model on both the training and validation sets.

    Args:
        model (nn.Module): The model to evaluate.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The hardware device.
        eval_iter (int): The number of batches to evaluate on the training set.

    Returns:
        tuple: The training loss and validation loss.
    """
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=None)

    model.train()
    return train_loss, val_loss

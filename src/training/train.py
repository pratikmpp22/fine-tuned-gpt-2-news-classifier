import json
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

from src.config.config import CONFIG, logger
from src.training.evaluate import calc_loss_batch, calc_loss_loader, evaluate_model
from src.training.metrics import calc_accuracy_loader

try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    xm = None
    _XLA_AVAILABLE = False


def _uses_xla(device):
    return _XLA_AVAILABLE and getattr(device, "type", str(device)) == "xla"


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, save_best=True):
    """Trains the GPT-2 based classifier.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device (CPU, GPU, or TPU) to run the training on.
        num_epochs (int): Number of complete passes through the dataset.
        eval_freq (int): The number of steps between intermediate validation evaluations.
        eval_iter (int): The number of batches to use for intermediate loss estimation.
        save_best (bool, optional): If True, saves the model checkpoint with the lowest validation loss. Defaults to True.

    Returns:
        tuple: Lists containing train_losses, val_losses, train_accs, val_accs, and the total examples_seen.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, 0

    best_val_loss = float("inf")
    checkpoint_saved = False

    logger.info("Computing initial pre-training loss...")
    with torch.no_grad():
        initial_train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        initial_val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    logger.info(
        f"Initial Train loss: {initial_train_loss:.3f}, Val loss: {initial_val_loss:.3f}"
    )

    start_time = time.time()
    train_acc_fraction = CONFIG["training"]["train_acc_batch_fraction"]
    train_batches_estimate = max(1, int(len(train_loader) * train_acc_fraction))

    for epoch in range(num_epochs):
        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
        )

        for input_batch, target_batch, attention_mask in progress_bar:
            optimizer.zero_grad()
            loss, _ = calc_loss_batch(
                input_batch, target_batch, attention_mask, model, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=CONFIG["training"]["max_grad_norm"],
            )

            if _uses_xla(device):
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(model, val_loss, epoch, global_step)
                    checkpoint_saved = True
                    logger.info(
                        f"  -> New best val loss: {val_loss:.3f}. Checkpoint saved."
                    )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=train_batches_estimate
        )
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=None)
        logger.info(
            f"Training accuracy (est.): {train_accuracy*100:.2f}% | "
            f"Validation accuracy (full): {val_accuracy*100:.2f}%"
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    if save_best and not checkpoint_saved:
        fallback_val_loss = val_losses[-1] if val_losses else initial_val_loss
        logger.warning(
            "No validation-loss improvement during training; saving final model state."
        )
        _save_checkpoint(model, fallback_val_loss, num_epochs - 1, global_step)
        checkpoint_saved = True

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    _save_training_metrics(train_losses, val_losses, train_accs, val_accs, examples_seen)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def _save_checkpoint(model, val_loss, epoch, global_step):
    """Saves the model weights and associated training metadata to disk."""
    save_path = CONFIG["paths"]["model_save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    metadata = {
        "gpt_version": "GPT-2 124M",
        "training_date": datetime.now().isoformat(),
        "best_val_loss": val_loss,
        "epoch": epoch + 1,
        "global_step": global_step,
        "config": CONFIG,
    }
    metadata_path = CONFIG["paths"]["metadata_save_path"]
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def _save_training_metrics(train_losses, val_losses, train_accs, val_accs, examples_seen):
    """Saves the recorded training metrics (losses and accuracies) as a JSON file."""
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "examples_seen": examples_seen,
    }
    metrics_path = CONFIG["paths"]["metrics_save_path"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

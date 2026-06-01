import torch
from torch.utils.data import DataLoader

from src.config.config import CONFIG


def custom_collate_fn(batch):
    """Dynamically pads a batch of sequences to the longest sequence in the batch.

    Args:
        batch (list): A list of tuples containing (input_tensor, label_tensor).

    Returns:
        tuple: Padded input tensors, target tensors, and attention masks.

    Raises:
        ValueError: If the batch is empty.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    pad_token_id = CONFIG["data"]["pad_token_id"]
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_len = max(len(x) for x in inputs)

    padded_inputs = []
    attention_masks = []

    for x in inputs:
        pad_len = max_len - len(x)
        padded_x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_inputs.append(padded_x)

        mask = torch.cat([
            torch.ones(len(x), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])
        attention_masks.append(mask)

    return (
        torch.stack(padded_inputs),
        torch.stack(targets),
        torch.stack(attention_masks),
    )


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Creates PyTorch DataLoaders for all dataset splits.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=CONFIG["training"]["num_workers"],
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        num_workers=CONFIG["training"]["num_workers"],
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        num_workers=CONFIG["training"]["num_workers"],
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    batch_size = CONFIG["training"]["batch_size"]
    if len(train_loader) == 0:
        raise ValueError(
            f"Training DataLoader is empty: dataset has {len(train_dataset)} samples "
            f"but batch_size={batch_size} with drop_last=True requires at least "
            f"{batch_size} samples."
        )

    return train_loader, val_loader, test_loader

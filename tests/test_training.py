import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.config import CONFIG
from src.training.evaluate import calc_loss_batch, calc_loss_loader
from src.training.logits import extract_last_token_logits


class _EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def test_calc_loss_batch(small_model):
    inputs = torch.randint(0, 50257, (4, 10))
    targets = torch.randint(0, CONFIG["data"]["num_classes"], (4,))
    masks = torch.ones(4, 10, dtype=torch.long)
    loss, logits = calc_loss_batch(inputs, targets, masks, small_model, device="cpu")
    assert loss.item() > 0
    assert logits.shape == (4, CONFIG["data"]["num_classes"])


def test_extract_last_token_logits_rejects_all_padding():
    logits = torch.randn(2, 5, CONFIG["data"]["num_classes"])
    attention_mask = torch.zeros(2, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="at least one real token"):
        extract_last_token_logits(logits, attention_mask)


def test_calc_loss_loader_empty_warns_and_returns_nan(small_model, caplog):
    empty_loader = DataLoader(_EmptyDataset())
    result = calc_loss_loader(empty_loader, small_model, device="cpu")
    assert result != result  # nan
    assert "empty DataLoader" in caplog.text

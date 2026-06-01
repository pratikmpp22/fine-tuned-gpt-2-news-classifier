import torch

from src.config.config import CONFIG
from src.utils.seed import set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_default_matches_config():
    set_seed()
    first = torch.randn(3)
    set_seed(CONFIG["training"]["seed"])
    second = torch.randn(3)
    assert torch.equal(first, second)

import pytest
import torch

from src.config.config import CONFIG
from src.data.dataset import NewsDataset
from src.data.dataloader import custom_collate_fn


def test_dataset_length(sample_df, tokenizer):
    ds = NewsDataset(data=sample_df, tokenizer=tokenizer, max_length=50)
    assert len(ds) == 3


def test_dataset_item_types(sample_df, tokenizer):
    ds = NewsDataset(data=sample_df, tokenizer=tokenizer, max_length=50)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_truncation(sample_df, tokenizer):
    ds = NewsDataset(data=sample_df, tokenizer=tokenizer, max_length=5)
    x, y = ds[0]
    assert len(x) <= 5


def test_max_length_capped_at_context_length(sample_df, tokenizer):
    ds = NewsDataset(data=sample_df, tokenizer=tokenizer, max_length=2000)
    assert ds.max_length <= CONFIG["gpt_model"]["context_length"]


def test_no_global_padding(sample_df, tokenizer):
    """Sequences are not padded to a global max length in the dataset."""
    ds = NewsDataset(data=sample_df, tokenizer=tokenizer, max_length=50)
    lengths = [len(ds[i][0]) for i in range(len(ds))]
    assert len(set(lengths)) > 1


def test_invalid_data_type(tokenizer):
    with pytest.raises(TypeError):
        NewsDataset(data="not a dataframe", tokenizer=tokenizer)


def test_invalid_label_in_dataset(sample_df, tokenizer):
    bad_df = sample_df.copy()
    bad_df.loc[0, "label"] = 99
    ds = NewsDataset(data=bad_df, tokenizer=tokenizer, max_length=50)
    with pytest.raises(ValueError, match="Invalid label"):
        _ = ds[0]


@pytest.fixture
def sample_batch():
    return [
        (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor(0, dtype=torch.long)),
        (torch.tensor([4, 5], dtype=torch.long), torch.tensor(1, dtype=torch.long)),
        (torch.tensor([6, 7, 8, 9], dtype=torch.long), torch.tensor(2, dtype=torch.long)),
    ]


def test_collate_pads_to_max_in_batch(sample_batch):
    inputs, targets, masks = custom_collate_fn(sample_batch)
    assert inputs.shape == (3, 4)
    assert targets.shape == (3,)
    assert masks.shape == (3, 4)


def test_attention_mask_correctness(sample_batch):
    inputs, targets, masks = custom_collate_fn(sample_batch)
    assert masks[0].tolist() == [1, 1, 1, 0]
    assert masks[1].tolist() == [1, 1, 0, 0]
    assert masks[2].tolist() == [1, 1, 1, 1]


def test_collate_empty_batch_raises():
    with pytest.raises(ValueError, match="Cannot collate an empty batch"):
        custom_collate_fn([])

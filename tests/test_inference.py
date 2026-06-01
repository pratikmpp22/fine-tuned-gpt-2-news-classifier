import pytest
import torch

from src.config.config import CONFIG, LABELS
from src.inference.predict import classify_news
from src.models.gpt2_classifier import GPTModel


def test_classify_news_returns_valid_label(small_model, tokenizer):
    result = classify_news(
        "Breaking news about the stock market today",
        small_model, tokenizer, device="cpu", max_length=32,
    )
    assert result in LABELS.values()


def test_classify_news_default_max_length(small_model, tokenizer):
    result = classify_news(
        "A short test",
        small_model, tokenizer, device="cpu",
    )
    assert result in LABELS.values()


def test_classify_news_invalid_max_length(small_model, tokenizer):
    with pytest.raises(ValueError, match="max_length must be strictly positive"):
        classify_news("test", small_model, tokenizer, device="cpu", max_length=-1)


def test_classify_news_empty_text(small_model, tokenizer):
    with pytest.raises(ValueError, match="text must contain at least one"):
        classify_news("   ", small_model, tokenizer, device="cpu", max_length=32)


def test_classify_news_requires_classification_head(small_config, tokenizer):
    model = GPTModel(small_config)
    with pytest.raises(ValueError, match="replace_classification_head"):
        classify_news("hello", model, tokenizer, device="cpu", max_length=16)

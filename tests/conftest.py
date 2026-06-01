import os
import sys

# Ensure the project root is on the path before any src/ imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import pytest
import torch

from src.config.config import CONFIG
from src.models.gpt2_classifier import GPTModel, replace_classification_head


@pytest.fixture
def small_config():
    gpt_cfg = CONFIG["gpt_model"]
    return {
        "vocab_size": 50257,
        "context_length": 64,
        "emb_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "layer_norm_eps": gpt_cfg["layer_norm_eps"],
        "ffn_multiplier": gpt_cfg["ffn_multiplier"],
    }


@pytest.fixture
def small_model(small_config):
    model = GPTModel(small_config)
    replace_classification_head(model, CONFIG["data"]["num_classes"])
    return model


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": [
            "This is a short news article about sports.",
            "Another article about business and finance topics.",
            "Science and technology advances in AI.",
        ],
        "label": [1, 2, 3],
    })


@pytest.fixture
def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")

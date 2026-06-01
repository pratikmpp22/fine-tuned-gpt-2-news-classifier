import torch

from src.config.config import CONFIG
from src.models.attention import MultiHeadAttention
from src.models.gpt2_classifier import (
    GPTModel,
    configure_transfer_learning,
    replace_classification_head,
)
from src.models.layers import FeedForward, LayerNorm
from src.models.transformer_block import TransformerBlock


def test_gpt_model_forward_shape(small_config):
    model = GPTModel(small_config)
    x = torch.randint(0, 50257, (2, 10))
    out = model(x)
    assert out.shape == (2, 10, small_config["vocab_size"])


def test_gpt_model_classification_head_shape(small_config):
    model = GPTModel(small_config)
    replace_classification_head(model)
    x = torch.randint(0, 50257, (2, 10))
    out = model(x)
    assert out.shape == (2, 10, CONFIG["data"]["num_classes"])


def test_gpt_model_with_attention_mask(small_config):
    model = GPTModel(small_config)
    replace_classification_head(model)
    x = torch.randint(0, 50257, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, 7:] = 0
    out = model(x, attention_mask=mask)
    assert out.shape == (2, 10, CONFIG["data"]["num_classes"])


def test_replace_classification_head_uses_bias_false(small_config):
    model = GPTModel(small_config)
    replace_classification_head(model)
    assert model.out_head.bias is None
    assert model.out_head.out_features == CONFIG["data"]["num_classes"]


def test_configure_transfer_learning_unfreezes_expected_params(small_config):
    model = GPTModel(small_config)
    configure_transfer_learning(model, num_blocks_to_unfreeze=1)

    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("out_head") for name in trainable_names)
    assert any(name.startswith("final_norm") for name in trainable_names)
    assert any(name.startswith("trf_blocks.1") for name in trainable_names)
    assert not any(name.startswith("trf_blocks.0") for name in trainable_names)


def test_gpt_model_at_context_boundary(small_config):
    model = GPTModel(small_config)
    replace_classification_head(model)
    seq_len = small_config["context_length"]
    x = torch.randint(0, 50257, (1, seq_len))
    out = model(x)
    assert out.shape == (1, seq_len, CONFIG["data"]["num_classes"])


def test_multi_head_attention_shape(small_config):
    attn = MultiHeadAttention(
        d_in=32, d_out=32, context_length=64,
        dropout=0.0, num_heads=4, qkv_bias=True,
    )
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == (2, 10, 32)


def test_layer_norm():
    ln = LayerNorm(32)
    x = torch.randn(2, 10, 32)
    out = ln(x)
    assert out.shape == x.shape


def test_feed_forward(small_config):
    ff = FeedForward(small_config)
    x = torch.randn(2, 10, 32)
    out = ff(x)
    assert out.shape == x.shape


def test_transformer_block(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_block_with_mask(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.long)
    out = block(x, attention_mask=mask)
    assert out.shape == x.shape

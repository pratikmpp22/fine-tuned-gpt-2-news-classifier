import copy

import pytest

from src.config.config import CONFIG, LABELS, validate_config


def test_config_has_required_sections():
    assert "paths" in CONFIG
    assert "data" in CONFIG
    assert "training" in CONFIG
    assert "gpt_model" in CONFIG
    assert "weights" in CONFIG


def test_labels_has_four_classes():
    assert len(LABELS) == 4
    assert LABELS[0] == "World"
    assert LABELS[1] == "Sports"
    assert LABELS[2] == "Business"
    assert LABELS[3] == "Sci/Tech"


def test_gpt_model_config_values():
    cfg = CONFIG["gpt_model"]
    assert cfg["vocab_size"] == 50257
    assert cfg["context_length"] == 1024
    assert cfg["emb_dim"] == 768
    assert cfg["n_layers"] == 12
    assert cfg["n_heads"] == 12
    assert cfg["qkv_bias"] is True
    assert cfg["layer_norm_eps"] == 1e-5
    assert cfg["ffn_multiplier"] == 4


def test_data_config_values():
    data_cfg = CONFIG["data"]
    assert data_cfg["num_classes"] == 4
    assert data_cfg["pad_token_id"] == 50256
    assert data_cfg["val_split_ratio"] == 0.15


def test_training_eval_config():
    training_cfg = CONFIG["training"]
    assert training_cfg["eval_iter"] == 5
    assert training_cfg["evals_per_epoch"] == 4


def test_plot_paths_configured():
    assert "accuracy_plot_path" in CONFIG["paths"]
    assert "confusion_matrix_path" in CONFIG["paths"]
    assert "loss_plot_path" in CONFIG["paths"]


def test_validate_config_rejects_non_divisible_heads():
    bad_config = copy.deepcopy(CONFIG)
    bad_config["gpt_model"] = {**bad_config["gpt_model"], "emb_dim": 770}
    with pytest.raises(ValueError, match="divisible"):
        validate_config(bad_config, LABELS)


def test_validate_config_rejects_gap_in_label_indices():
    bad_labels = {0: "A", 1: "B", 2: "C", 4: "D"}
    with pytest.raises(ValueError, match="contiguous"):
        validate_config(CONFIG, bad_labels)

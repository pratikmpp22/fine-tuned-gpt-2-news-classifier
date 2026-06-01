import logging
import sys

def setup_logging(force=False):
    """Configures application logging to stdout with a consistent format.

    Args:
        force (bool): If True, reconfigure even when handlers already exist.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=force,
    )


logger = logging.getLogger("GPT2NewsClassifier")
if not logging.getLogger().handlers:
    setup_logging()

# Centralized configuration for hyperparameters, architecture dimensions, and paths.
CONFIG = {
    "paths": {
        "train_csv": "data/news_train.csv",
        "val_csv": "data/news_validation.csv",
        "test_csv": "data/news_test.csv",
        "model_save_path": "models/News_classifier.pth",
        "loss_plot_path": "plots/loss-plot.pdf",
        "accuracy_plot_path": "plots/accuracy-plot.pdf",
        "confusion_matrix_path": "plots/confusion_matrix.png",
        "metrics_save_path": "models/training_metrics.json",
        "metadata_save_path": "models/News_classifier_metadata.json",
    },
    "data": {
        "val_split_ratio": 0.15,
        "pad_token_id": 50256,
        "num_classes": 4,
    },
    "training": {
        "batch_size": 32,
        "lr": 9e-6,
        "weight_decay": 0.1,
        "num_epochs": 3,
        "eval_iter": 5,
        "evals_per_epoch": 4,
        "num_workers": 0,
        "max_grad_norm": 1.0,
        "num_blocks_to_unfreeze": 3,
        "train_acc_batch_fraction": 0.2,
        "seed": 123,
    },
    "gpt_model": {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "layer_norm_eps": 1e-5,
        "ffn_multiplier": 4,
    },
    "weights": {
        "gpt2_checkpoint_sizes": ("124M", "355M", "774M", "1558M"),
    },
}

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def validate_config(config=None, labels=None):
    """Validate CONFIG structure and basic value constraints.

    Args:
        config (dict, optional): Configuration to validate. Defaults to module CONFIG.
        labels (dict, optional): Label map to validate. Defaults to module LABELS.

    Raises:
        ValueError: If any required key or constraint is violated.
    """
    config = CONFIG if config is None else config
    labels = LABELS if labels is None else labels

    required_sections = ("paths", "data", "training", "gpt_model", "weights")
    for section in required_sections:
        if section not in config:
            raise ValueError(f"CONFIG is missing required section: {section}")

    num_classes = config["data"]["num_classes"]
    if num_classes != len(labels):
        raise ValueError("CONFIG data.num_classes must match the number of LABELS")

    if set(labels.keys()) != set(range(num_classes)):
        raise ValueError("LABELS must define contiguous class indices from 0 to num_classes - 1")

    if not 0 < config["data"]["val_split_ratio"] < 1:
        raise ValueError("CONFIG data.val_split_ratio must be between 0 and 1")

    training = config["training"]
    for key in ("batch_size", "num_epochs", "eval_iter", "evals_per_epoch", "num_blocks_to_unfreeze"):
        if training[key] < 1:
            raise ValueError(f"CONFIG training.{key} must be at least 1")

    gpt = config["gpt_model"]
    if gpt["emb_dim"] % gpt["n_heads"] != 0:
        raise ValueError("CONFIG gpt_model.emb_dim must be divisible by n_heads")

    if gpt["ffn_multiplier"] < 1:
        raise ValueError("CONFIG gpt_model.ffn_multiplier must be at least 1")


validate_config()

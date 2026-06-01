import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import CONFIG, logger
from src.data.columns import REQUIRED_CSV_COLUMNS


def _dataset_csv_paths():
    return (
        CONFIG["paths"]["train_csv"],
        CONFIG["paths"]["val_csv"],
        CONFIG["paths"]["test_csv"],
    )


def validate_ag_news_frame(df, frame_name="dataset"):
    """Validate required columns and label range for AG News-style frames.

    Raises:
        ValueError: If columns are missing or labels fall outside 0..num_classes-1.
    """
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")

    num_classes = CONFIG["data"]["num_classes"]
    valid_labels = set(range(num_classes))
    label_values = set(df["label"].dropna().unique())
    invalid = label_values - valid_labels
    if invalid:
        raise ValueError(
            f"{frame_name} contains invalid labels {sorted(invalid)}; "
            f"expected {sorted(valid_labels)}"
        )


def _csvs_are_valid():
    """Return True when all split CSV files exist and pass validation."""
    num_classes = CONFIG["data"]["num_classes"]
    valid_labels = set(range(num_classes))

    for csv_path in _dataset_csv_paths():
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return False
        header = pd.read_csv(csv_path, nrows=0)
        if not all(column in header.columns for column in REQUIRED_CSV_COLUMNS):
            return False
        labels = pd.read_csv(csv_path, usecols=["label"])["label"]
        if labels.empty or not set(labels.unique()).issubset(valid_labels):
            return False
    return True


def prepare_datasets():
    """Downloads the AG News dataset, performs a stratified split, and saves to CSV.

    Retrieves the dataset from Hugging Face, splits it into training, validation,
    and test sets, and writes them to local CSV files to avoid re-downloading
    on subsequent runs.
    """
    train_csv_path, val_csv_path, test_csv_path = _dataset_csv_paths()

    if _csvs_are_valid():
        logger.info("Dataset CSVs already exist. Skipping download and split.")
        return

    logger.info("Loading dataset from Hugging Face parquet files...")
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }

    try:
        full_train_df = pd.read_parquet(
            "hf://datasets/fancyzhx/ag_news/" + splits["train"]
        )
        test_df = pd.read_parquet(
            "hf://datasets/fancyzhx/ag_news/" + splits["test"]
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download AG News from Hugging Face. "
            "Ensure network access and install fsspec, huggingface_hub, and pyarrow."
        ) from exc

    validate_ag_news_frame(full_train_df, "AG News train split")
    validate_ag_news_frame(test_df, "AG News test split")

    logger.info("Performing stratified split of the training data...")
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=CONFIG["data"]["val_split_ratio"],
        stratify=full_train_df["label"],
        random_state=CONFIG["training"]["seed"],
    )

    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)

    logger.info("Saving splits to CSV...")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logger.info(
        f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
    )


def load_data():
    """Loads the preprocessed dataset splits from local CSV files.

    Returns:
        tuple: A tuple containing (train_df, val_df, test_df) Pandas DataFrames.
    """
    prepare_datasets()

    train_df = pd.read_csv(CONFIG["paths"]["train_csv"])
    val_df = pd.read_csv(CONFIG["paths"]["val_csv"])
    test_df = pd.read_csv(CONFIG["paths"]["test_csv"])

    validate_ag_news_frame(train_df, "train CSV")
    validate_ag_news_frame(val_df, "validation CSV")
    validate_ag_news_frame(test_df, "test CSV")

    return train_df, val_df, test_df

import torch
from torch.utils.data import Dataset
import pandas as pd
from src.config.config import CONFIG
from src.data.columns import TOKENIZER_ALLOWED_SPECIAL


class NewsDataset(Dataset):
    """PyTorch Dataset for text classification.

    Tokenizes each article on access and truncates to ``max_length`` (defaults to the
    model context length). Sequences are not padded here; ``custom_collate_fn`` pads
    each batch only to the longest sequence in that batch.
    """
    def __init__(self, data, tokenizer, max_length=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a Pandas DataFrame")

        self.data = data
        self.tokenizer = tokenizer
        context_length = CONFIG["gpt_model"]["context_length"]
        if max_length is None:
            self.max_length = context_length
        else:
            self.max_length = min(max_length, context_length)

        self._num_classes = CONFIG["data"]["num_classes"]

    def _encode_text(self, text):
        encoded = self.tokenizer.encode(
            text, allowed_special=TOKENIZER_ALLOWED_SPECIAL
        )
        return encoded[: self.max_length]

    def __getitem__(self, index):
        """Retrieves a single unpadded tokenized sequence and its corresponding label."""
        encoded = self._encode_text(self.data.iloc[index]["text"])
        label = int(self.data.iloc[index]["label"])
        if not 0 <= label < self._num_classes:
            raise ValueError(
                f"Invalid label {label} at index {index}; "
                f"expected 0..{self._num_classes - 1}"
            )
        tensor_x = torch.tensor(encoded, dtype=torch.long)
        tensor_y = torch.tensor(label, dtype=torch.long)
        return tensor_x, tensor_y

    def __len__(self):
        return len(self.data)

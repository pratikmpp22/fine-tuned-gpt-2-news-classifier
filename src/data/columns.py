"""Shared column names and tokenisation constants for the data pipeline."""

REQUIRED_CSV_COLUMNS = ("text", "label")
TOKENIZER_ALLOWED_SPECIAL = frozenset({"<|endoftext|>"})

import torch


def extract_last_token_logits(logits, attention_mask):
    """Extract logits at the last non-padding token for each sequence in a batch.

    Args:
        logits (torch.Tensor): Model output of shape (batch, seq_len, num_classes).
        attention_mask (torch.Tensor): Mask of shape (batch, seq_len) where 1 marks real tokens.

    Returns:
        torch.Tensor: Logits of shape (batch, num_classes) for the last real token.

    Raises:
        ValueError: If any sequence has no real tokens.
    """
    token_counts = attention_mask.sum(dim=1)
    if (token_counts == 0).any():
        raise ValueError("Each sequence must contain at least one real token")

    last_token_indices = token_counts - 1
    batch_size = logits.shape[0]
    device = logits.device
    batch_indices = torch.arange(batch_size, device=device)
    return logits[batch_indices, last_token_indices, :]

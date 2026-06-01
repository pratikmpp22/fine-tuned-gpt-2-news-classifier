import torch

from src.config.config import CONFIG, LABELS
from src.data.columns import TOKENIZER_ALLOWED_SPECIAL
from src.training.logits import extract_last_token_logits


def classify_news(text, model, tokenizer, device, max_length=None):
    """Classifies a single news text into one of four categories.

    Tokenizes the input text, truncates to ``max_length`` if given (otherwise only
    to the model context limit), and runs inference at the actual sequence length
    without padding to a fixed width.

    Args:
        text (str): The raw text to classify.
        model (nn.Module): The trained GPT-2 classifier model.
        tokenizer: The BPE tokenizer instance.
        device (torch.device): The hardware device.
        max_length (int, optional): Maximum tokens to keep. Defaults to the model
            context length.

    Returns:
        str: The human-readable string representing the predicted class.

    Raises:
        ValueError: If text is empty, max_length is zero or negative, or the model
            output dimension does not match the number of configured classes.
    """
    if not text or not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")

    num_classes = CONFIG["data"]["num_classes"]
    if model.out_head.out_features != num_classes:
        raise ValueError(
            f"Model output head has {model.out_head.out_features} classes; "
            f"expected {num_classes}. Call replace_classification_head() before inference."
        )

    model.eval()
    supported_context_length = model.pos_emb.weight.shape[0]

    if max_length is None:
        max_length = supported_context_length
    elif max_length <= 0:
        raise ValueError("max_length must be strictly positive")

    max_length = min(max_length, supported_context_length)

    input_ids = tokenizer.encode(
        text, allowed_special=TOKENIZER_ALLOWED_SPECIAL
    )[:max_length]
    seq_len = len(input_ids)

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor, attention_mask=attention_mask)
        last_logits = extract_last_token_logits(logits, attention_mask)[0]

    predicted_label = torch.argmax(last_logits, dim=-1).item()
    return LABELS[predicted_label]

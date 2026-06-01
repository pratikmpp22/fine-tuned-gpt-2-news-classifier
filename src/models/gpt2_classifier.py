import torch
import torch.nn as nn
from src.config.config import CONFIG
from src.models.transformer_block import TransformerBlock
from src.models.layers import LayerNorm


class GPTModel(nn.Module):
    """The main GPT-2 language model architecture adapted for sequence classification.

    Supports weight tying between token embeddings and the LM output head until
    ``replace_classification_head()`` is called for fine-tuning.
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"], cfg.get("layer_norm_eps", 1e-5))

        # Initialized for LM weight tying; replaced before classification training.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, attention_mask=None):
        """Performs a forward pass through the GPT-2 model.

        Args:
            in_idx (torch.Tensor): Tensor of token indices.
            attention_mask (torch.Tensor, optional): Padding mask.

        Returns:
            torch.Tensor: The output logits.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def replace_classification_head(model, num_classes=None):
    """Replace the language-modeling head with a classification head.

    The new head is not weight-tied to token embeddings. Call this after loading
    pretrained LM weights and before fine-tuning or inference.

    Args:
        model (GPTModel): The GPT-2 model instance.
        num_classes (int, optional): Number of output classes. Defaults to CONFIG.

    Returns:
        GPTModel: The same model instance with an updated output head.
    """
    if num_classes is None:
        num_classes = CONFIG["data"]["num_classes"]
    model.out_head = nn.Linear(
        model.tok_emb.embedding_dim, num_classes, bias=False
    )
    return model


def configure_transfer_learning(model, num_classes=None, num_blocks_to_unfreeze=None):
    """Freeze the backbone, attach a classification head, and unfreeze top layers.

    Freezes all parameters, replaces the LM head, then enables gradients on the
    last ``num_blocks_to_unfreeze`` transformer blocks, final layer norm, and the
    new classification head.

    Args:
        model (GPTModel): Model with pretrained weights already loaded.
        num_classes (int, optional): Output classes. Defaults to CONFIG.
        num_blocks_to_unfreeze (int, optional): Blocks to unfreeze from the top.
            Defaults to CONFIG training.num_blocks_to_unfreeze.

    Returns:
        GPTModel: The same model, ready for optimizer setup.
    """
    if num_blocks_to_unfreeze is None:
        num_blocks_to_unfreeze = CONFIG["training"]["num_blocks_to_unfreeze"]

    for param in model.parameters():
        param.requires_grad = False

    replace_classification_head(model, num_classes=num_classes)

    for block_index in range(num_blocks_to_unfreeze):
        for param in model.trf_blocks[-(block_index + 1)].parameters():
            param.requires_grad = True

    for param in model.out_head.parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model

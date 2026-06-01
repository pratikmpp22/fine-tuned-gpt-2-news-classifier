import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.layers import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    """A single Transformer block implementing the Pre-Norm architecture.

    Contains a Multi-Head Attention mechanism followed by a Feed-Forward network,
    with Layer Normalization applied before each (Pre-Norm) and residual connections added after.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        layer_norm_eps = cfg.get("layer_norm_eps", 1e-5)
        self.norm1 = LayerNorm(cfg["emb_dim"], layer_norm_eps)
        self.norm2 = LayerNorm(cfg["emb_dim"], layer_norm_eps)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attention_mask=None):
        """Processes the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Padding mask for the attention layer.

        Returns:
            torch.Tensor: The processed tensor.
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # Pass attention_mask to MultiHeadAttention
        x = self.att(x, attention_mask=attention_mask)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Custom Layer Normalization module.

    Normalizes the input tensor across the last dimension to stabilize training.
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.

    Implements the exact mathematical approximation used in the original OpenAI GPT-2 
    TensorFlow implementation to ensure pretrained weights behave exactly as expected.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "_sqrt_2_over_pi",
            torch.tensor(2.0 / torch.pi).sqrt(),
            persistent=False,
        )

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            self._sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed-forward neural network block for the Transformer architecture.

    Expands the embedding dimension by a factor of 4, applies GELU activation, 
    and contracts back to the original embedding dimension.
    """
    def __init__(self, cfg):
        super().__init__()
        ffn_dim = cfg.get("ffn_multiplier", 4) * cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], ffn_dim),
            GELU(),
            nn.Linear(ffn_dim, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

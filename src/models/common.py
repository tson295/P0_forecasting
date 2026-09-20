"""Shared attention with separate Linear projections for future LoRA adapters."""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.dropout = heads, dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, causal=False):
        b, t, d = x.shape
        def project(layer):
            return layer(x).reshape(b, t, self.heads, d//self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(project(self.q_proj), project(self.k_proj),
                                          project(self.v_proj), is_causal=causal,
                                          dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class EncoderLayer(nn.Module):
    """Post-norm encoder, matching torch TransformerEncoderLayer's default."""
    def __init__(self, dim, heads, ffn_dim, dropout, activation=None, causal=False):
        super().__init__()
        self.self_attn = Attention(dim, heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, ffn_dim), nn.Linear(ffn_dim, dim)
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x):
        x = self.norm1(x+self.dropout(self.self_attn(x, self.causal)))
        return self.norm2(x+self.dropout(self.fc2(self.dropout(self.activation(self.fc1(x))))))


class ForecastModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.head_only = False

    def freeze_backbone(self):
        self.requires_grad_(False)
        self.head.requires_grad_(True)
        self.head_only = True
        self.train(self.training)

    def unfreeze(self):
        self.requires_grad_(True)
        self.head_only = False

    def train(self, mode=True):
        super().train(mode)
        if self.head_only:
            # Keep dropout and BatchNorm statistics frozen as well as parameters.
            for name, child in self.named_children():
                if name != "head":
                    child.eval()
        return self

    def save_pretrained(self, directory, metadata=None):
        from src.training.checkpoint import save_pretrained
        save_pretrained(self, directory, metadata)

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        from src.training.checkpoint import from_pretrained
        return from_pretrained(directory, device=device)

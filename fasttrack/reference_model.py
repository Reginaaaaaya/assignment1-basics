"""Readable decoder-only LM: pre-norm, RoPE, causal attention, SwiGLU."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    context: int = 256
    dim: int = 512
    layers: int = 4
    heads: int = 16
    hidden: int = 1344
    theta: float = 10000.0

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.dim, self.layers, self.heads, self.hidden) <= 0:
            raise ValueError("All dimensions must be positive")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim must be divisible by heads; head dimension must be even")


def rope(x, theta):
    # x: [batch, heads, time, head_dim]. Rotate adjacent coordinate pairs.
    t, d = x.shape[-2:]
    frequency = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angle = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.c.heads, d // self.c.heads).transpose(1, 2) for z in (q, k, v)]
        q, k = rope(q, self.c.theta), rope(k, self.c.theta)
        # is_causal prevents each position from seeing future tokens.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.out(y.transpose(1, 2).contiguous().reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(c.dim, eps=1e-5), nn.RMSNorm(c.dim, eps=1e-5)
        self.attention = Attention(c)
        self.gate, self.up = nn.Linear(c.dim, c.hidden, bias=False), nn.Linear(c.dim, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.RMSNorm(c.dim, eps=1e-5)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        if ids.shape[1] > self.config.context:
            raise ValueError("Input exceeds configured context")
        x = self.embedding(ids)  # [B,T] -> [B,T,D]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # [B,T,V], raw logits

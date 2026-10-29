"""基础版模型。按 LEARNING.md 第 2～5 关阅读，不必一次读完。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Config:
    """普通 Python 类：只用来把超参数放在一起。"""
    def __init__(self, vocab_size, context=32, dim=64, layers=2,
                 heads=4, hidden=128, theta=10000.0):
        if min(vocab_size, context, dim, layers, heads, hidden) <= 0:
            raise ValueError("各维度必须大于 0")
        if dim % heads != 0 or (dim // heads) % 2 != 0:
            raise ValueError("dim 必须整除 heads，每个 head 的维度必须是偶数")
        self.vocab_size = vocab_size
        self.context = context
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.hidden = hidden
        self.theta = theta


def rope(x, theta):
    """给 Q/K 加入位置。输入输出都是 [B,H,T,head_dim]。"""
    sequence_length = x.shape[-2]
    head_dim = x.shape[-1]
    coordinate_numbers = torch.arange(0, head_dim, 2, device=x.device).float()
    frequencies = 1.0 / (theta ** (coordinate_numbers / head_dim))
    positions = torch.arange(sequence_length, device=x.device).float()
    angles = positions[:, None] * frequencies[None, :]
    cosines = angles.cos().to(x.dtype)
    sines = angles.sin().to(x.dtype)
    even_coordinates = x[..., 0::2]
    odd_coordinates = x[..., 1::2]
    rotated_even = even_coordinates * cosines - odd_coordinates * sines
    rotated_odd = even_coordinates * sines + odd_coordinates * cosines
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads # 头的数量
        self.head_dim = config.dim // config.heads # 每一个头的维度
        self.theta = config.theta
        # 分开写 Q/K/V，方便对应到公式
        self.query_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.key_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.value_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.output_projection = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x):
        batch_size, sequence_length, model_dim = x.shape # (B, T, C)

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # [B,T,C] -> [B,T,H,head_dim] -> [B,H,T,head_dim]，reshape是为了分头
        # 第二步交换，是因为pytorch默认后两维是矩阵，前面的是多维独立计算，把H放在前面，是为了让每个头独立计算 
        query = query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = rope(query, self.theta)
        key = rope(key, self.theta)

        # 计算缩放点积注意力
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        future_mask = torch.ones(sequence_length, sequence_length, device=x.device)
        future_mask = future_mask.triu(diagonal=1).bool()

        scores = scores.masked_fill(future_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ value

        # 把多个头拼回一个向量
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, sequence_length, model_dim)
        return self.output_projection(output)


class FeedForward(nn.Module):
    """SwiGLU：两个升维分支相乘，再降回原维度。"""
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden, bias=False) # 一个线性层，特征维度升为hidden(128)
        self.up = nn.Linear(config.dim, config.hidden, bias=False) # 另一个线性层，同样升维
        self.down = nn.Linear(config.hidden, config.dim, bias=False) # 降维恢复为原维度

    def forward(self, x): # 做非线性变换
        gate_values = F.silu(self.gate(x))
        content_values = self.up(x)
        hidden = gate_values * content_values
        return self.down(hidden)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=1e-5) # 层归一化
        self.attention = Attention(config) # 注意力层
        self.feedforward_norm = nn.RMSNorm(config.dim, eps=1e-5) # 层归一化
        self.feedforward = FeedForward(config) # 前馈线性

    def forward(self, x):
        normalized = self.attention_norm(x)
        attention_output = self.attention(normalized)
        x = x + attention_output
        normalized = self.feedforward_norm(x)
        feedforward_output = self.feedforward(normalized)
        x = x + feedforward_output
        return x


class LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim) # 词嵌入层
        self.blocks = nn.ModuleList() # 完整block，有2层

        for _ in range(config.layers):
            self.blocks.append(Block(config))
        self.final_norm = nn.RMSNorm(config.dim, eps=1e-5)
        self.output_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        if token_ids.shape[1] > self.config.context:
            raise ValueError("输入长度超过 context")
        
        x = self.embedding(token_ids) # 词嵌入，增加特征维度

        for block in self.blocks: # 循环block
            x = block(x)

        x = self.final_norm(x) # 最后的归一化线性层

        logits = self.output_head(x)
        
        return logits

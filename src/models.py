import os
import sys
import math

import torch
import torch.nn as nn
from einops import *
from utils import *


class LinearModule(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device = device, dtype = dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean = 0, std = sigma, a = -3 * sigma, b = 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = 1.0, a = -3.0, b = 3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model, "Input tensor must have same hidden dimension as RMSNorm"

        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(torch.square(x), dim = -1, keepdim = True) / self.d_model + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int
    ):
        super().__init__()
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glu_x = silu(self.w1(x)) * self.w3(x)
        return self.w2(glu_x)


class RoPE(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int = 0,
        device: torch.device = None
    ):
        assert d_k % 2 == 0, "Dimension of query and key vectors for RoPE must be even"
        super().__init__()
        self.dim = d_k
        self.half_dim = d_k // 2
        freq = theta ** (-2 * torch.arange(0, self.half_dim, device = device) / d_k)
        self.register_buffer("freq", freq, persistent = False)

    def forward(self, in_query_or_key: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.
    
        Args:
            in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        pairs = rearrange(in_query_or_key, "... (q p) -> ... q p", q = self.half_dim, p = 2)
        angles = token_positions.unsqueeze(-1) * self.freq
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        rot1 = cos_angles * pairs[..., 0] - sin_angles * pairs[..., 1]
        rot2 = sin_angles * pairs[..., 0] + cos_angles * pairs[..., 1]
        result = rearrange(torch.stack((rot1, rot2), dim = -1), "... q p -> ... (q p)", q = self.half_dim, p = 2)
        return result


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope: RoPE = None):
        assert d_model % num_heads == 0, "Embedding dimension should be a multiple of number of heads"
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = LinearModule(d_model, d_model)
        self.k_proj = LinearModule(d_model, d_model)
        self.v_proj = LinearModule(d_model, d_model)
        self.output_proj = LinearModule(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        if self.rope is not None and token_positions is None:
            raise ValueError("Token position not found.")
        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        Q = rearrange(self.q_proj(x), "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads, d_head=self.d_head)
        K = rearrange(self.k_proj(x), "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads, d_head=self.d_head)
        if self.rope is not None:
            token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = rearrange(self.v_proj(x), "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads, d_head=self.d_head)
        attention = rearrange(scaled_dot_product_attention(Q, K, V, mask=mask), "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)", num_heads=self.num_heads, d_head=self.d_head)
        return self.output_proj(attention)


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model) ## Layer norm before self-attention
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope) ## Self-attention
        self.ln2 = RMSNorm(d_model=d_model) ## Layer norm before FFN
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff) ## Position-Wise Feed-Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_len = x.shape[-2]
        token_positions = torch.arange(sequence_len, device=x.device)
        x1 = x + self.attn(self.ln1(x), token_positions)
        x2 = x1 + self.ffn(self.ln2(x1))
        return x2


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope: RoPE = None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model) ## input embeddings
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope) for _ in range(num_layers)]) ## layers of transformer block
        self.ln_final = RMSNorm(d_model=d_model) ## layer norm after transformer block layers
        self.lm_head = LinearModule(d_model, vocab_size) ## output embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

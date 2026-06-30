import os
import sys

import torch
import torch.nn as nn
from einops import *
from utils import *


## trainiable parameter: output_dim * input_dim
class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, dtype=None):
        super(LinearModule, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=2.0/(input_dim + output_dim), a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


## trainable parameter: num_embeddings * embedding_dim
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        ## weight: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ## token_ids: (batch_size, sequence_length)
        return self.weight[token_ids]


## trainable parameter: d_model
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        y = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight
        y = y.to(in_dtype)
        return y


## trainable parameter: d_model * d_ff * 3
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            w1_weight (Float[Tensor, "d_ff d_model"]): Stored weight for W1
            w2_weight (Float[Tensor, "d_model d_ff"]): Stored weight for W2
            w3_weight (Float[Tensor, "d_ff d_model"]): Stored weight for W3
        """
        super(SwiGLU, self).__init__()
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        x2 = silu(x1) * x3
        y = self.w2(x2)
        return y


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, device=None):
        """
        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
        """
        super(RoPE, self).__init__()
        self.freqs = theta ** (-2*torch.arange(start=0, end=(d_k//2), step=1, device=device) / d_k)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.

        Args:
            in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        token_theta = token_positions.unsqueeze(-1) * self.freqs
        sin_theta = torch.sin(token_theta)
        cos_theta = torch.cos(token_theta)
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        rot_even = x_pairs[..., 0] * cos_theta - x_pairs[..., 1] * sin_theta
        rot_odd = x_pairs[..., 0] * sin_theta + x_pairs[..., 1] * cos_theta
        x_rot = torch.stack([rot_even, rot_odd], dim=-1).reshape_as(x)
        return x_rot


## trainable parameter: d_model * d_model * 4
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.rope = rope

        self.q_proj = LinearModule(d_model, d_model) ## nn.Linear(d_model, d_model, bias=False)
        self.k_proj = LinearModule(d_model, d_model)
        self.v_proj = LinearModule(d_model, d_model)
        self.output_proj = LinearModule(d_model, d_model)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x) ## ... seq_length d_model
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq (num_heads head_dim) -> ... num_heads seq head_dim", num_heads=self.num_heads, head_dim=self.head_dim) ## ... num_heads seq_length head_dim
        K = rearrange(K, "... seq (num_heads head_dim) -> ... num_heads seq head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        V = rearrange(V, "... seq (num_heads head_dim) -> ... num_heads seq head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        seq_len = x.shape[-2]  # or passed in
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions=token_positions)
            K = self.rope(K, token_positions=token_positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        W = scaled_dot_product_attention(query = Q, key = K, value = V, mask = mask) ## ... num_heads seq_length head_dim
        W = rearrange(W, "... num_heads seq_length head_dim -> ... seq_length (num_heads head_dim)") ## ... seq_length d_model
        return self.output_proj(W)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model ## dimensionality of the Transformer block inputs
        self.num_heads = num_heads ## Number of heads to use in multi-head self-attention
        self.d_ff = d_ff ## Dimensionality of the position-wise feed-forward inner layer
        self.rope = RoPE(theta = theta, d_k = self.d_model // self.num_heads)

        self.ln1 = RMSNorm(self.d_model)
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, rope=self.rope)
        self.ln2 = RMSNorm(self.d_model)
        self.ffn = SwiGLU(self.d_model, self.d_ff)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        ## First part: Multi-head self-attention with RoPE, with RMSNorm, then plus input
        y = x + self.attn(self.ln1(x), token_positions=token_positions)

        ## Second part: Position-wise feed-forward with RMSNorm, then plus output of first part
        z = y + self.ffn(self.ln2(y))
        return z
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model) ## Input token embedding
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta) for _ in range(num_layers)]) ## layers of transformer blocks
        self.ln_final = RMSNorm(self.d_model) ## RMSNorm applied to the output of the final transformer block
        self.lm_head = LinearModule(self.d_model, self.vocab_size) ## Linear layer to project to vocabulary size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_embeddings(x)
        for layer in self.layers:
            y = layer(y)
        y = self.ln_final(y)
        return self.lm_head(y)

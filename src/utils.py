import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified dimension.

    Args:
        x (Float[Tensor, "..."]): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        Float[Tensor, "..."]: Tensor with softmax applied along the specified dimension.
    """
    values, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - values)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        query: (batch_size, ..., seq_len, d_k)
        key: (batch_size, ..., seq_len, d_k)
        value: (batch_size, ..., seq_len, d_v)
        mask: (batch_size, ..., seq_len, seq_len) or None
    """
    d_k = query.shape[-1]
    scores = einsum(query, key, "... q_len d_k, ... k_len d_k -> ... q_len k_len") / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attention = softmax(scores, dim = -1)
    return einsum(attention, value, "... q_len k_len, ... k_len d_v -> ... q_len d_v")


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between predictions and targets.

    Args:
        inputs (Float[Tensor, "... num_classes"]): Predicted logits.
        targets (Long[Tensor, "..."]): Ground truth class indices.

    Returns:
        Float[Tensor, "..."]: Cross-entropy loss.
    """
    log_probs = nn.functional.log_softmax(inputs, dim = -1)
    return -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
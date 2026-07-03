import os
import sys
import math
from typing import IO, BinaryIO

import numpy as np
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


def softmax_temperature(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified dimension with temperature scaling.

    Args:
        x (Float[Tensor, "..."]): Input tensor.
        dim (int): Dimension along which to compute the softmax.
        temperature (float): Temperature parameter for scaling logits.

    Returns:
        Float[Tensor, "..."]: Tensor with softmax applied along the specified dimension.
    """
    _, indices = torch.max(x, dim=dim, keepdim=True)
    if temperature <= 0.0:
        dimensions = list(x.size())
        dst = torch.zeros(dimensions, device=x.device, dtype=x.dtype)
        dimensions[dim] = 1
        src = torch.ones(dimensions, device=x.device, dtype=x.dtype)
        dst.scatter_(dim, indices, src)
        return dst
    else:
        return softmax(x / temperature, dim=dim)


def top_sampling(probs: torch.Tensor, threshold: float) -> list():
    values, indices = torch.sort(probs, descending=True)
    prev_cumsum = torch.cat([torch.zeros(1, device=probs.device), torch.cumsum(values, dim=-1)[:-1]])
    mask = prev_cumsum < threshold
    mask[0] = True
    result = torch.zeros_like(probs, device=probs.device)
    result[indices[mask]] = values[mask]
    result = result / result.sum()
    return result


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


def learning_rate_schedule(lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int, current_iter: int) -> float:
    """
    Compute the learning rate based on a warmup and cosine decay schedule.

    Args:
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        warmup_iters (int): Number of iterations for linear warmup.
        cosine_cycle_iters (int): Number of iterations for one cosine cycle.
        current_iter (int): Current iteration number.

    Returns:
        float: Computed learning rate for the current iteration.
    """
    if current_iter < warmup_iters:
        return lr_max * current_iter / warmup_iters
    elif current_iter <= cosine_cycle_iters:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return lr_min + (lr_max - lr_min) * cosine_decay
    else:
        return lr_min
    

def gradient_clipping(parameters, max_norm: float = 1.0, eps: float = 1e-6):
    """
    Clip gradients of the given parameters to a maximum norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Parameters whose gradients will be clipped.
        max_norm (float): Maximum allowed norm of the gradients.
    """
    total_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None))
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coeff)


def data_loading(dataset: np.array, batch_size: int, context_length: int, device: torch.device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.array([dataset[i:i+context_length] for i in indices])
    y = np.array([dataset[i+1:i+context_length+1] for i in indices])
    return (torch.from_numpy(x).to(device).long(), torch.from_numpy(y).to(device).long())


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, hyperparams: dict, out: str | os.PathLike | BinaryIO | IO[bytes]):
    model_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(model_dict, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
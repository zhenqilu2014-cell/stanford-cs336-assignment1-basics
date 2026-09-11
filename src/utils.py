import os
import sys
import math
from typing import IO, BinaryIO, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def glu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    w1_x = einsum(w1, x, "d_ff d_model, ... d_model -> ... d_ff")
    w2_x = einsum(w2, x, "d_ff d_model, ... d_model -> ... d_ff")
    return torch.sigmoid(w1_x) * w2_x

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = torch.max(x, dim = dim, keepdim = True)
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim = dim, keepdim = True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        scores = torch.masked_fill(scores, ~mask, float("-inf"))
    attention = einsum(softmax(scores, dim = -1), V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention

def cross_entropy(
    inputs: torch.Tensor, 
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    log_probs = F.log_softmax(inputs, dim=-1)
    entropy = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
    return entropy

def learning_rate_schedule(
    lr_max: float, 
    lr_min: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int, 
    current_iter: int
):
    if current_iter < warmup_iters:
        return current_iter / warmup_iters * lr_max
    if warmup_iters <= current_iter <= cosine_cycle_iters:
        return lr_min + 0.5 * (1 + math.cos((current_iter - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    if current_iter > cosine_cycle_iters:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    if max_l2_norm <= 0:
        raise ValueError(f"Invalid maximum l2 norm: {max_l2_norm}")
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return
    l2_square = 0
    for p in parameters:
        l2_square += torch.sum(p.grad.data ** 2)
    l2_norm = torch.sqrt(l2_square)
    if l2_norm <= max_l2_norm:
        return
    alpha = max_l2_norm / (l2_norm + eps)
    with torch.no_grad():
        for p in parameters:
            p.grad.mul_(alpha)
    return

def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(dataset) <= context_length:
        raise ValueError(f"Invalid context length: {context_length}")
    if batch_size <= 0:
        raise ValueError(f"Invalid batch size: {batch_size}")
    if context_length <= 0:
        raise ValueError(f"Invalid context length: {context_length}")
    device = torch.device(device)
    indices = np.random.randint(0, len(dataset) - context_length, batch_size)
    temp = np.array([dataset[i: i + context_length] for i in indices])
    datain = torch.tensor(temp).to(device=device, dtype=torch.long)
    temp = np.array([dataset[i + 1: i + 1 + context_length] for i in indices])
    dataout = torch.tensor(temp).to(device=device, dtype=torch.long)
    return (datain, dataout)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] ## Get state associated with p
                t = state.get("t", 0) ## Get iteration number from the state, else 0
                grad = p.grad.data ## Get gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad ## Update weight tensor in-place
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if any(x < 0 or x >= 1 for x in betas):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                m1 = state.get("m1", torch.zeros_like(p))
                m2 = state.get("m2", torch.zeros_like(p))
                grad = p.grad.data
                lr_adj = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1)) ## Adjusted learning rate
                p.data -= lr * weight_decay * p.data ## Apply weight decay
                m1 = beta1 * m1 + (1 - beta1) * grad
                m2 = beta2 * m2 + (1 - beta2) * grad * grad
                p.data -= lr_adj * m1 / (torch.sqrt(m2) + eps)
                state["t"] = t + 1
                state["m1"] = m1
                state["m2"] = m2
        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())

        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
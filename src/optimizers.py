from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), lr=1e-3, weight_decay=0.1, eps=1e-8):
        """
        Args:
            betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            lr (float): Learning rate.
            weight_decay (float): Weight decay coefficient.
            eps (float): Small value to prevent division by zero.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            ## Get hyperparameters
            lr = group["lr"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                lr_adj = lr * math.sqrt(1 - beta_2 ** (t + 1)) / (1 - beta_1 ** (t + 1))
                p.data -= lr * weight_decay * p.data ## apply weight decay
                
                ## update moments
                if "moment_1" not in state:
                    state["moment_1"] = torch.zeros_like(p.data)
                if "moment_2" not in state:
                    state["moment_2"] = torch.zeros_like(p.data)
                moment_1 = state["moment_1"]
                moment_2 = state["moment_2"]
                moment_1 = beta_1 * moment_1 + (1 - beta_1) * grad
                moment_2 = beta_2 * moment_2 + (1 - beta_2) * grad * grad

                ## update state
                state["t"] = t + 1 # Increment iteration number.
                state["moment_1"] = moment_1
                state["moment_2"] = moment_2

                p.data -= lr_adj * moment_1 / (torch.sqrt(moment_2) + eps)
        
        return loss
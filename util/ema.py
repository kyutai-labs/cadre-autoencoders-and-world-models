"""Exponential moving average of the trained weights, with bias correction."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor


class EMA:
    """EMA of a model's trainable parameters. The step-n average weighs the parameters w_n = 1 / c_n with
    c_n = decay c_{n-1} + 1, c_0 = 0: the first step copies them, and w_n tends to 1 - decay (bias correction)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.count = 0.0
        self.state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.count = self.count * self.decay + 1
        w = 1.0 / self.count
        for name, param in model.named_parameters():
            if name in self.state:
                self.state[name].mul_(1 - w).add_(param.detach(), alpha=w)

    @contextmanager
    def swapped_in(self, model: nn.Module):
        """Run the block with the averaged weights in the model, then put the trained weights back."""
        params = dict(model.named_parameters())
        trained = {name: params[name].detach().clone() for name in self.state}
        with torch.no_grad():
            for name, value in self.state.items():
                params[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in trained.items():
                    params[name].copy_(value)

    def averaged_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        """The model's state_dict with every trained parameter replaced by its average (what checkpoints ship)."""
        return {name: self.state.get(name, value) for name, value in model.state_dict().items()}

    def state_dict(self) -> dict[str, Tensor | float | dict]:
        return {"state": self.state, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        assert state["state"].keys() == self.state.keys()
        self.count = state["count"]
        for name, value in state["state"].items():
            self.state[name].copy_(value)

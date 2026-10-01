"""Learning-rate schedule: linear warmup, constant, then an optional cosine decay to min_lr over the last
decay_steps of the run (warmup-stable-decay).

The rate is a pure function of the step and the run's own config, so a resumed or continued run follows its own
schedule shape from the checkpoint's step on.
"""

import math

import torch
from pydantic import BaseModel, ConfigDict


class LRSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    warmup_steps: int = 0
    decay_steps: int = 0  # 0 = constant after the warmup
    min_lr: float = 0.0


def lr_at(step: int, schedule: LRSchedule, total_steps: int) -> float:
    """The rate of optimizer step `step` (0-based) of a run of total_steps steps."""
    decay_start = total_steps - schedule.decay_steps
    if step < schedule.warmup_steps:
        return schedule.lr * step / schedule.warmup_steps
    if schedule.decay_steps == 0 or step < decay_start:
        return schedule.lr
    if step < total_steps:
        cosine = 0.5 * (1 + math.cos(math.pi * (step - decay_start) / schedule.decay_steps))
        return schedule.min_lr + (schedule.lr - schedule.min_lr) * cosine
    return schedule.min_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Write the rate into the optimizer's tensor rates, in place (a compiled step keeps reading the same tensor)."""
    for group in optimizer.param_groups:
        group["lr"].fill_(lr)

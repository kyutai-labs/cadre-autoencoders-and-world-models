"""Training of the inverse dynamics probe: smooth L1 on standardized per-frame actions, photometric augmentation only,
AdamW with linear warmup and cosine decay, DDP over the GPUs.

The data are `window`-frame clips of the world models' datasets (dataset.EnvironmentDataset), at their resolution and
frame rate; the probe reads the continuous actions and validates on the world models' held-out split. Training draws
dataset.clip_loader's shuffled epochs, whose windows shift by a random phase every epoch; evaluation reads the fixed
windows.
"""

import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dataset import ClipDataset, clip_loader
from model_idm import InverseDynamicsProbe, save_probe

STATS_BATCHES = 50  # training batches per rank that estimate the action statistics


class IDMTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    total_steps: int = 60_000
    grad_clip: float = 1.0
    batch_size: int = 48  # per GPU per micro-batch
    grad_accum: int = 1
    num_workers: int = 6
    seed: int = 2025
    log_every: int = 50
    eval_every: int = 2000
    eval_batches: int = 20
    ckpt_every: int = 5000


def lr_at(step: int, config: IDMTrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def photometric_augment(video: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """One brightness and contrast change (+-0.2) per clip, identical across its frames, plus pixel noise (sigma 0.02).
    Spatial augmentation would change the action (a flip inverts the steering), so there is none."""
    b = video.shape[0]
    brightness = (torch.rand(b, 1, 1, 1, 1, device=video.device, generator=generator) - 0.5) * 0.4
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=video.device, generator=generator) - 0.5) * 0.4
    video = (video - 0.5) * contrast + 0.5 + brightness
    video = video + torch.randn(video.shape, device=video.device, generator=generator) * 0.02
    return video.clamp(0.0, 1.0)


def probe_loss(pred: Tensor, actions: Tensor, mean: Tensor, std: Tensor) -> tuple[Tensor, Tensor]:
    """Smooth L1 per action dimension in standardized units -> (mean over dims, per-dim loss)."""
    target = (actions - mean) / (std + 1e-6)
    per_dim = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(0, 1))
    return per_dim.mean(), per_dim


def to_frames(video: Tensor, device) -> Tensor:
    """uint8 clips -> float frames in [0, 1] on `device`."""
    return video.to(device, non_blocking=True).float() / 255.0


def training_loss(
    net: torch.nn.Module,
    probe: InverseDynamicsProbe,
    video: Tensor,
    actions: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """One micro-batch: uint8 video (B, T, 3, H, W) and actions (B, T, D) on the CPU -> (loss, per-dim loss).
    `net` is the probe, or its DDP wrapper."""
    device = probe.action_mean.device
    frames = photometric_augment(to_frames(video, device), generator)
    actions = actions.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=frames.is_cuda):
        pred = net(frames)
        return probe_loss(pred, actions, probe.action_mean, probe.action_std)


@torch.no_grad()
def estimate_action_stats(probe: InverseDynamicsProbe, batches: Iterator[tuple[Tensor, ...]], world: int) -> None:
    """Per-dimension mean and std of the training actions, averaged over the ranks, frozen into the probe."""
    actions = torch.cat([next(batches)[1].reshape(-1, probe.config.action_dim) for _ in range(STATS_BATCHES)])
    probe.action_mean.copy_(actions.mean(0))
    probe.action_std.copy_(actions.std(0).clamp_min(1e-4))
    if world > 1:
        for buf in (probe.action_mean, probe.action_std):
            dist.all_reduce(buf)
            buf /= world


@torch.no_grad()
def evaluate(probe: InverseDynamicsProbe, dataset: ClipDataset, config: IDMTrainConfig) -> dict[str, float]:
    """Held-out R^2 per action dimension (the ARR ceiling on real video) and RMSE in raw units."""
    was_training = probe.training
    probe.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    device = probe.action_mean.device
    dim = probe.config.action_dim
    sse, total, total_sq = (torch.zeros(dim, device=device) for _ in range(3))
    count = 0
    for _, batch in zip(range(config.eval_batches), loader, strict=False):
        pred = probe.recover_actions(to_frames(batch["video"], device)).reshape(-1, dim)
        actions = batch["actions"].to(device).reshape(-1, dim).float()
        sse += ((pred - actions) ** 2).sum(0)
        total += actions.sum(0)
        total_sq += (actions**2).sum(0)
        count += actions.shape[0]
    mse = sse / count
    r2 = 1.0 - mse / (total_sq / count - (total / count) ** 2).clamp_min(1e-8)
    probe.train(was_training)
    return {"r2_mean": float(r2.mean()), "rmse_mean": float(mse.sqrt().mean())} | {
        f"r2_{d}": float(r2[d]) for d in range(dim)
    }


def train(
    probe: InverseDynamicsProbe,
    train_set: ClipDataset,
    val_set: ClipDataset,
    config: IDMTrainConfig,
    output_dir: str | Path,
    resume: str | None = None,
) -> None:
    """Train `probe` (already on its device, with a pretrained DINOv3) under torchrun or on one GPU. Rank 0 writes
    probe_step<k>.pt every ckpt_every steps and at the end, and log.jsonl. A resumed run continues exactly: the clip
    stream restarts at the clips already drawn, and the augmentation noise of each step is seeded by (rank, step)."""
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in probe.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        probe.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
    # A run's first STATS_BATCHES batches estimate the action statistics, then each step takes grad_accum of them.
    drawn = config.batch_size * (STATS_BATCHES + start_step * config.grad_accum) if resume else 0
    loader = clip_loader([train_set], [1.0], config.batch_size, rank, world, config.seed, drawn, config.num_workers)
    batches = ((batch["video"], batch["actions"]) for batch in loader)
    if not resume:
        estimate_action_stats(probe, batches, world)
    device = probe.action_mean.device
    net = DistributedDataParallel(probe, device_ids=[device]) if world > 1 else probe
    generator = torch.Generator(device)

    def log(record: dict) -> None:
        print(json.dumps(record), flush=True)
        with open(output_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def checkpoint(step: int) -> None:
        save_probe(probe, output_dir / f"probe_step{step}.pt", step=step, optimizer=optimizer.state_dict())

    probe.train()
    for step in range(start_step, config.total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, config)
        optimizer.zero_grad(set_to_none=True)
        generator.manual_seed(config.seed + 1_000_003 * rank + step)
        loss_sum = 0.0
        for micro in range(config.grad_accum):
            video, actions = next(batches)
            # One gradient all-reduce per optimizer step: skip it on all but the last micro-batch.
            last = micro == config.grad_accum - 1
            with contextlib.nullcontext() if (last or world == 1) else net.no_sync():
                loss, _ = training_loss(net, probe, video, actions, generator)
                (loss / config.grad_accum).backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()

        if rank == 0 and step % config.log_every == 0:
            log({"step": step, "loss": loss_sum / config.grad_accum, "lr": lr_at(step, config)})
        if step > 0 and step % config.eval_every == 0:
            if rank == 0:
                log({"step": step} | {f"val/{k}": v for k, v in evaluate(probe, val_set, config).items()})
            if world > 1:
                dist.barrier()
        if rank == 0 and step > 0 and step % config.ckpt_every == 0:
            checkpoint(step)
    if rank == 0:
        checkpoint(config.total_steps)

"""Training of the CADRE autoencoder.

One step: the flow pass and its losses under bf16 autocast (losses.AutoencoderLoss), backward, AdamW at the scheduled
rate, then the EMA of the weights. The trainer also tracks the mean and std of the latents, which every checkpoint
stores: a world model standardizes the latents with them.

A run resumes from its own latest checkpoint in output_dir if it has one; otherwise from continue_from (another
run's full training state: weights, EMA, optimizer, latent statistics and step); otherwise from finetune_from
(a release checkpoint: weights and latent statistics only, at step 0); otherwise it starts fresh.
"""

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from dataset import ClipDataset, clip_loader
from losses import AutoencoderLoss, AutoencoderLossConfig
from model_autoencoder import CADRE, CADRE_MODELS, RESOLUTION_MULTIPLE, CADREConfig
from util import distributed, hub
from util.checkpoint import RELEASE, TRAINING_STATE, save_checkpoint
from util.ema import EMA
from util.lr_schedule import LRSchedule, lr_at, set_lr

EARLY_LOG_STEPS = 10  # the first steps are all logged


class MultiresConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buckets: list[tuple[int, int]]  # (H, W), each a multiple of 32; step k trains at buckets[k % len(buckets)]
    rope_spatial_rescale: float

    @field_validator("buckets")
    @classmethod
    def _multiples(cls, buckets):
        assert all(h % RESOLUTION_MULTIPLE == 0 and w % RESOLUTION_MULTIPLE == 0 for h, w in buckets), buckets
        return buckets


class AutoencoderTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int
    batch_size: int = 96  # global: batch_size / world_size clips per GPU
    schedule: LRSchedule
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    latent_stats_decay: float = 0.99
    loss: AutoencoderLossConfig = AutoencoderLossConfig()
    # DINOv3 in train mode during the training steps: its RoPE box-rescale augmentation, in the encoder and loss_dino.
    dino_rope_augmentation: bool = True
    multires: MultiresConfig | None = None
    seed: int = 30
    checkpoint_every: int = 25_000
    keep_recent: int = 2  # checkpoints kept besides the permanent ones
    keep_every: int | None = 100_000  # checkpoints at multiples of this step are kept for good
    log_every: int = 200
    eval_every: int = 5_000
    num_workers: int = 6  # data loader workers per GPU
    compile: bool = True


def model_config(name: str, config: AutoencoderTrainConfig) -> CADREConfig:
    """The cell's geometry, with adaptive resolution and the RoPE rescale in the multi-resolution finetune."""
    cfg = CADRE_MODELS[name]
    if config.multires is None:
        return cfg
    return cfg.model_copy(
        update={"adaptive_resolution": True, "rope_spatial_rescale": config.multires.rope_spatial_rescale}
    )


def data_resolution(model: CADRE, config: AutoencoderTrainConfig) -> tuple[int, int]:
    """The size clips are loaded at: the model's, or the largest bucket of the multi-resolution finetune."""
    return max(config.multires.buckets) if config.multires is not None else model.config.resolution


def resize_video(video: Tensor, size: tuple[int, int]) -> Tensor:
    """uint8 (B, T, 3, H, W) -> (B, T, 3, *size), antialiased bilinear, rounded back to uint8."""
    if tuple(video.shape[-2:]) == tuple(size):
        return video
    b, t, c = video.shape[:3]
    x = F.interpolate(video.float().flatten(0, 1), size, mode="bilinear", antialias=True)
    return x.round().clamp_(0, 255).to(torch.uint8).view(b, t, c, *size)


class LatentStats:
    """EMAs of the mean and the std of the latents, one float64 scalar each, updated on every rank's own batches
    and averaged over the ranks at every log step."""

    def __init__(self, decay: float, device):
        self.decay = decay
        self.mean = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.std = torch.tensor(1.0, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, z: Tensor) -> None:
        z = z.float()
        self.mean = self.decay * self.mean + (1 - self.decay) * z.double().mean()
        self.std = self.decay * self.std + (1 - self.decay) * z.std(keepdim=True).double().mean()

    def all_reduce(self) -> None:
        distributed.all_reduce_mean_(self.mean)
        distributed.all_reduce_mean_(self.std)

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.mean = state["mean"].to(self.mean)
        self.std = state["std"].to(self.std)


class Trainer:
    """The model, its DDP wrapper (with more than one GPU), the loss, the optimizer, the EMA and the latent
    statistics of one run."""

    def __init__(self, model: CADRE, name: str, config: AutoencoderTrainConfig, device):
        self.model, self.name, self.config, self.device = model, name, config, device
        self.train_mode()
        self.net = DistributedDataParallel(model, device_ids=[device]) if distributed.world_size() > 1 else model
        self.loss = AutoencoderLoss(config.loss).to(device)
        # The rate is a tensor so that the compiled optimizer step does not recompile when it changes.
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=torch.tensor(config.schedule.lr),
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.ema = EMA(model, config.ema_decay)  # after the DDP wrap: rank 0's initial weights on every rank
        self.latent_stats = LatentStats(config.latent_stats_decay, device)
        self.optimizer_step = self.optimizer.step

    def train_mode(self) -> None:
        """The model in train mode, with DINOv3 in train mode too under dino_rope_augmentation."""
        self.model.train()
        self.model.encoder.dino.train(self.config.dino_rope_augmentation)

    def compile(self) -> None:
        """Compile the forward and the optimizer step (after any state is loaded)."""
        self.net.compile()
        self.optimizer_step = torch.compile(self.optimizer.step)

    def step(self, video: Tensor, step: int) -> dict[str, Tensor]:
        """One optimizer step on a uint8 batch (B, T, 3, H, W) on the device; returns the detached losses."""
        set_lr(self.optimizer, lr_at(step, self.config.schedule, self.config.steps))
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(video.device.type, dtype=torch.bfloat16):
            out = self.net(video)
            losses = self.loss(self.model, out)
        self.latent_stats.update(out.z)
        losses["loss_total"].backward()
        self.optimizer_step()
        self.ema.update(self.model)
        return {k: v.detach() for k, v in losses.items()}

    def release_checkpoint(self) -> dict:
        """The release checkpoint: averaged weights and the latent statistics (CADRE.from_checkpoint loads it)."""
        self.model.latent_mean, self.model.latent_std = self.latent_stats.mean.item(), self.latent_stats.std.item()
        return self.model.checkpoint(self.name) | {"state_dict": self.ema.averaged_state_dict(self.model)}

    def training_state(self, step: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "latent_stats": self.latent_stats.state_dict(),
            "step": step,
        }

    def load_training_state(self, state: dict) -> int:
        """Restore a full training state; returns the step to continue at."""
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.latent_stats.load_state_dict(state["latent_stats"])
        return state["step"] + 1

    def load_weights(self, checkpoint: dict) -> None:
        """Start from a release checkpoint's weights and latent statistics (the EMA restarts from them)."""
        self.model.load_state_dict(checkpoint["state_dict"])
        self.latent_stats.mean.fill_(checkpoint["latent_mean"].item())
        self.latent_stats.std.fill_(checkpoint["latent_std"].item())
        self.ema = EMA(self.model, self.config.ema_decay)

    def save(self, output_dir: Path, step: int, final: bool = False) -> None:
        c = self.config
        save_checkpoint(
            output_dir, step, self.release_checkpoint(), self.training_state(step), c.keep_recent, c.keep_every, final
        )


def restore(trainer: Trainer, source: tuple[str, Path] | None) -> int:
    """Load the state `source` names into the trainer; returns the first step to run."""
    if source is None:
        return 0
    kind, path = source
    if kind == "finetune":
        trainer.load_weights(hub.load_checkpoint(path / RELEASE if (path / RELEASE).is_file() else path))
        return 0
    return trainer.load_training_state(torch.load(path / TRAINING_STATE, map_location="cpu", weights_only=True))


def train(
    trainer: Trainer,
    datasets: list[ClipDataset],
    weights: list[float],
    output_dir: str | Path,
    source: tuple[str, Path] | None,
    evaluate: Callable[[CADRE, int], dict[str, float]] | None = None,
) -> None:
    """Run the recipe from where `source` leaves it. Rank 0 writes the checkpoints and log.jsonl. `evaluate`
    (model with the averaged weights, step) -> metrics runs every eval_every steps."""
    c, model, device = trainer.config, trainer.model, trainer.device
    output_dir = Path(output_dir)
    rank, world = distributed.rank(), distributed.world_size()
    assert c.batch_size % world == 0, f"batch_size {c.batch_size} is not a multiple of {world} GPUs"
    per_gpu = c.batch_size // world
    start = restore(trainer, source)
    if start >= c.steps:
        return
    if c.compile:
        trainer.compile()
    loader = clip_loader(datasets, weights, per_gpu, rank, world, c.seed, start * per_gpu, c.num_workers)
    batches: Iterator[dict] = iter(loader)

    def log(record: dict) -> None:
        if distributed.is_main():
            print(json.dumps(record), flush=True)
            with open(output_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

    sums: dict[str, Tensor] = defaultdict(lambda: torch.zeros((), device=device))
    counts: dict[str, int] = defaultdict(int)
    steps_since_log, last_log = 0, time.monotonic()
    step = start
    for step in range(start, c.steps):
        video = next(batches)["video"].to(device, non_blocking=True)
        if c.multires is not None:
            video = resize_video(video, c.multires.buckets[step % len(c.multires.buckets)])
        for k, v in trainer.step(video, step).items():
            sums[k] += v.float()
            counts[k] += 1
        steps_since_log += 1

        if step < EARLY_LOG_STEPS or step % c.log_every == 0:
            # The loss keys are the same on every rank (the PSD gate is shared), so the means reduce in order.
            keys = sorted(sums)
            means = distributed.all_reduce_mean_(torch.stack([sums[k] / counts[k] for k in keys]))
            trainer.latent_stats.all_reduce()
            now = time.monotonic()
            log(
                {"step": step, "lr": lr_at(step, c.schedule, c.steps)}
                | dict(zip(keys, means.tolist(), strict=True))
                | {"latent_mean": trainer.latent_stats.mean.item(), "latent_std": trainer.latent_stats.std.item()}
                | {"sec_per_step": (now - last_log) / steps_since_log}
            )
            sums.clear()
            counts.clear()
            steps_since_log, last_log = 0, now
        if evaluate is not None and step % c.eval_every == 0:
            with trainer.ema.swapped_in(model):
                model.eval()
                log({"step": step} | {f"val/{k}": v for k, v in evaluate(model, step).items()})
                trainer.train_mode()
        if step > 0 and step % c.checkpoint_every == 0:
            if distributed.is_main():
                trainer.save(output_dir, step)
            distributed.barrier()
    if distributed.is_main():
        trainer.save(output_dir, step, final=True)
    distributed.barrier()

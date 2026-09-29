"""Training of a CADRE world model, with one recipe per environment in configs/world_model.

One step: the frozen autoencoder (in eval mode) encodes the uint8 clips; the world model takes flow matching with
diffusion forcing on part of the batch and PSD-U on the rest (WorldModel.forward), under bf16 autocast; then
backward, gradient-norm clipping, AdamW at the scheduled rate and the EMA of the weights.

A run resumes from its own latest checkpoint in output_dir if it has one; otherwise from continue_from (another
run's full training state: weights, EMA, optimizer and step); otherwise it starts fresh.
"""

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from dataset import ENVIRONMENTS, EnvironmentDataset, clip_loader
from model_autoencoder import CADREConfig
from model_world_model import Autoencoder, WorldModel, WorldModelConfig, checkpoint_dict, encode_video, psd_split
from util import distributed
from util.checkpoint import TRAINING_STATE, save_checkpoint
from util.ema import EMA
from util.lr_schedule import LRSchedule, lr_at, set_lr

EARLY_LOG_STEPS = 10  # the first steps are all logged
WALL_CLOCK_EVERY = 50  # steps between two checks of max_wall_hours


class WorldModelTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: str  # a name of dataset.ENVIRONMENTS: the action columns
    camera_twist: bool = False  # the actions are camera twists, which the world model scales and keys (indoor)
    action_mean: list[float] | None = None  # z-score of the continuous actions; None: clamp and symlog
    action_std: list[float] | None = None
    # Shift of the flow-time schedule: a value, or "latent_grid" for sqrt(h w C / 4096) of the autoencoder's latents.
    noise_shift: float | Literal["latent_grid"]
    frames: int = 80
    psd_fraction: float = 0.25
    # Optimization.
    steps: int
    batch_size: int = 64  # global: batch_size / world_size clips per GPU
    schedule: LRSchedule
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    max_wall_hours: float | None = None  # stop cleanly (with a final checkpoint) after this much training time
    seed: int = 28
    checkpoint_every: int = 10_000
    keep_recent: int = 1  # checkpoints kept besides the permanent ones
    keep_every: int | None = None  # checkpoints at multiples of this step are kept for good
    log_every: int = 1_000
    eval_every: int = 2_000
    num_workers: int = 6  # data loader workers per GPU
    compile: bool = True

    @field_validator("environment")
    @classmethod
    def _known(cls, environment: str) -> str:
        assert environment in ENVIRONMENTS, f"unknown environment {environment!r}"
        return environment


def model_config(config: WorldModelTrainConfig, autoencoder: dict) -> WorldModelConfig:
    """The world model of an environment recipe on an autoencoder, given its release checkpoint dict."""
    cfg = CADREConfig.model_validate(autoencoder["config"])
    shift = config.noise_shift
    if shift == "latent_grid":
        h, w = cfg.latent_grid
        shift = math.sqrt(h * w * cfg.latent_dim / 4096)
    return WorldModelConfig(
        tau=cfg.tau,
        latent_channels=cfg.latent_dim,
        frames=config.frames,
        action_dim=ENVIRONMENTS[config.environment].action_dim,
        camera_twist=config.camera_twist,
        action_mean=config.action_mean,
        action_std=config.action_std,
        noise_shift=shift,
        psd_fraction=config.psd_fraction,
    )


class Trainer:
    """The world model, its DDP wrapper (with more than one GPU), the frozen autoencoder that encodes its inputs
    (and whose release checkpoint every world-model checkpoint embeds), the optimizer and the EMA of one run."""

    def __init__(
        self,
        model: WorldModel,
        autoencoder: Autoencoder,
        autoencoder_checkpoint: dict,
        config: WorldModelTrainConfig,
        device,
    ):
        self.model, self.config, self.device = model.train(), config, device
        self.autoencoder, self.autoencoder_checkpoint = autoencoder, autoencoder_checkpoint
        self.net = DistributedDataParallel(model, device_ids=[device]) if distributed.world_size() > 1 else model
        # The rate is a tensor so that the compiled optimizer step does not recompile when it changes.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=torch.tensor(config.schedule.lr),
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.ema = EMA(model, config.ema_decay)  # after the DDP wrap: rank 0's initial weights on every rank
        self.optimizer_step = self.optimizer.step

    def compile(self) -> None:
        """Compile the encoder, the transformer and the optimizer step (after any state is loaded). Not the whole
        loss, whose teacher passes run without gradients."""
        self.autoencoder.encode = torch.compile(self.autoencoder.encode)
        self.model.velocity = torch.compile(self.model.velocity)
        self.optimizer_step = torch.compile(self.optimizer.step)

    @torch.no_grad()
    def encode(self, video: Tensor) -> Tensor:
        """Standardized latents (B, T / tau, h, w, C) of a uint8 batch (B, T, 3, H, W), encoded under bf16 autocast."""
        with torch.autocast(video.device.type, dtype=torch.bfloat16):
            return encode_video(self.autoencoder, video)

    def step(self, z: Tensor, actions: Tensor, step: int) -> dict[str, Tensor]:
        """One optimizer step on latents z (B, T, h, w, C) and the clips' actions (B, frames, action_dim) on the
        device; returns the detached losses and the gradient norm before clipping."""
        set_lr(self.optimizer, lr_at(step, self.config.schedule, self.config.steps))
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(z.device.type, dtype=torch.bfloat16):
            losses = self.net(z, actions)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer_step()
        self.ema.update(self.model)
        return {k: v.detach() for k, v in losses.items()} | {"grad_norm": grad_norm}

    def release_checkpoint(self) -> dict:
        """The release checkpoint (model_world_model.load_checkpoint): averaged weights and the full autoencoder."""
        return checkpoint_dict(self.model.config, self.ema.averaged_state_dict(self.model), self.autoencoder_checkpoint)

    def training_state(self, step: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }

    def load_training_state(self, state: dict) -> int:
        """Restore a full training state; returns the step to continue at."""
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        return state["step"] + 1

    def save(self, output_dir: Path, step: int, final: bool = False) -> None:
        c = self.config
        save_checkpoint(
            output_dir, step, self.release_checkpoint(), self.training_state(step), c.keep_recent, c.keep_every, final
        )


def restore(trainer: Trainer, source: tuple[str, Path] | None) -> int:
    """Load the training state `source` names (a resume or a continue_from) into the trainer; returns the first step
    to run."""
    if source is None:
        return 0
    state = torch.load(source[1] / TRAINING_STATE, map_location="cpu", weights_only=True)
    return trainer.load_training_state(state)


def train(
    trainer: Trainer,
    dataset: EnvironmentDataset,
    output_dir: str | Path,
    source: tuple[str, Path] | None,
    evaluate: Callable[[WorldModel, Autoencoder, int], dict[str, float]] | None = None,
) -> None:
    """Run the recipe from where `source` leaves it, on the training split of the recipe's environment. Rank 0
    writes the checkpoints and log.jsonl. `evaluate` (world model with the averaged weights, the autoencoder, step)
    -> metrics runs every eval_every steps, from step 0."""
    c, model, device = trainer.config, trainer.model, trainer.device
    output_dir = Path(output_dir)
    rank, world = distributed.rank(), distributed.world_size()
    assert c.batch_size % world == 0, f"batch_size {c.batch_size} is not a multiple of {world} GPUs"
    per_gpu = c.batch_size // world
    n_psd = per_gpu - psd_split(per_gpu, c.psd_fraction)
    assert n_psd == per_gpu * c.psd_fraction, f"{per_gpu} clips per GPU cannot give PSD-U exactly {c.psd_fraction}"
    start = restore(trainer, source)
    if start >= c.steps:
        return
    if c.compile:
        trainer.compile()
    loader = clip_loader([dataset], [1.0], per_gpu, rank, world, c.seed, start * per_gpu, c.num_workers)
    batches: Iterator[dict] = iter(loader)

    def log(record: dict) -> None:
        if distributed.is_main():
            print(json.dumps(record), flush=True)
            with open(output_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

    sums: dict[str, Tensor] = defaultdict(lambda: torch.zeros((), device=device))
    steps_since_log, last_log, loop_start = 0, time.monotonic(), time.monotonic()
    step = start
    for step in range(start, c.steps):
        batch = next(batches)
        video = batch["video"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        for k, v in trainer.step(trainer.encode(video), actions, step).items():
            sums[k] += v.float()
        steps_since_log += 1

        if step < EARLY_LOG_STEPS or step % c.log_every == 0:
            keys = sorted(sums)
            means = distributed.all_reduce_mean_(torch.stack([sums[k] / steps_since_log for k in keys]))
            now = time.monotonic()
            log(
                {"step": step, "lr": lr_at(step, c.schedule, c.steps)}
                | dict(zip(keys, means.tolist(), strict=True))
                | {"sec_per_step": (now - last_log) / steps_since_log}
            )
            sums.clear()
            steps_since_log, last_log = 0, now
        if evaluate is not None and step % c.eval_every == 0:
            with trainer.ema.swapped_in(model):
                model.eval()
                log({"step": step} | {f"val/{k}": v for k, v in evaluate(model, trainer.autoencoder, step).items()})
                model.train()
        if 0 < step < c.steps - 1 and step % c.checkpoint_every == 0:  # the last step saves below
            if distributed.is_main():
                trainer.save(output_dir, step)
            distributed.barrier()
        if c.max_wall_hours is not None and step % WALL_CLOCK_EVERY == 0:
            # Rank 0's clock decides, so every rank stops at the same step.
            if distributed.broadcast_object(time.monotonic() - loop_start > 3600 * c.max_wall_hours):
                log({"step": step, "stopped": f"max_wall_hours={c.max_wall_hours}"})
                break
    if distributed.is_main():
        trainer.save(output_dir, step, final=True)
    distributed.barrier()

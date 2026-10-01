"""Evaluation of world-model rollouts and of autoencoder reconstructions.

Rollout: a manifest window is 136 frames of a test video with its actions. The autoencoder encodes the first 40
frames, the context; the world model generates the next 96 / tau latents, 10 denoising steps each, with context
noise 0.2; the autoencoder decodes context and rollout as the interactive demo streams them (StreamingDecoder, with
decode_steps 1 or 2), all under bf16 autocast. metrics.RolloutMetrics scores the 96 generated frames against the
recorded ones.

Reconstruction encodes and decodes each 16-frame window of a manifest in fp32, in one decoding step, and scores it
with metrics.ReconstructionMetrics.

Randomness and sharding: every window draws its noise (the rollout's, then the decoder's) from its own generator,
seeded by a hash of the window and the global seed, and rank r scores manifest windows r, r + world_size, ...
So the rollout of a window does not depend on the number of GPUs or on the order of the manifest (at batch_size 1:
a larger batch changes the rounding of the batched matmuls), and the metrics only through the order their sums are
added in. A window listed k times is scored k times, with different noise.
"""

import collections
import hashlib
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dataset import ClipDataset, EnvironmentDataset, read_manifest, read_split
from metrics import ARR_DIMS, SLICE_FRAMES, ReconstructionMetrics, RolloutMetrics
from model_idm import load_probe
from model_world_model import decode_latents, encode_video
from util import distributed

# The reported numbers: every metric on the last 16-frame slice (frames 81-96), except indoor navigation's PSNR, read
# on frames 33-48 (at 96 frames it mostly measures how far the camera moved).
PSNR_FRAMES = {"indoor_navigation": 48}


class RolloutEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_frames: int = 40
    generated_frames: int = 96
    denoise_steps: int = 10
    context_noise: float = 0.2
    decode_steps: int = 1
    decode_alpha: float = 0.99  # AR(1) coefficient of the streaming decoder's noise (StreamingDecoder)
    batch_size: int = 1  # windows per GPU and rollout
    num_workers: int = 4
    seed: int = 0
    compile: bool = True

    @property
    def clip_frames(self) -> int:
        return self.context_frames + self.generated_frames


class ReconstructionEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_frames: int = 16
    decode_steps: int = 1
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 0


def window_seeds(windows: list[tuple[str, int]], seed: int) -> list[int]:
    """One seed per manifest window, from a hash of "<key>@<start>" (its k-th repeat, k >= 1: "<key>@<start>#k")
    and the global seed."""
    repeats = collections.Counter()
    seeds = []
    for key, start in windows:
        name = f"{key}@{start}"
        k, repeats[name] = repeats[name], repeats[name] + 1
        digest = hashlib.sha256(f"{seed}/{name}{f'#{k}' if k else ''}".encode()).digest()
        seeds.append(int.from_bytes(digest[:8], "little") & (2**63 - 1))
    return seeds


def clip_generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device).manual_seed(seed)


class ShardWindows(Dataset):
    """This rank's windows of a manifest ClipDataset (every world_size-th, from `rank`), each with its seed."""

    def __init__(self, dataset: ClipDataset, manifest: str, seed: int, rank: int, world_size: int):
        self.dataset = dataset
        self.seeds = window_seeds(read_manifest(manifest), seed)
        self.indices = range(rank, len(dataset), world_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        index = self.indices[i]
        return self.dataset[index] | {"seed": self.seeds[index]}


def shard_loader(dataset: ClipDataset, manifest: str, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    """This rank's windows of the dataset (ShardWindows), in batches."""
    windows = ShardWindows(dataset, manifest, seed, distributed.rank(), distributed.world_size())
    return DataLoader(windows, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def rollout_noise(shape: tuple[int, ...], generators: list[torch.Generator], device, dtype) -> tuple[Tensor, Tensor]:
    """The rollout's (start, renoise) noise, (B, *shape) each: every clip draws its start noise, then its re-noise."""
    draws = [[torch.randn(shape, generator=g, device=device, dtype=dtype) for _ in range(2)] for g in generators]
    return torch.stack([d[0] for d in draws]), torch.stack([d[1] for d in draws])


@torch.no_grad()
def rollout_frames(
    world_model, autoencoder, video: Tensor, actions: Tensor, generators: list[torch.Generator], config
) -> Tensor:
    """The generated frames (B, generated_frames, 3, H, W), in [0, 1] and not clamped, of uint8 windows video
    (B, clip_frames, 3, H, W) with their actions (B, clip_frames, action_dim), clip b drawing its noise from
    generators[b]."""
    n_new = config.generated_frames // world_model.config.tau
    with torch.autocast(video.device.type, dtype=torch.bfloat16, enabled=video.is_cuda):
        z = encode_video(autoencoder, video[:, : config.context_frames])
        noise = rollout_noise((n_new, *z.shape[2:]), generators, z.device, z.dtype)
        z = world_model.rollout(
            z,
            actions,
            n_latents=n_new,
            denoise_steps=config.denoise_steps,
            context_noise=config.context_noise,
            noise=noise,
        )
        frames = torch.cat(
            [
                decode_latents(autoencoder, z[b : b + 1], config.decode_steps, generator=g, alpha=config.decode_alpha)
                for b, g in enumerate(generators)
            ]
        )
    return frames[:, config.context_frames : config.context_frames + config.generated_frames]


def evaluate_rollout(
    world_model, autoencoder, metrics: RolloutMetrics, loader, config: RolloutEvalConfig, device
) -> dict[str, float]:
    """Roll out every window of the loader (EnvironmentDataset items) and return metrics.compute() (collective: call
    it on every rank)."""
    scored = slice(config.context_frames, config.context_frames + config.generated_frames)
    for batch in loader:
        video, actions = batch["video"].to(device), batch["actions"].to(device)
        generators = [clip_generator(int(s), device) for s in batch["seed"]]
        generated = rollout_frames(world_model, autoencoder, video, actions, generators, config)
        metrics.update(generated, video[:, scored] / 255.0, actions[:, scored])
    return metrics.compute()


@torch.no_grad()
def reconstruct(autoencoder, video: Tensor, decode_steps: int, generators: list[torch.Generator]) -> Tensor:
    """Reconstructions in [0, 1] (not clamped) of uint8 clips (B, T, 3, H, W), clip b's decoder noise from
    generators[b]."""
    z = autoencoder.encode(autoencoder.preprocess(video))
    shape = (video.shape[1], 3, *autoencoder.config.resolution)
    noise = torch.stack([torch.randn(shape, generator=g, device=z.device, dtype=z.dtype) for g in generators])
    return autoencoder.decode(z, decode_steps, noise=noise) * 0.5 + 0.5


def evaluate_reconstruction(
    autoencoder, metrics: ReconstructionMetrics, loader, config: ReconstructionEvalConfig, device
) -> dict[str, float]:
    """Reconstruct every window of the loader and return metrics.compute() (collective)."""
    for batch in loader:
        video = batch["video"].to(device)
        generators = [clip_generator(int(s), device) for s in batch["seed"]]
        metrics.update(reconstruct(autoencoder, video, config.decode_steps, generators), video / 255.0)
    return metrics.compute()


def paper_numbers(results: dict[str, float], env: str | None) -> dict[str, float]:
    """The reported numbers among results: FID, FVD, LPIPS, ARR, its control and gap at 96 frames, PSNR at 96
    (indoor navigation: 48). Without a probe, the ARR entries are absent."""
    keys = ["fid_at_96", "fvd_at_96", f"psnr_at_{PSNR_FRAMES.get(env, 96)}", "lpips_at_96"]
    keys += ["arr_at_96", "arr_control_at_96", "arr_gap_at_96"]
    return {k: results[k] for k in keys if k in results}


def validation_manifest(root, clip_frames: int, n: int, path) -> str:
    """A manifest of the first n val videos long enough for one clip, each from frame 0, written to path."""
    lines = [f"{key} 0" for key, frames in read_split(root, "val") if frames >= clip_frames][:n]
    if distributed.is_main():
        Path(path).write_text("\n".join(lines) + "\n")
    distributed.barrier()
    return str(path)


def rollout_evaluator(env: str, root, n: int, output_dir, device, probe: str | None = None):
    """evaluate(world_model, autoencoder, step) for training: the paper's rollout metrics on n val clips of root."""
    config = RolloutEvalConfig(compile=False)
    probe_model = load_probe(probe).to(device) if probe else None
    manifest = validation_manifest(root, config.clip_frames, n, Path(output_dir) / "val_manifest.txt")

    def evaluate(world_model, autoencoder, step: int) -> dict[str, float]:
        dataset = EnvironmentDataset(env, root, "val", config.clip_frames, autoencoder.config.resolution, manifest)
        metrics = RolloutMetrics(config.generated_frames // SLICE_FRAMES, probe_model, ARR_DIMS.get(env), device)
        loader = shard_loader(dataset, manifest, config.seed, config.batch_size, config.num_workers)
        return paper_numbers(evaluate_rollout(world_model, autoencoder, metrics, loader, config, device), env)

    return evaluate


def reconstruction_evaluator(root, n: int, output_dir, device):
    """evaluate(autoencoder, step) for training: the paper's reconstruction metrics on n val clips of root."""
    config = ReconstructionEvalConfig()
    manifest = validation_manifest(root, config.clip_frames, n, Path(output_dir) / "val_manifest.txt")

    def evaluate(autoencoder, step: int) -> dict[str, float]:
        dataset = ClipDataset(root, "val", config.clip_frames, autoencoder.config.resolution, manifest, actions=False)
        loader = shard_loader(dataset, manifest, config.seed, config.batch_size, config.num_workers)
        return evaluate_reconstruction(autoencoder, ReconstructionMetrics(device), loader, config, device)

    return evaluate

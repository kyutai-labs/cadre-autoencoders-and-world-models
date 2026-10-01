"""Generic clip datasets for every CADRE consumer: autoencoder and world-model training, the IDM probe and the
rollout evaluation. The on-disk format (DATA.md):

    <root>/videos/<key>.mp4      one video per episode, 20 fps, any resolution
    <root>/actions/<key>.npy     optional: float32 (frames, D), actions[j] taken at frame j (it produces frame j + 1)
    <root>/splits/<split>.txt    the split's videos, one "<key> <frames>" line each (split = train, val or test)

A ClipDataset serves num_frames-frame clips of one split: its windows (each video tiled from frame 0 without
overlap, or the (key, start) windows of a manifest), or any (video, start) clip. Training shifts each video's tiling
by a random phase on every pass (MixtureSampler). Frames are resized to cover the target size and center-cropped; a
video already at the target size is used as is.

The world models and the IDM probes read the same datasets of four environments (ENVIRONMENTS), through an
EnvironmentDataset, which checks their action columns and applies the environment's per-clip action transform.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms.v2 import functional as TF

FPS = 20
RESOLUTION = (288, 512)


def normalize_speed(twists: np.ndarray) -> np.ndarray:
    """A clip's camera twists (T, 6), [v in SfM units / s, w in rad/s], with v divided by the median |v| of the clip's
    T - 1 steps (its last row reaches past the clip), or by 1 if that median is 0, as float32: SfM translation has an
    arbitrary scale per scene."""
    twists = twists.astype(np.float64)
    twists[:, :3] /= float(np.median(np.linalg.norm(twists[:-1, :3], axis=1))) or 1.0
    return twists.astype(np.float32)


@dataclass(frozen=True)
class Environment:
    action_dim: int  # the action columns of the environment's datasets (DATA.md)
    clip_actions: Callable[[np.ndarray], np.ndarray] | None = None  # applied to each clip's actions when loaded


ENVIRONMENTS = {
    "robot_manipulation": Environment(action_dim=14),  # ABC: joint-state deltas of the two YAM arms
    "driving": Environment(action_dim=2),  # PhysicalAI-AV: [speed m/s, curvature 1/m]
    "indoor_navigation": Environment(action_dim=6, clip_actions=normalize_speed),  # RE10K: the camera twist [v, w]
    "human_manipulation": Environment(action_dim=102),  # EgoDex: deltas of both hands' poses
}


def read_split(root: str | Path, split: str) -> list[tuple[str, int]]:
    """[(key, frames)] of <root>/splits/<split>.txt."""
    lines = (Path(root) / "splits" / f"{split}.txt").read_text().split("\n")
    return [(key, int(frames)) for key, frames in (line.split() for line in lines if line.strip())]


def read_manifest(path: str | Path) -> list[tuple[str, int]]:
    """[(key, start)] of a manifest: one "<key> <start frame>" line per clip."""
    lines = Path(path).read_text().split("\n")
    return [(key, int(start)) for key, start in (line.split() for line in lines if line.strip())]


def read_frames(path: str | Path, start: int, num_frames: int) -> Tensor:
    """Frames start .. start + num_frames - 1 of a 20 fps video -> uint8 (T, 3, H, W). Frame j is the one presented
    j / 20 s after the stream starts; decoding starts at the last keyframe at or before frame `start`."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin, time_base = stream.start_time or 0, stream.time_base
        container.seek(origin + int(start / FPS / time_base), stream=stream)
        frames = []
        for frame in container.decode(stream):
            if round((frame.pts - origin) * time_base * FPS) < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == num_frames:
                break
    assert len(frames) == num_frames, f"{path}: {len(frames)} of frames {start}..{start + num_frames - 1}"
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def resize_center_crop(frames: Tensor, size: tuple[int, int]) -> Tensor:
    """uint8 (T, 3, H, W) -> (T, 3, *size): an antialiased bilinear resize by the larger of the two scale factors,
    so the frame covers the target, then a center crop of the overflow. No bars, no aspect distortion."""
    h, w = frames.shape[-2:]
    if (h, w) == tuple(size):
        return frames
    scale = max(size[0] / h, size[1] / w)
    frames = TF.resize(frames, [max(size[0], round(h * scale)), max(size[1], round(w * scale))], antialias=True)
    return TF.center_crop(frames, list(size))


class ClipDataset(Dataset):
    """Clips of num_frames frames of one split -> {"video": uint8 (T, 3, H, W), "actions": float32 (T, D) (when
    the dataset has actions and `actions` is set), "key": "<video key>@<start frame>"}.

    Index it with a window number, or with (video number, start frame) for any clip. Without a manifest the windows
    tile each video from frame 0 without overlap, leaving `slack` frames past the last full window; with one, they
    are the manifest's (key, start) windows, each from this split, and no video has slack."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        num_frames: int,
        resolution: tuple[int, int] = RESOLUTION,
        manifest: str | Path | None = None,
        actions: bool = True,
    ):
        self.root, self.num_frames, self.resolution = Path(root), num_frames, tuple(resolution)
        self.with_actions = actions and (self.root / "actions").is_dir()
        self.videos = read_split(root, split)
        frames = torch.tensor([n for _, n in self.videos])
        if manifest is None:
            tiling = [range(0, n - num_frames + 1, num_frames) for _, n in self.videos]
            self.windows = [(v, s) for v, starts in enumerate(tiling) for s in starts]
            self.slack = torch.where(frames >= num_frames, frames % num_frames, 0)
        else:
            number = {key: v for v, (key, _) in enumerate(self.videos)}
            self.windows = [(number[key], start) for key, start in read_manifest(manifest)]
            for v, start in self.windows:
                key, n = self.videos[v]
                assert start + num_frames <= n, f"{key}@{start}: past the end of the video"
            self.slack = torch.zeros(len(self.videos), dtype=torch.long)
        assert self.windows, f"{root}: no {num_frames}-frame clip in split {split}"

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        video, start = self.windows[index] if isinstance(index, int) else index
        key = self.videos[video][0]
        frames = read_frames(self.root / "videos" / f"{key}.mp4", start, self.num_frames)
        item = {"video": resize_center_crop(frames, self.resolution), "key": f"{key}@{start}"}
        if self.with_actions:
            actions = np.load(self.root / "actions" / f"{key}.npy", mmap_mode="r")
            assert actions.ndim == 2 and len(actions) >= start + self.num_frames, f"{key}: actions {actions.shape}"
            item["actions"] = torch.from_numpy(np.array(actions[start : start + self.num_frames], dtype=np.float32))
        return item


class EnvironmentDataset(ClipDataset):
    """A split of one environment's dataset, as the world models and the IDM probes read it: clips with actions, of
    the environment's action_dim columns, passed through its clip_actions."""

    def __init__(
        self,
        environment: str,
        root: str | Path,
        split: str,
        num_frames: int,
        resolution: tuple[int, int] = RESOLUTION,
        manifest: str | Path | None = None,
    ):
        super().__init__(root, split, num_frames, resolution, manifest)
        assert self.with_actions, f"{root}: no actions/"
        self.environment = ENVIRONMENTS[environment]
        columns = np.load(self.root / "actions" / f"{self.videos[0][0]}.npy", mmap_mode="r").shape[1]
        expected = self.environment.action_dim
        assert columns == expected, f"{root}: {columns} action columns, {environment} has {expected}"

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        item = super().__getitem__(index)
        if self.environment.clip_actions is not None:
            item["actions"] = torch.from_numpy(self.environment.clip_actions(item["actions"].numpy()))
        return item


class MixtureSampler(Sampler[tuple[int, int, int]]):
    """Endless stream of (dataset, video, start) clips of a mixture of ClipDatasets, for one rank. Each draw picks a
    dataset with probability proportional to its weight, then the next window of that dataset's current epoch. An
    epoch is a pass over the dataset's windows in a random order, with each video's tiling shifted by a random
    phase, an integer uniform in [0, slack]; order and phases are drawn from
    (seed, dataset, epoch), the same on every rank, and each rank takes every world_size-th window. A run resumed
    at draw `start` continues the stream exactly."""

    def __init__(
        self,
        datasets: list[ClipDataset],
        weights: list[float],
        rank: int,
        world_size: int,
        seed: int,
        start: int = 0,
    ):
        assert len(datasets) == len(weights) and min(len(d) for d in datasets) >= world_size
        self.datasets, self.rank, self.world_size, self.seed, self.start = datasets, rank, world_size, seed, start
        self.weights = torch.tensor(weights, dtype=torch.float64)

    def epoch(self, source: int, epoch: int) -> list[tuple[int, int]]:
        """This rank's (video, start) clips of one epoch of dataset `source`, in order."""
        dataset = self.datasets[source]
        g = torch.Generator().manual_seed(self.seed + 1_000_003 * epoch + 7_919 * source)
        order = torch.randperm(len(dataset), generator=g)
        phase = (torch.rand(len(dataset.slack), generator=g, dtype=torch.float64) * (dataset.slack + 1)).long()
        mine = order[self.rank :: self.world_size][: len(dataset) // self.world_size].tolist()
        return [(v, s + int(phase[v])) for v, s in (dataset.windows[i] for i in mine)]

    def __iter__(self):
        picks = torch.Generator().manual_seed(self.seed + 1_000 * self.rank)
        position = [0] * len(self.datasets)  # draws of each dataset so far
        epochs: dict[int, tuple[int, list[tuple[int, int]]]] = {}  # dataset -> (epoch, this rank's clips)
        drawn = 0
        while True:
            for source in torch.multinomial(self.weights, 4096, replacement=True, generator=picks).tolist():
                epoch, i = divmod(position[source], len(self.datasets[source]) // self.world_size)
                position[source] += 1
                drawn += 1
                if drawn <= self.start:
                    continue
                if epochs.get(source, (None,))[0] != epoch:
                    epochs[source] = (epoch, self.epoch(source, epoch))
                yield (source, *epochs[source][1][i])


class Mixture(Dataset):
    """Datasets indexed by (dataset, video, start), the items of a MixtureSampler."""

    def __init__(self, datasets: list[ClipDataset]):
        self.datasets = datasets

    def __getitem__(self, item: tuple[int, int, int]) -> dict:
        source, video, start = item
        return self.datasets[source][video, start]


def clip_loader(
    datasets: list[ClipDataset],
    weights: list[float],
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    start: int = 0,
    num_workers: int = 6,
) -> DataLoader:
    """Endless per-rank training batches of a weighted mixture of datasets (a single dataset: its shuffled, randomly
    phased epochs); `start` is the number of clips this rank already drew, for a resumed run."""
    return DataLoader(
        Mixture(datasets),
        batch_size=batch_size,
        sampler=MixtureSampler(datasets, weights, rank, world_size, seed, start),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

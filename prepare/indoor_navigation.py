"""Indoor navigation (RE10K): the pixelSplat shards of chenchenshi/re10k-sc -> the release format (DATA.md).

    python prepare/indoor_navigation.py --src <re10k-sc> --out <root> [--split all] [--limit N] [--workers 8]

Reads <src>/train/index.json ({scene: shard file}) and the shards <src>/train/*.torch, each a list of scenes
{"key", "images": (n,) JPEG bytes as uint8 tensors, "cameras": (n, 18) [fx fy cx cy 0 0 | world-to-camera 3x4],
"timestamps": (n,) µs}. Converts the scenes of the paper's split lists (key = the scene id), as the paper read them:

- video: frame j is image round(j * fps / 20), for every j whose image exists, with fps = 1 / median(diff(t)) of
  the scene's timestamps (30 or 24). The JPEGs are decoded as the paper's loader did (libjpeg, torchvision) and kept
  at their size, 640x360.
- actions: the camera twist [v, w] of each frame to the next (DATA.md's camera_twists, copied below): the
  camera-to-world poses (OpenCV axes, the inverse of the world-to-camera matrix, rounded to float32 as the paper
  stored them) of the video's frames, Savitzky-Golay smoothed over the whole video, differenced at the frames'
  actual times; v in SfM units / s (EnvironmentDataset normalizes each clip's), w in rad/s, the last row repeated.
"""

import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import torch
from common import FPS, arguments, run, write_episode
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from torchvision.io import decode_jpeg


def sources(src: Path) -> dict[str, Path]:
    """{scene: its shard} for the shards the download holds."""
    shards = json.loads((src / "train" / "index.json").read_text())
    return {scene: src / "train" / shard for scene, shard in shards.items() if (src / "train" / shard).exists()}


@lru_cache(maxsize=1)
def load_shard(path: Path) -> dict[str, dict]:
    return {scene["key"]: scene for scene in torch.load(path, map_location="cpu", mmap=True)}


def frame_images(t: np.ndarray) -> list[int]:
    """The image of each 20 fps frame: round(j * fps / 20) while that image exists."""
    rate = 1.0 / float(np.median(np.diff(t))) / FPS
    images = []
    while round(len(images) * rate) < len(t):
        images.append(round(len(images) * rate))
    return images


def camera_to_world(cameras: torch.Tensor) -> np.ndarray:
    """(n, 18) pixelSplat camera rows -> (n, 4, 4) float32 camera-to-world poses, OpenCV axes."""
    w2c = np.tile(np.eye(4), (len(cameras), 1, 1))
    w2c[:, :3, :] = cameras[:, 6:].double().numpy().reshape(-1, 3, 4)
    return np.linalg.inv(w2c).astype(np.float32)


def camera_twists(c2w, t):
    """(n, 4, 4) poses and (n,) times in s of a video's 20 fps frames -> actions (n, 6) float32."""
    if len(c2w) >= 7:  # translation filtered directly, rotations entrywise then projected back to SO(3)
        p = savgol_filter(c2w[:, :3, 3], 7, 2, axis=0)
        U, _, Vt = np.linalg.svd(savgol_filter(c2w[:, :3, :3].reshape(-1, 9), 7, 2, axis=0).reshape(-1, 3, 3))
        U[np.linalg.det(U @ Vt) < 0, :, -1] *= -1
        c2w = np.tile(np.eye(4), (len(p), 1, 1))
        c2w[:, :3, :3], c2w[:, :3, 3] = U @ Vt, p
    R, p, dt = c2w[:, :3, :3], c2w[:, :3, 3], np.diff(t)[:, None]
    v = np.einsum("nij,ni->nj", R[:-1], p[1:] - p[:-1]) / dt  # in the camera frame at j
    w = Rotation.from_matrix(R[:-1].transpose(0, 2, 1) @ R[1:]).as_rotvec() / dt
    twists = np.concatenate([v, w], axis=1)
    return np.concatenate([twists, twists[-1:]]).astype(np.float32)


def convert(key: str, shard: Path, count: int, out: Path) -> None:
    scene = load_shard(shard)[key]
    t = scene["timestamps"].double().numpy() / 1e6
    images = frame_images(t)
    poses = camera_to_world(scene["cameras"])[images]
    # A one-frame video (one train scene, too short for any consumer) has no step: its action is zero.
    actions = camera_twists(poses, t[images]) if len(images) > 1 else np.zeros((1, 6))
    frames = (
        av.VideoFrame.from_ndarray(decode_jpeg(scene["images"][i]).permute(1, 2, 0).numpy(), format="rgb24")
        for i in images
    )
    write_episode(out, key, frames, actions, count)


if __name__ == "__main__":
    args = arguments(__doc__).parse_args()
    run("indoor_navigation", args, sources(args.src), convert, chunksize=16)

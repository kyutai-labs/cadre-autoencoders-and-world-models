"""Inverse dynamics probe: reads the commanded actions back from pixels, to score action following (ARR).

A frozen DINOv3 ViT-B/16 gives patch features from layers 2, 5, 8 and 11, combined by a learned softmax. Six factorized
space-time blocks with bidirectional attention follow, then a per-frame attention pool and a two-layer head give one
action per frame, in units standardized by training-set statistics stored with the weights.

Checkpoint: {"config": IDMConfig dict, "state_dict"}, the DINOv3 weights included under `dino.`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from model_autoencoder import load_dinov3
from util import hub

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16  # DINOv3 patch size


class IDMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_dim: int
    window: int = 8  # frames per probe call; one action per frame
    dino_layers: tuple[int, ...] = (2, 5, 8, 11)
    dino_dim: int = 768
    dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0


class LayerMix(nn.Module):
    """Softmax-weighted sum of the DINO layers, then a learned scale and per-channel shift."""

    def __init__(self, num_layers: int, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, feats: list[Tensor]) -> Tensor:
        """feats: one (B, T, C, h, w) map per layer -> (B, T, h, w, C)."""
        stacked = torch.stack(feats, dim=0)
        weights = self.w.softmax(0).view(-1, 1, 1, 1, 1, 1)
        mixed = (weights * stacked).sum(0).movedim(2, -1)
        return self.scale * mixed + self.shift


class Attention(nn.Module):
    """Pre-norm residual self-attention without a mask: the probe is bidirectional."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class SpaceTimeBlock(nn.Module):
    """Temporal attention across the frames at each token position, then spatial attention within each frame."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.temporal = Attention(dim, heads)
        self.spatial = Attention(dim, heads)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x: Tensor, t: int, n: int) -> Tensor:
        b = x.shape[0]
        x = rearrange(x, "b (t n) d -> (b n) t d", t=t, n=n)
        x = rearrange(self.temporal(x), "(b n) t d -> b (t n) d", b=b, n=n)
        x = rearrange(x, "b (t n) d -> (b t) n d", t=t, n=n)
        x = rearrange(self.spatial(x), "(b t) n d -> b (t n) d", b=b, t=t)
        return self.mlp(x)


class FramePool(nn.Module):
    """One learned query per frame, offset by a temporal embedding, attention-pools that frame's tokens."""

    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.zeros(window, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor, t: int, n: int) -> Tensor:
        b = x.shape[0]
        tokens = rearrange(self.norm(x), "b (t n) d -> (b t) n d", t=t, n=n)
        query = (self.query + self.temporal_pos[:t]).unsqueeze(0).expand(b, t, -1).reshape(b * t, 1, -1)
        out, _ = self.attn(query, tokens, tokens, need_weights=False)
        return rearrange(out.squeeze(1), "(b t) d -> b t d", b=b, t=t)


class InverseDynamicsProbe(nn.Module):
    """Maps a window of frames to one action per frame.

    `dino` is the DINOv3 ViT-B/16 backbone (model_autoencoder.load_dinov3: Meta's module, with its
    `get_intermediate_layers`); it stays frozen and in eval mode.
    """

    def __init__(self, config: IDMConfig, dino: nn.Module):
        super().__init__()
        self.config = config
        self.dino = dino.requires_grad_(False).eval()
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD)[None, :, None, None], persistent=False)
        self.layer_mix = LayerMix(len(config.dino_layers), config.dino_dim)
        self.proj = nn.Linear(config.dino_dim, config.dim)
        # Temporal position only: the DINO features already carry the spatial position.
        self.temporal_pos = nn.Parameter(torch.zeros(config.window, config.dim))
        self.blocks = nn.ModuleList(
            SpaceTimeBlock(config.dim, config.heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.pool = FramePool(config.dim, config.heads, config.window)
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.action_dim),
        )
        # Per-dimension action statistics of the training set; the head predicts standardized actions.
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def dino_features(self, frames: Tensor) -> list[Tensor]:
        """(B, T, 3, H, W) in [0, 1] -> one (B, T, C, H/16, W/16) map per layer."""
        b, t, _, h, w = frames.shape
        x = (rearrange(frames, "b t c h w -> (b t) c h w") - self.pixel_mean) / self.pixel_std
        x = F.interpolate(x, (PATCH * (h // PATCH), PATCH * (w // PATCH)), mode="bilinear", antialias=True)
        feats = self.dino.get_intermediate_layers(x, n=self.config.dino_layers, reshape=True, norm=True)
        return [rearrange(f, "(b t) c h w -> b t c h w", b=b, t=t) for f in feats]

    def forward(self, frames: Tensor) -> Tensor:
        """(B, T, 3, H, W) in [0, 1] with T <= window -> (B, T, action_dim) standardized actions."""
        with torch.no_grad():
            feats = self.dino_features(frames)
        mixed = self.layer_mix(feats)
        b, t, h, w, c = mixed.shape
        n = h * w
        x = self.proj(mixed.reshape(b, t, n, c)) + self.temporal_pos[:t].view(1, t, 1, -1)
        x = x.reshape(b, t * n, -1)
        for block in self.blocks:
            x = block(x, t, n)
        return self.head(self.pool(x, t, n))

    @torch.no_grad()
    def recover_actions(self, frames: Tensor) -> Tensor:
        """(B, T, 3, H, W) in [0, 1] -> (B, T, action_dim) actions in raw units, one probe call per window.

        Windows do not overlap, so each frame gets exactly one estimate; a short last window is padded by repeating
        its last frame and trimmed after.
        """
        w = self.config.window
        out = []
        for s in range(0, frames.shape[1], w):
            win = frames[:, s : s + w]
            n = win.shape[1]
            if n < w:
                win = torch.cat([win, win[:, -1:].expand(-1, w - n, -1, -1, -1)], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=frames.is_cuda):
                pred = self(win)
            out.append((pred.float() * self.action_std + self.action_mean)[:, :n])
        return torch.cat(out, dim=1)


def save_probe(probe: InverseDynamicsProbe, path, **extra) -> None:
    torch.save({"config": probe.config.model_dump(), "state_dict": probe.state_dict(), **extra}, path)


def load_probe(path) -> InverseDynamicsProbe:
    """Frozen probe in eval mode, DINOv3 weights included, from a .pt file, a Hugging Face folder or a Hub repo id with
    the environment as subfolder ("kyutai/cadre-action-probes/driving"; util.hub.load_checkpoint)."""
    ckpt = hub.load_checkpoint(path)
    probe = InverseDynamicsProbe(IDMConfig(**ckpt["config"]), load_dinov3())
    probe.load_state_dict(ckpt["state_dict"])
    return probe.eval().requires_grad_(False)


def load_pretrained_probe(path_or_repo_id, subfolder: str | None = None) -> InverseDynamicsProbe:
    """load_probe from a Hugging Face folder (util.hub), in a local directory or a Hub repo id such as
    "kyutai/cadre-action-probes" with the environment as `subfolder`."""
    return load_probe(f"{path_or_repo_id}/{subfolder}" if subfolder else path_or_repo_id)

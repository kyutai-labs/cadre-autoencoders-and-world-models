"""CADRE: causal diffusion representation autoencoder.

The encoder is a frozen DINOv3 ViT-B/16 whose layer-averaged features go through one trained strided convolution, the
bottleneck. The decoder is a space-time transformer trained by flow matching with diffusion forcing: it predicts the
clean clip x from the latent z and x_t = t x + (1 - t) eps (t = 0 is pure noise), one flow time per latent frame.
Clips are (B, T, 3, H, W) in [-1, 1]; latents are (B, T / tau, 64, h, w).

The training pass draws from torch's default generator, as DINOv3's train-mode augmentation does; decode takes its
noise (or a generator) explicitly. DINOv3 stays in eval mode when the model is put in train mode: its own train mode
(a RoPE box-rescale augmentation) is set separately, by the trainer (engine_autoencoder: dino_rope_augmentation).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from util import hub
from util.attention import SelfAttention, init_weights

# DINOv3 from torch.hub, pinned to the commit the released checkpoints were trained with (Meta's DINOv3 License).
DINOV3_HUB = "facebookresearch/dinov3:1e358a23bc05795715534db39fbca26fc442d8c8"
DINO_DIM, PATCH = 768, 16
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SPATIAL_ROPE_THETA, TEMPORAL_ROPE_THETA = 100.0, 64.0
RESOLUTION_MULTIPLE = 32  # adaptive resolution pads H and W to this multiple


class CADREConfig(BaseModel):
    """One CADRE autoencoder. The compression is tau x s x s with s = 16 * stride / unfold."""

    model_config = ConfigDict(extra="forbid")

    tau: int
    stride: int
    unfold: int = 1
    kernel_size: int | None = None  # spatial kernel of the bottleneck; None = stride
    latent_dim: int = 64
    resolution: tuple[int, int] = (288, 512)
    num_frames: int = 16
    dino_layers: tuple[int, ...] = (2, 4, 5, 6, 7, 8, 9, 10, 11)
    dim: int = 1024
    depth: int = 24  # space-time blocks, of which the last tail_depth run at the native frame rate
    tail_depth: int = 4
    num_heads: int = 16
    num_registers: int = 4
    num_time_registers: int = 1
    refine_dim: int = 16
    refine_dilations: tuple[int, ...] = (2, 4)
    # Multi-resolution finetune: keep each clip's own size (padded to a multiple of 32) instead of resizing it,
    # and rescale the decoder's spatial RoPE box by a log-uniform factor in [1/r, r] during training.
    adaptive_resolution: bool = False
    rope_spatial_rescale: float | None = None

    @property
    def latent_grid(self) -> tuple[int, int]:
        """(h, w) of the latents of a frame at `resolution`: its 16-pixel patch grid padded to a multiple of the
        stride, strided, then unfolded."""
        return tuple(-(-(size // PATCH) // self.stride) * self.unfold for size in self.resolution)


CADRE_MODELS = {
    "cadre_1x48x48": CADREConfig(tau=1, stride=3),
    "cadre_2x32x32": CADREConfig(tau=2, stride=2),
    "cadre_4x24x24": CADREConfig(tau=4, stride=3, unfold=2, kernel_size=5),
    "cadre_8x16x16": CADREConfig(tau=8, stride=1),
    "cadre_1x16x16": CADREConfig(tau=1, stride=1),
    "cadre_2x16x16": CADREConfig(tau=2, stride=1),
    "cadre_4x16x16": CADREConfig(tau=4, stride=1),
}


@dataclass
class AutoencoderOutputs:
    """The training pass: the clip x, the decoder's prediction x_hat from x_t, the noise eps of x_t, the latent
    and the encoder's DINO features of x (the target of loss_dino)."""

    x: Tensor
    x_hat: Tensor
    noise: Tensor
    z: Tensor
    features: list[Tensor]


def per_frame(values: Tensor, tau: int) -> Tensor:
    """(B, T_lat) per-latent-frame values -> (B, T, 1, 1, 1), shared by the tau frames of each latent frame."""
    b, t_lat = values.shape
    return values.repeat_interleave(tau, dim=1).view(b, t_lat * tau, 1, 1, 1)


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def _rope_cos_sin(positions: Tensor, dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables (N, dim) with each frequency repeated for a channel pair, as apply_rotary_emb expects."""
    k = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    freqs = positions.float()[:, None] * (theta ** (-2.0 * k / dim))[None, :]
    return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)


def spatial_rope(h: int, w: int, head_dim: int, device, rescale: float | None = None) -> tuple[Tensor, Tensor]:
    """Axial 2D RoPE in DINOv3's convention: pixel-center coordinates normalized per axis to [-1, 1] and scaled by
    2 pi, so a finer grid interpolates positions rather than extrapolating them. Half of each head encodes the row,
    half the column. `rescale` multiplies the coordinates by a log-uniform factor in [1/rescale, rescale]."""
    pos_h = (torch.arange(0.5, h, device=device) / h * 2.0 - 1.0) * (2 * math.pi)
    pos_w = (torch.arange(0.5, w, device=device) / w * 2.0 - 1.0) * (2 * math.pi)
    if rescale is not None:
        log_r = math.log(rescale)
        factor = torch.empty((), device=device).uniform_(-log_r, log_r).exp()
        pos_h, pos_w = pos_h * factor, pos_w * factor
    cos_h, sin_h = _rope_cos_sin(pos_h.repeat_interleave(w), head_dim // 2, SPATIAL_ROPE_THETA)
    cos_w, sin_w = _rope_cos_sin(pos_w.repeat(h), head_dim // 2, SPATIAL_ROPE_THETA)
    return torch.cat([cos_h, cos_w], dim=-1), torch.cat([sin_h, sin_w], dim=-1)


def temporal_rope(n: int, head_dim: int, device) -> tuple[Tensor, Tensor]:
    return _rope_cos_sin(torch.arange(n, device=device), head_dim, TEMPORAL_ROPE_THETA)


def load_dinov3() -> nn.Module:
    """Frozen DINOv3 ViT-B/16 (Meta's module and parameter names) from torch.hub at DINOV3_HUB, without weights."""
    return torch.hub.load(
        DINOV3_HUB, "dinov3_vitb16", pretrained=False, trust_repo=True, skip_validation=True, verbose=False
    ).requires_grad_(False)


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(1e-4 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = 256 * ((int(2 * 4 * dim / 3) + 255) // 256)
        self.swish_linear = nn.Linear(dim, hidden, bias=False)
        self.gate_linear = nn.Linear(dim, hidden, bias=False)
        self.output_linear = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))


class SpaceTimeBlock(nn.Module):
    """Spatial attention within each frame, then causal attention over time, then a SwiGLU MLP. Each sublayer is
    preceded by the spatial AdaLN: the shared per-token (scale, shift) plus this block's learned offset."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.adaln_offset = nn.Parameter(torch.zeros(6, dim))
        self.space_norm = nn.LayerNorm(dim, eps=1e-6)
        self.space_attn = SelfAttention(dim, num_heads, gating=True)
        self.space_ls = LayerScale(dim)
        self.time_norm = nn.LayerNorm(dim, eps=1e-6)
        self.time_attn = SelfAttention(dim, num_heads, gating=True, causal=True)
        self.time_ls = LayerScale(dim)
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim)
        self.mlp_ls = LayerScale(dim)

    def forward(self, x: Tensor, rope_space, rope_time, adaln: Tensor) -> Tensor:
        """x: (B, T, N, C) tokens; adaln: (B, T, N, 6C)."""
        b = x.shape[0]
        m = rearrange(adaln, "b t n (k c) -> k b t n c", k=6) + self.adaln_offset[:, None, None, None, :]
        h = rearrange(self.space_norm(x) * (1 + m[0]) + m[1], "b t n c -> (b t) n c")
        x = x + rearrange(self.space_ls(self.space_attn(h, rope_space)), "(b t) n c -> b t n c", b=b)
        h = rearrange(self.time_norm(x) * (1 + m[2]) + m[3], "b t n c -> (b n) t c")
        x = x + rearrange(self.time_ls(self.time_attn(h, rope_time)), "(b n) t c -> b t n c", b=b)
        return x + self.mlp_ls(self.mlp(self.mlp_norm(x) * (1 + m[4]) + m[5]))


class RefineHead(nn.Module):
    """Residual convolutions at pixel resolution after the unpatchify: per block a dilated depthwise 3x3 + pointwise
    conv, then a depthwise causal temporal conv. The latent enters through a zero-initialized 1x1 skip."""

    def __init__(self, dim: int, out_channels: int, latent_dim: int, dilations: tuple[int, ...]):
        super().__init__()
        self.spatial = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim),
                nn.Conv2d(dim, dim, 1),
            )
            for d in dilations
        )
        self.temporal = nn.ModuleList(nn.Conv3d(dim, dim, (3, 1, 1), groups=dim) for _ in dilations)
        self.latent_skip = nn.Conv2d(latent_dim, dim, 1)
        self.out_norm = nn.GroupNorm(8, dim)
        self.out_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """x: (B, T, dim, H, W); z: the latent repeated to the T frames, (B, T, latent_dim, h, w)."""
        b = x.shape[0]
        # The 1x1 projection commutes with nearest upsampling, so it runs at latent resolution.
        skip = F.interpolate(self.latent_skip(rearrange(z, "b t c h w -> (b t) c h w")), x.shape[-2:], mode="nearest")
        x = x + rearrange(skip, "(b t) c h w -> b t c h w", b=b)
        for spatial, temporal in zip(self.spatial, self.temporal, strict=True):
            h = spatial(rearrange(x, "b t c h w -> (b t) c h w"))
            x = x + rearrange(h, "(b t) c h w -> b t c h w", b=b)
            h = rearrange(x, "b t c h w -> b c t h w")
            h = temporal(torch.cat([h[:, :, :1].repeat(1, 1, 2, 1, 1), h], dim=2))  # causal: repeat the first frame
            x = x + rearrange(h, "b c t h w -> b t c h w")
        h = self.out_conv(F.silu(self.out_norm(rearrange(x, "b t c h w -> (b t) c h w"))))
        return rearrange(h, "(b t) c h w -> b t c h w", b=b)


class Encoder(nn.Module):
    """Frozen DINOv3 features, layer averaging, then the bottleneck: a convolution with kernel (tau, k, k) and stride
    (tau, s', s') (a 2D one for tau = 1). The 4x24x24 cell emits 2x2 channel groups that are reshaped into space.

    DINOv3 is built without weights: a CADRE checkpoint carries them, and training from scratch loads Meta's
    dinov3_vitb16 file with `encoder.dino.load_state_dict(torch.load(path))`."""

    def __init__(self, cfg: CADREConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.kernel_size or cfg.stride
        pad = (k - cfg.stride) // 2
        out_channels = cfg.latent_dim * cfg.unfold**2
        if cfg.tau > 1:
            self.bottleneck = nn.Conv3d(
                DINO_DIM, out_channels, (cfg.tau, k, k), stride=(cfg.tau, cfg.stride, cfg.stride), padding=(0, pad, pad)
            )
        else:
            self.bottleneck = nn.Conv2d(DINO_DIM, out_channels, k, stride=cfg.stride, padding=pad)
        self.apply(init_weights)
        self.dino = load_dinov3()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None], persistent=False)

    def train(self, mode: bool = True) -> "Encoder":
        super().train(mode)
        self.dino.eval()
        return self

    def dino_features(self, x: Tensor) -> list[Tensor]:
        """(B, T, 3, H, W) in [0, 1] -> the normalized patch features of the dino_layers blocks, each
        (B, T, 768, H / 16, W / 16)."""
        b = x.shape[0]
        x = (rearrange(x, "b t c h w -> (b t) c h w") - self.mean) / self.std
        size = (PATCH * (x.shape[-2] // PATCH), PATCH * (x.shape[-1] // PATCH))
        x = F.interpolate(x, size, mode="bilinear", antialias=True)
        features = self.dino.get_intermediate_layers(x, n=self.cfg.dino_layers, norm=True, reshape=True)
        return [rearrange(f, "(b t) c h w -> b t c h w", b=b) for f in features]

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        b = x.shape[0]
        with torch.no_grad():
            features = self.dino_features((x + 1) / 2)
        x = torch.stack(features, dim=0).mean(dim=0) + features[-1]
        # Zero-pad the feature grid (not the image) to a multiple of the stride; the decoder crops the lift back.
        s, u = self.cfg.stride, self.cfg.unfold
        pad_h, pad_w = (-x.shape[-2]) % s, (-x.shape[-1]) % s
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        if self.cfg.tau > 1:
            z = self.bottleneck(rearrange(x, "b t c h w -> b c t h w"))
            z = rearrange(z, "b (c u1 u2) t h w -> b t c (h u1) (w u2)", u1=u, u2=u)
        else:
            z = self.bottleneck(rearrange(x, "b t c h w -> (b t) c h w"))
            z = rearrange(z, "(b t) (c u1 u2) h w -> b t c (h u1) (w u2)", b=b, u1=u, u2=u)
        return z, features


class Decoder(nn.Module):
    """Causal flow decoder g(z, x_t, t, d) on a 16x16 patch grid; d is the jump length (0 = flow matching). The latent
    enters through the lift (summed with the patchified x_t), a spatial AdaLN in every block and a zero-initialized
    skip into the refine head."""

    def __init__(self, cfg: CADREConfig):
        super().__init__()
        self.cfg = cfg
        dim, s, u = cfg.dim, cfg.stride, cfg.unfold
        if u > 1:
            # 4x24x24: fold u x u latent cells into channels, one linear, unfold into s x s tokens (net factor s/u).
            self.lift = nn.Linear(cfg.latent_dim * u * u, dim * s * s)
        else:
            self.lift = nn.ConvTranspose2d(cfg.latent_dim, dim, kernel_size=s, stride=s)
        self.blocks = nn.ModuleList(SpaceTimeBlock(dim, cfg.num_heads) for _ in range(cfg.depth - cfg.tail_depth))
        self.tail_blocks = nn.ModuleList(SpaceTimeBlock(dim, cfg.num_heads) for _ in range(cfg.tail_depth))
        # Spatial registers join every frame; temporal register frames are prepended, so every frame attends them.
        self.registers = nn.Parameter(0.02 * torch.randn(cfg.num_registers, dim))
        self.time_registers = nn.Parameter(0.02 * torch.randn(cfg.num_time_registers, dim))
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.split_frames = nn.Linear(dim, dim * cfg.tau)  # one token per latent frame -> tau tokens, one per frame
        self.unpatchify = nn.Linear(dim, cfg.refine_dim * PATCH * PATCH)
        self.refine_head = RefineHead(cfg.refine_dim, 3, cfg.latent_dim, cfg.refine_dilations)
        self.patchify = nn.Linear(3 * cfg.tau * PATCH * PATCH, dim)
        self.time_embed = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.adaln_head = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.apply(init_weights)
        for layer in (self.patchify, self.adaln_head[1], self.refine_head.latent_skip):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight gradient balancing measures gradient norms on."""
        return self.refine_head.out_conv.weight

    def lift_latent(self, z: Tensor, grid: tuple[int, int]) -> Tensor:
        """(B, T, C, h, w) latent -> (B, T, H/16, W/16, dim) tokens, cropped to the token grid of the clip."""
        s, u = self.cfg.stride, self.cfg.unfold
        if u > 1:
            x = self.lift(rearrange(z, "b t c (h u1) (w u2) -> b t h w (c u1 u2)", u1=u, u2=u))
            x = rearrange(x, "b t h w (s1 s2 d) -> b t (h s1) (w s2) d", s1=s, s2=s)
        else:
            x = self.lift(rearrange(z, "b t c h w -> (b t) c h w"))
            x = rearrange(x, "(b t) c h w -> b t h w c", b=z.shape[0])
        assert x.shape[2] >= grid[0] and x.shape[3] >= grid[1], f"lifted grid {tuple(x.shape[2:4])} < {grid}"
        return x[:, :, : grid[0], : grid[1]]

    def flow_forward(self, z: Tensor, x_t: Tensor, t: Tensor, d: Tensor) -> Tensor:
        """Predict the clean clip from z and x_t (B, T, 3, H, W); t and d are (B, T_lat), one per latent frame."""
        b, t_lat = z.shape[:2]
        lat = self.lift_latent(z, (x_t.shape[-2] // PATCH, x_t.shape[-1] // PATCH))
        c = torch.cat([timestep_embedding(t.reshape(-1), 128), timestep_embedding(d.reshape(-1), 128)], dim=-1)
        c = self.time_embed(c.to(lat.dtype)).reshape(b, t_lat, -1)
        adaln = self.adaln_head(lat + c[:, :, None, None, :])
        patches = rearrange(
            x_t, "b (t pt) c (h p1) (w p2) -> b t h w (c pt p1 p2)", pt=self.cfg.tau, p1=PATCH, p2=PATCH
        )
        return self.run_trunk(lat + self.patchify(patches), adaln, z)

    def run_trunk(self, x: Tensor, adaln: Tensor, z: Tensor) -> Tensor:
        tau, head_dim = self.cfg.tau, self.cfg.dim // self.cfg.num_heads
        rescale = self.cfg.rope_spatial_rescale if self.training else None
        _, t, h, w, _ = x.shape
        rope_space = spatial_rope(h, w, head_dim, x.device, rescale)
        rope_time = temporal_rope(t + self.cfg.num_time_registers, head_dim, x.device)
        x = self._run_blocks(self.blocks, x, adaln, rope_space, rope_time, registers=True)
        # Temporal tail: split every token into tau tokens, one per frame, and run the last blocks at the frame rate.
        x = rearrange(self.split_frames(x), "b t h w (ut c) -> b (t ut) h w c", ut=tau).contiguous()
        adaln = adaln.repeat_interleave(tau, 1)
        rope_space = spatial_rope(h, w, head_dim, x.device, rescale)
        rope_time = temporal_rope(t * tau, head_dim, x.device)
        x = self._run_blocks(self.tail_blocks, x, adaln, rope_space, rope_time, registers=False)
        x = rearrange(self.unpatchify(self.norm_out(x)), "b t h w (c p1 p2) -> b t c (h p1) (w p2)", p1=PATCH, p2=PATCH)
        return torch.tanh(self.refine_head(x, z.repeat_interleave(tau, dim=1)))

    def _run_blocks(self, blocks, x: Tensor, adaln: Tensor, rope_space, rope_time, registers: bool) -> Tensor:
        b, t, h, w, _ = x.shape
        x = rearrange(x, "b t h w c -> b t (h w) c")
        adaln = rearrange(adaln, "b t h w c -> b t (h w) c")
        if registers:
            # Registers carry no position (identity RoPE rows) and no shared modulation, only the block offsets.
            k, kt = self.registers.shape[0], self.time_registers.shape[0]
            x = torch.cat([x, self.registers.to(x.dtype).expand(b, t, k, -1)], dim=2)
            cos, sin = rope_space
            ones = torch.ones(k, cos.shape[1], device=cos.device, dtype=cos.dtype)
            rope_space = (torch.cat([cos, ones]), torch.cat([sin, torch.zeros_like(ones)]))
            adaln = torch.cat([adaln, adaln.new_zeros(b, t, k, adaln.shape[-1])], dim=2)
            x = torch.cat([self.time_registers.to(x.dtype)[None, :, None, :].expand(b, kt, x.shape[2], -1), x], dim=1)
            adaln = torch.cat([adaln.new_zeros(b, kt, adaln.shape[2], adaln.shape[-1]), adaln], dim=1)
        for block in blocks:
            if self.training:
                x = checkpoint(block, x, rope_space, rope_time, adaln, use_reentrant=False)
            else:
                x = block(x, rope_space, rope_time, adaln)
        if registers:
            x = x[:, kt:]
        return rearrange(x[:, :, : h * w], "b t (h w) c -> b t h w c", h=h)


class CADRE(nn.Module):
    """Encoder + decoder, plus the latent statistics a world model standardizes latents with."""

    def __init__(self, config: CADREConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.latent_mean, self.latent_std = 0.0, 1.0

    def preprocess(self, video: Tensor) -> Tensor:
        """uint8 (B, T, 3, H, W) -> float in [-1, 1] at the model resolution: black-pad right or bottom to its aspect
        ratio, then an antialiased bilinear resize. With adaptive_resolution, only pad H and W to a multiple of 32."""
        x = video / 255.0
        h, w = x.shape[-2:]
        if self.config.adaptive_resolution:
            x = F.pad(x, (0, (-w) % RESOLUTION_MULTIPLE, 0, (-h) % RESOLUTION_MULTIPLE))
        else:
            th, tw = self.config.resolution
            if w * th < tw * h:
                x = F.pad(x, (0, round(tw / th * h) - w))
            elif w * th > tw * h:
                x = F.pad(x, (0, 0, 0, round(th / tw * w) - h))
            if x.shape[-2:] != (th, tw):
                x = F.interpolate(x.flatten(0, 1), size=(th, tw), mode="bilinear", antialias=True)
                x = x.unflatten(0, (video.shape[0], -1))
        return (x - 0.5) / 0.5

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, T, 3, H, W) in [-1, 1] with T a multiple of tau -> z: (B, T / tau, 64, h, w)."""
        return self.encoder(x)[0]

    def decode(
        self,
        z: Tensor,
        decode_steps: int,
        noise: Tensor | None = None,
        out_hw: tuple[int, int] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample the clip in decode_steps jumps of equal length from noise (B, T, 3, H, W) ~ N(0, I), drawn here
        unless given. out_hw: the clip size, needed when it differs from config.resolution. Returns [-1, 1]."""
        b, t_lat = z.shape[:2]
        out_h, out_w = out_hw or self.config.resolution
        if noise is None:
            shape = (b, t_lat * self.config.tau, 3, out_h, out_w)
            noise = torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator)
        x = noise
        steps = torch.linspace(0.0, 1.0, decode_steps + 1, device=z.device)
        for i in range(decode_steps):
            r, t = steps[i], steps[i + 1]
            x_hat = self.decoder.flow_forward(
                z,
                x,
                torch.full((b, t_lat), float(r), device=z.device),
                torch.full((b, t_lat), float(t - r), device=z.device),
            )
            x = x + ((t - r) / (1 - r)) * (x_hat - x)
        return x

    def forward(self, video: Tensor) -> AutoencoderOutputs:
        """Training pass on a uint8 clip: encode it and predict it from x_t, with t ~ U(0, 1) per latent frame."""
        x = self.preprocess(video)[:, : self.config.num_frames]
        z, features = self.encoder(x)
        t = torch.rand(z.shape[:2], device=x.device)
        noise = torch.randn_like(x)
        t_frames = per_frame(t, self.config.tau)
        x_hat = self.decoder.flow_forward(z, t_frames * x + (1 - t_frames) * noise, t, torch.zeros_like(t))
        return AutoencoderOutputs(x=x, x_hat=x_hat, noise=noise, z=z, features=features)

    def normalize(self, z: Tensor) -> Tensor:
        """Standardize latents with the scalar statistics measured during autoencoder training."""
        return (z - self.latent_mean) / self.latent_std

    def unnormalize(self, z: Tensor) -> Tensor:
        return self.latent_std * z + self.latent_mean

    def checkpoint(self, name: str) -> dict:
        return {
            "model": name,
            "config": self.config.model_dump(),
            "state_dict": self.state_dict(),
            "latent_mean": torch.tensor(self.latent_mean, dtype=torch.float64),
            "latent_std": torch.tensor(self.latent_std, dtype=torch.float64),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict | str | Path) -> "CADRE":
        """Build from a release checkpoint (a dict or its path) in eval mode, on the default device (wrap the call in
        `with torch.device("cuda"):` to build on the GPU directly)."""
        if not isinstance(checkpoint, dict):
            checkpoint = hub.load_checkpoint(checkpoint)
        model = cls(CADREConfig.model_validate(checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.latent_mean = checkpoint["latent_mean"].item()
        model.latent_std = checkpoint["latent_std"].item()
        return model.eval()

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str | Path, subfolder: str | None = None) -> "CADRE":
        """Build from a Hugging Face folder (util.hub: config.json + model.safetensors), a local directory or a Hub
        repo id such as "kyutai/cadre-2x16x16", like from_checkpoint."""
        return cls.from_checkpoint(f"{path_or_repo_id}/{subfolder}" if subfolder else path_or_repo_id)

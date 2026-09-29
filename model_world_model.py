"""The CADRE world model: a causal diffusion transformer over the standardized latents (B, T, h, w, C) of a frozen
CADRE autoencoder. It trains by flow matching with diffusion forcing plus PSD-U self-distillation, and rolls out one
latent frame at a time with a KV cache.

Actions (B, frames, action_dim) are at frame rate: a[j] is taken at frame j and produces frame j + 1, so latent k >= 1
is conditioned on a[k tau - 1 .. (k + 1) tau - 2] and latent 0 on a learned token. Flow times run from 0 (noise) to
1 (data), z_t = t z_1 + (1 - t) z_0; the model predicts z_1 - z_0 and also receives d, the jump of a denoising step
(0 for plain flow matching).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from model_autoencoder import CADRE
from util import hub
from util.attention import AdaptiveLayerNorm, SelfAttention, init_weights


@dataclass(frozen=True)
class DiTGeometry:
    hidden_dim: int
    depth: int
    num_heads: int
    num_kv_heads: int
    time_attention_every: int  # blocks 0, k, 2k, ... and the last one add causal time attention


WORLD_MODELS = {
    "dit_1.2b": DiTGeometry(hidden_dim=2048, depth=16, num_heads=16, num_kv_heads=4, time_attention_every=4),
}


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str = "dit_1.2b"
    # The autoencoder's latents: tau frames per latent frame, latent_channels channels.
    tau: int
    latent_channels: int = 64
    frames: int = 80  # training clip length; a rollout attends to at most frames // tau latent frames
    action_dim: int
    camera_twist: bool = False  # the actions are camera twists, scaled and keyed by the model (camera_inputs)
    # Per-environment z-score of the continuous actions; without it they are clamped to +-2048 and symlog-squashed.
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    # SD3/RAE shift of the flow-time schedule, in training and sampling. 1.0 = no shift.
    noise_shift: float
    psd_fraction: float = 0.25  # share of each per-GPU batch trained with PSD-U instead of flow matching
    activation_checkpointing: bool = True

    @model_validator(mode="after")
    def _check(self) -> "WorldModelConfig":
        assert self.architecture in WORLD_MODELS, f"unknown architecture {self.architecture!r}"
        assert self.frames % self.tau == 0
        assert (self.action_mean is None) == (self.action_std is None)
        if self.action_mean is not None:
            assert len(self.action_mean) == len(self.action_std) == self.action_dim
        assert not self.camera_twist or (self.action_dim == 6 and self.action_mean is None)
        return self


class Autoencoder(Protocol):
    """What the world model needs from its autoencoder (model_autoencoder.CADRE), which must be in eval mode."""

    config: Any  # its tau, num_frames (the length of a training clip) and resolution

    def preprocess(self, video: Tensor) -> Tensor:
        """uint8 video (B, T, 3, H, W) -> float in [-1, 1] at the autoencoder's resolution."""
        ...

    def encode(self, x: Tensor) -> Tensor:
        """(B, T, 3, H, W) in [-1, 1] -> raw latents (B, T // tau, C, h, w)."""
        ...

    def normalize(self, z: Tensor) -> Tensor:
        """Standardize raw latents with the autoencoder's latent statistics."""
        ...

    def unnormalize(self, z: Tensor) -> Tensor: ...

    def decode(
        self, z: Tensor, decode_steps: int, noise: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        """Raw latents (B, t, C, h, w) -> video in [-1, 1], (B, t * tau, 3, H, W), from noise (drawn from generator
        unless given)."""
        ...


def encode_video(autoencoder: Autoencoder, video: Tensor) -> Tensor:
    """Standardized latents (B, T // tau, h, w, C) of uint8 video (B, T, 3, H, W)."""
    z = autoencoder.encode(autoencoder.preprocess(video))
    return rearrange(autoencoder.normalize(z), "b t c h w -> b t h w c")


class StreamingDecoder:
    """Decodes standardized latents one at a time, as the rollout evaluation and the interactive demo stream them: a
    window of at most the autoencoder's training clip (num_frames / tau latents) slides one latent a step and only the
    newest latent's frames are kept. The decoder's noise is one buffer the size of the window, moved at every step by
    an AR(1) with coefficient alpha."""

    def __init__(
        self,
        autoencoder: Autoencoder,
        z: Tensor,
        decode_steps: int,
        alpha: float,
        generator: torch.Generator | None = None,
    ):
        """z (B, t, h, w, C): the latents so far, whose last ones fill the window."""
        cfg = autoencoder.config
        self.autoencoder, self.decode_steps, self.alpha, self.generator = autoencoder, decode_steps, alpha, generator
        self.window = z[:, -(cfg.num_frames // cfg.tau) :]
        self.noise = self._randn()

    def _randn(self) -> Tensor:
        b, n = self.window.shape[:2]
        shape = (b, n * self.autoencoder.config.tau, 3, *self.autoencoder.config.resolution)
        return torch.randn(shape, generator=self.generator, device=self.window.device, dtype=self.window.dtype)

    def decode_window(self) -> Tensor:
        """The frames of the window's latents, (B, t tau, 3, H, W) in [0, 1]."""
        z = self.autoencoder.unnormalize(rearrange(self.window, "b t h w c -> b t c h w"))
        return self.autoencoder.decode(z, self.decode_steps, noise=self.noise) * 0.5 + 0.5

    def push(self, z: Tensor) -> Tensor:
        """The frames (B, tau, 3, H, W) in [0, 1] of the next latent z (B, 1, h, w, C)."""
        self.window = torch.cat([self.window[:, 1:], z], dim=1)
        self.noise = self.alpha * self.noise + (1 - self.alpha**2) ** 0.5 * self._randn()
        return self.decode_window()[:, -self.autoencoder.config.tau :]


def decode_latents(
    autoencoder: Autoencoder,
    z: Tensor,
    decode_steps: int,
    generator: torch.Generator | None = None,
    alpha: float | None = None,
) -> Tensor:
    """Video in [0, 1] (B, T * tau, 3, H, W) of standardized latents (B, T, h, w, C), the decoder's noise drawn from
    generator: in one decoder call, or with alpha streamed as the rollout evaluation decodes (StreamingDecoder)."""
    if alpha is None:
        z = autoencoder.unnormalize(rearrange(z, "b t h w c -> b t c h w"))
        return autoencoder.decode(z, decode_steps, generator=generator) * 0.5 + 0.5
    cfg = autoencoder.config
    stream = StreamingDecoder(autoencoder, z[:, : cfg.num_frames // cfg.tau], decode_steps, alpha, generator)
    frames = [stream.decode_window()]
    frames += [stream.push(z[:, k : k + 1]) for k in range(stream.window.shape[1], z.shape[1])]
    return torch.cat(frames, dim=1)


def shift_time(t: Tensor, alpha: float) -> Tensor:
    """Noise-schedule shift: multiplies the noise odds (1 - t) / t by alpha; alpha > 1 spends more time near noise."""
    if alpha == 1.0:  # 1 - (1 - t) is not always t in floating point
        return t
    s = 1.0 - t
    s = alpha * s / (1.0 + (alpha - 1.0) * s)
    return 1.0 - s


def psd_split(batch_size: int, psd_fraction: float) -> int:
    """Samples of a per-GPU batch trained with flow matching; the rest get PSD-U. Both terms keep at least one."""
    assert batch_size >= 2, "flow matching and PSD-U need a per-GPU batch of at least 2"
    return min(max(1, round(batch_size * (1 - psd_fraction))), batch_size - 1)


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


# Indoor navigation: the model's inputs are the camera twist x 100 and 10 keys thresholded from it (camera_inputs).
CAMERA_KEYS = ("forward", "back", "left", "right", "up", "down", "fast", "slow", "roll_left", "roll_right")
CAMERA_SCALE = 100.0


def camera_inputs(twist: Tensor) -> tuple[Tensor, Tensor]:
    """Camera twists (..., 6), [v, w] = per-scene normalized velocity and angular rate in rad/s (y points down) -> the
    twist x 100 and the CAMERA_KEYS (..., 10) as 0/1 int64, thresholded on the unscaled twist."""
    twist = twist.float()
    v, w = twist[..., :3], twist[..., 3:]
    speed = torch.linalg.vector_norm(v, dim=-1)
    keys = [v[..., 2] > 0.25, v[..., 2] < -0.25, v[..., 0] < -0.25, v[..., 0] > 0.25, v[..., 1] < -0.25]
    keys += [v[..., 1] > 0.25, speed > 1.5, speed < 0.5, w[..., 2] < -0.15, w[..., 2] > 0.15]
    return twist * CAMERA_SCALE, torch.stack(keys, dim=-1).long()


class ActionEncoder(nn.Module):
    """One embedding per latent frame from its tau actions.

    Continuous actions are normalized and embedded per frame; each discrete key (of a camera twist, camera_inputs) has
    a two-entry embedding, zero-padded to half the width and passed through a linear
    layer (a constant in the other environments). Each half concatenates its tau per-frame embeddings and projects
    them, and an MLP joins the halves.
    """

    max_continuous = 2048.0

    def __init__(self, config: WorldModelConfig, dim: int):
        super().__init__()
        self.tau, self.camera_twist = config.tau, config.camera_twist
        continuous_dim = dim // 2
        discrete_dim = dim - continuous_dim
        n_keys = len(CAMERA_KEYS) if config.camera_twist else 0
        key_dim = 2 ** math.floor(math.log2(discrete_dim / n_keys)) if n_keys else 0
        self.discrete_pad = discrete_dim - n_keys * key_dim

        self.continuous_mlp = nn.Linear(config.action_dim, continuous_dim)
        z_score = config.action_mean is not None
        self.register_buffer("action_mean", torch.tensor(config.action_mean).view(1, 1, -1) if z_score else None)
        self.register_buffer("action_std", torch.tensor(config.action_std).view(1, 1, -1) if z_score else None)
        self.discrete_embeddings = nn.ModuleList(nn.Embedding(2, key_dim) for _ in range(n_keys))
        self.discrete_mlp = nn.Linear(discrete_dim, discrete_dim)
        self.continuous_pool = nn.Linear(self.tau * continuous_dim, continuous_dim)
        self.discrete_pool = nn.Linear(self.tau * discrete_dim, discrete_dim)
        self.joint_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.initial_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))

    def forward(self, actions: Tensor) -> Tensor:
        """actions (B, n tau, action_dim) -> (B, n + 1, dim), the initial token first."""
        batch, steps, _ = actions.shape
        continuous, keys = actions, []
        if self.camera_twist:
            continuous, discrete = camera_inputs(actions)
            keys = [embed(discrete[:, :, k]) for k, embed in enumerate(self.discrete_embeddings)]
        if self.action_mean is not None:
            x = (continuous - self.action_mean) / self.action_std
        else:
            m = self.max_continuous
            x = symlog(continuous.clamp(-m, m)) / symlog(torch.tensor(m, device=continuous.device))
        cont = self.continuous_mlp(x)

        if self.discrete_pad:
            keys.append(continuous.new_zeros(1, 1, self.discrete_pad).expand(batch, steps, -1))
        disc = self.discrete_mlp(torch.cat(keys, dim=-1))

        cont = self.continuous_pool(cont.unflatten(1, (-1, self.tau)).flatten(2))
        disc = self.discrete_pool(disc.unflatten(1, (-1, self.tau)).flatten(2))
        a = self.joint_mlp(torch.cat((cont, disc), dim=-1))
        return torch.cat([self.initial_token.expand(batch, -1, -1), a], dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal features of 1000 t (128 frequencies) and an MLP; one vector per sample and latent frame."""

    def __init__(self, dim: int, n_freq: int = 128):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        """t (B, T) -> (B, T, 1, 1, dim)."""
        b, n = t.shape
        t = 1000 * t.flatten()
        exponent = -math.log(10_000) * torch.arange(self.n_freq, dtype=torch.float32, device=t.device) / self.n_freq
        emb = t[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype)).view(b, n, 1, 1, -1)


class RotaryTables(nn.Module):
    """cos/sin tables of the rotary embeddings: 1D over latent frames (positions / 10, periods up to 64) and axial 2D
    over the latent grid (periods from 2 to 100 tokens), built on the frequencies' device."""

    def __init__(self, head_dim: int):
        super().__init__()
        # Built on the CPU even under a meta-device context (load_checkpoint): they are not in the state_dict.
        arange = torch.arange(head_dim // 2, dtype=torch.float32, device="cpu")
        self.register_buffer("time_freqs", torch.exp(arange * (-math.log(64.0) * 2 / head_dim)), persistent=False)
        n = head_dim // 4
        lo, hi = 2 * math.pi / 100.0, math.pi
        space_freqs = lo * (hi / lo) ** (torch.arange(n, dtype=torch.float32, device="cpu") / max(n - 1, 1))
        self.register_buffer("space_freqs", space_freqs, persistent=False)

    def time(self, n: int) -> tuple[Tensor, Tensor]:
        pos = torch.arange(n, dtype=torch.float32, device=self.time_freqs.device) / 10
        freqs = self.time_freqs * pos[:, None]
        return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)

    def space(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        def axis(size: int) -> tuple[Tensor, Tensor]:
            positions = torch.arange(size, dtype=torch.float32, device=self.space_freqs.device)
            freqs = positions[:, None] * self.space_freqs[None]
            return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)

        (cos_y, sin_y), (cos_x, sin_x) = axis(h), axis(w)
        cos_y, sin_y = (x[:, None, :].expand(h, w, -1) for x in (cos_y, sin_y))
        cos_x, sin_x = (x[None, :, :].expand(h, w, -1) for x in (cos_x, sin_x))
        cos = torch.cat([cos_y, cos_x], dim=-1).reshape(h * w, -1)
        sin = torch.cat([sin_y, sin_x], dim=-1).reshape(h * w, -1)
        return cos, sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 256):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.swish_linear = nn.Linear(dim, hidden, bias=False)
        self.gate_linear = nn.Linear(dim, hidden, bias=False)
        self.output_linear = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))


class SpaceTimeBlock(nn.Module):
    """Bidirectional attention within each latent frame, causal attention across frames at each grid position (in
    some blocks only), then a SwiGLU MLP. Each sub-layer is pre-normalized by AdaLN on the frame's conditioning."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, time_attention: bool):
        super().__init__()
        self.space_norm = AdaptiveLayerNorm(dim, dim)
        self.space_attn = SelfAttention(dim, num_heads, num_kv_heads, gating=True)
        self.time_norm = self.time_attn = None
        if time_attention:
            self.time_norm = AdaptiveLayerNorm(dim, dim)
            self.time_attn = SelfAttention(dim, num_heads, num_kv_heads, gating=True, causal=True)
        self.mlp_norm = AdaptiveLayerNorm(dim, dim)
        self.mlp = SwiGLU(dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        time_rope: tuple[Tensor, Tensor],
        space_rope: tuple[Tensor, Tensor],
        kv_cache: tuple[Tensor, Tensor] | None = None,
        return_kv: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        b, t, h, w, _ = x.shape
        x = rearrange(x, "b t h w c -> (b t) (h w) c")
        x = x + self.space_attn(self.space_norm(x, rearrange(cond, "b t h w c -> (b t) (h w) c")), space_rope)

        new_kv = None
        if self.time_attn is not None:
            x = rearrange(x, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
            cond_time = rearrange(cond, "b t h w c -> (b h w) t c")
            y = self.time_attn(self.time_norm(x, cond_time), time_rope, kv_cache, return_kv)
            if return_kv:
                y, new_kv = y
            x = rearrange(x + y, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        else:
            x = rearrange(x, "(b t) (h w) c -> b t h w c", b=b, t=t, h=h, w=w)

        return x + self.mlp(self.mlp_norm(x, cond)), new_kv


KVCache = list[tuple[Tensor, Tensor] | None]  # one (k, v) per block, None for blocks without time attention


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        geometry = WORLD_MODELS[config.architecture]
        dim = geometry.hidden_dim
        self.input_proj = nn.Linear(config.latent_channels, dim)
        self.time_embed = TimeEmbedding(dim)
        self.step_embed = TimeEmbedding(dim)
        self.blocks = nn.ModuleList(
            SpaceTimeBlock(
                dim,
                geometry.num_heads,
                geometry.num_kv_heads,
                time_attention=i % geometry.time_attention_every == 0 or i == geometry.depth - 1,
            )
            for i in range(geometry.depth)
        )
        self.head = nn.Linear(dim, config.latent_channels)
        self.action_encoder = ActionEncoder(config, dim)
        self.rope = RotaryTables(dim // geometry.num_heads)
        self.apply(init_weights)

    def encode_actions(self, actions: Tensor, first: int, n: int) -> Tensor:
        """Action embeddings (B, n, dim) of latent frames first .. first + n - 1 of a clip whose actions start at its
        first frame. The first of the n always gets the initial token, as in training."""
        tau = self.config.tau
        lo, hi = first * tau + tau - 1, (first + n - 1) * tau + tau - 1
        return self.action_encoder(actions[:, lo:hi])

    def velocity(
        self,
        z: Tensor,
        a: Tensor,
        t: Tensor,
        d: Tensor | None = None,
        kv_cache: KVCache | None = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """Predicted velocity (B, T, h, w, C) of latents z at flow times t and jumps d (B, T), given action embeddings a
        (B, T, dim). With a kv_cache of earlier frames, z holds only the newest frame."""
        b, n, h, w, _ = z.shape
        x = self.input_proj(z)
        t_emb = self.time_embed(t) + self.step_embed(torch.zeros_like(t) if d is None else d)
        cond = rearrange(a, "b t c -> b t 1 1 c").repeat(1, 1, h, w, 1) + t_emb.repeat(1, 1, h, w, 1)

        time_rope = self.rope.time(n if kv_cache is None else n + kv_cache[0][0].shape[1])
        space_rope = self.rope.space(h, w)
        new_kv = []
        for i, block in enumerate(self.blocks):
            if self.training and self.config.activation_checkpointing and torch.is_grad_enabled():
                x, _ = checkpoint(block, x, cond, time_rope, space_rope, use_reentrant=False)
            else:
                x, kv = block(x, cond, time_rope, space_rope, None if kv_cache is None else kv_cache[i], return_kv)
                new_kv.append(kv)
        v = self.head(x)
        return (v, new_kv) if return_kv else v

    def forward(self, z: Tensor, actions: Tensor, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Training losses on standardized latents z (B, T, h, w, C) of clips with their actions (B, >= T tau - 1,
        action_dim): loss_fm on the first psd_split(B) samples, loss_psd on the others, and their sum."""
        a = self.encode_actions(actions, 0, z.shape[1])
        n_fm = psd_split(z.shape[0], self.config.psd_fraction)
        loss_fm = self.loss_fm(z[:n_fm], a[:n_fm], generator)
        loss_psd = self.loss_psd(z[n_fm:], a[n_fm:], generator)
        return {"loss": loss_fm + loss_psd, "loss_fm": loss_fm, "loss_psd": loss_psd.detach()}

    def _noise(self, z: Tensor, generator: torch.Generator | None) -> Tensor:
        return torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)

    def _times(self, z: Tensor, generator: torch.Generator | None) -> Tensor:
        return torch.rand(z.shape[:2], generator=generator, device=z.device)

    def loss_fm(self, z: Tensor, a: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Flow matching under diffusion forcing: every latent frame draws its own (shifted) flow time."""
        z0 = self._noise(z, generator)
        v = z - z0
        t = shift_time(self._times(z, generator), self.config.noise_shift)
        t5 = t[..., None, None, None]
        pred = self.velocity(t5 * z + (1 - t5) * z0, a, t)
        return ((pred.float() - v.float()) ** 2).mean()

    def loss_psd(self, z: Tensor, a: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """PSD-U: the one-jump velocity from r to t regresses onto the stop-gradient two-hop target through
        u = g r + (1 - g) t, g ~ U[0, 1], with the current weights as teacher. (r, t) is uniform on r < t, then
        shifted, per latent frame."""
        z0 = self._noise(z, generator)
        t1, t2 = self._times(z, generator), self._times(z, generator)
        r = shift_time(torch.min(t1, t2), self.config.noise_shift)[..., None, None, None]
        t = shift_time(torch.max(t1, t2), self.config.noise_shift)[..., None, None, None]
        z_r = r * z + (1 - r) * z0
        g = torch.rand(r.shape, generator=generator, device=z.device)
        u = g * r + (1 - g) * t
        pred = self.velocity(z_r, a, r.flatten(1), (t - r).flatten(1))
        with torch.no_grad():
            v_ru = self.velocity(z_r, a, r.flatten(1), (u - r).flatten(1))
            v_ut = self.velocity(z_r + (u - r) * v_ru, a, u.flatten(1), (t - u).flatten(1))
            target = (1 - g) * v_ru + g * v_ut
        return ((pred.float() - target.float()) ** 2).mean()

    @torch.no_grad()
    def rollout(
        self,
        z: Tensor,
        actions: Tensor,
        *,
        n_latents: int,
        denoise_steps: int = 10,
        context_noise: float = 0.2,
        generator: torch.Generator | None = None,
        noise: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Extend the context latents z (B, n_ctx, h, w, C) by n_latents frames: (B, n_ctx + n_latents, h, w, C).

        The actions start at the first context frame, at frame rate, and cover (n_ctx + n_latents) tau - 1 frames. Each
        new frame attends to the n_ctx frames before it through the KV cache (next_latent). noise = (start, renoise),
        each (B, n_latents, h, w, C), else drawn from generator.
        """
        b, n_ctx = z.shape[:2]
        assert n_ctx < self.config.frames // self.config.tau, "the window would exceed the trained length"
        if noise is None:
            shape = (b, n_latents, *z.shape[2:])
            noise = tuple(torch.randn(shape, generator=generator, device=z.device, dtype=z.dtype) for _ in range(2))
        start, renoise = noise

        latents, kv = [z], None
        for k in range(n_latents):
            a = self.encode_actions(actions, k, n_ctx + 1)
            if kv is None:
                kv = self.context_cache(z, a[:, :-1])
            x, kv = self.next_latent(
                kv, a[:, -1:], start[:, k : k + 1], renoise[:, k : k + 1], denoise_steps, context_noise
            )
            latents.append(x)
        return torch.cat(latents, dim=1)

    @torch.no_grad()
    def context_cache(self, z: Tensor, a: Tensor) -> KVCache:
        """KV cache of clean latents z (B, n, h, w, C) with their action embeddings a (B, n, dim)."""
        _, kv = self.velocity(z, a, torch.ones(z.shape[:2], device=z.device, dtype=z.dtype), return_kv=True)
        # Copies: under CUDA graphs (torch.compile mode="reduce-overhead") velocity's outputs live in the graph's
        # memory, which its next replay overwrites, and the cache outlives the call.
        return [None if c is None else (c[0].clone(), c[1].clone()) for c in kv]

    @torch.no_grad()
    def next_latent(
        self,
        kv: KVCache,
        a: Tensor,
        noise: Tensor,
        renoise: Tensor,
        denoise_steps: int = 10,
        context_noise: float = 0.2,
    ) -> tuple[Tensor, KVCache]:
        """The latent frame (B, 1, h, w, C) after the cached ones, given its action embedding a (B, 1, dim): integrated
        from noise in denoise_steps Euler steps of the shifted linear schedule. Returns it with the cache moved one
        frame on: the oldest frame dropped, the new one appended as seen at flow time 1 - context_noise (re-noised with
        renoise)."""
        b = noise.shape[0]
        times = shift_time(torch.linspace(0, 1.0, denoise_steps + 1, device=noise.device), self.config.noise_shift)
        x = noise
        for t, dt in zip(times[:-1], times[1:] - times[:-1], strict=True):
            t_k = t * torch.ones((b, 1), device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, a, t_k, dt * torch.ones_like(t_k), kv_cache=kv)

        t_ctx = (1 - context_noise) * torch.ones((b, 1, 1, 1, 1), device=x.device, dtype=x.dtype)
        _, new_kv = self.velocity(t_ctx * x + (1 - t_ctx) * renoise, a, t_ctx.view(b, 1), kv_cache=kv, return_kv=True)
        kv = [
            None if old is None else tuple(torch.cat([o[:, 1:], n], dim=1) for o, n in zip(old, new, strict=True))
            for old, new in zip(kv, new_kv, strict=True)
        ]
        return x, kv


def checkpoint_dict(config: WorldModelConfig, state_dict: dict, autoencoder_checkpoint: dict) -> dict:
    """Standalone checkpoint: the world model's config and weights, plus its autoencoder's full checkpoint
    (CADRE.checkpoint)."""
    return {"config": config.model_dump(), "state_dict": state_dict, "autoencoder": autoencoder_checkpoint}


def save_checkpoint(path: str | Path, model: WorldModel, autoencoder_checkpoint: dict) -> None:
    torch.save(checkpoint_dict(model.config, model.state_dict(), autoencoder_checkpoint), path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[WorldModel, dict]:
    """The world model on `device`, and its autoencoder's checkpoint dict (build it with CADRE.from_checkpoint)."""
    ckpt = hub.load_checkpoint(path, device)
    with torch.device("meta"):
        model = WorldModel(WorldModelConfig.model_validate(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"], assign=True)
    return model.to(device), ckpt["autoencoder"]


def save_pretrained(checkpoint: dict, directory: str | Path, environment: str) -> None:
    """A release checkpoint in the Hugging Face layout (util.hub): the world model in <directory>/<environment>/ and
    its autoencoder in <directory>/autoencoder/, the one folder shared by the world models of a repo."""
    world_model = {"config": checkpoint["config"], "environment": environment, "autoencoder": "autoencoder"}
    hub.save_pretrained(world_model | {"state_dict": checkpoint["state_dict"]}, Path(directory) / environment)
    hub.save_pretrained(checkpoint["autoencoder"], Path(directory) / "autoencoder")


def load_pretrained(
    path_or_repo_id: str | Path, subfolder: str, device: str | torch.device = "cpu"
) -> tuple[WorldModel, CADRE]:
    """The world model of `subfolder` (its environment) in a local directory or a Hub repo id such as
    "kyutai/cadre-world-models-2x16x16", and its autoencoder, both on `device` in eval mode."""
    world_model, autoencoder_checkpoint = load_checkpoint(f"{path_or_repo_id}/{subfolder}", device)
    with torch.device(device):
        autoencoder = CADRE.from_checkpoint(autoencoder_checkpoint)
    return world_model.eval(), autoencoder

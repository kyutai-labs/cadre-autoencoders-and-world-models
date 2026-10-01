"""CADRE autoencoder losses.

The flow pass is supervised by l_rec = l1 + lambda_lpips l_lpips + lambda_dino l_dino + w_dwt lambda_dwt l_dwt, each
lambda set by gradient balancing. On a random fraction of steps (psd_prob) the decoder also distills its own two half
jumps into one jump, L_psd. Random draws (the PSD gate and times, the scored frames, DINOv3's train-mode augmentation)
come from torch's default generator.
"""

import math

import lpips
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from model_autoencoder import CADRE, AutoencoderOutputs, Decoder, per_frame


class AutoencoderLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lpips_weight: float = 1.0
    dino_weight: float = 1.0
    dwt_weight: float = 0.25
    psd_weight: float = 1.0
    lpips_frame_frac: float = 0.25  # LPIPS and DINO each score a random quarter of the frames
    dino_frame_frac: float = 0.25
    dwt_levels: int = 2
    dwt_lowpass_weight: float = 0.25
    psd_prob: float = 0.25  # fraction of steps that add L_psd
    grad_balance_max: float = 1e4


def haar_filters() -> Tensor:
    """The 8 orthonormal 3D Haar filters (t, h, w), band 0 the low-pass one: (8, 1, 2, 2, 2)."""
    lo = torch.tensor([1.0 / math.sqrt(2.0)] * 2)
    hi = torch.tensor([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)])
    bands = [
        ft[:, None, None] * fh[None, :, None] * fw[None, None, :]
        for ft in (lo, hi)
        for fh in (lo, hi)
        for fw in (lo, hi)
    ]
    return torch.stack(bands, dim=0).unsqueeze(1)


def _dwt_l1(filters: Tensor, x_hat: Tensor, x: Tensor, levels: int, lowpass_weight: float) -> Tensor:
    """L1 between Haar coefficients of (B, T, C, H, W) clips, recursing on the low-pass band. Detail bands are
    weighted 7/8 and the final low-pass band lowpass_weight / 8, so levels = 1, lowpass_weight = 1 is the plain L1
    over all 8 bands."""
    a, b = (rearrange(v.float(), "b t c h w -> b c t h w") for v in (x_hat, x))
    weight = filters.to(dtype=a.dtype, device=a.device)
    loss = a.new_zeros(())
    for _ in range(levels):
        a, b = (F.conv3d(v, weight, stride=2, groups=3) for v in (a, b))
        is_lowpass = torch.arange(a.shape[1], device=a.device) % 8 == 0
        loss = loss + (7.0 / 8.0) * F.l1_loss(a[:, ~is_lowpass], b[:, ~is_lowpass])
        a, b = a[:, is_lowpass], b[:, is_lowpass]
    return loss + lowpass_weight * (1.0 / 8.0) * F.l1_loss(a, b)


def loss_dwt(filters: Tensor, x_hat: Tensor, x: Tensor, levels: int, lowpass_weight: float) -> Tensor:
    """3D Haar wavelet L1 (as in LTX-Video), averaged with the same loss on the clip shifted by one frame: the
    stride-2 temporal pairs then also cover the transitions between them."""
    loss = _dwt_l1(filters, x_hat, x, levels, lowpass_weight)
    return 0.5 * (loss + _dwt_l1(filters, x_hat[:, 1:], x[:, 1:], levels, lowpass_weight))


def random_frames(num_frames: int, frac: float, device) -> Tensor:
    k = max(1, round(num_frames * frac))
    return torch.randperm(num_frames, device=device)[:k].sort().values


def loss_lpips(net: nn.Module, x_hat: Tensor, x: Tensor, frames: Tensor) -> Tensor:
    """LPIPS (VGG) on the given frames; inputs in [-1, 1]."""
    pred = rearrange(x_hat[:, frames], "b t c h w -> (b t) c h w")
    return net(pred, rearrange(x[:, frames], "b t c h w -> (b t) c h w")).mean()


def loss_dino(model: CADRE, x_hat: Tensor, features: list[Tensor], frames: Tensor) -> Tensor:
    """Squared distance between L2-normalized DINOv3 features of the prediction and of the clip, averaged over the
    encoder's feature layers; the clip's features are the encoder's own."""
    pred = model.encoder.dino_features(((x_hat + 1) / 2)[:, frames])
    terms = [
        F.mse_loss(F.normalize(p, dim=2, eps=1e-6), F.normalize(f[:, frames].detach(), dim=2, eps=1e-6))
        for p, f in zip(pred, features, strict=True)
    ]
    return torch.stack(terms).mean()


def grad_balance_weight(anchor: Tensor, other: Tensor, last_layer: nn.Parameter, max_weight: float) -> Tensor:
    """lambda = sg[||grad_W anchor|| / (||grad_W other|| + 1e-6)], clipped to max_weight. Rank-local: each rank
    balances its own batch and DDP averages the balanced gradients."""
    anchor_grad = torch.autograd.grad(anchor, last_layer, retain_graph=True)[0]
    other_grad = torch.autograd.grad(other, last_layer, retain_graph=True)[0]
    with torch.autocast(device_type=anchor_grad.device.type, enabled=False):
        ratio = anchor_grad.float().norm() / (other_grad.float().norm() + 1e-6)
    return ratio.clamp(0.0, max_weight).detach()


def psd_step(prob: float, device) -> bool:
    """Whether this step adds L_psd. Every rank draws, then all follow rank 0 so they run the same passes."""
    step = torch.rand((), device=device) < prob
    if dist.is_initialized():
        dist.broadcast(step, src=0)
    return bool(step)


def loss_psd(decoder: Decoder, z: Tensor, x: Tensor, noise: Tensor) -> Tensor:
    """Progressive self-distillation: the jump r -> t must land where two half jumps through
    u = (r + t) / 2 do, the halves taken by the current weights without gradient. Velocities are
    v = (x_hat - x_r) / (1 - r); the regression is on (1 - r) v. r < t are sorted draws per latent frame, and x_r
    reuses the flow pass's noise."""
    shape, tau = z.shape[:2], decoder.cfg.tau
    r1, r2 = torch.rand(shape, device=x.device), torch.rand(shape, device=x.device)
    r, t = torch.min(r1, r2), torch.max(r1, r2)
    u = (r + t) / 2
    r_frames, u_frames = per_frame(r, tau), per_frame(u, tau)
    x_r = r_frames * x + (1 - r_frames) * noise
    x_rt = decoder.flow_forward(z, x_r, r, t - r)
    with torch.no_grad():
        x_ru = decoder.flow_forward(z, x_r, r, (t - r) / 2)
        x_u = x_r + ((u_frames - r_frames) / (1 - r_frames)) * (x_ru - x_r)
        x_ut = decoder.flow_forward(z, x_u, u, t - u)
        v_target = 0.5 * ((x_ru - x_r) / (1 - r_frames) + (x_ut - x_u) / (1 - u_frames))
    v_rt = (x_rt - x_r) / (1 - r_frames)
    return F.mse_loss(((1 - r_frames) * v_rt).float(), ((1 - r_frames) * v_target).float())


class AutoencoderLoss(nn.Module):
    """All autoencoder losses of one training step, from the model's training pass."""

    def __init__(self, config: AutoencoderLossConfig):
        super().__init__()
        self.config = config
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)
        self.register_buffer("haar", haar_filters().repeat(3, 1, 1, 1, 1), persistent=False)

    def forward(self, model: CADRE, out: AutoencoderOutputs) -> dict[str, Tensor]:
        """Returns the unweighted terms, the balancing weights lambda_* and loss_total."""
        cfg = self.config
        x_hat, x = out.x_hat.float(), out.x.float()
        losses = {}
        if model.training and psd_step(cfg.psd_prob, x.device):
            losses["loss_psd"] = loss_psd(model.decoder, out.z, out.x, out.noise)
        losses["loss_l1"] = F.l1_loss(x_hat, x)
        losses["loss_dwt"] = loss_dwt(self.haar, x_hat, x, cfg.dwt_levels, cfg.dwt_lowpass_weight)
        frames = random_frames(x.shape[1], cfg.lpips_frame_frac, x.device)
        losses["loss_lpips"] = loss_lpips(self.lpips, x_hat, x, frames)
        frames = random_frames(x.shape[1], cfg.dino_frame_frac, x.device)
        losses["loss_dino"] = loss_dino(model, x_hat, out.features, frames)

        balanced = {name: losses[f"loss_{name}"] for name in ("dwt", "lpips", "dino")}
        if torch.is_grad_enabled():
            for name in ("lpips", "dino", "dwt"):
                lam = grad_balance_weight(
                    losses["loss_l1"], balanced[name], model.decoder.last_layer, cfg.grad_balance_max
                )
                losses[f"lambda_{name}"] = lam
                balanced[name] = lam * balanced[name]
        terms = [cfg.psd_weight * losses["loss_psd"]] if "loss_psd" in losses else []
        terms += [
            losses["loss_l1"],
            cfg.dwt_weight * balanced["dwt"],
            cfg.lpips_weight * balanced["lpips"],
            cfg.dino_weight * balanced["dino"],
        ]
        losses["loss_total"] = sum(terms)
        return losses

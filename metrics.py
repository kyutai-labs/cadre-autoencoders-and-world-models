"""Evaluation metrics for world-model rollouts and autoencoder reconstructions.

`RolloutMetrics` scores each 16-frame slice of a rollout: FID (per-frame Inception features), FVD (I3D logits per
slice), PSNR and LPIPS against the recorded future, and ARR, the action recoverability ratio read by a frozen inverse
dynamics probe. `ReconstructionMetrics` uses pytorch-fid's FID and cd-fvd's FVD, different estimators from the rollout
ones. Both accumulate per-rank sums in `update` and reduce them in `compute` (a collective: call it on every rank), so
results do not depend on the number of GPUs.
"""

import lpips
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from cdfvd.third_party.i3d.utils import load_i3d_model, preprocess_i3d
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch import Tensor

from model_idm import InverseDynamicsProbe

SLICE_FRAMES = 16
INCEPTION_DIM = 2048
I3D_DIM = 400
FID_BATCH = 64  # frames per Inception call in the reconstruction eval
I3D_BATCH = 16  # clips per I3D call in the reconstruction eval (cd-fvd's)
# A dimension the probe cannot read on real video says nothing about a rollout: it is left out of ARR.
MIN_CEILING = 0.05
# Action dimensions ARR is reported on (all when absent). EgoDex: the 12 wrist dims (translation 0:3 and rotation 3:6
# of each hand, hands at 0 and 51); the probe barely reads the finger joints.
ARR_DIMS = {"human_manipulation": [0, 1, 2, 3, 4, 5, 51, 52, 53, 54, 55, 56]}


def load_inception(device) -> torch.nn.Module:
    """pytorch-fid's InceptionV3 with the TF FID weights: (N, 3, H, W) in [0, 1] -> [(N, 2048, 1, 1)]."""
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_DIM]]).to(device).eval().requires_grad_(False)


def load_i3d(device) -> torch.nn.Module:
    """cd-fvd's Kinetics-400 I3D: (N, 3, T, 224, 224) in [-1, 1] -> (N, 400) logits."""
    return load_i3d_model(torch.device(device)).eval().requires_grad_(False)


def load_lpips(device) -> torch.nn.Module:
    """LPIPS on AlexNet features: two (N, 3, H, W) in [-1, 1] -> (N, 1, 1, 1)."""
    return lpips.LPIPS(net="alex", verbose=False).to(device).requires_grad_(False)


def all_reduce(tensors: list[Tensor]) -> None:
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t)


def i3d_features(i3d: torch.nn.Module, clips: Tensor) -> Tensor:
    """(N, T, 3, H, W) in [0, 1] -> (N, 400): rounded to uint8, then cd-fvd's preprocessing (224 px, [-1, 1])."""
    u8 = (clips.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
    return i3d(preprocess_i3d(u8).to(device=clips.device, dtype=torch.float32)).flatten(1)


class GaussianStats:
    """Per-slice float64 sums of x and x x^T: the sufficient statistics of a Gaussian fit to features."""

    def __init__(self, num_slices: int, dim: int, device):
        self.sum = torch.zeros(num_slices, dim, dtype=torch.double, device=device)
        self.outer = torch.zeros(num_slices, dim, dim, dtype=torch.double, device=device)
        self.n = torch.zeros(num_slices, dtype=torch.long, device=device)

    def tensors(self) -> list[Tensor]:
        return [self.sum, self.outer, self.n]

    def update(self, s: int, x: Tensor) -> None:
        x = x.double()
        self.n[s] += x.shape[0]
        self.sum[s] += x.sum(dim=0)
        self.outer[s] += x.T @ x

    def mean_cov(self, s: int, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        """Mean and unbiased covariance of slice s, regularized by eps * I."""
        assert self.n[s] > 1, "a Gaussian fit needs at least 2 samples"
        n = self.n[s].double()
        mean = self.sum[s] / n
        cov = (self.outer[s] - n * torch.outer(mean, mean)) / (n - 1.0)
        return mean, cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)


def frechet_distance(real: tuple[Tensor, Tensor], generated: tuple[Tensor, Tensor]) -> float:
    """||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 (C_1 C_2)^(1/2)) of two (mean, covariance) pairs."""
    mean1, cov1 = (t.cpu().numpy() for t in real)
    mean2, cov2 = (t.cpu().numpy() for t in generated)
    diff = mean1 - mean2
    cov_prod = np.asarray(linalg.sqrtm(cov1.dot(cov2)))
    if np.iscomplexobj(cov_prod):
        cov_prod = np.real(cov_prod)
    return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2 * cov_prod))


class R2Stats:
    """Per-slice, per-dimension sums for the R^2 of recovered against commanded actions over the whole eval set:
    a per-batch R^2 would divide by the variance of a single clip."""

    def __init__(self, num_slices: int, dim: int, device):
        self.sse = torch.zeros(num_slices, dim, device=device)
        self.sum = torch.zeros(num_slices, dim, device=device)
        self.sum_sq = torch.zeros(num_slices, dim, device=device)
        self.n = torch.zeros(num_slices, device=device)

    def tensors(self) -> list[Tensor]:
        return [self.sse, self.sum, self.sum_sq, self.n]

    def update(self, s: int, recovered: Tensor, commanded: Tensor) -> None:
        r = recovered.reshape(-1, recovered.shape[-1]).float()
        a = commanded.reshape(-1, commanded.shape[-1]).float()
        self.sse[s] += ((r - a) ** 2).sum(0)
        self.sum[s] += a.sum(0)
        self.sum_sq[s] += (a * a).sum(0)
        self.n[s] += a.shape[0]

    def r2(self) -> Tensor:
        """(num_slices, dim) R^2 = 1 - MSE / Var(commanded)."""
        n = self.n.clamp_min(1).unsqueeze(-1)
        var = (self.sum_sq / n - (self.sum / n) ** 2).clamp_min(1e-12)
        return 1 - (self.sse / n) / var


def dim_mean(r2: Tensor, dims: list[int] | None) -> list[float]:
    """Per-slice mean of R^2 over the reported dimensions (all when dims is None), each clamped at -1 first."""
    return (r2 if dims is None else r2[:, dims]).clamp_min(-1.0).mean(dim=-1).tolist()


class RolloutMetrics:
    """FID, FVD, PSNR, LPIPS and ARR of world-model rollouts, per 16-frame slice of the generated horizon.

    update(generated, real, actions) takes the scored span only: the num_slices * 16 frames after the context.
      generated: (B, T, 3, H, W) float, the decoded rollout mapped to [0, 1] by x * 0.5 + 0.5 and NOT clamped; PSNR,
        LPIPS, FID and the probe read it as is, FVD rounds it to uint8.
      real: (B, T, 3, H, W) float, the recorded frames uint8 / 255.
      actions: (B, T, D) commanded actions, actions[:, j] taken at frame j; needed with a probe.
    compute() -> fid_at_K, fvd_at_K, psnr_at_K, lpips_at_K for K = 16, 32, ..., and with a probe:
      arr_raw_at_K      mean R^2 of the recovered against the commanded actions (see `dim_mean`)
      arr_control_at_K  the same against the clip's own commands shifted by half the horizon
      arr_gap_at_K      arr_raw - arr_control
      arr_real_at_K     the probe's mean R^2 on the real frames
      arr_at_K          ARR = mean over the dims d with ceiling_d > MIN_CEILING of max(R^2_d, -1) / ceiling_d, where
                        ceiling_d is the probe's R^2_d on the real frames averaged over the slices
      arr_ceiling       mean ceiling_d over the reported dims
    `arr_dims` restricts the ARR scores to a subset of the action dimensions (ARR_DIMS).
    """

    def __init__(
        self,
        num_slices: int = 6,
        probe: InverseDynamicsProbe | None = None,
        arr_dims: list[int] | None = None,
        device="cuda",
    ):
        self.num_slices = num_slices
        self.probe = probe
        self.arr_dims = arr_dims
        self.inception = load_inception(device)
        self.i3d = load_i3d(device)
        self.lpips = load_lpips(device)
        self.fid_real = GaussianStats(num_slices, INCEPTION_DIM, device)
        self.fid_generated = GaussianStats(num_slices, INCEPTION_DIM, device)
        self.fvd_real = GaussianStats(num_slices, I3D_DIM, device)
        self.fvd_generated = GaussianStats(num_slices, I3D_DIM, device)
        self.psnr_sum = torch.zeros(num_slices, dtype=torch.double, device=device)
        self.psnr_n = torch.zeros(num_slices, dtype=torch.long, device=device)
        self.lpips_sum = torch.zeros(num_slices, dtype=torch.double, device=device)
        self.lpips_n = torch.zeros(num_slices, dtype=torch.long, device=device)
        if probe is not None:
            action_dim = probe.config.action_dim
            self.r2_generated = R2Stats(num_slices, action_dim, device)
            self.r2_real = R2Stats(num_slices, action_dim, device)
            self.r2_control = R2Stats(num_slices, action_dim, device)

    def tensors(self) -> list[Tensor]:
        """Every accumulated sum; compute() reduces them across ranks."""
        stats = [self.fid_real, self.fid_generated, self.fvd_real, self.fvd_generated]
        if self.probe is not None:
            stats += [self.r2_generated, self.r2_real, self.r2_control]
        return [t for s in stats for t in s.tensors()] + [self.psnr_sum, self.psnr_n, self.lpips_sum, self.lpips_n]

    @torch.no_grad()
    def update(self, generated: Tensor, real: Tensor, actions: Tensor | None = None) -> None:
        generated, real = generated.float(), real.float()
        assert generated.shape[1] == self.num_slices * SLICE_FRAMES
        w = SLICE_FRAMES
        if self.probe is not None:
            recovered = self.probe.recover_actions(generated)
            recovered_real = self.probe.recover_actions(real)
            commanded = actions.to(recovered.device)
            # The control: commands this rollout never received, but with the statistics of its own.
            shifted = commanded.roll(shifts=commanded.shape[1] // 2, dims=1)
            for s in range(self.num_slices):
                fs = slice(s * w, (s + 1) * w)
                self.r2_generated.update(s, recovered[:, fs], commanded[:, fs])
                self.r2_real.update(s, recovered_real[:, fs], commanded[:, fs])
                self.r2_control.update(s, recovered[:, fs], shifted[:, fs])
        for s in range(self.num_slices):
            fs = slice(s * w, (s + 1) * w)
            gen, ref = generated[:, fs], real[:, fs]
            b, t = gen.shape[:2]
            feats = self.inception(torch.cat([gen.reshape(b * t, *gen.shape[2:]), ref.reshape(b * t, *ref.shape[2:])]))
            gen_feats, real_feats = feats[0].squeeze(-1).squeeze(-1).chunk(2)
            self.fid_generated.update(s, gen_feats)
            self.fid_real.update(s, real_feats)
            psnr = -10 * torch.log10(F.mse_loss(gen, ref, reduction="none").mean(dim=(2, 3, 4)))
            self.psnr_sum[s] += psnr.sum().double()
            self.psnr_n[s] += psnr.numel()
            for i in range(b):
                dist_lpips = self.lpips(gen[i] * 2 - 1, ref[i] * 2 - 1)
                self.lpips_sum[s] += dist_lpips.sum().double()
                self.lpips_n[s] += dist_lpips.numel()
            gen_i3d, real_i3d = i3d_features(self.i3d, torch.cat([gen, ref])).chunk(2)
            self.fvd_generated.update(s, gen_i3d)
            self.fvd_real.update(s, real_i3d)

    def compute(self) -> dict[str, float]:
        all_reduce(self.tensors())
        curves = {
            "fid_at": [
                frechet_distance(self.fid_real.mean_cov(s), self.fid_generated.mean_cov(s))
                for s in range(self.num_slices)
            ],
            "fvd_at": [
                frechet_distance(self.fvd_real.mean_cov(s), self.fvd_generated.mean_cov(s))
                for s in range(self.num_slices)
            ],
            "psnr_at": (self.psnr_sum / self.psnr_n).tolist(),
            "lpips_at": (self.lpips_sum / self.lpips_n).tolist(),
        }
        out = {}
        if self.probe is not None:
            arr_curves, out["arr_ceiling"] = self._arr()
            curves |= arr_curves
        for name, values in curves.items():
            for s, v in enumerate(values):
                out[f"{name}_{(s + 1) * SLICE_FRAMES}"] = v
        return out

    def _arr(self) -> tuple[dict[str, list[float]], float]:
        r2, r2_real, r2_control = self.r2_generated.r2(), self.r2_real.r2(), self.r2_control.r2()
        per_dim, per_dim_real = r2.tolist(), r2_real.tolist()
        # The real-video ceiling has no horizon dependence, so it is pooled over the slices.
        ceiling = [sum(sl[d] for sl in per_dim_real) / len(per_dim_real) for d in range(r2.shape[1])]
        dims = list(range(r2.shape[1])) if self.arr_dims is None else self.arr_dims
        usable = [d for d in dims if ceiling[d] > MIN_CEILING]
        arr = [
            sum(max(-1.0, sl[d]) / ceiling[d] for d in usable) / len(usable) if usable else float("nan")
            for sl in per_dim
        ]
        raw, control = dim_mean(r2, self.arr_dims), dim_mean(r2_control, self.arr_dims)
        curves = {
            "arr_at": arr,
            "arr_raw_at": raw,
            "arr_control_at": control,
            "arr_gap_at": [a - b for a, b in zip(raw, control, strict=True)],
            "arr_real_at": dim_mean(r2_real, self.arr_dims),
        }
        return curves, sum(ceiling[d] for d in dims) / len(dims)


class ReconstructionMetrics:
    """FID, FVD, PSNR and LPIPS of autoencoder reconstructions.

    update(generated, real): (B, T, 3, H, W) in [0, 1] with T = 16; generated (the decoder output mapped by
      x * 0.5 + 0.5) is clamped to [0, 1] here. PSNR and LPIPS read the float frames; FID and FVD read them as uint8
      frames, x * 255 truncated.
    compute() -> {"fid", "fvd", "psnr", "lpips"}.
    FID is pytorch-fid's: mean and np.cov of every frame's Inception activations, gathered from all ranks, with no
    covariance regularization. FVD is cd-fvd's: float64 sums of the I3D logits and a biased covariance.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.inception = load_inception(device)
        self.i3d = load_i3d(device)
        self.lpips = load_lpips(device)
        self.psnr_sum = torch.zeros((), dtype=torch.double, device=device)
        self.psnr_n = torch.zeros((), dtype=torch.long, device=device)
        self.lpips_sum = torch.zeros((), dtype=torch.double, device=device)
        self.lpips_n = torch.zeros((), dtype=torch.long, device=device)
        # Index 0: generated, 1: real.
        self.activations: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        self.i3d_sum = np.zeros((2, I3D_DIM))
        self.i3d_outer = np.zeros((2, I3D_DIM, I3D_DIM))
        self.i3d_n = np.zeros(2, dtype=np.int64)

    @torch.no_grad()
    def update(self, generated: Tensor, real: Tensor) -> None:
        generated = generated.clamp(0, 1)
        psnr = -10 * torch.log10(F.mse_loss(generated, real, reduction="none").mean(dim=(2, 3, 4)))
        self.psnr_sum += psnr.sum().double()
        self.psnr_n += psnr.numel()
        for i in range(generated.shape[0]):
            dist_lpips = self.lpips(generated[i] * 2 - 1, real[i] * 2 - 1)
            self.lpips_sum += dist_lpips.sum().double()
            self.lpips_n += dist_lpips.numel()
        for k, video in enumerate((generated, real)):
            u8 = (video.cpu() * 255).to(torch.uint8)
            frames = u8.flatten(0, 1)
            for i in range(0, len(frames), FID_BATCH):
                x = frames[i : i + FID_BATCH].to(torch.float32).div(255).to(self.device)
                self.activations[k].append(self.inception(x)[0].squeeze(3).squeeze(2).cpu().numpy())
            clips = u8.permute(0, 1, 3, 4, 2).numpy()
            for i in range(0, len(clips), I3D_BATCH):
                logits = self.i3d(preprocess_i3d(clips[i : i + I3D_BATCH]).to(self.device, torch.float32))
                x = logits.cpu().numpy().astype(np.float64)
                self.i3d_sum[k] += x.sum(axis=0)
                self.i3d_outer[k] += x.T @ x
                self.i3d_n[k] += len(x)

    def compute(self) -> dict[str, float]:
        all_reduce([self.psnr_sum, self.psnr_n, self.lpips_sum, self.lpips_n])
        i3d_stats = [torch.as_tensor(a, device=self.device) for a in (self.i3d_sum, self.i3d_outer, self.i3d_n)]
        all_reduce(i3d_stats)
        i3d_sum, i3d_outer, i3d_n = (t.cpu().numpy() for t in i3d_stats)
        activations = [np.concatenate(a) for a in self.activations]
        if dist.is_available() and dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, activations)
            activations = [np.concatenate([shard[k] for shard in shards]) for k in range(2)]
        mu, sigma = [], []
        for k in range(2):
            act = activations[k].astype(np.float64)
            mu.append(np.mean(act, axis=0))
            sigma.append(np.cov(act, rowvar=False))
        fid = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
        mean = [i3d_sum[k] / i3d_n[k] for k in range(2)]
        cov = [i3d_outer[k] / i3d_n[k] - np.outer(mean[k], mean[k]) for k in range(2)]
        (mu_fake, mu_real), (sigma_fake, sigma_real) = mean, cov
        s, _ = linalg.sqrtm(np.dot(sigma_real, sigma_fake), disp=False)
        fvd = np.real(np.square(mu_real - mu_fake).sum() + np.trace(sigma_fake + sigma_real - s * 2))
        return {
            "fid": float(fid),
            "fvd": float(fvd),
            "psnr": float(self.psnr_sum / self.psnr_n),
            "lpips": float(self.lpips_sum / self.lpips_n),
        }

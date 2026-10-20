"""Attention and normalization layers shared by the CADRE decoder and the world model.

Parameter names (wqkv, wo, q_ln, k_ln, gamma_beta) match the released checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention.bias import causal_lower_right


class SelfAttention(nn.Module):
    """Multi-head self-attention with grouped KV heads, QK LayerNorm, an optional sigmoid output gate, rotary
    embeddings and an optional KV cache for causal rollout."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        gating: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.gating = gating
        self.causal = causal
        # Opt-in decoder inference specialization. This is a runtime policy,
        # not a parameter/buffer, and does not change checkpoint schemas.
        self.short_temporal_attention = False
        self.tiny_temporal_attention = False
        self._cached_causal_biases = {}

        kv_dim = num_kv_heads * self.head_dim
        self.split_sizes = [dim, kv_dim, kv_dim] + ([dim] if gating else [])
        self.wqkv = nn.Linear(dim, sum(self.split_sizes), bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_ln = QKLayerNorm((num_heads, self.head_dim))
        self.k_ln = QKLayerNorm((num_kv_heads, self.head_dim))

    def prepare_causal_bias(self, query_length: int, key_length: int) -> None:
        """Create an optional symbolic mask before compilation, never during forward."""
        if not 1 < query_length < key_length:
            raise ValueError("cached causal bias requires 1 < query_length < key_length")
        self._cached_causal_biases[query_length, key_length] = causal_lower_right(query_length, key_length)

    def forward(
        self,
        x: Tensor,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        return_kv: bool = False,
    ):
        """x: (B, L, dim). kv_cache: past (k, v), each (B, L_past, num_kv_heads, head_dim), prepended to this
        step's keys. With a cache, rotary_emb covers past + current positions and x contains the final positions."""
        bsz, seqlen, _ = x.shape
        q, k, v, *gate = self.wqkv(x).split(self.split_sizes, dim=-1)
        q = self.q_ln(q.view(bsz, seqlen, self.num_heads, self.head_dim))
        k = self.k_ln(k.view(bsz, seqlen, self.num_kv_heads, self.head_dim))
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # The cache stores keys before rotation: positions are re-applied over the full window each step.
        new_kv = (k.clone(), v.clone()) if return_kv else None
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        if rotary_emb is not None:
            q_emb = (rotary_emb[0][-seqlen:], rotary_emb[1][-seqlen:]) if kv_cache is not None else rotary_emb
            q = apply_rotary_emb(q, q_emb)
            k = apply_rotary_emb(k, rotary_emb)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        short = (
            self.short_temporal_attention
            and self.causal
            and not self.training
            and not torch.is_grad_enabled()
            and self.num_heads == self.num_kv_heads
            and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and (q.shape[-2], k.shape[-2]) in ((1, 1), (1, 2), (2, 2))
        )
        tiny = (
            self.tiny_temporal_attention
            and self.causal
            and q.device.type == "cuda"
            and not self.training
            and not torch.is_grad_enabled()
        )
        if tiny:
            from util.tiny_temporal_attention import supports_tiny_temporal_attention, tiny_temporal_attention

            tiny = supports_tiny_temporal_attention(q, k, v)
        if tiny:
            y = tiny_temporal_attention(q, k, v)
        elif short:
            y = short_causal_attention(q, k, v)
        else:
            # Native causality permits fused attention without allocating a mask.
            # A cached single query is last and can attend every key; is_causal
            # would incorrectly align it with the first key instead.
            causal = self.causal and q.shape[-2] == k.shape[-2]
            mask = None
            if self.causal and not causal and q.shape[-2] > 1:
                # A prepared symbolic bias uses Flash's lower-right causal
                # kernel. Constructing CausalBias here breaks fullgraph on
                # PyTorch 2.8; unprepared shapes retain the explicit mask.
                mask = self._cached_causal_biases.get((q.shape[-2], k.shape[-2]))
                if mask is None:
                    mask = causal_mask(q.shape[-2], k.shape[-2], x.device)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=self.num_kv_heads != self.num_heads
            )
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim)
        if self.gating:
            y = y * torch.sigmoid(gate[0])
        y = self.wo(y)
        return (y, new_kv) if return_kv else y


def short_causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal attention for one/two keys with queries at the final key positions.

    Inputs are (B, H, L, D) with equal Q/K/V head counts. The first of two
    causal queries can only see v0; the last query sees both keys. FP32 scores,
    softmax, and weighted values avoid tiny-sequence SDPA kernel overhead when
    compiled. Casting the result back matches the surrounding attention dtype,
    but rounding can differ from Flash attention; validate decoded images.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("short attention expects (batch, heads, length, dim) tensors")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1] or k.shape != v.shape:
        raise ValueError("short attention requires matching batch, head, and channel dimensions")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16) or q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("short attention requires matching float32, float16, or bfloat16 dtypes")
    q_len, k_len = q.shape[-2], k.shape[-2]
    if (q_len, k_len) not in ((1, 1), (1, 2), (2, 2)):
        raise ValueError("short attention supports Q/K lengths (1,1), (1,2), or (2,2)")
    if k_len == 1:
        return v
    scores = (q[..., -1:, :].float() * k.float()).sum(dim=-1) * (q.shape[-1] ** -0.5)
    weights = scores.softmax(dim=-1)
    last = (weights[..., None] * v.float()).sum(dim=-2, keepdim=True).to(q.dtype)
    return last if q_len == 1 else torch.cat([v[..., :1, :], last], dim=-2)


def causal_mask(q_len: int, k_len: int, device: torch.device) -> Tensor:
    """Boolean (q_len, k_len) mask, True where attention is allowed; queries are the last q_len keys."""
    return torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(diagonal=k_len - q_len)


def apply_rotary_emb(x: Tensor, freqs: tuple[Tensor, Tensor]) -> Tensor:
    """Rotate consecutive channel pairs of x (B, L, heads, C) by cos/sin tables of shape (L, C), whose values
    are repeated per pair. Computed in float32."""
    x = rearrange(x, "b t h c -> b h t c")
    cos, sin = freqs
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return rearrange(out, "b h t c -> b t h c")


class QKLayerNorm(nn.Module):
    """LayerNorm over the head dimension with a learned per-(head, channel) scale and no bias, in float32."""

    def __init__(self, shape: tuple[int, int], eps: float = 1e-5):
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return (self.qk_scale.float() * x).to(dtype)


class AdaptiveLayerNorm(nn.Module):
    """AdaLN: LayerNorm without affine parameters, then (1 + gamma) * x + beta with gamma, beta a linear map of
    the conditioning. Zero-initialized by init_weights, so it starts as a plain LayerNorm."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: Tensor, cond: Tensor | None, projected: Tensor | None = None) -> Tensor:
        """Optionally consume this layer's already projected inference conditioning."""
        gamma, beta = (self.gamma_beta(cond) if projected is None else projected).chunk(2, dim=-1)
        return (1 + gamma) * self.layer_norm(x) + beta


def init_weights(module: nn.Module) -> None:
    """Default init applied with Module.apply: N(0, 0.02) weights and zero biases for linear/conv/embedding
    layers, unit norms, and zero AdaLN projections."""
    if isinstance(module, AdaptiveLayerNorm):
        nn.init.zeros_(module.gamma_beta.weight)
        nn.init.zeros_(module.gamma_beta.bias)
    elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if isinstance(getattr(module, "bias", None), torch.Tensor):
            nn.init.zeros_(module.bias)

"""Optional inference kernel for full-context decoder temporal attention.

Every causal key is retained. This uses BF16 tensor-core products and FP32
accumulation/softmax, as in Flash attention, but floating-point rounding may
differ. Callers must opt in after matched output and rollout validation.
Training, autograd, other shapes and devices must retain their normal backend.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installations need not install the optional CUDA backend.
    triton, tl = None, None


SUPPORTED_LENGTHS = ((8, 9), (16, 16))


def supports_tiny_temporal_attention(query, key, value):
    """Pure metadata guard; never initializes CUDA or allocates a device tensor."""
    if triton is None or torch.is_grad_enabled():
        return False
    if any(tensor.ndim != 4 or tensor.requires_grad for tensor in (query, key, value)):
        return False
    if query.device.type != "cuda" or query.device != key.device or query.device != value.device:
        return False
    if any(tensor.dtype != torch.bfloat16 or tensor.stride(-1) != 1 for tensor in (query, key, value)):
        return False
    return (
        query.shape[0] > 0
        and query.shape[1] == 16
        and query.shape[-1] == 64
        and query.shape[:2] == key.shape[:2]
        and key.shape[-1] == 64
        and key.shape == value.shape
        and (query.shape[-2], key.shape[-2]) in SUPPORTED_LENGTHS
    )


if triton is not None:

    @triton.jit
    def _tiny_temporal_kernel(
        Q,
        K,
        V,
        OUT,
        B: tl.constexpr,
        H: tl.constexpr,
        QL: tl.constexpr,
        KL: tl.constexpr,
        QS0: tl.constexpr,
        QS1: tl.constexpr,
        QS2: tl.constexpr,
        KS0: tl.constexpr,
        KS1: tl.constexpr,
        KS2: tl.constexpr,
        VS0: tl.constexpr,
        VS1: tl.constexpr,
        VS2: tl.constexpr,
        OS0: tl.constexpr,
        OS1: tl.constexpr,
        OS2: tl.constexpr,
    ):
        # The leading tl.dot batch dimension keeps the four heads independent.
        heads = tl.program_id(0) * 4 + tl.arange(0, 4)
        batch, head = heads // H, heads % H
        rows, keys, channels = tl.arange(0, 16), tl.arange(0, 16), tl.arange(0, 64)
        q = tl.load(
            Q
            + batch[:, None, None] * QS0
            + head[:, None, None] * QS1
            + rows[None, :, None] * QS2
            + channels[None, None, :],
            (heads[:, None, None] < B * H) & (rows[None, :, None] < QL),
            other=0,
        )
        k = tl.load(
            K
            + batch[:, None, None] * KS0
            + head[:, None, None] * KS1
            + keys[None, None, :] * KS2
            + channels[None, :, None],
            (heads[:, None, None] < B * H) & (keys[None, None, :] < KL),
            other=0,
        )
        scores = tl.dot(q, k, out_dtype=tl.float32) * 0.125
        allowed = (keys[None, :] < KL) & (keys[None, :] <= rows[:, None] + KL - QL)
        scores = tl.where(allowed[None, :, :], scores, -float("inf"))
        # Padded queries get a finite dummy row; their output stores are masked.
        weights = tl.exp(scores - tl.max(scores, axis=2)[:, :, None])
        denominator = tl.sum(weights, axis=2)
        v = tl.load(
            V
            + batch[:, None, None] * VS0
            + head[:, None, None] * VS1
            + keys[None, :, None] * VS2
            + channels[None, None, :],
            (heads[:, None, None] < B * H) & (keys[None, :, None] < KL),
            other=0,
        )
        output = tl.dot(weights.to(tl.bfloat16), v, out_dtype=tl.float32) / denominator[:, :, None]
        tl.store(
            OUT
            + batch[:, None, None] * OS0
            + head[:, None, None] * OS1
            + rows[None, :, None] * OS2
            + channels[None, None, :],
            output.to(tl.bfloat16),
            (heads[:, None, None] < B * H) & (rows[None, :, None] < QL),
        )


def tiny_temporal_attention(query, key, value):
    """Return B,H,Q,64 BF16 attention, stored contiguously in consumer B,Q,H,64 order.

    Q8/K9 includes one cached prefix key, so query0 sees keys0 and1. Q16/K16
    uses ordinary lower-triangular causality. Arbitrary leading tensor strides
    and storage offsets are honored; channels must be contiguous.
    """
    if not supports_tiny_temporal_attention(query, key, value):
        raise ValueError("tiny temporal attention requires CUDA BF16 no-grad H16/D64 and Q/K8/9 or16/16")
    batch, heads, q_length, dim = query.shape
    output = torch.empty(
        (batch, q_length, heads, dim),
        dtype=query.dtype,
        device=query.device,
    ).transpose(1, 2)
    _tiny_temporal_kernel[(triton.cdiv(batch * heads, 4),)](
        query,
        key,
        value,
        output,
        batch,
        heads,
        q_length,
        key.shape[-2],
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        *output.stride()[:3],
        num_warps=4,
    )
    return output

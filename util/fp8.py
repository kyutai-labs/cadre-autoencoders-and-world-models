"""Opt-in FP8 inference using native PyTorch, with no additional dependencies.

This is a numerical quality/performance tradeoff, not an equivalent rewrite of
BF16 inference. Validate generated rollouts before enabling it for visitors.
The private torch._scaled_mm API is supported by the project's pinned torch 2.8.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def eligible_linear(module: nn.Module, min_features: int = 256) -> bool:
    """Shape eligibility only; hardware/dtype validation happens before conversion."""
    return (
        isinstance(module, nn.Linear)
        and min(module.in_features, module.out_features) >= min_features
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


class FP8Linear(nn.Module):
    """Inference-only E4M3 matmul, retaining BF16 parameters and checkpoint names.

    Small row counts use the original BF16 projection because quantization overhead
    can outweigh FP8 gains for conditioning vectors. Cached FP8 weights are omitted
    from state_dict; construct this wrapper after loading the final BF16 checkpoint.
    """

    def __init__(self, linear: nn.Linear, min_rows: int = 32, use_fast_accum: bool = False):
        super().__init__()
        if not eligible_linear(linear):
            raise ValueError("FP8 requires Linear dimensions >=256 and divisible by 16")
        if linear.weight.device.type != "cuda" or linear.weight.dtype != torch.bfloat16:
            raise ValueError("FP8 conversion requires CUDA BF16 Linear weights")
        if min_rows < 1:
            raise ValueError("min_rows must be positive")
        if linear.bias is not None and linear.bias.dtype != torch.bfloat16:
            raise ValueError("FP8 conversion requires BF16 bias")
        self.in_features, self.out_features, self.min_rows = linear.in_features, linear.out_features, min_rows
        self.use_fast_accum = use_fast_accum
        self.weight, self.bias = linear.weight, linear.bias
        with torch.no_grad():
            weights = linear.weight.float()
            scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
            quantized = (weights / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        self.register_buffer("weight_fp8", quantized.contiguous(), persistent=False)
        self.register_buffer("weight_scale", scale.t().contiguous(), persistent=False)
        self.train(linear.training)

    def forward(self, x: Tensor) -> Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FP8Linear is inference-only; call eval() and use torch.no_grad()")
        if x.shape[-1] != self.in_features:
            raise ValueError("input's last dimension must match Linear.in_features")
        # Explicit BF16 input casting matches the surrounding demo autocast policy.
        values_bf16 = x.to(torch.bfloat16)
        if x.numel() // self.in_features < self.min_rows:
            # Preserve the original shape/strides and hence the native BF16
            # kernel choice, including its rounding for small batched inputs.
            return F.linear(values_bf16, self.weight, self.bias)
        flat = values_bf16.reshape(-1, self.in_features)
        values = flat.float()
        scale = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        # reshape/to preserve strides when given an already-2D transposed input.
        # Native rowwise scaled_mm requires row-major A and contiguous row scales.
        quantized = (values / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
        output = torch._scaled_mm(
            quantized,
            self.weight_fp8.t(),
            scale_a=scale.contiguous(),
            scale_b=self.weight_scale,
            bias=None if self.bias is None else self.bias.contiguous(),
            out_dtype=torch.bfloat16,
            use_fast_accum=self.use_fast_accum,
        )
        return output.reshape(*x.shape[:-1], self.out_features)


def convert_fp8_linears(
    model: nn.Module, min_features: int = 256, min_rows: int = 32, use_fast_accum: bool = False
) -> dict:
    """Replace eligible child Linear modules after loading/casting/eval, and report them.

    Original BF16 parameters remain available for small-row fallback and unchanged
    checkpoint serialization. Model weights must stay immutable after conversion.
    Requires Hopper or newer CUDA hardware; conversion is never enabled implicitly.
    """
    if min_features < 256 or min_rows < 1:
        raise ValueError("min_features must be at least 256 and min_rows must be positive")
    if isinstance(model, nn.Linear):
        raise ValueError("wrap a root Linear in a container before calling convert_fp8_linears")
    candidates = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=False)
        if eligible_linear(module, min_features)
    ]
    if candidates:
        if model.training or any(module.training for _, module in candidates):
            raise ValueError("call model.eval() before FP8 conversion")
        if not callable(getattr(torch, "_scaled_mm", None)):
            raise RuntimeError("this PyTorch build does not provide torch._scaled_mm")
        for _, module in candidates:
            if module.weight.device.type != "cuda" or module.weight.dtype != torch.bfloat16:
                raise ValueError("FP8 conversion requires CUDA BF16 Linear weights")
            if module.bias is not None and module.bias.dtype != torch.bfloat16:
                raise ValueError("FP8 conversion requires BF16 bias")
        for device in {module.weight.device for _, module in candidates}:
            if torch.cuda.get_device_capability(device) < (9, 0):
                raise RuntimeError("this FP8 inference path requires Hopper or newer CUDA hardware")
    replacements = {}
    for name, module in candidates:
        if id(module) not in replacements:
            replacements[id(module)] = FP8Linear(module, min_rows, use_fast_accum)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacements[id(module)])
    return {
        "converted": len(replacements),
        "layers": [name for name, _ in candidates],
        "min_features": min_features,
        "min_rows": min_rows,
        "weight_dtype": "float8_e4m3fn",
        "activation_scaling": "per_row",
        "output_dtype": "bfloat16",
        "use_fast_accum": use_fast_accum,
        "retained_bf16_weight_bytes": sum(
            layer.weight.numel() * layer.weight.element_size() for layer in replacements.values()
        ),
        "cached_fp8_weight_bytes": sum(layer.weight_fp8.numel() for layer in replacements.values()),
    }

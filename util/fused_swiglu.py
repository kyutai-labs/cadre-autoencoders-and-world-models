"""Opt-in inference fusion of the two SwiGLU input projections.

Convert after loading/casting/eval and, when used, after FP8 conversion, but before
torch.compile. Original child modules and Parameters remain registered under the
same names. Concatenated inference weights are nonpersistent caches; model weights
and their dtype/device must stay unchanged after conversion, as with FP8Linear.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util.fp8 import FP8_DTYPE, FP8_MAX, FP8Linear


def _validate(module: nn.Module) -> None:
    if module.training or any(child.training for child in module.children()):
        raise ValueError("call model.eval() before SwiGLU fusion")
    swish, gate, output = module.swish_linear, module.gate_linear, module.output_linear
    if type(swish) is not type(gate) or not isinstance(swish, (nn.Linear, FP8Linear)):
        raise ValueError("SwiGLU input projections must both be Linear or both FP8Linear")
    if not isinstance(output, (nn.Linear, FP8Linear)):
        raise ValueError("SwiGLU output projection must be Linear or FP8Linear")
    if (swish.in_features, swish.out_features) != (gate.in_features, gate.out_features):
        raise ValueError("SwiGLU input projections must have matching shapes")
    if output.in_features != swish.out_features:
        raise ValueError("SwiGLU output projection must accept the hidden dimension")
    if (swish.weight.device, swish.weight.dtype) != (gate.weight.device, gate.weight.dtype):
        raise ValueError("SwiGLU input weights must have matching device and dtype")
    if (swish.bias is None) != (gate.bias is None):
        raise ValueError("SwiGLU input projections must agree on bias")
    if swish.bias is not None and any(
        (layer.bias.device, layer.bias.dtype) != (layer.weight.device, layer.weight.dtype) for layer in (swish, gate)
    ):
        raise ValueError("SwiGLU input bias must match its weight device and dtype")
    if isinstance(swish, FP8Linear) and (swish.min_rows, swish.use_fast_accum) != (
        gate.min_rows,
        gate.use_fast_accum,
    ):
        raise ValueError("SwiGLU FP8 input projections must have matching inference settings")


class FusedSwiGLU(nn.Module):
    """One input GEMM/quantization followed by the original output projection.

    A wider GEMM can round differently from separate projections. FP8 row scales
    and quantized weight values are preserved, and small-row FP8 fallback uses
    the original separate BF16 projections to preserve that path exactly.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        _validate(module)
        self.swish_linear = module.swish_linear
        self.gate_linear = module.gate_linear
        self.output_linear = module.output_linear
        self.in_features = self.swish_linear.in_features
        self.hidden_features = self.swish_linear.out_features
        self.use_fp8 = isinstance(self.swish_linear, FP8Linear)
        with torch.no_grad():
            if self.use_fp8:
                self.min_rows = self.swish_linear.min_rows
                self.use_fast_accum = self.swish_linear.use_fast_accum
                self.register_buffer(
                    "input_weight_fp8",
                    torch.cat((self.swish_linear.weight_fp8, self.gate_linear.weight_fp8), dim=0),
                    persistent=False,
                )
                self.register_buffer(
                    "input_weight_scale",
                    torch.cat((self.swish_linear.weight_scale, self.gate_linear.weight_scale), dim=1),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "input_weight",
                    torch.cat((self.swish_linear.weight, self.gate_linear.weight), dim=0),
                    persistent=False,
                )
            bias = (
                None
                if self.swish_linear.bias is None
                else torch.cat((self.swish_linear.bias, self.gate_linear.bias), dim=0)
            )
            self.register_buffer("input_bias", bias, persistent=False)
        self.train(module.training)

    def forward(self, x: Tensor) -> Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FusedSwiGLU is inference-only; call eval() and use torch.no_grad()")
        if x.shape[-1] != self.in_features:
            raise ValueError("input's last dimension must match SwiGLU input features")
        if self.use_fp8:
            if x.numel() // self.in_features < self.min_rows:
                return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))
            values = x.to(torch.bfloat16).reshape(-1, self.in_features).float()
            scale = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
            quantized = (values / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
            projected = torch._scaled_mm(
                quantized,
                self.input_weight_fp8.t(),
                scale_a=scale.contiguous(),
                scale_b=self.input_weight_scale,
                bias=self.input_bias,
                out_dtype=torch.bfloat16,
                use_fast_accum=self.use_fast_accum,
            ).reshape(*x.shape[:-1], 2 * self.hidden_features)
        else:
            projected = F.linear(x, self.input_weight, self.input_bias)
        swish, gate = projected.chunk(2, dim=-1)
        return self.output_linear(F.silu(swish) * gate)


def convert_fused_swiglu(model: nn.Module) -> dict:
    """Fuse this project's AE/WM SwiGLU children and report the opt-in conversion.

    Apply convert_fp8_linears first when FP8 is requested. Repeated fusion is a
    no-op. Shared SwiGLU modules retain sharing, and checkpoint keys/Parameters
    are unchanged. Wrap a root SwiGLU in a container before conversion.
    """
    from model_autoencoder import SwiGLU as AutoencoderSwiGLU
    from model_world_model import SwiGLU as WorldModelSwiGLU

    supported = (AutoencoderSwiGLU, WorldModelSwiGLU)
    if isinstance(model, supported):
        raise ValueError("wrap a root SwiGLU in a container before calling convert_fused_swiglu")
    candidates = [
        (name, module) for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, supported)
    ]
    if candidates and model.training:
        raise ValueError("call model.eval() before SwiGLU fusion")
    for _, module in candidates:
        _validate(module)
    replacements = {}
    for name, module in candidates:
        if id(module) not in replacements:
            replacements[id(module)] = FusedSwiGLU(module)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacements[id(module)])
    return {
        "converted": len(replacements),
        "layers": [name for name, _ in candidates],
        "fp8_layers": sum(layer.use_fp8 for layer in replacements.values()),
        "bf16_or_float_layers": sum(not layer.use_fp8 for layer in replacements.values()),
        "cached_weight_bytes": sum(
            buffer.numel() * buffer.element_size()
            for layer in replacements.values()
            for buffer in layer.buffers(recurse=False)
        ),
    }

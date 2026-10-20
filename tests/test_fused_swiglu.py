"""CPU fusion contracts; opt-in CUDA tests cover FP8 and compiled graph replay."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model_autoencoder import SwiGLU as AutoencoderSwiGLU
from model_world_model import SwiGLU as WorldModelSwiGLU
from util.fp8 import FP8_DTYPE, FP8_MAX, FP8Linear, convert_fp8_linears
from util.fused_swiglu import FusedSwiGLU, convert_fused_swiglu


def cpu_fp8_fixture(linear, *, min_rows=32, use_fast_accum=False):
    """Supply only FP8 state for a mocked native-op boundary, without CUDA init."""
    layer = FP8Linear.__new__(FP8Linear)
    nn.Module.__init__(layer)
    layer.in_features, layer.out_features = linear.in_features, linear.out_features
    layer.min_rows, layer.use_fast_accum = min_rows, use_fast_accum
    layer.weight, layer.bias = linear.weight, linear.bias
    weights = linear.weight.detach().float()
    scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    layer.register_buffer("weight_fp8", (weights / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), persistent=False)
    layer.register_buffer("weight_scale", scale.t().contiguous(), persistent=False)
    return layer.eval()


class FusedSwiGLUCpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_real_autoencoder_and_world_model_float_and_bfloat16_parity(self):
        for cls in (AutoencoderSwiGLU, WorldModelSwiGLU):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(model=cls.__module__, dtype=dtype):
                    model = nn.Sequential(cls(32)).to(dtype=dtype).eval()
                    # Preserve a transposed input through the original and fused paths.
                    x = torch.randn(2, 32, 17, dtype=dtype).transpose(1, 2)
                    with torch.no_grad():
                        expected = model(x)
                        summary = convert_fused_swiglu(model)
                        actual = model(x)
                    self.assertEqual(summary["converted"], 1)
                    self.assertEqual(summary["layers"], ["0"])
                    self.assertEqual(summary["fp8_layers"], 0)
                    self.assertEqual(actual.dtype, dtype)
                    tolerance = 8e-3 if dtype == torch.bfloat16 else 1e-6
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_checkpoint_parameters_sharing_and_nonpersistent_cache(self):
        original = WorldModelSwiGLU(32).eval()
        model = nn.ModuleDict({"first": original, "second": original}).eval()
        parameters = dict(model.named_parameters(remove_duplicate=False))
        checkpoint = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        summary = convert_fused_swiglu(model)
        self.assertEqual(summary["converted"], 1)
        self.assertEqual(summary["layers"], ["first", "second"])
        self.assertIs(model["first"], model["second"])
        self.assertEqual(set(checkpoint), set(model.state_dict()))
        for name, parameter in model.named_parameters(remove_duplicate=False):
            self.assertIs(parameter, parameters[name])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, checkpoint[name], rtol=0, atol=0)
        fused = model["first"]
        torch.testing.assert_close(fused.input_weight[: fused.hidden_features], original.swish_linear.weight)
        torch.testing.assert_close(fused.input_weight[fused.hidden_features :], original.gate_linear.weight)
        self.assertFalse(fused.input_weight.requires_grad)
        self.assertEqual(summary["cached_weight_bytes"], fused.input_weight.numel() * fused.input_weight.element_size())
        self.assertEqual(convert_fused_swiglu(model)["converted"], 0)

    def test_bias_is_concatenated_in_projection_order(self):
        original = WorldModelSwiGLU(32).eval()
        hidden = original.swish_linear.out_features
        original.swish_linear = nn.Linear(32, hidden).eval()
        original.gate_linear = nn.Linear(32, hidden).eval()
        x = torch.randn(11, 32)
        with torch.no_grad():
            fused = FusedSwiGLU(original)
            torch.testing.assert_close(fused(x), original(x), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(fused.input_bias[:hidden], original.swish_linear.bias)
        torch.testing.assert_close(fused.input_bias[hidden:], original.gate_linear.bias)

    def test_training_grad_and_invalid_pair_are_rejected_before_replacement(self):
        model = nn.Sequential(WorldModelSwiGLU(32), WorldModelSwiGLU(32))
        originals = list(model.children())
        with self.assertRaisesRegex(ValueError, "eval"):
            convert_fused_swiglu(model)
        model.eval()
        model[1].gate_linear = nn.Linear(16, 256).eval()
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            convert_fused_swiglu(model)
        self.assertEqual(list(model.children()), originals)
        with self.assertRaisesRegex(ValueError, "container"):
            convert_fused_swiglu(WorldModelSwiGLU(32).eval())
        fused = FusedSwiGLU(WorldModelSwiGLU(32).eval())
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            fused(torch.randn(2, 32))
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "last dimension"):
                fused(torch.randn(2, 16))
            fused.train()
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                fused(torch.randn(2, 32))

    def test_fp8_fuses_input_quantization_once_and_preserves_rowwise_weights(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                original = WorldModelSwiGLU(32).to(torch.bfloat16).eval()
                original.swish_linear = cpu_fp8_fixture(original.swish_linear, use_fast_accum=fast)
                original.gate_linear = cpu_fp8_fixture(original.gate_linear, use_fast_accum=fast)
                model = nn.Sequential(original).eval()
                checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
                summary = convert_fused_swiglu(model)
                fused = model[0]
                hidden = fused.hidden_features
                self.assertEqual(summary["fp8_layers"], 1)
                self.assertEqual(summary["bf16_or_float_layers"], 0)
                for start, layer in ((0, original.swish_linear), (hidden, original.gate_linear)):
                    torch.testing.assert_close(
                        fused.input_weight_fp8[start : start + hidden].float(), layer.weight_fp8.float()
                    )
                    torch.testing.assert_close(fused.input_weight_scale[:, start : start + hidden], layer.weight_scale)

                calls = []

                def emulate(a, b, scale_a, scale_b, bias, out_dtype, use_fast_accum, calls=calls, fast=fast):
                    calls.append((a.shape, b.shape))
                    self.assertTrue(a.is_contiguous())
                    self.assertTrue(b.t().is_contiguous())
                    self.assertTrue(scale_a.is_contiguous())
                    self.assertTrue(scale_b.is_contiguous())
                    self.assertIs(use_fast_accum, fast)
                    value = (a.float() * scale_a) @ (b.float() * scale_b)
                    if bias is not None:
                        value += bias.float()
                    return value.to(out_dtype)

                x = torch.randn(32, 33, dtype=torch.bfloat16).t()
                with torch.no_grad(), patch("util.fused_swiglu.torch._scaled_mm", side_effect=emulate):
                    expected = original(x)
                    self.assertEqual(len(calls), 2)
                    calls.clear()
                    actual = fused(x)
                    self.assertEqual(calls, [(torch.Size([33, 32]), torch.Size([32, hidden * 2]))])
                torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
                self.assertEqual(set(checkpoint), set(model.state_dict()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)

    def test_small_fp8_inputs_keep_original_bfloat16_fallback(self):
        original = WorldModelSwiGLU(32).to(torch.bfloat16).eval()
        original.swish_linear = cpu_fp8_fixture(original.swish_linear)
        original.gate_linear = cpu_fp8_fixture(original.gate_linear)
        fused = FusedSwiGLU(original)
        x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
        with torch.no_grad(), patch("util.fused_swiglu.torch._scaled_mm", side_effect=AssertionError("must use BF16")):
            torch.testing.assert_close(fused(x), original(x), rtol=0, atol=0)


@unittest.skipUnless(os.environ.get("CADRE_RUN_GPU_TESTS") == "1", "set CADRE_RUN_GPU_TESTS=1 to run CUDA checks")
class FusedSwiGLUCudaTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("Hopper or newer CUDA hardware required")
        torch.manual_seed(71)

    def test_fp8_parity_and_compiled_replay_consume_changed_inputs(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                model = nn.Sequential(WorldModelSwiGLU(256)).cuda().to(torch.bfloat16).eval()
                convert_fp8_linears(model, use_fast_accum=fast)
                original = model[0]
                checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
                convert_fused_swiglu(model)
                compiled_original = torch.compile(original, mode="reduce-overhead", fullgraph=True, dynamic=False)
                compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
                previous = None
                fusion_rmse = baseline_compile_rmse = 0.0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for replay in range(4):
                        x = torch.randn(2, 64, 256, device="cuda") + replay * 0.125
                        expected, eager = original(x), model(x)
                        torch.compiler.cudagraph_mark_step_begin()
                        compiled_reference = compiled_original(x).clone()
                        torch.compiler.cudagraph_mark_step_begin()
                        actual = compiled(x).clone()
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        # One wider GEMM may select a different tile/accumulation order.
                        torch.testing.assert_close(eager, expected, rtol=2e-2, atol=8e-3)
                        # Compare matching compile policies: Inductor fuses SiLU,
                        # multiplication and FP8 scaling, avoiding eager BF16
                        # rounding boundaries in BOTH original and fused modules.
                        # The baseline itself differs from eager, so that is not
                        # the numerical contract of this projection rewrite.
                        torch.testing.assert_close(actual, compiled_reference, rtol=2e-2, atol=8e-3)
                        fusion_error = actual.float() - compiled_reference.float()
                        compile_error = compiled_reference.float() - expected.float()
                        fusion_rmse = max(
                            fusion_rmse,
                            (
                                fusion_error.square().mean().sqrt() / compiled_reference.float().square().mean().sqrt()
                            ).item(),
                        )
                        baseline_compile_rmse = max(
                            baseline_compile_rmse,
                            (compile_error.square().mean().sqrt() / expected.float().square().mean().sqrt()).item(),
                        )
                        if previous is not None:
                            self.assertFalse(torch.equal(actual, previous))
                        previous = actual
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)
                print(
                    f"SwiGLU fast_accum={fast}: compiled fusion max normalized RMSE={fusion_rmse:.6g}; "
                    f"original compiled/eager max normalized RMSE={baseline_compile_rmse:.6g}"
                )


if __name__ == "__main__":
    unittest.main()

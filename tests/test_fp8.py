"""FP8 is optional; CPU tests exercise selection, and CUDA tests check its tradeoff."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from util.fp8 import FP8Linear, convert_fp8_linears, eligible_linear


class FP8SelectionTests(unittest.TestCase):
    def test_only_large_aligned_linear_shapes_are_eligible(self):
        for inputs, outputs, expected in ((256, 256, True), (272, 512, True), (128, 512, False), (300, 512, False)):
            with self.subTest(inputs=inputs, outputs=outputs):
                self.assertEqual(eligible_linear(nn.Linear(inputs, outputs)), expected)
        self.assertFalse(eligible_linear(nn.Conv2d(256, 256, 1)))

    def test_cpu_model_is_rejected_before_replacing_modules(self):
        model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256)).to(torch.bfloat16).eval()
        originals = list(model.children())
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            convert_fp8_linears(model)
        self.assertEqual(list(model.children()), originals)

    def test_ineligible_model_is_unchanged(self):
        model = nn.Sequential(nn.Linear(8, 8)).eval()
        original = model[0]
        summary = convert_fp8_linears(model)
        self.assertEqual(summary["converted"], 0)
        self.assertEqual(summary["layers"], [])
        self.assertFalse(summary["use_fast_accum"])
        self.assertIs(model[0], original)

    def test_summary_records_explicit_fast_accumulation_flag_without_cuda(self):
        for enabled in (False, True):
            with self.subTest(use_fast_accum=enabled):
                summary = convert_fp8_linears(nn.Sequential().eval(), use_fast_accum=enabled)
                self.assertIs(summary["use_fast_accum"], enabled)

    def test_training_invalid_thresholds_and_root_linear_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "eval"):
            convert_fp8_linears(nn.Sequential(nn.Linear(256, 256)))
        for kwargs in ({"min_features": 128}, {"min_rows": 0}):
            with self.assertRaises(ValueError):
                convert_fp8_linears(nn.Sequential(), **kwargs)
        with self.assertRaisesRegex(ValueError, "container"):
            convert_fp8_linears(nn.Linear(256, 256))
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            FP8Linear(nn.Linear(256, 256).to(torch.bfloat16))

    def test_operator_layout_contract_handles_real_row_counts_and_transposed_inputs(self):
        # Inspect the native-op boundary on CPU without creating a CUDA context.
        # Production construction deliberately rejects CPU; this fixture supplies
        # only the state needed to validate forward's quantization and layout.
        layer = FP8Linear.__new__(FP8Linear)
        nn.Module.__init__(layer)
        layer.in_features, layer.out_features, layer.min_rows = 256, 256, 32
        layer.use_fast_accum = False
        layer.weight = nn.Parameter(torch.zeros(256, 256, dtype=torch.bfloat16))
        layer.bias = nn.Parameter(torch.zeros(512, dtype=torch.bfloat16)[::2])
        layer.register_buffer("weight_fp8", torch.zeros(256, 256).to(torch.float8_e4m3fn))
        layer.register_buffer("weight_scale", torch.ones(1, 256))
        layer.eval()

        def inspect(a, b, scale_a, scale_b, bias, out_dtype, use_fast_accum):
            rows = a.shape[0]
            self.assertEqual(a.dtype, torch.float8_e4m3fn)
            self.assertEqual(b.dtype, torch.float8_e4m3fn)
            self.assertEqual(a.stride(), (256, 1))
            self.assertEqual(b.stride(), (1, 256))
            self.assertEqual(scale_a.shape, (rows, 1))
            self.assertEqual(scale_b.shape, (1, 256))
            self.assertEqual(scale_a.dtype, torch.float32)
            self.assertEqual(scale_b.dtype, torch.float32)
            self.assertEqual(scale_a.stride(0), 1)
            self.assertEqual(scale_b.stride(1), 1)
            self.assertEqual(bias.stride(), (1,))
            self.assertEqual(bias.dtype, torch.bfloat16)
            self.assertEqual(out_dtype, torch.bfloat16)
            self.assertIs(use_fast_accum, layer.use_fast_accum)
            self.assertTrue(torch.isfinite(scale_a).all())
            self.assertTrue((scale_a > 0).all())
            return torch.zeros(rows, 256, dtype=torch.bfloat16)

        with torch.no_grad(), patch("util.fp8.torch._scaled_mm", side_effect=inspect) as operation:
            for enabled in (False, True):
                layer.use_fast_accum = enabled
                for rows in (33, 576, 9280):
                    x = torch.zeros(256, rows, dtype=torch.bfloat16).t()
                    self.assertFalse(x.is_contiguous())
                    self.assertEqual(layer(x).shape, (rows, 256))
            self.assertEqual(operation.call_count, 6)
            with self.assertRaisesRegex(ValueError, "last dimension"):
                layer(torch.zeros(64, 128))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class FP8CudaTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("Hopper or newer hardware required")
        torch.manual_seed(31)
        self.model = nn.Sequential(nn.Linear(512, 256, device="cuda", dtype=torch.bfloat16)).eval()
        self.x = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)

    def test_fp8_error_is_bounded_and_state_dict_is_preserved(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        with torch.no_grad():
            expected = self.model(self.x)
            summary = convert_fp8_linears(self.model)
            actual = self.model(self.x)
        self.assertEqual(summary["converted"], 1)
        self.assertFalse(summary["use_fast_accum"])
        self.assertFalse(self.model[0].use_fast_accum)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        error = actual.float() - expected.float()
        normalized_rmse = error.square().mean().sqrt() / expected.float().square().mean().sqrt()
        # Dynamic E4M3 activation+weight quantization is lossy. This tests a bounded
        # local error, not whole-model visual quality or long-rollout stability.
        self.assertLess(normalized_rmse.item(), 0.08)
        print(
            f"FP8 Linear normalized RMSE={normalized_rmse.item():.6g}, "
            f"max absolute error={error.abs().max().item():.6g}"
        )
        self.assertEqual(set(before), set(self.model.state_dict()))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_small_row_fallback_preserves_bf16_and_bias(self):
        x = self.x[:, :1]
        with torch.no_grad():
            expected = self.model(x)
            convert_fp8_linears(self.model)
            with patch("util.fp8.torch._scaled_mm", side_effect=AssertionError("small rows must use BF16")):
                actual = self.model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_compiled_cuda_graph_replays_match_eager_fp8(self):
        convert_fp8_linears(self.model)
        compiled = torch.compile(self.model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        with torch.no_grad():
            expected = self.model(self.x)
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                actual = compiled(self.x).clone()
                torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=8e-3)

    def test_fast_accumulation_conversion_and_compiled_replays_have_bounded_error(self):
        original = self.model[0]
        summary = convert_fp8_linears(self.model, use_fast_accum=True)
        self.assertEqual(summary["converted"], 1)
        self.assertTrue(summary["use_fast_accum"])
        self.assertTrue(self.model[0].use_fast_accum)
        compiled = torch.compile(self.model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        max_normalized_rmse = max_absolute_error = max_eager_difference = 0.0
        with torch.no_grad():
            for replay in range(4):
                # Different values ensure a replay consumes current activations.
                x = self.x + replay * 0.125
                reference, eager = original(x), self.model(x)
                torch.compiler.cudagraph_mark_step_begin()
                actual = compiled(x).clone()
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, eager, rtol=1.6e-2, atol=8e-3)
                error = actual.float() - reference.float()
                normalized_rmse = error.square().mean().sqrt() / reference.float().square().mean().sqrt()
                # Same local E4M3 tolerance as slow accumulation; this does not
                # establish whole-model quality or long-rollout stability.
                self.assertLess(normalized_rmse.item(), 0.08)
                max_normalized_rmse = max(max_normalized_rmse, normalized_rmse.item())
                max_absolute_error = max(max_absolute_error, error.abs().max().item())
                max_eager_difference = max(max_eager_difference, (actual.float() - eager.float()).abs().max().item())
        print(
            f"FP8 fast accumulation: max normalized RMSE={max_normalized_rmse:.6g}, "
            f"max absolute BF16 error={max_absolute_error:.6g}, "
            f"max absolute eager/compiled difference={max_eager_difference:.6g}"
        )

    def test_cuda_actual_row_counts_transposes_and_bfloat16_autocast(self):
        convert_fp8_linears(self.model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for rows in (33, 576, 9280):
                with self.subTest(rows=rows):
                    # The demo's normalization can emit FP32 into BF16 autocast.
                    x = torch.randn(512, rows, device="cuda", dtype=torch.float32).t()
                    actual = self.model(x)
                    self.assertEqual(actual.shape, (rows, 256))
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()

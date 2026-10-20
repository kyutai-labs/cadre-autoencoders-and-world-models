"""Short causal attention preserves the reference path outside opt-in inference."""

import copy
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model_autoencoder import temporal_rope
from util.attention import SelfAttention, short_causal_attention


def reference_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[-2] == k.shape[-2])


def inputs(batch, heads, q_len, k_len, dim, *, dtype=torch.float32, device="cpu"):
    return tuple(
        torch.randn(batch, length, heads, dim, dtype=dtype, device=device).transpose(1, 2)
        for length in (q_len, k_len, k_len)
    )


class ShortAttentionCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(83)

    def test_float32_helper_matches_sdpa_for_all_supported_lengths_and_strides(self):
        with torch.no_grad():
            for q_len, k_len in ((1, 1), (1, 2), (2, 2)):
                with self.subTest(q_len=q_len, k_len=k_len):
                    q, k, v = inputs(3, 4, q_len, k_len, 16)
                    before = [value.clone() for value in (q, k, v)]
                    actual = short_causal_attention(q, k, v)
                    torch.testing.assert_close(actual, reference_attention(q, k, v), rtol=2e-6, atol=2e-6)
                    for value, original in zip((q, k, v), before, strict=True):
                        torch.testing.assert_close(value, original, rtol=0, atol=0)
                    if q_len == 2:
                        torch.testing.assert_close(actual[..., :1, :], v[..., :1, :], rtol=0, atol=0)

    def test_two_query_first_frame_cannot_see_future_value(self):
        q = torch.ones(1, 2, 2, 8)
        k = torch.cat([torch.zeros(1, 2, 1, 8), torch.ones(1, 2, 1, 8) * 100], dim=2)
        v = torch.cat([torch.full((1, 2, 1, 8), 3.0), torch.full((1, 2, 1, 8), 999.0)], dim=2)
        actual = short_causal_attention(q, k, v)
        torch.testing.assert_close(actual[..., :1, :], v[..., :1, :], rtol=0, atol=0)
        torch.testing.assert_close(actual[..., -1:, :], v[..., -1:, :], rtol=0, atol=0)

    def test_half_and_bfloat16_outputs_preserve_dtype_with_bounded_rounding(self):
        with torch.no_grad():
            for dtype in (torch.float16, torch.bfloat16):
                for q_len, k_len in ((1, 1), (1, 2), (2, 2)):
                    with self.subTest(dtype=dtype, q_len=q_len, k_len=k_len):
                        values = inputs(2, 4, q_len, k_len, 16, dtype=dtype)
                        actual = short_causal_attention(*values)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, reference_attention(*values), rtol=1.6e-2, atol=8e-3)

    def test_full_attention_matches_with_rotary_gating_and_cached_key(self):
        model = SelfAttention(32, 4, gating=True, causal=True).eval()
        self.assertFalse(model.short_temporal_attention)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        parameters = dict(model.named_parameters())
        reference = copy.deepcopy(model)
        model.short_temporal_attention = True
        with torch.no_grad():
            for length, cached in ((1, False), (2, False), (1, True)):
                with self.subTest(length=length, cached=cached):
                    cache = None
                    if cached:
                        _, cache = reference(torch.randn(2, 1, 32), temporal_rope(1, 8, "cpu"), return_kv=True)
                    x = torch.randn(2, length, 32)
                    rope = temporal_rope(length + int(cached), 8, "cpu")
                    rng = torch.get_rng_state().clone()
                    expected, expected_cache = reference(x, rope, kv_cache=cache, return_kv=True)
                    with patch("util.attention.short_causal_attention", wraps=short_causal_attention) as short:
                        actual, new_cache = model(x, rope, kv_cache=cache, return_kv=True)
                    short.assert_called_once()
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    for actual_kv, expected_kv in zip(new_cache, expected_cache, strict=True):
                        torch.testing.assert_close(actual_kv, expected_kv, rtol=0, atol=0)
        self.assertEqual(set(state), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, parameters[name])

    def test_training_and_eval_gradients_fall_back_to_original_sdpa(self):
        for training in (False, True):
            with self.subTest(training=training):
                model = SelfAttention(32, 4, gating=True, causal=True).train(training)
                reference = copy.deepcopy(model)
                model.short_temporal_attention = True
                x = torch.randn(2, 2, 32)
                outputs, gradients = [], []
                with patch("util.attention.short_causal_attention", side_effect=AssertionError("must use SDPA")):
                    for candidate in (model, reference):
                        value = x.clone().requires_grad_()
                        output = candidate(value, temporal_rope(2, 8, "cpu"))
                        output.square().mean().backward()
                        outputs.append(output)
                        gradients.append(value.grad)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_noncausal_long_gqa_and_float64_calls_retain_sdpa(self):
        for causal, length, kv_heads, dtype in (
            (False, 2, 4, torch.float32),
            (True, 3, 4, torch.float32),
            (True, 2, 2, torch.float32),
            (True, 2, 4, torch.float64),
        ):
            with self.subTest(causal=causal, length=length, kv_heads=kv_heads, dtype=dtype):
                model = SelfAttention(32, 4, num_kv_heads=kv_heads, gating=True, causal=causal).to(dtype=dtype).eval()
                reference = copy.deepcopy(model)
                model.short_temporal_attention = True
                x = torch.randn(2, length, 32, dtype=dtype)
                with (
                    torch.no_grad(),
                    patch("util.attention.short_causal_attention", side_effect=AssertionError("must use SDPA")),
                ):
                    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)

    def test_disabled_or_training_policy_retains_sdpa_even_without_gradients(self):
        for training, enabled in ((False, False), (True, True)):
            with self.subTest(training=training, enabled=enabled):
                model = SelfAttention(32, 4, causal=True).train(training)
                reference = copy.deepcopy(model)
                model.short_temporal_attention = enabled
                x = torch.randn(2, 2, 32)
                with (
                    torch.no_grad(),
                    patch("util.attention.short_causal_attention", side_effect=AssertionError("must use SDPA")),
                ):
                    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)

    def test_helper_rejects_unsupported_layouts_lengths_and_dtypes(self):
        q, k, v = inputs(2, 4, 1, 2, 16)
        for values in ((q[0], k, v), (q[:, :2], k, v), (q.double(), k, v), inputs(2, 4, 3, 3, 16)):
            with self.subTest(shapes=[value.shape for value in values]), self.assertRaises(ValueError):
                short_causal_attention(*values)


@unittest.skipUnless(os.environ.get("CADRE_RUN_GPU_TESTS") == "1", "set CADRE_RUN_GPU_TESTS=1 to run CUDA checks")
class ShortAttentionCudaTests(unittest.TestCase):
    def test_real_bfloat16_temporal_shapes_and_compiled_changed_input_replays(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for batch, q_len, k_len in ((3480, 1, 2), (3456, 2, 2)):
                with self.subTest(batch=batch, q_len=q_len, k_len=k_len):
                    compiled_reference = torch.compile(
                        reference_attention, mode="reduce-overhead", fullgraph=True, dynamic=False
                    )
                    compiled_short = torch.compile(
                        short_causal_attention, mode="reduce-overhead", fullgraph=True, dynamic=False
                    )
                    previous = None
                    for _ in range(3):
                        values = inputs(batch, 16, q_len, k_len, 64, dtype=torch.bfloat16, device="cuda")
                        torch.compiler.cudagraph_mark_step_begin()
                        expected = compiled_reference(*values).clone()
                        torch.compiler.cudagraph_mark_step_begin()
                        actual = compiled_short(*values).clone()
                        self.assertTrue(torch.isfinite(actual).all())
                        # Flash and FP32 explicit reductions round differently.
                        # BF16 epsilon=1/128; retain the established two-relative/
                        # one-absolute-epsilon local tolerance, not a quality claim.
                        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=8e-3)
                        if q_len == 2:
                            torch.testing.assert_close(actual[..., :1, :], values[2][..., :1, :], rtol=0, atol=0)
                        if previous is not None:
                            self.assertFalse(torch.equal(actual, previous))
                        previous = actual


if __name__ == "__main__":
    unittest.main()

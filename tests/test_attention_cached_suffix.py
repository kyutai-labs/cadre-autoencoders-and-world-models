"""Cached multi-query attention must use every query's true temporal position."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from model_autoencoder import temporal_rope
from util.attention import SelfAttention, causal_mask


class CachedAttentionSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, queries, past=1, kv_heads=4):
        with torch.random.fork_rng():
            torch.manual_seed(61)
            model = SelfAttention(32, 4, kv_heads, gating=True, causal=True).eval()
            values = torch.randn(2, queries + past, 32)
        return model, values, temporal_rope(queries + past, 8, values.device)

    def test_cached_suffix_matches_all_positions_with_rotary_and_gating(self):
        for queries, past in ((1, 1), (2, 1), (4, 3), (8, 1)):
            for kv_heads in (2, 4):
                with self.subTest(queries=queries, past=past, kv_heads=kv_heads), torch.no_grad():
                    model, values, rotary = self.fixture(queries, past, kv_heads)
                    expected, full_kv = model(values, rotary, return_kv=True)
                    _, prefix = model(values[:, :past], tuple(x[:past] for x in rotary), return_kv=True)
                    preserved = tuple(value.clone() for value in prefix)
                    actual, suffix_kv = model(values[:, past:], rotary, kv_cache=prefix, return_kv=True)
                    torch.testing.assert_close(actual, expected[:, past:], rtol=3e-5, atol=2e-6)
                    for value, full in zip(suffix_kv, full_kv, strict=True):
                        # Different Linear row counts may change FP32 GEMM rounding.
                        torch.testing.assert_close(value, full[:, past:], rtol=2e-5, atol=2e-6)
                    for value, before in zip(prefix, preserved, strict=True):
                        torch.testing.assert_close(value, before, rtol=0, atol=0)

    def test_cached_suffix_gradients_match_complete_attention(self):
        reference, values, rotary = self.fixture(8)
        candidate = copy.deepcopy(reference)
        gradient = torch.randn(2, 8, 32)
        results = []
        for model, cached in ((reference, False), (candidate, True)):
            inputs = values.clone().requires_grad_()
            if cached:
                _, prefix = model(inputs[:, :1], tuple(x[:1] for x in rotary), return_kv=True)
                output = model(inputs[:, 1:], rotary, kv_cache=prefix)
            else:
                output = model(inputs, rotary)[:, 1:]
            output.backward(gradient)
            results.append((inputs.grad, [parameter.grad for parameter in model.parameters()]))
        for value, expected in zip([results[1][0]] + results[1][1], [results[0][0]] + results[0][1], strict=True):
            torch.testing.assert_close(value, expected, rtol=3e-5, atol=1e-5)

    def test_future_suffix_tokens_cannot_change_earlier_outputs(self):
        model, values, rotary = self.fixture(8)
        with torch.no_grad():
            _, prefix = model(values[:, :1], tuple(x[:1] for x in rotary), return_kv=True)
            expected = model(values[:, 1:], rotary, kv_cache=prefix)
            changed = values[:, 1:].clone()
            changed[:, 4:] += 100
            actual = model(changed, rotary, kv_cache=prefix)
        torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[:, 4:], expected[:, 4:]))

    def test_symbolic_lower_right_bias_matches_materialized_mask(self):
        torch.manual_seed(17)
        for queries, keys in ((2, 3), (4, 7), (8, 9)):
            with self.subTest(queries=queries, keys=keys):
                query = torch.randn(2, 3, queries, 8)
                key, value = (torch.randn(2, 3, keys, 8) for _ in range(2))
                expected = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=causal_mask(queries, keys, query.device)
                )
                actual = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=causal_lower_right(queries, keys)
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prepared_bias_compiles_fullgraph_without_constructing_bias_in_forward(self):
        model, values, rotary = self.fixture(8)
        model.prepare_causal_bias(8, 9)
        self.assertNotIn("_cached_causal_biases", model.state_dict())
        with torch.no_grad():
            expected = model(values, rotary)[:, 1:]
            _, prefix = model(values[:, :1], tuple(x[:1] for x in rotary), return_kv=True)
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            with patch("util.attention.causal_lower_right", side_effect=AssertionError("constructor during forward")):
                actual = compiled(values[:, 1:], rotary, kv_cache=prefix)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
        for queries, keys in ((1, 2), (8, 8), (9, 8)):
            with self.subTest(queries=queries, keys=keys), self.assertRaises(ValueError):
                model.prepare_causal_bias(queries, keys)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_compiled_cuda_flash_suffix_and_changed_input_replay(self):
        model, values, _ = self.fixture(8)
        model, values = model.cuda(), values.cuda()
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.to(torch.bfloat16)
        rotary = temporal_rope(9, 8, values.device)
        model.prepare_causal_bias(8, 9)
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        # Same BF16 policy and native Flash backend as serving. Shape changes
        # may change accumulation order, so compare within BF16 precision.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for offset in (0.0, 0.2):
                torch.compiler.cudagraph_mark_step_begin()
                changed = values + offset
                expected = model(changed, rotary)[:, 1:].clone()
                _, prefix = model(changed[:, :1], tuple(x[:1] for x in rotary), return_kv=True)
                prefix = tuple(value.clone() for value in prefix)
                preserved = tuple(value.clone() for value in prefix)
                actual = compiled(changed[:, 1:], rotary, kv_cache=prefix).clone()
                torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=4e-3)
                replay = compiled(changed[:, 1:], rotary, kv_cache=prefix).clone()
                torch.testing.assert_close(replay, actual, rtol=0, atol=0)
                for value, before in zip(prefix, preserved, strict=True):
                    torch.testing.assert_close(value, before, rtol=0, atol=0)
                if offset == 0:
                    initial = actual.clone()
                else:
                    self.assertFalse(torch.equal(actual, initial))


if __name__ == "__main__":
    unittest.main()

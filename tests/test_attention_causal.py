"""Causal SDPA fast paths must preserve the original bottom-right mask semantics."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from util.attention import SelfAttention, causal_mask

SDPA = F.scaled_dot_product_attention


def legacy_causal_attention(q, k, v, **kwargs):
    kwargs["attn_mask"] = causal_mask(q.shape[-2], k.shape[-2], q.device)
    kwargs["is_causal"] = False
    return SDPA(q, k, v, **kwargs)


class CausalAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, queries, past, causal=True):
        torch.manual_seed(19)
        model = SelfAttention(16, 4, 2, gating=True, causal=causal)
        x = torch.randn(2, queries, 16)
        cache = tuple(torch.randn(2, past, 2, 4) for _ in range(2)) if past else None
        return model, x, cache

    def test_full_context_uses_native_causality(self):
        model, x, cache = self.fixture(3, 0)
        with patch("util.attention.F.scaled_dot_product_attention", wraps=SDPA) as attention:
            model(x, kv_cache=cache)
        self.assertTrue(attention.call_args.kwargs["is_causal"])
        self.assertIsNone(attention.call_args.kwargs["attn_mask"])

    def test_cached_single_query_can_attend_every_key(self):
        model, x, cache = self.fixture(1, 3)
        with patch("util.attention.F.scaled_dot_product_attention", wraps=SDPA) as attention:
            model(x, kv_cache=cache)
        self.assertFalse(attention.call_args.kwargs["is_causal"])
        self.assertIsNone(attention.call_args.kwargs["attn_mask"])

    def test_multiple_cached_queries_keep_offset_mask(self):
        model, x, cache = self.fixture(2, 3)
        with patch("util.attention.F.scaled_dot_product_attention", wraps=SDPA) as attention:
            model(x, kv_cache=cache)
        self.assertFalse(attention.call_args.kwargs["is_causal"])
        torch.testing.assert_close(attention.call_args.kwargs["attn_mask"], causal_mask(2, 5, x.device))

    def test_spatial_attention_stays_bidirectional(self):
        model, x, cache = self.fixture(3, 0, causal=False)
        with patch("util.attention.F.scaled_dot_product_attention", wraps=SDPA) as attention:
            model(x, kv_cache=cache)
        self.assertFalse(attention.call_args.kwargs["is_causal"])
        self.assertIsNone(attention.call_args.kwargs["attn_mask"])

    def test_outputs_caches_and_gradients_match_explicit_mask(self):
        for queries, past in ((3, 0), (1, 3), (2, 3)):
            with self.subTest(queries=queries, past=past):
                model, x, cache = self.fixture(queries, past)
                reference = copy.deepcopy(model)
                actual_x, expected_x = x.clone().requires_grad_(), x.clone().requires_grad_()
                phases = torch.randn(queries + past, 2).repeat_interleave(2, dim=-1)
                rotary = (phases.cos(), phases.sin())
                actual, actual_cache = model(actual_x, rotary, cache, return_kv=True)
                with patch("util.attention.F.scaled_dot_product_attention", legacy_causal_attention):
                    expected, expected_cache = reference(expected_x, rotary, cache, return_kv=True)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                for value, target in zip(actual_cache, expected_cache, strict=True):
                    torch.testing.assert_close(value, target, rtol=0, atol=0)
                actual.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=2e-5, atol=2e-6)
                for value, target in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(value.grad, target.grad, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16_flash_matches_explicit_mask_math(self):
        torch.manual_seed(29)
        model = SelfAttention(128, 4, 2, gating=True, causal=True).cuda().eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
        max_difference = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for queries, past in ((8, 0), (1, 7)):
                with self.subTest(queries=queries, past=past):
                    x = torch.randn(2, queries, 128, device="cuda")
                    cache = (
                        tuple(torch.randn(2, past, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2))
                        if past
                        else None
                    )
                    phases = torch.randn(queries + past, 16, device="cuda").repeat_interleave(2, dim=-1)
                    rotary = (phases.cos(), phases.sin())
                    # Forcing FLASH_ATTENTION makes this a backend eligibility test
                    # as well as a numerical check; silent fallback cannot pass it.
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        actual, actual_cache = model(x, rotary, cache, return_kv=True)
                    with (
                        sdpa_kernel(SDPBackend.MATH),
                        patch("util.attention.F.scaled_dot_product_attention", legacy_causal_attention),
                    ):
                        expected, expected_cache = model(x, rotary, cache, return_kv=True)
                    difference = (actual.float() - expected.float()).abs().max().item()
                    max_difference = max(max_difference, difference)
                    # BF16 attention kernels differ in softmax reduction/rounding.
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1.6e-2,
                        atol=4e-3,
                        msg=f"BF16 FlashAttention max absolute difference: {difference:.8g}",
                    )
                    for value, target in zip(actual_cache, expected_cache, strict=True):
                        torch.testing.assert_close(value, target, rtol=0, atol=0)
        print(f"CUDA BF16 FlashAttention maximum absolute difference: {max_difference:.8g}")


if __name__ == "__main__":
    unittest.main()

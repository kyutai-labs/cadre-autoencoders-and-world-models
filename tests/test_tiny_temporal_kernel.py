"""Narrow production guards and compiled CUDA replay; no model checkpoints."""

import copy
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from util.tiny_temporal_attention import supports_tiny_temporal_attention, tiny_temporal_attention


@dataclass
class TensorMetadata:
    shape: tuple
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16
    requires_grad: bool = False
    channel_stride: int = 1
    ndim: int = 4

    def stride(self, index):
        return self.channel_stride


class TinyTemporalGuardTests(unittest.TestCase):
    def test_metadata_guard_is_narrow_and_does_not_probe_gpu(self):
        q, k, v = [TensorMetadata(shape) for shape in ((580, 16, 8, 64), (580, 16, 9, 64), (580, 16, 9, 64))]
        with torch.no_grad(), patch("torch.cuda.is_available", side_effect=AssertionError("GPU probe")):
            self.assertTrue(supports_tiny_temporal_attention(q, k, v))
            for value in (q, k, v):
                value.requires_grad = True
                self.assertFalse(supports_tiny_temporal_attention(q, k, v))
                value.requires_grad = False
            for attribute, changed in (
                ("dtype", torch.float32),
                ("device", torch.device("cpu")),
                ("channel_stride", 2),
                ("ndim", 3),
            ):
                original = getattr(q, attribute)
                setattr(q, attribute, changed)
                self.assertFalse(supports_tiny_temporal_attention(q, k, v))
                setattr(q, attribute, original)
            q.shape = (580, 16, 9, 64)
            self.assertFalse(supports_tiny_temporal_attention(q, k, v))
        self.assertFalse(supports_tiny_temporal_attention(q, k, v))

    def test_cpu_tensors_and_grad_enabled_calls_stay_unsupported(self):
        q = torch.zeros(1, 16, 16, 64, dtype=torch.bfloat16)
        with torch.no_grad():
            self.assertFalse(supports_tiny_temporal_attention(q, q, q))
            with self.assertRaisesRegex(ValueError, "CUDA BF16"):
                tiny_temporal_attention(q, q, q)
        q, k, v = [TensorMetadata((2, 16, 16, 64)) for _ in range(3)]
        self.assertFalse(supports_tiny_temporal_attention(q, k, v))
        with torch.no_grad():
            self.assertTrue(supports_tiny_temporal_attention(q, k, v))
        q.shape = (2, 8, 16, 64)
        with torch.no_grad():
            self.assertFalse(supports_tiny_temporal_attention(q, k, v))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for kernel/graph checks")
class TinyTemporalCudaTests(unittest.TestCase):
    def reference(self, q, k, v):
        bias = None if q.shape[-2] == k.shape[-2] else causal_lower_right(q.shape[-2], k.shape[-2])
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=bias is None)

    def inputs(self, q_length, k_length, batch=3):
        # Distinct nonzero storage offsets and strides exercise pointer handling.
        result = []
        for index, length in enumerate((q_length, k_length, k_length), 1):
            storage = torch.randn(batch + 1, length + 2, 16 * index, 64, dtype=torch.bfloat16, device="cuda")
            result.append(storage[1:, 1 : length + 1, 16 * (index - 1) : 16 * index].transpose(1, 2))
        return result

    @torch.inference_mode()
    def test_fullgraph_replay_changes_all_inputs_and_retains_owned_outputs(self):
        for q_length, k_length in ((8, 9), (16, 16)):
            inputs = self.inputs(q_length, k_length)
            compiled = torch.compile(tiny_temporal_attention, fullgraph=True, dynamic=False, mode="reduce-overhead")
            torch.compiler.cudagraph_mark_step_begin()
            before = compiled(*inputs).clone()
            torch.testing.assert_close(before, self.reference(*inputs), rtol=0.016, atol=0.008)
            self.assertEqual(before.transpose(1, 2).reshape(3, q_length, 1024).shape, (3, q_length, 1024))
            saved = before.clone()
            for value in inputs:
                value.copy_(torch.randn_like(value))
            torch.compiler.cudagraph_mark_step_begin()
            after = compiled(*inputs).clone()
            torch.testing.assert_close(after, self.reference(*inputs), rtol=0.016, atol=0.008)
            torch.testing.assert_close(before, saved, rtol=0, atol=0)
            self.assertFalse(torch.equal(before, after))
            torch.compiler.cudagraph_mark_step_begin()
            repeated = compiled(*inputs).clone()
            torch.testing.assert_close(after, repeated, rtol=0, atol=0)

    @torch.inference_mode()
    def test_uniform_scores_head_sentinels_future_keys_and_concentrated_logits(self):
        for q_length, k_length in ((8, 9), (16, 16)):
            q, k, v = self.inputs(q_length, k_length)
            q.zero_()
            k.zero_()
            for head in range(16):
                v[:, head] = head + torch.arange(k_length, device="cuda")[:, None] / 16
            before = tiny_temporal_attention(q, k, v)
            expected = torch.stack(
                [v[..., :end, :].float().mean(-2) for end in range(k_length - q_length + 1, k_length + 1)], -2
            ).to(torch.bfloat16)
            torch.testing.assert_close(before, expected, rtol=0, atol=0)
            v[..., -1, :] += 32
            after = tiny_temporal_attention(q, k, v)
            torch.testing.assert_close(before[..., :-1, :], after[..., :-1, :], rtol=0, atol=0)
            self.assertFalse(torch.equal(before[..., -1, :], after[..., -1, :]))
            q.normal_().mul_(8)
            k.normal_().mul_(8)
            v.normal_()
            actual = tiny_temporal_attention(q, k, v)
            torch.testing.assert_close(actual, self.reference(q, k, v), rtol=0.016, atol=0.008)
            self.assertTrue(torch.isfinite(actual).all())

    @torch.inference_mode()
    def test_fullgraph_self_attention_with_rotary_cache_and_nonunit_qk_scales(self):
        from model_autoencoder import temporal_rope
        from util.attention import SelfAttention

        for q_length, k_length in ((8, 9), (16, 16)):
            module = SelfAttention(1024, 16, gating=True, causal=True).cuda().bfloat16().eval()
            module.q_ln.qk_scale.uniform_(0.7, 1.3)
            module.k_ln.qk_scale.uniform_(0.7, 1.3)
            native = copy.deepcopy(module)
            if q_length != k_length:
                module.prepare_causal_bias(q_length, k_length)
                native.prepare_causal_bias(q_length, k_length)
            module.tiny_temporal_attention = True
            compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="reduce-overhead")
            x = torch.randn(3, q_length, 1024, dtype=torch.bfloat16, device="cuda")
            cache = (
                None
                if q_length == k_length
                else tuple(torch.randn(3, 1, 16, 64, dtype=torch.bfloat16, device="cuda") for _ in range(2))
            )
            rope = temporal_rope(k_length, 64, x.device)
            preserved = None if cache is None else tuple(value.clone() for value in cache)
            torch.compiler.cudagraph_mark_step_begin()
            before = compiled(x, rope, kv_cache=cache).clone()
            torch.testing.assert_close(before, native(x, rope, kv_cache=cache), rtol=0.016, atol=0.008)
            if cache is not None:
                for value, original in zip(cache, preserved, strict=True):
                    torch.testing.assert_close(value, original, rtol=0, atol=0)
                    value.normal_()
            x.normal_()
            torch.compiler.cudagraph_mark_step_begin()
            after = compiled(x, rope, kv_cache=cache).clone()
            torch.testing.assert_close(after, native(x, rope, kv_cache=cache), rtol=0.016, atol=0.008)
            self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()

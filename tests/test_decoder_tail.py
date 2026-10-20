"""Suffix decoding preserves full-window predictions and sampling history."""

import copy
import types
import unittest
from types import SimpleNamespace

import torch
from einops import rearrange

from model_autoencoder import CADRE, PATCH, CADREConfig, Decoder, spatial_rope, temporal_rope
from model_world_model import StreamingDecoder


def legacy_trunk(self, x, adaln, z, last_frames=None):
    assert last_frames is None
    tau, head_dim = self.cfg.tau, self.cfg.dim // self.cfg.num_heads
    rescale = self.cfg.rope_spatial_rescale if self.training else None
    _, t, h, w, _ = x.shape
    rope_space = spatial_rope(h, w, head_dim, x.device, rescale)
    rope_time = temporal_rope(t + self.cfg.num_time_registers, head_dim, x.device)
    x = self._run_blocks(self.blocks, x, adaln, rope_space, rope_time, registers=True)
    x = rearrange(self.split_frames(x), "b t h w (ut c) -> b (t ut) h w c", ut=tau).contiguous()
    adaln = adaln.repeat_interleave(tau, 1)
    rope_space = spatial_rope(h, w, head_dim, x.device, rescale)
    rope_time = temporal_rope(t * tau, head_dim, x.device)
    x = self._run_blocks(self.tail_blocks, x, adaln, rope_space, rope_time, registers=False)
    x = rearrange(self.unpatchify(self.norm_out(x)), "b t h w (c p1 p2) -> b t c (h p1) (w p2)", p1=PATCH, p2=PATCH)
    return torch.tanh(self.refine_head(x, z.repeat_interleave(tau, dim=1)))


class DecoderTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.nnpack_context = torch.backends.nnpack.flags(enabled=False)
        cls.nnpack_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.nnpack_context.__exit__(None, None, None)
        torch.set_num_threads(cls.previous_threads)

    def fixture(self):
        torch.manual_seed(23)
        model = CADRE.__new__(CADRE)
        torch.nn.Module.__init__(model)
        model.config = CADREConfig(
            tau=2,
            stride=1,
            latent_dim=4,
            resolution=(16, 16),
            num_frames=16,
            dim=32,
            depth=2,
            tail_depth=1,
            num_heads=4,
            refine_dim=8,
            refine_dilations=(1, 2),
        )
        model.decoder = Decoder(model.config)
        with torch.no_grad():
            model.decoder.patchify.weight.normal_(std=0.02)
            model.decoder.adaln_head[1].weight.normal_(std=0.02)
            model.decoder.refine_head.latent_skip.weight.normal_(std=0.02)
        return model.eval(), torch.randn(2, 8, 4, 1, 1), torch.randn(2, 16, 3, 16, 16)

    def test_suffix_matches_full_decode_for_multiple_steps_and_lengths(self):
        model, z, noise = self.fixture()
        original_noise = noise.clone()
        with torch.no_grad():
            for steps in (1, 2, 3):
                expected = model.decode(z, steps, noise=noise)
                for frames in (1, 2, 5, 16):
                    with self.subTest(steps=steps, frames=frames):
                        actual = model.decode_tail(z, steps, frames, noise=noise)
                        torch.testing.assert_close(actual, expected[:, -frames:], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)

    def test_only_final_step_reduces_pixel_work_and_keeps_attention_context(self):
        model, z, noise = self.fixture()
        pixels, attention = [], []
        handles = [
            model.decoder.unpatchify.register_forward_pre_hook(lambda _, args: pixels.append(args[0].shape[1])),
            model.decoder.tail_blocks[0].register_forward_pre_hook(lambda _, args: attention.append(args[0].shape[1])),
        ]
        try:
            with torch.no_grad():
                model.decode(z, 3, noise=noise, last_frames=2)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(pixels, [16, 16, 6])  # two requested frames plus four frames of refinement history
        self.assertEqual(attention, [16, 16, 16])

    def test_streaming_optimized_and_generic_fallback_match(self):
        model, z, _ = self.fixture()
        common = dict(config=model.config, unnormalize=lambda z: z, decode=model.decode)
        optimized = SimpleNamespace(**common, decode_tail=model.decode_tail)
        generic = SimpleNamespace(**common)
        window = z.permute(0, 1, 3, 4, 2)
        actual = StreamingDecoder(optimized, window, 2, 0.99, torch.Generator().manual_seed(3))
        expected = StreamingDecoder(generic, window, 2, 0.99, torch.Generator().manual_seed(3))
        with torch.no_grad():
            for _ in range(3):
                next_z = torch.randn_like(window[:, :1])
                torch.testing.assert_close(actual.push(next_z), expected.push(next_z), rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual.noise, expected.noise, rtol=0, atol=0)

    def test_full_training_path_and_gradients_match_legacy(self):
        model, z, noise = self.fixture()
        reference = copy.deepcopy(model)
        reference.decoder.run_trunk = types.MethodType(legacy_trunk, reference.decoder)
        t, d = torch.rand(2, 8), torch.rand(2, 8)
        outputs, gradients = [], []
        for candidate in (model, reference):
            candidate.train()
            inputs = [value.clone().requires_grad_() for value in (z, noise)]
            output = candidate.decoder.flow_forward(*inputs, t, d)
            output.square().mean().backward()
            outputs.append(output)
            gradients.append([value.grad for value in inputs])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for actual, expected in zip(*gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            if actual.grad is None:
                self.assertIsNone(expected.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_invalid_suffix_is_rejected(self):
        model, z, noise = self.fixture()
        for frames in (0, -1, 17):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "last_frames"):
                model.decode(z, 1, noise=noise, last_frames=frames)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16_suffix_matches_full_decode(self):
        model, z, noise = self.fixture()
        model, z, noise = model.cuda(), z.cuda(), noise.cuda()
        # Match the demo: Linear weights are stored in BF16; autocast handles the
        # remaining operations. Different convolution shapes can round differently.
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
        max_difference = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for steps in (1, 2, 3):
                expected = model.decode(z, steps, noise=noise)
                for frames in (1, 2, 5):
                    with self.subTest(steps=steps, frames=frames):
                        actual = model.decode_tail(z, steps, frames, noise=noise)
                        target = expected[:, -frames:]
                        difference = (actual.float() - target.float()).abs().max().item()
                        max_difference = max(max_difference, difference)
                        # BF16 epsilon is 1/128. Allow roughly one absolute epsilon
                        # plus two relative epsilons for alternate convolution kernels.
                        torch.testing.assert_close(
                            actual,
                            target,
                            rtol=1.6e-2,
                            atol=8e-3,
                            msg=f"BF16 suffix max absolute difference: {difference:.8g}",
                        )
        print(f"CUDA BF16 decoder suffix maximum absolute difference: {max_difference:.8g}")


if __name__ == "__main__":
    unittest.main()

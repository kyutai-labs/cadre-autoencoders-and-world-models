"""Progressive refinement preserves suffix predictions and the full training path."""

import copy
import types
import unittest

import torch
from einops import rearrange
from torch.nn import functional as F

from model_autoencoder import CADRE, CADREConfig, Decoder, RefineHead


def legacy_refine(self, x, z, last_frames=None):
    """Unchanged pre-proposal arithmetic; caller chooses its final output suffix."""
    b = x.shape[0]
    skip = F.interpolate(self.latent_skip(rearrange(z, "b t c h w -> (b t) c h w")), x.shape[-2:], mode="nearest")
    x = x + rearrange(skip, "(b t) c h w -> b t c h w", b=b)
    for spatial, temporal in zip(self.spatial, self.temporal, strict=True):
        h = spatial(rearrange(x, "b t c h w -> (b t) c h w"))
        x = x + rearrange(h, "(b t) c h w -> b t c h w", b=b)
        h = rearrange(x, "b t c h w -> b c t h w")
        h = temporal(torch.cat([h[:, :, :1].repeat(1, 1, 2, 1, 1), h], dim=2))
        x = x + rearrange(h, "b c t h w -> b t c h w")
    h = self.out_conv(F.silu(self.out_norm(rearrange(x, "b t c h w -> (b t) c h w"))))
    return rearrange(h, "(b t) c h w -> b t c h w", b=b)


class RefineSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.nnpack = torch.backends.nnpack.flags(enabled=False)
        cls.nnpack.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.nnpack.__exit__(None, None, None)
        torch.set_num_threads(cls.previous_threads)

    def head(self, dilations=(1, 2)):
        with torch.random.fork_rng():
            torch.manual_seed(82)
            return RefineHead(8, 3, 4, dilations)

    def model(self):
        with torch.random.fork_rng():
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
            return model.eval()

    def test_suffix_outputs_input_gradients_and_parameter_gradients_match_legacy(self):
        generator = torch.Generator().manual_seed(64)
        for length in (2, 4, 6, 16):
            for suffix in sorted({value for value in (1, 2, 5, length) if value <= length}):
                with self.subTest(length=length, suffix=suffix):
                    expected_model, actual_model = self.head(), self.head()
                    x = torch.randn(2, length, 8, 8, 8, generator=generator)
                    z = torch.randn(2, length, 4, 2, 2, generator=generator)
                    gradient = torch.randn(2, suffix, 3, 8, 8, generator=generator)
                    results = []
                    for model, optimized in ((expected_model, False), (actual_model, True)):
                        inputs = [value.clone().requires_grad_() for value in (x, z)]
                        output = model(*inputs, last_frames=suffix) if optimized else legacy_refine(model, *inputs)
                        output = output[:, -suffix:]
                        output.backward(gradient)
                        results.append(
                            (output, [value.grad for value in inputs], [value.grad for value in model.parameters()])
                        )
                    torch.testing.assert_close(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)
                    pairs = zip(results[1][1] + results[1][2], results[0][1] + results[0][2], strict=True)
                    for actual, expected in pairs:
                        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)

    def test_default_training_path_is_bit_identical_to_legacy(self):
        x, z = torch.randn(2, 6, 8, 8, 8), torch.randn(2, 6, 4, 2, 2)
        expected_model, actual_model = self.head().train(), self.head().train()
        results = []
        for model, optimized in ((expected_model, False), (actual_model, True)):
            inputs = [value.clone().requires_grad_() for value in (x, z)]
            output = model(*inputs) if optimized else legacy_refine(model, *inputs)
            output.square().mean().backward()
            results.append((output, [value.grad for value in inputs], [value.grad for value in model.parameters()]))
        for actual, expected in zip(
            [results[1][0]] + results[1][1] + results[1][2],
            [results[0][0]] + results[0][1] + results[0][2],
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_short_context_padding_and_progressively_smaller_work(self):
        for length, spatial_frames, temporal_inputs, temporal_outputs in (
            (2, [2, 2], [4, 4], [2, 2]),
            (4, [4, 4], [6, 4], [4, 2]),
            (6, [6, 4], [6, 4], [4, 2]),
        ):
            with self.subTest(length=length):
                model = self.head().eval()
                observed = {"spatial": [], "temporal_input": [], "temporal_output": [], "output": []}
                handles = []

                def temporal_hook(_, args, output, observed=observed):
                    observed["temporal_input"].append(args[0].shape[2])
                    observed["temporal_output"].append(output.shape[2])

                for spatial, temporal in zip(model.spatial, model.temporal, strict=True):
                    handles.append(
                        spatial.register_forward_pre_hook(
                            lambda _, args, observed=observed: observed["spatial"].append(args[0].shape[0] // 2)
                        )
                    )
                    handles.append(temporal.register_forward_hook(temporal_hook))
                handles.append(
                    model.out_conv.register_forward_pre_hook(
                        lambda _, args, observed=observed: observed["output"].append(args[0].shape[0] // 2)
                    )
                )
                try:
                    with torch.no_grad():
                        model(torch.randn(2, length, 8, 8, 8), torch.randn(2, length, 4, 2, 2), last_frames=2)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(
                    observed,
                    {
                        "spatial": spatial_frames,
                        "temporal_input": temporal_inputs,
                        "temporal_output": temporal_outputs,
                        "output": [2],
                    },
                )

    def test_real_decoder_suffix_matches_legacy_for_multiple_sampling_steps(self):
        candidate = self.model()
        reference = copy.deepcopy(candidate)
        reference.decoder.refine_head.forward = types.MethodType(legacy_refine, reference.decoder.refine_head)
        generator = torch.Generator().manual_seed(71)
        for length in (2, 4, 6, 16):
            z = torch.randn(2, length // 2, 4, 1, 1, generator=generator)
            noise = torch.randn(2, length, 3, 16, 16, generator=generator)
            preserved_noise = noise.clone()
            for steps in (1, 2, 3):
                for suffix in sorted({value for value in (1, 2, 5, length) if value <= length}):
                    with self.subTest(length=length, steps=steps, suffix=suffix), torch.no_grad():
                        expected = reference.decode_tail(z, steps, suffix, noise=noise)
                        actual = candidate.decode_tail(z, steps, suffix, noise=noise)
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(noise, preserved_noise, rtol=0, atol=0)

    def test_checkpoint_parameters_and_buffers_remain_unchanged(self):
        model = self.model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            model.decode_tail(torch.randn(2, 8, 4, 1, 1), 2, 2, noise=torch.randn(2, 16, 3, 16, 16))
        self.assertEqual(state.keys(), model.state_dict().keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        fresh = self.model()
        fresh.load_state_dict(state, strict=True)

    def test_other_refinement_depths_and_invalid_suffixes(self):
        for dilations in ((), (1,), (1, 2, 3)):
            model = self.head(dilations).eval()
            x, z = torch.randn(2, 6, 8, 8, 8), torch.randn(2, 6, 4, 2, 2)
            with torch.no_grad():
                expected = legacy_refine(model, x, z)
                actual = model(x, z, last_frames=2)
            torch.testing.assert_close(actual, expected[:, -2:], rtol=2e-5, atol=2e-6)
        for invalid in (-1, 0, 7):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "last_frames"):
                self.head()(x, z, last_frames=invalid)


if __name__ == "__main__":
    unittest.main()

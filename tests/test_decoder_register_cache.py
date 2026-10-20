"""A cached constant temporal prefix must preserve the decoder's inference state."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from demo import ConcurrentDemo, build, parse_args
from model_autoencoder import CADREConfig, Decoder
from util.attention import SelfAttention


class DecoderRegisterCacheTests(unittest.TestCase):
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

    def fixture(self, *, time_registers=1):
        with torch.random.fork_rng():
            torch.manual_seed(83)
            decoder = Decoder(
                CADREConfig(
                    tau=2,
                    stride=1,
                    latent_dim=4,
                    resolution=(32, 32),
                    num_frames=16,
                    dim=32,
                    depth=4,
                    tail_depth=1,
                    num_heads=4,
                    num_registers=2,
                    num_time_registers=time_registers,
                    refine_dim=8,
                    refine_dilations=(1,),
                )
            ).eval()
            with torch.no_grad():
                decoder.patchify.weight.normal_(std=0.04)
                decoder.adaln_head[1].weight.normal_(std=0.04)
                decoder.refine_head.latent_skip.weight.normal_(std=0.04)
                # Default LayerScale=1e-4 would hide a broken prefix behind tiny
                # residuals. Exercise substantial learned prefix contributions.
                for block in (*decoder.blocks, *decoder.tail_blocks):
                    block.space_ls.gamma.fill_(0.4)
                    block.time_ls.gamma.fill_(0.4)
                    block.mlp_ls.gamma.fill_(0.4)
                    block.adaln_offset.normal_(std=0.1)
        return decoder

    def inputs(self, *, batch=2, frames=1, grid=(2, 2), seed=19):
        generator = torch.Generator().manual_seed(seed)
        h, w = grid
        return (
            torch.randn(batch, frames, 4, h, w, generator=generator),
            torch.randn(batch, frames * 2, 3, h * 16, w * 16, generator=generator),
            torch.rand(batch, frames, generator=generator),
            torch.rand(batch, frames, generator=generator),
        )

    def spatial_rows(self, decoder):
        rows = []
        handle = decoder.blocks[0].space_attn.register_forward_pre_hook(lambda _, args: rows.append(args[0].shape[0]))
        self.addCleanup(handle.remove)
        return rows

    def test_cached_single_frame_matches_reference_across_batches_and_inputs(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache()
        self.assertTrue(decoder.time_register_cache_ready)
        for block in decoder.blocks:
            self.assertEqual(
                set(block.time_attn._cached_causal_biases), {(frames, frames + 1) for frames in range(2, 9)}
            )
        rows = self.spatial_rows(decoder)
        with torch.no_grad():
            for batch in (1, 2, 3):
                for seed in (19, 73):
                    with self.subTest(batch=batch, seed=seed):
                        inputs = self.inputs(batch=batch, seed=seed)
                        expected = reference.flow_forward(*inputs)
                        actual = decoder.flow_forward(*inputs)
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
                        self.assertEqual(rows[-1], batch)  # prefix frame never enters runtime spatial attention
                        self.assertEqual(actual.shape, (batch, 2, 3, 32, 32))

    def test_state_dict_parameters_rng_inputs_and_cache_are_unchanged(self):
        decoder = self.fixture()
        state = {name: value.clone() for name, value in decoder.state_dict().items()}
        parameters = dict(decoder.named_parameters())
        rng = torch.get_rng_state().clone()
        decoder.prepare_time_register_cache()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(set(state), set(decoder.state_dict()))
        for name, value in decoder.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for name, parameter in decoder.named_parameters():
            self.assertIs(parameter, parameters[name])
        cached = {name: value.clone() for name, value in decoder.named_buffers()}
        self.assertTrue(cached)
        for name, buffer in decoder.named_buffers():
            self.assertFalse(buffer.requires_grad, name)
            self.assertNotIn(name, decoder.state_dict())
        inputs = self.inputs(frames=8)
        original_inputs = tuple(value.clone() for value in inputs)
        with torch.no_grad():
            decoder.flow_forward(*inputs)
            decoder.flow_forward(*self.inputs(batch=3, frames=4, seed=73))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for actual, expected in zip(inputs, original_inputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, buffer in decoder.named_buffers():
            torch.testing.assert_close(buffer, cached[name], rtol=0, atol=0)

    def test_full_context_matches_reference_and_skips_only_constant_prefix(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache()
        rows = self.spatial_rows(decoder)
        with torch.no_grad():
            for batch in (1, 2, 3):
                for frames in (2, 4, 8):
                    for seed in (19, 73):
                        with self.subTest(batch=batch, frames=frames, seed=seed):
                            inputs = self.inputs(batch=batch, frames=frames, seed=seed)
                            actual = decoder.flow_forward(*inputs)
                            expected = reference.flow_forward(*inputs)
                            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
                            self.assertEqual(rows[-1], batch * frames)
        self.assertTrue(decoder.time_register_cache_ready)

    def test_compiled_full_context_cache_matches_changed_inputs(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache()
        compiled = torch.compile(decoder.flow_forward, backend="eager", fullgraph=True)
        with torch.no_grad():
            for seed in (19, 73):
                inputs = self.inputs(batch=2, frames=8, seed=seed)
                actual = compiled(*inputs, last_frames=2)
                expected = reference.flow_forward(*inputs, last_frames=2)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    def test_other_spatial_grid_uses_original_path(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache()
        rows = self.spatial_rows(decoder)
        with torch.no_grad():
            inputs = self.inputs(batch=2, frames=4, grid=(1, 2))
            actual = decoder.flow_forward(*inputs)
            expected = reference.flow_forward(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(rows[-1], 2 * (4 + 1))
        self.assertTrue(decoder.time_register_cache_ready)

    def test_autocast_dtype_and_explicit_spatial_grid_select_matching_cache(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache(input_dtype=torch.float32)
        rows = self.spatial_rows(decoder)
        inputs = self.inputs()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = decoder.flow_forward(*inputs), reference.flow_forward(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(rows[-1], 4)  # BF16 activations cannot consume the FP32 cache
        summary = decoder.prepare_time_register_cache(input_dtype=torch.bfloat16)
        self.assertEqual(summary["input_dtype"], "torch.bfloat16")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = decoder.flow_forward(*inputs), reference.flow_forward(*inputs)
            torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=8e-3)
            self.assertEqual(rows[-1], 2)
        summary = decoder.prepare_time_register_cache(spatial_grid=(1, 2), input_dtype=torch.float32)
        self.assertEqual(summary["spatial_grid"], [1, 2])
        self.assertEqual(summary["layers"], len(decoder.blocks))
        self.assertGreater(summary["cached_bytes"], 0)
        inputs = self.inputs(grid=(1, 2))
        with torch.no_grad():
            torch.testing.assert_close(
                decoder.flow_forward(*inputs), reference.flow_forward(*inputs), rtol=2e-5, atol=3e-6
            )
        self.assertEqual(rows[-1], 2)

    def test_unsupported_prefix_and_invalid_preparation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.fixture(time_registers=2).prepare_time_register_cache()
        decoder = self.fixture().train()
        with self.assertRaisesRegex(ValueError, "eval"):
            decoder.prepare_time_register_cache()
        decoder.eval()
        for kwargs in ({"spatial_grid": (0, 2)}, {"input_dtype": torch.int32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                decoder.prepare_time_register_cache(**kwargs)
        self.assertFalse(decoder.time_register_cache_ready)

    def test_eval_with_gradients_uses_original_differentiable_path(self):
        decoder = self.fixture()
        reference = copy.deepcopy(decoder)
        decoder.prepare_time_register_cache()
        rows = self.spatial_rows(decoder)
        outputs, gradients = [], []
        for candidate in (decoder, reference):
            inputs = tuple(value.requires_grad_() for value in self.inputs())
            output = candidate.flow_forward(*inputs)
            output.square().mean().backward()
            outputs.append(output)
            gradients.append(tuple(value.grad.clone() for value in inputs))
        self.assertEqual(rows, [4])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for actual, expected in zip(*gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(decoder.parameters(), reference.parameters(), strict=True):
            if actual.grad is None:
                self.assertIsNone(expected.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_training_loading_and_dtype_changes_invalidate_cache(self):
        decoder = self.fixture()
        state = copy.deepcopy(decoder.state_dict())
        for change in (lambda: decoder.train(), lambda: decoder.load_state_dict(state), lambda: decoder.double()):
            decoder.eval()
            decoder.prepare_time_register_cache()
            self.assertTrue(decoder.time_register_cache_ready)
            change()
            self.assertFalse(decoder.time_register_cache_ready)
        decoder.float().eval()
        decoder.prepare_time_register_cache()
        decoder.clear_time_register_cache()
        self.assertFalse(decoder.time_register_cache_ready)
        reference = self.fixture().train()
        decoder.train()
        inputs = self.inputs()
        torch.testing.assert_close(decoder.flow_forward(*inputs), reference.flow_forward(*inputs), rtol=0, atol=0)

    def test_cli_build_validation_and_session_announcements(self):
        self.assertFalse(parse_args([]).cache_decoder_prefix)
        decoder = self.fixture()
        autoencoder = SimpleNamespace(config=decoder.cfg, decoder=decoder)
        common = ["--device", "cpu", "--no-compile", "--data", "driving=/unused", "--cache_decoder_prefix"]
        with (
            patch("demo.load_models", return_value=(autoencoder, {"driving": SimpleNamespace()})),
            patch("demo.EnvironmentDataset", return_value=[]) as dataset,
            patch.object(decoder, "prepare_time_register_cache") as prepare,
        ):
            args = parse_args(common)
            demo = build(args, checkpoints={"driving": "unused"})
            self.assertEqual(demo.decoder_context_frames, autoencoder.config.num_frames)
            dataset.assert_called_once()
        self.addCleanup(demo.executor.shutdown, wait=True)
        prepare.assert_called_once_with()
        pool = ConcurrentDemo(demo, 2)
        for slot in pool.slots:
            slot.reset()
            self.assertTrue(slot.cache_decoder_prefix)
            self.assertTrue(slot.hello()["cache_decoder_prefix"])
            self.assertFalse(slot.hello()["short_decoder_attention"])
            slot.rollouts["driving"] = SimpleNamespace(index=0, key="clip", seed=17, steps=1, remaining=lambda: 2)
            self.assertTrue(slot.info("driving")["cache_decoder_prefix"])

    def test_short_attention_build_changes_only_ae_time_layers_before_cache_preparation(self):
        self.assertFalse(parse_args([]).short_decoder_attention)
        decoder = self.fixture()
        autoencoder = SimpleNamespace(config=decoder.cfg, decoder=decoder)
        world_model = SimpleNamespace(space_attn=SelfAttention(32, 4), time_attn=SelfAttention(32, 4, causal=True))

        def prepared_after_policy_change():
            for block in (*decoder.blocks, *decoder.tail_blocks):
                self.assertTrue(block.time_attn.short_temporal_attention)
                self.assertFalse(block.space_attn.short_temporal_attention)
            self.assertFalse(world_model.time_attn.short_temporal_attention)
            self.assertFalse(world_model.space_attn.short_temporal_attention)

        args = parse_args(
            [
                "--device",
                "cpu",
                "--no-compile",
                "--data",
                "driving=/unused",
                "--short_decoder_attention",
                "--cache_decoder_prefix",
                "--decoder_context_frames",
                "2",
            ]
        )
        with (
            patch("demo.load_models", return_value=(autoencoder, {"driving": world_model})),
            patch("demo.EnvironmentDataset", return_value=[]),
            patch.object(decoder, "prepare_time_register_cache", side_effect=prepared_after_policy_change) as prepare,
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        prepare.assert_called_once_with()
        for slot in ConcurrentDemo(demo, 2).slots:
            slot.reset()
            self.assertTrue(slot.short_decoder_attention)
            self.assertTrue(slot.hello()["short_decoder_attention"])
            slot.rollouts["driving"] = SimpleNamespace(index=0, key="clip", seed=17, steps=1, remaining=lambda: 2)
            self.assertTrue(slot.info("driving")["short_decoder_attention"])


if __name__ == "__main__":
    unittest.main()

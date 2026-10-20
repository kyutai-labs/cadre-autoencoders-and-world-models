"""Final tail FFN pruning retains full temporal attention and checkpoint semantics."""

import copy
import types
import unittest

import torch

from model_autoencoder import CADRE, CADREConfig, Decoder, SpaceTimeBlock, spatial_rope, temporal_rope


def full_tail_blocks(self, blocks, x, adaln, rope_space, rope_time, registers, last_frames=None):
    return Decoder._run_blocks(self, blocks, x, adaln, rope_space, rope_time, registers, last_frames=None)


class DecoderTailFFNTests(unittest.TestCase):
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

    def model(self, tail_depth=3):
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
                depth=4,
                tail_depth=tail_depth,
                num_heads=4,
                refine_dim=8,
                refine_dilations=(1, 2),
            )
            model.decoder = Decoder(model.config)
            with torch.no_grad():
                model.decoder.patchify.weight.normal_(std=0.02)
                model.decoder.adaln_head[1].weight.normal_(std=0.02)
                model.decoder.refine_head.latent_skip.weight.normal_(std=0.02)
                for block in model.decoder.tail_blocks:
                    for layer_scale in (block.space_ls, block.time_ls, block.mlp_ls):
                        layer_scale.gamma.fill_(0.3)
            return model.eval()

    def test_block_suffix_outputs_and_gradients_match_full_block(self):
        generator = torch.Generator().manual_seed(47)
        for length in (2, 4, 16):
            for suffix in sorted({1, min(length, 6), length}):
                with self.subTest(length=length, suffix=suffix):
                    reference = SpaceTimeBlock(32, 4).eval()
                    with torch.no_grad():
                        for scale in (reference.space_ls, reference.time_ls, reference.mlp_ls):
                            scale.gamma.fill_(0.3)
                    candidate = copy.deepcopy(reference)
                    x = torch.randn(2, length, 4, 32, generator=generator)
                    modulation = torch.randn(2, length, 4, 6 * 32, generator=generator) * 0.1
                    rope_space = spatial_rope(2, 2, 8, x.device)
                    rope_time = temporal_rope(length, 8, x.device)
                    gradient = torch.randn(2, suffix, 4, 32, generator=generator)
                    results = []
                    for block, optimized in ((reference, False), (candidate, True)):
                        inputs = [value.clone().requires_grad_() for value in (x, modulation)]
                        output = block(
                            inputs[0], rope_space, rope_time, inputs[1], last_frames=suffix if optimized else None
                        )[:, -suffix:]
                        output.backward(gradient)
                        results.append(
                            (output, [value.grad for value in inputs], [value.grad for value in block.parameters()])
                        )
                    torch.testing.assert_close(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)
                    pairs = zip(results[1][1] + results[1][2], results[0][1] + results[0][2], strict=True)
                    for actual, expected in pairs:
                        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)

    def test_decoder_suffix_matches_unpruned_tail_across_steps_and_lengths(self):
        candidate = self.model()
        reference = copy.deepcopy(candidate)
        reference.decoder._run_blocks = types.MethodType(full_tail_blocks, reference.decoder)
        generator = torch.Generator().manual_seed(71)
        for length in (2, 4, 6, 16):
            z = torch.randn(2, length // 2, 4, 1, 1, generator=generator)
            noise = torch.randn(2, length, 3, 16, 16, generator=generator)
            preserved = [value.clone() for value in (z, noise)]
            for steps in (1, 2, 3):
                for suffix in sorted({value for value in (1, 2, 5, length) if value <= length}):
                    with self.subTest(length=length, steps=steps, suffix=suffix), torch.no_grad():
                        actual = candidate.decode_tail(z, steps, suffix, noise=noise)
                        expected = reference.decode_tail(z, steps, suffix, noise=noise)
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            for actual, expected in zip((z, noise), preserved, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_only_final_tail_ffn_of_final_sampling_step_is_shortened(self):
        model = self.model()
        observed = {"space": [], "time": [], "mlp": []}
        handles = []
        for index, block in enumerate(model.decoder.tail_blocks):
            for name, module in (("space", block.space_attn), ("time", block.time_attn), ("mlp", block.mlp)):

                def hook(_, args, name=name, index=index):
                    shape = args[0].shape
                    frames = shape[0] // 2 if name == "space" else shape[1]
                    observed[name].append((index, frames))

                handles.append(module.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                model.decode_tail(torch.randn(2, 8, 4, 1, 1), 3, 2, noise=torch.randn(2, 16, 3, 16, 16))
        finally:
            for handle in handles:
                handle.remove()
        full = [(index, 16) for _ in range(3) for index in range(3)]
        self.assertEqual(observed["space"], full)
        self.assertEqual(observed["time"], full)
        self.assertEqual(observed["mlp"], full[:-1] + [(2, 6)])

    def test_full_training_outputs_and_gradients_remain_bit_identical(self):
        candidate = self.model().train()
        reference = copy.deepcopy(candidate)
        reference.decoder._run_blocks = types.MethodType(full_tail_blocks, reference.decoder)
        z, noise = torch.randn(2, 8, 4, 1, 1), torch.randn(2, 16, 3, 16, 16)
        t, d = torch.rand(2, 8), torch.rand(2, 8)
        results = []
        for model in (reference, candidate):
            inputs = [value.clone().requires_grad_() for value in (z, noise)]
            output = model.decoder.flow_forward(*inputs, t, d)
            output.square().mean().backward()
            results.append((output, [value.grad for value in inputs], [value.grad for value in model.parameters()]))
        for actual, expected in zip(
            [results[1][0]] + results[1][1] + results[1][2],
            [results[0][0]] + results[0][1] + results[0][2],
            strict=True,
        ):
            if actual is None:
                self.assertIsNone(expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_and_no_tail_decoder_are_unchanged(self):
        for tail_depth in (0, 1, 3):
            with self.subTest(tail_depth=tail_depth):
                model = self.model(tail_depth)
                reference = copy.deepcopy(model)
                reference.decoder._run_blocks = types.MethodType(full_tail_blocks, reference.decoder)
                state = {name: value.clone() for name, value in model.state_dict().items()}
                z, noise = torch.randn(2, 8, 4, 1, 1), torch.randn(2, 16, 3, 16, 16)
                with torch.no_grad():
                    actual = model.decode_tail(z, 2, 2, noise=noise)
                    expected = reference.decode_tail(z, 2, 2, noise=noise)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(state.keys(), model.state_dict().keys())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                self.model(tail_depth).load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()

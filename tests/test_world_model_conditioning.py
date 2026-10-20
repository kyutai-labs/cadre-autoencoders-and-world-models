"""Compare compact frame conditioning with the original repeated spatial conditioning."""

import copy
import types
import unittest
from unittest.mock import patch

import torch
from einops import rearrange
from torch.utils._pytree import tree_leaves, tree_map

from model_world_model import WORLD_MODELS, DiTGeometry, WorldModel, WorldModelConfig
from util.attention import AdaptiveLayerNorm


def legacy_block_forward(self, x, cond, time_rope, space_rope, kv_cache=None, return_kv=False):
    """The original attention layout, including one conditioning projection per pixel."""
    b, t, h, w, _ = x.shape
    cond = cond.expand(-1, -1, h, w, -1).contiguous()
    x = rearrange(x, "b t h w c -> (b t) (h w) c")
    x = x + self.space_attn(self.space_norm(x, rearrange(cond, "b t h w c -> (b t) (h w) c")), space_rope)
    new_kv = None
    if self.time_attn is not None:
        x = rearrange(x, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
        cond_time = rearrange(cond, "b t h w c -> (b h w) t c")
        y = self.time_attn(self.time_norm(x, cond_time), time_rope, kv_cache, return_kv)
        if return_kv:
            y, new_kv = y
        x = rearrange(x + y, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
    else:
        x = rearrange(x, "(b t) (h w) c -> b t h w c", b=b, t=t, h=h, w=w)
    return x + self.mlp(self.mlp_norm(x, cond)), new_kv


class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def models(self):
        torch.manual_seed(7)
        geometry = DiTGeometry(hidden_dim=32, depth=3, num_heads=4, num_kv_heads=2, time_attention_every=2)
        with patch.dict(WORLD_MODELS, tiny=geometry):
            optimized = WorldModel(
                WorldModelConfig(architecture="tiny", tau=2, latent_channels=4, frames=16, action_dim=2, noise_shift=2)
            )
        # Fresh models zero-initialize AdaLN, which would make a conditioning test trivial.
        with torch.no_grad():
            for block in optimized.blocks:
                for norm in (block.space_norm, block.time_norm, block.mlp_norm):
                    if norm is not None:
                        norm.gamma_beta.weight.normal_(std=0.1)
                        norm.gamma_beta.bias.normal_(std=0.1)
        legacy = copy.deepcopy(optimized)
        for block in legacy.blocks:
            block.forward = types.MethodType(legacy_block_forward, block)
        return optimized, legacy

    def inputs(self, frames=3):
        return (
            torch.randn(2, frames, 2, 3, 4),
            torch.randn(2, frames, 32),
            torch.rand(2, frames),
            torch.rand(2, frames),
        )

    def assert_cache_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for pair, reference in zip(actual, expected, strict=True):
            if pair is None:
                self.assertIsNone(reference)
            else:
                for tensor, target in zip(pair, reference, strict=True):
                    torch.testing.assert_close(tensor, target, rtol=2e-5, atol=2e-6)

    def test_context_and_multiple_cached_steps_match_legacy(self):
        optimized, legacy = self.models()
        optimized.eval()
        legacy.eval()
        z, a, t, d = self.inputs()
        with torch.no_grad():
            actual, actual_kv = optimized.velocity(z, a, t, d, return_kv=True)
            expected, expected_kv = legacy.velocity(z, a, t, d, return_kv=True)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            self.assert_cache_close(actual_kv, expected_kv)
            for _ in range(3):
                noise, actions, _, _ = self.inputs(frames=1)
                renoise = torch.randn_like(noise)
                actual, actual_kv = optimized.next_latent(actual_kv, actions, noise, renoise, 2, 0.15)
                expected, expected_kv = legacy.next_latent(expected_kv, actions, noise, renoise, 2, 0.15)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                self.assert_cache_close(actual_kv, expected_kv)

    def test_training_values_and_gradients_match_with_checkpointing(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                optimized, legacy = self.models()
                inputs = self.inputs()
                outputs, input_grads = [], []
                for model in (optimized, legacy):
                    model.train()
                    model.config.activation_checkpointing = checkpointing
                    tensors = [value.clone().requires_grad_() for value in inputs]
                    output = model.velocity(*tensors)
                    output.square().mean().backward()
                    outputs.append(output)
                    input_grads.append([value.grad for value in tensors])
                torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
                for actual, expected in zip(*input_grads, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
                for (name, actual), (other_name, expected) in zip(
                    optimized.named_parameters(), legacy.named_parameters(), strict=True
                ):
                    self.assertEqual(name, other_name)
                    if actual.grad is None:
                        self.assertIsNone(expected.grad, name)
                    else:
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_projects_each_frame_once_and_preserves_checkpoint_schema(self):
        optimized, legacy = self.models()
        self.assertEqual(list(optimized.state_dict()), list(legacy.state_dict()))
        inputs = self.inputs()
        projected_rows = []
        for model in (optimized, legacy):
            shapes, handles = [], []
            for block in model.blocks:
                for norm in (block.space_norm, block.time_norm, block.mlp_norm):
                    if norm is not None:
                        handles.append(
                            norm.gamma_beta.register_forward_pre_hook(
                                lambda _, args, shapes=shapes: shapes.append(args[0])
                            )
                        )
            try:
                with torch.no_grad():
                    model.velocity(*inputs)
                projected_rows.append(sum(tensor.numel() // 32 for tensor in shapes))
                if model is optimized:
                    self.assertTrue(all(tensor.shape == (2, 3, 1, 1, 32) for tensor in shapes))
            finally:
                for handle in handles:
                    handle.remove()
        self.assertEqual(projected_rows[0], 8 * 2 * 3)  # three space/MLP pairs plus two temporal projections
        self.assertEqual(projected_rows[1], projected_rows[0] * 2 * 3)


def clone_tensor_tree(tree):
    return tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value, tree)


def tensor_leaves(tree):
    return [value for value in tree_leaves(tree) if isinstance(value, torch.Tensor)]


class PreprojectedConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, bf16=False, noise_shift=3):
        with torch.random.fork_rng(devices=[]), patch.dict(WORLD_MODELS, tiny=DiTGeometry(32, 3, 4, 2, 2)):
            torch.manual_seed(31)
            model = WorldModel(
                WorldModelConfig(
                    architecture="tiny", tau=2, latent_channels=4, frames=8, action_dim=2, noise_shift=noise_shift
                )
            ).eval()
            with torch.no_grad():
                # Nonzero learned conditioning is essential: zero AdaLN would
                # hide swapped schedule rows and mixed-up session projections.
                for module in model.modules():
                    if isinstance(module, AdaptiveLayerNorm):
                        module.gamma_beta.weight.normal_(std=0.1)
                        module.gamma_beta.bias.normal_(std=0.1)
                if bf16:
                    for module in model.modules():
                        if isinstance(module, torch.nn.Linear):
                            module.to(torch.bfloat16)
        return model

    def inputs(self, model, bf16=False, seed=19):
        generator = torch.Generator().manual_seed(seed)
        dtype = torch.bfloat16 if bf16 else torch.float32

        def randn(*shape):
            return torch.randn(shape, generator=generator, dtype=dtype)

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache = model.context_cache(randn(2, 3, 2, 3, 4), randn(2, 3, 32))
        return cache, randn(2, 1, 32), randn(2, 1, 2, 3, 4), randn(2, 1, 2, 3, 4)

    def assert_tree_close(self, actual, expected, rtol=2e-5, atol=2e-6):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        elif expected is None:
            self.assertIsNone(actual)
        else:
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_tree_close(a, b, rtol, atol)

    def test_production_flag_matches_original_with_changing_sessions(self):
        for bf16 in (False, True):
            for noise_shift in (1, 3):
                model = self.model(bf16=bf16, noise_shift=noise_shift)
                self.assertFalse(model.preproject_conditioning)
                original_state = {name: value.clone() for name, value in model.state_dict().items()}
                tolerance = {"rtol": 0.016, "atol": 0.008} if bf16 else {}
                for steps in (1, 3, 7):
                    with (
                        self.subTest(bf16=bf16, noise_shift=noise_shift, steps=steps),
                        torch.no_grad(),
                        torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16),
                    ):
                        cache, action, noise, renoise = self.inputs(model, bf16=bf16)
                        expected_cache, actual_cache = clone_tensor_tree(cache), clone_tensor_tree(cache)
                        previous = None
                        for update in range(3):
                            # Two distinct sessions change independently across successive cache updates.
                            delta = action.new_tensor([0.2, -0.3]).view(2, 1, 1) * update
                            current_action, current_noise = action + delta, noise + 0.1 * update
                            inputs = (actual_cache, current_action, current_noise, renoise)
                            preserved, rng = clone_tensor_tree(inputs), torch.random.get_rng_state().clone()
                            model.preproject_conditioning = False
                            expected, expected_cache = model.next_latent(
                                expected_cache, current_action, current_noise, renoise, steps, 0.15
                            )
                            model.preproject_conditioning = True
                            with patch.object(
                                model, "project_schedule_conditioning", wraps=model.project_schedule_conditioning
                            ) as projection:
                                actual, updated = model.next_latent(*inputs, steps, 0.15)
                            self.assertEqual(projection.call_count, 1)
                            self.assert_tree_close((actual, updated), (expected, expected_cache), **tolerance)
                            self.assert_tree_close(inputs, preserved, 0, 0)
                            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                            for old, new in zip(tensor_leaves(actual_cache), tensor_leaves(updated), strict=True):
                                self.assertNotEqual(old.untyped_storage().data_ptr(), new.untyped_storage().data_ptr())
                            if previous is not None:
                                self.assertGreater((actual - previous).abs().max().item(), 1e-4)
                            previous, actual_cache = actual.clone(), updated
                self.assertEqual(set(model.state_dict()), set(original_state))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)

    def test_training_and_grad_enabled_velocity_keep_original_path(self):
        model = self.model()
        cache, action, noise, renoise = self.inputs(model)
        with torch.no_grad():
            projected = model.project_schedule_conditioning(action, noise, 3, 0.15)
        for training in (False, True):
            with self.subTest(training=training):
                model.train(training)
                results = []
                for enabled in (False, True):
                    model.preproject_conditioning = enabled
                    z, a = noise.clone().requires_grad_(), action.clone().requires_grad_()
                    with patch.object(model, "project_schedule_conditioning", side_effect=AssertionError("not used")):
                        output = model.velocity(z, a, torch.full((2, 1), 0.4))
                        gradients = torch.autograd.grad(
                            output.square().sum(), (z, a, *model.parameters()), allow_unused=True
                        )
                    self.assertGreater(gradients[1].abs().max().item(), 0)
                    results.append((output.detach(), gradients))
                self.assert_tree_close(results[0], results[1], 0, 0)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            model.project_schedule_conditioning(action, noise, 3, 0.15)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            model.velocity(noise, action, torch.ones((2, 1)), projected_conditioning=tuple(p[0] for p in projected))
        with patch.object(model, "project_schedule_conditioning", side_effect=AssertionError("not used")):
            model.preproject_conditioning = False
            expected = model.next_latent(cache, action, noise, renoise, 3, 0.15)
            model.preproject_conditioning = True
            self.assert_tree_close(model.next_latent(cache, action, noise, renoise, 3, 0.15), expected, 0, 0)

    def test_projection_storage_is_owned_before_velocity_can_replay_a_graph(self):
        model = self.model()
        cache, action, noise, renoise = self.inputs(model)
        model.preproject_conditioning = True
        with torch.no_grad():
            expected = model.next_latent(cache, action, noise, renoise, 3, 0.15)
            transient = model.project_schedule_conditioning(action, noise, 3, 0.15)
            original_velocity = model.velocity

            def overwrite_transient_outputs(*args, **kwargs):
                # CPU surrogate for separately compiled graph output storage being reused.
                for value in transient:
                    value.fill_(123)
                return original_velocity(*args, **kwargs)

            with (
                patch.object(model, "project_schedule_conditioning", return_value=transient),
                patch.object(model, "velocity", side_effect=overwrite_transient_outputs),
            ):
                actual = model.next_latent(cache, action, noise, renoise, 3, 0.15)
            self.assert_tree_close(actual, expected, 0, 0)

    def test_projection_and_projected_velocity_compile_as_full_graphs_on_cpu(self):
        model = self.model(noise_shift=1)
        cache, action, noise, _ = self.inputs(model)
        project = torch.compile(model.project_schedule_conditioning, backend="eager", fullgraph=True, dynamic=False)
        velocity = torch.compile(model.velocity, backend="eager", fullgraph=True, dynamic=False)
        with torch.no_grad():
            previous = None
            for update in range(2):
                current_action, current_noise = action + 0.2 * update, noise - 0.1 * update
                expected = model.project_schedule_conditioning(current_action, current_noise, 3, 0.15)
                actual = project(current_action, current_noise, 3, 0.15)
                self.assert_tree_close(actual, expected, 0, 0)
                selected = tuple(value[-1] for value in actual)
                time = torch.full((2, 1), 0.85)
                expected_velocity = model.velocity(current_noise, current_action, time, kv_cache=cache, return_kv=True)
                actual_velocity = velocity(
                    current_noise, current_action, time, kv_cache=cache, return_kv=True, projected_conditioning=selected
                )
                self.assert_tree_close(actual_velocity, expected_velocity)
                if previous is not None:
                    self.assertFalse(torch.equal(previous, actual_velocity[0]))
                previous = actual_velocity[0].clone()


if __name__ == "__main__":
    unittest.main()

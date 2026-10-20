import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from demo import Rollout
from demo_batch import batch_key, decode_batches, decoder_batch_key, prepare_batch, step_batch
from model_world_model import WORLD_MODELS, DiTGeometry, StreamingDecoder, WorldModel, WorldModelConfig
from util.attention import AdaptiveLayerNorm


class TinyAutoencoder:
    """Cheap deterministic decoder that still depends on latents and decode noise."""

    config = SimpleNamespace(tau=2, num_frames=6, resolution=(4, 6))

    def unnormalize(self, z):
        return z * 0.7 + 0.1

    def decode(self, z, decode_steps, noise):
        batch, time = z.shape[:2]
        pixels = F.interpolate(z[:, :, :3].flatten(0, 1), self.config.resolution, mode="nearest")
        pixels = pixels.reshape(batch, time, 3, *self.config.resolution).repeat_interleave(self.config.tau, dim=1)
        return torch.tanh(pixels + noise * (0.1 / decode_steps))


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(), patch.dict(WORLD_MODELS, tiny=DiTGeometry(32, 3, 4, 2, 2)):
            torch.manual_seed(123)
            self.wm = WorldModel(
                WorldModelConfig(architecture="tiny", tau=2, latent_channels=4, frames=8, action_dim=2, noise_shift=2)
            ).eval()
            # Production checkpoints have learned conditioning; the default zero
            # AdaLN initialization would otherwise make an action mixing bug invisible.
            with torch.no_grad():
                for module in self.wm.modules():
                    if isinstance(module, AdaptiveLayerNorm):
                        module.gamma_beta.weight.normal_(0, 0.1)
        self.ae = TinyAutoencoder()

    def rollout(self, seed, context_frames=None):
        rollout = Rollout.__new__(Rollout)
        rollout.world_model, rollout.autoencoder, rollout.tau = self.wm, self.ae, 2
        rollout.device = torch.device("cpu")
        rollout.generator = torch.Generator().manual_seed(seed)
        context_generator = torch.Generator().manual_seed(seed + 1000)
        z = torch.randn((1, 3, 2, 3, 4), generator=context_generator)
        a = torch.randn((1, 3, 32), generator=context_generator)
        rollout.kv = self.wm.context_cache(z, a)
        rollout.decoder = StreamingDecoder(self.ae, z, 2, 0.6, rollout.generator, context_frames=context_frames)
        rollout.position, rollout.steps = 5, 0
        rollout.last_action = np.zeros(2, dtype=np.float32)
        return rollout

    def assert_state_equal(self, serial, batched):
        self.assertEqual(serial.position, batched.position)
        self.assertEqual(serial.steps, batched.steps)
        np.testing.assert_array_equal(serial.last_action, batched.last_action)
        self.assertTrue(torch.equal(serial.generator.get_state(), batched.generator.get_state()))
        torch.testing.assert_close(serial.decoder.noise, batched.decoder.noise, rtol=0, atol=0)
        torch.testing.assert_close(serial.decoder.window, batched.decoder.window, rtol=2e-5, atol=2e-5)
        for first, second in zip(serial.kv, batched.kv, strict=True):
            if first is None:
                self.assertIsNone(second)
            else:
                for x, y in zip(first, second, strict=True):
                    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)

    def test_real_model_matches_serial_over_multiple_steps(self):
        for denoise_steps, context_noise in ((1, 0.0), (3, 0.2)):
            with self.subTest(denoise_steps=denoise_steps, context_noise=context_noise):
                serial = [self.rollout(seed) for seed in (17, 43, 91)]
                batched = [self.rollout(seed) for seed in (17, 43, 91)]
                rng = np.random.default_rng(21)
                for _ in range(4):
                    actions = [rng.normal(size=(2, 2)).astype(np.float32) for _ in serial]
                    expected = [r.step(a, denoise_steps, context_noise) for r, a in zip(serial, actions, strict=True)]
                    actual = step_batch(batched, actions, denoise_steps, context_noise)
                    for r1, r2, v1, v2 in zip(serial, batched, expected, actual, strict=True):
                        self.assert_state_equal(r1, r2)
                        self.assertEqual(v2.dtype, torch.uint8)
                        self.assertEqual(v2.shape, (2, 3, 4, 6))
                        torch.testing.assert_close(v1, v2, rtol=0, atol=1)

    def test_session_state_and_returned_video_have_independent_storage(self):
        rollouts = [self.rollout(17), self.rollout(43)]
        actions = [np.zeros((2, 2), np.float32) for _ in rollouts]
        videos = step_batch(rollouts, actions, 1, 0.2)
        preserved = videos[0].clone()

        def tensors(rollout):
            return [rollout.decoder.window, rollout.decoder.noise, *(t for pair in rollout.kv if pair for t in pair)]

        for first, second in zip(tensors(rollouts[0]), tensors(rollouts[1]), strict=True):
            self.assertNotEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        self.assertNotEqual(videos[0].untyped_storage().data_ptr(), videos[1].untyped_storage().data_ptr())
        step_batch(rollouts, actions, 1, 0.2)
        self.assertTrue(torch.equal(videos[0], preserved))
        actions[0][:] = 99
        np.testing.assert_array_equal(rollouts[0].last_action, [0, 0])

    def test_single_session_and_empty_batch(self):
        serial, batched = self.rollout(17), self.rollout(17)
        actions = np.ones((2, 2), np.float32)
        torch.testing.assert_close(serial.step(actions, 2, 0.2), step_batch([batched], [actions], 2, 0.2)[0])
        self.assert_state_equal(serial, batched)
        self.assertEqual(step_batch([], [], 2, 0.2), [])

    def test_short_decoder_context_preserves_full_rng_and_world_model_state(self):
        full = [self.rollout(seed) for seed in (17, 43)]
        short = [self.rollout(seed, context_frames=4) for seed in (17, 43)]
        short_serial = [self.rollout(seed, context_frames=4) for seed in (17, 43)]
        self.assertNotEqual(batch_key(full[0], 2, 0.2), batch_key(short[0], 2, 0.2))
        rng = np.random.default_rng(21)
        for _ in range(3):
            actions = [rng.normal(size=(2, 2)).astype(np.float32) for _ in full]
            step_batch(full, actions, 2, 0.2)
            expected = [r.step(a, 2, 0.2) for r, a in zip(short_serial, actions, strict=True)]
            with patch.object(self.ae, "decode", wraps=self.ae.decode) as decode:
                actual = step_batch(short, actions, 2, 0.2)
            z, _ = decode.call_args.args
            noise = decode.call_args.kwargs["noise"]
            self.assertEqual(z.shape[:2], (2, 2))
            self.assertEqual(noise.shape[:2], (2, 4))
            for original, trimmed, serial, video, target in zip(
                full, short, short_serial, actual, expected, strict=True
            ):
                self.assert_state_equal(original, trimmed)
                self.assert_state_equal(serial, trimmed)
                self.assertEqual(trimmed.decoder.window.shape[1], 3)
                self.assertEqual(trimmed.decoder.noise.shape[1], 6)
                torch.testing.assert_close(video, target, rtol=0, atol=1)

    def test_short_decoder_context_is_used_by_batched_tail_api(self):
        calls = []

        def decode_tail(z, steps, last_frames, noise):
            calls.append((z.shape, noise.shape, last_frames))
            return self.ae.decode(z, steps, noise)[:, -last_frames:]

        rollouts = [self.rollout(seed, context_frames=2) for seed in (17, 43)]
        actions = [np.zeros((2, 2), np.float32) for _ in rollouts]
        with patch.object(self.ae, "decode_tail", decode_tail, create=True):
            step_batch(rollouts, actions, 1, 0.2)
        self.assertEqual(calls[0][0][:2], (2, 1))
        self.assertEqual(calls[0][1][:2], (2, 2))
        self.assertEqual(calls[0][2], 2)

    def test_source_strides_do_not_prevent_batching(self):
        first, second = self.rollout(17), self.rollout(43)
        serial = [self.rollout(17), self.rollout(43)]
        # A fresh prompt's window can be a slice of a longer context; after a
        # step it is a contiguous owned allocation. Both can join the same batch.
        second.decoder.window = torch.cat([second.decoder.window] * 2, dim=1)[:, :3]
        self.assertNotEqual(first.decoder.window.stride(), second.decoder.window.stride())
        second.decoder.noise = second.decoder.noise.transpose(-1, -2).contiguous().transpose(-1, -2)
        second.kv = [
            None if pair is None else tuple(t.transpose(-1, -2).contiguous().transpose(-1, -2) for t in pair)
            for pair in second.kv
        ]
        self.assertEqual(batch_key(first, 2, 0.2), batch_key(second, 2, 0.2))
        actions = [np.ones((2, 2), np.float32) * i for i in range(2)]
        expected = [r.step(a, 2, 0.2) for r, a in zip(serial, actions, strict=True)]
        actual = step_batch([first, second], actions, 2, 0.2)
        for r1, r2, v1, v2 in zip(serial, [first, second], expected, actual, strict=True):
            self.assert_state_equal(r1, r2)
            torch.testing.assert_close(v1, v2, rtol=0, atol=1)

    def test_incompatible_batches_fail_before_random_draws(self):
        first, second = self.rollout(17), self.rollout(43)
        action = np.zeros((2, 2), np.float32)
        rng_state = first.generator.get_state().clone()
        self.assertEqual(batch_key(first, 2, 0.2), batch_key(second, 2, 0.2))
        self.assertNotEqual(batch_key(first, 2, 0.2), batch_key(first, 3, 0.2))
        self.assertNotEqual(batch_key(first, 2, 0.2), batch_key(first, 2, 0.3))
        second.decoder.alpha = 0.7
        with self.assertRaisesRegex(ValueError, "incompatible"):
            step_batch([first, second], [action, action], 2, 0.2)
        with self.assertRaisesRegex(ValueError, "only once"):
            step_batch([first, first], [action, action], 2, 0.2)
        with self.assertRaisesRegex(ValueError, "one action"):
            step_batch([first], [], 2, 0.2)
        with self.assertRaisesRegex(ValueError, "shape"):
            step_batch([first], [action[:1]], 2, 0.2)
        self.assertTrue(torch.equal(first.generator.get_state(), rng_state))
        self.assertEqual(first.steps, 0)

    def test_different_world_models_share_one_decode_and_match_serial_rng_and_state(self):
        with torch.random.fork_rng(), patch.dict(WORLD_MODELS, tiny=DiTGeometry(32, 3, 4, 2, 2)):
            torch.manual_seed(917)
            other = WorldModel(
                WorldModelConfig(architecture="tiny", tau=2, latent_channels=4, frames=8, action_dim=6, noise_shift=1)
            ).eval()
        serial = [self.rollout(seed, context_frames=2) for seed in (17, 43, 91)]
        mixed = [self.rollout(seed, context_frames=2) for seed in (17, 43, 91)]
        with patch.object(self, "wm", other):
            serial[1], mixed[1] = self.rollout(43, context_frames=2), self.rollout(43, context_frames=2)
        rng = np.random.default_rng(157)
        for _ in range(3):
            actions = [
                rng.normal(size=(2, rollout.world_model.config.action_dim)).astype(np.float32) for rollout in mixed
            ]
            expected = [
                rollout.step(action, 7 if i != 1 else 3, 0.15 if i != 1 else 0.2)
                for i, (rollout, action) in enumerate(zip(serial, actions, strict=True))
            ]
            first = prepare_batch([mixed[0], mixed[2]], [actions[0], actions[2]], 7, 0.15)
            second = prepare_batch([mixed[1]], [actions[1]], 3, 0.2)
            self.assertNotEqual(batch_key(mixed[0], 7, 0.15), batch_key(mixed[1], 3, 0.2))
            self.assertEqual(decoder_batch_key(first), decoder_batch_key(second))
            with patch.object(self.ae, "decode", wraps=self.ae.decode) as decode:
                videos = decode_batches([first, second])
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(decode.call_args.args[0].shape[:2], (3, 1))
            self.assertEqual([len(group) for group in videos], [2, 1])
            actual = [videos[0][0], videos[1][0], videos[0][1]]
            for baseline, rollout, output, target in zip(serial, mixed, actual, expected, strict=True):
                self.assert_state_equal(baseline, rollout)
                torch.testing.assert_close(output, target, rtol=0, atol=1)

    def test_pending_state_and_videos_are_private_until_successful_shared_decode(self):
        first, second = self.rollout(17), self.rollout(43)
        actions = [np.zeros((2, 2), np.float32) for _ in range(2)]
        original = (first.decoder.window.clone(), first.decoder.noise.clone(), first.position, first.steps)
        pending = prepare_batch([first], [actions[0]], 3, 0.15)
        preserved = (pending.window.clone(), pending.noise.clone(), pending.caches[0][0][0].clone())
        other = prepare_batch([second], [actions[1]], 3, 0.15)
        self.assertTrue(torch.equal(pending.window, preserved[0]))
        self.assertTrue(torch.equal(pending.noise, preserved[1]))
        self.assertTrue(torch.equal(pending.caches[0][0][0], preserved[2]))
        self.assertTrue(torch.equal(first.decoder.window, original[0]))
        self.assertTrue(torch.equal(first.decoder.noise, original[1]))
        self.assertEqual((first.position, first.steps), original[2:])
        actions[0][:] = 99  # pending action metadata must already be owned.
        with patch.object(self.ae, "decode", side_effect=RuntimeError("decode failed")):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                decode_batches([pending, other])
        self.assertEqual((first.position, first.steps), original[2:])
        self.assertFalse(pending.consumed)
        videos = decode_batches([pending, other])
        np.testing.assert_array_equal(first.last_action, [0, 0])
        for a, b in (
            (first.decoder.window, second.decoder.window),
            (first.decoder.noise, second.decoder.noise),
            (first.kv[0][0], second.kv[0][0]),
            (videos[0][0], videos[1][0]),
        ):
            self.assertNotEqual(a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        retained = videos[0][0].clone()
        step_batch([second], [actions[1]], 3, 0.15)
        self.assertTrue(torch.equal(videos[0][0], retained))
        with self.assertRaisesRegex(ValueError, "only once"):
            decode_batches([pending])

    def test_shared_decoder_rejects_incompatible_or_stale_pending_state(self):
        full, short = self.rollout(17), self.rollout(43, context_frames=2)
        actions = np.zeros((2, 2), np.float32)
        first = prepare_batch([full], [actions], 1, 0.2)
        second = prepare_batch([short], [actions], 1, 0.2)
        rng = full.generator.get_state().clone()
        with self.assertRaisesRegex(ValueError, "incompatible decoder"):
            decode_batches([first, second])
        with self.assertRaisesRegex(ValueError, "only once"):
            decode_batches([first, first])
        self.assertEqual(full.steps, 0)
        self.assertTrue(torch.equal(full.generator.get_state(), rng))
        full.position += 2
        with self.assertRaisesRegex(ValueError, "state changed"):
            decode_batches([first])
        self.assertEqual(decode_batches([]), [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_batch([], [], 1, 0.2)


if __name__ == "__main__":
    unittest.main()

"""Short decoder context is opt-in and never changes the world-model RNG stream."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from demo import ConcurrentDemo, Demo, build, parse_args
from model_world_model import StreamingDecoder
from tests import test_decoder_tail


class DecoderContextTests(unittest.TestCase):
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

    def test_context_validation_and_full_default(self):
        cfg = SimpleNamespace(tau=2, num_frames=16)
        self.assertEqual(StreamingDecoder.validate_context_frames(cfg, None), 16)
        for frames in (2, 4, 6, 8, 16):
            self.assertEqual(StreamingDecoder.validate_context_frames(cfg, frames), frames)
        for frames in (-2, 0, 1, 3, 18, True, 8.0):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "multiple of tau"):
                StreamingDecoder.validate_context_frames(cfg, frames)

    def test_real_decoder_matches_explicit_slicing_and_keeps_full_state(self):
        model, z, _ = test_decoder_tail.DecoderTailTests().fixture()
        ae = SimpleNamespace(
            config=model.config, unnormalize=lambda x: x, decode=model.decode, decode_tail=model.decode_tail
        )
        window = z.permute(0, 1, 3, 4, 2)
        full = StreamingDecoder(ae, window, 2, 0.99, torch.Generator().manual_seed(17))
        short = StreamingDecoder(ae, window, 2, 0.99, torch.Generator().manual_seed(17), context_frames=8)
        with torch.no_grad():
            for _ in range(2):
                next_z = torch.randn_like(window[:, :1])
                full.push(next_z)
                actual = short.push(next_z)
                raw = short.window[:, -4:].permute(0, 1, 4, 2, 3)
                expected = model.decode_tail(raw, 2, 2, noise=short.noise[:, -8:]) * 0.5 + 0.5
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(short.window, full.window, rtol=0, atol=0)
                torch.testing.assert_close(short.noise, full.noise, rtol=0, atol=0)
                self.assertEqual(short.window.shape[1], 8)
                self.assertEqual(short.noise.shape[1], 16)
                self.assertTrue(torch.equal(short.generator.get_state(), full.generator.get_state()))

    def test_full_context_default_and_explicit_setting_match(self):
        model, z, _ = test_decoder_tail.DecoderTailTests().fixture()
        ae = SimpleNamespace(config=model.config, unnormalize=lambda x: x, decode=model.decode)
        window = z.permute(0, 1, 3, 4, 2)
        default = StreamingDecoder(ae, window, 1, 0.99, torch.Generator().manual_seed(17))
        explicit = StreamingDecoder(ae, window, 1, 0.99, torch.Generator().manual_seed(17), context_frames=16)
        with torch.no_grad():
            next_z = torch.randn_like(window[:, :1])
            torch.testing.assert_close(default.push(next_z), explicit.push(next_z), rtol=0, atol=0)
        self.assertTrue(torch.equal(default.generator.get_state(), explicit.generator.get_state()))

    def test_cli_validation_build_and_session_announcements(self):
        self.assertIsNone(parse_args([]).decoder_context_frames)
        with self.assertRaises(SystemExit):
            parse_args(["--decoder_context_frames", "0"])
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)))
        for frames in (3, 18):
            args = parse_args(["--device", "cpu", "--no-compile", "--decoder_context_frames", str(frames)])
            with (
                patch("demo.load_models", return_value=(ae, {"driving": SimpleNamespace()})),
                patch("demo.EnvironmentDataset") as dataset,
                self.assertRaisesRegex(ValueError, "multiple of tau"),
            ):
                build(args, checkpoints={"driving": "unused"})
            dataset.assert_not_called()
        args = parse_args(
            ["--device", "cpu", "--no-compile", "--decoder_context_frames", "8", "--data", "driving=/unused"]
        )
        with (
            patch("demo.load_models", return_value=(ae, {"driving": SimpleNamespace()})),
            patch("demo.EnvironmentDataset", return_value=[]),
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        pool = ConcurrentDemo(demo, 2)
        for slot in pool.slots:
            slot.reset()
            self.assertEqual(slot.decoder_context_frames, 8)
            self.assertEqual(slot.hello()["decoder_context_frames"], 8)
            slot.rollouts["driving"] = SimpleNamespace(index=0, key="clip", seed=17, steps=1, remaining=lambda: 2)
            self.assertEqual(slot.info("driving")["decoder_context_frames"], 8)

    def test_demo_passes_context_to_new_rollout_without_altering_prompt_cache(self):
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16))
        clip = {"key": "clip", "actions": torch.zeros(80, 2)}
        demo = Demo(
            ae,
            "test",
            {"driving": SimpleNamespace()},
            {"driving": [clip]},
            torch.device("cpu"),
            3,
            0.15,
            decoder_context_frames=8,
        )
        self.addCleanup(demo.executor.shutdown, wait=True)
        window = torch.randn(1, 8, 1, 1, 4)
        rollout = SimpleNamespace(
            decoder=SimpleNamespace(window=window), kv=[(window.clone(), window.clone())], last_action=[0.0, 0.0]
        )
        with patch("demo.Rollout", return_value=rollout) as create:
            demo.new_prompt("driving", 0, seed=17)
            self.assertEqual(create.call_args.kwargs["decoder_context_frames"], 8)
            cached = demo.prompt_cache[("driving", 0)][1][0]
            self.assertEqual(cached.shape[1], 8)  # eight latents = full 16 frames
            demo.new_prompt("driving", 0, seed=17)
            self.assertEqual(create.call_args.kwargs["decoder_context_frames"], 8)
            torch.testing.assert_close(create.call_args.args[6][0], window, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

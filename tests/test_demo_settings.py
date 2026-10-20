import asyncio
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from demo import ConcurrentDemo, Demo, build, parse_args, serve


class SettingsTests(unittest.TestCase):
    def make_demo(self, fixed_settings=False):
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16))
        demo = Demo(
            ae, "test", {"driving": SimpleNamespace()}, {}, torch.device("cpu"), 3, 0.15, fixed_settings=fixed_settings
        )
        self.addCleanup(demo.executor.shutdown, wait=True)
        return demo

    def test_cli_defaults_to_unlocked_and_accepts_fixed_settings(self):
        self.assertFalse(parse_args([]).fixed_settings)
        self.assertTrue(parse_args(["--fixed_settings"]).fixed_settings)

    def test_preprojection_is_opt_in_compiled_and_announced(self):
        self.assertFalse(parse_args([]).preproject_conditioning)
        args = parse_args(["--device", "cpu", "--data", "driving=/unused", "--preproject_conditioning"])
        velocity, projection, decoder_forward = (object() for _ in range(3))
        world_model = SimpleNamespace(velocity=velocity, project_schedule_conditioning=projection)
        ae = SimpleNamespace(
            config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)),
            decoder=SimpleNamespace(flow_forward=decoder_forward),
        )
        with (
            patch("demo.load_models", return_value=(ae, {"driving": world_model})),
            patch("demo.EnvironmentDataset", return_value=[]),
            patch("torch.compile", side_effect=lambda method, **kwargs: method) as compile_method,
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        self.assertTrue(world_model.preproject_conditioning)
        self.assertEqual(
            [call.args[0] for call in compile_method.call_args_list], [velocity, projection, decoder_forward]
        )
        for slot in ConcurrentDemo(demo, 2).slots:
            self.assertTrue(slot.hello()["preproject_conditioning"])

    def test_lock_rejects_setting_changes_but_preserves_driving_input(self):
        demo = self.make_demo(fixed_settings=True)
        demo.apply({"type": "settings", "denoise_steps": 1, "context_noise": 0.8})
        # Ignored settings never reach parsing or compilation, even for a client
        # that bypasses the browser's disabled controls.
        demo.apply({"type": "settings", "denoise_steps": "invalid", "context_noise": "invalid"})
        self.assertEqual((demo.denoise_steps, demo.context_noise), (3, 0.15))
        demo.apply({"type": "keys", "held": ["KeyW"], "sequence": 17})
        self.assertEqual(demo.held, {"KeyW"})
        self.assertEqual(demo.input_sequence, 17)

    def test_lock_survives_slot_copy_and_reset_and_is_announced(self):
        demo = self.make_demo(fixed_settings=True)
        pool = ConcurrentDemo(demo, 2)
        for slot in pool.slots:
            slot.reset()
            slot.apply({"type": "settings", "denoise_steps": 2, "context_noise": 0.4})
            hello = slot.hello()
            self.assertTrue(slot.fixed_settings)
            self.assertTrue(hello["settings_locked"])
            self.assertEqual(hello["max_denoise_steps"], 3)
            self.assertEqual((hello["denoise_steps"], hello["context_noise"]), (3, 0.15))

    def test_unlocked_clients_can_adjust_settings_within_server_maximum(self):
        demo = self.make_demo()
        demo.apply({"type": "settings", "denoise_steps": 2, "context_noise": 0.3})
        hello = demo.hello()
        self.assertFalse(hello["settings_locked"])
        self.assertEqual((hello["denoise_steps"], hello["context_noise"]), (2, 0.3))
        self.assertEqual(hello["max_denoise_steps"], 3)
        demo.apply({"type": "settings", "denoise_steps": 50})
        self.assertEqual(demo.denoise_steps, 3)

    def test_build_passes_cli_lock_to_demo(self):
        args = parse_args(["--device", "cpu", "--no-compile", "--data", "driving=/unused", "--fixed_settings"])
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)))
        with (
            patch("demo.load_models", return_value=(ae, {"driving": SimpleNamespace()})),
            patch("demo.EnvironmentDataset", return_value=[]),
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        self.assertTrue(demo.fixed_settings)
        self.assertTrue(demo.hello()["settings_locked"])

    def test_fused_swiglu_is_opt_in_and_conversion_follows_fp8(self):
        self.assertFalse(parse_args([]).fused_swiglu)
        args = parse_args(["--device", "cpu", "--no-compile", "--data", "driving=/unused", "--fp8", "--fused_swiglu"])
        decoder, world_model, calls = object(), SimpleNamespace(), []
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)), decoder=decoder)

        def record(kind, module):
            calls.append((kind, module))
            return {"converted": 0}

        with (
            patch("demo.load_models", return_value=(ae, {"driving": world_model})),
            patch("demo.EnvironmentDataset", return_value=[]),
            patch("util.fp8.convert_fp8_linears", side_effect=lambda model, **kwargs: record("fp8", model)),
            patch("util.fused_swiglu.convert_fused_swiglu", side_effect=lambda model: record("fused", model)),
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        self.assertEqual(calls, [(kind, model) for kind in ("fp8", "fused") for model in (decoder, world_model)])
        for slot in ConcurrentDemo(demo, 2).slots:
            slot.reset()
            self.assertTrue(slot.hello()["fused_swiglu"])

    def test_world_model_only_fusion_leaves_decoder_unchanged(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--fused_swiglu_world_model_only"])
        args = parse_args(
            [
                "--device",
                "cpu",
                "--no-compile",
                "--data",
                "driving=/unused",
                "--fused_swiglu",
                "--fused_swiglu_world_model_only",
            ]
        )
        world_model = SimpleNamespace()
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)), decoder=object())
        with (
            patch("demo.load_models", return_value=(ae, {"driving": world_model})),
            patch("demo.EnvironmentDataset", return_value=[]),
            patch("util.fused_swiglu.convert_fused_swiglu", return_value={"converted": 0}) as convert,
        ):
            demo = build(args, checkpoints={"driving": "unused"})
        self.addCleanup(demo.executor.shutdown, wait=True)
        convert.assert_called_once_with(world_model)
        for slot in ConcurrentDemo(demo, 2).slots:
            self.assertTrue(slot.hello()["fused_swiglu_world_model_only"])

    def test_decoder_only_fp8_requires_fp8_and_selects_only_decoder(self):
        self.assertFalse(parse_args([]).fp8_decoder_only)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--fp8_decoder_only"])
        for decoder_only, fast in ((False, False), (True, False), (True, True)):
            with self.subTest(decoder_only=decoder_only, fast=fast):
                flags = ["--fp8_decoder_only"] if decoder_only else []
                if fast:
                    flags.append("--fp8_fast_accum")
                args = parse_args(["--device", "cpu", "--no-compile", "--data", "driving=/unused", "--fp8", *flags])
                decoder, world_model = object(), SimpleNamespace()
                ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)), decoder=decoder)
                with (
                    patch("demo.load_models", return_value=(ae, {"driving": world_model})),
                    patch("demo.EnvironmentDataset", return_value=[]),
                    patch("util.fp8.convert_fp8_linears", return_value={"converted": 0}) as convert,
                ):
                    demo = build(args, checkpoints={"driving": "unused"})
                self.addCleanup(demo.executor.shutdown, wait=True)
                selected = [call.args[0] for call in convert.call_args_list]
                self.assertEqual(selected, [decoder] if decoder_only else [decoder, world_model])
                self.assertTrue(all(call.kwargs == {"use_fast_accum": fast} for call in convert.call_args_list))
                decoder_precision = "fp8-fast" if fast else "fp8"
                expected = f"bf16-wm/{decoder_precision}-decoder" if decoder_only else decoder_precision
                pool = ConcurrentDemo(demo, 2)
                for slot in pool.slots:
                    slot.reset()
                    slot.rollouts["driving"] = SimpleNamespace(
                        index=0, key="clip", seed=17, steps=1, remaining=lambda: 2
                    )
                    for info in (slot.hello(), slot.info("driving")):
                        self.assertEqual(info["precision"], expected)
                        self.assertEqual(info["fp8_decoder_only"], decoder_only)
                        self.assertEqual(info["decoder_precision"], decoder_precision)
                        self.assertEqual(info["world_model_precision"], "bf16" if decoder_only else decoder_precision)
                with patch("uvicorn.run") as run:
                    serve(demo, "127.0.0.1", 0, pool=pool)
                app = run.call_args.args[0]
                health = next(route.endpoint for route in app.routes if route.path == "/health")
                info = json.loads(asyncio.run(health(None)).body)
                self.assertEqual(info["precision"], expected)
                self.assertEqual(info["fp8_decoder_only"], decoder_only)
                self.assertEqual(info["world_model_precision"], "bf16" if decoder_only else decoder_precision)


if __name__ == "__main__":
    unittest.main()

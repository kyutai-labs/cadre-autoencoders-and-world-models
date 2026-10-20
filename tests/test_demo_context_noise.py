import asyncio
import contextlib
import io
import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from dataset import ENVIRONMENTS
from demo import CONTEXT_NOISE_BY_ENV, ConcurrentDemo, Demo, build, parse_args, produce_batch, serve, warm_up


class ContextNoiseTests(unittest.TestCase):
    def models(self):
        return {
            env: SimpleNamespace(
                config=SimpleNamespace(
                    action_dim=ENVIRONMENTS[env].action_dim, action_std=[1.0] * ENVIRONMENTS[env].action_dim
                )
            )
            for env in CONTEXT_NOISE_BY_ENV
        }

    def autoencoder(self):
        return SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16, resolution=(2, 2)))

    def make_demo(self, override=None, locked=False):
        demo = Demo(
            self.autoencoder(),
            "test",
            self.models(),
            {env: [] for env in CONTEXT_NOISE_BY_ENV},
            torch.device("cpu"),
            7,
            override,
            fixed_settings=locked,
        )
        self.addCleanup(demo.executor.shutdown, wait=True)
        return demo

    def rollout(self, demo, env):
        decoder = SimpleNamespace(
            autoencoder=demo.autoencoder,
            decode_steps=1,
            alpha=0.99,
            context_frames=16,
            window=torch.zeros(1, 8, 1, 1, 3),
            noise=torch.zeros(1, 16, 3, 2, 2),
        )
        return SimpleNamespace(
            index=0,
            key="clip",
            seed=17,
            steps=0,
            remaining=lambda: 48,
            tau=2,
            world_model=demo.world_models[env],
            autoencoder=demo.autoencoder,
            decoder=decoder,
            kv=[None],
            device=demo.device,
            step=Mock(return_value=torch.zeros(2, 3, 2, 2, dtype=torch.uint8)),
            recorded_actions=lambda: np.zeros((2, ENVIRONMENTS[env].action_dim), dtype=np.float32),
        )

    def test_cli_default_map_and_explicit_global_override(self):
        self.assertIsNone(parse_args([]).context_noise)
        for value in (0.0, 0.15, 0.35, 1.0):
            self.assertEqual(parse_args(["--context_noise", str(value)]).context_noise, value)
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--context_noise", value])

    def test_build_uses_environment_defaults_unless_global_override_is_explicit(self):
        for override in (None, 0.0, 0.15):
            with self.subTest(override=override):
                flags = [] if override is None else ["--context_noise", str(override)]
                args = parse_args(
                    [
                        "--device",
                        "cpu",
                        "--no-compile",
                        "--envs",
                        *CONTEXT_NOISE_BY_ENV,
                        "--data",
                        *(f"{env}=/unused" for env in CONTEXT_NOISE_BY_ENV),
                        *flags,
                    ]
                )
                with (
                    patch("demo.load_models", return_value=(self.autoencoder(), self.models())),
                    patch("demo.EnvironmentDataset", return_value=[]),
                ):
                    demo = build(args, checkpoints=dict.fromkeys(CONTEXT_NOISE_BY_ENV, "unused"))
                self.addCleanup(demo.executor.shutdown, wait=True)
                expected = (
                    dict(CONTEXT_NOISE_BY_ENV) if override is None else dict.fromkeys(CONTEXT_NOISE_BY_ENV, override)
                )
                self.assertEqual(demo.context_noise_defaults, expected)
                self.assertEqual(demo.context_noises, expected)

    def test_environment_switches_keep_private_settings_and_fixed_defaults(self):
        for locked in (False, True):
            with self.subTest(locked=locked):
                demo = self.make_demo(locked=locked)
                for env, expected in CONTEXT_NOISE_BY_ENV.items():
                    demo.apply({"type": "env", "env": env})
                    self.assertEqual(demo.context_noise, expected)
                    demo.apply({"type": "settings", "context_noise": 0.45})
                    self.assertEqual(demo.context_noise, expected if locked else 0.45)
                for env, expected in CONTEXT_NOISE_BY_ENV.items():
                    demo.apply({"type": "env", "env": env})
                    self.assertEqual(demo.context_noise, expected if locked else 0.45)
                demo.reset()
                self.assertEqual(demo.context_noises, CONTEXT_NOISE_BY_ENV)
                self.assertEqual(demo.context_noise, 0.35)

    def test_slot_copy_and_reset_do_not_leak_noise_changes(self):
        demo = self.make_demo()
        first, second = ConcurrentDemo(demo, 2).slots
        self.assertIsNot(first.context_noises, second.context_noises)
        first.apply({"type": "settings", "context_noise": 0.1})
        first.apply({"type": "env", "env": "robot_manipulation"})
        first.apply({"type": "settings", "context_noise": 0.4})
        self.assertEqual(second.context_noises, CONTEXT_NOISE_BY_ENV)
        self.assertEqual(first.context_noises["driving"], 0.1)
        self.assertEqual(first.context_noise, 0.4)
        first.reset()  # The session admission path runs this before hello.
        self.assertEqual(first.hello()["context_noise_by_env"], CONTEXT_NOISE_BY_ENV)
        self.assertIsNot(first.context_noises, second.context_noises)

    def test_produce_uses_selected_environment_noise_after_pending_switch(self):
        demo = self.make_demo(locked=True)
        for env, expected in CONTEXT_NOISE_BY_ENV.items():
            rollout = demo.rollouts[env] = self.rollout(demo, env)
            demo.modes[env] = "replay"
            demo.commands.put({"type": "env", "env": env})
            demo.commands.put({"type": "settings", "context_noise": 0.9})
            with patch("demo.to_jpeg", return_value=[b"frame", b"frame"]):
                chunk = demo.produce()
            self.assertEqual(rollout.step.call_args.args[1:], (7, expected))
            self.assertEqual(chunk.info["context_noise"], expected)
            self.assertEqual(chunk.info["context_noise_by_env"], CONTEXT_NOISE_BY_ENV)

    def test_single_and_mixed_warmup_use_correct_values_and_restore_defaults(self):
        demo = self.make_demo(locked=True)
        singles, batches = [], []

        def produce():
            singles.append((demo.env, demo.context_noise))
            return SimpleNamespace(info={"compute_ms": 1.0})

        def batch(slots):
            batches.append(tuple((slot.env, slot.context_noise) for slot in slots))
            return [SimpleNamespace(info={}) for _ in slots]

        pool = ConcurrentDemo(demo, 2, batch_size=2)
        with (
            patch.object(Demo, "new_prompt"),
            patch.object(demo, "produce", side_effect=produce),
            patch("demo.produce_batch", side_effect=batch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warm_up(demo, steps=1)
            pool.warm_up_batches(steps=1)
        self.assertEqual(singles, list(CONTEXT_NOISE_BY_ENV.items()))
        patterns = [(env, env) for env in CONTEXT_NOISE_BY_ENV]
        patterns.extend(itertools.combinations(CONTEXT_NOISE_BY_ENV, 2))
        self.assertEqual(batches, [tuple((env, CONTEXT_NOISE_BY_ENV[env]) for env in pattern) for pattern in patterns])
        for slot in pool.slots:
            self.assertEqual(slot.context_noises, CONTEXT_NOISE_BY_ENV)
            self.assertEqual(slot.env, "driving")

    def test_batch_partition_uses_noise_and_shared_decoder_keeps_session_order(self):
        demo = self.make_demo()
        slots = ConcurrentDemo(demo, 3, batch_size=3).slots
        slots[1].env = "robot_manipulation"
        slots[2].context_noise = 0.45  # Same WM as slot0, but incompatible sampling setting.
        for slot in slots:
            slot.rollouts[slot.env] = self.rollout(slot, slot.env)
            slot.modes[slot.env] = "replay"

        def prepare(rollouts, actions, steps, noise):
            return SimpleNamespace(rollouts=rollouts)

        with (
            patch("demo.prepare_batch", side_effect=prepare) as prepare_call,
            patch("demo.decoder_batch_key", return_value="shared"),
            patch(
                "demo.decode_batches",
                side_effect=lambda groups: [
                    [torch.zeros(2, 3, 2, 2, dtype=torch.uint8) for _ in group.rollouts] for group in groups
                ],
            ) as decode_call,
            patch("demo.to_jpeg", return_value=[b"frame"] * 6),
        ):
            chunks = produce_batch(slots)
        self.assertTrue(all(not isinstance(chunk, Exception) for chunk in chunks))
        self.assertEqual([call.args[3] for call in prepare_call.call_args_list], [0.35, 0.20, 0.45])
        self.assertEqual([len(call.args[0]) for call in prepare_call.call_args_list], [1, 1, 1])
        self.assertEqual(decode_call.call_count, 1)
        self.assertEqual([chunk.info["context_noise"] for chunk in chunks], [0.35, 0.20, 0.45])

    def test_hello_frame_and_health_expose_current_and_configured_values(self):
        for override in (None, 0.15):
            with self.subTest(override=override):
                demo = self.make_demo(override=override)
                expected = (
                    dict(CONTEXT_NOISE_BY_ENV) if override is None else dict.fromkeys(CONTEXT_NOISE_BY_ENV, override)
                )
                demo.apply({"type": "settings", "context_noise": 0.1})
                hello = demo.hello()
                self.assertEqual(hello["context_noise"], 0.1)
                self.assertEqual({env["name"]: env["context_noise"] for env in hello["envs"]}, demo.context_noises)
                hello["context_noise_by_env"]["driving"] = 0.9
                self.assertEqual(demo.context_noise, 0.1)
                demo.rollouts["robot_manipulation"] = self.rollout(demo, "robot_manipulation")
                self.assertEqual(demo.info("robot_manipulation")["context_noise"], expected["robot_manipulation"])
                with patch("uvicorn.run") as run:
                    serve(demo, "127.0.0.1", 0)
                app = run.call_args.args[0]
                endpoint = next(route.endpoint for route in app.routes if route.path == "/health")
                health = json.loads(asyncio.run(endpoint(None)).body)
                self.assertEqual(health["context_noise_by_env"], expected)
                self.assertEqual(health["context_noise_override"], override)


if __name__ == "__main__":
    unittest.main()

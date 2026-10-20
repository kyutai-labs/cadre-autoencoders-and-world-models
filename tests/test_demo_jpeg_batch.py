import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from demo import Demo, produce_batch


class JPEGBatchTests(unittest.TestCase):
    def slot(self, values, group="shared", waiting=False, started=0.1, decoder_group="shared"):
        video = torch.stack([torch.full((3, 2, 2), value, dtype=torch.uint8) for value in values])
        rollout = SimpleNamespace(video=video, group=group, decoder_group=decoder_group, step=lambda *args: video)
        prepared = ("driving", np.zeros((len(values), 2), dtype=np.float32), 0.0, started, len(values))
        slot = SimpleNamespace(
            denoise_steps=7,
            context_noise=0.15,
            rollouts={"driving": rollout},
            line=[object()] * (2 if waiting else 1),
            preview_encoded_at=None,
            prepare_step=lambda: prepared,
            info=lambda env, compute_ms: {"env": env, "compute_ms": compute_ms},
        )
        slot.finish_step = MethodType(Demo.finish_step, slot)
        return slot

    def encode(self, frames, quality):
        values = [int(frame[0, 0, 0]) for frame in frames]
        self.encoded.append(values)
        return [torch.tensor([value], dtype=torch.uint8) for value in values]

    def setUp(self):
        self.encoded = []

    def test_group_encodes_once_and_keeps_each_users_frames_and_preview(self):
        slots = [self.slot([11, 12]), self.slot([21, 22, 23], waiting=True, started=0.2)]
        with (
            patch("demo.batch_key", side_effect=lambda rollout, *args: rollout.group),
            patch("demo.step_batch", side_effect=lambda rollouts, *args: [r.video for r in rollouts]) as inference,
            patch("demo.encode_jpeg", side_effect=self.encode),
            patch("demo.time.perf_counter", side_effect=[1.0, 2.0, 2.1, 3.0]),
        ):
            chunks = produce_batch(slots)
        self.assertEqual(self.encoded, [[11, 12, 21, 22, 23]])
        self.assertEqual(chunks[0].frames, [bytes([11]), bytes([12])])
        self.assertEqual(chunks[1].frames, [bytes([21]), bytes([22]), bytes([23])])
        self.assertIsNone(chunks[0].preview)
        self.assertIs(chunks[1].preview, chunks[1].frames[0])
        self.assertEqual(inference.call_count, 1)
        self.assertEqual([chunk.info["batch_size"] for chunk in chunks], [2, 2])
        # Both timings end when all frames/previews are ready, rather than when
        # an earlier member of the group finishes its metadata.
        self.assertEqual([chunk.info["compute_ms"] for chunk in chunks], [2900.0, 2800.0])

    def test_single_user_uses_existing_encoder_path(self):
        with (
            patch("demo.batch_key", return_value="one"),
            patch("demo.step_batch") as inference,
            patch("demo.encode_jpeg", side_effect=self.encode),
        ):
            chunks = produce_batch([self.slot([7, 8])])
        inference.assert_not_called()
        self.assertEqual(self.encoded, [[7, 8]])
        self.assertEqual(chunks[0].frames, [bytes([7]), bytes([8])])
        self.assertEqual(chunks[0].info["batch_size"], 1)

    def test_waiting_previews_are_limited_to_four_per_second_per_slot(self):
        first, second = self.slot([11, 12], waiting=True), self.slot([21, 22], waiting=True)
        first.preview = b"last displayed preview"
        previews = []
        with patch("demo.encode_jpeg") as encode:
            for now in (0.0, 0.1, 0.249, 0.25, 0.3, 0.5):
                with patch("demo.time.perf_counter", return_value=now):
                    chunk = first.finish_step(
                        first.prepare_step(), first.rollouts["driving"].video, frames=[b"active one", b"active two"]
                    )
                previews.append(chunk.preview)
                self.assertEqual(chunk.frames, [b"active one", b"active two"])
            # Another slot gets its first preview immediately, even while this
            # slot is inside its rate limit. Skipped chunks retain the old preview.
            with patch("demo.time.perf_counter", return_value=0.51):
                other = second.finish_step(second.prepare_step(), second.rollouts["driving"].video, frames=[b"active"])
        self.assertEqual(previews, [b"active one", None, None, b"active one", None, b"active one"])
        self.assertEqual(other.preview, b"active")
        encode.assert_not_called()
        self.assertEqual(first.preview, b"last displayed preview")
        self.assertEqual(first.preview_encoded_at, 0.5)
        self.assertEqual(second.preview_encoded_at, 0.51)

    def test_interleaved_compatibility_groups_preserve_session_order(self):
        slots = [self.slot([11, 12]), self.slot([21, 22], group="other"), self.slot([31, 32])]
        with (
            patch("demo.batch_key", side_effect=lambda rollout, *args: rollout.group),
            patch(
                "demo.prepare_batch", side_effect=lambda rollouts, *args: SimpleNamespace(rollouts=rollouts)
            ) as prepare,
            patch("demo.decoder_batch_key", return_value="shared"),
            patch(
                "demo.decode_batches", side_effect=lambda batches: [[r.video for r in b.rollouts] for b in batches]
            ) as decode,
            patch("demo.step_batch") as legacy,
            patch("demo.encode_jpeg", side_effect=self.encode),
        ):
            chunks = produce_batch(slots)
        legacy.assert_not_called()
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual([len(batch.rollouts) for batch in decode.call_args.args[0]], [2, 1])
        self.assertEqual(self.encoded, [[11, 12, 31, 32, 21, 22]])
        self.assertEqual([chunk.frames for chunk in chunks], [[bytes([x]), bytes([x + 1])] for x in (11, 21, 31)])
        self.assertEqual([chunk.info["batch_size"] for chunk in chunks], [2, 1, 2])
        self.assertEqual([chunk.info["decoder_batch_size"] for chunk in chunks], [3, 3, 3])

    def test_failed_world_model_group_does_not_discard_other_prepared_sessions(self):
        slots = [self.slot([11, 12]), self.slot([21, 22], group="broken"), self.slot([31, 32])]

        def prepare(rollouts, *args):
            if rollouts[0].group == "broken":
                raise RuntimeError("world model failed")
            return SimpleNamespace(rollouts=rollouts)

        with (
            patch("demo.batch_key", side_effect=lambda rollout, *args: rollout.group),
            patch("demo.prepare_batch", side_effect=prepare),
            patch("demo.decoder_batch_key", return_value="shared"),
            patch("demo.decode_batches", side_effect=lambda batches: [[r.video for r in b.rollouts] for b in batches]),
            patch("demo.encode_jpeg", side_effect=self.encode),
        ):
            chunks = produce_batch(slots)
        self.assertIsInstance(chunks[1], RuntimeError)
        self.assertEqual(self.encoded, [[11, 12, 31, 32]])
        self.assertEqual([chunks[i].info["decoder_batch_size"] for i in (0, 2)], [2, 2])
        self.assertEqual(chunks[2].frames, [bytes([31]), bytes([32])])

    def test_decoder_failure_is_isolated_to_compatible_decoder_group(self):
        slots = [
            self.slot([11, 12], group="a", decoder_group="broken"),
            self.slot([21, 22], group="b", decoder_group="broken"),
            self.slot([31, 32], group="c", decoder_group="valid"),
        ]

        def decode(batches):
            if batches[0].rollouts[0].decoder_group == "broken":
                raise RuntimeError("shared decoder failed")
            return [[r.video for r in batch.rollouts] for batch in batches]

        with (
            patch("demo.batch_key", side_effect=lambda rollout, *args: rollout.group),
            patch("demo.prepare_batch", side_effect=lambda rollouts, *args: SimpleNamespace(rollouts=rollouts)),
            patch("demo.decoder_batch_key", side_effect=lambda batch: batch.rollouts[0].decoder_group),
            patch("demo.decode_batches", side_effect=decode),
            patch("demo.encode_jpeg", side_effect=self.encode),
        ):
            chunks = produce_batch(slots)
        self.assertTrue(all(isinstance(chunks[i], RuntimeError) for i in (0, 1)))
        self.assertEqual(self.encoded, [[31, 32]])
        self.assertEqual(chunks[2].frames, [bytes([31]), bytes([32])])
        self.assertEqual(chunks[2].info["decoder_batch_size"], 1)


if __name__ == "__main__":
    unittest.main()

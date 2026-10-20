import asyncio
import json
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from demo import Chunk, Demo


class TimedSocket:
    def __init__(self, frames):
        self.limit = frames
        self.frames = []
        self.closed = asyncio.Event()

    async def send_bytes(self, message):
        length = struct.unpack("<I", message[:4])[0]
        header = json.loads(message[4 : 4 + length])
        self.frames.append((time.perf_counter(), header))
        if len(self.frames) >= self.limit:
            self.closed.set()


class PacingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16))
        self.demo = Demo(ae, "test", {"driving": SimpleNamespace()}, {}, torch.device("cpu"), 7, 0.15)

        async def heartbeat():
            while True:
                await asyncio.sleep(0.003)

        self.ticker = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.ticker.cancel()
        await asyncio.gather(self.ticker, return_exceptions=True)
        self.demo.executor.shutdown(wait=True)

    async def test_variable_inference_keeps_frame_pace_with_fresh_actions(self):
        socket = TimedSocket(12)
        self.demo.line = [socket]
        durations = iter([0.040, 0.055, 0.045, 0.050, 0.045, 0.050, 0.040])

        def produce():
            sampled = time.perf_counter()
            time.sleep(next(durations))
            info = {
                "sampled_at": sampled,
                "worker_started": sampled,
                "compute_ms": 1000 * (time.perf_counter() - sampled),
                "prompt_ms": 0.0,
            }
            return Chunk([b"jpeg", b"jpeg"], info, sampled)

        receiver = asyncio.create_task(socket.closed.wait())
        try:
            with patch.object(self.demo, "produce", produce):
                await asyncio.wait_for(self.demo.stream(socket, receiver), 2)
            first_frames = [(shown, header) for shown, header in socket.frames if header["i"] == 0]
            self.assertEqual(len(first_frames), 6)
            intervals = [
                later[0] - earlier[0] for earlier, later in zip(first_frames[:-1], first_frames[1:], strict=True)
            ]
            self.assertAlmostEqual(sum(intervals) / len(intervals), 0.1, delta=0.02)
            input_to_display = [shown - header["sampled_at"] for shown, header in first_frames[2:]]
            # Sampling immediately after the preceding chunk is produced makes
            # later actions ~150 ms old. Just-in-time production stays near the
            # recent 55 ms service budget while still sending 20 frames/s.
            self.assertLess(sum(input_to_display) / len(input_to_display), 0.085)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def test_disconnect_awaits_inflight_work_and_stops_playback(self):
        socket = TimedSocket(100)
        self.demo.line = [socket]
        entered, finished = threading.Event(), threading.Event()
        calls = 0

        def produce():
            nonlocal calls
            calls += 1
            sampled = time.perf_counter()
            if calls == 2:
                entered.set()
                time.sleep(0.080)
                finished.set()
            return Chunk(
                [b"jpeg", b"jpeg"],
                {"worker_started": sampled, "compute_ms": 100.0, "prompt_ms": 0.0},
                sampled,
            )

        receiver = asyncio.create_task(socket.closed.wait())
        try:
            with patch.object(self.demo, "produce", produce):
                stream = asyncio.create_task(self.demo.stream(socket, receiver))
                async with asyncio.timeout(1):
                    while not entered.is_set():
                        await asyncio.sleep(0.003)
                socket.closed.set()
                await asyncio.wait_for(stream, 1)
            self.assertTrue(finished.is_set())
            self.assertEqual(calls, 2)
            count = len(socket.frames)
            await asyncio.sleep(0.060)
            self.assertEqual(len(socket.frames), count)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()

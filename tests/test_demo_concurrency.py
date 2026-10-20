import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from starlette.websockets import WebSocketDisconnect

from demo import Chunk, ConcurrentDemo, Demo, parse_args


class Socket:
    client = None

    def __init__(self):
        self.messages = []
        self.closed = asyncio.Event()

    async def send_text(self, message):
        self.messages.append(json.loads(message))

    async def send_bytes(self, message):
        self.messages.append(message)
        if sum(isinstance(m, bytes) for m in self.messages) >= 4:
            self.closed.set()

    async def receive_text(self):
        await self.closed.wait()
        raise WebSocketDisconnect()

    async def receive(self):
        await self.closed.wait()
        return {"type": "websocket.disconnect"}

    async def close(self, **kwargs):
        self.closed.set()


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def make_demo(self):
        model = SimpleNamespace()
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16))
        return Demo(ae, "test", {"driving": model}, {}, torch.device("cpu"), 7, 0.15)

    async def test_global_counts_follow_admission_environment_and_queue_changes(self):
        demo = self.make_demo()
        demo.envs = ["driving", "indoor_navigation"]
        pool = ConcurrentDemo(demo, 2)
        first, second = pool.slots
        try:
            self.assertEqual(
                demo.concurrency(),
                {
                    "active": 0,
                    "max_active": 2,
                    "waiting": 0,
                    "by_env": {"driving": 0, "indoor_navigation": 0},
                },
            )
            first.line = [object(), object(), object()]
            second.line = [object()]
            second.env = "indoor_navigation"
            expected = {
                "active": 2,
                "max_active": 2,
                "waiting": 2,
                "by_env": {"driving": 1, "indoor_navigation": 1},
            }
            self.assertEqual(first.concurrency(), expected)
            self.assertEqual(second.turn()["concurrency"], expected)
            second.line.clear()
            self.assertEqual(first.concurrency()["active"], 1)
            self.assertEqual(first.concurrency()["by_env"]["indoor_navigation"], 0)
            self.assertEqual(first.concurrency()["waiting"], 2)
        finally:
            demo.executor.shutdown(wait=True)

    async def test_hello_and_queue_report_global_counts(self):
        demo = self.make_demo()
        pool = ConcurrentDemo(demo, 2)
        first, second = pool.slots
        waiting = Socket()
        first.line = [object(), waiting]
        second.line = [object()]
        task = asyncio.create_task(first.wait_turn(waiting))
        try:
            await asyncio.sleep(0.02)
            snapshot = {"active": 2, "max_active": 2, "waiting": 1, "by_env": {"driving": 2}}
            self.assertEqual(first.hello()["concurrency"], snapshot)
            self.assertEqual(waiting.messages[0]["concurrency"], snapshot)
        finally:
            waiting.closed.set()
            await asyncio.wait_for(task, 1)
            demo.executor.shutdown(wait=True)

    async def test_two_users_stream_with_independent_state(self):
        demo = self.make_demo()
        pool = ConcurrentDemo(demo, 2)
        first, second = pool.slots
        self.assertIs(first.executor, second.executor)
        self.assertIs(first.world_models, second.world_models)
        self.assertIs(first.prompt_cache, second.prompt_cache)
        for attr in ("commands", "held", "controls", "rollouts", "rng", "line"):
            self.assertIsNot(getattr(first, attr), getattr(second, attr))
        first.held.add("KeyW")
        self.assertFalse(second.held)
        second.apply({"type": "settings", "denoise_steps": 50})
        self.assertEqual(second.denoise_steps, 7)

        def produce(slot):
            time.sleep(0.005)
            return Chunk([b"jpeg", b"jpeg"], {"compute_ms": 5}, time.perf_counter())

        sockets = [Socket(), Socket()]

        async def heartbeat():
            # Match uvicorn's periodic wakeup; this container does not wake the
            # selector reliably for thread completion notifications alone.
            while True:
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(heartbeat())
        try:
            with patch.object(Demo, "produce", produce):
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*(pool.session(s) for s in sockets), return_exceptions=True), 3
                )
            self.assertTrue(all(isinstance(result, WebSocketDisconnect) for result in outcomes))
            for socket in sockets:
                self.assertEqual(socket.messages[0]["type"], "hello")
                self.assertGreaterEqual(sum(isinstance(m, bytes) for m in socket.messages), 4)
            self.assertFalse(first.line)
            self.assertFalse(second.line)
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            demo.executor.shutdown(wait=True)

    async def test_disconnected_waiter_is_removed(self):
        demo = self.make_demo()
        active, waiting = Socket(), Socket()
        demo.line.append(active)
        waiting.closed.set()
        try:
            await asyncio.wait_for(demo.session(waiting), 1)
            self.assertEqual(demo.line, [active])
        finally:
            demo.executor.shutdown(wait=True)

    async def test_three_users_share_batches_and_finish_before_slot_reuse(self):
        demo = self.make_demo()
        pool = ConcurrentDemo(demo, 3, batch_size=3, batch_wait_ms=2)
        batches = []

        def produce(slots):
            batches.append(tuple(id(slot) for slot in slots))
            time.sleep(0.005)
            return [Chunk([b"jpeg", b"jpeg"], {"batch_size": len(slots)}, time.perf_counter()) for _ in slots]

        demo.scheduler.produce = produce

        async def heartbeat():
            while True:
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(heartbeat())
        try:
            sockets = [Socket() for _ in pool.slots]
            results = await asyncio.wait_for(
                asyncio.gather(*(pool.session(socket) for socket in sockets), return_exceptions=True), 3
            )
            self.assertTrue(all(isinstance(result, WebSocketDisconnect) for result in results), results)
            self.assertTrue(any(len(batch) == 3 for batch in batches), batches)
            self.assertTrue(all(len(batch) == len(set(batch)) for batch in batches))
            self.assertTrue(all(not slot.line for slot in pool.slots))
            self.assertTrue(demo.scheduler.pending.empty())
            self.assertFalse(demo.scheduler.current)
        finally:
            await demo.scheduler.close()
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            demo.executor.shutdown(wait=True)

    async def test_waiting_user_takes_another_slot_when_it_becomes_free(self):
        class PersistentSocket(Socket):
            def __init__(self):
                super().__init__()
                self.hello = asyncio.Event()
                self.waiting = asyncio.Event()
                self.receivers = 0
                self.max_receivers = 0

            async def send_text(self, message):
                await super().send_text(message)
                kind = json.loads(message)["type"]
                if kind == "hello":
                    self.hello.set()
                elif kind == "line":
                    self.waiting.set()

            async def send_bytes(self, message):
                self.messages.append(message)

            async def receive(self):
                self.receivers += 1
                self.max_receivers = max(self.max_receivers, self.receivers)
                try:
                    return await super().receive()
                finally:
                    self.receivers -= 1

            async def receive_text(self):
                self.receivers += 1
                self.max_receivers = max(self.max_receivers, self.receivers)
                try:
                    return await super().receive_text()
                finally:
                    self.receivers -= 1

        demo = self.make_demo()
        pool = ConcurrentDemo(demo, 2)
        first, departing, waiting = [PersistentSocket() for _ in range(3)]

        def produce(slot):
            time.sleep(0.005)
            return Chunk([b"jpeg", b"jpeg"], {}, time.perf_counter())

        async def heartbeat():
            while True:
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(heartbeat())
        tasks = []
        try:
            with patch.object(Demo, "produce", produce):
                tasks = [asyncio.create_task(pool.session(socket)) for socket in (first, departing)]
                await asyncio.wait_for(asyncio.gather(first.hello.wait(), departing.hello.wait()), 1)
                tasks.append(asyncio.create_task(pool.session(waiting)))
                await asyncio.wait_for(waiting.waiting.wait(), 1)
                self.assertEqual(pool.slots[0].line, [first, waiting])
                departing.closed.set()
                await asyncio.wait_for(waiting.hello.wait(), 1)
                self.assertFalse(tasks[0].done(), "the first user must retain its turn")
                self.assertEqual(pool.slots[0].line, [first])
                self.assertEqual(pool.slots[1].line, [waiting])
                self.assertEqual(waiting.max_receivers, 1, "migration must never overlap websocket receivers")
        finally:
            for socket in (first, departing, waiting):
                socket.closed.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            demo.executor.shutdown(wait=True)

    def test_cached_prompt_preserves_seed_and_isolates_tensors(self):
        demo = self.make_demo()
        model = demo.world_models["driving"]
        model.config = SimpleNamespace(tau=2)
        model.encode_actions = lambda actions, start, count: actions[:, :count]
        model.context_cache = lambda z, actions: [(z.clone(), z.clone() + 1)]
        clip = {"key": "clip", "video": torch.zeros(80, 3, 2, 2), "actions": torch.zeros(80, 2)}
        demo.datasets = {"driving": [clip]}

        class Decoder:
            def __init__(self, autoencoder, z, steps, alpha, generator, context_frames=None):
                self.window = z
                self.noise = torch.randn(z.shape, generator=generator)

        try:
            with patch("demo.encode_video", return_value=torch.ones(1, 20, 2, 2, 1)) as encode:
                with patch("demo.StreamingDecoder", Decoder):
                    demo.new_prompt("driving", 0, seed=17)
                    original = demo.rollouts["driving"]
                    demo.new_prompt("driving", 0, seed=17)
                    cached = demo.rollouts["driving"]
                    self.assertEqual(encode.call_count, 1)
                    self.assertTrue(torch.equal(original.decoder.noise, cached.decoder.noise))
                    self.assertTrue(torch.equal(original.kv[0][0], cached.kv[0][0]))
                    cached.kv[0][0].zero_()
                    self.assertTrue(original.kv[0][0].any())
                    self.assertTrue(demo.prompt_cache[("driving", 0)][1][1][0][0].any())
        finally:
            demo.executor.shutdown(wait=True)

    def test_driving_default_and_invalid_capacity(self):
        self.assertEqual(parse_args([]).envs, ["driving"])
        with self.assertRaises(SystemExit):
            parse_args(["--max_active", "0"])


if __name__ == "__main__":
    unittest.main()

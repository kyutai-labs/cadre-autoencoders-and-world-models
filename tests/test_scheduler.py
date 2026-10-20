"""Planned submissions coalesce staggered streams without delaying absent peers."""

import asyncio
import json
import struct
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from demo import Chunk, ConcurrentDemo, Demo
from demo_scheduler import BatchScheduler


class PlannedBatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.scheduler = None

        async def heartbeat():
            while True:
                await asyncio.sleep(0.002)

        self.ticker = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        if self.scheduler:
            await self.scheduler.close()
        self.ticker.cancel()
        await asyncio.gather(self.ticker, return_exceptions=True)
        self.executor.shutdown(wait=True)

    def make_scheduler(self, produce, **kwargs):
        self.scheduler = BatchScheduler(self.executor, produce, 2, **kwargs)
        return self.scheduler

    async def test_lone_active_user_does_not_wait_for_collection(self):
        scheduler = self.make_scheduler(lambda rows: rows, wait_ms=500)
        session = object()
        scheduler.activate(session)
        self.assertIs(await asyncio.wait_for(scheduler.submit(session), 0.05), session)

    async def test_known_peer_plan_coalesces_and_samples_latest_input_at_worker_start(self):
        batches = []

        def produce(rows):
            batches.append(rows)
            return [row.action for row in rows]

        scheduler = self.make_scheduler(produce)
        first, second = SimpleNamespace(action=1), SimpleNamespace(action=2)
        scheduler.activate(first)
        scheduler.activate(second)
        scheduler.plan(second, asyncio.get_running_loop().time() + 0.045)
        pending = scheduler.submit(first)
        await asyncio.sleep(0.020)
        self.assertFalse(pending.done())
        first.action = 9  # Input arriving during collection must reach this job.
        await asyncio.sleep(0.020)
        other = scheduler.submit(second)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(pending, other), 0.05), [9, 2])
        self.assertEqual(batches, [[first, second]])

    async def test_departure_wakes_collector_and_new_admission_has_no_old_plan(self):
        scheduler = self.make_scheduler(lambda rows: rows)
        first, second = object(), object()
        for session in (first, second):
            scheduler.activate(session)
        scheduler.plan(second, asyncio.get_running_loop().time() + 1)
        pending = scheduler.submit(first)
        await asyncio.sleep(0.010)
        self.assertFalse(pending.done())
        scheduler.deactivate(second)
        self.assertIs(await asyncio.wait_for(pending, 0.025), first)
        scheduler.activate(second)
        scheduler.plan(second, asyncio.get_running_loop().time() + 0.020)
        pending = scheduler.submit(first)
        await asyncio.sleep(0.010)
        self.assertFalse(pending.done())
        other = scheduler.submit(second)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(pending, other), 0.05), [first, second])

    async def test_stale_peer_plan_only_gets_normal_grace_repeatedly(self):
        scheduler = self.make_scheduler(lambda rows: rows)
        first, stalled = object(), object()
        scheduler.activate(first)
        scheduler.activate(stalled)
        scheduler.plan(stalled, asyncio.get_running_loop().time() - 1)
        for _ in range(3):
            self.assertIs(await asyncio.wait_for(scheduler.submit(first), 0.025), first)

    async def test_earlier_plan_wakes_collector_promptly(self):
        scheduler = self.make_scheduler(lambda rows: rows)
        first, second = object(), object()
        scheduler.activate(first)
        scheduler.activate(second)
        scheduler.plan(second, asyncio.get_running_loop().time() + 1)
        pending = scheduler.submit(first)
        await asyncio.sleep(0.010)
        self.assertFalse(pending.done())
        scheduler.plan(second, asyncio.get_running_loop().time() - 0.010)
        self.assertIs(await asyncio.wait_for(pending, 0.025), first)

    async def test_plan_updates_cannot_extend_first_request_hard_cap(self):
        scheduler = self.make_scheduler(lambda rows: rows, max_hold_ms=35)
        first, stalled = object(), object()
        scheduler.activate(first)
        scheduler.activate(stalled)
        scheduler.plan(stalled, asyncio.get_running_loop().time() + 1)
        pending = scheduler.submit(first)

        async def reschedule():
            for _ in range(12):
                scheduler.plan(stalled, asyncio.get_running_loop().time() + 1)
                await asyncio.sleep(0.005)

        updates = asyncio.create_task(reschedule())
        began = asyncio.get_running_loop().time()
        self.assertIs(await asyncio.wait_for(pending, 0.060), first)
        self.assertGreaterEqual(asyncio.get_running_loop().time() - began, 0.025)
        await updates

    async def test_cancelled_collected_request_does_not_block_later_peer(self):
        batches = []

        def produce(rows):
            batches.append(rows)
            return rows

        scheduler = self.make_scheduler(produce)
        first, second = object(), object()
        scheduler.activate(first)
        scheduler.activate(second)
        scheduler.plan(second, asyncio.get_running_loop().time() + 1)
        canceled = scheduler.submit(first)
        await asyncio.sleep(0.010)
        survivor = scheduler.submit(second)
        canceled.cancel()
        scheduler.deactivate(first)
        self.assertIs(await asyncio.wait_for(survivor, 0.025), second)
        self.assertEqual(batches, [[second]])

    async def test_shutdown_cancels_collected_and_queued_requests(self):
        scheduler = self.make_scheduler(lambda rows: rows)
        first, second, third = object(), object(), object()
        for session in (first, second, third):
            scheduler.activate(session)
            scheduler.plan(session, asyncio.get_running_loop().time() + 1)
        collected = scheduler.submit(first)
        await asyncio.sleep(0.010)
        self.assertFalse(collected.done())
        queued = scheduler.submit(third)
        await scheduler.close()
        self.assertTrue(collected.cancelled())
        self.assertTrue(queued.cancelled())
        self.assertFalse(scheduler.plans)
        self.assertTrue(scheduler.pending.empty())


class StreamCoalescingTests(unittest.IsolatedAsyncioTestCase):
    async def test_staggered_singleton_streams_converge_to_paired_fresh_chunks(self):
        model = SimpleNamespace()
        ae = SimpleNamespace(config=SimpleNamespace(tau=2, num_frames=16))
        demo = Demo(ae, "test", {"driving": model}, {}, torch.device("cpu"), 3, 0.35)
        pool = ConcurrentDemo(demo, 2, batch_size=2, batch_wait_ms=2)
        batches, sockets = [], []
        began = time.perf_counter()

        def produce(slots):
            sampled = time.perf_counter()
            batches.append((sampled, len(slots)))
            # Reproduce the observed pathological regime: two singleton jobs
            # consume almost the entire100ms chunk period without coalescing.
            time.sleep(0.0488 if len(slots) == 1 else 0.089)
            return [
                Chunk(
                    [b"jpeg", b"jpeg"],
                    {
                        "sampled_at": sampled,
                        "worker_started": sampled,
                        "compute_ms": 1000 * (time.perf_counter() - sampled),
                        "prompt_ms": 0,
                    },
                    sampled,
                )
                for _ in slots
            ]

        demo.scheduler.produce = produce

        class Socket:
            def __init__(self):
                self.closed = asyncio.Event()
                self.frames = []

            async def send_bytes(self, message):
                length = struct.unpack("<I", message[:4])[0]
                self.frames.append((time.perf_counter(), json.loads(message[4 : 4 + length])))

        async def client(index):
            await asyncio.sleep(0.037 * index)
            socket = Socket()
            sockets.append(socket)
            slot = pool.slots[index]
            slot.line = [socket]
            receiver = asyncio.create_task(socket.closed.wait())
            try:
                await slot.stream(socket, receiver)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        async def heartbeat():
            while True:
                await asyncio.sleep(0.002)

        ticker = asyncio.create_task(heartbeat())
        clients = [asyncio.create_task(client(index)) for index in range(2)]
        try:
            await asyncio.sleep(1.8)
            stopped = time.perf_counter()
            for socket in sockets:
                socket.closed.set()
            await asyncio.wait_for(asyncio.gather(*clients), 0.5)
            measured = [width for started, width in batches if began + 0.5 <= started < stopped]
            self.assertGreaterEqual(len(measured), 10)
            self.assertGreaterEqual(measured.count(2) / len(measured), 0.95)
            for socket in sockets:
                frames = [
                    (shown, header) for shown, header in socket.frames if shown >= began + 0.5 and header["i"] == 0
                ]
                intervals = [right[0] - left[0] for left, right in zip(frames[:-1], frames[1:], strict=True)]
                self.assertAlmostEqual(sum(intervals) / len(intervals), 0.1, delta=0.008)
                self.assertLess(max(shown - header["sampled_at"] for shown, header in frames), 0.13)
            self.assertFalse(demo.scheduler.plans)
        finally:
            for socket in sockets:
                socket.closed.set()
            await asyncio.gather(*clients, return_exceptions=True)
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            await demo.scheduler.close()
            demo.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()

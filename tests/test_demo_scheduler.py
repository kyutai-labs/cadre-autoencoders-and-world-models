import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from demo_scheduler import BatchScheduler


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.scheduler = None

        async def heartbeat():
            while True:
                await asyncio.sleep(0.005)

        self.ticker = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        if self.scheduler:
            await self.scheduler.close()
        self.ticker.cancel()
        await asyncio.gather(self.ticker, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def test_bounded_fifo_batches_and_canceled_requests(self):
        batches = []

        def produce(values):
            batches.append(values)
            time.sleep(0.002)
            return [value * 10 for value in values]

        self.scheduler = BatchScheduler(self.executor, produce, 2, 1)
        futures = [self.scheduler.submit(i) for i in range(6)]
        futures[2].cancel()
        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), 2)
        self.assertEqual([r for r in results if not isinstance(r, BaseException)], [0, 10, 30, 40, 50])
        self.assertEqual([item for batch in batches for item in batch], [0, 1, 3, 4, 5])
        self.assertTrue(all(len(batch) <= 2 for batch in batches))
        self.assertEqual(batches[0], [0, 1])

    async def test_failure_is_isolated_and_worker_continues(self):
        def produce(values):
            return [ValueError("bad session") if value == 0 else value for value in values]

        self.scheduler = BatchScheduler(self.executor, produce, 2, 1)
        results = await asyncio.wait_for(
            asyncio.gather(self.scheduler.submit(0), self.scheduler.submit(1), return_exceptions=True), 2
        )
        self.assertIsInstance(results[0], ValueError)
        self.assertEqual(results[1], 1)
        self.assertEqual(await asyncio.wait_for(self.scheduler.submit(2), 2), 2)

    async def test_full_batch_starts_before_collection_deadline(self):
        batches = []

        def produce(values):
            batches.append(values)
            return values

        self.scheduler = BatchScheduler(self.executor, produce, 2, 500)
        first = self.scheduler.submit(0)
        await asyncio.sleep(0.020)  # the worker is collecting, rather than idle
        self.assertFalse(first.done())
        second = self.scheduler.submit(1)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(first, second), 0.2), [0, 1])
        self.assertEqual(batches, [[0, 1]])

    async def test_partial_batch_keeps_first_deadline_despite_later_arrival(self):
        batches = []
        began = time.perf_counter()

        def produce(values):
            batches.append((values, time.perf_counter() - began))
            return values

        self.scheduler = BatchScheduler(self.executor, produce, 3, 200)
        first = self.scheduler.submit(0)
        await asyncio.sleep(0.120)
        self.assertFalse(first.done())
        second = self.scheduler.submit(1)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(first, second), 0.16), [0, 1])
        self.assertEqual(batches[0][0], [0, 1])
        self.assertGreaterEqual(batches[0][1], 0.180)
        self.assertLess(batches[0][1], 0.280)

    async def test_canceled_requests_do_not_take_batch_slots_or_reorder_survivors(self):
        batches = []

        def produce(values):
            batches.append(values)
            return values

        self.scheduler = BatchScheduler(self.executor, produce, 2, 500)
        first = self.scheduler.submit(0)
        await asyncio.sleep(0.020)
        first.cancel()  # canceled while the collector is waiting for another item
        canceled = self.scheduler.submit(1)
        canceled.cancel()
        remaining = [self.scheduler.submit(i) for i in (2, 3)]
        self.assertEqual(await asyncio.wait_for(asyncio.gather(*remaining), 0.2), [2, 3])
        self.assertEqual(batches, [[2, 3]])

    async def test_zero_wait_dispatches_partial_batch_without_waiting(self):
        self.scheduler = BatchScheduler(self.executor, lambda values: values, 3, 0)
        self.assertEqual(await asyncio.wait_for(self.scheduler.submit(7), 0.2), 7)

    async def test_shutdown_resolves_pending_requests(self):
        self.scheduler = BatchScheduler(self.executor, lambda values: values, 2, 100)
        first, second = self.scheduler.submit(1), self.scheduler.submit(2)
        await asyncio.sleep(0)
        await self.scheduler.close()
        self.assertTrue(first.cancelled())
        self.assertTrue(second.cancelled())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.scheduler.submit(3)


if __name__ == "__main__":
    unittest.main()

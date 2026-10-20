"""Collect ready sessions into bounded batches on one inference worker."""

import asyncio


class BatchScheduler:
    def __init__(self, executor, produce, max_batch: int, wait_ms: float = 2.0, *, max_hold_ms: float = 100.0):
        self.executor, self.produce = executor, produce
        self.max_batch, self.wait_seconds = max_batch, wait_ms / 1000
        self.pending = asyncio.Queue()
        self.task = None
        self.current = []
        self.closed = False
        self.max_hold_seconds = max_hold_ms / 1000
        self.plans = {}
        self.changed = asyncio.Event()

    def activate(self, session):
        """Register a streaming participant; callers without plans keep legacy batching."""
        self.plans[id(session)] = asyncio.get_running_loop().time()
        self.changed.set()

    def plan(self, session, ready_at: float):
        """Publish when the stream intends to submit, without sampling its input."""
        if id(session) in self.plans:
            self.plans[id(session)] = ready_at
            self.changed.set()

    def deactivate(self, session):
        self.plans.pop(id(session), None)
        self.changed.set()

    def collection_deadline(self, requests, started):
        if not self.plans:
            return started + self.wait_seconds
        present = {id(session) for session, _ in requests}
        missing = sorted(ready_at for session, ready_at in self.plans.items() if session not in present)
        if not missing:
            return started  # A lone active user does not wait for an absent partner.
        needed = min(self.max_batch - len(requests), len(missing))
        # All wakes share the original cap. A past/stalled forecast only gets
        # the normal collection grace, never a fresh max-hold reservation.
        return min(
            started + self.max_hold_seconds,
            max(started + self.wait_seconds, missing[needed - 1] + self.wait_seconds),
        )

    def submit(self, session):
        if self.closed:
            raise RuntimeError("inference scheduler is closed")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if id(session) in self.plans:
            self.plans[id(session)] = loop.time()  # consume the previous forecast
        future.add_done_callback(lambda done: self.changed.set() if done.cancelled() else None)
        self.pending.put_nowait((session, future))
        self.changed.set()
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        return future

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.pending.get()
            if first[1].cancelled():
                continue
            requests = [first]
            self.current = requests
            started = loop.time()
            while len(requests) < self.max_batch:
                requests[:] = [(session, future) for session, future in requests if not future.cancelled()]
                if not requests:
                    break
                if not self.pending.empty():
                    request = self.pending.get_nowait()
                else:
                    deadline = self.collection_deadline(requests, started)
                    if loop.time() >= deadline:
                        break
                    self.changed.clear()
                    try:
                        async with asyncio.timeout_at(deadline):
                            await self.changed.wait()
                    except TimeoutError:
                        break
                    continue
                # A full batch starts immediately; a partial batch keeps the
                # original hard cap even as requests arrive or are canceled.
                requests[:] = [(session, future) for session, future in requests if not future.cancelled()]
                if not request[1].cancelled():
                    requests.append(request)
            requests[:] = [(session, future) for session, future in requests if not future.cancelled()]
            if not requests:
                continue
            try:
                results = await loop.run_in_executor(self.executor, self.produce, [s for s, _ in requests])
                if len(results) != len(requests):
                    raise RuntimeError("inference worker returned the wrong number of results")
            except Exception as error:
                results = [error] * len(requests)
            for (_, future), result in zip(requests, results, strict=True):
                if not future.done():
                    if isinstance(result, Exception):
                        future.set_exception(result)
                    else:
                        future.set_result(result)
            self.current = []

    async def close(self):
        self.closed = True
        self.plans.clear()
        self.changed.set()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        for _, future in self.current:
            future.cancel()
        self.current = []
        while not self.pending.empty():
            _, future = self.pending.get_nowait()
            future.cancel()

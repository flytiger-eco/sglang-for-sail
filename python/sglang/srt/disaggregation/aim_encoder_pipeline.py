"""Bounded CPU preparation for the encoder's ordered GPU dispatcher.

The coordinator owns no GPU work. Each job runs its asynchronous media loading
and CPU preprocessing with a copy of the submitting runtime context. The
existing encoder I/O and preprocessing pools remain responsible for that work.
"""

import asyncio
import concurrent.futures
import contextvars
import threading
from time import perf_counter


class _Preparation:
    def __init__(self, record=None):
        self.future = None
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.loop = None
        self.task = None
        self.record = record
        self.queued_at = perf_counter()

    def observe(self, stage, result, seconds):
        if self.record is not None:
            self.record(stage, result, seconds, 1)

    def run(self, prepare):
        async def run_async():
            with self.lock:
                self.loop = asyncio.get_running_loop()
                self.task = asyncio.current_task()
            try:
                if self.cancelled.is_set():
                    raise asyncio.CancelledError
                return await prepare()
            finally:
                with self.lock:
                    self.loop = self.task = None

        start = perf_counter()
        self.observe("encoder_prepare_queue", "success", start - self.queued_at)
        result = "success"
        try:
            return asyncio.run(run_async())
        except asyncio.CancelledError:
            result = "cancel"
            raise
        except BaseException:
            result = "error"
            raise
        finally:
            self.observe("encoder_prepare", result, perf_counter() - start)

    def cancel(self):
        already_cancelled = self.cancelled.is_set()
        self.cancelled.set()
        if self.future is not None:
            if self.future.cancel() and not already_cancelled:
                self.observe(
                    "encoder_prepare_queue", "cancel", perf_counter() - self.queued_at
                )
        with self.lock:
            if self.loop is not None and self.task is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)


class CPUPreparationQueue:
    """At most current + next batch, with one CPU coordinator at a time.

    All public methods are called by the encoder event-loop thread. Cancellation
    stops the coordinator's await; already-running I/O/CPU pool calls finish under
    their own timeouts, and do not hold up the dispatcher shutdown.
    """

    def __init__(self, record=None):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aim-encoder-prepare"
        )
        self._jobs = {}
        self._closed = False
        self._record = record

    def submit(self, batch_id, prepare):
        if self._closed:
            raise RuntimeError("Encoder preparation queue stopped")
        if batch_id in self._jobs:
            raise RuntimeError(f"Duplicate encoder preparation: {batch_id}")
        if len(self._jobs) >= 2:
            raise RuntimeError("Encoder preparation queue is full")
        job = _Preparation(record=self._record)
        job.future = self._executor.submit(
            contextvars.copy_context().run, job.run, prepare
        )
        self._jobs[batch_id] = job

    async def wait_ready(self, batch_id):
        # The result/error is consumed in batch_encode on every TP rank. A
        # canceled HTTP waiter never owns this await; only scheduler shutdown
        # cancels it, and shutdown also closes preparation on the TP peers.
        await asyncio.wrap_future(self._jobs[batch_id].future)

    async def take(self, batch_id):
        job = self._jobs[batch_id]
        try:
            return await asyncio.wrap_future(job.future)
        except asyncio.CancelledError:
            job.cancel()
            raise
        finally:
            self._jobs.pop(batch_id, None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for job in self._jobs.values():
            job.cancel()
        self._jobs.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

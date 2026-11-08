"""CPU lifecycle tests for the production encoder preparation/scheduler code.

The scheduler is loaded from its AST so these tests also run on a development
host without torch, CUDA, transformers or a model. Only GPU/protocol endpoints
are substituted; the preparation queue and scheduler methods are unchanged.
"""

import ast
import asyncio

# These imports also populate the namespace for AST-loaded runtime methods.
import concurrent.futures  # noqa: F401
import contextlib  # noqa: F401
import contextvars
import enum
import importlib.util
import logging
import threading
import time  # noqa: F401
import unittest
import uuid  # noqa: F401
from collections import defaultdict  # noqa: F401
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple  # noqa: F401

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/disaggregation/encoder/runtime.py"
SERVER = ROOT / "python/sglang/srt/disaggregation/encoder/server.py"
spec = importlib.util.spec_from_file_location(
    "aim_encoder_pipeline",
    ROOT / "python/sglang/srt/disaggregation/aim_encoder_pipeline.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
CPUPreparationQueue = pipeline.CPUPreparationQueue


class Modality(enum.Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

    @classmethod
    def from_str(cls, value):
        return cls(value.lower())


def load_scheduler():
    names = {
        "MMError",
        "BadRequestError",
        "InternalError",
        "PendingRequest",
        "EncoderScheduler",
        "_handle_encoder_worker_request",
    }
    server_tree = ast.parse(SERVER.read_text())
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        n
        for n in server_tree.body
        if getattr(n, "name", None)
        in {
            "MMError",
            "BadRequestError",
            "InternalError",
            "_handle_encoder_worker_request",
        }
    ]
    nodes += [
        n
        for n in tree.body
        if getattr(n, "name", None)
        in {"PendingRequest", "EncoderScheduler", "validate_encode_request"}
    ]
    namespace = dict(globals())
    namespace.update(
        logger=logging.getLogger("aim-encoder-test"),
        cuda_failure_detected=lambda: False,
        _BATCHABLE_MODALITIES={Modality.IMAGE, Modality.AUDIO},
        zmq=SimpleNamespace(Socket=object),
        MMEncoder=object,
        ProfileReq=type("ProfileReq", (), {}),
        envs=SimpleNamespace(
            SGLANG_AIM_ENCODER_MAX_PENDING=SimpleNamespace(get=lambda: 2)
        ),
        sock_send=lambda socket, message: socket.append(message),
        wrap_as_pickle=lambda message: message,
        server_module=SimpleNamespace(
            ENCODER_REQ_TIMEOUT=5, encoder_metrics_collector=None
        ),
    )
    # Error classes precede EncoderScheduler's default arguments.
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    for name in ["MMError", "BadRequestError", "InternalError"]:
        setattr(namespace["server_module"], name, namespace[name])
    encoder = next(
        n for n in server_tree.body if getattr(n, "name", None) == "MMEncoder"
    )
    prepare = next(
        n
        for n in encoder.body
        if getattr(n, "name", None) == "_prepare_encode_context_on_all_ranks"
    )
    namespace["EncodeContext"] = object
    namespace["_preprocess_layout_digest"] = lambda ctx: (1, 2)
    exec(
        compile(ast.Module(body=[prepare], type_ignores=[]), str(SERVER), "exec"),
        namespace,
    )
    return namespace


NS = load_scheduler()
EncoderScheduler = NS["EncoderScheduler"]


def request(index, modality="image"):
    return {
        "req_id": str(index),
        "modality": modality,
        "mm_items": [index],
        "num_parts": 1,
        "part_idx": 0,
    }


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


class FakeEncoder:
    def __init__(self, prepare=None, forward=None):
        self.aim_encoder_pipeline_enabled = True
        self._aim_preparation = CPUPreparationQueue()
        self.encode_dispatch_lock = asyncio.Lock()
        self.embedding_to_send = {}
        self.prepared = []
        self.forwarded = []
        self.prepare_fn = prepare
        self.forward_fn = forward

    async def release_request(self, req_id):
        self.embedding_to_send.pop(req_id, None)

    async def abandon_request(self, req_id):
        await self.release_request(req_id)

    def clear_abandoned_request(self, req_id):
        pass

    def prepare_batch(self, batch_id, requests, modality):
        self.prepared.append([r["req_id"] for r in requests])

        async def prepare():
            if self.prepare_fn:
                await self.prepare_fn(requests)
            return [r["req_id"] for r in requests]

        self._aim_preparation.submit(batch_id, prepare)

    async def batch_encode(self, requests, modality, prepared_batch_id=None):
        if prepared_batch_id is not None:
            ids = await self._aim_preparation.take(prepared_batch_id)
            assert ids == [r["req_id"] for r in requests]
        self.forwarded.append([r["req_id"] for r in requests])
        if self.forward_fn:
            self.forward_fn(requests)
        for req in requests:
            self.embedding_to_send[req["req_id"]] = object()
        return [(1, 2, 3, None, None) for _ in requests]


class TestCPUPreparationQueue(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_distinguish_running_and_queued_cancellation(self):
        events = []
        queue = CPUPreparationQueue(record=lambda *event: events.append(event))
        started = threading.Event()

        async def blocked():
            started.set()
            await asyncio.sleep(60)

        queue.submit("running", blocked)
        queue.submit("queued", blocked)
        await eventually(started.is_set)
        queue.close()
        await eventually(
            lambda: any(event[:2] == ("encoder_prepare", "cancel") for event in events)
        )
        self.assertEqual(
            sorted((stage, outcome) for stage, outcome, _, _ in events),
            [
                ("encoder_prepare", "cancel"),
                ("encoder_prepare_queue", "cancel"),
                ("encoder_prepare_queue", "success"),
            ],
        )

    async def test_metrics_distinguish_queue_and_execution_outcomes(self):
        events = []
        queue = CPUPreparationQueue(record=lambda *event: events.append(event))

        async def success():
            return "pixels"

        async def failure():
            raise ValueError("invalid image")

        try:
            queue.submit("good", success)
            self.assertEqual(await queue.take("good"), "pixels")
            queue.submit("bad", failure)
            with self.assertRaisesRegex(ValueError, "invalid image"):
                await queue.take("bad")
            self.assertEqual(
                [(stage, outcome) for stage, outcome, _, _ in events],
                [
                    ("encoder_prepare_queue", "success"),
                    ("encoder_prepare", "success"),
                    ("encoder_prepare_queue", "success"),
                    ("encoder_prepare", "error"),
                ],
            )
            self.assertTrue(
                all(seconds >= 0 and count == 1 for _, _, seconds, count in events)
            )
        finally:
            queue.close()

    async def test_bounded_jobs_and_runtime_context(self):
        queue = CPUPreparationQueue()
        release = threading.Event()
        context = contextvars.ContextVar("model_session")
        context.set("qwen38")

        async def prepare():
            while not release.is_set():
                await asyncio.sleep(0.001)
            return context.get(), threading.get_ident()

        try:
            queue.submit("first", prepare)
            queue.submit("second", prepare)
            with self.assertRaisesRegex(RuntimeError, "full"):
                queue.submit("third", prepare)
            context.set("another_session")
            release.set()
            results = [await queue.take("first"), await queue.take("second")]
            self.assertEqual([r[0] for r in results], ["qwen38", "qwen38"])
            self.assertTrue(all(r[1] != threading.get_ident() for r in results))
            self.assertEqual(results[0][1], results[1][1])
        finally:
            queue.close()

    async def test_close_cancels_running_coroutine_and_queued_job(self):
        queue = CPUPreparationQueue()
        started = threading.Event()
        canceled = threading.Event()
        queued_started = threading.Event()

        async def prepare():
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                canceled.set()

        async def queued():
            queued_started.set()

        queue.submit("running", prepare)
        queue.submit("queued", queued)
        await eventually(started.is_set)
        queue.close()
        await eventually(canceled.is_set)
        self.assertFalse(queued_started.is_set())
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            queue.submit("after-close", queued)
        queue.close()


class TestEncoderPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_preparation_error_does_not_stop_following_batch(self):
        async def prepare(requests):
            if requests[0]["req_id"] == "0":
                raise ValueError("invalid image")

        encoder = FakeEncoder(prepare)
        scheduler = EncoderScheduler(encoder, [], 1)
        scheduler.start()
        try:
            with self.assertLogs("aim-encoder-test", level="ERROR"):
                results = await asyncio.gather(
                    scheduler.submit(request(0)),
                    scheduler.submit(request(1)),
                    return_exceptions=True,
                )
            self.assertIsInstance(results[0], ValueError)
            self.assertEqual(results[1], (1, 2, 3, None, None))
            self.assertEqual(encoder.forwarded, [["1"]])
            self.assertFalse(scheduler._stopped)
        finally:
            await scheduler.stop()

    async def test_bad_modality_is_rejected_without_stopping_valid_batch(self):
        for modalities in [("image", "invalid"), ("invalid", "image")]:
            with self.subTest(modalities=modalities):
                encoder = FakeEncoder()
                scheduler = EncoderScheduler(encoder, [], 2, coalesce_same_turn=True)
                scheduler.start()
                try:
                    results = await asyncio.gather(
                        *(
                            scheduler.submit(request(i, modality))
                            for i, modality in enumerate(modalities)
                        ),
                        return_exceptions=True,
                    )
                    for modality, result in zip(modalities, results):
                        if modality == "image":
                            self.assertEqual(result, (1, 2, 3, None, None))
                        else:
                            self.assertIsInstance(result, NS["BadRequestError"])
                    self.assertFalse(scheduler._stopped)
                finally:
                    await scheduler.stop()

    async def test_next_cpu_batch_overlaps_blocking_ordered_forward(self):
        release_first = threading.Event()
        gpu_running = threading.Event()
        second_ready = threading.Event()
        main_thread = threading.get_ident()

        async def prepare(requests):
            if requests[0]["req_id"] == "0":
                while not release_first.is_set():
                    await asyncio.sleep(0.001)
            else:
                self.assertTrue(gpu_running.wait(1))
                second_ready.set()

        def forward(requests):
            self.assertEqual(threading.get_ident(), main_thread)
            if requests[0]["req_id"] == "0":
                gpu_running.set()
                # Blocking the event loop models the real synchronous ViT.
                self.assertTrue(second_ready.wait(1))

        encoder = FakeEncoder(prepare, forward)
        wire = []
        scheduler = EncoderScheduler(encoder, [wire], 1)
        scheduler.start()
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(2)]
        try:
            await eventually(lambda: len(encoder.prepared) == 2)
            release_first.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertEqual(encoder.forwarded, [["0"], ["1"]])
            self.assertEqual(encoder.prepared, [["0"], ["1"]])
            self.assertEqual(
                [m["type"] for m in wire],
                ["prepare_batch", "prepare_batch", "batch_encode", "batch_encode"],
            )
        finally:
            await scheduler.stop()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_stop_releases_current_lookahead_and_queued_waiters(self):
        canceled = threading.Event()

        async def prepare(_):
            try:
                await asyncio.sleep(60)
            finally:
                canceled.set()

        encoder = FakeEncoder(prepare)
        scheduler = EncoderScheduler(encoder, [], 1)
        scheduler.start()
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(2)]
        await eventually(lambda: len(encoder.prepared) == 2)
        tasks.append(asyncio.create_task(scheduler.submit(request(2))))
        await eventually(lambda: scheduler.pending_queue.qsize() == 1)
        await asyncio.wait_for(scheduler.stop(), 1)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
        await eventually(canceled.is_set)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            await scheduler.submit(request(3))

    async def test_backpressure_and_cancel_before_preparation(self):
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(encoder, [], 1)
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(2)]
        await eventually(lambda: scheduler.pending_queue.qsize() == 2)
        with self.assertRaises(NS["MMError"]) as caught:
            await scheduler.submit(request(2))
        self.assertEqual(caught.exception.code, HTTPStatus.SERVICE_UNAVAILABLE)
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        scheduler.start()
        try:
            await tasks[1]
            self.assertEqual(encoder.prepared, [["1"]])
        finally:
            await scheduler.stop()

    async def test_non_image_batch_bypasses_preparation(self):
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(encoder, [], 1)
        scheduler.start()
        try:
            await scheduler.submit(request(0, modality="audio"))
            self.assertEqual(encoder.prepared, [])
            self.assertEqual(encoder.forwarded, [["0"]])
        finally:
            await scheduler.stop()

    async def test_canceled_prepared_request_finishes_tp_order_and_releases_result(
        self,
    ):
        release = threading.Event()

        async def prepare(_):
            while not release.is_set():
                await asyncio.sleep(0.001)

        encoder = FakeEncoder(prepare)
        scheduler = EncoderScheduler(encoder, [], 1)
        scheduler.start()
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(2)]
        try:
            await eventually(lambda: len(encoder.prepared) == 2)
            tasks[1].cancel()
            await asyncio.gather(tasks[1], return_exceptions=True)
            release.set()
            await tasks[0]
            await eventually(lambda: len(encoder.forwarded) == 2)
            self.assertEqual(encoder.forwarded, [["0"], ["1"]])
            # Forwarding is observed before the scheduler's awaited cleanup.
            await eventually(lambda: "1" not in encoder.embedding_to_send)
            self.assertNotIn("1", encoder.embedding_to_send)
        finally:
            await scheduler.stop()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_tp_preprocessing_error_consensus_happens_before_forward(self):
        queue = CPUPreparationQueue()
        main_thread = threading.get_ident()
        sync_threads = []

        async def prepare():
            return SimpleNamespace()

        def sync(error, **kwargs):
            self.assertIsNone(error)
            sync_threads.append(threading.get_ident())
            raise NS["BadRequestError"]("Image preprocessing failed on another TP rank")

        queue.submit("batch", prepare)
        encoder = SimpleNamespace(
            _aim_preparation=queue, image_pruning_rate=0.5, _sync_tp_prepare_status=sync
        )
        try:
            with self.assertRaisesRegex(NS["BadRequestError"], "another TP rank"):
                await NS["_prepare_encode_context_on_all_ranks"](
                    encoder,
                    [request(0)],
                    Modality.IMAGE,
                    use_global_cache=False,
                    prepared_batch_id="batch",
                )
            self.assertEqual(sync_threads, [main_thread])
        finally:
            queue.close()

    async def test_disabled_pipeline_uses_existing_batch_path(self):
        encoder = FakeEncoder()
        encoder.aim_encoder_pipeline_enabled = False
        scheduler = EncoderScheduler(encoder, [], 1)
        scheduler.start()
        try:
            await scheduler.submit(request(0))
            self.assertEqual(encoder.prepared, [])
            self.assertEqual(encoder.forwarded, [["0"]])
            self.assertEqual(scheduler.pending_queue.maxsize, 0)
        finally:
            await scheduler.stop()
            encoder._aim_preparation.close()

    async def test_tp_worker_consumes_same_preparation_ticket(self):
        encoder = FakeEncoder()
        handler = NS["_handle_encoder_worker_request"]
        requests = [request(0)]
        try:
            await handler(
                encoder,
                {
                    "type": "prepare_batch",
                    "batch_id": "batch0",
                    "requests": requests,
                    "modality": "IMAGE",
                },
            )
            await handler(
                encoder,
                {
                    "type": "batch_encode",
                    "prepared_batch_id": "batch0",
                    "requests": requests,
                    "modality": "IMAGE",
                },
            )
            self.assertEqual(encoder.prepared, [["0"]])
            self.assertEqual(encoder.forwarded, [["0"]])
            await handler(encoder, {"type": "close_preparation"})
        finally:
            encoder._aim_preparation.close()


if __name__ == "__main__":
    unittest.main()

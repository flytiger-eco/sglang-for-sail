"""CPU-only visual receive/preparation lane; no scheduler or device operations.

The worker owns both ZMQ sockets and all mutable preparation objects. Scheduler
threads only snapshot immutable terminal results and enqueue retirement. Early
embedding parts are bounded and may arrive before the tokenized prompt.
"""
import contextvars
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import zmq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedResult:
    status: int
    request: object = None
    error: str = None
    code: int = None


class BackgroundReceiver:
    def __init__(self, host, factory, decode_metadata, resolve_rid,
                 timeout=120, max_requests=128, max_bytes=512 * 1024**2):
        self.host, self.factory = host, factory
        self.decode_metadata, self.resolve_rid = decode_metadata, resolve_rid
        self.timeout, self.max_requests, self.max_bytes = timeout, max_requests, max_bytes
        self.epoch = uuid.uuid4().hex
        self.records, self.retired = {}, {}
        self.bytes = 0
        self.results = {}
        self.lock = threading.Lock()
        self.commands = queue.SimpleQueue()
        self.stop_event, self.started = threading.Event(), threading.Event()
        self.failure = None
        context = contextvars.copy_context()
        self.thread = threading.Thread(target=lambda: context.run(self._run),
                                       name="encoder-background-ready", daemon=True)
        self.thread.start()
        if not self.started.wait(10) or self.failure is not None:
            raise RuntimeError(f"Background receiver startup failed: {self.failure}")

    def snapshot(self, rid):
        with self.lock:
            return self.results.get(rid)

    def discard(self, rid):
        self.commands.put(rid)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _publish(self, rid, result):
        with self.lock:
            self.results[rid] = result

    def _drop(self, rid):
        rec = self.records.pop(rid, None)
        if rec is not None:
            self.bytes -= rec["bytes"]
        with self.lock:
            self.results.pop(rid, None)
        self.retired[rid] = time.monotonic()

    def _fail(self, rid, error, code=400):
        rec = self.records.get(rid)
        if rec is not None:
            self.bytes -= rec["bytes"]
            rec.update(bytes=0, parts=[], waiting=None)
        self._publish(rid, PreparedResult(-1, error=error, code=code))

    def _record(self, rid):
        if rid in self.retired:
            return None
        rec = self.records.get(rid)
        if rec is None:
            if len(self.records) >= self.max_requests:
                # Keep the oldest admitted request; reject new input explicitly.
                # Bound rejection records as well; a scheduler-side timeout is
                # the final guard when admission itself is saturated.
                with self.lock:
                    if len(self.results) < 2 * self.max_requests:
                        self.results[rid] = PreparedResult(-1, error="Background visual queue full", code=503)
                self.retired[rid] = time.monotonic()
                return None
            rec = {"created": time.monotonic(), "bytes": 0, "parts": [], "waiting": None}
            self.records[rid] = rec
        return rec

    def _prepare(self, data):
        rid = data.get("rid")
        if not isinstance(rid, str) or not rid:
            return
        if data.get("epoch") != self.epoch:
            self._fail(rid, "Stale background receiver epoch", 503)
            return
        rec = self._record(rid)
        if rec is None or self.snapshot(rid) is not None:
            return
        if rec["waiting"] is not None:
            self._fail(rid, "Duplicate visual preparation request")
            return
        try:
            prompt = data["prompt"]
            req = SimpleNamespace(rid=rid, input_text=prompt["text"], input_ids=prompt["ids"],
                                  epd_prompt_pretokenized=prompt["pretokenized"], num_items_assigned={})
            rec["waiting"] = self.factory(req, self.embedding_port)
            parts, rec["parts"] = rec["parts"], []
            for meta, frames in parts:
                self._consume(rid, rec, meta, frames)
        except Exception as exc:
            logger.exception("Background prompt preparation failed: %s", rid)
            self._fail(rid, str(exc))

    def _consume(self, rid, rec, meta, frames):
        waiting = rec["waiting"]
        if waiting is None or self.snapshot(rid) is not None:
            return
        waiting.consume_parts(frames, decoded=meta)
        if waiting.status != 0:
            self._publish(rid, PreparedResult(int(waiting.status), waiting.recv_req,
                                             waiting.error_msg, waiting.error_code))
            # No more mutation after publishing. Retain the charged byte budget
            # until Scheduler consumes/retires this request.
            rec["waiting"] = None

    def _embedding(self, frames):
        rid = None
        try:
            meta = self.decode_metadata(frames[0])
            rid = self.resolve_rid(meta)
            if not rid or self.snapshot(rid) is not None:
                return
            rec = self._record(rid)
            if rec is None:
                return
            # Charge raw buffers plus CPU materialization, conservatively.
            size = 2 * sum(len(x) for x in frames)
            if self.bytes + size > self.max_bytes:
                self._fail(rid, "Background visual byte budget exceeded", 503)
                return
            self.bytes += size
            rec["bytes"] += size
            if rec["waiting"] is None:
                rec["parts"].append((meta, frames))
            else:
                self._consume(rid, rec, meta, frames)
        except Exception as exc:
            logger.exception("Background embedding receive failed: %s", rid)
            if rid:
                self._fail(rid, str(exc))

    def _maintenance(self):
        while True:
            try:
                self._drop(self.commands.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic()
        for rid, rec in list(self.records.items()):
            age = now - rec["created"]
            if age > self.timeout and self.snapshot(rid) is None:
                self._fail(rid, "Background visual preparation timed out", 408)
            if age > 2 * self.timeout:
                self._drop(rid)
        for rid, retired in list(self.retired.items()):
            if now - retired > 2 * self.timeout:
                self.retired.pop(rid, None)
                with self.lock:
                    self.results.pop(rid, None)

    def _run(self):
        context = zmq.Context()
        control, embedding = context.socket(zmq.PULL), context.socket(zmq.PULL)
        try:
            for socket in (control, embedding):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVHWM, 128)
            self.control_port = control.bind_to_random_port("tcp://127.0.0.1")
            self.embedding_port = embedding.bind_to_random_port(f"tcp://{self.host}")
            poller = zmq.Poller()
            poller.register(control, zmq.POLLIN)
            poller.register(embedding, zmq.POLLIN)
            self.started.set()
            while not self.stop_event.is_set():
                self._maintenance()
                ready = dict(poller.poll(2))
                if control in ready:
                    try:
                        self._prepare(control.recv_json())
                    except Exception:
                        logger.exception("Invalid background preparation control message")
                if embedding in ready:
                    self._embedding(embedding.recv_multipart(copy=False))
        except BaseException as exc:
            self.failure = repr(exc)
            self.started.set()
            logger.exception("Background receiver stopped")
            for rid in list(self.records):
                self._fail(rid, f"Background receiver stopped: {exc}", 500)
        finally:
            control.close()
            embedding.close()
            context.term()

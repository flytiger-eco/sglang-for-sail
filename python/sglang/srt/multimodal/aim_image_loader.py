"""AIM image batching with bounded work on a caller-owned I/O executor.

The coordinator must run outside ``executor``. Keeping the executor at the
processor/Encoder lifetime avoids a thread pool for every multimodal request.
This module deliberately has no eager torch or image-library imports.
"""

from concurrent.futures import FIRST_COMPLETED, CancelledError, wait
from threading import current_thread
from time import perf_counter


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Image loading was cancelled")


def bounded_map(function, items, *, executor, max_in_flight, cancel_event=None):
    """Return ordered results with at most ``max_in_flight`` submitted tasks.

    On error/cancellation, queued tasks are cancelled; already-running I/O must
    finish under its own timeout. The caller retains ownership of the executor.
    Do not call from one of its workers: even a one-worker pool must remain usable.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    # ThreadPoolExecutor has no public worker-membership API. This guard catches
    # accidental nesting in our processor/Encoder pools before it deadlocks.
    if current_thread() in getattr(executor, "_threads", ()):
        raise RuntimeError("Image batch coordinator cannot run in its I/O executor")
    _check_cancelled(cancel_event)
    results = [None] * len(items)
    pending = {}
    next_index = 0

    def run(item):
        _check_cancelled(cancel_event)
        return function(item)

    try:
        while next_index < len(items) or pending:
            _check_cancelled(cancel_event)
            while next_index < len(items) and len(pending) < max_in_flight:
                future = executor.submit(run, items[next_index])
                pending[future] = next_index
                next_index += 1
            done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            # Inspect all completed tasks before replenishing, so an error
            # cannot cause the next wave of downloads to be submitted.
            for future in sorted(done, key=pending.__getitem__):
                index = pending.pop(future)
                results[index] = future.result()
        _check_cancelled(cancel_event)
        return results
    finally:
        for future in pending:
            future.cancel()


def _record(record, stage, result, seconds, count):
    if record is not None:
        record(stage, result, seconds, count)


def _record_failure(record, stage, error, started, count):
    # Match image_stage for both asyncio and concurrent.futures cancellation.
    outcome = "cancel" if type(error).__name__ == "CancelledError" else "error"
    _record(record, stage, outcome, perf_counter() - started, count)


def _resolve_redkv_batch(image_files, *, mget, record):
    from sglang.srt.connector.redkv import fallback_url, is_redkv_url

    # ImageData carries a URL; load_image also unwraps it before loading. Other
    # inputs stay untouched, including PIL images and caller-specific objects.
    sources = [getattr(source, "url", source) for source in image_files]
    indices = [
        index
        for index, source in enumerate(sources)
        if isinstance(source, str) and is_redkv_url(source)
    ]
    if not indices:
        return image_files
    if mget is None:
        from sglang.srt.connector.redkv import mget_bytes_from_redis_cache_with_status

        mget = mget_bytes_from_redis_cache_with_status

    keys = list(dict.fromkeys(sources[index] for index in indices))
    start = perf_counter()
    fetched = mget(keys)
    hits = sum(fetched.values.get(key) is not None for key in keys)
    if fetched.error is not None:
        outcome = "error"
    elif hits == len(keys):
        outcome = "all_hit"
    elif hits == 0:
        outcome = "all_miss"
    else:
        outcome = "partial_miss"
    _record(record, "redkv_mget", outcome, perf_counter() - start, len(keys))
    resolved = list(image_files)
    counts = {"hit": 0, "miss": 0, "error": 0}
    for index in indices:
        key = sources[index]
        value = fetched.values.get(key)
        if value is not None:
            resolved[index] = value
            counts["hit"] += 1
        else:
            # Do not send the original key through load_image again: that would
            # issue a duplicate GET and misclassify connection failures as misses.
            resolved[index] = fallback_url(key)
            counts["error" if fetched.error is not None else "miss"] += 1
    for outcome, count in counts.items():
        if count:
            _record(record, "redkv_cache", outcome, 0.0, count)
    return resolved


class _ReadyImage:
    """An error fallback already produced by the caller's single-image policy."""

    def __init__(self, result):
        self.result = result


def decode_jpeg_cpu(image_bytes):
    """Decode one JPEG to a fully materialized RGB PIL image on CPU.

    Imports stay lazy so opting out requires neither torchvision nor its JPEG
    extension. RGB mode also handles grayscale JPEGs without one-channel tensors
    reaching the multimodal processor. EXIF orientation follows load_image's
    existing PIL behavior (no automatic rotation).
    """
    import torch
    from PIL import Image
    from torchvision.io import ImageReadMode, decode_jpeg

    # bytearray provides writable storage, avoiding torch.frombuffer's warning
    # for immutable bytes; the tensor stays alive for the duration of decoding.
    encoded = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
    decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
    pixels = decoded.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(pixels, mode="RGB")


def _load_images_cpu(
    sources,
    *,
    executor,
    max_in_flight,
    decode_max_in_flight,
    load_one,
    get_bytes,
    decoder,
    on_error,
    cancel_event,
    record,
):
    if get_bytes is None:
        from sglang.srt.utils.common import get_image_bytes

        get_bytes = get_image_bytes
    decoder = decoder or decode_jpeg_cpu

    def fetch(source):
        unwrapped = getattr(source, "url", source)
        if not isinstance(unwrapped, str):
            return unwrapped
        try:
            return get_bytes(unwrapped)
        except CancelledError:
            raise
        except Exception as error:
            if on_error is None:
                raise
            return _ReadyImage(on_error(source, error))

    fetch_count = sum(
        isinstance(getattr(source, "url", source), str) for source in sources
    )
    start = perf_counter()
    try:
        fetched = bounded_map(
            fetch,
            sources,
            executor=executor,
            max_in_flight=max_in_flight,
            cancel_event=cancel_event,
        )
    except BaseException as error:
        _record_failure(record, "image_fetch", error, start, fetch_count)
        raise
    if fetch_count:
        outcome = (
            "fallback"
            if any(isinstance(source, _ReadyImage) for source in fetched)
            else "success"
        )
        _record(record, "image_fetch", outcome, perf_counter() - start, fetch_count)

    def decode(source):
        if isinstance(source, _ReadyImage):
            return source.result, None
        if isinstance(source, bytes) and source.startswith(b"\xff\xd8"):
            start = perf_counter()
            try:
                image = decoder(source)
            except (CancelledError, MemoryError):
                raise
            except Exception as error:
                if type(error).__name__ == "OutOfMemoryError":
                    raise
                # The existing loader applies PIL/HEIF and per-image error
                # policy to these already-fetched bytes; no network retry.
                elapsed = perf_counter() - start
                return load_one(source), ("fallback", elapsed)
            elapsed = perf_counter() - start
            return load_one(image), ("success", elapsed)
        return load_one(source), None

    start = perf_counter()
    try:
        decoded = bounded_map(
            decode,
            fetched,
            executor=executor,
            max_in_flight=decode_max_in_flight,
            cancel_event=cancel_event,
        )
    except BaseException as error:
        _record_failure(record, "image_decode", error, start, len(sources))
        raise
    _record(record, "image_decode", "success", perf_counter() - start, len(sources))
    # Aggregate per-JPEG work separately from batch wall time. Stats callbacks
    # always run on the coordinator, never concurrently in the I/O workers.
    for outcome in ("success", "fallback"):
        timings = [
            stat[1] for _, stat in decoded if stat is not None and stat[0] == outcome
        ]
        if timings:
            _record(record, "jpeg_cpu_decode", outcome, sum(timings), len(timings))
    return [result for result, _ in decoded]


def load_images_batch(
    image_files,
    *,
    executor,
    max_in_flight,
    load_one,
    enable_mget=False,
    mget=None,
    cancel_event=None,
    record=None,
    cpu_jpeg_decode=False,
    decode_max_in_flight=None,
    get_bytes=None,
    decode_cpu_jpeg=None,
    on_error=None,
):
    """Load mixed image inputs once, preserving order and ``load_one``'s contract.

    ``load_one`` is the existing single-image loader with its decode/fallback
    options bound. Optional ``record(stage, result, seconds, count)`` callbacks
    run on the coordinator thread and contain bounded labels, never image URLs.
    Cache failures and misses both use HTTP fallback but retain distinct stats.

    CPU batch decode fetches bytes once and retains ``load_one``'s output
    contract by passing decoded JPEGs back as PIL images. Bind ``load_one`` with
    GPU decode disabled when enabling this option; JPEG decode failures then
    use its per-image CPU/PIL fallback. ``on_error(source, error)`` applies the
    same failure policy to a fetch error without retrying the failed request.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    if decode_max_in_flight is None:
        decode_max_in_flight = max_in_flight
    if decode_max_in_flight < 1:
        raise ValueError("decode_max_in_flight must be positive")
    if not image_files:
        _check_cancelled(cancel_event)
        return []
    # image_load covers the entire batch, including MGET and HTTP fallbacks.
    start = perf_counter()
    try:
        _check_cancelled(cancel_event)
        sources = (
            _resolve_redkv_batch(image_files, mget=mget, record=record)
            if enable_mget
            else image_files
        )
        _check_cancelled(cancel_event)
        if cpu_jpeg_decode:
            results = _load_images_cpu(
                sources,
                executor=executor,
                max_in_flight=max_in_flight,
                decode_max_in_flight=decode_max_in_flight,
                load_one=load_one,
                get_bytes=get_bytes,
                decoder=decode_cpu_jpeg,
                on_error=on_error,
                cancel_event=cancel_event,
                record=record,
            )
        else:
            results = bounded_map(
                load_one,
                sources,
                executor=executor,
                max_in_flight=max_in_flight,
                cancel_event=cancel_event,
            )
    except BaseException as error:
        _record_failure(record, "image_load", error, start, len(image_files))
        raise
    _record(record, "image_load", "success", perf_counter() - start, len(sources))
    return results

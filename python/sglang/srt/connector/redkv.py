# SPDX-License-Identifier: Apache-2.0
"""AIM image cache access via EDS, adapted from v0.0.0+aim0728_0.

Only imagetransfer_http(s) inputs use this connector. redis/edsredis are lazy
optional dependencies; a cache failure preserves the caller's HTTP fallback.
No direct Redis address fallback is used when EDS configuration is absent.
"""

import atexit
import inspect
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedKVGetResult:
    value: Optional[bytes]
    error: Optional[str] = None


@dataclass(frozen=True)
class RedKVMgetResult:
    values: Dict[str, Optional[bytes]]
    error: Optional[str] = None


_redis_client = None
_redis_client_lock = threading.Lock()
_retry_after = 0.0
_RETRY_COOLDOWN = 60.0


def is_redkv_url(source):
    return isinstance(source, str) and source.startswith(
        ("imagetransfer_http://", "imagetransfer_https://")
    )


def fallback_url(key):
    if not is_redkv_url(key):
        raise ValueError("Expected an imagetransfer_http(s) image key")
    return key[len("imagetransfer_") :]


def _positive_env(name, default, convert=float):
    try:
        value = convert(os.environ.get(name, default))
        if value > 0 and math.isfinite(value):
            return value
    except (ValueError, TypeError):
        pass
    logger.warning("Invalid %s; using default %s", name, default)
    return default


def _create_redis_client():
    # The internal EDS library must come from the company's package registry.
    from edsredis import EdsBlockingConnectionPool
    from redis import Connection, Redis

    socket_timeout = _positive_env("SGLANG_REDKV_SOCKET_TIMEOUT", 2.0)
    kwargs = {
        "socket_timeout": socket_timeout,
        "socket_connect_timeout": _positive_env(
            "SGLANG_REDKV_SOCKET_CONNECT_TIMEOUT", 2.0
        ),
        "timeout": socket_timeout,
        "max_connections": _positive_env("SGLANG_REDKV_MAX_CONNECTIONS", 32, int),
    }
    if os.environ.get("SGLANG_REDKV_PROTOCOL", "2") != "2":
        logger.warning("RedKV requires RESP2; ignoring SGLANG_REDKV_PROTOCOL")
    # redis-py versions before protocol selection already use RESP2. Newer
    # versions can default to RESP3, which the EDS RedKV service does not speak.
    parameters = inspect.signature(Connection.__init__).parameters
    if "protocol" in parameters:
        kwargs["protocol"] = 2
    pool = EdsBlockingConnectionPool(cluster_name=os.environ["CACHE_SERVICE"], **kwargs)
    try:
        return Redis(connection_pool=pool, decode_responses=False)
    except Exception:
        pool.disconnect()
        raise


def get_redis_client():
    global _redis_client, _retry_after
    if _redis_client is not None:
        return _redis_client
    with _redis_client_lock:
        if _redis_client is not None:
            return _redis_client
        if time.monotonic() < _retry_after:
            return None
        if not os.environ.get("CACHE_SERVICE") or not os.environ.get("EDS_HTTP_HOST"):
            logger.warning(
                "RedKV requires CACHE_SERVICE and EDS_HTTP_HOST; using HTTP fallback"
            )
            _retry_after = time.monotonic() + _RETRY_COOLDOWN
            return None
        try:
            _redis_client = _create_redis_client()
        except Exception as exc:
            # Do not expose image/cache keys, URLs, or connection credentials.
            logger.warning(
                "RedKV initialization failed (%s); using HTTP fallback",
                type(exc).__name__,
            )
            _retry_after = time.monotonic() + _RETRY_COOLDOWN
            return None
        return _redis_client


def cleanup_redis_client():
    global _redis_client, _retry_after
    with _redis_client_lock:
        client, _redis_client = _redis_client, None
        _retry_after = 0.0
        if client is not None:
            try:
                client.close()
            finally:
                # Redis does not own an explicitly supplied connection pool.
                client.connection_pool.disconnect()


def _after_fork():
    global _redis_client, _redis_client_lock, _retry_after
    # Do not acquire an inherited lock or reuse sockets owned by the parent.
    _redis_client = None
    _redis_client_lock = threading.Lock()
    _retry_after = 0.0


atexit.register(cleanup_redis_client)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _error_status(exc):
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import ResponseError
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except ImportError:
        RedisConnectionError, RedisTimeoutError, ResponseError = (
            ConnectionError,
            TimeoutError,
            (),
        )
    if isinstance(exc, (TimeoutError, RedisTimeoutError)):
        return "timeout"
    if isinstance(exc, (ConnectionError, RedisConnectionError)):
        return "connection_error"
    if isinstance(exc, ResponseError):
        return "response_error"
    return "unexpected_error"


def _valid_key(key):
    return isinstance(key, str) and bool(key.strip())


def get_bytes_from_redis_cache_with_status(key):
    if not _valid_key(key):
        return RedKVGetResult(None, "invalid_key")
    client = get_redis_client()
    if client is None:
        return RedKVGetResult(None, "client_unavailable")
    try:
        value = client.get(key)
        if value is not None and (not isinstance(value, bytes) or not value):
            return RedKVGetResult(None, "invalid_value")
        return RedKVGetResult(value)
    except Exception as exc:
        return RedKVGetResult(None, _error_status(exc))


def mget_bytes_from_redis_cache_with_status(keys):
    if not keys:
        return RedKVMgetResult({})
    if not all(_valid_key(key) for key in keys):
        return RedKVMgetResult({}, "invalid_key")
    keys = list(dict.fromkeys(keys))
    missing = dict.fromkeys(keys)
    client = get_redis_client()
    if client is None:
        return RedKVMgetResult(missing, "client_unavailable")
    try:
        values = client.mget(keys)
        if not isinstance(values, (tuple, list)) or len(values) != len(keys):
            return RedKVMgetResult(missing, "invalid_response")
        if any(
            value is not None and (not isinstance(value, bytes) or not value)
            for value in values
        ):
            return RedKVMgetResult(missing, "invalid_value")
        return RedKVMgetResult(dict(zip(keys, values)))
    except Exception as exc:
        return RedKVMgetResult(missing, _error_status(exc))


def resolve_image_bytes(source, http_fetch, record=None):
    """Resolve cache bytes, keeping the original HTTP downloader/URL policy.

    ``record(stage, result, seconds, count)`` receives bounded labels. Errors
    are counted independently of real cache misses, even though both fall back.
    """
    if not is_redkv_url(source):
        return http_fetch(source)
    started = time.perf_counter()
    result = get_bytes_from_redis_cache_with_status(source)
    outcome = "error" if result.error else "hit" if result.value is not None else "miss"
    if record is not None:
        record("redkv_get", outcome, time.perf_counter() - started, 1)
        record("redkv_cache", outcome, 0.0, 1)
    if result.value is not None:
        return result.value
    started = time.perf_counter()
    try:
        value = http_fetch(fallback_url(source))
    except Exception:
        if record is not None:
            record("redkv_fallback_http", "error", time.perf_counter() - started, 1)
        raise
    if record is not None:
        record("redkv_fallback_http", "success", time.perf_counter() - started, 1)
    return value

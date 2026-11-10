# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the actual RedKV module, without importing GPU serving APIs."""
import concurrent.futures
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/connector/redkv.py"
spec = importlib.util.spec_from_file_location("aim_redkv_under_test", SOURCE)
redkv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = redkv
spec.loader.exec_module(redkv)


class TestRedKV(unittest.TestCase):
    def setUp(self):
        redkv.cleanup_redis_client()

    def tearDown(self):
        redkv.cleanup_redis_client()

    def test_hit_miss_and_error_are_distinct(self):
        client = Mock()
        client.get.side_effect = [b"image", None, TimeoutError()]
        with patch.object(redkv, "get_redis_client", return_value=client):
            self.assertEqual(redkv.get_bytes_from_redis_cache_with_status("key").value, b"image")
            self.assertEqual(redkv.get_bytes_from_redis_cache_with_status("key"), redkv.RedKVGetResult(None))
            result = redkv.get_bytes_from_redis_cache_with_status("key")
            self.assertEqual(result.error, "timeout")

    def test_empty_and_non_binary_cache_values_are_errors(self):
        client = Mock()
        with patch.object(redkv, "get_redis_client", return_value=client):
            for value in [b"", "decoded string", 3]:
                client.get.return_value = value
                self.assertEqual(redkv.get_bytes_from_redis_cache_with_status("key").error, "invalid_value")

    def test_prefixed_key_uses_cache_and_original_https_fallback(self):
        key = "imagetransfer_https://example.test/image"
        fetch = Mock(return_value=b"http")
        with patch.object(redkv, "get_bytes_from_redis_cache_with_status", return_value=redkv.RedKVGetResult(b"cached")):
            self.assertEqual(redkv.resolve_image_bytes(key, fetch), b"cached")
            fetch.assert_not_called()
        for error in [None, "timeout", "client_unavailable"]:
            with patch.object(redkv, "get_bytes_from_redis_cache_with_status", return_value=redkv.RedKVGetResult(None, error)):
                self.assertEqual(redkv.resolve_image_bytes(key, fetch), b"http")
                fetch.assert_called_with("https://example.test/image")

    def test_normal_http_does_not_initialize_cache(self):
        with patch.object(redkv, "get_redis_client") as client:
            self.assertEqual(redkv.resolve_image_bytes("https://example.test/a", lambda _: b"ok"), b"ok")
            client.assert_not_called()
        self.assertFalse(redkv.is_redkv_url("imagetransfer_httpnot-a-url"))
        with self.assertRaises(ValueError):
            redkv.fallback_url("imagetransfer_file:///etc/passwd")

    def test_mget_deduplicates_and_checks_response_length(self):
        client = Mock()
        client.mget.return_value = [b"A", None]
        with patch.object(redkv, "get_redis_client", return_value=client):
            result = redkv.mget_bytes_from_redis_cache_with_status(["a", "b", "a"])
            client.mget.assert_called_once_with(["a", "b"])
            self.assertEqual(result.values, {"a": b"A", "b": None})
            self.assertIsNone(result.error)
            client.mget.return_value = [b"A"]
            result = redkv.mget_bytes_from_redis_cache_with_status(["a", "b"])
            self.assertEqual(result.error, "invalid_response")
            self.assertEqual(result.values, {"a": None, "b": None})

    def test_invalid_keys_do_not_connect(self):
        with patch.object(redkv, "get_redis_client") as client:
            self.assertEqual(redkv.get_bytes_from_redis_cache_with_status("").error, "invalid_key")
            self.assertEqual(redkv.mget_bytes_from_redis_cache_with_status([]).values, {})
            self.assertEqual(redkv.mget_bytes_from_redis_cache_with_status(["a", None]).error, "invalid_key")
            client.assert_not_called()

    def test_missing_eds_configuration_and_failure_cooldown(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(redkv, "_create_redis_client") as create:
            self.assertIsNone(redkv.get_redis_client())
            self.assertIsNone(redkv.get_redis_client())
            create.assert_not_called()
        redkv.cleanup_redis_client()
        with patch.dict(os.environ, {"CACHE_SERVICE": "cluster", "EDS_HTTP_HOST": "eds"}), patch.object(redkv, "_create_redis_client", side_effect=ImportError("edsredis")) as create, patch.object(redkv.time, "monotonic", return_value=100) as now:
            self.assertIsNone(redkv.get_redis_client())
            self.assertIsNone(redkv.get_redis_client())
            self.assertEqual(create.call_count, 1)
            now.return_value = 161
            self.assertIsNone(redkv.get_redis_client())
            self.assertEqual(create.call_count, 2)

    def test_client_initialization_is_thread_safe_and_cleanup_closes_pool(self):
        client = Mock()
        with patch.dict(os.environ, {"CACHE_SERVICE": "cluster", "EDS_HTTP_HOST": "eds"}), patch.object(redkv, "_create_redis_client", return_value=client) as create:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: redkv.get_redis_client(), range(20)))
            self.assertTrue(all(item is client for item in results))
            create.assert_called_once()
        redkv.cleanup_redis_client()
        client.close.assert_called_once()
        client.connection_pool.disconnect.assert_called_once()

    def test_resp2_and_bounded_connection_waits(self):
        class Connection:
            def __init__(self, protocol=3):
                pass

        fake_redis = types.SimpleNamespace(Connection=Connection, Redis=Mock())
        pool = Mock()
        fake_eds = types.SimpleNamespace(EdsBlockingConnectionPool=pool)
        with patch.dict(sys.modules, {"redis": fake_redis, "edsredis": fake_eds}), patch.dict(os.environ, {"CACHE_SERVICE": "cluster", "EDS_HTTP_HOST": "eds", "SGLANG_REDKV_PROTOCOL": "3"}):
            redkv._create_redis_client()
        self.assertEqual(pool.call_args.kwargs["protocol"], 2)
        self.assertEqual(pool.call_args.kwargs["cluster_name"], "cluster")
        self.assertGreater(pool.call_args.kwargs["socket_timeout"], 0)
        self.assertGreater(pool.call_args.kwargs["timeout"], 0)
        self.assertGreater(pool.call_args.kwargs["max_connections"], 0)
        self.assertFalse(fake_redis.Redis.call_args.kwargs["decode_responses"])


if __name__ == "__main__":
    unittest.main()

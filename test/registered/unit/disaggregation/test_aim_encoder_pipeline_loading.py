"""CPU-only integration of the production Encoder image batching method."""

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(os.environ.get(
    "AIM_ENCODER_TEST_SOURCE",
    str(ROOT / "python/sglang/srt/disaggregation/encoder/preprocessor.py"),
))
spec = importlib.util.spec_from_file_location(
    "aim_image_loader", ROOT / "python/sglang/srt/multimodal/aim_image_loader.py"
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
tree = ast.parse(SOURCE.read_text())
encoder = next(n for n in tree.body if getattr(n, "name", None) == "EncoderPreprocessor")
method = next(n for n in encoder.body if getattr(n, "name", None) == "_flatten_and_load_images")
flags = {
    "SGLANG_ENABLE_REDKV_MGET": True,
    "SGLANG_ENABLE_BATCH_JPEG_DECODE": False,
    "SGLANG_ENABLE_JPEG_GPU_DECODE": True,
    "SGLANG_JPEG_CPU_DECODE_WORKERS": 1,
    "SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": False,
}
envs = SimpleNamespace(**{
    name: SimpleNamespace(get=lambda name=name: flags[name]) for name in flags
})
namespace = dict(globals(), Modality=SimpleNamespace(IMAGE="image"))
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
load_encoder_images = namespace["_flatten_and_load_images"]


class TestEncoderBatchImageLoading(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.io_pool = ThreadPoolExecutor(max_workers=1)
        self.preproc_pool = ThreadPoolExecutor(max_workers=1)
        self.loads = []
        self.mgets = []

        def mget(keys):
            self.mgets.append(keys)
            return SimpleNamespace(values={keys[0]: b"hit"}, error=None)

        redkv = ModuleType("sglang.srt.connector.redkv")
        redkv.is_redkv_url = lambda value: value.startswith("imagetransfer_http")
        redkv.fallback_url = lambda value: value.removeprefix("imagetransfer_")
        redkv.mget_bytes_from_redis_cache_with_status = mget
        image_policy = ModuleType("sglang.srt.multimodal.aim_image")
        image_policy.image_load_error_fallback = lambda error, **_: (_ for _ in ()).throw(error)
        self.modules = patch.dict(sys.modules, {
            "sglang.srt.multimodal.aim_image_loader": loader,
            "sglang.srt.connector.redkv": redkv,
            "sglang.srt.multimodal.aim_image": image_policy,
        })
        self.modules.start()

        def load_one(value, **_):
            self.loads.append(value)
            return value

        async def original(value, _):
            return ("original-contract", value)

        self.encoder = SimpleNamespace(
            io_executor=self.io_pool,
            preproc_executor=self.preproc_pool,
            _load_single_item=load_one,
            _flatten_and_load_data_by_modality=original,
        )

    def tearDown(self):
        self.modules.stop()
        self.io_pool.shutdown()
        self.preproc_pool.shutdown()

    async def test_nested_order_duplicate_key_and_single_worker_pools(self):
        hit = "imagetransfer_http://images/hit"
        miss = "imagetransfer_http://images/miss"
        result = await asyncio.wait_for(
            load_encoder_images(self.encoder, [[hit, "regular"], [miss, hit]]), 2
        )
        self.assertEqual(result, [[b"hit", "regular"], ["http://images/miss", b"hit"]])
        self.assertEqual(self.mgets, [[hit, miss]])
        self.assertEqual(len(self.loads), 4)

    async def test_scalar_and_empty_shape(self):
        self.assertEqual(await load_encoder_images(self.encoder, "regular"), "regular")
        self.assertEqual(await load_encoder_images(self.encoder, []), [])

    async def test_content_hash_dict_uses_existing_validation(self):
        data = [{"url": "imagetransfer_http://images/hit", "content_hash": "digest"}]
        result = await load_encoder_images(self.encoder, data)
        self.assertEqual(result, ("original-contract", data))
        self.assertEqual(self.mgets, [])
        self.assertEqual(self.loads, [])

    async def test_cancellation_signals_batch_loader(self):
        started = threading.Event()
        observed_cancel = threading.Event()

        def blocked_loader(_, *, cancel_event, **kwargs):
            started.set()
            if cancel_event.wait(2):
                observed_cancel.set()
            return ["unused"]

        with patch.object(loader, "load_images_batch", blocked_loader):
            task = asyncio.create_task(load_encoder_images(self.encoder, ["regular"]))
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(await asyncio.to_thread(observed_cancel.wait, 1))


if __name__ == "__main__":
    unittest.main()

"""Exercise actual split Encoder health/dispatch boundaries without GPU faults."""
import asyncio
import importlib.util
import io
import os
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from PIL import Image
from requests.exceptions import HTTPError

from sglang.srt.disaggregation.encoder import http_server as http
from sglang.srt.disaggregation.encoder import runtime, server
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal import aim_cuda_health as health
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import common

SUPPORT_PATH = Path(__file__).with_name('test_aim_encoder_pipeline.py')
spec = importlib.util.spec_from_file_location('aim_health_pipeline_support', SUPPORT_PATH)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class TestSplitCudaHealth(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        publish(ServerArgs(model_path='dummy'), role='encoder')
        state = patch.object(health, '_state', health.CudaFailureState())
        state.start()
        self.addCleanup(state.stop)
        env = patch.dict(os.environ, {
            'SGLANG_EXIT_ON_CUDA_FATAL': 'false',
            'SGLANG_ENCODER_EXIT_ON_CUDA_FATAL': 'false',
            'SGLANG_ENABLE_JPEG_GPU_DECODE': 'true',
            'SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE': 'true',
        })
        env.start()
        self.addCleanup(env.stop)

    def busy_encoder(self):
        encoder = SimpleNamespace(
            supports_modality=lambda modality: modality == Modality.IMAGE,
            has_pending_embeddings=Mock(return_value=True),
            encode_dispatch_lock=asyncio.Lock(),
            encode=AsyncMock(),
        )
        for key, value in [('encoder', encoder), ('dp_dispatcher', None), ('send_sockets', [])]:
            mocked = patch.object(http, key, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        return encoder

    async def test_fatal_marker_precedes_busy_liveness_shortcut(self):
        encoder = self.busy_encoder()
        self.assertEqual((await http.health_generate()).status_code, 200)
        encoder.has_pending_embeddings.reset_mock()
        health.record_cuda_failure(RuntimeError('CUDA error: illegal memory access'))
        self.assertEqual((await http.health_generate()).status_code, 503)
        encoder.has_pending_embeddings.assert_not_called()
        encoder.encode.assert_not_called()

    async def test_health_rechecks_failure_after_waiting_for_dispatch_lock(self):
        encoder = self.busy_encoder()
        await encoder.encode_dispatch_lock.acquire()
        task = asyncio.create_task(http.health_generate())
        await asyncio.sleep(0)
        health.record_cuda_failure(RuntimeError('misaligned address'))
        encoder.encode_dispatch_lock.release()
        self.assertEqual((await task).status_code, 503)
        encoder.encode.assert_not_called()
        self.assertFalse(encoder.encode_dispatch_lock.locked())

    async def test_failed_http_encode_is_rejected_before_enqueue(self):
        health.record_cuda_failure(RuntimeError('CUDA error 700'))
        self.assertEqual((await http.handle_encode_request({})).status_code, 503)

    async def test_shared_pipeline_and_dp_health_reject_poisoned_context(self):
        health.record_cuda_failure(RuntimeError('CUDA error 717'))
        for call in [lambda: runtime.execute_encode_pipeline(None, None, {}),
                     lambda: runtime._dp_worker_health_encode(None)]:
            with self.assertRaises(server.MMError) as caught:
                await call()
            self.assertEqual(caught.exception.code, 503)

    async def test_preparation_preserves_native_fatal_exception_chain(self):
        encoder = server.MMEncoder.__new__(server.MMEncoder)
        fatal = RuntimeError('CUDA error: an illegal memory access was encountered')
        encoder.preprocessor = SimpleNamespace(process_batch_mm_items=AsyncMock(side_effect=fatal))
        with self.assertRaises(server.InternalError) as caught:
            await encoder._prepare_encode_context([], Modality.IMAGE, use_global_cache=False)
        self.assertIs(caught.exception.__cause__, fatal)
        self.assertTrue(health.cuda_failure_detected())

    async def test_staged_errors_record_fatal_wrapped_by_client_error(self):
        encoder = server.MMEncoder.__new__(server.MMEncoder)
        encoder.rank = 0
        staged = []
        encoder._stage_embedding = staged.append
        error = server.BadRequestError('wrapped preprocessing failure')
        error.__cause__ = RuntimeError('CUDA error: device-side assert triggered')
        result = encoder._stage_errors(
            [{'req_id': 'failed', 'num_parts': 1, 'part_idx': 0}], Modality.IMAGE, error)
        self.assertTrue(health.cuda_failure_detected())
        self.assertEqual(result[0][4], 500)
        self.assertEqual(staged[0].error_code, 500)

    async def test_real_gpu_decode_boundary_records_fault_without_1px_fallback(self):
        data = io.BytesIO()
        Image.new('RGB', (3, 4)).save(data, format='JPEG')
        failure = RuntimeError('CUDA error: illegal memory access')
        with patch.object(common, 'is_cuda', return_value=True), \
             patch('torchvision.io.decode_jpeg', side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                common.load_image(data.getvalue(), gpu_image_decode=True)
        self.assertIs(caught.exception, failure)
        self.assertTrue(health.cuda_failure_detected())

    async def test_known_failure_prevents_next_gpu_decode(self):
        health.record_cuda_failure(RuntimeError('misaligned address'))
        with patch.object(common, 'is_cuda', return_value=True), \
             patch('torchvision.io.decode_jpeg') as decode:
            with self.assertRaisesRegex(RuntimeError, 'unhealthy'):
                common.load_image(b'\xff\xd8\xff\xd9', gpu_image_decode=True)
        decode.assert_not_called()

    async def test_url_error_and_oom_do_not_poison_health(self):
        encoder = self.busy_encoder()
        with patch.object(common, 'download_remote_media', side_effect=HTTPError(
                'https://images.example/illegal memory access')):
            with self.assertRaises(HTTPError):
                common.load_image('https://images.example/not-found')
        self.assertFalse(health.record_cuda_failure(torch.OutOfMemoryError('CUDA out of memory')))
        self.assertFalse(health.cuda_failure_detected())
        self.assertEqual((await http.health_generate()).status_code, 200)
        encoder.encode.assert_not_called()

    async def test_dispatch_owner_rechecks_health_under_lock_before_broadcast(self):
        ns = support.load_scheduler()
        ns['cuda_failure_detected'] = health.cuda_failure_detected
        encoder = support.FakeEncoder()
        self.addCleanup(encoder._aim_preparation.close)
        wire = []
        scheduler = ns['EncoderScheduler'](encoder, [wire], 1)
        pending = ns['PendingRequest'](support.request(0), asyncio.get_running_loop())
        await encoder.encode_dispatch_lock.acquire()
        task = asyncio.create_task(scheduler._dispatch_group([pending], support.Modality.IMAGE))
        await asyncio.sleep(0)
        health.record_cuda_failure(RuntimeError('illegal memory access'))
        encoder.encode_dispatch_lock.release()
        await task
        self.assertEqual(pending.future.exception().code, 503)
        self.assertEqual(encoder.forwarded, [])
        self.assertNotIn('batch_encode', [message['type'] for message in wire])

    async def test_dispatch_rechecks_health_after_cpu_preparation(self):
        ns = support.load_scheduler()
        ns['cuda_failure_detected'] = health.cuda_failure_detected
        started, release = threading.Event(), threading.Event()
        async def prepare(requests):
            started.set()
            while not release.is_set():
                await asyncio.sleep(.001)
        encoder = support.FakeEncoder(prepare)
        self.addCleanup(encoder._aim_preparation.close)
        wire = []
        scheduler = ns['EncoderScheduler'](encoder, [wire], 1)
        request = support.request(0)
        encoder.prepare_batch('ticket', [request], support.Modality.IMAGE)
        pending = ns['PendingRequest'](request, asyncio.get_running_loop())
        task = asyncio.create_task(scheduler._dispatch_group([pending], support.Modality.IMAGE, 'ticket'))
        await support.eventually(started.is_set)
        health.record_cuda_failure(RuntimeError('device-side assert triggered'))
        release.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(pending.future.exception().code, 503)
        self.assertEqual(encoder.forwarded, [])
        self.assertNotIn('batch_encode', [message['type'] for message in wire])

    async def test_grpc_health_reports_not_serving_after_failure(self):
        from sglang.srt.disaggregation.encoder.grpc_server import EncoderHealthServicer, health_pb2
        servicer = EncoderHealthServicer()
        servicer.set_serving()
        self.assertEqual((await servicer.Check(None, None)).status, health_pb2.HealthCheckResponse.SERVING)
        health.record_cuda_failure(RuntimeError("illegal memory access"))
        self.assertEqual((await servicer.Check(None, None)).status, health_pb2.HealthCheckResponse.NOT_SERVING)

    async def test_grpc_encode_rechecks_after_waiting_for_dispatch_lock(self):
        import grpc
        from sglang.srt.disaggregation.encoder.grpc_server import SGLangEncoderServer
        servicer = SGLangEncoderServer.__new__(SGLangEncoderServer)
        servicer.encoder = SimpleNamespace(encode_dispatch_lock=asyncio.Lock(), release_request=AsyncMock())
        servicer._dispatch_encode = AsyncMock()
        request = SimpleNamespace(mm_items=["unused"], req_id="grpc-fatal", num_parts=1, part_idx=0)
        context = Mock()
        await servicer.encoder.encode_dispatch_lock.acquire()
        task = asyncio.create_task(servicer.Encode(request, context))
        await asyncio.sleep(0)
        health.record_cuda_failure(RuntimeError("misaligned address"))
        servicer.encoder.encode_dispatch_lock.release()
        await task
        context.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)
        servicer._dispatch_encode.assert_not_called()

"""CPU checks for the real pre-launch accounting methods and failure paths."""
import ast
import copy
import logging
import os
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('SGLANG_LOAD_TRACKER_TEST_SOURCE', str(ROOT/'python/sglang/srt'))
SOURCE = Path(os.environ['SGLANG_LOAD_TRACKER_TEST_SOURCE'])


def method(file, cls, name, env):
    tree = ast.parse((SOURCE / file).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(env)
    exec(compile(module, str(SOURCE / file), 'exec'), namespace)
    return namespace[name]


METHODS = [
    '_update_prefill_scheduled_chunk', '_begin_prefill_gpu_pending',
    '_record_prefill_gpu_pending', '_cancel_prefill_load_ticket',
    '_run_prefill_batch_with_load', 'get_prefill_gpu_pending_tokens',
    'get_prefill_scheduled_chunk_end',
]


def harness(enabled=True):
    obj = NS(chunked_req=None, snapshots=[])
    env = {'envs': NS(SGLANG_PREFILL_GPU_PENDING_LOAD=NS(get=lambda: enabled)),
           'SimpleNamespace': NS, 'logger': logging.getLogger(__name__)}
    for name in METHODS:
        fn = method('disaggregation/prefill.py',
                              'SchedulerDisaggregationPrefillMixin', name, env)
        setattr(obj, name, MethodType(fn, obj))

    def publish(**kwargs):
        obj.snapshots.append((obj.get_prefill_gpu_pending_tokens(),
                              obj.get_prefill_scheduled_chunk_end()))
    obj.publish_load_snapshot = publish
    return obj


def batch(rid='a', end=32768, tokens=32768):
    req = NS(rid=rid, extend_range=NS(end=end))
    return NS(reqs=[req], forward_mode=NS(is_extend=lambda: True),
              is_dllm=lambda: False, extend_num_tokens=tokens)


def event(ready=False):
    return NS(query=lambda: ready)


class PrefillPrelaunchLoadTest(unittest.TestCase):
    def test_prelaunch_is_visible_before_run_batch(self):
        obj, work = harness(), batch()
        obj.chunked_req = work.reqs[0]
        result = NS(copy_done=event())

        def run(work):
            self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 32768)
            self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 32768)
            self.assertEqual(obj.snapshots, [(32768, 32768)])
            return result

        obj.run_batch = run
        self.assertIs(obj._run_prefill_batch_with_load(work), result)
        self.assertEqual(obj.snapshots, [(32768, 32768), (32768, 32768)])
        self.assertEqual(list(obj._prefill_active_token_batches), [id(result)])

    def test_none_event_inflight_event_and_ready_event_are_distinct(self):
        obj, work = harness(), batch()
        obj.chunked_req = work.reqs[0]
        ticket = obj._begin_prefill_gpu_pending(work)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 32768)
        self.assertIsNone(obj._prefill_active_token_batches[ticket.key][1])
        ready = [False]
        result = NS(copy_done=NS(query=lambda: ready[0]))
        obj._record_prefill_gpu_pending(work, result, ticket)
        self.assertEqual(len(obj._prefill_active_token_batches), 1)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 32768)
        ready[0] = True
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 0)
        self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 32768)
        obj._prefill_active_token_batches.pop(id(result))
        self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 32768)

    def test_reused_batch_object_gets_independent_plan_tickets(self):
        obj, work = harness(), batch()
        obj.chunked_req = work.reqs[0]
        first = obj._begin_prefill_gpu_pending(work)
        second = obj._begin_prefill_gpu_pending(work)
        self.assertIsNot(first.key, second.key)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 65536)
        result = NS(copy_done=event())
        obj._record_prefill_gpu_pending(work, result, first)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 65536)
        self.assertEqual(len(obj._prefill_active_token_batches), 2)

    def test_launch_exception_restores_previous_chunk_and_other_results(self):
        obj, old = harness(), batch(end=32768)
        obj.chunked_req = old.reqs[0]
        result = NS(copy_done=event())
        obj._record_prefill_gpu_pending(old, result)
        new = batch(end=65536)
        obj.chunked_req = new.reqs[0]
        original = RuntimeError('launch failed')

        def fail(_):
            self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 65536)
            self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 65536)
            raise original

        obj.run_batch = fail
        with self.assertRaises(RuntimeError) as caught:
            obj._run_prefill_batch_with_load(new)
        self.assertIs(caught.exception, original)
        self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 32768)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 32768)
        self.assertEqual(list(obj._prefill_active_token_batches), [id(result)])
        self.assertEqual(obj.snapshots[-1], (32768, 32768))

    def test_begin_publish_exception_does_not_leave_reservation(self):
        obj, work = harness(), batch()
        obj.chunked_req = work.reqs[0]
        obj.publish_load_snapshot = lambda **kwargs: (_ for _ in ()).throw(ValueError('publish'))
        with self.assertRaises(ValueError):
            obj._begin_prefill_gpu_pending(work)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 0)
        self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 0)
        self.assertEqual(obj._prefill_active_token_batches, {})

    def test_bind_publish_exception_also_removes_result_key(self):
        obj, work = harness(), batch()
        obj.chunked_req = work.reqs[0]
        count = [0]

        def publish(**kwargs):
            count[0] += 1
            if count[0] == 2:
                raise ValueError('bind publication')

        obj.publish_load_snapshot = publish
        obj.run_batch = lambda _: NS(copy_done=event())
        with self.assertRaises(ValueError):
            obj._run_prefill_batch_with_load(work)
        self.assertEqual(obj.get_prefill_gpu_pending_tokens(), 0)
        self.assertEqual(obj.get_prefill_scheduled_chunk_end(), 0)
        self.assertEqual(obj._prefill_active_token_batches, {})

    def test_disabled_observation_does_not_publish_or_register(self):
        obj, work = harness(False), batch()
        result = NS(copy_done=event())
        obj.run_batch = lambda _: result
        self.assertIs(obj._run_prefill_batch_with_load(work), result)
        self.assertEqual(obj.snapshots, [])
        self.assertFalse(hasattr(obj, '_prefill_active_token_batches'))

    def test_nonextend_and_dllm_are_not_registered(self):
        obj, work = harness(), batch()
        work.is_dllm = lambda: True
        self.assertIsNone(obj._begin_prefill_gpu_pending(work))
        work.is_dllm = lambda: False
        work.forward_mode.is_extend = lambda: False
        self.assertIsNone(obj._begin_prefill_gpu_pending(work))
        self.assertEqual(obj.snapshots, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)

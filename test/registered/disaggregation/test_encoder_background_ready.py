import importlib.util,json,time,threading,unittest,sys
from pathlib import Path
from types import SimpleNamespace
import zmq
from sglang.srt.disaggregation.encoder import background_receiver as bg
class Waiting:
 def __init__(self,req,port):self.recv_req=req;self.status=0;self.error_msg=None;self.error_code=None
 def consume_parts(self,frames,decoded=None):
  if decoded.get('block'): gate.wait(2)
  if decoded.get('error'):self.status=-1;self.error_msg=decoded['error'];self.error_code=400;return
  self.recv_req.mm_inputs={'ids':self.recv_req.input_ids,'thread':threading.current_thread().name}
  self.status=1
class TestBackground(unittest.TestCase):
 def setUp(self):
  global gate;gate=threading.Event()
  self.w=bg.BackgroundReceiver('127.0.0.1',Waiting,lambda x:json.loads(bytes(x)),lambda x:x['rid'],timeout=.15,max_bytes=1024)
  self.ctx=zmq.Context();self.c=self.ctx.socket(zmq.PUSH);self.e=self.ctx.socket(zmq.PUSH)
  self.c.connect(f'tcp://127.0.0.1:{self.w.control_port}');self.e.connect(f'tcp://127.0.0.1:{self.w.embedding_port}')
 def tearDown(self):
  gate.set();self.w.close();self.c.close(0);self.e.close(0);self.ctx.term()
 def prepare(self,rid,epoch=None):self.c.send_json({'rid':rid,'epoch':epoch or self.w.epoch,'prompt':{'text':'x','ids':[1,2],'pretokenized':True}})
 def part(self,rid,**kw):self.e.send_multipart([json.dumps({'rid':rid,**kw}).encode(),b'abcd'])
 def result(self,rid):
  stop=time.monotonic()+2
  while time.monotonic()<stop:
   r=self.w.snapshot(rid)
   if r:return r
   time.sleep(.003)
  self.fail('timeout')
 def test_embedding_before_prompt(self):
  self.part('a');time.sleep(.01);self.prepare('a');r=self.result('a');self.assertEqual(r.status,1);self.assertEqual(r.request.mm_inputs['ids'],[1,2]);self.assertNotEqual(r.request.mm_inputs['thread'],threading.current_thread().name)
 def test_prompt_before_embedding(self):
  self.prepare('b');self.part('b');self.assertEqual(self.result('b').status,1)
 def test_pending_assembly_does_not_block_scheduler_snapshot(self):
  self.prepare('c');self.part('c',block=True);time.sleep(.02);t=time.monotonic();self.assertIsNone(self.w.snapshot('c'));self.assertLess(time.monotonic()-t,.02);gate.set();self.assertEqual(self.result('c').status,1)
 def test_retire_discards_late_parts(self):
  self.prepare('d');self.w.discard('d');time.sleep(.02);self.part('d');time.sleep(.02);self.assertIsNone(self.w.snapshot('d'));self.assertNotIn('d',self.w.records)
 def test_bad_epoch(self):
  self.prepare('e','stale');self.assertEqual(self.result('e').code,503)
 def test_byte_limit(self):
  self.prepare('f');self.part('f',padding='x'*1024);self.assertEqual(self.result('f').code,503)
 def test_error_propagation(self):
  self.prepare('g');self.part('g',error='broken image');self.assertEqual(self.result('g').error,'broken image')
 def test_timeout_and_cleanup(self):
  self.prepare('h');self.assertEqual(self.result('h').code,408);time.sleep(.35);self.assertNotIn('h',self.w.records);self.assertEqual(self.w.bytes,0)
 def test_terminal_result_not_mutated_by_duplicate(self):
  self.prepare('i');self.part('i');r=self.result('i');self.part('i',error='late');time.sleep(.02);self.assertIs(self.w.snapshot('i'),r);self.assertEqual(r.status,1)
if __name__=='__main__':unittest.main()

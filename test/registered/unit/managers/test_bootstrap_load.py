from types import SimpleNamespace as NS
import unittest
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.disaggregation.utils import DisaggregationMode

def request(n, matched=0):return NS(seqlen=n,num_matched_prefix_tokens=matched,prefix_indices=list(range(matched)))
def load(waiting=(),bootstrap=(),chunk=None,matched=True,hit=.5,mode=DisaggregationMode.PREFILL):
 obj=NS(disaggregation_mode=mode,get_waiting_queue=lambda:waiting,waiting_queue_prefix_matched=lambda:matched,get_recent_cache_hit_rate=lambda:hit,get_chunked_req=lambda:chunk,get_disagg_prefill_bootstrap_queue=lambda:NS(queue=bootstrap))
 return SchedulerLoadInquirer.get_num_waiting_uncached_tokens(obj)
class BootstrapLoad(unittest.TestCase):
 def test_growing_bootstrap_increases_score_with_same_waiting_queue(self):
  wait=[request(100,40)];a=request(30);b=request(20)
  self.assertEqual(load(wait,[a]),90);self.assertEqual(load(wait,[a,b]),110)
 def test_transition_to_matched_waiting_does_not_double_count(self):
  a=request(100,40)
  self.assertEqual(load([], [a]),100)
  self.assertEqual(load([a],[a]),60)
  self.assertEqual(load([a],[]),60)
 def test_chunk_counts_once_if_still_in_bootstrap(self):
  a=request(100,40)
  self.assertEqual(load([], [a], a),60)
 def test_duplicate_bootstrap_entry_counts_once(self):
  a=request(30);self.assertEqual(load([], [a,a]),30)
 def test_unmatched_waiting_keeps_existing_cache_estimator(self):
  self.assertEqual(load([request(100)], [request(20)], matched=False,hit=.75),45)
 def test_bootstrap_does_not_use_unprotected_prefix_match(self):
  self.assertEqual(load([], [request(100,90)]),100)
 def test_decode_and_non_disagg_paths_unchanged(self):
  self.assertEqual(load([request(100)], [request(20)],mode=DisaggregationMode.DECODE),0)
  self.assertEqual(load([request(100,40)], [request(20)],mode=DisaggregationMode.NULL),60)
 def test_empty_and_negative_bootstrap_length(self):
  self.assertEqual(load(),0);self.assertEqual(load([], [request(-1)]),0)
if __name__=='__main__':unittest.main()

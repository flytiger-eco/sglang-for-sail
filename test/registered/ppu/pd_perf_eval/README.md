# PPU disaggregated serving performance — 1p1d pilot

This directory holds the prefill/decode-disaggregated (PD) serving-performance
entry for PPU CI, together with the versioned execution contract it reads. The
evaluator lives in `python/sglang/test/kits/pd_perf_eval_kit.py` and the
on-machine driver in `python/sglang/test/kits/pd_perf_suite_kit.py`.

```
pd_perf_eval/
├── configs/<model family>/<model config>.json   reviewed execution contracts
└── test_ppu_*.py                                 registered entry points
```

It is a sibling of `perf_eval/` rather than a subdirectory of it, and reads its
own environment variables (`SGLANG_PPU_PD_PERF_TEST_CONFIG`,
`SGLANG_PPU_PD_PERF_RESULTS_DIR`), for the reason `perf_eval/` is separate from
`answer_eval/`: a change made for one line must not be able to move the other's
config. What the two performance lines *do* share is the report — a PD run emits
the same `ppu-perf-report/v1` a colocated run does, which is why
`collect_perf_evidence.sh` needs no PD-specific variant — and the hardware,
model and workload halves of the config contract, which `pd_perf_eval_kit`
validates through `perf_eval_kit.validate_hardware`, `validate_model` and
`validate_workload` rather than through a second copy of those rules.

**Four configs across two suites have now measured green**, each producing a full
set of numbers on two boards; the results are under
[What the dispatches established](#what-the-dispatches-established). The workflows
are still on no schedule and wired into no nightly caller — they are dispatched by
hand — but the measurement path itself is now proven end to end.

## What runs on the two boards

A PD group is one K8s job of two gang-scheduled pods, each holding a whole
ZW-M890P board, each running `scripts/ci/ppu/run_pd_perf_suite_node.sh`. Three
server-side processes exist across them:

| Process | Node | Port | Role |
| --- | --- | --- | --- |
| prefill server | rank 0 | 21000 | `--disaggregation-mode prefill`, tp 8 |
| decode server | rank 1 | 21001 | `--disaggregation-mode decode`, tp 8, dp 8 |
| mini-lb router | rank 0 | 12345 | what `bench_serving` posts to |

Which role a rank serves is the config's answer to the rank the action injects
(`prefill_nodes` ranks first, then `decode_nodes`), resolved in
`pd_perf_eval_kit.pd_role_for_node_rank`. Rank 0 additionally owns the router,
the benchmark and the report; rank 1 serves its half of the KV path and holds its
board until rank 0 publishes completion.

### The two nodes exchange endpoints, not a rendezvous

The multi-node colocated entries derive a rendezvous address because their ranks
join one process group. A PD group has nothing to meet at: each server is a rank
0 of its own eight devices, and what crosses the node boundary is a KV transfer
over Mooncake plus the router's HTTP. So each node instead publishes its own
`http://<addr>:<port>` into `ranks/rank-N-endpoint.json` on the shared results
directory — a staged write followed by a rename, as with the device inventories
— and rank 0 reads the peers' files back.

The address is `get_local_ip_auto()` rather than the injected `MASTER_ADDR`,
because these pods run with host networking and `ClusterFirst` DNS, where the
injected name is not resolvable.

The publish happens **immediately after the server process is created, before its
`/health` is awaited**. That ordering is the whole anti-deadlock argument: if a
node published only once healthy, and both nodes' health depended on the peer
(the decode server's KV handshake does), neither would ever publish. Rank 0 then
waits in a fixed order — peer endpoint files, its own `/health`, each peer's
`/health`, then the router's — so a peer that never comes up fails as a peer-wait
timeout rather than as an unexplained hang.

Because the shared directory is the transport for all of this, the suite refuses
a relative `SGLANG_PPU_PD_PERF_RESULTS_DIR`: the two pods must name the same
bytes.

## What this line measures, and what it does not

Same contract as the colocated line, and the same refusal to judge: a
measurement is red only when it produced no usable numbers at all, per
`perf_eval_kit.REASON_CODES`. A slow server is never red. The nineteen fields of
`perf_eval_kit.METRIC_FIELDS` are recorded for every measurement, reaching the
run page as one annotation per measurement with the machine-readable copy in the
artifact.

### TPOT and ITL are recorded here, because this case decodes

Unlike the prefill line — where `output_len` is 1 and decode-side metrics would
be degenerate — this case decodes 1500 tokens per request, and the source case's
metric block asks for `TPOT_AVG` and its percentiles. `bench_serving` emits
`mean_tpot_ms`, `median_tpot_ms`, `std_tpot_ms`, `p99_tpot_ms` and the ITL
family, and the shared `perf_eval_kit.DECODE_METRIC_FIELDS` records all nine —
but only when a measurement's `output_len` is above one. A single-token prefill
measurement still omits them rather than record the zero an empty `tpots` list
produces, so extending the schema for this line changed no colocated report: the
colocated configs decode one token and take the prefill set unchanged, while
this one decodes 1500 and takes the decode set as well. End-to-end p90 and the
output-throughput peak, defined from the first token, were folded into the
always-recorded set at the same time.

`TTFT_P90` is omitted for the reason the colocated line records: the source
metric block reads a `tp_90` percentile `bench_serving` does not emit, and this
line does not change `bench_serving`. The p99 tail stands in its place.

### The recorded pass is the second, by construction

The source harness reports a second pass over a first: it warms the server, runs
the workload once, and reports the run after that. This line matches it with
`workload.warmup_passes`, which the shared kit resolves into each measurement
and the disaggregated suite honours by running that many full passes at the
measurement's own shape and discarding them before the recorded one. These
cases set one; the colocated line's default is zero. The KV cache is flushed
after each warmup pass, so the recorded pass runs warm on compilation and cold on
cache — which is the point: run 34317611899 measured a first 4k prefill at 23.6
tok/s against a second at 1495 tok/s, a ~170s per-batch-shape compilation cost
that `warmup_requests` (capped at 32 output tokens) cannot reach and that
otherwise lands entirely in the reported TTFT tail. The discarded pass's raw
output is kept under a `-warmupN` name next to the recorded one as evidence it
ran.

## The suites

| Suite | Test file | Model | Configs | Topology | Boards |
| --- | --- | --- | --- | --- | --- |
| `nightly-pd-perf-16-glm52-ppu` | `test_ppu_glm52_pd_perf.py` | GLM-5.2 | fp8-channelwise, mxfp4-fp8 | 1p1d | 2 |
| `nightly-pd-perf-16-qwen35-ppu` | `test_ppu_qwen35_pd_perf.py` | Qwen3.5-397B-A17B | fp8-channelwise, mxfp4-fp8 | 1p1d | 2 |

One suite per model, two reviewed configs behind each, one measurement per config:
a job names the config in `SGLANG_PPU_PD_PERF_TEST_CONFIG` and the class falls back
to its default when unset. The device count leads each name as it does on the
colocated line, because it is the scheduling fact a reader needs first: a dispatch
of either suite asks the cluster for sixteen devices across two whole boards.

## Alignment with the source cases

The fp8-channelwise config renders the two `Chapter 2` launch commands of
`server_cmds/LLM_Serving/BTV1.5/GLM-5.2.md` under `fp8-channel` /
`4096-4096/1500-1500`, and takes its workload from
`testcases/btv1.5/.../PD-Disaggregation/notune/glm-5_2_fp8-channel_1p1d_4096_1500_0001.json`
(`TC_id` `glm-5.2_0004`, `TC_name` `glm-5.2_pd_notune_4096_1500_mtp`). Every
reviewed server parameter of both roles comes from those two commands; the
`pd_disaggregation_dist` block of the case file (`num_prefill_worker` 1,
`num_decode_worker` 1, both world size 1) is what the config's `prefill_nodes`
and `decode_nodes` state.

### Departures, and why

- **`SGLANG_WARMUP_TIMEOUT=3600` in both roles' `env`.** Not in the source case.
  It is the colocated line's precedent, for the same reason: a cold checkpoint
  read off the NAS can exceed the default warmup budget, and the failure it
  prevents is an infrastructure one, not a property of the model.
- **`--disaggregation-transfer-backend mooncake` and
  `--disaggregation-bootstrap-port 8998` are rendered explicitly.** The red-zone
  commands state neither. Both values equal `ServerArgs`' defaults
  (`server_args.py`: `disaggregation_transfer_backend = "mooncake"`,
  `disaggregation_bootstrap_port = 8998`), so behaviour is unchanged; what
  changes is that they are now reviewed rather than inherited. The bootstrap port
  in particular is *load-bearing and invisible* in the source: the router command
  passes no bootstrap port, so the decode server can only find the prefill
  server's at the default. The schema therefore pins it to 8998 as a hard check —
  a config that changed it would break the router silently.
- **`random-ids` with `--tokenize-prompt`, not the source's length
  distribution.** The case declares `min_input_len == max_input_len == 4096` and
  the same for 1500 output; `random-ids` at `random_range_ratio` 1.0 is how
  `bench_serving` pins both, and the report records the observed length summary
  so a drift is visible rather than assumed. Same substitution the colocated line
  makes.
- **`--log-level info` is not in the schema.** It appears on both red-zone
  commands and is SGLang's default, so rendering it would add a parameter to the
  contract that changes nothing.
- **`tc_name` says `mtp`, and this config has no speculative decoding.** That is
  the source case's own name (`glm-5.2_pd_notune_4096_1500_mtp`) and is recorded
  verbatim for traceability. Neither of its launch commands carries any
  `--speculative-*` flag — the MTP-bearing variant is the MXFP4 one, ported
  alongside it (below). The name is inherited, not a description of what runs.
- **The KV-cache flush goes to each server directly, not through the router.**
  `args.flush_cache` is `False` and the suite POSTs `/flush_cache` to both
  endpoints itself, asserting the status. Both router implementations in this
  tree fan `/flush_cache` out to their workers, but whether `--mini-lb` proxies
  admin routes at all is not established here, and a flush that silently went
  nowhere would be worse than no flush. With one measurement in the plan this
  only affects the warmup request, and both roles run
  `--disable-radix-cache` anyway.
- **`MC_LOG_LEVEL=TRACE` is passed through as the source sets it.** It is the
  Mooncake transfer engine at its most verbose on both nodes for the whole run.
  It is kept because the first dispatch is diagnostic and this is the KV path's
  only voice; if a later run is drowned by it, the level belongs in the config,
  which is why it is a reviewed key rather than a hard-coded export.
- **The servers listen on 21000 and 21001, not the source's 30000 and 40000.**
  The only departure here a run actually forced: 30000 is where Kubernetes starts
  handing out node ports, and on these nodes traffic to `<node ip>:30000` is
  rewritten before it reaches a socket bound to exactly that address — measured
  twice, in the dispatches below. 40000 sits in the ephemeral range, where an
  outbound connection can take the port first; nothing has been seen to, but it
  is the same class of problem and moving one port without the other would leave
  it standing. The red-zone commands are not wrong to use either, since they do
  not run under Kubernetes. `pd_perf_eval_kit.PD_HIGHEST_BINDABLE_PORT` holds the
  bound so a config written by copying a red-zone command is refused by the
  schema rather than by two boards eight minutes in.

### The three MTP configs alongside it

The other three configs decode with Multi-Token-Prediction, which the
fp8-channelwise case above does not, and are otherwise the same 1p1d shape and
workload:

- **GLM-5.2 mxfp4-fp8.** The source names `speculative_algorithm EAGLE` on both
  roles with an empty `speculative_draft_model_path`, which SGLang resolves to
  EAGLE over the base checkpoint's own MTP layers; the config omits the empty draft
  key, since an absent and an empty draft path both fall back to the model path.
  Its weights are on the NAS this suite reads (`model_weight_path.csv`).
- **Qwen3.5-397B-A17B fp8-channelwise and mxfp4-fp8.** Both name `NEXTN`, resolved
  the same way to the Qwen3.5 checkpoint's own MTP layers; the two differ only in
  weights and, on fp8-channelwise, a triton draft attention backend the mxfp4 case
  does not name.

**Departure, all three MTP configs: `disable_radix_cache: true` in both roles.**
The GLM mxfp4 config carries it as the fp8-channelwise one does; the two Qwen
configs needed it added. These configs pair speculative decoding with
`--mamba-scheduler-strategy no_buffer`, and the PPU SDK refuses that combination
with the radix cache on — the prefill server exits at startup with `Speculative
decoding ... is not compatible with radix cache when using
--mamba-scheduler-strategy no_buffer`. The workload is `random-ids` at
`random_range_ratio` 1.0, so there is no shared prefix for a radix cache to reuse
and disabling it is throughput-neutral; the alternative (`extra_buffer` plus
`SGLANG_ENABLE_SPEC_V2=1`) changes more and departs further from the notune
sources. The incompatibility surfaces only when a pod actually starts the server —
JSON and schema validation pass — so it was found by dispatch, not by review.

## Deferred cases, and why

`144G/Daily/PD-Disaggregation/notune` holds 21 source cases. Nine belong to the
five checkpoint families this board has already stood up; four of those are now
ported and measured green — both GLM-5.2 configs and both Qwen3.5 configs. The
rest are deferred, each for a stated reason rather than for lack of time:

| Source case | Deferred because |
| --- | --- |
| `minimax-m2_7_{fp8-channel,mxfp4-fp8}_1p1d_4096_1500_0001.json` | Both need `--speculative-draft-model-path .../MiniMax-M2.5-Eagle3`. That checkpoint is referenced nowhere in this tree, and no existence evidence for it was found. |
| `kimi-k2.6_{int4,mxfp4-fp8}_1p2d_4096_1500_0001.json` | 1p2d — three boards. Nothing about the topology is unsupported by the schema, and the 1p1d shape it waited on is now green; what remains is standing up the three-board group. |
| `qwen3_8_mxfp4-fp8_2p4d_4096_1500_0001.json` | 2p4d — six boards, same reason. |

The remaining twelve (DeepSeek-V3.2 and five DeepSeek-V4 cases across four
variant names, GLM-5.1, GLM-5.3, Qwen3.7) are families no line on this board has
stood up, with no weight-existence evidence gathered for them.

The order is deliberate: the schema already validates any `NpMd` topology and the
runtime already resolves roles by rank, and the 1p1d pilot is now green, so the
deferred multi-board cases are config-only additions. The MiniMax cases are not —
they need a draft checkpoint established outside this repository first.

## Workflow

| Workflow | Suite | Boards | Trigger |
| --- | --- | --- | --- |
| `test-ppu-pd-perf-k8s.yml` | `nightly-pd-perf-16-glm52-ppu` | 2 | dispatch, `workflow_call` |
| `test-ppu-qwen35-pd-perf-k8s.yml` | `nightly-pd-perf-16-qwen35-ppu` | 2 | dispatch, `workflow_call` |

A workflow of its own rather than a lane in `test-ppu-perf-k8s.yml`, for the
reason the colocated 16-board entry has one: every lane in that file claims a
single board, so folding a two-board entry in would make each dispatch ask the
cluster for two more.

It gang-schedules both pods into a PodGroup with `minMember` 2, which matters more
here than on the colocated line: a prefill server whose decode peer never arrives
holds a board for the entire 5400 s peer-wait budget and then reports nothing.

One thing `run_pd_perf_suite_node.sh` deliberately does not do, recorded in its
header: it derives **no rendezvous address** (see above). It does set the **RoCE
GID index**, through the collective line's `answer_gid_index.sh` and under
Mooncake's own `MC_GID_INDEX`, and it sets a **proxy bypass** for the cluster
addresses its own HTTP hops use — both because the dispatches below proved they
were needed.
`GLOO_SOCKET_IFNAME=lo` is correct here for the single-board reason: each
server's ranks are processes in one network namespace, and its peer is reached
over HTTP and RDMA, never over gloo.

Evidence collection is the colocated line's `collect_perf_evidence.sh`
unchanged. What comes back under `ranks/` is each node's published endpoint, each
node's device inventory, each node's own log and each node's exit code — the
per-node log matters more here than on any other line, because the action streams
only worker-0's and the decode server's failure would otherwise be invisible.

## Capacity and time budget

Two ZW-M890P boards (144 GiB × 8 each), held for as long as the group runs.

`PERF_TIMEOUT_PER_FILE` is 13800 s, the pod 240 min and the job 270 min. The
reasoning: one 5400 s startup budget rather than two, since the loads proceed in
parallel; plus the endpoint exchange, one measurement of 80 requests at 4096 in /
1500 out at concurrency 8, and the dependency install. These were estimates a cold
clone had to survive rather than measured budgets — as is the
`register_ppu_ci(est_time=7200)` in the test file — and the green runs below now
give the real figures: a single measurement takes 3–10 min depending on whether
the config decodes speculatively, comfortably inside the budget.

## What the dispatches established

Three shakeout dispatches — the same config at 4 requests and 256 output tokens,
on a throwaway branch, 2026-09-09 — ran first. The first died in the prefill server
twenty seconds after launch and the other two died two minutes after that server
was listening, and between them they turned two guesses in this port into measured
facts and left the second failure with one candidate cause.

The first found that the Mooncake transfer engine opened none of the four bonds:
`No suitable GID found on mlx5_bond_1/`, then `Failed to open device mlx5_bond_1
on port  with GID -1`, on every bond, then `Mooncake Transfer Engine
initialization failed` and a server killed at `-9`. Its automatic GID discovery
requires an IPv4-mapped-IPv6 GID and these bonds carry no IPv4, which is upstream
Mooncake #1593. `run_pd_perf_suite_node.sh` now derives the index with the
collective line's `answer_gid_index.sh` and exports it as `MC_GID_INDEX`, and the
second dispatch confirmed the effect: `Using user-specified GID index: 3` on
every bond, the engine up, the checkpoint loaded and the server listening.

The second then died in the server's own warmup. Its `/model_info` request to its
own routable address came back as an nginx 404 for the full two minutes of the
warmup loop, while the uvicorn bound to that exact address logged no request at
all — so the request never arrived. Nothing in this repository names a proxy, so
the script exported a `no_proxy` covering loopback and this node's own /24 and
printed the proxy variables it inherited.

The third answered that: `inherited proxy: http_proxy=unset https_proxy=unset
HTTP_PROXY=unset HTTPS_PROXY=unset`, and the identical 404. No proxy was ever
configured, so no bypass could have helped — the bypass stays as the cheaper of
the two defences, but the cause is elsewhere. What is left that fits a bound
socket receiving nothing while nginx answers for it is a rewrite on the node, and
the port the servers used, 30000, is where Kubernetes starts handing out node
ports: kube-proxy's rules for a node port apply to traffic the node originates
too, so a server that binds one is answered for by whatever backs that service.
It also explains why the colocated line has never seen this — it talks to
`127.0.0.1`, and PD is the first line here that has to bind an address its peer
can reach. The ports moved to 21000/21001 and three probes now run before the
install, at addresses nothing of ours is listening on yet: the old port on this
node's address, the old port on loopback, and the port this run will use.

Then it ran green. The port move to 21000/21001 held, the three preflight probes
confirmed the rewrite reading above, and four configs each produced a full
measurement on two boards:

| Suite | Config | output tok/s | total tok/s | TTFT p50 | TTFT p99 | duration | Run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glm52 | fp8-channelwise | 194.91 | 727.16 | 546.38ms | 4249.09ms | 615.66s | 34317611899 |
| glm52 | mxfp4-fp8 | 453.32 | 1691.18 | 459.79ms | 4424.03ms | 264.71s | 34337978737 |
| qwen35 | fp8-channelwise | 605.75 | 2259.86 | 323.72ms | 1637.88ms | 198.10s | 34342244447 |
| qwen35 | mxfp4-fp8 | 689.83 | 2573.52 | 292.93ms | 1470.46ms | 173.96s | 34342255481 |

(The glm52 fp8-channelwise row is the recorded second pass of the double-pass
diagnostic that isolated the ~170 s cold-start compilation cost noted above; it is
the only config here without speculative decoding, which is why it decodes its
1500 tokens more slowly than the three MTP configs.)

That the numbers exist at all answers the two questions the shakeout runs left
open:

1. **`sglang_router`'s mini-lb comes up on this board.** `launch_router
   --pd-disaggregation --mini-lb` accepted its arguments, routed `bench_serving`'s
   requests to the two PPU servers, and returned completions — the throughput above
   is measured through it.
2. **The Mooncake KV handshake completes across two boards.** No prefilled block
   would reach the decode server otherwise, and every request decoded its full
   output; the fabric between two ZW-M890P boards carried the KV path for the whole
   run. `MC_LOG_LEVEL=TRACE` stays on until a run is drowned by it.

The two-pod harness itself worked throughout: the group gang-scheduled onto two
separate boards, each node published its endpoint and had the role it claimed
checked against the role its rank was assigned, and the report, the annotations and
the per-node evidence came back — with the one caveat, seen on some runs, that the
artifact upload reports success while arriving empty, in which case the numbers are
still in the run log.

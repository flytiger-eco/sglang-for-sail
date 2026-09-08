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

**This is a pilot, and it has never had a measured run.** Its workflow is on no
schedule and wired into no nightly caller. What it is dispatched to establish is
listed under [The two unknowns](#the-two-unknowns-the-first-dispatch-answers);
until one dispatch has answered them, nothing here should be read as a working
measurement path.

## What runs on the two boards

A PD group is one K8s job of two gang-scheduled pods, each holding a whole
ZW-M890P board, each running `scripts/ci/ppu/run_pd_perf_suite_node.sh`. Three
server-side processes exist across them:

| Process | Node | Port | Role |
| --- | --- | --- | --- |
| prefill server | rank 0 | 30000 | `--disaggregation-mode prefill`, tp 8 |
| decode server | rank 1 | 40000 | `--disaggregation-mode decode`, tp 8, dp 8 |
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
`perf_eval_kit.REASON_CODES`. A slow server is never red. The seventeen fields of
`perf_eval_kit.METRIC_FIELDS` are recorded, reaching the run page as one
annotation per measurement with the machine-readable copy in the artifact.

### TPOT and ITL are not recorded, and here that is a real gap

Unlike the prefill line — where `output_len` is 1 and decode-side metrics would
be degenerate — this case decodes 1500 tokens per request, and the source case's
metric block asks for `TPOT_AVG` and its percentiles. `bench_serving` does emit
`mean_tpot_ms`, `median_tpot_ms`, `std_tpot_ms`, `p99_tpot_ms` and the ITL
family, so nothing prevents recording them except that `METRIC_FIELDS` is shared
with the colocated line: extending it would change every colocated report's
schema from inside a PD-scoped change, and populate six fields there with the
zeros an empty `tpots` list produces.

They are left out deliberately, and the omission costs less than it looks:
`TPOT_AVG` is `primary: false` in the source case, both of its primary metrics
(`total_token_throughput`, `output_token_throughput`) are recorded, and at a
pinned concurrency the output throughput carries the same decode-side signal.
Adding a decode-metric block is the first follow-up once a measured run exists,
and it belongs in a change of its own that touches both lines.

`TTFT_P90` is omitted for the reason the colocated line records: the source
metric block reads a `tp_90` percentile `bench_serving` does not emit, and this
line does not change `bench_serving`. The p99 tail stands in its place.

## The suite

| Suite | Test file | Model | Topology | Measurements | Boards |
| --- | --- | --- | --- | --- | --- |
| `nightly-pd-perf-16-glm52-ppu` | `test_ppu_glm52_pd_perf.py` | GLM-5.2 FP8-Channelwise | 1p1d | 1 | 2 |

One config, one measurement, one suite. The device count leads the name as it
does on the colocated line, because it is the scheduling fact a reader needs
first: a dispatch of this suite asks the cluster for sixteen devices across two
whole boards.

## Alignment with the source case

The config renders the two `Chapter 2` launch commands of
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
  `--speculative-*` flag — the MTP-bearing variant is the MXFP4 one, which is
  deferred below. The name is inherited, not a description of what runs.
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

## Deferred cases, and why

`144G/Daily/PD-Disaggregation/notune` holds 21 source cases. Nine belong to the
five checkpoint families this board has already stood up; one is ported. The rest
are deferred, each for a stated reason rather than for lack of time:

| Source case | Deferred because |
| --- | --- |
| `glm-5_2_mxfp4-fp8_1p1d_4096_1500_0001.json` | `--speculative-algorithm EAGLE` with **no** `--speculative-draft-model-path`, so the draft head has to come from inside `GLM-5.2-MXFP4-FP8-fromBF16`. Nothing in this tree evidences that checkpoint carries an MTP head. |
| `minimax-m2_7_{fp8-channel,mxfp4-fp8}_1p1d_4096_1500_0001.json` | Both need `--speculative-draft-model-path .../MiniMax-M2.5-Eagle3`. That checkpoint is referenced nowhere in this tree, and no existence evidence for it was found. |
| `qwen3_5-397b-a17b_{fp8-channel,mxfp4-fp8}_1p1d_4096_1500_0001.json` | Both carry `--prefill-round-robin-balance`, which `server_args.py` accepts only to say it is deprecated. Porting it would encode a flag that no longer does anything. |
| `kimi-k2.6_{int4,mxfp4-fp8}_1p2d_4096_1500_0001.json` | 1p2d — three boards. Nothing about the topology is unsupported by the schema; it waits on the 1p1d shape being shown to work. |
| `qwen3_8_mxfp4-fp8_2p4d_4096_1500_0001.json` | 2p4d — six boards, same reason. |

The remaining twelve (DeepSeek-V3.2 and five DeepSeek-V4 cases across four
variant names, GLM-5.1, GLM-5.3, Qwen3.7) are families no line on this board has
stood up, with no weight-existence evidence gathered for them.

The order is deliberate: the schema already validates any `NpMd` topology and the
runtime already resolves roles by rank, so the deferred multi-board cases are
config-only additions once the pilot is green. The speculative and deprecated-flag
cases are not — they need a fact established outside this repository first.

## Workflow

| Workflow | Suite | Boards | Trigger |
| --- | --- | --- | --- |
| `test-ppu-pd-perf-k8s.yml` | `nightly-pd-perf-16-glm52-ppu` | 2 | dispatch, `workflow_call` |

A workflow of its own rather than a lane in `test-ppu-perf-k8s.yml`, for the
reason the colocated 16-board entry has one: every lane in that file claims a
single board, so folding a two-board entry in would make each dispatch ask the
cluster for two more.

It gang-schedules both pods into a PodGroup with `minMember` 2, which matters more
here than on the colocated line: a prefill server whose decode peer never arrives
holds a board for the entire 5400 s peer-wait budget and then reports nothing.

Two things `run_pd_perf_suite_node.sh` deliberately does not do, both recorded in
its header: it derives **no rendezvous address** (see above) and sets **no RoCE
GID index** — the cross-node path is Mooncake's over the devices
`disaggregation.ib_devices` names, not a collective's. Should the KV handshake
turn out to need its own hint, it belongs in the config next to those devices.
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
1500 out at concurrency 8, and the dependency install. **Every one of these
numbers is an estimate a cold clone has to survive, not a measured budget** — as
is the `register_ppu_ci(est_time=7200)` in the test file. The first green run is
what should replace them.

## The two unknowns the first dispatch answers

1. **Does `sglang_router`'s mini-lb come up on this board?**
   `ppu_install_dependency.sh` records `sglang-router 0.3.2+v0.1.0.ppu2.1.1` as
   present in the image and deliberately left alone (verified in-image
   2026-08-13), so the package exists. What is untested is whether that build's
   `launch_router --pd-disaggregation --mini-lb` accepts these arguments and
   routes to two PPU servers. The in-tree `disaggregation_fixture.py` is the only
   evidence for the flag shape.
2. **Does the Mooncake KV handshake complete across two of these boards?**
   The `mlx5_bond_*` devices come from the source commands and the transfer
   backend is SGLang's default, but no run in this repository has moved a KV
   block between two ZW-M890P boards. If it needs an environment hint beyond what
   the source cases set, this is the run that will say so — which is why
   `MC_LOG_LEVEL=TRACE` is left on.

# PPU Answer MVP

This directory holds the registered PPU Answer nightly tests together with the
public, versioned inputs they read. The evaluator lives in
`python/sglang/test/kits/answer_eval_kit.py` and the on-machine driver in
`python/sglang/test/kits/answer_suite_kit.py`.

```
answer_eval/
├── configs/<model family>/<model config>.json   reviewed execution contracts
├── dataset/                                     corpus, quality profile, schema
└── test_ppu_*.py                                registered entry points
```

The test files sit here rather than in `test/registered/ppu/` so that the suites,
the contracts they execute, and the corpus they are judged against are one
reviewable unit. Nothing about discovery changes: `run_suite.py` and the
`check-registered-tests` hook both walk `test/registered/**/*.py`, and a suite is
owned by the `register_ppu_ci` call inside the file, not by its directory.

Three models are covered, one registered file and one suite each because
`register_ppu_ci` registers per file and the three claim different device counts:

| Suite | Test file | Model | Devices |
| --- | --- | --- | --- |
| `nightly-answer-1-ppu` | `test_ppu_qwen38_answer.py` | Qwen3.8-27B, BF16 | 1 |
| `nightly-answer-8-ppu` | `test_ppu_qwen35_answer.py` | Qwen3.5-397B-A17B-W8A8-INT8 | 8 |
| `nightly-answer-32-ppu` | `test_ppu_qwen38_a95b_answer.py` | Qwen3.8-2.4T-A95B-FP8 | 32, over 4 nodes |

The first two suites are executed on two boards each. That is a matter of
configuration rather than of registration: the model, the corpus, and the judging
standard are the same, so a board is chosen by handing the suite a different
reviewed config.

| Board | Config suffix | Workflow | Trigger |
| --- | --- | --- | --- |
| ZW810E, 96GiB | none | `nightly-test-ppu-answer.yml` | cron 05:00 Beijing, dispatch |
| ZW-M890P, 144GiB | `-144g` | `test-ppu-answer-k8s.yml` | dispatch only |

`nightly-answer-32-ppu` has one config and one board. Its checkpoint is 2324.7
GiB over 213 shards, which no 96GiB node count this cluster can gang-schedule
would hold, so there is no ZW810E sibling to compare against; it has a workflow
of its own, `test-ppu-answer-32-k8s.yml`, dispatch only — see
[The four-node line](#the-four-node-line).

The dedicated `.github/workflows/nightly-test-ppu-answer.yml` workflow runs both
ZW810E entries as a `max-parallel: 1` matrix. It has its own workflow instead of
being dispatched by the general `nightly-test-ppu.yml` workflow, so Answer
failures, timeouts, scheduling, and artifacts remain isolated; its cron is
offset to 05:00 Beijing so the two workflows do not contend for the same runner.
Both entries are executed through `run_suite.py`, so the executed set is exactly
what the registry declares. `fail-fast` is off: one model's verdict must not
suppress the other's evidence, and the single-card entry runs first so a break in
the shared serving path appears hours before the 8-card entry would report it.
The ZW-M890P workflow keeps all of that and differs only where the cluster forces
it to; see [The ZW-M890P line](#the-zw-m890p-line).

The current phase enforces only request integrity and deterministic facts and
quality checks. LLM-as-Judge is intentionally deferred. Open-ended cases are
therefore marked `hard_constraints_only` in the result rather than being
presented as fully semantic evaluations.

## Runner configuration

`configs/<model family>/<model config>.json` are the reviewed sources of truth
for hardware selection, checkpoint identity and path, SGLang server parameters,
request generation parameters, timeouts, dataset, and quality profile. The file
name spells the identity its `test_id` declares, with the ZW810E baseline
carrying no board suffix: `configs/qwen3.8/27b-bf16.json` is
`qwen3.8-27b-bf16-answer-96g` and its ZW-M890P sibling
`configs/qwen3.8/27b-bf16-144g.json` is `qwen3.8-27b-bf16-answer-144g`. The
suffix is the per-device capacity rather than the board name, which is how the
internal plans separate these same two boards (`answer_96g`, `answer_144g`) and
what `answer_expected_hardware` renders into provenance (`zw810e-8x96g`,
`zw-m890p-8x144g`, and `zw-m890p-4nx8x144g` for the four-node entry). The
workflow reads the same file the test will read to select
the visible devices and to warm the checkpoint page cache, then exports its path
as `SGLANG_PPU_ANSWER_TEST_CONFIG`. This keeps the Python tests generic and makes
the exact execution contract visible in the workflow log.

Only the `hardware` section and the checkpoint path differ between a board pair.
Every server and generation parameter is held identical on purpose, so that a
difference between two verdicts is attributable to the board alone.

The two models share `dataset/answer_cases_zh_v1.json` and
`dataset/quality_profile.json`: the ten prompts are public general knowledge with
reviewed reference facts, so they identify a judgement standard rather than a
model, and a shared file keeps the two suites comparable by construction.

`evaluation.dataset` and `evaluation.quality_profile` are resolved against this
directory, which each test file declares as `data_root`, rather than against the
directory holding the config. Resolving from the config would put the shared
corpus above it, reachable only through `..`, which `validate_test_config`
rejects so that no config can name inputs outside the reviewed tree. `data_root`
is a class attribute with no environment override, unlike the config path: the
corpus a verdict was produced against stays pinned to the checkout.

CI runs each suite the same way every other registered test runs:

```bash
cd test
python3 run_suite.py --hw ppu --suite nightly-answer-8-ppu --nightly \
  --timeout-per-file 14400
```

`run_suite.py` invokes each file as `python3 <file> -f`, so the configuration
arrives through the environment variable. A local run can select the contract
the same way:

```bash
SGLANG_PPU_ANSWER_TEST_CONFIG=test/registered/ppu/answer_eval/configs/qwen3.5/397b-a17b-w8a8-int8.json \
  python3 test/registered/ppu/answer_eval/test_ppu_qwen35_answer.py
```

Each test file also names its own config as the default, so the variable is only
needed to depart from it. A config handed to the wrong file still fails loudly
rather than silently testing something else: validation ties the parallel
degrees to the declared devices and the preflight ties the checkpoint to the
devices actually visible.

For local pytest users, `test/registered/ppu/conftest.py` exposes the same
selection as `--answer-test-config <path>`; that option is not part of the CI
path.

The tests validate the configuration, checkpoint directory, and `config.json`
before starting SGLang. The test configuration digest and checkpoint
configuration digest are included in provenance; the first on-machine run must
establish the reviewed checkpoint digest baseline.

Candidate generation is deterministic for every model: temperature 0, top-p 1,
and at most 2048 output tokens. The 397B server configuration is TP=8, FA3
attention, static memory fraction 0.8, and `w8a8_int8` quantization; the 27B is
TP=1, FA3, static memory fraction 0.85, and `unquant`, which is how
`server_args` spells an explicit opt-out for a BF16 checkpoint; the 2.4T is
TP=32, FA3, static memory fraction 0.8, and no quantization flag at all, so the
checkpoint's own declaration stands — see
[Parity with the internal launcher](#parity-with-the-internal-launcher). All
three carry `watchdog_timeout` 600, the value the internal answer cases use.

## The ZW-M890P line

`.github/workflows/test-ppu-answer-k8s.yml` runs the same two suites against the
144GiB board, which is reachable only through the K8s cluster. It is a sibling
workflow rather than a second matrix dimension of the bare-metal one because
everything around the test differs, while the test itself does not:

- **Selection.** The board is chosen by `node_selector: board-type=ZW-M890P`, and
  the work runs in a worker pod submitted by `ppu-distributed-action` from a
  CPU-only orchestration shell that holds no device.
- **Devices.** `nproc_per_node` is a resource request, not isolation. A pod that
  asks for one PPU still sees all eight `/dev/alixpu_ppu*` nodes, and
  `torch.cuda.device_count()` still returns 8, so the preflight — which requires
  the visible device count to equal the configured one — would fail the
  single-card entry outright. `CUDA_VISIBLE_DEVICES` is therefore exported inside
  the pod from the same config the test reads, and both entries request all eight
  PPUs so that no second pod can land on the board and contend for a device with
  a server that has already claimed most of its memory. The internal btv1.5 plan
  schedules both answer cases as `1node8ppu` for the same reason.
- **Checkpoint path.** The 397B weights live under `T-HEAD/v3.5/` on this NAS
  rather than under `qwen/v3.5/` as on the ZW810E line; the 27B path is the same
  on both. The path is a per-config field, so this costs nothing beyond the two
  new configs — `checkpoint_name` is unchanged, because only the parent directory
  differs.
- **Evidence channel.** The action returns pod logs but not pod files, so the
  report travels over the NAS that both sides mount: the pod writes
  `SGLANG_PPU_ANSWER_RESULTS_DIR` under `/mnt/wl_nas/devops/<run>/<job>/`, and the
  orchestration shell reads the same bytes under `/wl_nas/...`, publishes the
  summary, and uploads the artifact. The suite name is part of the path because
  `github.job` is identical for every matrix entry. The NAS copy is read and left
  in place rather than moved: the pod writes as root and the orchestration shell
  is a different, non-root uid, so it can read those bytes but not unlink them.
  A step summary does not survive this runner at all, so the failing cases reach
  the run page as annotations instead; see
  [Results and annotations](#results-and-annotations).
- **Provenance.** `base_image_digest` is null on this path. The orchestration
  shell has no Docker daemon to inspect the image with and the pod cannot see its
  own digest, so only the image tag is recorded; on the bare-metal line the digest
  is resolved by `docker image inspect`.
- **Trigger.** No cron. GitHub honours `schedule` only on the default branch, so a
  cron on a version branch would never fire while claiming the workflow is
  scheduled. The line is dispatched by hand or through `workflow_call`.
- **Warm.** The page-cache warm runs inside the pod, immediately before
  `run_suite.py`. It has to: the cache it warms belongs to the node that will then
  load the weights, which no orchestration-shell step can reach.

The board's identity and capacity are measured rather than assumed. An inventory
probe submitted to this cluster on 2026-09-02 reported eight devices named
`ZW-M890P` with compute capability 8.9, 39 multiprocessors, 32MB of L2, and
`total_memory` 147456MB — 144.0GiB exactly — under torch 2.11.0, with both
checkpoints present (94 and 18 shards) and the JIT and Hugging Face caches in
place. `hardware.generation` is a free-form string that nothing validates against
a list, so a guessed value would be indistinguishable from a measured one in the
provenance record; `zw-m890p` and `memory_gib_per_device: 144` are what the
hardware reported. Board type alone would not have settled the capacity in any
case: the internal plans schedule this same board in both a 96GiB and a 144GiB
configuration.

Both entries have since been run on this board; see
[Measured baseline (ZW-M890P)](#measured-baseline-zw-m890p).

## The four-node line

`nightly-answer-32-ppu` serves one checkpoint across four ZW-M890P nodes at
TP=32. Its internal plan,
`btv1.5_P1_sglang_1node_func_answer_4nodes.csv`, schedules it as `4node8ppu` on
`M890P`, which is where the topology comes from: the case JSON itself carries only
`tp: 32`. The suite body is the same ten prompts and the same deterministic
evaluator; what is new is that four processes have to become one server, and
that only one of them can grade the result.

**The config states the topology, the environment states the rendezvous.**
`hardware.nnodes` is the reviewed fact — it says the checkpoint is served across
four nodes — and `hardware.visible_devices` keeps meaning *per node*, so
`validate_test_config` requires `tp_size * pp_size == len(visible_devices) *
nnodes`, which is the same product SGLang's own `check_server_args` measures
against the node count. There
is deliberately no `devices_per_node` field: it would restate the length of a
list that is already there, and two spellings of one number drift. `nnodes`
defaults to 1, so every existing config and the string
`answer_expected_hardware` renders for it are untouched.

**The 32 devices are 8-way tensor parallel over 4 pipeline stages, not 32-way
tensor parallel.** This is the one place the entry departs from the number the
internal case states, and the reason is arithmetic in the checkpoint rather than
a preference. `config.json` declares `moe_intermediate_size: 2048` and
`quantization_config.weight_block_size: [128, 128]`, and the FP8 path shards the
expert weight column-wise before it lays out the scales, so
`create_weights` in `layers/quantization/fp8.py` requires
`2048 / tp_size` to be a multiple of 128 — satisfied up to `tp_size` 16, and
`tp_size` 32 gives 64. Run `33813065317` is that failure, reached only after the
cross-node group had formed and weight loading had begun:

```
ValueError: The output_size of gate's and up's weight = 64 is not divisible by
weight quantization block_n = 128.
```

A second constraint is independent of quantization: the hybrid layers declare
`linear_num_key_heads: 16`, which 32 does not divide, so the linear-attention
heads cannot be split 32 ways either. Both are satisfied by `tp_size` 8, and the
remaining factor of four goes to the 92 layers: `pp_size` 4. SGLang lays a rank
out as `pp_rank * tp_size + tp_rank`, so each node ends up holding exactly one
pipeline stage — every tensor-parallel collective stays inside a node, and the
only cross-node traffic is the pipeline's point-to-point hand-off. That is a
lighter demand on the fabric than `tp_size` 32 would have made, not a heavier
one.

This is also the shape the vendor documents for this checkpoint on four nodes:
`server_cmds/LLM_Serving/BTV1.5/Qwen3.8-2.4T-A95B.md` in `model-test-cases`
gives `--tp-size 8 --pp-size 4` for the FP8 weights, and its 32-way command is a
different configuration entirely — expert parallel over DeepEP with data-parallel
attention, which needs a transport PPU does not have here. The vendor's own
evaluation command for the sibling MXFP4 checkpoint reads `--nnodes 2 --tp-size
16`, where `2048 / 16` is exactly 128.

`pp_size` is optional in the schema and absent from every other config, so those
entries emit the command line they emitted before the field existed; a config
that names it emits `--pp-size` immediately after `--tp-size`.

Which node a given process is, and where the group meets, are not reviewable
facts: they are decided by whatever launched the four pods. `resolve_distributed_runtime`
reads them from the environment and returns `None` for a single-node config:

| Variable | Meaning |
| --- | --- |
| `NODE_RANK` | this pod's rank, injected per pod by `ppu-distributed-action` |
| `NNODES` | the group size the launcher started, checked against `hardware.nnodes` |
| `MASTER_ADDR`, `MASTER_PORT` | rank 0's address, composed into `--dist-init-addr` |
| `SGLANG_PPU_ANSWER_NODE_RANK` | overrides the rank |
| `SGLANG_PPU_ANSWER_DIST_INIT_ADDR` | overrides the address, as `host:port` |

`NNODES` is checked rather than used: the group size is stated twice, once in the
reviewed config and once in the workflow that asks the cluster for boards, and a
launcher that started too few pods leaves every rank it did start with a valid
one. The shortfall would then surface only as a rendezvous that never completes,
after the group had held the boards it did get for the whole 5400s startup
budget. A launcher that states nothing is left alone, which is the bare-metal
case, and a single-node config is never asked, so the `NNODES=1` the action
injects for every single-board entry cannot fail one of those.

The two overrides are not conveniences. `ppu-distributed-action` asks for host
networking while leaving `spec.dnsPolicy` at ClusterFirst, so the pods get the
host resolver and the cluster-internal name it puts in `MASTER_ADDR` does not
resolve — measured identically on all four ranks of run `33750074634`. A patch
is with the action's owners; until it lands, the caller discovers rank 0's
address another way and passes it through the override. Everything the launch
needs is otherwise derived: the resolved rendezvous becomes `--nnodes`,
`--node-rank`, and `--dist-init-addr`, appended to the argument list the
single-node path already produced, and `build_answer_server_args` refuses a
multi-node config launched without one — the alternative is four independent rank
0 processes each trying to fit 2324.7 GiB onto eight devices.

**Every node runs the same file.** `AnswerSuiteMixin` does not branch until after
the server is up, because it does not have to: for `node_rank >= 1`, SGLang's
`launch_server` brings up its schedulers and then serves a dummy health endpoint,
so `popen_launch_server` returns on all four nodes and one launch path covers the
group. The branch is in what happens next.

| | Rank 0 | Ranks 1–3 |
| --- | --- | --- |
| Holds | the tokenizer and the HTTP API | its eight devices in the group |
| Does | generates, grades, writes the report | waits |
| Passes when | the corpus passes | it held the group until rank 0 was done |
| Fails when | a case fails or the server dies | its own server dies first, or rank 0 never finishes |

A worker cannot return early: its schedulers own their slice of every request, so
leaving would tear the group down under rank 0. It also cannot wait forever, so
the hold is bounded by the reviewed request budget for the whole corpus plus
`WORKER_HOLD_MARGIN_SECONDS`.

**The nodes address each other through the results directory.** They have no
other shared surface — the action streams worker-0's log and returns no files —
so `SGLANG_PPU_ANSWER_RESULTS_DIR` must be an absolute path on the NAS every pod
mounts, and the multi-node path refuses a relative one rather than letting each
pod write to its own copy of the same name. Underneath it, `ranks/` carries:

- `rank-<n>-devices.json`, each node's own device inventory, written staged and
  renamed because a reader on NFS can otherwise observe a partial file. Rank 0
  reads all four into the report's `accelerator` record, so the provenance
  describes the thirty-two devices the verdict was produced on rather than the
  eight rank 0 could see. A node whose inventory never arrived is named in
  `node_ranks_without_inventory` rather than dropped.
- `rank0-complete`, written when rank 0 is done. Workers poll it with a
  `listdir` rather than an `exists`, since a readdir revalidates the negative
  lookup NFS would otherwise cache.
- `rendezvous`, the address rank 0 published for the group to meet at, written by
  `scripts/ci/ppu/answer_rendezvous.sh` before anything else and read by the
  other three. Also staged and renamed, for the same reason.
- `rank-<n>.log` and `rank-<n>.status`, each node's own output and its own exit
  code, written by `scripts/ci/ppu/run_answer_suite_node.sh`.

Rank 0 writes its report where the workflow collects it; a worker writes its own
under `ranks/rank-<n>/`, so a worker's view of a run it did not grade cannot
overwrite the verdict. The single-node report is byte-identical to what it was:
the multi-node keys are additive.

The sentinel is written *before* rank 0 kills its own server. In the other order,
killing rank 0 makes the workers' schedulers exit, each worker sees its own server
die, and a healthy run reports three failed pods.

The exchange is under test without a board.
`TestPPUAnswerMultiNodeExchange` in `test_ppu_answer_eval_unit.py` binds the mixin
to each of four ranks against a real temporary directory with a stubbed device,
and covers the report destinations, the inventory round trip including a node that
reported nothing, the release, the three worker outcomes, and the teardown order.
It skips itself where torch is absent, since the driver needs it and the evaluator
does not.

**Measured baseline, four nodes.**
[Run 33849322347](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/33849322347),
2026-09-04, four ZW-M890P boards (thirty-two devices), driver 1.6.1, SDK 2.1.1,
at `b7b48ee`. **Ten of ten cases passed**, verdict `passed`, no suspect case, and
all four ranks exited 0.

| Phase | Cost | Observation |
| --- | --- | --- |
| `torch.distributed` init | 29.5s | the group forms across all four boards |
| Weight load | 14m42s | 213 shards, `type=Qwen3_5MoeForCausalLM`, `quant=fp8`, 72.36 GiB per device |
| CUDA graph capture | 5m30s | against the 46.4 GiB left free per device |
| Launch to ready | 22m44s | `/generate` warm-up 200 OK, well inside the 5400s `startup_timeout_seconds` |
| Ten graded requests | 57s | every one 200 OK; 16:01:35 to 16:02:32 |
| Registered file, total | 1455s | against `est_time` 7200 |

`max_total_num_tokens` comes out at 3875020 with a 262144 context and
`max_running_requests` 96. The 503s on `/health_generate` before ready are the
driver polling a scheduler that has not finished warming, not a fault.

The reasoning split is visible in the report: at `reasoning_effort: low` the
longest case spends 389 of its 418 completion tokens on reasoning and none of it
reaches the graded text — `final_answer` for the letter-count case is `4个。` — and
`reasoning_sha256` records the reasoning separately for every case. The widest
case uses 418 of the 8192 token budget, so `max_tokens` is not the binding
constraint at this effort.

`startup_timeout_seconds` 5400 and `est_time` 7200 remain as they were: both hold
comfortably against these numbers, and leaving headroom for a cold page cache is
deliberate on a node whose memory is smaller than the checkpoint tree.
`WORKER_HOLD_MARGIN_SECONDS` 900 was never approached — the workers finished
within 27s of rank 0.

**The workflow is its own file, and dispatch only.**
`.github/workflows/test-ppu-answer-32-k8s.yml` claims four whole boards, so it is
not a third entry in the btv1.5 matrix — sharing that matrix would make every
routine btv1.5 dispatch ask the cluster for four more boards — and it is not
wired into any nightly caller until it has passed once. `nnodes: 4` is what makes
the action gang-schedule: it creates a PodGroup with `minMember: 4`, so the group
either gets all four boards or waits, rather than half a group holding sixteen
devices while the rest never arrives. `nproc_per_node: 8` is the whole board on
each, which is both what TP=32 across four nodes needs and the only fence that
keeps a second pod off a board this run has claimed.

Four things differ from the single-board entries, each for a measured reason.

**Rank 0 publishes the rendezvous, first thing.**
`scripts/ci/ppu/answer_rendezvous.sh` runs before the dependency install: rank 0
detects its own address and writes it into `ranks/rendezvous`, and the other three
read it there and export it as `SGLANG_PPU_ANSWER_DIST_INIT_ADDR`. It has to be
done this way because the name the action injects as `MASTER_ADDR` does not
resolve in these pods, and it is done *early* so a worker waits only on its own
install rather than on rank 0's. The address is detected in the order SGLang's own
`get_local_ip_auto` uses — explicit host IP, then the source address the kernel
would pick for an outbound route, then the hostname — so the group meets at the
address the server would have chosen for itself. The detection is written against
the standard library rather than imported from `sglang`, because it runs before the
editable install and the rendezvous of a test should not need the tree under test
to be importable.

**The group is told which RoCE GID to use.** `scripts/ci/ppu/answer_gid_index.sh`
runs next and exports `NCCL_IB_GID_INDEX`. Left to itself pccl picks index 0 on
these hosts, which is the link-local `fe80::<EUI-64>` address formed from the
bond's MAC: it does not cross the L3 fabric between two nodes, so the group
builds its queue pairs over the out-of-band TCP channel, logs `Connected all
rings`, and then dies on its first payload with `IBV_WC_RETRY_EXC_ERR` — packets
leave and the peer never acknowledges them. What the bonds do carry is a global
ULA under `fd03::/8`, present as a consecutive GID pair, RoCE v1 then v2. A
two-node sweep of all six indices on these boards measured exactly one that
works:

| GID | address | result |
| --- | --- | --- |
| 0 | `fe80:…:<MAC>`, link-local | hang, then `SIGABRT` — **this is what pccl picks by itself** |
| 1 | `fe80:…:<MAC>`, link-local | `ibv_modify_qp` fails, connection timed out |
| 2 | `fd03:45c2:1:XXXX::1`, global | `IBV_WC_RETRY_EXC_ERR` |
| 3 | `fd03:45c2:1:XXXX::1`, global | `all_reduce` completes, checksum correct |
| 4, 5 | `fe80:45c2:…` | `IBV_WC_RETRY_EXC_ERR` |

The script derives the index — highest non-link-local GID, which is the v2 entry
of the routable pair — rather than writing `3` down, because the table is built
in whatever order the kernel added the addresses, and a node that ordered them
differently would otherwise fall silently back to a GID that drops every packet.
It exits non-zero if a device exposes no routable GID or if the four devices
disagree, both being conditions under which the group must not start: failing at
the entrance costs seconds, failing on the fabric costs the whole run.

Note the boundary: this compensates for a *host* condition, RoCE bonds with no
IPv4, which is what leads pccl's own preference — a v2 GID derived from the
device's IPv4 — to find no match and fall back. On a host whose bonds carry
IPv4, the automatic choice is already correct and this script would simply agree
with it.

**No page-cache warm.** A warm reads the whole checkpoint tree, and at 2324.7 GiB
that is larger than one node's 2266 GiB of memory, so the beginning of it is
already evicted by the time the read ends; run on all four nodes it would also
read four times what the group needs, since each node loads roughly its own
quarter. It would cost hours of NAS traffic to leave the cache no warmer than it
started. The single-board entries keep their warm, where the checkpoint does fit.

**Each node keeps its own log and its own exit code.** The action streams
worker-0's log and no other pod's, and a worker is exactly the node that would
report a device shortfall or a lost server, so
`scripts/ci/ppu/run_answer_suite_node.sh` tees each node's output into `ranks/`
and records its status there. `tee` rather than a redirect, so rank 0's log still
arrives live over a run this long; `PIPESTATUS` rather than the pipeline's status,
because handing back `tee`'s exit code would turn every failed suite green. The
collect step turns those status files into annotations: rank 0's verdict is
written before the workers are released, so a worker that failed afterwards is
visible in nothing else. The suite itself still runs through `run_suite.py`, so
the executed set is what `register_ppu_ci` declares.

## Relation to the internal test cases

All three suites are ports of internal `llm_infer_sglang_evalscope` answer cases
(`qwen3.5-397b-a17b-w8a8-int8_3001`, `qwen3.8-27b-bf16_3001`, and the btv1.5
`answer_4nodes/qwen3.8-2.4t-a95b-fp8_3001`). The port keeps
the checkpoint, device count, attention backend, memory fraction, and
quantization intent, and deliberately departs in two places:

- generation parameters are pinned to the deterministic house line (temperature
  0, top-p 1) instead of the sampling values the internal cases carry, because a
  rule-based verdict must be reproducible;
- serving parameters that the reviewed schema does not model — `page_size`,
  `stream_interval`, `max_running_requests`, `cuda_graph_max_bs_decode`,
  `dist_timeout`, and the `disable_*` flags — are
  left at SGLang defaults. `validate_test_config` accepts exactly the reviewed
  parameter set, so adding one of them is a schema change with its own review
  rather than a silent config edit.

The btv1.5 plan for the 144GiB board carries `qwen3.8-27b-fp8_3001` where this
repository keeps BF16, and no BF16 entry of its own. The port stays on BF16
deliberately: the two `-144g` configs are the `-96g` ones with a different board,
and nothing else, so a difference in verdicts has one candidate explanation. An
FP8 entry is a separate case with its own baseline, not a substitution.

The four-node case departs in one more place, which is not a choice: its internal
plan reaches the group through the cluster's own launcher, and this repository
reaches it through `ppu-distributed-action`, so the rank and the rendezvous arrive
by different means. `served_model_name` is `Qwen3.8-2.4T-A95B` while
`checkpoint_name` is `Qwen3.8-2.4T-A95B-FP8`, which is the internal case's own
spelling: the served name is what a client asks for and the checkpoint name is
what was loaded, and the reviewed schema keeps them separate for exactly this.

## Parity with the internal launcher

The internal cases are executed by `model_mate`, whose `SGLangServerCmd`
(`utils/commands/ServerCmd.py`, branch `sglang/h20_golden`) composes the launch.
The command it builds and the one `popen_launch_server` builds were compared
flag by flag; what follows is what that established.

**Identical.** Both invoke `sglang serve` rather than
`python3 -m sglang.launch_server`. Both reach a group through
`--dist-init-addr host:port`, `--nnodes`, and `--node-rank`, and both emit those
three only when the node count exceeds one, so a single-node launch is given no
rendezvous on either side. Both derive `--tp-size` from the case's `tp`, with the
one deliberate exception recorded above: the 2.4T FP8 entry splits the group as
`--tp-size 8 --pp-size 4`, because 32-way tensor parallelism cannot shard that
checkpoint's FP8 blocks at all.

**Quantization, which was a defect here.** `model_mate` does not derive the flag
from the case's `data_type` — that field only names the log directory — so a case
states a format or does not. Across the 51 internal btv1.5 sglang evalscope and
answer cases the line is sharp: only `data_type: w8a8-int8` carries
`quantization: w8a8_int8`, while `fp8`, `fp8-channel`, `mxfp4-fp8`, `awq`,
`gptq-int4`, `gptq-int8`, and `bf16` all leave it unset. The reason is visible in
`server_args.py`, where `fp8` is annotated *MOE + linear online quantization*:
naming it asks for unquantised weights to be quantised at load time, which
overrides what an already-quantised checkpoint declares in its own
`config.json`. The 2.4T config therefore states `null`, and
`build_answer_server_args` omits a null rather than rendering it. `unquant` on
the two BF16 configs is not the same case: `server_args` maps it to `None` on
arrival, and the flag it sets alongside is read only under `is_sm100_supported()`,
so on PPU it is exactly the internal cases' silence, spelled explicitly.

**`watchdog_timeout` 600** is now part of the reviewed schema and matches the
internal answer cases, which set it almost uniformly.

**`reasoning_parser`, a deliberate departure.** The internal Qwen answer cases do
not pass it; these configs pass one on all five. The requests here ask for
`separate_reasoning`, and the parser is what keeps a reasoning block out of the
graded text; the two entries with measured baselines were produced with it. The
internal cases that do pass one are the models whose templates need a different
parser (`glm45`, `deepseek-v4`, `minimax-append-think`).

Four configs pass `qwen3` and ask their template for `enable_thinking: false`.
The 2.4T entry cannot: its checkpoint ships a template that answers
`raise_exception('Disabling thinking is not supported.')` to exactly that
argument, which is why every request in run 33841139864 came back `400 Bad
Request` while `/generate` was answering 200. That template grades the pass
instead of switching it off — `reasoning_effort`, one of `xhigh` (its default),
`medium`, or `low`, rejected by name otherwise — so the config states `low`, and
`max_tokens` rises to 8192 with a 600s request timeout because a reasoning pass
that cannot be disabled still has to fit. The parser becomes `qwen3-thinking`,
which is the same `Qwen3Detector` with `force_reasoning` set: the template emits
`<|im_start|>assistant\n<think>\n` itself, so `<think>` arrives in the prompt and
the completion opens mid-reasoning with only `</think>` to come. `qwen3` decides
it is looking at reasoning by finding `<think>` in the text, so on this
checkpoint it would hand the whole reasoning pass to the grader as content.
Both findings were reproduced against the checkpoint's own template under
jinja2 3.1.2, in the image the pods run.

**Bind address, unchanged.** `model_mate` binds `--host 0.0.0.0`; here the host
comes from `DEFAULT_URL_FOR_TEST`, which is the loopback. Nothing off-node
reaches the HTTP API: the only client runs in rank 0's own pod, and the workers
coordinate through the rendezvous and the NAS, never over HTTP.

**Group environment variables.** `SGLangServerCmd._config_env` exports
`MASTER_ADDR`, `NNODES`, and `RANK` around the launch, and `_group_environment`
now does the same for a multi-node launch — for parity, not for a consumer this
repository can point at. On the path these configs take SGLang reads the
rendezvous from `--dist-init-addr` alone: `MASTER_PORT` matters only behind an
`env://` init-method override, and `MASTER_ADDR` only behind a `nixl` a2a
backend. What the PPU runtime below reads is not visible here, so the group
states itself the way the framework whose runs are the baseline states it.
`MASTER_ADDR` is taken from the rendezvous actually in force rather than copied
from the injected variable of the same name, since an override exists precisely
when the injected one does not resolve.

One mechanism has no analogue and needs none: `model_mate` skips any parameter
whose value is falsy, which is why an internal case's `ep: 0` produces no flag.
The reviewed schema lists its parameters exhaustively instead, so there is no
value that silently disappears.

## Measured baseline (ZW810E)

The first on-machine run was executed on `ptg-ppu-02` (16 × PPU-ZW810E, 96GiB
per device, driver 1.6.1, SDK 2.1.1) on 2026-09-01 using eight devices, and it
establishes the reference cost of the 397B test:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 26m17s | ~250MB/s across eight parallel readers |
| Weight load | 66s | 47.35 GB per device, 47.64 GB free afterwards |
| CUDA graph capture | ~3m | 18.66 GB free afterwards |
| Ten candidate generations | ~4m | 98.6 decode tokens/s |
| Test body total | 7m31s | well inside the 280-minute step budget |

Two properties of this host are load-bearing and are the reason the warm step
carries the comment it does: the checkpoint is only reachable over NFS at about
57MB/s per stream, and the page cache is reclaimed back to its baseline within
30 minutes regardless of free memory. Neither is a property of the test, and a
run that skips or defers the warm will exhaust `startup_timeout_seconds`
instead of producing a verdict.

During capture each of the eight TP ranks emits one
`Scheduler watchdog timeout (soft=True)` record with a py-spy dump. The
watchdog is soft, no process is killed, and the run proceeds normally; the only
effect is several hundred extra log lines.

The run reported nine of ten cases passing. `deepseek-letter-count` answered
`3` where the reviewed fact is `4`, with `finish_reason=stop` and complete
usage accounting, so the failure is a model-capability result rather than a
serving or evaluation defect. `fact_rule_failed` is classified `hard_fail`
by the kit, and neither the quality profile nor the case schema carries a
severity field, so this case fails the suite until the judgement contract is
revised upstream.

The 27B entry was measured on the same host on 2026-09-02 using one device:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 3m34s | 51.7GiB in 18 shards, ~247MB/s in parallel |
| Weight load | 11.4s | 51.05 GB used, 44.93 GB free afterwards |
| CUDA graph capture | 87.5s | batch sizes up to 33, 13.96 GB free afterwards |
| Server startup, launch to ready | 2m43s | against the 1800s `startup_timeout_seconds` |
| Ten candidate generations | ~27s | ~37 decode tokens/s |
| Test body total | 3m10s | reported elapsed 199s against `est_time` 1200 |
| Whole job | 8m41s | against the 60-minute job budget |

The generous budgets are kept deliberately: they cover a cold checkpoint. At the
measured single-stream 57MB/s, 51.7GiB needs about 15m30s, which together with
capture still fits inside `startup_timeout_seconds`, so this entry produces a
verdict even if the warm step is skipped — unlike the 397B entry, for which the
warm is a hard dependency. `max_total_num_tokens` came out at 264724 with
`context_len` 262144, so `mem_fraction_static` 0.85 leaves the KV pool ample
room on a 96GiB device.

That run reported seven of ten cases passing, all three failures being
`fact_rule_failed` on objective cases:

| Case | Answer | Reviewed fact |
| --- | --- | --- |
| `deepseek-letter-count` | `2` | 4 |
| `henan-bordering-provinces` | includes 湖南, omits 河北 | the six bordering provinces |
| `red-ball-probability` | `12.5` | 1/7, about 14.29% |

The first two are model-capability results: the request path is clean and the
rules are correct — the second even reproduces the error that the reviewed rule
was written to correct in the internal golden. The third is a wrong answer as
well, but it also exposed a gap in the `probability` rule, which accepted `a/b`,
`X%`, or a bare number already in `[0, 1]`. The prompt asks for a percentage, so
a bare `14.29` was discarded exactly as `12.5` was, and no correctly formed
bare-percentage answer could pass at all. A case now declares how a bare number
may be read, and this one admits either reading:

```json
{"type": "probability", "target": 0.142857142857, "tolerance": 0.0006,
 "bare_number_units": ["percent", "probability"], "description": "..."}
```

The declaration is the only way in: the default stays the probability reading
alone and an unknown unit fails the run, because inferring `percent` from a value
above 1 would admit a rounded `14` against a target of 1/7 under a loose enough
tolerance. The two readings of one bare number are alternatives for a single
claim, so the matching one is kept — otherwise `答案是0.1429` would contradict
itself, its percentage reading being both asserted and wrong. Neither the
tolerance nor the severity classification is changed, so `12.5` and a rounded
`14` still fail and the verdict for the run above stands. Changing what a rule
accepts changes the judging standard, so `dataset/answer_cases_zh_v1.json` is now
`revision` 2: an annotation keyed on revision 1 must not be read as though it had
been produced under the current standard.

## Measured baseline (ZW-M890P)

Both entries were run on the btv1.5 cluster on 2026-09-02 (run `33645062958`,
eight ZW-M890P devices of 144GiB, torch 2.11.0). The verdicts are the ZW810E ones,
case for case:

| Suite | Passed | Failing cases | Job | Test body |
| --- | --- | --- | --- | --- |
| `nightly-answer-1-ppu` (27B, 1 device) | 7/10 | `deepseek-letter-count`, `henan-bordering-provinces`, `red-ball-probability` | 19m06s | 399.8s |
| `nightly-answer-8-ppu` (397B, 8 devices) | 9/10 | `deepseek-letter-count` | 26m39s | 770.6s |

The failing answers are the ZW810E ones character for character: `2`, the same
Henan sentence that includes 湖南 and omits 河北, and `12.5` from the 27B; `3` from
the 397B. At temperature 0 that is the expected result, and it is what makes the
board a controlled variable — the `-144g` configs differ from the `-96g` ones only
in `hardware` and, for the 397B, in the checkpoint's parent directory, and no
verdict moves.

Where the board does show is in capacity and in the cost of using it:

| Phase | 27B, 1 device | 397B, 8 devices | ZW810E counterpart |
| --- | --- | --- | --- |
| Free device memory before load | 143.98 GB | 143.07 GB per rank | 96GiB board |
| Weight load | 3.8s, 51.05 GB | 21.9–24.5s, 47.35 GB per rank | 11.4s / 66s |
| CUDA graph capture | 260.8s, batch sizes to 78 | 621.7s | 87.5s (to 33) / ~3m |
| Launch to ready | 6m08s | 12m23s | 2m43s / — |
| `max_total_num_tokens` | 616103 | 2467089 | 264724 / — |
| Free memory after capture | 20.88 GB | 26.33 GB | 13.96 GB / 18.66 GB |

Capture is what grew. `mem_fraction_static` is a fraction, so the same 0.85 on a
144GiB device yields a KV pool 2.3× the 96GiB one; the captured batch-size list is
capped by `max_running_requests`, which the pool raises from 33 to 78 for the 27B,
and every additional batch size is another graph to record. Both entries still
reach ready far inside the 1800s `startup_timeout_seconds` and finish well inside
their step budgets. The 397B also reproduces the soft `Scheduler watchdog timeout`
record on each of the eight ranks during capture that the ZW810E line reports, with
the same absence of consequence.

Dependency install and the page-cache warm cannot be separated in these logs — the
pod's stdout reaches the runner in batches, so both land on a single flush
timestamp — but together they account for roughly 12 minutes for the 27B and 13.5
for the 397B. The install dominates: a worker pod starts from the base image and
re-resolves the wheels on every run, where the bare-metal host reuses what is
already installed. That is a property of the K8s path rather than of the test, and
it is why this line's pod budgets are 120 and 330 minutes.

One defect surfaced in that run and is fixed in the workflow as it now stands: the
evidence step also tried to delete the NAS copy after taking it, which fails with
`EPERM` on every file and turned the step red on both entries even though the copy
and the upload had succeeded. The step now only reads.

The fix was re-run on the same cluster (run `33690732486`, both entries against
`2f48d78`): `Collect Answer evidence` and `Upload Answer evidence` are green on
both, each artifact carries all five files, and the verdicts and failing cases are
unchanged — 7/10 and 9/10, same case ids. The remaining red step is the suite's own
assertion, which is the intended signal. That run also shows what the warm buys:
with the checkpoints already in the node's page cache from the run above, the jobs
took 10m56s and 18m15s instead of 19m06s and 26m39s. The table above keeps the cold
numbers, since a nightly on an otherwise idle board is the cold case.

## Results and annotations

The workflow uploads `result.json` (rule findings and provenance), `summary.md`,
`junit.xml`, and — because `SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS` is set to
`1` — `result.raw.json` and `label_candidates.jsonl`, which carry the candidate
answers verbatim. Publishing the candidates is deliberate: the prompts are
public general knowledge that already lives in this directory, the answers are
this project's own model output, and a red nightly is otherwise not diagnosable
without occupying eight devices for a second run. `result.json` stays redacted
so the schema-stable report keeps one shape whether or not raw collection is
enabled.

The same switch also prints the candidates into the job log, so a failure can be
read without downloading anything: after the summary table the test emits one
block per case with the prompt, the answer, and the observed value that tripped
each rule. `ci_utils.run_unittest_files` runs the test file as a plain
subprocess with an inherited stdout, so the block reaches the log whether the
run ends green or red. Because both the artifact and the log carry candidate
text, set the switch back to `0` for any dataset whose prompts or answers cannot
be published — one switch covers both surfaces.

One step further out, the run page itself names the failures. `summary.md` opens
with a `### Failing cases` list — one line per failing case carrying the rule
sentence the dataset declares, for example `deepseek-letter-count` against
`答案包含数字 4` — and both workflows turn each of those lines into a GitHub
annotation, plus one notice with the pass count. Annotations were chosen over the
step summary as the primary channel because **a step summary is not always
collected**: the K8s runner drives its job through a container hook that does not
share the job container's filesystem with the runner process, so bytes written to
`GITHUB_STEP_SUMMARY` there are dropped without an error — the check runs of the
first two runs on that board reported a summary of length zero while their
annotations arrived intact. Annotations travel over the step's stdout, which
reaches the runner on both paths. The step summary is still written, since it is
the richer surface where it works.

The two workflows read that list back with a `grep` for the bullet prefix, because
the orchestration container is not guaranteed a JSON parser — no `python3`, no
`jq`. That prefix appears on no other line of the document, and
`test_summary_names_the_failing_cases_for_annotations` locks the shape on the
evaluator's side, which is where a bash snippet cannot.

Both raw files are written with escaped non-ASCII (`\uXXXX`) so that an unpaired
surrogate in a candidate answer cannot fail the write. Read them with a JSON
tool, which decodes the escapes: `jq '.cases[] | select(.case_id=="...")'`, or
`jq -r '.cases[] | "[\(.verdict)] \(.case_id)\n  \(.final_answer)\n"'` for every
answer at once. The log block, in contrast, is already plain text: only
characters that UTF-8 cannot encode stay escaped there.

In provenance, `source_revision` is the authoritative identifier of what ran: it
is the full commit SHA of the checkout, injected by the workflow.
`package_versions.sglang` reads `0.0.0`, which is a property of the CI install
path rather than a collection defect. `ppu_install_dependency.sh` swaps in
`pyproject_other.toml`, whose version is dynamic, and installs with
`--no-build-isolation`, so `setuptools-scm` is never present to supply one; had
it run and found no tag, the configured `fallback_version` would have produced
`0.0.0.dev0` instead. The checkout is `--no-tags --depth=2` in any case, so no
tag is reachable on the machine to describe against. Recover the version number
from the SHA in a full clone:

```bash
git checkout <source_revision>
python3 python/tools/get_version_tag.py   # e.g. 0.5.13+v0.1.0-121-g5c9c6bce54
```

Public data belongs here:

- prompts, reviewed reference facts, and quality profiles;
- synthetic mutations that are safe to disclose;
- explicitly released calibration snapshots.

Pending blind reviews, adjudication records, and unrevealed holdout samples
belong in a private annotation repository. Records should conform to
`dataset/annotation_record.schema.json`. GitHub artifacts are evidence for a run,
not the long-term annotation system of record, so promote a candidate into that
repository rather than relying on the 30-day artifact retention.

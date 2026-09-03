"""On-machine driver shared by the registered PPU Answer suites.

`answer_eval_kit` deliberately stays hardware-free so the CPU suite can own the
evaluator; this module is the half that needs torch and a live server, so only
the PPU Answer test files import it.  Those files exist one per device topology
because `register_ppu_ci` registers a suite per file, and everything else that
differs between models lives in the reviewed test config, so the mixin below is
the entire body of such a test.

It is a plain mixin rather than a `TestCase` subclass on purpose: the test files
are executed as `python3 <file> -f`, and `unittest.main()` collects every
`TestCase` subclass it finds in the module namespace, so a shared base class
would also run itself with no configuration selected.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.answer_eval_kit import (
    CandidateRequestError,
    answer_expected_hardware,
    build_answer_server_args,
    build_report,
    canonical_digest,
    default_provenance,
    load_json,
    render_candidates,
    render_summary,
    request_chat_completion,
    resolve_distributed_runtime,
    resolve_evaluation_paths,
    validate_test_config,
    write_report_files,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

TEST_CONFIG_PATH_ENV = "SGLANG_PPU_ANSWER_TEST_CONFIG"
RESULTS_DIR_ENV = "SGLANG_PPU_ANSWER_RESULTS_DIR"
INCLUDE_RAW_OUTPUTS_ENV = "SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS"

# What a worker node allows rank 0 beyond the reviewed request budget for the
# whole corpus: grading is local and takes seconds, so this covers the report
# write and the interval between two nodes observing the same NAS directory.
# It is a guard against an unbounded hold, not the run's real fence, which is
# the pod timeout the workflow sets.
WORKER_HOLD_MARGIN_SECONDS = 900


class AnswerSuiteMixin:
    """The reviewed Answer contract, executed against a live server.

    A test file supplies the configuration and nothing else::

        class TestPPUQwen35Answer(AnswerSuiteMixin, unittest.TestCase):
            data_root = Path(__file__).parent
            default_test_config_path = data_root / "configs" / ... / "....json"

    `SGLANG_PPU_ANSWER_TEST_CONFIG` overrides that default, which is how the
    workflow keeps the file it warms and the file the test reads identical.  A
    config that does not match the file it is handed to still fails loudly:
    validation ties `tp_size` to the declared devices, and the preflight below
    ties the checkpoint to the devices actually visible.

    `data_root` is not overridable that way on purpose: the dataset and the
    quality profile are reviewed repository assets, so the corpus a verdict was
    produced against stays pinned to the checkout even when the config does not.

    A config that declares `hardware.nnodes` greater than one turns the same
    file into one node of a group: every node runs this mixin, rank 0 owns the
    HTTP API and the verdict, and the other ranks hold their devices in the
    group until rank 0 is done.  Which node this process is comes from the
    environment, not from the config; see `resolve_distributed_runtime`.
    """

    data_root = None
    default_test_config_path = None
    distributed = None
    node_rank = 0

    @classmethod
    def _load_test_config(cls):
        configured_path = os.environ.get(TEST_CONFIG_PATH_ENV)
        config_path = (
            Path(configured_path) if configured_path else cls.default_test_config_path
        )
        if config_path is None:
            raise RuntimeError(
                "no Answer test config selected: set default_test_config_path on the "
                f"test class or {TEST_CONFIG_PATH_ENV} in the environment"
            )
        config = load_json(config_path)
        validate_test_config(config)
        return config, Path(config_path)

    @classmethod
    def _provenance(cls, accelerator):
        provenance = default_provenance(
            cls.model_config["served_model_name"],
            cls.model_path or "",
            server_config=cls.server_config,
            generation_config=cls.request_config["generation"],
            expected_hardware=answer_expected_hardware(cls.test_config),
            accelerator=accelerator,
        )
        provenance["test_config_id"] = cls.test_config["test_id"]
        provenance["test_config_sha256"] = canonical_digest(cls.test_config)
        return provenance

    @classmethod
    def _local_accelerator(cls):
        device_count = torch.cuda.device_count()
        return {
            "visible_device_count": device_count,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(device_count)
            ],
        }

    @classmethod
    def _accelerator(cls):
        """What this run actually held, as one record per node.

        `visible_device_count` keeps meaning "what this process can see", so a
        single-node report is byte-identical to the ones already collected; the
        multi-node keys are additive.  A node whose inventory is absent is named
        rather than silently dropped, because "the report claims 32 devices" and
        "three nodes reported eight devices each and the fourth said nothing"
        are different pieces of evidence.
        """

        accelerator = cls._local_accelerator()
        if cls.distributed is None:
            return accelerator
        accelerator["nnodes"] = cls.distributed["nnodes"]
        accelerator["node_rank"] = cls.node_rank
        accelerator["node_name"] = os.environ.get("NODE_NAME")
        accelerator["dist_init_addr"] = cls.distributed["dist_init_addr"]
        nodes, missing = cls._read_node_inventories()
        accelerator["nodes"] = nodes
        accelerator["total_device_count"] = sum(
            node["visible_device_count"] for node in nodes
        )
        if missing:
            accelerator["node_ranks_without_inventory"] = missing
        return accelerator

    @classmethod
    def _node_inventory_path(cls, node_rank):
        return cls.rank_dir / f"rank-{node_rank}-devices.json"

    @classmethod
    def _write_node_inventory(cls):
        """Publish this node's devices where rank 0 can read them.

        The action streams the log of worker-0 only, so without this the report
        could describe eight of the thirty-two devices the verdict was produced
        on.  The write is staged and renamed because the four nodes reach this
        directory over NFS, where a reader can otherwise observe a partial file.
        """

        inventory = {
            "node_rank": cls.node_rank,
            "node_name": os.environ.get("NODE_NAME"),
            **cls._local_accelerator(),
        }
        path = cls._node_inventory_path(cls.node_rank)
        staged = path.parent / f"{path.name}.partial"
        staged.write_text(
            json.dumps(inventory, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)

    @classmethod
    def _read_node_inventories(cls):
        nodes = []
        missing = []
        for node_rank in range(cls.distributed["nnodes"]):
            try:
                nodes.append(load_json(cls._node_inventory_path(node_rank)))
            except (OSError, ValueError):
                missing.append(node_rank)
        return nodes, missing

    @classmethod
    def _sentinel_path(cls):
        return cls.rank_dir / "rank0-complete"

    @classmethod
    def _release_worker_nodes(cls):
        """Tell the worker nodes that rank 0 is done with them.

        Called from the teardown and from the setup failure path both: a setup
        that never launched a server still has to release nodes that did, or
        three pods sit on twenty-four devices until their own budget expires.
        """

        if cls.distributed is None or cls.node_rank != 0:
            return
        try:
            cls.rank_dir.mkdir(parents=True, exist_ok=True)
            cls._sentinel_path().write_text(
                f"{os.environ.get('GITHUB_RUN_ID', 'local')}\n", encoding="utf-8"
            )
        except OSError as error:
            print(
                "failed to release the worker nodes: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    @classmethod
    def _write_setup_failure(cls, stage, failure_class, exc):
        try:
            accelerator = cls._local_accelerator()
        except Exception:
            accelerator = {"visible_device_count": None}
        if cls.distributed is not None:
            accelerator["nnodes"] = cls.distributed["nnodes"]
            accelerator["node_rank"] = cls.node_rank
        responses = {
            case["id"]: {
                "error": type(exc).__name__,
                "reason_code": f"{stage}_failed",
                "failure_class": failure_class,
            }
            for case in cls.dataset["cases"]
        }
        provenance = cls._provenance(accelerator)
        provenance["setup_stage"] = stage
        report = build_report(cls.dataset, cls.profile, responses, provenance)
        write_report_files(report, cls.report_dir)
        print(render_summary(report), flush=True)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_config, cls.test_config_path = cls._load_test_config()
        cls.model_config = cls.test_config["model"]
        cls.server_config = cls.test_config["server"]["parameters"]
        cls.request_config = cls.test_config["request"]
        if cls.data_root is None:
            raise RuntimeError(
                "no Answer data root selected: set data_root on the test class"
            )
        dataset_path, profile_path = resolve_evaluation_paths(
            cls.test_config, Path(cls.data_root)
        )
        cls.dataset = load_json(dataset_path)
        cls.profile = load_json(profile_path)
        cls.output_dir = Path(os.environ.get(RESULTS_DIR_ENV, "ppu-answer-artifacts"))
        # Before the try: a config that asks for four nodes without giving them a
        # way to reach each other is a misconfiguration of the caller, and the
        # structured evidence below is written to a directory this very check is
        # what establishes.
        cls.distributed = resolve_distributed_runtime(cls.test_config)
        cls.node_rank = 0 if cls.distributed is None else cls.distributed["node_rank"]
        cls.rank_dir = cls._resolve_rank_dir()
        # Rank 0 publishes the verdict where the workflow collects it; a worker
        # node writes its own evidence one level down, because all nodes share
        # this directory and a worker's report would otherwise overwrite the
        # verdict with its own view of a run it did not grade.
        cls.report_dir = (
            cls.output_dir
            if cls.node_rank == 0
            else cls.rank_dir / f"rank-{cls.node_rank}"
        )
        cls.model_path = cls.model_config["path"]
        stage = "runner_preflight"
        failure_class = "runner_error"
        try:
            if not Path(cls.model_path).is_dir():
                raise RuntimeError(
                    f"the model directory configured by {cls.test_config_path} is missing"
                )
            model_dir = Path(cls.model_path)
            if model_dir.name != cls.model_config["checkpoint_name"]:
                raise RuntimeError(
                    "model.path must identify " f"{cls.model_config['checkpoint_name']}"
                )
            config_path = model_dir / "config.json"
            if not config_path.is_file():
                raise RuntimeError(f"{cls.model_path} does not contain config.json")
            model_config = load_json(config_path)
            accepted_model_types = cls.model_config["accepted_model_types"]
            if model_config.get("model_type") not in accepted_model_types:
                raise RuntimeError(
                    f"checkpoint config model_type {model_config.get('model_type')!r} "
                    f"is not one of the accepted types {accepted_model_types}"
                )
            expected_device_count = len(cls.test_config["hardware"]["visible_devices"])
            if torch.cuda.device_count() != expected_device_count:
                raise RuntimeError(
                    f"{cls.test_config['test_id']} requires {expected_device_count} "
                    f"visible PPU devices per node; node {cls.node_rank} found "
                    f"{torch.cuda.device_count()}"
                )

            if cls.distributed is not None:
                cls.rank_dir.mkdir(parents=True, exist_ok=True)
                if cls.node_rank == 0:
                    # A worker only looks for this after its own server is ready,
                    # which cannot happen before rank 0 has joined the group, so
                    # clearing it here cannot race a worker into an early exit.
                    cls._sentinel_path().unlink(missing_ok=True)

            cls.base_url = DEFAULT_URL_FOR_TEST
            stage = "server_start"
            failure_class = "server_error"
            cls.process = popen_launch_server(
                model=cls.model_path,
                base_url=cls.base_url,
                timeout=cls.test_config["server"]["startup_timeout_seconds"],
                other_args=build_answer_server_args(
                    cls.test_config, distributed=cls.distributed
                ),
            )
            # A worker rank serves a dummy health endpoint once its own
            # schedulers are ready, so the launch above returns on every node and
            # this is the point where each one knows what it holds.
            if cls.distributed is not None:
                stage = "node_inventory"
                failure_class = "runner_error"
                cls._write_node_inventory()
        except Exception as exc:
            try:
                cls._write_setup_failure(stage, failure_class, exc)
            except Exception as report_error:
                print(
                    "failed to write structured Answer setup evidence: "
                    f"{type(report_error).__name__}",
                    flush=True,
                )
            cls._release_worker_nodes()
            raise

    @classmethod
    def _resolve_rank_dir(cls):
        """The directory the nodes of a group use to address each other.

        None for a single-node config.  The multi-node case insists on an
        absolute results directory because the nodes exchange their device
        inventory and rank 0's completion through it, and the default is a
        relative path that resolves to a different directory in each pod.
        """

        if cls.distributed is None:
            return None
        if not cls.output_dir.is_absolute():
            raise RuntimeError(
                f"a multi-node Answer config needs {RESULTS_DIR_ENV} set to a "
                "path every node of the group shares; "
                f"got {cls.output_dir}"
            )
        return cls.output_dir / "ranks"

    @classmethod
    def tearDownClass(cls):
        # Released before the server is killed, and not the other way round: a
        # worker treats the loss of its own server as a failure, and killing rank
        # 0's process is what makes the workers' schedulers exit, so a teardown
        # in the other order would report three failed worker pods on a healthy
        # run.
        cls._release_worker_nodes()
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
        super().tearDownClass()

    def _hold_until_rank_zero_completes(self):
        """A worker node's share of the suite is to stay in the group.

        Only rank 0 holds the tokenizer and the HTTP API; this node's schedulers
        own their slice of every request, so returning early would tear the group
        down under rank 0.  The pass condition is therefore that the node held
        its devices until rank 0 published its completion, and the failure
        conditions are its own server dying first, or that completion never
        arriving.
        """

        sentinel = self._sentinel_path()
        budget = (
            self.request_config["timeout_seconds"] * len(self.dataset["cases"])
            + WORKER_HOLD_MARGIN_SECONDS
        )
        deadline = time.monotonic() + budget
        print(
            f"node {self.node_rank} is holding the group and waiting for {sentinel}",
            flush=True,
        )
        while time.monotonic() < deadline:
            # listdir rather than exists: the directory is on NFS, and a readdir
            # revalidates the entry that a cached negative lookup would not.
            try:
                released = sentinel.name in os.listdir(self.rank_dir)
            except OSError:
                released = sentinel.exists()
            if released:
                print(f"node {self.node_rank} was released by rank 0", flush=True)
                return
            exit_code = self.process.poll()
            if exit_code is not None:
                self.fail(
                    f"node {self.node_rank} lost its server with exit code "
                    f"{exit_code} before rank 0 completed"
                )
            time.sleep(10)
        self.fail(
            f"node {self.node_rank} held the group for {budget}s and rank 0 never "
            f"published its completion at {sentinel}"
        )

    def test_public_answer_suite(self):
        if self.node_rank != 0:
            self._hold_until_rank_zero_completes()
            return
        responses = {}
        for case in self.dataset["cases"]:
            try:
                responses[case["id"]] = request_chat_completion(
                    self.base_url,
                    self.model_config["served_model_name"],
                    case["prompt"],
                    self.dataset["request_suffix"],
                    timeout_seconds=self.request_config["timeout_seconds"],
                    generation_config=self.request_config["generation"],
                )
            except CandidateRequestError as exc:
                responses[case["id"]] = {
                    "error": str(exc),
                    "reason_code": exc.reason_code,
                }

        report = build_report(
            self.dataset,
            self.profile,
            responses,
            self._provenance(self._accelerator()),
        )
        include_raw_outputs = os.environ.get(INCLUDE_RAW_OUTPUTS_ENV, "0") in {
            "1",
            "true",
            "TRUE",
        }
        write_report_files(
            report, self.report_dir, include_raw_outputs=include_raw_outputs
        )
        print(render_summary(report), flush=True)
        if include_raw_outputs:
            # The log is a disclosure surface like the artifact, so the same
            # switch governs both.  ci_utils runs this file as a plain
            # subprocess with an inherited stdout, so the block lands in the job
            # log whether or not the assertion below fails.  Relaxing the error
            # handler keeps a candidate that stdout cannot encode from turning
            # the nightly red over log formatting alone.
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="backslashreplace")
            print(render_candidates(report, self.dataset), flush=True)

        self.assertEqual(
            report["summary"]["failed"],
            0,
            json.dumps(
                {
                    result["case_id"]: [
                        finding["reason_code"]
                        for finding in result["findings"]
                        if finding["action"] == "hard_fail"
                    ]
                    for result in report["cases"]
                    if result["verdict"] == "failed"
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

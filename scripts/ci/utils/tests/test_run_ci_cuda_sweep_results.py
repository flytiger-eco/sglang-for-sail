import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("cuda_baseline", ROOT / "scripts/ci/ppu/cuda_ci_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class SweepResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "worker"
        self.junit = self.directory / "junit"
        self.junit.mkdir(parents=True)
        self.file = "test/registered/unit/test_example.py"
        self.metadata = {
            "suite": "base-a-test-1-gpu-small", "board": "810E", "n_devices": 1,
            "run_id": "123", "attempt": 1, "sha": "a" * 40,
            "image_digest": "sha256:" + "b" * 64, "source_root": "/workspace/source",
            "timeout_per_file": 600, "files": [self.file], "ppu_supports_fp8": 0, "plan_attempt": 1,
        }
        (self.directory / "sweep.json").write_text(json.dumps(self.metadata))

    def parse(self, xml=None, code=0, error="", log="", elapsed=1.5):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"), "sweep result parser is missing")
        record = {"file": "/workspace/source/" + self.file, "junit": "case.xml", "log": "case.log", "elapsed": elapsed}
        if code is not None:
            record["returncode"] = code
        if error:
            record["error"] = error
        (self.junit / "summary.json").write_text(json.dumps({"selected": [record["file"]], "files": [record]}))
        (self.junit / "case.log").write_text(log)
        if xml is not None:
            (self.junit / "case.xml").write_text(xml)
        rows = baseline.parse_sweep_results(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], self.file)
        return rows[0]

    def frozen_plan(self, files=None, attempt=1, **kwargs):
        files = files or [self.file]
        registrations = [dict(file=f, suite=self.metadata["suite"], backend="CUDA",
                              est_time_s=10, gpu_mem_gb=80, n_devices=1) for f in files]
        plan = baseline.build_sweep_plan(registrations, suites=self.metadata["suite"],
                                        sha=self.metadata["sha"], partition_size=len(files),
                                        board="810E", **kwargs)
        plan.update(run_id=self.metadata["run_id"], image_digest=self.metadata["image_digest"], attempt=attempt)
        return plan

    def worker_evidence(self, name, plan, index=0, attempt=1, fail=False):
        directory = self.root / name
        junit = directory / "junit"
        junit.mkdir(parents=True, exist_ok=True)
        job = plan["matrix"]["include"][index]
        metadata = dict(self.metadata, **job)
        metadata.update(attempt=attempt, plan_attempt=plan["attempt"])
        (directory / "sweep.json").write_text(json.dumps(metadata))
        selected = ["/workspace/source/" + f for f in job["files"]]
        records = []
        for i, f in enumerate(selected):
            records.append(dict(file=f, junit=f"{i}.xml", log=f"{i}.log", returncode=int(fail)))
            (junit / f"{i}.xml").write_text('<testsuite><testcase classname="Example" name="ok">' +
                                           ('<failure message="wrong"/>' if fail else '') + '</testcase></testsuite>')
            (junit / f"{i}.log").write_text("failed" if fail else "passed")
        (junit / "summary.json").write_text(json.dumps(dict(selected=selected, files=records)))
        return directory

    def test_result_exposes_counts_paths_and_observed_fp8(self):
        row = self.parse('<testsuite><testcase name="ok"/></testsuite>')
        self.assertEqual(row.get("tests"), 1)
        self.assertEqual(row.get("ppu_supports_fp8"), 0)
        self.assertEqual(row.get("log_path"), str(self.junit / "case.log"))
        self.assertEqual(row.get("junit_path"), str(self.junit / "case.xml"))

    def test_missing_or_board_inconsistent_runtime_fp8_never_passes(self):
        for board, value in (("810E", None), ("810E", 1), ("890P", 0), ("890P", "1"), ("890P", True)):
            with self.subTest(board=board, value=value):
                self.metadata["board"] = board
                self.metadata["ppu_supports_fp8"] = value
                (self.directory / "sweep.json").write_text(json.dumps(self.metadata))
                row = self.parse('<testsuite><testcase name="ok"/></testsuite>')
                self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))
                self.assertEqual(row.get("ppu_supports_fp8"), value)
                self.assertIn("ppu_supports_fp8", row["reason"])

    def test_attempted_failures_are_gaps_not_raw_status_verdicts(self):
        for status in ("FAIL", "TIMEOUT", "CRASH", "OOM"):
            with self.subTest(status=status):
                self.assertEqual(baseline._triage(dict(auto_status=status)), "GAP")
        self.assertEqual(baseline._triage(dict(auto_status="OOM", gpu_mem_gb=180)), "NEEDS_TOPOLOGY")
        self.assertEqual(baseline._triage(dict(auto_status="INFRA_MODEL", file="test_nvfp4.py")), "INFRA_BLOCKED")

    def test_failed_nvfp4_concrete_evidence_is_gap_and_marked(self):
        for source in ("filename", "testcase", "error", "log", "failure"):
            with self.subTest(source=source):
                self.file = "test/registered/unit/test_nvfp4.py" if source == "filename" else "test/registered/unit/test_example.py"
                self.metadata["files"] = [self.file]
                (self.directory / "sweep.json").write_text(json.dumps(self.metadata))
                case = "test_nvfp4" if source == "testcase" else "test_case"
                failure = "nvfp4: CUDA out of memory" if source == "failure" else "CUDA out of memory"
                row = self.parse(f'<testsuite><testcase classname="Example" name="{case}"><failure message="{failure}"/></testcase></testsuite>',
                                 code=1, error="nvfp4 failed" if source == "error" else "", log="NVFP4 failed" if source == "log" else "")
                row["gpu_mem_gb"] = 180
                self.assertEqual(baseline._triage(row), "GAP")
                self.assertTrue(row["failure_signature"].startswith("NVFP4"))
                self.assertNotEqual(row.get("reason_code"), "NV_DTYPE")

    def test_mxfp4_is_generic_gap_and_infra_is_not_relabelled(self):
        row = self.parse('<testsuite><testcase><failure message="MXFP4 unavailable"/></testcase></testsuite>', code=1)
        self.assertEqual(row["triage_status"], "GAP")
        row = self.parse('<testsuite><testcase><error type="LocalEntryNotFoundError" message="NVFP4 model absent"/></testcase></testsuite>', code=2)
        self.assertEqual((row["auto_status"], row["triage_status"]), ("INFRA_MODEL", "INFRA_BLOCKED"))

    def test_partial_or_missing_selected_keeps_planned_rows_unexecuted(self):
        plan = self.frozen_plan()
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        self.worker_evidence("worker", plan)
        summary = self.junit / "summary.json"
        original = json.loads(summary.read_text())
        for selected in ([], None):
            with self.subTest(selected=selected):
                data = dict(original)
                if selected is None:
                    data.pop("selected")
                else:
                    data["selected"] = selected
                summary.write_text(json.dumps(data))
                row, = baseline.parse_sweep_results(self.root)
                self.assertEqual(row["auto_status"], "NOT_EXECUTED")
        summary.write_text(json.dumps(dict(original, selected=original["selected"] * 2)))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_partial_rerun_uses_producer_plan_and_keeps_successful_job(self):
        second = "test/registered/unit/test_second.py"
        plan = self.frozen_plan([self.file, second])
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        self.worker_evidence("worker", plan, 0, attempt=1)
        self.worker_evidence("failed", plan, 1, attempt=1, fail=True)
        self.worker_evidence("retry", plan, 1, attempt=2)
        rows = baseline.parse_sweep_results(self.root)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["plan_attempt"] == 1 for r in rows))
        manifest = baseline.build_parity_manifest(plan["expected"], rows)
        self.assertEqual([r["auto_status"] for r in manifest], ["PASS", "PASS"])
        self.assertEqual([len(r["attempt_history"]) for r in manifest], [1, 2])
        self.assertTrue(manifest[1]["flaky"])
        new_plan_dir = self.root / "new-plan"
        new_plan_dir.mkdir()
        (new_plan_dir / "sweep_plan.json").write_text(json.dumps(dict(plan, attempt=2)))
        rows = baseline.parse_sweep_results(self.root)
        manifest = baseline.build_parity_manifest(plan["expected"], rows)
        self.assertEqual([r["auto_status"] for r in manifest], ["NOT_EXECUTED", "NOT_EXECUTED"])

    def test_new_shared_plan_does_not_reuse_old_exact_suite_pass(self):
        plan = self.frozen_plan()
        other = dict(plan["expected"][0], cuda_suite="extra-test-1-gpu-large", suite="extra-test-1-gpu-large")
        plan["expected"].append(other)
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        self.worker_evidence("worker", plan)
        new_plan = dict(plan, attempt=2, expected=list(reversed(plan["expected"])),
                        matrix={"include": [dict(plan["matrix"]["include"][0], suite=other["cuda_suite"])]})
        new_dir = self.root / "new-plan"
        new_dir.mkdir()
        (new_dir / "sweep_plan.json").write_text(json.dumps(new_plan))
        rows = baseline.parse_sweep_results(self.root)
        manifest = baseline.build_parity_manifest(new_plan["expected"], rows)
        self.assertEqual([r["auto_status"] for r in manifest], ["NOT_EXECUTED", "NOT_EXECUTED"])
        for row in manifest:
            self.assertEqual(row["plan_attempt"], 2)
            self.assertEqual([r["auto_status"] for r in row["attempt_history"]], ["PASS", "NOT_EXECUTED"])
        self.worker_evidence("new-worker", new_plan, attempt=2, fail=True)
        manifest = baseline.build_parity_manifest(new_plan["expected"], baseline.parse_sweep_results(self.root))
        self.assertEqual([r["auto_status"] for r in manifest], ["FAIL", "FAIL"])

    def test_frozen_plan_rejects_worker_capability_and_provenance_drift(self):
        plan = self.frozen_plan()
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        for change in ({"ppu_supports_fp8": 1}, {"ppu_supports_fp8": None}, {"plan_attempt": 2},
                       {"run_id": "124"}, {"sha": "c" * 40}, {"image_digest": "sha256:" + "d" * 64},
                       {"key": "other"}, {"partition_size": 2}):
            with self.subTest(change=change):
                self.worker_evidence("worker", plan)
                metadata = json.loads((self.directory / "sweep.json").read_text())
                metadata.update(change)
                (self.directory / "sweep.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    baseline.parse_sweep_results(self.root)

    def test_undispatched_worker_cannot_borrow_expected_rows(self):
        plan = self.frozen_plan()
        self.worker_evidence("worker", plan)
        plan["matrix"]["include"] = []
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_worker_must_identify_immutable_producer_plan(self):
        plan = self.frozen_plan()
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        self.worker_evidence("worker", plan)
        metadata = json.loads((self.directory / "sweep.json").read_text())
        metadata.pop("plan_attempt")
        (self.directory / "sweep.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "plan_attempt"):
            baseline.parse_sweep_results(self.root)

    def test_failure_body_and_parameterized_testcase_are_concrete_nvfp4_evidence(self):
        row = self.parse('<testsuite><testcase classname="Example" name="test_dtype[float4]"><failure type="AssertionError" message="failed">NVFP4 path unavailable</failure></testcase></testsuite>', code=1)
        self.assertTrue(row["failure_signature"].startswith("NVFP4"))
        self.assertEqual(row["triage_status"], "GAP")

    def test_absent_classname_does_not_fabricate_exact_skip_node(self):
        row = self.parse('<testsuite><testcase name="ok"/><testcase name="Example.test_skip"><skipped message="unsupported"/></testcase></testsuite>')
        self.assertNotIn("Example.test_skip", row["skipped_nodes"])

    def test_plan_does_not_share_a_missing_job_across_different_timeout(self):
        plan = self.frozen_plan()
        other = dict(plan["expected"][0], cuda_suite="extra-test-1-gpu-large", suite="extra-test-1-gpu-large", timeout_per_file=1200)
        plan["expected"].append(other)
        job = dict(plan["matrix"]["include"][0], key="second", suite=other["cuda_suite"], timeout_per_file=1200)
        plan["matrix"]["include"].append(job)
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        self.worker_evidence("worker", plan)
        rows = baseline.parse_sweep_results(self.root)
        self.assertEqual(len(rows), 2)
        manifest = baseline.build_parity_manifest(plan["expected"], rows)
        self.assertEqual([r["auto_status"] for r in manifest], ["PASS", "NOT_EXECUTED"])
        self.assertEqual([r["evidence_shared_with"] for r in manifest], [[], []])

    def test_pass_requires_exit_zero_and_junit(self):
        row = self.parse('<testsuites><testsuite><testcase name="ok"/></testsuite></testsuites>')
        self.assertEqual(row["auto_status"], "PASS")
        self.assertEqual(row["passed"], 1)
        self.assertEqual(row["wall_s"], 1.5)

    def test_pass_with_skips_retains_reason(self):
        row = self.parse('<testsuite><testcase name="ok"/><testcase name="skip"><skipped message="platform-only"/></testcase></testsuite>')
        self.assertEqual(row["auto_status"], "PASS_WITH_SKIPS")
        self.assertEqual(row["skipped"], 1)
        self.assertIn("platform-only", row["skip_reasons"])

    def test_runner_skip_policy_error_allows_later_exact_review(self):
        runner_spec = importlib.util.spec_from_file_location("sweep_test_runner", ROOT / "python/sglang/test/ci/pr_runner.py")
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        xml = '<testsuite><testcase classname="Example" name="ok"/><testcase classname="Example" name="test_skip"><skipped message="platform-only"/></testcase></testsuite>'
        path = self.junit / "case.xml"
        path.write_text(xml)
        with self.assertRaises(runner.InvalidResults) as rejected:
            runner.inspect_junit(path)
        error = str(rejected.exception)
        row = self.parse(xml, error=error)
        self.assertEqual(row["auto_status"], "PASS_WITH_SKIPS")
        approval = dict(file=self.file, junit_id="Example.test_skip", cuda_suites=self.metadata["suite"],
                        reason_code="HW_CAPABILITY", evidence="fixture:10", reviewer="ci-owner", expires_at="2026-11-01")
        approved = self.root / "approved_skips.csv"
        with approved.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(approval))
            writer.writeheader()
            writer.writerow(approval)
        manifest = baseline.build_parity_manifest([row], [row], approved, today=date(2026, 10, 29))
        self.assertEqual(manifest[0]["triage_status"], "PASS")
        for message in ("unrelated runner error", error + "; runtime failure", error.replace("'failures': 0", "'failures': 1")):
            with self.subTest(error=message):
                self.assertEqual(self.parse(xml, error=message)["auto_status"], "FAIL")
        self.assertEqual(self.parse(xml, code=1, error=error)["auto_status"], "FAIL")

    def test_failure_and_error_counts_are_separate(self):
        row = self.parse('<testsuite><testcase><failure type="AssertionError" message="wrong"/></testcase><testcase><error type="ImportError" message="missing"/></testcase></testsuite>', code=1)
        self.assertEqual(row["auto_status"], "FAIL")
        self.assertEqual((row["failed"], row["errors"]), (1, 1))
        self.assertIn("AssertionError", row["failure_signature"])

    def test_timeout_keeps_missing_exit_code(self):
        row = self.parse(code=None, error="Command python timed out after 600 seconds", elapsed=600.2)
        self.assertEqual(row["auto_status"], "TIMEOUT")
        self.assertIn(row["exit_code"], (None, ""))

    def test_module_skip_is_zero_collected(self):
        row = self.parse('<testsuite><testcase name="test_example"><skipped message="module skip"/></testcase></testsuite>', code=5)
        self.assertEqual(row["auto_status"], "ZERO_COLLECTED")

    def test_crash_requires_abnormal_exit(self):
        row = self.parse(code=-11, log="Segmentation fault")
        self.assertEqual(row["auto_status"], "CRASH")

    def test_missing_model_overrides_generic_error(self):
        row = self.parse('<testsuite><testcase><error type="LocalEntryNotFoundError" message="model absent"/></testcase></testsuite>', code=2)
        self.assertEqual(row["auto_status"], "INFRA_MODEL")

    def test_oom_overrides_generic_failure(self):
        row = self.parse('<testsuite><testcase><failure type="OutOfMemoryError" message="CUDA out of memory"/></testcase></testsuite>', code=1)
        self.assertEqual(row["auto_status"], "OOM")

    def test_missing_exit_code_never_becomes_pass(self):
        row = self.parse('<testsuite><testcase name="ok"/></testsuite>', code=None)
        self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))
        self.assertIn(row["exit_code"], (None, ""))

    def test_slow_success_and_offline_environment_are_not_failures(self):
        row = self.parse('<testsuite><testcase name="ok"/></testsuite>', elapsed=600.1, log="HF_HUB_OFFLINE=1\nnot out of memory\n1 passed")
        self.assertEqual(row["auto_status"], "PASS")

    def test_malformed_and_entity_junit_never_become_pass(self):
        for xml in ('<testsuite>', '<!DOCTYPE x [<!ENTITY a "expanded">]><testsuite><testcase name="&a;"/></testsuite>'):
            with self.subTest(xml=xml):
                try:
                    row = self.parse(xml)
                except ValueError:
                    continue
                self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))

    def test_junit_path_cannot_escape_artifact_directory(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"), "sweep result parser is missing")
        outside = self.root / "outside.xml"
        outside.write_text('<testsuite><testcase name="ok"/></testsuite>')
        (self.junit / "summary.json").write_text(json.dumps({"selected": ["/workspace/source/" + self.file], "files": [{"file": "/workspace/source/" + self.file, "junit": "../../outside.xml", "log": "case.log", "returncode": 0, "elapsed": 1}]}))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_selected_but_unstarted_file_is_not_executed(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"))
        (self.junit / "summary.json").write_text(json.dumps({"selected": ["/workspace/source/" + self.file], "files": []}))
        row, = baseline.parse_sweep_results(self.root)
        self.assertEqual(row["auto_status"], "NOT_EXECUTED")

    def test_missing_worker_summary_retains_expected_file(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"))
        row, = baseline.parse_sweep_results(self.root)
        self.assertEqual(row["auto_status"], "NOT_EXECUTED")

    def test_missing_log_cannot_produce_pass(self):
        self.parse('<testsuite><testcase name="ok"/></testsuite>')
        (self.junit / "case.log").unlink()
        row, = baseline.parse_sweep_results(self.root)
        self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))

    def test_legacy_metadata_cannot_produce_pass(self):
        self.metadata["mode"] = "legacy"
        (self.directory / "sweep.json").write_text(json.dumps(self.metadata))
        row = self.parse('<testsuite><testcase name="ok"/></testsuite>')
        self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))
        self.assertEqual(row["triage_status"], "RUNNER_INCOMPATIBLE")

    def test_unexpected_file_rejected_instead_of_shared(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"))
        (self.junit / "summary.json").write_text(json.dumps({"selected": [], "files": [{"file": "/workspace/source/test/registered/other.py", "returncode": 0}]}))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_duplicate_record_and_symlink_artifacts_rejected(self):
        self.parse('<testsuite><testcase name="ok"/></testsuite>')
        summary = self.junit / "summary.json"
        data = json.loads(summary.read_text())
        data["files"] *= 2
        summary.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)
        data["files"] = data["files"][:1]
        summary.write_text(json.dumps(data))
        (self.junit / "case.xml").unlink()
        outside = self.root / "payload.xml"
        outside.write_text('<testsuite><testcase name="ok"/></testsuite>')
        (self.junit / "case.xml").symlink_to(outside)
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_oversized_junit_fails_closed(self):
        self.parse('<testsuite><testcase name="ok"/></testsuite>')
        with (self.junit / "case.xml").open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        try:
            row, = baseline.parse_sweep_results(self.root)
        except ValueError:
            return
        self.assertNotIn(row["auto_status"], ("PASS", "PASS_WITH_SKIPS"))

    def test_plan_without_worker_and_all_preblocked_preserves_expected(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"))
        (self.directory / "sweep.json").unlink()
        expected = dict(file=self.file, cuda_suite=self.metadata["suite"], board="810E", n_devices=1, ppu_supports_fp8=0)
        plan = dict(sha=self.metadata["sha"], image_digest=self.metadata["image_digest"], run_id="123", attempt=2,
                    expected=[expected], blocked=[], matrix={"include": []})
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        row, = baseline.parse_sweep_results(self.root)
        self.assertEqual(row["auto_status"], "NOT_EXECUTED")
        plan["blocked"] = [dict(expected, auto_status="INFRA_MODEL", reason="missing model")]
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        row, = baseline.parse_sweep_results(self.root)
        self.assertEqual(row["auto_status"], "INFRA_MODEL")
        self.assertEqual(row["attempt"], 2)

    def test_later_plan_without_worker_keeps_both_attempts(self):
        for attempt in (1, 2):
            directory = self.root / ("plan-%s" % attempt)
            directory.mkdir()
            plan = self.frozen_plan(attempt=attempt)
            (directory / "sweep_plan.json").write_text(json.dumps(plan))
            if attempt == 1:
                self.worker_evidence("worker", plan)
        rows = baseline.parse_sweep_results(self.root)
        self.assertEqual([(r["attempt"], r["auto_status"]) for r in rows], [(1, "PASS"), (2, "NOT_EXECUTED")])
        row, = baseline.build_parity_manifest(plan["expected"], rows)
        self.assertEqual(row["auto_status"], "NOT_EXECUTED")

    def test_unspecified_plan_digest_does_not_conflict_with_worker_digest(self):
        plan = self.frozen_plan()
        plan["image_digest"] = ""
        self.worker_evidence("worker", plan)
        (self.root / "sweep_plan.json").write_text(json.dumps(plan))
        row, = baseline.parse_sweep_results(self.root)
        self.assertEqual(row["auto_status"], "PASS")

    def test_worker_partition_metadata_cannot_drift_from_plan(self):
        self.parse('<testsuite><testcase name="ok"/></testsuite>')
        expected = dict(file=self.file, cuda_suite=self.metadata["suite"], board="810E", n_devices=1, ppu_supports_fp8=0)
        job = dict(key="fixture", suite=self.metadata["suite"], board="810E", n_devices=1, ppu_supports_fp8=0, files=[self.file],
                   partition_id=0, partition_size=1, timeout_per_file=600)
        self.metadata.update(job, timeout_per_file=1)
        (self.directory / "sweep.json").write_text(json.dumps(self.metadata))
        (self.root / "sweep_plan.json").write_text(json.dumps(dict(sha=self.metadata["sha"], image_digest=self.metadata["image_digest"], expected=[expected], blocked=[], matrix={"include": [job]})))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)

    def test_plan_and_worker_selection_must_match(self):
        self.assertTrue(hasattr(baseline, "parse_sweep_results"))
        expected = dict(file=self.file, cuda_suite=self.metadata["suite"], board="890P", n_devices=4)
        (self.root / "sweep_plan.json").write_text(json.dumps(dict(sha=self.metadata["sha"], expected=[expected], blocked=[], matrix={"include": []})))
        with self.assertRaises(ValueError):
            baseline.parse_sweep_results(self.root)


class ParityManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = "test/registered/unit/test_example.py"
        self.suite = "base-a-test-1-gpu-small"
        self.expected = [{"file": self.file, "cuda_suite": self.suite, "board": "810E", "n_devices": 1, "ppu_supports_fp8": 0}]
        self.result = dict(self.expected[0], auto_status="PASS", run_id="123", attempt=1,
                           sha="a" * 40, image_digest="sha256:" + "b" * 64,
                           skipped=0, skipped_nodes=[], mode="pytest")

    def join(self, results, approved=None):
        self.assertTrue(hasattr(baseline, "build_parity_manifest"), "parity join is missing")
        return baseline.build_parity_manifest(self.expected, results, approved_skips=approved, today=date(2026, 10, 29))

    def approval_file(self, **changes):
        row = dict(file=self.file, junit_id="Example.test_skip", cuda_suites=self.suite,
                   reason_code="HW_CAPABILITY", evidence="test/registered/unit/test_example.py:10",
                   reviewer="ci-owner", expires_at="2026-11-01")
        row.update(changes)
        path = self.root / "approved_skips.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_every_expected_file_suite_is_retained(self):
        self.expected.append(dict(self.expected[0], cuda_suite="extra-test-1-gpu-large"))
        rows = self.join([])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["auto_status"] for r in rows}, {"NOT_EXECUTED"})

    def test_same_topology_can_share_evidence_with_explicit_suite_list(self):
        other = "extra-test-1-gpu-large"
        self.expected.append(dict(self.expected[0], cuda_suite=other))
        rows = self.join([self.result])
        self.assertEqual({r["auto_status"] for r in rows}, {"PASS"})
        self.assertIn(other, rows[0]["evidence_shared_with"])
        self.assertIn(self.suite, rows[1]["evidence_shared_with"])

    def test_different_board_or_device_count_cannot_share_evidence(self):
        self.expected.extend([dict(self.expected[0], cuda_suite="extra-test-1-gpu-large", board="890P"),
                              dict(self.expected[0], cuda_suite="extra-test-2-gpu-large", n_devices=2)])
        rows = self.join([self.result])
        self.assertEqual([r["auto_status"] for r in rows], ["PASS", "NOT_EXECUTED", "NOT_EXECUTED"])

    def test_attempts_are_preserved_and_changed_outcome_is_flaky(self):
        rows = self.join([dict(self.result, auto_status="FAIL"), dict(self.result, attempt=2)])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["flaky"])
        self.assertIn("FAIL", str(rows[0]["attempt_history"]))
        self.assertIn("PASS", str(rows[0]["attempt_history"]))

    def test_frozen_sha_and_digest_mismatches_are_not_flaky(self):
        for change in ({"sha": "c" * 40}, {"image_digest": "sha256:" + "d" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.join([self.result, dict(self.result, attempt=2, **change)])

    def test_skip_requires_exact_unexpired_complete_approval(self):
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=["Example.test_skip"])
        row, = self.join([result], self.approval_file())
        self.assertEqual(row["auto_status"], "PASS_WITH_SKIPS")
        self.assertEqual(row["triage_status"], "PASS")
        for change in ({"junit_id": "Example.*"}, {"junit_id": "Example::test_skip"},
                       {"expires_at": "2026-10-28"}, {"reviewer": ""},
                       {"cuda_suites": "extra-test-1-gpu-large"}):
            with self.subTest(change=change):
                row, = self.join([result], self.approval_file(**change))
                self.assertNotEqual(row.get("triage_status"), "PASS")

    def test_one_approved_skip_does_not_approve_other_nodes(self):
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=2,
                      skipped_nodes=["Example.test_skip", "Example.test_unapproved"])
        row, = self.join([result], self.approval_file())
        self.assertNotEqual(row.get("triage_status"), "PASS")

    def test_exact_parameterized_junit_skip_can_be_approved(self):
        node = "unit.Example.test_skip[float8-e4m3]"
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=[node])
        row, = self.join([result], self.approval_file(junit_id=node))
        self.assertEqual(row["triage_status"], "PASS")

    def test_blank_suite_approval_applies_to_all_suites(self):
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=["Example.test_skip"])
        row, = self.join([result], self.approval_file(cuda_suites=""))
        self.assertEqual(row["triage_status"], "PASS")

    def test_skip_approval_requires_allowed_reason_and_exact_node(self):
        for reason in ("NV_DTYPE", "NV_ARCH", "NV_LIB", "NV_INTERCONNECT", "NV_TOOLING", "HW_CAPABILITY"):
            result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=["Example.test_skip"])
            with self.subTest(reason=reason):
                row, = self.join([result], self.approval_file(reason_code=reason))
                self.assertEqual(row["triage_status"], "PASS")
        for node, reason in (("Example.*", "HW_CAPABILITY"), ("Example", "HW_CAPABILITY"),
                             ("test_example.py", "HW_CAPABILITY"), ("Example::test_skip", "HW_CAPABILITY"),
                             ("Example.test_skip", "TYPO")):
            with self.subTest(node=node, reason=reason):
                result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=[node])
                row, = self.join([result], self.approval_file(junit_id=node, reason_code=reason))
                self.assertNotEqual(row["triage_status"], "PASS")

    def test_nvfp4_support_is_not_authorized_by_nv_dtype_skip(self):
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=["Example.test_skip"], skip_reasons=["NVFP4 unsupported"])
        row, = self.join([result], self.approval_file(reason_code="NV_DTYPE", evidence="NVFP4 unsupported"))
        self.assertNotEqual(row["triage_status"], "PASS")

    def test_board_capability_requires_reviewed_supporting_board(self):
        result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=1, skipped_nodes=["Example.test_skip"])
        approval = dict(reason_code="BOARD_CAPABILITY", board="810E", supported_board="890P", evidence="FP8 supported on 890P; reviewed device result")
        row, = self.join([result], self.approval_file(**approval))
        self.assertEqual(row["triage_status"], "PASS")
        self.assertEqual((row["reason_code"], row["supported_board"]), ("BOARD_CAPABILITY", "890P"))
        self.assertIn("890P", row["evidence"])
        for change in ({"supported_board": ""}, {"board": ""}, {"evidence": "unattributed unsupported dtype"}, {"reviewer": ""}, {"expires_at": "2020-01-01"}):
            with self.subTest(change=change):
                row, = self.join([result], self.approval_file(**dict(approval, **change)))
                self.assertNotEqual(row["triage_status"], "PASS")
        self.expected[0].update(board="890P", ppu_supports_fp8=1)
        row, = self.join([dict(result, board="890P", ppu_supports_fp8=1)], self.approval_file(**approval))
        self.assertNotEqual(row["triage_status"], "PASS")

    def test_incomplete_skips_and_zero_collection_cannot_be_approved(self):
        for nodes in ([], ["Example.test_skip"], "Example.test_skip"):
            with self.subTest(nodes=nodes):
                result = dict(self.result, auto_status="PASS_WITH_SKIPS", skipped=2, skipped_nodes=nodes)
                row, = self.join([result], self.approval_file())
                self.assertNotEqual(row["triage_status"], "PASS")
        row, = self.join([dict(self.result, auto_status="ZERO_COLLECTED", skipped=1, skipped_nodes=["Example.test_skip"])], self.approval_file())
        self.assertEqual(row["triage_status"], "RUNNER_INCOMPATIBLE")

    def test_independent_executions_are_not_labelled_shared(self):
        other = "extra-test-1-gpu-large"
        self.expected.append(dict(self.expected[0], cuda_suite=other))
        rows = self.join([self.result, dict(self.result, cuda_suite=other)])
        self.assertEqual([r["evidence_shared_with"] for r in rows], [[], []])

    def test_runtime_capability_mismatch_is_not_overwritten_or_shared(self):
        for value in (None, 1):
            with self.subTest(value=value):
                row, = self.join([dict(self.result, ppu_supports_fp8=value)])
                self.assertNotEqual(row["triage_status"], "PASS")
                self.assertEqual(row["ppu_supports_fp8"], value)
        self.expected.append(dict(self.expected[0], cuda_suite="extra-test-1-gpu-large", ppu_supports_fp8=1))
        rows = self.join([self.result])
        self.assertEqual(rows[1]["auto_status"], "NOT_EXECUTED")
        self.assertEqual(rows[0]["evidence_shared_with"], [])

    def test_legacy_evidence_never_approves_a_file(self):
        row, = self.join([dict(self.result, mode="legacy")])
        self.assertNotEqual(row["auto_status"], "PASS")
        self.assertEqual(row["triage_status"], "RUNNER_INCOMPATIBLE")


if __name__ == "__main__":
    unittest.main()

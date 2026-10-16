"""Cheap PR execution contracts: no torch, model, or accelerator required."""

import ast
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[4]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("ppu_runner", "python/sglang/test/ci/pr_runner.py")
REPORT = load("ppu_report", "scripts/ci/ppu/pr_report.py")
MOUNTS = load("ppu_mounts", "scripts/ci/ppu/prepare_pr_action.py")
MODEL = load("ppu_model", "python/sglang/test/ppu_pr_utils.py")


class StrictRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, source, **options):
        file = self.root / "test_fixture.py"
        file.write_text(source)
        directory = self.root / "results"
        result = RUNNER.run_pytest_files(
            [file], directory, options.pop("timeout", 15), **options
        )
        report = json.loads((directory / "summary.json").read_text())
        return result, report

    def test_pass_and_per_file_evidence(self):
        result, report = self.run_fixture("def test_ok():\n    assert 2 + 2 == 4\n")
        self.assertEqual(result, 0)
        self.assertTrue(report["success"])
        record = report["files"][0]
        self.assertEqual(record["executed"], 1)
        self.assertTrue((self.root / "results" / record["junit"]).is_file())
        self.assertGreater(record["elapsed"], 0)

    def test_failure_retains_counts(self):
        result, report = self.run_fixture("def test_bad():\n    assert False\n")
        self.assertEqual(result, 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["files"][0]["executed"], 1)
        self.assertEqual(report["files"][0]["failures"], 1)

    def test_empty_selection(self):
        self.assertEqual(RUNNER.run_pytest_files([], self.root, 1), 1)
        self.assertFalse(
            json.loads((self.root / "summary.json").read_text())["success"]
        )

    def test_duplicate_selection(self):
        self.assertEqual(RUNNER.run_pytest_files([__file__, __file__], self.root, 1), 1)

    def test_zero_collected(self):
        result, report = self.run_fixture("value = 1\n")
        self.assertEqual(result, 1)
        self.assertEqual(report["files"][0]["executed"], 0)

    def test_all_skipped_is_failure_even_if_allowed(self):
        result, report = self.run_fixture(
            "import pytest\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n",
            allowed_skips=("test_fixture.test_skip",),
        )
        self.assertEqual(result, 1)
        self.assertEqual(report["files"][0]["skipped"], 1)

    def test_explicit_partial_skip_allowlist(self):
        source = "import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n"
        result, report = self.run_fixture(
            source, allowed_skips=("test_fixture.test_skip",)
        )
        self.assertEqual(result, 0)
        self.assertEqual(report["files"][0]["executed"], 1)
        self.assertEqual(report["files"][0]["skipped"], 1)

    def test_unexpected_partial_skip(self):
        result, _ = self.run_fixture(
            "import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n"
        )
        self.assertEqual(result, 1)

    def test_collection_error(self):
        result, report = self.run_fixture("import missing_ppu_fixture_dependency\n")
        self.assertEqual(result, 1)
        self.assertGreater(report["files"][0]["failures"], 0)

    def test_timeout(self):
        result, report = self.run_fixture(
            "import time\ndef test_wait(): time.sleep(60)\n", timeout=1
        )
        self.assertEqual(result, 1)
        self.assertIn("timed out", report["files"][0]["error"])
        self.assertLess(report["elapsed"], 5)

    def test_suite_deadline(self):
        result, report = self.run_fixture("def test_ok(): pass\n", suite_timeout=0)
        self.assertEqual(result, 1)
        self.assertIn("budget exhausted", report["files"][0]["error"])

    def test_missing_report_is_failure(self):
        result, report = self.run_fixture("import os\nos._exit(0)\n")
        self.assertEqual(result, 1)
        self.assertIn("error", report["files"][0])

    def test_stale_report_rejected(self):
        self.run_fixture("def test_ok(): pass\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_fixture("def test_ok(): pass\n")
        self.assertFalse(
            json.loads((self.root / "results/summary.json").read_text())["success"]
        )

    def test_xml_declarations_rejected(self):
        path = self.root / "report.xml"
        for text in ('<!DOCTYPE x [<!ENTITY y "hi">]><testsuite/>', "<testsuite/>\x00"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                RUNNER.inspect_junit(path)

    def test_secrets_are_sanitized(self):
        with patch.dict(
            os.environ, {"PPU_ARTIFACTORY_PASSWORD": "private-fixture-value"}
        ):
            cleaned = RUNNER.sanitize(
                "https://user:pass@host password=private-fixture-value"
            )
        self.assertNotIn("private-fixture-value", cleaned)
        self.assertNotIn("user:pass", cleaned)

    def test_cancellation_kills_descendants_and_stays_red(self):
        # The child inherits pytest's session, just like the serving fixtures.
        heartbeat = self.root / "heartbeat"
        child = (
            "import pathlib,time\np=pathlib.Path(%r)\nwhile True:\n p.write_text(str(time.time())); time.sleep(.05)\n"
            % str(heartbeat)
        )
        fixture = self.root / "test_cancel.py"
        fixture.write_text(
            "import subprocess,sys,time\ndef test_child():\n subprocess.Popen([sys.executable, '-c', %r])\n time.sleep(60)\n"
            % child
        )
        driver = "import importlib.util,sys\ns=importlib.util.spec_from_file_location('r',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nsys.exit(m.run_pytest_files([sys.argv[2]],sys.argv[3],60))\n"
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                driver,
                str(ROOT / "python/sglang/test/ci/pr_runner.py"),
                str(fixture),
                str(self.root / "cancel"),
            ],
            stdout=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while not heartbeat.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(heartbeat.exists())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 1)
            before = heartbeat.read_text()
            time.sleep(0.2)
            self.assertEqual(heartbeat.read_text(), before)
            report = json.loads((self.root / "cancel/summary.json").read_text())
            self.assertFalse(report["success"])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()

    def test_mandatory_evidence_is_revalidated(self):
        suite = REPORT.SUITES["cpu"][0]
        file = self.root / "test_ok.py"
        file.write_text("def test_ok(): pass\n")
        self.assertEqual(RUNNER.run_pytest_files([file], self.root / suite, 15), 0)
        metadata = {
            "worker.json": {"exit_code": 0, "elapsed": 1},
            "setup.json": {"lane": "cpu", "setup_seconds": 1},
            "environment.json": {
                key: "fixture"
                for key in ("commit", "python", "torch", "checkout_import", "backend")
            },
        }
        for name, value in metadata.items():
            (self.root / name).write_text(json.dumps(value))
        self.assertEqual(REPORT.verify(self.root, "cpu")["executed"], 1)
        for name in metadata:
            path = self.root / name
            original = path.read_text()
            path.write_text("{}")
            with self.assertRaises(ValueError):
                REPORT.verify(self.root, "cpu")
            path.write_text(original)
        xml = next((self.root / suite).glob("*.xml"))
        xml.write_text(
            '<testsuite><testcase name="bad"><failure/></testcase></testsuite>'
        )
        with self.assertRaises(ValueError):
            REPORT.verify(self.root, "cpu")


class FinalStatusTest(unittest.TestCase):
    def needs(self, lanes):
        needs = {
            "check-changes": {
                "result": "success",
                "outputs": {
                    key: str(key in lanes).lower()
                    for key in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi")
                },
            },
            "pr-gate": {"result": "success" if lanes else "skipped"},
        }
        for job, lane in REPORT.JOBS.items():
            needs[job] = {
                "result": "success" if lane in lanes else "skipped",
                "outputs": {"evidence_ok": "true"},
            }
        return needs

    def test_only_explicitly_unselected_jobs_may_skip(self):
        for lanes in (
            set(),
            {"cpu"},
            {"cpu", "ppu_single"},
            {"cpu", "ppu_single", "ppu_kernel", "ppu_multi"},
        ):
            needs = self.needs(lanes)
            self.assertTrue(REPORT.aggregate(needs))
            for job, record in needs.items():
                for status in ("failure", "cancelled", "skipped"):
                    if record["result"] == status:
                        continue
                    changed = copy.deepcopy(needs)
                    changed[job]["result"] = status
                    self.assertFalse(REPORT.aggregate(changed), (lanes, job, status))

    def test_missing_evidence_or_selection_is_red(self):
        needs = self.needs({"cpu", "ppu_single"})
        for job in (
            "stage-a-test-cpu-ppu",
            "stage-a-test-1-gpu-ppu",
            "stage-b-test-1-gpu-ppu",
        ):
            changed = copy.deepcopy(needs)
            changed[job]["outputs"] = {}
            self.assertFalse(REPORT.aggregate(changed))
        needs["check-changes"]["outputs"].pop("ppu_multi")
        self.assertFalse(REPORT.aggregate(needs))
        self.assertFalse(REPORT.aggregate(self.needs({"ppu_multi"})))


class SetupContractTest(unittest.TestCase):
    def test_read_only_mount_patch_rendering_and_drift(self):
        entries = (
            "nas-aisw|/nas_aisw|/nas_aisw",
            "wl-nas|/mnt/wl_nas|/wl_nas",
            "wl-nas-direct|/wl_nas|/wl_nas",
        )
        source = (
            "DEFAULT_NAS_VOLUME_MOUNTS=''\nDEFAULT_NAS_VOLUMES=("
            + " ".join(repr(e) for e in entries)
            + ")\n"
        )
        source += 'for _def_entry in "${DEFAULT_NAS_VOLUMES[@]}"; do\n  IFS=\'|\' read -r _vname _mpath _hpath <<< "$_def_entry"\n  DEFAULT_NAS_VOLUME_MOUNTS+="        - name: ${_vname}"$\'\\n\'\n'
        source += (
            MOUNTS.MOUNT_LINE + '\ndone\nprintf "%s" "$DEFAULT_NAS_VOLUME_MOUNTS"\n'
        )
        rendered = subprocess.check_output(
            ["bash", "-c", MOUNTS.constrain(source)], text=True
        )
        mounts = yaml.safe_load(rendered)
        self.assertEqual(len(mounts), 3)
        self.assertTrue(all(m["readOnly"] is True for m in mounts))
        with self.assertRaises(ValueError):
            MOUNTS.constrain(source.replace(MOUNTS.MOUNT_LINE, ""))

    def test_observer_is_inserted_at_resolved_submit_step(self):
        source = "runs:\n  steps:\n    - shell: bash\n" + MOUNTS.SUBMIT_LINE + "\n"
        # Optionally validate the exact pinned checkout without submitting pods.
        checkout = os.environ.get("PPU_ACTION_CONTRACT_SOURCE")
        if checkout:
            directory = Path(checkout)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
                ).strip(),
                MOUNTS.REVISION,
            )
            source = (directory / "action.yml").read_text()
            constrained = MOUNTS.constrain(
                (directory / "scripts/submit.sh").read_text()
            )
            subprocess.run(["bash", "-n"], input=constrained, text=True, check=True)
            self.assertIn("${DEFAULT_NAS_VOLUME_MOUNTS}", constrained)
            start = constrained.index('DEFAULT_NAS_VOLUME_MOUNTS=""')
            end = constrained.index("# --- 循环创建 N 个 Pod ---", start)
            entries = 'DEFAULT_NAS_VOLUMES=("nas-aisw|/nas_aisw|/nas_aisw" "wl-nas|/mnt/wl_nas|/wl_nas" "wl-nas-direct|/wl_nas|/wl_nas")\n'
            rendered = subprocess.check_output(
                [
                    "bash",
                    "-c",
                    entries
                    + constrained[start:end]
                    + '\nprintf "%s" "$DEFAULT_NAS_VOLUME_MOUNTS"',
                ],
                text=True,
            )
            self.assertTrue(all(m["readOnly"] for m in yaml.safe_load(rendered)))
        patched = MOUNTS.observe_before_submit(source)
        action = yaml.safe_load(patched)
        step = next(
            s
            for s in action["runs"]["steps"]
            if "observe_pod_environment.sh" in s.get("run", "")
        )
        self.assertLess(
            step["run"].index("observe_pod_environment.sh"),
            step["run"].index("scripts/submit.sh"),
        )
        self.assertIn('PPU_ENV_JOB_NAME="$JOB_NAME"', step["run"])
        subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
        with self.assertRaises(ValueError):
            MOUNTS.observe_before_submit(patched)

    def test_summary_separates_observed_and_unknown_timings(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"GITHUB_STEP_SUMMARY": ""}
        ):
            root = Path(directory)
            result = REPORT.summarize(root, "smoke")
            self.assertIsNone(result["scheduling_seconds"])
            self.assertEqual(result["aot_cache"], "unknown")
            (root / "setup.json").write_text('{"setup_seconds": 40}')
            (root / "aot.json").write_text('{"aot_seconds": 30}')
            (root / "setup.log").write_text(
                "Wheel cache key: abc123\nPPU sgl-kernel installed from cache (no compile)"
            )
            (root / "environment-pods.timing.tsv").write_text(
                "2026-01-01T00:00:00Z\t2026-01-01T00:00:03Z\t2026-01-01T00:00:11Z\n"
            )
            result = REPORT.summarize(root, "smoke")
            self.assertEqual(result["scheduling_seconds"], 3)
            self.assertEqual(result["pod_startup_seconds"], 8)
            self.assertEqual(result["setup_seconds"], 40)
            self.assertEqual(result["aot_seconds"], 30)
            self.assertEqual(result["aot_cache"], "cache hit")
            self.assertIsNone(result["github_queue_seconds"])

    def test_gsm8k_sampler_has_no_retries_and_retains_errors(self):
        tree = ast.parse(
            (ROOT / "test/registered/ppu/test_ppu_fa3_eval.py").read_text()
        )
        sampler_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_StrictCompletionSampler"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[sampler_node], type_ignores=[]),
                "<sampler contract>",
                "exec",
            ),
            namespace,
        )
        client = Mock()
        client.completions.create.return_value.choices = [Mock(text="42")]
        sampler = namespace["_StrictCompletionSampler"](client, "local-model")
        self.assertEqual(sampler([{"content": "question"}]), "42")
        self.assertEqual(client.completions.create.call_args.kwargs["max_tokens"], 512)
        client.completions.create.side_effect = TimeoutError("request expired")
        with self.assertRaises(TimeoutError):
            sampler([{"content": "question"}])
        self.assertEqual(client.completions.create.call_count, 2)
        self.assertEqual(len(sampler.errors), 1)
        self.assertEqual(len(sampler.completed), 1)

    def test_action_uses_only_pr_writable_mounts(self):
        action = yaml.safe_load(
            (ROOT / ".github/actions/ppu-pr-suite/action.yml").read_text()
        )
        worker = next(s for s in action["runs"]["steps"] if s.get("id") == "worker")
        mounts = worker["with"]["host_volumes"].split(",")
        self.assertEqual(len(mounts), 2)
        self.assertTrue(all(m.startswith("/wl_nas/pr-ci/") for m in mounts))
        self.assertEqual(worker["with"]["node_selector"], "board-type=OAM-810E")
        checkout = next(
            s for s in action["runs"]["steps"] if s.get("with", {}).get("repository")
        )
        self.assertEqual(checkout["with"]["ref"], MOUNTS.REVISION)

    def test_local_model_missing_files_fail_without_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PPU_SMALL_MODEL_PATH": directory}
        ):
            with self.assertRaises(FileNotFoundError):
                MODEL.small_model()
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model.safetensors",
            ):
                (Path(directory) / name).write_text("{}")
            self.assertEqual(MODEL.small_model(), str(Path(directory).resolve()))
            self.assertTrue(MODEL.model_identity()["metadata_sha256"])
            (Path(directory) / "model.safetensors.index.json").write_text(
                '{"weight_map":{"weight":"missing.safetensors"}}'
            )
            with self.assertRaises(FileNotFoundError):
                MODEL.small_model()

    def test_legacy_runner_keeps_default_execution_contract(self):
        # Execute the unchanged legacy branch with mocked infrastructure boundaries;
        # this cheap contract test does not claim runtime import coverage.
        tree = ast.parse((ROOT / "python/sglang/test/ci/ci_utils.py").read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_unittest_files"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        fake = Mock(returncode=0)
        registry = type("CIRegistry", (), {})
        namespace = dict(
            CIRegistry=registry,
            cuda_coredump=Mock(is_enabled=lambda: False),
            time=time,
            os=os,
            subprocess=Mock(Popen=Mock(return_value=fake)),
            logger=Mock(),
            json=json,
            run_with_timeout=lambda f, args, kwargs, timeout: f(*args, **kwargs),
            _repo_relative_path=lambda p: p,
        )
        exec(
            compile(ast.fix_missing_locations(module), "<legacy contract>", "exec"),
            namespace,
        )
        run = namespace["run_unittest_files"]
        self.assertEqual(run([]), 0)
        self.assertEqual(run([], require_nonempty=True), 1)
        self.assertEqual(
            run(
                [type("File", (), {"name": "fixture.py", "estimated_time": 1})()],
                timeout_per_file=1,
            ),
            0,
        )
        namespace["subprocess"].Popen.assert_called_once_with(
            ["python3", os.path.join(os.getcwd(), "fixture.py"), "-f"],
            stdout=None,
            stderr=None,
        )
        with self.assertRaises(ValueError):
            run(
                [],
                pytest_mode=True,
                enable_retry=True,
                junit_dir="fixture",
                timeout_per_file=1,
            )


if __name__ == "__main__":
    unittest.main()

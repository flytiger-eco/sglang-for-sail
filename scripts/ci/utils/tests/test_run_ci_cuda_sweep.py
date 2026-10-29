import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/ppu-cuda-sweep.yml"
WORKER = ROOT / "scripts/ci/ppu/run_cuda_sweep.sh"


class CudaSweepWorkflowTests(unittest.TestCase):
    def test_manual_only_and_serialized_workflow(self):
        self.assertTrue(WORKFLOW.is_file(), "manual sweep workflow is missing")
        workflow = yaml.safe_load(WORKFLOW.read_text())
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        self.assertEqual(workflow["concurrency"]["group"], "ppu-cuda-sweep")
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        inputs = triggers["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["suites"]["default"], "base-a-test-1-gpu-small")
        self.assertEqual(inputs["max_parallel"]["default"], 24)
        self.assertEqual(inputs["board"]["default"], "auto")
        self.assertIn("890P", inputs["board"]["description"])
        self.assertTrue(inputs["ref"]["required"])

    def test_cluster_contract_and_evidence_survive_failure(self):
        self.assertTrue(WORKFLOW.is_file(), "manual sweep workflow is missing")
        workflow = yaml.safe_load(WORKFLOW.read_text())
        job = workflow["jobs"]["sweep"]
        self.assertFalse(job["strategy"]["fail-fast"])
        self.assertIn("max_jobs", job["strategy"]["max-parallel"])
        worker = next(step for step in job["steps"] if step.get("id") == "worker")
        self.assertEqual(
            worker["uses"],
            "flytiger-eco/shared-workflows/cluster-task@4692d51c552e8cdf4f267adfdef1e484ca685a6a",
        )
        self.assertEqual(worker["with"]["nnodes"], 0)
        self.assertEqual(worker["with"]["nproc_per_node"], "${{ matrix.n_devices }}")
        self.assertEqual(worker["with"]["node_selector"], "board-type=${{ matrix.node_selector }}")
        self.assertIn("PPU_BASE_IMAGE_REF", worker["with"]["image"])
        self.assertIn("HF_HUB_OFFLINE=1", worker["with"]["extra_env"])
        self.assertIn("HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub", worker["with"]["extra_env"])
        self.assertIn("PPU_SUPPORTS_FP8=${{ matrix.ppu_supports_fp8 }}", worker["with"]["extra_env"])
        self.assertIn("PPU_SWEEP_BOARD=${{ matrix.board }}", worker["with"]["extra_env"])
        self.assertNotRegex(worker["with"]["extra_env"], r"PPU_SUPPORTS_FP8=[01](?:,|$)")
        export = next(step for step in job["steps"] if step.get("id") == "export")
        self.assertEqual(export["if"], "always()")
        self.assertIn("pr_report.py export", export["run"])
        upload = next(step for step in job["steps"] if "upload-artifact@" in step.get("uses", ""))
        self.assertIn("always()", upload["if"])
        self.assertIn("sanitized", upload["if"])
        self.assertEqual(upload["with"]["if-no-files-found"], "error")

    def test_worker_preserves_cuda_selection_and_dependencies(self):
        self.assertTrue(WORKER.is_file(), "sweep worker is missing")
        script = WORKER.read_text()
        self.assertIn("test/run_suite.py --hw cuda", script)
        for flag in ("--pytest", "--junit-dir", "--continue-on-error", "--timeout-per-file", "--auto-partition-id", "--auto-partition-size"):
            self.assertIn(flag, script)
        for forbidden in ("--suite-timeout", "--enable-retry", "--nightly", "ppu_install_dependency.sh", "pip install", "git config", "register_ppu_ci"):
            self.assertNotIn(forbidden, script)
        self.assertIn("PPU_SWEEP_SHA", script)
        self.assertIn("PPU_BASE_IMAGE_DIGEST", script)
        self.assertIn("sweep.json", script)

    def test_untrusted_dispatch_inputs_are_passed_as_environment(self):
        self.assertTrue(WORKFLOW.is_file(), "manual sweep workflow is missing")
        workflow = yaml.safe_load(WORKFLOW.read_text())
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                self.assertNotIn("${{ inputs.", step.get("run", ""))
        self.assertIn("--sweep-plan", WORKFLOW.read_text())
        self.assertIn("--cache-root", WORKFLOW.read_text())

    def test_worker_handles_extracted_checkout_ownership_without_config_writes(self):
        script = WORKER.read_text()
        exports = "\n".join(line for line in script.splitlines() if line.startswith("export GIT_CONFIG"))
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp).resolve())
            subprocess.run(["git", "init", "-q", root], check=True)
            env = dict(os.environ, ROOT=root, GIT_TEST_ASSUME_DIFFERENT_OWNER="1")
            before = subprocess.run(["git", "-C", root, "status", "--porcelain"], env=env, capture_output=True)
            self.assertNotEqual(before.returncode, 0)
            config = (Path(root) / ".git/config").read_bytes()
            after = subprocess.run(["bash", "-c", exports + '\ngit -C "$ROOT" status --porcelain'], env=env, capture_output=True)
            self.assertEqual(after.returncode, 0, after.stderr.decode())
            self.assertEqual(config, (Path(root) / ".git/config").read_bytes())

    def test_mounted_kubectl_and_occupancy_upload_allowlist(self):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        step = next(s for s in workflow["jobs"]["prepare"]["steps"] if s.get("name") == "Check pinned image and cluster occupancy")
        self.assertIn("/opt/bin/kubectl", step["run"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "mounted/kubectl"
            binary.parent.mkdir()
            pods = {"items": [{"metadata": {"name": "existing-ci"}, "status": {"phase": "Running"}, "spec": {"containers": [{"resources": {"requests": {"alibabacloud.com/ppu": "1"}}, "env": [{"name": "PASSWORD", "value": "must-not-publish"}]}]}}]}
            binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + json.dumps(pods) + "'\n")
            binary.chmod(0o700)
            env = dict(os.environ, PATH="/usr/bin:/bin", PPU_BASE_IMAGE_DIGEST="sha256:" + "b" * 64, PPU_BASE_IMAGE_REF="example@sha256:" + "b" * 64)
            command = step["run"].replace("/opt/bin/kubectl", str(binary)).replace("/opt/bin:", str(binary.parent) + ":")
            result = subprocess.run(["/bin/bash", "-eo", "pipefail", "-c", command], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("defer sweep", result.stderr)
            exported = (root / "sweep-plan/occupancy.json").read_text()
            self.assertNotIn("must-not-publish", exported)
            self.assertEqual(json.loads(exported), {"active_ppu_jobs": ["existing-ci"]})

    def test_worker_binds_provenance_to_actual_execution_parameters(self):
        code = re.search(r"python3 - <<'PY'[^\n]*\n(.*?)\nPY", WORKER.read_text(), re.S).group(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sweep-plan").mkdir()
            (root / "evidence").mkdir()
            (root / "bin").mkdir()
            git = root / "bin/git"
            git.write_text("#!/bin/sh\nprintf '%s\\n' '" + "a" * 40 + "'\n")
            git.chmod(0o700)
            entry = dict(key="dryrun-0", suite="base-a-test-1-gpu-small", board="890P", n_devices=1,
                         ppu_supports_fp8=1, partition_id=0, partition_size=1, timeout_per_file=600,
                         files=["test/registered/unit/test_example.py"])
            plan = dict(sha="a" * 40, image_digest="sha256:" + "b" * 64,
                        run_id="123", attempt=1, matrix={"include": [entry]})
            env = dict(os.environ, PATH=str(git.parent) + os.pathsep + os.environ["PATH"],
                       PPU_EVIDENCE_DIR=str(root / "evidence"), PPU_SWEEP_SHA="a" * 40,
                       PPU_BASE_IMAGE_DIGEST="sha256:" + "b" * 64,
                       PPU_BASE_IMAGE_REF="image@sha256:" + "b" * 64,
                       PPU_SWEEP_RUN_ID="123", PPU_SWEEP_ATTEMPT="2", PPU_SWEEP_KEY=entry["key"],
                       PPU_SWEEP_SUITE=entry["suite"], PPU_SWEEP_N_DEVICES="1",
                       PPU_SWEEP_PARTITION_ID="0", PPU_SWEEP_PARTITION_SIZE="1", PPU_SWEEP_TIMEOUT="600")
            for board, fp8 in (("890P", 1), ("810E", 0)):
                entry.update(board=board, ppu_supports_fp8=fp8)
                (root / "sweep-plan/sweep_plan.json").write_text(json.dumps(plan))
                env.update(PPU_SWEEP_BOARD=board, PPU_SUPPORTS_FP8=str(fp8))
                result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads((root / "evidence/sweep.json").read_text())
                self.assertEqual(metadata.get("mode"), "pytest")
                self.assertEqual(metadata.get("ppu_supports_fp8"), fp8)
                self.assertEqual(metadata.get("plan_attempt"), 1)
                self.assertEqual(metadata["attempt"], 2)
                for key, value in (("PPU_SWEEP_SUITE", "extra-test-1-gpu-large"),
                                   ("PPU_SWEEP_N_DEVICES", "2"), ("PPU_SWEEP_PARTITION_ID", "1"),
                                   ("PPU_SWEEP_PARTITION_SIZE", "2"), ("PPU_SWEEP_TIMEOUT", "601"),
                                   ("PPU_SWEEP_RUN_ID", "456"), ("PPU_SWEEP_ATTEMPT", "0"),
                                   ("PPU_SWEEP_BOARD", "810E" if board == "890P" else "890P"),
                                   ("PPU_SUPPORTS_FP8", str(1 - fp8))):
                    with self.subTest(board=board, key=key):
                        failed = subprocess.run([sys.executable, "-c", code], cwd=root,
                                                env=dict(env, **{key: value}), capture_output=True, text=True)
                        self.assertNotEqual(failed.returncode, 0, "accepted execution metadata mismatch: " + key)
                entry["ppu_supports_fp8"] = 1 - fp8
                (root / "sweep-plan/sweep_plan.json").write_text(json.dumps(plan))
                failed = subprocess.run([sys.executable, "-c", code], cwd=root,
                                        env=dict(env, PPU_SUPPORTS_FP8=str(1 - fp8)), capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0, "accepted board-inconsistent plan and runtime")

    def test_environment_records_actual_partial_card_visibility(self):
        code = re.findall(r"python3 - <<'PY'[^\n]*\n(.*?)\nPY", WORKER.read_text(), re.S)[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prelude = (
                "import sys\nfrom types import SimpleNamespace\n"
                "sys.modules['sglang'] = SimpleNamespace(__file__=" + repr(str(root / "python/sglang/__init__.py")) + ")\n"
                "sys.modules['torch'] = SimpleNamespace(__version__='fixture', cuda=SimpleNamespace("
                "is_available=lambda: True, device_count=lambda: int(sys.argv[1]), get_device_name=lambda i: 'fixture-device'))\n"
            )
            env = dict(os.environ, PPU_SWEEP_BOARD="890P", PPU_SUPPORTS_FP8="1", PPU_SWEEP_N_DEVICES="1")
            result = subprocess.run([sys.executable, "-O", "-c", prelude + code, "1"], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(result.stdout)
            self.assertEqual(evidence.get("requested_devices"), 1)
            self.assertEqual(evidence.get("visible_devices"), 1)
            self.assertEqual(evidence.get("ppu_supports_fp8"), 1)
            self.assertEqual(evidence.get("board"), "890P")
            mismatch = subprocess.run([sys.executable, "-O", "-c", prelude + code, "2"], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0, "partial allocation check must not disappear under Python optimization")

    def test_pre_runner_failures_keep_diagnostic_logs(self):
        script = WORKER.read_text()
        self.assertRegex(script, r"python3 - <<'PY'.*preflight\.log.*2>&1")
        self.assertRegex(script, r"python3 - <<'PY'.*environment\.json.*2>.*environment\.log")

    def test_partial_rerun_downloads_the_producing_attempt_plan(self):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        prepare = workflow["jobs"]["prepare"]
        self.assertEqual(prepare["outputs"].get("plan_artifact"), "${{ steps.plan.outputs.artifact_name }}")
        download = next(step for step in workflow["jobs"]["sweep"]["steps"] if "download-artifact@" in step.get("uses", ""))
        self.assertEqual(download["with"]["name"], "${{ needs.prepare.outputs.plan_artifact }}")
        step = next(s for s in prepare["steps"] if s.get("id") == "plan")
        code = re.search(r"python3 - <<'PY'[^\n]*\n(.*?)\nPY", step["run"], re.S).group(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sweep-plan").mkdir()
            (root / "sweep-plan/sweep_plan.json").write_text(json.dumps({"matrix": {"include": []}, "max_jobs": 16}))
            env = dict(os.environ, GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1", GITHUB_OUTPUT=str(root / "output"))
            subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, capture_output=True)
            outputs = dict(line.split("=", 1) for line in (root / "output").read_text().splitlines())
            self.assertEqual(outputs["artifact_name"], "ppu-cuda-sweep-plan-123-1")
            self.assertNotEqual(outputs["artifact_name"], "ppu-cuda-sweep-plan-123-2")


if __name__ == "__main__":
    unittest.main()

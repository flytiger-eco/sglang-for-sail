"""CPU-only regression fixtures for the static CUDA/PPU CI inventory."""

import csv
import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "cuda_ci_baseline", ROOT / "scripts/ci/ppu/cuda_ci_baseline.py"
)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)

PR_SCRIPT = """\
#!/usr/bin/env bash
set -euo pipefail
lane="${1:?cpu|smoke|body|multi}"
case "$lane" in
    cpu) hw=cpu; suite=stage-a-test-cpu-ppu; cap=300 ;;
    smoke) hw=ppu; suite=stage-a-test-1-gpu-ppu; cap=300 ;;
    body) hw=ppu; suite=stage-b-test-1-gpu-ppu; cap=1080 ;;
    multi) hw=ppu; suite=stage-c-test-2-gpu-ppu; cap=600 ;;
    *) exit 2 ;;
esac
source scripts/ci/ppu/not-invoked.sh
python3 test/run_suite.py --hw "$hw" --suite "$suite" --require-nonempty \\
    --timeout-per-file "$cap"
if [[ "$lane" == body && "${PPU_RUN_KERNEL:-false}" == true ]]; then
    python3 test/run_suite.py --hw ppu --suite stage-b-kernel-test-1-gpu-ppu --require-nonempty
fi
"""


class BaselineFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.write(".github/workflows/.keep", "")
        self.write("scripts/ci/runner_configs.yml", "runner_configs: {}\n")
        self.write(
            "test/run_suite.py",
            "PER_COMMIT_SUITES = {HWBackend.CUDA: [], HWBackend.PPU: []}\n"
            "NIGHTLY_SUITES = {HWBackend.CUDA: [], HWBackend.PPU: []}\n",
        )
        self.write("scripts/ci/ppu/run_pr_suite.sh", PR_SCRIPT)
        for name, value in (
            ("REPO", str(self.repo)),
            ("WORKFLOW_DIR", str(self.repo / ".github/workflows")),
        ):
            patch = mock.patch.object(baseline, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        # These unrelated report sections need neither git nor a partition subprocess.
        for name, value in (
            ("git_head", ("fixture", "date", "branch")),
            ("diffusion_plan", {}),
        ):
            patch = mock.patch.object(baseline, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content))

    def workflow(self, name, content):
        self.write(f".github/workflows/{name}", content)

    def registered(self, backend, suite):
        self.write(
            f"test/registered/test_{suite}.py",
            f'register_{backend}_ci(est_time=1, suite="{suite}")\n',
        )

    def build(self):
        result = baseline.build(str(self.repo / "out"))
        self.assertEqual(
            json.loads((self.repo / "out/summary.json").read_text())["cuda"],
            result["cuda"],
        )
        return result

    def entries(self):
        return baseline.scan_workflows(baseline.load_workflows(), {})

    def cuda_leaf(self, suite):
        self.registered("cuda", suite)
        self.workflow(
            "leaf.yml",
            f"""
            on: workflow_call
            jobs:
              test:
                runs-on: 1-gpu-h100
                steps:
                  - run: python3 test/run_suite.py --hw cuda --suite {suite}
            """,
        )

    def test_cuda_cadence_uses_transitive_root_and_trigger_evidence(self):
        for root, trigger, suite, expected in (
            (
                "pr-test.yml",
                "pull_request: {}",
                "base-a-test-1-gpu-small",
                "per_commit",
            ),
            (
                "pr-test-extra.yml",
                "pull_request: {}",
                "extra-a-test-1-gpu-large",
                "extra",
            ),
            (
                "nightly-test-nvidia.yml",
                "schedule: [{cron: '0 14 */2 * *'}]",
                "nightly-test-1-gpu",
                "nightly",
            ),
        ):
            with self.subTest(root=root):
                self.cuda_leaf(suite)
                self.workflow(
                    root,
                    f"on:\n  {trigger}\njobs:\n  call:\n    uses: ./.github/workflows/middle.yml\n",
                )
                self.workflow(
                    "middle.yml",
                    "on: workflow_call\njobs:\n  call:\n    uses: ./.github/workflows/leaf.yml\n",
                )
                meta = self.build()["cuda"]["by_suite"][suite]
                self.assertEqual(meta["cadence"], expected)
                self.assertEqual(meta.get("cadence_verification"), "verified")
                evidence = meta.get("cadence_evidence", [])
                self.assertTrue(
                    any(e["workflow"] == root and e["triggers"] for e in evidence)
                )
                if expected == "nightly":
                    self.assertTrue(
                        any(e["crons"] == ["0 14 */2 * *"] for e in evidence)
                    )
                (self.repo / ".github/workflows" / root).unlink()

    def test_cuda_unverified_cadences_are_reported_as_anomalies(self):
        suite = "nightly-test-1-gpu"
        self.cuda_leaf(suite)
        cases = (
            (None, None, "missing_root"),
            ("nightly-test-nvidia.yml", "workflow_dispatch: {}", "dispatch_only"),
            ("pr-test.yml", "pull_request: {}", "mismatch"),
            ("unclassified.yml", "repository_dispatch: {}", "unknown"),
            ("nightly-test-nvidia.yml", "schedule: []", "unknown"),
        )
        for root, trigger, status in cases:
            with self.subTest(status=status, trigger=trigger):
                if root:
                    self.workflow(
                        root,
                        f"on:\n  {trigger}\njobs:\n  call:\n    uses: ./.github/workflows/leaf.yml\n",
                    )
                result = self.build()
                meta = result["cuda"]["by_suite"][suite]
                self.assertEqual(meta["cadence"], "nightly")
                self.assertEqual(meta.get("cadence_verification"), status)
                self.assertIn(
                    suite,
                    result["anomalies"]
                    .get("cuda_cadence_verification", {})
                    .get(status, []),
                )
                if root:
                    (self.repo / ".github/workflows" / root).unlink()

    def test_reports_distinguish_nominal_cadence_from_verification(self):
        self.cuda_leaf("nightly-test-1-gpu")
        self.build()
        markdown = (self.repo / "out/summary.md").read_text()
        self.assertIn("Nominal cadence", markdown)
        self.assertIn("missing_root", markdown)
        with (self.repo / "out/registrations.csv").open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row["cadence_verification"], "missing_root")
        self.assertEqual(json.loads(row["cadence_evidence"]), [])

    def test_ppu_summary_includes_invocation_evidence(self):
        suite = "stage-a-test-1-gpu-ppu"
        self.registered("ppu", suite)
        self.workflow(
            "pr-test-ppu.yml",
            """
            on: pull_request
            jobs:
              test:
                runs-on: ppu
                steps:
                  - run: bash scripts/ci/ppu/run_pr_suite.sh smoke
        """,
        )
        meta = self.build()["ppu"]["by_suite"][f"other|{suite}"]
        self.assertEqual(meta.get("reachable_from"), "pr-test-ppu.yml")
        self.assertTrue(
            any("run_pr_suite.sh" in e["script"] for e in meta.get("invocations", []))
        )

    def test_missing_reusable_workflow_is_not_dispatch_evidence(self):
        suite = "base-a-test-1-gpu-small"
        self.registered("cuda", suite)
        self.workflow(
            "pr-test.yml",
            f"""
            on: pull_request
            jobs:
              test:
                uses: ./.github/workflows/_pr-test-stage.yml
                with:
                  self_name: {suite}
                  runner_config: 1-gpu-small
        """,
        )
        meta = self.build()["cuda"]["by_suite"][suite]
        self.assertEqual(meta["reachable_from"], "")
        self.assertEqual(meta.get("cadence_verification"), "missing_root")

    def test_pr_shell_resolves_only_invoked_lane(self):
        for lane, hw, suite in (
            ("cpu", "cpu", "stage-a-test-cpu-ppu"),
            ("smoke", "ppu", "stage-a-test-1-gpu-ppu"),
            ("body", "ppu", "stage-b-test-1-gpu-ppu"),
            ("multi", "ppu", "stage-c-test-2-gpu-ppu"),
        ):
            with self.subTest(lane=lane):
                self.workflow(
                    "pr-test-ppu.yml",
                    f"""
                    on: pull_request
                    jobs:
                      worker:
                        runs-on: ppu
                        steps:
                          - run: bash scripts/ci/ppu/run_pr_suite.sh {lane}
                """,
                )
                entries = [e for e in self.entries() if e["kind"] == "registered"]
                self.assertEqual(
                    [(e["hw"], e["suite"]) for e in entries], [(hw, suite)]
                )
                self.assertEqual(entries[0]["reachable_from"], ["pr-test-ppu.yml"])
                self.assertIn("run_pr_suite.sh", entries[0].get("script", ""))

    def test_action_command_resolves_optional_kernel_as_conditional(self):
        self.workflow(
            "pr-test-ppu.yml",
            """
            on: workflow_call
            jobs:
              worker:
                runs-on: ppu
                steps:
                  - uses: flytiger-eco/shared-workflows/cluster-task@fixture
                    with:
                      command: bash /workspace/source/scripts/ci/ppu/run_pr_suite.sh body
                      extra_env: PPU_RUN_KERNEL=${{ inputs.kernel }}
        """,
        )
        self.workflow(
            "ci.yaml",
            "on: pull_request\njobs:\n  ppu:\n    uses: ./.github/workflows/pr-test-ppu.yml\n",
        )
        entries = [e for e in self.entries() if e["hw"] == "ppu"]
        self.assertEqual(
            {e["suite"] for e in entries},
            {"stage-b-test-1-gpu-ppu", "stage-b-kernel-test-1-gpu-ppu"},
        )
        kernel = next(e for e in entries if "kernel" in e["suite"])
        self.assertIn("PPU_RUN_KERNEL", kernel.get("condition", ""))
        self.assertEqual(kernel["reachable_from"], ["ci.yaml"])

    def test_uncalled_script_and_suite_prose_do_not_imply_reachability(self):
        suite = "stage-b-test-1-gpu-ppu"
        self.registered("ppu", suite)
        self.write(
            "scripts/ci/ppu/not-invoked.sh",
            f"python3 test/run_suite.py --hw ppu --suite {suite}\n",
        )
        self.write("scripts/ci/ppu/cuda_ci_baseline.py", f'EXAMPLE = "{suite}"\n')
        self.workflow(
            "pr-test-ppu.yml",
            f"""
            on: pull_request
            jobs:
              prose:
                runs-on: ppu
                steps:
                  - name: {suite}
                    run: |
                      # python3 test/run_suite.py --hw ppu --suite {suite}
                      echo 'python3 test/run_suite.py --hw ppu --suite {suite}'
                      echo 'bash scripts/ci/ppu/run_pr_suite.sh body'
                  - run: bash scripts/ci/ppu/run_pr_suite.sh smoke
        """,
        )
        result = self.build()
        self.assertIn(
            suite,
            result["anomalies"].get("ppu_suites_without_reachable_invocation", []),
        )
        self.assertFalse(
            any(e["suite"] == suite and e["reachable_from"] for e in self.entries())
        )

    def test_call_only_ppu_workflow_has_no_reachable_root(self):
        self.workflow(
            "unused.yml",
            """
            on: workflow_call
            jobs:
              test:
                runs-on: ppu
                steps:
                  - run: bash scripts/ci/ppu/run_pr_suite.sh smoke
        """,
        )
        entries = [e for e in self.entries() if e["hw"] == "ppu"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["reachable_from"], [])

    def test_variable_nightly_suite_in_invoked_board_script(self):
        self.write(
            "scripts/ci/ppu/run_answer_suite_board.sh",
            'python3 test/run_suite.py --hw ppu --suite "$ANSWER_SUITE" --nightly\n',
        )
        self.workflow(
            "test-ppu-answer.yml",
            """
            on: workflow_dispatch
            jobs:
              test:
                runs-on: ppu
                steps:
                  - uses: flytiger-eco/shared-workflows/cluster-task@fixture
                    with:
                      command: bash /workspace/source/scripts/ci/ppu/run_answer_suite_board.sh
                      extra_env: ANSWER_SUITE=nightly-answer-1-ppu,OTHER=value
        """,
        )
        entries = [e for e in self.entries() if e["hw"] == "ppu"]
        self.assertEqual([e["suite"] for e in entries], ["nightly-answer-1-ppu"])
        self.assertEqual(entries[0]["reachable_from"], ["test-ppu-answer.yml"])

    def test_variable_nightly_suite_in_node_script_from_matrix(self):
        self.write(
            "scripts/ci/ppu/run_answer_suite_node.sh",
            'suite="${1:-}"\npython3 run_suite.py --hw ppu --suite "$suite" --nightly\n',
        )
        self.workflow(
            "test-ppu-answer-16.yml",
            """
            on: workflow_dispatch
            jobs:
              test:
                strategy:
                  matrix:
                    include:
                      - suite: nightly-answer-16-ppu
                      - suite: nightly-answer-16-kimi26-ppu
                env:
                  ANSWER_SUITE: ${{ matrix.suite }}
                runs-on: ppu
                steps:
                  - uses: flytiger-eco/ppu-distributed-action@fixture
                    with:
                      command: |
                        cd test
                        bash ../scripts/ci/ppu/run_answer_suite_node.sh \\
                          "${{ env.ANSWER_SUITE }}" 7200 /tmp/logs
        """,
        )
        entries = [e for e in self.entries() if e["hw"] == "ppu"]
        self.assertEqual(
            {e["suite"] for e in entries},
            {"nightly-answer-16-ppu", "nightly-answer-16-kimi26-ppu"},
        )

    def test_direct_variable_suite_in_docker_command(self):
        self.workflow(
            "nightly-test-ppu.yml",
            """
            on:
              schedule: [{cron: '0 16 * * *'}]
            jobs:
              test:
                runs-on: ppu
                env:
                  SUITE: nightly-1-ppu
                steps:
                  - run: |
                      docker run image bash -c "
                        cd test && \\
                        python3 run_suite.py --hw ppu --suite \\"$SUITE\\" --nightly
                      "
        """,
        )
        entries = [e for e in self.entries() if e["hw"] == "ppu"]
        self.assertEqual([e["suite"] for e in entries], ["nightly-1-ppu"])


class RepositoryWorkflowContract(unittest.TestCase):
    def test_all_five_nightly_ppu_calls_require_nonempty(self):
        with mock.patch.object(
            baseline, "WORKFLOW_DIR", str(ROOT / ".github/workflows")
        ):
            workflow = baseline.load_workflows()["nightly-test-ppu.yml"]
        calls = [
            line.strip()
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            for line in step.get("run", "").splitlines()
            if "python3 run_suite.py --hw ppu" in line
        ]
        self.assertEqual(len(calls), 5)
        self.assertTrue(all("--require-nonempty" in line for line in calls), calls)


if __name__ == "__main__":
    unittest.main()

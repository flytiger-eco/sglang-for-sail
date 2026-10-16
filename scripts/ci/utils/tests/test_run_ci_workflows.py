import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


class RunCiWorkflowContractTest(unittest.TestCase):
    def read_workflow(self, name):
        return (ROOT / ".github/workflows" / name).read_text()

    def test_slash_handler_executes_default_branch_code(self):
        workflow = self.read_workflow("slash-command-handler.yml")

        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertNotIn("refs/pull/", workflow)
        self.assertNotIn("headRefName", workflow)

    def test_ci_pipeline_uses_event_aware_run_ci_gate(self):
        workflow = self.read_workflow("ci.yaml")

        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, labeled]",
            workflow,
        )
        self.assertIn("needs: registration-validation", workflow)
        self.assertIn(
            "contains(github.event.pull_request.labels.*.name, 'run-ci')",
            workflow,
        )
        self.assertIn(
            "github.event.action != 'labeled' || "
            "github.event.label.name == 'run-ci'",
            workflow,
        )
        self.assertNotIn("require_run_ci_label: false", workflow)
        self.assertIn("github.event.action == 'labeled'", workflow)
        self.assertIn("github.event.label.name != 'run-ci'", workflow)

    def test_ppu_workflow_retains_defense_in_depth_label_gate(self):
        workflow = yaml.safe_load(self.read_workflow("pr-test-ppu.yml"))
        gate = workflow["jobs"]["pr-gate"]
        self.assertEqual(gate["uses"], "./.github/workflows/pr-gate.yml")
        self.assertEqual(
            gate["with"]["require-run-ci"], "${{ inputs.require_run_ci_label }}"
        )
        self.assertEqual(
            gate["if"], "needs.check-changes.outputs.changes_exist == 'true'"
        )
        stage = workflow["jobs"]["stage-a-test-1-gpu-ppu"]
        self.assertIn("pr-gate", stage["needs"])
        self.assertIn("needs.pr-gate.result == 'success'", stage["if"])


class PpuPathSelectionTest(unittest.TestCase):
    """Exercise the YAML rules with both actions' actual glob engines.

    Requires PyYAML and Node with picomatch@2.3.1 and minimatch@9.0.5
    resolvable through NODE_PATH; ci.yaml installs these on its CPU runner.
    """

    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        )
        cls.check = cls.workflow["jobs"]["check-changes"]
        cls.filter_step = next(
            step for step in cls.check["steps"] if step.get("id") == "filter"
        )
        cls.filters = yaml.safe_load(cls.filter_step["with"]["filters"])
        cls.labeler = yaml.safe_load((ROOT / ".github/labeler.yml").read_text())

    def match_paths(self, paths, execution=False):
        # Do not emulate extglob with Python fnmatch: its semantics differ.
        script = textwrap.dedent("""
            const fs = require('node:fs');
            const picomatch = require('picomatch');
            const {minimatch} = require('minimatch');
            const {paths, filters, labels, quantifier} = JSON.parse(fs.readFileSync(0, 'utf8'));
            const result = {
                filters: Object.entries(filters).filter(([name, patterns]) =>
                    paths.some(path => patterns[quantifier](pattern =>
                        picomatch(pattern, {dot: true})(path)
                    ))
                ).map(([name]) => name).sort(),
                labels: Object.entries(labels).filter(([name, rule]) => {
                    const all = rule['all-globs-to-any-file'];
                    const patterns = all || rule['any-glob-to-any-file'];
                    return paths.some(path => patterns[all ? 'every' : 'some'](pattern =>
                        minimatch(path, pattern, {dot: true})
                    ));
                }).map(([name]) => name).sort(),
            };
            process.stdout.write(JSON.stringify(result));
            """)
        labels = {
            name: rules[0]["changed-files"][0]
            for name, rules in self.labeler.items()
            if name.startswith("ppu-") or name == "documentation"
        }
        result = subprocess.run(
            ["node", "-e", script],
            input=json.dumps(
                {
                    "paths": paths,
                    "filters": self.filters,
                    "labels": labels,
                    "quantifier": self.filter_step["with"].get(
                        "predicate-quantifier", "some"
                    ),
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        matched = json.loads(result.stdout)
        if not execution:
            matched["filters"] = [
                g
                for g in matched["filters"]
                if g in ("ppu_runtime", "ppu_tests", "ppu_ci")
            ]
        return matched

    def lanes(self, paths, force=False):
        matches = self.match_paths(paths, execution=True)["filters"]
        selected = set()
        for lane in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi"):
            expression = self.check["outputs"][lane]
            # Evaluate the actual OR-only output contract, not a separate router.
            terms = expression.removeprefix("${{ ").removesuffix(" }}").split(" || ")
            for term in terms:
                if term == "steps.run-mode.outputs.run_all_tests == 'true'":
                    value = force
                else:
                    prefix, suffix = "steps.filter.outputs.", " == 'true'"
                    self.assertTrue(term.startswith(prefix) and term.endswith(suffix))
                    value = term[len(prefix) : -len(suffix)] in matches
                if value:
                    selected.add(lane)
        return selected

    def test_docs_inside_code_and_tests_do_not_select_ppu(self):
        # Previously test/registered/** matched README edits and ran all stages.
        paths = [
            f"{directory}/{filename}"
            for directory in (
                ".",
                "docs/inference",
                "python/sglang",
                "python/sglang/srt/hardware_backend/ppu",
                "python/sglang/kernels/aot",
                "test/registered/ppu/answer_eval",
                "scripts/ci/ppu",
                ".github/actions/example",
            )
            for filename in (
                "guide.md",
                "guide.MD",
                "guide.mDx",
                "guide.RsT",
                "README.txt",
                "LICENSE",
            )
        ]
        paths = [path.removeprefix("./") for path in paths]
        self.assertEqual(
            self.match_paths(paths), {"filters": [], "labels": ["documentation"]}
        )

    def test_relevant_runtime_tests_and_build_inputs_are_selected(self):
        cases = {
            "ppu_runtime": (
                "python/sglang/__init__.py",
                "python/sglang/srt/platforms/ppu.py",
                "python/sglang/srt/managers/scheduler.py",
                "python/sglang/kernels/aot/csrc/attention/kernel.cu",
                "python/sglang/kernels/aot/CMakeLists.txt",
                "python/sglang/kernels/aot/requirements.txt",
            ),
            "ppu_tests": (
                "test/run_suite.py",
                "test/registered/ppu/test_ppu_basic.py",
                "test/registered/ppu/accuracy/configs/model.json",
                "test/registered/ppu/fixtures/prompt.txt",
                "test/registered/ppu/config.yaml",
            ),
            "ppu_ci": (
                ".github/workflows/ci.yaml",
                ".github/workflows/pr-test-ppu.yml",
                ".github/workflows/pr-gate.yml",
                ".github/workflows/labeler.yml",
                ".github/labeler.yml",
                ".github/actions/example/action.yml",
                "scripts/ci/ppu/ppu_install_dependency.sh",
                "scripts/ci/utils/common.py",
                "python/pyproject.toml",
                "python/pyproject_other.toml",
                "python/pyproject_ppu.toml",
                "python/setup.cfg",
                "requirements.txt",
                "python/requirements-dev.in",
            ),
        }
        for group, paths in cases.items():
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(
                        self.match_paths([path]),
                        {"filters": [group], "labels": [group.replace("_", "-")]},
                    )

    def test_unrelated_components_and_empty_diff_do_not_select_ppu(self):
        for paths in (
            [],
            ["sgl-model-gateway/src/main.rs"],
            ["python/sglang/multimodal_gen/runtime/pipeline.py"],
            ["python/sglang/multimodal_gen/__init__.py"],
            [".github/workflows/pr-test-xpu.yml"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(self.match_paths(paths), {"filters": [], "labels": []})

    def test_mixed_documentation_and_code_still_selects_tests(self):
        self.assertEqual(
            self.match_paths(["README.md", "test/registered/ppu/test_ppu_basic.py"]),
            {"filters": ["ppu_tests"], "labels": ["documentation", "ppu-tests"]},
        )
        self.assertEqual(
            self.match_paths(
                ["python/sglang/srt/platforms/ppu.py", "python/pyproject_other.toml"]
            )["filters"],
            ["ppu_ci", "ppu_runtime"],
        )

    def test_label_and_selector_rules_stay_in_sync(self):
        for group in ("ppu_runtime", "ppu_tests", "ppu_ci"):
            patterns = self.filters[group]
            with self.subTest(group=group):
                rules = self.labeler[group.replace("_", "-")]
                self.assertEqual(
                    rules[0]["changed-files"][0]["all-globs-to-any-file"], patterns
                )
        self.assertNotIn("run-ci", self.labeler)

    def test_path_selection_controls_every_stage_and_preserves_force_option(self):
        selector = self.check["outputs"]["changes_exist"]
        terms = [
            "steps.filter.outputs.cpu == 'true'",
            "steps.run-mode.outputs.run_all_tests == 'true'",
        ]
        self.assertEqual(selector, "${{ " + " || ".join(terms) + " }}")
        self.assertEqual(
            self.filter_step["if"], "steps.run-mode.outputs.run_all_tests != 'true'"
        )
        # BaseLoader preserves the YAML 1.2 key 'on' as a string.
        events = yaml.load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text(),
            Loader=yaml.BaseLoader,
        )["on"]
        for event in ("workflow_dispatch", "workflow_call"):
            self.assertEqual(
                events[event]["inputs"]["run_all_tests"]["type"], "boolean"
            )
            self.assertEqual(
                events[event]["inputs"]["run_all_tests"]["default"], "false"
            )
        for stage, lane in (
            ("stage-a-test-cpu-ppu", "cpu"),
            ("stage-a-test-1-gpu-ppu", "ppu_single"),
            ("stage-b-test-1-gpu-ppu", "ppu_single"),
            ("stage-c-test-2-gpu-ppu", "ppu_multi"),
        ):
            self.assertIn(
                f"needs.check-changes.outputs.{lane} == 'true'",
                self.workflow["jobs"][stage]["if"],
            )
        finish = self.workflow["jobs"]["pr-test-ppu-finish"]
        self.assertEqual(finish["if"], "always()")
        self.assertIn("pr_report.py aggregate", finish["steps"][-1]["run"])

    def test_execution_routing_and_rename_union(self):
        cpu = {"cpu"}
        single = cpu | {"ppu_single"}
        kernel = single | {"ppu_kernel"}
        multi = kernel | {"ppu_multi"}
        cases = [
            (["test/registered/unit/server_args/test_server_args.py"], cpu),
            (["python/sglang/srt/parser/reasoning_parser.py"], cpu),
            (["test/ppu/module/test_ppu_deepep_hooks.py"], cpu),
            (
                [
                    "test/registered/ppu/pr/test_cpu_dispatch.py",
                    "python/sglang/srt/new_module.py",
                ],
                single,
            ),
            (["python/sglang/srt/new_module.py", "README.md"], single),
            (["python/sglang/srt/layers/attention/new.py"], kernel),
            (["python/sglang/srt/mem_cache/radix_cache.py"], kernel),
            (["sgl-kernel/csrc/attention.cu"], kernel),
            (["python/sglang/srt/hardware_backend/ppu/moe/ppu_deepep_hooks.py"], multi),
            (["python/sglang/srt/distributed/parallel_state.py"], multi),
            (["python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh"], multi),
            (["test/registered/ppu/pr/test_two_ppu.py"], multi),
            (["test/run_suite.py"], multi),
            (["python/sglang/test/ci/ci_register.py"], multi),
            (["python/pyproject_other.toml"], multi),
            (["docker/Dockerfile.ppu"], multi),
            ([".github/actions/ppu-pr-suite/action.yml"], multi),
            (["python/sglang/kernels/ops/diffusion/attention.py"], set()),
            (["test/registered/diffusion/test_model.py"], set()),
            (["sgl-model-gateway/src/main.rs"], set()),
            (["python/sglang/srt/distributed/deep/README.MD"], set()),
            ([], set()),
        ]
        for paths, expected in cases:
            with self.subTest(paths=paths):
                self.assertEqual(self.lanes(paths), expected)
        # paths-filter v3 represents renames as Added(new) + Deleted(previous).
        # Both directions must retain the union, including a deleted protected file.
        protected = "python/sglang/srt/distributed/old.py"
        for previous, current in (
            (protected, "docs/moved.py"),
            ("docs/moved.py", protected),
        ):
            self.assertEqual(self.lanes([current, previous]), multi)
        self.assertEqual(self.lanes([protected]), multi)  # deletion
        self.assertEqual(self.lanes(["README.md"], force=True), multi)
        for patterns in self.filters.values():
            self.assertEqual(sum(not p.startswith("!") for p in patterns), 1)

    def test_cuda_style_dag_and_caps(self):
        jobs = self.workflow["jobs"]
        cpu, smoke = "stage-a-test-cpu-ppu", "stage-a-test-1-gpu-ppu"
        body, multi = "stage-b-test-1-gpu-ppu", "stage-c-test-2-gpu-ppu"
        for job in (body, multi):
            self.assertEqual(set(jobs[job]["needs"]), {"check-changes", cpu, smoke})
            for prerequisite in (cpu, smoke):
                self.assertIn(
                    f"needs.{prerequisite}.result == 'success'", jobs[job]["if"]
                )
            self.assertIn("!cancelled()", jobs[job]["if"])
        for job, cap in ((cpu, 10), (smoke, 12), (body, 25), (multi, 20)):
            self.assertEqual(jobs[job]["timeout-minutes"], cap)
            self.assertIn("evidence_ok", jobs[job]["outputs"])
            self.assertIn(job, jobs["pr-test-ppu-finish"]["needs"])
        for job in (smoke, body, multi):
            self.assertEqual(jobs[job]["runs-on"], "k8s-runner-group-cpu-flytiger")
            action = next(s for s in jobs[job]["steps"] if s.get("id") == "suite")
            self.assertEqual(action["uses"], "./.github/actions/ppu-pr-suite")
            self.assertLess(
                action["with"]["worker-minutes"], jobs[job]["timeout-minutes"]
            )
        self.assertEqual(jobs[cpu]["runs-on"], "ubuntu-latest")

    def test_summary_reports_documentation_skip_and_manual_override(self):
        step = next(
            step
            for step in self.check["steps"]
            if step.get("name") == "Summarize PPU test selection"
        )
        for force in ("false", "true"):
            with self.subTest(force=force), tempfile.TemporaryDirectory() as directory:
                summary = Path(directory) / "summary"
                subprocess.run(
                    ["bash", "-e", "-c", step["run"]],
                    env={
                        **os.environ,
                        "GITHUB_STEP_SUMMARY": str(summary),
                        "CPU": force,
                        "SINGLE": force,
                        "KERNEL": force,
                        "MULTI": force,
                    },
                    check=True,
                    timeout=30,
                )
                expected = (
                    "Selected lanes remain subject to PR authorization"
                    if force == "true"
                    else "skip the PR gate and all PPU stages"
                )
                self.assertIn(expected, summary.read_text())


if __name__ == "__main__":
    unittest.main()

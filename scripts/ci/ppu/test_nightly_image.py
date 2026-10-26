import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/nightly-test-ppu.yml"
SCRIPTS = ROOT / "scripts/ci/ppu"
FAMILIES = {
    "pd-glm52": "test-ppu-pd-perf-glm52.yml",
    "pd-qwen35": "test-ppu-pd-perf-qwen35.yml",
    "perf-8": "test-ppu-perf.yml",
    "perf-16": "test-ppu-perf-16.yml",
    "perf-32": "test-ppu-perf-32.yml",
    "answer-8": "test-ppu-answer.yml",
    "answer-16": "test-ppu-answer-16.yml",
    "answer-32": "test-ppu-answer-32.yml",
    "accuracy": "test-ppu-accuracy.yml",
}


class TestNightlyResults(unittest.TestCase):
    def verify(self, log):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.log"
            path.write_text(log)
            return subprocess.run(
                [sys.executable, str(SCRIPTS / "verify_nightly_result.py"), str(path)],
                capture_output=True,
                text=True,
            )

    def test_verified_execution_is_required(self):
        for log, success in (
            ("Ran 2 tests in 1.0s\n\nOK\nTest Summary: 1/1 passed\n", True),
            ("Ran 3 tests in 1.0s\n\nOK (skipped=1)\nTest Summary: 1/1 passed\n", True),
            (
                "Ran 2 tests in 1.0s\n\nOK (skipped=2)\nTest Summary: 1/1 passed\n",
                False,
            ),
            ("Ran 0 tests in 0.0s\n\nOK\nTest Summary: 1/1 passed\n", False),
            ("Test Summary: 0/0 passed\n", False),
            ("Ran 2 tests in 1.0s\n\nOK\nTest Summary: 1/2 passed\n", False),
            ("Ran 2 tests in 1.0s\n\nOK\nTest Summary: 2/2 passed\n", False),
            (
                "Ran 2 tests in 1.0s\n\nFAILED (errors=1)\nTest Summary: 1/1 passed\n",
                False,
            ),
            ("Ran 2 tests in 1.0s\n\nOK\n", False),
            ("", False),
        ):
            with self.subTest(log=log):
                result = self.verify(log)
                self.assertEqual(result.returncode == 0, success, result.stderr)


class TestNightlyWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.jobs = self.workflow["jobs"]

    def test_builds_ci_target_with_immutable_tag(self):
        build = self.jobs["build-image"]
        step = next(
            s
            for s in build["steps"]
            if s.get("uses", "").startswith("docker/build-push-action@")
        )
        self.assertEqual(step["with"]["target"], "ci")
        self.assertEqual(step["with"]["file"], "docker/ppu.Dockerfile")
        self.assertNotIn("latest", step["with"]["tags"])
        self.assertIn("digest", str(build["outputs"]))
        self.assertIn("ppu-eic=", step["with"]["build-contexts"])
        self.assertNotIn("/nas_aisw", step["with"]["build-contexts"])
        self.assertIs(self.workflow["concurrency"]["cancel-in-progress"], False)

    def test_every_family_consumes_the_same_built_digest(self):
        for name, workflow in FAMILIES.items():
            with self.subTest(job=name):
                self.assertIn(name, self.jobs)
                job = self.jobs[name]
                self.assertIn("build-image", job.get("needs", []))
                self.assertEqual(job["uses"], f"./.github/workflows/{workflow}")
                self.assertEqual(
                    job["with"]["image_ref"], "${{ needs.build-image.outputs.image }}"
                )
                self.assertEqual(
                    job["with"]["image_digest"],
                    "${{ needs.build-image.outputs.digest }}",
                )
                self.assertEqual(
                    job["with"]["ref"], "${{ needs.build-image.outputs.source-sha }}"
                )

    def test_full_coverage_and_resource_stage_order(self):
        self.assertEqual(
            self.jobs.get("accuracy", {}).get("with", {}).get("mode"), "full"
        )
        self.assertEqual(self.jobs["accuracy"]["with"]["entries"], "all")
        prior = {"build-image"}
        for stage in (
            ("pd-glm52", "pd-qwen35"),
            ("perf-8", "perf-16", "perf-32"),
            ("answer-8", "answer-16", "answer-32"),
            ("accuracy",),
        ):
            for name in stage:
                self.assertEqual(set(self.jobs[name]["needs"]), prior)
                self.assertIn("!cancelled()", self.jobs[name]["if"])
            prior.update(stage)
        for name in ("pd-glm52", "pd-qwen35"):
            self.assertEqual(self.jobs[name]["with"]["pd_config"], "all")

    def test_only_one_schedule_uses_global_resource_lock(self):
        self.assertEqual(self.workflow["concurrency"]["group"], "ppu-nightly-btv15")
        answer = yaml.safe_load(
            (ROOT / ".github/workflows/nightly-test-ppu-answer.yml").read_text()
        )
        triggers = answer.get("on", answer.get(True, {}))
        self.assertNotIn("schedule", triggers)
        self.assertEqual(answer["concurrency"]["group"], "ppu-nightly-btv15")

    def test_finish_rejects_skipped_cancelled_and_failed_jobs(self):
        finish = self.jobs["nightly-test-ppu-finish"]
        self.assertEqual(set(finish["needs"]), {"build-image", *FAMILIES})
        step = next(s for s in finish["steps"] if "NEEDS_JSON" in s.get("env", {}))
        good = {key: {"result": "success"} for key in finish["needs"]}
        cases = [
            (good, True),
            ({}, False),
            ({**good, "unexpected": {"result": "success"}}, False),
        ]
        for status in ("failure", "cancelled", "skipped", "unknown"):
            for name in finish["needs"]:
                cases.append(({**good, name: {"result": status}}, False))
        for needs, success in cases:
            with self.subTest(needs=needs):
                result = subprocess.run(
                    ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
                    env={**os.environ, "NEEDS_JSON": json.dumps(needs)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, success, result.stderr)

    def test_branch_advancement_during_registry_setup_prevents_promotion(self):
        runner = """
const steps = JSON.parse(process.argv[1]);
const advance = process.argv[2] === 'true';
let head = process.env.SOURCE_SHA;
let promoted = false;
const outputs = {};
const events = [];
const github = {rest: {git: {getRef: async () => {
  events.push('check');
  return {data: {object: {sha: head}}};
}}}};
const context = {repo: {owner: 'fixture', repo: 'sglang'}};
const core = {setOutput: (name, value) => {outputs[name] = value;}, notice: () => {}};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
(async () => {
  for (const step of steps) {
    if (step.if) {
      if (step.if !== "steps.current-revision.outputs.current == 'true'")
        throw new Error('Unrecognized promotion condition');
      if (outputs.current !== 'true') continue;
    }
    if ((step.uses || '').startsWith('docker/login-action@')) {
      events.push('login');
      if (advance) head = 'b'.repeat(40);
    }
    if ((step.uses || '').startsWith('actions/github-script@'))
      await new AsyncFunction('github', 'context', 'core', step.with.script)(github, context, core);
    if ((step.run || '').includes('imagetools create')) promoted = true;
  }
  console.log(JSON.stringify({promoted, events}));
})().catch(error => {console.error(error); process.exit(1);});
"""
        for advance in (False, True):
            with self.subTest(branch_advanced=advance):
                result = subprocess.run(
                    [
                        "node",
                        "-e",
                        runner,
                        json.dumps(self.jobs["promote-image"]["steps"]),
                        str(advance).lower(),
                    ],
                    env={**os.environ, "SOURCE_SHA": "a" * 40},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                observed = json.loads(result.stdout)
                self.assertEqual(observed["promoted"], not advance)
                self.assertLess(
                    observed["events"].index("login"), observed["events"].index("check")
                )

    def test_promotion_only_retags_a_successful_current_revision(self):
        job = self.jobs["promote-image"]
        self.assertIn("nightly-test-ppu-finish", job["needs"])
        self.assertNotIn("always()", job.get("if", ""))
        self.assertIn("current-revision", str(job["steps"]))
        steps = [s for s in job["steps"] if "imagetools create" in s.get("run", "")]
        self.assertEqual(len(steps), 1)
        self.assertIn("--prefer-index=false", steps[0]["run"])
        self.assertIn('"$PPU_BASE_IMAGE"', steps[0]["run"])
        self.assertIn("v0.5.18-latest", str(job))
        self.assertNotIn("build-push-action", str(job))


if __name__ == "__main__":
    unittest.main()

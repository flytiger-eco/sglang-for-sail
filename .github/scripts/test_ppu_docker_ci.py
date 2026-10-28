import copy
import fnmatch
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/pr-test-ppu.yml"
BASELINE = "f51071a18f5e5d3ce889d1ced7328c61a124c833"
STAGES = [
    f"stage-{stage}-test-{gpus}-gpu-ppu"
    for stage, gpus in (("a", 1), ("b", 1), ("c", 2))
]


class DockerCIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(WORKFLOW.read_text())
        cls.jobs = cls.workflow["jobs"]

    def finish(self, docker_changes, failure=None):
        self.assertIn("build-ppu-docker", self.jobs["pr-test-ppu-finish"]["needs"])
        needs = {
            name: {"result": "success", "outputs": {}}
            for name in self.jobs["pr-test-ppu-finish"]["needs"]
        }
        if failure:
            needs[failure]["result"] = "failure"
        if not docker_changes:
            needs["build-ppu-docker"]["result"] = "skipped"
        script = self.jobs["pr-test-ppu-finish"]["steps"][0]["run"]
        with tempfile.NamedTemporaryFile() as summary:
            return subprocess.run(
                ["bash", "-e", "-c", script],
                capture_output=True,
                env={
                    **os.environ,
                    "NEEDS_JSON": json.dumps(needs),
                    "DOCKER_CHANGES": str(docker_changes).lower(),
                    "CANDIDATE_TAG": "example.invalid/ppu:ci-commit-run-attempt",
                    "TESTED_IMAGE": "example.invalid/ppu@sha256:" + "a" * 64,
                    "GITHUB_STEP_SUMMARY": summary.name,
                },
            ).returncode

    def promote(
        self,
        event="push",
        ref="refs/heads/v0.5.21",
        changes="true",
        build="success",
        finish="success",
    ):
        values = {
            "github.event_name": event,
            "github.ref": ref,
            "needs.check-changes.outputs.docker_changes": changes,
            "needs.build-ppu-docker.result": build,
            "needs.pr-test-ppu-finish.result": finish,
        }
        expression = self.jobs["promote-qualified-image"]["if"]
        expression = re.sub(
            r"\b(?:github|needs)\.[\w.-]+",
            lambda match: repr(values[match[0]]),
            expression,
        )
        return eval(
            expression.replace("&&", " and ").replace("||", " or "),
            {"__builtins__": {}},
        )

    def test_docker_changes_trigger_build_but_not_ordinary_source(self):
        check = self.jobs["check-changes"]
        filters = yaml.safe_load(
            next(step for step in check["steps"] if step.get("id") == "filter")["with"][
                "filters"
            ]
        )
        self.assertIn("ppu_docker", filters)
        paths = filters["ppu_docker"]
        for path in (
            "docker/ppu.Dockerfile",
            ".github/workflows/pr-test-ppu.yml",
            ".dockerignore",
            ".gitignore",
            "python/pyproject_ppu.toml",
            "python/setup.py",
            "python/sglang/srt/rust_extensions/torch_build.py",
            "rust/Cargo.toml",
        ):
            self.assertTrue(any(fnmatch.fnmatch(path, pattern) for pattern in paths))
        for path in (
            "python/sglang/srt/server_args.py",
            "test/registered/ppu/test_ppu_basic.py",
            "scripts/ci/ppu/ppu_install_dependency.sh",
        ):
            self.assertFalse(any(fnmatch.fnmatch(path, pattern) for pattern in paths))
        self.assertEqual(paths, self.workflow[True]["push"]["paths"])
        self.assertIn(
            "steps.filter.outputs.ppu_docker == 'true'",
            check["outputs"]["docker_changes"],
        )
        self.assertIn(
            "needs.check-changes.outputs.docker_changes == 'true'",
            self.jobs["build-ppu-docker"]["if"],
        )
        self.assertIn(
            "github.event.pull_request.head.repo.full_name == github.repository",
            self.jobs["build-ppu-docker"]["if"],
        )
        self.assertIn("'run-ci'", self.jobs["build-ppu-docker"]["if"])

    def test_candidate_digest_reaches_existing_stages(self):
        build = self.jobs["build-ppu-docker"]
        self.assertEqual(build["runs-on"], "ubuntu-latest")
        self.assertIn("tested_image", build["outputs"])
        publish = next(step for step in build["steps"] if step.get("id") == "candidate")
        self.assertEqual(publish["env"]["DIGEST"], "${{ steps.build.outputs.digest }}")
        self.assertIn("${PPU_DOCKER_REPOSITORY}@${DIGEST}", publish["run"])
        for digest, valid in (("sha256:" + "a" * 64, True), ("latest", False)):
            result = subprocess.run(
                ["bash", "-e", "-c", publish["run"]],
                env={
                    **os.environ,
                    "DIGEST": digest,
                    "PPU_DOCKER_REPOSITORY": "example.invalid/ppu",
                    "GITHUB_OUTPUT": os.devnull,
                },
            )
            self.assertEqual(result.returncode == 0, valid)
        for stage in STAGES:
            self.assertIn("build-ppu-docker", self.jobs[stage]["needs"])
            action = next(
                step
                for step in self.jobs[stage]["steps"]
                if "ppu-distributed-action@" in step.get("uses", "")
            )
            self.assertEqual(
                action["env"]["PPU_BASE_IMAGE"],
                "${{ needs.build-ppu-docker.outputs.tested_image || env.PPU_BASE_IMAGE }}",
            )
            self.assertEqual(action["with"]["image"], "${{ env.PPU_BASE_IMAGE }}")

    def test_pr_and_dispatch_never_promote(self):
        self.assertFalse(self.promote(event="pull_request"))
        self.assertFalse(self.promote(event="workflow_dispatch"))

    def test_only_successful_integration_push_promotes(self):
        self.assertTrue(self.promote())
        self.assertFalse(self.promote(ref="refs/heads/main"))
        self.assertFalse(self.promote(changes="false"))
        for result in ("failure", "cancelled", "skipped"):
            self.assertFalse(self.promote(build=result))
            self.assertFalse(self.promote(finish=result))

    def test_each_stage_and_build_failure_blocks_promotion(self):
        self.assertEqual(self.finish(True), 0)
        for job in ["build-ppu-docker", *STAGES]:
            with self.subTest(job=job):
                self.assertNotEqual(self.finish(True, job), 0)
                self.assertFalse(self.promote(finish="failure"))

    def test_promote_retags_tested_digest_without_rebuild(self):
        promote = self.jobs["promote-qualified-image"]
        self.assertEqual(promote["runs-on"], "ubuntu-latest")
        self.assertEqual(
            promote["env"]["TESTED_IMAGE"],
            "${{ needs.build-ppu-docker.outputs.tested_image }}",
        )
        script = promote["steps"][-1]["run"]
        self.assertIn("docker buildx imagetools create --prefer-index=false", script)
        self.assertIn('--tag "${PPU_DOCKER_REPOSITORY}:qualified"', script)
        self.assertIn('"${TESTED_IMAGE}"', script)
        self.assertFalse(
            any(
                "build-push-action" in step.get("uses", "") for step in promote["steps"]
            )
        )

    def test_no_docker_change_preserves_existing_ci(self):
        self.assertEqual(self.finish(False), 0)
        baseline = yaml.safe_load(
            subprocess.check_output(
                ["git", "show", f"{BASELINE}:.github/workflows/pr-test-ppu.yml"],
                cwd=ROOT,
                text=True,
            )
        )
        for stage in STAGES:
            before, after = baseline["jobs"][stage], self.jobs[stage]
            for field in ("runs-on", "container", "timeout-minutes"):
                self.assertEqual(before[field], after[field])
            old_action = next(
                step
                for step in before["steps"]
                if "ppu-distributed-action@" in step.get("uses", "")
            )
            new_action = copy.deepcopy(
                next(
                    step
                    for step in after["steps"]
                    if "ppu-distributed-action@" in step.get("uses", "")
                )
            )
            new_action.pop("env")
            self.assertEqual(old_action, new_action)
        image = self.workflow["env"]["PPU_BASE_IMAGE"]
        self.assertLess(
            image.index("inputs.runtime_image"), image.index("vars.PPU_CI_BASE_IMAGE")
        )
        self.assertIn(baseline["env"]["PPU_BASE_IMAGE"], image)


if __name__ == "__main__":
    unittest.main()

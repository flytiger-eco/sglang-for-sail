"""PR consumers resolve the last successful nightly once, then use only its digest."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/ci/ppu"
WORKFLOWS = ROOT / ".github/workflows"
RESOLVER = SCRIPTS / "resolve_nightly_image.sh"
REPOSITORY = "registry.example.test:5000/team/ppu-image"
IMAGE = f"{REPOSITORY}:v0.5.18-latest"
DIGEST = "sha256:" + "abc01234" * 8
REF = f"{REPOSITORY}@{DIGEST}"
HARDWARE = (
    "stage-a-test-1-gpu-ppu",
    "stage-b-test-1-gpu-ppu",
    "stage-c-test-2-gpu-ppu",
)


def workflow(name):
    # BaseLoader keeps GitHub's `on` key and string-valued expressions intact.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


class TestNightlyResolver(unittest.TestCase):
    def setUp(self):
        self.assertTrue(RESOLVER.is_file(), "fail-closed nightly resolver is missing")
        self.tmp = tempfile.TemporaryDirectory(dir=SCRIPTS)
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.env_file = self.directory / "github-env"
        self.output_file = self.directory / "github-output"
        self.calls = self.directory / "curl-calls"
        self.env_file.write_text("existing_env=kept\n")
        self.output_file.write_text("existing_output=kept\n")
        curl = self.directory / "curl"
        curl.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with open(os.environ['CURL_CALLS'], 'a') as output:\n"
            "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "mode = os.environ['CURL_MODE']\n"
            "if mode == 'unreachable':\n"
            "    sys.exit(7)\n"
            "if mode == 'missing':\n"
            "    print('HTTP/1.1 404 Not Found\\r\\n')\n"
            "elif mode == 'auth' and '--get' in sys.argv:\n"
            '    print(\'{"token":"anonymous-pull-token"}\')\n'
            "elif mode == 'auth' and not any('Bearer' in a for a in sys.argv):\n"
            "    print('HTTP/1.1 401 Unauthorized\\r\\nWww-Authenticate: Bearer '\n"
            "          'realm=\"https://registry.example.test/token\",'\n"
            '          \'service="registry",scope="repository:team/ppu-image:pull"\\r\\n\')\n'
            "else:\n"
            "    print('HTTP/1.1 200 OK\\r\\nDocker-Content-Digest: '\n"
            "          + os.environ['CURL_DIGEST'] + '\\r\\n')\n"
        )
        curl.chmod(0o700)

    def run_resolver(self, image=IMAGE, digest=DIGEST, mode="success"):
        env = os.environ.copy()
        env.update(
            PATH=f"{self.directory}:{env['PATH']}",
            PPU_BASE_IMAGE=image,
            # Do not accept stale inherited identities on a lookup failure.
            PPU_BASE_IMAGE_REF=REF,
            PPU_BASE_IMAGE_DIGEST=DIGEST,
            GITHUB_ENV=str(self.env_file),
            GITHUB_OUTPUT=str(self.output_file),
            RUNNER_TEMP=str(self.directory),
            CURL_CALLS=str(self.calls),
            CURL_MODE=mode,
            CURL_DIGEST=digest,
        )
        return subprocess.run(
            ["bash", str(RESOLVER)],
            cwd=self.directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def assert_no_outputs(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.env_file.read_text(), "existing_env=kept\n")
        self.assertEqual(self.output_file.read_text(), "existing_output=kept\n")
        self.assertFalse(list(self.directory.glob("ppu-nightly-image.*")))

    def test_empty_or_malformed_image_rejected_before_network(self):
        for image in (
            "",
            ":v0.5.18-latest",
            "registry/repo",
            "registry/repo:latest",
            "registry/repo:v0.5.17-latest",
            REF,
            "https://" + IMAGE,
            "registry/repo:old:v0.5.18-latest",
            "registry//repo:v0.5.18-latest",
            "registry/../repo:v0.5.18-latest",
            "registry/Repo:v0.5.18-latest",
            IMAGE + "\nimage=injected",
            "registry/repo with spaces:v0.5.18-latest",
            "user:password@registry/repo:v0.5.18-latest",
        ):
            with self.subTest(image=image):
                self.assert_no_outputs(self.run_resolver(image=image))
                self.assertFalse(self.calls.exists())

    def test_missing_first_nightly_and_network_failure_fail_closed(self):
        for mode in ("missing", "unreachable"):
            with self.subTest(mode=mode):
                self.assert_no_outputs(self.run_resolver(mode=mode))

    def test_malformed_digest_never_published_or_executed(self):
        marker = self.directory / "injected"
        for digest in (
            "",
            "sha256:abc",
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            "sha256:" + "g" * 64,
            "sha512:" + "a" * 64,
            DIGEST + " ",
            DIGEST + ";touch " + str(marker),
            "sha256:$(touch " + str(marker) + ")",
        ):
            with self.subTest(digest=digest):
                self.assert_no_outputs(self.run_resolver(digest=digest))
                self.assertFalse(marker.exists())

    def test_publishes_only_validated_matching_digest(self):
        result = self.run_resolver()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.env_file.read_text(),
            f"existing_env=kept\nPPU_BASE_IMAGE_REF={REF}\nPPU_BASE_IMAGE_DIGEST={DIGEST}\n",
        )
        self.assertEqual(
            self.output_file.read_text(),
            f"existing_output=kept\nimage={REF}\ndigest={DIGEST}\n",
        )
        requests = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(requests), 1)
        self.assertIn(
            "https://registry.example.test:5000/v2/team/ppu-image/manifests/v0.5.18-latest",
            requests[0],
        )
        self.assertFalse(list(self.directory.glob("ppu-nightly-image.*")))

    def test_anonymous_pull_challenge_uses_existing_resolver(self):
        result = self.run_resolver(mode="auth")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"image={REF}\n", self.output_file.read_text())
        requests = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(requests), 3)
        self.assertIn("Authorization: Bearer anonymous-pull-token", requests[-1])
        self.assertNotIn("--user", [arg for request in requests for arg in request])


class TestPrNightlyWorkflows(unittest.TestCase):
    def setUp(self):
        self.pr = workflow("pr-test-ppu.yml")
        self.e2e = workflow("pr-test-ppu-e2e.yml")
        self.entry = workflow("pr-test-ppu-e2e-entry.yml")

    def test_single_authorized_resolution_job_without_registry_write_credentials(self):
        jobs = self.pr["jobs"]
        self.assertIn("resolve-nightly-image", jobs)
        job = jobs["resolve-nightly-image"]
        self.assertEqual(set(job["needs"]), {"check-changes", "pr-gate"})
        self.assertIn("inputs.cpu_only != true", job["if"])
        self.assertIn("needs.pr-gate.result == 'success'", job["if"])
        for flag in ("ppu_single", "ppu_kernel", "ppu_multi"):
            self.assertIn(f"needs.check-changes.outputs.{flag} == 'true'", job["if"])
        self.assertIn("inputs.e2e_requested == true", job["if"])
        self.assertEqual(job["runs-on"], "k8s-runner-group-cpu-flytiger")
        self.assertEqual(
            job["container"]["image"], "ghcr.io/actions/actions-runner:latest"
        )
        self.assertEqual(
            job["env"]["PPU_BASE_IMAGE"],
            "${{ vars.PPU_NIGHTLY_IMAGE_REPOSITORY }}:v0.5.18-latest",
        )
        step = next(
            s for s in job["steps"] if "resolve_nightly_image.sh" in s.get("run", "")
        )
        for output in ("image", "digest"):
            self.assertEqual(
                job["outputs"][output],
                "${{ steps." + step["id"] + ".outputs." + output + " }}",
            )
        self.assertNotIn("secrets.", json.dumps(job))
        self.assertNotIn("write", self.pr["permissions"].values())
        self.assertIn("resolve-nightly-image", jobs["pr-test-ppu-finish"]["needs"])
        combined = json.dumps([self.pr, self.e2e, self.entry])
        self.assertEqual(combined.count("resolve_nightly_image.sh"), 1)
        self.assertNotIn("resolve_image_digest.sh", combined)
        self.assertNotIn("sglang0.5.17", combined)

    def test_all_hardware_lanes_use_shared_outputs(self):
        for name in HARDWARE:
            with self.subTest(job=name):
                job = self.pr["jobs"][name]
                self.assertIn("resolve-nightly-image", job["needs"])
                self.assertIn(
                    "needs.resolve-nightly-image.result == 'success'", job["if"]
                )
                for key, output in (
                    ("PPU_BASE_IMAGE", "image"),
                    ("PPU_BASE_IMAGE_REF", "image"),
                    ("PPU_BASE_IMAGE_DIGEST", "digest"),
                ):
                    self.assertEqual(
                        job["env"][key],
                        "${{ needs.resolve-nightly-image.outputs." + output + " }}",
                    )
                worker = next(s for s in job["steps"] if s.get("id") == "worker")
                self.assertEqual(
                    worker["with"]["image"], "${{ env.PPU_BASE_IMAGE_REF }}"
                )
                for key in (
                    "PPU_BASE_IMAGE",
                    "PPU_BASE_IMAGE_REF",
                    "PPU_BASE_IMAGE_DIGEST",
                ):
                    self.assertIn(
                        key + "=${{ env." + key + " }}", worker["with"]["extra_env"]
                    )
                self.assertTrue(any(s.get("id") == "evidence" for s in job["steps"]))
        self.assertEqual(
            [self.pr["jobs"][j]["timeout-minutes"] for j in HARDWARE],
            ["45", "60", "60"],
        )

    def test_cpu_stage_remains_independent_of_image_and_gate(self):
        cpu = self.pr["jobs"]["stage-a-test-cpu-ppu"]
        self.assertEqual(cpu["needs"], "check-changes")
        self.assertEqual(cpu["if"], "needs.check-changes.outputs.cpu == 'true'")
        self.assertNotIn("PPU_BASE_IMAGE", self.pr["env"])
        self.assertNotIn("resolve-nightly-image", json.dumps(cpu))
        self.assertIn("inputs.cpu_only != true", self.pr["jobs"]["pr-gate"]["if"])
        self.assertEqual(
            self.pr["jobs"]["pr-gate"]["with"]["require-run-ci"],
            "${{ inputs.require_run_ci_label }}",
        )

    def test_e2e_chain_forwards_required_digest_inputs_without_lookups(self):
        job = self.pr["jobs"]["e2e"]
        self.assertIn("resolve-nightly-image", job["needs"])
        self.assertIn("needs.resolve-nightly-image.result == 'success'", job["if"])
        for name, output in (("image_ref", "image"), ("image_digest", "digest")):
            self.assertEqual(
                job["with"][name],
                "${{ needs.resolve-nightly-image.outputs." + output + " }}",
            )
            for reusable in (self.e2e, self.entry):
                self.assertEqual(
                    reusable["on"]["workflow_call"]["inputs"][name],
                    {"required": "true", "type": "string"},
                )
            for phase in ("default", "extended", "distributed"):
                self.assertEqual(
                    self.e2e["jobs"][phase]["with"][name], "${{ inputs." + name + " }}"
                )
        env = self.entry["jobs"]["execute"]["env"]
        self.assertEqual(env["PPU_BASE_IMAGE_REF"], "${{ inputs.image_ref }}")
        self.assertEqual(env["PPU_BASE_IMAGE_DIGEST"], "${{ inputs.image_digest }}")
        self.assertEqual(env["PPU_BASE_IMAGE"], "${{ inputs.image_ref }}")

    def test_entry_rejects_bad_digest_references_before_allocation(self):
        steps = self.entry["jobs"]["execute"]["steps"]
        validation = next(
            (s for s in steps if s.get("name") == "Validate immutable image inputs"),
            None,
        )
        self.assertIsNotNone(
            validation, "entry must validate digest-pinned image_ref inputs"
        )
        self.assertLess(
            steps.index(validation),
            next(i for i, s in enumerate(steps) if s.get("id") == "worker"),
        )
        for image, digest, valid in (
            (REF, DIGEST, True),
            (IMAGE, DIGEST, False),
            ("@" + DIGEST, DIGEST, False),
            ("https://" + REF, DIGEST, False),
            (REF + "\n", DIGEST, False),
            (REF, "sha256:" + "a" * 64, False),
            (REPOSITORY + "@sha256:abc", "sha256:abc", False),
            ("registry//repo@" + DIGEST, DIGEST, False),
        ):
            with self.subTest(image=image, digest=digest):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", validation["run"]],
                    env={
                        **os.environ,
                        "PPU_BASE_IMAGE_REF": image,
                        "PPU_BASE_IMAGE_DIGEST": digest,
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode == 0, valid, result.stdout + result.stderr
                )

    def test_aggregate_accepts_extra_resolution_need_and_rejects_blocked_consumers(
        self,
    ):
        for cpu_only, resolver_result in (
            (False, "success"),
            (True, "skipped"),
            (False, "failure"),
        ):
            with self.subTest(cpu_only=cpu_only, resolver_result=resolver_result):
                needs = {
                    "check-changes": {
                        "result": "success",
                        "outputs": dict.fromkeys(
                            ("cpu", "ppu_single", "ppu_kernel", "ppu_multi"), "true"
                        ),
                    },
                    "pr-gate": {"result": "skipped" if cpu_only else "success"},
                    "resolve-nightly-image": {"result": resolver_result},
                    "e2e": {"result": "skipped"},
                    "stage-a-test-cpu-ppu": {
                        "result": "success",
                        "outputs": {"evidence_ok": "true"},
                    },
                }
                for name in HARDWARE:
                    needs[name] = {
                        "result": (
                            "success"
                            if not cpu_only and resolver_result == "success"
                            else "skipped"
                        ),
                        "outputs": {"evidence_ok": "true"},
                    }
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / "pr_report.py"), "aggregate"]
                    + (["--cpu-only"] if cpu_only else []),
                    env={
                        **os.environ,
                        "NEEDS_JSON": json.dumps(needs),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode == 0, resolver_result != "failure", result.stderr
                )


if __name__ == "__main__":
    unittest.main()

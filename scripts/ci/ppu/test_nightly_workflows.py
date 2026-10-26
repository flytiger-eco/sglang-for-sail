import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
WORKER_COUNTS = {
    "test-ppu-answer.yml": 11,
    "test-ppu-answer-16.yml": 1,
    "test-ppu-answer-32.yml": 1,
    "test-ppu-perf.yml": 9,
    "test-ppu-perf-16.yml": 1,
    "test-ppu-perf-32.yml": 1,
    "test-ppu-pd-perf-glm52.yml": 2,
    "test-ppu-pd-perf-qwen35.yml": 2,
    "test-ppu-accuracy.yml": 28,
}
LEGACY_IMAGE = (
    "pkg.flytiger-eco.com/docker_release/llm:"
    "sdk2.2.0-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.17-py312-20261001"
)
HELPER = "/workspace/source/scripts/ci/ppu/use_nightly_image.sh"
VALIDATION = "Validate candidate image inputs"
BAKED_GUARD = 'if [ "${PPU_TEST_BAKED_IMAGE:-0}" = "1" ]; then'
LEGACY_GUARD = 'if [ "${PPU_TEST_BAKED_IMAGE:-0}" != "1" ]; then'
SETUP = re.compile(
    r"\Aset -e\n"
    + re.escape(BAKED_GUARD)
    + r"\n  source "
    + re.escape(HELPER)
    + r"\nelse\n(.*?)\nfi\n",
    re.DOTALL,
)
INSTALL = re.compile(re.escape(LEGACY_GUARD) + r"\n(.*?)\nfi\n", re.DOTALL)
CHANGED_ENV = {
    "PPU_TEST_BAKED_IMAGE",
    "PPU_CI_SOURCE_SHA",
    "PPU_BASE_IMAGE",
    "PPU_BASE_IMAGE_DIGEST",
    "PPU_ARTIFACTORY_USER",
    "PPU_ARTIFACTORY_PASSWORD",
}
# Freeze fields outside the candidate-image contract, including complete legacy commands.
LEGACY_CONTRACT_HASHES = {
    "test-ppu-answer.yml": "88370f06e1595ff9a78a11ace59224ab4420bb00c24d4aaadb755cd70171e7bc",
    "test-ppu-answer-16.yml": "3f7b3c4c835213026c9ca82cb2c76e99cca2f1ae334b9482b70ef1cb3fbbb94f",
    "test-ppu-answer-32.yml": "b1150dd2631009e9c4ba86e2eead4ba60608581952c7ad2848cfbfa478cec3b9",
    "test-ppu-perf.yml": "2c02578fc3f04c5e1f33acbfebee290c5d643f5e8fd2569c18ab2b0c56a8e393",
    "test-ppu-perf-16.yml": "7a53e046cb5a8dc44744501f2e30a9df2f87e13cdbc02d5e0a2cbbaa653390d1",
    "test-ppu-perf-32.yml": "167657ddcdfc2f998c03f59b5ef0797fae253ace2249fbeb65d132de0665ea3c",
    "test-ppu-pd-perf-glm52.yml": "1b13118c863b1496671bd749d291c1b21c28b1b1a2ed6b6583209cc89eed1cb6",
    "test-ppu-pd-perf-qwen35.yml": "9f65775792766b84a001037b6615a40d02a97d8dad77ba6ed4a3ecee7176d748",
    "test-ppu-accuracy.yml": "ad93114ff8753da00360a85d98dbdf360e7e42ba478ed9fd7535791d19d828fe",
}


def workers(document):
    for job_id, job in document["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            if re.search(r"/ppu[^/]*action@", step.get("uses", "")):
                yield job_id, job, index, step


def legacy_contract(document):
    document = copy.deepcopy(document)
    for key in ("image_ref", "image_digest"):
        document["on"]["workflow_call"]["inputs"].pop(key, None)
    for key in ("PPU_BASE_IMAGE", "PPU_BASE_IMAGE_DIGEST"):
        document["env"].pop(key, None)
    for job in document["jobs"].values():
        job["steps"] = [s for s in job.get("steps", []) if s.get("name") != VALIDATION]
        for step in job["steps"]:
            if re.match(r"Resolve .* image digest", step.get("name", "")):
                step.pop("if", None)
    for _, _, _, step in workers(document):
        settings = step["with"]
        settings.pop("image")
        settings["extra_env"] = ",".join(
            part
            for part in re.split(r",(?=[A-Z][A-Z0-9_]*=)", settings["extra_env"])
            if part.split("=", 1)[0] not in CHANGED_ENV
        )
        command = settings["command"]
        command = SETUP.sub(
            lambda m: re.sub(r"^  ", "", m[1], flags=re.MULTILINE) + "\n", command
        )
        command = INSTALL.sub(lambda m: textwrap.dedent(m[1]) + "\n", command)
        settings["command"] = command
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


class NightlyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {
            name: yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)
            for name in WORKER_COUNTS
        }

    def all_workers(self):
        for name, document in self.documents.items():
            for job_id, job, index, step in workers(document):
                yield name, job_id, job, index, step

    def test_enumerates_every_worker(self):
        for name, document in self.documents.items():
            with self.subTest(workflow=name):
                self.assertEqual(len(list(workers(document))), WORKER_COUNTS[name])
                self.assertEqual(
                    (WORKFLOWS / name).read_text().count("extra_env:"),
                    WORKER_COUNTS[name],
                )

    def test_optional_inputs_and_legacy_default(self):
        for name, document in self.documents.items():
            with self.subTest(workflow=name):
                inputs = document["on"]["workflow_call"]["inputs"]
                for key in ("image_ref", "image_digest"):
                    self.assertIn(key, inputs)
                    self.assertEqual(inputs[key]["type"], "string")
                    self.assertEqual(inputs[key]["required"], "false")
                    self.assertEqual(inputs[key]["default"], "")
                self.assertEqual(
                    document["env"]["PPU_BASE_IMAGE"],
                    "${{ inputs.image_ref || '" + LEGACY_IMAGE + "' }}",
                )
                self.assertEqual(
                    document["env"]["PPU_BASE_IMAGE_DIGEST"],
                    "${{ inputs.image_digest }}",
                )

    def test_every_worker_pins_image_and_propagates_identity_without_duplicates(self):
        for name, job_id, _, _, step in self.all_workers():
            with self.subTest(workflow=name, job=job_id):
                settings = step["with"]
                fallback = "env.PPU_BASE_IMAGE_REF || " if "perf" in name else ""
                image = "${{ inputs.image_ref || " + fallback + "env.PPU_BASE_IMAGE }}"
                self.assertEqual(settings["image"], image)
                extra = settings["extra_env"]
                keys = re.findall(r"(?:^|,)([A-Z][A-Z0-9_]*)=", extra)
                self.assertEqual(len(keys), len(set(keys)))
                expected = {
                    "PPU_BASE_IMAGE": image,
                    "PPU_BASE_IMAGE_DIGEST": "${{ env.PPU_BASE_IMAGE_DIGEST }}",
                    "PPU_CI_SOURCE_SHA": "${{ inputs.ref }}",
                    "PPU_TEST_BAKED_IMAGE": "${{ inputs.image_ref != '' && '1' || '0' }}",
                }
                for key, value in expected.items():
                    self.assertIn(key + "=" + value + ",", extra)

    def test_candidate_workers_do_not_receive_artifactory_secrets(self):
        for name, job_id, _, _, step in self.all_workers():
            with self.subTest(workflow=name, job=job_id):
                extra = step["with"]["extra_env"]
                for key in ("PPU_ARTIFACTORY_USER", "PPU_ARTIFACTORY_PASSWORD"):
                    self.assertIn(
                        key
                        + "=${{ inputs.image_ref == '' && secrets."
                        + key
                        + " || '' }}",
                        extra,
                    )

    def test_legacy_resolvers_are_skipped_for_candidates(self):
        count = 0
        for name, document in self.documents.items():
            for job in document["jobs"].values():
                for step in job.get("steps", []):
                    if "resolve_image_digest.sh" in step.get("run", ""):
                        count += 1
                        with self.subTest(workflow=name, step=step["name"]):
                            self.assertEqual(
                                step.get("if"), "${{ inputs.image_ref == '' }}"
                            )
        self.assertEqual(count, 15)

    def test_validation_is_fail_closed_after_checkout_before_every_worker(self):
        for name, job_id, job, index, step in self.all_workers():
            with self.subTest(workflow=name, job=job_id):
                steps = job["steps"]
                validations = [
                    i for i, s in enumerate(steps) if s.get("name") == VALIDATION
                ]
                self.assertEqual(len(validations), 1)
                validation_index = validations[0]
                self.assertLess(validation_index, index)
                checkouts = [
                    i
                    for i, s in enumerate(steps)
                    if s.get("uses", "").startswith("actions/checkout@")
                ]
                self.assertTrue(checkouts)
                self.assertGreater(validation_index, max(checkouts))
                validation = steps[validation_index]
                self.assertNotIn("if", validation)
                self.assertNotIn("continue-on-error", validation)
                self.assertNotIn("if", step)
                self.assertEqual(validation["shell"], "bash")
                self.assertEqual(
                    validation["env"],
                    {
                        "IMAGE_REF": "${{ inputs.image_ref }}",
                        "IMAGE_DIGEST": "${{ inputs.image_digest }}",
                    },
                )
                self.assertNotIn("${{", validation["run"])
                self.assertIn("=~", validation["run"])

    def test_validation_shell_accepts_only_paired_matching_digest_references(self):
        digest = "sha256:" + "a1" * 32
        reference = "registry.example:5000/team/sglang@" + digest
        cases = [
            ("", "", True),
            (reference, digest, True),
            ("sglang@" + digest, digest, True),
            (reference, "", False),
            ("", digest, False),
            (reference, "sha256:" + "b" * 64, False),
            ("registry.example/sglang:latest", digest, False),
            ("@" + digest, digest, False),
            ("https://registry.example/sglang@" + digest, digest, False),
            ("registry.example/sglang@@" + digest, digest, False),
            ("registry.example/sglang@" + digest + "\n", digest, False),
            (reference + ",INJECTED=1", digest, False),
            ("$(exit 0)/sglang@" + digest, digest, False),
            ("registry.example/space name@" + digest, digest, False),
        ]
        for bad_digest in (
            "a" * 64,
            "sha512:" + "a" * 64,
            "sha256:" + "g" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            digest + "\n",
        ):
            cases.append(("registry.example/sglang@" + bad_digest, bad_digest, False))
        for name, job_id, job, _, _ in self.all_workers():
            validations = [s for s in job["steps"] if s.get("name") == VALIDATION]
            with self.subTest(workflow=name, job=job_id):
                self.assertEqual(len(validations), 1)
            if not validations:
                continue
            for ref, sha, accepted in cases:
                with self.subTest(workflow=name, job=job_id, ref=ref, digest=sha):
                    result = subprocess.run(
                        [
                            "bash",
                            "--noprofile",
                            "--norc",
                            "-e",
                            "-c",
                            validations[0]["run"],
                        ],
                        env={**os.environ, "IMAGE_REF": ref, "IMAGE_DIGEST": sha},
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)

    def test_inline_setup_and_mutations_are_guarded(self):
        count = 0
        for name, job_id, _, _, step in self.all_workers():
            command = step["with"]["command"]
            if "\n" not in command:
                with self.subTest(workflow=name, job=job_id):
                    self.assertNotIn(HELPER, command)
                    self.assertRegex(
                        command, r"^bash /workspace/source/scripts/ci/ppu/run_.*\.sh$"
                    )
                continue
            count += 1
            with self.subTest(workflow=name):
                setup = SETUP.search(command)
                install = INSTALL.search(command)
                self.assertIsNotNone(
                    setup, "baked helper must precede legacy setup and source revision"
                )
                self.assertIsNotNone(
                    install, "dependency install and JIT touch must be legacy-only"
                )
                self.assertTrue(
                    command[setup.end() :].startswith(
                        "SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)\n"
                    )
                )
                self.assertIn("~/.pip/pip.conf", setup[1])
                self.assertIn("cd /workspace/source", setup[1])
                self.assertIn(
                    "git config --global --add safe.directory /workspace/source",
                    setup[1],
                )
                self.assertIn("ppu_install_dependency.sh", install[1])
                self.assertIn("JIT_MTIME=$(git log", install[1])
                self.assertIn("-exec touch", install[1])
                remainder = command.replace(setup[0], "").replace(install[0], "")
                self.assertNotRegex(
                    remainder,
                    r"pip\.conf|ppu_install_dependency|JIT_MTIME|\btouch\b|use_nightly_image",
                )
                result = subprocess.run(
                    ["bash", "-n"], input=command, text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for mode in ("0", "1"):
                    for fragment in (setup[0], install[0]):
                        result = subprocess.run(
                            [
                                "bash",
                                "--noprofile",
                                "--norc",
                                "-e",
                                "-c",
                                """
source() { printf 'helper:%s\n' "$1"; }
mkdir() { printf 'mkdir\n'; }
cat() { while IFS= read -r line; do :; done; }
cd() { printf 'cd:%s\n' "$1"; }
git() { printf 'git:%s\n' "$*" >&2; printf '1\n'; }
bash() { printf 'install:%s\n' "$1"; }
find() { printf 'touch\n'; }
"""
                                + fragment.replace("~/.pip/pip.conf", "/dev/null"),
                            ],
                            env={**os.environ, "PPU_TEST_BAKED_IMAGE": mode},
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        if mode == "1":
                            self.assertNotRegex(
                                result.stdout + result.stderr,
                                r"mkdir|cd:|git:|install:|touch",
                            )
                            self.assertEqual(
                                result.stdout,
                                (
                                    "helper:" + HELPER + "\n"
                                    if fragment == setup[0]
                                    else ""
                                ),
                            )
                        else:
                            self.assertNotIn("helper:", result.stdout)
                            self.assertIn(
                                "mkdir" if fragment == setup[0] else "install:",
                                result.stdout,
                            )
                            if fragment == install[0]:
                                self.assertIn("touch", result.stdout)
                failed_helper = setup[0].replace("source " + HELPER, "false")
                result = subprocess.run(
                    [
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        failed_helper + "printf continued",
                    ],
                    env={**os.environ, "PPU_TEST_BAKED_IMAGE": "1"},
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("continued", result.stdout)
        self.assertEqual(count, 4)

    def test_preserves_legacy_lanes_models_paths_commands_evidence_and_trends(self):
        self.assertEqual(set(LEGACY_CONTRACT_HASHES), set(WORKER_COUNTS))
        for name, document in self.documents.items():
            with self.subTest(workflow=name):
                self.assertEqual(
                    legacy_contract(document), LEGACY_CONTRACT_HASHES[name]
                )


if __name__ == "__main__":
    unittest.main()

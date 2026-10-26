"""Host-only candidate runtime contracts; no Docker, pip, or PPU execution."""

import json
import os
import shlex
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/ci/ppu"
EMBEDDED = "/sgl-workspace/sglang"
SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
KINDS = ("ANSWER", "PERF", "ACCURACY", "PD_PERF")
BOARDS = (
    "run_answer_suite_board.sh",
    "run_perf_suite_board.sh",
    "run_accuracy_suite_board.sh",
    "run_pd_perf_suite_node.sh",
)
GOOD_LOG = "Ran 2 tests in 0.01s\n\nOK (skipped=1)\nTest Summary: 1/1 passed\n"


class RuntimeHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.embedded = self.root / "embedded"
        self.checkout = self.root / "checkout"
        self.bin = self.root / "bin"
        self.home = self.root / "home"
        self.log = self.root / "commands"
        for path in (self.embedded / "test", self.checkout, self.bin, self.home):
            path.mkdir(parents=True)
        self.package = self.embedded / "python/sglang"
        self.package.mkdir(parents=True)
        (self.package / "__init__.py").write_text("")
        self.python = self.root / "python-venv/bin/python3"
        venv.EnvBuilder(with_pip=False).create(self.python.parent.parent)
        site = subprocess.check_output(
            [str(self.python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
        self.pth = Path(site) / "embedded.pth"
        self.pth.write_text(str(self.embedded / "python") + "\n")
        self.executable(
            self.bin / "python3",
            f'#!/bin/bash\nexec {shlex.quote(str(self.python))} "$@"\n',
        )
        self.executable(
            self.bin / "git",
            '#!/bin/bash\nprintf "git %s\\n" "$*" >> "$COMMAND_LOG"\n'
            '[[ "$*" == *"rev-parse HEAD" ]] || exit 91\n'
            'printf "%s\\n" "$EMBEDDED_HEAD"\nexit "${GIT_RC:-0}"\n',
        )
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PPU_", "SGLANG_", "PYTHON", "EVALSCOPE_"))
        }
        self.env.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            HOME=str(self.home),
            PYTHONDONTWRITEBYTECODE="1",
            COMMAND_LOG=str(self.log),
            EMBEDDED_HEAD=SHA,
            PPU_TEST_BAKED_IMAGE="1",
            PPU_BASE_IMAGE=f"registry.example/sglang:candidate@{DIGEST}",
            PPU_BASE_IMAGE_DIGEST=DIGEST,
            PPU_CI_SOURCE_SHA=SHA,
            SGLANG_BUILD_COMMIT=SHA,
            GITHUB_WORKSPACE=str(self.checkout),
        )

    def executable(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.write_text(text)
        path.chmod(0o755)

    def copy_script(self, name):
        source = SCRIPTS / name
        self.assertTrue(source.is_file(), f"missing candidate helper: {name}")
        target = self.embedded / "scripts/ci/ppu" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text().replace(EMBEDDED, str(self.embedded)))
        return target

    def run_shell(self, command, **env):
        return subprocess.run(
            ["bash", "-c", command],
            cwd=self.checkout,
            env=dict(self.env, **env),
            text=True,
            capture_output=True,
        )

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def prepare_suite(self):
        helper = self.copy_script("run_baked_suite.sh")
        self.copy_script("use_nightly_image.sh")
        self.copy_script("verify_nightly_result.py")
        (self.embedded / "test/run_suite.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['SUITE_ARGS']).write_text(json.dumps({"
            "'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'workspace': os.environ.get('GITHUB_WORKSPACE')}))\n"
            "print(os.environ['SUITE_LOG'], end='', flush=True)\n"
            "sys.exit(int(os.environ.get('SUITE_RC', '0')))\n"
        )
        self.env.update(SUITE_ARGS=str(self.root / "args.json"), SUITE_LOG=GOOD_LOG)
        return helper


class TestImageSelection(RuntimeHarness):
    def source(self, **env):
        helper = self.copy_script("use_nightly_image.sh")
        return self.run_shell(
            f"set -eu; source {shlex.quote(str(helper))}; "
            "python3 -c 'import os, json; print(json.dumps(dict(os.environ), sort_keys=True))'",
            **env,
        )

    def test_valid_candidate_sanitizes_imports_and_remaps_only_configs(self):
        values = {
            f"SGLANG_PPU_{kind}_TEST_CONFIG": f"/workspace/source/test/{kind}.json"
            for kind in KINDS
        }
        results = {
            f"SGLANG_PPU_{kind}_RESULTS_DIR": f"/workspace/source/results/{kind}"
            for kind in KINDS
        }
        (self.checkout / "sglang.py").write_text(
            "raise RuntimeError('checkout import')\n"
        )
        result = self.source(
            PYTHONPATH=str(self.checkout), PYTHONHOME="/missing", **values, **results
        )
        self.assert_ok(result)
        actual = json.loads(result.stdout.splitlines()[-1])
        for kind in KINDS:
            self.assertEqual(
                actual[f"SGLANG_PPU_{kind}_TEST_CONFIG"],
                str(self.embedded / f"test/{kind}.json"),
            )
        for key, value in results.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual["GITHUB_WORKSPACE"], str(self.embedded))
        self.assertEqual(actual["PWD"], str(self.embedded))
        self.assertEqual(actual["SGLANG_PPU_SOURCE_REVISION"], SHA)
        self.assertEqual(actual["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("PYTHONPATH", actual)
        self.assertNotIn("PYTHONHOME", actual)

    def test_external_configs_and_unset_configs_are_unchanged(self):
        result = self.source(
            SGLANG_PPU_ANSWER_TEST_CONFIG="/nas/config.json",
            SGLANG_PPU_PERF_TEST_CONFIG="/workspace/source-other/config.json",
        )
        self.assert_ok(result)
        actual = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(actual["SGLANG_PPU_ANSWER_TEST_CONFIG"], "/nas/config.json")
        self.assertEqual(
            actual["SGLANG_PPU_PERF_TEST_CONFIG"], "/workspace/source-other/config.json"
        )
        self.assertNotIn("SGLANG_PPU_ACCURACY_TEST_CONFIG", actual)

    def test_requires_explicit_candidate_flag_and_forbids_pr(self):
        for change in (
            {"PPU_TEST_BAKED_IMAGE": ""},
            {"PPU_TEST_BAKED_IMAGE": "0"},
            {"PPU_TEST_BAKED_IMAGE": "true"},
            {"PPU_E2E_PLAN": "/plan.json"},
            {"PPU_INSTALL_PROFILE": "pr"},
        ):
            with self.subTest(change=change):
                result = self.source(**change)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("PPU_", result.stderr)

    def test_missing_malformed_or_mismatched_digest_is_rejected(self):
        for image, digest in (
            ("", DIGEST),
            ("registry.example/sglang:latest", DIGEST),
            (f"registry.example/sglang@{DIGEST}", ""),
            (f"registry.example/sglang@{DIGEST}", "sha256:" + "c" * 64),
            ("registry.example/sglang@sha256:abc", "sha256:abc"),
            ("registry.example/sglang@sha256:" + "g" * 64, "sha256:" + "g" * 64),
            (f"@{DIGEST}", DIGEST),
            (f"registry.example/sglang@{DIGEST}\n", DIGEST),
        ):
            with self.subTest(image=image, digest=digest):
                result = self.source(PPU_BASE_IMAGE=image, PPU_BASE_IMAGE_DIGEST=digest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("PPU_BASE_IMAGE", result.stderr)

    def test_source_sha_metadata_and_git_head_must_agree(self):
        for key in ("PPU_CI_SOURCE_SHA", "SGLANG_BUILD_COMMIT", "EMBEDDED_HEAD"):
            for value in ("", "unknown", "a" * 7, "g" * 40, "c" * 40):
                with self.subTest(key=key, value=value):
                    result = self.source(**{key: value})
                    self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.source(GIT_RC="42").returncode, 0)

    def test_failures_propagate_when_sourced_without_errexit(self):
        helper = self.copy_script("use_nightly_image.sh")
        result = self.run_shell(
            f"if source {shlex.quote(str(helper))}; then exit 0; else exit 17; fi",
            PPU_BASE_IMAGE_DIGEST="bad",
        )
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)

    def test_import_must_succeed_and_resolve_inside_embedded_python(self):
        (self.package / "__init__.py").write_text(
            "raise ImportError('native extension unavailable')\n"
        )
        result = self.source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native extension unavailable", result.stderr)
        (self.package / "__init__.py").write_text("")
        external = self.root / "external/sglang"
        external.mkdir(parents=True)
        (external / "__init__.py").write_text("")
        self.pth.write_text(str(external.parent) + "\n")
        self.assertNotEqual(self.source().returncode, 0)
        self.pth.write_text(str(self.embedded / "python") + "\n")
        (self.package / "__init__.py").unlink()
        self.package.rmdir()
        self.package.symlink_to(external, target_is_directory=True)
        self.assertNotEqual(self.source().returncode, 0)


class TestBakedSuite(RuntimeHarness):
    def test_runs_embedded_suite_with_original_arguments_and_verifies_log(self):
        helper = self.prepare_suite()
        args = [
            "--hw",
            "ppu",
            "--suite",
            "answer suite",
            "--nightly",
            "--timeout-per-file",
            "37",
        ]
        result = self.run_shell(shlex.join(["bash", str(helper), *args]))
        self.assert_ok(result)
        actual = json.loads((self.root / "args.json").read_text())
        self.assertEqual(actual["args"], args + ["--require-nonempty"])
        self.assertEqual(actual["cwd"], str(self.embedded / "test"))
        self.assertEqual(actual["workspace"], str(self.embedded))
        self.assertIn('"executed": 1', result.stdout)

    def test_python_failure_is_not_masked_by_tee_or_valid_summary(self):
        helper = self.prepare_suite()
        result = self.run_shell(
            shlex.join(["bash", str(helper), "--suite", "fixture"]), SUITE_RC="37"
        )
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)

    def test_empty_all_skipped_missing_summary_and_failed_logs_are_rejected(self):
        helper = self.prepare_suite()
        for log in (
            "",
            "Test Summary: 0/0 passed\n",
            "Ran 0 tests in 0.01s\nOK\nTest Summary: 1/1 passed\n",
            "Ran 2 tests in 0.01s\nOK (skipped=2)\nTest Summary: 1/1 passed\n",
            "Ran 1 test in 0.01s\nOK\n",
            "Ran 1 test in 0.01s\nFAILED (failures=1)\nTest Summary: 1/1 passed\n",
            GOOD_LOG
            + "Ran 1 test in 0.01s\nOK (skipped=1)\nTest Summary: 2/2 passed\n",
        ):
            with self.subTest(log=log):
                self.assertNotEqual(
                    self.run_shell(
                        shlex.join(["bash", str(helper)]), SUITE_LOG=log
                    ).returncode,
                    0,
                )

    def test_tee_failure_is_not_masked(self):
        helper = self.prepare_suite()
        self.executable(self.bin / "tee", '#!/bin/bash\ncat > "$1"\nexit 23\n')
        result = self.run_shell(shlex.join(["bash", str(helper)]))
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

    def test_node_wrappers_preserve_candidate_and_legacy_rank_status(self):
        self.prepare_suite()
        for kind in ("answer", "perf"):
            wrapper = self.copy_script(f"run_{kind}_suite_node.sh")
            for candidate, rc, log in (
                ("1", "0", GOOD_LOG),
                ("1", "37", GOOD_LOG),
                ("1", "0", "Test Summary: 0/0 passed\n"),
                ("0", "19", GOOD_LOG),
            ):
                with self.subTest(kind=kind, candidate=candidate, rc=rc, log=log):
                    rank_dir = self.root / f"{kind} ranks"
                    result = self.run_shell(
                        f"cd {shlex.quote(str(self.embedded / 'test'))}; "
                        + shlex.join(
                            ["bash", str(wrapper), "fixture", "37", str(rank_dir)]
                        ),
                        PPU_TEST_BAKED_IMAGE=candidate,
                        SUITE_RC=rc,
                        SUITE_LOG=log,
                        **{f"SGLANG_PPU_{kind.upper()}_NODE_RANK": "2"},
                    )
                    expected = int(rc) if rc != "0" else (0 if log == GOOD_LOG else 1)
                    self.assertEqual(
                        result.returncode, expected, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        (rank_dir / "rank-2.status").read_text().strip(), str(expected)
                    )
                    self.assertIn(log.strip(), (rank_dir / "rank-2.log").read_text())
                    actual = json.loads((self.root / "args.json").read_text())["args"]
                    self.assertEqual("--require-nonempty" in actual, candidate == "1")


class TestBakedEvalScope(RuntimeHarness):
    def setUp(self):
        super().setUp()
        self.evalvenv = self.root / "evalscope-venv"
        self.env["EVALSCOPE_VENV"] = str(self.evalvenv)
        self.executable(
            self.bin / "python3",
            '#!/bin/bash\nprintf "python3 %s\\n" "$*" >> "$COMMAND_LOG"\nexit 93\n',
        )

    def run_setup(self, **env):
        return self.run_shell(
            shlex.join(["bash", str(SCRIPTS / "setup_evalscope.sh")]), **env
        )

    def test_missing_baked_venv_fails_without_install_attempt(self):
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baked", result.stderr.lower())
        self.assertFalse(
            self.log.exists(), "baked setup must not create a venv or invoke pip"
        )

    def make_evalscope(self, fields):
        venv.EnvBuilder(with_pip=False).create(self.evalvenv)
        site = subprocess.check_output(
            [
                str(self.evalvenv / "bin/python"),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            text=True,
        ).strip()
        pkg = Path(site) / "evalscope"
        (pkg / "api/metric").mkdir(parents=True)
        (pkg / "__init__.py").write_text('__version__ = "fixture"\n')
        (pkg / "api/metric/semantics.py").write_text("class MetricIdentity: pass\n")
        (pkg / "report.py").write_text(f"class Report: model_fields = {fields!r}\n")
        self.executable(
            self.evalvenv / "bin/evalscope", '#!/bin/bash\n[[ "$1" == --help ]]\n'
        )
        return pkg

    def test_baked_venv_validates_schema_without_pip(self):
        self.make_evalscope(
            dict.fromkeys(("schema_version", "metrics", "primary_metric_identity"))
        )
        result = self.run_setup()
        self.assert_ok(result)
        self.assertIn("report schema v2 present", result.stdout)
        self.assertFalse(self.log.exists())

    def test_baked_venv_bad_schema_or_import_fails_without_pip(self):
        pkg = self.make_evalscope({})
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Report.schema_version", result.stderr)
        (pkg / "__init__.py").write_text("raise ImportError('broken evalscope')\n")
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertFalse(self.log.exists())

    def test_legacy_still_attempts_to_create_missing_venv(self):
        result = self.run_setup(PPU_TEST_BAKED_IMAGE="0")
        self.assertEqual(result.returncode, 93)
        self.assertIn("-m venv", self.log.read_text())


class TestScriptIntegration(unittest.TestCase):
    def test_board_candidate_branch_precedes_config_and_skips_mutations(self):
        for name in BOARDS:
            with self.subTest(script=name):
                text = (SCRIPTS / name).read_text()
                candidate = text.find(
                    'if [[ "${PPU_TEST_BAKED_IMAGE:-0}" == "1" ]]; then'
                )
                self.assertGreaterEqual(candidate, 0, "missing candidate setup branch")
                self.assertIn(
                    "use_nightly_image.sh",
                    text[candidate : text.index("CUDA_VISIBLE_DEVICES=$(")],
                )
                self.assertIn("else\n  cd /workspace/source", text)
                self.assertIn(
                    'if [[ "${PPU_TEST_BAKED_IMAGE:-0}" != "1" ]]; then\n',
                    text[: text.index("mkdir -p ~/.pip")],
                )
                install = text.index("bash scripts/ci/ppu/ppu_install_dependency.sh")
                self.assertEqual(
                    text[:install].rstrip().splitlines()[-1],
                    'if [[ "${PPU_TEST_BAKED_IMAGE:-0}" != "1" ]]; then',
                )
                self.assertIn(
                    "fi\nfi",
                    text[install : text.index("bash scripts/ci/warm_page_cache.sh")],
                )
                self.assertIn('if [[ -n "${PPU_E2E_PLAN:-}" ]]; then', text)
                if name != "run_pd_perf_suite_node.sh":
                    self.assertIn("exec bash scripts/ci/ppu/run_baked_suite.sh", text)
                else:
                    self.assertIn("run_perf_suite_node.sh", text)

    def test_ci_target_bakes_isolated_evalscope_before_metadata(self):
        text = (ROOT / "docker/ppu.Dockerfile").read_text()
        ci = text.split("FROM framework AS ci", 1)[1].split("FROM ", 1)[0]
        self.assertIn(
            "bash /sgl-workspace/sglang/scripts/ci/ppu/setup_evalscope.sh", ci
        )
        self.assertLess(
            ci.index("setup_evalscope.sh"), ci.index("ARG SGLANG_BUILD_COMMIT")
        )


if __name__ == "__main__":
    unittest.main()

"""Dependency-free contracts for the PPU per-commit port."""

import ast
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[4]
STAGES = (
    "stage-a-test-1-gpu-ppu",
    "stage-b-test-1-gpu-ppu",
    "stage-c-test-2-gpu-ppu",
)


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


def load_registry():
    return load_module("ppu_port_registry", "python/sglang/test/ci/ci_register.py")


def load_suite(registry):
    ci_utils = types.ModuleType("sglang.test.ci.ci_utils")
    ci_utils.run_unittest_files = Mock(return_value=0)
    with patch.dict(
        sys.modules,
        {
            "sglang.test.ci.ci_register": registry,
            "sglang.test.ci.ci_utils": ci_utils,
            "tabulate": types.ModuleType("tabulate"),
        },
    ):
        return load_module("ppu_port_suite", "test/run_suite.py")


class PpuRegistryTest(unittest.TestCase):
    def test_legacy_and_modern_ppu_markers(self):
        registry = load_registry()
        visitor = registry.RegistryVisitor("test_ppu.py")
        visitor.visit(
            ast.parse(
                'register_ppu_ci(5, suite="stage-a-test-1-gpu-ppu")\n'
                'register_ppu_ci(10, stage="stage-b", runner_config="1-gpu-ppu")\n'
            )
        )
        self.assertEqual(len(visitor.registries), 2)
        self.assertEqual(
            [r.effective_suite for r in visitor.registries], list(STAGES[:2])
        )
        self.assertTrue(
            all(r.backend == registry.HWBackend.PPU for r in visitor.registries)
        )
        self.assertIsNone(registry.register_ppu_ci(5, suite=STAGES[0]))
        self.assertIn("register_ppu_ci", registry.__all__)

    def test_ppu_suites_and_target_cuda_suites(self):
        registry = load_registry()
        suite = load_suite(registry)
        ppu = suite.HW_MAPPING["ppu"]
        self.assertEqual(suite.PER_COMMIT_SUITES[ppu], list(STAGES))
        self.assertIn(ppu, suite._SUITE_CHECKED_BACKENDS)
        with self.assertRaisesRegex(ValueError, "invalid suites"):
            suite.validate_all_suites(
                [registry.CIRegistry(ppu, "test_ppu.py", 1, suite="misspelled")]
            )
        suite.validate_all_suites(
            [
                registry.CIRegistry(registry.HWBackend.CUDA, "test_cuda.py", 1, suite=s)
                for s in ("nightly-1-gpu", "weekly-8-gpu-h200")
            ]
        )

    def test_global_collection_and_target_exclusions(self):
        registry = load_registry()
        suite = load_suite(registry)
        args = types.SimpleNamespace(
            hw="cpu",
            suite="base-a-test-cpu",
            nightly=False,
            auto_partition_id=None,
            auto_partition_size=None,
            timeout_per_file=1200,
            enable_retry=False,
            continue_on_error=False,
            max_attempts=2,
            retry_wait_seconds=60,
        )
        with (
            patch.object(suite, "pretty_print_tests"),
            patch.object(
                suite, "collect_tests", wraps=registry.collect_tests
            ) as collect,
        ):
            self.assertEqual(suite.run_a_suite(args), 0)
        files = collect.call_args.args[0]
        self.assertTrue(files)
        self.assertFalse(any(f.endswith("/cpu/utils.py") for f in files))
        self.assertFalse(any(f.endswith("/run_tests.py") for f in files))


class RegistrationCheckerTest(unittest.TestCase):
    def check_files(self, files, content):
        checker = load_module(
            "ppu_port_checker", "scripts/lint/check_registered_tests.py"
        )
        registry = load_registry()
        # Keep the checker's local import dependency-free and independent of cwd.
        with (
            patch.object(
                checker.importlib.util, "module_from_spec", return_value=registry
            ),
            patch.object(
                checker.importlib.util,
                "spec_from_file_location",
                return_value=types.SimpleNamespace(loader=Mock()),
            ),
            patch.object(checker.glob, "glob", return_value=files),
            patch("builtins.open", mock_open(read_data=content)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return checker.main()

    def test_existing_cuda_nightly_weekly_and_stress_are_valid(self):
        for suite in ("nightly-1-gpu", "weekly-8-gpu-h200", "stress"):
            with self.subTest(suite=suite):
                self.assertEqual(
                    self.check_files(
                        ["test/registered/test_case.py"],
                        f'register_cuda_ci(1, suite="{suite}")\n',
                    ),
                    0,
                )

    def test_obsolete_cuda_pr_suite_is_rejected(self):
        self.assertEqual(
            self.check_files(
                ["test/registered/test_case.py"],
                'register_cuda_ci(1, suite="base-b-kernel-unit-1-gpu-large")\n',
            ),
            1,
        )

    def test_checker_matches_target_suite_exclusions(self):
        self.assertEqual(
            self.check_files(
                [
                    "test/registered/cpu/utils.py",
                    "test/registered/cpu/run_tests.py",
                    "test/registered/conftest.py",
                    "test/registered/__init__.py",
                ],
                "pass\n",
            ),
            0,
        )


class PpuKernelSourceTest(unittest.TestCase):
    def test_all_local_common_sources_exist(self):
        root = ROOT / "python/sglang/kernels/aot"
        tree = ast.parse((root / "setup_ppu.py").read_text())
        sources = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "common_sources"
                for t in node.targets
            )
        )
        self.assertEqual([s for s in sources if not (root / s).is_file()], [])

    def test_unavailable_operators_are_not_bound(self):
        cpp = (
            ROOT / "python/sglang/kernels/aot/csrc/common_extension_ppu.cc"
        ).read_text()
        for name in (
            "bmm_fp8",
            "qserve_w4a8_per_chn_gemm",
            "qserve_w4a8_per_group_gemm",
        ):
            self.assertNotIn(f'm.impl("{name}"', cpp)


class SlashHandlerCompatibilityTest(unittest.TestCase):
    def test_legacy_scheduled_cuda_rerun_mapping_is_preserved(self):
        from test_run_ci_permissions import handler

        for suite, runners in (
            ("nightly-1-gpu", ["1-gpu-large"]),
            ("weekly-8-gpu-h200", ["8-gpu-h200"]),
            ("nightly-8-gpu-common", ["8-gpu-h200", "8-gpu-b200"]),
        ):
            with (
                self.subTest(suite=suite),
                patch(
                    "builtins.open",
                    mock_open(read_data=f'register_cuda_ci(1, suite="{suite}")'),
                ),
                patch.object(
                    handler,
                    "_resolve_runner_config",
                    side_effect=lambda rc, path, suite: (rc, suite),
                ),
            ):
                self.assertEqual(
                    handler.detect_suite("registered/test_cuda.py"),
                    [(rc, suite) for rc in runners],
                )

    def test_repository_role_adds_only_tag_permission_without_mutation(self):
        from test_run_ci_permissions import handler

        repo = Mock()
        repo.get_collaborator_permission.return_value = "write"
        configured = {"can_rerun_failed_ci": False}
        with contextlib.redirect_stdout(io.StringIO()):
            resolved = handler.resolve_tag_run_ci_permissions(
                repo, "writer", configured
            )
        self.assertEqual(configured, {"can_rerun_failed_ci": False})
        self.assertEqual(
            resolved, {"can_rerun_failed_ci": False, "can_tag_run_ci_label": True}
        )

    def test_main_resolves_tag_commands_but_not_plain_reruns(self):
        from test_run_ci_permissions import handler

        for command in ("/tag-run-ci-label", "/tag-and-rerun-ci", "/rerun-failed-ci"):
            with self.subTest(command=command):
                repo = Mock()
                repo.get_pull.return_value.user.login = "another-author"
                repo.get_collaborator_permission.return_value = "write"
                env = {
                    "GITHUB_TOKEN": "test-token",
                    "REPO_FULL_NAME": "owner/repo",
                    "PR_NUMBER": "1",
                    "COMMENT_ID": "2",
                    "COMMENT_BODY": command,
                    "USER_LOGIN": "writer",
                }
                with (
                    patch.object(handler, "get_env_var", side_effect=env.__getitem__),
                    patch.object(handler, "load_permissions", return_value=None),
                    patch.object(handler, "Github") as github,
                    patch.object(
                        handler, "handle_tag_run_ci", return_value=False
                    ) as tag,
                    patch.object(handler, "handle_rerun_failed_ci", return_value=False),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    github.return_value.get_repo.return_value = repo
                    handler.main()
                if command.startswith("/tag"):
                    repo.get_collaborator_permission.assert_called_once_with("writer")
                    self.assertTrue(tag.call_args.args[3]["can_tag_run_ci_label"])
                else:
                    repo.get_collaborator_permission.assert_not_called()
                    tag.assert_not_called()


class PpuWorkflowStagesTest(unittest.TestCase):
    def test_full_ci_does_not_wait_for_human_review(self):
        workflow = (ROOT / ".github/workflows/ci.yaml").read_text()
        full_ci = workflow.split("  ci:\n", 1)[1]
        self.assertIn("    needs: registration-validation\n", full_ci)
        self.assertNotIn("human-review", full_ci)
        self.assertIn("  human-review:\n", workflow)

    def test_k8s_stages_are_serial_and_keep_checkout_ref(self):
        workflow = (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        self.assertIn("PPU_SDK_VERSION: 2.2.0", workflow)
        for index, stage in enumerate(STAGES):
            block = workflow.split(f"  {stage}:\n", 1)[1].split("\n  #", 1)[0]
            needs = ["check-changes"] + ([STAGES[index - 1]] if index else [])
            self.assertIn(f"    needs: [{', '.join(needs)}]\n", block)
            self.assertIn("runs-on: k8s-runner-group-cpu-flytiger", block)
            self.assertIn("uses: flytiger-eco/ppu-distributed-action@main", block)
            self.assertIn("ref: ${{ inputs.ref || github.sha }}", block)
            self.assertIn(f"--hw ppu --suite {stage}", block)
            if index:
                self.assertIn("!failure()", block)
                self.assertIn("!cancelled()", block)
        self.assertNotIn("test_anthropic_server.py", workflow)
        self.assertNotIn("27 no-runtime", workflow)
        self.assertIn("Stage C has no registered per-commit PPU tests", workflow)


if __name__ == "__main__":
    unittest.main()

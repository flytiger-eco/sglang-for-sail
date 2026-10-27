import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = ROOT / "scripts/ci/utils/slash_command_handler.py"

requests_stub = types.ModuleType("requests")
github_stub = types.ModuleType("github")


class _Auth:
    @staticmethod
    def Token(token):
        return token


github_stub.Auth = _Auth
github_stub.Github = object
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("github", github_stub)
sys.modules.setdefault("runner_configs", types.ModuleType("runner_configs"))

spec = importlib.util.spec_from_file_location("slash_command_handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class ResolveTagRunCiPermissionsTest(unittest.TestCase):
    def test_explicit_permission_skips_repository_lookup(self):
        repo = Mock()

        result = handler.resolve_tag_run_ci_permissions(
            repo, "configured-user", {"can_tag_run_ci_label": True}
        )

        self.assertTrue(result["can_tag_run_ci_label"])
        repo.get_collaborator_permission.assert_not_called()

    def test_write_or_higher_repository_role_is_authorized(self):
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "maintainer", None
                )

                self.assertTrue(result["can_tag_run_ci_label"])

    def test_lower_repository_role_is_not_authorized(self):
        for permission in ("none", "read", "triage"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "external-user", None
                )

                self.assertNotIn("can_tag_run_ci_label", result)

    def test_lookup_failure_fails_closed(self):
        repo = Mock()
        repo.get_collaborator_permission.side_effect = RuntimeError("API unavailable")

        result = handler.resolve_tag_run_ci_permissions(repo, "unknown-user", None)

        self.assertNotIn("can_tag_run_ci_label", result)


class RerunFailedCiTest(unittest.TestCase):
    def run_record(
        self,
        run_id=1,
        workflow_id=10,
        conclusion="failure",
        status="completed",
        path=".github/workflows/pr-test.yml",
    ):
        run = Mock()
        run.id = run_id
        run.workflow_id = workflow_id
        run.conclusion = conclusion
        run.status = status
        run.path = path
        run.name = path.rsplit("/", 1)[-1]
        return run

    def invoke(self, runs):
        repo = Mock()
        repo.get_workflow_runs.return_value = runs
        pr = Mock()
        pr.head.sha = "head-sha"
        pr.get_files.return_value = []
        comment = Mock()
        result = handler.handle_rerun_failed_ci(
            repo,
            pr,
            comment,
            {"can_rerun_failed_ci": True},
        )
        repo.get_workflow_runs.assert_called_once_with(head_sha="head-sha")
        return result

    def test_rerun_covers_all_unsuccessful_conclusions(self):
        for conclusion in ("failure", "skipped", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                run = self.run_record(conclusion=conclusion)
                self.assertTrue(self.invoke([run]))
                if conclusion == "skipped":
                    run.rerun.assert_called_once_with()
                    run.rerun_failed_jobs.assert_not_called()
                else:
                    run.rerun_failed_jobs.assert_called_once_with()
                    run.rerun.assert_not_called()

    def test_only_latest_run_per_workflow_and_recoverable_states_are_used(self):
        older = self.run_record(run_id=1, conclusion="failure")
        newer = self.run_record(run_id=2, conclusion="success")
        active = self.run_record(run_id=3, workflow_id=11, status="in_progress")
        approval = self.run_record(
            run_id=4, workflow_id=12, conclusion="action_required"
        )

        self.assertFalse(self.invoke([older, newer, active, approval]))
        for run in (older, newer, active, approval):
            run.rerun.assert_not_called()
            run.rerun_failed_jobs.assert_not_called()

    def test_selective_rerun_rejection_falls_back_to_full_rerun(self):
        run = self.run_record(conclusion="cancelled")
        run.rerun_failed_jobs.side_effect = RuntimeError("no failed jobs")

        self.assertTrue(self.invoke([run]))
        run.rerun_failed_jobs.assert_called_once_with()
        run.rerun.assert_called_once_with()

    def test_ppu_pipeline_always_requires_a_full_rerun(self):
        for conclusion in ("failure", "skipped", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                run = self.run_record(
                    conclusion=conclusion, path=".github/workflows/ci.yaml"
                )
                self.assertTrue(self.invoke([run]))
                run.rerun.assert_called_once_with()
                run.rerun_failed_jobs.assert_not_called()


if __name__ == "__main__":
    unittest.main()

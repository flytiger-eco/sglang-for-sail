import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "scripts" / "lint" / "check_registered_tests.py"
_CI_REGISTER = _REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"


class TestCheckRegisteredTests(unittest.TestCase):
    def run_checker(self, source: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            ci_register = root / "python" / "sglang" / "test" / "ci" / "ci_register.py"
            ci_register.parent.mkdir(parents=True)
            shutil.copyfile(_CI_REGISTER, ci_register)

            test_file = root / "test" / "registered" / "example.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text(source, encoding="utf-8")

            return subprocess.run(
                [sys.executable, str(_CHECKER)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_rejects_enabled_pytest_registry_without_main(self):
        result = self.run_checker("""
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base", runner_config="1-gpu")


def test_example():
    assert True
""")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("missing", result.stdout.lower())
        self.assertIn("__main__", result.stdout)

    def test_accepts_enabled_pytest_registry_with_main(self):
        result = self.run_checker("""
import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base", runner_config="1-gpu")


def test_example():
    assert True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
""")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_accepts_disabled_only_registry_without_main(self):
        result = self.run_checker("""
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=1,
    stage="base",
    runner_config="1-gpu",
    disabled="not supported",
)


def test_example():
    assert True
""")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_unittest_registry_without_main(self):
        result = self.run_checker("""
import unittest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base", runner_config="1-gpu")


class ExampleTest(unittest.TestCase):
    def test_example(self):
        self.assertTrue(True)
""")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("__main__", result.stdout)


if __name__ == "__main__":
    unittest.main()

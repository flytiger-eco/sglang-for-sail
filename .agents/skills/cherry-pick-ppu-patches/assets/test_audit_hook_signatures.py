import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("audit_hook_signatures.py")
SPEC = importlib.util.spec_from_file_location("audit_hook_signatures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class TestAuditHookSignatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_root = self.root / "python"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, relative_path, source):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def audit(self, hook_path):
        findings, errors = AUDIT.run_audit([hook_path], [self.source_root])
        self.assertEqual(errors, [])
        self.assertEqual(len(findings), 1)
        return findings[0]

    def test_replace_hook_missing_new_target_parameter_fails(self):
        self.write(
            "python/pkg/target.py",
            """
            class Backend:
                def apply(self, plan, bs, mode=None):
                    pass
            """,
        )
        hook = self.write(
            "hooks.py",
            """
            @plugin_hook("pkg.target.Backend.apply", type=HookType.REPLACE)
            def replacement(self, bs, mode=None):
                pass
            """,
        )

        finding = self.audit(hook)

        self.assertEqual(finding.status, "MISMATCH")
        self.assertIn("不兼容", finding.detail)

    def test_around_hook_resolves_static_fstring_and_inherited_method(self):
        self.write(
            "python/pkg/base.py",
            """
            class Base:
                def run(self, value, *, strict=False):
                    pass
            """,
        )
        self.write(
            "python/pkg/target.py",
            """
            from pkg.base import Base

            class Backend(Base):
                pass
            """,
        )
        hook = self.write(
            "hooks.py",
            """
            _BACKEND = "pkg.target.Backend"

            @plugin_hook(f"{_BACKEND}.run", type=HookType.AROUND)
            def around(original_fn, self, value, *, strict=False):
                return original_fn(self, value, strict=strict)
            """,
        )

        finding = self.audit(hook)

        self.assertEqual(finding.status, "OK")

    def test_universal_variadic_replace_accepts_any_existing_target_signature(self):
        self.write(
            "python/external/binary.py",
            """
            def kernel(value, *, scale=1):
                pass
            """,
        )
        hook = self.write(
            "hooks.py",
            """
            @plugin_hook("external.binary.kernel", type=HookType.REPLACE)
            def replacement(*args, **kwargs):
                pass
            """,
        )

        finding = self.audit(hook)

        self.assertEqual(finding.status, "OK")
        self.assertIn("*args, **kwargs", finding.detail)

    def test_missing_target_fails_closed_even_for_variadic_hook(self):
        hook = self.write(
            "hooks.py",
            """
            @plugin_hook("missing.binary.kernel", type=HookType.REPLACE)
            def replacement(*args, **kwargs):
                pass
            """,
        )

        finding = self.audit(hook)

        self.assertEqual(finding.status, "ERROR")
        self.assertIn("找不到 target 模块源码", finding.detail)

    def test_non_static_target_fails_closed(self):
        hook = self.write(
            "hooks.py",
            """
            def target_name():
                return "pkg.target.fn"

            @plugin_hook(target_name(), type=HookType.REPLACE)
            def replacement(value):
                pass
            """,
        )

        findings, errors = AUDIT.run_audit([hook], [self.source_root])

        self.assertEqual(findings, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("不是静态字符串表达式", errors[0])

    def test_around_hook_requires_original_fn_prefix(self):
        self.write(
            "python/pkg/target.py",
            """
            def run(value):
                pass
            """,
        )
        hook = self.write(
            "hooks.py",
            """
            @plugin_hook("pkg.target.run", type=HookType.AROUND)
            def around(value):
                pass
            """,
        )

        finding = self.audit(hook)

        self.assertEqual(finding.status, "MISMATCH")
        self.assertIn("original_fn", finding.detail)


if __name__ == "__main__":
    unittest.main()

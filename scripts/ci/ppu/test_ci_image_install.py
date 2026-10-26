"""Host-only contracts: exercise shell logic with pip/git/container paths mocked.

No Docker build, package installation, Rust compilation or PPU inference is run.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_ppu_build_kernel_cache import KernelCacheHarness

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/ci/ppu"
EMBEDDED = "/sgl-workspace/sglang"
KERNEL_INPUTS = (
    "python/sglang/kernels/aot",
    "python/sglang/kernels/jit/include",
    "scripts/ci/ppu/ppu_build_kernel.sh",
)
NATIVE_COMPONENTS = (
    ("sglang-kernel", KERNEL_INPUTS[0]),
    ("sglang-router", "sgl-model-gateway"),
)


class TestCIImageContract(unittest.TestCase):
    def test_ci_target_bakes_dependencies_kernel_eic_and_provenance(self):
        text = (ROOT / "docker/ppu.Dockerfile").read_text()
        stages = re.findall(r"^FROM .+ AS (\w+)", text, re.M)
        self.assertIn("ci", stages, "missing explicit CI target")
        self.assertEqual(stages[-1], "runtime")
        ci = text.split("FROM framework AS ci", 1)[1].split("FROM ", 1)[0]
        for required in (
            "pytest==8.3.5",
            "PyYAML==6.0.3",
            "jsonschema",
            "tabulate",
            "maturin",
            "dill>=0.3.8,<0.3.9",
            "uvicorn>=0.37.0",
            "-c /sgl-workspace/ppu_constraints.txt",
            "GITHUB_WORKSPACE=/sgl-workspace/sglang",
            "SGL_KERNEL_WHEEL_CACHE=0",
            "ppu_build_kernel.sh",
            "from=ppu-eic",
            "target=/tmp/ppu-eic",
            "install_ci_image.sh",
            "ARG SGLANG_BUILD_COMMIT",
            "ARG SGLANG_BUILD_URL",
            "ARG SGLANG_IMAGE_TAG",
            "ENV SGLANG_BUILD_COMMIT",
            "LABEL",
        ):
            self.assertIn(required, ci)
        self.assertNotIn("python[test", ci)
        self.assertLess(
            ci.index("ppu_build_kernel.sh"), ci.index("ARG SGLANG_BUILD_COMMIT")
        )
        self.assertNotRegex(
            (ROOT / ".dockerignore").read_text(), r"(?m)^\s*\.git/?\s*$"
        )

    def test_pr_dispatch_precedes_all_legacy_dependency_installs(self):
        text = (SCRIPTS / "ppu_install_dependency.sh").read_text()
        dispatch = text.find(
            'exec bash "${REPO_ROOT}/scripts/ci/ppu/install_pr_from_nightly.sh"'
        )
        self.assertGreater(dispatch, 0, "PR must exec its isolated installer")
        self.assertLess(dispatch, text.index("--upgrade pip wheel"))
        self.assertNotIn('if [[ "$PPU_INSTALL_PROFILE" == "pr" ]]', text)


class TestPRInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "pr"
        self.embedded = self.root / "embedded"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.dist_info = self.root / "dist-info"
        self.dist_info.mkdir()
        self.project = '[build-system]\nrequires = []\n[project]\nname = "sglang"\ndependencies = []\n'
        for repo in (self.repo, self.embedded):
            (repo / "python").mkdir(parents=True)
            (repo / "python/pyproject_ppu.toml").write_text(self.project)
            (repo / "python/pyproject.toml").write_text("old metadata\n")
            (repo / "scripts/ci/ppu").mkdir(parents=True)
            (repo / "sgl-model-gateway/bindings/python").mkdir(parents=True)
            (repo / "sgl-model-gateway/bindings/python/pyproject.toml").write_text(
                self.project
            )
            (repo / "scripts/ci/ppu/ppu_build_kernel.sh").write_text(
                '#!/bin/bash\npython3 -m kernel-test "$GITHUB_WORKSPACE" "${SGL_KERNEL_WHEEL_CACHE:-}"\nexit "${KERNEL_RC:-0}"\n'
            )
        self.env = dict(
            os.environ,
            GITHUB_WORKSPACE=str(self.repo),
            SGLANG_BUILD_COMMIT="a" * 40,
            TEST_EMBEDDED=str(self.embedded),
            COMMAND_LOG=str(self.log),
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TEST_DIST_INFO=str(self.dist_info),
            PYTHONPATH=os.pathsep.join(
                filter(None, (str(self.dist_info), os.environ.get("PYTHONPATH")))
            ),
            PPU_INSTALL_EVIDENCE_DIR=str(self.evidence),
            SGLANG_BUILD_RUST_EXTS="none",
            PPU_E2E_REQUIRE_PD="true",
        )
        self.executable(
            "git",
            f"""#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as f: f.write(json.dumps(['git', *args]) + '\\n')
repo = args[args.index('-C') + 1]
obj = args[-1]
if obj == 'HEAD': print('a' * 40 if repo == os.environ['TEST_EMBEDDED'] else 'b' * 40)
elif obj.startswith('HEAD:'):
    path = obj[5:]
    print(('changed-' if repo != os.environ['TEST_EMBEDDED'] and path == os.environ.get('CHANGED_TREE') else 'same-') + path)
else: sys.exit(1)
""",
        )
        self.executable(
            "python3",
            f"""#!{sys.executable}
import json, os, pathlib, re, sys
if len(sys.argv) > 2 and sys.argv[1] == '-m':
    with open(os.environ['COMMAND_LOG'], 'a') as f:
        f.write(json.dumps(['python3', *sys.argv[1:], 'rust=' + os.environ.get('SGLANG_BUILD_RUST_EXTS', '')]) + '\\n')
    component = None
    if sys.argv[2] == 'pip':
        rc = int(os.environ.get('PIP_RC', '0'))
        if rc: sys.exit(rc)
        if any(arg.endswith('.whl') for arg in sys.argv): component = 'sglang-router'
    elif sys.argv[2] == 'kernel-test' and os.environ.get('KERNEL_RC', '0') == '0':
        component = 'sglang-kernel'
    version = json.loads(os.environ.get('REBUILT_VERSIONS', '{{}}')).get(component)
    if version:
        # Simulate installation only by updating this test's private metadata.
        metadata = pathlib.Path(os.environ['TEST_DIST_INFO']) / (component.replace('-', '_') + '.dist-info/METADATA')
        metadata.write_text(re.sub(r'(?m)^Version: .*$', 'Version: ' + version, metadata.read_text()))
    sys.exit(0)
os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""",
        )
        for cmd in ("cargo", "maturin"):
            self.executable(
                cmd,
                f"""#!{sys.executable}
import json, os, pathlib, sys
with open(os.environ['COMMAND_LOG'], 'a') as f: f.write(json.dumps([{cmd!r}, *sys.argv[1:], 'cwd=' + os.getcwd()]) + '\\n')
if {cmd!r} == 'maturin':
    out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'sglang_router-0.3.2-py3-none-any.whl').write_text('fixture')
else:
    out = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / 'release/sgl-model-gateway'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text('fixture binary')
sys.exit(int(os.environ.get('GATEWAY_RC', '0')))
""",
            )

    def executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def run_install(self, **env):
        script = SCRIPTS / "install_pr_from_nightly.sh"
        self.assertTrue(script.exists(), "PR source installer is missing")
        text = script.read_text()
        self.assertIn(EMBEDDED, text)
        # Map immutable container locations only at the Docker boundary.
        text = text.replace(EMBEDDED, str(self.embedded)).replace(
            "/usr/local/bin/sgl-model-gateway", str(self.bin / "sgl-model-gateway")
        )
        fixture = self.root / "install.sh"
        fixture.write_text(text)
        return subprocess.run(
            ["bash", str(fixture)],
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
        )

    def commands(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def fake_distribution(self, name, version="1.0", extras=(), requires=()):
        path = self.dist_info / f"{name.replace('-', '_')}.dist-info"
        path.mkdir(exist_ok=True)
        (path / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
            + "".join(f"Requires-Dist: {req}\n" for req in requires)
        )

    def native_requirement(self, name, before="1.0", after="2.0", build=False):
        self.fake_distribution(name)
        # Exercise PEP 503 name normalization rather than literal spelling.
        requirement_name = name.upper().replace("-", ".")
        for repo, version in ((self.embedded, before), (self.repo, after)):
            (repo / "python/pyproject_ppu.toml").write_text(
                self.project.replace(
                    "requires = []" if build else "dependencies = []",
                    f'{"requires" if build else "dependencies"} = ["{requirement_name}=={version}"]',
                )
            )

    def assert_native_built(self, name):
        commands = self.commands()
        if name == "sglang-kernel":
            self.assertTrue(any("kernel-test" in c for c in commands), commands)
        else:
            self.assertTrue(any(c[0] == "maturin" for c in commands), commands)
            self.assertTrue(any(c[0] == "cargo" for c in commands), commands)
            self.assertTrue(
                any(
                    c[:3] == ["python3", "-m", "pip"]
                    and any(arg.endswith(".whl") for arg in c)
                    for c in commands
                ),
                commands,
            )

    def assert_no_install(self):
        self.assertFalse(any(c[0] != "git" for c in self.commands()), self.commands())

    def test_current_pr_editable_no_dependency_mutation_and_kernel_reuse(self):
        self.assert_ok(self.run_install())
        commands = self.commands()
        installs = [c for c in commands if c[:3] == ["python3", "-m", "pip"]]
        self.assertEqual(len(installs), 1, installs)
        for flag in ("--no-deps", "--no-build-isolation", "-e", "rust=all"):
            self.assertIn(flag, installs[0])
        self.assertIn(str(self.repo / "python"), installs[0])
        self.assertNotIn("--upgrade", installs[0])
        self.assertEqual(
            (self.repo / "python/pyproject.toml").read_text(), self.project
        )
        self.assertEqual(
            (self.embedded / "python/pyproject.toml").read_text(), "old metadata\n"
        )
        self.assertFalse(
            any("kernel-test" in c or c[0] in ("cargo", "maturin") for c in commands)
        )
        self.assertEqual(
            json.loads((self.evidence / "aot.json").read_text())["aot_seconds"], 0
        )
        for path in KERNEL_INPUTS:
            self.assertTrue(any(f"HEAD:{path}" in c for c in commands), path)

    def test_each_changed_kernel_input_builds_from_pr_not_a_stale_cache(self):
        for path in KERNEL_INPUTS:
            with self.subTest(path=path):
                self.log.unlink(missing_ok=True)
                self.assert_ok(self.run_install(CHANGED_TREE=path))
                build = [c for c in self.commands() if "kernel-test" in c]
                self.assertEqual(len(build), 1)
                self.assertEqual(build[0][3:5], [str(self.repo), "0"])
                self.assertIn(
                    "aot_seconds", json.loads((self.evidence / "aot.json").read_text())
                )

    def test_kernel_build_failure_is_fatal(self):
        result = self.run_install(CHANGED_TREE=KERNEL_INPUTS[0], KERNEL_RC="37")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertTrue((self.evidence / "aot.json").exists())

    def test_embedded_checkout_rejected_even_through_symlink(self):
        alias = self.root / "alias"
        alias.symlink_to(self.embedded, target_is_directory=True)
        for checkout in (self.embedded, alias):
            result = self.run_install(GITHUB_WORKSPACE=str(checkout))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("embedded", result.stderr.lower())
        self.assertFalse(
            any(c[:3] == ["python3", "-m", "pip"] for c in self.commands())
        )

    def test_missing_or_mismatching_provenance_rejected(self):
        for commit in ("", "unknown", "c" * 40):
            result = self.run_install(SGLANG_BUILD_COMMIT=commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SGLANG_BUILD_COMMIT", result.stderr)
        self.assertFalse(
            any(c[:3] == ["python3", "-m", "pip"] for c in self.commands())
        )

    def test_gateway_change_rebuilds_bindings_and_binary(self):
        self.assert_ok(self.run_install(CHANGED_TREE="sgl-model-gateway"))
        commands = self.commands()
        for cmd in ("maturin", "cargo"):
            self.assertTrue(
                any(
                    c[0] == cmd and str(self.repo / "sgl-model-gateway") in " ".join(c)
                    for c in commands
                )
            )
        wheel_installs = [
            c
            for c in commands
            if c[:3] == ["python3", "-m", "pip"] and any(".whl" in a for a in c)
        ]
        self.assertEqual(len(wheel_installs), 1)
        self.assertIn("--no-deps", wheel_installs[0])
        self.assertTrue((self.bin / "sgl-model-gateway").exists())

    def test_gateway_failure_is_fatal(self):
        result = self.run_install(CHANGED_TREE="sgl-model-gateway", GATEWAY_RC="41")
        self.assertEqual(result.returncode, 41, result.stdout + result.stderr)

    def test_unsatisfied_dependency_change_requires_candidate_image(self):
        (self.repo / "python/pyproject_ppu.toml").write_text(
            self.project.replace(
                "dependencies = []",
                'dependencies = ["ci-fixture-missing-package==999"]',
            )
        )
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate image", result.stdout + result.stderr)
        self.assertFalse(
            any(c[:3] == ["python3", "-m", "pip"] for c in self.commands())
        )

    def test_satisfied_dependency_change_does_not_install_dependencies(self):
        (self.repo / "python/pyproject_ppu.toml").write_text(
            self.project.replace(
                "dependencies = []", 'dependencies = ["packaging>=20"]'
            )
        )
        self.assert_ok(self.run_install())
        installs = [c for c in self.commands() if c[:3] == ["python3", "-m", "pip"]]
        self.assertEqual(len(installs), 1)
        self.assertNotIn("packaging>=20", installs[0])

    def test_unavailable_dependency_extra_requires_candidate_image(self):
        (self.repo / "python/pyproject_ppu.toml").write_text(
            self.project.replace(
                "dependencies = []",
                'dependencies = ["packaging[ci-fixture-missing-extra]"]',
            )
        )
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate image", result.stdout + result.stderr)

    def test_inactive_optional_dependency_changes_do_not_block_pr(self):
        for relative in (
            "python/pyproject_ppu.toml",
            "sgl-model-gateway/bindings/python/pyproject.toml",
        ):
            (self.repo / relative).write_text(
                self.project
                + '[project.optional-dependencies]\ntest = ["ci-fixture-missing-package==999"]\n'
            )
        self.assert_ok(self.run_install())
        installs = [c for c in self.commands() if c[:3] == ["python3", "-m", "pip"]]
        self.assertEqual(len(installs), 1)
        self.assertIn(str(self.repo / "python"), installs[0])

    def test_changed_native_pin_accepted_after_matching_rebuild(self):
        for name, tree in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.native_requirement(name)
                result = self.run_install(
                    CHANGED_TREE=tree, REBUILT_VERSIONS=json.dumps({name: "2.0"})
                )
                self.assert_ok(result)
                self.assert_native_built(name)

    def test_changed_native_pin_without_source_change_rejected_before_install(self):
        for name, _ in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.native_requirement(name)
                result = self.run_install(REBUILT_VERSIONS=json.dumps({name: "2.0"}))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("installed 1.0", result.stderr)
                self.assert_no_install()

    def test_wrong_rebuilt_version_rejected_after_build(self):
        for name, tree in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.native_requirement(name)
                result = self.run_install(
                    CHANGED_TREE=tree, REBUILT_VERSIONS=json.dumps({name: "3.0"})
                )
                self.assertNotEqual(result.returncode, 0)
                self.assert_native_built(name)
                self.assertIn("installed 3.0", result.stderr)

    def test_unchanged_native_pin_still_checked_after_rebuild(self):
        for name, tree in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.native_requirement(name, after="1.0")
                result = self.run_install(
                    CHANGED_TREE=tree, REBUILT_VERSIONS=json.dumps({name: "2.0"})
                )
                self.assert_native_built(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("installed 2.0", result.stderr)

    def test_native_build_requirement_not_deferred(self):
        for name, tree in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.native_requirement(name, build=True)
                result = self.run_install(
                    CHANGED_TREE=tree, REBUILT_VERSIONS=json.dumps({name: "2.0"})
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("installed 1.0", result.stderr)
                self.assert_no_install()

    def test_changed_external_requirement_not_deferred_for_native_build(self):
        (self.repo / "python/pyproject_ppu.toml").write_text(
            self.project.replace(
                "dependencies = []",
                'dependencies = ["ci-fixture-missing-package==999"]',
            )
        )
        result = self.run_install(CHANGED_TREE=KERNEL_INPUTS[0])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate image", result.stderr)
        self.assert_no_install()

    def test_changed_native_url_requirement_not_deferred(self):
        for name, tree in NATIVE_COMPONENTS:
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                (self.repo / "python/pyproject_ppu.toml").write_text(
                    self.project.replace(
                        "dependencies = []",
                        f'dependencies = ["{name} @ https://example.invalid/native.whl"]',
                    )
                )
                result = self.run_install(CHANGED_TREE=tree)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("URL requirement", result.stderr)
                self.assert_no_install()

    def test_requested_extras_check_transitive_dependencies_and_markers(self):
        self.fake_distribution(
            "ci-fixture-parent",
            extras=("foo",),
            requires=(
                'ci-fixture-child[bar]; extra == "foo"',
                'ci-fixture-absent; extra == "inactive"',
                'ci-fixture-absent; python_version < "2"',
            ),
        )
        self.fake_distribution(
            "ci-fixture-child",
            extras=("bar",),
            requires=('ci-fixture-leaf==2; extra == "bar"',),
        )
        (self.repo / "python/pyproject_ppu.toml").write_text(
            self.project.replace(
                "dependencies = []", 'dependencies = ["ci-fixture-parent[foo]"]'
            )
        )
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing ci-fixture-leaf", result.stderr)
        self.assert_no_install()
        self.fake_distribution("ci-fixture-leaf", version="2")
        self.assert_ok(self.run_install())

    def test_rebuilt_native_requested_extras_are_validated(self):
        self.native_requirement("sglang-kernel")
        path = self.repo / "python/pyproject_ppu.toml"
        path.write_text(
            path.read_text().replace("SGLANG.KERNEL==", "SGLANG.KERNEL[foo]==")
        )
        self.fake_distribution(
            "sglang-kernel",
            extras=("foo",),
            requires=('ci-fixture-missing-package; extra == "foo"',),
        )
        result = self.run_install(
            CHANGED_TREE=KERNEL_INPUTS[0],
            REBUILT_VERSIONS=json.dumps({"sglang-kernel": "2.0"}),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assert_native_built("sglang-kernel")
        self.assertIn("missing ci-fixture-missing-package", result.stderr)

    def test_source_install_failure_is_fatal(self):
        result = self.run_install(PIP_RC="43")
        self.assertEqual(result.returncode, 43, result.stdout + result.stderr)


class TestEICBuildInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pkg = self.root / "eic"
        self.pkg.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands"
        for name in (
            "ali-rdma-core_1.deb",
            "accl-barex-cuda13-1.deb",
            "u2mm.deb",
            "mooncake_transfer_engine-0.3.6.post1-py3-none-any.whl",
        ):
            (self.pkg / name).write_text("fixture")
        for cmd in ("dpkg", "python3", "ldconfig"):
            path = self.bin / cmd
            path.write_text(
                f"""#!/bin/sh
printf '%s\\n' '{cmd}' "$*" >> "$COMMAND_LOG"
if [ "${{FAIL_COMMAND:-}}" = '{cmd}' ]; then exit 39; fi
"""
            )
            path.chmod(0o755)

    def run_install(self, fail=""):
        path = SCRIPTS / "install_ci_image.sh"
        self.assertTrue(path.exists(), "build-time EIC installer missing")
        # Map ldconfig paths out of the host system; never call real dpkg/pip.
        script = self.root / "install.sh"
        script.write_text(
            path.read_text()
            .replace("/usr/lib64", str(self.root / "lib64"))
            .replace("/etc/ld.so.conf.d", str(self.root / "ld.conf.d"))
        )
        return subprocess.run(
            ["bash", str(script), str(self.pkg)],
            env=dict(
                os.environ,
                PATH=f"{self.bin}:{os.environ['PATH']}",
                COMMAND_LOG=str(self.log),
                FAIL_COMMAND=fail,
            ),
            text=True,
            capture_output=True,
        )

    def test_required_eic_and_barex_wheel_installed_without_dependencies(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self.log.read_text()
        for name in (
            "ali-rdma-core",
            "accl-barex-cuda13",
            "u2mm.deb",
            "mooncake_transfer_engine",
        ):
            self.assertIn(name, log)
        self.assertIn("--no-deps", log)

    def test_core_deb_or_wheel_failure_aborts_build(self):
        for cmd in ("dpkg", "python3"):
            result = self.run_install(fail=cmd)
            self.assertEqual(result.returncode, 39, result.stdout + result.stderr)
            self.assertNotIn("installation complete", result.stdout)

    def test_missing_named_context_artifacts_are_fatal(self):
        (self.pkg / "u2mm.deb").unlink()
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("u2mm", result.stdout + result.stderr)


class TestKernelBuildFailure(unittest.TestCase):
    def test_failed_pipeline_cannot_install_stale_wheel(self):
        harness = KernelCacheHarness()
        self.addCleanup(harness.close)
        (harness.aot / "dist").mkdir()
        (harness.aot / "dist/sglang_kernel-stale.whl").write_text("stale")
        harness._write_executable(
            "python3",
            """#!/bin/sh
case "$1" in
  setup_ppu.py) printf 'error: simulated compiler failure\n'; exit 37 ;;
  -m) printf 'STALE_WHEEL_INSTALLED\n'; exit 0 ;;
  -c) printf '2\n'; exit 0 ;;
esac
exit 1
""",
        )
        result = harness.run()
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertNotIn("STALE_WHEEL_INSTALLED", result.stdout)


if __name__ == "__main__":
    unittest.main()

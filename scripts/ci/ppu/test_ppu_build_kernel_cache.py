import hashlib
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts/ci/ppu/ppu_build_kernel.sh"
AOT_TREE = "aot-tree-object"
TOOLCHAIN = "2.11.0|13.0|cp312"


class KernelCacheHarness:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        self.aot = self.repo / "python/sglang/kernels/aot"
        self.aot.mkdir(parents=True)
        (self.aot / "setup_ppu.py").write_text("# test fixture\n")
        self.cache = self.root / "cache"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self._write_executable(
            "git",
            """#!/bin/sh
case "$*" in
  *HEAD:python/sglang/kernels/aot*) printf '%s\\n' 'aot-tree-object' ;;
  *--short*) printf '%s\\n' 'deadbee' ;;
  *) exit 1 ;;
esac
""",
        )
        self._write_executable(
            "python3",
            """#!/bin/sh
if [ "${1:-}" = "-c" ]; then
  case "${2:-}" in
    *"import sys"*) printf '%s\\n' '2.11.0|13.0|cp312' ;;
    *) exit 0 ;;
  esac
elif [ "${1:-}" = "-m" ]; then
  exit 0
fi
exit 1
""",
        )

    def close(self):
        self._tmp.cleanup()

    def _write_executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    @staticmethod
    def cache_key(image_identity, sdk_version):
        material = "|".join((AOT_TREE, TOOLCHAIN, image_identity, sdk_version))
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def run(self, *, image="", digest="", sdk_version="", cached_key=None):
        if cached_key:
            key_dir = self.cache / cached_key
            key_dir.mkdir(parents=True)
            (key_dir / "sglang_kernel-0.0.0-py3-none-any.whl").write_bytes(b"wheel")
        env = os.environ.copy()
        env.update(
            {
                "GITHUB_WORKSPACE": str(self.repo),
                "PATH": f"{self.bin}:{env['PATH']}",
                "SGL_KERNEL_WHEEL_CACHE_DIR": str(self.cache),
                "PPU_BASE_IMAGE": image,
                "PPU_BASE_IMAGE_DIGEST": digest,
                "PPU_SDK_VERSION": sdk_version,
            }
        )
        return subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


class TestKernelCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.harness = KernelCacheHarness()

    def tearDown(self):
        self.harness.close()

    def assert_cache_hit(self, result, key):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Cache HIT (key {key})", result.stdout)
        self.assertIn(f"SGL_KERNEL_LOCAL_VERSION: ppu.src.{key}", result.stdout)

    def test_image_tag_changes_cache_key(self):
        for image in ("registry/llm:sdk2.1.1", "registry/llm:sdk2.2.0"):
            with self.subTest(image=image):
                key = self.harness.cache_key(image, "2.2.0")
                result = self.harness.run(
                    image=image, sdk_version="2.2.0", cached_key=key
                )
                self.assert_cache_hit(result, key)

    def test_digest_takes_precedence_over_tag(self):
        digest = "sha256:" + "a" * 64
        key = self.harness.cache_key(digest, "2.2.0")
        result = self.harness.run(
            image="registry/llm:mutable",
            digest=digest,
            sdk_version="2.2.0",
            cached_key=key,
        )
        self.assert_cache_hit(result, key)
        self.assertIn("Cache image identity (digest)", result.stdout)

    def test_sdk_version_changes_cache_key(self):
        image = "registry/llm:stable"
        for sdk_version in ("2.1.1", "2.2.0"):
            with self.subTest(sdk_version=sdk_version):
                key = self.harness.cache_key(image, sdk_version)
                result = self.harness.run(
                    image=image, sdk_version=sdk_version, cached_key=key
                )
                self.assert_cache_hit(result, key)

    def test_missing_image_identity_bypasses_cache(self):
        result = self.harness.run(sdk_version="2.2.0")
        self.assertNotIn("Cache HIT", result.stdout)
        self.assertIn("Wheel cache unavailable (missing image identity)", result.stdout)

    def test_missing_sdk_version_bypasses_cache(self):
        result = self.harness.run(image="registry/llm:sdk2.2.0")
        self.assertNotIn("Cache HIT", result.stdout)
        self.assertIn(
            "Wheel cache unavailable (missing PPU_SDK_VERSION)", result.stdout
        )


if __name__ == "__main__":
    unittest.main()

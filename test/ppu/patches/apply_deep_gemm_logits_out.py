"""Apply the Python-only out interface to the verified PPU DeepGEMM package.

Run inside the task container before importing DeepGEMM in server processes.
Unknown package contents fail closed; native libraries are never modified.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

BEFORE = "a7ebbc6eacb8229f4f3a176f6c55d333760ca1eed09c2c605576a1c5cc919f24"
AFTER = "9543d6578077e4897ce515ecd33e566b60da3d456c98c941588915b1cf3171eb"


def apply_unified(original: str, patch: str) -> str:
    source = original.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    result = []
    position = 0
    i = 0
    while i < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        if not match:
            i += 1
            continue
        start = int(match[1]) - 1
        if start < position:
            raise ValueError("Overlapping dependency patch hunks")
        result.extend(source[position:start])
        position = start
        old_count = new_count = 0
        i += 1
        while i < len(lines) and lines[i][:1] in (" ", "+", "-"):
            operation, content = lines[i][0], lines[i][1:]
            if operation in (" ", "-"):
                if position >= len(source) or source[position] != content:
                    raise ValueError(
                        f"Dependency patch context differs at line {position + 1}"
                    )
                position += 1
                old_count += 1
            if operation in (" ", "+"):
                result.append(content)
                new_count += 1
            i += 1
        if old_count != int(match[2] or 1) or new_count != int(match[4] or 1):
            raise ValueError("Dependency patch hunk length differs")
    result.extend(source[position:])
    return "".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root
    if root is None:
        spec = importlib.util.find_spec("deep_gemm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Installed DeepGEMM Python package not found")
        root = Path(spec.origin).parent
    target = root / "jit_kernels" / "attention.py"
    original = target.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == AFTER:
        print(
            json.dumps(
                {"state": "already-applied", "path": str(target), "sha256": digest}
            )
        )
        return
    if digest != BEFORE:
        raise RuntimeError(
            f"Unsupported DeepGEMM Python source SHA256 {digest}; inspect before porting patch"
        )
    patch = Path(__file__).with_name("deep_gemm_mqa_logits_out.patch").read_text()
    updated = apply_unified(original.decode(), patch).encode()
    if hashlib.sha256(updated).hexdigest() != AFTER:
        raise RuntimeError(
            "Patched DeepGEMM source hash differs; no package file changed"
        )
    compile(updated, str(target), "exec")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / "attention.py"
    if backup.exists() and backup.read_bytes() != original:
        raise RuntimeError("Backup directory contains different original contents")
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_suffix(".py.workspace-tmp")
    temporary.write_bytes(updated)
    temporary.chmod(target.stat().st_mode)
    temporary.replace(target)
    for cached in (target.parent / "__pycache__").glob("attention.*.pyc"):
        cached.unlink()
    print(
        json.dumps(
            {
                "state": "applied",
                "path": str(target),
                "before": BEFORE,
                "after": AFTER,
                "backup": str(backup),
            }
        )
    )


if __name__ == "__main__":
    main()

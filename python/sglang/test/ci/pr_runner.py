"""Opt-in, fail-closed pytest execution. Stdlib-only for cheap contract tests."""

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize(text):
    # Never publish credentials in URLs, headers, environment dumps, or logs.
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    # Authorization carries scheme + credential ("Bearer abc", "Basic xyz");
    # redact the whole value, not just the scheme word.
    text = re.sub(
        r"(?im)(authorization\s*[:=][ \t]*)(?:[^\s,]+[ \t]+)?[^\s,]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?im)((?:password|token|secret|api[_-]?key)\s*[:=]\s*)[^\s,]+",
        r"\1[REDACTED]",
        text,
    )
    for key, value in os.environ.items():
        if re.search(r"PASSWORD|TOKEN|SECRET|API_KEY", key) and len(value) >= 4:
            text = text.replace(value, "[REDACTED]")
    return text


class InvalidResults(ValueError):
    def __init__(self, counts, bad_skips):
        self.counts = counts
        super().__init__(
            f"Invalid test results: {counts}; unexpected skips: {bad_skips}"
        )


def inspect_junit(path, allowed_skips=()):
    # Reports are untrusted input. Bound memory and prohibit DTD/entity expansion;
    # decode UTF-8 first so alternate encodings cannot bypass the declaration guard.
    with Path(path).open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("JUnit report exceeds 16 MiB")
    text = data.decode("utf-8")
    if "\x00" in text or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
        raise ValueError("JUnit DTD/entities are forbidden")
    root = ET.fromstring(text)
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError("Invalid JUnit root")
    cases = list(root.iter("testcase"))
    skipped = [c for c in cases if c.find("skipped") is not None]
    bad_skips = [
        f"{c.get('classname')}.{c.get('name')}"
        for c in skipped
        if f"{c.get('classname')}.{c.get('name')}" not in allowed_skips
    ]
    counts = {
        "collected": len(cases),
        "executed": len(cases) - len(skipped),
        "skipped": len(skipped),
        "failures": len(list(root.iter("failure"))) + len(list(root.iter("error"))),
    }
    if counts["executed"] == 0 or counts["failures"] or bad_skips:
        raise InvalidResults(counts, bad_skips)
    return counts


def stop_process(process):
    # pytest and all of its server/distributed descendants share this session.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _version_tuple(version):
    return tuple(int(part) for part in re.findall(r"\d+", version)[:4])


def _package_installed(spec):
    if "--force-reinstall" in spec:
        return False
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:(==|>=)([0-9.]+))?$", spec)
    if not match:
        return False
    name, comparator, required = match.groups()
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    if comparator == "==":
        return installed == required
    if comparator == ">=":
        return _version_tuple(installed) >= _version_tuple(required)
    return True


def _dependency_action(root, filename):
    manifest = root / "scripts/ci/ppu/test_dependency_manifest.tsv"
    with manifest.open(newline="") as stream:
        rows = csv.DictReader(
            (line for line in stream if not line.startswith("#")), delimiter="\t"
        )
        for row in rows:
            if str(filename).endswith(row["file"]):
                return row
    return None


def select_manifest_registrations(all_tests, repo_root, manifest_path, nightly_suite):
    """Select exact source registrations assigned to a nightly suite."""
    root = Path(repo_root).resolve()
    with Path(manifest_path).open(newline="") as stream:
        rows = csv.DictReader(
            (line for line in stream if not line.startswith("#")), delimiter="\t"
        )
        wanted = {
            (row["file"], row["source_suite"])
            for row in rows
            if row["nightly_suite"] == nightly_suite
        }

    selected = []
    found = set()
    for test in all_tests:
        relative = Path(
            os.path.relpath(Path(test.filename).resolve(), root)
        ).as_posix()
        key = (relative, test.effective_suite)
        if key in wanted:
            selected.append(test)
            found.add(key)

    missing = wanted - found
    if missing:
        formatted = ", ".join(f"{suite}:{filename}" for filename, suite in missing)
        raise ValueError(f"Manifest registrations not found: {formatted}")

    enabled = [test for test in selected if test.disabled is None]
    skipped = [test for test in selected if test.disabled is not None]
    return enabled, skipped


def partition_manifest_registrations(all_tests, size):
    """Balance estimated time without repeating a filename in one partition."""
    partitions = [[] for _ in range(size)]
    totals = [0.0] * size
    filenames = [set() for _ in range(size)]
    for test in sorted(all_tests, key=lambda item: item.est_time, reverse=True):
        candidates = [
            index for index in range(size) if test.filename not in filenames[index]
        ]
        if not candidates:
            raise ValueError(
                f"Cannot place duplicate manifest registration: {test.filename}"
            )
        index = min(
            candidates, key=lambda item: (totals[item], len(partitions[item]), item)
        )
        partitions[index].append(test)
        totals[index] += test.est_time
        filenames[index].add(test.filename)
    return partitions


_COMPLETED_BUILDS = set()


def _ensure_rust_toolchain(root):
    key = ("rust-toolchain", root)
    if key in _COMPLETED_BUILDS:
        return
    subprocess.run(
        ["bash", str(root / "scripts/ci/utils/install_rustup.sh")], check=True
    )
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    os.environ["PATH"] = (
        str(cargo_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    toolchain = root / "rust/rust-toolchain.toml"
    match = re.search(
        r'^channel\s*=\s*"([^"]+)"',
        toolchain.read_text() if toolchain.is_file() else "",
        re.MULTILINE,
    )
    os.environ["RUSTUP_TOOLCHAIN"] = match.group(1) if match else "stable"
    _COMPLETED_BUILDS.add(key)


def _pip_install_command(packages):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        *packages,
    ]
    if any("DeepGEMM-for-sail.git" in package for package in packages):
        setup = (
            "source /usr/local/PPU_SDK/envsetup.sh && "
            "export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/CUDA_SDK/include"
            "${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}} && "
            "export CFLAGS=\"-D__CUDA_FP16_H__ -D__CUDA_BF16_H__ "
            "-D__CUDA_FP8_H__ ${CFLAGS:-}\" && "
            "export CXXFLAGS=\"-D__CUDA_FP16_H__ -D__CUDA_BF16_H__ "
            "-D__CUDA_FP8_H__ ${CXXFLAGS:-}\" && "
            "export MAX_JOBS=20 PYTORCH_SAIL_ARCH='ppu_10;ppu_15' && "
        )
        return ["bash", "-c", setup + shlex.join(command)]
    return command


def ensure_test_dependencies(filename):
    root = Path(os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[4]))
    record = _dependency_action(root, filename)
    if record is None:
        return {}
    action = record["action"]
    if action == "pythonpath":
        return {
            "PYTHONPATH": str(root / "test")
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        }
    if action == "build-rust":
        key = (action, root)
        if key not in _COMPLETED_BUILDS:
            _ensure_rust_toolchain(root)
            from sglang.srt.rust_extensions import load_rust_extension

            load_rust_extension("sglang.srt.rust_extensions._server")
            _COMPLETED_BUILDS.add(key)
        return {}
    if action == "build-kernel":
        key = (action, root)
        if key not in _COMPLETED_BUILDS:
            subprocess.run(
                ["bash", str(root / "scripts/ci/ppu/ppu_build_kernel.sh")], check=True
            )
            _COMPLETED_BUILDS.add(key)
        return {}
    if action != "pip":
        raise ValueError(f"Unsupported test dependency action: {action}")
    packages = shlex.split(record["packages"])
    missing = [package for package in packages if not _package_installed(package)]
    if missing:
        subprocess.run(_pip_install_command(missing), check=True)
    return {}


def run_pytest_files(
    files,
    report_dir,
    timeout_per_file,
    suite_timeout=None,
    continue_on_error=False,
    allowed_skips=(),
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    selected = [str(Path(f).resolve()) for f in files]
    manifest = {
        "selected": selected,
        "files": [],
        "success": False,
        "allowed_skips": list(allowed_skips),
    }
    summary = directory / "summary.json"

    def save():
        summary.write_text(json.dumps(manifest, indent=2) + "\n")

    save()  # An interrupted run leaves an explicitly incomplete report.
    if not selected or len(set(selected)) != len(selected):
        return 1
    started = time.monotonic()
    deadline = started + suite_timeout if suite_timeout is not None else float("inf")
    for filename in selected:
        name = (
            Path(filename).stem
            + "-"
            + hashlib.sha256(filename.encode()).hexdigest()[:12]
        )
        junit = directory / f"{name}.xml"
        log = directory / f"{name}.log"
        if junit.exists():
            # Reject stale evidence rather than accepting a previous run's XML.
            raise ValueError(f"Report already exists: {junit}")
        record = {
            "file": filename,
            "junit": junit.name,
            "log": log.name,
            "passed": False,
            "executed": 0,
            "skipped": 0,
        }
        manifest["files"].append(record)
        tic = time.monotonic()
        process = None
        handlers = {}

        def interrupted(signum, frame):
            raise InterruptedError(f"runner received signal {signum}")

        try:
            handlers = {
                sig: signal.signal(sig, interrupted)
                for sig in (signal.SIGTERM, signal.SIGINT)
            }
            remaining = min(timeout_per_file, deadline - tic)
            if remaining <= 0:
                raise TimeoutError("suite budget exhausted")
            env = dict(
                os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""
            )
            # -o addopts= prevents repository configuration from changing selection.
            command = [
                sys.executable,
                "-m",
                "pytest",
                filename,
                "-v",
                "-o",
                "addopts=",
                "-o",
                "junit_family=xunit2",
                f"--junitxml={junit}",
            ]
            env.update(ensure_test_dependencies(filename))
            with log.open("w+", errors="replace") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )

                try:
                    record["returncode"] = process.wait(timeout=remaining)
                finally:
                    stop_process(process)
                    process = None
                    output.seek(0)
                    cleaned = sanitize(output.read())
                    output.seek(0)
                    output.write(cleaned)
                    output.truncate()
            counts = inspect_junit(junit, allowed_skips)
            record.update(counts)
            record["passed"] = record["returncode"] == 0
        except (
            OSError,
            ValueError,
            ET.ParseError,
            TimeoutError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
        ) as exc:
            if isinstance(exc, InvalidResults):
                record.update(exc.counts)
            if isinstance(exc, InterruptedError):
                continue_on_error = False
            record["error"] = sanitize(str(exc))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if process is not None:
                stop_process(process)
            if junit.exists():
                junit.write_text(sanitize(junit.read_text(errors="replace")))
            record["elapsed"] = round(time.monotonic() - tic, 3)
            save()
        print(json.dumps(record), flush=True)
        if not record["passed"] and not continue_on_error:
            break
    manifest["elapsed"] = round(time.monotonic() - started, 3)
    manifest["success"] = len(manifest["files"]) == len(selected) and all(
        r["passed"] for r in manifest["files"]
    )
    save()
    return 0 if manifest["success"] else 1

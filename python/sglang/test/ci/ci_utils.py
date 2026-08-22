import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from sglang.srt.debug_utils import cuda_coredump
from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import CIRegistry

# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


# Patterns that indicate retriable accuracy/performance failures
RETRIABLE_PATTERNS = [
    r"AssertionError:.*not greater than",
    r"AssertionError:.*not less than",
    r"AssertionError:.*not equal to",
    r"AssertionError:.*!=.*expected",
    r"accuracy",
    r"score",
    r"latency",
    r"throughput",
    r"timeout",
]

# Patterns that indicate non-retriable failures (real code errors)
NON_RETRIABLE_PATTERNS = [
    r"SyntaxError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"NameError",
    r"TypeError",
    r"AttributeError",
    r"RuntimeError",
    r"CUDA out of memory",
    r"OOM",
    r"Segmentation fault",
    r"core dumped",
    r"ConnectionRefusedError",
    r"FileNotFoundError",
]


def is_retriable_failure(output: str) -> tuple[bool, str]:
    """
    Determine if a test failure is retriable based on output patterns.

    Returns:
        tuple: (is_retriable, reason)
    """
    # Check for non-retriable patterns first
    for pattern in NON_RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return False, f"non-retriable error: {pattern}"

    # Check for retriable patterns
    for pattern in RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return True, f"retriable pattern: {pattern}"

    # If we have an AssertionError but didn't match non-retriable, assume retriable
    if re.search(r"AssertionError", output):
        return True, "AssertionError (assuming retriable)"

    # Default: not retriable
    return False, "unknown failure type"


def _parse_unittest_counts(line: str) -> Optional[int]:
    """Number of tests executed from a unittest 'Ran N test(s) in ...' line."""
    m = re.search(r"\bRan (\d+) tests?\b", line)
    return int(m.group(1)) if m else None


def _parse_pytest_counts(line: str) -> dict:
    """Counts from a pytest terminal summary line ('= 2 passed, 3 skipped in 1s =')."""
    counts = {}
    for m in re.finditer(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)", line):
        key = "error" if m.group(2) == "errors" else m.group(2)
        counts[key] = counts.get(key, 0) + int(m.group(1))
    return counts


def _is_all_tests_skipped(output: str) -> bool:
    """Detect 'tests were collected but every one of them skipped'.

    The third state between pass and fail that exit codes cannot express:
    unittest.main() exits 0 with 'OK (skipped=N)' and pytest exits 0 with
    an 'N skipped' summary, both indistinguishable from a real pass by
    return code alone. Recognises both runners:

    - unittest: the summary line after 'Ran N tests' must be exactly
      'OK (skipped=M)' with M == N -- a single item whose count covers
      every executed test. Any extra item such as 'expected failures' or
      'unexpected successes' means a test body actually ran (an xfail
      that ran is not a skip), and skipped < ran means the remaining
      tests passed for real; neither form is zero coverage
    - pytest: the last summary line carrying outcome counts, which must be
      skips only -- any passed/failed/error/xfailed/xpassed disqualifies
    """
    lines = output.splitlines()
    # unittest: 'Ran N tests in ...' is always immediately followed by the
    # summary line ('OK (...)' or 'FAILED (...)'), anchored at column 0 so
    # quoted test output cannot fake it.
    for idx, line in enumerate(lines):
        ran = _parse_unittest_counts(line)
        if ran is None or ran == 0:
            continue
        # The summary line follows 'Ran N tests' (possibly after blank
        # lines) and is anchored at column 0 so quoted test output cannot
        # fake it.
        for tail in lines[idx + 1 :]:
            if not tail.strip():
                continue
            m = re.match(r"^OK \(([^)]*)\)", tail)
            if m:
                # All-skipped only when skipped=N is the SOLE item AND N
                # equals the 'Ran N tests' count: any other item
                # (expected failures, unexpected successes) proves a test
                # body ran, and skipped < ran means the remaining tests
                # passed for real (e.g. 'Ran 52 / OK (skipped=3)'
                # contributed 49 true passes -- not zero coverage).
                items = [item.strip() for item in m.group(1).split(",")]
                if len(items) != 1:
                    return False
                sk = re.fullmatch(r"skipped=(\d+)", items[0])
                return sk is not None and int(sk.group(1)) == ran
            break
        return False
    # pytest: the last summary line carrying outcome counts decides; it is
    # all-skipped only when that line reports skips and nothing else.
    for line in reversed(lines):
        counts = _parse_pytest_counts(line)
        if counts:
            return (
                counts.get("skipped", 0) > 0
                and sum(v for k, v in counts.items() if k != "skipped") == 0
            )
    return False


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def _repo_relative_path(p: str) -> str:
    """Return path stripped to repo-relative form (e.g. 'test/srt/foo.py').

    Used in the machine-readable TIMINGS block so downstream scrapers
    get a stable key regardless of CI runner checkout layout.
    """
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    marker = "/sglang/"
    idx = p.rfind(marker)
    return p[idx + len(marker) :] if idx >= 0 else p


def run_unittest_files(
    files: Union[List[TestFile], List[CIRegistry]],
    timeout_per_file: float,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_attempts: int = 2,
    retry_wait_seconds: int = 60,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        enable_retry: If True, retry failed tests that appear to be accuracy/performance
                     assertion failures (not code errors).
        max_attempts: Maximum number of attempts per file including initial run (default: 2).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
    """
    coredump_enabled = cuda_coredump.is_enabled()
    if coredump_enabled:
        cuda_coredump.cleanup_dump_dir()

    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    retried_tests = []  # Track which tests were retried
    # Third state: the file exited successfully but every collected test
    # skipped, so it contributed zero coverage. Kept out of both
    # passed_tests and failed_tests and reported on its own at the end.
    all_skipped_tests = []
    # Per-file elapsed seconds, latest attempt wins. Consumed by the
    # TIMINGS block emitted at the end of this function.
    file_elapsed: Dict[str, float] = {}

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        process = None
        output_lines = []

        def run_one_file(filename, capture_output=True):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n"
            )
            file_tic = time.perf_counter()

            cmd = ["python3", full_path, "-f"]

            if capture_output:
                # Capture output for the retry decision and for
                # all-skipped detection. The capture branch tees every line
                # to the logger as it arrives, so CI-log visibility is the
                # same as the inherited passthrough mode.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="ignore",  # Ignore non-UTF-8 bytes to prevent UnicodeDecodeError
                )
                output_lines = []
                for line in process.stdout:
                    logger.info(line.rstrip())
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(cmd, stdout=None, stderr=None)
                process.wait()

            elapsed = time.perf_counter() - file_tic
            file_elapsed[filename] = elapsed

            logger.info(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n"
            )
            return process.returncode

        # Retry loop for each file
        attempt = 1
        file_passed = False
        was_retried = False

        while attempt <= (max_attempts if enable_retry else 1):
            if attempt > 1:
                logger.info(
                    f"\n[CI Retry] Attempt {attempt}/{max_attempts} for {filename}\n"
                )
                was_retried = True

            try:
                # Output is always captured (tee'd to the logger live) so
                # the all-skipped state can be detected even without retry.
                ret_code = run_with_timeout(
                    run_one_file,
                    args=(filename,),
                    kwargs={"capture_output": True},
                    timeout=timeout_per_file,
                )

                if ret_code == 0 or ret_code == 5:
                    file_passed = True
                    if ret_code == 5:
                        logger.info(
                            f"\n⊘ SKIPPED (no tests collected): {filename}\n"
                        )
                    elif _is_all_tests_skipped("".join(output_lines)):
                        # Exited 0 but contributed zero coverage. Not a
                        # failure (exit-code semantics unchanged), not a
                        # pass either: third state, reported at the end.
                        logger.info(
                            f"\n⊘ ALL SKIPPED (zero coverage): {filename}\n"
                        )
                        all_skipped_tests.append(filename)
                        break
                    if was_retried:
                        logger.info(
                            f"\n✓ PASSED on retry (attempt {attempt}): {filename}\n"
                        )
                        retried_tests.append((filename, attempt, "passed"))
                    passed_tests.append(filename)
                    break
                else:
                    # Check if we should retry
                    if enable_retry and attempt < max_attempts:
                        output = "".join(output_lines)
                        is_retriable, reason = is_retriable_failure(output)

                        if is_retriable:
                            logger.info(f"\n[CI Retry] {filename} failed with {reason}")
                            logger.info(
                                f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n"
                            )
                            time.sleep(retry_wait_seconds)
                            attempt += 1
                            continue
                        else:
                            logger.info(
                                f"\n[CI Retry] {filename} failed with {reason} - not retrying\n"
                            )

                    # No retry or not retriable
                    logger.info(
                        f"\n✗ FAILED: {filename} returned exit code {ret_code}\n"
                    )
                    if was_retried:
                        retried_tests.append((filename, attempt, "failed"))
                    failed_tests.append((filename, f"exit code {ret_code}"))
                    break

            except TimeoutError:
                kill_process_tree(process.pid)
                time.sleep(5)
                # TimeoutError aborts run_one_file before its elapsed write;
                # record the timeout cap as an upper bound so the file still
                # appears in the TIMINGS block below.
                file_elapsed[filename] = float(timeout_per_file)
                logger.info(
                    f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n"
                )
                if was_retried:
                    retried_tests.append((filename, attempt, "timeout"))
                failed_tests.append((filename, f"timeout after {timeout_per_file}s"))
                break

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if coredump_enabled and not success:
        cuda_coredump.report()

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Test Summary: {len(passed_tests)}/{len(files)} passed"
        + (f", {len(all_skipped_tests)} all-skipped" if all_skipped_tests else "")
    )
    if enable_retry and retried_tests:
        logger.info(f"Retries: {len(retried_tests)} test(s) were retried")
    logger.info(f"{'='*60}")
    if passed_tests:
        logger.info("✓ PASSED:")
        for test in passed_tests:
            logger.info(f"  {test}")
    if failed_tests:
        logger.info("\n✗ FAILED:")
        for test, reason in failed_tests:
            logger.info(f"  {test} ({reason})")
    if retried_tests:
        logger.info("\n↻ RETRIED:")
        for test, attempts, result in retried_tests:
            logger.info(f"  {test} ({attempts} attempts, {result})")
    if all_skipped_tests:
        logger.info(
            f"\n⊘ ALL SKIPPED ({len(all_skipped_tests)} file(s) ran but every "
            f"collected test skipped -- zero coverage, not counted as passed):"
        )
        for test in all_skipped_tests:
            logger.info(f"  {test}")
    logger.info(f"{'='*60}\n")

    # Machine-readable timings block for downstream scrapers/dashboards.
    # One JSON object per executed file (post-retry: only the latest
    # attempt's elapsed is recorded). Files skipped via fail-fast
    # (continue_on_error=False) are omitted. Job wall-clock is read
    # separately from the GitHub Actions API by consumers, so we don't
    # emit any aggregate fields here.
    passed_set = set(passed_tests)
    all_skipped_set = set(all_skipped_tests)
    logger.info("========== TIMINGS BEGIN ==========")
    for fname, elapsed in file_elapsed.items():
        if fname in passed_set:
            status = "passed"
        elif fname in all_skipped_set:
            status = "all_skipped"
        else:
            status = "failed"
        logger.info(
            json.dumps(
                {
                    "file": _repo_relative_path(fname),
                    "passed": fname in passed_set,
                    # Third state for scrapers; "passed" stays false for
                    # all-skipped files while the run itself is not failed.
                    "status": status,
                    "elapsed": round(elapsed),
                }
            )
        )
    logger.info("========== TIMINGS END ==========")

    # Write GitHub Step Summary when retries occurred or zero-coverage
    # files were detected
    if all_skipped_tests:
        write_github_step_summary(
            f"**⊘ {len(all_skipped_tests)} file(s) all-skipped (zero coverage):**\n"
            + "".join(f"- {t}\n" for t in all_skipped_tests)
        )
    if retried_tests:
        passed_on_retry = [t for t, _, r in retried_tests if r == "passed"]
        failed_after_retry = [t for t, _, r in retried_tests if r != "passed"]
        summary = f"**↻ Retried {len(retried_tests)} test(s):**\n"
        if passed_on_retry:
            summary += f"- ✓ Passed on retry: {', '.join(passed_on_retry)}\n"
        if failed_after_retry:
            summary += f"- ✗ Still failed: {', '.join(failed_after_retry)}\n"
        write_github_step_summary(summary)

    return 0 if success else -1

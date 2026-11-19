# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression test for the DSA RAGGED top-k row-caliber fix.

What it guards
--------------
With an FP8 KV cache and the ``flashmla_sparse`` DSA prefill backend, the unfused
(fused top-k disabled) context-parallel prefill path used to fail on the very first
forward pass inside ``sglang.srt.layers.attention.dsa_backend.forward_extend``:
``topk_indices`` carries one row per padded/aligned query row while the RAGGED offset
is built per logical EXTEND token, so the masked addition raised a broadcast error
(or, after the fix, the row-caliber guards). This test starts a server with that
exact configuration, runs a short load and asserts that neither a traceback nor one
of those guards ever fires in the server log.

How to run
----------
Manual test (needs a checkpoint and four accelerators, so it is not CI-discovered).
From the repository root, on a machine with at least 4 accelerator devices of the
required capability and the model checkpoint available locally::

    SGLANG_TEST_MODEL_PATH=/path/to/glm-moe-mxfp4-checkpoint \\
    python3 test/manual/attention/test_dsa_fp8_kv_ragged_topk_rowcaliber_e2e.py -v

Environment variables (typed in ``sglang.srt.environ``; all read through ``.get()``)
-----------------------------------------------------------------------------------
``SGLANG_TEST_MODEL_PATH`` (``EnvStr``)
    Path to a GLM MoE MXFP4 checkpoint with FP8 per-channel dense layers (the model
    family that uses the DSA indexer). If it is unset or missing the test is skipped.
``SGLANG_TEST_PORT`` (``EnvInt``, default 31010)
    Port the spawned server listens on.
``SGLANG_TEST_SERVER_TIMEOUT`` (``EnvInt``, default 1800)
    Start-up budget in seconds.
``SGLANG_TEST_E2E_PROMPTS`` (``EnvInt``, default 4)
    Number of load requests sent by ``test_02``.
``SGLANG_TEST_E2E_INPUT_LEN`` (``EnvInt``, default 4096)
    Prompt length in tokens for those requests.
``SGLANG_TEST_DISABLE_MEGAMOE`` (``EnvBool``, default ``False``)
    Set to ``1``/``true`` to drop the Megamoe expert-path flags. The Megamoe per-rank
    token alignment is what pads query rows past the logical EXTEND count, so the
    default (Megamoe enabled) is the configuration that exercises the guard.

``SGLANG_DSA_FUSE_TOPK`` is forced to ``0`` by the test itself; export it to make the
intent explicit in the launch environment.

Proxy configuration of the host is deliberately irrelevant: this test only ever talks to the
server it spawned, on the loopback address, and it does so through an explicit no-proxy opener
while removing the proxy variables from the child process, so a host that sets ``http_proxy``
without a loopback exemption gets the same result as a host that sets nothing.

The test skips itself (with an explicit reason) when the checkpoint is unavailable,
fewer than four accelerator devices are visible, their compute capability does not
match, or the framework refuses to run the Megamoe expert path that the default
configuration asks for (that combination cannot start a server at all). Set
``SGLANG_TEST_DISABLE_MEGAMOE=1`` on such a host to exercise the same guard without
Megamoe. So it is safe on CPU-only or single-accelerator hosts.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from sglang.srt.environ import envs
from sglang.test.test_utils import (
    CustomTestCase,
    collect_process_tree_pids,
    terminate_and_kill_process_tree,
    wait_for_gpu_release,
)

REQUIRED_COMPUTE_CAPABILITY = (8, 9)
# How long the whole process group is given to react to SIGTERM before it is
# SIGKILLed, and how long to wait for the group (and the device memory it holds)
# to actually disappear afterwards.
PGRP_TERM_GRACE_SECS = 60
PGRP_KILL_GRACE_SECS = 30
PGRP_POLL_SECS = 2
READY_MARKER = b"The server is fired up and ready to roll"
# Every HTTP request this test makes itself goes to the loopback address of the server it
# spawned, so it must never depend on how the host is configured for outbound traffic:
# ``urllib.request.urlopen`` uses the default opener, which honours ``http_proxy`` from the
# environment and would happily send a 127.0.0.1 request through that proxy unless the host
# also happens to exempt loopback in ``NO_PROXY``. On such a host the server is perfectly
# healthy while every request of the test fails, which is not what this test measures. An
# explicit no-proxy opener is built once here and used for all of them; the spawned server
# gets its own proxy variables removed in ``setUpClass`` for the same reason.
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FORBIDDEN_MARKERS = (
    b"Traceback (most recent call last)",
    b"AssertionError",
    b"torch.AcceleratorError",
)


def _visible_device_count() -> int:
    try:
        import torch
    except Exception:  # torch is a hard dependency of the server, not of this helper
        return 0
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _capabilities() -> list:
    try:
        import torch
    except Exception:
        return []
    out = []
    for idx in range(_visible_device_count()):
        try:
            major, minor = torch.cuda.get_device_capability(idx)
            out.append((int(major), int(minor)))
        except Exception:
            continue
    return out


def _capability_ok() -> bool:
    caps = _capabilities()
    if not caps:
        return False
    return all(c >= REQUIRED_COMPUTE_CAPABILITY for c in caps)


def _megamoe_supported(model_path: str) -> tuple:
    """Ask the framework's own Megamoe compatibility check about this device.

    ``--moe-a2a-backend megamoe`` is not accepted everywhere (the production hook
    rejects, for example, CUDA GPUs that are neither SM90 nor SM100 class), and a
    rejected launch surfaces as a failed server start rather than as a skip. The
    probe delegates to that hook instead of repeating its platform matrix here, so
    this test skips on exactly the hosts where the server would refuse to start.
    """
    try:
        from sglang.srt.arg_groups.mega_moe_hook import check_mega_moe_compat
        from sglang.srt.server_args import ServerArgs
    except Exception as exc:
        # A tree without the hook: keep the previous behaviour and let the server speak.
        return True, "Megamoe compatibility hook unavailable: %r" % (exc,)
    try:
        check_mega_moe_compat(
            ServerArgs(model_path=model_path, moe_a2a_backend="megamoe")
        )
    except ValueError as exc:
        return False, str(exc)
    return True, ""


class DsaFp8RaggedTopkRowCaliberTest(CustomTestCase):
    """Start the vulnerable configuration and prove the server stays healthy."""

    proc = None
    log_path = None
    log_dir = None
    log_file = None
    port = 0
    server_timeout = 0

    @classmethod
    def setUpClass(cls):
        model_path = envs.SGLANG_TEST_MODEL_PATH.get()
        if not model_path or not os.path.isdir(model_path):
            raise unittest.SkipTest(
                "SGLANG_TEST_MODEL_PATH must point to a local GLM MoE MXFP4 checkpoint"
            )
        if _visible_device_count() < 4:
            raise unittest.SkipTest(
                "this end-to-end test needs at least 4 visible accelerator devices"
            )
        if not _capability_ok():
            raise unittest.SkipTest(
                "this end-to-end test needs accelerator devices with compute "
                "capability >= %s" % (REQUIRED_COMPUTE_CAPABILITY,)
            )
        # The default configuration drives the token alignment through the Megamoe
        # expert path. Where the framework will not accept Megamoe at all, starting
        # the server can only fail, so skip explicitly instead of reporting a failure
        # - and never "fix" it by dropping Megamoe silently, that would no longer be
        # the configuration this regression guards.
        disable_megamoe = envs.SGLANG_TEST_DISABLE_MEGAMOE.get()
        if not disable_megamoe:
            supported, refusal = _megamoe_supported(model_path)
            if not supported:
                raise unittest.SkipTest(
                    "the default configuration of this test launches "
                    "--moe-a2a-backend megamoe, which this host rejects (%s); set "
                    "SGLANG_TEST_DISABLE_MEGAMOE=1 to exercise the same RAGGED "
                    "top-k row-caliber guard on the non-Megamoe configuration" % refusal
                )

        cls.port = envs.SGLANG_TEST_PORT.get()
        cls.server_timeout = envs.SGLANG_TEST_SERVER_TIMEOUT.get()
        cls.log_dir = tempfile.mkdtemp(prefix="dsa_fp8_ragged_topk_")
        cls.log_path = os.path.join(cls.log_dir, "server.log")

        # The defect only reproduces on the unfused DSA top-k transform.
        env = os.environ.copy()
        env["SGLANG_DSA_FUSE_TOPK"] = "0"
        # Drop the proxy variables for the child: the server talks to itself over loopback
        # (engine stats, health probes) and a host-level proxy would distort that too. This
        # covers the spawned process only - requests made by this test itself go through
        # NO_PROXY_OPENER, so neither layer depends on the host's proxy setup any more.
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.port),
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--attn-cp-size",
            "4",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--chunked-prefill-size",
            "32768",
            "--mem-fraction-static",
            "0.9",
            "--attention-backend",
            "dsa",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "flashmla_kv",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--max-running-requests",
            "16",
            "--num-continuous-decode-steps",
            "1",
            "--trust-remote-code",
            "--weight-loader-disable-mmap",
            "--watchdog-timeout",
            "24000",
            "--dist-timeout",
            "24000",
        ]
        # The reproducing configuration uses the Megamoe FP4 expert path: the token
        # alignment it imposes on each rank is what pads the query rows past the
        # logical EXTEND count. SGLANG_TEST_DISABLE_MEGAMOE drops those flags.
        if not disable_megamoe:
            cmd += [
                "--moe-a2a-backend",
                "megamoe",
                "--enable-w4a4-mxfp4-megamoe",
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
            ]
        cls.log_file = open(cls.log_path, "wb")
        cls.proc = subprocess.Popen(
            cmd,
            stdout=cls.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    @staticmethod
    def _pgrp_pids(pgid: int) -> list:
        """Every live pid currently in process group ``pgid``.

        Used instead of a pid-tree walk because the launcher may already be gone:
        its children are then reparented and invisible to the tree, yet they still
        hold the devices. ``os.getpgid`` is used because process-group membership is
        not exposed uniformly by every psutil version.
        """
        try:
            import psutil
        except Exception:
            return []
        pids = []
        for candidate in psutil.process_iter():
            try:
                if os.getpgid(candidate.pid) == pgid:
                    pids.append(candidate.pid)
            except (OSError, psutil.Error):
                continue
        return pids

    @staticmethod
    def _pgrp_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except Exception:
            # Undeterminable (e.g. not our group): treat it as alive so the sweep
            # is retried rather than silently skipped.
            return True
        return True

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "proc", None)
        if proc is not None:
            # The launcher was spawned with start_new_session=True, so it leads its
            # own process group and its pid is that group id. Clean the group as a
            # unit and unconditionally: a launcher that exited early leaves its
            # scheduler/worker children running on the devices, which a
            # ``proc.poll()`` guard would miss.
            pgid = proc.pid
            before = set(cls._pgrp_pids(pgid))
            before.update(collect_process_tree_pids(pgid))
            if proc.poll() is None:
                try:
                    # SIGTERM so the server unwinds its device contexts in userspace,
                    # then SIGKILL whatever is left, then wait for that memory to come
                    # back. Shared with every launched server in this repository.
                    terminate_and_kill_process_tree(proc)
                except Exception as exc:
                    print("TEARDOWN_HELPER_ERROR=%r" % (exc,))
            for sig, grace in (
                (signal.SIGTERM, PGRP_TERM_GRACE_SECS),
                (signal.SIGKILL, PGRP_KILL_GRACE_SECS),
            ):
                if not cls._pgrp_alive(pgid):
                    break
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                except Exception as exc:
                    print("TEARDOWN_PGRP_SIGNAL_ERROR=sig=%s %r" % (sig, exc))
                    break
                deadline = time.monotonic() + grace
                while cls._pgrp_alive(pgid) and time.monotonic() < deadline:
                    time.sleep(PGRP_POLL_SECS)
            # Orphans killed only through the group are not part of the helper's
            # wait, so confirm the device memory of every pid seen above.
            wait_for_gpu_release(sorted(before))
            still = sorted(cls._pgrp_pids(pgid))
            print(
                "TEARDOWN_PGRP=%s pids_seen=%d still_alive=%s"
                % (pgid, len(before), still)
            )
            sys.stdout.flush()
        log_file = getattr(cls, "log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        log_dir = getattr(cls, "log_dir", None)
        if log_dir:
            print("SERVER_LOG=%s" % os.path.join(log_dir, "server.log"))
            sys.stdout.flush()

    def _read_log(self) -> bytes:
        try:
            with open(self.log_path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return b""

    def _wait_until_ready(self):
        deadline = time.time() + self.server_timeout
        while time.time() < deadline:
            blob = self._read_log()
            for marker in FORBIDDEN_MARKERS:
                if marker in blob:
                    tail = blob.decode("utf-8", "replace")[-4000:]
                    self.fail(
                        "server log contains %r before readiness;\n%s" % (marker, tail)
                    )
            if READY_MARKER in blob:
                return
            if self.proc.poll() is not None:
                tail = self._read_log().decode("utf-8", "replace")[-4000:]
                self.fail(
                    "server exited with code %s before readiness:\n%s"
                    % (self.proc.returncode, tail)
                )
            time.sleep(5)
        self.fail("server not ready within %ss" % self.server_timeout)

    def _generate(self, input_ids, max_new_tokens=1):
        body = json.dumps(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/generate" % self.port,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with NO_PROXY_OPENER.open(req, timeout=self.server_timeout) as handle:
            return json.loads(handle.read().decode())

    @staticmethod
    def _response_verdict(payload: dict) -> tuple:
        """Return (finish_reason, completion_tokens) of one /generate response.

        HTTP 200 alone is not a pass: the scheduler can abort a request -- for
        instance when a rank dies while the batch is in flight -- and the endpoint
        still answers 200 with the abort recorded in ``meta_info``.
        """
        meta = payload.get("meta_info") or {}
        reason = meta.get("finish_reason")
        reason_type = reason.get("type") if isinstance(reason, dict) else reason
        try:
            completion_tokens = int(meta.get("completion_tokens", 0))
        except (TypeError, ValueError):
            completion_tokens = 0
        return reason, reason_type, completion_tokens

    def test_01_server_reaches_ready_state(self):
        """The regression crashed at start-up warmup, so readiness alone is a gate."""
        self._wait_until_ready()

    def test_02_short_load_reports_no_error(self):
        self._wait_until_ready()
        num_prompts = envs.SGLANG_TEST_E2E_PROMPTS.get()
        input_len = envs.SGLANG_TEST_E2E_INPUT_LEN.get()
        failures = []
        for idx in range(num_prompts):
            # Deterministic pseudo-random prompt; the trigger depends on the padded
            # row layout, not on the prompt semantics.
            rnd = (idx * 7919 + 1013) % 90000
            input_ids = [rnd + ((j * 31) % 50000) for j in range(input_len)]
            try:
                payload = self._generate(input_ids)
            except (urllib.error.URLError, TimeoutError) as exc:
                failures.append((idx, repr(exc)))
                continue
            reason, reason_type, completion_tokens = self._response_verdict(payload)
            # An aborted request, or one that produced no token even though
            # max_new_tokens=1 was asked for, is a failure even on HTTP 200.
            if reason_type == "abort" or completion_tokens < 1:
                failures.append(
                    (
                        idx,
                        "finish_reason=%r completion_tokens=%r response_keys=%r"
                        % (reason, completion_tokens, sorted(payload)),
                    )
                )
        self.assertEqual([], failures, "load requests failed: %s" % failures)

    def test_03_server_log_is_clean(self):
        self._wait_until_ready()
        blob = self._read_log()
        for marker in FORBIDDEN_MARKERS:
            self.assertNotIn(
                marker, blob, "forbidden marker %r found in server log" % marker
            )
        self.assertIn(
            "fp8_e4m3".encode(),
            blob,
            "server did not report the FP8 KV cache dtype; check the launch arguments",
        )
        self.assertIn(b"flashmla_sparse", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)

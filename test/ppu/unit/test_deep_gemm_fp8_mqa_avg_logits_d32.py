"""Standalone regression for PPU DeepGEMM FP8 avg-MQA illegal access at D32.

Dependencies: Python stdlib, torch, and the PPU deep_gemm package. No SGLang
imports or helpers, pytest dependency, model weights, or server are needed.
Inputs are generated and quantized on CPU; device preparation only copies them.
Each shape runs in a fresh process, so a failed device context cannot affect the
next case. A native crash is a FAILURE, not xfail or an expected success.

Run inside a verified PPU-890P task container through run-validation:
    python3 test/ppu/unit/test_deep_gemm_fp8_mqa_avg_logits_d32.py \
        --output-dir "$PPU_DEV_RUN_DIR/deep-gemm-d32" -v

Minimal single-case entry (same container requirement):
    python3 test/ppu/unit/test_deep_gemm_fp8_mqa_avg_logits_d32.py \
        --case 32 128 32 --output-dir "$PPU_DEV_RUN_DIR/deep-gemm-d32-single"

Default suite: three D32 repro shapes, plus D64/D128 controls. This is a
version-pinned reproducer for the legacy clean_logits/global-column ABI, not
a smoke test for the current compressed-output API. On the tested older PPU
deep-gemm 0.0.0 build, all three D32 cases raised illegal memory access, while
the three D64/D128 controls completed without importing SGLang. Those failures
remain failures. Current API documentation rejects D32 before kernel dispatch;
do not interpret an unsupported-keyword error on a newer build as the old
device fault. Use test_qsa_ppu_compressed_logits.py for current API coverage.
Raw logs, phase markers, inputs' metadata, and results are retained.
"""

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _case_name(rows, keys, dim):
    return f"q{rows}-k{keys}-h4-d{dim}"


def _worker(rows, keys, dim, output_dir):
    """The only process that imports torch/deep_gemm or opens a device."""
    record = {"shape": [rows, keys, dim], "heads": 4, "seed": 42}
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f"{_case_name(rows, keys, dim)}.json"

    def phase(name):
        record["phase"] = name
        print(json.dumps({"phase": name, "shape": record["shape"]}), flush=True)

    try:
        phase("prerequisites")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires a PPU-890P CUDA-compatible runtime")
        device = torch.cuda.get_device_properties(0)
        capability = list(torch.cuda.get_device_capability(0))
        record.update(
            python=sys.version,
            torch=torch.__version__,
            device_name=device.name,
            capability=capability,
        )
        if "890" not in device.name.upper() or capability != [8, 9]:
            raise unittest.SkipTest(f"unvalidated device: {device.name}, {capability}")

        import deep_gemm

        record["deep_gemm_module"] = deep_gemm.__file__
        for distribution in ("deep-gemm", "sgl-deep-gemm"):
            try:
                record[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        # Do not silently use a framework helper via a transitive import.
        if any(m == "sglang" or m.startswith("sglang.") for m in sys.modules):
            raise RuntimeError("SGLang was imported; direct-call isolation is invalid")
        record["sglang_loaded"] = False
        native = deep_gemm.fp8_mqa_avg_logits

        phase("cpu_input_preparation")
        generator = torch.Generator(device="cpu").manual_seed(42)
        q = torch.randn(rows, 4, dim, generator=generator, dtype=torch.bfloat16)
        k = torch.randn(keys, dim, generator=generator, dtype=torch.bfloat16)
        q[0].zero_()

        def quantize_cpu(x):
            xf = x.float()
            scale = xf.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
            fp8 = (xf / scale[:, None]).to(torch.float8_e4m3fn)
            return fp8.contiguous(), scale.contiguous()

        q8, qs = quantize_cpu(q.reshape(rows, -1))
        q8 = q8.reshape(rows, 4, dim)
        k8, ks = quantize_cpu(k)
        starts = torch.zeros(rows, dtype=torch.int32)
        starts[rows // 2 :] = keys // 2
        ends = starts + torch.arange(rows, dtype=torch.int32) % max(1, keys // 2)
        assert bool(((0 <= starts) & (starts <= ends) & (ends <= keys)).all())

        # FP64 reference from the actual quantized values; BF16->FP8 rounding is
        # not an error in this kernel-safety regression.
        q_ref = q8.float().double() * qs.double()[:, None, None]
        k_ref = k8.float().double() * ks.double()[:, None]
        expected = torch.relu(torch.einsum("mhd,nd->mnh", q_ref, k_ref)).sum(-1)
        expected /= math.sqrt(dim)
        columns = torch.arange(keys)[None, :]
        valid = (columns >= starts[:, None]) & (columns < ends[:, None])
        expected.masked_fill_(~valid, -float("inf"))

        phase("device_input_copy")
        q_device, k_device = q8.cuda(), k8.cuda()
        qs_device, ks_device = qs.cuda(), ks.cuda()
        starts_device, ends_device = starts.cuda(), ends.cuda()
        record["inputs"] = {
            name: {
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": str(t.dtype),
            }
            for name, t in (
                ("q", q_device),
                ("k", k_device),
                ("q_scale", qs_device),
                ("k_scales", ks_device),
                ("starts", starts_device),
                ("ends", ends_device),
            )
        }
        assert all(
            t.is_contiguous()
            for t in (
                q_device,
                k_device,
                qs_device,
                ks_device,
                starts_device,
                ends_device,
            )
        )
        torch.cuda.synchronize()
        record["pre_call_sync_passed"] = True

        phase("native_call")
        actual = native(
            q_device,
            (k_device, ks_device),
            starts_device,
            ends_device,
            clean_logits=True,
            q_scale=qs_device,
            logits_dtype=torch.float32,
        )
        phase("native_synchronize")
        torch.cuda.synchronize()
        record["post_call_sync_passed"] = True

        phase("result_validation")
        assert actual.dtype == torch.float32 and tuple(actual.shape) == (rows, keys)
        actual = actual.cpu().double()
        assert torch.equal(torch.isneginf(actual), ~valid), "invalid logits mask"
        assert bool(torch.isfinite(actual[valid]).all()), "nonfinite valid logits"
        a, b = actual[valid], expected[valid]
        denominator = (a.square() + b.square()).sum()
        diff = float((a - b).square().sum() / denominator) if denominator else 0.0
        record.update(
            max_abs_vs_quantized_reference=(
                float((a - b).abs().max()) if a.numel() else 0.0
            ),
            normalized_squared_error=diff,
        )
        # Basic numerical sanity only; this is not a BF16 model-quality test.
        assert diff < 1e-3, f"quantized-reference normalized squared error: {diff}"
        record["status"] = "passed"
        print(json.dumps(record), flush=True)
    except unittest.SkipTest as error:
        record.update(status="skipped", error=str(error))
        print(json.dumps(record), flush=True)
        return 77
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(json.dumps(record), flush=True)
        raise
    finally:
        report.write_text(json.dumps(record, indent=2) + "\n")
    return 0


class TestDeepGemmFp8MqaAvgLogitsD32(unittest.TestCase):
    output_root = None
    case_timeout = 180

    @classmethod
    def setUpClass(cls):
        root = cls.output_root or os.environ.get("PPU_DEV_RUN_DIR")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        cls.results = Path(tempfile.mkdtemp(prefix="deep-gemm-d32-", dir=root))
        print(f"DeepGEMM repro artifacts: {cls.results}", flush=True)

    def _run_case(self, rows, keys, dim):
        name = _case_name(rows, keys, dim)
        log_path = self.results / f"{name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            str(rows),
            str(keys),
            str(dim),
            "--output-dir",
            str(self.results),
        ]
        timed_out = False
        with log_path.open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.case_timeout,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                timed_out, code = True, 124
        report_path = self.results / f"{name}.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(exit_code=code, timed_out=timed_out, log=str(log_path))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        output = log_path.read_text(errors="replace")
        if code == 77:
            self.skipTest(report.get("error", output[-2000:]))
        self.assertEqual(code, 0, f"{name}: {report}\n{output[-6000:]}")
        self.assertEqual(report.get("status"), "passed", report)

    def test_d32_shapes(self):
        for rows, keys in ((3, 7), (32, 128), (17, 91)):
            with self.subTest(rows=rows, keys=keys, dim=32):
                self._run_case(rows, keys, 32)

    def test_d64_d128_controls(self):
        for rows, keys, dim in ((32, 128, 64), (32, 128, 128), (3, 7, 128)):
            with self.subTest(rows=rows, keys=keys, dim=dim):
                self._run_case(rows, keys, dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, type=int, metavar=("M", "N", "D"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=180, help="seconds per child")
    args, unittest_args = parser.parse_known_args()
    if args.case:
        if any(x <= 0 for x in args.case):
            parser.error("M, N, and D must be positive")
        destination = args.output_dir or Path(tempfile.mkdtemp(prefix="deep-gemm-d32-"))
        sys.exit(_worker(*args.case, destination))
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    TestDeepGemmFp8MqaAvgLogitsD32.output_root = args.output_dir
    TestDeepGemmFp8MqaAvgLogitsD32.case_timeout = args.timeout
    unittest.main(argv=[sys.argv[0], *unittest_args])

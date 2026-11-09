# SPDX-License-Identifier: Apache-2.0
"""Frozen Qwen38 EP4 kernel regression, excluding host/communication overhead.

Optional baseline JSON maps "gate_up/M" or "down/M" to an old library, its
sha256, torch operator namespace, and integer variant (benchmark ABI).
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from qwen38_kineto import profile
from sgl_kernel.w4a8_sm import (
    cutlass_w4a8_sm_moe_mm,
    cutlass_w4a8_sm_moe_mm_swiglu,
    prepare_w4a8_sm_weight,
    select_w4a8_sm_config,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    baselines = (
        json.loads(Path(args.baseline_configs).read_text())
        if args.baseline_configs
        else {}
    )
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "global_experts": 512,
        "local_experts": 128,
        "topk": 10,
        "rows": [],
    }
    for gate in (True, False):
        projection = "gate_up" if gate else "down"
        e, n, k = (128, 1280, 2560) if gate else (128, 2560, 640)
        q = torch.randint(-7, 8, (e, n, k), device="cuda", dtype=torch.int8)
        scales = (
            torch.rand((e, n, k // 128), device="cuda") * 0.015 + 0.002
        ).bfloat16()
        real_w = (
            q.float().reshape(e, n, k // 128, 128) * scales.float().unsqueeze(-1)
        ).reshape(e, n, k)
        packed = (
            (q[..., ::2].to(torch.int16) & 15)
            | ((q[..., 1::2].to(torch.int16) & 15) << 4)
        ).to(torch.int8)
        prepared = prepare_w4a8_sm_weight(packed, scales, fused_gate_up=gate)
        del q, packed, scales
        for m in args.m:
            rows = (m * 10 + 3) // 4
            counts = [(rows + e - 1 - i) // e for i in range(e)]
            ct = torch.tensor(counts, device="cuda", dtype=torch.int32)
            offsets = torch.cat(
                (torch.zeros(1, device="cuda", dtype=torch.int32), ct.cumsum(0).int())
            )
            ps = torch.stack((torch.full_like(ct, n), ct, torch.full_like(ct, k)), 1)
            config, alignment = select_w4a8_sm_config(projection, rows, e)
            weight_scale = prepared.scales_k512 if alignment == 4 else prepared.scales
            sk = torch.full((e, 3), k, device="cuda", dtype=torch.int64)
            sn = torch.full((e, 3), n, device="cuda", dtype=torch.int64)
            sd = torch.full(
                (e, 3), 640 if gate else n, device="cuda", dtype=torch.int64
            )
            a = (torch.randn((rows, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
            asc = torch.tensor([0.5], device="cuda")
            osc = torch.tensor([0.03125], device="cuda")
            output = torch.empty(
                (rows, 640 if gate else n),
                device="cuda",
                dtype=torch.float8_e4m3fn if gate else torch.bfloat16,
            )
            common = [output, a, prepared.weight, asc, weight_scale]
            if gate:
                common += [osc]
            common += [offsets[:-1], ps, sk, sk, sd, sn]
            kernel = cutlass_w4a8_sm_moe_mm_swiglu if gate else cutlass_w4a8_sm_moe_mm

            def run():
                kernel(*common, config_id=config)
                return output

            run()
            torch.cuda.synchronize()
            ref = []
            start = 0
            for expert, count in enumerate(counts):
                x = (a[start : start + count].float() * asc) @ real_w[expert].T
                if gate:
                    x = (
                        ((torch.nn.functional.silu(x[:, :640]) * x[:, 640:]) / osc)
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                    )
                else:
                    x = x.bfloat16()
                ref.append(x)
                start += count
            ref = torch.cat(ref).float()
            start = 0
            errors = []
            cosines = []
            for count in counts:
                if count:
                    x = output[start : start + count].float().flatten()
                    y = ref[start : start + count].flatten()
                    assert torch.isfinite(x).all()
                    errors.append(
                        float((x - y).abs().sum() / y.abs().sum().clamp_min(1e-8))
                    )
                    cosines.append(
                        float(torch.nn.functional.cosine_similarity(x, y, dim=0))
                    )
                start += count
            assert max(errors) < 0.035 and min(cosines) > 0.995
            eager = output.clone()
            row = {
                "projection": projection,
                "global_m": m,
                "local_rows": rows,
                "config_id": config,
                "worst_relative_l1": max(errors),
                "worst_cosine": min(cosines),
            }
            row["new"] = profile(
                run,
                f"{projection}_{m}_new",
                out / "traces",
                rounds=args.rounds,
                iters=args.iters,
            )
            assert torch.equal(eager.view(torch.uint8), output.view(torch.uint8))
            if f"{projection}/{m}" in baselines:
                base = baselines[f"{projection}/{m}"]
                assert sha(base["library"]) == base["sha256"]
                torch.ops.load_library(base["library"])
                op = getattr(torch.ops, base["namespace"]).run

                def old():
                    op(*common, 128, base["variant"])
                    return output

                old()
                torch.cuda.synchronize()
                assert torch.equal(eager.view(torch.uint8), output.view(torch.uint8)), (
                    "integration differs from frozen kernel"
                )
                row["baseline"] = profile(
                    old,
                    f"{projection}_{m}_baseline",
                    out / "traces",
                    rounds=args.rounds,
                    iters=args.iters,
                )
                row["baseline_library_sha256"] = base["sha256"]
            result["rows"].append(row)
            (out / "results.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "projection": projection,
                        "m": m,
                        "new_us": row["new"]["gpu_us"]["median"],
                        "baseline_us": row.get("baseline", {})
                        .get("gpu_us", {})
                        .get("median"),
                    }
                ),
                flush=True,
            )
            del ref, eager
        del prepared, real_w
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument(
        "--m", type=int, nargs="+", default=[30, 32, 36, 4096, 5120, 6144, 7168, 8192]
    )
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--baseline-configs")
    main(p.parse_args())

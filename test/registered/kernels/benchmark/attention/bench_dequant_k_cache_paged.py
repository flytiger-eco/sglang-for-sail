"""Benchmark the DSv4 paged K-cache dequantize kernel across shapes.

Compares the original one-thread-per-token scalar kernel against the vectorized
kernel at each of its threads-per-token instantiations, over a sweep of token
counts and page sizes. Reports median latency, speedup and the effective
bandwidth implied by 583 cache bytes + 4 table bytes read and 1024 bytes written
per token (excluding the unused scale padding byte).

The token counts include 20096 and 40960, the per-call sizes the DeepSeek-V4.1
100K-input sparse prefill actually issues; page sizes 48 and 96 are not powers
of two and exercise the kernel's integer-division path rather than its shift
path.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE, get_benchmark_range
from sglang.kernels.ops.attention.dsa.triton_kernel import _supports_fp8
from sglang.kernels.ops.attention.dsv4 import (
    dequantize_k_cache_paged_cuda,
    dequantize_k_cache_paged_v2_cuda,
)
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    _vector_access_ok,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

OUT_DIM = DIM_NOPE + DIM_ROPE
SM_COUNT = torch.cuda.get_device_properties(DEFAULT_DEVICE).multi_processor_count

TOKEN_RANGE = get_benchmark_range(
    full_range=[
        1,
        16,
        64,
        256,
        1024,
        4096,
        8192,
        16384,
        20096,
        32768,
        40960,
        65536,
        131072,
    ],
    ci_range=[1, 64, 1024, 8192, 20096, 65536],
)
PAGE_SIZE_RANGE = get_benchmark_range(
    full_range=[64, 128, 48, 96],
    ci_range=[64, 128],
)

LINE_VALS = ["scalar", "vec8", "vec16", "vec32"]
# Logical traffic: data + seven scale bytes + int32 token ID + BF16 output.
BYTES_PER_TOKEN = NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN - 1 + 4 + OUT_DIM * 2


def bytes_per_page_for(page_size: int) -> int:
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    return -(-raw // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES


def benchmark(num_tokens: int, provider: str, page_size: int) -> marker.BenchResult:
    torch.manual_seed(0)
    num_pages = -(-num_tokens // page_size) + 8
    bpp = bytes_per_page_for(page_size)
    cache = torch.randint(
        0, 256, (num_pages, bpp), dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    table = torch.randint(
        0,
        num_pages * page_size,
        (num_tokens,),
        dtype=torch.int32,
        device=DEFAULT_DEVICE,
    )
    out = torch.empty(
        (num_tokens, 1, OUT_DIM), dtype=torch.bfloat16, device=DEFAULT_DEVICE
    )
    # Same predicate the library uses, so both providers take the conversion
    # path this device would actually run.
    native = _supports_fp8()

    if provider == "scalar":

        def fn():
            dequantize_k_cache_paged_cuda(
                out=out,
                k_cache=cache,
                page_table=table,
                page_size=page_size,
                use_fp8_native=native,
            )

    elif provider in ("vec8", "vec16", "vec32"):
        tpt = int(provider[3:])
        assert _vector_access_ok(cache, out), "benchmark must not hit the fallback"

        def fn():
            dequantize_k_cache_paged_v2_cuda(
                out=out,
                k_cache=cache,
                page_table=table,
                page_size=page_size,
                use_fp8_native=native,
                threads_per_token=tpt,
            )

    else:
        raise ValueError(f"Unknown benchmark provider: {provider}")

    fn()
    torch.cuda.synchronize()
    # The marker's non-graph path flushes L2 between reps, as the original did.
    return marker.do_bench(
        fn,
        use_cuda_graph=False,
        metrics=(0.5, 0.2, 0.8),
        estimated_time_ms=200,
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=num_tokens * BYTES_PER_TOKEN,
    )


def effective_gbps(num_tokens: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return num_tokens * BYTES_PER_TOKEN / (us * 1e-6) / 1e9


def run_console_benchmark() -> None:
    print(
        f"device={torch.cuda.get_device_name(DEFAULT_DEVICE)} CUs/SMs={SM_COUNT}",
        flush=True,
    )
    slower = []
    for page_size in PAGE_SIZE_RANGE:
        print(
            f"\n=== page_size={page_size} "
            f"(bytes_per_page={bytes_per_page_for(page_size)}, "
            f"{'shift' if (page_size & (page_size - 1)) == 0 else 'divide'} path) ===",
            flush=True,
        )
        header = f"{'num_tokens':>11} {'scalar':>10}"
        header += "".join(f"{'tpt=' + p[3:]:>10}" for p in LINE_VALS[1:])
        header += f"{'best':>9} {'speedup':>9} {'GB/s':>8}"
        print(header, flush=True)

        for num_tokens in TOKEN_RANGE:
            medians = {
                provider: benchmark(
                    num_tokens=num_tokens, provider=provider, page_size=page_size
                ).times[0]
                * 1e6
                for provider in LINE_VALS
            }
            base = medians["scalar"]
            vec = {p: medians[p] for p in LINE_VALS[1:]}
            best_provider = min(vec, key=vec.get)
            best = vec[best_provider]
            speedup = base / best if best > 0 else float("inf")
            if speedup < 1.0:
                slower.append((page_size, num_tokens, best_provider, speedup))
            print(
                f"{num_tokens:>11} {base:>10.3f}"
                + "".join(f"{medians[p]:>10.3f}" for p in LINE_VALS[1:])
                + f" {best_provider:>8} {speedup:>8.2f}x "
                f"{effective_gbps(num_tokens, best):>7.0f}",
                flush=True,
            )

    print("\n=== generalization verdict ===", flush=True)
    if slower:
        for page_size, num_tokens, provider, speedup in slower:
            print(
                f"  NOT FASTER: page_size={page_size} num_tokens={num_tokens} "
                f"{provider} {speedup:.2f}x",
                flush=True,
            )
    else:
        print("  vector kernel faster than scalar at every shape swept", flush=True)


if __name__ == "__main__":
    run_console_benchmark()

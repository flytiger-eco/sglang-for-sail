"""Finite device regression; run in the authorized PPU container, no server."""

import json
import os
from pathlib import Path

import torch
from deep_gemm import fp8_fp4_mqa_logits
from deep_gemm.jit_kernels.attention import fp8_fp4_mqa_logits as fp8_fp4_mqa_logits_out
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.layers.attention.dsv4.ppu_logits_workspace import LogitsWorkspace


class RejectCudaEmpty(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.empty.memory_format:
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                raise AssertionError("out path attempted a new CUDA empty allocation")
        return func(*args, **kwargs)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    arena = LogitsWorkspace(
        device="cuda:0", capacity_bytes=4096 * 1024 * 1024, preallocate_mask=True
    )
    arena.initialize()
    pointer = arena.storage.data_ptr()
    mask_pointer = arena.mask_storage.data_ptr()
    cases = []
    for rows, width in [(1, 19), (5, 257), (32, 1025), (4, 800000)]:
        q = torch.randint(-128, 128, (rows, 32, 64), dtype=torch.int8, device="cuda")
        kv = torch.randint(-128, 128, (width, 64), dtype=torch.int8, device="cuda")
        q_sf = torch.full((rows, 32), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
        kv_sf = torch.full((width,), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
        weights = torch.rand(rows, 32, dtype=torch.float32, device="cuda")
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        lengths = torch.linspace(1, width, rows, device="cuda").to(torch.int32)
        if rows == 1:
            lengths.fill_(width)
        args = ((q, q_sf), (kv, kv_sf), weights, starts, lengths, False, width)
        reference = fp8_fp4_mqa_logits(*args)
        with arena.lease(rows, width) as out:
            # First call also warms this optional-output frontend/JIT path.
            fp8_fp4_mqa_logits_out(*args, out=out)
            with RejectCudaEmpty():
                actual = fp8_fp4_mqa_logits_out(*args, out=out)
            assert actual.data_ptr() == pointer
            valid = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
            assert torch.equal(actual[valid], reference[valid]), (rows, width)
            cases.append(
                {
                    "rows": rows,
                    "width": width,
                    "stride": list(actual.stride()),
                    "storage_ptr": actual.untyped_storage().data_ptr(),
                    "bitwise_equal_valid": True,
                }
            )
        del (
            actual,
            reference,
            q,
            kv,
            q_sf,
            kv_sf,
            weights,
            starts,
            lengths,
            valid,
            args,
            out,
        )

    # Include the actual 800K tile and the full 1 GiB mask boundary. Compare in
    # row chunks so the reference itself does not need another large buffer.
    mask_cases = []
    for rows, width in [(5, 24), (1248, 800000), (1024, 1048576)]:
        columns = torch.arange(width, device="cuda")[None, :]
        lengths = torch.linspace(0, width, rows, device="cuda").to(torch.int32)[:, None]
        with arena.lease(rows, width):
            mask = arena.causal_mask(rows, width)
            with RejectCudaEmpty():
                torch.ge(columns, lengths, out=mask)
            assert mask.data_ptr() == mask_pointer
            for start in range(0, rows, 32):
                assert torch.equal(
                    mask[start : start + 32], columns >= lengths[start : start + 32]
                )
            mask_cases.append({"rows": rows, "width": width, "equal": True})
        del mask, columns, lengths

    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(first), arena.lease(128, 1024) as out:
        out.fill_(3)
        arena.causal_mask(128, 1024).fill_(True)
    with torch.cuda.stream(second), arena.lease(128, 1024) as out:
        assert torch.all(out == 3).item()
        assert arena.causal_mask(128, 1024).all().item()
        arena.causal_mask(128, 1024).fill_(False)
        out.fill_(4)
    with torch.cuda.stream(first), arena.lease(128, 1024) as out:
        assert torch.all(out == 4).item()
        assert not arena.causal_mask(128, 1024).any().item()
    first.synchronize()
    second.synchronize()
    torch.cuda.synchronize()
    result = {
        "passed": True,
        "capacity_bytes": arena.capacity_bytes,
        "storage_ptr": pointer,
        "cases": cases,
        "mask_capacity_bytes": arena.mask_storage.numel(),
        "mask_cases": mask_cases,
        "reference_entry_module": fp8_fp4_mqa_logits.__module__,
        "workspace_entry_module": fp8_fp4_mqa_logits_out.__module__,
        "cross_stream_handoff": True,
        "no_cuda_empty_in_out_path": True,
    }
    Path(os.environ["PPU_DEV_RUN_DIR"], "workspace-device-result.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Cleanup helpers for residual GPU processes in CI tests."""

import os
from typing import List, Tuple

from sglang.srt.utils import kill_process_tree


def residual_gpu_process_ids(pynvml, gpu_indices: List[int]) -> List[int]:
    """Return other compute-process PIDs on visible GPUs, excluding init and self."""
    self_pid = os.getpid()
    pids = set()
    for index in gpu_indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError:
            continue
        for process in processes or []:
            if process.pid not in (self_pid, 1):
                pids.add(process.pid)
    return sorted(pids)


def cleanup_residual_gpu_processes(
    pynvml, gpu_indices: List[int]
) -> Tuple[List[int], List[str]]:
    """Kill residual GPU process trees after the idle wait expires."""
    killed = []
    errors = []
    for pid in residual_gpu_process_ids(pynvml, gpu_indices):
        try:
            kill_process_tree(pid, wait_timeout=10)
            killed.append(pid)
        except Exception as error:
            errors.append(f"pid={pid}: {error}")
    return killed, errors

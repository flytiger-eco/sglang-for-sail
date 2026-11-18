#!/usr/bin/env python3
"""Validate the assigned PPU board family and record an evidence fingerprint."""

import argparse
import json
import re
from pathlib import Path

EXPECTED_BOARD_NAMES = {
    "zw-810e": "ZW810E",
    "zw-m890p": "ZWM890P",
}


def normalized_device_name(name):
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def check_board(board, devices, output):
    import torch

    expected_name = EXPECTED_BOARD_NAMES[board]
    device_count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    report = {
        "board": board,
        "expected_devices": devices,
        "device_count": device_count,
        "device_names": names,
    }
    error = None
    if report["device_count"] != devices:
        error = f"Expected {devices} assigned devices, found {report['device_count']}"
    elif any(expected_name not in normalized_device_name(name) for name in names):
        error = f"Expected board {board}, found device names {names}"

    if error:
        report["error"] = error
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if error:
        raise RuntimeError(error)
    print(f"Validated {board}: {names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=sorted(EXPECTED_BOARD_NAMES), required=True)
    parser.add_argument("--devices", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_board(args.board, args.devices, args.output)

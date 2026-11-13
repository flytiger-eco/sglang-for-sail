#!/usr/bin/env python3
"""Temporary read-only audit for resources listed in Missing_model_datasets.xlsx."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

MODEL_IDS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "RadixArk/Kimi-K3-DSpark",
    "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "lightseekorg/kimi-k2.5-eagle3-mla",
    "lmsys/sglang-ci-dsv3-channel-int8-test",
    "lmsys/sglang-ci-dsv3-channel-int8-test-NextN",
    "lmsys/sglang-ci-dsv3-test",
    "lmsys/sglang-ci-dsv3-test-NextN",
    "meta-llama/Llama-2-7b-hf",
    "shanearora/2025-sep-a-base-model",
    "swiss-ai/Apertus-8B",
    "thinkingmachines/Inkling",
    "thinkingmachines/Inkling-NVFP4",
    "thinkingmachines/Inkling-Small-NVFP4",
]

LORA_IDS = [
    "opherlie/lora-test-case-NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "opherlie/lora-test-case-Qwen3.5-35B-A3B",
    "opherlie/lora-test-case-Qwen3.5-4B",
    "yushengsu/lora-diff-DeepSeek-V3.1-Base",
    "yushengsu/lora-diff-Kimi-K2.5",
    "yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507",
    "yushengsu/lora-diff-Qwen3-8B",
    "yushengsu/lora-diff-Qwen3-VL-30B-A3B-Instruct",
    "yushengsu/lora-diff-gpt-oss-20b",
]

DATASETS = [
    {
        "id": "THUDM/LongBench-v2",
        "repo_id": "THUDM/LongBench-v2",
        "requirement": "train",
        "runtime": "/nas_aisw/datasets/datasets/LLM/ZhipuAI/v2/LongBench-v2",
    },
    {
        "id": "MMMU/MMMU",
        "repo_id": "MMMU/MMMU",
        "requirement": "math-test",
        "runtime": "/nas_aisw/datasets/datasets/LLM/Comprehensive_Examination/v1/MMMU",
    },
    {
        "id": "HellaSwag validation",
        "repo_id": "Rowan/hellaswag",
        "requirement": "hellaswag",
        "runtime": None,
    },
    {
        "id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "repo_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "requirement": "sharegpt",
        "runtime": "/nas_aisw/datasets/datasets/LLM/anon8231489123/v1.0/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json",
    },
]

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
DATA_SUFFIXES = {".arrow", ".json", ".jsonl", ".parquet"}


def repo_directory(cache, repo_id, repo_type):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise ValueError(f"Unsafe repository ID: {repo_id}")
    prefix = "models--" if repo_type == "model" else "datasets--"
    return cache / (prefix + repo_id.replace("/", "--"))


def nonempty_file(path):
    return path.is_file() and path.stat().st_size > 0


def matches_requirement(path, requirement):
    relative = str(path).lower()
    if requirement == "train":
        return "train" in relative and Path(relative).suffix in DATA_SUFFIXES
    if requirement == "math-test":
        return "math" in relative and "test" in relative and Path(relative).suffix in DATA_SUFFIXES
    if requirement == "hellaswag":
        return path.name == "hellaswag_val.jsonl" or (
            "validation" in relative and Path(relative).suffix in DATA_SUFFIXES
        )
    if requirement == "sharegpt":
        return path.name == "ShareGPT_V3_unfiltered_cleaned_split.json"
    raise ValueError(f"Unknown dataset requirement: {requirement}")


def audit_snapshot(cache, repo_id, repo_type, revision, kind):
    result = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "complete": False,
        "issues": [],
        "files": [],
        "weight_count": 0,
        "size_bytes": 0,
        "weight_target_examples": [],
    }
    issues = result["issues"]
    repo = repo_directory(cache, repo_id, repo_type)
    result["repo_path"] = str(repo)
    if not repo.is_dir():
        issues.append("repository cache directory is missing")
        return result
    ref = repo / "refs" / revision
    if not ref.is_file():
        issues.append(f"refs/{revision} is missing")
        return result
    commit = ref.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        issues.append(f"refs/{revision} does not contain a full commit SHA")
        return result
    snapshot = repo / "snapshots" / commit
    result["snapshot"] = str(snapshot)
    if not snapshot.is_dir():
        issues.append("referenced snapshot directory is missing")
        return result

    sizes = {}
    snapshot_files = []
    weight_files = []
    for directory, dirnames, filenames in os.walk(snapshot):
        for name in dirnames:
            path = Path(directory) / name
            if path.is_symlink() and not path.exists():
                issues.append(f"broken directory symlink: {path.relative_to(snapshot)}")
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(snapshot)
            snapshot_files.append(path)
            result["files"].append(str(relative))
            if path.is_symlink() and not path.exists():
                issues.append(f"broken symlink: {relative}")
                continue
            if not path.is_file():
                issues.append(f"not a regular file: {relative}")
                continue
            stat = path.stat()
            sizes[(stat.st_dev, stat.st_ino)] = stat.st_size
            is_lora_weight = kind == "lora" and name.startswith("adapter_model.")
            is_model_weight = kind == "model" and path.suffix in WEIGHT_SUFFIXES
            if is_lora_weight or is_model_weight:
                weight_files.append(path)
                if stat.st_size == 0:
                    issues.append(f"empty weight file: {relative}")
                if not path.is_symlink():
                    issues.append(f"weight is not a symlink: {relative}")
                elif len(result["weight_target_examples"]) < 3:
                    result["weight_target_examples"].append(str(path.resolve()))
            if name.endswith((".safetensors.index.json", ".bin.index.json")):
                try:
                    shards = set(json.loads(path.read_text())["weight_map"].values())
                    if not shards:
                        issues.append(f"empty weight index: {relative}")
                    for shard in shards:
                        shard_path = Path(shard)
                        if shard_path.is_absolute() or ".." in shard_path.parts:
                            issues.append(f"unsafe shard path in {relative}: {shard}")
                        elif not nonempty_file(path.parent / shard_path):
                            issues.append(f"missing or empty shard {shard} in {relative}")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append(f"invalid weight index {relative}: {error}")

    result["size_bytes"] = sum(sizes.values())
    result["weight_count"] = len(weight_files)
    result["file_count"] = len(snapshot_files)
    if not snapshot_files:
        issues.append("snapshot is empty")
    if kind == "model" and not weight_files:
        issues.append("no model weights found")
    if kind == "lora" and not weight_files:
        issues.append("no adapter_model weights found")

    required_files = []
    if kind == "model":
        required_files.append("config.json")
    elif kind == "lora":
        required_files.extend(["adapter_config.json", "compare_sample_train_data.pt"])
    for name in required_files:
        path = snapshot / name
        if not nonempty_file(path):
            issues.append(f"required file is missing or empty: {name}")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text())
            except (OSError, ValueError) as error:
                issues.append(f"invalid JSON in {name}: {error}")

    result["issues"] = list(dict.fromkeys(issues))
    result["complete"] = not result["issues"]
    return result


def audit_runtime_path(path_value, requirement):
    path = Path(path_value)
    result = {"path": path_value, "complete": False, "issues": [], "matched_files": []}
    if not path.exists():
        result["issues"].append("runtime path is missing")
        return result
    paths = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
    for item in paths:
        if item.is_symlink() and not item.exists():
            result["issues"].append(f"broken runtime symlink: {item}")
        elif matches_requirement(item, requirement) and nonempty_file(item):
            result["matched_files"].append(str(item))
    if not result["matched_files"]:
        result["issues"].append(f"no nonempty runtime file matches {requirement}")
    result["complete"] = not result["issues"]
    return result


def audit_model(cache, repo_id, kind):
    revisions = ["main", "test"] if repo_id == "thinkingmachines/Inkling" else ["main"]
    repo_type = "dataset" if kind == "lora" else "model"
    snapshots = [audit_snapshot(cache, repo_id, repo_type, revision, kind) for revision in revisions]
    return {
        "id": repo_id,
        "type": "LoRA" if kind == "lora" else "model",
        "complete": all(item["complete"] for item in snapshots),
        "snapshots": snapshots,
    }


def audit_dataset(cache, spec):
    snapshot = audit_snapshot(cache, spec["repo_id"], "dataset", "main", "dataset")
    snapshot_path = Path(snapshot["snapshot"]) if "snapshot" in snapshot else None
    matched = []
    if snapshot_path is not None and snapshot_path.is_dir():
        matched = [
            str(path.relative_to(snapshot_path))
            for path in snapshot_path.rglob("*")
            if path.is_file() and matches_requirement(path, spec["requirement"]) and nonempty_file(path)
        ]
    if not matched:
        snapshot["issues"].append(f"no nonempty HF cache file matches {spec['requirement']}")
        snapshot["complete"] = False
    snapshot["matched_files"] = matched
    runtime = audit_runtime_path(spec["runtime"], spec["requirement"]) if spec["runtime"] else None
    return {
        "id": spec["id"],
        "type": "dataset",
        "complete": snapshot["complete"] and (runtime is None or runtime["complete"]),
        "snapshots": [snapshot],
        "runtime": runtime,
    }


def write_report(output, cache, results):
    complete = sum(item["complete"] for item in results)
    report = {
        "cache": str(cache),
        "resource_count": len(results),
        "complete_count": complete,
        "incomplete_count": len(results) - complete,
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "cache_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "type", "status", "issues"])
        writer.writeheader()
        for item in results:
            issues = [issue for snapshot in item["snapshots"] for issue in snapshot["issues"]]
            if item.get("runtime"):
                issues.extend(item["runtime"]["issues"])
            writer.writerow({"id": item["id"], "type": item["type"], "status": "PASS" if item["complete"] else "FAIL", "issues": "; ".join(issues)})
    lines = [
        "# Missing model and dataset cache audit",
        "",
        f"Resources: {len(results)}; complete: {complete}; incomplete: {len(results) - complete}.",
        "",
        "| Status | Type | Resource | Details |",
        "|---|---|---|---|",
    ]
    for item in results:
        issues = [issue for snapshot in item["snapshots"] for issue in snapshot["issues"]]
        if item.get("runtime"):
            issues.extend(item["runtime"]["issues"])
        details = "; ".join(issues) if issues else "HF structure, required files, and symlinks passed"
        lines.append(f"| {'PASS' if item['complete'] else 'FAIL'} | {item['type']} | `{item['id']}` | {details} |")
    (output / "cache_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", "/nas_aisw/datasets/hf_cache/hub")))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.cache.is_dir():
        raise SystemExit(f"HF cache is not readable: {args.cache}")
    results = [audit_model(args.cache, repo_id, "model") for repo_id in MODEL_IDS]
    results.extend(audit_model(args.cache, repo_id, "lora") for repo_id in LORA_IDS)
    results.extend(audit_dataset(args.cache, spec) for spec in DATASETS)
    if len(results) != 27:
        raise AssertionError("The workbook manifest must contain exactly 27 resources")
    report = write_report(args.output, args.cache, results)
    return 0 if report["incomplete_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check workbook resources through the same offline HF path used by tests."""

import argparse
import json
import os
from pathlib import Path

MODELS = [
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

ADAPTERS = [
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
        "repo": "THUDM/LongBench-v2",
        "selector": "longbench",
        "config": None,
        "split": "train",
        "runtime": "/nas_aisw/datasets/datasets/LLM/ZhipuAI/v2/LongBench-v2",
    },
    {
        "id": "MMMU/MMMU",
        "repo": "MMMU/MMMU",
        "selector": "mmmu",
        "config": "Math",
        "split": "test",
        "runtime": "/nas_aisw/datasets/datasets/LLM/Comprehensive_Examination/v1/MMMU",
    },
    {
        "id": "HellaSwag validation",
        "repo": "Rowan/hellaswag",
        "selector": "hellaswag",
        "config": None,
        "split": "validation",
        "runtime": None,
    },
    {
        "id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "selector": "sharegpt",
        "config": None,
        "split": None,
        "runtime": "/nas_aisw/datasets/datasets/LLM/anon8231489123/v1.0/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json",
    },
]

MODEL_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
DATA_SUFFIXES = {".arrow", ".json", ".jsonl", ".parquet"}


def selected_dataset_file(path, selector):
    name = path.name.lower()
    full_path = str(path).lower()
    if path.suffix.lower() not in DATA_SUFFIXES:
        return False
    if selector == "longbench":
        return "train" in full_path
    if selector == "mmmu":
        return "math" in full_path and "test" in full_path
    if selector == "hellaswag":
        return name == "hellaswag_val.jsonl" or "validation" in full_path
    if selector == "sharegpt":
        return path.name == "ShareGPT_V3_unfiltered_cleaned_split.json"
    raise ValueError(f"Unknown dataset selector: {selector}")


def open_resource(path):
    """Resolve any cache layout and prove that the target is readable and nonempty."""
    target = path.resolve(strict=True)
    if not target.is_file():
        raise ValueError("cache entry does not resolve to a regular file")
    with path.open("rb") as stream:
        prefix = stream.read(4096)
    if not prefix:
        raise ValueError("cache target is empty")
    return {
        "path": str(path),
        "target": str(target),
        "size_bytes": target.stat().st_size,
        "probe_bytes": len(prefix),
        "symlink": path.is_symlink(),
    }


def resolve_snapshot(cache, repo, repo_type, revision):
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=repo,
        repo_type=repo_type,
        revision=revision,
        cache_dir=str(cache),
        local_files_only=True,
    )
    return Path(snapshot)


def check_config(snapshot, name):
    path = snapshot / name
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def cache_files(root):
    """Include files below directory links used by the NAS cache layout."""
    files = []
    for directory, _, names in os.walk(root, followlinks=True):
        for name in names:
            path = Path(directory) / name
            if path.is_file() or path.is_symlink():
                files.append(path)
    return sorted(files)


def check_model_revision(cache, repo, revision, adapter):
    repo_type = "dataset" if adapter else "model"
    snapshot = resolve_snapshot(cache, repo, repo_type, revision)
    errors = []
    config_name = "adapter_config.json" if adapter else "config.json"
    try:
        config = check_config(snapshot, config_name)
        config_type = "adapter" if adapter else config.get("model_type")
    except Exception as error:
        errors.append(f"{config_name}: {type(error).__name__}: {error}")
        config_type = None

    files = cache_files(snapshot)
    if adapter:
        weights = [path for path in files if path.name.startswith("adapter_model.")]
    else:
        weights = [path for path in files if path.suffix.lower() in MODEL_SUFFIXES]
    if not weights:
        errors.append("no adapter weights found" if adapter else "no model weights found")

    probes = []
    for path in weights:
        try:
            probes.append(open_resource(path))
        except Exception as error:
            errors.append(f"{path.relative_to(snapshot)}: {type(error).__name__}: {error}")

    if adapter:
        try:
            open_resource(snapshot / "compare_sample_train_data.pt")
        except Exception as error:
            errors.append(f"reference data: {type(error).__name__}: {error}")

    for index in files:
        if not index.name.endswith((".safetensors.index.json", ".bin.index.json")):
            continue
        try:
            shards = set(check_config(index.parent, index.name)["weight_map"].values())
            if not shards:
                raise ValueError("empty weight map")
            for shard in shards:
                shard_path = Path(shard)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    raise ValueError(f"unsafe shard path: {shard}")
                open_resource(index.parent / shard_path)
        except Exception as error:
            errors.append(f"{index.relative_to(snapshot)}: {type(error).__name__}: {error}")

    return {
        "revision": revision,
        "snapshot": str(snapshot),
        "config_type": config_type,
        "weight_count": len(weights),
        "weight_bytes": sum(probe["size_bytes"] for probe in probes),
        "weight_targets": [probe["target"] for probe in probes[:3]],
        "readable": not errors,
        "errors": errors,
    }


def check_model(cache, repo, adapter, references):
    revisions = ["main", "test"] if repo == "thinkingmachines/Inkling" else ["main"]
    attempts = []
    for revision in revisions:
        try:
            attempts.append(check_model_revision(cache, repo, revision, adapter))
        except Exception as error:
            attempts.append(
                {
                    "revision": revision,
                    "readable": False,
                    "errors": [f"{type(error).__name__}: {error}"],
                }
            )
    return {
        "id": repo,
        "type": "LoRA" if adapter else "model",
        "references": references.get(repo, []),
        "readable": all(attempt["readable"] for attempt in attempts),
        "attempts": attempts,
    }


def dataset_files(root, selector):
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = cache_files(root)
    else:
        raise FileNotFoundError(f"dataset path does not exist: {root}")
    return [path for path in candidates if selected_dataset_file(path, selector)]


def probe_dataset_path(root, spec):
    if spec["selector"] == "sharegpt":
        paths = dataset_files(root, spec["selector"])
        if not paths:
            raise FileNotFoundError(f"no ShareGPT test data found under {root}")
        return [open_resource(path) for path in paths], []

    from datasets import load_dataset

    dataset = load_dataset(
        str(root),
        name=spec["config"],
        split=spec["split"],
        streaming=True,
    )
    row = next(iter(dataset))
    if not row:
        raise ValueError("dataset returned an empty first row")
    return [], sorted(row)


def check_dataset(cache, spec, references):
    attempts = []
    try:
        snapshot = resolve_snapshot(cache, spec["repo"], "dataset", "main")
        probes, row_fields = probe_dataset_path(snapshot, spec)
        attempts.append(
            {
                "source": "HF cache",
                "snapshot": str(snapshot),
                "matched_files": len(probes),
                "targets": [probe["target"] for probe in probes[:3]],
                "row_fields": row_fields,
                "readable": True,
                "errors": [],
            }
        )
    except Exception as error:
        attempts.append(
            {
                "source": "HF cache",
                "readable": False,
                "errors": [f"{type(error).__name__}: {error}"],
            }
        )
    if spec["runtime"]:
        try:
            probes, row_fields = probe_dataset_path(Path(spec["runtime"]), spec)
            attempts.append(
                {
                    "source": "test runtime path",
                    "matched_files": len(probes),
                    "targets": [probe["target"] for probe in probes[:3]],
                    "row_fields": row_fields,
                    "readable": True,
                    "errors": [],
                }
            )
        except Exception as error:
            attempts.append(
                {
                    "source": "test runtime path",
                    "readable": False,
                    "errors": [f"{type(error).__name__}: {error}"],
                }
            )
    return {
        "id": spec["id"],
        "type": "dataset",
        "references": references.get(spec["repo"], []),
        "readable": attempts[-1]["readable"],
        "attempts": attempts,
    }


def find_references():
    repos = set(MODELS + ADAPTERS + [spec["repo"] for spec in DATASETS])
    references = {repo: [] for repo in repos}
    for root_name in ("test", "benchmark"):
        root = Path(root_name)
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for repo in repos:
                if repo in text:
                    references[repo].append(str(path))
    return references


def run_audit(cache):
    references = find_references()
    results = [check_model(cache, repo, False, references) for repo in MODELS]
    results.extend(check_model(cache, repo, True, references) for repo in ADAPTERS)
    results.extend(check_dataset(cache, spec, references) for spec in DATASETS)
    if len(results) != 27:
        raise AssertionError("workbook resource count changed")
    failures = [result["id"] for result in results if not result["readable"]]
    return {
        "cache": str(cache),
        "offline": {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        },
        "resource_count": len(results),
        "readable_count": len(results) - len(failures),
        "failed_count": len(failures),
        "failures": failures,
        "results": results,
    }


def publish(report):
    lines = [
        "# Missing resource startup audit",
        "",
        f"Readable: {report['readable_count']}/{report['resource_count']}; failed: {report['failed_count']}.",
        "",
        "| Status | Type | Resource | Details |",
        "|---|---|---|---|",
    ]
    for result in report["results"]:
        errors = [error for attempt in result["attempts"] for error in attempt["errors"]]
        details = "Test startup path and file reads passed" if result["readable"] else "; ".join(errors)
        lines.append(
            f"| {'PASS' if result['readable'] else 'FAIL'} | {result['type']} | `{result['id']}` | {details} |"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return report["failed_count"] == 0


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--publish", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", "/nas_aisw/datasets/hf_cache/hub")),
    )
    args = parser.parse_args()
    if args.output:
        report = run_audit(args.cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"RESOURCE_AUDIT: {report['readable_count']}/{report['resource_count']} readable; "
            f"failures={report['failures']}",
            flush=True,
        )
        return 0
    report = json.loads(args.publish.read_text(encoding="utf-8"))
    return 0 if publish(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())

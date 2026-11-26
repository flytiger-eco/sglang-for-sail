"""Collect fixed GLM-5.3 outputs or compare two collected result files.

Collection uses an already running 1P1D pair and the native /generate router.
No server is started or stopped by this script. All prompts are synthetic.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def compare(before, after):
    assert before["manifest"] == after["manifest"], "Input or sampling mismatch"
    assert len(before["cases"]) == len(after["cases"])
    rows = []
    for left, right in zip(before["cases"], after["cases"]):
        assert left["id"] == right["id"]
        a, b = left["token_ids"], right["token_ids"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        teacher_a, teacher_b = left["teacher_logprobs"], right["teacher_logprobs"]
        assert left["teacher_token_ids"] == right["teacher_token_ids"]
        assert len(teacher_a) == len(teacher_b)
        assert all(math.isfinite(x) for x in teacher_a + teacher_b)
        differences = sorted(abs(x - y) for x, y in zip(teacher_a, teacher_b))
        rows.append(
            {
                "id": left["id"],
                "tokens_equal": a == b,
                "first_different_token": first,
                "teacher_tokens": len(differences),
                "max_abs_logprob_diff": max(differences, default=0),
                "mean_abs_logprob_diff": sum(differences) / max(1, len(differences)),
                "cached_tokens_before": left["cached_tokens"],
                "cached_tokens_after": right["cached_tokens"],
            }
        )
    return {
        "cases": len(rows),
        "identical_output_cases": sum(row["tokens_equal"] for row in rows),
        "max_abs_logprob_diff": max(row["max_abs_logprob_diff"] for row in rows),
        "byte_identical_outputs": all(row["tokens_equal"] for row in rows),
        "rows": rows,
    }


def collect(args):
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    session = requests.Session()

    def post(url, payload):
        response = session.post(url, json=payload, timeout=600)
        if not response.ok:
            raise RuntimeError(
                f"{url}: HTTP {response.status_code}: {response.text[:4000]}"
            )
        if "/flush_cache" in url:
            return response.text
        result = response.json()
        if isinstance(result, list):
            assert len(result) == 1
            result = result[0]
        return result

    def generate(ids, count, logprob_start=-1):
        return post(
            args.url + "/generate",
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": count,
                    "ignore_eos": True,
                    "sampling_seed": 223072263,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start,
            },
        )

    fixture = []
    for index in range(args.samples):
        length = (128, 2048, 8192, 16384)[index % 4]
        text = (
            "Reference notes: red=17; blue=29; green=41. "
            "The notes repeat to exercise cached context pages.\n"
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        prefix = (tokens * (length // len(tokens) + 1))[:length]
        suffix = tokenizer.encode(
            f"\nQuestion {index}: What is red plus blue? Explain briefly.\nAnswer:",
            add_special_tokens=False,
        )
        fixture.append((prefix, suffix))
    manifest = {
        "model": Path(args.model).name,
        "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "seed": 223072263,
        "topology": "1P1D; TP8/EP8; prefill CP8 LayerSplit",
        "fixture_sha256": hashlib.sha256(json.dumps(fixture).encode()).hexdigest(),
    }
    if reference:
        assert reference["manifest"] == manifest
    result = {"manifest": manifest, "label": args.label, "cases": []}
    for index, (prefix, suffix) in enumerate(fixture):
        for mode in ("cold", "prefix_hit"):
            for worker in (args.prefill, args.decode):
                post(worker + "/flush_cache?timeout=30", {})
            if mode == "prefix_hit":
                generate(prefix, 1)
            ids = prefix + suffix
            generated = generate(ids, args.max_new_tokens)
            metadata = generated["meta_info"]
            entries = metadata["output_token_logprobs"]
            output_ids = [int(entry[1]) for entry in entries]
            assert len(output_ids) == args.max_new_tokens
            row_id = f"{index:02d}-{mode}"
            if reference:
                ref = reference["cases"][len(result["cases"])]
                assert ref["id"] == row_id
                teacher_ids = ref["token_ids"]
            else:
                teacher_ids = output_ids
            for worker in (args.prefill, args.decode):
                post(worker + "/flush_cache?timeout=30", {})
            # The API marks the first requested input-logprob position as None.
            # Include the final prompt token so every continuation token is scored.
            teacher = generate(ids + teacher_ids, 1, len(ids) - 1)
            scored = teacher["meta_info"]["input_token_logprobs"]
            scored = [entry for entry in scored if entry[0] is not None]
            assert [int(entry[1]) for entry in scored] == teacher_ids, (
                [int(entry[1]) for entry in scored],
                teacher_ids,
                teacher["meta_info"],
            )
            row = {
                "id": row_id,
                "input_length": len(ids),
                "token_ids": output_ids,
                "text": generated["text"],
                "cached_tokens": metadata.get("cached_tokens", 0),
                "teacher_token_ids": teacher_ids,
                "teacher_logprobs": [float(entry[0]) for entry in scored],
            }
            if mode == "prefix_hit":
                assert row["cached_tokens"] > 0, "Cache-hit path did not run"
            result["cases"].append(row)
            Path(args.output).write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            print(
                json.dumps(
                    {
                        "id": row_id,
                        "input_tokens": len(ids),
                        "cached_tokens": row["cached_tokens"],
                        "time": time.time(),
                    }
                ),
                flush=True,
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--url", default="http://127.0.0.1:17680")
    parser.add_argument("--prefill", default="http://127.0.0.1:17699")
    parser.add_argument("--decode", default="http://127.0.0.1:17690")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--reference")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.before and arguments.after:
        summary = compare(
            json.loads(Path(arguments.before).read_text()),
            json.loads(Path(arguments.after).read_text()),
        )
        Path(arguments.output).write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    else:
        if not arguments.model:
            parser.error("--model is required when collecting")
        collect(arguments)

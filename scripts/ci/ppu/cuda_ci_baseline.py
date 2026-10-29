#!/usr/bin/env python3
"""Static CUDA CI baseline for PPU parity planning.

Inventories every CI registration, resolves which workflow dispatches each CUDA
suite and on which runner, lists CUDA entry points that bypass the registry
(pytest / multimodal runners / scripts on GPU runners), and records which CUDA
files already carry a PPU registration.

Static only: times are the in-source `est_time` values (CUDA hardware), and no
test is collected or executed. Keyword hints are triage aids, not verdicts.

    uv run --no-project --with pyyaml scripts/ci/ppu/cuda_ci_baseline.py --out-dir /tmp/baseline
"""

import argparse
import ast
import collections
import csv
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKFLOW_DIR = os.path.join(REPO, ".github", "workflows")

# Nominal per-device memory: CUDA runner GPUs in GB, PPU boards in GiB.
CUDA_GPUS = {
    "5090": ("RTX5090", 32),
    "h100": ("H100", 80),
    "h20": ("H20", 96),
    "h200": ("H200", 141),
    "b200": ("B200", 180),
    "gb200": ("GB200", 186),
    "b300": ("B300", 288),
    "gb300": ("GB300", 288),
}
PPU_BOARDS = {"810E": 96, "890P": 144}
PPU_FP8 = {"810E": 0, "890P": 1}

# Nominal grouping only; verify_cadence independently checks reachable root triggers.
CADENCE_PREFIXES = (
    ("base-", "per_commit"),
    ("extra-", "extra"),
    ("nightly-", "nightly"),
    ("weekly-", "weekly"),
    ("stress", "stress"),
)
CUDA_CADENCES = ("per_commit", "extra", "nightly", "weekly", "stress", "other")

# Triage hints only: NVIDIA-specific stacks or Blackwell/Hopper-only formats.
HINT_PATTERNS = {
    "fp4": r"nvfp4|mxfp4|\bfp4\b|modelopt_fp4",
    "deepgemm": r"deep_gemm|deepgemm",
    "deepep": r"deep_ep|deepep",
    "flashinfer": r"flashinfer",
    "fa3_fa4": r"\bfa3\b|\bfa4\b|flash_attn_interface|flash_attn\.cute",
    "trtllm": r"trtllm|tensorrt",
    "cutlass_cute": r"cutlass|cute_dsl|cutedsl|\bcute\b",
    "arch_gate": r"is_blackwell|is_hopper|sm100|sm90|is_sm\d+|get_device_capability",
    "rdma_disagg": r"mooncake|nixl|\bmlx5|ib_device|rdma",
    "triton": r"\btriton\b",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci_register = load_module(
    "ci_register",
    os.path.join(REPO, "python", "sglang", "test", "ci", "ci_register.py"),
)
compute_partitions = load_module(
    "compute_partitions",
    os.path.join(REPO, "scripts", "ci", "utils", "compute_partitions.py"),
)


def rel(path):
    return os.path.relpath(path, REPO)


def cadence_of(suite):
    for prefix, cadence in CADENCE_PREFIXES:
        if (suite or "").startswith(prefix):
            return cadence
    return "other"


def verify_cadence(suite, entries, workflows):
    """Check nominal cadence against root events, not the suite name again.

    Extra and stress are CI policy lanes, not event frequencies. All other
    observations come from PR/push events or the simple cron shapes used here.
    Reachability is potential: job conditions, labels and runtime inputs are
    not evaluated. A matching root verifies the grouping, not actual execution.
    """
    expected = cadence_of(suite)
    roots = sorted({root for e in entries for root in e["reachable_from"]})
    evidence = []
    for root in roots:
        info = workflows[root]
        triggers = set(info["triggers"]) - {"workflow_call"}
        observed = set()
        if triggers & {"pull_request", "pull_request_target", "push", "merge_group"}:
            observed.add("extra" if root == "pr-test-extra.yml" else "per_commit")
        if "schedule" in triggers:
            for cron in info["crons"]:
                fields = (cron or "").split()
                if len(fields) != 5:
                    continue
                _, _, day, month, weekday = fields
                if (
                    month == "*"
                    and weekday == "*"
                    and re.fullmatch(r"\*(?:/\d+)?", day)
                ):
                    observed.add("nightly")
                elif month == day == "*" and re.fullmatch(r"[0-6](?:,[0-6])*", weekday):
                    observed.add("weekly")
        if root == "stress-test.yml" and "workflow_dispatch" in triggers:
            observed.add("stress")
        if not observed:
            observed.add(
                "dispatch_only" if triggers == {"workflow_dispatch"} else "unknown"
            )
        evidence.append(
            {
                "workflow": root,
                "triggers": info["triggers"],
                "crons": info["crons"],
                "observed_cadences": sorted(observed),
            }
        )
    observed = {c for e in evidence for c in e["observed_cadences"]}
    if not roots:
        status = "missing_root"
    elif expected == "other":
        status = "unknown"
    elif expected in observed:
        status = "verified"
    elif observed == {"dispatch_only"}:
        status = "dispatch_only"
    elif observed - {"unknown", "dispatch_only"}:
        status = "mismatch"
    else:
        status = "unknown"
    return {
        "cadence_basis": "suite_prefix",
        "cadence_verification": status,
        "cadence_evidence": evidence,
    }


def suite_lists():
    """PER_COMMIT_SUITES / NIGHTLY_SUITES from run_suite.py, parsed via AST."""
    with open(os.path.join(REPO, "test", "run_suite.py")) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        name = node.targets[0].id
        if name not in ("PER_COMMIT_SUITES", "NIGHTLY_SUITES"):
            continue
        out[name] = {
            key.attr: [elt.value for elt in value.elts]
            for key, value in zip(node.value.keys, node.value.values)
        }
    return out


def runner_configs():
    with open(os.path.join(REPO, "scripts", "ci", "runner_configs.yml")) as f:
        configs = yaml.safe_load(f)["runner_configs"]
    # `$b200_runner` resolves to 4-gpu-b200 unless sgl-kernel changed.
    return {
        name: ("4-gpu-b200" if cfg["runs_on"] == "$b200_runner" else cfg["runs_on"])
        for name, cfg in configs.items()
    }


def parse_label(label):
    """'8-gpu-h200' -> (8, 'H200', 141); unknown labels -> (None, label, None)."""
    m = re.match(r"(\d+)-gpu-([a-z0-9]+)", label or "")
    if not m:
        return None, label, None
    model, mem = CUDA_GPUS.get(m.group(2), (m.group(2), None))
    return int(m.group(1)), model, mem


def run_text_commands(run):
    text = run.replace("\\\n", " ")
    cwd = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"cd\s+(\S+)", line)
        if m:
            cwd = m.group(1)
        yield cwd, line


def load_workflows():
    workflows = {}
    for name in sorted(os.listdir(WORKFLOW_DIR)):
        if not name.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(WORKFLOW_DIR, name)) as f:
            data = yaml.safe_load(f) or {}
        triggers = data.get("on", data.get(True)) or {}
        if isinstance(triggers, str):
            triggers = {triggers: None}
        elif isinstance(triggers, list):
            triggers = {t: None for t in triggers}
        crons = [s.get("cron") for s in (triggers.get("schedule") or [])]
        workflows[name] = {
            "triggers": sorted(triggers),
            "crons": crons,
            "jobs": data.get("jobs") or {},
            "env": data.get("env") or {},
        }
    return workflows


# Deliberately not a shell interpreter. Only these current entry points and
# literal/env/matrix suite arguments are resolved; source/eval are never run.
PPU_SCRIPTS = {
    "run_pr_suite.sh",
    "run_answer_suite_board.sh",
    "run_answer_suite_node.sh",
    "run_perf_suite_board.sh",
    "run_perf_suite_node.sh",
    "run_accuracy_suite_board.sh",
}
PPU_COMMAND_ACTIONS = (
    "flytiger-eco/shared-workflows/cluster-task@",
    "flytiger-eco/ppu-distributed-action@",
)
SHELL_TOKEN = r"""(?:\$\{\{.*?\}\}|"[^"]*"|'[^']*'|[^\s]+)"""


def literal_values(value, variables, seen=()):
    """Resolve whole-token references only; unknown expressions remain unknown."""
    if isinstance(value, (list, set)):
        return {v for item in value for v in literal_values(item, variables, seen)}
    text = str(value).replace('\\"', '"').replace("\\$", "$").strip("\"'")
    if re.fullmatch(r"[a-zA-Z0-9_.-]+", text):
        return {text}
    ref = re.fullmatch(r"\$\{\{\s*(env|matrix)\.(\w+)\s*\}\}", text)
    if ref:
        key = ref[2] if ref[1] == "env" else "matrix." + ref[2]
    else:
        ref = re.fullmatch(r"\$(?:\{(\w+)\}|(\w+))", text)
        if not ref:
            return set()
        key = ref[1] or ref[2]
    if key in seen or key not in variables:
        return set()
    return literal_values(variables[key], variables, (*seen, key))


def suite_commands(text):
    """Read command lines/continuations, excluding comments, prose and heredocs.

    Physical lines also cover commands inside the existing Docker bash -c
    blocks. This does not attempt arbitrary shell control flow or expansion.
    """
    lines = iter(text.splitlines())
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if heredoc:
            for body in lines:
                if body.strip() == heredoc[1]:
                    break
            continue
        if not re.match(
            r"(?:python3?(?: -u)?\s+\S*run_suite\.py\b|bash\s+\S*scripts/ci/ppu/\w+\.sh\b)",
            line,
        ):
            continue
        # A continued shell chain starts a new command, not more arguments to
        # this wrapper (notably ppu_install_dependency.sh && in Docker blocks).
        while line.endswith("\\") and not re.search(r"(?:&&|\|\||;)\s*\\$", line):
            line = line[:-1] + " " + next(lines, "").strip()
        yield line.replace('\\"', '"').replace("\\$", "$")


def scan_ppu_step(info, job, step, entry):
    """Follow workflow commands into the allowlisted PPU suite wrappers only."""
    variables = {**info["env"], **(job.get("env") or {}), **(step.get("env") or {})}
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if isinstance(matrix, dict):
        for key, values in matrix.items():
            if key not in ("include", "exclude") and isinstance(values, list):
                variables["matrix." + key] = list(values)
        for row in matrix.get("include", []):
            for key, value in row.items():
                variables.setdefault("matrix." + key, []).append(value)
    with_ = step.get("with") or {}
    run = step.get("run") or ""
    if (step.get("uses") or "").startswith(PPU_COMMAND_ACTIONS):
        run = with_.get("command") or ""
        for match in re.finditer(
            r"(?:^|,)\s*(\w+)=([^,]*)", with_.get("extra_env") or ""
        ):
            variables[match[1]] = literal_values(match[2], variables) or match[2]
    for match in re.finditer(r"(?m)^\s*(?:export )?(\w+)=(\S+)\s*$", run):
        variables[match[1]] = match[2]

    def registered(command, bindings, script="", invocation="", condition=""):
        hw = re.search(r"--hw[ =]+(" + SHELL_TOKEN + ")", command)
        suite = re.search(r"--suite[ =]+(" + SHELL_TOKEN + ")", command)
        if not hw or not suite:
            return []
        backends = literal_values(hw[1], bindings)
        if not backends & ({"ppu", "cpu"} if script else {"ppu"}):
            return []
        suites = literal_values(suite[1], bindings)
        return [
            dict(
                entry,
                kind="registered" if suites else "unresolved_registered",
                hw=backend,
                suite=name,
                command=command,
                script=script,
                invocation=invocation,
                condition=condition,
            )
            for backend in sorted(backends)
            for name in sorted(suites) or [None]
        ]

    entries = []
    for command in suite_commands(run):
        if command.startswith("python"):
            entries.extend(registered(command, variables))
            continue
        script = re.search(r"scripts/ci/ppu/(\w+\.sh)\b(.*)", command)
        if not script or script[1] not in PPU_SCRIPTS:
            continue
        path = "scripts/ci/ppu/" + script[1]
        try:
            with open(os.path.join(REPO, path)) as f:
                text = f.read()
        except OSError:
            continue
        args = re.findall(SHELL_TOKEN, script[2].strip())
        lanes = [None]
        if script[1] == "run_pr_suite.sh":
            lanes = sorted(literal_values(args[0], variables)) if args else []
            lanes = lanes or [None]
        for lane in lanes:
            bindings = dict(variables)
            if script[1] == "run_pr_suite.sh":
                branch = re.search(
                    r"(?m)^\s*" + re.escape(lane) + r"\)\s*hw=(\w+);\s*suite=([\w-]+);",
                    text,
                ) if lane is not None else None
                if not branch:
                    entries.append(dict(entry, kind="unresolved_registered", hw="ppu",
                                        suite=None, command=command, script=path,
                                        invocation=command, condition="unresolved PR lane"))
                    continue
                bindings.update(hw=branch[1], suite=branch[2])
            elif script[1].endswith("_node.sh"):
                bindings["suite"] = literal_values(args[0], variables) if args else set()
            for call in suite_commands(text):
                condition = ""
                if lane is not None and "--suite stage-b-kernel-test-1-gpu-ppu" in call:
                    flag = variables.get("PPU_RUN_KERNEL", "false")
                    if lane != "body" or literal_values(flag, variables) == {"false"}:
                        continue
                    condition = "lane == body && PPU_RUN_KERNEL == true (conditional)"
                entries.extend(registered(call, bindings, path, command, condition))
    return entries


def scan_workflows(workflows, rc_map):
    """Return registry/native command entries annotated with reachable roots."""
    edges = collections.defaultdict(set)
    passed_inputs = collections.defaultdict(set)
    entries = []
    for wf, info in workflows.items():
        for job_id, job in info["jobs"].items():
            if not isinstance(job, dict):
                continue
            uses = job.get("uses") or ""
            if uses.startswith("./.github/workflows/"):
                callee = os.path.basename(uses)
                edges[wf].add(callee)
                with_ = job.get("with") or {}
                passed_inputs[callee].update(with_)
                if callee in workflows and callee in (
                    "_pr-test-stage.yml",
                    "_pr-test-stage-cpu.yml",
                ):
                    rc = with_.get("runner_config")
                    entries.append(
                        {
                            "workflow": wf,
                            "job": job_id,
                            "kind": "registered",
                            "hw": "cuda" if callee == "_pr-test-stage.yml" else "cpu",
                            "suite": with_.get("self_name", job_id),
                            "runs_on": rc_map.get(rc, rc),
                            "cwd": "test",
                            "command": f"{callee} self_name={with_.get('self_name', job_id)}",
                        }
                    )
                continue
            runs_on = job.get("runs-on")
            runs_on = runs_on if isinstance(runs_on, str) else json.dumps(runs_on)
            gpu_job = "-gpu-" in runs_on or "runs_on_map" in runs_on
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                entry = {"workflow": wf, "job": job_id, "runs_on": runs_on, "cwd": None}
                entries.extend(scan_ppu_step(info, job, step, entry))
                run = step.get("run")
                if not run:
                    continue
                for cwd, line in run_text_commands(run):
                    if line.startswith(("#", "echo ", "printf ")):
                        continue
                    entry = {
                        "workflow": wf,
                        "job": job_id,
                        "runs_on": runs_on,
                        "cwd": cwd,
                    }
                    m = re.search(r"(\S*run_suite\.py)(.*)", line)
                    if m:
                        hw = re.search(r"--hw[ =]+(\S+)", m.group(2))
                        suite = re.search(r"--suite[ =]+(\S+)", m.group(2))
                        native = "multimodal_gen" in m.group(1)
                        if hw and hw[1].strip("\"'") == "ppu":
                            continue  # Resolved from actual commands above, not prose.
                        entries.append(
                            dict(
                                entry,
                                kind="multimodal_run_suite" if native else "registered",
                                hw=None if native else (hw.group(1) if hw else None),
                                suite=suite.group(1) if suite else None,
                                command=line,
                            )
                        )
                        continue
                    if not gpu_job:
                        continue
                    if re.match(r"(timeout \d+ )?(python3? -m )?pytest\b", line):
                        kind = "pytest"
                    elif re.search(
                        r"python3?( -u)? [\"']?(\S+\.py|\$\{?\w+)", line
                    ) and not re.search(
                        r"install|runner_configs|metrics|upload|dashboard|publish|warmup",
                        line,
                    ):
                        kind = "script"
                    else:
                        continue
                    entries.append(
                        dict(entry, kind=kind, hw="cuda", suite=None, command=line)
                    )

    # A command in a call-only workflow driven solely by inputs no caller passes never runs.
    def dead(e):
        refs = re.findall(r"inputs\.(\w+)", e["command"])
        call_only = workflows[e["workflow"]]["triggers"] == ["workflow_call"]
        return (
            e["kind"] != "registered"
            and call_only
            and refs
            and not set(refs) & passed_inputs[e["workflow"]]
        )

    entries = [e for e in entries if not dead(e)]
    roots = [
        wf
        for wf, info in workflows.items()
        if set(info["triggers"]) - {"workflow_call"}
    ]
    reaching = collections.defaultdict(set)
    for root in roots:
        stack, seen = [root], set()
        while stack:
            wf = stack.pop()
            if wf in seen:
                continue
            seen.add(wf)
            reaching[wf].add(root)
            stack.extend(edges.get(wf, ()))
    for e in entries:
        e["reachable_from"] = sorted(reaching.get(e["workflow"], ()))
    return entries


def diffusion_plan():
    """Case-level partition plan of the CUDA diffusion runner, if it can be computed."""
    script = os.path.join(
        REPO, "scripts", "ci", "utils", "diffusion", "compute_diffusion_partitions.py"
    )
    with tempfile.NamedTemporaryFile("r", suffix=".txt") as out:
        env = dict(os.environ, GITHUB_OUTPUT=out.name)
        proc = subprocess.run(
            [
                sys.executable,
                script,
                "--min-time",
                "1200",
                "--target-time",
                "1800",
                "--max-time",
                "2400",
                "--max-partitions",
                "10",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip().splitlines()[-1:]}
        plans = {}
        for line in out.read().splitlines():
            key, _, value = line.partition("=")
            if key.startswith("plan-"):
                plan = json.loads(value)
                parts = plan["partitions"]
                plans[plan["suite"]] = {
                    "partitions": len(parts),
                    "cases": sum(len(p["case_ids"]) for p in parts),
                    "standalone_files": sum(len(p["standalone_files"]) for p in parts),
                    "est_hours": round(
                        sum(p["estimated_time"] for p in parts) / 3600, 2
                    ),
                }
        return plans


def keyword_hints(path):
    try:
        with open(os.path.join(REPO, path), errors="ignore") as f:
            text = f.read().lower()
    except OSError:
        text = ""
    return [k for k, pat in HINT_PATTERNS.items() if re.search(pat, text)]


def git_head():
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    date = subprocess.run(
        ["git", "show", "-s", "--format=%cI", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return sha.stdout.strip(), date.stdout.strip(), branch.stdout.strip()


def build(out_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        files = compute_partitions.discover_files(REPO)
        regs = ci_register.collect_tests(files, sanity_check=False)
    unregistered = sorted(
        rel(str(w.message).split(" in ", 1)[1])
        for w in caught
        if "No CI registry" in str(w.message)
    )

    lists = suite_lists()
    rc_map = runner_configs()
    workflows = load_workflows()
    entries = scan_workflows(workflows, rc_map)

    suite_entries = collections.defaultdict(list)
    for e in entries:
        if e["kind"] == "registered" and e["suite"]:
            suite_entries[(e["hw"], e["suite"])].append(e)

    def resources(suite, hw_entries):
        rc = suite.rsplit("-test-", 1)[1] if "-test-" in suite else None
        label = rc_map.get(rc)
        if label is None:
            literal = [
                e["runs_on"] for e in hw_entries if "-gpu-" in (e["runs_on"] or "")
            ]
            label = literal[0] if literal else rc
        n, model, mem = parse_label(label)
        return label, n, model, mem

    rows = []
    for r in regs:
        suite = r.effective_suite or ""
        backend = r.backend.name
        row = {
            "file": rel(r.filename),
            "backend": backend,
            "suite": suite,
            "est_time_s": r.est_time,
            "disabled": r.disabled or "",
            "nightly_flag": r.nightly,
        }
        if backend == "CUDA":
            hw_entries = suite_entries.get(("cuda", suite), [])
            label, n, model, mem = resources(suite, hw_entries)
            row.update(
                cadence=cadence_of(suite),
                runner_label=label,
                n_devices=n,
                gpu_model=model,
                gpu_mem_gb=mem,
                reachable_from=";".join(
                    sorted({w for e in hw_entries for w in e["reachable_from"]})
                ),
                **verify_cadence(suite, hw_entries, workflows),
            )
        else:
            hw_entries = suite_entries.get((backend.lower(), suite), [])
            per_commit = suite in lists["PER_COMMIT_SUITES"].get(backend, [])
            nightly = suite in lists["NIGHTLY_SUITES"].get(backend, [])
            m = re.search(r"-(\d+)-(?:gpu-)?ppu$|^nightly-(?:[a-z]+-)?(\d+)-", suite)
            row.update(
                cadence=(
                    "per_commit" if per_commit else ("nightly" if nightly else "other")
                ),
                runner_label="",
                n_devices=(
                    int(next(g for g in m.groups() if g))
                    if (backend == "PPU" and m)
                    else None
                ),
                gpu_model="",
                gpu_mem_gb=None,
                reachable_from=";".join(
                    sorted({w for e in hw_entries for w in e["reachable_from"]})
                ),
            )
        rows.append(row)

    cuda = [r for r in rows if r["backend"] == "CUDA"]
    ppu = [r for r in rows if r["backend"] == "PPU"]
    ppu_by_file = collections.defaultdict(list)
    for r in ppu:
        ppu_by_file[r["file"]].append(r)

    # Per CUDA file view for triage.
    file_rows = []
    by_file = collections.defaultdict(list)
    for r in cuda:
        by_file[r["file"]].append(r)
    for path, regs_ in sorted(by_file.items()):
        enabled = [r for r in regs_ if not r["disabled"]]
        p = ppu_by_file.get(path, [])
        p_enabled = [r for r in p if not r["disabled"]]
        parts = path.split("/")
        file_rows.append(
            {
                "file": path,
                "area": (
                    parts[2]
                    if parts[:2] == ["test", "registered"] and len(parts) > 3
                    else parts[0]
                ),
                "cuda_cadences": ";".join(sorted({r["cadence"] for r in enabled})),
                "cuda_suites": ";".join(sorted({r["suite"] for r in enabled})),
                "cuda_disabled_suites": ";".join(
                    sorted({r["suite"] for r in regs_ if r["disabled"]})
                ),
                "max_devices": max(
                    (r["n_devices"] or 1 for r in enabled), default=None
                ),
                "gpu_models": ";".join(
                    sorted({r["gpu_model"] for r in enabled if r["gpu_model"]})
                ),
                "cuda_est_time_s": sum(r["est_time_s"] for r in enabled),
                "ppu_suites": ";".join(sorted({r["suite"] for r in p_enabled})),
                "ppu_cadences": ";".join(sorted({r["cadence"] for r in p_enabled})),
                "ppu_disabled": ";".join(
                    sorted(
                        {f"{r['suite']}: {r['disabled']}" for r in p if r["disabled"]}
                    )
                ),
                "keyword_hints": ";".join(keyword_hints(path)),
            }
        )

    # Aggregates.
    def agg(items, key):
        out = collections.OrderedDict()
        for r in sorted(items, key=key):
            k = key(r)
            a = out.setdefault(
                k,
                {"registrations": 0, "files": set(), "serial_h": 0.0, "device_h": 0.0},
            )
            a["registrations"] += 1
            a["files"].add(r["file"])
            a["serial_h"] += r["est_time_s"] / 3600
            a["device_h"] += r["est_time_s"] * (r["n_devices"] or 1) / 3600
        for a in out.values():
            a["files"] = len(a["files"])
            a["serial_h"] = round(a["serial_h"], 2)
            a["device_h"] = round(a["device_h"], 2)
        return out

    cuda_enabled = [r for r in cuda if not r["disabled"]]
    by_cadence = agg(cuda_enabled, lambda r: r["cadence"])
    by_suite = agg(cuda_enabled, lambda r: (r["cadence"], r["suite"]))
    by_gpu = agg(
        cuda_enabled,
        lambda r: (r["cadence"], r["gpu_model"] or "?", r["n_devices"] or 0),
    )

    suite_meta = {}
    for (cadence, suite), a in by_suite.items():
        sample = next(r for r in cuda_enabled if r["suite"] == suite)
        suite_meta[suite] = {
            "cadence": cadence,
            "runner_label": sample["runner_label"],
            "n_devices": sample["n_devices"],
            "gpu_model": sample["gpu_model"],
            "gpu_mem_gb": sample["gpu_mem_gb"],
            "reachable_from": sample["reachable_from"],
            **{
                key: sample[key]
                for key in ("cadence_basis", "cadence_verification", "cadence_evidence")
            },
            **a,
        }

    overlap = {}
    for cadence in CUDA_CADENCES:
        cadence_files = {r["file"] for r in cuda_enabled if r["cadence"] == cadence}
        if not cadence_files:
            continue
        with_ppu = {
            f
            for f in cadence_files
            if any(not r["disabled"] for r in ppu_by_file.get(f, []))
        }
        ppu_pc = {
            f
            for f in with_ppu
            if any(
                r["cadence"] == "per_commit" and not r["disabled"]
                for r in ppu_by_file[f]
            )
        }
        ppu_disabled_only = {f for f in cadence_files - with_ppu if ppu_by_file.get(f)}
        overlap[cadence] = {
            "cuda_files": len(cadence_files),
            "with_enabled_ppu_registration": len(with_ppu),
            "of_which_ppu_per_commit": len(ppu_pc),
            "ppu_registered_but_disabled": len(ppu_disabled_only),
        }
    cuda_files = {r["file"] for r in cuda}
    ppu_enabled = [r for r in ppu if not r["disabled"]]

    registry_cuda_suites = set(
        lists["PER_COMMIT_SUITES"].get("CUDA", [])
        + lists["NIGHTLY_SUITES"].get("CUDA", [])
    )
    anomalies = {
        "cuda_suites_with_registrations_but_no_dispatch": sorted(
            s for s, m in suite_meta.items() if not m["reachable_from"]
        ),
        "cuda_suites_listed_in_run_suite_without_registrations": sorted(
            registry_cuda_suites - set(suite_meta)
        ),
        "cuda_disabled_registrations": sum(1 for r in cuda if r["disabled"]),
        "cuda_cadence_verification": {
            status: sorted(
                s for s, m in suite_meta.items() if m["cadence_verification"] == status
            )
            for status in ("missing_root", "mismatch", "dispatch_only", "unknown")
        },
        "ppu_suites_without_reachable_invocation": sorted(
            {r["suite"] for r in ppu_enabled if not r["reachable_from"]}
        ),
        "ppu_unresolved_suite_commands": [
            e
            for e in entries
            if e["hw"] == "ppu" and e["kind"] == "unresolved_registered"
        ],
        "files_without_registration": unregistered,
    }

    native = [
        e
        for e in entries
        if e["kind"] not in ("registered", "unresolved_registered") and e["hw"] != "cpu"
    ]
    native = [
        e
        for e in native
        if not e["workflow"].startswith(
            (
                "pr-test-amd",
                "nightly-test-amd",
                "release-",
                "pr-test-musa",
                "nightly-test-musa",
                "pr-test-npu",
                "nightly-test-npu",
                "pr-test-xpu",
                "pr-test-ppu",
                "nightly-test-ppu",
                "test-ppu",
            )
        )
    ]

    ppu_suites = agg(ppu_enabled, lambda r: f"{r['cadence']}|{r['suite']}")
    for key, meta in ppu_suites.items():
        invocations = suite_entries.get(("ppu", key.split("|", 1)[1]), [])
        meta["reachable_from"] = ";".join(
            sorted({w for e in invocations for w in e["reachable_from"]})
        )
        meta["invocations"] = invocations

    sha, date, branch = git_head()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_head": {"sha": sha, "commit_date": date, "branch": branch},
        "method": "static: ci_register AST + workflow root/trigger graph + allowlisted PPU shell invocations; times are CUDA est_time",
        "limitations": (
            "Nominal cadence is grouped by suite prefix (CUDA) or run_suite lists (PPU), "
            "not proof of scheduling. CUDA verification checks reachable root triggers; "
            "PR extra/stress are policy lanes, nightly includes multi-day cron intervals. "
            "Reachability is potential, not execution: conditions, labels, dynamic dispatch, "
            "remote workflows/actions and arbitrary shell/Python are not interpreted. "
            "Only the known PPU worker actions, suite scripts and literal/env/matrix "
            "arguments are resolved; an unresolved or absent path is not proof a suite never runs."
        ),
        "discovered_files": len(files),
        "registrations_by_backend": collections.Counter(r["backend"] for r in rows),
        "cuda": {
            "by_cadence": by_cadence,
            "by_suite": suite_meta,
            "by_gpu": {f"{c}|{g}|{n}": a for (c, g, n), a in by_gpu.items()},
        },
        "ppu": {
            "enabled_registrations": len(ppu_enabled),
            "by_cadence": agg(ppu_enabled, lambda r: r["cadence"]),
            "by_suite": ppu_suites,
            "files_also_registered_for_cuda": len(
                {r["file"] for r in ppu_enabled} & cuda_files
            ),
            "ppu_only_files": len({r["file"] for r in ppu_enabled} - cuda_files),
        },
        "cuda_file_overlap_with_ppu": overlap,
        "workflow_native_cuda_entries": native,
        "diffusion_plan": diffusion_plan(),
        "anomalies": anomalies,
        "ppu_boards_gib": PPU_BOARDS,
    }

    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "registrations.csv"), rows)
    write_csv(os.path.join(out_dir, "cuda_files.csv"), file_rows)
    write_csv(
        os.path.join(out_dir, "workflow_entries.csv"),
        [dict(e, reachable_from=";".join(e["reachable_from"])) for e in entries],
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o),
        )
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(render_markdown(summary, workflows))
    return summary


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    out += [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in rows
    ]
    return "\n".join(out) + "\n"


def render_markdown(s, workflows):
    head = s["repo_head"]
    roots = {
        root
        for backend in ("cuda", "ppu")
        for meta in s[backend]["by_suite"].values()
        for root in meta["reachable_from"].split(";")
        if root
    }
    md = [
        "# CUDA CI baseline (static)\n",
        f"- Repo HEAD: `{head['sha']}` ({head['branch']}, {head['commit_date']})",
        f"- Generated: {s['generated_at']}",
        f"- Method: {s['method']}",
        f"- Limits: {s['limitations']}",
        f"- Discovered files: {s['discovered_files']}; registrations by backend: {dict(s['registrations_by_backend'])}\n",
        "## CUDA registered tests by nominal cadence (enabled)\n",
        table(
            ["cadence", "registrations", "files", "serial h", "device h"],
            [
                [c, a["registrations"], a["files"], a["serial_h"], a["device_h"]]
                for c, a in s["cuda"]["by_cadence"].items()
            ],
        ),
        "## CUDA suites\n",
        table(
            [
                "nominal cadence",
                "verification",
                "suite",
                "runner",
                "GPU",
                "n",
                "mem GB",
                "regs",
                "files",
                "serial h",
                "device h",
                "reachable roots",
            ],
            [
                [
                    m["cadence"],
                    m["cadence_verification"],
                    suite,
                    m["runner_label"],
                    m["gpu_model"],
                    m["n_devices"],
                    m["gpu_mem_gb"],
                    m["registrations"],
                    m["files"],
                    m["serial_h"],
                    m["device_h"],
                    m["reachable_from"] or "**none**",
                ]
                for suite, m in s["cuda"]["by_suite"].items()
            ],
        ),
        "## CUDA demand by GPU (device-hours per full run, CUDA est_time)\n",
        table(
            ["cadence", "GPU", "devices/job", "regs", "serial h", "device h"],
            [
                k.split("|") + [a["registrations"], a["serial_h"], a["device_h"]]
                for k, a in s["cuda"]["by_gpu"].items()
            ],
        ),
        "## Current PPU registrations (enabled)\n",
        table(
            [
                "nominal cadence",
                "suite",
                "regs",
                "files",
                "serial h",
                "reachable roots",
            ],
            [
                k.split("|")
                + [
                    a["registrations"],
                    a["files"],
                    a["serial_h"],
                    a["reachable_from"] or "**none**",
                ]
                for k, a in s["ppu"]["by_suite"].items()
            ],
        ),
        f"PPU files also registered for CUDA: {s['ppu']['files_also_registered_for_cuda']}; "
        f"PPU-only files: {s['ppu']['ppu_only_files']}\n",
        "## CUDA files already registered for PPU\n",
        table(
            [
                "CUDA cadence",
                "CUDA files",
                "with enabled PPU reg",
                "of which PPU per-commit",
                "PPU reg disabled only",
            ],
            [
                [
                    c,
                    o["cuda_files"],
                    o["with_enabled_ppu_registration"],
                    o["of_which_ppu_per_commit"],
                    o["ppu_registered_but_disabled"],
                ]
                for c, o in s["cuda_file_overlap_with_ppu"].items()
            ],
        ),
        "## Workflow-native CUDA entries (outside the registry)\n",
        table(
            ["workflow", "job", "runs-on", "kind", "cwd", "command", "reached from"],
            [
                [
                    e["workflow"],
                    e["job"],
                    e["runs_on"],
                    e["kind"],
                    e["cwd"],
                    "`" + e["command"][:110] + "`",
                    ";".join(e["reachable_from"]),
                ]
                for e in s["workflow_native_cuda_entries"]
            ],
        ),
        "### Diffusion runner partition plan\n",
        "```json\n" + json.dumps(s["diffusion_plan"], indent=2) + "\n```\n",
        "## Anomalies\n",
        "```json\n"
        + json.dumps({k: v for k, v in s["anomalies"].items()}, indent=2)
        + "\n```\n",
        "## Root workflow triggers\n",
        table(
            ["workflow", "triggers", "crons"],
            [
                [wf, ",".join(map(str, info["triggers"])), ";".join(info["crons"])]
                for wf, info in workflows.items()
                if wf in roots
            ],
        ),
    ]
    return "\n".join(md)


# P1 deliberately consumes P0's registration rows rather than re-discovering a
# second inventory. No test module or SGLang package is imported here.
SWEEP_CADENCES = {"per_commit", "extra", "nightly"}
DEFAULT_SWEEP_SUITE = "base-a-test-1-gpu-small"
RESULT_FIELDS = [
    "file", "cuda_suite", "board", "n_devices", "auto_status", "triage_status",
    "run_id", "attempt", "plan_attempt", "sha", "image_digest", "ppu_supports_fp8",
    "exit_code", "wall_s", "tests", "passed", "failed", "errors", "skipped",
    "skipped_nodes", "skip_reasons", "log_path", "junit_path",
    "failure_signature", "reason", "reason_code", "evidence", "supported_board", "mode", "issue", "owner",
]


def target_registrations(registrations):
    return [dict(r) for r in registrations if r.get("backend") == "CUDA"
            and not r.get("disabled")
            and str(r.get("nightly_flag", False)).lower() in ("false", "0", "")
            and r.get("cadence", cadence_of(r.get("suite"))) in SWEEP_CADENCES]


def _integer(value, name, minimum=1, maximum=256):
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _repo_file(value):
    path = PurePosixPath(str(value))
    if (path.is_absolute() or ".." in path.parts or "\\" in str(value)
            or not str(path).startswith("test/registered/")
            or path.suffix != ".py" or str(path) != str(value)):
        raise ValueError(f"Invalid registered file: {value!r}")
    return str(path)


def _resource(row, board="auto"):
    if board not in ("auto", "810E", "890P"):
        raise ValueError("board must be auto, 810E or 890P")
    cards = _integer(row.get("n_devices"), "n_devices")
    try:
        mem = float(row["gpu_mem_gb"])
        est = float(row["est_time_s"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Unknown resource/estimate for {row.get('suite')}") from None
    if not math.isfinite(mem) or mem <= 0 or not math.isfinite(est) or est < 0:
        raise ValueError(f"Invalid resource/estimate for {row.get('suite')}")
    # Phase 1 always targets 890P; 810E is explicit calibration/phase 2 only.
    selected = "890P" if board == "auto" else board
    if selected == "810E" and mem > 96:
        raise ValueError("810E is smaller than the required mapping; use 890P or auto")
    return selected, cards, est


def build_model_inventory(registrations, repo_root=None, overrides_path=None):
    """Static, recall-favoring file model sets; empty means unknown, not model-free."""
    repo_root = Path(repo_root or REPO)
    extractor = load_module("sweep_stage_models", Path(__file__).parents[1] / "list_stage_models.py")
    default = repo_root / "scripts/ci/stage_models_overrides.json"
    overrides = extractor.load_overrides(overrides_path or (str(default) if default.exists() else None))
    deny = set(overrides["deny"])
    global_table, errors = extractor.build_global_constant_table(str(repo_root), deny)

    def comma_models(value):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return {v.strip() for v in value.value.split(",")
                    if extractor.looks_like_model_id(v.strip(), deny)}
        if isinstance(value, (ast.Tuple, ast.List)):
            return set().union(*(comma_models(v) for v in value.elts))
        return set()

    def constants(source):
        result = extractor.extract_constants_from_source(source, deny)
        for stmt in ast.parse(source).body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        result.setdefault(target.id, set()).update(comma_models(stmt.value))
        return result

    # Supplement the shared extractor's table only for comma-list constants.
    for path in sorted((repo_root / "python/sglang/test").rglob("*.py")):
        try:
            for name, models in constants(path.read_text()).items():
                global_table.setdefault(name, set()).update(models)
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors[str(path.relative_to(repo_root))] = str(exc)
    by_file = collections.defaultdict(set)
    for row in target_registrations(registrations):
        by_file[_repo_file(row["file"])].add(row["suite"])
    file_models, file_datasets, unknowns = {}, {}, {}
    for file, suites in sorted(by_file.items()):
        models, datasets = set(), set()
        try:
            source = (repo_root / file).read_text()
            tree = ast.parse(source)
            table = dict(global_table)
            table.update(constants(source))
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in global_table:
                            table[alias.asname or alias.name] = global_table[alias.name]
                        if (node.module or "").startswith("sglang.test"):
                            modules.add(alias.asname or alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("sglang.test"):
                            modules.add(alias.asname or alias.name)
            models.update(extractor.extract_models_from_source(source, table, deny))
            dataset_helpers = {alias.asname or alias.name for node in ast.walk(tree)
                               if isinstance(node, ast.ImportFrom) and node.module == "sglang.benchmark.utils"
                               for alias in node.names if alias.name == "download_and_cache_hf_file"}
            dataset_nodes, dataset_names = set(), set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                keywords = {k.arg: k.value for k in node.keywords}
                dataset_helper = ast.unparse(node.func) in dataset_helpers
                repo_type = keywords.get("repo_type", node.args[2] if dataset_helper and len(node.args) > 2 else None)
                is_dataset = isinstance(repo_type, ast.Constant) and repo_type.value == "dataset"
                is_dataset |= repo_type is None and dataset_helper
                value = keywords.get("repo_id", node.args[0] if node.args else None)
                if is_dataset and value is not None:
                    dataset_nodes.update(id(n) for n in ast.walk(value))
                    datasets.update(comma_models(value))
                    if isinstance(value, ast.Name):
                        dataset_names.add(value.id)
                        datasets.update(table.get(value.id, set()))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if node.value and all(isinstance(t, ast.Name) and t.id in dataset_names for t in targets):
                        dataset_nodes.update(id(n) for n in ast.walk(node.value))
            fragments = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                         for part in node.values if isinstance(part, ast.Constant)}
            fragments.update(id(node.value) for node in ast.walk(tree)
                             if isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Constant))
            other_refs, formatted = set(), set()
            for node in ast.walk(tree):
                found = set()
                if isinstance(node, ast.Constant):
                    if id(node) in fragments:
                        formatted.update(comma_models(node))
                    else:
                        found.update(comma_models(node))
                        models.update(found)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    found.update(table.get(node.id, set()))
                elif isinstance(node, ast.Attribute):
                    dotted = ast.unparse(node.value)
                    if dotted in modules or dotted.startswith("sglang.test."):
                        found.update(global_table.get(node.attr, set()))
                        models.update(found)
                elif isinstance(node, ast.JoinedStr) and any(
                    isinstance(v, ast.Constant) and isinstance(v.value, str) and "/" in v.value
                    for v in node.values
                ):
                    unknowns[file] = "dynamic string reference; static extraction may be incomplete"
                if id(node) not in dataset_nodes:
                    other_refs.update(found)
            models.difference_update((datasets | formatted) - other_refs)
            if datasets:
                unknowns[file] = "; ".join(filter(None, (unknowns.get(file), "dataset references recorded separately; availability not checked")))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors[file] = f"{type(exc).__name__}: {exc}"
            unknowns[file] = errors[file]
        models.update(overrides["by_file"].get(file, []))
        for suite in suites:
            models.update(overrides["by_suite"].get(suite, []))
        models.difference_update(deny)
        file_models[file] = sorted(models)
        if datasets:
            file_datasets[file] = sorted(datasets - deny)
        if not models:
            unknowns.setdefault(file, "no statically resolved models; not proof of model-free execution")
    return {"file_models": file_models, "file_datasets": file_datasets, "unresolved_files": unknowns,
            "parse_failures": errors, "all_models": sorted({m for ms in file_models.values() for m in ms}),
            "method": "static literals, referenced helper constants and overrides; datasets separate and unchecked; not load proof"}


def probe_model_cache(model, cache_root):
    """Read-only static default-revision completeness, NOT model load proof.

    Self-contained Python 3.6 stdlib: serialized verbatim for the remote probe.
    Follow NAS checkpoint symlinks, count unique target inodes, read only small
    metadata/ref files, never weights. Directory presence alone is insufficient.
    """
    import json
    import os
    import re
    import stat
    import time

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", model):
        raise ValueError("Invalid model id")
    cache_root = str(cache_root)
    if not os.path.isabs(cache_root) or any(ord(c) < 32 for c in cache_root):
        raise ValueError("cache_root must be an absolute path without control characters")
    # Distinguish a missing/unmounted/unreadable root from absent models.
    with os.scandir(cache_root):
        pass
    row = dict(model=model, exists=False, size_bytes=0, cache_status="incomplete", reason="")
    entry = os.path.join(cache_root, "models--" + model.replace("/", "--"))
    if not os.path.lexists(entry):
        row.update(cache_status="missing", reason="model cache entry absent")
        return row
    try:
        ref = os.path.join(entry, "refs", "main")
        if not stat.S_ISREG(os.stat(ref).st_mode) or os.stat(ref).st_size > 512:
            raise ValueError("invalid refs/main")
        with open(ref) as stream:
            revision = stream.read(513).strip()
        if not re.match(r"^[A-Za-z0-9_-]+$", revision):
            raise ValueError("invalid refs/main revision")
        snapshot = os.path.join(entry, "snapshots", revision)
        if not os.path.isdir(snapshot):
            raise ValueError("refs/main snapshot missing or dangling")
        stack, seen_dirs, seen_files, files = [(snapshot, "")], set(), set(), {}
        broken = False
        deadline = time.monotonic() + 15
        count = 0
        while stack:
            directory, prefix = stack.pop()
            info = os.stat(directory)
            inode = (info.st_dev, info.st_ino)
            if inode in seen_dirs:
                continue
            seen_dirs.add(inode)
            with os.scandir(directory) as children:
                for child in children:
                    count += 1
                    if count > 100000 or time.monotonic() > deadline:
                        raise ValueError("snapshot metadata traversal limit exceeded")
                    relative = prefix + child.name
                    try:
                        info = child.stat(follow_symlinks=True)
                    except OSError:
                        broken = True
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        stack.append((child.path, relative + "/"))
                    elif stat.S_ISREG(info.st_mode):
                        files[relative] = (child.path, info.st_size)
                        inode = (info.st_dev, info.st_ino)
                        if inode not in seen_files:
                            row["size_bytes"] += info.st_size
                            seen_files.add(inode)
                    else:
                        broken = True
        def payload(name):
            return name in files and files[name][1] > 0
        def metadata(name):
            if not payload(name) or files[name][1] > 2 * 1024 * 1024:
                return None
            with open(files[name][0]) as stream:
                return json.load(stream)
        if broken or any(name.endswith((".incomplete", ".lock")) for name in files):
            raise ValueError("dangling, partial or non-regular snapshot payload")
        weights = [name for name in files if name.endswith((".safetensors", ".bin", ".pt", ".pth"))
                   and not name.startswith(("optimizer", "training_args"))]
        weight_ok = bool(weights) and all(payload(name) for name in weights)
        for name in weights:
            match = re.search(r"-(\d+)-of-(\d+)(\.[^.]+)$", name)
            if match:
                total = int(match.group(2))
                if total > 10000:
                    raise ValueError("invalid shard count")
                for number in range(1, total + 1):
                    sibling = name[:match.start()] + "-" + str(number).zfill(len(match.group(1))) + "-of-" + match.group(2) + match.group(3)
                    weight_ok = weight_ok and payload(sibling)
        indexes = [name for name in files if name.endswith((".safetensors.index.json", ".bin.index.json"))]
        for name in indexes:
            index = metadata(name)
            mapping = index.get("weight_map") if isinstance(index, dict) else None
            weight_ok = weight_ok and isinstance(mapping, dict) and bool(mapping) and all(payload(v) for v in mapping.values())
        tokenizer = metadata("tokenizer.json") if payload("tokenizer.json") else None
        tokenizer_ok = (isinstance(tokenizer, dict) and isinstance(tokenizer.get("model"), dict)
                        and bool(tokenizer["model"].get("vocab"))) or payload("tokenizer.model") or payload("spiece.model")
        tokenizer_ok = tokenizer_ok or (isinstance(metadata("vocab.json"), dict) and payload("merges.txt"))
        adapter_weights = {name for name in weights
                           if re.match(r"^adapter_model(?:-\d+-of-\d+)?\.(?:safetensors|bin)$", name)}
        base_weights = set(weights) - adapter_weights
        config_ok = bool(base_weights) and isinstance(metadata("config.json"), dict)
        adapter = metadata("adapter_config.json")
        adapter_ok = (bool(adapter_weights) and isinstance(adapter, dict)
                      and all(isinstance(adapter.get(k), str) and adapter[k].strip()
                              for k in ("peft_type", "base_model_name_or_path")))
        complete_weights = (weight_ok and (not base_weights or config_ok)
                            and (not (adapter_weights or "adapter_config.json" in files) or adapter_ok))
        # A partially cached model must not pass merely because its tokenizer exists.
        if (weights or indexes or "adapter_config.json" in files) and not complete_weights:
            raise ValueError("incomplete weights/config or shard set")
        if not (complete_weights or tokenizer_ok):
            raise ValueError("no complete config+weights or tokenizer payload")
        payload_kind = "+".join(kind for kind, valid in (("config+weights", config_ok), ("adapter config+weights", adapter_ok)) if valid) if complete_weights else "tokenizer payload"
        row.update(exists=True, cache_status="present", reason="static completeness only: " + payload_kind + "; not load proof")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        row["reason"] = str(exc)[:300]
    return row


def probe_model_caches(models, cache_root, cache_ssh=None):
    """One bounded, read-only remote Python process; no remote packages/files."""
    cache_root = str(cache_root)
    if not os.path.isabs(cache_root) or any(ord(c) < 32 for c in cache_root):
        raise ValueError("cache_root must be an absolute path without control characters")
    models = sorted(set(models))
    if cache_ssh is None:
        # Validate even an empty inventory's mount.
        with os.scandir(cache_root):
            pass
        return [probe_model_cache(model, cache_root) for model in models]
    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*", cache_ssh):
        raise ValueError("Invalid SSH hostname (optional user@hostname only)")
    source = inspect.getsource(probe_model_cache)
    payload = json.dumps({"models": models, "root": cache_root})
    source += ("\nimport json, os\np = json.loads(" + repr(payload) + ")\n"
               "with os.scandir(p['root']):\n    pass\n"
               "print(json.dumps([probe_model_cache(m, p['root']) for m in p['models']]))\n")
    process = subprocess.run(
        ["ssh", "-oBatchMode=yes", "-oConnectTimeout=15", "--", cache_ssh, "python3", "-"],
        input=source, capture_output=True, text=True, timeout=180,
    )
    if process.returncode:
        raise ValueError("Remote cache probe failed: " + process.stderr[-1000:])
    rows = json.loads(process.stdout)
    if not isinstance(rows, list) or sorted(r["model"] for r in rows) != models:
        raise ValueError("Invalid remote cache response")
    return rows


def model_availability(inventory, cache_root, cache_ssh=None):
    references = collections.defaultdict(list)
    for file, models in inventory["file_models"].items():
        for model in models:
            references[model].append(file)
    rows = []
    blocked = collections.defaultdict(list)
    for result in probe_model_caches(references, cache_root, cache_ssh):
        files = sorted(references[result["model"]])
        rows.append(dict(result, referencing_file_count=len(files), referencing_files=files))
        if result["cache_status"] != "present":
            for file in files:
                blocked[file].append(f"{result['model']}: {result['cache_status']} ({result['reason']})")
    return rows, {file: "; ".join(reasons) for file, reasons in blocked.items()}


def _suite_groups(registrations):
    groups = collections.defaultdict(list)
    for row in target_registrations(registrations):
        groups[row["suite"]].append(row)
    return groups


def _suite_resources(rows, board, timeout):
    resources = [_resource(row, board) for row in rows]
    if len({(b, n) for b, n, _ in resources}) != 1:
        raise ValueError(f"Inconsistent resources for {rows[0]['suite']}")
    b, cards, _ = resources[0]
    limit = timeout or int(math.ceil(min(max(3 * max(t for _, _, t in resources), 600), 3600)))
    return b, cards, limit


def build_sweep_suites(registrations, blocked_files=None):
    """Offline R1 proposal for ALL target suites, including unmappable resources."""
    blocked_files = blocked_files or {}
    groups = _suite_groups(registrations)
    occurrences = collections.Counter(r["file"] for rows in groups.values() for r in rows)
    result = []
    for suite, rows in sorted(groups.items()):
        row = dict(suite=suite, cadence=rows[0]["cadence"], file_count=len(rows),
                   runner_label=rows[0].get("runner_label"), gpu_model=rows[0].get("gpu_model"),
                   gpu_mem_gb=rows[0].get("gpu_mem_gb"), board="", n_devices="", ppu_supports_fp8="",
                   timeout_per_file="", proposed_partitions="", serial_est_s="", device_est_s="",
                   blocked_files=sum(r["file"] in blocked_files for r in rows), reason="")
        try:
            b, cards, limit = _suite_resources(rows, "auto", 0)
            # Singleton proposals preserve LPT and isolate any cache/dedup decisions.
            singleton = any(r["file"] in blocked_files or occurrences[r["file"]] > 1 for r in rows)
            max_files = max(1, (360 - 40) * 60 // limit)
            proposed = len(rows) if singleton else max(1, math.ceil(len(rows) / max_files))
            estimate = sum(float(r["est_time_s"]) for r in rows)
            row.update(board=b, n_devices=cards, ppu_supports_fp8=PPU_FP8[b], timeout_per_file=limit,
                       proposed_partitions=proposed, serial_est_s=estimate, device_est_s=cards * estimate,
                       reason="singleton isolation recommended for blocked/shared files" if singleton else "")
        except ValueError as exc:
            row["reason"] = str(exc)
        result.append(row)
    return result


def build_sweep_plan(registrations, suites=DEFAULT_SWEEP_SUITE, partition_size=1,
                     board="auto", timeout_per_file=0, max_parallel=24, sha=None,
                     blocked_files=None):
    """Frozen original-registry LPT only; never filter a mixed shard at execution."""
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("ref must be a full 40-character lowercase SHA")
    size = _integer(partition_size, "partition_size", maximum=100000)
    cap = _integer(max_parallel, "max_parallel", maximum=256)
    timeout = _integer(timeout_per_file, "timeout_per_file", minimum=0, maximum=3600)
    if board not in ("auto", "810E", "890P"):
        raise ValueError("board must be auto, 810E or 890P")
    names = [s.strip() for s in suites.split(",")] if isinstance(suites, str) else list(suites)
    groups = _suite_groups(registrations)
    if not names or any(not s or s not in groups for s in names):
        raise ValueError("Empty, unknown or out-of-scope suite; only enabled per_commit/extra/nightly CUDA suites are supported")
    blocked_files = blocked_files or {}
    plan = dict(sha=sha, image_digest=os.environ.get("PPU_BASE_IMAGE_DIGEST", ""),
                run_id=os.environ.get("GITHUB_RUN_ID", ""),
                attempt=_integer(os.environ.get("GITHUB_RUN_ATTEMPT", "1"), "attempt", maximum=100000),
                expected=[], blocked=[], matrix={"include": []}, max_jobs=0)
    seen, maximum_cards = set(), 0
    for suite in dict.fromkeys(names):
        rows = groups[suite]
        if size > len(rows):
            raise ValueError(f"partition_size exceeds file_count={len(rows)} for {suite}")
        if len({r["file"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate registrations within suite {suite}")
        b, cards, limit = _suite_resources(rows, board, timeout)
        if cards > cap:
            raise ValueError(f"{suite} requests {cards} cards, above max_parallel={cap}")
        maximum_cards = max(maximum_cards, cards)
        expected = {}
        for row in rows:
            f = _repo_file(row["file"])
            item = dict(row, cuda_suite=suite, board=b, n_devices=cards,
                        ppu_supports_fp8=PPU_FP8[b], timeout_per_file=limit)
            expected[f] = item
            plan["expected"].append(item)
            if f in blocked_files:
                plan["blocked"].append(dict(item, auto_status="INFRA_MODEL", reason=blocked_files[f]))
        originals = [SimpleNamespace(filename=r["file"], est_time=float(r["est_time_s"])) for r in rows]
        for rank in range(size):
            files = [r.filename for r in ci_register.auto_partition(originals, rank, size)]
            blocked = [f in blocked_files for f in files]
            duplicates = [(f, b, cards, limit) in seen for f in files]
            if any(blocked) and not all(blocked):
                raise ValueError(f"Mixed ready/blocked shard in {suite}; set partition_size=file_count ({len(rows)}) for singleton isolation")
            if all(blocked):
                continue
            if any(duplicates) and not all(duplicates):
                raise ValueError(f"Mixed new/duplicate shard in {suite}; set partition_size=file_count ({len(rows)}) for singleton isolation")
            if all(duplicates):
                continue
            minutes = math.ceil(len(files) * limit / 60) + 30  # queue + setup margin
            if minutes + 10 > 360:
                raise ValueError(f"{suite} shard exceeds GitHub 6h job limit; increase partition_size (up to file_count={len(rows)})")
            key = f"s{len(plan['matrix']['include']):03d}-{b}-{cards}c-p{rank}"
            plan["matrix"]["include"].append(dict(
                key=key, suite=suite, board=b, n_devices=cards, ppu_supports_fp8=PPU_FP8[b],
                node_selector="OAM-810E" if b == "810E" else "ZW-M890P",
                partition_id=rank, partition_size=size, timeout_per_file=limit,
                timeout_minutes=minutes, job_minutes=minutes + 10, files=files,
            ))
            seen.update((f, b, cards, limit) for f in files)
    if len(plan["matrix"]["include"]) > 256:
        raise ValueError("Worker matrix exceeds 256 jobs; request fewer suites in this dispatch")
    plan["max_jobs"] = cap // maximum_cards
    return plan


def _artifact(base, value):
    """Artifact paths may neither escape nor traverse symlinks (even internal)."""
    value = str(value)
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"Unsafe artifact path: {value!r}")
    path = Path(base)
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink artifact: {path}")
    return path


def _read_artifact(path, limit=16 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Invalid or oversized artifact: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Oversized artifact: {path}")
    return data


def _json_artifact(path):
    try:
        return json.loads(_read_artifact(path))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON artifact {path}: {exc}") from exc


def _identity(row):
    return (row["file"], row.get("cuda_suite", row.get("suite")), row.get("board"),
            _integer(row.get("n_devices"), "n_devices"))


def _topology(row):
    file, _, board, cards = _identity(row)
    return file, board, cards


def _valid_fp8(row):
    value = row.get("ppu_supports_fp8")
    return type(value) is int and row.get("board") in PPU_FP8 and value == PPU_FP8[row["board"]]


def _triage(row):
    status = row["auto_status"]
    if row.get("mode") == "legacy" or status == "ZERO_COLLECTED":
        return "RUNNER_INCOMPATIBLE"
    if status == "INFRA_MODEL":
        return "INFRA_BLOCKED"
    if status in ("FAIL", "TIMEOUT", "CRASH", "OOM"):
        # NVFP4 support is unconfirmed. Concrete failed-test evidence is a gap,
        # never an automatically approved NVIDIA dtype/capability exclusion.
        if "nvfp4" in (str(row.get("file", "")) + str(row.get("failure_signature", ""))).lower():
            return "GAP"
        if status == "OOM":
            try:
                if float(row.get("gpu_mem_gb") or 0) > 144:
                    return "NEEDS_TOPOLOGY"
            except (ValueError, TypeError):
                pass
        return "GAP"
    return status


def _empty_result(item, metadata=None):
    row = dict.fromkeys(RESULT_FIELDS, "")
    row.update(item)
    row.update({k: (metadata or {}).get(k, item.get(k, ""))
                for k in ("run_id", "attempt", "plan_attempt", "sha", "image_digest")})
    if metadata is not None:
        row["ppu_supports_fp8"] = metadata.get("ppu_supports_fp8", item.get("ppu_supports_fp8"))
    row.update(cuda_suite=item.get("cuda_suite", item.get("suite")), auto_status="NOT_EXECUTED",
               mode=(metadata or {}).get("mode", "pytest"), tests=0, passed=0, failed=0, errors=0,
               skipped=0, skipped_nodes=[], skip_reasons=[], issue="", owner="")
    row["triage_status"] = _triage(row)
    return row


def _record_result(item, metadata, record, directory):
    row = _empty_result(item, metadata)
    code = record.get("returncode")
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise ValueError("Per-file returncode must be an integer or absent")
    row.update(exit_code=code, wall_s=record.get("elapsed", ""),
               ppu_supports_fp8=metadata.get("ppu_supports_fp8"))
    error = str(record.get("error", ""))
    junit = _artifact(directory, record["junit"]) if record.get("junit") else None
    log_path = _artifact(directory, record["log"]) if record.get("log") else None
    row.update(junit_path=str(junit) if junit else "", log_path=str(log_path) if log_path else "")
    log = _read_artifact(log_path, 64 * 1024 * 1024).decode("utf-8", "replace") if log_path and log_path.exists() else None
    valid_xml, failures, case_ids = False, [], []
    if junit and junit.exists():
        xml = _read_artifact(junit)
        if re.search(br"<!\s*(?:DOCTYPE|ENTITY)", xml, re.I) or b"\x00" in xml:
            raise ValueError("DTD/entity or non-UTF8 JUnit is forbidden")
        try:
            tree = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise ValueError(f"Malformed JUnit: {junit}: {exc}") from exc
        if tree.tag.split("}")[-1] not in ("testsuites", "testsuite"):
            raise ValueError(f"Invalid JUnit root: {tree.tag}")
        valid_xml = True
        for case in tree.iter():
            if case.tag.split("}")[-1] != "testcase":
                continue
            row["tests"] += 1
            case_ids.append(".".join((case.get("classname", ""), case.get("name", ""))))
            children = {kind: [c for c in case if c.tag.split("}")[-1] == kind]
                        for kind in ("failure", "error", "skipped")}
            row["failed"] += bool(children["failure"])
            row["errors"] += bool(children["error"])
            row["skipped"] += bool(children["skipped"])
            if not any(children.values()):
                row["passed"] += 1
            for child in list(case):
                kind = child.tag.split("}")[-1]
                if kind in ("failure", "error"):
                    failures.append((child.get("type") or kind) + ": " + " ".join(filter(None, (child.get("message"), child.text))))
                elif kind == "skipped":
                    # Missing classname/name is incomplete evidence, not a nodeid.
                    node = ""
                    if case.get("classname") and case.get("name"):
                        node = case.get("classname") + "." + case.get("name")
                    row["skipped_nodes"].append(node)
                    row["skip_reasons"].append(child.get("message") or child.text or "")
    counts = dict(collected=row["tests"], executed=row["tests"] - row["skipped"],
                  skipped=row["skipped"], failures=row["failed"] + row["errors"])
    skip_policy_error = bool(row["skipped"]) and error == (
        f"Invalid test results: {counts}; unexpected skips: {row['skipped_nodes']}"
    )
    execution_error = "" if skip_policy_error else error
    row["failure_signature"] = re.sub(r"\s+", " ", failures[0] if failures else execution_error)[:200]
    abnormal = code not in (None, 0) or bool(failures) or bool(execution_error)
    diagnostic = "\n".join(failures + [execution_error, log or ""])
    positive_lines = "\n".join(line for line in diagnostic.splitlines()
                               if not re.search(r"\b(?:not|no)\s+(?:an?\s+)?(?:\w+\s+)?out of memory", line, re.I))
    if row["mode"] == "legacy":
        status = "RUNNER_INCOMPATIBLE"
    elif re.search(r"timed out|suite budget exhausted|TimeoutExpired", error, re.I) or code == 124:
        status = "TIMEOUT"
    elif abnormal and re.search(r"LocalEntryNotFoundError|OfflineModeIsEnabled|Cannot find.*(?:cached|cache)|not found in.*cache", diagnostic, re.I):
        status = "INFRA_MODEL"
    elif abnormal and re.search(r"OutOfMemoryError|(?:CUDA|PPU|HIP) out of memory|\bout of memory\b", positive_lines, re.I):
        status = "OOM"
    elif row["failed"] or row["errors"]:
        status = "FAIL"
    elif code == 5 or (valid_xml and not row["passed"]):
        status = "ZERO_COLLECTED"
    elif code == 0 and valid_xml and row["passed"] and log is not None and not execution_error:
        status = "PASS_WITH_SKIPS" if row["skipped"] else "PASS"
    elif not valid_xml and code not in (None, 0):
        status = "CRASH"
    else:
        status = "FAIL"  # Missing code/log/XML or unexplained runner error is not success.
    row["reason"] = error or ("missing log" if log is None else "")
    if not _valid_fp8(row):
        row["reason"] = "; ".join(filter(None, [row["reason"], "missing or board-inconsistent runtime ppu_supports_fp8"]))
        if status in ("PASS", "PASS_WITH_SKIPS"):
            status = "FAIL"
    if status in ("FAIL", "TIMEOUT", "CRASH", "OOM") and "nvfp4" in "\n".join([row["file"], diagnostic, *case_ids]).lower():
        row["failure_signature"] = "NVFP4: " + row["failure_signature"]
    row["auto_status"] = status
    row["triage_status"] = _triage(row)
    return row


def _artifact_inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Sweep result root must be an existing non-symlink directory")
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Symlink artifact: {path}")
        paths.extend(Path(directory) / f for f in files if f in ("sweep_plan.json", "sweep.json"))
    return sorted(paths)


def _sweep_plans(paths):
    plans = [_json_artifact(p) for p in paths if p.name == "sweep_plan.json"]
    for plan in plans:
        if (not isinstance(plan, dict) or not isinstance(plan.get("expected"), list)
                or any(not isinstance(item, dict) or not {"file", "cuda_suite", "board", "n_devices"} <= item.keys()
                       for item in plan["expected"])):
            raise ValueError("Malformed sweep plan expected rows")
    return plans


def parse_sweep_results(root):
    """Parse every attempt; planned but unstarted and preblocked files survive."""
    paths = _artifact_inventory(root)
    plans = _sweep_plans(paths)
    expected = {}
    for plan in plans:
        identities = set()
        for item in plan.get("expected", []):
            _repo_file(item["file"])
            identity = _identity(item)
            if identity in identities:
                raise ValueError("Duplicate expected tuple in plan")
            if not _valid_fp8(item):
                raise ValueError("Missing or board-inconsistent ppu_supports_fp8 in plan")
            identities.add(identity)
            expected[identity] = item
        if any(_identity(item) not in identities for item in plan.get("blocked", [])):
            raise ValueError("Preblocked file not in expected plan")
    def same_producer(row, plan):
        return (str(row.get("plan_attempt") or row.get("attempt") or 1) == str(plan.get("attempt") or 1)
                and all(not plan.get(k) or str(row.get(k, "")) == str(plan[k])
                        for k in ("run_id", "sha", "image_digest")))
    rows, seen, provenance = [], set(), set()
    for path in paths:
        if path.name != "sweep.json":
            continue
        meta = _json_artifact(path)
        if (not isinstance(meta, dict) or not {"suite", "board", "n_devices", "files"} <= meta.keys()
                or not isinstance(meta["files"], list)):
            raise ValueError("Malformed worker metadata or expected file list")
        provenance.add((meta.get("sha", ""), meta.get("image_digest", "")))
        files = [_repo_file(f) for f in meta["files"]]
        if len(files) != len(set(files)):
            raise ValueError("Duplicate expected files in worker")
        if plans and "plan_attempt" not in meta:
            raise ValueError("Worker must record its producer plan_attempt")
        meta["attempt"] = _integer(meta.get("attempt", 1), "attempt", maximum=100000)
        meta["plan_attempt"] = _integer(meta.get("plan_attempt", meta["attempt"]), "plan_attempt", maximum=100000)
        if meta["plan_attempt"] > meta["attempt"]:
            raise ValueError("Worker attempt predates producer plan")
        matching_plans = [p for p in plans if same_producer(meta, p)]
        if plans and not matching_plans:
            raise ValueError("Worker run/producer plan/SHA/digest differs from plan")
        planned_jobs = [j for p in matching_plans for j in p.get("matrix", {}).get("include", [])]
        if plans:
            matches = [j for j in planned_jobs if j.get("key") == meta.get("key")]
            if not matches or not all(j["files"] == files and j["suite"] == meta["suite"]
                       and j["board"] == meta["board"] and int(j["n_devices"]) == int(meta["n_devices"])
                       and _valid_fp8(meta) and j.get("ppu_supports_fp8") == meta["ppu_supports_fp8"]
                       and all(k not in j or meta.get(k) == j[k]
                               for k in ("partition_id", "partition_size", "timeout_per_file", "node_selector"))
                       for j in matches):
                raise ValueError("Worker selection/resource/FP8/timeout differs from frozen plan")
        blocked_identities = {_identity(item) for p in matching_plans for item in p.get("blocked", [])}
        allowed = {_identity(item): item for p in matching_plans for item in p.get("expected", [])}
        items = {}
        for f in files:
            item = dict(file=f, cuda_suite=meta["suite"], board=meta["board"], n_devices=meta["n_devices"],
                        ppu_supports_fp8=meta.get("ppu_supports_fp8"))
            identity = _identity(item)
            if plans and identity not in allowed:
                raise ValueError("Unexpected worker file/suite/topology")
            if plans and (not _valid_fp8(meta) or meta["ppu_supports_fp8"] != allowed[identity]["ppu_supports_fp8"]):
                raise ValueError("Worker ppu_supports_fp8 differs from frozen plan")
            if identity in blocked_identities:
                raise ValueError("Worker executed a model-blocked file")
            items[f] = allowed.get(identity, item)
        directory = path.parent / "junit"
        summary_path = directory / "summary.json"
        summary = _json_artifact(summary_path) if summary_path.exists() else {"files": []}
        if (not isinstance(summary, dict) or not isinstance(summary.get("selected", []), list)
                or not isinstance(summary.get("files", []), list)):
            raise ValueError("Worker summary must contain selected/files lists")
        def normalize(value):
            raw = PurePosixPath(value)
            if raw.is_absolute():
                try:
                    raw = raw.relative_to(PurePosixPath(meta["source_root"]))
                except (KeyError, ValueError):
                    raise ValueError("Record outside worker source_root") from None
            file = _repo_file(str(raw))
            if file not in items:
                raise ValueError(f"Unexpected worker record: {file}")
            return file
        selected = [normalize(f) for f in summary.get("selected", [])]
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate worker selection")
        records = {}
        for record in summary.get("files", []):
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise ValueError("Per-file record requires a file path")
            f = normalize(record["file"])
            if f in records:
                raise ValueError(f"Duplicate per-file record: {f}")
            records[f] = record
        for f, item in items.items():
            key = (_identity(item), str(meta.get("run_id", "")), meta["attempt"], meta["plan_attempt"])
            if key in seen:
                raise ValueError(f"Duplicate worker evidence: {key}")
            seen.add(key)
            rows.append(_record_result(item, meta, records[f], directory) if f in records and f in selected
                        else _empty_result(item, meta))
    if len(provenance) > 1:
        raise ValueError("Mixed SHA/image digest evidence; report frozen runs separately")
    frozen = dict(zip(("sha", "image_digest"), next(iter(provenance), ("", ""))))
    for key in ("sha", "image_digest"):
        values = {p[key] for p in plans if p.get(key)}
        if frozen[key]:
            values.add(frozen[key])
        if len(values) > 1:
            raise ValueError("Mixed SHA/image digest in plans and evidence")
        if values:
            frozen[key] = next(iter(values))
    for plan in plans:
        blocked = {_identity(item): item for item in plan.get("blocked", [])}
        for item in plan.get("expected", []):
            identity = _identity(item)
            attempt_rows = [r for r in rows if same_producer(r, plan)]
            if any(_identity(r) == identity for r in attempt_rows):
                continue
            if identity not in blocked and any(_topology(r) == _topology(item)
                    and r.get("ppu_supports_fp8") == item.get("ppu_supports_fp8")
                    and r.get("timeout_per_file") == item.get("timeout_per_file") for r in attempt_rows):
                continue  # Shared explicitly by parity, not fabricated per-suite executions.
            row = _empty_result(item, dict(plan, plan_attempt=plan.get("attempt", 1), **frozen))
            if identity in blocked:
                row.update(auto_status="INFRA_MODEL", reason=blocked[identity].get("reason", "model cache unavailable"))
                row["triage_status"] = _triage(row)
            rows.append(row)
    return rows


def build_parity_manifest(expected, results, approved_skips=None, today=None):
    """Join same-topology evidence only; preserve raw status and attempt history."""
    today = today or date.today()
    provenance = {(r.get("sha", ""), r.get("image_digest", "")) for r in results}
    if len(provenance) > 1:
        raise ValueError("Mixed SHA/image digest evidence; not comparable attempts")
    if approved_skips:
        with open(approved_skips, newline="") as stream:
            approvals = list(csv.DictReader(stream))
    else:
        approvals = []
    identities = {_identity(e) for e in expected}
    groups, seen = collections.defaultdict(list), set()
    for result in results:
        if _identity(result) not in identities:
            raise ValueError(f"Unexpected result: {_identity(result)}")
        key = (_identity(result), str(result.get("run_id", "")), str(result.get("attempt", "")),
               str(result.get("plan_attempt") or result.get("attempt") or 1))
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        groups[_topology(result)].append(result)
    def order(row):
        return (int(row.get("run_id") or 0), int(row.get("plan_attempt") or row.get("attempt") or 0),
                int(row.get("attempt") or 0))
    def approved(item, node):
        # Exact classname.name only: no wildcard, file, module or pytest nodeid.
        if (not isinstance(node, str) or "::" in node
                or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\[[^\]\r\n*?]+\])?", node)
                or node.endswith(".py")):
            return None
        for a in approvals:
            try:
                valid_date = date.fromisoformat(a.get("expires_at", "")) >= today
            except ValueError:
                continue
            reason = a.get("reason_code")
            if reason not in {"NV_DTYPE", "NV_ARCH", "NV_LIB", "NV_INTERCONNECT", "NV_TOOLING", "HW_CAPABILITY", "BOARD_CAPABILITY"}:
                continue
            if reason == "NV_DTYPE" and "nvfp4" in " ".join(str(item.get(k, "")) for k in
                    ("file", "failure_signature", "skip_reasons", "skipped_nodes")).lower() + str(a.get("evidence", "")).lower():
                continue
            if reason == "BOARD_CAPABILITY" and not (
                    a.get("board") == item.get("board") == "810E" and a.get("supported_board") == "890P"
                    and re.search(r"\b890P\b", a.get("evidence", ""))):
                continue
            suites = a.get("cuda_suites", "").strip()
            if (a.get("file") == item["file"] and a.get("junit_id") == node
                    and (not suites or item["cuda_suite"] in re.split(r"[;,]\s*", suites))
                    and valid_date and all(a.get(k, "").strip() for k in ("evidence", "reviewer"))):
                return a
        return None
    histories = {}
    for item in expected:
        candidates = groups.get(_topology(item), [])
        producers = collections.defaultdict(list)
        for r in candidates:
            if (_identity(r) == _identity(item) or (_valid_fp8(r)
                    and r.get("ppu_supports_fp8") == item.get("ppu_supports_fp8")
                    and r.get("timeout_per_file") == item.get("timeout_per_file"))):
                producers[order(r)[:2]].append(r)
        history = []
        for producer in sorted(producers):
            attempts = producers[producer]
            exact = [r for r in attempts if _identity(r) == _identity(item)]
            history.extend(sorted(exact or attempts, key=order))
        histories[_identity(item)] = history
    manifest = []
    for item in expected:
        history = histories[_identity(item)]
        row = _empty_result(item)
        if history:
            row.update(history[-1])
            # Keep observed runtime/provenance; only registration annotations and
            # the destination suite belong to the expected row.
            row.update({k: v for k, v in item.items() if k not in RESULT_FIELDS and k != "timeout_per_file"})
            row.update(file=item["file"], cuda_suite=item["cuda_suite"], board=item["board"], n_devices=item["n_devices"])
            if item.get("sha") and item["sha"] != history[-1].get("sha"):
                raise ValueError("Expected SHA differs from evidence")
            if item.get("image_digest") and item["image_digest"] != history[-1].get("image_digest"):
                raise ValueError("Expected image digest differs from evidence")
            if row["auto_status"] in ("PASS", "PASS_WITH_SKIPS") and (not _valid_fp8(row)
                    or row.get("ppu_supports_fp8") != item.get("ppu_supports_fp8")):
                row.update(auto_status="FAIL", reason="missing or inconsistent runtime ppu_supports_fp8")
        row["evidence_shared_with"] = sorted({e["cuda_suite"] for e in expected
                if e["cuda_suite"] != item["cuda_suite"] and histories[_identity(e)]
                and histories[_identity(e)][-1] is history[-1]}) if history and history[-1]["auto_status"] != "NOT_EXECUTED" else []
        row["attempt_history"] = [{k: r.get(k, "") for k in ("cuda_suite", "run_id", "attempt", "plan_attempt", "auto_status", "sha", "image_digest", "ppu_supports_fp8")} for r in history]
        outcomes = collections.defaultdict(set)
        for r in history:
            outcomes[_identity(r)].add(r["auto_status"])
        row["flaky"] = any(len(statuses) > 1 for statuses in outcomes.values())
        if row.get("mode") == "legacy":
            row["auto_status"] = "RUNNER_INCOMPATIBLE"
        row["triage_status"] = _triage(row)
        if row["auto_status"] == "PASS_WITH_SKIPS":
            nodes = row.get("skipped_nodes", [])
            if isinstance(nodes, list) and nodes and len(nodes) == int(row.get("skipped", 0)):
                reviews = [approved(row, n) for n in nodes]
                if all(reviews):
                    row["triage_status"] = "PASS"
                    for key in ("reason_code", "evidence", "supported_board"):
                        row[key] = "; ".join(sorted({a[key] for a in reviews if a.get(key)}))
        row.update(issue="", owner="")
        manifest.append(row)
    return manifest


def _report_csv(path, rows, fields):
    """P1 reports always have headers, including empty evidence/cache inventories."""
    keys = list(dict.fromkeys(fields + [k for row in rows for k in row]))
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                          for k, v in row.items()} for row in rows)


def _report_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _report_parity(out_dir, evidence, registrations, approved_skips):
    # Reusing an output directory must never retain successful stale evidence.
    for name in ("sweep_results.csv", "parity_manifest.csv"):
        _report_csv(out_dir / name, [], RESULT_FIELDS)
    rows = parse_sweep_results(evidence)
    _report_csv(out_dir / "sweep_results.csv", rows, RESULT_FIELDS)
    plans = _sweep_plans(_artifact_inventory(evidence))
    planned = {_identity(item): item for plan in plans for item in plan.get("expected", [])}
    if not plans:
        planned = {_identity(r): r for r in rows}
    file_meta = {r["file"]: r for r in _read_csv(out_dir / "cuda_files.csv")}
    expected = []
    for r in target_registrations(registrations):
        item = dict(file_meta.get(r["file"], {}), **r, cuda_suite=r["suite"])
        matches = [p for p in planned.values() if (p["file"], p["cuda_suite"]) == (r["file"], r["suite"])]
        if matches:
            for p in matches:
                expected.append(dict(item, board=p["board"], n_devices=p["n_devices"],
                                     ppu_supports_fp8=p.get("ppu_supports_fp8"),
                                     timeout_per_file=p.get("timeout_per_file")))
        else:
            try:
                board, cards, _ = _resource(r)
            except ValueError:
                # Visible unknown topology, never silently inferred as one card.
                board, cards = "UNKNOWN", 0
            expected.append(dict(item, board=board, n_devices=cards, ppu_supports_fp8=PPU_FP8.get(board)))
    # Sharing applies only inside requested scope, not all inventory suites.
    # Unknown topology remains visibly NOT_EXECUTED, never an inferred one card.
    requested = [r for r in expected if r["n_devices"] and _identity(r) in planned]
    joined = {_identity(r): r for r in build_parity_manifest(requested, rows, approved_skips)}
    manifest = [joined.get(_identity(r), _empty_result(r)) if r["n_devices"] else _empty_result(r)
                for r in expected]
    _report_csv(out_dir / "parity_manifest.csv", manifest, RESULT_FIELDS + ["evidence_shared_with", "attempt_history", "flaky"])
    return int(not planned or any(r["triage_status"] != "PASS" for r in manifest
                                  if r["n_devices"] and _identity(r) in planned))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-root", help="Read-only HF hub cache root (no downloads)")
    parser.add_argument("--cache-ssh", help="Probe with one read-only SSH Python 3.6 process")
    parser.add_argument("--sweep-plan", action="store_true")
    parser.add_argument("--suites", default=DEFAULT_SWEEP_SUITE)
    parser.add_argument("--partition-size", default="1")
    parser.add_argument("--board", default="auto")
    parser.add_argument("--timeout-per-file", default="0")
    parser.add_argument("--max-parallel", default="24", help="Total card cap, not number of jobs")
    parser.add_argument("--ref", help="Full lowercase SHA; must match checkout")
    parser.add_argument("--sweep-results", help="Downloaded sweep artifact tree")
    parser.add_argument("--approved-skips", help="Human-reviewed exact JUnit skip approval CSV")
    args = parser.parse_args(argv)
    if args.cache_ssh and not args.cache_root:
        parser.error("--cache-ssh requires --cache-root")
    if args.approved_skips and not args.sweep_results:
        parser.error("--approved-skips requires --sweep-results")
    out = Path(args.out_dir)
    try:
        if args.sweep_plan:
            if (out / "sweep_plan.json").exists():
                (out / "sweep_plan.json").unlink()
            sha = git_head()[0]
            if args.ref is not None and args.ref != sha:
                raise ValueError("--ref must match the current checkout SHA")
        s = build(args.out_dir)
        registrations = _read_csv(out / "registrations.csv")
        inventory = build_model_inventory(registrations)
        _report_json(out / "model_inventory.json", inventory)
        blocked = {}
        if args.cache_root:
            availability, blocked = model_availability(inventory, args.cache_root, args.cache_ssh)
            _report_csv(out / "model_availability.csv", availability,
                        ["model", "referencing_file_count", "exists", "size_bytes", "cache_status", "referencing_files", "reason"])
            _report_csv(out / "model_blocked_files.csv",
                        [dict(file=f, auto_status="INFRA_MODEL", reason=reason) for f, reason in sorted(blocked.items())],
                        ["file", "auto_status", "reason"])
        suites = build_sweep_suites(registrations, blocked)
        _report_csv(out / "sweep_suites.csv", suites,
                    ["suite", "cadence", "file_count", "board", "n_devices", "timeout_per_file", "proposed_partitions", "reason"])
        totals = collections.defaultdict(lambda: dict(suites=0, files=0, serial_est_s=0, device_est_s=0))
        for row in suites:
            t = totals[row["board"] or "UNKNOWN"]
            t["suites"] += 1
            t["files"] += row["file_count"]
            t["serial_est_s"] += row["serial_est_s"] or 0
            t["device_est_s"] += row["device_est_s"] or 0
        _report_json(out / "sweep_resources.json", dict(totals))
        if args.sweep_plan:
            plan = build_sweep_plan(registrations, args.suites, args.partition_size, args.board,
                                    args.timeout_per_file, args.max_parallel, args.ref or sha, blocked)
            _report_json(out / "sweep_plan.json", plan)
        status = _report_parity(out, args.sweep_results, registrations, args.approved_skips) if args.sweep_results else 0
        for cadence, a in s["cuda"]["by_cadence"].items():
            print(f"CUDA {cadence:10} regs={a['registrations']:4} files={a['files']:4} serial_h={a['serial_h']:7} device_h={a['device_h']:7}")
        print(f"PPU enabled registrations: {s['ppu']['enabled_registrations']}")
        print(f"Wrote {args.out_dir}")
        if (out / "sweep_error.json").exists():
            (out / "sweep_error.json").unlink()
        return status
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"P1 report failed: {exc}", file=sys.stderr)
        if out.is_dir():
            _report_json(out / "sweep_error.json", {"error": str(exc)})
            if args.sweep_results:
                for name in ("sweep_results.csv", "parity_manifest.csv"):
                    _report_csv(out / name, [], RESULT_FIELDS)
        return 2


if __name__ == "__main__":
    sys.exit(main())

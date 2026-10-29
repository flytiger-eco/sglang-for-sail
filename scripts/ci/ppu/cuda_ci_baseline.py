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
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone

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
        while line.endswith("\\"):
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
        bindings = dict(variables)
        lane = None
        if script[1] == "run_pr_suite.sh":
            lanes = literal_values(args[0], variables) if args else set()
            if len(lanes) != 1:
                continue
            lane = next(iter(lanes))
            branch = re.search(
                r"(?m)^\s*" + re.escape(lane) + r"\)\s*hw=(\w+);\s*suite=([\w-]+);",
                text,
            )
            if not branch:
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    s = build(args.out_dir)
    for cadence, a in s["cuda"]["by_cadence"].items():
        print(
            f"CUDA {cadence:10} regs={a['registrations']:4} files={a['files']:4} serial_h={a['serial_h']:7} device_h={a['device_h']:7}"
        )
    print(f"PPU enabled registrations: {s['ppu']['enabled_registrations']}")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()

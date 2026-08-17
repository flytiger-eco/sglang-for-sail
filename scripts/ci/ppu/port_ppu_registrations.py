#!/usr/bin/env python3
"""Port `register_ppu_ci(...)` calls from one sglang checkout to another.

One-shot migration tool. Reads every registration in SOURCE and replays it into
the same repo-relative path in TARGET, adding the import if needed. Writes a
report of what could not be matched so the gaps are visible rather than silent.

Usage:
    python3 port_ppu_registrations.py --source <src_repo> --target <dst_repo> [--apply]

Without --apply it is a dry run: nothing is written, the report is still printed.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CI_REGISTER_MODULE = "sglang.test.ci.ci_register"
PPU_FUNC = "register_ppu_ci"
# black's default; used to decide single-line vs parenthesized import.
LINE_LENGTH = 88

SCAN_GLOBS = (
    "test/registered/**/*.py",
    "python/sglang/jit_kernel/tests/**/test_*.py",
    "python/sglang/jit_kernel/benchmark/**/bench_*.py",
)


@dataclass
class Report:
    ported: List[str] = field(default_factory=list)
    ported_moved: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    missing_in_target: List[str] = field(default_factory=list)
    ambiguous: List[str] = field(default_factory=list)
    no_anchor: List[str] = field(default_factory=list)
    parse_error: List[str] = field(default_factory=list)
    new_in_target: List[str] = field(default_factory=list)


def resolve_moved(rel: str, by_basename: Dict[str, List[str]]) -> Optional[str]:
    """Locate a file that kept its name but changed directory.

    The target reorganized `test/registered/`, so an exact-path miss is usually a
    move, not a deletion. Only accept an unambiguous single candidate: guessing
    between e.g. the generic and the `amd/` variant would silently register the
    wrong file. Vendor-specific and non-registered locations are filtered out
    first, since a PPU registration never belongs in either.
    """
    candidates = by_basename.get(os.path.basename(rel), [])
    candidates = [
        c
        for c in candidates
        if c.startswith("test/registered/")
        and "/amd/" not in c
        and "/musa/" not in c
        and "/xpu/" not in c
        and "/ascend/" not in c
        and "/cpu/" not in c
    ]
    return candidates[0] if len(candidates) == 1 else None


def scan(repo: str) -> List[str]:
    """Repo-relative paths of candidate test files."""
    out: List[str] = []
    for pattern in SCAN_GLOBS:
        for path in glob.glob(os.path.join(repo, pattern), recursive=True):
            rel = os.path.relpath(path, repo)
            base = os.path.basename(rel)
            if base in ("conftest.py", "__init__.py"):
                continue
            out.append(rel)
    return sorted(set(out))


def find_ppu_call(src: str, tree: ast.AST) -> Optional[str]:
    """Exact source text of the first module-level register_ppu_ci call."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == PPU_FUNC
        ):
            return ast.get_source_segment(src, node)
    return None


def find_register_anchor(tree: ast.AST) -> Optional[int]:
    """1-based end line of the last module-level `register_*_ci(...)` statement.

    Anchoring on the *statement* (not the Call) keeps multi-line calls intact.
    """
    last = None
    for stmt in getattr(tree, "body", []):
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id.startswith("register_")
            and call.func.id.endswith("_ci")
        ):
            last = stmt.end_lineno
    return last


def find_ci_import(src: str, tree: ast.AST):
    """The ImportFrom node for ci_register, plus its names, or None."""
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == CI_REGISTER_MODULE:
            return stmt, [a.name for a in stmt.names]
    return None, None


def render_import(names: List[str]) -> str:
    """Emit an import in black's style: one line if it fits, else parenthesized."""
    names = sorted(set(names))
    single = f"from {CI_REGISTER_MODULE} import " + ", ".join(names)
    if len(single) <= LINE_LENGTH:
        return single
    body = "".join(f"    {n},\n" for n in names)
    return f"from {CI_REGISTER_MODULE} import (\n{body})"


def port_one(target_path: str, ppu_call: str) -> str:
    """Return the new file text with the PPU registration + import added.

    Raises LookupError if the file has no `register_*_ci` call to anchor against
    — inserting at a guessed position risks landing above an import it needs — or
    if it has no plain `from ... import` to extend. A handful of files bind the
    registrar through a try/except stub or importlib instead; there we cannot add
    the name, and writing the call anyway yields a NameError at import that takes
    down every suite the file belongs to, not just PPU.
    """
    src = open(target_path, encoding="utf-8").read()
    tree = ast.parse(src)

    anchor = find_register_anchor(tree)
    if anchor is None:
        raise LookupError("no register_*_ci anchor")

    imp, names = find_ci_import(src, tree)
    if imp is None:
        raise LookupError("no plain ci_register import to extend")
    lines = src.splitlines(keepends=True)

    # Insert the call first, then rewrite the import. Doing it in this order means
    # the import's line numbers are still valid when we splice it.
    call_text = ppu_call if ppu_call.endswith("\n") else ppu_call + "\n"
    lines.insert(anchor, call_text)

    if PPU_FUNC not in names:
        new_import = render_import(names + [PPU_FUNC])
        # imp line numbers are 1-based and unaffected: the call went in below them.
        start, end = imp.lineno - 1, imp.end_lineno
        lines[start:end] = [new_import + "\n"]

    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="repo to read registrations from")
    ap.add_argument("--target", required=True, help="repo to write registrations into")
    ap.add_argument(
        "--apply", action="store_true", help="write changes (default: dry run)"
    )
    ap.add_argument("--report", default=None, help="write the report to this path")
    args = ap.parse_args()

    source, target = os.path.abspath(args.source), os.path.abspath(args.target)
    for repo in (source, target):
        if not os.path.isdir(repo):
            print(f"error: not a directory: {repo}", file=sys.stderr)
            return 2

    rep = Report()

    # Collect every PPU registration in source, keyed by repo-relative path.
    registrations: Dict[str, str] = {}
    for rel in scan(source):
        path = os.path.join(source, rel)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            rep.parse_error.append(f"{rel} (source)")
            continue
        call = find_ppu_call(src, tree)
        if call is not None:
            registrations[rel] = call

    target_files = set(scan(target))
    target_by_basename: Dict[str, List[str]] = {}
    for rel in sorted(target_files):
        target_by_basename.setdefault(os.path.basename(rel), []).append(rel)

    # rel_in_target may differ from rel when the file moved between versions.
    resolved: Dict[str, str] = {}
    for rel in registrations:
        if rel in target_files:
            resolved[rel] = rel
            continue
        moved = resolve_moved(rel, target_by_basename)
        if moved is not None:
            resolved[rel] = moved

    for rel, call in sorted(registrations.items()):
        target_rel = resolved.get(rel)
        if target_rel is None:
            same_name = target_by_basename.get(os.path.basename(rel), [])
            if len(same_name) > 1:
                rep.ambiguous.append(f"{rel} -> {same_name}")
            else:
                rep.missing_in_target.append(rel)
            continue
        if target_rel != rel:
            rep.ported_moved.append(f"{rel} -> {target_rel}")
        path = os.path.join(target, target_rel)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            rep.parse_error.append(f"{rel} (target)")
            continue
        if find_ppu_call(src, tree) is not None:
            rep.already_present.append(rel)
            continue
        try:
            new_text = port_one(path, call)
        except LookupError:
            rep.no_anchor.append(rel)
            continue
        # Never write something we cannot parse back.
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            rep.parse_error.append(f"{rel} (post-edit: {exc})")
            continue
        if args.apply:
            open(path, "w", encoding="utf-8").write(new_text)
        rep.ported.append(rel)

    # Target files that carry CI registrations but got no PPU one from source: new
    # in this version, so a human has to pick suite/est_time. Subtract the *resolved*
    # target paths, not the source keys — otherwise every moved file lands here too.
    covered = set(resolved.values())
    for rel in sorted(target_files - covered):
        path = os.path.join(target, rel)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        if find_register_anchor(tree) is not None:
            rep.new_in_target.append(rel)

    mode = "APPLIED" if args.apply else "DRY RUN (no files written)"
    out = [f"=== PPU registration port — {mode} ===", ""]
    out.append(f"source registrations found  : {len(registrations)}")
    out.append(f"ported                      : {len(rep.ported)}")
    out.append(f"  of which path moved       : {len(rep.ported_moved)}")
    out.append(f"already present             : {len(rep.already_present)}")
    out.append(f"MISSING in target           : {len(rep.missing_in_target)}")
    out.append(f"ambiguous (needs manual)    : {len(rep.ambiguous)}")
    out.append(f"no anchor (needs manual)    : {len(rep.no_anchor)}")
    out.append(f"parse errors                : {len(rep.parse_error)}")
    out.append(f"NEW in target (needs triage): {len(rep.new_in_target)}")

    for title, items in (
        ("Ported to a moved path", rep.ported_moved),
        ("MISSING in target (registration dropped)", rep.missing_in_target),
        ("Ambiguous basename (pick one manually)", rep.ambiguous),
        ("No register_*_ci anchor (insert manually)", rep.no_anchor),
        ("Parse errors", rep.parse_error),
        ("New in target (decide suite/est_time)", rep.new_in_target),
    ):
        if items:
            out += ["", f"--- {title} ({len(items)}) ---"] + [f"  {i}" for i in items]

    text = "\n".join(out)
    print(text)
    if args.report:
        open(args.report, "w", encoding="utf-8").write(text + "\n")
        print(f"\nreport written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Port PPU `@skip_if_*` guards from one sglang checkout to another.

Companion to port_ppu_registrations.py. Registering a test is only half the job:
without these guards a test whose model is not on NAS fails instead of skipping,
so the whole nightly reads red. This replays the guards onto the same class or
function name in the target.

Matching is by (file, class/function name). Anything that does not match exactly
is reported, never guessed — silently guarding the wrong class would drop coverage
without any signal.

Usage:
    python3 port_ppu_skip_guards.py --source <src> --target <dst> [--apply]
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_ppu_registrations import LINE_LENGTH, resolve_moved, scan

SKIP_MODULE = "sglang.test.ci.ppu_skip_utils"
SKIP_NAMES = (
    "skip_if_model_missing",
    "skip_if_no_fp8",
    "skip_if_no_fp4",
    "model_exists",
    "is_ppu_platform",
)


@dataclass
class Report:
    applied: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    file_missing: List[str] = field(default_factory=list)
    name_missing: List[str] = field(default_factory=list)
    unresolved_symbol: List[str] = field(default_factory=list)
    parse_error: List[str] = field(default_factory=list)


def decorator_free_symbols(deco_src: str) -> set:
    """Bare names a decorator expression depends on, minus the guard itself.

    Guards are often parameterised by a constant rather than a literal, e.g.
    `@skip_if_model_missing(DEFAULT_MODEL_NAME_FOR_TEST)`. Copying that text into a
    file that does not import the constant is a NameError at class-definition time,
    which takes down every suite the file belongs to. Callers use this to check
    resolvability before writing.
    """
    try:
        expr = ast.parse(deco_src, mode="eval")
    except SyntaxError:
        return set()
    return {
        n.id
        for n in ast.walk(expr)
        if isinstance(n, ast.Name) and n.id not in SKIP_NAMES
    }


def module_bound_names(tree: ast.AST) -> set:
    """Names a module defines or imports at top level (plus builtins)."""
    names = set(dir(__builtins__)) if isinstance(__builtins__, type) else set()
    import builtins

    names |= set(dir(builtins))
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Try):
            for sub in stmt.body + [s for h in stmt.handlers for s in h.body]:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
                elif isinstance(
                    sub, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
    return names


def decorator_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node, ast.Name):
        return node.id
    return None


def collect_guards(src: str, tree: ast.AST) -> Dict[str, List[str]]:
    """{class_or_func_name: [decorator source, ...]} for PPU skip guards."""
    out: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if decorator_name(deco) in SKIP_NAMES:
                seg = ast.get_source_segment(src, deco)
                if seg is not None:
                    out.setdefault(node.name, []).append(seg)
    return out


def find_defs(tree: ast.AST) -> Dict[str, ast.AST]:
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def existing_guards(node: ast.AST) -> set:
    return {
        decorator_name(d)
        for d in node.decorator_list
        if decorator_name(d) in SKIP_NAMES
    }


def find_skip_import(src: str, tree: ast.AST) -> Tuple[Optional[ast.AST], List[str]]:
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == SKIP_MODULE:
            return stmt, [a.name for a in stmt.names]
    return None, []


def render_import(names: List[str]) -> str:
    names = sorted(set(names))
    single = f"from {SKIP_MODULE} import " + ", ".join(names)
    if len(single) <= LINE_LENGTH:
        return single
    body = "".join(f"    {n},\n" for n in names)
    return f"from {SKIP_MODULE} import (\n{body})"


def last_import_line(tree: ast.AST) -> int:
    """1-based end line of the final top-level import, for placing a new one."""
    last = 0
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            last = max(last, stmt.end_lineno)
    return last


def port_file(
    target_path: str, guards: Dict[str, List[str]]
) -> Tuple[Optional[str], List[str], List[str], List[str]]:
    """Add guards to target_path.

    Returns (new_text|None, applied, name_missing, unresolved).
    """
    src = open(target_path, encoding="utf-8").read()
    tree = ast.parse(src)
    defs = find_defs(tree)
    bound = module_bound_names(tree)

    applied: List[str] = []
    missing: List[str] = []
    unresolved: List[str] = []
    needed_names: set = set()
    # (line_to_insert_before, indent, text) — collected then applied bottom-up so
    # earlier insertions do not shift the line numbers of later ones.
    edits: List[Tuple[int, int, str]] = []

    for name, decos in sorted(guards.items()):
        node = defs.get(name)
        if node is None:
            missing.append(name)
            continue
        have = existing_guards(node)
        for deco in decos:
            fn = deco.split("(")[0].strip()
            if fn in have:
                continue
            absent = decorator_free_symbols(deco) - bound
            if absent:
                unresolved.append(f"{name}: {deco} needs {sorted(absent)}")
                continue
            needed_names.add(fn)
            # node.lineno is the `class`/`def` keyword line, below any decorators,
            # so this lands the guard innermost. Skip decorators are order-independent.
            edits.append((node.lineno, node.col_offset, f"@{deco}"))
            applied.append(name)

    if not edits:
        return None, [], missing, unresolved

    lines = src.splitlines(keepends=True)
    for lineno, indent, text in sorted(edits, key=lambda e: -e[0]):
        lines.insert(lineno - 1, " " * indent + text + "\n")

    # Import last: every edit above was below the import block, so recomputing
    # positions from the original tree is still valid.
    imp, have_names = find_skip_import(src, tree)
    if imp is not None:
        new_import = render_import(have_names + sorted(needed_names))
        # Line numbers shift by the number of insertions made *above* the import;
        # all guard insertions are below it, so no adjustment is needed.
        lines[imp.lineno - 1 : imp.end_lineno] = [new_import + "\n"]
    else:
        anchor = last_import_line(tree)
        lines.insert(anchor, render_import(sorted(needed_names)) + "\n")

    return "".join(lines), applied, missing, unresolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    source, target = os.path.abspath(args.source), os.path.abspath(args.target)
    rep = Report()

    target_files = set(scan(target))
    by_base: Dict[str, List[str]] = {}
    for rel in sorted(target_files):
        by_base.setdefault(os.path.basename(rel), []).append(rel)

    for rel in scan(source):
        path = os.path.join(source, rel)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        guards = collect_guards(src, tree)
        if not guards:
            continue

        trel = rel if rel in target_files else resolve_moved(rel, by_base)
        if trel is None:
            rep.file_missing += [f"{rel}::{n}" for n in guards]
            continue

        tpath = os.path.join(target, trel)
        try:
            new_text, applied, missing, unresolved = port_file(tpath, guards)
        except (OSError, SyntaxError) as exc:
            rep.parse_error.append(f"{trel}: {exc}")
            continue

        rep.name_missing += [f"{trel}::{n}" for n in missing]
        rep.unresolved_symbol += [f"{trel}::{u}" for u in unresolved]
        if new_text is None:
            rep.already_present += [f"{trel}::{n}" for n in guards if n not in missing]
            continue
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            rep.parse_error.append(f"{trel} (post-edit): {exc}")
            continue
        if args.apply:
            open(tpath, "w", encoding="utf-8").write(new_text)
        rep.applied += [f"{trel}::{n}" for n in applied]

    mode = "APPLIED" if args.apply else "DRY RUN (no files written)"
    out = [f"=== PPU skip-guard port — {mode} ===", ""]
    out.append(f"guards applied            : {len(rep.applied)}")
    out.append(f"already present           : {len(rep.already_present)}")
    out.append(f"file missing in target    : {len(rep.file_missing)}")
    out.append(f"class/func name not found : {len(rep.name_missing)}")
    out.append(f"unresolved symbol         : {len(rep.unresolved_symbol)}")
    out.append(f"parse errors              : {len(rep.parse_error)}")
    for title, items in (
        ("Class/func name not found (add manually)", rep.name_missing),
        ("Unresolved symbol — import it, then re-run", rep.unresolved_symbol),
        ("File missing in target", rep.file_missing),
        ("Parse errors", rep.parse_error),
    ):
        if items:
            out += ["", f"--- {title} ({len(items)}) ---"] + [f"  {i}" for i in items]

    text = "\n".join(out)
    print(text)
    if args.report:
        open(args.report, "w", encoding="utf-8").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

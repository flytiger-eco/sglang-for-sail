"""静态审计 ``@plugin_hook`` 与目标 callable 的签名兼容性。

用法：
    python audit_hook_signatures.py [hook_file ...]

默认扫描 ``python/**/*.py`` 中的全部 ``@plugin_hook``。脚本只解析 AST，不导入
sglang、torch 或硬件扩展。发现签名不兼容、动态 target 无法求值、目标不存在或
源码无法解析时返回非零退出码。
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

HOOK_TYPES = {"BEFORE", "AFTER", "AROUND", "REPLACE"}
SPECIAL_PREFIX = {"AFTER": "result", "AROUND": "original_fn"}


@dataclass(frozen=True)
class HookDecl:
    path: Path
    line: int
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    target: str
    hook_type: str


@dataclass(frozen=True)
class Finding:
    status: str
    hook: HookDecl
    detail: str

    @property
    def is_error(self) -> bool:
        return self.status in {"ERROR", "MISMATCH"}


@dataclass(frozen=True)
class ResolvedTarget:
    path: Path
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class SourceResolver:
    """Resolve dotted targets to AST nodes without importing their modules."""

    def __init__(self, source_roots: Sequence[Path]):
        self.source_roots = tuple(root.resolve() for root in source_roots)
        self._trees: dict[Path, ast.Module] = {}

    def _module_file(self, module: str) -> Path | None:
        rel = Path(*module.split("."))
        for root in self.source_roots:
            for candidate in (
                root / rel.with_suffix(".py"),
                root / rel / "__init__.py",
            ):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    def _tree(self, path: Path) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        return self._trees[path]

    def resolve(self, target: str) -> ResolvedTarget:
        parts = target.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:split])
            path = self._module_file(module)
            if path is None:
                continue
            return self._resolve_members(path, parts[split:])
        raise LookupError(f"找不到 target 模块源码: {target}")

    @staticmethod
    def _named_node(
        body: Sequence[ast.stmt], name: str
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
        return next(
            (
                item
                for item in body
                if isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and item.name == name
            ),
            None,
        )

    def _import_bindings(self, path: Path) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for node in self._tree(path).body:
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    if alias.name != "*":
                        bindings[alias.asname or alias.name] = (
                            f"{node.module}.{alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        return bindings

    def _base_target(self, path: Path, base: ast.expr) -> str | None:
        bindings = self._import_bindings(path)
        if isinstance(base, ast.Name):
            return bindings.get(base.id)
        if isinstance(base, ast.Attribute):
            parts = []
            current: ast.AST = base
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name):
                return None
            root = bindings.get(current.id, current.id)
            return ".".join([root, *reversed(parts)])
        return None

    def _resolve_inherited_member(
        self,
        path: Path,
        class_node: ast.ClassDef,
        member: str,
        visited: set[tuple[Path, str, str]],
    ) -> ResolvedTarget | None:
        key = (path, class_node.name, member)
        if key in visited:
            return None
        visited.add(key)
        tree = self._tree(path)
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                local_base = self._named_node(tree.body, base.id)
                if isinstance(local_base, ast.ClassDef):
                    direct = self._named_node(local_base.body, member)
                    if direct is not None:
                        return ResolvedTarget(path, direct)
                    inherited = self._resolve_inherited_member(
                        path, local_base, member, visited
                    )
                    if inherited is not None:
                        return inherited
            target = self._base_target(path, base)
            if target is None:
                continue
            try:
                return self.resolve(f"{target}.{member}")
            except LookupError:
                continue
        return None

    def _resolve_members(self, path: Path, members: Sequence[str]) -> ResolvedTarget:
        body: Sequence[ast.stmt] = self._tree(path).body
        current: ast.AST | None = None
        for index, name in enumerate(members):
            current = self._named_node(body, name)
            if current is None:
                imported = self._import_bindings(path).get(name) if index == 0 else None
                if imported is not None:
                    suffix = ".".join(members[index + 1 :])
                    return self.resolve(f"{imported}.{suffix}" if suffix else imported)
                raise LookupError(
                    f"{path} 中找不到 target 成员: {'.'.join(members[: index + 1])}"
                )
            if index < len(members) - 1:
                if not isinstance(current, ast.ClassDef):
                    raise LookupError(
                        f"target 中间成员不是类: {'.'.join(members[: index + 1])}"
                    )
                next_name = members[index + 1]
                if self._named_node(current.body, next_name) is None:
                    inherited = self._resolve_inherited_member(
                        path, current, next_name, set()
                    )
                    if inherited is None or index + 1 != len(members) - 1:
                        raise LookupError(
                            f"{path} 中找不到 target 成员: "
                            f"{'.'.join(members[: index + 2])}"
                        )
                    return inherited
                body = current.body
        assert isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
        return ResolvedTarget(path, current)


def _eval_string(node: ast.AST, constants: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_string(node.left, constants) + _eval_string(node.right, constants)
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                pieces.append(_eval_string(value.value, constants))
            else:
                raise ValueError("f-string 含非静态表达式")
        return "".join(pieces)
    raise ValueError("不是静态字符串表达式")


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                pending.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                pending.append((node.target.id, node.value))

    while pending:
        remaining = []
        changed = False
        for name, value in pending:
            try:
                constants[name] = _eval_string(value, constants)
                changed = True
            except ValueError:
                remaining.append((name, value))
        if not changed:
            break
        pending = remaining
    return constants


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _hook_type(call: ast.Call) -> str:
    type_expr = next((kw.value for kw in call.keywords if kw.arg == "type"), None)
    if type_expr is None:
        return "AFTER"
    if isinstance(type_expr, ast.Attribute):
        value = type_expr.attr
    elif isinstance(type_expr, ast.Name):
        value = type_expr.id
    elif isinstance(type_expr, ast.Constant) and isinstance(type_expr.value, str):
        value = type_expr.value.upper()
    else:
        raise ValueError("HookType 不是静态枚举值")
    if value not in HOOK_TYPES:
        raise ValueError(f"未知 HookType: {value}")
    return value


def collect_hooks(path: Path) -> tuple[list[HookDecl], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [], [f"{path}: AST 解析失败: {exc}"]

    constants = _module_constants(tree)
    hooks: list[HookDecl] = []
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _decorator_name(decorator.func) != "plugin_hook":
                continue
            if not decorator.args:
                errors.append(f"{path}:{node.lineno}: @plugin_hook 缺少 target")
                continue
            try:
                target = _eval_string(decorator.args[0], constants)
                hook_type = _hook_type(decorator)
            except ValueError as exc:
                errors.append(f"{path}:{node.lineno}: {exc}")
                continue
            hooks.append(
                HookDecl(path, node.lineno, node.name, node, target, hook_type)
            )
    return hooks, errors


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> inspect.Signature:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    default_offset = len(positional) - len(args.defaults)
    parameters: list[inspect.Parameter] = []
    for index, arg in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = inspect.Parameter.empty
        if index >= default_offset:
            default = object()
        parameters.append(inspect.Parameter(arg.arg, kind, default=default))
    if args.vararg is not None:
        parameters.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        default = inspect.Parameter.empty if default_node is None else object()
        parameters.append(
            inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=default)
        )
    if args.kwarg is not None:
        parameters.append(
            inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    return inspect.Signature(parameters)


def _strip_special_prefix(
    signature: inspect.Signature, hook_type: str
) -> inspect.Signature:
    expected = SPECIAL_PREFIX.get(hook_type)
    if expected is None:
        return signature
    parameters = list(signature.parameters.values())
    if not parameters:
        raise ValueError(f"{hook_type} Hook 缺少首参数 {expected}")
    first = parameters[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(f"{hook_type} Hook 首参数必须是位置参数 {expected}")
    if first.name != expected:
        raise ValueError(
            f"{hook_type} Hook 首参数应命名为 {expected}，实际为 {first.name}"
        )
    return inspect.Signature(parameters[1:])


def _representative_calls(
    signature: inspect.Signature,
) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    params = list(signature.parameters.values())
    positional_only = [p for p in params if p.kind == p.POSITIONAL_ONLY]
    positional_or_keyword = [p for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
    keyword_only = [p for p in params if p.kind == p.KEYWORD_ONLY]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    min_args = tuple(
        object()
        for p in positional_only + positional_or_keyword
        if p.default is inspect.Parameter.empty
    )
    min_kwargs = {
        p.name: object() for p in keyword_only if p.default is inspect.Parameter.empty
    }
    calls.append(("最小实参", min_args, min_kwargs))

    all_pos_args = tuple(object() for _ in positional_only + positional_or_keyword)
    all_kw = {p.name: object() for p in keyword_only}
    if has_varargs:
        all_pos_args += (object(),)
    if has_varkw:
        all_kw["__audit_extra_kw"] = object()
    calls.append(("全部位置实参", all_pos_args, all_kw))

    for split in range(len(positional_or_keyword) + 1):
        args = tuple(object() for _ in positional_only + positional_or_keyword[:split])
        kwargs = {
            p.name: object() for p in positional_or_keyword[split:] + keyword_only
        }
        calls.append((f"位置/关键字分界 {split}", args, kwargs))

    unique = []
    seen = set()
    for label, args, kwargs in calls:
        key = (len(args), tuple(kwargs))
        if key not in seen:
            seen.add(key)
            unique.append((label, args, kwargs))
    return unique


def _is_universal_variadic(
    node: ast.FunctionDef | ast.AsyncFunctionDef, hook_type: str
) -> bool:
    try:
        signature = _strip_special_prefix(_signature(node), hook_type)
    except ValueError:
        return False
    params = list(signature.parameters.values())
    return (
        len(params) == 2
        and params[0].kind == inspect.Parameter.VAR_POSITIONAL
        and params[1].kind == inspect.Parameter.VAR_KEYWORD
    )


def compare_signatures(
    target: ast.FunctionDef | ast.AsyncFunctionDef,
    hook: ast.FunctionDef | ast.AsyncFunctionDef,
    hook_type: str,
) -> str | None:
    target_signature = _signature(target)
    try:
        hook_signature = _strip_special_prefix(_signature(hook), hook_type)
    except ValueError as exc:
        return str(exc)

    for label, args, kwargs in _representative_calls(target_signature):
        try:
            hook_signature.bind(*args, **kwargs)
        except TypeError as exc:
            return (
                f"{label} 不兼容: target{target_signature} -> "
                f"hook{hook_signature}: {exc}"
            )
    return None


def audit_hook(hook: HookDecl, resolver: SourceResolver) -> Finding:
    if isinstance(hook.node, ast.ClassDef):
        if hook.hook_type != "REPLACE":
            return Finding("ERROR", hook, "类 Hook 只能使用 REPLACE")
        try:
            target = resolver.resolve(hook.target)
        except (LookupError, OSError, UnicodeError, SyntaxError) as exc:
            return Finding("ERROR", hook, str(exc))
        if not isinstance(target.node, ast.ClassDef):
            return Finding("MISMATCH", hook, "类 Hook 的 target 不是类")
        return Finding("OK", hook, f"类 REPLACE target={target.path}")

    try:
        target = resolver.resolve(hook.target)
    except (LookupError, OSError, UnicodeError, SyntaxError) as exc:
        return Finding("ERROR", hook, str(exc))
    if isinstance(target.node, ast.ClassDef):
        return Finding("MISMATCH", hook, "函数 Hook 的 target 是类")
    if _is_universal_variadic(hook.node, hook.hook_type):
        return Finding(
            "OK",
            hook,
            f"target={target.path}:{target.node.lineno}; "
            "Hook 使用 *args, **kwargs，覆盖任意 target 调用签名",
        )

    mismatch = compare_signatures(target.node, hook.node, hook.hook_type)
    if mismatch is not None:
        return Finding(
            "MISMATCH", hook, f"target={target.path}:{target.node.lineno}; {mismatch}"
        )
    return Finding(
        "OK",
        hook,
        f"target={target.path}:{target.node.lineno} 签名兼容",
    )


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_source_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root / "python" / "sglang" / "kernels" / "aot" / "python",
        repo_root / "python",
    ]


def default_hook_paths(repo_root: Path) -> list[Path]:
    excluded = {"build", "dist", "__pycache__", ".git"}
    paths = []
    for path in (repo_root / "python").rglob("*.py"):
        if excluded.intersection(path.relative_to(repo_root).parts):
            continue
        try:
            if "@plugin_hook" not in path.read_text(encoding="utf-8"):
                continue
        except (OSError, UnicodeError):
            paths.append(path)
            continue
        paths.append(path)
    return sorted(paths)


def run_audit(
    hook_paths: Iterable[Path], source_roots: Sequence[Path]
) -> tuple[list[Finding], list[str]]:
    resolver = SourceResolver(source_roots)
    hooks: list[HookDecl] = []
    errors: list[str] = []
    for path in hook_paths:
        found, parse_errors = collect_hooks(path.resolve())
        hooks.extend(found)
        errors.extend(parse_errors)
    return [audit_hook(hook, resolver) for hook in hooks], errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hook_files",
        nargs="*",
        type=Path,
        help="要扫描的 Python 文件；省略时扫描 python/**/*.py",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root(),
        help="sglang 仓库根目录",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=None,
        help="模块源码根目录；可重复指定",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    source_roots = (
        [path.resolve() for path in args.source_root]
        if args.source_root
        else default_source_roots(repo_root)
    )
    hook_paths = (
        [path.resolve() for path in args.hook_files]
        if args.hook_files
        else default_hook_paths(repo_root)
    )

    findings, parse_errors = run_audit(hook_paths, source_roots)
    for error in parse_errors:
        print(f"[ERROR] {error}")
    for finding in findings:
        hook = finding.hook
        try:
            display_path = hook.path.relative_to(repo_root)
        except ValueError:
            display_path = hook.path
        print(
            f"[{finding.status}] {display_path}:{hook.line} {hook.name} "
            f"({hook.hook_type}) -> {hook.target}"
        )
        print(f"    {finding.detail}")

    failed = len(parse_errors) + sum(finding.is_error for finding in findings)
    print(
        f"\nHOOK_SIGNATURE_AUDIT hooks={len(findings)} "
        f"errors={failed} status={'FAIL' if failed else 'PASS'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

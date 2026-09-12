"""Report the public objects of a package that carry no docstring.

``sphinx-build -b coverage`` measures a different and complementary thing:
whether every object of every imported module reaches a page of the site.
Because ``autodoc_default_options`` sets ``undoc-members`` -- which is what
puts the whole public surface on the API pages in the first place -- an object
with no docstring still gets an entry there and still counts as covered, so the
coverage builder cannot see a docstring gap.  This script is the docstring half
of the same question.

It works from the source alone: it parses the tree with :mod:`ast`, so it needs
no OpenCL stack, no import of the package and no Sphinx, and it can run on a
machine that could not build the documentation.  *Public* means every
module-level class and function, and every method of a public class, whose name
does not begin with an underscore, in a module whose dotted name has no
underscore-prefixed component.  The two halves of a property count once.

Usage::

    python doc/tools/docstring_gaps.py
    python doc/tools/docstring_gaps.py --max-gaps 21 --output gaps.txt

``--max-gaps`` is the ratchet CI applies: it fails when the number of
undocumented public objects rises above the recorded value.  Lower that value
in the same commit that lowers the count.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]

# Kept in step with ``coverage_ignore_modules`` in ``doc/source/conf.py`` and
# with the filter in ``doc/source/_templates/autosummary/module.rst``: a 2019
# finite-element experiment that nothing in the tree imports and that is not
# part of the supported API.
IGNORED_MODULES = frozenset({"volumential.qbfem"})

_FUNCTION_NODES = (ast.AsyncFunctionDef, ast.FunctionDef)


class Gap(NamedTuple):
    """One public object that has no docstring."""

    path: str
    lineno: int
    kind: str
    name: str


def _module_name(path: Path, package_root: Path) -> str:
    """Dotted module name of *path* inside the package rooted at *package_root*."""
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_property_mutator(node: ast.AST) -> bool:
    """Is *node* the ``@x.setter`` or ``@x.deleter`` half of a property?

    Sphinx documents a property once, under its getter, so counting the
    mutators separately would overstate the gap.
    """
    return any(
        isinstance(decorator, ast.Attribute)
        and decorator.attr in {"deleter", "setter"}
        for decorator in node.decorator_list
    )


def _scan_module(path: Path, module: str) -> tuple[list[Gap], int]:
    """Undocumented public objects of one module, and its public-object count."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(REPO_ROOT).as_posix()

    gaps: list[Gap] = []
    total = 0

    def record(node, kind: str, name: str) -> None:
        nonlocal total
        total += 1
        if ast.get_docstring(node) is None:
            gaps.append(Gap(relative, node.lineno, kind, name))

    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)):
            continue
        if not _is_public(node.name):
            continue

        if isinstance(node, ast.ClassDef):
            record(node, "class", f"{module}.{node.name}")
            for member in node.body:
                if not isinstance(member, _FUNCTION_NODES):
                    continue
                if not _is_public(member.name) or _is_property_mutator(member):
                    continue
                record(member, "method", f"{module}.{node.name}.{member.name}")
        else:
            record(node, "function", f"{module}.{node.name}")

    return gaps, total


def collect(package_root: Path) -> tuple[list[Gap], int]:
    """Undocumented public objects of a package, and its public-object count."""
    gaps: list[Gap] = []
    total = 0

    for path in sorted(package_root.rglob("*.py")):
        module = _module_name(path, package_root)
        if module in IGNORED_MODULES:
            continue
        if any(part.startswith("_") for part in module.split(".")):
            continue

        module_gaps, module_total = _scan_module(path, module)
        gaps.extend(module_gaps)
        total += module_total

    return gaps, total


def format_report(gaps: list[Gap], total: int) -> str:
    """Render the gap list as the plain-text report CI keeps as an artifact."""
    title = "Public objects without a docstring"
    lines = [title, "=" * len(title), ""]

    current_path = None
    for gap in gaps:
        if gap.path != current_path:
            if current_path is not None:
                lines.append("")
            lines.append(gap.path)
            current_path = gap.path
        lines.append(f"  {gap.lineno:>6}  {gap.kind:<8}  {gap.name}")

    documented = total - len(gaps)
    share = 100.0 * documented / total if total else 100.0
    lines.extend([
        "",
        f"{len(gaps)} of {total} public objects have no docstring "
        f"({share:.1f}% documented).",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "volumential",
        help="package directory to scan (default: the volumential package)",
    )
    parser.add_argument(
        "--max-gaps",
        type=int,
        default=None,
        help="fail when more public objects than this have no docstring",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="also write the report to this file",
    )
    args = parser.parse_args(argv)

    gaps, total = collect(args.package.resolve())
    report = format_report(gaps, total)
    print(report, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    if args.max_gaps is not None and len(gaps) > args.max_gaps:
        print(
            f"error: {len(gaps)} undocumented public objects, "
            f"more than the {args.max_gaps} recorded in CI; "
            f"document the new ones or raise --max-gaps deliberately.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

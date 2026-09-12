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
underscore-prefixed component.  A definition nested in module-level control
flow -- the ``except ImportError`` fallback for an optional dependency, say --
counts, and so does a method defined inside class-level control flow: both bind
an attribute like any other.  Definitions that bind one name count once, as
the object a reader ends up with: the two halves of a property under its
getter, an ``@overload`` set under its implementation, a rebinding under the
last of them, and a conditional alternative under whichever branch is missing
its documentation.

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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]

# Kept in step with ``coverage_ignore_modules`` in ``doc/source/conf.py`` and
# with the filter in ``doc/source/_templates/autosummary/module.rst``: a 2019
# finite-element experiment that nothing in the tree imports and that is not
# part of the supported API.
# Each entry ignores that module *and its descendants*, so an addition below
# ``volumential.qbfem`` cannot fail the ratchet for a package that the other
# two mechanisms leave out of the documentation entirely.
IGNORED_MODULES = frozenset({"volumential.qbfem"})

_FUNCTION_NODES = (ast.AsyncFunctionDef, ast.FunctionDef)

# Statements a definition can be nested in and still belong to the enclosing
# module or class: an optional-dependency fallback under ``try`` or a version
# check under ``if`` binds a name every importer sees.  ``ast.Match`` is
# handled separately below, because its suites hang off ``cases`` rather than
# off the fields these nodes share.
_CONTROL_FLOW_NODES = (ast.For, ast.If, ast.Try, ast.TryStar, ast.While, ast.With)


class Gap(NamedTuple):
    """One public object that has no docstring."""

    path: str
    lineno: int
    kind: str
    name: str


class Carrier(NamedTuple):
    """The definition that carries one public name's documentation."""

    #: The definition the report points at.
    node: ast.stmt
    #: True when the name is a gap whatever docstring *node* carries.
    always_a_gap: bool
    #: Every definition that binds this name, the carrier included.
    group: list[ast.stmt]


class Descriptor(NamedTuple):
    """A class member created by assigning a descriptor, not by ``def``."""

    #: The assignment statement.
    node: ast.stmt
    #: What the report calls it: ``"property"``, ``"method"``.
    kind: str
    #: Whether the object it creates ends up with a docstring.
    documented: bool


#: Builtins whose result is a class member, and what to call the member.
_DESCRIPTOR_KINDS = {
    "property": "property",
    "classmethod": "method",
    "staticmethod": "method",
}


def _module_name(path: Path, package_root: Path) -> str:
    """Dotted module name of *path* inside the package rooted at *package_root*."""
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _dominates(suite: tuple, other: tuple) -> bool:
    """Does a binding in *suite* overwrite one already made in *other*?

    Only if *other* is nested inside it: a definition in the enclosing suite
    runs whatever the branches did, so it replaces anything they bound.  Two
    branches of one ``if`` are nested in neither, and both survive.
    """
    return other[: len(suite)] == suite


def _bound_name(node: ast.stmt) -> str:
    """The name *node* binds, whether it is a definition or an assignment."""
    if isinstance(node, ast.AnnAssign):
        return node.target.id
    if isinstance(node, ast.Assign):
        return node.targets[0].id
    return node.name


def _is_ignored(module: str) -> bool:
    """Is *module* one of :data:`IGNORED_MODULES`, or inside one?"""
    return any(
        module == ignored or module.startswith(f"{ignored}.")
        for ignored in IGNORED_MODULES
    )


def _definitions(
    body: list[ast.stmt], suite: tuple = ()
) -> Iterator[tuple[ast.stmt, tuple]]:
    """Yield ``(definition, suite)`` for one suite, in source order.

    Control flow is entered, because a class or method defined in an ``except
    ImportError`` fallback or under a version check binds an attribute like any
    other.  A function or class body is *not* entered: what it defines belongs
    to that scope, not to the suite this was called on.

    *suite* identifies the branch a definition sits in, and the caller needs it
    to tell two situations apart: two definitions in the same suite are a
    rebinding, where only the later one survives, while two in different suites
    are alternatives, either of which a reader may end up importing.
    """
    for node in body:
        if isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)):
            yield node, suite
        elif isinstance(node, _CONTROL_FLOW_NODES):
            for field in ("body", "orelse", "finalbody"):
                yield from _definitions(
                    getattr(node, field, []), (*suite, node.lineno, field)
                )
            for index, handler in enumerate(getattr(node, "handlers", [])):
                yield from _definitions(
                    handler.body, (*suite, node.lineno, "handler", index)
                )
        elif isinstance(node, ast.Match):
            for index, case in enumerate(node.cases):
                yield from _definitions(
                    case.body, (*suite, node.lineno, "case", index)
                )


def _overload_names(tree: ast.Module) -> frozenset[str]:
    """The names this module binds to ``typing.overload``.

    ``from typing import overload as _overload`` is as much an overload
    decorator as the plain name, and a module that defines an ``overload`` of
    its own is not one at all, so the decorator is recognised by what was
    imported rather than by how it is spelled.  The attribute form,
    ``@typing.overload``, is matched separately.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "typing",
            "typing_extensions",
        }:
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "overload"
            )
    return frozenset(names)


def _is_typing_overload(node: ast.AST, overload_names: frozenset[str]) -> bool:
    """Is *node* an ``@overload`` stub rather than the runtime definition?

    A stub carries no docstring by convention and is not the object anyone
    imports; the implementation that follows the set is.
    """
    return any(
        (isinstance(decorator, ast.Name) and decorator.id in overload_names)
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
        for decorator in node.decorator_list
    )


def _is_property_mutator(node: ast.AST) -> bool:
    """Is *node* the ``@x.setter`` or ``@x.deleter`` half of a property?"""
    return any(
        isinstance(decorator, ast.Attribute)
        and decorator.attr in {"deleter", "setter"}
        for decorator in node.decorator_list
    )


def _descriptor_documentation(
    call: ast.Call, kind: str, function_docstrings: dict[str, bool]
) -> bool:
    """Will the object *call* creates have a docstring?

    ``property`` takes an explicit ``doc`` and otherwise copies its getter's;
    ``classmethod`` and ``staticmethod`` copy the function they wrap.  A
    ``lambda`` has no docstring to copy, and a getter this module cannot
    resolve to a definition in the same class body counts as undocumented,
    which errs towards reporting a gap rather than hiding one.
    """
    if kind == "property":
        for keyword in call.keywords:
            if keyword.arg == "doc":
                return (
                    isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                    and bool(keyword.value.value.strip())
                )

    wrapped = call.args[0] if call.args else None
    if wrapped is None:
        for keyword in call.keywords:
            if keyword.arg in {"fget", "f"}:
                wrapped = keyword.value
                break
    if isinstance(wrapped, ast.Name):
        return function_docstrings.get(wrapped.id, False)
    return False


def _descriptor_assignments(
    body: Iterable[ast.stmt],
    function_docstrings: dict[str, bool],
    suite: tuple = (),
) -> dict[str, dict[tuple, Descriptor]]:
    """Public members a class body creates by assigning a descriptor.

    ``value = property()``, ``value: property = property(...)``,
    ``public = classmethod(_impl)`` and the ``staticmethod`` form all bind a
    class member that no ``def`` carries the name of, so nothing else in this
    module would see them.  Control flow is traversed as everywhere else, and
    a later binding in the same suite replaces an earlier one.

    A bare string literal following the assignment counts as documentation --
    the attribute-docstring convention autodoc reads -- and so does whatever
    :func:`_descriptor_documentation` finds in the call itself.
    """
    statements = list(body)
    found: dict[str, dict[tuple, Descriptor]] = {}

    def merge(other: dict[str, dict[tuple, Descriptor]]) -> None:
        for name, alternatives in other.items():
            bindings = found.setdefault(name, {})
            for other_suite in list(bindings):
                for new_suite in alternatives:
                    if _dominates(new_suite, other_suite):
                        del bindings[other_suite]
                        break
            bindings.update(alternatives)

    for index, node in enumerate(statements):
        if isinstance(node, _CONTROL_FLOW_NODES):
            for field in ("body", "orelse", "finalbody"):
                merge(
                    _descriptor_assignments(
                        getattr(node, field, []),
                        function_docstrings,
                        (*suite, node.lineno, field),
                    )
                )
            for handler_index, handler in enumerate(getattr(node, "handlers", [])):
                merge(
                    _descriptor_assignments(
                        handler.body,
                        function_docstrings,
                        (*suite, node.lineno, "handler", handler_index),
                    )
                )
            continue
        if isinstance(node, ast.Match):
            for case_index, case in enumerate(node.cases):
                merge(
                    _descriptor_assignments(
                        case.body,
                        function_docstrings,
                        (*suite, node.lineno, "case", case_index),
                    )
                )
            continue

        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not isinstance(target, ast.Name) or not _is_public(target.id):
            continue
        if not isinstance(value, ast.Call):
            continue

        function = value.func
        if isinstance(function, ast.Name):
            builtin = function.id
        elif isinstance(function, ast.Attribute):
            builtin = function.attr
        else:
            continue
        kind = _DESCRIPTOR_KINDS.get(builtin)
        if kind is None:
            continue

        following = statements[index + 1] if index + 1 < len(statements) else None
        documented = (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ) or _descriptor_documentation(value, kind, function_docstrings)

        merge({target.id: {suite: Descriptor(node, kind, documented)}})
    return found


def _carriers(
    nodes: Iterable[tuple[ast.stmt, tuple]],
    overload_names: frozenset[str],
    descriptors: dict[str, dict[tuple, Descriptor]] | None = None,
) -> Iterator[Carrier]:
    """One :class:`Carrier` per public name a suite binds.

    Several definitions can bind one name -- an ``@overload`` set, a branch
    that defines a name two ways, a property's getter and its setter -- and the
    docstring that matters is the one on whichever of them the reader ends up
    with.  Four rules cover that:

    * a property takes its ``__doc__`` from its getter, never from a mutator,
      so a name bound only by mutators (a write-only ``value = property()`` and
      its setter) is a gap however well the setter is written;
    * an ``@overload`` stub loses to the implementation that follows the set;
    * a binding is gone once a later one in the same suite, or in a suite that
      encloses it, replaces it -- see :func:`_dominates`;
    * of the bindings that remain, *any* of them can be the one a reader
      imports, since alternative branches of a version check bind different
      ones on different interpreters, so the name is a gap unless they all
      carry a docstring, and the earliest that does not is what the report
      points at.

    *descriptors* adds the members a class body creates by assignment rather
    than by ``def``, from :func:`_descriptor_assignments`; a ``def`` of the
    same name that is not a property mutator supersedes them by the third rule.
    """
    descriptors = descriptors or {}
    # Keyed by suite, so a rebinding keeps only the last of its suite while the
    # branches of a conditional keep all of theirs.
    groups: dict[str, dict[tuple, ast.stmt]] = {}
    kinds: dict[str, str] = {}
    for name, alternatives in descriptors.items():
        groups[name] = {suite: entry.node for suite, entry in alternatives.items()}
        kinds[name] = next(iter(alternatives.values())).kind

    def bind(name: str, suite: tuple, node: ast.stmt) -> None:
        bucket = groups.setdefault(name, {})
        for bound in list(bucket):
            # ``("mutator", ...)`` keys are not suites and never dominated.
            if isinstance(bound, tuple) and bound and bound[0] == "mutator":
                continue
            if _dominates(suite, bound):
                del bucket[bound]
        bucket[suite] = node

    for node, suite in nodes:
        if not _is_public(node.name):
            continue
        if isinstance(node, _FUNCTION_NODES) and _is_property_mutator(node):
            # A mutator augments the property its getter created rather than
            # rebinding the name, so it must not displace that getter.
            groups.setdefault(node.name, {})[("mutator", suite, node.lineno)] = node
        else:
            bind(node.name, suite, node)

    for name, by_suite in groups.items():
        group = list(by_suite.values())
        surviving_descriptors = {
            suite: entry
            for suite, entry in descriptors.get(name, {}).items()
            if by_suite.get(suite) is entry.node
        }
        definitions = [
            node
            for node in group
            if node not in {entry.node for entry in surviving_descriptors.values()}
        ]
        implementations = [
            node
            for node in definitions
            if not (
                isinstance(node, _FUNCTION_NODES)
                and _is_typing_overload(node, overload_names)
            )
        ]
        candidates = implementations or definitions
        getters = [
            node
            for node in candidates
            if not (
                isinstance(node, _FUNCTION_NODES) and _is_property_mutator(node)
            )
        ]
        alternatives = list(surviving_descriptors.values())

        undocumented = [
            entry.node for entry in alternatives if not entry.documented
        ]
        undocumented += [
            node for node in getters if ast.get_docstring(node) is None
        ]

        if undocumented:
            # Point at whichever branch is missing its documentation.
            first = min(undocumented, key=lambda node: node.lineno)
            yield Carrier(first, True, group)
        elif getters:
            yield Carrier(getters[-1], False, group)
        elif alternatives:
            yield Carrier(alternatives[-1].node, False, group)
        else:
            yield Carrier(candidates[0], True, group)


def _scan_module(
    path: Path, module: str, report_root: Path
) -> tuple[list[Gap], int]:
    """Undocumented public objects of one module, and its public-object count.

    Paths in the report are relative to *report_root*, the directory the
    scanned package sits in, so that ``--package`` works outside this
    repository as well as inside it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(report_root).as_posix()

    overload_names = _overload_names(tree)
    gaps: list[Gap] = []
    total = 0

    def record(node, kind: str, name: str, *, always_a_gap: bool = False) -> None:
        nonlocal total
        total += 1
        if isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)):
            undocumented = always_a_gap or ast.get_docstring(node) is None
        else:
            # A member created by assigning a descriptor carries no docstring
            # node of its own; the carrier decided whether it is documented.
            undocumented = always_a_gap
        if undocumented:
            gaps.append(Gap(relative, node.lineno, kind, name))

    for carrier in _carriers(_definitions(tree.body), overload_names):
        node = carrier.node
        record(
            node,
            "class" if isinstance(node, ast.ClassDef) else "function",
            f"{module}.{node.name}",
            always_a_gap=carrier.always_a_gap,
        )

        # Any branch of a conditionally defined class can be the one that binds
        # the name, so the members of all of them are in play -- including when
        # another branch binds that name to a function and carries it.
        class_bodies = [
            definition.body
            for definition in carrier.group
            if isinstance(definition, ast.ClassDef)
        ]
        if not class_bodies:
            continue

        # Attribute docstrings are positional, so each class body is scanned
        # on its own: flattening them first would let one class's docstring sit
        # where the previous class's trailing assignment looks for its own.
        descriptors: dict[str, dict[tuple, Descriptor]] = {}
        descriptor_kinds: dict[str, str] = {}
        for index, class_body in enumerate(class_bodies):
            # A descriptor can name a private getter, so every function of the
            # body is in the lookup, not just the public ones.
            function_docstrings = {
                member.name: ast.get_docstring(member) is not None
                for member, _ in _definitions(class_body)
                if isinstance(member, _FUNCTION_NODES)
            }
            for assigned_name, alternatives in _descriptor_assignments(
                class_body, function_docstrings, (index,)
            ).items():
                descriptors.setdefault(assigned_name, {}).update(alternatives)
                descriptor_kinds[assigned_name] = next(
                    iter(alternatives.values())
                ).kind
        methods = (
            (member, (index, suite))
            for index, class_body in enumerate(class_bodies)
            for member, suite in _definitions(class_body)
            if isinstance(member, _FUNCTION_NODES)
        )
        for member in _carriers(methods, overload_names, descriptors):
            name = _bound_name(member.node)
            kind = (
                "method"
                if isinstance(member.node, _FUNCTION_NODES)
                else descriptor_kinds.get(name, "property")
            )
            record(
                member.node,
                kind,
                f"{module}.{node.name}.{name}",
                always_a_gap=member.always_a_gap,
            )

    # Grouping by name can pick a carrier that is not the first definition of
    # the group, so restore source order for the report.
    gaps.sort(key=lambda gap: gap.lineno)
    return gaps, total


def collect(package_root: Path) -> tuple[list[Gap], int]:
    """Undocumented public objects of a package, and its public-object count."""
    gaps: list[Gap] = []
    total = 0

    for path in sorted(package_root.rglob("*.py")):
        module = _module_name(path, package_root)
        if _is_ignored(module):
            continue
        if any(part.startswith("_") for part in module.split(".")):
            continue

        module_gaps, module_total = _scan_module(path, module, package_root.parent)
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

    package_root = args.package.resolve()
    # Without this, a misspelled ``--package`` scans nothing, reports "0 of 0"
    # and exits successfully -- a green run that checked no code at all.
    if not (package_root / "__init__.py").is_file():
        parser.error(f"{package_root} is not a Python package directory")

    gaps, total = collect(package_root)
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

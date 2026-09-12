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
machine that could not build the documentation.

*Public* means every module-level class and function, and every method or
property of a public class, whose name does not begin with an underscore, in a
module whose dotted name has no underscore-prefixed component.

What binds such a name is read from the source in order, because several
statements can bind one and only the object a reader ends up importing counts:

* a ``def`` or a ``class``, wherever it sits -- including inside module- or
  class-level control flow, such as the ``except ImportError`` fallback for an
  optional dependency;
* a class member created by assigning a descriptor: ``value = property(...)``,
  ``public = classmethod(_impl)``, the ``staticmethod`` form, and the plain
  alias ``public = _implementation``;
* any other assignment to the name, which does not create a documentable
  function or class and therefore removes it from the report.

:func:`_dominates` decides which of them survive: a binding replaces one made
earlier in the same suite or in a suite it encloses, while the branches of an
``if`` or the handlers of a ``try`` are alternatives that all survive, since
either can be the one that runs.  A name is a gap unless *every* surviving
binding is documented.  The two halves of a property count once, under the
getter; an ``@overload`` set counts once, under its implementation; and the
body of an ``if TYPE_CHECKING:`` counts not at all, because it never runs.

Two limits are deliberate.  Nothing outside the file is resolved, so a getter
or a base class in another module cannot be inspected; where that decides a
docstring -- ``@Base.value.setter``, which inherits one -- the name is taken as
documented, because a false gap breaks a ratchet while a missed one only fails
to tighten it.  And nothing is executed, so a binding made by anything other
than a statement of the forms above is invisible.

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
# part of the supported API.  Each entry ignores that module *and its
# descendants*, so an addition below ``volumential.qbfem`` cannot fail the
# ratchet for a package the documentation leaves out entirely.
IGNORED_MODULES = frozenset({"volumential.qbfem"})

_FUNCTION_NODES = (ast.AsyncFunctionDef, ast.FunctionDef)

# Statements a binding can be nested in and still belong to the enclosing
# module or class.  ``ast.Match`` is handled beside them in :func:`_suites_of`,
# because its suites hang off ``cases`` rather than the shared fields.
_CONTROL_FLOW_NODES = (ast.For, ast.If, ast.Try, ast.TryStar, ast.While, ast.With)

#: Builtins whose result is a class member, and what the report calls it.
_DESCRIPTOR_KINDS = {
    "property": "property",
    "classmethod": "method",
    "staticmethod": "method",
}

#: Kind for a binding that is not a documentable function, class or descriptor.
#: It still dominates whatever it replaces; it is simply never reported.
_OPAQUE = "value"

#: Kind for the marker an exhaustive ``if``/``else`` emits for a name its
#: branches all rebind.  It dominates what came before and is then discarded:
#: the branch bindings themselves are the alternatives.
_SHADOW = "shadow"


class Gap(NamedTuple):
    """One public object that has no docstring."""

    path: str
    lineno: int
    kind: str
    name: str


class Binding(NamedTuple):
    """One statement binding one name."""

    #: The name it binds.  A chained assignment binds several, one per
    #: :class:`Binding`, so this is not always derivable from *node*.
    name: str
    #: The statement the report points at.
    node: ast.stmt
    #: The suite it sits in; see :func:`_dominates`.
    suite: tuple
    #: What the report calls it, or :data:`_OPAQUE`.
    kind: str
    #: ``True``/``False`` when the statement settles it, ``None`` to read the
    #: docstring of *node* itself.
    documented: bool | None


# {{{ guards and decorators


def _imported_as(tree: ast.Module, wanted: str) -> frozenset[str]:
    """The names this module binds to ``typing.<wanted>``.

    ``from typing import overload as _overload`` is as much an overload
    decorator as the plain name, and a module that defines an ``overload`` of
    its own is not one at all, so both are recognised by what was imported
    rather than by how they are spelled.  The attribute forms,
    ``@typing.overload`` and ``if typing.TYPE_CHECKING:``, are matched by
    attribute name instead.
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
                if alias.name == wanted
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


def _is_type_checking_guard(node: ast.stmt, guard_names: frozenset[str]) -> bool:
    """Is *node* an ``if TYPE_CHECKING:``, whose body never runs?

    ``typing.TYPE_CHECKING`` is ``False`` at run time, so nothing under it
    binds anything a reader can import.  Only the ``if`` body is type-only; the
    ``else`` runs and is scanned as usual.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id in guard_names) or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _property_mutator_decorators(node: ast.AST) -> list[ast.Attribute]:
    """The ``@x.setter`` / ``@x.deleter`` decorators *node* carries."""
    return [
        decorator
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Attribute)
        and decorator.attr in {"deleter", "setter"}
    ]


def _is_property_mutator(node: ast.AST) -> bool:
    """Is *node* the ``@x.setter`` or ``@x.deleter`` half of a property?"""
    return bool(_property_mutator_decorators(node))


def _overrides_inherited_property(node: ast.AST) -> bool:
    """Does *node* override one half of a property defined on another class?

    ``@Base.value.setter`` builds a new property from ``Base.value``, so it
    keeps that property's getter and its docstring; ``@value.setter`` refers to
    a property of this class body, which has to carry its own.
    """
    decorators = _property_mutator_decorators(node)
    return bool(decorators) and all(
        isinstance(decorator.value, ast.Attribute) for decorator in decorators
    )


# }}}


# {{{ walking suites


def _suites_of(
    node: ast.stmt, suite: tuple, guard_names: frozenset[str]
) -> Iterator[tuple[list, tuple]]:
    """Yield ``(statements, suite)`` for each branch of a control-flow node.

    Three subtleties are encoded here rather than at every call site.  A
    ``finally`` suite runs on every path out of the ``try``, so its bindings
    belong to the *enclosing* suite and dominate whatever the body and the
    handlers bound.  A ``try``'s ``else`` runs after the body on the successful
    path, so it shares the body's suite and supersedes it, while the handlers
    stay alternatives.  And the body of an ``if TYPE_CHECKING:`` is not yielded
    at all, because it never runs.
    """
    if isinstance(node, ast.Match):
        for index, case in enumerate(node.cases):
            yield case.body, (*suite, node.lineno, "case", index)
        return

    body_suite = (*suite, node.lineno, "body")
    if not _is_type_checking_guard(node, guard_names):
        yield node.body, body_suite

    if isinstance(node, (ast.Try, ast.TryStar)):
        yield node.orelse, body_suite
    else:
        yield getattr(node, "orelse", []), (*suite, node.lineno, "orelse")

    for index, handler in enumerate(getattr(node, "handlers", [])):
        yield handler.body, (*suite, node.lineno, "handler", index)
    if getattr(node, "finalbody", []):
        yield node.finalbody, suite


def _dominates(suite: tuple, other: tuple) -> bool:
    """Does a binding in *suite* overwrite one already made in *other*?

    Only if *other* is nested inside it, or is it: a binding in the enclosing
    suite runs whatever the branches did, so it replaces anything they bound.
    Two branches of one ``if`` are nested in neither, and both survive.
    """
    return other[: len(suite)] == suite


def _is_public(name: str) -> bool:
    return not name.startswith("_")


# }}}


# {{{ reading one suite into bindings


def _assignment_targets(node: ast.stmt) -> list[ast.Name]:
    """The plain names an assignment statement binds.

    A chained assignment binds each of its targets, and a tuple or attribute
    target binds nothing this module tracks.
    """
    targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
    return [target for target in targets if isinstance(target, ast.Name)]


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


def _assignment_bindings(
    node: ast.stmt,
    following: ast.stmt | None,
    suite: tuple,
    function_docstrings: dict[str, bool],
    in_class: bool,
) -> Iterator[Binding]:
    """Read an assignment to a public name as a :class:`Binding`.

    An assignment can create a documentable object: a descriptor inside a
    class body (``value = property(...)``, ``public = classmethod(_impl)``), or
    anywhere an alias of a function defined in the same suite
    (``public = _implementation``) or a ``lambda``.  A bare string literal after
    it is that object's docstring, which is the attribute-docstring convention
    autodoc reads.  Anything else binds a value that is not a documentable
    function or class: it still replaces whatever the name held, so it is
    recorded as :data:`_OPAQUE` rather than dropped, and the name leaves the
    report.
    """
    targets = _assignment_targets(node)
    if not targets:
        return
    if isinstance(node, ast.AnnAssign) and node.value is None:
        # ``public: Callable[..., None]`` annotates the name without binding
        # it, so whatever the name already held is still what it holds.
        return

    value = node.value
    callable_kind = "method" if in_class else "function"
    kind: str | None = None
    from_value = False
    if in_class and isinstance(value, ast.Call):
        function = value.func
        builtin = (
            function.id
            if isinstance(function, ast.Name)
            else getattr(function, "attr", None)
        )
        kind = _DESCRIPTOR_KINDS.get(builtin or "")
        if kind is not None:
            from_value = _descriptor_documentation(
                value, kind, function_docstrings
            )
    elif isinstance(value, ast.Name) and value.id in function_docstrings:
        # ``public = _implementation`` exposes the private function under a
        # public name, and the public object's ``__doc__`` is that function's.
        kind = callable_kind
        from_value = function_docstrings[value.id]
    elif isinstance(value, ast.Lambda):
        # A lambda is a function with no docstring to expose.
        kind = callable_kind

    if kind is None:
        for target in targets:
            yield Binding(target.id, node, suite, _OPAQUE, True)
        return

    attribute_docstring = (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    )
    documented = attribute_docstring or from_value
    for target in targets:
        yield Binding(target.id, node, suite, kind, documented)


def _bindings(
    body: Iterable[ast.stmt],
    guard_names: frozenset[str],
    function_docstrings: dict[str, bool],
    *,
    in_class: bool,
    suite: tuple = (),
) -> Iterator[Binding]:
    """Yield every name binding in one suite, in source order.

    Private names are yielded too, because a descriptor's getter is usually
    one; the caller filters.  Function and class bodies are not entered: what
    they define belongs to that scope, not to this one.
    """
    statements = list(body)
    for index, node in enumerate(statements):
        if isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)):
            kind = (
                "class"
                if isinstance(node, ast.ClassDef)
                else ("method" if in_class else "function")
            )
            yield Binding(node.name, node, suite, kind, None)
        elif isinstance(node, (*_CONTROL_FLOW_NODES, ast.Match)):
            branches = list(_suites_of(node, suite, guard_names))
            for name in _rebound_by_every_branch(
                node, branches, guard_names, function_docstrings, in_class
            ):
                yield Binding(name, node, suite, _SHADOW, None)
            for statements_of_branch, branch in branches:
                yield from _bindings(
                    statements_of_branch,
                    guard_names,
                    function_docstrings,
                    in_class=in_class,
                    suite=branch,
                )
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            next_index = index + 1
            following = (
                statements[next_index] if next_index < len(statements) else None
            )
            yield from _assignment_bindings(
                node, following, suite, function_docstrings, in_class
            )


def _rebound_by_every_branch(
    node: ast.stmt,
    branches: list[tuple[list, tuple]],
    guard_names: frozenset[str],
    function_docstrings: dict[str, bool],
    in_class: bool,
) -> set[str]:
    """Names that an exhaustive ``if``/``else`` rebinds whichever way it goes.

    Only ``if``/``else`` is treated as exhaustive, and only when both suites
    bind the name: then whatever the name held before the statement is
    unreachable afterwards, and a marker in the enclosing suite says so.  A
    ``try`` is not exhaustive -- its body can stop part way -- and a ``match``
    need not be, so neither is considered here.
    """
    if not isinstance(node, ast.If) or not node.orelse:
        return set()
    if _is_type_checking_guard(node, guard_names):
        return set()

    per_branch = [
        {
            binding.name
            for binding in _bindings(
                statements,
                guard_names,
                function_docstrings,
                in_class=in_class,
                suite=branch,
            )
        }
        for statements, branch in branches
    ]
    return set.intersection(*per_branch) if per_branch else set()


def _function_documentation(
    body: Iterable[ast.stmt], guard_names: frozenset[str]
) -> dict[str, bool]:
    """Whether each function defined in one suite has a docstring, by name.

    Private names are included, because that is what a descriptor's getter or
    an aliased implementation usually is, and a name defined in two surviving
    branches counts as documented only when both are.
    """
    known = _function_documentation_pass(body, guard_names, {})
    # ``_alias = _impl`` then ``public = _alias``: the second assignment can
    # only be read as a function once the first one has been.  Two names is
    # already an unusual chain, so a handful of passes is ample, and the loop
    # stops as soon as nothing new is learned.
    for _ in range(4):
        wider = _function_documentation_pass(body, guard_names, known)
        if wider == known:
            break
        known = wider
    return known


def _function_documentation_pass(
    body: Iterable[ast.stmt],
    guard_names: frozenset[str],
    function_docstrings: dict[str, bool],
) -> dict[str, bool]:
    """One sweep of :func:`_function_documentation`."""
    body = list(body)
    surviving: dict[str, dict[tuple, bool | None]] = {}
    for binding in _bindings(
        body, guard_names, function_docstrings, in_class=False
    ):
        node = binding.node
        if isinstance(node, ast.ClassDef) or binding.kind == _SHADOW:
            continue
        if isinstance(node, _FUNCTION_NODES):
            documented: bool | None = ast.get_docstring(node) is not None
        elif binding.kind == _OPAQUE:
            # Rebound to something this pass cannot call a function -- a
            # constant, an import, a call.  ``None`` drops the name from the
            # result, so an alias of it is not read as a function and a
            # descriptor built on it gets no docstring from it.
            documented = None
        else:
            # A ``lambda``, or an alias an earlier pass resolved.
            documented = binding.documented
        bucket = surviving.setdefault(binding.name, {})
        for bound in list(bucket):
            if _dominates(binding.suite, bound):
                del bucket[bound]
        bucket[binding.suite] = documented

    return {
        name: all(states.values())
        for name, states in surviving.items()
        if None not in states.values()
    }


# }}}


# {{{ choosing what carries each name


class Carrier(NamedTuple):
    """What the report says about one public name."""

    #: The name.
    name: str
    #: The statement the report points at.
    node: ast.stmt
    #: What the report calls it.
    kind: str
    #: Whether the object the name ends up holding has a docstring.
    documented: bool
    #: The bindings that survived, which is where a class's members come from:
    #: a superseded ``class`` body is not part of the API any more.
    surviving: list[Binding]


def _carriers(
    bindings: Iterable[Binding], overload_names: frozenset[str]
) -> Iterator[Carrier]:
    """One :class:`Carrier` per public name a source-ordered stream binds.

    Bindings of one name are collected by suite, so that a rebinding keeps only
    the last of its suite while the branches of a conditional keep all of
    theirs, and a binding in an enclosing suite clears the nested ones it
    replaces.  What survives is then judged together:

    * a name whose last surviving binding is not a documentable object --
      ``public = 3`` over a former ``def public()`` -- is not reported at all;
    * an ``@overload`` stub loses to the implementation that follows the set;
    * a property takes its ``__doc__`` from its getter, never from a mutator,
      so a name bound only by mutators is a gap -- unless every one of them is
      qualified, as in ``@Base.value.setter``, which keeps the base property's
      getter and therefore its docstring;
    * any surviving binding can be the one a reader imports, so the name is a
      gap unless all of them are documented, and the earliest that is not is
      what the report points at.
    """
    groups: dict[str, dict[tuple, Binding]] = {}
    for binding in bindings:
        if not _is_public(binding.name):
            continue
        bucket = groups.setdefault(binding.name, {})
        if binding.kind == _SHADOW:
            for bound in list(bucket):
                if isinstance(bound, tuple) and bound and bound[0] == "mutator":
                    continue
                if _dominates(binding.suite, bound):
                    del bucket[bound]
            continue
        if isinstance(binding.node, _FUNCTION_NODES) and _is_property_mutator(
            binding.node
        ):
            # A mutator augments the property its getter created rather than
            # rebinding the name, so it must not displace that getter.
            bucket[("mutator", binding.suite, binding.node.lineno)] = binding
            continue
        for bound in list(bucket):
            if isinstance(bound, tuple) and bound and bound[0] == "mutator":
                continue
            if _dominates(binding.suite, bound):
                del bucket[bound]
        bucket[binding.suite] = binding

    for bucket in groups.values():
        surviving = list(bucket.values())
        if all(binding.kind == _OPAQUE for binding in surviving):
            # Nothing documentable is left under this name.
            continue

        implementations = [
            binding
            for binding in surviving
            if not (
                isinstance(binding.node, _FUNCTION_NODES)
                and _is_typing_overload(binding.node, overload_names)
            )
        ]
        candidates = implementations or surviving
        carriers = [
            binding
            for binding in candidates
            if not (
                isinstance(binding.node, _FUNCTION_NODES)
                and _is_property_mutator(binding.node)
            )
        ]

        if not carriers:
            # Mutators only.  Qualified ones inherit the base property's
            # docstring; unqualified ones have no getter here to take one from.
            inherited = all(
                _overrides_inherited_property(binding.node)
                for binding in candidates
            )
            first_binding = candidates[0]
            yield Carrier(
                first_binding.name,
                first_binding.node,
                first_binding.kind,
                inherited,
                surviving,
            )
            continue

        undocumented = [
            binding
            for binding in carriers
            if binding.kind != _OPAQUE and not _is_documented(binding)
        ]
        if undocumented:
            first = min(undocumented, key=lambda binding: binding.node.lineno)
            yield Carrier(first.name, first.node, first.kind, False, surviving)
        else:
            last = carriers[-1]
            yield Carrier(last.name, last.node, last.kind, True, surviving)


def _is_documented(binding: Binding) -> bool:
    """Does the object *binding* creates have a docstring?"""
    if binding.documented is not None:
        return binding.documented
    return ast.get_docstring(binding.node) is not None


# }}}


# {{{ scanning modules


def _module_name(path: Path, package_root: Path) -> str:
    """Dotted module name of *path* inside the package rooted at *package_root*."""
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_ignored(module: str) -> bool:
    """Is *module* one of :data:`IGNORED_MODULES`, or inside one?"""
    return any(
        module == ignored or module.startswith(f"{ignored}.")
        for ignored in IGNORED_MODULES
    )


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
    overload_names = _imported_as(tree, "overload")
    guard_names = _imported_as(tree, "TYPE_CHECKING") | {"TYPE_CHECKING"}

    gaps: list[Gap] = []
    total = 0

    def record(carrier: Carrier, name: str) -> None:
        nonlocal total
        total += 1
        if not carrier.documented:
            gaps.append(Gap(relative, carrier.node.lineno, carrier.kind, name))

    module_functions = _function_documentation(tree.body, guard_names)
    module_bindings = list(
        _bindings(tree.body, guard_names, module_functions, in_class=False)
    )
    for carrier in _carriers(module_bindings, overload_names):
        name = carrier.name
        record(carrier, f"{module}.{name}")

        # Any branch of a conditionally defined class can be the one that binds
        # the name, so the members of all of them are in play -- including when
        # another branch binds that name to a function and carries it.
        class_bodies = [
            binding.node.body
            for binding in carrier.surviving
            if isinstance(binding.node, ast.ClassDef)
        ]
        if not class_bodies:
            continue

        # Attribute docstrings are positional and the class index keeps the
        # suites of different class bodies apart, so each body is read on its
        # own: concatenating them first would let one class's docstring sit
        # where the previous class's trailing assignment looks for its own.
        members: list[Binding] = []
        for index, class_body in enumerate(class_bodies):
            function_docstrings = _function_documentation(class_body, guard_names)
            members.extend(
                binding._replace(suite=(index, *binding.suite))
                for binding in _bindings(
                    class_body, guard_names, function_docstrings, in_class=True
                )
            )
        for member in _carriers(members, overload_names):
            record(member, f"{module}.{name}.{member.name}")

    # Grouping by name can pick a carrier that is not the first binding of the
    # group, so restore source order for the report.
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


# }}}


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

# vim: foldmethod=marker

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


class Known(NamedTuple):
    """What a name defined in this suite holds, for resolving aliases."""

    #: ``"class"`` or ``"function"``.
    kind: str
    #: Whether that definition carries a docstring.
    documented: bool
    #: The definition itself, so an alias of a class can be scanned.
    node: ast.stmt | None


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
    #: For an alias, the definition it resolves to, whose body a class scan
    #: needs; ``None`` otherwise.
    target: ast.stmt | None = None


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


def _typing_module_names(tree: ast.Module) -> frozenset[str]:
    """The names this module binds to ``typing`` or ``typing_extensions``."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"typing", "typing_extensions"}
            )
    return frozenset(names)


class Guards(NamedTuple):
    """How this module can spell ``TYPE_CHECKING``."""

    #: Names imported from a typing module, ``TYPE_CHECKING`` or an alias.
    direct: frozenset[str]
    #: Names a typing module itself is bound to, for the qualified form.
    modules: frozenset[str]


def _is_type_checking_guard(node: ast.stmt, guards: Guards) -> bool:
    """Is *node* an ``if TYPE_CHECKING:``, whose body never runs?

    ``typing.TYPE_CHECKING`` is ``False`` at run time, so nothing under it
    binds anything a reader can import.  Only the ``if`` body is type-only; the
    ``else`` runs and is scanned as usual.

    The attribute form is accepted only when its base is a typing module this
    file imported: a project's own ``flags.TYPE_CHECKING`` can perfectly well
    be true, and its body is ordinary code.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id in guards.direct
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in guards.modules
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


def _inherited_property_sources(node: ast.AST) -> list[tuple[str, str]]:
    """The ``(class, property)`` pairs a qualified mutator of *node* names.

    ``@Base.value.setter`` builds a new property from ``Base.value``, so it
    keeps that property's getter and its docstring; ``@value.setter`` refers to
    a property of this class body, which has to carry its own, and contributes
    no pair.
    """
    pairs = []
    for decorator in _property_mutator_decorators(node):
        base = decorator.value
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
        ):
            pairs.append((base.value.id, base.attr))
    return pairs


def _overrides_inherited_property(
    node: ast.AST, inherited: dict[str, dict[str, bool]]
) -> bool | None:
    """Is *node* an override of a property documented on another class?

    ``None`` when *node* is not a qualified mutator at all.  When the base
    class is defined in this file its getter settles the answer; when it is
    not, the answer is ``True``, because a false gap breaks a ratchet while a
    missed one only fails to tighten it.
    """
    decorators = _property_mutator_decorators(node)
    pairs = _inherited_property_sources(node)
    if not decorators or len(pairs) != len(decorators):
        return None

    documented = True
    for class_name, property_name in pairs:
        local = inherited.get(class_name)
        if local is not None and property_name in local:
            documented = documented and local[property_name]
    return documented


# }}}


# {{{ walking suites


def _suites_of(
    node: ast.stmt, suite: tuple, guards: Guards
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
    if not _is_type_checking_guard(node, guards):
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


def _has_docstring(node: ast.AST) -> bool:
    """Does *node* carry a docstring with something in it?

    ``ast.get_docstring`` normalises ``\"\"\"   \"\"\"`` to an empty string
    rather than ``None``, and an empty docstring puts nothing on the page.
    """
    return bool((ast.get_docstring(node) or "").strip())


# }}}


# {{{ reading one suite into bindings


def _names_in(target: ast.expr) -> Iterator[str]:
    """The names a destructuring target binds, however deeply nested."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _names_in(target.value)
    elif isinstance(target, (ast.List, ast.Tuple)):
        for element in target.elts:
            yield from _names_in(element)


def _assignment_targets(node: ast.stmt) -> tuple[list[str], list[str]]:
    """The names an assignment binds, split by whether the value is theirs.

    A chained assignment gives each of its plain targets the whole value, so
    those can be classified.  A destructuring target takes the value apart, and
    this module does not follow that -- those names are returned separately and
    recorded as opaque, which is enough for them to replace what they shadow.
    """
    targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
    plain = [target.id for target in targets if isinstance(target, ast.Name)]
    unpacked = [
        name
        for target in targets
        if not isinstance(target, ast.Name)
        for name in _names_in(target)
    ]
    return plain, unpacked


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
            if keyword.arg != "doc":
                continue
            explicit = keyword.value
            if isinstance(explicit, ast.Constant) and explicit.value is None:
                # ``doc=None`` is the default, and tells ``property`` to copy
                # the getter's docstring rather than to have none.
                break
            return (
                isinstance(explicit, ast.Constant)
                and isinstance(explicit.value, str)
                and bool(explicit.value.strip())
            )

    wrapped = call.args[0] if call.args else None
    if wrapped is None:
        for keyword in call.keywords:
            if keyword.arg in {"fget", "f"}:
                wrapped = keyword.value
                break
    if isinstance(wrapped, ast.Name):
        known = function_docstrings.get(wrapped.id)
        return known.documented if known is not None else False
    return False


def _assignment_bindings(
    node: ast.stmt,
    following: ast.stmt | None,
    suite: tuple,
    function_docstrings: dict[str, Known],
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
    targets, unpacked = _assignment_targets(node)
    if not targets and not unpacked:
        return
    if isinstance(node, ast.AnnAssign) and node.value is None:
        # ``public: Callable[..., None]`` annotates the name without binding
        # it, so whatever the name already held is still what it holds.
        return

    for name in unpacked:
        yield Binding(name, node, suite, _OPAQUE, True)
    if not targets:
        return

    value = node.value
    callable_kind = "method" if in_class else "function"
    kind: str | None = None
    from_value = False
    target: ast.stmt | None = None
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
        # ``Public = _Implementation`` exposes a private definition under a
        # public name, and the public object's ``__doc__`` is that one's.
        known = function_docstrings[value.id]
        kind = "class" if known.kind == "class" else callable_kind
        from_value = known.documented
        target = known.node
    elif isinstance(value, ast.Lambda):
        # A lambda is a function with no docstring to expose.
        kind = callable_kind

    if kind is None:
        for name in targets:
            yield Binding(name, node, suite, _OPAQUE, True)
        return

    attribute_docstring = (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    )
    documented = attribute_docstring or from_value
    for name in targets:
        yield Binding(name, node, suite, kind, documented, target)


def _bindings(
    body: Iterable[ast.stmt],
    guards: Guards,
    function_docstrings: dict[str, Known],
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
            branches = list(_suites_of(node, suite, guards))
            for name in _rebound_by_every_branch(
                node, branches, guards, function_docstrings, in_class
            ):
                yield Binding(name, node, suite, _SHADOW, None)
            for statements_of_branch, branch in branches:
                yield from _bindings(
                    statements_of_branch,
                    guards,
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
    guards: Guards,
    function_docstrings: dict[str, Known],
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
    if _is_type_checking_guard(node, guards):
        return set()

    per_branch = [
        _definitely_bound(
            statements, guards, function_docstrings, in_class, branch
        )
        for statements, branch in branches
    ]
    return set.intersection(*per_branch) if per_branch else set()


def _definitely_bound(
    statements: Iterable[ast.stmt],
    guards: Guards,
    function_docstrings: dict[str, Known],
    in_class: bool,
    suite: tuple,
) -> set[str]:
    """Names this suite binds on *every* path through it.

    A binding inside a conditional that may not run tells us nothing: the name
    can still hold what it held before.  So only statements that always execute
    count -- a definition or assignment in the suite itself, the body of a
    ``with``, the ``finally`` of a ``try``, and an ``if`` with an ``else``
    where both arms bind the name.
    """
    bound: set[str] = set()
    for node in statements:
        if isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)):
            bound.add(node.name)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            bound.update(
                binding.name
                for binding in _assignment_bindings(
                    node, None, suite, function_docstrings, in_class
                )
            )
        elif isinstance(node, ast.With):
            bound |= _definitely_bound(
                node.body, guards, function_docstrings, in_class, suite
            )
        elif isinstance(node, (ast.Try, ast.TryStar)):
            bound |= _definitely_bound(
                node.finalbody, guards, function_docstrings, in_class, suite
            )
        elif (
            isinstance(node, ast.If)
            and node.orelse
            and not _is_type_checking_guard(node, guards)
        ):
            bound |= _definitely_bound(
                node.body, guards, function_docstrings, in_class, suite
            ) & _definitely_bound(
                node.orelse, guards, function_docstrings, in_class, suite
            )
    return bound


def _function_documentation(
    body: Iterable[ast.stmt], guards: Guards
) -> dict[str, Known]:
    """What each name defined in one suite holds, for resolving aliases.

    Private names are included, because that is what a descriptor's getter or
    an aliased implementation usually is, and a name defined in two surviving
    branches counts as documented only when both are.
    """
    # ``_alias = _impl`` then ``public = _alias``: the second assignment can
    # only be read as a function once the first one has been, so sweep until
    # nothing new is learned.  Each sweep can only add names or turn one from
    # documented to not, and there are finitely many, so this terminates.
    known: dict[str, Known] = {}
    while True:
        wider = _function_documentation_pass(body, guards, known)
        if wider == known:
            return known
        known = wider


def _function_documentation_pass(
    body: Iterable[ast.stmt],
    guards: Guards,
    function_docstrings: dict[str, Known],
) -> dict[str, Known]:
    """One sweep of :func:`_function_documentation`."""
    body = list(body)
    surviving: dict[str, dict[tuple, Known | None]] = {}
    for binding in _bindings(
        body, guards, function_docstrings, in_class=False
    ):
        node = binding.node
        if binding.kind == _SHADOW:
            bucket = surviving.setdefault(binding.name, {})
            for bound in list(bucket):
                if _dominates(binding.suite, bound):
                    del bucket[bound]
            continue
        if isinstance(node, ast.ClassDef):
            held: Known | None = Known("class", _has_docstring(node), node)
        elif isinstance(node, _FUNCTION_NODES):
            held = Known("function", _has_docstring(node), node)
        elif binding.kind == _OPAQUE:
            # Rebound to something this pass cannot resolve -- a constant, an
            # import, a call.  ``None`` drops the name from the result, so an
            # alias of it is not read as a definition and a descriptor built on
            # it gets no docstring from it.
            held = None
        else:
            # A ``lambda``, or an alias an earlier pass resolved.
            held = Known(
                "class" if binding.kind == "class" else "function",
                bool(binding.documented),
                binding.target,
            )
        bucket = surviving.setdefault(binding.name, {})
        for bound in list(bucket):
            if _dominates(binding.suite, bound):
                del bucket[bound]
        bucket[binding.suite] = held

    resolved: dict[str, Known] = {}
    for name, states in surviving.items():
        held_states = list(states.values())
        if None in held_states:
            continue
        first = held_states[0]
        resolved[name] = Known(
            first.kind,
            all(state.documented for state in held_states),
            first.node if len(held_states) == 1 else None,
        )
    return resolved


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
    bindings: Iterable[Binding],
    overload_names: frozenset[str],
    inherited: dict[str, dict[str, bool]] | None = None,
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
    inherited = inherited or {}
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
            overrides = [
                _overrides_inherited_property(binding.node, inherited)
                for binding in candidates
            ]
            documented_override = all(
                answer is not None and answer for answer in overrides
            )
            first_binding = candidates[0]
            yield Carrier(
                first_binding.name,
                first_binding.node,
                first_binding.kind,
                documented_override,
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
    return _has_docstring(binding.node)


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


def _local_property_documentation(
    tree: ast.Module, guards: Guards
) -> dict[str, dict[str, bool]]:
    """Per class defined in this file, whether each property getter is documented.

    Only the decorated form is read, which is the one a subclass overrides half
    of.  A class this file does not define is simply absent, and
    :func:`_overrides_inherited_property` then falls back to assuming the base
    is documented.
    """
    found: dict[str, dict[str, bool]] = {}
    for binding in _bindings(tree.body, guards, {}, in_class=False):
        node = binding.node
        if not isinstance(node, ast.ClassDef):
            continue
        properties = found.setdefault(node.name, {})
        for member, _ in ((m.node, m.suite) for m in _bindings(
            node.body, guards, {}, in_class=True
        )):
            if not isinstance(member, _FUNCTION_NODES):
                continue
            if any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in member.decorator_list
            ):
                properties[member.name] = _has_docstring(member)
    return found


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
    guards = Guards(
        direct=_imported_as(tree, "TYPE_CHECKING"),
        modules=_typing_module_names(tree),
    )

    gaps: list[Gap] = []
    total = 0

    def record(carrier: Carrier, name: str) -> None:
        nonlocal total
        total += 1
        if not carrier.documented:
            gaps.append(Gap(relative, carrier.node.lineno, carrier.kind, name))

    inherited = _local_property_documentation(tree, guards)
    module_functions = _function_documentation(tree.body, guards)
    module_bindings = list(
        _bindings(tree.body, guards, module_functions, in_class=False)
    )
    for carrier in _carriers(module_bindings, overload_names, inherited):
        name = carrier.name
        record(carrier, f"{module}.{name}")

        # Any branch of a conditionally defined class can be the one that binds
        # the name, so the members of all of them are in play -- including when
        # another branch binds that name to a function and carries it.
        class_bodies = []
        for binding in carrier.surviving:
            definition = binding.target if binding.target else binding.node
            if isinstance(definition, ast.ClassDef):
                class_bodies.append(definition.body)
        if not class_bodies:
            continue

        # Attribute docstrings are positional and the class index keeps the
        # suites of different class bodies apart, so each body is read on its
        # own: concatenating them first would let one class's docstring sit
        # where the previous class's trailing assignment looks for its own.
        members: list[Binding] = []
        for index, class_body in enumerate(class_bodies):
            function_docstrings = _function_documentation(class_body, guards)
            members.extend(
                binding._replace(suite=(index, *binding.suite))
                for binding in _bindings(
                    class_body, guards, function_docstrings, in_class=True
                )
            )
        for member in _carriers(members, overload_names, inherited):
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

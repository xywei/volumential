"""Behaviour of ``doc/tools/docstring_gaps.py``, one case per code shape.

The checker decides from the source alone what object a public name ends up
holding and whether it is documented, and the shapes below are the ones that
make that decision non-obvious: conditional and superseded definitions,
``@overload`` sets, properties in each of their forms, descriptor and alias
assignments, and ``if TYPE_CHECKING:``.  Each case is a whole module, and the
expectation is how many of its public objects have no docstring.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_checker():
    """Import the checker, which lives outside any importable package."""
    path = Path(__file__).resolve().parents[1] / "doc" / "tools"
    spec = importlib.util.spec_from_file_location(
        "docstring_gaps", path / "docstring_gaps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


docstring_gaps = _load_checker()


# Each entry is a module body and the number of undocumented public objects
# the checker should find in it.
CASES = {
    'getter+setter one suite': (
        '''
"""F."""
class W:
    """W."""
    @property
    def size(self):
        """Doc."""
        return 1
    @size.setter
    def size(self, v):
        pass
''',
        0,
    ),
    'write-only property, documented setter': (
        '''
"""F."""
class W:
    """W."""
    value = property()
    @value.setter
    def value(self, v):
        """Doc."""
''',
        1,
    ),
    'bare property assignment': (
        '''
"""F."""
class W:
    """W."""
    value = property()
''',
        1,
    ),
    'property with attribute docstring': (
        '''
"""F."""
class W:
    """W."""
    value = property()
    """Doc."""
''',
        0,
    ),
    'annotated property assignment': (
        '''
"""F."""
class W:
    """W."""
    value: property = property()
''',
        1,
    ),
    'property doc= argument': (
        '''
"""F."""
class W:
    """W."""
    value = property(fget=lambda self: 1, doc="Doc")
''',
        0,
    ),
    'property from documented private getter': (
        '''
"""F."""
class W:
    """W."""
    def _g(self):
        """Doc."""
        return 1
    value = property(_g)
''',
        0,
    ),
    'property from undocumented private getter': (
        '''
"""F."""
class W:
    """W."""
    def _g(self):
        return 1
    value = property(_g)
''',
        1,
    ),
    'getter documented in only one branch': (
        '''
"""F."""
import sys
class W:
    """W."""
    if sys.platform == "linux":
        def _g(self):
            return 1
    else:
        def _g(self):
            """Doc."""
            return 1
    value = property(_g)
''',
        1,
    ),
    'getter rebound documented in same suite': (
        '''
"""F."""
class W:
    """W."""
    def _g(self):
        return 1
    def _g(self):
        """Doc."""
        return 1
    value = property(_g)
''',
        0,
    ),
    'classmethod assignment undocumented': (
        '''
"""F."""
class W:
    """W."""
    def _impl(cls):
        return 1
    public = classmethod(_impl)
''',
        1,
    ),
    'staticmethod assignment documented': (
        '''
"""F."""
class W:
    """W."""
    def _impl():
        """Doc."""
        return 1
    public = staticmethod(_impl)
''',
        0,
    ),
    'plain alias undocumented': (
        '''
"""F."""
class W:
    """W."""
    def _impl(self):
        return 1
    public = _impl
''',
        1,
    ),
    'plain alias documented': (
        '''
"""F."""
class W:
    """W."""
    def _impl(self):
        """Doc."""
        return 1
    public = _impl
''',
        0,
    ),
    'getter then later property assignment': (
        '''
"""F."""
class W:
    """W."""
    @property
    def value(self):
        """Doc."""
        return 1
    value = property()
''',
        1,
    ),
    'property assignment then documented getter': (
        '''
"""F."""
class W:
    """W."""
    value = property()
    @property
    def value(self):
        """Doc."""
        return 1
''',
        0,
    ),
    'inherited property setter override': (
        '''
"""F."""
from base import Base
class W(Base):
    """W."""
    @Base.value.setter
    def value(self, new):
        pass
''',
        0,
    ),
    'overload set': (
        '''
"""F."""
from typing import overload
@overload
def widen(v: int) -> int: ...
@overload
def widen(v: str) -> str: ...
def widen(v):
    """Doc."""
    return v
''',
        0,
    ),
    'aliased overload set': (
        '''
"""F."""
from typing import overload as _ov
@_ov
def widen(v: int) -> int: ...
def widen(v):
    """Doc."""
    return v
''',
        0,
    ),
    'local decorator named overload': (
        '''
"""F."""
def overload(fn):
    """Doc."""
    return fn
@overload
def thing():
    pass
''',
        1,
    ),
    'conditional alternatives, one undocumented': (
        '''
"""F."""
import sys
if sys.platform == "linux":
    def go():
        pass
else:
    def go():
        """Doc."""
''',
        1,
    ),
    'branch then unconditional documented': (
        '''
"""F."""
import sys
if sys.platform == "linux":
    def go():
        pass
def go():
    """Doc."""
''',
        0,
    ),
    'class method: branch then unconditional': (
        '''
"""F."""
import sys
class W:
    """W."""
    if sys.platform == "linux":
        def go(self):
            pass
    def go(self):
        """Doc."""
''',
        0,
    ),
    'sequential rebinding, later documented': (
        '''
"""F."""
def public():
    pass
def public():
    """Doc."""
''',
        0,
    ),
    'sequential rebinding, later undocumented': (
        '''
"""F."""
def public():
    """Doc."""
def public():
    pass
''',
        1,
    ),
    'match-case definitions': (
        '''
"""F."""
import sys
match sys.platform:
    case "linux":
        def go():
            pass
    case _:
        def go():
            """Doc."""
''',
        1,
    ),
    'fallback class in except': (
        '''
"""F."""
try:
    from x import Thing
except ImportError:
    class Thing:
        def go(self):
            pass
''',
        2,
    ),
    'finally dominates try': (
        '''
"""F."""
try:
    def public():
        pass
finally:
    def public():
        """Doc."""
''',
        0,
    ),
    'try-else supersedes try body': (
        '''
"""F."""
try:
    import x
    def public():
        pass
except ImportError:
    def public():
        """Doc."""
else:
    def public():
        """Doc."""
''',
        0,
    ),
    'TYPE_CHECKING guard': (
        '''
"""F."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    class TypeOnly:
        def go(self):
            pass
''',
        0,
    ),
    'aliased TYPE_CHECKING guard': (
        '''
"""F."""
from typing import TYPE_CHECKING as TC
if TC:
    class TypeOnly:
        def go(self):
            pass
''',
        0,
    ),
    'qualified TYPE_CHECKING guard': (
        '''
"""F."""
import typing
if typing.TYPE_CHECKING:
    def type_only():
        pass
''',
        0,
    ),
    'TYPE_CHECKING else branch still scanned': (
        '''
"""F."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    def go():
        """Doc."""
else:
    def go():
        pass
''',
        1,
    ),
    'lambda replaces documented def': (
        '''
"""F."""
def public():
    """Doc."""
public = lambda: None
''',
        1,
    ),
    'module-level alias of an undocumented function': (
        '''
"""F."""
def _impl():
    pass
public = _impl
''',
        1,
    ),
    'module-level alias of a documented function': (
        '''
"""F."""
def _impl():
    """Doc."""
public = _impl
''',
        0,
    ),
    'value replaces documented def': (
        '''
"""F."""
def public():
    """Doc."""
public = 3
''',
        0,
    ),
    'conditional class, undocumented method early branch': (
        '''
"""F."""
import sys
if sys.version_info < (3, 13):
    class P:
        """Doc."""
        def go(self):
            pass
else:
    class P:
        """Doc."""
        def go(self):
            """Doc."""
''',
        1,
    ),
    'function in else hides class method in if': (
        '''
"""F."""
import sys
if sys.version_info < (3, 13):
    class P:
        """Doc."""
        def go(self):
            pass
else:
    def P():
        """Doc."""
''',
        1,
    ),
    'class docstring not stolen across bodies': (
        '''
"""F."""
import sys
if sys.platform == "linux":
    class W:
        """W."""
        value = property()
else:
    class W:
        """W."""
        value = property()
        """Doc."""
''',
        1,
    ),
    'superseded class body not scanned': (
        '''
"""F."""
class Public:
    """Doc."""
    def go(self):
        pass
class Public:
    """Doc."""
    def go(self):
        """Doc."""
''',
        0,
    ),
    'bare annotation does not rebind': (
        '''
"""F."""
from typing import Callable
def public():
    pass
public: Callable[..., None]
''',
        1,
    ),
    'getter replaced by a lambda': (
        '''
"""F."""
class W:
    """W."""
    def _g(self):
        """Doc."""
        return 1
    _g = lambda self: 1
    value = property(_g)
''',
        1,
    ),
    'constant alias is not a function': (
        '''
"""F."""
VERSION = "1.0"
MIN_VERSION = VERSION
''',
        0,
    ),
    'exhaustive if/else shadows, both documented': (
        '''
"""F."""
import sys
def public():
    pass
if sys.platform == "linux":
    def public():
        """Doc."""
else:
    def public():
        """Doc."""
''',
        0,
    ),
    'if without else does not shadow': (
        '''
"""F."""
import sys
def public():
    pass
if sys.platform == "linux":
    def public():
        """Doc."""
''',
        1,
    ),
    'chained assignment rebinds both targets': (
        '''
"""F."""
def public():
    """Doc."""
public = alias = lambda: None
''',
        2,
    ),
    'alias chain of an undocumented function': (
        '''
"""F."""
def _impl():
    pass
_alias = _impl
public = _alias
''',
        1,
    ),
    'alias chain of a documented function': (
        '''
"""F."""
def _impl():
    """Doc."""
_alias = _impl
public = _alias
''',
        0,
    ),
    'whitespace-only docstring is no docstring': (
        '''
"""F."""
def public():
    """   """
''',
        1,
    ),
    'doc=None falls back to the getter': (
        '''
"""F."""
class W:
    """W."""
    def _g(self):
        """Doc."""
        return 1
    value = property(_g, doc=None)
''',
        0,
    ),
    'five-hop alias chain': (
        '''
"""F."""
def _impl():
    pass
_a = _impl
_b = _a
_c = _b
_d = _c
public = _d
''',
        1,
    ),
    'project flag named TYPE_CHECKING is not a guard': (
        '''
"""F."""
import flags
if flags.TYPE_CHECKING:
    def public():
        pass
''',
        1,
    ),
    'typing.TYPE_CHECKING is a guard': (
        '''
"""F."""
import typing
if typing.TYPE_CHECKING:
    def public():
        pass
''',
        0,
    ),
    'exhaustive if/else shadows a getter': (
        '''
"""F."""
import sys
class W:
    """W."""
    def _g(self):
        return 1
    if sys.platform == "linux":
        def _g(self):
            """Doc."""
            return 1
    else:
        def _g(self):
            """Doc."""
            return 1
    value = property(_g)
''',
        0,
    ),
    'nested non-exhaustive if does not shadow': (
        '''
"""F."""
import sys
def public():
    pass
if sys.platform == "linux":
    if sys.maxsize > 2**32:
        def public():
            """Doc."""
else:
    def public():
        """Doc."""
''',
        1,
    ),
    'nested exhaustive if does shadow': (
        '''
"""F."""
import sys
def public():
    pass
if sys.platform == "linux":
    if sys.maxsize > 2**32:
        def public():
            """Doc."""
    else:
        def public():
            """Doc."""
else:
    def public():
        """Doc."""
''',
        0,
    ),
    'unpacking rebinds a documented def': (
        '''
"""F."""
def public():
    """Doc."""
public, other = (lambda: None), 0
''',
        0,
    ),
    'alias of an undocumented local class': (
        '''
"""F."""
class _Implementation:
    def go(self):
        pass
Public = _Implementation
''',
        2,
    ),
    'alias of a documented local class': (
        '''
"""F."""
class _Implementation:
    """Doc."""
    def go(self):
        """Doc."""
Public = _Implementation
''',
        0,
    ),
    'override of an undocumented local base property': (
        '''
"""F."""
class _Base:
    @property
    def value(self):
        return 1
class Public(_Base):
    """Doc."""
    @_Base.value.setter
    def value(self, new):
        pass
''',
        1,
    ),
    'override of a documented local base property': (
        '''
"""F."""
class _Base:
    @property
    def value(self):
        """Doc."""
        return 1
class Public(_Base):
    """Doc."""
    @_Base.value.setter
    def value(self, new):
        pass
''',
        0,
    ),
}


@pytest.mark.parametrize("label", list(CASES))
def test_docstring_gap_count(tmp_path, label):
    source, expected = CASES[label]
    package = tmp_path / "fixturepkg"
    package.mkdir()
    (package / "__init__.py").write_text(source, encoding="utf-8")

    gaps, total = docstring_gaps.collect(package)

    assert len(gaps) == expected, docstring_gaps.format_report(gaps, total)


def test_rejects_a_path_that_is_not_a_package(tmp_path):
    with pytest.raises(SystemExit):
        docstring_gaps.main(["--package", str(tmp_path)])


def test_max_gaps_is_a_ratchet(tmp_path):
    package = tmp_path / "fixturepkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Fixture."""\n\n\ndef public():\n    pass\n', encoding="utf-8"
    )
    arguments = ["--package", str(package)]

    assert docstring_gaps.main([*arguments, "--max-gaps", "1"]) == 0
    assert docstring_gaps.main([*arguments, "--max-gaps", "0"]) == 1

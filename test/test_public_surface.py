"""Guard the top-level public surface of :mod:`volumential`.

These checks are import-level only: they pin the names that
``volumential/__init__.py`` and ``volumential/version.py`` promise, so that
reorganizing the package cannot silently drop one.
"""

__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import pytest

import volumential
import volumential.version
from volumential.nearfield_potential_table import NearFieldInteractionTable
from volumential.singular_integral_2d import box_quad
from volumential.table_manager import NearFieldInteractionTableManager


EXPECTED_TOP_LEVEL_NAMES = (
    "CACHING_ENABLED",
    "CacheMode",
    "NearFieldInteractionTable",
    "NearFieldInteractionTableManager",
    "OPT_ENABLED",
    "box_quad",
    "code_cache",
    "nearfield_potential_table",
    "set_caching_enabled",
    "set_optimization_enabled",
    "volumential_version",
)


def test_all_is_declared_and_complete():
    assert set(EXPECTED_TOP_LEVEL_NAMES) <= set(volumential.__all__)


@pytest.mark.parametrize("name", EXPECTED_TOP_LEVEL_NAMES)
def test_exported_name_resolves(name):
    assert hasattr(volumential, name)


def test_reexports_match_their_defining_modules():
    assert volumential.NearFieldInteractionTable is NearFieldInteractionTable
    assert (
        volumential.NearFieldInteractionTableManager
        is NearFieldInteractionTableManager
    )
    assert volumential.box_quad is box_quad


def test_version_metadata():
    version = volumential.version

    assert version.VERSION_TEXT.startswith(
        ".".join(str(part) for part in version.VERSION)
    )
    assert volumential.volumential_version == version.VERSION_TEXT
    assert version.KERNEL_VERSION[0] == version.VERSION
    assert set(version.__all__) == {
        "KERNEL_VERSION",
        "LOOPY_LANG_VERSION",
        "VERSION",
        "VERSION_STATUS",
        "VERSION_TEXT",
    }


def test_caching_switches_round_trip():
    original = volumential.CACHING_ENABLED
    try:
        volumential.set_caching_enabled(not original)
        assert volumential.CACHING_ENABLED is (not original)

        with volumential.CacheMode(original):
            assert volumential.CACHING_ENABLED is original
        assert volumential.CACHING_ENABLED is (not original)
    finally:
        volumential.set_caching_enabled(original)

    assert volumential.CACHING_ENABLED is original


def test_optimization_switch_round_trip():
    original = volumential.OPT_ENABLED
    try:
        volumential.set_optimization_enabled(not original)
        assert volumential.OPT_ENABLED is (not original)
    finally:
        volumential.set_optimization_enabled(original)

    assert volumential.OPT_ENABLED is original

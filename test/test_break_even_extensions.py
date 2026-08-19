"""Tests for the E3 extensions of the break-even driver: the lazy-direct
provisioning strategy and the operation-counter summary columns."""

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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_break_even():
    path = _REPOSITORY_ROOT / "benchmarks" / "break_even_validation.py"
    spec = importlib.util.spec_from_file_location("break_even_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    finally:
        sys.path[:] = previous_sys_path
    return module


def test_eager_provisioning_reproduces_committed_levels():
    module = _load_break_even()
    assert module._resolve_direct_levels(
        smoke=True, provisioning="eager", nlevels=2
    ) == [1, 2]
    assert module._resolve_direct_levels(
        smoke=False, provisioning="eager", nlevels=5
    ) == [0, 1, 2, 3, 4, 5]


def test_lazy_provisioning_builds_only_the_touched_level():
    module = _load_break_even()
    # only the leaf level owns List 1 work on the uniform benchmark tree
    assert module._resolve_direct_levels(
        smoke=True, provisioning="lazy", nlevels=2
    ) == [2]
    assert module._resolve_direct_levels(
        smoke=False, provisioning="lazy", nlevels=5
    ) == [5]


def test_unknown_provisioning_is_rejected():
    module = _load_break_even()
    with pytest.raises(ValueError, match="provisioning"):
        module._resolve_direct_levels(
            smoke=True, provisioning="opportunistic", nlevels=2
        )


def test_summary_fields_extend_the_committed_layout():
    module = _load_break_even()
    fields = list(module.SUMMARY_FIELDS)
    # append-only contract: the historical columns keep their positions
    assert fields.index("mode") == 0
    assert fields.index("benchmark_total_s") < fields.index(
        "direct_provisioning"
    )
    for name in (
        "direct_provisioning",
        "ops_reduced_entries_per_table",
        "ops_direct_build_routing",
        "ops_direct_singular_node_evals",
        "ops_direct_special_function",
        "ops_rke_channel_singular_node_evals",
        "ops_nearfield_point_pairs_per_solve",
        "ops_split_series_nmax_per_parameter",
    ):
        assert name in fields


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

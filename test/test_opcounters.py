"""Tests for the explicit operation counters (Paper 1, experiment E3).

The analytic-count helpers are pinned against the constants of the ops cost
model (kb: paper1-ops-cost-model), which were derived by executing the node
builders and entry enumerations at the artifact commit; the instrumented
counters are then checked to reproduce those analytic counts in situ.
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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pytest

import volumential.opcounters as oc
from volumential.rke_table_assembly import (
    _duffy_channel_entry_values,
    _normalized_windowed_channel_profile,
    _smooth_remainder_entry_values,
    _windowed_channel_skeleton,
    windowed_remainder_profile,
)


ROOT_EXTENT = 2.0
WINDOW_THETA = 16.0


# {{{ counter mechanics

def test_add_is_noop_without_active_counters():
    # must not raise and must not leak state anywhere
    oc.add(oc.SINGULAR_NODES, "duffy_radial", 123)


def test_counting_context_collects_and_nests():
    outer = oc.OpCounters()
    inner = oc.OpCounters()
    with oc.counting(outer):
        oc.add(oc.SMOOTH_NODES, "tensor_gauss", 5)
        with oc.counting(inner):
            oc.add(oc.SMOOTH_NODES, "tensor_gauss", 7)
    oc.add(oc.SMOOTH_NODES, "tensor_gauss", 1000)  # inactive: dropped

    assert outer.total(oc.SMOOTH_NODES) == 12
    assert inner.total(oc.SMOOTH_NODES) == 7
    assert outer.labels(oc.SMOOTH_NODES) == {"tensor_gauss": 12}


def test_counting_context_recovers_after_exception():
    counters = oc.OpCounters()
    with pytest.raises(RuntimeError):
        with oc.counting(counters):
            raise RuntimeError("boom")
    oc.add(oc.SMOOTH_NODES, "tensor_gauss", 3)
    assert counters.total(oc.SMOOTH_NODES) == 0


@pytest.mark.parametrize("bad", [True, 1.5, -1, "3"])
def test_counts_must_be_nonnegative_integers(bad):
    counters = oc.OpCounters()
    with pytest.raises(ValueError):
        counters.add(oc.SMOOTH_NODES, "tensor_gauss", bad)


def test_by_function_listing_is_sorted_and_compact():
    counters = oc.OpCounters()
    counters.add(oc.SPECIAL_EVALS, "k0", 3)
    counters.add(oc.SPECIAL_EVALS, "expn", 10)
    counters.add(oc.SPECIAL_EVALS, "k0", np.int64(4))
    assert counters.by_function(oc.SPECIAL_EVALS) == "expn:10;k0:7"


def test_counting_rejects_non_counter():
    with pytest.raises(ValueError):
        with oc.counting({}):
            pass

# }}}


# {{{ analytic helpers against the ops-model constants

def test_surviving_radial_node_counts_match_ops_model():
    # kb: paper1-ops-cost-model, "Surviving radial nodes"
    expected = {
        20: 29, 21: 29, 31: 35, 40: 40, 45: 43, 61: 49,
        101: 64, 121: 70, 141: 75, 160: 80, 161: 81, 201: 90, 320: 114,
    }
    for order, count in expected.items():
        assert oc.surviving_radial_node_count(order) == count


def test_batched_nodes_per_entry_match_ops_model():
    # kb: paper1-ops-cost-model, "(b) singular-quadrature node evaluations"
    assert oc.batched_duffy_nodes_per_entry(2, 16, 40) == 5120
    assert oc.batched_duffy_nodes_per_entry(2, 24, 61) == 9408
    assert oc.batched_duffy_nodes_per_entry(2, 48, 160) == 30720
    assert oc.batched_duffy_nodes_per_entry(3, 16, 45) == 528384
    assert oc.batched_duffy_nodes_per_entry(3, 24, 61) == 1354752


@pytest.mark.parametrize(
    ("dim", "q_order", "n_rep", "n_blocks", "n_regions"),
    [
        (2, 3, 357, 63, 165),
        (2, 4, 1084, 112, 296),
        (3, 2, 256, 74, 1044),
    ],
)
def test_duffy_block_geometry_matches_ops_model(
    dim, q_order, n_rep, n_blocks, n_regions
):
    # kb: paper1-ops-cost-model, "Block and region counts"
    table = _windowed_channel_skeleton(
        dim, q_order, 0, ROOT_EXTENT, WINDOW_THETA, 0
    )
    geometry = oc.duffy_block_geometry(table)
    assert geometry["n_reduced_entries"] == n_rep
    assert geometry["n_blocks"] == n_blocks
    assert geometry["n_active_regions"] == n_regions
    assert oc.reduced_entry_count(table) == n_rep
    # blocks partition the reduced entries and every region belongs to a
    # block, so the per-entry weighting dominates the per-block count
    assert geometry["entry_weighted_active_regions"] >= n_regions


def test_entry_weighted_regions_match_ops_model_mean():
    # kb: paper1-ops-cost-model: scalar 2D q=4 entry-weighted mean 2.54 of 4
    table = _windowed_channel_skeleton(2, 4, 0, ROOT_EXTENT, WINDOW_THETA, 0)
    geometry = oc.duffy_block_geometry(table)
    mean = geometry["entry_weighted_active_regions"] / geometry[
        "n_reduced_entries"
    ]
    assert mean == pytest.approx(2.54, abs=0.01)

# }}}


# {{{ instrumented counts reproduce the analytic counts in situ

def test_channel_build_counts_match_grouped_analytic():
    dim, q_order, level = 2, 2, 3
    regular_order, radial_order = 8, 21
    table = _windowed_channel_skeleton(
        dim, q_order, level, ROOT_EXTENT, WINDOW_THETA, 0
    )
    geometry = oc.duffy_block_geometry(table)
    window_scale = (ROOT_EXTENT * 0.5**level / WINDOW_THETA) ** 2

    counters = oc.OpCounters()
    with oc.counting(counters):
        _duffy_channel_entry_values(
            table,
            _normalized_windowed_channel_profile(dim, 0, window_scale),
            regular_order,
            radial_order,
        )

    analytic = oc.grouped_duffy_singular_nodes(
        table, regular_order, radial_order, geometry=geometry
    )
    assert counters.total(oc.SINGULAR_NODES) == analytic
    # 2D channel profiles cost one expn evaluation per singular node
    assert counters.labels(oc.SPECIAL_EVALS)["expn"] == analytic
    assert counters.total(oc.PROFILE_NODES) == analytic


def test_smooth_remainder_counts_match_analytic():
    dim, q_order, level, p_star = 2, 2, 3, 3
    smooth_quad_order = 6
    table = _windowed_channel_skeleton(
        dim, q_order, level, ROOT_EXTENT, WINDOW_THETA, 0
    )
    geometry = oc.duffy_block_geometry(table)
    window_scale = (ROOT_EXTENT * 0.5**level / WINDOW_THETA) ** 2
    # a bare skeleton stores no data, so ask the invariant enumeration
    entry_ids = np.asarray(
        table._get_invariant_entry_info()["entry_ids"], dtype=np.int64
    )

    kernel_calls = oc.OpCounters()

    def kernel_radial(r):
        oc.add(oc.KERNEL_EVALS, "test_kernel", np.asarray(r).size)
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    remainder = windowed_remainder_profile(
        dim, 4.0, kernel_radial, window_scale, p_star
    )
    with oc.counting(kernel_calls):
        _smooth_remainder_entry_values(
            table, entry_ids, remainder, smooth_quad_order
        )

    smooth_nodes = geometry["n_blocks"] * smooth_quad_order**dim
    assert kernel_calls.total(oc.SMOOTH_NODES) == smooth_nodes
    assert kernel_calls.labels(oc.KERNEL_EVALS)["test_kernel"] == smooth_nodes
    # each smooth node passes through all p_star channel profiles
    assert kernel_calls.total(oc.PROFILE_NODES) == p_star * smooth_nodes

# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

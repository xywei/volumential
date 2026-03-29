__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

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

import logging
import os
import subprocess
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

if (
    sys.platform == "darwin"
    and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
):
    pytest.skip(
        "volume FMM tests are unstable on macOS OpenCL CI "
        "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)",
        allow_module_level=True,
    )

import pyopencl as cl
import pyopencl.array  # noqa: F401

import volumential.meshgen as mg


logger = logging.getLogger(__name__)


class _FakeDeviceArray:
    def __init__(self, ary):
        self._ary = np.asarray(ary)

    def get(self, queue=None):
        return self._ary


def test_list1_gallery_includes_mixed_source_levels():
    from volumential.list1_gallery import generate_interactions

    interactions = generate_interactions(2)
    source_radii = {float(sbox.radius) for _, sbox in interactions}

    assert len(interactions) > 0
    assert len(source_radii) > 1


def test_validate_table_box_particle_layout_accepts_q_order_layout():
    from volumential.expansion_wrangler_fpnd import _validate_table_box_particle_layout

    tree = SimpleNamespace(
        box_target_counts_nonchild=_FakeDeviceArray(np.array([16, 0, 16, 16]))
    )

    _validate_table_box_particle_layout(
        queue=None,
        tree=tree,
        target_boxes=np.array([0, 2], dtype=np.int32),
        source_boxes=np.array([0, 2, 3], dtype=np.int32),
        n_q_points=16,
    )


def test_validate_table_box_particle_layout_rejects_non_q_order_layout():
    from volumential.expansion_wrangler_fpnd import _validate_table_box_particle_layout

    tree = SimpleNamespace(
        box_target_counts_nonchild=_FakeDeviceArray(np.array([16, 24, 16, 31]))
    )

    with pytest.raises(ValueError, match="requires exactly 16 quadrature points"):
        _validate_table_box_particle_layout(
            queue=None,
            tree=tree,
            target_boxes=np.array([0, 1], dtype=np.int32),
            source_boxes=np.array([0, 3], dtype=np.int32),
            n_q_points=16,
        )


def test_validate_table_box_particle_layout_cached_reuses_result(monkeypatch):
    import volumential.expansion_wrangler_fpnd as wrangler_mod

    calls = []

    def _fake_validate(queue, tree, target_boxes, source_boxes, n_q_points):
        calls.append((id(target_boxes), id(source_boxes), int(n_q_points)))

    monkeypatch.setattr(
        wrangler_mod, "_validate_table_box_particle_layout", _fake_validate
    )

    target_boxes = np.array([0, 1], dtype=np.int32)
    source_boxes = np.array([1, 2, 3], dtype=np.int32)
    cache = set()

    wrangler_mod._validate_table_box_particle_layout_cached(
        queue=None,
        tree=object(),
        target_boxes=target_boxes,
        source_boxes=source_boxes,
        n_q_points=16,
        validation_cache=cache,
    )
    wrangler_mod._validate_table_box_particle_layout_cached(
        queue=None,
        tree=object(),
        target_boxes=target_boxes,
        source_boxes=source_boxes,
        n_q_points=16,
        validation_cache=cache,
    )

    assert len(calls) == 1


def test_validate_table_box_particle_layout_cached_separates_q_order(monkeypatch):
    import volumential.expansion_wrangler_fpnd as wrangler_mod

    calls = []

    def _fake_validate(queue, tree, target_boxes, source_boxes, n_q_points):
        calls.append(int(n_q_points))

    monkeypatch.setattr(
        wrangler_mod, "_validate_table_box_particle_layout", _fake_validate
    )

    target_boxes = np.array([0], dtype=np.int32)
    source_boxes = np.array([0], dtype=np.int32)
    cache = set()

    wrangler_mod._validate_table_box_particle_layout_cached(
        queue=None,
        tree=object(),
        target_boxes=target_boxes,
        source_boxes=source_boxes,
        n_q_points=16,
        validation_cache=cache,
    )
    wrangler_mod._validate_table_box_particle_layout_cached(
        queue=None,
        tree=object(),
        target_boxes=target_boxes,
        source_boxes=source_boxes,
        n_q_points=25,
        validation_cache=cache,
    )

    assert calls == [16, 25]


def test_rebuild_tob_from_geometry_restores_unit_level_edges():
    from boxtree import make_tree_of_boxes_root, refine_and_coarsen_tree_of_boxes

    from volumential.tree_interactive_build import _rebuild_tob_from_geometry

    tob = make_tree_of_boxes_root((np.array([0.0, 0.0]), np.array([1.0, 1.0])))

    # Step 1: split root -> level-1 leaves
    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[0] = True
    tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)

    # Step 2: split one level-1 leaf -> introduces level-2 leaves
    level1_leaves = np.where(
        (tob.box_levels == 1) & (np.all(tob.box_child_ids == 0, axis=0))
    )[0]
    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[int(level1_leaves[0])] = True
    tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)

    # Step 3: split one level-2 leaf -> introduces level-3 leaves
    level2_leaves = np.where(
        (tob.box_levels == 2) & (np.all(tob.box_child_ids == 0, axis=0))
    )[0]
    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[int(level2_leaves[0])] = True
    tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)

    # Step 4: split another level-1 leaf -> triggers index corruption in upstream
    level1_leaves = np.where(
        (tob.box_levels == 1) & (np.all(tob.box_child_ids == 0, axis=0))
    )[0]
    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[int(level1_leaves[0])] = True
    tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)

    levels = np.asarray(tob.box_levels, dtype=np.int32)
    bad_edges_before = 0
    for parent in range(tob.nboxes):
        plevel = int(levels[parent])
        for child in tob.box_child_ids[:, parent]:
            child = int(child)
            if child != 0 and int(levels[child]) != plevel + 1:
                bad_edges_before += 1

    if bad_edges_before == 0:
        pytest.skip("boxtree no longer reproduces the invalid-edge topology")

    repaired = _rebuild_tob_from_geometry(tob)
    repaired_levels = np.asarray(repaired.box_levels, dtype=np.int32)
    bad_edges_after = 0
    for parent in range(repaired.nboxes):
        plevel = int(repaired_levels[parent])
        for child in repaired.box_child_ids[:, parent]:
            child = int(child)
            if child != 0 and int(repaired_levels[child]) != plevel + 1:
                bad_edges_after += 1

    assert bad_edges_after == 0
    assert np.all(repaired_levels[:-1] <= repaired_levels[1:])

    level_starts = np.asarray(repaired.level_start_box_nrs, dtype=np.int32)
    for lev in range(len(level_starts) - 1):
        start = int(level_starts[lev])
        stop = int(level_starts[lev + 1])
        if start == stop:
            continue
        assert np.all(repaired_levels[start:stop] == lev)


def test_normalize_source_fields_accepts_field_major_matrix():
    from volumential.volume_fmm import _normalize_source_fields

    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    normalized = _normalize_source_fields(
        values,
        np.float64,
        expected_length=4,
        field_name="src_weights",
    )

    assert len(normalized) == 3
    assert np.allclose(normalized[0], values[0])
    assert np.allclose(normalized[2], values[2])


def test_normalize_source_fields_accepts_column_vector_single_field():
    from volumential.volume_fmm import _normalize_source_fields

    values = np.arange(4, dtype=np.float64).reshape(4, 1)
    normalized = _normalize_source_fields(
        values,
        np.float64,
        expected_length=4,
        field_name="src_weights",
    )

    assert len(normalized) == 1
    assert np.allclose(normalized[0], values[:, 0])


def test_normalize_source_fields_treats_scalar_sequence_as_single_field():
    from volumential.volume_fmm import _normalize_source_fields

    values = [1.0, 2.0, 3.0, 4.0]
    normalized = _normalize_source_fields(
        values,
        np.float64,
        expected_length=4,
        field_name="src_weights",
    )

    assert len(normalized) == 1
    assert np.allclose(normalized[0], np.array(values, dtype=np.float64))


def test_normalize_source_fields_rejects_point_major_matrix():
    from volumential.volume_fmm import _normalize_source_fields

    values = np.arange(12, dtype=np.float64).reshape(4, 3)

    with pytest.raises(ValueError, match="point-major shape"):
        _normalize_source_fields(
            values,
            np.float64,
            expected_length=4,
            field_name="src_weights",
        )


def test_normalize_source_fields_casts_object_matrix_rows():
    from volumential.volume_fmm import _normalize_source_fields

    values = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=object)
    normalized = _normalize_source_fields(
        values,
        np.float64,
        expected_length=3,
        field_name="src_weights",
    )

    assert len(normalized) == 2
    assert normalized[0].dtype == np.float64
    assert normalized[1].dtype == np.float64
    assert np.allclose(normalized[0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(normalized[1], np.array([10.0, 20.0, 30.0]))


def test_build_box_mode_to_source_ids_raises_on_unmatched_nodes(monkeypatch):
    import volumential.volume_fmm as volume_fmm

    monkeypatch.setattr(volume_fmm.cl.array, "to_device", lambda queue, ary: ary)

    q_order = 2
    traversal = SimpleNamespace(
        target_boxes=_FakeDeviceArray(np.array([0], dtype=np.int32))
    )
    tree = SimpleNamespace(
        dimensions=2,
        root_extent=1.0,
        box_target_starts=_FakeDeviceArray(np.array([0], dtype=np.int32)),
        box_target_counts_cumul=_FakeDeviceArray(np.array([4], dtype=np.int32)),
        box_levels=_FakeDeviceArray(np.array([0], dtype=np.int32)),
        box_centers=_FakeDeviceArray(np.array([[0.5], [0.5]], dtype=np.float64)),
        sources=np.empty(2, dtype=object),
    )

    # One coordinate is deliberately off the interpolation node by O(1e-1),
    # which should trigger strict mapping failure.
    tree.sources[0] = _FakeDeviceArray(np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float64))
    tree.sources[1] = _FakeDeviceArray(np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="could not build mode-to-source ids"):
        volume_fmm._build_box_mode_to_source_ids(
            tree=tree,
            traversal=traversal,
            q_order=q_order,
            interp_points_1d=np.array([0.0, 1.0], dtype=np.float64),
            queue=None,
        )


def test_build_box_mode_to_source_ids_accepts_float32_roundoff(monkeypatch):
    import volumential.volume_fmm as volume_fmm

    monkeypatch.setattr(volume_fmm.cl.array, "to_device", lambda queue, ary: ary)

    q_order = 2
    traversal = SimpleNamespace(
        target_boxes=_FakeDeviceArray(np.array([0], dtype=np.int32))
    )
    tree = SimpleNamespace(
        dimensions=2,
        root_extent=np.float32(1.0),
        box_target_starts=_FakeDeviceArray(np.array([0], dtype=np.int32)),
        box_target_counts_cumul=_FakeDeviceArray(np.array([4], dtype=np.int32)),
        box_levels=_FakeDeviceArray(np.array([0], dtype=np.int32)),
        box_centers=_FakeDeviceArray(np.array([[0.5], [0.5]], dtype=np.float32)),
        sources=np.empty(2, dtype=object),
    )

    # Valid tensor-product nodes in float32 with small coordinate drift that is
    # larger than 1e-12 but should be accepted for float32 matching.
    drift = np.float32(5.0e-7)
    tree.sources[0] = _FakeDeviceArray(
        np.array([0.0 + drift, 0.0 + drift, 1.0 - drift, 1.0 - drift], dtype=np.float32)
    )
    tree.sources[1] = _FakeDeviceArray(
        np.array([0.0 + drift, 1.0 - drift, 0.0 + drift, 1.0 - drift], dtype=np.float32)
    )

    mode_to_source = volume_fmm._build_box_mode_to_source_ids(
        tree=tree,
        traversal=traversal,
        q_order=q_order,
        interp_points_1d=np.array([0.0, 1.0], dtype=np.float32),
        queue=None,
    )

    np.testing.assert_array_equal(mode_to_source, np.arange(4, dtype=np.int32))


def test_build_source_only_wrangler_preserves_self_extra_kwargs(monkeypatch):
    import volumential.volume_fmm as volume_fmm

    class _DummyBoxtreeActx:
        def __init__(self, queue):
            self.queue = queue

    class _DummyTraversalBuilder:
        def __init__(self, actx):
            self.actx = actx

        def __call__(self, actx, source_tree):
            return SimpleNamespace(tree=source_tree), None

    monkeypatch.setattr("boxtree.array_context.PyOpenCLArrayContext", _DummyBoxtreeActx)
    monkeypatch.setattr("boxtree.traversal.FMMTraversalBuilder", _DummyTraversalBuilder)

    source_tree = object()
    monkeypatch.setattr(
        volume_fmm,
        "_clone_source_side_tree_as_targets",
        lambda tree, queue: source_tree,
    )

    class _MockWrangler:
        def __init__(self, **kwargs):
            if kwargs:
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            self.tree_indep = object()
            self.near_field_table = object()
            self.dtype = np.float64
            self.quad_order = 4
            self.potential_kind = 1
            self.source_extra_kwargs = {"source_scale": 2.0}
            self.kernel_extra_kwargs = {"kernel_scale": 3.0}
            self.self_extra_kwargs = {
                "exclude_self": True,
                "target_to_source": np.array([0, 1], dtype=np.int32),
            }
            self.list1_extra_kwargs = {"case_encoding_bias": 1.0e-10}
            self.level_orders = (4, 5)
            self.translation_classes_data = object()
            self.preprocessed_mpole_dtype = np.float32

    queue = object()
    traversal = SimpleNamespace(tree=object())
    wrangler = _MockWrangler()

    source_traversal, source_wrangler = volume_fmm._build_source_only_wrangler(
        traversal, wrangler, queue
    )

    assert source_traversal.tree is source_tree
    assert source_wrangler.self_extra_kwargs is not wrangler.self_extra_kwargs
    assert source_wrangler.self_extra_kwargs["exclude_self"] is True
    np.testing.assert_array_equal(
        source_wrangler.self_extra_kwargs["target_to_source"],
        wrangler.self_extra_kwargs["target_to_source"],
    )
    assert source_wrangler.list1_extra_kwargs == wrangler.list1_extra_kwargs
    assert source_wrangler.translation_classes_data is wrangler.translation_classes_data
    assert source_wrangler.preprocessed_mpole_dtype == wrangler.preprocessed_mpole_dtype


def test_build_source_only_wrangler_rebuilds_target_to_source_mapping(monkeypatch):
    import volumential.volume_fmm as volume_fmm

    class _DummyBoxtreeActx:
        def __init__(self, queue):
            self.queue = queue

    class _DummyTraversalBuilder:
        def __init__(self, actx):
            self.actx = actx

        def __call__(self, actx, source_tree):
            return SimpleNamespace(tree=source_tree), None

    monkeypatch.setattr("boxtree.array_context.PyOpenCLArrayContext", _DummyBoxtreeActx)
    monkeypatch.setattr("boxtree.traversal.FMMTraversalBuilder", _DummyTraversalBuilder)

    source_tree = SimpleNamespace(ntargets=3)
    monkeypatch.setattr(
        volume_fmm,
        "_clone_source_side_tree_as_targets",
        lambda tree, queue: source_tree,
    )

    class _MockWrangler:
        def __init__(self, **kwargs):
            if kwargs:
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            self.tree_indep = object()
            self.near_field_table = object()
            self.dtype = np.float64
            self.quad_order = 4
            self.potential_kind = 1
            self.source_extra_kwargs = {}
            self.kernel_extra_kwargs = {}
            # Simulate already-processed non-NumPy mapping from original wrangler.
            self.self_extra_kwargs = {
                "exclude_self": True,
                "target_to_source": _FakeDeviceArray(
                    np.array([9, 8, 7], dtype=np.int32)
                ),
            }
            self.list1_extra_kwargs = {}
            self.level_orders = (4,)

    queue = object()
    traversal = SimpleNamespace(tree=object())
    wrangler = _MockWrangler()

    _, source_wrangler = volume_fmm._build_source_only_wrangler(
        traversal, wrangler, queue
    )

    rebuilt = source_wrangler.self_extra_kwargs["target_to_source"]
    assert isinstance(rebuilt, np.ndarray)
    np.testing.assert_array_equal(rebuilt, np.arange(3, dtype=np.int32))


def test_looks_like_coincident_source_target_setup_matches_user_ids():
    import volumential.volume_fmm as volume_fmm

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=4,
        ntargets=4,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([2, 0, 3, 1], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([2, 0, 3, 1], dtype=np.int32)),
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.sources[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.2], dtype=np.float64)
    )
    tree.targets[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.targets[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.2], dtype=np.float64)
    )

    assert volume_fmm._looks_like_coincident_source_target_setup(tree, queue=None)


def test_looks_like_coincident_source_target_setup_rejects_equal_ids_mismatched_coords():
    import volumential.volume_fmm as volume_fmm

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=4,
        ntargets=4,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([2, 0, 3, 1], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([2, 0, 3, 1], dtype=np.int32)),
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.sources[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.2], dtype=np.float64)
    )
    tree.targets[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.targets[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.25], dtype=np.float64)
    )

    assert not volume_fmm._looks_like_coincident_source_target_setup(tree, queue=None)


def test_looks_like_coincident_source_target_setup_rejects_mismatched_user_ids():
    import volumential.volume_fmm as volume_fmm

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=4,
        ntargets=4,
        user_source_ids=_FakeDeviceArray(np.array([0, 1, 2, 3], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([3, 2, 1, 0], dtype=np.int32)),
    )

    assert not volume_fmm._looks_like_coincident_source_target_setup(tree, queue=None)


def test_looks_like_coincident_source_target_setup_matches_offset_target_ids():
    import volumential.volume_fmm as volume_fmm

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=4,
        ntargets=4,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([2, 0, 3, 1], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([14, 12, 15, 13], dtype=np.int32)),
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.sources[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.2], dtype=np.float64)
    )
    tree.targets[0] = _FakeDeviceArray(
        np.array([0.2, -0.5, 0.8, 0.1], dtype=np.float64)
    )
    tree.targets[1] = _FakeDeviceArray(
        np.array([1.5, -0.2, 0.3, 1.2], dtype=np.float64)
    )

    assert volume_fmm._looks_like_coincident_source_target_setup(tree, queue=None)


def test_looks_like_coincident_source_target_setup_matches_without_user_target_ids():
    import volumential.volume_fmm as volume_fmm

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=3,
        ntargets=3,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([0, 1, 2], dtype=np.int32)),
        user_target_ids=None,
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.sources[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))
    tree.targets[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.targets[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))

    assert volume_fmm._looks_like_coincident_source_target_setup(tree, queue=None)


def test_maybe_guard_coincident_source_target_tree_warns_once(caplog, monkeypatch):
    import volumential.volume_fmm as volume_fmm

    monkeypatch.delenv("VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE", raising=False)
    monkeypatch.setattr(volume_fmm, "_COINCIDENT_TREE_WARNING_EMITTED", False)

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=3,
        ntargets=3,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([0, 1, 2], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([0, 1, 2], dtype=np.int32)),
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.sources[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))
    tree.targets[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.targets[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))

    with caplog.at_level(logging.WARNING):
        assert volume_fmm._maybe_guard_coincident_source_target_tree(tree, queue=None)
        assert volume_fmm._maybe_guard_coincident_source_target_tree(tree, queue=None)

    warning_records = [
        rec for rec in caplog.records if "targets=None" in rec.getMessage()
    ]
    assert len(warning_records) == 1


def test_maybe_guard_coincident_source_target_tree_strict_mode(monkeypatch):
    import volumential.volume_fmm as volume_fmm

    monkeypatch.setenv("VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE", "1")
    monkeypatch.setattr(volume_fmm, "_COINCIDENT_TREE_WARNING_EMITTED", False)

    tree = SimpleNamespace(
        sources_are_targets=False,
        nsources=3,
        ntargets=3,
        dimensions=2,
        user_source_ids=_FakeDeviceArray(np.array([0, 1, 2], dtype=np.int32)),
        user_target_ids=_FakeDeviceArray(np.array([0, 1, 2], dtype=np.int32)),
        sources=np.empty(2, dtype=object),
        targets=np.empty(2, dtype=object),
    )
    tree.sources[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.sources[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))
    tree.targets[0] = _FakeDeviceArray(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    tree.targets[1] = _FakeDeviceArray(np.array([-0.4, 0.5, 0.6], dtype=np.float64))

    with pytest.raises(ValueError, match="targets=None"):
        volume_fmm._maybe_guard_coincident_source_target_tree(tree, queue=None)


def test_treebuilder_targets_none_sets_sources_are_targets(ctx_factory):
    from boxtree import TreeBuilder
    from boxtree.array_context import PyOpenCLArrayContext
    from pytools.obj_array import new_1d as obj_array_1d

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    source_points_host = np.array(
        [
            [-0.25, -0.25, -0.25],
            [-0.25, -0.25, +0.25],
            [-0.25, +0.25, -0.25],
            [-0.25, +0.25, +0.25],
            [+0.25, -0.25, -0.25],
            [+0.25, -0.25, +0.25],
            [+0.25, +0.25, -0.25],
            [+0.25, +0.25, +0.25],
        ],
        dtype=np.float64,
    )
    source_points_t = np.ascontiguousarray(source_points_host.T)
    source_points = obj_array_1d(
        [
            cl.array.to_device(queue, source_points_t[iaxis])
            for iaxis in range(source_points_t.shape[0])
        ]
    )

    tb = TreeBuilder(actx)
    tree_coincident, _ = tb(
        actx,
        particles=source_points,
        targets=None,
        max_particles_in_box=8,
        kind="adaptive-level-restricted",
    )
    tree_split, _ = tb(
        actx,
        particles=source_points,
        targets=source_points,
        max_particles_in_box=8,
        kind="adaptive-level-restricted",
    )

    assert bool(tree_coincident.sources_are_targets)
    assert not bool(tree_split.sources_are_targets)


def _create_non_intel_opencl_context_or_skip():
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    gpu_candidates = []
    fallback_candidates = []
    for platform in platforms:
        if platform.name == "Intel(R) OpenCL":
            continue
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                gpu_candidates.append(device)
            else:
                fallback_candidates.append(device)

    devices = gpu_candidates or fallback_candidates
    if not devices:
        pytest.skip("No non-Intel OpenCL device available for convergence regression")

    return cl.Context(devices=[devices[0]])


def _get_laplace_3d_table(queue, table_path, q_order):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    if q_order <= 2:
        regular_quad_order = 6
        radial_quad_order = 21
    else:
        regular_quad_order = 8
        radial_quad_order = 31

    with NearFieldInteractionTableManager(
        str(table_path), root_extent=2.0, queue=queue
    ) as tm:
        build_config = DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=regular_quad_order,
            radial_quad_order=radial_quad_order,
        )
        table, _ = tm.get_table(
            3,
            "Laplace",
            q_order,
            force_recompute=False,
            queue=queue,
            build_config=build_config,
        )

    return table


def _run_3d_gaussian_case(
    ctx,
    queue,
    table,
    *,
    q_order,
    nlevels,
    fmm_order,
    split_targets,
    return_state=False,
    tree_mode="mesh_aligned",
    alpha=80.0,
):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.volume_fmm import drive_volume_fmm

    dim = 3

    def rhs_f(x, y, z):
        r2 = x * x + y * y + z * z
        return (2 * dim * alpha - 4 * alpha * alpha * r2) * np.exp(-alpha * r2)

    def u_exact(x, y, z):
        return np.exp(-alpha * (x * x + y * y + z * z))

    mesh = mg.MeshGen3D(q_order, nlevels, -0.5, 0.5, queue=queue)

    if tree_mode == "mesh_aligned":
        from dataclasses import replace

        from boxtree.array_context import PyOpenCLArrayContext
        from boxtree.traversal import FMMTraversalBuilder

        from volumential.volume_fmm import _clone_source_side_tree_as_targets

        q_points, source_weights, tree, traversal = mg.build_geometry_info(
            ctx,
            queue,
            dim,
            q_order,
            mesh,
            bbox=np.array([[-0.5, 0.5]] * dim, dtype=np.float64),
        )
        source_coords_host = np.array([coords.get(queue) for coords in q_points])
        source_points_host = np.ascontiguousarray(source_coords_host.T)
        source_vals = cl.array.to_device(
            queue,
            np.ascontiguousarray(
                rhs_f(
                    source_coords_host[0],
                    source_coords_host[1],
                    source_coords_host[2],
                )
            ),
        )

        if split_targets:
            split_tree = _clone_source_side_tree_as_targets(tree, queue)
            if split_tree is None:
                raise ValueError("unable to build split target tree from source tree")

            split_tree = replace(split_tree, sources_are_targets=False)

            actx = PyOpenCLArrayContext(queue)
            tg = FMMTraversalBuilder(actx)
            traversal, _ = tg(actx, split_tree)
            tree = split_tree
    elif tree_mode == "treebuilder":
        from boxtree import TreeBuilder
        from boxtree.array_context import PyOpenCLArrayContext
        from boxtree.traversal import FMMTraversalBuilder
        from pytools.obj_array import new_1d as obj_array_1d

        source_points_host = np.ascontiguousarray(mesh.get_q_points())
        source_weights_host = np.ascontiguousarray(mesh.get_q_weights())
        source_points_t = np.ascontiguousarray(source_points_host.T)

        source_points = obj_array_1d(
            [
                cl.array.to_device(queue, source_points_t[iaxis])
                for iaxis in range(source_points_t.shape[0])
            ]
        )
        source_vals = cl.array.to_device(
            queue,
            np.ascontiguousarray(
                rhs_f(
                    source_points_host[:, 0],
                    source_points_host[:, 1],
                    source_points_host[:, 2],
                )
            ),
        )
        source_weights = cl.array.to_device(queue, source_weights_host)

        actx = PyOpenCLArrayContext(queue)
        tb = TreeBuilder(actx)
        targets = source_points if split_targets else None
        tree, _ = tb(
            actx,
            particles=source_points,
            targets=targets,
            max_particles_in_box=q_order**3 * 8 - 1,
            kind="adaptive-level-restricted",
        )

        tg = FMMTraversalBuilder(actx)
        traversal, _ = tg(actx, tree)
    else:
        raise ValueError(f"unknown tree_mode '{tree_mode}'")

    knl = LaplaceKernel(dim)
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        [knl],
        exclude_self=True,
    )

    self_extra_kwargs = {}
    if tree.sources_are_targets:
        self_extra_kwargs = {
            "target_to_source": np.arange(tree.ntargets, dtype=np.int32)
        }

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
    )

    (potentials,) = drive_volume_fmm(
        traversal,
        wrangler,
        source_vals * source_weights,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )

    exact = u_exact(
        source_points_host[:, 0],
        source_points_host[:, 1],
        source_points_host[:, 2],
    )
    abs_err = np.abs(potentials.get(queue) - exact)
    rel_l2 = np.linalg.norm(abs_err) / np.linalg.norm(exact)

    result = {
        "tree_sources_are_targets": bool(tree.sources_are_targets),
        "n_points": int(source_points_host.shape[0]),
        "root_extent": float(tree.root_extent),
        "rel_l2": float(rel_l2),
        "alpha": alpha,
    }

    if return_state:
        result.update(
            {
                "traversal": traversal,
                "wrangler": wrangler,
                "potentials": potentials,
                "source_vals": source_vals,
                "source_weights": source_weights,
            }
        )

    return result


def test_volume_fmm_strict_guard_rejects_split_tree_source_nodes(tmp_path, monkeypatch):
    ctx = _create_non_intel_opencl_context_or_skip()
    queue = cl.CommandQueue(ctx)

    q_order = 2
    table = _get_laplace_3d_table(
        queue,
        tmp_path / "nft-strict-guard-3d.sqlite",
        q_order,
    )

    monkeypatch.setenv("VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE", "1")
    with pytest.raises(ValueError, match="targets=None"):
        _run_3d_gaussian_case(
            ctx,
            queue,
            table,
            q_order=q_order,
            nlevels=2,
            fmm_order=8,
            split_targets=True,
            tree_mode="treebuilder",
        )


def test_volume_fmm_split_tree_auto_interpolation_matches_manual_backends(
    tmp_path,
    monkeypatch,
):
    import volumential.volume_fmm as volume_fmm

    from pytools.obj_array import new_1d as obj_array_1d

    from volumential.volume_fmm import (
        _build_source_only_wrangler,
        drive_volume_fmm,
    )

    orig_interpolate = volume_fmm.interpolate_volume_potential
    auto_interp_flags = []

    def _record_interpolate(*args, **kwargs):
        auto_interp_flags.append(bool(kwargs.get("use_numpy_interpolation", False)))
        return orig_interpolate(*args, **kwargs)

    monkeypatch.setattr(volume_fmm, "interpolate_volume_potential", _record_interpolate)

    ctx = _create_non_intel_opencl_context_or_skip()
    queue = cl.CommandQueue(ctx)

    q_order = 2
    table = _get_laplace_3d_table(
        queue,
        tmp_path / "nft-split-tree-auto-interp.sqlite",
        q_order,
    )

    split_tree_case = _run_3d_gaussian_case(
        ctx,
        queue,
        table,
        q_order=q_order,
        nlevels=3,
        fmm_order=12,
        split_targets=True,
        return_state=True,
        tree_mode="mesh_aligned",
    )

    assert not split_tree_case["tree_sources_are_targets"]

    split_traversal = split_tree_case["traversal"]
    split_wrangler = split_tree_case["wrangler"]
    source_vals = split_tree_case["source_vals"]
    source_weights = split_tree_case["source_weights"]

    source_traversal, source_wrangler = _build_source_only_wrangler(
        split_traversal,
        split_wrangler,
        queue,
    )
    assert source_traversal.tree.sources_are_targets

    (source_potential_tree_order,) = drive_volume_fmm(
        source_traversal,
        source_wrangler,
        source_vals * source_weights,
        source_vals,
        direct_evaluation=False,
        reorder_sources=True,
        reorder_potentials=False,
        auto_interpolate_targets=False,
    )

    interp_kwargs = {
        "potential_in_tree_order": True,
        "use_mode_to_source_ids": True,
    }
    interp_tree_cl = None
    interp_cl_error = None
    try:
        interp_tree_cl = orig_interpolate(
            split_traversal.tree.targets,
            split_traversal,
            split_wrangler,
            source_potential_tree_order,
            **interp_kwargs,
        )
    except (cl.LogicError, cl.RuntimeError) as err:
        interp_cl_error = err

    interp_tree_numpy = orig_interpolate(
        split_traversal.tree.targets,
        split_traversal,
        split_wrangler,
        source_potential_tree_order,
        use_numpy_interpolation=True,
        **interp_kwargs,
    )

    def _to_user_order_host(potential_tree_order):
        result = split_wrangler.reorder_potentials(obj_array_1d([potential_tree_order]))
        result = split_wrangler.finalize_potentials(result)
        return result[0].get(queue)

    auto_host = split_tree_case["potentials"].get(queue)
    interp_numpy_host = _to_user_order_host(interp_tree_numpy)

    assert np.all(np.isfinite(interp_numpy_host))

    assert auto_interp_flags
    assert auto_interp_flags[0] is False

    if interp_tree_cl is None:
        assert auto_interp_flags[:2] == [False, True], interp_cl_error
        assert np.allclose(auto_host, interp_numpy_host, rtol=1.0e-11, atol=1.0e-11)
    else:
        interp_cl_host = _to_user_order_host(interp_tree_cl)
        assert np.all(np.isfinite(interp_cl_host))
        assert np.allclose(
            interp_cl_host,
            interp_numpy_host,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        assert np.allclose(auto_host, interp_cl_host, rtol=1.0e-11, atol=1.0e-11)


def test_volume_fmm_3d_gaussian_convergence_regression(tmp_path):
    ctx = _create_non_intel_opencl_context_or_skip()
    queue = cl.CommandQueue(ctx)

    q_order = 3
    table = _get_laplace_3d_table(
        queue,
        tmp_path / "nft-convergence-3d.sqlite",
        q_order,
    )

    coarse = _run_3d_gaussian_case(
        ctx,
        queue,
        table,
        q_order=q_order,
        nlevels=3,
        fmm_order=12,
        split_targets=False,
        tree_mode="mesh_aligned",
    )
    fine = _run_3d_gaussian_case(
        ctx,
        queue,
        table,
        q_order=q_order,
        nlevels=4,
        fmm_order=12,
        split_targets=False,
        tree_mode="mesh_aligned",
    )

    err_ratio = fine["rel_l2"] / coarse["rel_l2"]
    observed_order = -np.log2(err_ratio)

    assert coarse["tree_sources_are_targets"]
    assert fine["tree_sources_are_targets"]
    assert abs(coarse["root_extent"] - 1.0) < 1.0e-12
    assert abs(fine["root_extent"] - 1.0) < 1.0e-12
    assert err_ratio < 0.2, (
        "3D regression: rel-L2 error did not improve enough under refinement "
        f"(coarse={coarse['rel_l2']:.3e}, fine={fine['rel_l2']:.3e}, "
        f"ratio={err_ratio:.3f})"
    )
    assert observed_order > 2.0, (
        "3D regression: observed refinement order is too low "
        f"(order={observed_order:.3f}, coarse={coarse['rel_l2']:.3e}, "
        f"fine={fine['rel_l2']:.3e}, points={coarse['n_points']}->{fine['n_points']})"
    )


def test_volume_fmm_3d_calculus_patch_residual_regression(tmp_path):
    from sumpy.point_calculus import CalculusPatch

    from volumential.volume_fmm import interpolate_volume_potential

    ctx = _create_non_intel_opencl_context_or_skip()
    queue = cl.CommandQueue(ctx)

    q_order = 3
    table = _get_laplace_3d_table(
        queue,
        tmp_path / "nft-calculus-patch-3d.sqlite",
        q_order,
    )

    # Use a moderately concentrated manufactured profile so the residual remains
    # sensitive to FMM/interpolation behavior without being dominated by
    # calculus-patch truncation error.
    alpha = 20.0

    coarse = _run_3d_gaussian_case(
        ctx,
        queue,
        table,
        q_order=q_order,
        nlevels=3,
        fmm_order=12,
        split_targets=False,
        return_state=True,
        tree_mode="mesh_aligned",
        alpha=alpha,
    )
    fine = _run_3d_gaussian_case(
        ctx,
        queue,
        table,
        q_order=q_order,
        nlevels=4,
        fmm_order=12,
        split_targets=False,
        return_state=True,
        tree_mode="mesh_aligned",
        alpha=alpha,
    )

    patch = CalculusPatch(center=[0.0, 0.0, 0.0], h=0.3, order=5)
    patch_targets = np.empty(3, dtype=object)
    patch_targets[0] = cl.array.to_device(queue, np.ascontiguousarray(patch.x))
    patch_targets[1] = cl.array.to_device(queue, np.ascontiguousarray(patch.y))
    patch_targets[2] = cl.array.to_device(queue, np.ascontiguousarray(patch.z))

    u_patch_coarse = interpolate_volume_potential(
        patch_targets,
        coarse["traversal"],
        coarse["wrangler"],
        coarse["potentials"],
    ).get(queue)
    u_patch_fine = interpolate_volume_potential(
        patch_targets,
        fine["traversal"],
        fine["wrangler"],
        fine["potentials"],
    ).get(queue)

    alpha = float(coarse["alpha"])

    def rhs_f(x, y, z):
        r2 = x * x + y * y + z * z
        return (6 * alpha - 4 * alpha * alpha * r2) * np.exp(-alpha * r2)

    rho_patch = rhs_f(patch.x, patch.y, patch.z)
    rel_residual_coarse = np.linalg.norm(
        -patch.laplace(u_patch_coarse) - rho_patch
    ) / np.linalg.norm(rho_patch)
    rel_residual_fine = np.linalg.norm(
        -patch.laplace(u_patch_fine) - rho_patch
    ) / np.linalg.norm(rho_patch)

    assert rel_residual_fine < rel_residual_coarse, (
        "3D calculus-patch residual did not improve under refinement "
        f"(coarse={rel_residual_coarse:.3e}, fine={rel_residual_fine:.3e})"
    )
    assert rel_residual_fine < 0.7 * rel_residual_coarse, (
        "3D calculus-patch residual improvement is too weak under refinement "
        f"(coarse={rel_residual_coarse:.3e}, fine={rel_residual_fine:.3e})"
    )
    assert rel_residual_fine < 2.0e-1, (
        "3D calculus-patch residual remains too large after refinement "
        f"(fine={rel_residual_fine:.3e}, coarse={rel_residual_coarse:.3e})"
    )


def test_volume_fmm_list1_multi_source_superposition():
    from volumential.expansion_wrangler_interface import ExpansionWranglerInterface
    from volumential.volume_fmm import drive_volume_fmm

    class _MockWrangler(ExpansionWranglerInterface):
        dtype = np.float64

        def multipole_expansion_zeros(self):
            return None

        def local_expansion_zeros(self):
            return None

        def output_zeros(self):
            return np.array([0.0], dtype=self.dtype)

        def reorder_sources(self, source_array):
            return source_array

        def reorder_targets(self, target_array):
            return target_array

        def reorder_potentials(self, potentials):
            return potentials

        def form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        ):
            return None, None

        def coarsen_multipoles(
            self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
        ):
            return mpoles, None

        def eval_direct(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            mode_coefs,
        ):
            return np.array([np.sum(mode_coefs)], dtype=self.dtype), None

        def multipole_to_local(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def eval_multipoles(
            self,
            level_start_target_box_nrs,
            target_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def form_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        ):
            return local_exps, None

        def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
            return np.array([0.0], dtype=self.dtype), None

        def finalize_potentials(self, potentials):
            return potentials

    traversal = SimpleNamespace(
        tree=SimpleNamespace(nsources=3, ntargets=5),
        level_start_source_box_nrs=np.array([], dtype=np.int32),
        source_boxes=np.array([], dtype=np.int32),
        level_start_source_parent_box_nrs=np.array([], dtype=np.int32),
        source_parent_boxes=np.array([], dtype=np.int32),
        target_boxes=np.array([], dtype=np.int32),
        neighbor_source_boxes_starts=np.array([], dtype=np.int32),
        neighbor_source_boxes_lists=np.array([], dtype=np.int32),
    )

    src0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    src1 = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    src_weights = np.empty(2, dtype=object)
    src_weights[:] = [src0, src1]
    src_func = np.empty(2, dtype=object)
    src_func[:] = [src0, src1]

    result = drive_volume_fmm(
        traversal,
        _MockWrangler(),
        src_weights,
        src_func,
        list1_only=True,
    )

    if isinstance(result, np.ndarray) and result.dtype == object:
        assert len(result) == 1
        result = result[0]

    assert np.allclose(result, np.array([66.0]))


def test_volume_fmm_reorders_src_func_with_source_permutation():
    from volumential.expansion_wrangler_interface import ExpansionWranglerInterface
    from volumential.volume_fmm import drive_volume_fmm

    class _MockWrangler(ExpansionWranglerInterface):
        dtype = np.float64

        def __init__(self):
            self._source_map = np.array([2, 0, 1], dtype=np.int32)
            self._target_map = np.array([3, 1, 4, 0, 2], dtype=np.int32)
            self.reordered_src_weights = None
            self.reordered_src_func = None

        def multipole_expansion_zeros(self):
            return None

        def local_expansion_zeros(self):
            return None

        def output_zeros(self):
            return np.array([0.0], dtype=self.dtype)

        def reorder_sources(self, source_array):
            src = np.asarray(source_array)
            assert src.shape[0] == self._source_map.shape[0]
            return src[self._source_map]

        def reorder_targets(self, target_array):
            tgt = np.asarray(target_array)
            if tgt.shape[0] != self._target_map.shape[0]:
                raise AssertionError(
                    "source fields were reordered with target permutation"
                )
            return tgt[self._target_map]

        def reorder_potentials(self, potentials):
            return potentials

        def form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        ):
            self.reordered_src_weights = np.asarray(src_weights[0]).copy()
            return None, None

        def coarsen_multipoles(
            self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
        ):
            return mpoles, None

        def eval_direct(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            mode_coefs,
        ):
            self.reordered_src_func = np.asarray(mode_coefs).copy()
            return np.array([np.sum(mode_coefs)], dtype=self.dtype), None

        def multipole_to_local(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def eval_multipoles(
            self,
            level_start_target_box_nrs,
            target_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def form_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        ):
            return local_exps, None

        def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
            return np.array([0.0], dtype=self.dtype), None

        def finalize_potentials(self, potentials):
            return potentials

    traversal = SimpleNamespace(
        tree=SimpleNamespace(nsources=3, ntargets=5),
        level_start_source_box_nrs=np.array([], dtype=np.int32),
        source_boxes=np.array([], dtype=np.int32),
        level_start_source_parent_box_nrs=np.array([], dtype=np.int32),
        source_parent_boxes=np.array([], dtype=np.int32),
        target_boxes=np.array([], dtype=np.int32),
        neighbor_source_boxes_starts=np.array([], dtype=np.int32),
        neighbor_source_boxes_lists=np.array([], dtype=np.int32),
    )

    src_weights = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    src_func = np.array([10.0, 20.0, 40.0], dtype=np.float64)

    wrangler = _MockWrangler()
    result = drive_volume_fmm(
        traversal,
        wrangler,
        src_weights,
        src_func,
        list1_only=True,
    )

    assert np.allclose(wrangler.reordered_src_weights, np.array([4.0, 1.0, 2.0]))
    assert np.allclose(wrangler.reordered_src_func, np.array([40.0, 10.0, 20.0]))
    assert np.allclose(result, np.array([70.0]))


def test_volume_fmm_list1_falls_back_to_p2p_on_nonfinite_table_values():
    from volumential.expansion_wrangler_interface import ExpansionWranglerInterface
    from volumential.volume_fmm import drive_volume_fmm

    class _MockWrangler(ExpansionWranglerInterface):
        dtype = np.float64

        def __init__(self):
            self.p2p_calls = 0
            self.p2p_src_weights = None

        def multipole_expansion_zeros(self):
            return None

        def local_expansion_zeros(self):
            return None

        def output_zeros(self):
            return np.zeros(2, dtype=self.dtype)

        def reorder_sources(self, source_array):
            return source_array

        def reorder_targets(self, target_array):
            return target_array

        def reorder_potentials(self, potentials):
            return potentials

        def form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        ):
            return None, None

        def coarsen_multipoles(
            self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
        ):
            return mpoles, None

        def eval_direct(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            mode_coefs,
        ):
            return np.array([np.nan, 1.0], dtype=self.dtype), None

        def eval_direct_p2p(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            src_weights,
        ):
            self.p2p_calls += 1
            self.p2p_src_weights = np.asarray(src_weights[0]).copy()
            return np.array([3.0, 4.0], dtype=self.dtype), None

        def multipole_to_local(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def eval_multipoles(
            self,
            level_start_target_box_nrs,
            target_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def form_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        ):
            return local_exps, None

        def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
            return np.array([0.0], dtype=self.dtype), None

        def finalize_potentials(self, potentials):
            return potentials

    traversal = SimpleNamespace(
        tree=SimpleNamespace(nsources=3, ntargets=2, sources_are_targets=True),
        level_start_source_box_nrs=np.array([], dtype=np.int32),
        source_boxes=np.array([], dtype=np.int32),
        level_start_source_parent_box_nrs=np.array([], dtype=np.int32),
        source_parent_boxes=np.array([], dtype=np.int32),
        target_boxes=np.array([], dtype=np.int32),
        neighbor_source_boxes_starts=np.array([], dtype=np.int32),
        neighbor_source_boxes_lists=np.array([], dtype=np.int32),
    )

    src_weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    src_func = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    wrangler = _MockWrangler()
    result = drive_volume_fmm(
        traversal,
        wrangler,
        src_weights,
        src_func,
        list1_only=True,
        allow_list1_p2p_fallback=True,
    )

    assert wrangler.p2p_calls == 1
    np.testing.assert_array_equal(wrangler.p2p_src_weights, src_weights)
    assert np.allclose(result, np.array([3.0, 4.0]))


def test_volume_fmm_rejects_multi_source_full_sumpy_path(monkeypatch):
    from volumential.expansion_wrangler_interface import ExpansionWranglerInterface
    import volumential.volume_fmm as volume_fmm

    class _MockSumpyWrangler(ExpansionWranglerInterface):
        dtype = np.float64

    monkeypatch.setattr(
        volume_fmm,
        "FPNDSumpyExpansionWrangler",
        _MockSumpyWrangler,
    )

    traversal = SimpleNamespace(tree=SimpleNamespace(nsources=3, ntargets=5))

    src0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    src1 = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    src_weights = np.empty(2, dtype=object)
    src_weights[:] = [src0, src1]
    src_func = np.empty(2, dtype=object)
    src_func[:] = [src0, src1]

    with pytest.raises(NotImplementedError, match="list1_only=True"):
        volume_fmm.drive_volume_fmm(
            traversal,
            _MockSumpyWrangler(),
            src_weights,
            src_func,
        )


def test_volume_fmm_direct_eval_accepts_fmmlib_plain_arrays(monkeypatch):
    from volumential.expansion_wrangler_interface import ExpansionWranglerInterface
    import volumential.volume_fmm as volume_fmm

    class _DummyP2P:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, setup_actx, targets, sources, strengths, **kwargs):
            return (np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64),)

    class _MockFMMLibWrangler(ExpansionWranglerInterface):
        dtype = np.float64

        def __init__(self):
            queue = object()
            self.queue = queue
            self.tree_indep = SimpleNamespace(
                target_kernels=(object(),),
                exclude_self=False,
                _setup_actx=SimpleNamespace(queue=queue),
            )

        def multipole_expansion_zeros(self):
            return None

        def local_expansion_zeros(self):
            return None

        def output_zeros(self):
            return np.zeros(4, dtype=self.dtype)

        def reorder_sources(self, source_array):
            return source_array

        def reorder_targets(self, target_array):
            return target_array

        def reorder_potentials(self, potentials):
            return potentials

        def form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        ):
            return np.array([0.0], dtype=self.dtype), None

        def coarsen_multipoles(
            self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
        ):
            return mpoles, None

        def eval_direct(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            mode_coefs,
        ):
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=self.dtype), None

        def eval_direct_p2p(
            self,
            target_boxes,
            neighbor_sources_starts,
            neighbor_sources_lists,
            src_weights,
        ):
            return np.array([0.5, 0.5, 0.5, 0.5], dtype=self.dtype), None

        def multipole_to_local(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def eval_multipoles(
            self,
            level_start_target_box_nrs,
            target_boxes,
            starts,
            lists,
            mpole_exps,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def form_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        ):
            return np.array([0.0], dtype=self.dtype), None

        def refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        ):
            return local_exps, None

        def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
            return np.array([0.0], dtype=self.dtype), None

        def finalize_potentials(self, potentials):
            return potentials

    class _Traversal(SimpleNamespace):
        def get(self, queue):
            return self

    monkeypatch.setattr(volume_fmm, "FPNDFMMLibExpansionWrangler", _MockFMMLibWrangler)
    monkeypatch.setattr(volume_fmm.cl.array, "to_device", lambda queue, ary: ary)
    monkeypatch.setitem(sys.modules, "sumpy", SimpleNamespace(P2P=_DummyP2P))

    traversal = _Traversal(
        tree=SimpleNamespace(
            nsources=3,
            ntargets=4,
            targets=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            sources=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        ),
        level_start_source_box_nrs=np.array([], dtype=np.int32),
        source_boxes=np.array([], dtype=np.int32),
        level_start_source_parent_box_nrs=np.array([], dtype=np.int32),
        source_parent_boxes=np.array([], dtype=np.int32),
        target_boxes=np.array([], dtype=np.int32),
        neighbor_source_boxes_starts=np.array([], dtype=np.int32),
        neighbor_source_boxes_lists=np.array([], dtype=np.int32),
        from_sep_close_smaller_starts=None,
        from_sep_close_bigger_starts=None,
    )

    src_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    src_func = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = volume_fmm.drive_volume_fmm(
        traversal,
        _MockFMMLibWrangler(),
        src_weights,
        src_func,
        direct_evaluation=True,
        reorder_sources=False,
        reorder_potentials=False,
    )

    assert np.allclose(result, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64))


# {{{ make sure context getter works


def test_cl_ctx_getter(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = np.random.default_rng(seed=42)
    a_np = rng.random(50000, dtype=np.float32)
    b_np = rng.random(50000, dtype=np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(
        ctx,
        """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g,
        __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """,
    ).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    assert np.linalg.norm(res_np - (a_np + b_np)) < 1e-12


@pytest.mark.skipif(
    mg.provider != "meshgen_boxtree",
    reason="Meshgen boxtree backend is unavailable",
)
def test_meshgen_boxtree_gmsh_export(ctx_factory, tmp_path):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen3D(degree=2, nlevels=1, a=-1.0, b=1.0, queue=queue)
    mesh_file = tmp_path / "box_grid.msh"
    mesh.generate_gmsh(str(mesh_file))

    assert mesh_file.exists()
    mesh_text = mesh_file.read_text(encoding="ascii")
    assert "$MeshFormat" in mesh_text
    assert "$Nodes" in mesh_text
    assert "$Elements" in mesh_text


@pytest.mark.skipif(
    mg.provider != "meshgen_boxtree",
    reason="Meshgen boxtree backend is unavailable",
)
def test_meshgen_boxtree_gmsh_export_non_dyadic_bounds(ctx_factory, tmp_path):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen2D(degree=2, nlevels=2, a=0.1, b=0.9, queue=queue)
    mesh_file = tmp_path / "box_grid_non_dyadic.msh"
    mesh.generate_gmsh(str(mesh_file))

    lines = mesh_file.read_text(encoding="ascii").splitlines()
    node_count = int(lines[lines.index("$Nodes") + 1])
    element_count = int(lines[lines.index("$Elements") + 1])
    level = int(mesh._leaf_levels()[0])
    expected_nodes = (2**level + 1) ** mesh.dim
    expected_elements = (2**level) ** mesh.dim

    assert node_count == expected_nodes
    assert element_count == expected_elements


@pytest.mark.skipif(
    mg.provider != "meshgen_boxtree",
    reason="Meshgen boxtree backend is unavailable",
)
def test_meshgen_boxtree_gmsh_export_rejects_mixed_levels(ctx_factory, tmp_path):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen2D(degree=2, nlevels=2, a=-1.0, b=1.0, queue=queue)
    for _ in range(3):
        criteria = np.arange(mesh.n_active_cells(), dtype=float)
        mesh.update_mesh(
            criteria,
            top_fraction_of_cells=0.25,
            bottom_fraction_of_cells=0.0,
        )

    if len(np.unique(mesh._leaf_levels())) == 1:
        pytest.skip("could not create mixed-level mesh on this backend")

    mesh_file = tmp_path / "box_grid_adaptive.msh"
    with pytest.raises(NotImplementedError, match="uniform-level box meshes"):
        mesh.generate_gmsh(str(mesh_file))


# }}}

# {{{ laplace volume potential


@pytest.fixture
def laplace_problem(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dim = 2
    dtype = np.float64

    q_order = 2  # quadrature order
    n_levels = 3  # 2^(n_levels-1) subintervals in 1D

    # adaptive_mesh = True
    n_refinement_loops = 100
    refined_n_cells = 1000
    rratio_top = 0.2
    rratio_bot = 0.5

    # bounding box
    a = -1.0
    b = 1.0

    m_order = 15  # multipole order

    alpha = 160 / np.sqrt(2)

    def source_field(x):
        assert len(x) == dim
        assert dim == 2
        norm2 = x[0] ** 2 + x[1] ** 2
        lap_u = (4 * alpha**2 * norm2 - 4 * alpha) * np.exp(-alpha * norm2)
        return -lap_u

    def exact_solu(x, y):
        norm2 = x**2 + y**2
        return np.exp(-alpha * norm2)

    # {{{ generate quad points

    mesh = mg.MeshGen2D(q_order, n_levels, a, b)
    iloop = 0
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array(
            [
                np.abs(source_field(c) * m)
                for (c, m) in zip(mesh.get_cell_centers(), mesh.get_cell_measures())
            ]
        )
        mesh.update_mesh(crtr, rratio_top, rratio_bot)
        if iloop > n_refinement_loops:
            print("Max number of refinement loops reached.")
            break

    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    # q_radii = None

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points_org = q_points
    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import new_1d as obj_array_1d

    q_points = obj_array_1d(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)]
    )

    q_weights = cl.array.to_device(queue, q_weights)
    # q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(
        queue, np.array([source_field(qp) for qp in q_points_org])
    )

    # particle_weigt = source_val * q_weight

    # }}} End discretize the source field

    # {{{ build tree and traversals

    # tune max_particles_in_box to reconstruct the mesh
    from boxtree.array_context import PyOpenCLArrayContext
    from volumential.tree_interactive_build import build_particle_tree_from_box_tree

    actx = PyOpenCLArrayContext(queue)
    tree = build_particle_tree_from_box_tree(actx, mesh.boxtree, q_points_org)

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(actx)
    trav, _ = tg(actx, tree)

    # }}} End build tree and traversals

    # {{{ build near field potential table

    from volumential.table_manager import NearFieldInteractionTableManager

    subprocess.check_call(["rm", "-f", "nft-test-volume-fmm.hdf5"])
    tm = NearFieldInteractionTableManager("nft-test-volume-fmm.hdf5")
    nftable, _ = tm.get_table(dim, "Laplace", q_order)

    # }}} End build near field potential table

    # {{{ sumpy expansion for laplace kernel

    from sumpy.expansion.local import LinearPDEConformingVolumeTaylorLocalExpansion

    # from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    # from sumpy.expansion.local import VolumeTaylorLocalExpansion
    from sumpy.expansion.multipole import (
        LinearPDEConformingVolumeTaylorMultipoleExpansion,
    )
    from sumpy.kernel import LaplaceKernel

    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion
    # local_expn_class = partial(VolumeTaylorLocalExpansion, use_rscale=None)
    # mpole_expn_class = partial(VolumeTaylorMultipoleExpansion, use_rscale=None)

    exclude_self = True
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        out_kernels,
        exclude_self=exclude_self,
    )

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=trav,
        near_field_table=nftable,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
    )

    # }}} End sumpy expansion for laplace kernel

    exact_vals = cl.array.to_device(
        queue,
        np.array([exact_solu(qp[0], qp[1]) for qp in q_points_org], dtype=dtype),
    )

    return trav, wrangler, source_vals, q_weights, exact_vals


# }}} End laplace volume potential


@pytest.mark.skipif(
    mg.provider != "meshgen_boxtree",
    reason="Adaptive mesh module is not available",
)
def test_volume_fmm_laplace(laplace_problem):

    trav, wrangler, source_vals, q_weights, exact_vals = laplace_problem

    from volumential.volume_fmm import drive_volume_fmm

    (fmm_pot,) = drive_volume_fmm(
        trav, wrangler, source_vals * q_weights, source_vals, direct_evaluation=False
    )

    exact_host = exact_vals.get()
    fmm_host = wrangler.tree_indep._setup_actx.to_numpy(fmm_pot)
    if os.environ.get("VOLUMENTIAL_DEBUG_NAN"):
        print("fmm sample", fmm_host[:8])
        print("exact sample", exact_host[:8])
        print("abs err sample", np.abs(exact_host - fmm_host)[:8])
        print("max abs fmm", np.nanmax(np.abs(fmm_host)))
        print("max abs exact", np.nanmax(np.abs(exact_host)))
    max_err = np.nanmax(np.abs(exact_host - fmm_host))
    assert np.isfinite(max_err)
    assert float(max_err) < 5e-2


@pytest.mark.skipif(
    mg.provider != "meshgen_boxtree",
    reason="Adaptive mesh module is not available",
)
@pytest.mark.parametrize("bbox_radius", [1.0, 1.3])
def test_volume_fmm_calculus_patch_matches_source_density(
    ctx_factory, tmp_path, bbox_radius
):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from sumpy.point_calculus import CalculusPatch

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.geometry import BoundingBoxFactory, BoxFMMGeometryFactory
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm, interpolate_volume_potential

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dim = 2
    q_order = 4
    nlevels = 5
    fmm_order = 15

    bbox = BoundingBoxFactory(
        dim=dim,
        center=np.array([0.0, 0.0]),
        radius=float(bbox_radius),
    )
    geo = BoxFMMGeometryFactory(
        ctx,
        dim=dim,
        order=q_order,
        nlevels=nlevels,
        bbox_getter=bbox,
    )(queue)

    patch = CalculusPatch(center=[0.0, 0.0], h=0.36, order=7)
    patch_targets = np.empty(dim, dtype=object)
    patch_targets[0] = cl.array.to_device(queue, np.ascontiguousarray(patch.x))
    patch_targets[1] = cl.array.to_device(queue, np.ascontiguousarray(patch.y))

    def source_density(x, y):
        return np.exp(0.3 * x - 0.2 * y) * np.cos(0.8 * x + 0.6 * y)

    src_vals_user = source_density(geo.nodes[0].get(queue), geo.nodes[1].get(queue))
    src_vals = cl.array.to_device(queue, np.ascontiguousarray(src_vals_user))
    src_weights = src_vals * geo.weights

    knl = LaplaceKernel(dim)
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    table_file = tmp_path / "nft-calculus-patch.sqlite"
    tm = NearFieldInteractionTableManager(str(table_file), queue=queue)
    nftable, _ = tm.get_table(dim, "Laplace", q_order, force_recompute=True)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        [knl],
        exclude_self=False,
    )
    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        traversal=geo.traversal,
        near_field_table=nftable,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        queue=queue,
    )

    (u_src,) = drive_volume_fmm(
        geo.traversal,
        wrangler,
        src_weights,
        src_vals,
        direct_evaluation=False,
        reorder_sources=True,
        reorder_potentials=True,
        allow_list1_p2p_fallback=False,
    )

    u_patch = interpolate_volume_potential(
        patch_targets,
        geo.traversal,
        wrangler,
        u_src,
    ).get(queue)

    minus_lap = -patch.laplace(u_patch)
    rho_patch = source_density(patch.x, patch.y)

    rel_residual = np.linalg.norm(minus_lap - rho_patch) / np.linalg.norm(rho_patch)
    assert float(rel_residual) < 5.0e-4


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])


# vim: filetype=pyopencl:foldmethod=marker

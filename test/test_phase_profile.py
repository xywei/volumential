"""Tests for the per-phase solve profiler and the per-phase FMM operation
counts (Paper 1, experiment E6).

Everything here is pure Python/numpy: the phase timer is exercised against a
scripted clock and a scripted sync callback, and the FMM stage counts are
exercised against hand-built traversal stand-ins whose expected counts are
worked out by hand in the test itself.  No OpenCL context is created.
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
import volumential.phase_profile as pp


# {{{ phase timer mechanics

def test_phase_is_inert_without_an_active_profile():
    calls = []

    def sync():
        calls.append("sync")

    profile = pp.PhaseProfile(sync=sync)
    with pp.phase("far_eval_locals"):
        pass
    # nothing activated, so neither the clock nor the sync ran
    assert calls == []
    assert profile.total_seconds() == 0.0
    assert not pp.active()


def test_phase_records_and_syncs_at_both_boundaries():
    calls = []
    profile = pp.PhaseProfile(sync=lambda: calls.append("sync"))
    with pp.profiling(profile):
        assert pp.active()
        with pp.phase("nearfield_table_apply"):
            calls.append("body")
    assert calls == ["sync", "body", "sync"]
    assert profile.calls("nearfield_table_apply") == 1
    assert profile.seconds("nearfield_table_apply") >= 0.0
    assert not pp.active()


def test_repeated_phases_accumulate_and_mean_divides():
    profile = pp.PhaseProfile()
    profile.record("far_eval_locals", 0.5)
    profile.record("far_eval_locals", 1.5)
    profile.record("nearfield_table_apply", 2.0)
    assert profile.calls("far_eval_locals") == 2
    assert profile.seconds("far_eval_locals") == pytest.approx(2.0)
    assert profile.mean_seconds("far_eval_locals") == pytest.approx(1.0)
    assert profile.total_seconds() == pytest.approx(4.0)
    assert profile.shares() == {
        "far_eval_locals": pytest.approx(0.5),
        "nearfield_table_apply": pytest.approx(0.5),
    }
    assert profile.shares(denominator=8.0)["far_eval_locals"] == (
        pytest.approx(0.25)
    )


def test_unrecorded_phase_reads_as_zero_not_as_an_error():
    profile = pp.PhaseProfile()
    assert profile.seconds("split_correction") == 0.0
    assert profile.calls("split_correction") == 0
    assert profile.mean_seconds("split_correction") == 0.0
    assert profile.shares() == {}


def test_profile_recovers_after_an_exception_in_the_body():
    profile = pp.PhaseProfile()
    with pytest.raises(RuntimeError):
        with pp.profiling(profile):
            with pp.phase("far_form_locals"):
                raise RuntimeError("boom")
    # the phase is still recorded, and nothing stays active
    assert profile.calls("far_form_locals") == 1
    assert not pp.active()


def test_nesting_is_flagged_rather_than_silently_double_counted():
    profile = pp.PhaseProfile()
    with pp.profiling(profile):
        with pp.phase("far_form_multipoles"):
            with pp.phase("far_eval_locals"):
                pass
    assert profile.nested_names == frozenset({"far_eval_locals"})


def test_multiple_active_profiles_all_receive_the_phase():
    outer = pp.PhaseProfile()
    inner = pp.PhaseProfile()
    with pp.profiling(outer):
        with pp.phase("far_refine_locals"):
            pass
        with pp.profiling(inner):
            with pp.phase("far_refine_locals"):
                pass
    assert outer.calls("far_refine_locals") == 2
    assert inner.calls("far_refine_locals") == 1


def test_a_concurrent_solve_does_not_join_another_thread_s_profile():
    """Activation is per execution context, not per process.

    Two threads each profiling their own solve must record only their own
    phases.  With a process-global active list the second thread's blocks
    would be added to the first thread's profile as well.
    """
    import threading

    started = threading.Barrier(2)
    profiles = {}
    errors = []

    def solve(tag, phase_name):
        try:
            profile = pp.PhaseProfile()
            profiles[tag] = profile
            with pp.profiling(profile):
                # both threads are inside their own profiling block here
                started.wait(timeout=10.0)
                with pp.phase(phase_name):
                    pass
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=solve, args=("a", "far_form_multipoles")),
        threading.Thread(target=solve, args=("b", "far_eval_locals")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)

    assert errors == []
    assert profiles["a"].names() == ["far_form_multipoles"]
    assert profiles["b"].names() == ["far_eval_locals"]
    assert not pp.active()


def test_a_concurrent_solve_does_not_drain_another_thread_s_queue():
    """The wrong queue is drained too, not merely the wrong counter written.

    Each profile carries its own ``sync`` (in practice its own
    ``queue.finish``).  A phase block in one thread must invoke only its own.
    """
    import threading

    started = threading.Barrier(2)
    syncs = {"a": [], "b": []}
    errors = []

    def solve(tag):
        try:
            profile = pp.PhaseProfile(sync=lambda tag=tag: syncs[tag].append(1))
            with pp.profiling(profile):
                started.wait(timeout=10.0)
                with pp.phase("nearfield_table_apply"):
                    pass
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=solve, args=(tag,)) for tag in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)

    assert errors == []
    # entry sync plus exit sync, from that thread's phase block only
    assert len(syncs["a"]) == 2
    assert len(syncs["b"]) == 2


def test_a_thread_started_inside_a_profile_does_not_inherit_it():
    """A worker thread is a separate execution context, so it starts clean."""
    import threading

    profile = pp.PhaseProfile()
    seen = []

    def worker():
        seen.append(pp.active())
        with pp.phase("split_correction"):
            pass

    with pp.profiling(profile):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=20.0)

    assert seen == [False]
    assert profile.calls("split_correction") == 0


@pytest.mark.parametrize("bad", [-1.0, -1e-9])
def test_negative_seconds_are_rejected(bad):
    profile = pp.PhaseProfile()
    with pytest.raises(ValueError):
        profile.record("far_eval_locals", bad)


def test_profiling_and_sync_reject_non_callables():
    with pytest.raises(ValueError):
        pp.PhaseProfile(sync=object())
    with pytest.raises(ValueError):
        with pp.profiling({}):
            pass


def test_solve_phase_names_cover_the_far_and_near_field_lists():
    assert set(pp.SOLVE_PHASES) == set(pp.FAR_FIELD_PHASES) | set(
        pp.NEAR_FIELD_PHASES
    )
    assert len(pp.SOLVE_PHASES) == len(set(pp.SOLVE_PHASES))
    # the opcounters stage names are the far-field phase names, unprefixed
    assert tuple(
        name[len("far_"):] for name in pp.FAR_FIELD_PHASES
    ) == oc.FMM_FAR_FIELD_STAGES

# }}}


# {{{ FMM stage operation counts on a hand-built two-level tree

def _two_level_quadtree():
    """A 2D root with four leaves, uniform ``q**2 = 4`` points per leaf.

    Boxes: 0 is the root (level 0, no own sources/targets), 1..4 are its
    children (level 1).  Every leaf holds 4 sources and 4 targets.  Every
    leaf neighbours every other leaf, so List 1 is complete and Lists 2, 3,
    4 are empty -- which is exactly the uniform benchmark tree's shape, one
    level down.
    """
    nboxes = 5
    box_levels = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    counts = np.array([0, 4, 4, 4, 4], dtype=np.int64)
    child_ids = np.zeros((4, nboxes), dtype=np.int64)
    child_ids[:, 0] = [1, 2, 3, 4]
    leaves = np.array([1, 2, 3, 4], dtype=np.int64)
    return {
        "box_levels": box_levels,
        "box_source_counts_nonchild": counts,
        "box_target_counts_nonchild": counts,
        "box_child_ids": child_ids,
        "source_boxes": leaves,
        "source_parent_boxes": np.array([0], dtype=np.int64),
        "target_boxes": leaves,
        "target_or_target_parent_boxes": np.array(
            [0, 1, 2, 3, 4], dtype=np.int64
        ),
        "from_sep_siblings_starts": np.zeros(6, dtype=np.int64),
        "from_sep_bigger_starts": np.zeros(6, dtype=np.int64),
        "from_sep_bigger_lists": np.zeros(0, dtype=np.int64),
        "sep_smaller_by_level": [
            (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)),
            (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)),
        ],
        "multipole_coeff_counts": [7, 7],
        "local_coeff_counts": [9, 9],
    }


def test_stage_counts_on_a_uniform_two_level_tree():
    counts = oc.fmm_stage_operation_counts(**_two_level_quadtree())

    # P2M: 4 leaves x 4 sources x 7 multipole coefficients
    assert counts["form_multipoles"] == 4 * 4 * 7
    # M2M: sumpy's loop is range(nlevels-1, 2, -1) = range(1, 2, -1), empty
    assert counts["coarsen_multipoles"] == 0
    # empty interaction lists 2, 3 and 4
    assert counts["multipole_to_local"] == 0
    assert counts["eval_multipoles"] == 0
    assert counts["form_locals"] == 0
    # L2L: one parent-to-child translation per level-1 box in the
    # target-or-target-parent list, 9 x 9 each
    assert counts["refine_locals"] == 4 * 9 * 9
    # L2P: 4 leaves x 4 targets x 9 local coefficients
    assert counts["eval_locals"] == 4 * 4 * 9
    assert counts["far_total"] == sum(
        counts[stage] for stage in oc.FMM_FAR_FIELD_STAGES
    )


def test_list2_and_list4_counts_use_the_csr_lists():
    data = _two_level_quadtree()
    # give box 1 two List 2 entries and box 2 one, all at level 1
    data["from_sep_siblings_starts"] = np.array(
        [0, 0, 2, 3, 3, 3], dtype=np.int64
    )
    # give box 3 a List 4 entry pointing at box 4 (4 sources)
    data["from_sep_bigger_starts"] = np.array(
        [0, 0, 0, 0, 1, 1], dtype=np.int64
    )
    data["from_sep_bigger_lists"] = np.array([4], dtype=np.int64)

    counts = oc.fmm_stage_operation_counts(**data)
    assert counts["multipole_to_local"] == 3 * 7 * 9
    assert counts["form_locals"] == 4 * 9


def test_list3_counts_use_the_per_source_level_lists():
    data = _two_level_quadtree()
    # two List 3 source boxes at level 1 for target box 1, one for box 2
    data["sep_smaller_by_level"] = [
        (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)),
        (
            np.array([1, 2], dtype=np.int64),
            np.array([0, 2, 3], dtype=np.int64),
        ),
    ]
    counts = oc.fmm_stage_operation_counts(**data)
    # (4 targets x 2 lists + 4 targets x 1 list) x 7 multipole coefficients
    assert counts["eval_multipoles"] == (4 * 2 + 4 * 1) * 7


def test_m2m_runs_only_over_the_source_levels_sumpy_visits():
    # five levels, so sumpy's range(nlevels-1, 2, -1) = [4, 3] makes target
    # levels 3 and 2 the only M2M targets; a chain of one box per level
    # isolates the loop bounds from everything else
    nlevels = 5
    nboxes = 5
    data = {
        "box_levels": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "box_source_counts_nonchild": np.zeros(nboxes, dtype=np.int64),
        "box_target_counts_nonchild": np.zeros(nboxes, dtype=np.int64),
        "box_child_ids": np.zeros((4, nboxes), dtype=np.int64),
        "source_boxes": np.zeros(0, dtype=np.int64),
        # one parent box on every level 0..3, each owning two children
        "source_parent_boxes": np.array([0, 1, 2, 3], dtype=np.int64),
        "target_boxes": np.zeros(0, dtype=np.int64),
        "target_or_target_parent_boxes": np.zeros(0, dtype=np.int64),
        "from_sep_siblings_starts": np.zeros(1, dtype=np.int64),
        "from_sep_bigger_starts": np.zeros(1, dtype=np.int64),
        "from_sep_bigger_lists": np.zeros(0, dtype=np.int64),
        "sep_smaller_by_level": [
            (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64))
            for _ in range(nlevels)
        ],
        "multipole_coeff_counts": [2] * nlevels,
        "local_coeff_counts": [2] * nlevels,
    }
    data["box_child_ids"][0, :4] = [1, 2, 3, 4]
    data["box_child_ids"][1, :4] = [1, 2, 3, 4]

    counts = oc.fmm_stage_operation_counts(**data)
    # target levels 3 and 2 contribute (2 children x 2 x 2) each; levels 0
    # and 1 are never M2L targets, so sumpy never translates into them
    assert counts["coarsen_multipoles"] == 2 * (2 * 2 * 2)


def test_m2m_is_empty_on_a_shallow_tree():
    data = _two_level_quadtree()
    # the root has four children but nlevels = 2, so sumpy's loop is empty
    assert oc.fmm_stage_operation_counts(**data)["coarsen_multipoles"] == 0


def test_coefficient_count_lists_must_agree_in_length():
    data = _two_level_quadtree()
    data["local_coeff_counts"] = [9]
    with pytest.raises(ValueError, match="per level"):
        oc.fmm_stage_operation_counts(**data)


def test_expansion_size_scaling_is_linear_in_the_stage_rules():
    base = oc.fmm_stage_operation_counts(**_two_level_quadtree())
    doubled = _two_level_quadtree()
    doubled["multipole_coeff_counts"] = [14, 14]
    scaled = oc.fmm_stage_operation_counts(**doubled)
    # P2M is linear in the multipole size; L2P does not see it at all
    assert scaled["form_multipoles"] == 2 * base["form_multipoles"]
    assert scaled["eval_locals"] == base["eval_locals"]

# }}}


# {{{ near-field pair counts with an explicit source set

def _nearfield_case():
    # three target boxes, each seeing itself and the next one
    return {
        "target_boxes": np.array([1, 2, 3], dtype=np.int64),
        "neighbor_source_boxes_starts": np.array([0, 2, 4, 5], dtype=np.int64),
        "neighbor_source_boxes_lists": np.array(
            [1, 2, 2, 3, 3], dtype=np.int64
        ),
        "box_target_counts_nonchild": np.array(
            [0, 4, 4, 4], dtype=np.int64
        ),
    }


def test_nearfield_pairs_default_to_the_box_fmm_convention():
    case = _nearfield_case()
    # (4 x 8) + (4 x 8) + (4 x 4)
    assert oc.nearfield_point_pairs_from_counts(**case) == 32 + 32 + 16


def test_nearfield_pairs_scale_with_an_explicit_smooth_source_set():
    case = _nearfield_case()
    base = oc.nearfield_point_pairs_from_counts(**case)
    # the split remainder runs on an interpolated smooth quadrature: four
    # times the sources per box at q_smooth = 2 q in 2D
    smooth = oc.nearfield_point_pairs_from_counts(
        **case,
        box_source_counts_nonchild=np.array([0, 16, 16, 16], dtype=np.int64),
    )
    assert smooth == 4 * base


def test_nearfield_pairs_handle_empty_lists():
    assert (
        oc.nearfield_point_pairs_from_counts(
            target_boxes=np.zeros(0, dtype=np.int64),
            neighbor_source_boxes_starts=np.zeros(1, dtype=np.int64),
            neighbor_source_boxes_lists=np.zeros(0, dtype=np.int64),
            box_target_counts_nonchild=np.zeros(3, dtype=np.int64),
        )
        == 0
    )

# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

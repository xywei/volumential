"""Focused tests for :mod:`volumential.list1_symmetry`.

These pin the behavior of the symmetry-tag parser and the case-vector
reduction so that the module can be refactored safely. They deliberately do
*not* pin the merge path for two already-overlapping swap groups: that path is
index-fragile in the module under test and no caller in this repository
reaches it, so pinning it would freeze behavior that still needs a decision.
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

import numpy as np
import pytest

from volumential.list1_symmetry import CaseVecReduction, Flip, Swap


def test_flip_and_swap_repr_and_ordering():
    assert repr(Flip(2)) == "Flip(2)"
    assert repr(Swap(2, 0)) == "Swap(0,2)"

    assert Flip(0) < Flip(1)
    assert not Flip(1) < Flip(0)
    assert Swap(0, 1) < Swap(0, 2)

    # different operation types compare by repr
    assert Flip(0) < Swap(0, 1)


def test_parse_symmetry_tags_none_means_maximum_symmetry():
    red = CaseVecReduction(vecs=[(-1, -1)], sym_tags=None)

    assert np.all(red.flippable == 1)
    assert red.swappable_groups == [{0, 1}]


def test_parse_symmetry_tags_empty_means_no_symmetry():
    red = CaseVecReduction(vecs=[(-1, -1)], sym_tags=[])

    assert np.all(red.flippable == 0)
    assert red.swappable_groups == []


def test_parse_symmetry_tags_flip_marks_only_named_axes():
    red = CaseVecReduction(vecs=[(-1, -1, -1)], sym_tags=[Flip(0), Flip(2)])

    assert list(red.flippable) == [1, 0, 1]
    assert red.swappable_groups == []


def test_parse_symmetry_tags_extends_existing_group():
    red = CaseVecReduction(
        vecs=[(-1, -1, -1)], sym_tags=[Swap(0, 1), Swap(1, 2)], do_reduction=False
    )

    assert red.swappable_groups == [{0, 1, 2}]


def test_parse_symmetry_tags_keeps_disjoint_groups_separate():
    red = CaseVecReduction(
        vecs=[(-1, -1, -1, -1)],
        sym_tags=[Swap(0, 1), Swap(2, 3)],
        do_reduction=False,
    )

    assert red.swappable_groups == [{0, 1}, {2, 3}]


def test_parse_symmetry_tags_repeated_swap_is_idempotent():
    red = CaseVecReduction(
        vecs=[(-1, -1)], sym_tags=[Swap(0, 1), Swap(0, 1)], do_reduction=False
    )

    assert red.swappable_groups == [{0, 1}]


def test_parse_symmetry_tags_rejects_unknown_tag():
    class _Bogus:
        pass

    red = CaseVecReduction(vecs=[(-1, -1)], sym_tags=[], do_reduction=False)

    with pytest.raises(NotImplementedError):
        red.parse_symmetry_tags([_Bogus()])


def test_case_vec_reduction_rejects_non_symmetry_tag_at_construction():
    class _Bogus:
        pass

    with pytest.raises(AssertionError):
        CaseVecReduction(vecs=[(-1, -1)], sym_tags=[_Bogus()], do_reduction=False)


def test_case_vec_reduction_rejects_non_list_vecs():
    with pytest.raises(RuntimeError):
        CaseVecReduction(vecs=None)

    with pytest.raises(RuntimeError):
        CaseVecReduction(vecs=((-1, -1),))


def test_find_base_vecs_under_full_symmetry_2d():
    vecs = [
        (-2, -1),
        (-1, -2),
        (-1, -1),
        (1, 1),
        (0, 0),
        (0, -1),
        (-1, 0),
    ]
    red = CaseVecReduction(vecs=vecs, sym_tags=None)

    base_vecs, base_vec_ids = red.find_base_vecs()

    # nonpositive entries, sorted ascending within the single swappable group
    assert base_vecs == [(-2, -1), (-1, -1), (0, 0), (-1, 0)]
    assert base_vec_ids == [0, 2, 4, 6]
    assert red.get_reduced_vecs() == base_vecs
    assert red.get_reduced_vec_ids() == base_vec_ids


def test_find_base_vecs_without_symmetry_keeps_everything():
    vecs = [(-1, 1), (1, -1), (2, 2)]
    red = CaseVecReduction(vecs=vecs, sym_tags=[])

    assert red.get_reduced_vecs() == vecs
    assert red.get_reduced_vec_ids() == [0, 1, 2]
    assert red.get_inter_box_reduction_ratio() == 1.0


def test_find_invariant_group_of_zero_vector_is_full_group():
    red = CaseVecReduction(vecs=[(0, 0)], sym_tags=None)

    (group,) = red.get_reduced_invariant_groups()

    assert sorted(repr(op) for op in group) == ["Flip(0)", "Flip(1)", "Swap(0,1)"]


def test_find_invariant_group_of_generic_vector_is_trivial():
    red = CaseVecReduction(vecs=[(-2, -1)], sym_tags=None)

    (group,) = red.get_reduced_invariant_groups()

    assert group == []


def test_reduction_ratios_for_single_zero_vector():
    red = CaseVecReduction(vecs=[(0, 0)], sym_tags=None)

    assert red.get_inter_box_reduction_ratio() == 1.0
    # two flips (2**2) and one swappable group of size two (2!)
    assert red.get_intra_box_reduction_ratio() == pytest.approx(1.0 / 8.0)
    assert red.get_full_reduction_ratio() == pytest.approx(1.0 / 8.0)


def test_do_reduction_false_defers_the_getters():
    red = CaseVecReduction(vecs=[(0, 0)], sym_tags=None, do_reduction=False)

    assert red.reduced is False
    with pytest.raises(AssertionError):
        red.get_reduced_vecs()

    red.reduce()

    assert red.reduced is True
    assert red.get_reduced_vecs() == [(0, 0)]

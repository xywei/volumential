"""Symmetry bookkeeping for the *list 1* interaction gallery.

This module owns the description of the discrete symmetries a near-field
interaction table may exploit (axis flips and axis swaps), and the reduction of
a set of case vectors to the subset of symmetry representatives together with
each representative's invariant group.

.. autoclass:: SymmetryOperationBase
.. autoclass:: Flip
.. autoclass:: Swap
.. autoclass:: CaseVecReduction
"""

__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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

import math
from collections.abc import Sequence

import numpy as np


# {{{ symmetry operations


class SymmetryOperationBase:
    """Base class for the symmetry operations a case vector may admit."""

    def __init__(self, index) -> None:
        self._index = index

    def __lt__(self, other: "SymmetryOperationBase") -> bool:
        if type(self) is type(other):
            return self._index < other._index

        # different operations in lexicographical order
        return repr(self) < repr(other)


class Flip(SymmetryOperationBase):
    """
    Flip the sign of an axis, spanning S_2^dim
    """

    def __init__(self, iaxis: int) -> None:
        super().__init__(iaxis)
        self.axis = iaxis

    def __repr__(self) -> str:
        return f"Flip({self.axis:d})"


class Swap(SymmetryOperationBase):
    """
    Swap two axes, spanning S_dim
    """

    def __init__(self, iaxis: int, jaxis: int) -> None:
        self.axes = (iaxis, jaxis)
        super().__init__(sorted(self.axes))

    def __repr__(self) -> str:
        first, second = sorted(self.axes)
        return f"Swap({first:d},{second:d})"


# }}} End symmetry operations


class CaseVecReduction:
    """
    Reduce a set of case vectors based on symmetry.
    """

    def __init__(
        self,
        vecs: list | None = None,
        sym_tags: list[SymmetryOperationBase] | None = None,
        do_reduction: bool = True,
    ) -> None:
        """
        sym_tags is a list of SymmetryOperationBase objects.
        sym_tags is [] if no symmetry can be used.
        sym_tags is None if maximum symmetry can be used.
        """
        if isinstance(vecs, list):
            assert len(vecs) >= 1
        else:
            raise RuntimeError("Invalid list of case vecs.")

        self.dim = len(vecs[0])
        for vec in vecs:
            assert len(vec) == self.dim
        self.full_vecs = vecs

        if sym_tags is not None:
            for tag in sym_tags:
                assert isinstance(tag, SymmetryOperationBase)
        self.symmetry_tags = sym_tags
        self.flippable, self.swappable_groups = self.parse_symmetry_tags(
            self.symmetry_tags
        )

        self.reduced = False
        if do_reduction:
            self.reduce()

    def parse_symmetry_tags(
        self, tags: list[SymmetryOperationBase] | None
    ) -> tuple[np.ndarray, list[set[int]]]:
        """Split *tags* into a per-axis flippable mask and swappable axis groups.

        *tags* of ``None`` means maximum symmetry: every axis is flippable and
        all axes form a single swappable group.
        """
        flippable = np.zeros(self.dim)
        swappable_groups: list[set[int]] = []

        if tags is None:
            flippable += 1
            swappable_groups.append(set(range(self.dim)))
            return flippable, swappable_groups

        for tag in tags:
            if isinstance(tag, Flip):
                flippable[tag.axis] = 1

            elif isinstance(tag, Swap):
                iaxis, jaxis = tag.axes
                gi = None
                gj = None
                for gid, group in enumerate(swappable_groups):
                    if iaxis in group:
                        assert gi is None
                        gi = gid
                    if jaxis in group:
                        assert gj is None
                        gj = gid

                if gi is None and gj is None:
                    # New group
                    swappable_groups.append({iaxis, jaxis})
                elif gi is None:
                    # Update group[gj]
                    swappable_groups[gj].add(iaxis)
                elif gj is None:
                    # Update group[gi]
                    swappable_groups[gi].add(jaxis)
                elif gi != gj:
                    # Merge groups
                    swappable_groups.append(
                        set().union(swappable_groups[gi], swappable_groups[gj])
                    )
                    swappable_groups.remove(swappable_groups[gi])
                    swappable_groups.remove(swappable_groups[gj])

            else:
                raise NotImplementedError

        return flippable, swappable_groups

    def find_base_vecs(self) -> tuple[list, list[int]]:
        """Return the symmetry representatives and their ids in the full list."""
        vecs = self.full_vecs
        base_vecs = []
        base_vec_ids = []
        for vid, vec in enumerate(vecs):
            is_base = True
            # Check for flips
            for d in range(self.dim):
                if not self.flippable[d]:
                    continue
                if vec[d] > 0:
                    is_base = False
                    break
            # Check for swaps
            if is_base:
                for group in self.swappable_groups:
                    group_view = list(group)
                    vec_part = np.array(vec)[group_view]
                    if sorted(vec_part) != list(vec_part):
                        is_base = False
                        break
            if is_base:
                base_vecs.append(vec)
                base_vec_ids.append(vid)
        return base_vecs, base_vec_ids

    def find_invariant_group(
        self, vec: Sequence[int]
    ) -> list[SymmetryOperationBase]:
        """
        For a given case vector, within the allowed symmetry tags,
        return a generating set of its invariant group as a list of
        SymmetryOperationBase objects.
        """
        ivgp: list[SymmetryOperationBase] = []
        n = len(vec)
        assert n == self.dim
        for iaxis in range(n):
            if not self.flippable[iaxis]:
                continue
            if vec[iaxis] == -vec[iaxis]:
                ivgp.append(Flip(iaxis))
        for iaxis in range(n):
            for jaxis in range(iaxis + 1, n):
                if vec[iaxis] == vec[jaxis]:
                    # only if swap(i,j) is allowed
                    for group in self.swappable_groups:
                        if (iaxis in group) and (jaxis in group):
                            ivgp.append(Swap(iaxis, jaxis))
                            break
        return ivgp

    def reduce(self) -> None:
        """Compute the representatives and their invariant groups."""
        self.reduced_vecs, self.reduced_vec_ids = self.find_base_vecs()
        self.reduced_invariant_groups = [
            self.find_invariant_group(v) for v in self.reduced_vecs
        ]
        self.reduced = True

    # call reduce() before calling getters

    def get_reduced_vecs(self) -> list:
        """Return the symmetry representatives."""
        assert self.reduced
        return self.reduced_vecs

    def get_reduced_vec_ids(self) -> list[int]:
        """Return the indices of the representatives into the full vector list."""
        assert self.reduced
        return self.reduced_vec_ids

    def get_inter_box_reduction_ratio(self) -> float:
        """Return the fraction of case vectors that survive as representatives."""
        assert self.reduced
        return len(self.reduced_vecs) / len(self.full_vecs)

    def get_intra_box_reduction_ratio(self) -> float:
        """Return the mean fraction of entries needed within a representative box."""
        assert self.reduced
        total_ratio = 0
        for vid in range(len(self.reduced_vecs)):
            ratio = 1
            fable, sgroups = self.parse_symmetry_tags(
                self.reduced_invariant_groups[vid]
            )
            ratio = ratio / (2 ** (sum(fable)))
            for grp in sgroups:
                ratio = ratio / math.factorial(len(grp))
            total_ratio += ratio
        return total_ratio / len(self.reduced_vecs)

    def get_full_reduction_ratio(self) -> float:
        """Return the combined inter-box and intra-box reduction ratio."""
        return (
            self.get_inter_box_reduction_ratio() * self.get_intra_box_reduction_ratio()
        )

    def get_reduced_invariant_groups(self) -> list[list[SymmetryOperationBase]]:
        """Return the invariant group generators of each representative."""
        assert self.reduced
        return self.reduced_invariant_groups

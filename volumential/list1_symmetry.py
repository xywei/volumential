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

import numpy as np
import math

# {{{ symmetry operations


class SymmetryOperationBase(object):
    def __lt__(self, other):
        if type(self) == type(other):
            return self._index < other._index

        # differnt operations in lexicographical order
        return repr(self) < repr(other)


class Flip(SymmetryOperationBase):
    """
    Flip the sign of an axis, spanning S_2^dim
    """

    def __init__(self, iaxis):
        self.axis = iaxis
        self._index = iaxis

    def __repr__(self):
        return "Flip(%d)" % self.axis


class Swap(SymmetryOperationBase):
    """
    Swap two axes, spanning S_dim
    """

    def __init__(self, iaxis, jaxis):
        self.axes = (iaxis, jaxis)
        self._index = sorted(self.axes)

    def __repr__(self):
        return "Swap(%d,%d)" % tuple(sorted(self.axes))


# }}} End symmetry operations


class CaseVecReduction(object):
    """
    Reduce a set of case vectors based on symmetry.
    """

    def __init__(self, vecs=None, sym_tags=None, do_reduction=True):
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

    def parse_symmetry_tags(self, tags):
        flippable = np.zeros(self.dim)
        swappable_groups = []

        if tags is None:
            flippable += 1
            swappable_groups.append({i for i in range(self.dim)})
            return flippable, swappable_groups

        for tag in tags:

            if isinstance(tag, Flip):
                flippable[tag.axis] = 1

            elif isinstance(tag, Swap):
                iaxis, jaxis = tag.axes
                gi = None
                gj = None
                for gid, group in zip(
                        range(len(swappable_groups)),
                        swappable_groups
                        ):
                    if iaxis in group:
                        assert gi is None
                        gi = gid
                    if jaxis in group:
                        assert gj is None
                        gj = gid
                if gi is None:
                    if gj is None:
                        # New group
                        swappable_groups.append({iaxis, jaxis})
                    else:
                        # Update group[gj]
                        swappable_groups[gj].add(iaxis)
                else:
                    if gj is None:
                        # Update group[gi]
                        swappable_groups[gi].add(jaxis)
                    else:
                        if gi == gj:
                            pass
                        else:
                            # Merge groups
                            swappable_groups.append(
                                set().union(
                                    swappable_groups[gi],
                                    swappable_groups[gj])
                                )
                            swappable_groups.remove(swappable_groups[gi])
                            swappable_groups.remove(swappable_groups[gj])

            else:
                raise NotImplementedError

        return flippable, swappable_groups

    def find_base_vecs(self):
        vecs = self.full_vecs
        base_vecs = []
        base_vec_ids = []
        for vid, vec in zip(range(len(vecs)), vecs):
            is_base = True
            # Check for flips
            for d in range(self.dim):
                v = vec[d]
                if not self.flippable[d]:
                    continue
                else:
                    if v > 0:
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

    def find_invariant_group(self, vec):
        """
        For a given case vector, within the allowed symmetry tags,
        return a generating set of its invariant group as a list of
        SymmetryOperationBase objects.
        """
        ivgp = []
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

    def reduce(self):
        self.reduced_vecs, self.reduced_vec_ids = self.find_base_vecs()
        self.reduced_invariant_groups = [
            self.find_invariant_group(v) for v in self.reduced_vecs
        ]
        self.reduced = True

    # call reduce() before calling getters

    def get_reduced_vecs(self):
        assert self.reduced
        return self.reduced_vecs

    def get_reduced_vec_ids(self):
        assert self.reduced
        return self.reduced_vec_ids

    def get_inter_box_reduction_ratio(self):
        assert self.reduced
        return len(self.reduced_vecs) / len(self.full_vecs)

    def get_intra_box_reduction_ratio(self):
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

    def get_full_reduction_ratio(self):
        return (
            self.get_inter_box_reduction_ratio()
            * self.get_intra_box_reduction_ratio()
        )

    def get_reduced_invariant_groups(self):
        assert self.reduced
        return self.reduced_invariant_groups

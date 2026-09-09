"""Enumeration of the *list 1* (near-neighbor) interaction gallery.

This module owns the geometric enumeration of every distinct
target-box-center-to-source-box-center displacement that can occur between a
box and one of its near neighbors in a level-restricted box tree. The gallery
is expressed in integer units of a quarter box width so that all involved
lengths stay exact integers, which lets downstream table code index cases by a
cheap positional encoding.

.. autoclass:: List1Gallery
.. autofunction:: generate_list1_gallery
"""

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

from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import numpy as np


# {{{ TreeBox class


class TreeBox:
    """A minimalistic tree box used only to enumerate the gallery.

    Courtesy of: Andreas Klockner
    """

    def __init__(
        self, center: np.ndarray, radius: int, child_nlevels: int
    ) -> None:
        self.center = center
        self.radius = radius

        self.children: list[TreeBox] = []

        if child_nlevels:
            child_radius = radius // 2
            assert child_radius

            dimensions = len(center)

            for i in range(2**dimensions):
                child_center = center.copy()
                for idim in range(dimensions):
                    # 1 if that dimension bit is set, 0 if not
                    dim_indicator = int(bool(i & 1 << idim))
                    child_center[idim] += (2 * dim_indicator - 1) * child_radius

                self.children.append(
                    TreeBox(child_center, child_radius, child_nlevels - 1)
                )

    def draw(self) -> None:
        """Draw the box outline with :mod:`matplotlib` (2D boxes only)."""
        lx, ly = self.center - self.radius * 0.95
        hx, hy = self.center + self.radius * 0.95

        import matplotlib.pyplot as plt

        plt.plot([lx, lx, hx, hx, lx], [ly, hy, hy, ly, ly])


# }}} End TreeBox class


def build_tree(dimensions: int) -> TreeBox:
    """Build the four-level reference tree the gallery is enumerated on.

    Four levels deep means box centers land on a ``2**4 x 2**4`` integer grid.

    Courtesy of: Andreas Klockner
    """
    nlevels = 4
    root_radius = 2 ** (nlevels - 1)
    root = TreeBox(
        center=np.array(dimensions * [root_radius], np.int64),
        radius=root_radius,
        child_nlevels=nlevels - 1,
    )

    return root


def generate_boxes_on_level(box: TreeBox, ilevel: int) -> Iterator[TreeBox]:
    """Yield the descendants of *box* that sit *ilevel* levels below it.

    Courtesy of: Andreas Klockner
    """
    if ilevel:
        for child in box.children:
            yield from generate_boxes_on_level(child, ilevel - 1)
    else:
        yield box


def generate_boxes(box: TreeBox) -> Iterator[TreeBox]:
    """Yield *box* and all of its descendants.

    Courtesy of: Andreas Klockner
    """
    yield box

    for child in box.children:
        yield from generate_boxes(child)


def linf_dist(box1: TreeBox, box2: TreeBox) -> float:
    """Return the :math:`\\ell^\\infty` gap between two boxes (0 if adjacent).

    Courtesy of: Andreas Klockner
    """
    return np.max(np.abs(box1.center - box2.center) - (box1.radius + box2.radius))


def generate_interactions(dimensions: int) -> list[tuple[TreeBox, TreeBox]]:
    """Return every (target box, source box) pair that touches, in *dimensions*.

    Target boxes are taken from level 2 of the reference tree and kept away
    from the domain boundary, so that their neighborhoods are complete. Source
    boxes are drawn from all levels, which is what makes the gallery cover
    mixed-level interactions.

    Courtesy of: Andreas Klockner
    """
    root = build_tree(dimensions)
    root_radius = root.radius

    min_cutoff = root_radius >> 2
    max_cutoff = 2 * root_radius - min_cutoff

    target_boxes = [
        box
        for box in generate_boxes_on_level(root, 2)
        if np.min(box.center) > min_cutoff and np.max(box.center) < max_cutoff
    ]

    return [
        (tbox, sbox)
        for tbox in target_boxes
        for sbox in generate_boxes(root)
        if linf_dist(tbox, sbox) == 0
    ]


def postprocess_interactions(
    near_neighbor_interactions: Sequence[tuple[TreeBox, TreeBox]],
) -> list[tuple[int, ...]]:
    """Reduce touching box pairs to the sorted set of distinct case vectors."""
    unique_interaction_vectors = set()

    for tbox, sbox in near_neighbor_interactions:
        unique_interaction_vectors.add(tuple(tbox.center - sbox.center))

    # Add interactions within the same box
    tb0, _ = near_neighbor_interactions[0]
    unique_interaction_vectors.add(tuple(tb0.center - tb0.center))

    return sorted(unique_interaction_vectors)


class List1Gallery(NamedTuple):
    """The list 1 gallery for one dimension.

    .. attribute:: vec_list

        Sorted list of distinct case vectors, each a tuple of integers in units
        of a quarter of the target box width.

    .. attribute:: case_encode

        Maps a case vector to its index into :attr:`case_indices`.

    .. attribute:: case_indices

        Lookup table from encoded case vector to case id, ``-1`` where no case
        vector maps to that slot.
    """

    vec_list: list[tuple[int, ...]]
    case_encode: Callable[[Sequence[int]], int]
    case_indices: np.ndarray


def generate_list1_gallery(dim: int) -> List1Gallery:
    """Generate a *list1* that serves as the gallery for all possible *list1*
    interactions with given dimension and order.

    The returned value is a :class:`List1Gallery`, which unpacks as the
    ``(vec_list, case_encode, case_indices)`` triple it has always been.
    """
    # contains each sourcebox.center-to-targetbox.center vector
    # source box is 4x4 to make all involved lengths to be integers
    vec_list = postprocess_interactions(generate_interactions(dim))

    distinct_numbers = set()
    for vec in vec_list:
        for cvc in vec:
            distinct_numbers.add(cvc)

    # contains a lookup table for case indices
    base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
    case_indices = -np.ones(base**dim, dtype=int)
    shift = -min(distinct_numbers)

    def case_encode(case_vec: Sequence[int]) -> int:
        table_id = 0
        for cvc in case_vec:
            table_id = table_id * base + (cvc + shift)
        return int(table_id)

    case_id = 0
    for vec in vec_list:
        case_indices[case_encode(vec)] = case_id
        case_id += 1

    assert len(vec_list) == case_id

    return List1Gallery(vec_list, case_encode, case_indices)


# vim: ft=pyopencl:fdm=marker

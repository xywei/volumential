'''
    Generate a list 1 that servers as the gallery for all possible list 1
    interactions with given dimension and order.
'''

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

import numpy as np


# {{{ TreeBox class
class TreeBox:
    '''
    A minimalistic tree class.
    Courtesy of: Andreas Klockner
    '''

    def __init__(self, center, radius,
                 child_nlevels):
        self.center = center
        self.radius = radius

        self.children = []

        if child_nlevels:
            child_radius = radius // 2
            assert child_radius

            dimensions = len(center)

            for i in range(
                    2**dimensions):
                child_center = center.copy(
                )
                for idim in range(
                        dimensions):
                    # 1 if that dimension bit is set, 0 if not
                    dim_indicator = int(
                        bool(
                            i &
                            1 << idim))
                    child_center[
                        idim] += (
                            2 *
                            dim_indicator
                            - 1
                    ) * child_radius

                self.children.append(
                    TreeBox(
                        child_center,
                        child_radius,
                        child_nlevels -
                        1))

    def draw(self):
        lx, ly = self.center - self.radius * 0.95
        hx, hy = self.center + self.radius * 0.95

        import matplotlib.pyplot as plt
        plt.plot([lx, lx, hx, hx, lx],
                 [ly, hy, hy, ly, ly])


# }}} End TreeBox class


def build_tree(dimensions):
    '''
    Courtesy of: Andreas Klockner
    '''
    # four levels deep
    # -> centers on a 2**4 x 2**4 grid

    nlevels = 4
    root_radius = 2**(nlevels - 1)
    root = TreeBox(
        center=np.array(
            dimensions * [root_radius],
            np.int),
        radius=root_radius,
        child_nlevels=nlevels - 1)

    return root


def generate_boxes_on_level(box,
                            ilevel):
    '''
    Courtesy of: Andreas Klockner
    '''
    if ilevel:
        for child in box.children:
            for result in generate_boxes_on_level(
                    child, ilevel - 1):
                yield result
    else:
        yield box


def generate_boxes(box):
    '''
    Courtesy of: Andreas Klockner
    '''
    yield box

    for child in box.children:
        for result in generate_boxes(
                child):
            yield result


def linf_dist(box1, box2):
    '''
    Courtesy of: Andreas Klockner
    '''
    return np.max(
        np.abs(box1.center - box2.center)
        - (box1.radius + box2.radius))


def generate_interactions(dimensions):
    '''
    Courtesy of: Andreas Klockner
    '''
    root = build_tree(dimensions)
    root_radius = root.radius

    min_cutoff = root_radius >> 2
    max_cutoff = 2 * root_radius - min_cutoff

    target_boxes = [
        box
        for box in
        generate_boxes_on_level(
            root, 2)
        if np.min(
            box.center) > min_cutoff and
        np.max(box.center) < max_cutoff
    ]

    near_neighbor_interactions = [
        (tbox, sbox)
        for tbox in target_boxes
        for sbox in generate_boxes(root)
        if linf_dist(tbox, sbox) == 0
    ]

    if 0:
        import matplotlib.pyplot as plt
        for tbox, sbox in near_neighbor_interactions:
            plt.figure()
            plt.gca().set_aspect(
                "equal")
            tbox.draw()
            sbox.draw()
    return near_neighbor_interactions


def postprocess_interactions(
        near_neighbor_interactions):
    unique_interaction_vectors = set()

    for (
            tbox, sbox
    ) in near_neighbor_interactions:
        unique_interaction_vectors.add(
            tuple(tbox.center -
                  sbox.center))

    # Add interactions within the same box
    tb0, wb0 = near_neighbor_interactions[0]
    unique_interaction_vectors.add(tuple(tb0.center - tb0.center))

    list1_interactions = sorted(
        list(
            unique_interaction_vectors))

    return list1_interactions


def generate_list1_gallery(dim):
    # contains each sourcebox.center-to-targetbox.center vector
    # source box is 4x4 to make all involved lengths to be integers
    vec_list = postprocess_interactions(
        generate_interactions(dim))

    distinct_numbers = set()
    for vec in vec_list:
        for l in vec:
            distinct_numbers.add(l)

    # contains a lookup table for case indices
    base = len(
        range(
            min(distinct_numbers),
            max(distinct_numbers) + 1))
    case_indices = -np.ones(
        base**dim, dtype=int)
    shift = -min(distinct_numbers)

    def case_encode(case_vec):
        table_id = 0
        for l in case_vec:
            table_id = table_id * base + (
                l + shift)
        return int(table_id)

    case_id = 0
    for vec in vec_list:
        case_indices[case_encode(
            vec)] = case_id
        case_id += 1

    assert (len(vec_list) == case_id)

    return (vec_list, case_encode,
            case_indices)


# vim: ft=pyopencl:fdm=marker

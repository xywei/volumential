from __future__ import absolute_import, division, print_function

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

import numpy as np
import pyopencl as cl

import boxtree as bt
import volumential.meshgen as mg

dim = 2

q_order = 2  # quadrature order
n_levels = 4

a = -1
b = 1

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# {{{ generate quad points

q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
    degree=q_order, level=n_levels)

assert (len(q_points) == len(q_weights))
assert (q_points.shape[1] == dim)

q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array
q_points = make_obj_array(
    [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)
q_radii = cl.array.to_device(queue, q_radii)

# }}}

# {{{ build tree and traversals

from boxtree.tools import AXIS_NAMES
axis_names = AXIS_NAMES[:dim]

from pytools import single_valued
coord_dtype = single_valued(coord.dtype for coord in q_points)
from boxtree.bounding_box import make_bounding_box_dtype
bbox_type, _ = make_bounding_box_dtype(ctx.devices[0], dim, coord_dtype)

bbox = np.empty(1, bbox_type)
for ax in axis_names:
    bbox["min_" + ax] = a
    bbox["max_" + ax] = b

# tune max_particles_in_box to reconstruct the mesh
# passing target_radii to align the mesh messes up the traversal
# (produces list 4 and list 3 close)
# instead we can use specified bbox to do the trick
from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(
    queue,
    particles=q_points,
    targets=q_points,
    #target_radii=q_radii,
    #stick_out_factor=0,
    bbox = bbox,
    max_particles_in_box=q_order**2 * 4 - 1,
    kind="adaptive-level-restricted")

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# }}} End build tree and traversals

# {{{ plot the tree

import matplotlib.pyplot as plt

if dim == 2:
    plt.plot(q_points[0].get(), q_points[1].get(), "x")

from boxtree.visualization import TreePlotter
plotter = TreePlotter(tree.get(queue=queue))
plotter.draw_tree(fill=False, edgecolor="black")
plotter.draw_box_numbers()
plotter.set_bounding_box()
plt.gca().set_aspect("equal")
plt.savefig("tree.png")

# }}} End plot the tree

# vim: filetype=pyopencl:foldmethod=marker

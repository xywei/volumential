__copyright__ = "Copyright (C) 2017 Xiaoyu Wei"

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

from volumential.volume_fmm import interpolate_volume_potential

bdry_nodes_x = bdry_discr.nodes()[0].with_queue(queue).get()
bdry_nodes_y = bdry_discr.nodes()[1].with_queue(queue).get()
bdry_nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, bdry_nodes_x),
         cl.array.to_device(queue, bdry_nodes_y)])

bdry_pot = interpolate_volume_potential(bdry_nodes, trav, wrangler, pot)
bdry_pot = bdry_pot.get()
assert(len(bdry_pot) == bdry_discr.nnodes)

bdry_condition = exact_solu(bdry_nodes_x, bdry_nodes_y)
bdry_vals = bdry_condition - bdry_pot

bdry_vals = cl.array.to_device(queue, bdry_vals)

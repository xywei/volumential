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

# output the whole box
from meshmode.discretization.visualization import make_visualizer
vis = make_visualizer(queue, box_discr, visual_order)

qbx_stick_out = qbx.copy(target_stick_out_factor=0.05)

# interpolate solution
from volumential.volume_fmm import interpolate_volume_potential
box_nodes_x = box_discr.nodes()[0].with_queue(queue).get()
box_nodes_y = box_discr.nodes()[1].with_queue(queue).get()
box_nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, box_nodes_x),
         cl.array.to_device(queue, box_nodes_y)])

box_vol_pot = interpolate_volume_potential(box_nodes,
        trav, wrangler, pot)

box_bvp_sol = bind(
        (qbx_stick_out, box_discr),
        op.representation(sym_sigma, map_potentials=None, qbx_forced_limit=None
            ))(queue, sigma=sigma)

box_solu = box_bvp_sol + box_vol_pot

vis.write_vtk_file("solution_box.vtu", [
    ("x", box_discr.nodes()[0]),
    ("y", box_discr.nodes()[1]),
    ("bvp_sol", box_bvp_sol),
    ("vol_pot", box_vol_pot),
    ("solu", box_solu)
    ])



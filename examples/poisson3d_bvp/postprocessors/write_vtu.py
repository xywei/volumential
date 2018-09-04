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

# write file
from meshmode.discretization.visualization import make_visualizer
vis = make_visualizer(queue, vol_discr, visual_order)
bdry_vis = make_visualizer(queue, bdry_discr, visual_order)

vis.write_vtk_file("solution.vtu", [
    ("x", vol_discr.nodes()[0]),
    ("y", vol_discr.nodes()[1]),
    ("z", vol_discr.nodes()[2]),
    ("bvp_sol", bvp_sol),
    ("vol_pot", vol_pot),
    ("solu", solu)
    ])

bdry_normals = bind(bdry_discr, sym.normal(dim))(queue).as_vector(dtype=object)

bdry_vis.write_vtk_file("solution-bdry.vtu", [
    ("bdry_normals", bdry_normals),
    ("BC_vals", bdry_condition),
    ("VP_vals", bdry_pot),
    ("BC-VP", bdry_vals),
    ])

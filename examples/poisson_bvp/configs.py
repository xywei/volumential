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

dim = 2
dtype = np.float64
nftable_datafile = "./nft_brick.hdf5"

# {{{ order control

q_order = 4         # (volumetric) quadrature order

# FIXME: high multipole order unstable
# appears to be some overflow in sumpy/codegen
m_order = 8         # multipole order

mesh_order = 4      # meshmode's volume mesh order
bdry_quad_order = 4 # meshmode's boundary discr order

bdry_ovsmp_quad_order = 4*bdry_quad_order # qbx's fine_order
qbx_order = 4                             # qbx order
qbx_fmm_order = m_order                   # qbx's fmm_order

# }}} End order control

# {{{ mesh control

# bounding box
a = -0.25
b = 0.25

# boundary mesh control
h = 0.01

# volume (adaptive) mesh control
n_levels = 3  # 2^(n_levels-1) subintervals in 1D, must be at least 2 if not adaptive.
              # If adaptive mesh, this is the intial number of levels before further
              # adaptive refinements
adaptive_mesh = True
n_refinement_loops = 100
refined_n_cells = 5000
rratio_top = 0.2
rratio_bot = 0.5

def refinement_flag(f, u=0):
    '''
    Input source field value f and solution value u (if available),
    output non-negative flags used for mesh refinement.
    '''
    return f

# }}} End mesh control

# {{{ equation control

loc_sign = -1 # +1 for exterior, -1 for interior

source_type = "./source_field/constant_one.py"
physical_domain = "circle"

# }}} End equation control

# {{{ postprocessing options

visual_order = 10

plot_boxtree  = True
write_vtu     = True

# FIXME
write_box_vtu = True

# }}} End postprocessing options

if write_vtu:
    assert(adaptive_mesh)

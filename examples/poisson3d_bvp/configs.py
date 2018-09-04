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

dim = 3
dtype = np.float64
nftable_datafile = "./nft_brick.hdf5"
verbose = True

if verbose:
        logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# {{{ order control

q_order = 3         # (volumetric) quadrature order

m_order = 10         # multipole order

mesh_order = 2      # meshmode's volume mesh order
bdry_quad_order = 3 # meshmode's boundary discr order

bdry_ovsmp_quad_order = 6 # qbx's fine_order
qbx_order = 6             # qbx order
qbx_fmm_order = 10        # qbx's fmm_order

# }}} End order control

# {{{ mesh control

# bounding box
a = -0.3
b = 0.3

# boundary mesh control
h = 0.01

# volume (adaptive) mesh control
n_levels = 3  # 2^(n_levels-1) subintervals in 1D, must be at least 2 if not adaptive.
              # If adaptive mesh, this is the intial number of levels before further
              # adaptive refinements
adaptive_mesh = True
n_refinement_loops = 100
refined_n_cells = 50000
rratio_top = 0.1
rratio_bot = 0.5

def refinement_flag(f, u=0):
    '''
    Input source field value f and solution value u (if available),
    output non-negative flags used for mesh refinement.
    '''
    return np.abs(f)

# }}} End mesh control

# {{{ equation control

loc_sign = -1 # +1 for exterior, -1 for interior

source_type = "./source_field/constant_one.py"
physical_domain = "dice"

# }}} End equation control

# {{{ postprocessing options

visual_order = 2

write_vtu     = True
write_box_vtu = False

# }}} End postprocessing options

if write_vtu:
    assert(adaptive_mesh)

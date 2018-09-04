dim = 2
dtype = np.float64
nftable_datafile = "../nft.hdf5"

# {{{ order control

q_order = 2         # (volumetric) quadrature order

# FIXME: high multipole order unstable
# appears to be some overflow in sumpy/codegen
m_order = 8         # multipole order

mesh_order = 2      # meshmode's volume mesh order
bdry_quad_order = 2 # meshmode's boundary discr order

bdry_ovsmp_quad_order = 4*bdry_quad_order # qbx's fine_order
qbx_order = 4                             # qbx order
qbx_fmm_order = m_order                   # qbx's fmm_order

# }}} End order control

# {{{ mesh control

# bounding box
a = -0.25
b = 0.25

# boundary mesh control
h = 0.05

# volume (adaptive) mesh control
n_levels = 3  # 2^(n_levels-1) subintervals in 1D, must be at least 2 if not adaptive.
              # If adaptive mesh, this is the intial number of levels before further
              # adaptive refinements
adaptive_mesh = True
n_refinement_loops = 100
refined_n_cells = 1000
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

visual_order = 5

plot_boxtree  = True
write_vtu     = True
write_box_vtu = False

# }}} End postprocessing options

if write_vtu:
    assert(adaptive_mesh)

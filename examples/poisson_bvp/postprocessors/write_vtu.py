# write file
from meshmode.discretization.visualization import make_visualizer
vis = make_visualizer(queue, vol_discr, visual_order)

vis.write_vtk_file("solution.vtu", [
    ("x", vol_discr.nodes()[0]),
    ("y", vol_discr.nodes()[1]),
    ("bvp_sol", bvp_sol),
    ("vol_pot", vol_pot),
    ("solu", solu)
    ])



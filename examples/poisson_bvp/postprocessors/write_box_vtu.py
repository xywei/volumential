# output the whole box
from meshmode.discretization.visualization import make_visualizer
vis = make_visualizer(queue, box_discr, visual_order)

# interpolate solution
from volumential.volume_fmm import interpolate_volume_potential
box_nodes_x = box_discr.nodes()[0].with_queue(queue).get()
box_nodes_y = box_discr.nodes()[1].with_queue(queue).get()
box_nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, box_nodes_x),
         cl.array.to_device(queue, box_nodes_y)])

box_vol_pot = interpolate_volume_potential(box_nodes,
        trav, wrangler, pot)

# FIXME
box_bvp_sol = bind(
        (qbx, box_discr),
        op.representation(sym_sigma))(queue, sigma=sigma)

box_solu = box_bvp_sol + box_vol_pot

vis.write_vtk_file("solution_box.vtu", [
    ("x", vol_discr.nodes()[0]),
    ("y", vol_discr.nodes()[1]),
    ("bvp_sol", box_bvp_sol),
    ("vol_pot", box_vol_pot),
    ("solu", box_solu)
    ])



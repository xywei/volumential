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

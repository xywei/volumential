bvp_sol = bind(
        (qbx, vol_discr),
        op.representation(sym_sigma))(queue, sigma=sigma)

solu = bvp_sol + vol_pot

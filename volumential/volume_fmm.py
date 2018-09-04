from __future__ import division

import logging
logger = logging.getLogger(__name__)

from volumential.expansion_wrangler_interface import ExpansionWranglerInterface

import numpy as np


def drive_volume_fmm(traversal, expansion_wrangler, src_weights, src_func,
        direct_evaluation=False):
    """
    Top-level driver routine for volume potential calculation
    via fast multiple method.

    This function, and the interface it utilizes, is adapted from boxtree/fmm.py

    The fast multipole method is a two-pass algorithm:
    - During the fist (upward) pass, the multipole expansions for all boxes
      at all levels are formed from bottom up.
    - In the second (downward) pass, the local expansions for all boxes
      at all levels at formed from top down.

    :arg traversal: A :class:`boxtree.traversal.FMMTraversalInfo` instance.
    :arg expansion_wrangler: An object implementing the expansion
                    wrangler interface.
    :arg src_weights: Source 'density/weights/charges' time quad weights..
        Passed unmodified to *expansion_wrangler*.
    :arg src_func: Source 'density/weights/charges' function.
        Passed unmodified to *expansion_wrangler*.

    Returns the potentials computed by *expansion_wrangler*.
    """
    wrangler = expansion_wrangler
    assert (issubclass(type(wrangler), ExpansionWranglerInterface))

    logger.info("start fmm")

    logger.debug("reorder source weights")

    src_weights = wrangler.reorder_sources(src_weights)
    src_func = wrangler.reorder_sources(src_func)

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps = wrangler.form_multipoles(traversal.level_start_source_box_nrs,
                                          traversal.source_boxes, src_weights)

    # print(max(abs(mpole_exps)))

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    wrangler.coarsen_multipoles(traversal.level_start_source_parent_box_nrs,
                                traversal.source_parent_boxes, mpole_exps)

    # print(max(abs(mpole_exps)))

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    # look up in the prebuilt table
    # this step also constructs the output array
    potentials = wrangler.eval_direct(traversal.target_boxes,
                                      traversal.neighbor_source_boxes_starts,
                                      traversal.neighbor_source_boxes_lists,
                                      src_func
                                      )

    # potentials = wrangler.output_zeros()

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage X:" direct evaluation of everything and return
    if direct_evaluation:

        print("Warning: NOT USING FMM (forcing global p2p)")

        # list 2 and beyond
        # First call global p2p, then substract list 1

        from sumpy import P2P
        p2p = P2P(wrangler.queue.context, wrangler.code.out_kernels,
                exclude_self=wrangler.code.exclude_self)
        evt, (ref_pot,) = p2p(wrangler.queue,
                traversal.tree.targets, traversal.tree.sources,
                (src_weights,), **wrangler.self_extra_kwargs)
        potentials[0] += ref_pot

        potentials = potentials - wrangler.eval_direct_p2p(
                traversal.target_boxes,
                traversal.neighbor_source_boxes_starts,
                traversal.neighbor_source_boxes_lists,
                src_weights)

        # list 3
        assert (traversal.from_sep_close_smaller_starts is None)

        # list 4
        assert (traversal.from_sep_close_bigger_starts is None)

        logger.debug("reorder potentials")
        result = wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = wrangler.finalize_potentials(result)

        logger.info("direct p2p complete")

        return result

    # }}} End Stage

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps = wrangler.multipole_to_local(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts, traversal.from_sep_siblings_lists,
        mpole_exps)

    # print(max(abs(local_exps)))

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
        traversal.level_start_target_box_nrs, traversal.target_boxes,
        traversal.from_sep_smaller_by_level, mpole_exps)

    # these potentials are called beta in [1]

    # volume fmm does not work with list 3 close currently
    # but list 3 should be empty with our use cases
    assert (traversal.from_sep_close_smaller_starts is None)
    # if traversal.from_sep_close_smaller_starts is not None:
    #    logger.debug("evaluate separated close smaller interactions directly "
    #                 "('list 3 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_boxes, traversal.from_sep_close_smaller_starts,
    #        traversal.from_sep_close_smaller_lists, src_weights)

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    logger.debug(
        "form locals for separated bigger source boxes ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts, traversal.from_sep_bigger_lists,
        src_weights)

    # volume fmm does not work with list 4 currently
    assert (traversal.from_sep_close_bigger_starts is None)
    # if traversal.from_sep_close_bigger_starts is not None:
    #    logger.debug("evaluate separated close bigger interactions directly "
    #                 "('list 4 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_or_target_parent_boxes,
    #        traversal.from_sep_close_bigger_starts,
    #        traversal.from_sep_close_bigger_lists, src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")
    # import numpy.linalg as la

    wrangler.refine_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes, local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")

    potentials = potentials + wrangler.eval_locals(
        traversal.level_start_target_box_nrs, traversal.target_boxes,
        local_exps)

    # }}}

    logger.debug("reorder potentials")
    result = wrangler.reorder_potentials(potentials)

    logger.debug("finalize potentials")
    result = wrangler.finalize_potentials(result)

    logger.info("fmm complete")

    return result


def compute_barycentric_lagrange_params(q_order):

    # 1d quad points and weights
    q_points_1d, q_weights_1d = np.polynomial.legendre.leggauss(q_order)
    q_points_1d = (q_points_1d + 1) * 0.5
    q_weights_1d *= 0.5

    # interpolation weights for barycentric Lagrange interpolation
    from scipy.interpolate import BarycentricInterpolator as Interpolator
    interp = Interpolator(xi=q_points_1d, yi=None)
    interp_weights = interp.wi
    interp_points = interp.xi

    return (interp_points, interp_weights)


def interpolate_volume_potential(target_points, traversal, wrangler, potential,
                                 target_radii=None):
    """
    Interpolate the volume potential.
    target_points and potential should be an cl array.
    """

    dim = wrangler.near_field_table.dim
    tree = wrangler.tree
    queue = wrangler.queue
    q_order = wrangler.quad_order
    ctx = queue.context
    dtype = wrangler.dtype
    n_points = len(target_points[0])

    assert(dim == 2)

    from boxtree.area_query import LeavesToBallsLookupBuilder
    lookup_builder = LeavesToBallsLookupBuilder(ctx)

    if target_radii is None:
        import pyopencl as cl
        # Set this number small enough so that all points found
        # are inside the box
        target_radii = cl.array.to_device(queue,
                                          np.ones(n_points, dtype=dtype) * 1e-32)

    lbl_lookup, evt = lookup_builder(queue, tree, target_points, target_radii)

    pout = cl.array.zeros(queue, n_points, dtype=dtype)
    multiplicity = cl.array.zeros(queue, n_points, dtype=dtype)
    pout.add_event(evt)

    # map all boxes to a template [0,1]^2 so that the interpolation
    # weights and modes can be precomputed
    (blpoints, blweights) = compute_barycentric_lagrange_params(q_order)
    blpoints = cl.array.to_device(queue, blpoints)
    blweights = cl.array.to_device(queue, blweights)

# {{{ loopy kernel for interpolation

    import loopy
    lpknl = loopy.make_kernel(  # NOQA
            [
                "{ [ tbox ] : 0 <= tbox < n_tgt_boxes }",
                "{ [ tpt, mid, mjd, mkd ] : tpt_begin <= tpt < tpt_end "
                "and 0 <= mid < n_box_modes and 0 <= mjd < Q_ORDER "
                "and 0 <= mkd < Q_ORDER }".replace(
                    "Q_ORDER", str(q_order))
            ], """
            for tbox
                <> target_box_id  = target_boxes[tbox]

                <> tpt_begin = balls_near_box_starts[target_box_id]
                <> tpt_end   = balls_near_box_starts[target_box_id+1]

                <> box_level     = box_levels[target_box_id]
                <> box_mode_beg  = box_target_starts[target_box_id]
                <> n_box_modes   = box_target_counts_cumul[target_box_id]

                <> box_extent   = root_extent * (1.0 / (2**box_level))
                <> box_center_x = box_centers[0, target_box_id]
                <> box_center_y = box_centers[1, target_box_id]

                for tpt
                    <> target_point_id = balls_near_box_lists[tpt]
                    <> real_x = target_coords_x[target_point_id]
                    <> real_y = target_coords_y[target_point_id]

                    # Count how many times the potential is computed
                    multiplicity[target_point_id] = multiplicity[target_point_id] + 1

                    # Map target point to template box
                    <> x = (real_x - box_center_x) / box_extent + 0.5
                    <> y = (real_y - box_center_y) / box_extent + 0.5

                    # Precompute denominators
                    <> denom_x = 0.0 {id=reinit_denom_x}
                    <> denom_y = 0.0 {id=reinit_denom_y}
                    for mjd
                         diff_x[mjd] = if( \
                                          x == barycentric_lagrange_points[mjd], \
                                          1, \
                                          x - barycentric_lagrange_points[mjd]) \
                            {id=diff_x, dep=reinit_denom_x}
                         denom_x = denom_x + \
                                 barycentric_lagrange_weights[mjd] / diff_x[mjd] \
                                 {id=denom_x, dep=diff_x}

                         diff_y[mjd] = if( \
                                          y == barycentric_lagrange_points[mjd], \
                                          1, \
                                          y - barycentric_lagrange_points[mjd]) \
                            {id=diff_y, dep=reinit_denom_y}
                         denom_y = denom_y + \
                                 barycentric_lagrange_weights[mjd] / diff_y[mjd] \
                                 {id=denom_y, dep=diff_y}
                    end

                    for mid
                        # Find the coeff of each mode
                        <> mode_id      = box_mode_beg + mid
                        <> mode_id_user = user_mode_ids[mode_id]
                        <> mode_coeff   = potential[mode_id_user]

                        # Mode id in each direction
                        idx0 = mid / Q_ORDER
                        idx1 = mid % Q_ORDER

                        # Interpolate mode value in x direction
                        <> numerator_x = barycentric_lagrange_weights[idx0] / \
                                         diff_x[idx0] \
                            {id=numer_x, dep=diff_x}
                        <> mode_val_x = numerator_x / denom_x \
                            {id=mode_val_x, dep=denom_x}

                        # Interpolate mode value in y direction
                        <> numerator_y = barycentric_lagrange_weights[idx1] / \
                                         diff_y[idx1] \
                            {id=numer_y, dep=diff_y}
                        <> mode_val_y = numerator_y / denom_y \
                            {id=mode_val_y, dep=denom_y}

                        for mkd  # Fix when target point coincide with a quad point
                            mode_val_x = if( \
                                    x == barycentric_lagrange_points[mkd], \
                                    if(mkd == idx0, 1, 0), \
                                    mode_val_x) \
                                {id=fix_mode_val_x, dep=mode_val_x}
                            mode_val_y = if( \
                                    y == barycentric_lagrange_points[mkd], \
                                    if(mkd == idx1, 1, 0), \
                                    mode_val_y) \
                                {id=fix_mode_val_y, dep=mode_val_y}
                        end

                        # Evaluate each mode at the target point
                        p_out[target_point_id] = p_out[target_point_id] + \
                            mode_coeff * mode_val_x * mode_val_y \
                            {id=p_out, dep=fix_mode_val_x, dep=fix_mode_val_y}

                    end

                end

            end

            """.replace("Q_ORDER", str(q_order)),
            [
                loopy.TemporaryVariable("idx0", np.int32),
                loopy.TemporaryVariable("idx1", np.int32),
                loopy.TemporaryVariable("diff_x", dtype, str(q_order)),
                loopy.TemporaryVariable("diff_y", dtype, str(q_order)),
                loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
                loopy.GlobalArg("balls_near_box_lists", None, None),
                loopy.ValueArg("aligned_nboxes", np.int32),
                loopy.ValueArg("dim", np.int32), "..."
            ])
# }}} End loopy kernel for interpolation

    lpknl = loopy.set_options(lpknl, return_dict=True)
    evt, res_dict = lpknl(
        queue,
        p_out=pout,
        multiplicity=multiplicity,
        box_centers=tree.box_centers,
        box_levels=tree.box_levels,
        balls_near_box_starts=lbl_lookup.balls_near_box_starts,
        balls_near_box_lists=lbl_lookup.balls_near_box_lists,
        barycentric_lagrange_weights=blweights,
        barycentric_lagrange_points=blpoints,
        box_target_starts=tree.box_target_starts,
        box_target_counts_cumul=tree.box_target_counts_cumul,
        potential=potential,
        user_mode_ids=tree.user_source_ids,
        target_coords_x=target_points[0],
        target_coords_y=target_points[1],
        target_boxes=traversal.target_boxes,
        dim=dim,
        root_extent=tree.root_extent,
        n_tgt_boxes=len(traversal.target_boxes)
    )

    assert(pout is res_dict["p_out"])
    assert(multiplicity is res_dict["multiplicity"])
    pout.add_event(evt)
    multiplicity.add_event(evt)

    return pout / multiplicity


# vim: filetype=pyopencl:fdm=marker

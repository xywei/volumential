__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

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

__doc__ = """
.. autofunction:: drive_volume_fmm
.. autofunction:: interpolate_volume_potential
"""

import logging

import numpy as np

import pyopencl as cl
from boxtree.timing import TimingRecorder
from pytools.obj_array import make_obj_array

from volumential.expansion_wrangler_fpnd import (
    FPNDFMMLibExpansionWrangler, FPNDSumpyExpansionWrangler)
from volumential.expansion_wrangler_interface import ExpansionWranglerInterface


logger = logging.getLogger(__name__)


# {{{ volume FMM driver

def drive_volume_fmm(traversal, expansion_wrangler, src_weights, src_func,
                     direct_evaluation=False, timing_data=None,
                     reorder_sources=True, reorder_potentials=True,
                     **kwargs):
    """
    Top-level driver routine for volume potential calculation
    via fast multiple method.

    This function, and the interface it utilizes, is adapted from boxtree/fmm.py

    The fast multipole method is a two-pass algorithm:

    1. During the fist (upward) pass, the multipole expansions for all boxes
    at all levels are formed from bottom up.

    2. In the second (downward) pass, the local expansions for all boxes
    at all levels at formed from top down.

    :arg traversal: A :class:`boxtree.traversal.FMMTraversalInfo` instance.
    :arg expansion_wrangler: An object implementing the expansion
                    wrangler interface.
    :arg src_weights: Source 'density/weights/charges' time quad weights..
        Passed unmodified to *expansion_wrangler*.
    :arg src_func: Source 'density/weights/charges' function.
        Passed unmodified to *expansion_wrangler*.
    :arg reorder_sources: Whether sources are in user order (if True, sources
        are reordered into tree order before conducting FMM).
    :arg reorder_potentials: Whether potentials should be in user order (if True,
        potentials are reordered into user order before return).

    Returns the potentials computed by *expansion_wrangler*.
    """
    wrangler = expansion_wrangler
    assert issubclass(type(wrangler), ExpansionWranglerInterface)

    recorder = TimingRecorder()
    logger.info("start fmm")

    # accept unpacked inputs when doing fmm for just one source field
    dtype = wrangler.dtype
    if src_weights.ndim == 1:
        src_weights = make_obj_array([src_weights.astype(dtype)])
    if src_func.ndim == 1:
        src_func = make_obj_array([src_func.astype(dtype)])

    assert (ns := len(src_weights)) == len(src_func)
    if ns > 1:
        raise NotImplementedError("Multiple outputs are not yet supported")

    if isinstance(expansion_wrangler, FPNDSumpyExpansionWrangler):
        assert all(isinstance(sw, cl.array.Array) for sw in src_weights)
        assert all(isinstance(sf, cl.array.Array) for sf in src_func)

    elif isinstance(expansion_wrangler, FPNDFMMLibExpansionWrangler):

        traversal = traversal.get(wrangler.queue)

        if isinstance(src_weights, cl.array.Array):
            src_weights = src_weights.get(wrangler.queue)
        if isinstance(src_func, cl.array.Array):
            src_func = src_func.get(wrangler.queue)

    if reorder_sources:
        logger.debug("reorder source weights")
        for idx_s in range(ns):
            src_weights[idx_s] = wrangler.reorder_sources(src_weights[idx_s])
            src_func[idx_s] = wrangler.reorder_targets(src_func[idx_s])

    # {{{ Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps, timing_future = wrangler.form_multipoles(
        traversal.level_start_source_box_nrs, traversal.source_boxes, src_weights
    )
    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    mpole_exps, timing_future = wrangler.coarsen_multipoles(
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        mpole_exps,
    )
    recorder.add("coarsen_multipoles", timing_future)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    # look up in the prebuilt table
    # this step also constructs the output array
    potentials, timing_future = wrangler.eval_direct(
        traversal.target_boxes,
        traversal.neighbor_source_boxes_starts,
        traversal.neighbor_source_boxes_lists,
        src_func[0],  # FIXME: handle multiple source fields
    )
    recorder.add("eval_direct", timing_future)

    # Return list 1 only, for debugging
    # 'list1_only' takes precedence over 'exclude_list1'
    if "list1_only" in kwargs and kwargs["list1_only"]:

        if reorder_potentials:
            logger.debug("reorder potentials")
            result = wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = wrangler.finalize_potentials(result)

        logger.info("fmm complete with list 1 only")
        logger.info("fmm complete")
        logger.warn("only list 1 results are returned")

        return result

    # Do not include list 1
    if "exclude_list1" in kwargs and kwargs["exclude_list1"]:
        logger.info("Using zeros for list 1")
        logger.warn("list 1 interactions are not included")
        potentials = wrangler.output_zeros()

    # these potentials are called alpha in [1]

    # }}}

    # {{{ (Optional) direct evaluation of everything and return
    if direct_evaluation:

        print("Warning: NOT USING FMM (forcing global p2p)")
        if len(src_weights) != len(src_func):
            print("Using P2P with different src/tgt discretizations can be "
                  "unstable when targets are close to the sources while not "
                  "be exactly the same")

        # list 2 and beyond
        # First call global p2p, then substract list 1

        from sumpy import P2P

        p2p = P2P(
            wrangler.queue.context,
            wrangler.code.target_kernels,
            exclude_self=wrangler.code.exclude_self,
            value_dtypes=[wrangler.dtype],
        )

        if hasattr(wrangler, "self_extra_kwargs"):
            p2p_extra_kwargs = dict(wrangler.self_extra_kwargs)
        else:
            p2p_extra_kwargs = {}

        if hasattr(wrangler, "self_extra_kwargs"):
            p2p_extra_kwargs.update(wrangler.kernel_extra_kwargs)

        for iw, sw in enumerate(src_weights):
            evt, (ref_pot,) = p2p(
                wrangler.queue,
                traversal.tree.targets,
                traversal.tree.sources,
                (sw,),
                **p2p_extra_kwargs
            )
            potentials[iw] += ref_pot

        l1_potentials, timing_future = wrangler.eval_direct_p2p(
                traversal.target_boxes,
                traversal.neighbor_source_boxes_starts,
                traversal.neighbor_source_boxes_lists,
                src_weights,
                )

        potentials = potentials - l1_potentials

        # list 3
        assert traversal.from_sep_close_smaller_starts is None

        # list 4
        assert traversal.from_sep_close_bigger_starts is None

        if reorder_potentials:
            logger.debug("reorder potentials")
            result = wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = wrangler.finalize_potentials(result)

        logger.info("direct p2p complete")

        return result

    # }}} End Stage

    # {{{ Translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps, timing_future = wrangler.multipole_to_local(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        mpole_exps,
    )
    recorder.add("multipole_to_local", timing_future)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ Evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
        traversal.target_boxes_sep_smaller_by_source_level,
        traversal.from_sep_smaller_by_level,
        mpole_exps,
    )
    recorder.add("eval_multipoles", timing_future)

    potentials = potentials + mpole_result

    # these potentials are called beta in [1]

    # volume fmm does not work with list 3 close currently
    # but list 3 should be empty with our use cases
    assert traversal.from_sep_close_smaller_starts is None
    # if traversal.from_sep_close_smaller_starts is not None:
    #    logger.debug("evaluate separated close smaller interactions directly "
    #                 "('list 3 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_boxes, traversal.from_sep_close_smaller_starts,
    #        traversal.from_sep_close_smaller_lists, src_weights)

    # }}}

    # {{{ Form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_result, timing_future = wrangler.form_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts,
        traversal.from_sep_bigger_lists,
        src_weights,
    )

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    # volume fmm does not work with list 4 currently
    assert traversal.from_sep_close_bigger_starts is None
    # if traversal.from_sep_close_bigger_starts is not None:
    #    logger.debug("evaluate separated close bigger interactions directly "
    #                 "('list 4 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_or_target_parent_boxes,
    #        traversal.from_sep_close_bigger_starts,
    #        traversal.from_sep_close_bigger_lists, src_weights)

    # }}}

    # {{{ Propagate local_exps downward

    logger.debug("propagate local_exps downward")
    # import numpy.linalg as la

    local_exps, timing_future = wrangler.refine_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        local_exps,
    )

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ Evaluate locals

    logger.debug("evaluate locals")

    local_result, timing_future = wrangler.eval_locals(
        traversal.level_start_target_box_nrs, traversal.target_boxes, local_exps
    )

    recorder.add("eval_locals", timing_future)

    potentials = potentials + local_result

    # }}}

    # {{{ Reorder potentials
    if reorder_potentials:
        logger.debug("reorder potentials")
        result = wrangler.reorder_potentials(potentials)

    logger.debug("finalize potentials")
    result = wrangler.finalize_potentials(result)

    logger.info("fmm complete")
    # }}} End Reorder potentials

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result

# }}} End volume FMM driver


# {{{ free form interpolation of potentials

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
                                 potential_in_tree_order=False,
                                 target_radii=None, lbl_lookup=None, **kwargs):
    """
    Interpolate the volume potential, only works for tensor-product quadrature
    formulae. target_points and potential should be an cl array.

    :arg wrangler: Used only for general info (nothing sumpy kernel specific).
        May also be None if the needed information is passed by kwargs.
    :arg potential_in_tree_order: Whether the potential is in tree order (as
        opposed to in user order).
    :lbl_lookup: a :class:`boxtree.LeavesToBallsLookup` that has the lookup
        information for target points. Can be None if the lookup lists are
        provided separately in kwargs. If it is None and no other information is
        provided, the lookup will be built from scratch.
    """
    if wrangler is not None:
        dim = next(iter(wrangler.near_field_table.values()))[0].dim
        tree = wrangler.tree
        queue = wrangler.queue
        q_order = wrangler.quad_order
        dtype = wrangler.dtype
    else:
        dim = kwargs["dim"]
        tree = kwargs["tree"]
        queue = kwargs["queue"]
        q_order = kwargs["q_order"]
        dtype = kwargs["dtype"]

    ctx = queue.context
    coord_dtype = tree.coord_dtype
    n_points = len(target_points[0])

    from volumential.expansion_wrangler_fpnd import FPNDFMMLibExpansionWrangler
    if isinstance(wrangler, FPNDFMMLibExpansionWrangler):
        tree = tree.to_device(queue)
        traversal = traversal.to_device(queue)

    assert dim == len(target_points)
    evt = None

    if ("balls_near_box_starts" in kwargs) and ("balls_near_box_lists" in kwargs):
        balls_near_box_starts = kwargs["balls_near_box_starts"]
        balls_near_box_lists = kwargs["balls_near_box_lists"]

    else:
        # Building the lookup takes O(n*log(n))
        if lbl_lookup is None:
            from boxtree.area_query import LeavesToBallsLookupBuilder
            lookup_builder = LeavesToBallsLookupBuilder(ctx)

            if target_radii is None:
                # Set this number small enough so that all points found
                # are inside the box
                target_radii = cl.array.to_device(
                    queue, np.ones(n_points, dtype=coord_dtype) * 1e-12)

            lbl_lookup, evt = lookup_builder(
                queue, tree, target_points, target_radii)

        balls_near_box_starts = lbl_lookup.balls_near_box_starts
        balls_near_box_lists = lbl_lookup.balls_near_box_lists

    pout = cl.array.zeros(queue, n_points, dtype=dtype)
    multiplicity = cl.array.zeros(queue, n_points, dtype=dtype)

    if evt is not None:
        pout.add_event(evt)

    # map all boxes to a template [0,1]^2 so that the interpolation
    # weights and modes can be precomputed
    (blpoints, blweights) = compute_barycentric_lagrange_params(q_order)
    blpoints = cl.array.to_device(queue, blpoints)
    blweights = cl.array.to_device(queue, blweights)

    # {{{ loopy kernel for interpolation

    if dim == 1:
        code_target_coords_assignment = """target_coords_x[target_point_id]"""
    if dim == 2:
        code_target_coords_assignment = """if(
        iaxis == 0, target_coords_x[target_point_id],
                    target_coords_y[target_point_id])"""
    elif dim == 3:
        code_target_coords_assignment = """if(
        iaxis == 0, target_coords_x[target_point_id], if(
        iaxis == 1, target_coords_y[target_point_id],
                    target_coords_z[target_point_id]))"""
    else:
        raise NotImplementedError

    if dim == 1:
        code_mode_index_assignment = """mid"""
    elif dim == 2:
        code_mode_index_assignment = """if(
        iaxis == 0, mid / Q_ORDER,
                    mid % Q_ORDER)""".replace(
            "Q_ORDER", "q_order"
        )
    elif dim == 3:
        code_mode_index_assignment = """if(
        iaxis == 0, mid / (Q_ORDER * Q_ORDER), if(
        iaxis == 1, mid % (Q_ORDER * Q_ORDER) / Q_ORDER,
                    mid % (Q_ORDER * Q_ORDER) % Q_ORDER))""".replace(
            "Q_ORDER", "q_order"
        )
    else:
        raise NotImplementedError

    import loopy

    lpknl = loopy.make_kernel(
        [
            "{ [ tbox, iaxis ] : 0 <= tbox < n_tgt_boxes " "and 0 <= iaxis < dim }",
            "{ [ tpt, mid, mjd, mkd ] : tpt_begin <= tpt < tpt_end "
            "and 0 <= mid < n_box_modes and 0 <= mjd < q_order "
            "and 0 <= mkd < q_order }",
        ],
        """
            for tbox
                <> target_box_id  = target_boxes[tbox]

                <> tpt_begin = balls_near_box_starts[target_box_id]
                <> tpt_end   = balls_near_box_starts[target_box_id+1]

                <> box_level     = box_levels[target_box_id]
                <> box_mode_beg  = box_target_starts[target_box_id]
                <> n_box_modes   = box_target_counts_cumul[target_box_id]

                <> box_extent   = root_extent * (1.0 / (2**box_level))

                for iaxis
                    <> box_center[iaxis] = box_centers[iaxis, target_box_id] {dup=iaxis}
                end

                for tpt
                    <> target_point_id = balls_near_box_lists[tpt]

                    for iaxis
                        <> real_coord[iaxis] = TARGET_COORDS_ASSIGNMENT {dup=iaxis}
                    end

                    # Count how many times the potential is computed
                    multiplicity[target_point_id] = multiplicity[target_point_id] + 1  {atomic}

                    # Map target point to template box
                    for iaxis
                        <> tplt_coord[iaxis] = (real_coord[iaxis] - box_center[iaxis]
                            ) / box_extent + 0.5 {dup=iaxis}
                    end

                    # Precompute denominators
                    for iaxis
                        <> denom[iaxis] = 0.0 {id=reinit_denom,dup=iaxis}
                    end

                    for iaxis, mjd
                         <> diff[iaxis, mjd] = if( \
                                          tplt_coord[iaxis] == barycentric_lagrange_points[mjd], \
                                          1, \
                                          tplt_coord[iaxis] - barycentric_lagrange_points[mjd]) \
                                          {id=diff, dep=reinit_denom, dup=iaxis:mjd}
                         denom[iaxis] = denom[iaxis] + \
                                 barycentric_lagrange_weights[mjd] / diff[iaxis, mjd] \
                                 {id=denom, dep=diff, dup=iaxis:mjd}
                    end

                    for mid
                        # Find the coeff of each mode
                        <> mode_id      = box_mode_beg + mid
                        <> mode_id_user = user_mode_ids[mode_id]
                        <> mode_coeff   = potential[mode_id_user]

                        # Mode id in each direction
                        for iaxis
                            idx[iaxis] = MODE_INDEX_ASSIGNMENT {id=mode_indices,dup=iaxis}
                        end

                        # Interpolate mode value in each direction
                        for iaxis
                            <> numerator[iaxis] = (barycentric_lagrange_weights[idx[iaxis]]
                                                / diff[iaxis, idx[iaxis]]) {id=numer,dep=diff:mode_indices,dup=iaxis}
                            <> mode_val[iaxis] = numerator[iaxis] / denom[iaxis] {id=mode_val,dep=numer:denom,dup=iaxis}
                        end

                        # Fix when target point coincide with a quad point
                        for mkd, iaxis
                            mode_val[iaxis] = if(
                                    tplt_coord[iaxis] == barycentric_lagrange_points[mkd],
                                    if(mkd == idx[iaxis], 1, 0),
                                    mode_val[iaxis]) {id=fix_mode_val, dep=mode_val:mode_indices, dup=iaxis}
                        end

                        <> prod_mode_val = product(iaxis,
                            mode_val[iaxis]) {id=pmod,dep=fix_mode_val,dup=iaxis}

                    end

                    p_out[target_point_id] = p_out[target_point_id] + sum(mid,
                        mode_coeff * prod_mode_val
                        ) {id=p_out,dep=pmod,atomic}

                end

            end

            """.replace(    # noqa: E501
            "TARGET_COORDS_ASSIGNMENT", code_target_coords_assignment
        )
        .replace("MODE_INDEX_ASSIGNMENT", code_mode_index_assignment)
        .replace("Q_ORDER", "q_order"),
        [
            loopy.TemporaryVariable("idx", np.int32, "dim,"),
            # loopy.TemporaryVariable("denom", dtype, "dim,"),
            # loopy.TemporaryVariable("diff", dtype, "dim, q_order"),
            loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
            loopy.GlobalArg("balls_near_box_lists", None, None),
            loopy.GlobalArg("multiplicity", None, None, for_atomic=True),
            loopy.GlobalArg("p_out", None, None, for_atomic=True),
            loopy.ValueArg("aligned_nboxes", np.int32),
            loopy.ValueArg("dim", np.int32),
            loopy.ValueArg("q_order", np.int32),
            "...",
        ],
        lang_version=(2018, 2),
    )
    # }}} End loopy kernel for interpolation

    # loopy does not directly support object arrays
    if dim == 1:
        target_coords_knl_kwargs = {"target_coords_x": target_points[0]}
    elif dim == 2:
        target_coords_knl_kwargs = {
            "target_coords_x": target_points[0],
            "target_coords_y": target_points[1],
        }
    elif dim == 3:
        target_coords_knl_kwargs = {
            "target_coords_x": target_points[0],
            "target_coords_y": target_points[1],
            "target_coords_z": target_points[2],
        }
    else:
        raise NotImplementedError

    if potential_in_tree_order:
        user_mode_ids = cl.array.arange(
            queue, len(tree.user_source_ids), dtype=tree.user_source_ids.dtype)
    else:
        # fetching from user_source_ids converts potential to tree order
        user_mode_ids = tree.user_source_ids

    lpknl = loopy.set_options(lpknl, return_dict=True)
    lpknl = loopy.fix_parameters(lpknl, dim=int(dim), q_order=int(q_order))
    lpknl = loopy.split_iname(lpknl, "tbox", 128, outer_tag="g.0", inner_tag="l.0")
    evt, res_dict = lpknl(
        queue,
        p_out=pout,
        multiplicity=multiplicity,
        box_centers=tree.box_centers,
        box_levels=tree.box_levels,
        balls_near_box_starts=balls_near_box_starts,
        balls_near_box_lists=balls_near_box_lists,
        barycentric_lagrange_weights=blweights,
        barycentric_lagrange_points=blpoints,
        box_target_starts=tree.box_target_starts,
        box_target_counts_cumul=tree.box_target_counts_cumul,
        potential=potential,
        user_mode_ids=user_mode_ids,
        target_boxes=traversal.target_boxes,
        root_extent=tree.root_extent,
        n_tgt_boxes=len(traversal.target_boxes),
        **target_coords_knl_kwargs)

    assert pout is res_dict["p_out"]
    assert multiplicity is res_dict["multiplicity"]
    pout.add_event(evt)
    multiplicity.add_event(evt)

    return pout / multiplicity

# }}} End free form interpolation of potentials

# vim: filetype=pyopencl:fdm=marker

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

__doc__ = """Batched far-field stages for the pyfmmlib backend.

Two optional accelerations of the per-box pyfmmlib calls: a batched P2M via
the ``*formmp_imany`` routines, and a GEMM-based L2P that reuses one dense
evaluation matrix per level when the level's target boxes share a layout.
Both fall back to the inherited per-box implementation when unavailable.
"""

import logging

import numpy as np

from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from pytools import memoize_method


logger = logging.getLogger(__name__)


class FMMLibBatchedStagesMixin:
    """Batched P2M and GEMM-based L2P for :class:`FPNDFMMLibExpansionWrangler`."""

    def _get_batched_formmp_routine(self):
        """Return the per-level batched P2M routine
        ``{l,h}{2,3}dformmp_imany`` from :mod:`pyfmmlib`, or *None* if the
        batched code path is unavailable for this wrangler configuration.
        """
        if self.use_dipoles:
            return None
        if self.dim not in (2, 3):
            return None

        eqn_letter = self.tree_indep.eqn_letter
        if eqn_letter not in ("l", "h"):
            return None

        import pyfmmlib

        name = f"{eqn_letter}{self.dim}dformmp_imany"
        return getattr(pyfmmlib, name, None)

    def _form_multipoles_batched(
        self, formmp_imany, level_start_source_box_nrs, source_boxes, src_weights
    ):
        """Batched (one Fortran call per level) version of
        :meth:`boxtree.pyfmmlib_integration.FMMLibExpansionWrangler.form_multipoles`.

        Mirrors the inherited semantics exactly: same per-box source slices,
        the same per-level ``rscale`` and ``nterms``, and the same output
        layout (including the transpose that boxtree applies when storing
        expansions).

        .. note::

            All offset/start *values* passed to the ``_imany`` routines are
            0-based (the Fortran arrays are declared ``(0:*)``).
        """
        (src_weights,) = src_weights
        tree = self.tree

        mpoles = self.multipole_expansion_zeros()
        charge = np.ascontiguousarray(src_weights, dtype=np.complex128)
        all_sources = self._get_single_sources_array()
        all_centers = self._get_single_box_centers_array()

        for lev in range(tree.nlevels):
            start, stop = level_start_source_box_nrs[lev:lev + 2]
            if start == stop:
                continue

            boxes = np.asarray(source_boxes[start:stop])
            counts = tree.box_source_counts_nonchild[boxes]

            # Empty boxes are excluded from the batched call; their
            # expansion entries stay zero, matching the inherited behavior.
            nonempty = counts > 0
            boxes = boxes[nonempty]
            counts = counts[nonempty]
            nboxes = len(boxes)
            if nboxes == 0:
                continue

            level_start_ibox, mpoles_view = self.multipole_expansions_view(
                mpoles, lev
            )

            rscale = self.level_to_rscale(lev)
            nterms = self.level_orders[lev]

            # One segment per box (0-based offset values)
            seg_starts = np.arange(nboxes + 1, dtype=np.int32)
            box_offsets = np.asarray(
                tree.box_source_starts[boxes], dtype=np.int32
            )
            count_offsets = np.arange(nboxes, dtype=np.int32)
            center_offsets = np.asarray(boxes, dtype=np.int32)

            ier, expn = formmp_imany(
                **self.kernel_kwargs,
                rscale=rscale,
                sources=all_sources,
                sources_offsets=box_offsets,
                sources_starts=seg_starts,
                charge=charge,
                charge_offsets=box_offsets,
                charge_starts=seg_starts,
                nsources=np.asarray(counts, dtype=np.int32),
                nsources_offsets=count_offsets,
                nsources_starts=seg_starts,
                centers=all_centers,
                centers_offsets=center_offsets,
                nterms=nterms,
            )

            if (np.asarray(ier) != 0).any():
                raise RuntimeError(
                    f"formmp_imany failed with nonzero ier "
                    f"on level {lev} ({nboxes} boxes)"
                )

            # expn has shape (*reversed(expansion_shape), nvcount); boxtree
            # stores mpole.T per box, so expn.T has exactly the layout of
            # mpoles_view[boxes], i.e. (nvcount, *expansion_shape).
            mpoles_view[boxes - level_start_ibox] = expn.T

        return mpoles

    _L2P_OFFSET_RTOL = 1e-12

    def _gemm_l2p_supported(self):
        """Whether the GEMM-based L2P code path applies to this wrangler."""
        return (
            not self.tree_indep.ifgrad
            and self.dim in (2, 3)
            and self.tree_indep.eqn_letter in ("l", "h")
        )

    def _l2p_reference_offsets(self, lev, target_boxes_of_level):
        """Return ``(ref_ibox, ref_offsets)`` for level *lev*, where
        *ref_offsets* is the ``(dim, ntargets_per_box)`` array of target
        coordinates relative to the box center of the first nonempty target
        box on that level. Returns ``(None, None)`` if the level has no
        nonempty target box.
        """
        for tgt_ibox in target_boxes_of_level:
            pslice = self._get_target_slice(tgt_ibox)
            if pslice.stop - pslice.start == 0:
                continue
            offsets = (
                self._get_targets(pslice)
                - self.tree.box_centers[:, tgt_ibox].reshape(-1, 1)
            )
            return tgt_ibox, offsets
        return None, None

    def _l2p_level_layout_ok(self, lev, boxes, counts, ref_offsets):
        """Verify (on a sample of boxes) that every target box on level
        *lev* shares the same target-node layout relative to its center.
        This holds for volumential's tensor-product node placement; if it is
        violated, the caller falls back to the per-box reference loop.
        """
        nref = ref_offsets.shape[1]
        if not (counts == nref).all():
            return False

        box_size = self.tree.root_extent * 2.0 ** (-lev)
        tol = self._L2P_OFFSET_RTOL * box_size

        nboxes = len(boxes)
        sample = np.unique(
            np.asarray([0, nboxes // 3, (2 * nboxes) // 3, nboxes - 1])
        )
        for i in sample:
            tgt_ibox = boxes[i]
            pslice = self._get_target_slice(tgt_ibox)
            offsets = (
                self._get_targets(pslice)
                - self.tree.box_centers[:, tgt_ibox].reshape(-1, 1)
            )
            if np.abs(offsets - ref_offsets).max() >= tol:
                return False
        return True

    @memoize_method
    def _l2p_matrix(self, lev, ref_ibox):
        """Build the dense L2P evaluation matrix for level *lev*, using
        the target offsets of box *ref_ibox* as the reference layout.

        *ref_ibox* must be the same box whose offsets
        :meth:`_l2p_level_layout_ok` verified (i.e. the first nonempty box
        among the traversal's target boxes of the level), so the matrix is
        never built from an unverified reference. Memoized on
        ``(lev, ref_ibox)``; for a fixed traversal the reference box is
        stable, so the memoization stays effective across calls.

        Returns an array of shape ``(ntargets_per_box, ncoefs)`` whose
        column *j* is the potential of the *j*-th unit coefficient vector,
        evaluated (via the same scalar ``taeval`` routine boxtree uses) at
        the reference target offsets. Rows are matched to the C-order
        raveling of boxtree's per-box expansion view, so at runtime

            pot = exps_view[boxes].reshape(nboxes, ncoefs) @ M.T
        """
        pslice = self._get_target_slice(ref_ibox)
        ref_offsets = (
            self._get_targets(pslice)
            - self.tree.box_centers[:, ref_ibox].reshape(-1, 1)
        )

        taeval = self.tree_indep.get_expn_eval_routine("ta")
        rscale = self.level_to_rscale(lev)
        expn_shape = self.expansion_shape(self.level_orders[lev])
        ncoefs = int(np.prod(expn_shape))
        ntargets = ref_offsets.shape[1]
        center = np.zeros(self.dim)

        mat = np.zeros((ntargets, ncoefs), dtype=np.complex128)
        unit = np.zeros(expn_shape, dtype=np.complex128)
        for j in range(ncoefs):
            unit.flat[j] = 1
            pot, _grad = taeval(
                rscale=rscale,
                center=center,
                expn=unit.T,
                ztarg=ref_offsets,
                **self.kernel_kwargs,
            )
            mat[:, j] = pot
            unit.flat[j] = 0

        return mat

    def _eval_locals_level_reference(
        self, lev, boxes, local_exps_view, level_start_ibox, output
    ):
        """Per-box scalar L2P for one level; identical math to the loop in
        :meth:`boxtree.pyfmmlib_integration.FMMLibExpansionWrangler.eval_locals`.

        .. note::

            This mirrors boxtree's ``eval_locals`` per-box loop combined
            with the non-grad branch of its ``add_potgrad_onto_output``
            (i.e. ``output[tgt_pslice] += pot``). If boxtree's
            implementation changes, this method must be updated to match,
            or the agreement tests will surface the drift.
        """
        taeval = self.tree_indep.get_expn_eval_routine("ta")
        rscale = self.level_to_rscale(lev)

        for tgt_ibox in boxes:
            tgt_pslice = self._get_target_slice(tgt_ibox)
            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tmp_pot, _tmp_grad = taeval(
                rscale=rscale,
                center=self.tree.box_centers[:, tgt_ibox],
                expn=local_exps_view[tgt_ibox - level_start_ibox].T,
                ztarg=self._get_targets(tgt_pslice),
                **self.kernel_kwargs,
            )
            output[tgt_pslice] += tmp_pot

    def _eval_locals_gemm(
        self, level_start_target_box_nrs, target_boxes, local_exps
    ):
        """GEMM-based version of ``eval_locals``: per level, gather the
        local expansion coefficients of all target boxes into a matrix and
        apply a single precomputed evaluation matrix, instead of one scalar
        ``taeval`` call per box.
        """
        output = FMMLibExpansionWrangler.output_zeros(self)

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev + 2]
            if start == stop:
                continue

            level_start_ibox, local_exps_view = self.local_expansions_view(
                local_exps, lev
            )

            boxes = np.asarray(target_boxes[start:stop])
            counts = self.box_target_counts_nonchild()[boxes]

            nonempty = counts > 0
            boxes_ne = boxes[nonempty]
            counts_ne = counts[nonempty]
            if len(boxes_ne) == 0:
                continue

            ref_ibox, ref_offsets = self._l2p_reference_offsets(lev, boxes_ne)

            if not self._l2p_level_layout_ok(
                lev, boxes_ne, counts_ne, ref_offsets
            ):
                logger.warning(
                    "eval_locals: target-node layout is not uniform on "
                    "level %d; falling back to per-box L2P for this level",
                    lev,
                )
                self._eval_locals_level_reference(
                    lev, boxes_ne, local_exps_view, level_start_ibox, output
                )
                continue

            mat = self._l2p_matrix(lev, ref_ibox)

            nboxes = len(boxes_ne)
            exps = local_exps_view[boxes_ne - level_start_ibox].reshape(
                nboxes, -1
            )
            pots = exps @ mat.T

            # The layout check guarantees every box on this level has
            # exactly nref targets, and the target slices of distinct
            # boxes are disjoint, so a single fancy-indexed accumulate
            # (no duplicate indices) replaces the per-box loop safely.
            nref = ref_offsets.shape[1]
            box_target_starts = self.box_target_starts()
            flat_idx = (
                box_target_starts[boxes_ne][:, None]
                + np.arange(nref)[None, :]
            ).ravel()
            output[flat_idx] += pots.ravel()

        return output

# vim: filetype=pyopencl:foldmethod=marker

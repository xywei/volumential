from __future__ import division

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

import logging

logger = logging.getLogger(__name__)

from abc import ABCMeta, abstractmethod

# {{{ expansion wrangler interface

# NOTE: abstractmethod's signatures (arguement lists) are not enforced


class ExpansionWranglerInterface(metaclass=ABCMeta):
    """
        Abstract expansion handling interface.
        The interface is adapted from, and stays compatible with boxtree/fmm.
        TODO: Update docstrings
    """

    @abstractmethod
    def multipole_expansion_zeros(self):
        """
        Construct arrays to store multipole expansions for all boxes
        """
        pass

    @abstractmethod
    def local_expansion_zeros(self):
        """
        Construct arrays to store multipole expansions for all boxes
        """
        pass

    @abstractmethod
    def output_zeros(self):
        """
        Construct arrays to store potential values for all target points
        """
        pass

    @abstractmethod
    def reorder_sources(self, source_array):
        """
        Return a copy of *source_array* in tree source order.
        *source_array* is in user source order.
        """
        pass

    @abstractmethod
    def reorder_potentials(self, potentials):
        """
        Return a copy of *potentials* in user target order.
        *source_weights* is in tree target order.
        """
        pass

    @abstractmethod
    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        """
        Return an expansions array containing multipole expansions
        in *source_boxes* due to sources with *src_weights*.
        """
        pass

    @abstractmethod
    def coarsen_multipoles(
        self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        """
        For each box in *source_parent_boxes*, gather (and translate)
        the box's children's multipole expansions in *mpoles* and add
        the resulting expansion into the box's multipole expansion
        in *mpoles*.

        :returns: *mpoles*
        """
        pass

    @abstractmethod
    def eval_direct(
        self, target_boxes, neighbor_sources_starts, neighbor_sources_lists
    ):
        """
        For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weights*

        This step amounts to looking up the corresponding entries in a
        pre-built table.

        :returns: a new potential array, see :meth:`output_zeros`.
        """
        pass

    @abstractmethod
    def multipole_to_local(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        mpole_exps,
    ):
        """
        For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.

        :returns: a new (local) expansion array.
        """
        pass

    @abstractmethod
    def eval_multipoles(
        self, level_start_target_box_nrs, target_boxes, starts, lists, mpole_exps
    ):
        """
        For each box in *target_boxes*, evaluate the multipole expansion in
        *mpole_exps* in the nearby boxes given in *starts* and *lists*, and
        return a new potential array.

        :returns: a new potential array, see :meth:`output_zeros`.
        """
        pass

    @abstractmethod
    def form_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        src_weights,
    ):
        """
        For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.

        :returns: a new local expansion array
        """
        pass

    @abstractmethod
    def refine_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        """
        For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: *local_exps*
        """
        pass

    @abstractmethod
    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        """For each box in *target_boxes*, evaluate the local expansion in
        *local_exps* and return a new potential array.

        :returns: a new potential array, see :meth:`output_zeros`.
        """
        pass

    @abstractmethod
    def finalize_potentials(self, potentials):
        """
        Postprocess the reordered potentials. This is where global scaling
        factors could be applied.
        """
        pass


# }}} End expansion wrangler interface

# {{{ sumpy based expansion wrangler code container

# from sumpy import (P2EFromSingleBox, P2EFromCSR, E2PFromSingleBox, E2PFromCSR,
# P2PFromCSR, E2EFromCSR, E2EFromChildren, E2EFromParent)

# from pytools import memoize_method

from sumpy.fmm import SumpyExpansionWranglerCodeContainer


class ExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    """Objects of this type serve as a place to keep the code needed
    for ExpansionWrangler if it is using sumpy to perform multipole
    expansion and manipulations.

    Since :class:`SumpyExpansionWrangler` necessarily must have a
    :class:`pyopencl.CommandQueue`, but this queue is allowed to be
    more ephemeral than the code, the code's lifetime
    is decoupled by storing it in this object.
    """


# }}} End sumpy based expansion wrangler code container

# vim: filetype=pyopencl:fdm=marker

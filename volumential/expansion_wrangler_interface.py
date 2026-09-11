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

__doc__ = """The stage-by-stage contract every volumential wrangler implements.

An *expansion wrangler* is the object the FMM driver in
:mod:`volumential.volume_fmm` calls once per tree stage: form multipoles,
coarsen them upward, translate, refine locals downward, evaluate.  The
interface here mirrors :mod:`boxtree.fmm` so that a volumential wrangler is a
drop-in for a boxtree one; the concrete implementations live in
:mod:`volumential.wranglers`.

.. autoclass:: ExpansionWranglerInterface
.. autoclass:: TreeIndependentDataForWranglerInterface
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, TypeAlias


logger = logging.getLogger(__name__)


#: Whatever array flavour a stage hands to the next one: a
#: :class:`pyopencl.array.Array`, a :class:`numpy.ndarray`, or an object array
#: holding one of those per output kernel.  Backends disagree on the flavour,
#: so the interface deliberately does not pin it down.
FMMArray: TypeAlias = Any

#: A per-box or per-box-pair index array, in tree order.
BoxIndexArray: TypeAlias = Any

#: What one FMM stage hands back: the stage's array (see :data:`FMMArray`)
#: paired with a timing future for the driver's profiling hooks, or ``None``
#: in place of the future when the backend does not time the stage.
StageResult: TypeAlias = Any


# {{{ expansion wrangler interface


class ExpansionWranglerInterface:
    """
    Abstract expansion handling interface.
    The interface is adapted from, and stays compatible with boxtree/fmm.

    .. note::

        ``__metaclass__`` is Python 2 spelling and has no effect here, so the
        :func:`~abc.abstractmethod` decorators below are documentation rather
        than an enforced contract: subclasses are not checked for completeness,
        and this class stays instantiable with method bodies that return
        ``None``.  Argument lists are not enforced either.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def multipole_expansion_zeros(self) -> FMMArray:
        """
        Construct arrays to store multipole expansions for all boxes
        """

    @abstractmethod
    def local_expansion_zeros(self) -> FMMArray:
        """
        Construct arrays to store multipole expansions for all boxes
        """

    @abstractmethod
    def output_zeros(self) -> FMMArray:
        """
        Construct arrays to store potential values for all target points
        """

    @abstractmethod
    def reorder_sources(self, source_array: FMMArray) -> FMMArray:
        """
        Return a copy of *source_array* in tree source order.
        *source_array* is in user source order.
        """

    @abstractmethod
    def reorder_targets(self, source_array: FMMArray) -> FMMArray:
        """
        Return a copy of *target_array* in tree source order.
        *target_array* is in user target order.
        """

    @abstractmethod
    def reorder_potentials(self, potentials: FMMArray) -> FMMArray:
        """
        Return a copy of *potentials* in user target order.
        *source_weights* is in tree target order.
        """

    @abstractmethod
    def form_multipoles(
        self,
        level_start_source_box_nrs: BoxIndexArray,
        source_boxes: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        """
        Return an expansions array containing multipole expansions
        in *source_boxes* due to sources with *src_weights*.
        """

    @abstractmethod
    def coarsen_multipoles(
        self,
        level_start_source_parent_box_nrs: BoxIndexArray,
        source_parent_boxes: BoxIndexArray,
        mpoles: FMMArray,
    ) -> StageResult:
        """
        For each box in *source_parent_boxes*, gather (and translate)
        the box's children's multipole expansions in *mpoles* and add
        the resulting expansion into the box's multipole expansion
        in *mpoles*.

        :returns: *mpoles*
        """

    @abstractmethod
    def eval_direct(
        self,
        target_boxes: BoxIndexArray,
        neighbor_sources_starts: BoxIndexArray,
        neighbor_sources_lists: BoxIndexArray,
    ) -> StageResult:
        """
        For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weights*

        This step amounts to looking up the corresponding entries in a
        pre-built table.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    @abstractmethod
    def multipole_to_local(
        self,
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        starts: BoxIndexArray,
        lists: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
        """
        For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.

        :returns: a new (local) expansion array.
        """

    @abstractmethod
    def eval_multipoles(
        self,
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        starts: BoxIndexArray,
        lists: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
        """
        For each box in *target_boxes*, evaluate the multipole expansion in
        *mpole_exps* in the nearby boxes given in *starts* and *lists*, and
        return a new potential array.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    @abstractmethod
    def form_locals(
        self,
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        starts: BoxIndexArray,
        lists: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        """
        For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.

        :returns: a new local expansion array
        """

    @abstractmethod
    def refine_locals(
        self,
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        """
        For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: *local_exps*
        """

    @abstractmethod
    def eval_locals(
        self,
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        """For each box in *target_boxes*, evaluate the local expansion in
        *local_exps* and return a new potential array.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    @abstractmethod
    def finalize_potentials(self, potentials: FMMArray) -> FMMArray:
        """
        Postprocess the reordered potentials. This is where global scaling
        factors could be applied.
        """


# }}} End expansion wrangler interface

# {{{ tree-independent-data interface


class TreeIndependentDataForWranglerInterface:
    """
    Abstract tree-independent data interface.
    The interface is adapted from, and stays compatible with boxtree/fmm.

    Holds the parts of a wrangler that outlive any single
    :class:`pyopencl.CommandQueue` -- generated code, expansion factories,
    kernel lists -- so that wranglers themselves can be short-lived.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_wrangler(self, *args: Any, **kwargs: Any) -> ExpansionWranglerInterface:
        """Makes a wrangler object."""


# }}} End tree-independent-data interface

# vim: filetype=pyopencl:fdm=marker


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

import volumential.expansion_wrangler_interface as ewi


def test_interface_allocation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()

    wrglr.multipole_expansion_zeros()
    wrglr.local_expansion_zeros()
    wrglr.output_zeros()


def test_interface_utilities():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()

    wrglr.reorder_sources(None)
    wrglr.reorder_potentials(None)
    wrglr.finalize_potentials(None)


def test_interface_multipole_formation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()

    wrglr.form_multipoles(
        level_start_source_box_nrs=None, source_boxes=None, src_weights=None
    )
    wrglr.form_locals(
        level_start_target_or_target_parent_box_nrs=None,
        target_or_target_parent_boxes=None,
        starts=None,
        lists=None,
        src_weights=None,
    )


def test_interface_multipole_evaluation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()

    wrglr.eval_direct(
        target_boxes=None,
        neighbor_sources_starts=None,
        neighbor_sources_lists=None
    )
    wrglr.eval_locals(
        level_start_target_box_nrs=None, target_boxes=None, local_exps=None
    )
    wrglr.eval_multipoles(
        level_start_target_box_nrs=None,
        target_boxes=None,
        starts=None,
        lists=None,
        mpole_exps=None,
    )


def test_interface_multipole_manipulation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()

    wrglr.coarsen_multipoles(
        level_start_source_parent_box_nrs=None,
        source_parent_boxes=None,
        mpoles=None,
    )
    wrglr.multipole_to_local(
        level_start_target_or_target_parent_box_nrs=None,
        target_or_target_parent_boxes=None,
        starts=None,
        lists=None,
        mpole_exps=None,
    )
    wrglr.refine_locals(
        level_start_target_or_target_parent_box_nrs=None,
        target_or_target_parent_boxes=None,
        local_exps=None,
    )

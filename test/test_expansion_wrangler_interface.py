import volumential.expansion_wrangler_interface as ewi


def test_interface_allocation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()
    try:
        wrglr.multipole_expansion_zeros()
        wrglr.local_expansion_zeros()
        wrglr.output_zeros()
    except NotImplementedError:
        assert (False)
    assert (True)


def test_interface_utilities():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()
    try:
        wrglr.reorder_sources(None)
        wrglr.reorder_potentials(None)
        wrglr.finalize_potentials(None)
    except NotImplementedError:
        assert (False)
    assert (True)


def test_interface_multipole_formation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()
    try:
        wrglr.form_multipoles(
            level_start_source_box_nrs=None,
            source_boxes=None,
            src_weights=None)
        wrglr.form_locals(
            level_start_target_or_target_parent_box_nrs=None,
            target_or_target_parent_boxes=None,
            starts=None,
            lists=None,
            src_weights=None)
    except NotImplementedError:
        assert (False)
    assert (True)


def test_interface_multipole_evaluation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()
    try:
        wrglr.eval_direct(
            target_boxes=None,
            neighbor_sources_starts=None,
            neighbor_sources_lists=None)
        wrglr.eval_locals(
            level_start_target_box_nrs=None,
            target_boxes=None,
            local_exps=None)
        wrglr.eval_multipoles(
            level_start_target_box_nrs=None,
            target_boxes=None,
            starts=None,
            lists=None,
            mpole_exps=None)
    except NotImplementedError:
        assert (False)
    assert (True)


def test_interface_multipole_manipulation():
    # make the interface instantiable by overriding its set of abstract methods
    ewi.ExpansionWranglerInterface.__abstractmethods__ = frozenset()
    wrglr = ewi.ExpansionWranglerInterface()
    try:
        wrglr.coarsen_multipoles(
            level_start_source_parent_box_nrs=None,
            source_parent_boxes=None,
            mpoles=None)
        wrglr.multipole_to_local(
            level_start_target_or_target_parent_box_nrs=None,
            target_or_target_parent_boxes=None,
            starts=None,
            lists=None,
            mpole_exps=None)
        wrglr.refine_locals(
            level_start_target_or_target_parent_box_nrs=None,
            target_or_target_parent_boxes=None,
            local_exps=None)
    except NotImplementedError:
        assert (False)
    assert (True)

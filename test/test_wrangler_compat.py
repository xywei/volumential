__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

__doc__ = """Compatibility guards for the split of ``expansion_wrangler_fpnd``.

The wrangler implementation moved into the :mod:`volumential.wranglers`
package.  These tests pin down what callers are allowed to rely on:

* every name that used to live in :mod:`volumential.expansion_wrangler_fpnd`
  is still importable from there, and is the very same object as in its new
  home;
* the method-resolution order of the two wranglers still puts
  :class:`~volumential.expansion_wrangler_interface.ExpansionWranglerInterface`
  ahead of the sumpy/fmmlib base wrangler, as it did before the split;
* patching a helper on the legacy module still affects its callers.
"""

import numpy as np
import pytest

from volumential.expansion_wrangler_interface import ExpansionWranglerInterface


# Names importable from volumential.expansion_wrangler_fpnd before the split
# into volumential.wranglers.  Two groups are absent on purpose:
#
# * ``_evaluate_scalar_arithmetic_entry`` -- unreferenced dead code (an exact
#   copy of the live one in volumential.orbit_arithmetic), removed with the
#   split;
# * the names the pre-split module only happened to bind because it imported
#   them (``np``, ``cl``, ``json``, ``LaplaceKernel``,
#   ``NearFieldInteractionTable``, ...).  Nothing in the tree imports those
#   through this module, and each is still importable from its real home.
LEGACY_NAMES = (
    "FPNDExpansionWrangler",
    "FPNDFMMLibExpansionWrangler",
    "FPNDFMMLibTreeIndependentDataForWrangler",
    "FPNDSumpyExpansionWrangler",
    "FPNDSumpyTreeIndependentDataForWrangler",
    "FPNDTreeIndependentDataForWrangler",
    "HelmholtzSplitCacheAccounting",
    "SumpyTimingFuture",
    "_ARITHMETIC_RECONSTRUCTION_KINDS",
    "_HelmholtzSplitSeriesRemainderKernel",
    "_ORBIT_RECONSTRUCTION_HASH_MULTIPLIER",
    "_RadialPowerKernel",
    "_RadialPowerLogKernel",
    "_array_layout_cache_token",
    "_barycentric_interp_matrix",
    "_build_arithmetic_orbit_reconstruction",
    "_build_fast_scalar_arithmetic_orbit_reconstruction",
    "_build_generated_orbit_reconstruction",
    "_build_open_addressed_int_lookup",
    "_build_scalar_arithmetic_orbit_reconstruction",
    "_build_sparse_sign_lookup",
    "_canonical_case_from_axis_descriptors",
    "_case_arithmetic_axis_descriptors",
    "_compact_arithmetic_case_value_count",
    "_compute_box_local_ids",
    "_derive_source_kernels_from_target_kernels",
    "_entry_has_odd_reconstruction_stabilizer",
    "_evaluate_arithmetic_orbit_entry",
    "_evaluate_compact_arithmetic_orbit_entry",
    "_evaluate_generated_orbit_reconstruction",
    "_extract_symmetry_source_direction",
    "_find_directional_source_derivative_kernel",
    "_format_helmholtz_split_term_key",
    "_gauss_legendre_nodes_and_weights",
    "_kernel_axis_preserving_arithmetic_symmetry",
    "_nearfield_table_payload_bytes",
    "_normalize_helmholtz_split_term_key",
    "_prepare_table_data_and_entry_map",
    "_queue_from_array_like",
    "_rank_multiset",
    "_resolve_queue",
    "_select_split_order_from_rho",
    "_select_split_order_from_rho_components",
    "_table_data_fingerprint",
    "_target_kernels_include_source_derivatives",
    "_validate_table_box_particle_layout",
    "_validate_table_box_particle_layout_cached",
    "inverse_id_map",
    "level_to_rscale",
    "logger",
)


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_legacy_name_is_importable(name):
    import volumential.expansion_wrangler_fpnd as legacy

    assert hasattr(legacy, name), f"{name} is no longer importable from the shim"


def test_legacy_names_are_the_same_objects_as_in_the_package():
    import volumential.expansion_wrangler_fpnd as legacy
    from volumential.wranglers import arithmetic_orbits, sumpy_backend, table_data

    assert legacy.FPNDExpansionWrangler is sumpy_backend.FPNDExpansionWrangler
    assert (
        legacy._prepare_table_data_and_entry_map
        is table_data._prepare_table_data_and_entry_map
    )
    assert (
        legacy._ARITHMETIC_RECONSTRUCTION_KINDS
        is arithmetic_orbits._ARITHMETIC_RECONSTRUCTION_KINDS
    )


def test_every_legacy_name_is_the_package_object_itself():
    """The shim must re-export, never re-implement.

    ``logger`` is the one exception: the shim keeps a logger of its own so that
    ``volumential.expansion_wrangler_fpnd`` stays a usable logger name.
    """
    import sys

    import volumential.expansion_wrangler_fpnd as legacy

    rebound = []
    for name in LEGACY_NAMES:
        if name == "logger":
            continue
        obj = getattr(legacy, name)
        home = getattr(obj, "__module__", None)
        if home is None or not home.startswith("volumential.wranglers"):
            continue
        if getattr(sys.modules[home], name, None) is not obj:
            rebound.append(f"{name} ({home})")
    assert not rebound, f"shim rebinds instead of re-exporting: {rebound}"


def test_package_public_api():
    import volumential.wranglers as wr

    for name in wr.__all__:
        assert hasattr(wr, name), f"{name} is advertised in __all__ but missing"


def test_no_mixin_attribute_is_shadowed_or_shadowing():
    """The split is only inert while the mixins collide with nobody.

    A mixin sits between :class:`ExpansionWranglerInterface` and the sumpy or
    fmmlib base wrangler.  If a name it defines ever also appears earlier in
    the MRO (the interface) or later (the backend base), the method that runs
    stops being the one that ran before the split.
    """
    from volumential.wranglers import (
        FPNDFMMLibExpansionWrangler,
        FPNDSumpyExpansionWrangler,
    )

    problems = []
    for cls in (FPNDSumpyExpansionWrangler, FPNDFMMLibExpansionWrangler):
        mro = cls.__mro__
        for pos, entry in enumerate(mro):
            if not entry.__name__.endswith("Mixin"):
                continue
            for name in vars(entry):
                if name.startswith("__"):
                    continue
                others = [
                    other.__name__
                    for i, other in enumerate(mro)
                    if i != pos and name in vars(other)
                ]
                if others:
                    problems.append(f"{cls.__name__}.{name}: also on {others}")
    assert not problems, f"mixin name collisions in the MRO: {problems}"


def test_default_aliases_track_the_sumpy_backend():
    import volumential.expansion_wrangler_fpnd as legacy

    assert issubclass(
        legacy.FPNDExpansionWrangler, legacy.FPNDSumpyExpansionWrangler
    )
    assert issubclass(
        legacy.FPNDTreeIndependentDataForWrangler,
        legacy.FPNDSumpyTreeIndependentDataForWrangler,
    )


@pytest.mark.parametrize(
    "wrangler_name",
    ["FPNDSumpyExpansionWrangler", "FPNDFMMLibExpansionWrangler"],
)
def test_interface_still_precedes_the_backend_base(wrangler_name):
    """The interface's ``pass``-bodied methods win over the backend's.

    This ordering predates the split and several methods rely on it, so the
    mixins introduced by the split must sit *after* the interface.
    """
    import volumential.expansion_wrangler_fpnd as legacy

    mro = getattr(legacy, wrangler_name).__mro__
    assert ExpansionWranglerInterface in mro

    interface_pos = mro.index(ExpansionWranglerInterface)
    backend_base = [
        cls for cls in mro[1:]
        if cls.__module__.startswith(("sumpy.", "boxtree."))
    ]
    assert backend_base, "no sumpy/boxtree base class in the MRO"
    assert interface_pos < mro.index(backend_base[0])


def test_mixins_are_in_the_wrangler_mro():
    from volumential.wranglers import (
        FMMLibBatchedStagesMixin,
        FPNDFMMLibExpansionWrangler,
        FPNDSumpyExpansionWrangler,
        HelmholtzSplitCorrectionMixin,
        NearFieldPayloadCacheMixin,
    )

    assert issubclass(FPNDSumpyExpansionWrangler, NearFieldPayloadCacheMixin)
    assert issubclass(FPNDSumpyExpansionWrangler, HelmholtzSplitCorrectionMixin)
    assert issubclass(FPNDFMMLibExpansionWrangler, NearFieldPayloadCacheMixin)
    assert issubclass(FPNDFMMLibExpansionWrangler, FMMLibBatchedStagesMixin)


def test_both_backends_share_one_nearfield_payload_cache_implementation():
    from volumential.wranglers import (
        FPNDFMMLibExpansionWrangler,
        FPNDSumpyExpansionWrangler,
    )

    assert (
        FPNDSumpyExpansionWrangler._get_cached_nearfield_payload
        is FPNDFMMLibExpansionWrangler._get_cached_nearfield_payload
    )


def test_patching_the_legacy_validator_still_reaches_the_cached_wrapper(monkeypatch):
    """``_validate_table_box_particle_layout`` is patched on the legacy module."""
    import volumential.expansion_wrangler_fpnd as legacy

    calls = []

    def _fake_validate(queue, tree, target_boxes, source_boxes, n_q_points):
        calls.append(int(n_q_points))

    monkeypatch.setattr(
        legacy, "_validate_table_box_particle_layout", _fake_validate
    )

    boxes = np.array([0, 1], dtype=np.int32)
    legacy._validate_table_box_particle_layout_cached(
        queue=None,
        tree=object(),
        target_boxes=boxes,
        source_boxes=boxes,
        n_q_points=16,
        validation_cache=None,
    )

    assert calls == [16]


def test_the_fft_plan_hook_is_sumpys_cached_one():
    """The FFT plan must come from sumpy's memoizing hook.

    This class used to reimplement ``opencl_fft_app`` as a bare call to
    ``sumpy.tools.get_opencl_fft_app``, which is a copy of sumpy's body with
    the ``memoize_in`` cache dropped: it rebuilt the loopy translation unit
    or VkFFT plan on every FFT stage.  Inheriting also keeps us out of the
    way of that signature, which inducer/sumpy@16e0e3c5 ("refactor fft
    apps") turned keyword-only.
    """
    import inspect

    from sumpy.fmm import SumpyTreeIndependentDataForWrangler

    from volumential.wranglers.sumpy_backend import (
        FPNDSumpyTreeIndependentDataForWrangler,
    )

    assert (
        FPNDSumpyTreeIndependentDataForWrangler.opencl_fft_app
        is SumpyTreeIndependentDataForWrangler.opencl_fft_app
    )

    inspect.signature(
        FPNDSumpyTreeIndependentDataForWrangler.opencl_fft_app
    ).bind(
        object(),
        shape=(4,),
        dtype=np.dtype(np.complex128),
        inverse=False,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__, "-v"])

# vim: filetype=pyopencl:foldmethod=marker

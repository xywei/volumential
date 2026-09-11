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

__doc__ = """Compatibility shim for the fpnd expansion wranglers.

The implementation now lives in :mod:`volumential.wranglers`, one module per
concern.  This module keeps the historical import path working: every name that
was importable from ``volumential.expansion_wrangler_fpnd`` -- public classes
and internal helpers alike -- is re-exported here and refers to the same object
as its new home.

New code should import from :mod:`volumential.wranglers` (or the specific
submodule) instead.
"""

import logging

from volumential.wranglers import (
    FPNDExpansionWrangler,
    FPNDFMMLibExpansionWrangler,
    FPNDFMMLibTreeIndependentDataForWrangler,
    FPNDSumpyExpansionWrangler,
    FPNDSumpyTreeIndependentDataForWrangler,
    FPNDTreeIndependentDataForWrangler,
    HelmholtzSplitCacheAccounting,
    SumpyTimingFuture,
    inverse_id_map,
    level_to_rscale,
)
from volumential.wranglers.arithmetic_orbits import (  # noqa: F401
    _ARITHMETIC_RECONSTRUCTION_KINDS,
    _build_arithmetic_orbit_reconstruction,
    _build_fast_scalar_arithmetic_orbit_reconstruction,
    _build_scalar_arithmetic_orbit_reconstruction,
    _canonical_case_from_axis_descriptors,
    _case_arithmetic_axis_descriptors,
    _compact_arithmetic_case_value_count,
    _entry_has_odd_reconstruction_stabilizer,
    _evaluate_arithmetic_orbit_entry,
    _evaluate_compact_arithmetic_orbit_entry,
    _kernel_axis_preserving_arithmetic_symmetry,
    _rank_multiset,
)
from volumential.wranglers.barycentric import (  # noqa: F401
    _barycentric_interp_matrix,
    _gauss_legendre_nodes_and_weights,
)
from volumential.wranglers.box_layout import (  # noqa: F401
    _array_layout_cache_token,
    _compute_box_local_ids,
    _validate_table_box_particle_layout,
    _validate_table_box_particle_layout_cached,
)
from volumential.wranglers.device_arrays import (  # noqa: F401
    _queue_from_array_like,
    _resolve_queue,
)
from volumential.wranglers.kernel_symmetry import (  # noqa: F401
    _derive_source_kernels_from_target_kernels,
    _extract_symmetry_source_direction,
    _find_directional_source_derivative_kernel,
    _target_kernels_include_source_derivatives,
)
from volumential.wranglers.kernels import (  # noqa: F401
    _HelmholtzSplitSeriesRemainderKernel,
    _RadialPowerKernel,
    _RadialPowerLogKernel,
)
from volumential.wranglers.orbit_generated import (  # noqa: F401
    _build_generated_orbit_reconstruction,
    _evaluate_generated_orbit_reconstruction,
)
from volumential.wranglers.orbit_lookup import (  # noqa: F401
    _ORBIT_RECONSTRUCTION_HASH_MULTIPLIER,
    _build_open_addressed_int_lookup,
    _build_sparse_sign_lookup,
)
from volumential.wranglers.split_terms import (  # noqa: F401
    _format_helmholtz_split_term_key,
    _nearfield_table_payload_bytes,
    _normalize_helmholtz_split_term_key,
    _select_split_order_from_rho,
    _select_split_order_from_rho_components,
)
from volumential.wranglers.table_data import (  # noqa: F401
    _prepare_table_data_and_entry_map,
    _table_data_fingerprint,
)


logger = logging.getLogger(__name__)


__all__ = [
    "FPNDExpansionWrangler",
    "FPNDFMMLibExpansionWrangler",
    "FPNDFMMLibTreeIndependentDataForWrangler",
    "FPNDSumpyExpansionWrangler",
    "FPNDSumpyTreeIndependentDataForWrangler",
    "FPNDTreeIndependentDataForWrangler",
    "HelmholtzSplitCacheAccounting",
    "SumpyTimingFuture",
    "inverse_id_map",
    "level_to_rscale",
]

# vim: filetype=pyopencl:foldmethod=marker

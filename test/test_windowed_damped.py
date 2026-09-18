"""Tests for the damped complex-frequency windowed assembly (E8).

The branch contract is the module's pointwise selection: the decaying root
on the Yukawa ray and the outgoing lower-half-plane limit on the negative
(Helmholtz) ray.  Both real rays must therefore reproduce the real-kernel
assemblies exactly from the same cached real channel family, and interior
phases must carry the |zeta|-only conditioning certificate.
"""

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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys

import numpy as np
import pytest

from volumential.rke_table_assembly import (
    RKEWindowCoverageError,
    assemble_windowed_damped_table,
    assemble_windowed_parameterized_table,
    damped_kernel_radial,
)


ROOT_EXTENT = 2.0
WINDOW_THETA = 16.0
# small, fast configuration; the orders only need to be legal, not converged
FAST_KW = {
    "source_box_level": 3,
    "root_extent": ROOT_EXTENT,
    "window_theta": WINDOW_THETA,
    "p_star": 4,
    "smooth_quad_order": 12,
    "chan_regular_order": 8,
    "chan_radial_order": 21,
}


# {{{ kernel branch contract

def test_damped_kernel_reduces_to_yukawa_on_positive_ray():
    import scipy.special as sps

    lam = 3.0
    r = np.linspace(0.05, 1.5, 40)
    kernel = damped_kernel_radial(2, lam * lam)
    expected = sps.k0(lam * r) / (2.0 * np.pi)
    assert np.max(np.abs(kernel(r) - expected)) < 1e-15
    assert np.max(np.abs(np.imag(kernel(r)))) == 0.0

    kernel_3d = damped_kernel_radial(3, lam * lam)
    expected_3d = np.exp(-lam * r) / (4.0 * np.pi * r)
    assert np.max(np.abs(kernel_3d(r) - expected_3d)) < 1e-15


def test_damped_kernel_is_outgoing_on_negative_ray():
    import scipy.special as sps

    k = 3.0
    r = np.linspace(0.05, 1.5, 40)
    kernel = damped_kernel_radial(2, -(k * k))
    expected = 0.25j * sps.hankel1(0, k * r)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(kernel(r) - expected)) < 1e-14 * scale

    kernel_3d = damped_kernel_radial(3, -(k * k))
    expected_3d = np.exp(1j * k * r) / (4.0 * np.pi * r)
    scale_3d = np.max(np.abs(expected_3d))
    assert np.max(np.abs(kernel_3d(r) - expected_3d)) < 1e-14 * scale_3d


@pytest.mark.parametrize("bad", [0.0, complex(np.nan, 0.0), complex(0.0, np.inf)])
def test_damped_kernel_rejects_zero_and_nonfinite_zeta(bad):
    with pytest.raises(ValueError):
        damped_kernel_radial(2, bad)

# }}}


# {{{ assembly parity and certificates

@pytest.fixture(scope="module")
def damped_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("damped-cache") / "channels.db"


def test_damped_assembly_matches_yukawa_on_positive_ray(damped_cache):
    lam = 4.0
    table_y, cert_y = assemble_windowed_parameterized_table(
        damped_cache, 2, "Yukawa", 2, lam, **FAST_KW
    )
    table_d, cert_d = assemble_windowed_damped_table(
        damped_cache, 2, 2, lam * lam, **FAST_KW
    )
    ids = np.asarray(table_y.get_reduced_entry_ids(), dtype=np.int64)
    values_y = np.asarray(table_y.get_entry_data_for_full_indices(ids))
    values_d = np.asarray(table_d.get_entry_data_for_full_indices(ids))
    scale = np.max(np.abs(values_y))
    assert np.max(np.abs(values_d - values_y)) < 1e-14 * scale
    assert cert_d["kernel_type"] == "Damped"
    assert cert_d["zeta_phase_fraction"] == 0.0
    assert cert_d["condition_number"] == pytest.approx(
        cert_y["condition_number"], rel=1e-12
    )


def test_damped_assembly_matches_helmholtz_on_negative_ray(damped_cache):
    k = 4.0
    table_h, _ = assemble_windowed_parameterized_table(
        damped_cache, 2, "Helmholtz", 2, k, **FAST_KW
    )
    table_d, cert_d = assemble_windowed_damped_table(
        damped_cache, 2, 2, -(k * k), **FAST_KW
    )
    ids = np.asarray(table_h.get_reduced_entry_ids(), dtype=np.int64)
    values_h = np.asarray(table_h.get_entry_data_for_full_indices(ids))
    values_d = np.asarray(table_d.get_entry_data_for_full_indices(ids))
    scale = np.max(np.abs(values_h))
    assert np.max(np.abs(values_d - values_h)) < 1e-13 * scale
    assert cert_d["zeta_phase_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("phase_fraction", [0.25, 0.5, 0.75])
def test_damped_interior_phases_certify_by_modulus(
    damped_cache, phase_fraction
):
    # |zeta| fixed at the phi = 0 value: the conditioning contract depends
    # on the modulus only, so interior phases must certify with the same
    # coefficient bound and a comparable condition number
    lam = 4.0
    zeta = (lam * lam) * np.exp(1j * np.pi * phase_fraction)
    _, cert_real = assemble_windowed_damped_table(
        damped_cache, 2, 2, lam * lam, **FAST_KW
    )
    table, cert = assemble_windowed_damped_table(
        damped_cache, 2, 2, zeta, **FAST_KW
    )
    assert cert["zeta_phase_fraction"] == pytest.approx(phase_fraction)
    assert cert["coefficient_bound"] == pytest.approx(
        cert_real["coefficient_bound"]
    )
    assert cert["truncation_tail_bound"] == 0.0
    assert cert["condition_number"] < 10.0 * max(
        cert_real["condition_number"], 1.0
    )
    ids = np.asarray(table.get_reduced_entry_ids(), dtype=np.int64)
    values = np.asarray(table.get_entry_data_for_full_indices(ids))
    assert np.all(np.isfinite(values))
    # a genuinely damped point has genuinely complex entries
    assert np.max(np.abs(values.imag)) > 0.0


def test_damped_assembly_refuses_outside_coverage(damped_cache):
    box_extent = ROOT_EXTENT * 0.5 ** FAST_KW["source_box_level"]
    mu_outside = 1.5 * WINDOW_THETA / box_extent
    zeta = (mu_outside**2) * np.exp(0.5j * np.pi)
    with pytest.raises(RKEWindowCoverageError):
        assemble_windowed_damped_table(damped_cache, 2, 2, zeta, **FAST_KW)


def test_damped_assembly_rejects_zero_zeta(damped_cache):
    with pytest.raises(ValueError, match="nonzero"):
        assemble_windowed_damped_table(damped_cache, 2, 2, 0.0, **FAST_KW)

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

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

from itertools import pairwise

import numpy as np
import pytest

from volumential.rke_table_assembly import (
    _assemble_windowed_for_zeta,
    _duffy_channel_entry_values,
    _tensor_product_gauss_points,
    _windowed_channel_kernel_func,
    _windowed_channel_skeleton,
    assemble_parameterized_table,
    assemble_windowed_parameterized_table,
    windowed_channel_profile,
    windowed_remainder_profile,
)

WINDOW_THETA = 16.0
ROOT_EXTENT = 2.0

# Channel quadrature orders used consistently per dimension so every test
# shares one cached channel family per (dim, q, level).  2D q=3 puts
# interpolation nodes close to the box edge, and the resulting high-aspect
# Duffy triangles converge slowly in the angular order (the same effect
# governs direct builds), so the 2D channels use angular order 48 — which
# is also the module's resolved 2D default (see
# test_default_channel_orders); the 3D q=2 geometry is mild (measured
# drift of 14/45 versus 28/61 is below 1e-7 relative) and 14/45 keeps the
# one-time build under a minute.
CHAN_ORDERS_2D = {"chan_regular_order": 48, "chan_radial_order": 61}
CHAN_ORDERS_3D = {"chan_regular_order": 14, "chan_radial_order": 45}


def _get_queue_or_skip():
    import pyopencl as cl

    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception as exc:
        pytest.skip(f"no OpenCL context available: {exc}")
    return cl.CommandQueue(ctx)


def _box_extent(source_box_level):
    return ROOT_EXTENT * 0.5**source_box_level


def _kernel_radial(dim, kernel_type, parameter):
    import scipy.special as sps

    parameter = complex(parameter)
    if kernel_type == "Yukawa":
        if dim == 2:
            return lambda r: sps.kv(0, parameter * np.asarray(r)) / (
                2.0 * np.pi
            )
        return lambda r: np.exp(-parameter * np.asarray(r)) / (
            4.0 * np.pi * np.asarray(r)
        )
    assert kernel_type == "Helmholtz"
    if dim == 2:
        return lambda r: 0.25j * sps.hankel1(
            0, parameter * np.asarray(r, dtype=np.complex128)
        )
    return lambda r: np.exp(1j * parameter * np.asarray(r)) / (
        4.0 * np.pi * np.asarray(r)
    )


def _windowed_remainder(dim, kernel_type, parameter, source_box_level, p_star):
    box_extent = _box_extent(source_box_level)
    window_scale = (box_extent / WINDOW_THETA) ** 2
    if kernel_type == "Yukawa":
        zeta = complex(parameter) ** 2
    else:
        zeta = -(complex(parameter) ** 2)
    prefactor = 1.0 / (2.0 * np.pi) if dim == 2 else 1.0 / (4.0 * np.pi)
    profiles = [
        windowed_channel_profile(dim, m, window_scale) for m in range(p_star)
    ]
    kernel = _kernel_radial(dim, kernel_type, parameter)

    def remainder(r):
        coefficient = 1.0 + 0.0j
        channel_sum = np.zeros(np.shape(np.asarray(r)), dtype=np.complex128)
        for m in range(p_star):
            channel_sum = channel_sum + coefficient * profiles[m](r)
            coefficient = coefficient * (-zeta) / (m + 1)
        return kernel(r) - prefactor * channel_sum

    def channel_sum_only(r):
        coefficient = 1.0 + 0.0j
        acc = np.zeros(np.shape(np.asarray(r)), dtype=np.complex128)
        for m in range(p_star):
            acc = acc + coefficient * profiles[m](r)
            coefficient = coefficient * (-zeta) / (m + 1)
        return prefactor * acc

    return kernel, channel_sum_only, remainder


def _scalar_direct_entries(
    dim,
    q_order,
    source_box_level,
    entry_ids,
    kernel_radial,
    deg_theta,
    radial_quad_order,
    complex_valued=False,
):
    """Independent scalar-DuffyRadial reference entries for a radial kernel
    (real and imaginary parts separately when complex)."""
    parts = ("real", "imag") if complex_valued else ("real",)
    part_values = {}
    # only ``kernel_func`` differs between the parts, so one skeleton serves
    table = _windowed_channel_skeleton(
        dim, q_order, source_box_level, ROOT_EXTENT, WINDOW_THETA, 0
    )
    for part in parts:

        def kernel_func(x, y=None, z=None, _part=part):
            coords = [c for c in (x, y, z) if c is not None][:dim]
            r = np.sqrt(
                sum(np.asarray(c, dtype=np.float64) ** 2 for c in coords)
            )
            value = kernel_radial(r)
            return getattr(np, _part)(value)

        table.kernel_func = kernel_func
        part_values[part] = np.array(
            [
                table.compute_table_entry_duffy_radial(
                    int(entry_id),
                    radial_rule="tanh-sinh-fast",
                    deg_theta=deg_theta,
                    radial_quad_order=radial_quad_order,
                    mp_dps=50,
                )[1]
                for entry_id in entry_ids
            ],
            dtype=np.float64,
        )
    values = part_values["real"].astype(np.complex128)
    if complex_valued:
        values = values + 1j * part_values["imag"]
    return values


def _tensor_gauss_direct_q1_entries(table, entry_ids, kernel_radial, order):
    """Independent direct quadrature for nonsingular q=1 table entries."""
    assert int(table.quad_order) == 1
    extent = float(table.source_box_extent)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axis_nodes = 0.5 * extent * (nodes + 1.0)
    axis_weights = 0.5 * extent * weights
    grids = np.meshgrid(axis_nodes, axis_nodes, axis_nodes, indexing="ij")
    weight_tensor = (
        axis_weights[:, np.newaxis, np.newaxis]
        * axis_weights[np.newaxis, :, np.newaxis]
        * axis_weights[np.newaxis, np.newaxis, :]
    )

    values = []
    for entry_id in entry_ids:
        case_index = int(entry_id // table.n_pairs)
        pair_id = int(entry_id % table.n_pairs)
        source_mode = pair_id // table.n_q_points
        target_index = pair_id % table.n_q_points
        assert source_mode == 0
        target = np.asarray(
            table.find_target_point(target_index, case_index)
        )
        assert np.max(np.abs(target.imag)) == 0.0
        target = target.real
        radius = np.sqrt(
            sum((grids[axis] - target[axis]) ** 2 for axis in range(3))
        )
        values.append(np.sum(kernel_radial(radius) * weight_tensor))
    return np.asarray(values)


def _spot_entry_ids(entry_ids, values, n_top=6, n_spread=6):
    """A deterministic subset: the largest-magnitude entries plus an even
    spread across the reduced list."""
    order = np.argsort(-np.abs(np.asarray(values)))
    chosen = list(order[:n_top])
    chosen.extend(
        int(i) for i in np.linspace(0, len(entry_ids) - 1, n_spread)
    )
    positions = sorted({int(i) for i in chosen})
    return positions


# {{{ T1: channel profiles against mpmath, and vectorized-vs-scalar Duffy

@pytest.mark.parametrize("dim", [2, 3])
def test_channel_profiles_match_mpmath(dim):
    import mpmath as mp

    old_dps = mp.mp.dps
    mp.mp.dps = 40
    try:
        box_extent = 0.25 if dim == 2 else 0.5
        window_scale = (box_extent / WINDOW_THETA) ** 2
        radii = np.geomspace(
            1e-6 * box_extent, 3.0 * np.sqrt(dim) * box_extent, 40
        )
        for m in range(6):
            profile = windowed_channel_profile(dim, m, window_scale)
            values = profile(radii)
            for radius, value in zip(radii, values, strict=True):
                x = mp.mpf(radius) ** 2 / (4 * mp.mpf(window_scale))
                if dim == 2:
                    reference = (
                        mp.mpf("0.5")
                        * (mp.mpf(radius) ** 2 / 4) ** m
                        * mp.gammainc(-m, x, mp.inf)
                    )
                else:
                    reference = (
                        (mp.mpf(radius) ** 2 / 4)
                        ** (mp.mpf(m) - mp.mpf("0.5"))
                        * mp.gammainc(mp.mpf("0.5") - m, x, mp.inf)
                        / (2 * mp.sqrt(mp.pi))
                    )
                reference = float(reference)
                assert abs(value - reference) <= 1e-13 * max(
                    1.0, abs(reference)
                ), (dim, m, radius)
    finally:
        mp.mp.dps = old_dps


@pytest.mark.parametrize(("dim", "m", "x"), [
    (2, 20, 60.0),
    (3, 20, 60.0),
    (2, 60, 240.0),
    (3, 60, 240.0),
    (3, 200, 240.0),
])
def test_high_order_channel_profiles_remain_positive_and_stable(dim, m, x):
    import mpmath as mp

    radius = 2.0 * np.sqrt(x)
    actual = windowed_channel_profile(dim, m, 1.0)(radius)

    old_dps = mp.mp.dps
    mp.mp.dps = 100
    try:
        x_mp = mp.mpf(str(x))
        if dim == 2:
            expected = mp.mpf("0.5") * x_mp**m * mp.gammainc(
                -m, x_mp, mp.inf
            )
        else:
            expected = x_mp ** (m - mp.mpf("0.5")) * mp.gammainc(
                mp.mpf("0.5") - m, x_mp, mp.inf
            ) / (2 * mp.sqrt(mp.pi))
        expected = float(expected)
    finally:
        mp.mp.dps = old_dps

    assert actual > 0.0
    assert actual == pytest.approx(expected, rel=2.0e-12, abs=0.0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("zeta_sign", [-1, 1])
def test_high_order_small_extent_remainder_matches_mpmath(dim, zeta_sign):
    import mpmath as mp

    box_extent = 1.0e-3
    window_scale = (box_extent / WINDOW_THETA) ** 2
    zeta = zeta_sign * (WINDOW_THETA / box_extent) ** 2
    radii = box_extent * np.array([0.05, 0.1, 0.25])

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    remainder = windowed_remainder_profile(
        dim, zeta, zero_kernel, window_scale, 60
    )
    values = np.asarray(remainder(radii), dtype=np.complex128)
    assert np.all(np.isfinite(values))

    old_dps = mp.mp.dps
    mp.mp.dps = 80
    try:
        t_w = mp.mpf(window_scale)
        scaled_zeta = mp.mpf(zeta_sign)
        prefactor = 1 / (2 * mp.pi) if dim == 2 else 1 / (4 * mp.pi)
        reference = []
        for radius in radii:
            x = mp.mpf(radius) ** 2 / (4 * t_w)
            coefficient = mp.mpf(1)
            channel_sum = mp.mpc(0)
            for m in range(60):
                if dim == 2:
                    profile = mp.mpf("0.5") * x**m * mp.gammainc(
                        -m, x, mp.inf
                    )
                else:
                    profile = (
                        t_w ** mp.mpf("-0.5")
                        * x ** (mp.mpf(m) - mp.mpf("0.5"))
                        * mp.gammainc(mp.mpf("0.5") - m, x, mp.inf)
                        / (2 * mp.sqrt(mp.pi))
                    )
                channel_sum += coefficient * profile
                coefficient *= -scaled_zeta / (m + 1)
            reference.append(complex(-prefactor * channel_sum))
    finally:
        mp.mp.dps = old_dps

    np.testing.assert_allclose(
        values, np.asarray(reference), rtol=2.0e-11, atol=1.0e-12
    )


def test_high_order_small_extent_assembly_stays_finite(tmp_path):
    table, certificate = assemble_windowed_parameterized_table(
        tmp_path / "p60.sqlite",
        2,
        "Yukawa",
        1,
        1.6e4,
        root_extent=1.0e-3,
        window_theta=WINDOW_THETA,
        p_star=60,
        smooth_quad_order=8,
        chan_regular_order=2,
        chan_radial_order=7,
    )
    values = np.asarray(table.get_reduced_table_data()[1])
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(certificate["per_channel_peak"]))
    assert certificate["coefficient_bound"] == pytest.approx(1.0)


def test_vectorized_channel_builder_matches_scalar_duffy():
    # identical node sets make the agreement order-independent, so low
    # orders suffice to certify the vectorized evaluator entry-for-entry
    cases = [
        (2, 3, 3, 1, 8, 15, (0, 120, 356)),
        (3, 2, 2, 0, 4, 9, (0, 128)),
    ]
    for dim, q_order, level, m, deg, radial, picks in cases:
        table = _windowed_channel_skeleton(
            dim, q_order, level, ROOT_EXTENT, WINDOW_THETA, m
        )
        window_scale = (_box_extent(level) / WINDOW_THETA) ** 2
        entry_ids, values = _duffy_channel_entry_values(
            table,
            windowed_channel_profile(dim, m, window_scale),
            deg,
            radial,
        )
        table.kernel_func = _windowed_channel_kernel_func(
            dim, m, window_scale
        )
        # the picks are fixed positions in the reduced list; a shrunk entry
        # count must fail as an assertion here, not as an opaque IndexError
        assert len(entry_ids) > max(picks), (dim, q_order, len(entry_ids))
        for pick in picks:
            _, reference = table.compute_table_entry_duffy_radial(
                int(entry_ids[pick]),
                radial_rule="tanh-sinh-fast",
                deg_theta=deg,
                radial_quad_order=radial,
                mp_dps=50,
            )
            assert abs(values[pick] - reference) <= 1e-12 * max(
                1.0, abs(reference)
            ), (dim, pick)

# }}}


# {{{ T2: germ cancellation at the design edge theta = Theta

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_type", ["Yukawa", "Helmholtz"])
def test_germ_cancellation_at_design_theta(dim, kernel_type):
    source_box_level = 3 if dim == 2 else 2
    box_extent = _box_extent(source_box_level)
    parameter = WINDOW_THETA / box_extent
    kernel, channel_sum, remainder = _windowed_remainder(
        dim, kernel_type, parameter, source_box_level, p_star=6
    )

    r_tiny = np.array([1e-7 * box_extent, 1e-5 * box_extent])
    kernel_tiny = np.asarray(kernel(r_tiny), dtype=np.complex128)
    channels_tiny = np.asarray(channel_sum(r_tiny), dtype=np.complex128)
    remainder_tiny = np.asarray(remainder(r_tiny), dtype=np.complex128)

    assert np.all(np.isfinite(remainder_tiny))

    # the kernel and the channel sum individually grow as r -> 0 ...
    kernel_growth = abs(kernel_tiny[0] - kernel_tiny[1])
    channel_growth = abs(channels_tiny[0] - channels_tiny[1])
    assert kernel_growth > 0.1
    assert channel_growth > 0.1

    # ... while their difference does not: the log (2D) / 1-over-r (3D)
    # germ coefficients of kernel and channel sum agree
    remainder_growth = abs(remainder_tiny[0] - remainder_tiny[1])
    assert remainder_growth <= 1e-6 * max(1.0, kernel_growth)

    # R stays at the scale it has on the resolved part of the range
    moderate = np.abs(
        np.asarray(
            remainder(np.linspace(0.1 * box_extent, box_extent, 64)),
            dtype=np.complex128,
        )
    )
    assert np.max(np.abs(remainder_tiny)) <= 100.0 * max(
        1e-3, float(np.max(moderate))
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_type", ["Yukawa", "Helmholtz"])
def test_germ_cancellation_implementation_remainder(dim, kernel_type):
    """Same germ-cancellation certificate as above, but through the
    implementation's own remainder path (``windowed_remainder_profile`` is
    exactly what the assembler integrates), so a sign/prefactor/coefficient
    bug on the implementation side cannot hide behind the test helper's
    independent re-derivation."""
    source_box_level = 3 if dim == 2 else 2
    box_extent = _box_extent(source_box_level)
    window_scale = (box_extent / WINDOW_THETA) ** 2
    parameter = WINDOW_THETA / box_extent
    if kernel_type == "Yukawa":
        zeta = complex(parameter) ** 2
    else:
        zeta = -(complex(parameter) ** 2)

    kernel = _kernel_radial(dim, kernel_type, parameter)
    remainder = windowed_remainder_profile(
        dim, zeta, kernel, window_scale, 6
    )

    r_tiny = np.array([1e-7 * box_extent, 1e-5 * box_extent])
    kernel_tiny = np.asarray(kernel(r_tiny), dtype=np.complex128)
    remainder_tiny = np.asarray(remainder(r_tiny), dtype=np.complex128)

    assert np.all(np.isfinite(remainder_tiny))
    kernel_growth = abs(kernel_tiny[0] - kernel_tiny[1])
    assert kernel_growth > 0.1
    remainder_growth = abs(remainder_tiny[0] - remainder_tiny[1])
    assert remainder_growth <= 1e-6 * max(1.0, kernel_growth)

    moderate = np.abs(
        np.asarray(
            remainder(np.linspace(0.1 * box_extent, box_extent, 64)),
            dtype=np.complex128,
        )
    )
    assert np.max(np.abs(remainder_tiny)) <= 100.0 * max(
        1e-3, float(np.max(moderate))
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_type", ["Yukawa", "Helmholtz"])
def test_windowed_remainder_defines_origin_limit(dim, kernel_type):
    import mpmath as mp

    source_box_level = 3 if dim == 2 else 2
    box_extent = _box_extent(source_box_level)
    window_scale = (box_extent / WINDOW_THETA) ** 2
    parameter = 4.0
    zeta = parameter**2 if kernel_type == "Yukawa" else -(parameter**2)
    remainder = windowed_remainder_profile(
        dim,
        zeta,
        _kernel_radial(dim, kernel_type, parameter),
        window_scale,
        6,
    )

    old_dps = mp.mp.dps
    mp.mp.dps = 100
    try:
        r = mp.sqrt(window_scale) * mp.mpf("1e-40")
        x = r * r / (4 * window_scale)
        if kernel_type == "Yukawa":
            kernel = (
                mp.besselk(0, parameter * r) / (2 * mp.pi)
                if dim == 2
                else mp.exp(-parameter * r) / (4 * mp.pi * r)
            )
        else:
            kernel = (
                0.25j * mp.hankel1(0, parameter * r)
                if dim == 2
                else mp.exp(1j * parameter * r) / (4 * mp.pi * r)
            )

        if dim == 2:
            value = mp.e1(x)
            profiles = [mp.mpf("0.5") * value]
            for m in range(1, 6):
                value = (mp.exp(-x) - x * value) / m
                profiles.append(mp.mpf("0.5") * value)
            prefactor = 1 / (2 * mp.pi)
        else:
            value = mp.sqrt(mp.pi) * mp.erfc(mp.sqrt(x)) / mp.sqrt(x)
            scale = 1 / (2 * mp.sqrt(mp.pi * window_scale))
            profiles = [scale * value]
            for m in range(1, 6):
                value = (mp.exp(-x) - x * value) / (m - mp.mpf("0.5"))
                profiles.append(scale * value)
            prefactor = 1 / (4 * mp.pi)

        coefficient = mp.mpc(1)
        channel_sum = mp.mpc(0)
        for m, profile in enumerate(profiles):
            channel_sum += coefficient * profile
            coefficient *= -mp.mpc(zeta * window_scale) / (m + 1)
        expected = complex(kernel - prefactor * channel_sum)
    finally:
        mp.mp.dps = old_dps

    at_origin = remainder(0.0)
    values = remainder(np.array([0.0, 1.0e-7 * box_extent]))
    assert np.isfinite(at_origin)
    assert values[0] == pytest.approx(at_origin)
    assert at_origin == pytest.approx(expected, rel=2.0e-12, abs=2.0e-12)

# }}}


# {{{ T2b: table geometry pinned independently of the implementation

@pytest.mark.parametrize(("dim", "q_order"), [(2, 3), (3, 2)])
def test_tensor_gauss_points_explicit_ordering(dim, q_order):
    """Pin the queue-free q-point construction against an explicit
    nested-loop re-derivation of the documented ordering (axis 0 slowest,
    last axis fastest — independent of np.meshgrid semantics)."""
    extent = _box_extent(3 if dim == 2 else 2)
    points = _tensor_product_gauss_points(q_order, dim, extent)

    nodes = np.polynomial.legendre.leggauss(q_order)[0]
    axis = 0.5 * extent * (nodes + 1.0)
    expected = []
    if dim == 2:
        for i in range(q_order):
            for j in range(q_order):
                expected.append((axis[i], axis[j]))
    else:
        for i in range(q_order):
            for j in range(q_order):
                for k in range(q_order):
                    expected.append((axis[i], axis[j], axis[k]))
    expected = np.asarray(expected, dtype=np.float64)
    assert points.shape == expected.shape
    assert np.max(np.abs(points - expected)) <= 1e-14 * extent


@pytest.mark.parametrize(("dim", "q_order"), [(2, 3), (3, 2)])
def test_tensor_gauss_points_match_meshgen(dim, q_order):
    """Pin the queue-free q-point construction against the real
    mesh-generator-backed table constructor when one is available (the
    geometry T3/T4 exercise through the table manager)."""
    extent = _box_extent(3 if dim == 2 else 2)
    points = _tensor_product_gauss_points(q_order, dim, extent)

    from volumential.nearfield_potential_table import (
        NearFieldInteractionTable,
    )

    # Only environment unavailability may skip: a missing meshgen module or
    # no OpenCL platform to run it on.  A q-point, API or geometry failure in
    # the constructor is precisely what this test exists to catch and must
    # not be swallowed into a green skip.
    try:
        import pyopencl as cl
    except ImportError:
        unavailable: tuple[type[BaseException], ...] = (ImportError,)
    else:
        unavailable = (ImportError, cl.Error)

    try:
        real_table = NearFieldInteractionTable(
            quad_order=q_order,
            dim=dim,
            kernel_func=None,
            kernel_type=None,
            sumpy_kernel=None,
            source_box_extent=extent,
            dtype=np.float64,
            progress_bar=False,
        )
    except unavailable as exc:
        pytest.skip(f"no mesh-generator table construction available: {exc}")
    assert (
        np.max(np.abs(np.asarray(real_table.q_points) - points))
        <= 1e-13 * extent
    )

# }}}


# {{{ T3: assembled-vs-direct parity at the classical configurations

@pytest.fixture(scope="session")
def channel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("windowed-channels") / "channels.sqlite"


def _direct_reference_table(
    queue, cache_path, dim, kernel_type, q_order, parameter, level,
    build_config,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    manager_kwargs = {}
    get_kwargs = {}
    if kernel_type == "Helmholtz":
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(dim)
        manager_kwargs["dtype"] = np.complex128
        get_kwargs["sumpy_knl"] = knl
        get_kwargs[knl.helmholtz_k_name] = float(parameter)
        kernel_request = "Helmholtz-Reference"
    else:
        get_kwargs["lam"] = float(parameter)
        kernel_request = "Yukawa"

    with NearFieldInteractionTableManager(
        str(cache_path),
        root_extent=ROOT_EXTENT,
        queue=queue,
        **manager_kwargs,
    ) as table_manager:
        table, _ = table_manager.get_table(
            dim,
            kernel_request,
            q_order,
            source_box_level=level,
            force_recompute=True,
            queue=queue,
            build_config=build_config,
            **get_kwargs,
        )
    return table


@pytest.mark.parametrize(
    ("dim", "kernel_type", "q_order", "parameter", "level"),
    [
        (2, "Yukawa", 3, 4.0, 3),
        (2, "Helmholtz", 3, 4.0, 3),
        pytest.param(
            3, "Yukawa", 2, 2.0, 2, marks=pytest.mark.full_accuracy
        ),
        pytest.param(
            3, "Helmholtz", 2, 2.0, 2, marks=pytest.mark.full_accuracy
        ),
    ],
)
def test_windowed_matches_direct_batched(
    tmp_path, channel_cache, dim, kernel_type, q_order, parameter, level
):
    chan_kwargs = CHAN_ORDERS_3D if dim == 3 else CHAN_ORDERS_2D
    assembled, certificate = assemble_windowed_parameterized_table(
        channel_cache,
        dim,
        kernel_type,
        q_order,
        parameter,
        source_box_level=level,
        window_theta=WINDOW_THETA,
        **chan_kwargs,
    )

    if dim == 2:
        # full manager-path reference: real mesh-generator geometry, batched
        # OpenCL Duffy engine, full entry IDs
        queue = _get_queue_or_skip()
        from volumential.nearfield_potential_table import DuffyBuildConfig

        direct_table = _direct_reference_table(
            queue,
            tmp_path / "direct.sqlite",
            dim,
            kernel_type,
            q_order,
            parameter,
            level,
            DuffyBuildConfig(
                radial_rule="tanh-sinh-fast",
                regular_quad_order=16,
                radial_quad_order=45,
            ),
        )
        ids_direct, direct_values = direct_table.get_reduced_table_data()
        ids_direct = np.asarray(ids_direct)
        direct_values = np.asarray(direct_values)
    else:
        # The 3D batched Duffy builder fails on constrained OpenCL runtimes
        # (observed clEnqueueReadBuffer OUT_OF_RESOURCES under pocl) and its
        # full-table scalar fallback is prohibitively slow, so reference a
        # deterministic spot subset of the reduced entries with the
        # pre-existing scalar Duffy engine instead (queue-free; the q-point
        # geometry parity with the real mesh generator is pinned by
        # test_tensor_gauss_points_match_meshgen, and the scalar engine's
        # per-entry cost caps the affordable subset size).
        entry_ids = np.asarray(assembled.get_reduced_entry_ids())
        _, values_all = assembled.get_reduced_table_data()
        positions = _spot_entry_ids(entry_ids, values_all, 6, 6)
        ids_direct = entry_ids[positions]
        direct_values = _scalar_direct_entries(
            dim,
            q_order,
            level,
            ids_direct,
            _kernel_radial(dim, kernel_type, parameter),
            deg_theta=8,
            radial_quad_order=25,
            complex_valued=(kernel_type == "Helmholtz"),
        )
        if kernel_type == "Yukawa":
            direct_values = direct_values.real
    assembled_values = np.asarray(
        assembled.get_entry_data_for_full_indices(ids_direct)
    )

    scale = max(float(np.max(np.abs(direct_values))), 1e-300)
    deviation = float(
        np.max(np.abs(assembled_values - direct_values)) / scale
    )
    assert deviation < 5.0e-6, (deviation, certificate)
    assert certificate["condition_number"] < 50.0
    assert certificate["truncation_tail_bound"] == 0.0
    if kernel_type == "Yukawa":
        assert assembled.dtype == np.float64
        assert certificate["assembled_max_abs_imag"] <= 1.0e-10 * scale
    else:
        assert assembled.dtype == np.complex128
    # like the classical assembler, the result must not inherit a
    # scale-reuse kernel identity
    assert assembled.kernel_type is None

# }}}


# {{{ T4: high-theta parity at the design edge, with classical contrast

@pytest.fixture(scope="module")
def high_theta_direct_refs(tmp_path_factory):
    """Direct references at theta = 16 under two quadrature policies; their
    agreement estimates the direct references' own accuracy floor."""
    queue = _get_queue_or_skip()
    from volumential.nearfield_potential_table import DuffyBuildConfig

    cache_dir = tmp_path_factory.mktemp("high-theta-direct")
    policies = {
        "mid": DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=24,
            radial_quad_order=61,
        ),
        "tight": DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=48,
            radial_quad_order=160,
        ),
    }
    refs = {}
    for kernel_type in ("Yukawa", "Helmholtz"):
        per_policy = {}
        for name, build_config in policies.items():
            table = _direct_reference_table(
                queue,
                cache_dir / f"{kernel_type}-{name}.sqlite",
                2,
                kernel_type,
                3,
                64.0,
                3,
                build_config,
            )
            ids, values = table.get_reduced_table_data()
            per_policy[name] = (np.asarray(ids), np.asarray(values))
        refs[kernel_type] = per_policy
    return refs


@pytest.mark.parametrize("kernel_type", ["Yukawa", "Helmholtz"])
@pytest.mark.full_accuracy
def test_high_theta_parity(
    tmp_path, channel_cache, high_theta_direct_refs, kernel_type
):
    ids_tight, tight_values = high_theta_direct_refs[kernel_type]["tight"]
    ids_mid, mid_values = high_theta_direct_refs[kernel_type]["mid"]
    assert np.array_equal(ids_tight, ids_mid)

    scale = max(float(np.max(np.abs(tight_values))), 1e-300)
    direct_floor = float(
        np.max(np.abs(mid_values - tight_values)) / scale
    )

    assembled, certificate = assemble_windowed_parameterized_table(
        channel_cache,
        2,
        kernel_type,
        3,
        64.0,
        source_box_level=3,
        window_theta=WINDOW_THETA,
        **CHAN_ORDERS_2D,
    )
    assembled_values = np.asarray(
        assembled.get_entry_data_for_full_indices(ids_tight)
    )
    deviation = float(
        np.max(np.abs(assembled_values - tight_values)) / scale
    )

    assert deviation < 1.0e-4, (deviation, direct_floor, certificate)
    # the assembled table's own accuracy floor is the one-time channel
    # quadrature (angular order 48 holds the 2D q = 3 channel family to
    # about 2e-7 relative — see _resolve_channel_orders), so the deviation
    # must be explained by that floor or by the direct references' floor
    assert deviation <= max(10.0 * direct_floor, 5.0e-7), (
        deviation,
        direct_floor,
    )
    assert certificate["condition_number"] < 50.0
    assert certificate["theta"] == pytest.approx(16.0)

    # the classical series assembly cannot serve this point: it refuses in
    # choose_truncation_order before any table build (the pinned message
    # distinguishes the principled refusal from an unrelated crash)
    queue = _get_queue_or_skip()
    with pytest.raises(ValueError, match="cannot certify"):
        assemble_parameterized_table(
            queue,
            tmp_path / "classical-contrast.sqlite",
            2,
            kernel_type,
            3,
            64.0,
            source_box_level=3,
            tolerance=1.0e-11,
        )


@pytest.mark.parametrize("kernel_type", ["Yukawa", "Helmholtz"])
def test_3d_design_edge_table_parity_and_refinement(tmp_path, kernel_type):
    dim, q_order, level, parameter = 3, 1, 3, 64.0  # theta = 16
    cache = tmp_path / "3d-design-edge.sqlite"
    tables = {}
    certificates = {}
    for name, smooth_order in (("coarse", 10), ("fine", 24)):
        tables[name], certificates[name] = assemble_windowed_parameterized_table(
            cache,
            dim,
            kernel_type,
            q_order,
            parameter,
            source_box_level=level,
            window_theta=WINDOW_THETA,
            p_star=4,
            smooth_quad_order=smooth_order,
            chan_regular_order=8,
            chan_radial_order=25,
        )

    fine = tables["fine"]
    extent = float(fine.source_box_extent)
    candidates = []
    for entry_id in np.asarray(fine.get_reduced_entry_ids()):
        case_index = int(entry_id // fine.n_pairs)
        pair_id = int(entry_id % fine.n_pairs)
        target_index = pair_id % fine.n_q_points
        target = np.asarray(
            fine.find_target_point(target_index, case_index)
        ).real
        outside = np.maximum(np.maximum(-target, target - extent), 0.0)
        distance = float(np.linalg.norm(outside))
        # Four window widths from the box still exercise a visible channel
        # correction while avoiding a singular direct-reference integral.
        if 0.2 * extent <= distance <= 0.5 * extent:
            candidates.append((distance, int(entry_id)))
    assert len(candidates) >= 3
    spot_ids = np.asarray(
        [entry_id for _, entry_id in sorted(candidates)[:3]], dtype=np.int64
    )

    kernel_radial = _kernel_radial(dim, kernel_type, parameter)
    reference_mid = _tensor_gauss_direct_q1_entries(
        fine, spot_ids, kernel_radial, 28
    )
    reference = _tensor_gauss_direct_q1_entries(
        fine, spot_ids, kernel_radial, 40
    )
    if kernel_type == "Yukawa":
        reference_mid = reference_mid.real
        reference = reference.real
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    direct_floor = float(np.max(np.abs(reference_mid - reference)) / scale)
    deviations = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(table.get_entry_data_for_full_indices(spot_ids))
                    - reference
                )
            )
            / scale
        )
        for name, table in tables.items()
    }

    assert direct_floor < 1.0e-9, direct_floor
    assert deviations["fine"] < 2.0e-6, deviations
    assert deviations["fine"] <= 0.2 * deviations["coarse"], deviations
    assert certificates["fine"]["theta"] == pytest.approx(WINDOW_THETA)
    assert certificates["fine"]["condition_number"] < 10.0


@pytest.mark.full_accuracy
def test_3d_design_edge_singular_nonconstant_mode(tmp_path):
    from volumential.rke_table_assembly import get_windowed_channel_table

    dim, q_order, level, parameter = 3, 2, 3, 64.0  # theta = 16
    cache = tmp_path / "3d-design-edge-singular.sqlite"
    table, certificate = assemble_windowed_parameterized_table(
        cache,
        dim,
        "Yukawa",
        q_order,
        parameter,
        source_box_level=level,
        window_theta=WINDOW_THETA,
        p_star=2,
        smooth_quad_order=24,
        chan_regular_order=6,
        chan_radial_order=15,
    )

    extent = float(table.source_box_extent)
    self_cases = []
    for case_index in range(table.n_cases):
        target = np.asarray(table.find_target_point(0, case_index)).real
        if np.all((target > 0.0) & (target < extent)):
            self_cases.append(case_index)
    assert len(self_cases) == 1

    source_mode = 1
    target_index = 0
    mode_axes = np.asarray(table._get_all_mode_axes())[source_mode]
    assert tuple(mode_axes) != (0, 0, 0)
    entry_id = (
        self_cases[0] * table.n_pairs
        + source_mode * table.n_q_points
        + target_index
    )
    entry_ids = np.asarray([entry_id], dtype=np.int64)
    assembled = np.asarray(
        table.reconstruct_full_table_from_symmetry()[entry_ids]
    )

    # The normalized m=1 channel must itself carry a finite, nonzero value at
    # this singular self entry, rather than passing through only the m=0 germ.
    channel_m1 = get_windowed_channel_table(
        cache,
        dim,
        q_order,
        1,
        source_box_level=level,
        window_theta=WINDOW_THETA,
        chan_regular_order=6,
        chan_radial_order=15,
    )
    channel_value = np.asarray(
        channel_m1.reconstruct_full_table_from_symmetry()[entry_ids]
    )
    assert channel_m1._windowed_cache_disposition == "hit"
    assert np.all(np.isfinite(channel_value))
    assert np.max(np.abs(channel_value)) > 0.0

    reference = _scalar_direct_entries(
        dim,
        q_order,
        level,
        entry_ids,
        _kernel_radial(dim, "Yukawa", parameter),
        deg_theta=6,
        radial_quad_order=15,
    ).real
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    deviation = float(np.max(np.abs(assembled - reference)) / scale)
    assert deviation < 5.0e-3, (deviation, assembled, reference)
    assert certificate["theta"] == pytest.approx(WINDOW_THETA)
    assert certificate["condition_number"] < 10.0


# }}}


# {{{ T5: smooth-remainder order convergence

def test_smooth_order_convergence(channel_cache):
    dim, q_order, level, parameter = 2, 3, 3, 32.0  # theta = 8
    probe_smooth_orders = (8, 16, 24, 32)

    tables = {}
    for smooth_order in probe_smooth_orders:
        table, certificate = assemble_windowed_parameterized_table(
            channel_cache,
            dim,
            "Yukawa",
            q_order,
            parameter,
            source_box_level=level,
            window_theta=WINDOW_THETA,
            p_star=6,
            smooth_quad_order=smooth_order,
            **CHAN_ORDERS_2D,
        )
        assert certificate["smooth_quad_order"] == smooth_order
        tables[smooth_order] = table

    entry_ids = np.asarray(tables[32].get_reduced_entry_ids())
    _, finest_values = tables[32].get_reduced_table_data()
    positions = _spot_entry_ids(entry_ids, finest_values, 6, 6)
    spot_ids = entry_ids[positions]

    reference = _scalar_direct_entries(
        dim,
        q_order,
        level,
        spot_ids,
        _kernel_radial(dim, "Yukawa", parameter),
        deg_theta=48,
        radial_quad_order=160,
    ).real
    scale = max(float(np.max(np.abs(reference))), 1e-300)

    deviations = []
    for smooth_order in probe_smooth_orders:
        values = np.asarray(
            tables[smooth_order].get_entry_data_for_full_indices(spot_ids)
        )
        deviations.append(
            float(np.max(np.abs(values - reference)) / scale)
        )

    # non-increasing within noise, and converged (plateaued) by order 32
    for coarse, fine in pairwise(deviations):
        assert fine <= 1.25 * coarse + 1e-9, deviations
    assert deviations[-1] < 1e-6, deviations

# }}}


# {{{ T6: complex squared frequency (damped Helmholtz)

def test_complex_zeta_damped_helmholtz(channel_cache):
    import scipy.special as sps

    dim, q_order, level = 2, 3, 3
    k_complex = 8.0 * (1.0 + 0.05j)
    zeta = -(k_complex * k_complex)

    def kernel_radial(r):
        return 0.25j * sps.hankel1(
            0, k_complex * np.asarray(r, dtype=np.complex128)
        )

    table, certificate = _assemble_windowed_for_zeta(
        channel_cache,
        dim,
        q_order,
        zeta,
        kernel_radial,
        source_box_level=level,
        window_theta=WINDOW_THETA,
        **CHAN_ORDERS_2D,
    )
    assert certificate["condition_number"] < 50.0
    assert complex(certificate["zeta_real"], certificate["zeta_imag"]) == (
        pytest.approx(zeta)
    )

    entry_ids = np.asarray(table.get_reduced_entry_ids())
    _, values_all = table.get_reduced_table_data()
    positions = _spot_entry_ids(entry_ids, values_all, 3, 3)
    spot_ids = entry_ids[positions]
    assembled_values = np.asarray(
        table.get_entry_data_for_full_indices(spot_ids)
    )

    reference = _scalar_direct_entries(
        dim,
        q_order,
        level,
        spot_ids,
        kernel_radial,
        deg_theta=48,
        radial_quad_order=160,
        complex_valued=True,
    )
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    deviation = float(
        np.max(np.abs(assembled_values - reference)) / scale
    )
    assert deviation < 1e-5, deviation

# }}}


# {{{ T7: p_star knob at the design edge

def test_p_star_knob_at_design_theta(channel_cache):
    dim, q_order, level, parameter = 2, 3, 3, 64.0  # theta = 16

    values_by_p_star = {}
    for p_star in (4, 6):
        table, certificate = assemble_windowed_parameterized_table(
            channel_cache,
            dim,
            "Yukawa",
            q_order,
            parameter,
            source_box_level=level,
            window_theta=WINDOW_THETA,
            p_star=p_star,
            **CHAN_ORDERS_2D,
        )
        assert certificate["condition_number"] < 50.0
        assert certificate["p_star"] == p_star
        # pin the window-units contract directly: t_w = (b/Theta)^2, and at
        # theta = Theta the coefficient bound (theta/Theta)^{2m}/m! peaks
        # at exactly 1 — a (1/Theta)^2-style units bug would break both
        assert certificate["window_scale_t_w"] == pytest.approx(
            (_box_extent(level) / WINDOW_THETA) ** 2, rel=1e-14
        )
        assert certificate["coefficient_bound"] == pytest.approx(
            1.0, rel=1e-12
        )
        values_by_p_star[p_star] = table

    entry_ids = np.asarray(values_by_p_star[6].get_reduced_entry_ids())
    _, finest_values = values_by_p_star[6].get_reduced_table_data()
    positions = _spot_entry_ids(entry_ids, finest_values, 4, 4)
    spot_ids = entry_ids[positions]

    reference = _scalar_direct_entries(
        dim,
        q_order,
        level,
        spot_ids,
        _kernel_radial(dim, "Yukawa", parameter),
        deg_theta=48,
        radial_quad_order=160,
    ).real
    scale = max(float(np.max(np.abs(reference))), 1e-300)

    deviations = {}
    for p_star, table in values_by_p_star.items():
        values = np.asarray(
            table.get_entry_data_for_full_indices(spot_ids)
        )
        deviations[p_star] = float(
            np.max(np.abs(values - reference)) / scale
        )

    # more retained channels must not lose accuracy (equality within the
    # shared quadrature floor is allowed) ...
    assert deviations[6] <= 1.5 * deviations[4] + 1e-8, deviations
    # ... and the flagship p_star must be absolutely accurate against the
    # independent scalar reference at the design edge — this queue-free
    # absolute bound catches recombination bugs (e.g. a dropped channel)
    # that leave the p_star ratio above intact
    assert deviations[6] < 1e-5, deviations


def test_high_theta_helmholtz_scalar_parity(channel_cache):
    """Queue-free absolute parity for Helmholtz at the design edge
    theta = Theta (the Yukawa counterpart is the absolute bound in
    test_p_star_knob_at_design_theta): catches wrapper-side zeta-sign or
    normalization bugs that are invisible at small theta."""
    dim, q_order, level, parameter = 2, 3, 3, 64.0  # theta = 16

    table, certificate = assemble_windowed_parameterized_table(
        channel_cache,
        dim,
        "Helmholtz",
        q_order,
        parameter,
        source_box_level=level,
        window_theta=WINDOW_THETA,
        **CHAN_ORDERS_2D,
    )
    assert certificate["condition_number"] < 50.0

    entry_ids = np.asarray(table.get_reduced_entry_ids())
    _, values_all = table.get_reduced_table_data()
    positions = _spot_entry_ids(entry_ids, values_all, 4, 4)
    spot_ids = entry_ids[positions]
    assembled_values = np.asarray(
        table.get_entry_data_for_full_indices(spot_ids)
    )

    reference = _scalar_direct_entries(
        dim,
        q_order,
        level,
        spot_ids,
        _kernel_radial(dim, "Helmholtz", parameter),
        deg_theta=48,
        radial_quad_order=160,
        complex_valued=True,
    )
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    deviation = float(
        np.max(np.abs(assembled_values - reference)) / scale
    )
    assert deviation < 1e-5, deviation

# }}}


# {{{ T8: default channel orders serve the flagship configuration

def test_default_channel_orders(channel_cache):
    from volumential.rke_table_assembly import _resolve_channel_orders

    # 2D resolves to the reference-grade angular order the parity tests use;
    # 3D keeps the mild-geometry default; explicit values pass through
    assert _resolve_channel_orders(2, None, None) == (48, 61)
    assert _resolve_channel_orders(3, None, None) == (20, 61)
    assert _resolve_channel_orders(2, 20, 45) == (20, 45)

    # the all-defaults public path resolves to the same channel family the
    # parity tests certify (same cache key), hence identical entries
    common = {
        "source_box_level": 3,
        "window_theta": WINDOW_THETA,
    }
    default_table, default_certificate = (
        assemble_windowed_parameterized_table(
            channel_cache, 2, "Yukawa", 3, 4.0, **common
        )
    )
    explicit_table, _ = assemble_windowed_parameterized_table(
        channel_cache, 2, "Yukawa", 3, 4.0, **common, **CHAN_ORDERS_2D
    )
    assert default_certificate["channel_quadrature_orders"] == {
        "regular": 48,
        "radial": 61,
    }
    ids_default, values_default = default_table.get_reduced_table_data()
    ids_explicit, values_explicit = explicit_table.get_reduced_table_data()
    assert np.array_equal(np.asarray(ids_default), np.asarray(ids_explicit))
    assert np.array_equal(
        np.asarray(values_default), np.asarray(values_explicit)
    )

# }}}


# {{{ T9: channel cache self-heals after torn or corrupted writes

@pytest.mark.parametrize("dim", [2, 3])
def test_high_order_normalized_channel_cache(tmp_path, dim):
    from pathlib import Path

    import volumential.rke_table_assembly as rta

    cache = tmp_path / "normalized.sqlite"
    table = rta.get_windowed_channel_table(
        cache,
        dim,
        1,
        59,
        root_extent=1.0e-3,
        window_theta=WINDOW_THETA,
        chan_regular_order=2,
        chan_radial_order=7,
    )
    entry_ids, values = table.get_reduced_table_data()
    entry_ids = np.asarray(entry_ids)
    values = np.asarray(values)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) > 0.0
    assert table._windowed_cache_disposition == "rebuilt"

    cache_dir = Path(str(cache) + ".windowed")
    (cache_file,) = sorted(cache_dir.glob("*.npz"))
    with np.load(cache_file, allow_pickle=False) as payload:
        assert payload["key_cache_schema"].item() == 2
        assert payload["key_normalization"].item() == "psi=chi/t_w**m"
        assert payload["payload_checksum"].item() == (
            rta._windowed_channel_payload_checksum(entry_ids, values)
        )

    reloaded = rta.get_windowed_channel_table(
        cache,
        dim,
        1,
        59,
        root_extent=1.0e-3,
        window_theta=WINDOW_THETA,
        chan_regular_order=2,
        chan_radial_order=7,
    )
    assert np.array_equal(
        values, np.asarray(reloaded.get_reduced_table_data()[1])
    )
    assert reloaded._windowed_cache_disposition == "hit"


def test_channel_cache_disposition(tmp_path):
    from pathlib import Path

    from volumential.rke_table_assembly import get_windowed_channel_table

    cache = tmp_path / "disposition.sqlite"
    kwargs = {
        "chan_regular_order": 2,
        "chan_radial_order": 7,
    }
    rebuilt = get_windowed_channel_table(cache, 2, 1, 0, **kwargs)
    values = np.asarray(rebuilt.get_reduced_table_data()[1])
    assert rebuilt._windowed_cache_disposition == "rebuilt"

    hit = get_windowed_channel_table(cache, 2, 1, 0, **kwargs)
    assert hit._windowed_cache_disposition == "hit"
    assert np.array_equal(values, np.asarray(hit.get_reduced_table_data()[1]))

    cache_dir = Path(str(cache) + ".windowed")
    (cache_file,) = sorted(cache_dir.glob("*.npz"))
    with np.load(cache_file, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["payload_checksum"] = np.asarray("0" * 64)
    np.savez(cache_file, **arrays)

    recovered = get_windowed_channel_table(cache, 2, 1, 0, **kwargs)
    assert recovered._windowed_cache_disposition == "rebuilt"
    assert np.array_equal(
        values, np.asarray(recovered.get_reduced_table_data()[1])
    )
    reloaded = get_windowed_channel_table(cache, 2, 1, 0, **kwargs)
    assert reloaded._windowed_cache_disposition == "hit"


def test_channel_cache_self_heals(tmp_path):
    from pathlib import Path

    from volumential.rke_table_assembly import get_windowed_channel_table

    cache = tmp_path / "heal.sqlite"
    kwargs = {
        "source_box_level": 3,
        "root_extent": ROOT_EXTENT,
        "window_theta": WINDOW_THETA,
        "chan_regular_order": 6,
        "chan_radial_order": 15,
    }
    table = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    ids, values = table.get_reduced_table_data()

    cache_dir = Path(str(cache) + ".windowed")
    cache_files = sorted(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    cache_file = cache_files[0]
    assert not list(cache_dir.glob("*.tmp-*"))

    # a torn write (interrupted np.savez / concurrent process) must rebuild,
    # not raise zipfile.BadZipFile forever
    payload = cache_file.read_bytes()
    cache_file.write_bytes(payload[: len(payload) // 2])
    healed = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    ids_healed, values_healed = healed.get_reduced_table_data()
    assert np.array_equal(np.asarray(ids), np.asarray(ids_healed))
    assert np.array_equal(np.asarray(values), np.asarray(values_healed))

    # the heal must have rewritten the same deterministic cache key, or the
    # next corruption step would poison an orphan file and assert nothing
    refreshed_files = sorted(cache_dir.glob("*.npz"))
    assert refreshed_files == cache_files
    cache_files = refreshed_files
    cache_file = cache_files[0]

    # arbitrary garbage (stale format) likewise self-heals
    cache_file.write_bytes(b"not a zip file")
    healed = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        np.asarray(values), np.asarray(healed.get_reduced_table_data()[1])
    )
    refreshed_files = sorted(cache_dir.glob("*.npz"))
    assert refreshed_files == cache_files
    cache_files = refreshed_files
    cache_file = cache_files[0]

    # a loadable file whose key and entry IDs check out but whose value array
    # has the wrong shape must also rebuild: accepting it would raise out of
    # ``set_reduced_table_data`` and wedge every later call on the bad file
    with np.load(cache_file, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["values"] = np.asarray(arrays["values"])[:-1]
    np.savez(cache_file, **arrays)
    healed = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        np.asarray(values), np.asarray(healed.get_reduced_table_data()[1])
    )
    refreshed_files = sorted(cache_dir.glob("*.npz"))
    assert refreshed_files == cache_files
    cache_files = refreshed_files
    cache_file = cache_files[0]

    # A physical-channel or otherwise stale normalization identity must never
    # alias the normalized psi_m cache, even if placed at the active path.
    with np.load(cache_file, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["key_normalization"] = np.asarray("physical-chi")
    np.savez(cache_file, **arrays)
    healed = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        np.asarray(values), np.asarray(healed.get_reduced_table_data()[1])
    )
    refreshed_files = sorted(cache_dir.glob("*.npz"))
    assert refreshed_files == cache_files
    cache_files = refreshed_files
    cache_file = cache_files[0]

    # Finite, in-bound corruption is invisible to the analytic magnitude
    # check, so the payload checksum must force a rebuild.
    with np.load(cache_file, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    corrupted = np.asarray(arrays["values"]).copy()
    corrupted[0] = np.nextafter(corrupted[0], np.inf)
    arrays["values"] = corrupted
    np.savez(cache_file, **arrays)
    healed = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        np.asarray(values), np.asarray(healed.get_reduced_table_data()[1])
    )
    refreshed_files = sorted(cache_dir.glob("*.npz"))
    assert refreshed_files == cache_files
    cache_files = refreshed_files
    cache_file = cache_files[0]

    # and the rebuilt file is a valid cache again (no rebuild artifacts)
    reloaded = get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        np.asarray(values), np.asarray(reloaded.get_reduced_table_data()[1])
    )
    assert not list(cache_dir.glob("*.tmp-*"))

# }}}


# {{{ T10: declaration and guard paths

def test_bool_rejection_and_max_terms_guard():
    import volumential.rke_table_assembly as rta

    for bad_bool in (True, np.bool_(False)):
        with pytest.raises(ValueError, match="value must be an integer"):
            rta._require_integer("value", bad_bool)

    with pytest.raises(ValueError, match="q_order must be an integer"):
        _tensor_product_gauss_points(True, 2, 1.0)
    with pytest.raises(ValueError, match="max_terms must be an integer"):
        rta.choose_truncation_order(3, 0.0, 1.0, 1.0e-12, max_terms=True)
    with pytest.raises(ValueError, match="max_terms must be >= 1"):
        rta.choose_truncation_order(3, 0.0, 1.0, 1.0e-12, max_terms=0)

    n_terms, bound = rta.choose_truncation_order(
        3, 0.0, 1.0, 1.0e-12, max_terms=np.int64(1)
    )
    assert n_terms == 1
    assert bound == 0.0


def test_windowed_integer_parameters_require_index(tmp_path):
    import volumential.rke_table_assembly as rta

    cache = tmp_path / "integer-guards.sqlite"

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    with pytest.raises(ValueError, match="dim must be an integer"):
        rta.choose_truncation_order(np.float64(2.0), 1j, 1.0, 1.0e-6)
    with pytest.raises(ValueError, match="dim must be an integer"):
        assemble_parameterized_table(None, cache, 2.0, "Yukawa", 1, 1.0)
    with pytest.raises(ValueError, match="dim must be an integer"):
        windowed_channel_profile(2.0, 0, 1.0)
    with pytest.raises(ValueError, match="dim must be an integer"):
        _tensor_product_gauss_points(1, 2.0, 1.0)
    with pytest.raises(ValueError, match="dim must be an integer"):
        _windowed_channel_skeleton(2.0, 1, 0, ROOT_EXTENT, WINDOW_THETA, 0)
    with pytest.raises(ValueError, match="dim must be an integer"):
        rta._windowed_channel_cache_file(
            cache, 2.0, 1, 0, ROOT_EXTENT, WINDOW_THETA, 0, 2, 7
        )
    with pytest.raises(ValueError, match="dim must be an integer"):
        rta.get_windowed_channel_table(cache, 2.0, 1, 0)
    with pytest.raises(ValueError, match="dim must be an integer"):
        _assemble_windowed_for_zeta(cache, 2.0, 1, 1.0, zero_kernel)
    with pytest.raises(ValueError, match="dim must be an integer"):
        assemble_windowed_parameterized_table(
            cache, 2.0, "Yukawa", 1, 1.0
        )
    with pytest.raises(ValueError, match="dim must be an integer"):
        rta._resolve_channel_orders(2.0, 2, 7)
    with pytest.raises(NotImplementedError, match="only 2D and 3D"):
        _tensor_product_gauss_points(1, 4, 1.0)

    with pytest.raises(ValueError, match="q_order must be an integer"):
        assemble_parameterized_table(None, cache, 2, "Yukawa", 1.5, 1.0)
    with pytest.raises(ValueError, match="q_order must be an integer"):
        _tensor_product_gauss_points(1.5, 2, 1.0)
    with pytest.raises(ValueError, match="q_order must be an integer"):
        _windowed_channel_skeleton(2, 1.5, 0, ROOT_EXTENT, WINDOW_THETA, 0)
    with pytest.raises(ValueError, match="q_order must be an integer"):
        rta._windowed_channel_cache_file(
            cache, 2, 1.5, 0, ROOT_EXTENT, WINDOW_THETA, 0, 2, 7
        )
    with pytest.raises(ValueError, match="q_order must be an integer"):
        rta.get_windowed_channel_table(cache, 2, 1.5, 0)
    with pytest.raises(ValueError, match="q_order must be an integer"):
        _assemble_windowed_for_zeta(cache, 2, 1.5, 1.0, zero_kernel)
    with pytest.raises(ValueError, match="q_order must be an integer"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1.5, 1.0
        )
    with pytest.raises(ValueError, match="q_order must be >= 1"):
        _tensor_product_gauss_points(0, 2, 1.0)
    with pytest.raises(ValueError, match="q_order must be >= 1"):
        rta.get_windowed_channel_table(cache, 2, 0, 0)

    with pytest.raises(ValueError, match="source_box_level must be an integer"):
        assemble_parameterized_table(
            None, cache, 2, "Yukawa", 1, 1.0, source_box_level=0.5
        )
    with pytest.raises(ValueError, match="source_box_level must be an integer"):
        _windowed_channel_skeleton(2, 1, 0.5, ROOT_EXTENT, WINDOW_THETA, 0)
    with pytest.raises(ValueError, match="source_box_level must be an integer"):
        rta.get_windowed_channel_table(
            cache, 2, 1, 0, source_box_level=np.float64(0.0)
        )
    with pytest.raises(ValueError, match="source_box_level must be an integer"):
        _assemble_windowed_for_zeta(
            cache, 2, 1, 1.0, zero_kernel, source_box_level=0.5
        )
    with pytest.raises(ValueError, match="source_box_level must be an integer"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1, 1.0, source_box_level=0.5
        )
    with pytest.raises(ValueError, match="source_box_level must be >= 0"):
        _windowed_channel_skeleton(2, 1, -1, ROOT_EXTENT, WINDOW_THETA, 0)

    with pytest.raises(ValueError, match="m must be an integer"):
        windowed_channel_profile(2, 0.5, 1.0)
    with pytest.raises(ValueError, match="m must be an integer"):
        _windowed_channel_skeleton(2, 1, 0, ROOT_EXTENT, WINDOW_THETA, 0.5)
    with pytest.raises(ValueError, match="m must be an integer"):
        rta.get_windowed_channel_table(cache, 2, 1, 0.5)

    with pytest.raises(ValueError, match="p_star must be an integer"):
        windowed_remainder_profile(2, 1.0, zero_kernel, 1.0, 1.5)
    with pytest.raises(ValueError, match="p_star must be an integer"):
        _assemble_windowed_for_zeta(
            cache, 2, 1, 1.0, zero_kernel, p_star=1.5
        )
    with pytest.raises(ValueError, match="p_star must be an integer"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1, 1.0, p_star=1.5
        )

    with pytest.raises(ValueError, match="smooth_quad_order must be an integer"):
        rta._smooth_remainder_entry_values(None, None, None, 8.5)
    with pytest.raises(ValueError, match="smooth_quad_order must be an integer"):
        _assemble_windowed_for_zeta(
            cache, 2, 1, 1.0, zero_kernel, smooth_quad_order=8.5
        )
    with pytest.raises(ValueError, match="smooth_quad_order must be an integer"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1, 1.0, smooth_quad_order=8.5
        )

    with pytest.raises(ValueError, match="chan_regular_order must be an integer"):
        rta._resolve_channel_orders(2, 6.5, 15)
    with pytest.raises(ValueError, match="chan_radial_order must be an integer"):
        rta._resolve_channel_orders(2, 6, 15.5)
    with pytest.raises(ValueError, match="chan_regular_order must be an integer"):
        _duffy_channel_entry_values(None, None, 6.5, 15)
    with pytest.raises(ValueError, match="chan_radial_order must be an integer"):
        _duffy_channel_entry_values(None, None, 6, 15.5)
    with pytest.raises(ValueError, match="chan_regular_order must be an integer"):
        rta.get_windowed_channel_table(
            cache, 2, 1, 0, chan_regular_order=6.5, chan_radial_order=15
        )

    assert rta._require_integer("value", np.int64(3), minimum=0) == 3
    profile = windowed_channel_profile(np.int64(2), np.int64(0), 1.0)
    assert np.isfinite(profile(1.0))
    table = _windowed_channel_skeleton(
        np.int64(2),
        np.int64(1),
        np.int64(0),
        ROOT_EXTENT,
        WINDOW_THETA,
        np.int64(0),
    )
    assert table.source_box_level == 0
    points = _tensor_product_gauss_points(
        np.int64(1), np.int64(2), 1.0
    )
    assert points.shape == (1, 2)
    _, cache_key = rta._windowed_channel_cache_file(
        cache,
        np.int64(2),
        np.int64(1),
        np.int64(0),
        ROOT_EXTENT,
        WINDOW_THETA,
        np.int64(0),
        np.int64(2),
        np.int64(7),
    )
    assert cache_key["dim"] == 2
    assert cache_key["q_order"] == 1
    assert rta._resolve_channel_orders(
        2, np.int64(6), np.int64(15)
    ) == (6, 15)
    assert not (tmp_path / "integer-guards.sqlite.windowed").exists()


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_invalid_window_scales_and_declarations(tmp_path, invalid):
    from pathlib import Path

    import volumential.rke_table_assembly as rta

    cache = tmp_path / "window-guards.sqlite"

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    with pytest.raises(ValueError, match="window_scale must be finite and positive"):
        windowed_channel_profile(2, 0, invalid)
    with pytest.raises(ValueError, match="window_scale must be finite and positive"):
        windowed_remainder_profile(2, 1.0, zero_kernel, invalid, 1)
    with pytest.raises(ValueError, match="window_theta must be finite and positive"):
        _windowed_channel_skeleton(2, 1, 0, ROOT_EXTENT, invalid, 0)
    with pytest.raises(ValueError, match="window_theta must be finite and positive"):
        rta.get_windowed_channel_table(cache, 2, 1, 0, window_theta=invalid)
    with pytest.raises(ValueError, match="window_theta must be finite and positive"):
        _assemble_windowed_for_zeta(
            cache, 2, 1, 1.0, zero_kernel, window_theta=invalid
        )
    with pytest.raises(ValueError, match="window_theta must be finite and positive"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1, 1.0, window_theta=invalid
        )
    assert not Path(str(cache) + ".windowed").exists()


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_invalid_window_condition_thresholds(tmp_path, invalid):
    from pathlib import Path

    cache = tmp_path / "condition-guards.sqlite"

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    condition_message = "max_condition must be finite and positive"
    with pytest.raises(ValueError, match=condition_message):
        _assemble_windowed_for_zeta(
            cache, 2, 1, 1.0, zero_kernel, max_condition=invalid
        )
    with pytest.raises(ValueError, match=condition_message):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 1, 1.0, max_condition=invalid
        )
    with pytest.raises(ValueError, match=condition_message):
        assemble_parameterized_table(
            None, cache, 2, "Yukawa", 1, 1.0, max_condition=invalid
        )
    assert not cache.exists()
    assert not Path(str(cache) + ".windowed").exists()


def test_zero_zeta_is_rejected_before_channel_side_effects(tmp_path):
    from pathlib import Path

    cache = tmp_path / "zero-zeta-guard.sqlite"

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    with pytest.raises(ValueError, match="zeta must be nonzero"):
        windowed_remainder_profile(2, 0.0, zero_kernel, 1.0, 1)
    with pytest.raises(ValueError, match="zeta must be nonzero"):
        _assemble_windowed_for_zeta(cache, 2, 1, 0.0, zero_kernel)
    assert not Path(str(cache) + ".windowed").exists()


@pytest.mark.parametrize("invalid", [
    complex(np.nan, 0.0),
    complex(0.0, np.nan),
    complex(np.inf, 0.0),
    complex(0.0, -np.inf),
])
def test_nonfinite_zeta_is_rejected_before_channel_side_effects(
    tmp_path, invalid
):
    from pathlib import Path

    cache = tmp_path / "nonfinite-zeta-guard.sqlite"

    def zero_kernel(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    with pytest.raises(ValueError, match="zeta must be finite"):
        windowed_remainder_profile(2, invalid, zero_kernel, 1.0, 1)
    with pytest.raises(ValueError, match="zeta must be finite"):
        _assemble_windowed_for_zeta(cache, 2, 1, invalid, zero_kernel)
    assert not Path(str(cache) + ".windowed").exists()


def test_windowed_declaration_guards(tmp_path):
    cache = tmp_path / "guards.sqlite"

    # theta = parameter * b beyond the declared window Theta
    with pytest.raises(ValueError, match="exceeds the declared window"):
        assemble_windowed_parameterized_table(
            cache,
            2,
            "Yukawa",
            3,
            65.0,  # theta = 65 * 0.25 = 16.25 > 16
            source_box_level=3,
            window_theta=WINDOW_THETA,
        )

    # nonpositive parameter
    for bad_parameter in (0.0, -4.0):
        with pytest.raises(ValueError, match="positive parameter"):
            assemble_windowed_parameterized_table(
                cache, 2, "Yukawa", 3, bad_parameter, source_box_level=3
            )

    # p_star must retain at least the m = 0 channel
    with pytest.raises(ValueError, match="p_star"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 3, 4.0, source_box_level=3, p_star=0
        )

    # unsupported kernel family
    with pytest.raises(NotImplementedError):
        assemble_windowed_parameterized_table(
            cache, 2, "Stokeslet", 3, 4.0, source_box_level=3
        )

    # box extents outside the certified O(1) range must refuse (parallel to
    # the classical assembler) instead of silently dropping the singular
    # channel content (extent-scaled Duffy degeneracy tests)
    with pytest.raises(ValueError, match="outside the supported"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 2, 1.0, source_box_level=30
        )
    with pytest.raises(ValueError, match="outside the supported"):
        assemble_windowed_parameterized_table(
            cache, 2, "Yukawa", 2, 1.0, root_extent=1e-6, source_box_level=5
        )
    from volumential.rke_table_assembly import get_windowed_channel_table

    with pytest.raises(ValueError, match="outside the supported"):
        get_windowed_channel_table(cache, 2, 2, 0, source_box_level=30)

    # the complex-zeta entry point enforces the declaration disk
    # |zeta| <= (Theta/b)^2 itself (not only the real-parameter wrapper)
    import scipy.special as sps

    k_out = 100.0  # theta = 25 > Theta = 16

    def kernel_radial(r):
        return 0.25j * sps.hankel1(
            0, k_out * np.asarray(r, dtype=np.complex128)
        )

    with pytest.raises(ValueError, match="coverage disk"):
        _assemble_windowed_for_zeta(
            cache,
            2,
            3,
            complex(-(k_out * k_out)),
            kernel_radial,
            source_box_level=3,
            window_theta=WINDOW_THETA,
        )


def test_windowed_condition_guard(tmp_path):
    # the peak-sum condition estimate is >= 1 by the triangle inequality, so
    # a sub-unit max_condition must always trip the guard after assembly
    cache = tmp_path / "guards.sqlite"
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        assemble_windowed_parameterized_table(
            cache,
            2,
            "Yukawa",
            3,
            4.0,
            source_box_level=3,
            p_star=2,
            smooth_quad_order=8,
            max_condition=0.5,
            chan_regular_order=6,
            chan_radial_order=15,
        )


def test_windowed_reality_guard(tmp_path):
    # requesting a real (Yukawa-typed) result for a genuinely complex kernel
    # must fail loudly, proving the reality check precedes the .real cast
    import scipy.special as sps

    cache = tmp_path / "guards.sqlite"
    k = 4.0

    def kernel_radial(r):
        return 0.25j * sps.hankel1(0, k * np.asarray(r, dtype=np.complex128))

    with pytest.raises(RuntimeError, match="imaginary part"):
        _assemble_windowed_for_zeta(
            cache,
            2,
            3,
            complex(-(k * k)),
            kernel_radial,
            source_box_level=3,
            p_star=2,
            smooth_quad_order=8,
            chan_regular_order=6,
            chan_radial_order=15,
            result_dtype=np.float64,
        )

# }}}


# {{{ T11: channel quadrature order legality and the entry-magnitude bound

# orders the tanh-sinh-fast radial builder does not honour: 3 and below hit
# its silent ``max(3, order)`` clamp, and 3-6 produce rules whose weights
# miss unity by more than 1e-8 (order 3: 3.3e-5, order 6: 6.1e-8)
@pytest.mark.parametrize("bad_radial", [-5, 0, 1, 2, 3, 5, 6])
def test_illegal_channel_radial_order_refused(tmp_path, bad_radial):
    from volumential.rke_table_assembly import get_windowed_channel_table

    cache = tmp_path / "orders.sqlite"
    with pytest.raises(ValueError, match="chan_radial_order"):
        get_windowed_channel_table(
            cache,
            3,
            2,
            0,
            source_box_level=3,
            root_extent=ROOT_EXTENT,
            window_theta=WINDOW_THETA,
            chan_regular_order=6,
            chan_radial_order=bad_radial,
        )

    # the public assembly entry point refuses before any build too
    with pytest.raises(ValueError, match="chan_radial_order"):
        assemble_windowed_parameterized_table(
            cache,
            2,
            "Yukawa",
            3,
            4.0,
            source_box_level=3,
            window_theta=WINDOW_THETA,
            chan_regular_order=6,
            chan_radial_order=bad_radial,
        )


@pytest.mark.parametrize("bad_regular", [-4, 0, 1])
def test_illegal_channel_regular_order_refused(tmp_path, bad_regular):
    from volumential.rke_table_assembly import get_windowed_channel_table

    cache = tmp_path / "orders.sqlite"
    with pytest.raises(ValueError, match="chan_regular_order"):
        get_windowed_channel_table(
            cache,
            3,
            2,
            0,
            source_box_level=3,
            root_extent=ROOT_EXTENT,
            window_theta=WINDOW_THETA,
            chan_regular_order=bad_regular,
            chan_radial_order=15,
        )


def test_legal_channel_orders_pass_validation():
    from volumential.rke_table_assembly import _resolve_channel_orders

    # every order this test module, the module defaults, and the published
    # 2D/3D sweeps use must validate (validation is a legality gate, not an
    # accuracy gate)
    for radial in (12, 15, 25, 45, 61, 101, 121, 141, 161, 201):
        assert _resolve_channel_orders(3, 6, radial) == (6, radial)
    for regular in (2, 6, 14, 20, 28, 36, 48):
        assert _resolve_channel_orders(2, regular, 61) == (regular, 61)


def test_channel_entry_bound_rejects_poisoned_table(tmp_path, monkeypatch):
    from pathlib import Path

    import volumential.rke_table_assembly as rta

    cache = tmp_path / "poison.sqlite"
    kwargs = {
        "source_box_level": 3,
        "root_extent": ROOT_EXTENT,
        "window_theta": WINDOW_THETA,
        "chan_regular_order": 6,
        "chan_radial_order": 15,
    }
    window_scale = (_box_extent(3) / WINDOW_THETA) ** 2
    good = rta.get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    values = np.asarray(good.get_reduced_table_data()[1])

    # the analytic bound is a real bound, not a fudge: the honest table sits
    # comfortably under it
    bound = rta._channel_entry_magnitude_bound(good, 0, window_scale)
    assert np.max(np.abs(values)) < bound

    # an entry astronomically above the chi_0 mass scale is provably wrong
    poisoned = values.copy()
    poisoned[0] = 1.0e101
    with pytest.raises(RuntimeError, match="analytic bound"):
        rta._check_channel_table_values(good, poisoned, 0, window_scale)

    non_finite = values.copy()
    non_finite[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        rta._check_channel_table_values(good, non_finite, 0, window_scale)

    # a cache poisoned on disk is discarded and rebuilt, never trusted
    cache_dir = Path(str(cache) + ".windowed")
    (cache_file,) = sorted(cache_dir.glob("*.npz"))
    with np.load(cache_file, allow_pickle=False) as payload:
        stored = {name: payload[name] for name in payload.files}
    stored["values"] = poisoned
    with open(cache_file, "wb") as stream:
        np.savez(stream, **stored)
    healed = rta.get_windowed_channel_table(cache, 2, 3, 0, **kwargs)
    assert np.array_equal(
        values, np.asarray(healed.get_reduced_table_data()[1])
    )

    # and a builder that returns garbage fails the build instead of caching
    # a poisoned table that every downstream certificate field would rate as
    # perfectly conditioned
    real_builder = rta._duffy_channel_entry_values

    def poisoned_builder(table, profile, regular_order, radial_order):
        entry_ids, entry_values = real_builder(
            table, profile, regular_order, radial_order
        )
        entry_values = np.asarray(entry_values).copy()
        entry_values[0] = 1.0e101
        return entry_ids, entry_values

    monkeypatch.setattr(rta, "_duffy_channel_entry_values", poisoned_builder)
    fresh = tmp_path / "poison-build.sqlite"
    with pytest.raises(RuntimeError, match="analytic bound"):
        rta.get_windowed_channel_table(fresh, 2, 3, 0, **kwargs)
    assert not list(Path(str(fresh) + ".windowed").glob("*.npz"))


def test_3d_channel_order_convergence(tmp_path):
    # Radial orders whose smallest tanh-sinh-fast node sits on the float64
    # floor 5.551115e-17 (orders 10, 12, 36, 101, 201, ...) place a Duffy
    # node closer to the target than one ulp of the target coordinate.
    # Differencing absolute box coordinates absorbed such a node onto the
    # target, and the 3D germ's small-argument clip then reported
    # sqrt(pi/1e-300) ~ 1.8e150 instead of a pole, poisoning chi_0 by ~1e101.
    # Order 12 reproduces that geometry at negligible cost.
    from volumential.rke_table_assembly import get_windowed_channel_table

    cache = tmp_path / "conv.sqlite"

    def entries(regular, radial):
        table = get_windowed_channel_table(
            cache,
            3,
            2,
            0,
            source_box_level=3,
            root_extent=ROOT_EXTENT,
            window_theta=WINDOW_THETA,
            chan_regular_order=regular,
            chan_radial_order=radial,
        )
        return np.asarray(table.get_reduced_table_data()[1])

    reference = entries(14, 45)
    scale = float(np.max(np.abs(reference)))
    # chi_0 carries the full channel mass 4 pi t_w on the self case
    assert abs(scale / (4.0 * np.pi * (_box_extent(3) / WINDOW_THETA) ** 2)
               - 1.0) < 0.05

    deviations = [
        float(np.max(np.abs(entries(regular, radial) - reference))) / scale
        for regular, radial in ((6, 12), (8, 20), (10, 25))
    ]
    # the floor-node order is merely coarse, not catastrophic
    assert deviations[0] < 1.0e-2, deviations
    # and refining the pair improves monotonically
    assert deviations[2] < deviations[1] < deviations[0], deviations

# }}}


if __name__ == "__main__":
    import sys

    pytest.main([sys.argv[0], "-x", "-q"])

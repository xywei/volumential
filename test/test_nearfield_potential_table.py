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

import os
import sys

import numpy as np
import pyopencl as cl
import pytest
from numpy.polynomial.chebyshev import chebval, chebval2d, chebval3d

if (
    sys.platform == "darwin"
    and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
):
    pytest.skip(
        "nearfield potential table tests are unstable on macOS OpenCL CI "
        "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)",
        allow_module_level=True,
    )

import volumential.nearfield_potential_table as npt
from volumential.table_manager import ConstantKernel


def _make_build_queue_or_skip():
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    for platform in platforms:
        if platform.name == "Intel(R) OpenCL":
            continue
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    pytest.skip("No OpenCL devices available for table builds")


def test_const_order_1():
    queue = _make_build_queue_or_skip()
    table = npt.NearFieldInteractionTable(
        quad_order=1,
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(2),
        progress_bar=False,
    )
    table.build_table(queue=queue)
    for entry_id in range(len(table.data)):
        assert np.allclose(table.get_entry_data(entry_id), 1)


def test_const_order_2(longrun):
    queue = _make_build_queue_or_skip()
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(2),
        progress_bar=False,
    )
    table.build_table(queue=queue)
    for entry_id in range(len(table.data)):
        assert np.allclose(table.get_entry_data(entry_id), 0.25)


def interp_modes(q_order):
    table = npt.NearFieldInteractionTable(quad_order=q_order, progress_bar=False)

    modes = [table.get_mode(i) for i in range(table.n_q_points)]

    def interpolate_function(x, y):
        return sum([mode(x, y) for mode in modes])

    h = 0.2
    xx = yy = np.arange(-1.0, 1.0, h)
    xi, yi = np.meshgrid(xx, yy)
    xi = xi.flatten()
    yi = yi.flatten()

    val = np.zeros(xi.shape)
    print(xi)
    print(yi)

    for i in range(len(xi)):
        val[i] = interpolate_function(xi[i], yi[i])

    return val


def test_modes():
    for q in [1, 2, 3, 4, 5, 6, 7, 8]:
        val = interp_modes(q)
        assert np.allclose(val, 1)


def test_sumpy_kernel_to_lambda_lambdifies_once(monkeypatch):
    from sumpy.kernel import LaplaceKernel

    import sympy

    call_count = {"n": 0}
    original_lambdify = sympy.lambdify

    def wrapped_lambdify(*args, **kwargs):
        call_count["n"] += 1
        return original_lambdify(*args, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", wrapped_lambdify)

    knl_func = npt.sumpy_kernel_to_lambda(LaplaceKernel(2))
    assert np.isfinite(knl_func(0.5, 0.25))
    assert np.isfinite(knl_func(0.25, 0.5))
    assert call_count["n"] == 1


def test_sumpy_kernel_to_lambda_applies_wrapper_postprocess():
    class FakeKernel:
        dim = 1

        def get_expression(self, args):
            return args[0]

        def postprocess_at_source(self, expr, bvec):
            return expr + 1

        def postprocess_at_target(self, expr, bvec):
            return 2 * expr

        def get_global_scaling_const(self):
            return 3

    knl_func = npt.sumpy_kernel_to_lambda(FakeKernel())
    assert knl_func(4.0) == 30.0


def cheb_eval(dim, coefs, coords):
    if dim == 1:
        return chebval(coords[0], coefs)
    elif dim == 2:
        return chebval2d(coords[0], coords[1], coefs)
    elif dim == 3:
        return chebval3d(coords[0], coords[1], coords[2], coefs)
    else:
        raise NotImplementedError("dimension %d not supported" % dim)


def drive_test_modes_cheb_coeffs(dim, q, cheb_order):
    if not cheb_order >= q:
        raise RuntimeError("Insufficient cheb_order to fully resolve the modes")

    rng = np.random.default_rng(seed=42)
    sample_mode = rng.integers(q**dim)
    table = npt.NearFieldInteractionTable(quad_order=q, dim=dim, progress_bar=False)
    ccoefs = table.get_mode_cheb_coeffs(sample_mode, cheb_order)
    shape = (cheb_order,) * dim
    ccoefs = ccoefs.reshape(*shape)

    # Evaluate the mode at the interpolation nodes via
    # inverse Chebyshev transform.
    #
    # NOTE: table.q_points are over [0, 1]^dim,
    # while cheb_eval assumes points are over [-1, 1]^dim
    targets = np.array([[qpt[i] for qpt in table.q_points] for i in range(dim)]) * 2 - 1

    mode_vals = cheb_eval(dim, ccoefs, targets)
    mode_vals[np.abs(mode_vals) < 8 * np.finfo(mode_vals.dtype).eps] = 0

    mode_interp_vals = np.zeros(q**dim)
    mode_interp_vals[sample_mode] = 1

    assert np.allclose(mode_vals, mode_interp_vals)


def test_modes_cheb_coeffs():
    drive_test_modes_cheb_coeffs(1, 5, 5)
    drive_test_modes_cheb_coeffs(1, 5, 10)
    drive_test_modes_cheb_coeffs(1, 10, 10)
    drive_test_modes_cheb_coeffs(1, 15, 15)

    drive_test_modes_cheb_coeffs(2, 5, 5)
    drive_test_modes_cheb_coeffs(2, 5, 10)
    drive_test_modes_cheb_coeffs(2, 10, 10)
    drive_test_modes_cheb_coeffs(2, 15, 15)

    drive_test_modes_cheb_coeffs(3, 5, 5)
    drive_test_modes_cheb_coeffs(3, 5, 10)
    drive_test_modes_cheb_coeffs(3, 10, 10)


@pytest.mark.parametrize("q_order", [1, 2, 3, 5, 8])
def test_build_normalizer_table_matches_gauss_legendre_weights(monkeypatch, q_order):
    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=2,
        build_method="DuffyRadial",
        progress_bar=False,
    )

    def fail_compute_nmlz(_mode_id):
        raise AssertionError("normalizer fallback path should not be used")

    monkeypatch.setattr(table, "compute_nmlz", fail_compute_nmlz)

    table.build_normalizer_table()

    _, weights_1d = np.polynomial.legendre.leggauss(q_order)
    weights_1d = (0.5 * table.source_box_extent * weights_1d).astype(table.dtype)
    expected = np.multiply.outer(weights_1d, weights_1d).reshape(-1)

    atol = 256 * np.finfo(table.dtype).eps
    assert np.allclose(table.mode_normalizers, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    "dim, q_order",
    [
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 2),
        (3, 3),
    ],
)
def test_invariant_entry_info_matches_orbit_decode_metadata(dim, q_order):
    from volumential.table_manager import ConstantKernel

    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(dim),
        progress_bar=False,
    )

    invariant_info = table._get_invariant_entry_info()
    invariant_entry_ids = np.array(invariant_info["entry_ids"], dtype=np.int64)
    canonical_entry_ids = np.array(
        invariant_info["canonical_entry_ids"], dtype=np.int64
    )

    assert canonical_entry_ids.shape == (len(table.data),)
    assert np.array_equal(np.unique(canonical_entry_ids), invariant_entry_ids)

    decoded_case_ids = np.array(
        [
            table.decode_index(int(entry_id))["case_index"]
            for entry_id in invariant_entry_ids
        ],
        dtype=np.int32,
    )
    decoded_target_ids = np.array(
        [
            table.decode_index(int(entry_id))["target_point_index"]
            for entry_id in invariant_entry_ids
        ],
        dtype=np.int32,
    )
    decoded_source_ids = np.array(
        [
            table.decode_index(int(entry_id))["source_mode_index"]
            for entry_id in invariant_entry_ids
        ],
        dtype=np.int32,
    )

    assert np.array_equal(invariant_info["case_indices"], decoded_case_ids)
    assert np.array_equal(invariant_info["target_point_indices"], decoded_target_ids)
    assert np.array_equal(invariant_info["source_mode_indices"], decoded_source_ids)

    expected_mode_axes = np.array(
        [table.unwrap_mode_index(int(mode_id)) for mode_id in decoded_source_ids],
        dtype=np.int32,
    )
    assert np.array_equal(invariant_info["mode_axes"], expected_mode_axes)


def test_orbit_canonicalization_preserves_value_lookup_equivalence():
    from volumential.table_manager import ConstantKernel

    table = npt.NearFieldInteractionTable(
        quad_order=3,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(2),
        progress_bar=False,
    )

    info = table._get_invariant_entry_info()
    canonical_entry_ids = np.array(info["canonical_entry_ids"], dtype=np.int64)

    rng = np.random.default_rng(seed=11)
    table.data = np.asarray(rng.normal(size=len(table.data)), dtype=table.dtype)
    table.table_data_is_symmetry_reduced = False

    # populate canonical entries with deterministic values and mirror all entries
    # from canonical orbit representatives.
    canonical_vals = np.asarray(
        rng.normal(size=len(info["entry_ids"])), dtype=table.dtype
    )
    for irep, entry_id in enumerate(info["entry_ids"]):
        table.data[int(entry_id)] = canonical_vals[irep]
    table.data[:] = table.data[canonical_entry_ids]
    table.table_data_is_symmetry_reduced = True

    for _ in range(32):
        source_mode_id = int(rng.integers(0, table.n_q_points))
        target_point_id = int(rng.integers(0, table.n_q_points))
        case_id = int(rng.integers(0, table.n_cases))

        full_entry_id = (
            case_id * table.n_pairs
            + source_mode_id * table.n_q_points
            + target_point_id
        )
        mapped_entry_id = table.get_entry_index(
            source_mode_id, target_point_id, case_id
        )
        expected_entry_id = int(canonical_entry_ids[full_entry_id])
        assert mapped_entry_id == expected_entry_id
        assert table.data[mapped_entry_id] == table.data[expected_entry_id]


def test_orbit_canonicalization_tracks_sign_for_target_derivative_kernel():
    from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

    table = npt.NearFieldInteractionTable(
        quad_order=3,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="inv_power",
        sumpy_kernel=AxisTargetDerivative(0, LaplaceKernel(2)),
        derive_kernel_func=False,
        progress_bar=False,
    )

    info = table._get_invariant_entry_info()
    scales = np.asarray(info["canonical_scales"])

    # Derivative kernels are odd under some reflections; orbit metadata must
    # keep sign information rather than assuming all +1.
    assert np.any(scales < 0)


def test_orbit_canonicalization_tracks_sign_for_directional_source_derivative():
    from sumpy.kernel import DirectionalSourceDerivative, LaplaceKernel

    table = npt.NearFieldInteractionTable(
        quad_order=3,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="inv_power",
        sumpy_kernel=DirectionalSourceDerivative(LaplaceKernel(2), "dir_vec"),
        derive_kernel_func=False,
        symmetry_source_direction=np.array([1.0, 0.0]),
        progress_bar=False,
    )

    info = table._get_invariant_entry_info()
    scales = np.asarray(info["canonical_scales"])
    assert np.any(scales < 0)


def test_directional_source_orbit_updates_when_direction_changes():
    from sumpy.kernel import DirectionalSourceDerivative, LaplaceKernel

    table = npt.NearFieldInteractionTable(
        quad_order=3,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="inv_power",
        sumpy_kernel=DirectionalSourceDerivative(LaplaceKernel(2), "dir_vec"),
        derive_kernel_func=False,
        symmetry_source_direction=np.array([1.0, 0.0]),
        progress_bar=False,
    )

    info_x = table._get_orbit_canonical_info()
    scales_x = np.asarray(info_x["canonical_scales"]).copy()

    table.symmetry_source_direction = np.array([0.0, 1.0])
    info_y = table._get_orbit_canonical_info()
    scales_y = np.asarray(info_y["canonical_scales"]).copy()

    assert scales_x.shape == scales_y.shape
    assert not np.array_equal(scales_x, scales_y)


def test_duffy_radial_routes_queue_to_batched_builder(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        self.mode_normalizers[:] = 1

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["queue"] = queue
        seen["called"] = True
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    q = object()
    table.build_table_via_duffy_radial(queue=q)

    assert seen["called"]
    assert seen["queue"] is q


def test_build_table_uses_supplied_cl_ctx_when_queue_missing(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {"create_some_context_called": False}
    sentinel_ctx = object()
    sentinel_queue = object()

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["queue"] = queue
        self.is_built = True

    def fake_create_some_context(interactive=False):
        seen["create_some_context_called"] = True
        raise AssertionError("unexpected auto context creation")

    def fake_command_queue(ctx):
        assert ctx is sentinel_ctx
        return sentinel_queue

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )
    monkeypatch.setattr(npt.cl, "create_some_context", fake_create_some_context)
    monkeypatch.setattr(npt.cl, "CommandQueue", fake_command_queue)

    table.build_table(
        cl_ctx=sentinel_ctx,
        queue=None,
        build_config=npt.DuffyBuildConfig(),
    )

    assert seen["queue"] is sentinel_queue
    assert not seen["create_some_context_called"]


def test_duffy_radial_keeps_legacy_deg_theta_alias(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["called"] = True
        seen["regular_quad_order"] = deg_theta

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(
        queue=object(),
        deg_theta=17,
    )

    assert seen["called"]
    assert seen["regular_quad_order"] == 17


def test_duffy_radial_accepts_build_config_dataclass(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["called"] = True
        seen["radial_rule"] = radial_rule
        seen["regular_quad_order"] = deg_theta
        seen["radial_quad_order"] = radial_quad_order
        seen["mp_dps"] = mp_dps
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(
        queue=object(),
        build_config=npt.DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=9,
            radial_quad_order=41,
            mp_dps=70,
        ),
    )

    assert seen["called"]
    assert seen["radial_rule"] == "tanh-sinh-fast"
    assert seen["regular_quad_order"] == 9
    assert seen["radial_quad_order"] == 41
    assert seen["mp_dps"] == 70


def test_duffy_radial_adaptive_rule_uses_scalar_fallback(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=1,
        build_method="DuffyRadial",
        dim=2,
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=None,
        progress_bar=False,
    )

    seen = {"batched_called": False, "scalar_calls": 0}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        self.mode_normalizers[:] = 1

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["batched_called"] = True
        raise AssertionError("adaptive rule should not use batched builder")

    def fake_compute_entry(
        self,
        entry_id,
        radial_rule="adaptive",
        deg_theta=20,
        radial_quad_order=61,
        mp_dps=50,
    ):
        seen["scalar_calls"] += 1
        assert radial_rule == "adaptive"
        return entry_id, float(entry_id + 1)

    def identity_lookup_by_symmetry(self, entry_id):
        return entry_id, entry_id

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "compute_table_entry_duffy_radial",
        fake_compute_entry,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "lookup_by_symmetry",
        identity_lookup_by_symmetry,
    )

    table.build_table(
        build_config=npt.DuffyBuildConfig(
            radial_rule="adaptive",
            regular_quad_order=6,
            radial_quad_order=41,
        ),
    )

    assert not seen["batched_called"]
    invariant_entry_ids = table._get_invariant_entry_info()["entry_ids"]
    assert seen["scalar_calls"] == len(invariant_entry_ids)
    assert table.is_built
    for entry_id in range(len(table.data)):
        assert np.isfinite(table.get_entry_data(entry_id))


def test_duffy_radial_auto_tune_orders_routes_to_batched_builder(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_autotune(
        self,
        radial_rule,
        mp_dps,
        queue=None,
        sample_count=5,
        candidates=None,
        floor_factor=8.0,
        kernel_kwargs=None,
    ):
        seen["auto_called"] = True
        seen["sample_count"] = sample_count
        seen["floor_factor"] = floor_factor
        return (
            8,
            31,
            {
                "auto_tuned": True,
                "selected_regular_quad_order": 8,
                "selected_radial_quad_order": 31,
            },
        )

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["called"] = True
        seen["regular_quad_order"] = deg_theta
        seen["radial_quad_order"] = radial_quad_order
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_auto_tune_duffy_radial_orders",
        fake_autotune,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(
        queue=object(),
        auto_tune_orders=True,
        auto_tune_samples=3,
        auto_tune_floor_factor=6.0,
    )

    assert seen["auto_called"]
    assert seen["sample_count"] == 3
    assert seen["floor_factor"] == 6.0
    assert seen["called"]
    assert seen["regular_quad_order"] == 8
    assert seen["radial_quad_order"] == 31


def test_duffy_radial_auto_keyword_orders_trigger_autotune(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_autotune(
        self,
        radial_rule,
        mp_dps,
        queue=None,
        sample_count=5,
        candidates=None,
        floor_factor=8.0,
        kernel_kwargs=None,
    ):
        seen["auto_called"] = True
        return (
            6,
            21,
            {
                "auto_tuned": True,
                "selected_regular_quad_order": 6,
                "selected_radial_quad_order": 21,
            },
        )

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["called"] = True
        seen["regular_quad_order"] = deg_theta
        seen["radial_quad_order"] = radial_quad_order
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_auto_tune_duffy_radial_orders",
        fake_autotune,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(
        queue=object(),
        regular_quad_order="auto",
        radial_quad_order="auto",
    )

    assert seen["auto_called"]
    assert seen["called"]
    assert seen["regular_quad_order"] == 6
    assert seen["radial_quad_order"] == 21


def test_duffy_radial_partial_auto_order_keeps_explicit_value(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_autotune(
        self,
        radial_rule,
        mp_dps,
        queue=None,
        sample_count=5,
        candidates=None,
        floor_factor=8.0,
        kernel_kwargs=None,
    ):
        seen["auto_called"] = True
        return (
            6,
            21,
            {
                "auto_tuned": True,
                "selected_regular_quad_order": 6,
                "selected_radial_quad_order": 21,
            },
        )

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["called"] = True
        seen["regular_quad_order"] = deg_theta
        seen["radial_quad_order"] = radial_quad_order
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_auto_tune_duffy_radial_orders",
        fake_autotune,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(
        queue=object(),
        regular_quad_order=12,
        radial_quad_order="auto",
    )

    assert seen["auto_called"]
    assert seen["called"]
    assert seen["regular_quad_order"] == 12
    assert seen["radial_quad_order"] == 21


def test_duffy_autotune_two_candidates_prefers_fine_rule(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    monkeypatch.setattr(
        table,
        "_duffy_autotune_sample_entry_ids",
        lambda sample_count: [0, 1],
    )

    def fake_compute_table_entry_duffy_radial(
        self,
        entry_id,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
    ):
        if (deg_theta, radial_quad_order) == (4, 11):
            values = np.array([1.0, -1.0], dtype=np.float64)
        elif (deg_theta, radial_quad_order) == (12, 61):
            values = np.array([1.1, -0.9], dtype=np.float64)
        else:
            raise AssertionError("unexpected candidate pair")

        return entry_id, values[entry_id]

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "compute_table_entry_duffy_radial",
        fake_compute_table_entry_duffy_radial,
    )

    selected_regular, selected_radial, info = table._auto_tune_duffy_radial_orders(
        radial_rule="tanh-sinh-fast",
        mp_dps=50,
        queue=None,
        sample_count=2,
        candidates=[(4, 11), (12, 61)],
        floor_factor=8.0,
    )

    assert (selected_regular, selected_radial) == (12, 61)
    assert info["relative_errors_vs_best"][0] > info["acceptance_threshold"]


def test_mode_remap_is_elementwise_for_vectorized_inputs():
    table = npt.NearFieldInteractionTable(quad_order=3, progress_bar=False)

    mode = table.get_mode(0)

    x = np.array([-0.8, 0.25], dtype=np.float64)
    y = np.array([0.4, 0.5], dtype=np.float64)

    scalar_vals = np.array(
        [mode(float(ix), float(iy)) for ix, iy in zip(x, y)],
        dtype=np.float64,
    )
    vector_vals = mode(x, y)

    assert np.allclose(scalar_vals, vector_vals)


def test_quad_order_one_mode_preserves_numpy_broadcasting():
    table = npt.NearFieldInteractionTable(quad_order=1, dim=2, progress_bar=False)

    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)[:, np.newaxis]
    y = np.array([0.4, 0.5], dtype=np.float64)[np.newaxis, :]

    mode = table.get_mode(0)
    template_mode = table.get_template_mode(0)

    mode_vals = mode(x, y)
    template_mode_vals = template_mode(x, y)

    assert mode_vals.shape == (3, 2)
    assert template_mode_vals.shape == (3, 2)
    assert np.all(mode_vals == 1)
    assert np.all(template_mode_vals == 1)
    assert np.isscalar(mode(0.1, 0.2))
    assert np.isscalar(template_mode(0.1, 0.2))


def test_duffy_radial_batched_initializes_normalizers(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        progress_bar=False,
    )
    table.integral_knl = object()

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        self.mode_normalizers[:] = 2
        seen["normalizers"] = True

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        assert seen["normalizers"]
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    table.build_table_via_duffy_radial(queue=object())

    assert seen["normalizers"]
    assert table.has_normalizers
    assert table.mode_normalizers[0] == 2


def test_duffy_radial_batched_clamps_decomposition_vertex(monkeypatch):
    class DummyIntegralKernel:
        def get_expression(self, _dvec):
            return 0

        def get_global_scaling_const(self):
            return 1

    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        progress_bar=False,
    )
    table.integral_knl = DummyIntegralKernel()

    captured = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        self.mode_normalizers[:] = 1

    def fake_invariant_entry_info(self):
        return {
            "entry_ids": np.array([0], dtype=np.int32),
            "case_indices": np.array([0], dtype=np.int32),
            "target_point_indices": np.array([0], dtype=np.int32),
            "source_mode_indices": np.array([0], dtype=np.int32),
            "mode_axes": np.array([[0, 0]], dtype=np.int32),
        }

    def fake_case_points(self):
        return np.array(
            [
                [[2.10, 0.15, 0.32, 0.75]],
                [[-0.10, 1.20, 0.60, -0.10]],
            ],
            dtype=np.float64,
        )

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    def fake_node_data(
        self, radial_rule, regular_quad_order, radial_quad_order, mp_dps
    ):
        return {
            "n_nodes": 1,
            "node_u": np.ones((self.dim, 1), dtype=self.dtype),
            "node_sign": np.ones((self.dim, 1), dtype=self.dtype),
            "node_jac_base": np.ones(1, dtype=self.dtype),
        }

    def fake_program_factory(self, queue, n_entries, n_nodes):
        def fake_program(*args, **kwargs):
            captured["decomposition_targets"] = kwargs["decomposition_targets"]
            captured["target_points"] = kwargs["target_points"]
            return None, {"result": FakeResult(np.zeros(n_entries, dtype=table.dtype))}

        return fake_program

    def identity_lookup_by_symmetry(self, entry_id):
        return entry_id, entry_id

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_invariant_entry_info",
        fake_invariant_entry_info,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_case_target_points",
        fake_case_points,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_duffy_radial_node_data",
        fake_node_data,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_fused_invariant_duffy_table_program",
        fake_program_factory,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "lookup_by_symmetry",
        identity_lookup_by_symmetry,
    )

    table.build_table_via_duffy_radial(queue=object())

    decomposition_targets = captured["decomposition_targets"]
    target_points = captured["target_points"]

    assert np.all(decomposition_targets >= 0)
    assert np.all(decomposition_targets <= table.source_box_extent)
    assert np.allclose(decomposition_targets[:, 0], np.array([1.0, 0.0]))
    assert np.allclose(target_points[:, 0], np.array([2.10, -0.10]))


def test_duffy_radial_routes_queue_to_batched_builder_1d(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=1,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fail_build_normalizer_table(self, pool=None, pb=None):
        raise AssertionError("normalizer table should not be built in 1D")

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["queue"] = queue
        seen["dim"] = self.dim
        seen["called"] = True
        self.is_built = True
        self.last_duffy_build_timings = {
            "invariant_info_s": 0.0,
            "quadrature_s": 0.0,
            "scatter_s": 0.0,
            "total_s": 0.0,
            "n_entries": 0,
        }

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fail_build_normalizer_table,
    )

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    q = object()
    table.build_table_via_duffy_radial(queue=q)

    assert seen["called"]
    assert seen["queue"] is q
    assert seen["dim"] == 1
    assert table.last_duffy_build_timings["normalizer_s"] == 0.0


def test_duffy_radial_routes_queue_to_batched_builder_3d(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=3,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {}

    def fail_build_normalizer_table(self, pool=None, pb=None):
        raise AssertionError("normalizer table should not be built in 3D")

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["queue"] = queue
        seen["dim"] = self.dim
        seen["called"] = True
        self.is_built = True
        self.last_duffy_build_timings = {
            "invariant_info_s": 0.0,
            "quadrature_s": 0.0,
            "scatter_s": 0.0,
            "total_s": 0.0,
            "n_entries": 0,
        }

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fail_build_normalizer_table,
    )

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )

    q = object()
    table.build_table_via_duffy_radial(queue=q)

    assert seen["called"]
    assert seen["queue"] is q
    assert seen["dim"] == 3
    assert table.last_duffy_build_timings["normalizer_s"] == 0.0


def test_duffy_radial_wrapped_kernel_does_not_fallback_to_scalar(monkeypatch):
    class WrappedKernel:
        def __init__(self):
            self._base = object()

        def get_base_kernel(self):
            return self._base

    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=WrappedKernel(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {"scalar_called": False}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_batched(
        self,
        queue,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        raise RuntimeError("batched failed")

    def fake_scalar(
        self,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["scalar_called"] = True
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched",
        fake_batched,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_build_table_via_duffy_radial_scalar",
        fake_scalar,
    )

    with pytest.raises(RuntimeError, match="scalar fallback is disabled"):
        table.build_table_via_duffy_radial(queue=object())

    assert not seen["scalar_called"]


def test_duffy_radial_adaptive_rejects_wrapped_kernel_scalar_path(monkeypatch):
    class WrappedKernel:
        def __init__(self):
            self._base = object()

        def get_base_kernel(self):
            return self._base

    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=WrappedKernel(),
        derive_kernel_func=False,
        progress_bar=False,
    )

    seen = {"scalar_called": False}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_scalar(
        self,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
        kernel_kwargs=None,
    ):
        seen["scalar_called"] = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_build_table_via_duffy_radial_scalar",
        fake_scalar,
    )

    with pytest.raises(RuntimeError, match="Adaptive DuffyRadial scalar build"):
        table.build_table_via_duffy_radial(
            radial_rule="adaptive",
            regular_quad_order=8,
            radial_quad_order=31,
            mp_dps=50,
        )

    assert not seen["scalar_called"]


def test_duffy_runtime_kernel_kwargs_reject_none_or_nonnumeric():
    class FakeLoopyArg:
        def __init__(self, name):
            self.name = name

    class FakeKernelArg:
        def __init__(self, name):
            self.loopy_arg = FakeLoopyArg(name)

    class FakeKernel:
        def get_args(self):
            return [FakeKernelArg("lam")]

    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        sumpy_kernel=object(),
        derive_kernel_func=False,
        progress_bar=False,
    )
    table.integral_knl = FakeKernel()

    with pytest.raises(TypeError, match="lam=None"):
        table._extract_integral_kernel_runtime_kwargs({"lam": None})

    with pytest.raises(TypeError, match="expected numeric scalar"):
        table._extract_integral_kernel_runtime_kwargs({"lam": "abc"})

    with pytest.raises(TypeError, match="expected finite numeric scalar"):
        table._extract_integral_kernel_runtime_kwargs({"lam": np.nan})

    with pytest.raises(TypeError, match="expected finite numeric scalar"):
        table._extract_integral_kernel_runtime_kwargs({"lam": np.inf})


def _get_cpu_queue_or_skip(ctx_factory):
    import pyopencl as cl

    ctx = ctx_factory()
    if not any(dev.type & cl.device_type.CPU for dev in ctx.devices):
        pytest.skip("batched-vs-reference table checks run on CPU contexts only")
    return cl.CommandQueue(ctx)


@pytest.mark.parametrize(
    "dim, q_order, regular_quad_order, radial_quad_order, rtol",
    [
        (1, 2, 2, 21, 1e-12),
        (2, 2, 8, 31, 1e-12),
        (3, 2, 5, 21, 1e-12),
        (2, 3, 10, 41, 1e-10),
    ],
)
def test_duffy_radial_batched_matches_scalar_reference_entries(
    ctx_factory,
    dim,
    q_order,
    regular_quad_order,
    radial_quad_order,
    rtol,
):
    from volumential.table_manager import ConstantKernel

    queue = _get_cpu_queue_or_skip(ctx_factory)
    sumpy_knl = ConstantKernel(dim)

    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=sumpy_knl,
        progress_bar=False,
    )
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )

    invariant_entry_ids = table._get_invariant_entry_info()["entry_ids"]
    assert np.all(np.isfinite(table.data[invariant_entry_ids]))

    sample_entry_ids = sorted(
        {
            int(invariant_entry_ids[0]),
            int(invariant_entry_ids[len(invariant_entry_ids) // 2]),
            int(invariant_entry_ids[-1]),
        }
    )

    reference_table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=sumpy_knl,
        progress_bar=False,
    )
    for entry_id in sample_entry_ids:
        _, ref_val = reference_table.compute_table_entry_duffy_radial(
            entry_id,
            radial_rule="tanh-sinh-fast",
            deg_theta=regular_quad_order,
            radial_quad_order=radial_quad_order,
        )
        rel_err = abs(table.data[entry_id] - ref_val) / max(1.0, abs(ref_val))
        assert rel_err < rtol


def test_duffy_radial_batched_laplace_center_case_is_finite(ctx_factory):
    from sumpy.kernel import LaplaceKernel

    queue = _get_cpu_queue_or_skip(ctx_factory)

    table = npt.NearFieldInteractionTable(
        quad_order=4,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.get_laplace(2),
        kernel_type="log",
        sumpy_kernel=LaplaceKernel(2),
        progress_bar=False,
    )

    invariant_info = table._get_invariant_entry_info()
    center_case_id = table.case_indices[table.case_encode([0, 0])]
    center_local_indices = np.flatnonzero(
        invariant_info["case_indices"] == center_case_id
    ).astype(np.int64)

    values = table._batched_duffy_values_for_local_indices(
        queue,
        invariant_info,
        center_local_indices,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=50,
        radial_quad_order=100,
        mp_dps=50,
    )

    assert np.all(np.isfinite(values))


def test_duffy_radial_batched_keeps_symmetry_reduced_storage(ctx_factory):
    queue = _get_cpu_queue_or_skip(ctx_factory)

    table = npt.NearFieldInteractionTable(
        quad_order=4,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(2),
        progress_bar=False,
    )
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=8,
        radial_quad_order=31,
    )

    invariant_entry_ids = np.asarray(
        table._get_invariant_entry_info()["entry_ids"], dtype=np.int64
    )
    non_invariant_ids = np.setdiff1d(
        np.arange(len(table.data), dtype=np.int64),
        invariant_entry_ids,
    )

    assert non_invariant_ids.size > 0
    assert table.table_data_is_symmetry_reduced
    assert np.all(np.isfinite(table.data[invariant_entry_ids]))
    assert np.all(np.isnan(table.data[non_invariant_ids]))

    # Reduced tables must still provide finite values through get_entry_index.
    for case_id in range(table.n_cases):
        entry_id = table.get_entry_index(0, 0, case_id)
        assert np.isfinite(table.data[entry_id])


def test_prepare_table_data_and_entry_map_rejects_mixed_storage_modes():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    full = SimpleNamespace(
        data=np.array([1.0, 2.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=False,
    )
    reduced = SimpleNamespace(
        data=np.array([1.0, np.nan], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
    )

    with pytest.raises(RuntimeError, match="mixed full/reduced"):
        _prepare_table_data_and_entry_map([full, reduced])


def test_prepare_table_data_and_entry_map_accepts_finite_full_entries():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    table0 = SimpleNamespace(
        data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0, 2.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0, 0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=False,
    )
    table1 = SimpleNamespace(
        data=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        mode_normalizers=np.array([4.0, 5.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0, 0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=False,
    )

    (
        table_data_combined,
        mode_nmlz_combined,
        exterior_mode_nmlz_combined,
        table_entry_ids,
        table_entry_scales,
    ) = _prepare_table_data_and_entry_map([table0, table1])

    assert table_data_combined.shape == (2, 3)
    np.testing.assert_allclose(table_data_combined[0], table0.data)
    np.testing.assert_allclose(table_data_combined[1], table1.data)
    np.testing.assert_array_equal(table_entry_ids, np.array([0, 1, 2]))
    np.testing.assert_allclose(table_entry_scales, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(mode_nmlz_combined[0], table0.mode_normalizers)
    np.testing.assert_allclose(mode_nmlz_combined[1], table1.mode_normalizers)
    np.testing.assert_allclose(
        exterior_mode_nmlz_combined[0], table0.kernel_exterior_normalizers
    )


def test_prepare_table_data_and_entry_map_rejects_nonfinite_full_entries():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    table0 = SimpleNamespace(
        data=np.array([1.0, np.inf, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=False,
    )
    table1 = SimpleNamespace(
        data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=False,
    )

    with pytest.raises(RuntimeError, match="full near-field table"):
        _prepare_table_data_and_entry_map([table0, table1])


def test_prepare_table_data_and_entry_map_rejects_mismatched_reduced_masks():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    table0 = SimpleNamespace(
        data=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
    )
    table1 = SimpleNamespace(
        data=np.array([1.0, 2.0, np.nan], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
    )

    with pytest.raises(RuntimeError, match="symmetry-reduced entry ids"):
        _prepare_table_data_and_entry_map([table0, table1])


def test_prepare_table_data_and_entry_map_accepts_consistent_reduced_masks():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    table0 = SimpleNamespace(
        data=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
    )
    table1 = SimpleNamespace(
        data=np.array([10.0, np.nan, 30.0], dtype=np.float64),
        mode_normalizers=np.array([2.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
    )

    (
        table_data_combined,
        mode_nmlz_combined,
        _,
        table_entry_ids,
        table_entry_scales,
    ) = _prepare_table_data_and_entry_map([table0, table1])

    assert table_data_combined.shape == (2, 2)
    np.testing.assert_allclose(table_data_combined[0], np.array([1.0, 3.0]))
    np.testing.assert_allclose(table_data_combined[1], np.array([10.0, 30.0]))
    np.testing.assert_array_equal(table_entry_ids, np.array([0, -1, 1]))
    np.testing.assert_allclose(table_entry_scales, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(mode_nmlz_combined[0], table0.mode_normalizers)
    np.testing.assert_allclose(mode_nmlz_combined[1], table1.mode_normalizers)


def test_prepare_table_data_and_entry_map_uses_orbit_canonical_mapping():
    from types import SimpleNamespace

    from volumential.expansion_wrangler_fpnd import _prepare_table_data_and_entry_map

    orbit_info = {
        "entry_ids": np.array([0, 2], dtype=np.int64),
        "canonical_entry_ids": np.array([0, 0, 2], dtype=np.int64),
        "canonical_scales": np.array([1.0, -1.0, 1.0], dtype=np.float64),
    }

    table0 = SimpleNamespace(
        data=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
        _get_orbit_canonical_info=lambda: orbit_info,
    )
    table1 = SimpleNamespace(
        data=np.array([10.0, np.nan, 30.0], dtype=np.float64),
        mode_normalizers=np.array([2.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        table_data_is_symmetry_reduced=True,
        _get_orbit_canonical_info=lambda: orbit_info,
    )

    (
        table_data_combined,
        mode_nmlz_combined,
        _,
        table_entry_ids,
        table_entry_scales,
    ) = _prepare_table_data_and_entry_map([table0, table1])

    assert table_data_combined.shape == (2, 2)
    np.testing.assert_allclose(table_data_combined[0], np.array([1.0, 3.0]))
    np.testing.assert_allclose(table_data_combined[1], np.array([10.0, 30.0]))
    np.testing.assert_array_equal(table_entry_ids, np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(table_entry_scales, np.array([1.0, -1.0, 1.0]))
    np.testing.assert_allclose(mode_nmlz_combined[0], table0.mode_normalizers)
    np.testing.assert_allclose(mode_nmlz_combined[1], table1.mode_normalizers)


def test_table_payload_serialization_excludes_nan_sentinels_for_reduced_tables():
    import io

    from volumential.table_manager import (
        _deserialize_table_payload,
        _serialize_table_payload,
    )

    table = npt.NearFieldInteractionTable(
        quad_order=2,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(2),
        progress_bar=False,
    )

    table.data[:] = np.array([1.0, np.nan, 2.0] + [np.nan] * (len(table.data) - 3))
    table.table_data_is_symmetry_reduced = True

    payload_blob = _serialize_table_payload(table)
    payload = _deserialize_table_payload(payload_blob)

    assert "data" not in payload
    assert "reduced_entry_ids" in payload
    assert "reduced_data" in payload
    assert np.all(np.isfinite(payload["reduced_data"]))
    assert not np.isnan(payload["reduced_data"]).any()


def test_batched_duffy_non_cl_executor_signature(monkeypatch):
    table = npt.NearFieldInteractionTable(quad_order=1, dim=2, progress_bar=False)

    invariant_info = {
        "mode_axes": np.array([[0, 0]], dtype=np.int32),
        "case_indices": np.array([0], dtype=np.int32),
        "target_point_indices": np.array([0], dtype=np.int32),
    }

    def fake_node_data(
        self, radial_rule, regular_quad_order, radial_quad_order, mp_dps
    ):
        del radial_rule, regular_quad_order, radial_quad_order, mp_dps
        return {
            "n_nodes": 1,
            "node_u": np.zeros((self.dim, 1), dtype=self.dtype),
            "node_sign": np.ones((self.dim, 1), dtype=self.dtype),
            "node_jac_base": np.ones(1, dtype=self.dtype),
        }

    def fake_case_target_points(self):
        return np.zeros((self.dim, 1, self.n_q_points), dtype=self.dtype)

    def fake_bary(self):
        return np.array([0.5], dtype=self.dtype), np.array([1], dtype=self.dtype)

    recorded = {}

    class FakeProgram:
        def executor(self, *args):
            recorded["executor_args"] = args

            def run(*args, **kwargs):
                recorded["call_args"] = args
                recorded["call_kwargs"] = kwargs
                return None, {"result": np.array([3.14], dtype=table.dtype)}

            return run

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_duffy_radial_node_data",
        fake_node_data,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_case_target_points",
        fake_case_target_points,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_barycentric_data",
        fake_bary,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "_get_fused_invariant_duffy_table_program",
        lambda self, queue, n_entries, n_nodes: FakeProgram(),
    )

    values = table._batched_duffy_values_for_local_indices(
        queue=object(),
        invariant_info=invariant_info,
        local_entry_indices=np.array([0], dtype=np.int64),
        radial_rule="tanh-sinh-fast",
        regular_quad_order=8,
        radial_quad_order=31,
        mp_dps=50,
    )

    assert recorded["executor_args"] == ()
    assert recorded["call_args"] == ()
    assert recorded["call_kwargs"]["source_box_extent"] == table.dtype(
        table.source_box_extent
    )
    assert values.shape == (1,)
    assert np.allclose(values, np.array([3.14], dtype=table.dtype))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main

        main([__file__])

# vim: foldmethod=marker:filetype=pyopencl

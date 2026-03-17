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

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebval2d, chebval3d

import volumential.nearfield_potential_table as npt


def test_const_order_1():
    table = npt.NearFieldInteractionTable(
        quad_order=1, build_method="Transform", progress_bar=False
    )
    table.build_table()
    for ary in table.data:
        assert np.allclose(ary, 1)


def test_const_order_2(longrun):
    table = npt.NearFieldInteractionTable(
        quad_order=2, build_method="Transform", progress_bar=False
    )
    table.build_table()
    for ary in table.data:
        assert np.allclose(ary, 0.25)


def interp_modes(q_order):
    table = npt.NearFieldInteractionTable(
        quad_order=q_order, build_method="Transform", progress_bar=False
    )

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


def test_droste_sum_routes_queue_to_batched_duffy(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DrosteSum",
        dim=2,
        sumpy_kernel=object(),
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        self.mode_normalizers[:] = 1

    def fake_batched(self, queue, radial_rule, deg_theta, radial_quad_order, mp_dps):
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
        "build_table_via_duffy_radial_batched_2d",
        fake_batched,
    )

    q = object()
    table.build_table_via_droste_bricks(queue=q)

    assert seen["called"]
    assert seen["queue"] is q


def test_droste_sum_keeps_legacy_deg_theta_alias(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DrosteSum",
        dim=2,
        sumpy_kernel=object(),
        progress_bar=False,
    )

    seen = {}

    def fake_build_normalizer_table(self, pool=None, pb=None):
        pass

    def fake_batched(self, queue, radial_rule, deg_theta, radial_quad_order, mp_dps):
        seen["called"] = True
        seen["regular_quad_order"] = deg_theta

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched_2d",
        fake_batched,
    )

    table.build_table_via_droste_bricks(
        queue=object(),
        deg_theta=17,
    )

    assert seen["called"]
    assert seen["regular_quad_order"] == 17


def test_mode_remap_is_elementwise_for_vectorized_inputs():
    table = npt.NearFieldInteractionTable(
        quad_order=3,
        build_method="Transform",
        progress_bar=False,
    )

    mode = table.get_mode(0)

    x = np.array([-0.8, 0.25], dtype=np.float64)
    y = np.array([0.4, 0.5], dtype=np.float64)

    scalar_vals = np.array(
        [mode(float(ix), float(iy)) for ix, iy in zip(x, y)],
        dtype=np.float64,
    )
    vector_vals = mode(x, y)

    assert np.allclose(scalar_vals, vector_vals)


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

    def fake_batched(self, queue, radial_rule, deg_theta, radial_quad_order, mp_dps):
        assert seen["normalizers"]
        self.is_built = True

    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_normalizer_table",
        fake_build_normalizer_table,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "build_table_via_duffy_radial_batched_2d",
        fake_batched,
    )

    table.build_table_via_duffy_radial(queue=object())

    assert seen["normalizers"]
    assert table.has_normalizers
    assert table.mode_normalizers[0] == 2


def test_duffy_radial_batched_clamps_decomposition_vertex(monkeypatch):
    table = npt.NearFieldInteractionTable(
        quad_order=2,
        build_method="DuffyRadial",
        dim=2,
        progress_bar=False,
    )
    table.integral_knl = object()

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

    def fake_program_factory(self, queue, n_entries, n_tri, n_theta, n_rho):
        def fake_program(*args, **kwargs):
            captured["tri_v0"] = kwargs["tri_v0"]
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
        "_get_fused_invariant_duffy_table_2d_program",
        fake_program_factory,
    )
    monkeypatch.setattr(
        npt.NearFieldInteractionTable,
        "lookup_by_symmetry",
        identity_lookup_by_symmetry,
    )

    table.build_table_via_duffy_radial(queue=object())

    tri_v0 = captured["tri_v0"]
    target_points = captured["target_points"]

    assert np.all(tri_v0 >= 0)
    assert np.all(tri_v0 <= table.source_box_extent)
    assert np.allclose(tri_v0[:, 0, 0], np.array([1.0, 0.0]))
    assert np.allclose(target_points[:, 0], np.array([2.10, -0.10]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main

        main([__file__])

# vim: foldmethod=marker:filetype=pyopencl

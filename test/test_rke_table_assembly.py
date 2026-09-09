"""Tests for the recursive-kernel-expansion (RKE) table assembly: truncation
order certification, conditioning guards, and agreement with a direct
batched build.
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

import numpy as np
import pytest

from volumential.rke_table_assembly import (
    assemble_parameterized_table,
    choose_truncation_order,
)


def _get_queue_or_skip():
    import pyopencl as cl

    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception as exc:
        pytest.skip(f"no OpenCL context available: {exc}")
    return cl.CommandQueue(ctx)


def test_truncation_order_monotone_and_certifiable():
    n1, b1 = choose_truncation_order(2, 1j * 2.0, 0.75, 1.0e-8)
    n2, b2 = choose_truncation_order(2, 1j * 2.0, 0.75, 1.0e-13)
    assert n2 >= n1
    assert b1 <= 1.0e-8 and b2 <= 1.0e-13

    with pytest.raises(ValueError):
        choose_truncation_order(2, 1j * 40.0, 4.0, 1.0e-12, max_terms=10)


def test_uncertifiable_tolerance_raises_value_error(tmp_path):
    # lam * radius ~ 68 at level 0 exhausts the default series window, so
    # the truncation selector must refuse before any channel is built.
    with pytest.raises(ValueError):
        choose_truncation_order(2, 1j * 8.0, 8.49, 1.0e-12)


def test_condition_guard_rejects_ill_conditioned_assembly(tmp_path):
    queue = _get_queue_or_skip()
    # A certifiable but cancellation-heavy configuration: the tolerance is
    # reachable within the series window, so channels assemble, and the
    # deliberately tight max_condition must then trip the RuntimeError
    # guard (level-3 2D at lam = 4 measures condition ~ 1.6).
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        assemble_parameterized_table(
            queue,
            tmp_path / "chan2d.sqlite",
            2,
            "Yukawa",
            2,
            4.0,
            source_box_level=3,
            tolerance=1.0e-8,
            max_condition=1.0,
        )


@pytest.mark.parametrize(
    ("dim", "kernel_type", "q_order", "parameter", "level"),
    [
        (2, "Yukawa", 3, 4.0, 3),
        (2, "Helmholtz", 3, 4.0, 3),
        # the 3D cases are the two slowest tests in the whole suite
        pytest.param(3, "Yukawa", 2, 2.0, 2, marks=pytest.mark.slow),
        pytest.param(3, "Helmholtz", 2, 2.0, 2, marks=pytest.mark.slow),
    ],
)
def test_assembled_matches_direct_batched(
    tmp_path, dim, kernel_type, q_order, parameter, level
):
    queue = _get_queue_or_skip()
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=16,
        radial_quad_order=45,
    )

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
        str(tmp_path / "direct.sqlite"),
        root_extent=2.0,
        queue=queue,
        **manager_kwargs,
    ) as table_manager:
        direct_table, _ = table_manager.get_table(
            dim,
            kernel_request,
            q_order,
            source_box_level=level,
            force_recompute=True,
            queue=queue,
            build_config=build_config,
            **get_kwargs,
        )

    assembled, certificate = assemble_parameterized_table(
        queue,
        tmp_path / "channels.sqlite",
        dim,
        kernel_type,
        q_order,
        parameter,
        source_box_level=level,
        tolerance=1.0e-11,
        build_config=build_config,
    )

    ids_direct, direct_values = direct_table.get_reduced_table_data()
    ids_assembled, assembled_values = assembled.get_reduced_table_data()
    assert np.array_equal(np.asarray(ids_direct), np.asarray(ids_assembled))

    direct_values = np.asarray(direct_values)
    assembled_values = np.asarray(assembled_values)
    scale = max(float(np.max(np.abs(direct_values))), 1.0e-300)
    deviation = float(
        np.max(np.abs(assembled_values - direct_values)) / scale
    )

    # Both paths share the batched quadrature character; the deviation is
    # bounded by the certified truncation plus the (common-order) quadrature
    # difference between the elementary-channel and direct integrands.
    assert deviation < 1.0e-6, (deviation, certificate)
    assert certificate["condition_number"] < 1.0e6
    if kernel_type == "Yukawa":
        assert certificate["assembled_max_abs_imag"] <= 1.0e-10 * max(
            float(np.max(np.abs(assembled_values))), 1.0e-300
        )
    # the assembled table must not inherit the base channel's scale-reuse
    # identity (a fixed-parameter table is not scale reusable)
    assert assembled.kernel_type is None


def test_classical_assembly_refuses_non_o1_box_extents(tmp_path):
    # The float64 recombination is certified only on an O(1) source-box
    # extent, and the refusal must come from the validation prologue --
    # before any channel table (or OpenCL queue) is touched.
    cache = tmp_path / "chan2d.sqlite"
    for kwargs in (
        {"source_box_level": 30},
        {"root_extent": 1.0e-6, "source_box_level": 5},
        {"root_extent": 1.0e6},
    ):
        with pytest.raises(ValueError, match="outside the supported"):
            assemble_parameterized_table(
                None, cache, 2, "Yukawa", 2, 1.0, **kwargs
            )
    assert not cache.exists()


def test_classical_assembly_validation_order_is_queue_free(tmp_path):
    # Every rejected request must be refused without a queue: unsupported
    # kernel families, the undefined zero-parameter 2D series, and
    # unsupported dimensions all raise from the prologue.
    cache = tmp_path / "chan2d.sqlite"
    with pytest.raises(NotImplementedError, match="Helmholtz and Yukawa"):
        assemble_parameterized_table(None, cache, 2, "Stokeslet", 2, 1.0)
    with pytest.raises(ValueError, match="zero-parameter 2D assembly"):
        assemble_parameterized_table(None, cache, 2, "Yukawa", 2, 0.0)
    with pytest.raises(NotImplementedError, match="only 2D and 3D"):
        assemble_parameterized_table(None, cache, 4, "Yukawa", 2, 1.0)
    assert not cache.exists()

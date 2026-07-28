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


def test_condition_guard_rejects_unresolved_parameters(tmp_path):
    queue = _get_queue_or_skip()
    # lam * radius ~ 23 at level 0: certified float64 recombination must
    # refuse rather than deliver a silently cancelled result.
    with pytest.raises((RuntimeError, ValueError)):
        assemble_parameterized_table(
            queue,
            tmp_path / "chan2d.sqlite",
            2,
            "Yukawa",
            2,
            8.0,
            source_box_level=0,
            tolerance=1.0e-12,
        )


@pytest.mark.parametrize(
    ("dim", "kernel_type", "q_order", "parameter", "level"),
    [
        (2, "Yukawa", 3, 4.0, 3),
        (2, "Helmholtz", 3, 4.0, 3),
        (3, "Yukawa", 2, 2.0, 2),
        (3, "Helmholtz", 2, 2.0, 2),
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
    assert certificate["assembled_max_abs_imag"] <= (
        1.0e-10 * max(float(np.max(np.abs(assembled_values))), 1.0e-300)
        if kernel_type == "Yukawa"
        else np.inf
    )

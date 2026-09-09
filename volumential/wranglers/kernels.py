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

__doc__ = """Auxiliary :mod:`sumpy` expression kernels for the Helmholtz split.

The Helmholtz near-field split expands the kernel into radial power and
power-times-log terms plus a series remainder; each of those needs its own
:class:`~sumpy.kernel.ExpressionKernel` so that sumpy can generate P2P code
for it.
"""

import numpy as np

from sumpy.kernel import ExpressionKernel


class _RadialPowerKernel(ExpressionKernel):
    init_arg_names = ("dim", "power")
    mapper_method = "map_expression_kernel"

    def __init__(self, dim, power):
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        if power < 0:
            raise ValueError("power must be non-negative")

        self.power = int(power)
        dim = int(dim)

        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = 1 if self.power == 0 else r**self.power

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return False

    def __getinitargs__(self):
        return (self.dim, self.power)

    def __repr__(self):
        return f"RadialPowerKernel(dim={self.dim}, power={self.power})"


class _RadialPowerLogKernel(ExpressionKernel):
    init_arg_names = ("dim", "power")
    mapper_method = "map_expression_kernel"

    def __init__(self, dim, power):
        from pymbolic import var
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        self.power = int(power)
        if self.power <= 0:
            raise ValueError("power must be positive for r**power * log(r)")

        dim = int(dim)
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = (r**self.power) * var("log")(r)

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return False

    def __getinitargs__(self):
        return (self.dim, self.power)

    def __repr__(self):
        return f"RadialPowerLogKernel(dim={self.dim}, power={self.power})"


class _HelmholtzSplitSeriesRemainderKernel(ExpressionKernel):
    """Analytic near-field split remainder for ``G_k - G_0``.

    ``split_order`` controls which non-smooth terms are removed from this
    remainder and handled by prebuilt near-field tables:

    - 2D: remove :math:`r^{2n}\\log r` for :math:`n=1,\\dots,p-1`.
    - 3D: remove odd powers :math:`r^{2j-1}` for :math:`j=1,\\dots,p-1`.

    The remaining smooth polynomial terms stay in this kernel.
    """

    init_arg_names = (
        "dim",
        "wave_number_real",
        "wave_number_imag",
        "split_order",
        "series_nmax",
    )
    mapper_method = "map_expression_kernel"

    def __init__(
        self,
        dim,
        wave_number_real,
        wave_number_imag,
        split_order,
        series_nmax,
    ):
        from math import factorial

        from pymbolic import var
        from pymbolic.primitives import Comparison, If
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        dim = int(dim)
        self.wave_number_real = float(wave_number_real)
        self.wave_number_imag = float(wave_number_imag)
        self.split_order = int(split_order)
        self.series_nmax = int(series_nmax)

        if dim not in (2, 3):
            raise NotImplementedError("split remainder kernel supports only 2D/3D")
        if self.split_order < 1:
            raise ValueError("split_order must be >= 1")

        k = np.complex128(self.wave_number_real + 1j * self.wave_number_imag)
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = 0

        if dim == 2:
            if np.abs(k) == 0.0:
                expr = np.complex128(0.0)
            else:
                expr = expr + np.complex128(
                    0.25j
                    - (1.0 / (2.0 * np.pi))
                    * (np.log(0.5 * k) + np.complex128(np.euler_gamma))
                )

                log_k_half = np.log(0.5 * k)
                euler_gamma = np.complex128(np.euler_gamma)
                for n in range(1, self.series_nmax + 1):
                    series_scale = (
                        ((-1) ** n) * (k * k / 4.0) ** n / (factorial(n) * factorial(n))
                    )
                    harmonic_n = np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64))
                    common = np.complex128(series_scale)

                    coeff_log = np.complex128(-common / (2.0 * np.pi))
                    coeff_power = np.complex128(
                        common
                        * (
                            (harmonic_n - (log_k_half + euler_gamma)) / (2.0 * np.pi)
                            + 0.25j
                        )
                    )

                    power = 2 * n
                    if n >= self.split_order:
                        log_term = If(
                            Comparison(r, "<=", np.float64(1.0e-300)),
                            np.float64(0.0),
                            (r**power) * var("log")(r),
                        )
                        expr = expr + coeff_log * log_term
                    expr = expr + coeff_power * (r**power)
        else:
            max_extracted_n = 2 * max(0, self.split_order - 1)
            for n in range(1, self.series_nmax + 1):
                if n % 2 == 0 and n <= max_extracted_n:
                    continue
                coeff = (1j * k) ** n / (4.0 * np.pi * factorial(n))
                expr = expr + np.complex128(coeff) * (r ** (n - 1))

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return True

    def __getinitargs__(self):
        return (
            self.dim,
            self.wave_number_real,
            self.wave_number_imag,
            self.split_order,
            self.series_nmax,
        )

    def __repr__(self):
        return (
            "HelmholtzSplitSeriesRemainderKernel("
            f"dim={self.dim}, k=({self.wave_number_real}+{self.wave_number_imag}j), "
            f"split_order={self.split_order}, series_nmax={self.series_nmax})"
        )

# vim: filetype=pyopencl:foldmethod=marker

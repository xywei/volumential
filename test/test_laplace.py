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
import pytest
import volumential.nearfield_potential_table as npt
from sumpy.point_calculus import CalculusPatch

LONGRUN = pytest.mark.skipif(not pytest.config.option.longrun,
                             reason="needs --longrun option to run")

# Directly evaluate volume integrals on a calculus patch to
# test the correctness of singular quadrature and formulation

# NOTE: the laplace equation being concerned here is:
#   - Delta u = f
# (notice the negative sign!)

# The correctness of table build is then tested against the direct
# evaluation direct_quad() tested here

dim = 2

patch_order = 3


def make_patch(center, size):
    return CalculusPatch(center=center, h=size, order=patch_order)


def test_patch_laplace():

    rep = 3
    size = 0.05

    import random
    for r in range(rep):
        center_x = random.uniform(-1, 1)
        center_y = random.uniform(-1, 1)
        center = [center_x, center_y]
        patch = make_patch(center, size)

        f_values = 0.25 * (patch.x**2 + patch.y**2)

        lap = patch.laplace(f_values)

        print(center, size, len(patch.x), lap)

        assert (np.max(np.abs(lap - 1)) < 1e-4)


def direct_quad(source_func, target_point):

    knl_func = npt.get_laplace(dim)

    def integrand(x, y):
        return source_func(x, y) * knl_func(target_point[0] - x,
                                            target_point[1] - y)

    import volumential.singular_integral_2d as squad
    integral, error = squad.box_quad(
        func=integrand,
        a=0,
        b=1,
        c=0,
        d=1,
        singular_point=target_point,
        maxiter=1000)

    return integral


def eval_f(patch, source_func=None):
    n = len(patch.x)
    ff = np.zeros(n)

    def const_source(x, y):
        return 1

    if source_func is None:
        source_func = const_source

    for i in range(n):
        ff[i] = direct_quad(source_func, (patch.x[i], patch.y[i]))

    return ff


@LONGRUN
def test_laplace_same_box_on_patch():

    rep = 3
    # Prevent numerical instability when point spacings are too small
    size = patch_order * 0.01

    # inside the box
    import random
    for r in range(rep):
        center_x = random.uniform(size / 2, 1 - size / 2)
        center_y = random.uniform(size / 2, 1 - size / 2)
        center = [center_x, center_y]
        patch = make_patch(center, size)

        f_values = eval_f(patch)

        lap = patch.laplace(f_values)

        print("patch_order", patch_order)
        print(r, center, size, len(patch.x))
        print("x", patch.x)
        print("y", patch.y)
        print("f_vals", f_values)
        print("lap", lap)

        assert (max(abs(lap + 1)) < 0.1)

    # outside of the box

    rep = 3
    for r in range(rep):
        center_x = 0.5
        while center_x <= 1 + size / 2 and center_x >= -size / 2:
            center_x = random.uniform(-1, 2)

        center_y = 0.5
        while center_y <= 1 + size / 2 and center_y >= -size / 2:
            center_y = random.uniform(-1, 2)

        center = [center_x, center_y]
        patch = make_patch(center, size)

        f_values = eval_f(patch)

        lap = patch.laplace(f_values)

        print(r, center, size, len(patch.x))
        print("x", patch.x)
        print("y", patch.y)
        print("f_vals", f_values)
        print("lap", lap)

        assert (max(abs((lap))) < 0.1)

__copyright__ = "Copyright (C) 2017 Xiaoyu Wei"

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

import volumential.nearfield_potential_table as npt
from sumpy.point_calculus import CalculusPatch

dim = 2
patch_order = 6

s = 1.5
epsilon = 0.01
delta_t = 0.05
final_t = delta_t * 1
theta_y = 60.0 / 180.0 * np.pi

b = s / (epsilon ** 2)
c = 1.0 / (epsilon * delta_t)


def make_patch(center, size):
    return CalculusPatch(center=center, h=size, order=patch_order)


def test_patch_cahn_hilliard():

    rep = 3
    size = 0.05

    import random

    for r in range(rep):
        center_x = random.uniform(-1, 1)
        center_y = random.uniform(-1, 1)
        center = [center_x, center_y]
        patch = make_patch(center, size)

        f_values = 0.5 * 1 / (4 * 3 * 2) * (patch.x ** 4 + patch.y ** 4)

        lap = patch.laplace(f_values)
        laplap = patch.laplace(lap)

        # print(center, size, len(patch.x), lap, laplap)

        assert np.max(np.abs(laplap - 1)) < 5e-6


def test_kernel_func():

    # With/without special expansion at origin
    knl_func = npt.get_cahn_hilliard(dim, b, c)
    alternate_knl_func = npt.get_cahn_hilliard(dim, b, c, True)

    for i in range(50):
        ri = 0.001 * 2 ** (-i)
        assert abs(knl_func(ri, 0) - alternate_knl_func(ri, 0)) < 1e-12


def direct_quad(source_func, target_point):

    knl_func = npt.get_cahn_hilliard(dim, b, c)

    def integrand(x, y):
        return source_func(x, y) * knl_func(target_point[0] - x, target_point[1] - y)

    import volumential.singular_integral_2d as squad

    integral, error = squad.box_quad(
        func=integrand, a=0, b=1, c=0, d=1, singular_point=target_point, maxiter=100
    )

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


def test_cahn_hilliard_same_box_on_patch():

    rep = 1
    # Prevent numerical instability when point spacings are too small
    size = 0.5

    # inside the box
    import random

    for r in range(rep):
        center_x = random.uniform(size / 2, 1 - size / 2)
        center_y = random.uniform(size / 2, 1 - size / 2)
        center = [center_x, center_y]
        patch = make_patch(center, size)

        def source_func(x, y):
            return c

        f_values = eval_f(patch, source_func)

        lap = patch.laplace(f_values)
        laplap = patch.laplace(lap)

        # print(c)
        # print(center)
        # print((laplap - b * lap + c * f_values))
        # print((laplap - b * lap + c * f_values - c) / c)

        assert np.max(np.abs(laplap - b * lap + c * f_values - c) / c) < 5e-2


# def test_cahn_hilliard_table():
# assert False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main

        main([__file__])

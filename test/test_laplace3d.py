
__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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

# import volumential.nearfield_potential_table as npt
from sumpy.point_calculus import CalculusPatch

dim = 3
patch_order = 3


def make_patch(center, size):
    return CalculusPatch(center=center, h=size, order=patch_order)


def test_patch_laplace():

    rep = 3
    size = 0.05

    import random

    for r in range(rep):
        center = [random.uniform(-1, 1) for d in range(dim)]
        patch = make_patch(center, size)

        f_values = 1 / 6 * (patch.x ** 2 + patch.y ** 2 + patch.z ** 2)

        lap = patch.laplace(f_values)

        assert np.max(np.abs(lap - 1)) < 1e-4

# Use this script to pre-compute nftables
from __future__ import absolute_import, division, print_function

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

import logging
logger = logging.getLogger(__name__)
verbose = False
if verbose:
    logging.basicConfig(level=logging.INFO)

import numpy as np
import pyopencl as cl
import boxtree as bt  # noqa: F401
import sumpy as sp  # noqa: F401
import volumential as vm  # noqa: F401

from volumential.table_manager import NearFieldInteractionTableManager

ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)

from time import time

dim = 3
last_cvg_order = 35
root_extent = 2.00

table_filename = "nft_laplace3d_4_multilayer.hdf5"

# a list of parameters that are found via iterating.
param_alpha = 0.1
n_levels1 = 15

param_bq_orders = np.array([10 for i in range(20)])

for i in range(4, 5):
    for sl in range(1):

        t0 = time()

        print("")
        print("*********************************")
        print("* Quad order =", i, "   Level", sl)
        print("*********************************")

        diff = 1

        order1 = int(param_bq_orders[i-1])
        last_diff = 10

        alpha = param_alpha

        print("Brick quad order =", order1)
        tm1 = NearFieldInteractionTableManager(table_filename,
                root_extent=root_extent)
        table1, rec_flag = tm1.get_table(dim, "Laplace", q_order=i,
                source_box_level=sl,
                force_recompute=False, compute_method="DrosteSum", queue=queue,
                n_brick_quad_points=order1, adaptive_level=False, alpha=alpha,
                use_symmetry=True, n_levels=n_levels1)

        while diff <= last_diff:

            order2 = max(int(order1 * 1.1), order1 + 5)
            print("Brick quad order =", order2)
            tm2 = NearFieldInteractionTableManager(table_filename,
                    root_extent=root_extent)
            # last 2 extra kwargs are for logging (all kwargs are stored in the hdf5)
            table2, _ = tm2.get_table(dim, "Laplace", q_order=i, source_box_level=sl,
                    force_recompute=True, compute_method="DrosteSum", queue=queue,
                    n_brick_quad_points=order2, adaptive_level=False, alpha=alpha,
                    use_symmetry=True, n_levels=n_levels1)

            last_diff = diff
            diff = np.max(
                    np.abs(table1.data - table2.data)) / np.max(np.abs(table2.data))
            print("diff =", diff)

            tm1 = tm2
            order1 = order2
            table1 = table2

        last_cvg_order = order1

        t1 = time()
        print("Wall time:", t1 - t0)

        import gc
        gc.collect()

__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

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
import time

import pyopencl as cl

from volumential.table_manager import NearFieldInteractionTableManager


logging.basicConfig(format="%(name)s:%(levelname)s: %(message)s")


def bench_table_build(queue):

    dim = 3
    root_table_source_extent = 2
    q_order = 1
    force_recompute = True

    table_filename = "nft.hdf5"

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent
    )

    queue.finish()
    t0 = time.time()
    nftable, _ = tm.get_table(
        dim,
        "Laplace",
        q_order,
        force_recompute=force_recompute,
        compute_method="DrosteSum",
        queue=queue,
        n_brick_quad_points=120,
        adaptive_level=True,
        adaptive_quadrature=True,
        use_symmetry=True,
        alpha=0, n_levels=1,
    )
    queue.finish()
    t1 = time.time()

    print("Table built in:", t1 - t0)


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    bench_table_build(queue)


if __name__ == "__main__":
    main()

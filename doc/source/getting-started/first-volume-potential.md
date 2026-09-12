# A first volume potential

The script below is `examples/laplace2d.py` reduced to its load-bearing
twenty-odd lines. It evaluates

$$
u(\boldsymbol{x}) = \int_{[-1/2,\,1/2]^2}
\frac{-1}{2\pi} \log \lVert \boldsymbol{x} - \boldsymbol{y} \rVert \,
f(\boldsymbol{y}) \, \mathrm{d}\boldsymbol{y}
$$

for a source $f$ manufactured so that the answer is the Gaussian
$u(\boldsymbol{x}) = e^{-\alpha \lVert \boldsymbol{x} \rVert^2}$, which is what
the error print at the end compares against.

Set `PYOPENCL_CTX` before running it. Otherwise `cl.create_some_context()`
stops and asks which device to use at a terminal, and picks one in an
implementation-defined manner anywhere else:

```bash
export PYOPENCL_CTX=portable:0
```

```python
import numpy as np
import pymbolic as pmbl
import pyopencl as cl
import pyopencl.array  # noqa: F401
from boxtree import TreeBuilder
from boxtree.array_context import PyOpenCLArrayContext
from boxtree.traversal import FMMTraversalBuilder
from functools import partial
from pytools.obj_array import new_1d as obj_array_1d
from sumpy.expansion import DefaultExpansionFactory
from sumpy.kernel import LaplaceKernel

import volumential.meshgen as mg
from volumential.expansion_wrangler_fpnd import (
    FPNDExpansionWrangler, FPNDTreeIndependentDataForWrangler)
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.tools import ScalarFieldExpressionEvaluation as Eval
from volumential.volume_fmm import drive_volume_fmm

dim, q_order, n_levels, m_order, alpha = 2, 9, 6, 20, 160
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
actx = PyOpenCLArrayContext(queue)

# 1. The source density, as a symbolic expression evaluated on device.
x, y, exp = pmbl.var("x"), pmbl.var("y"), pmbl.var("exp")
norm2 = x**2 + y**2
source_expr = -(4 * alpha**2 * norm2 - 4 * alpha) * exp(-alpha * norm2)

# 2. A box mesh over [-1/2, 1/2]^2 and its volume quadrature nodes.
mesh = mg.MeshGen2D(q_order, n_levels, -0.5, 0.5, queue=queue)
q_points = np.ascontiguousarray(mesh.get_q_points().T)
q_weights = cl.array.to_device(queue, mesh.get_q_weights())
source_vals = cl.array.to_device(
    queue, Eval(dim, source_expr, [x, y])(queue, q_points))

# 3. A boxtree over those nodes, plus the FMM traversal.
particles = obj_array_1d([actx.from_numpy(q_points[i]) for i in range(dim)])
tree, _ = TreeBuilder(actx)(
    actx, particles=particles, targets=None,
    max_particles_in_box=q_order**dim * 4 - 1,
    kind="adaptive-level-restricted")
trav, _ = FMMTraversalBuilder(actx)(actx, tree)

# 4. The near-field interaction table, built once and cached in SQLite.
tm = NearFieldInteractionTableManager("nft_laplace2d.sqlite",
                                      root_extent=2, queue=queue)
nftable, _ = tm.get_table(dim, "Laplace", q_order, queue=queue)

# 5. The wrangler: sumpy expansions for the far field, the table for the near.
knl = LaplaceKernel(dim)
factory = DefaultExpansionFactory()
tree_indep = FPNDTreeIndependentDataForWrangler(
    ctx,
    partial(factory.get_multipole_expansion_class(knl), knl),
    partial(factory.get_local_expansion_class(knl), knl),
    [knl],
    exclude_self=True)
wrangler = FPNDExpansionWrangler(
    tree_indep=tree_indep, queue=queue, traversal=trav,
    near_field_table=nftable, dtype=np.float64,
    fmm_level_to_order=lambda *args: m_order, quad_order=q_order,
    self_extra_kwargs={
        "target_to_source": np.arange(tree.ntargets, dtype=np.int32)})

# 6. Evaluate.
(pot,) = drive_volume_fmm(trav, wrangler, source_vals * q_weights, source_vals)

exact = np.exp(-alpha * (q_points[0] ** 2 + q_points[1] ** 2))
print("max error =", np.max(np.abs(exact - pot.get())))
```

Run the maintained version, which also carries the plotting and direct-P2P
branches this excerpt drops, and pins an explicit `DuffyBuildConfig` (the
`tanh-sinh-fast` radial rule at regular/radial quadrature orders 50/100) where
the excerpt takes the defaults — that is a tighter near-field table, and the
reason the example's error is smaller than this one's:

```bash
uv run python examples/laplace2d.py
```

## What just happened

Step 2 is the only part that is specific to volume potentials: the source
density is discretized at tensor-product Gauss-Legendre nodes inside each leaf
box, and the particle weights handed to the FMM are `source_vals * q_weights`.
Steps 3 and 5 are an ordinary `boxtree`/`sumpy` FMM, with one substitution —
`FPNDExpansionWrangler` replaces the point-to-point near-field stage with a
table lookup. The reason is the integrand: over the box containing the target
it is genuinely singular and point quadrature does not converge at all, and
over the neighbouring boxes it is finite but near-singular and converges far
too slowly to be useful at `q_order = 9`.

Step 4 is where the cost is. The first run builds the near-field table by
Duffy-transformed quadrature and writes it to `nft_laplace2d.sqlite`; later
runs load it in milliseconds. The table depends on the kernel, the dimension,
`q_order` and the *scale* of the source box — the manager's `root_extent` and
the request's `source_box_level`, from which `source_box_extent` is derived —
but not on the source density, the tree's topology or the target points. So the
cache file is worth keeping and reusing across runs of this example, and is
**not** reusable at a different root extent. See {doc}`../user-guide/table-build-routing` for how a
build is routed and how to tell a cached table's provenance, and
{doc}`../user-guide/nearfield_symmetry` for why the stored table is much
smaller than the number of interactions it serves.

## Faster, for a first look

The example honours `VOLUMENTIAL_EXAMPLE_SMOKE=1`, which drops to
`q_order = 3`, two levels and multipole order 8 and uses a separate cache file.
That is what CI runs; it finishes in seconds and is accurate to about a
percent.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/laplace2d.py
```

## Next

- Other maintained examples: `examples/laplace3d.py`,
  `examples/helmholtz2d.py`, `examples/helmholtz3d.py`,
  `examples/poisson3d.py`, `examples/branched_flow_helmholtz2d.py`, and the
  two `*_split_p_convergence.py` drivers. The Helmholtz ones pick their own
  device instead of reading `PYOPENCL_CTX` — see {doc}`device-selection`.
- Choosing the device the run lands on: {doc}`device-selection`.
- The whole pipeline, stage by stage:
  {doc}`../user-guide/volume-fmm-workflow`.

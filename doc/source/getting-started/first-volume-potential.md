# A first volume potential

Start with the result and work backward into the machinery.

This example evaluates the 2-D Laplace volume potential

$$
\nu(\boldsymbol{x}) =
\int_{[-1/2,\,1/2]^2}
\frac{-1}{2\pi}\log \lVert \boldsymbol{x}-\boldsymbol{y}\rVert
\,f(\boldsymbol{y})\,\mathrm{d}\boldsymbol{y}
$$

for a manufactured source chosen so that the exact answer is the Gaussian

$$
u(\boldsymbol{x}) = e^{-\alpha\lVert\boldsymbol{x}\rVert^2}.
$$

That gives us something unusually useful for a first example: we can run the
whole volume-FMM pipeline and check the answer everywhere.

## Run it first

Pick an OpenCL device and run the maintained smoke configuration:

```bash
export PYOPENCL_CTX=portable:0
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/laplace2d.py
```

The smoke configuration is deliberately small: it is for seeing the path work,
not for measuring performance or quoting an accuracy result. The first run may
still spend most of its time building the near-field table; later runs reuse
the SQLite cache.

```{figure} ../images/volumential-overview.svg
:alt: A source density is sampled on volume nodes, interactions are split into near-field table lookup and far-field FMM, and the resulting potential is checked against an exact manufactured solution.
:align: center

The rest of this page opens that picture one stage at a time.
```

## 1. Define the source

The manufactured source is represented symbolically and evaluated on the
volume quadrature nodes:

```python
x, y, exp = pmbl.var("x"), pmbl.var("y"), pmbl.var("exp")
norm2 = x**2 + y**2
source_expr = -(4 * alpha**2 * norm2 - 4 * alpha) * exp(-alpha * norm2)
```

Nothing FMM-specific has happened yet. This is just the function
$f(\boldsymbol y)$ that will be integrated against the Laplace kernel.

## 2. Put quadrature nodes in boxes

Volumential's mesh generator fills each leaf box with tensor-product
Gauss-Legendre nodes:

```python
mesh = mg.MeshGen2D(q_order, n_levels, -0.5, 0.5, queue=queue)
q_points = np.ascontiguousarray(mesh.get_q_points().T)
q_weights = cl.array.to_device(queue, mesh.get_q_weights())
source_vals = cl.array.to_device(
    queue, Eval(dim, source_expr, [x, y])(queue, q_points))
```

The far field will see those nodes as weighted particles. The near field cannot
simply do the same thing: in the target box the kernel is singular, and in the
neighboring boxes it is near-singular.

## 3. Build the tree and decide what is near

The nodes go into `boxtree`, which builds the adaptive level-restricted tree and
its FMM traversal:

```python
particles = obj_array_1d([actx.from_numpy(q_points[i]) for i in range(dim)])
tree, _ = TreeBuilder(actx)(
    actx,
    particles=particles,
    targets=None,
    max_particles_in_box=q_order**dim * 4 - 1,
    kind="adaptive-level-restricted",
)
trav, _ = FMMTraversalBuilder(actx)(actx, tree)
```

```{figure} ../images/near-far.svg
:alt: The target box and its List 1 neighbors use precomputed near-field tables, while well-separated boxes are handled by the particle FMM.
:align: center

This is the key Volumential idea. **Near** means the target's own box plus its
List 1 neighbors. Those interactions use precomputed integrals. **Far** means
well-separated boxes, and those use an ordinary particle FMM.
```

## 4. Build or load the near-field table

```python
tm = NearFieldInteractionTableManager(
    "nft_laplace2d.sqlite", root_extent=2, queue=queue)
nftable, _ = tm.get_table(dim, "Laplace", q_order, queue=queue)
```

On a cache miss, this is the expensive one-time step: singular and
near-singular box interactions are integrated with desingularizing quadrature.
The result is written to SQLite and reused on later runs.

The table depends on the kernel, dimension, quadrature order and source-box
scale. It does **not** depend on this particular Gaussian source or on the
topology of this particular tree. See
{doc}`../user-guide/table-build-routing` for routing and provenance, and
{doc}`../user-guide/nearfield_symmetry` for why the stored table is much
smaller than the set of interactions it serves.

## 5. Give one wrangler both jobs

The wrangler combines sumpy expansions for the far field with the table for
the near field:

```python
knl = LaplaceKernel(dim)
factory = DefaultExpansionFactory()
tree_indep = FPNDTreeIndependentDataForWrangler(
    ctx,
    partial(factory.get_multipole_expansion_class(knl), knl),
    partial(factory.get_local_expansion_class(knl), knl),
    [knl],
    exclude_self=True,
)
wrangler = FPNDExpansionWrangler(
    tree_indep=tree_indep,
    queue=queue,
    traversal=trav,
    near_field_table=nftable,
    dtype=np.float64,
    fmm_level_to_order=lambda *args: m_order,
    quad_order=q_order,
    self_extra_kwargs={
        "target_to_source": np.arange(tree.ntargets, dtype=np.int32),
    },
)
```

The name `fpnd` is literal: **f**ar field by **p**article approximation,
**n**ear field **d**irect.

## 6. Evaluate and check the answer

```python
(pot,) = drive_volume_fmm(
    trav, wrangler, source_vals * q_weights, source_vals)

exact = np.exp(-alpha * (q_points[0] ** 2 + q_points[1] ** 2))
print("max error =", np.max(np.abs(exact - pot.get())))
```

Notice the two source arrays passed to `drive_volume_fmm`:

- `source_vals * q_weights` is what the particle far field integrates;
- `source_vals` is the bare density contracted against the near-field table.

Passing the same array for both is a quiet but important mistake.

:::{dropdown} Complete minimal driver

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

x, y, exp = pmbl.var("x"), pmbl.var("y"), pmbl.var("exp")
norm2 = x**2 + y**2
source_expr = -(4 * alpha**2 * norm2 - 4 * alpha) * exp(-alpha * norm2)

mesh = mg.MeshGen2D(q_order, n_levels, -0.5, 0.5, queue=queue)
q_points = np.ascontiguousarray(mesh.get_q_points().T)
q_weights = cl.array.to_device(queue, mesh.get_q_weights())
source_vals = cl.array.to_device(
    queue, Eval(dim, source_expr, [x, y])(queue, q_points))

particles = obj_array_1d([actx.from_numpy(q_points[i]) for i in range(dim)])
tree, _ = TreeBuilder(actx)(
    actx, particles=particles, targets=None,
    max_particles_in_box=q_order**dim * 4 - 1,
    kind="adaptive-level-restricted")
trav, _ = FMMTraversalBuilder(actx)(actx, tree)

tm = NearFieldInteractionTableManager(
    "nft_laplace2d.sqlite", root_extent=2, queue=queue)
nftable, _ = tm.get_table(dim, "Laplace", q_order, queue=queue)

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

(pot,) = drive_volume_fmm(
    trav, wrangler, source_vals * q_weights, source_vals)

exact = np.exp(-alpha * (q_points[0] ** 2 + q_points[1] ** 2))
print("max error =", np.max(np.abs(exact - pot.get())))
```

:::

## The maintained example

`examples/laplace2d.py` is the version to copy. It carries the extra
diagnostic branches omitted above and pins an explicit `DuffyBuildConfig`
(the `tanh-sinh-fast` radial rule at regular/radial quadrature orders
50/100), giving a tighter near-field table than the minimal excerpt's defaults.

```bash
uv run python examples/laplace2d.py
```

## Next

- See the problems by output and application: {doc}`../examples/gallery`.
- Choose the OpenCL device deliberately: {doc}`device-selection`.
- Read the full pipeline, including interpolation and alternative wranglers:
  {doc}`../user-guide/volume-fmm-workflow`.
- For 3-D spatial diagnostics, continue to `examples/poisson3d.py`.

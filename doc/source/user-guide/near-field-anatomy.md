# Anatomy of a near-field interaction

The central numerical idea in Volumential is easier to understand as a picture
than as a wrangler class name.

```{figure} ../gallery/near-far-anatomy.svg
:alt: Schematic. A five by five block of boxes. The target box and its eight neighbors are table lookups, the target box's own interaction included; the outer ring is handled by the FMM with multipole and local expansions.
:width: 100%

Schematic, not computed output: a uniform neighborhood. Adaptive traversals
add relationships between levels, but the split is the same: difficult local
integrals are tabulated; well-separated interactions stay on the ordinary FMM
path.
```

## Why the neighboring boxes are special

For a volume potential

$$
u(\boldsymbol{x}) =
\int_\Omega G(\boldsymbol{x}, \boldsymbol{y}) f(\boldsymbol{y})\,
\mathrm{d}\boldsymbol{y},
$$

the target can lie inside the source box. The kernel is then genuinely singular.
In neighboring boxes it is finite but near-singular. The tensor-product point
quadrature that is perfectly useful for well-separated particles is the wrong
tool for those interactions.

Volumential therefore changes *how the local interaction is evaluated*, not the
far-field algorithm.

## What is tabulated

For a fixed kernel, dimension, quadrature order and source-box scale, the
near-field table stores the integral of the kernel against the source basis for
each target node and relative interaction case. Building those entries uses
singular-aware quadrature; applying them later is a lookup/reconstruction
operation.

That is why the table can be reused when the **source density changes**. It is
about the local geometry and kernel, not about the values of $f$ in one run.

## What still goes through the FMM

Well-separated boxes use the volume quadrature nodes as weighted particles.
Multipole and local expansions are ordinary `sumpy`/FMM machinery. The two
paths are accumulated into the same target potential.

```{figure} ../gallery/volume-fmm-workflow.svg
:alt: Schematic flow chart. The source density at the box quadrature nodes goes into a tree with its traversal, then splits into a near-field path, a precomputed table lookup, and a far-field path, ordinary FMM expansions; the two paths add up to the potential.
:width: 100%

Schematic, not computed output: the two paths of the computation.
{doc}`volume-fmm-workflow` names the objects and modules behind each box.
```

(graded-tree-list4)=

## On a graded tree

An adaptive tree is kept 2:1 balanced: leaves that touch differ by at most one
level. That keeps List 1 inside the cases the table stores, neighbours of
half, equal and twice the size of the target box, and the far field needs
nothing more ({mod}`volumential.tree_interactive_build`).

It does let one kind of far-field pair come closer than the others. A box can
be split while a neighbour of the same size stays a leaf, and that leaf then
reaches the children of the split box that do not touch it through List 4. It
is twice the size of such a target box and only half its own size away from
it; a List 2 or List 3 source is never closer than its own size. The far field
integrates it like every other source, by point quadrature at its nodes, and
at half a box size that quadrature is much worse. On 2-D trees graded over two
or three levels, at `q_order` 4 to 8, the error these pairs add is 7 to 2800
times the error of the far pairs one to two source sizes away, and the gap
grows with `q_order`.

In the total error they are a minor term so far. On a 16 x 16 tree refined
twice around a narrow source (535 leaves on levels 4 to 6), at `q_order` 8 they
account for 21% of the max error, which is 2.9e-9, and 2% of the relative L2
error; at `q_order` 4 they do not show. Integrating them from an upsampled
source is [#178](https://github.com/xywei/volumential/issues/178), which has
the measurements.

## Where symmetry enters

The tabulated neighborhood contains many interactions that are equivalent under
box symmetries. Volumential stores canonical cases and reconstructs the others
online instead of storing every geometrically equivalent entry.

Continue with:

- {doc}`nearfield_symmetry` for orbit canonicalization, compact storage and
  reconstruction;
- {doc}`table-build-routing` for how a cache miss chooses a builder and how
  provenance is recorded;
- {doc}`volume-fmm-workflow` for the complete pipeline and the modules that own
  each stage.

This page is deliberately conceptual. Those pages remain the source of truth for
the exact interaction-case encoding, adaptive traversal details and cache
payload.

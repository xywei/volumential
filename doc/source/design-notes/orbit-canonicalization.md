# ORBIT canonicalization

*Implemented in {mod}`volumential.list1_symmetry`,
{mod}`volumential.orbit_arithmetic` and
`volumential.wranglers.arithmetic_orbits`. The storage format and the runtime
reconstruction modes are documented in {doc}`../user-guide/nearfield_symmetry`;
this page is about why the scheme has the shape it has.*

## The redundancy

A near-field interaction table is indexed by
`(source_mode, target_mode, interaction_case)`: one entry per source basis
function, per target node, per relative position of a source box to the target
box in List 1. That index set is large — it grows like
$n_{\rm cases} \times q^{2d}$ — and it is massively redundant, because the
kernel and the box geometry share symmetries. Reflecting a case vector across
an axis, or permuting two axes, maps one entry onto another entry of the same
table up to a sign.

{mod}`volumential.list1_symmetry` discovers which operations are available:
they depend on the dimension and on the symmetry properties of the kernel, so a
scalar Laplace table in 3D admits the full hyperoctahedral group while an
axis-differentiated kernel admits only the subgroup that preserves the
differentiation axis.

## Two ways to canonicalize

**Dense.** Build the full orbit structure once — a map from every full entry id
to its canonical representative, plus a per-entry sign from signed union-find —
and store the representatives. This is straightforward, it is the
construction-time oracle, and it remains the fallback for any symmetry group
the arithmetic path does not cover.

Its cost is the map itself. It is $O(n_{\rm cases} q^{2d})$ integers that have
to be built at construction time and, in the naive online scheme, *shipped to
the device* so the List 1 kernel can dereference it per interaction.

**Arithmetic.** {mod}`volumential.orbit_arithmetic` derives, per interaction
case and from the case vector alone, an axis permutation, an axis sign pattern
and an axis grouping. Given those, the canonical address of an entry is an
integer computation on the mode digits — the same computation on the host and
inside the generated List 1 OpenCL kernel — so no full-entry map exists at
runtime at all. Per case the kernel carries `case_orbit_ranks`
(`uint16[n_cases]`) and the packed `uint8`/`int8` descriptor arrays
`case_axis_perm`, `case_axis_sign`, `case_axis_group`; per entry it carries
nothing.

`build_arithmetic_case_metadata` builds those descriptors, and
`enumerate_scalar_arithmetic_representatives` lists the full-table entry id of
every compact orbit representative, which is what lets a symmetry-reduced Duffy
build fill a compact `table.data` array directly instead of building a dense
buffer and discarding most of it.

## What makes it safe

The arithmetic path is a closed-form claim about a group action, so it is
checked against the oracle rather than trusted. Payload preparation validates
the generated representative ids and the generated signs against
`table_entry_ids` and `table_entry_scales` — the dense construction-time
metadata — before the generated descriptor is used. A disagreement is a build
failure, not a silent fall back to dense.

Derivative kernels are the case where signs matter: the orbit metadata is
sign-aware, and the runtime lookup applies a per-entry sign factor when
reconstructing an entry from its canonical representative. Under the FMM
wrangler path the relevant `symmetry_source_direction` is inferred from the
active directional source vector in `source_extra_kwargs` and applied to the
lookup metadata for that evaluation; see {doc}`../user-guide/derivative_support`.

## What it buys

Two separate wins, often conflated:

- **Storage.** Only canonical entries are built and persisted. Reduced tables
  store representative values in a compact `table.data` array with the
  corresponding full ids in `reduced_entry_ids`, and cache payloads carry
  `reduced_entry_ids` plus `reduced_data` — never a dense
  $n_{\rm cases} \times q^{2d}$ buffer, and never NaN sentinels.
- **Build time.** The expensive singular quadrature runs once per orbit rather
  than once per entry, which is where most of the table build time is saved.

The on-device arithmetic addressing adds a third, smaller win: it removes a
large read-only buffer from the List 1 kernel's working set.

## Where to look

- Storage format, reconstruction modes and GPU scheduling:
  {doc}`../user-guide/nearfield_symmetry`.
- Symmetry discovery: {mod}`volumential.list1_symmetry`.
- Case enumeration: {mod}`volumential.list1_gallery`.
- Arithmetic addressing: {mod}`volumential.orbit_arithmetic`.
- Symmetry-reduction counts as measured evidence (`cache_economics.csv`):
  {doc}`../benchmarks/index`.

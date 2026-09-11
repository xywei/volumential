# User guide

How the volume FMM is put together, and what each part of it costs you.

```{toctree}
:maxdepth: 1

volume-fmm-workflow
nearfield_symmetry
table-build-routing
derivative_support
helmholtz_split
m1_kernels
validation_matrix
```

{doc}`volume-fmm-workflow` is the one to read first: it walks the pipeline from
a box mesh to a potential and says which module owns each stage. The remaining
pages go deep on the parts that are not obvious from the source —
{doc}`nearfield_symmetry` on why the stored table is far smaller than the
number of interactions it serves and what the SQLite cache actually holds;
{doc}`table-build-routing` on how a table build is routed, when it silently
becomes slow, and how to refuse a table whose provenance cannot be verified;
{doc}`derivative_support` on the derivative wrappers and their sign
bookkeeping; {doc}`helmholtz_split` on the near-field split that makes one
table family serve a range of wave numbers.

{doc}`m1_kernels` and {doc}`validation_matrix` record what is supported and
what is actually tested, so a gap shows up as a gap rather than as an absent
test name.

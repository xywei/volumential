# Examples

`examples/` holds eight self-contained programs and three notebooks. Each one
solves a whole problem — build a mesh, build or load a near-field table, run
the volume FMM, report an error — rather than demonstrating a single call, so
the shortest path from {doc}`../getting-started/first-volume-potential` to your
own driver is usually to copy the closest example and change it.

Two things decide what a run costs, and neither is visible in the file name.

**Smoke mode.** Six of the eight scripts drop to a small configuration when
`VOLUMENTIAL_EXAMPLE_SMOKE=1` is set (`branched_flow_helmholtz2d.py` uses its
own `--smoke` flag instead); `laplace3d.py` has no smoke mode at all. A smoke
run is a correctness check, never a measurement — see
{doc}`../benchmarks/index` for what a measurement has to record.

**Near-field tables.** Every script writes its table cache to an SQLite file in
the working directory, named in the table below. The *first* run of a script
builds that table and is dominated by the build; later runs load it in
milliseconds. Deleting the file, or running from a different directory, pays
the build again. Which build path runs, and how long it takes, is the subject
of {doc}`../user-guide/table-build-routing`.

## At a glance

| Script | Small run | Table cache | Device |
| --- | --- | --- | --- |
| [`laplace2d.py`][laplace2d] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_laplace2d[_smoke].sqlite` | `PYOPENCL_CTX` |
| [`laplace3d.py`][laplace3d] | **none** | `nft_laplace3d.sqlite` | `PYOPENCL_CTX` |
| [`poisson3d.py`][poisson3d] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_poisson3d[_smoke].sqlite` | `PYOPENCL_CTX` |
| [`helmholtz2d.py`][helmholtz2d] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_laplace2d_for_helmholtz[_smoke].sqlite` | picks its own |
| [`helmholtz3d.py`][helmholtz3d] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_laplace3d_for_helmholtz[_smoke].sqlite` | picks its own |
| [`helmholtz2d_split_p_convergence.py`][h2dsplit] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_laplace2d_split_p_convergence[_smoke].sqlite` | picks its own |
| [`helmholtz3d_split_p_convergence.py`][h3dsplit] | `VOLUMENTIAL_EXAMPLE_SMOKE=1` | `nft_laplace3d_split_p_convergence[_smoke].sqlite` | `--backend` |
| [`branched_flow_helmholtz2d.py`][branched] | `--smoke` | under `--output-dir` | picks its own |

"Picks its own" means the script builds an OpenCL context directly and
**ignores `PYOPENCL_CTX`**, preferring the first fp64-capable GPU it finds and
falling back to a CPU device. On a host with a GPU that silently changes the
cost class of a run. {doc}`../getting-started/device-selection` lists which
script does what, and how to pin the device you meant.

Only `laplace2d.py`, `helmholtz2d.py` and `helmholtz3d.py` run on pull
requests, under `VOLUMENTIAL_EXAMPLE_SMOKE=1`, in a job with a 30-minute
budget it shares with six benchmark smoke runs. The rest run only in the
`Examples` job of `CI Full`, at full settings, in a 240-minute budget shared by
all eight — so treat "full settings" as *tens of minutes each*, and expect the
3D ones to sit at the expensive end. See {doc}`../development/ci`.

## Laplace

### `laplace2d.py`

The reference example, and the one {doc}`../getting-started/first-volume-potential`
walks through line by line. It evaluates the volume potential of a manufactured
density over $[-1, 1]^2$ with the Laplace kernel, compares it against the exact
potential at the quadrature nodes, and prints the relative error. It also
carries the plotting and direct-particle-to-particle branches the walkthrough
leaves out, and pins an explicit `DuffyBuildConfig`, which is why its error is
smaller than the excerpt's.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/laplace2d.py   # seconds
uv run python examples/laplace2d.py                               # full settings
```

### `laplace3d.py`

The same problem over $[-1, 1]^3$. **It has no smoke mode**: the only way to
run it is at full settings, and the first run builds a 3D near-field table,
which is the expensive part — `CI Full` caches `nft_laplace3d.sqlite` between
runs for exactly that reason. Keep the file.

```bash
uv run python examples/laplace3d.py
```

### `poisson3d.py`

A manufactured 3D Poisson solve over $[-0.5, 0.5]^3$ — two shifted Gaussian
bumps and the forcing $f = -\Delta u$ — that also produces pictures: orthogonal
slice plots and a point-cloud error plot, written as PNGs into
`poisson3d_output/` (override with `VOLUMENTIAL_POISSON3D_OUTPUT_DIR`). Use it
when you want to *see* where the error lives rather than read one number.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/poisson3d.py
```

## Helmholtz

### `helmholtz2d.py`

A q-order convergence study for the 2D Helmholtz volume potential. It is the
smallest complete demonstration of the routing described in
{doc}`../user-guide/table-build-routing`: rather than tabulating the Helmholtz
kernel, it reads a *Laplace* near-field table and adds an online
Helmholtz-minus-Laplace smooth correction for the list-1 interactions. What it
reports is a manufactured-source PDE residual $(-\Delta - k^2)u - \rho$ on an
interior calculus patch, not an error against a closed-form potential.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/helmholtz2d.py
```

### `helmholtz3d.py`

The 3D counterpart, with a complex-valued source over $[-0.5, 0.5]^3$ and the
same PDE-residual diagnostic. Its `run_convergence_study()` is what the
Helmholtz notebook below imports, so the two stay in step.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/helmholtz3d.py
```

### `helmholtz2d_split_p_convergence.py`

A sweep over `helmholtz_split_order` for a manufactured Gaussian solution,
which is the driver behind the claims in {doc}`../user-guide/helmholtz_split`.
It writes a JSON summary, a plot-friendly CSV and, when matplotlib is present,
a PNG. `--input-json` re-renders the CSV and the plot from a previous run
without computing anything, and
`examples/helmholtz2d_split_p_convergence_from_final_case.csv` is a recorded
output of the tuned case to compare against.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python \
    examples/helmholtz2d_split_p_convergence.py
```

### `helmholtz3d_split_p_convergence.py`

The 3D version of the sweep, with `examples/helmholtz3d_split_p_convergence_from_q5n3.csv`
as its recorded output. It is the one example with an explicit `--backend`
flag, and its default is mode-dependent in a way worth knowing before you time
it: `auto` under smoke, but **`pocl-cpu` at full settings**. A full run
therefore lands on the CPU even on a host with a GPU unless you pass
`--backend cuda-gpu`.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python \
    examples/helmholtz3d_split_p_convergence.py
```

### `branched_flow_helmholtz2d.py`

:::{warning}
The default configuration of this script is a publication-scale pilot, not an
example run. Give it `--smoke` unless you are on a machine set aside for it.
:::

Free-space 2D Helmholtz branched flow through a smooth random medium, solved as
a constant-background Lippmann-Schwinger equation with GMRES — no artificial
outer boundary, no BIE, no PML. It is the largest program in the tree and the
only one that exercises the optional fixed right preconditioners, including the
Lagrange palette of fixed-wave-number tables. `--smoke` uses the `sumpy`
far-field backend and needs nothing extra; the default configuration uses
`fmmlib`, so it needs the `fmmlib` extra installed (`uv sync --extra test
--extra doc --extra fmmlib`) and a `pyfmmlib` built as `DEVELOPMENT.md`
describes. Output lands under `--output-dir`, `build/branched-flow-helmholtz2d`
by default.

```bash
uv run python examples/branched_flow_helmholtz2d.py --smoke
```

[laplace2d]: https://github.com/xywei/volumential/blob/main/examples/laplace2d.py
[laplace3d]: https://github.com/xywei/volumential/blob/main/examples/laplace3d.py
[poisson3d]: https://github.com/xywei/volumential/blob/main/examples/poisson3d.py
[helmholtz2d]: https://github.com/xywei/volumential/blob/main/examples/helmholtz2d.py
[helmholtz3d]: https://github.com/xywei/volumential/blob/main/examples/helmholtz3d.py
[h2dsplit]: https://github.com/xywei/volumential/blob/main/examples/helmholtz2d_split_p_convergence.py
[h3dsplit]: https://github.com/xywei/volumential/blob/main/examples/helmholtz3d_split_p_convergence.py
[branched]: https://github.com/xywei/volumential/blob/main/examples/branched_flow_helmholtz2d.py

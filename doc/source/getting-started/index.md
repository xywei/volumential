# Getting started

Three pages, in order: provision an environment that produces trustworthy
numbers, evaluate a volume potential, and make sure the computation ran on the
device you intended.

```{toctree}
:maxdepth: 1

installation
first-volume-potential
device-selection
```

Volumential needs a working OpenCL runtime. Everything below assumes one is
installed and that `python -c "import pyopencl"` succeeds; if it does not,
start at {doc}`installation`.

Before running anything, export `PYOPENCL_CTX`:

```bash
export PYOPENCL_CTX=portable:0   # the PoCL platform, device 0
```

Without it, `pyopencl.create_some_context()` — which `laplace2d.py`,
`laplace3d.py` and `poisson3d.py` call — decides for you: at a terminal it
stops and asks, and anywhere else (a batch job, a pipeline, CI) it picks a
device in an implementation-defined manner. The first wastes your time, the
second quietly makes the run unreproducible.

The variable is not universal. `helmholtz2d.py` and `helmholtz3d.py` enumerate
the platforms themselves and build a `cl.Context` directly, so `PYOPENCL_CTX`
does not reach them; both prefer an fp64 GPU where there is one.
`branched_flow_helmholtz2d.py` does the same only when the variable is unset.
Check the device the run reports rather than assuming the variable settled it.
{doc}`device-selection` covers the rest; this one line is enough to get through
{doc}`first-volume-potential`.

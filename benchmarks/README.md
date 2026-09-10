# Benchmarks

These scripts emit reproducible CSV artifacts for manuscript evidence. Keep smoke modes lightweight enough for CI/local checks and reserve full sweeps for a suitable, currently idle remote compute host. Keep infrastructure-identifying metadata private and redact it from public artifacts.

## Performance Suite Driver

Use the suite driver to run the maintained benchmark set with a shared output
layout and JSON command manifest:

```bash
python benchmarks/performance_suite.py --mode smoke --out-dir build/benchmarks/performance-suite
```

For full runs, execute on a controlled remote compute host and wrap the suite
with the paper repository metadata tool before promoting results:

```bash
python benchmarks/performance_suite.py --mode full --out-dir /path/to/raw-runs/performance-suite
```

The suite currently covers canonical table equivalence/cache economics,
accuracy preservation, split-parameter coverage, and adaptive timing. Use
`--case <name>` to run a subset, `--dry-run` to emit the command manifest
without executing benchmarks, `--backend <auto|pocl-cpu|cuda-gpu>` to preserve
device selection for benchmark scripts that expose a backend option, and
`--list-cases` to inspect the registered cases without importing
OpenCL-dependent benchmark modules. The manifest is written even when a case
fails, so partial controlled runs remain auditable. The suite passes each case
an output-local cache directory under `<out-dir>/<case>/cache`, keeping table
caches with the promoted run artifacts instead of relying on script defaults.

## Table Equivalence And Cache Economics

```bash
python benchmarks/table_equivalence_cache.py --mode smoke --out-dir build/benchmarks/table-equivalence-cache
```

Full paper runs should be wrapped from the paper repository so hardware, OpenCL, package, commit, parameter, and result-file metadata are captured:

```bash
python /path/to/boxcode-paper/tools/run_benchmark_with_metadata.py \
  --out /path/to/raw-runs/table-equivalence-cache-metadata.json \
  --case-id table-equivalence-cache \
  --mode full \
  --repo volumential=/path/to/volumential \
  --param kernel=laplace \
  --param dimension=3 \
  --param cache_state=cold-and-warm \
  --result-file table_equivalence.csv \
  --result-file cache_economics.csv \
  --cwd /path/to/volumential \
  -- python benchmarks/table_equivalence_cache.py --mode full --out-dir /path/to/raw-runs
```

The script writes:

- `table_equivalence.csv`: max absolute/relative mismatch between direct per-level tables and a scaled canonical level-0 table.
- `cache_economics.csv`: cold build, warm load, payload bytes, cache bytes, and symmetry-reduction counts from the table manager and table diagnostics.

## Accuracy Preservation

```bash
python benchmarks/accuracy_preservation.py --mode smoke --out build/benchmarks/accuracy-preservation.csv
```

The benchmark compares canonical rescaled tables, direct per-level tables, and direct evaluation on a smooth manufactured 3D Poisson problem. Full runs should be wrapped with the paper repository metadata tool before their CSVs are promoted to manuscript data.

## Helmholtz/Yukawa Split Parameter Sweep

```bash
python benchmarks/split_parameter_sweep.py --mode smoke --out build/benchmarks/split-parameter-sweep.csv
```

The benchmark sweeps scalar Helmholtz wave numbers and Yukawa screening parameters in the dimension selected by `--dim` (2, the default, or 3). Each row compares the full implemented split evaluator against a direct fixed-parameter near-field table at the same parameter and application level. Full mode prevents the split-order trend from being limited by quadrature noise: at `q=4`, Yukawa direct references use regular/radial Duffy orders 80/320, channel tables use 48/160, and retained orders above one use smooth-remainder order `2q`. The effective orders are recorded in every row, and full runs reject a nonconvergent Yukawa `p=1,2,3` sweep. The benchmark separately records direct-table and RKE-channel setup/load costs, payload, repeated full applications, isolated coefficient and residual diagnostics, cold/warm strategy totals, and a linear break-even model. `--direct-levels` controls the levels provisioned by the direct setup strategy, while `--nlevels` is the application level; `--repeat-count` is the number of applications per parameter, and `break_even_repeat_count` uses the same per-parameter unit.

The sweep additionally runs a windowed-assembled table-provisioning strategy (rows tagged `table_strategy=windowed_assembled`): for each `--windowed-thetas` value (a theta at the application level, up to the `--window-theta` declaration, default 16), a fixed-parameter table is offline-assembled from the windowed channel family, registered through the standard table manager (`register_external_table`), reloaded through the ordinary cache path, and applied through the identical evaluator against the same direct fixed-parameter reference. Rows carry the certificate status (`ok`/`refused`/`failed`), condition number, per-parameter assembly/registration/load costs, and a polynomial-completion certificate probe per theta (`--classical-probe`: cheap truncation-only in smoke, full assembly in full mode) so one CSV holds the evaluator-level windowed + polynomial + direct comparison. The driver fails on any `failed` row, on a refusal inside the declaration, and on small-theta disagreement with the direct reference.

`--dim 3` runs the same accounting on `MeshGen3D` geometry over `[-0.5, 0.5]^3` with the 3D Helmholtz `exp(i k r)/(4 pi r)` and Yukawa `exp(-lambda r)/(4 pi r)` kernels. It uses the conventions the other Paper 1 3D drivers already exercise: production source order `q = 3`, table root extent 2, the loose `16/45` and tight `24/61` direct Duffy policies of the committed 3D production dispatch of `windowed_rke_sweep.py` (`--direct-policies '16,45;24,61'`; that script's own default pair is the 2D one, `24,61;48,160`) with loose in smoke and tight in full, the `12/35` auto-built channel policy of `rke_field_demo_3d.py`, and the windowed channel orders `20/61` that `rke_table_assembly._resolve_channel_orders` declares for three dimensions (2D keeps `48/61`). Case ids and cache files are dimension-tagged, every row carries `dim`, and the full-mode Yukawa split-order gate is one order of magnitude in 3D against three in 2D, the range the committed 3D field demo measures.

`--fmm-order-rule resolved` prescribes the surrounding FMM expansion order per windowed row from that row's own Helmholtz wave number,

```
p(k) = max(--fmm-order, ceil(k*a + 3*ln(k*a + pi))),   a = sqrt(dim)/4,
```

with `a` the half-diagonal of a level-1 box of the unit root box (`sqrt(2)/4` in 2D, `sqrt(3)/4` in 3D). Yukawa rows stay at the floor, the screened kernel being non-oscillatory. The floor is a floor and never a ceiling, so no row is less resolved than a pinned-order run of the same configuration, and because the prescription happens inside one invocation the run keeps a single cold cache and its cost columns stay quotable. `--max-fmm-order` caps the prescription: a row whose prescribed order exceeds the cap keeps its (evaluator-independent) certificate columns, records `far_field_status = refused_order_cap` and is not solved, rather than reporting agreement between two unresolved far fields. Every solved Helmholtz row records `implied_reference_norm = linf_error / rel_l2_error`; the driver writes the CSV and then exits 2, printing `FAR-FIELD-UNRESOLVED`, if any of them leaves the O(1) band, if either error column is non-finite, or if both are exactly zero (the two ends of the diverged-far-field signature). `--min-targets` fails before any device is selected or any geometry is built, from the pure node count `q**dim * 2**((nlevels-1)*dim)`, if the geometry is smaller than the configuration requires.

A cap-refused row is a measurement, not an error: it carries `windowed_status = ok`, its certificate, condition-number and probe columns, its prescribed order in `fmm_order`, `far_field_status = refused_order_cap`, and the cap message in `windowed_refusal` (that column therefore holds either a certificate refusal or a far-field refusal; `far_field_status` and `windowed_status` distinguish them). Solve, error and timing columns are blank. Every gate the driver applies at the end of a run --- the Yukawa split-order gate, the windowed `failed`/in-declaration-refusal/small-theta gates, and the far-field resolution check --- reports **after** the CSV has been written, so a multi-hour run never trades its measurements for a verdict; the first three still exit through the exception (`GATE-FAILED` on stdout), the last returns 2.

## Adaptive Timing

```bash
python benchmarks/adaptive_timing.py --mode smoke --out build/benchmarks/adaptive-timing.csv
```

The benchmark runs 2D Laplace evaluations on deterministically adapted meshes and writes one cold-cache and one warm-cache row per case. Rows report mesh/adaptation setup, geometry construction, table build or load, FMM wall time, and the timing categories exposed by `drive_volume_fmm`. Full paper runs should be wrapped with the paper repository metadata tool before their CSVs are promoted to manuscript data.

## Paper 1 Mechanism And Application Drivers

The following evidence drivers have structured outputs or campaign-specific
acceptance gates and are intentionally not part of `performance_suite.py`:

`adaptive_split_composition.py` compares direct and RKE setup using total table-manager build time and serialized cache payload bytes on both paths. It reports the RKE base and channel payloads separately and in total so storage comparisons include every required table. Three composition extensions sit behind flags with the committed defaults unchanged (the committed CSV column prefix is preserved byte-position-identically; extension columns are appended): `--kernels Yukawa Helmholtz` adds Helmholtz composition rows on the same graded trees, reading the shared `--parameters` list as wave numbers `k` (case ids `helmholtz2d-...-k<param>-...`); `--quadrature-policy {auto,default,high-accuracy}` pins the table quadrature policy explicitly (`auto`, the default, keeps the historical behavior: default policy in smoke, the split-parameter sweep's high-accuracy Yukawa policy in full), and full high-accuracy runs reject a nonconvergent Yukawa split-order trend with the sweep's calibration (`p=2` must improve on `p=1` by three orders of magnitude, no consecutive-order degradation), so a `--split-orders 1 2 3` re-run demonstrably lifts the committed default-quadrature floor; `--include-windowed` appends one windowed-assembled composition row per kernel and parameter (`table_strategy=windowed_assembled`): the parameter-independent windowed channel family is built per populated source level, a fixed-parameter table is offline-assembled per level, registered through the standard table manager (`register_external_table`), reloaded through the ordinary cache path, and run through the identical graded-tree evaluator against the same direct per-level reference, with the `ok`/`refused`/`failed` taxonomy, per-level theta and smooth-order records, and the sweep's small-theta agreement gate.

```bash
python benchmarks/adaptive_timing_3d.py --mode smoke
python benchmarks/graded_tree_convergence.py --mode smoke
python benchmarks/rke_field_demo_3d.py --mode smoke --force-recompute
python benchmarks/complex_channel_closure.py --mode smoke
python benchmarks/complex_bessel_parameterized.py --mode smoke
python benchmarks/derivative_log_preservation.py --mode smoke
python benchmarks/adaptive_split_composition.py --mode smoke
python benchmarks/break_even_validation.py --mode smoke
python benchmarks/keller_segel_continuation.py --mode smoke
```

The Keller--Segel driver supports `--strategy windowed`: per mass factor it
runs a direct-strategy baseline at its quantized lambda ladder (resolved
regime, `theta <= --theta-max`) and a windowed continuation whose
screened-Yukawa state advances through per-step windowed offline-assembled
tables at the exact unquantized `lambda = 1/sqrt(dt)`, with no `theta` step
floor beyond the declared certificate `theta <= --window-theta` (default 16)
and the advective CFL cap kept. Both runs land on shared
`--checkpoint-fractions` of `t_end`; `ks_windowed_checkpoints.csv` records the
weighted relative L2 trajectory agreement there, the summary reports the
per-step provisioning costs, the achieved dt/theta ranges, the
refused/skipped/failed taxonomy, and a binding-constraint histogram with an
explicit go/no-go verdict naming which constraint (CFL, `dt` cap, checkpoint
landing, or theta floor) actually bound the step size.

`graded_tree_convergence.py` (E9) measures manufactured-solution continuum
convergence, error against degrees of freedom, on genuinely graded 2:1 trees
--- the committed continuum-accuracy studies are all uniform-tree. A tight
off-center Gaussian (`--source-alpha`, `--source-center`) drives
source-adapted refinement through `refine_and_coarsen_tree_of_boxes` plus a
colleague-preserving 2:1 balance closure (MeshGen's public update path would
turn these compact cases uniform), refining leaves whose local
interpolation-error proxy `|f(center)| * h**q_order` is within
`--adapt-fraction` of the maximum. The 3D Laplace potential runs through the
canonical rescaled-table path against the closed-form analytic Gaussian
potential; the Gaussian mass omitted outside the box is gated at `1e-10` so
the modeling gap cannot masquerade as continuum error. Two ladders share one
table, FMM order, and table quadrature: uniform over `--uniform-nlevels` and
adaptive over `--adapt-steps` from `--base-nlevels` (full defaults: `q=3`,
uniform `3,4,5`, adaptive steps `0..6`). Every row records the leaf-level
histogram, 2:1 balance, cross-level List 1 fractions, and a `grading_status`;
the driver fails hard if any adaptive rung is not genuinely graded or if a
ladder does not converge. Observed orders in DOF between consecutive rungs,
the per-ladder asymptotic-regime verdict (two consecutive rungs agreeing
within 25%, or an explicit `limitation: ... extend the ladder` statement),
and the matched-error DOF advantage of the adaptive ladder land in the CSV
and the JSON metadata sidecar (`<out stem>-metadata.json`). If the full-mode
verdict states a limitation, extend with `--uniform-nlevels 3,4,5,6` and/or
more adaptive steps rather than requoting a pre-asymptotic order.

The break-even driver selects its direct-baseline provisioning policy with
`--direct-provisioning {eager,lazy}`. `eager` (default, the committed-artifact
policy) builds every anticipated level per parameter; `lazy` builds only the
leaf level the priced workload touches, which on the uniform benchmark tree
owns all List 1 work. Both policies produce identical answers --- the strategy
changes provisioning, not results --- so the pair isolates how much of the
measured cold-build advantage belongs to the mechanism rather than to eager
defaults. Either policy also emits `ops_*` operation-count columns (entries
built, singular and smooth node evaluations, special-function evaluations by
function, recombination flops, near-field point pairs per solve), computed
after every timed phase from the executed node builders and degeneracy
predicates in `volumential/opcounters.py` rather than from constants, so the
counts confirm or refute the analytic cost model in situ. Run the two policies
as separate metadata-wrapped invocations with distinct cache directories.

The windowed sweep accepts `--complex-phases` to add damped complex-frequency
rows at `zeta = Theta^2 exp(i pi f)` for each requested fraction `f` in
`(0, 1)` (the bare flag defaults to `0.25,0.5,0.75`; the endpoints are the real
Yukawa and Helmholtz rays the sweep already covers). Damped rows use the same
real channel family and carry every real-row certificate and deviation column
plus `zeta_phase_fraction`, `zeta_real`, and `zeta_imag`; the direct reference
applies the selected-branch kernel's real and imaginary parts separately at
both Duffy policies, so the reference-floor semantics are unchanged. The
polynomial-completion assembler has no complex path and is recorded as
`skipped` on these rows, distinct from a certificate `refused`.

The complex Bessel driver additionally requires the benchmark extra:
`python -m pip install -e ".[benchmark]"`.

The 3D field driver's full mode always clears its direct and RKE caches, uses
separate higher-order direct-reference and channel-table quadrature policies,
raises the smooth-remainder order above retained order one, and rejects a
nonconvergent `p=1,2,3` path comparison. Its CSV records the effective orders,
and its emitted JSON uses infrastructure-sanitized paths and host labels. Wrap
full runs with the paper metadata tool to retain the private raw environment
record separately.

## DMK Effective-Density Diagnostic

```bash
python benchmarks/dmk_effective_density.py --mode smoke --slice-size 8
```

The benchmark isolates a controlled 3D Laplace DMK far-plus-residual effective
density model. The far density is the analytic Gaussian-filtered source with
multiplier `exp(-sigma_l^2 |k|^2 / 4)` and smoothing standard deviation
`sigma_l / sqrt(2)`, where `sigma_l = box_side_length / sqrt(log(1/epsilon))`.
The residual uses the all-space fourth-order Taylor/asymptotic model for
`erfc(r/sigma_l)/(4*pi*r)`, converts `u_R ~= c0 rho + c1 Delta rho + c2 Delta^2 rho`
to an equivalent density by `rho_R_eff = -Delta u_R`, and applies Volumential's
singular Laplace path to the total density. CSV/JSON/NPZ diagnostics report the
Volumential error against the analytic asymptotic split and the residual split
bias against the unsmoothed Gaussian reference. Smoke mode uses a lightweight
uniform `q=4`, `nlevels=2`, root-half-width `0.5` mesh; full mode uses the
resolved `q=4`, `nlevels=4`, root-half-width `1.0` configuration.

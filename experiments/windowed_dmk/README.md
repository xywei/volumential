# Windowed RKE / DMK unification experiments

Exploratory scripts for the windowed RKE / DMK unification study (companion to
the DMK-inspired table-build experiment of #104). One script per experiment
of the campaign brief. Nothing here is part of Volumential's API or test suite,
and nothing here is benchmark-grade: no warm-up, no repetition statistics, no
claim that any quadrature rule used is optimal. The numbers are evidence for
write-ups kept alongside the manuscripts, not for a manuscript table.

The derivations and the findings write-ups live in the private `boxcode-paper`
manuscript repository and are not reproduced here.

**Lint status.** `pyproject.toml` excludes `experiments/` from `interrogate` and
`basedpyright`, but both of those are defensive: CI scopes them to
`volumential/`.  The gate that does reach this directory is
`ruff check --select E9,F63,F7,F82`, which CI runs from the repository root with
no path argument, and `ruff.toml` does not exclude `experiments/`.  These files
byte-compile cleanly and carry no undefined names, but they have not been run
through ruff itself; do that before the branch goes to CI, or add
`experiments` to `extend-exclude` in `ruff.toml` if the directory is meant to be
outside the linted surface.

## Experiment A: identity and normalization

**Purpose.** Check the brief's derivations D1, D2, D3, D4 and D7 and Appendix
E's Lemma E.1 germ constants against closed forms, high-precision references,
and Volumential's own Duffy-built windowed channel entries. The highest-value
item is check 3: the D7 separability proposition, tested against production
channel entries rather than against a second implementation of the same idea.

| Script | Checks | Needs |
|---|---|---|
| `experiment_a_identity.py` | 1 (`chi_0` is the Ewald short-range kernel), 2 (adjacent-window slab is DMK's `D_l`), 5 (Lemma E.1 germ constants, `m <= 5`) | numpy, scipy, mpmath; Volumential optional |
| `experiment_a_separability.py` | 3 (D7 separability against Duffy-built channel entries) | numpy, scipy, **Volumential** |
| `experiment_a_yukawa_window.py` | 4 (D3 heat-time window vs Fourier-Gaussian multiplier; D4 closed form and its float64 stability) | numpy, scipy; mpmath optional |
| `experiment_a_plots.py` | none (renders the PNGs from the CSVs) | numpy, matplotlib only |

**Invocation.**

```bash
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_a_identity.py      --out <dir>
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_a_separability.py  --out <dir> \
    --dims 2,3 --duffy-orders 12,20,32,48,72 --duffy-radial-orders 31,61,91,121
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_a_yukawa_window.py --out <dir>
python experiments/windowed_dmk/experiment_a_plots.py --out <dir>
```

The two sweep flags above both default to empty, so a run without them writes
neither `experiment_a_check3_duffy_order_sweep.csv` nor the `duffy_order_sweep`
block of the check-3 JSON summary; the command shown is the one the 2026-09-17
campaign run used.

`experiment_a_separability.py` also takes `--q-order-2d`, `--q-order-3d`,
`--channels` and `--theta`.  The defaults are `Theta = 16`, `p* = 6` channels,
`q = 3` in both dimensions, root extent 2.  `q = 3` is Paper 1's committed
*three*-dimensional source order; its committed two-dimensional order is
`q = 4`, so the 2D default is a deliberate reduction, taken because the Duffy
build dominates the run time and grows as `q**(2d)`.  Report it as a reduction
wherever the 2D numbers are quoted.  It compares against
`volumential.rke_table_assembly._duffy_channel_entry_values_{2d,3d}` evaluated
on a *narrowed* Duffy context, so only the handful of `(case, target)` groups of
interest are built; the comparison therefore uses the production node sets,
basis evaluation and normalization (the stored `psi_m = chi_m / t_w**m`, not
Appendix E's unrelated polynomial-completion `psi_m`) rather than a
reimplementation of them. `--duffy-orders` refines the module's own angular
order so a disagreement can be attributed to one side or the other.

`experiment_a_plots.py` exists because the Volumential environment that check 3
needs has SciPy but no matplotlib; it reads only the CSVs and recomputes
nothing, so it can be run afterwards with any interpreter that does have
matplotlib. The check scripts skip their own plotting gracefully.

**Outputs** (all written into `--out`):

| File | Contents |
|---|---|
| `experiment_a_check1_chi0.csv` | `chi_0` from Volumential, the `erfc` closed form and DMK's `sigma` form against a 40-digit reference |
| `experiment_a_check2_shell.csv` | adjacent-window difference vs DMK's `D_l` and vs the fixed scaled profile, each against a cancellation-free 60-digit `erfc`-difference reference (the two float64 forms lose relative accuracy at opposite ends of the range, so a form-vs-form deviation is not a test of the identity) |
| `experiment_a_check5_germs.csv` | `chi_m - c_m psi_m` against the explicit entire completion, `m <= 5`, both dimensions |
| `experiment_a_identity_summary.json` | headline numbers for checks 1, 2, 5 |
| `experiment_a_check3_entries.csv` | per-entry Duffy vs separable `u`-integral values |
| `experiment_a_check3_v_convergence.csv` | self-convergence of the graded `v`-rule: nodes vs relative error |
| `experiment_a_check3_duffy_order_sweep.csv` | Duffy entries vs the separable value as the module's angular order is refined |
| `experiment_a_check3_summary.json` | agreement digits, node counts for 1e-13, target geometry |
| `experiment_a_check4_yukawa.csv` | both splits at `r = h, 2h, 3h` over `theta/Theta`, with the `(e^{zeta T} - 1) K_zeta` tail |
| `experiment_a_check4_d4_stability.csv` | naive vs `erfcx` D4 evaluation past the declaration |
| `experiment_a_check4_summary.json` | headline numbers for check 4 |
| `*.png` | the six plots written by `experiment_a_plots.py` |

**Caveat.** The `v`-quadrature node counts are for the specific graded rule in
`v_quadrature` (dyadic grading plus panel breaks at the target-to-face
transition scales) and are an upper bound, not a minimum: no attempt was made to
optimize the rule.

Findings and verdicts:
the experiment A write-up in the private manuscript repository.

## Experiment B: parameterized shell test

**Purpose.** Build the adjacent-window slab (scale shell)

```
D_l(r) = chi_0(r; t_l) - chi_0(r; t_{l+1})
```

on a uniform dyadic 3D tree with `h_l = 2**-l`, `t_l = (h_l/Theta)**2`, for
Yukawa with `mu` set so that `theta_L = Theta` at the finest level `L = 6`, and
compare three constructions: the `chi_m` channel series truncated at
`p* = 4, 6, 8, 12`; the closed form D4 of the brief (naive `erfc` form, stable
`erfcx` form, and the cancellation-free `u`-integral); and numerical inversion
of the D3 Fourier multiplier. Then count the per-coordinate Fourier modes a
tensor trapezoidal rule needs on the colleague range at `1e-6`, `1e-9`,
`1e-12`, under a mode-count definition stated in the script docstring. The
point of the experiment is the regime split of brief D5 and the cost rule of
D6, with numbers.

**Invocation.**

```bash
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_b_shells.py --out <dir>
# add --quick for a smoke run with a shorter Fourier sweep
# redraw the PNGs from the CSVs with an interpreter that does have matplotlib:
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_b_shells.py --out <dir> --plots-only
```

Dependencies: numpy, scipy, mpmath; matplotlib optional (plots are skipped
gracefully when it is missing). The 2026-09-17 campaign run used an
interpreter without matplotlib, so the three PNGs were produced by a second
`--plots-only` pass; that pass recomputes nothing and reads only the CSVs, so
the JSON summary of the compute pass records `"matplotlib": null` and an empty
`"plots"` list.

**Outputs** (all written into `--out`):

| File | Contents |
|---|---|
| `experiment_b_closed_form.csv` | naive vs stable D4, `u`-integral, window difference and Fourier inversion against an mpmath reference, per level and radius |
| `experiment_b_series.csv` | `chi_m` series truncation error versus `p*`, per level and radius |
| `experiment_b_conditioning.csv` | companion one-parameter sweep in `zeta t_l`: digits lost, shell amplitude, double-precision summation error |
| `experiment_b_fourier_modes.csv` | `n_f`, `N_1 = 2 n_f + 1`, band limit, aliasing admissibility, per `Theta`, level, period factor `nu` and tolerance |
| `experiment_b_regime_split.csv` | the D5 regime classification per level |
| `experiment_b_summary.json` | every headline number, environment versions, configuration |
| `experiment_b_series_truncation.png` | truncation error versus `p*` per level |
| `experiment_b_fourier_modes.png` | `N_1` for the Laplace shell versus `Theta` and tolerance |
| `experiment_b_conditioning.png` | digits lost versus `zeta t_l`, against the D5 prediction and the shell's own decay |

**Caveat.** `N_1` depends on the periodization convention (`nu`), so the
script also reports the convention-independent band limit `K h_l`. The numbers
are *not* a reproduction of DMK Table 3.1; that table uses a PSWF window and a
different periodization rule, and is quoted in the JSON summary for context
only.

## Experiment C: leaf residual replacement

Purpose: in 2D Laplace, replace only the finest local residual of a DMK-style
telescoping hierarchy by the windowed RKE prefix `W_{t_L}(r) = E_1(r^2/(4 t_L))
/ (4 pi)` evaluated on the *physical* leaf geometry, and compare it against the
legacy DMK-style leaf treatment (the asymptotic Laplacian series `sum_j
t_L^{j+1}/(j+1)! (Delta^j rho)(x)`, equivalently the polynomial density
integrated over the whole box) as a target approaches a cut. Three levels over
`[-1, 1]^2`, `Theta = 8`, piecewise tensor-polynomial densities of degree at
most 3, four leaf-geometry cases (interior full box, half-plane cut,
right-angle wedge, 60-degree wedge) and ten targets per case. This also gives
the 2D analogue of the D7 separability check: the windowed-prefix entry over a
rectangle (optionally clipped by axis-aligned cuts) is computed both by the
separable one-dimensional `u`-integral of erf moments and by target-centred
polar quadrature.

References are target-centred polar (Duffy) quadrature over the fan of
triangles with the target as apex, with the `r^k log r` moments in closed form,
so only the angular variable is quadrature; the script reports the
self-convergence of that rule (about `1e-16` relative here).

| Script | Needs |
|---|---|
| `experiment_c_leaf_residual.py` | numpy, scipy |
| `experiment_c_plots.py` | matplotlib (run separately if the numerics interpreter lacks it) |

Invocation (from the repository root, with `PYTHONPATH` set to it):

```bash
python experiments/windowed_dmk/experiment_c_leaf_residual.py --out OUT --targets 10
python experiments/windowed_dmk/experiment_c_plots.py --out OUT
```

Flags: `--targets` (targets per case), `--n-ang` / `--n-rad` (polar quadrature
orders), `--polar-every` (how often the separable prefix is cross-checked
against the polar prefix), `--no-refine` (skip the quadrature self-check).
The full run takes about half a minute on one core.

Outputs:

- `experiment_c_kernel_split.csv` (telescoping identity at the kernel level)
- `experiment_c_case1_interior_global.csv`,
  `experiment_c_case1_interior_piecewise.csv`,
  `experiment_c_case2_halfplane.csv`, `experiment_c_case3_wedge90.csv`,
  `experiment_c_case4_wedge60.csv`
- `experiment_c_smooth_quadrature.csv` (tensor-Gauss order convergence of each
  split piece over the target's own leaf)
- `experiment_c_uquad_nodes.csv` (`v`-node count of the separable prefix)
- `experiment_c_summary.json`
- `experiment_c_boundary_sweep.png`, `experiment_c_smooth_convergence.png`,
  `experiment_c_leaf_columns.png`

Findings and verdicts:
the experiment C write-up in the private manuscript repository.

## Experiment E: baseline (asymptotic local part and Nystrom solver)

`experiment_e_baseline.py` carries the two reference implementations that the
two-dimensional assembly on complex geometry is measured against.  It imports
its vocabulary from `experiment_e_common.py` and adds nothing to it.

Part (a), `fryklund_VL(target, closest_point_b, kappa_b, rho_jet, delta)`, is
the asymptotic local volume potential of Fryklund, Greengard, Jiang and Potter
(2024), Lemma 4.5, through order `delta^2` with an `O(delta^{5/2})` remainder.
It is the same quantity as the windowed leaf closure with `delta = t_L`, namely
`(4 pi)^{-1} int_Omega E_1(|x - y|^2 / (4 delta)) f(y) dy`.  The local frame puts
the target at the origin with `xi` along the tangent at the closest boundary
point `b` and `eta` along the inward normal; `kappa_b` is positive where the
domain is locally convex (`+1/R` inside a disk of radius `R`); `rho_jet` holds
the partial derivatives of the density at the target in that frame.  Exterior
targets (`sign=-1`), which the published lemma does not cover, are evaluated
through the exact complement identity documented in the module docstring.

One transcription correction is recorded in `LEMMA_4_5_NOTES` and reproduced by
a test: the `delta^{3/2}` bracket as typeset in the paper's display (4.6) reads
`(kappa_b f - 2 f_eta)`, which disagrees in sign with the same paper's
on-boundary corollary (4.11), `(2 f_eta - kappa_b f)`.  A term-by-term
evaluation of the half-plane integral agrees with the corollary, so the module
uses `(2 f_eta - kappa_b f)`.  The `delta` and `delta^2` groups are used exactly
as printed.  `lemma_4_5_terms()` prints all three groups as implemented.

References are target-centred polar quadratures of the same kernel with the
radial moments `int_0^a E_1(s^2 / 4 delta) s^n ds` in closed form
(`radial_log_moment`, an incomplete gamma function), so only the angular
variable is quadrature: `halfplane_VL_reference` for a straight boundary and
`disk_VL_reference` for a curved one, both for polynomial densities.  For a
straight boundary and a density of degree at most two the expansion is *exact*,
which is the sharpest available check on the printed coefficients.

Part (b) is a Nystrom solver for the interior Dirichlet Laplace problem on a
smooth closed curve by the double-layer representation: `star_curve(n)` samples
`r(phi) = 0.55 (1 + 0.3 cos 5 phi)`, `double_layer_matrix` assembles
`-I/2 + D` with the periodic trapezoidal rule and the Kress limiting diagonal
value `-kappa |gamma'| / (4 pi)`, `solve_interior_dirichlet` solves it, and
`double_layer_eval` / `double_layer_grad` evaluate the representation and its
gradient at interior points (smooth rule only, so accuracy degrades within a few
node spacings of the curve).  `closest_point_on_curve` is a Newton search for
the closest boundary point, which the baseline comparison needs.

The manufactured boundary value problem of the plan uses
`u_exact(x, y) = sin(2 x) cosh(2 y) + exp(-2 (x^2 + y^2))`, so
`rho = -Laplacian u_exact = exp(-2 (x^2 + y^2)) (8 - 16 (x^2 + y^2))` is smooth,
non-polynomial and not identically zero; `u_exact`, `grad_u_exact` and
`rho_manufactured` are module-level callables.

| Script | Needs |
|---|---|
| `experiment_e_baseline.py` | numpy, scipy |

Invocation (from this directory, with `PYTHONPATH` including it):

```bash
python experiment_e_baseline.py --out OUT      # add --quick to skip the slow checks
```

Outputs `experiment_e_baseline_tests.json`, a record per check with its key
number.  The full self test takes a few seconds on one core.  Its thirteen
checks cover the closed-form radial moments, the self-convergence of both polar
references, the exactness of the expansion for a straight boundary and a
quadratic density (interior and exterior), the sign of the `delta^{3/2}` term,
the `O(delta^{5/2})` remainder law, the curvature term on a disk, and the
Nystrom operator, its spectral convergence, its gradient and the manufactured
pair.

## Experiment E: the stage hierarchy (`experiment_e_hierarchy.py`)

**Purpose.** Assemble the three stages of the windowed telescoping
decomposition in two dimensions on a leaf list carrying physical pieces and a
per-leaf tensor Lagrange density proxy, and validate each stage directly
against the equation rather than against a second implementation of itself.

The decomposition is `K = W_0 + sum_l D_l + chi_0(.; t_L)` with
`t_l = (h_l / Theta)^2`, and the three stages are evaluated by three different
mechanisms:

1. `w0_stage` takes `W_0 = -log r - chi_0(r; t_0)` by direct tensor Gauss
   quadrature over every leaf piece. Order 18 (10 points per direction) is the
   default, fixed by the `w0_order_convergence` check: at `L = 3`, where the
   leaf side equals `sqrt(t_0)` and the rule is hardest, order 6 gives `5e-11`,
   order 10 gives `1e-15` and order 18 reproduces the order-40 value to
   `6e-17`.
2. `shell_stage` takes one shell by plane waves: per-box coefficients on the
   tensor grid of `experiment_e_common.plane_wave_grid` (closed Legendre /
   spherical-Bessel form for box pieces, `experiment_e_geometry`'s
   divergence-theorem and curved routes otherwise), colleague-only translation
   by `e^{i k . delta}`, the `expm1`-formed multiplier, and direct evaluation of
   the inverse transform. The per-coordinate mode count is level independent:
   147 at `Theta = 8` and 219 at `Theta = 12`, with a truncation of `6e-13`
   relative to the shell peak. That count is convention dependent through the
   periodization factor `nu` and is not DMK's `N_1`.
3. `closure_stage` takes `chi_0(.; t_L)` over the colleague leaves, by the
   separable `v = sqrt(u)` quadrature of the A-wave separability check on
   axis-aligned box pieces and by the target-centred polar rule on polygon and
   curved pieces. The gradient on those pieces is
   `experiment_e_geometry.polar_gradient`, the vector form of the same fan,
   which dispatches a curved piece's arcs to the arc rule; an earlier revision
   took it on a 32-chord polygon through each arc instead, which cost about
   `1e-6` at a target `1e-4` leaf widths inside a cut star leaf.

Every stage returns `u`, `grad u` and `Delta u` analytically; `assemble` returns
the per-stage fields and the total, and `residual_report` compares each stage's
`-Delta` with the closed-form heat-smoothed right-hand sides `G_{t_0} * rho`,
`(G_{t_{l+1}} - G_{t_l}) * rho` and `rho - G_{t_L} * rho`. Those right-hand
sides are computed exactly per leaf by `heat_smoothed_proxy` (closed-form
Gaussian box moments, polar rule elsewhere) rather than by a globally graded
quadrature, whose node count would grow like `t^{-1}`.

Two readings of `residual_report` need care. The `rho 1_Omega` factor of its
right-hand side is the *domain's* analytic predicate, passed in as `indicator`,
not the predicate `closure_stage` subtracts; if the two were the same, a
geometric misclassification would cancel between them and the residual could not
see it. The report returns `indicator_mismatch`, the number of targets where
they disagree, and anything but zero invalidates its rows. And the `closure` row
compares that stage's Gaussian term against a right-hand side built from the
same closed forms, so it measures the colleague truncation of the closure and
not the separable or polar quadrature that the closure actually uses; the
independent checks on those are `closure_separable_vs_polar_u/grad/lap` and the
assembled total against `polar_reference_total`.

| Script | Needs |
|---|---|
| `experiment_e_hierarchy.py` | numpy, scipy, `experiment_e_common`, `experiment_a_separability`, `experiment_c_leaf_residual`; `experiment_e_geometry` is used when importable |

Invocation (with `PYTHONPATH` set to this directory):

```bash
python experiments/windowed_dmk/experiment_e_hierarchy.py --out OUT --levels 3,4,5
```

Flags: `--levels` (level counts for the full-box control study), `--quick`
(level 3 only). The whole run, unit checks included, takes about half a minute
on one core.

Outputs: `experiment_e_hierarchy.json`, holding the unit checks and, per level
count, the per-stage residuals broken down by probe family (leaf centre, leaf
face, leaf corner, bulk), the mode counts, and the assembled total against a
direct polar-quadrature reference at 24 probes.

**What the checks establish.** The plane-wave shells agree with direct polar
quadrature of the same shell kernel over the same colleague range to `2e-15`
relative; the closed-form box plane-wave coefficients agree with quadrature and
with the geometry module's independent closed form to `3e-16`; the separable
box closure agrees with the polar closure of the same square to `9e-16` in `u`,
`4e-16` in `grad u` and `2e-16` in `Delta u`; every stage's analytic gradient
and Laplacian match central differences to the finite-difference truncation
level; and the assembled total matches the direct polar reference to `1e-12`
absolute, which is that reference's own self-convergence floor.

**The one finding that is not a tolerance.** With `Theta = 8` the per-stage
residual is at the plane-wave and quadrature tolerances only for targets at
leaf centres (`8e-13` at `L = 3`); at targets sitting on a leaf face or corner
it is `2e-9` to `3e-8`, and it grows with the level count because each
additional shell contributes another such term. The cause is geometric, not a
quadrature defect: the window radius `12 sqrt(t_l) = 1.5 h_l` equals the
distance from a box *centre* to the edge of its colleague range, but a target
on a box face is only `h_l` from the nearest source the colleague list omits,
so the truncated Gaussian tail is `e^{-Theta^2/4} = 1.1e-7` instead of
`e^{-36}`. `Theta = 12` removes it outright: the same study at `L = 3` gives
`1.7e-13` in total and `6e-17` for the closure, with the mode count rising from
147 to 219 per coordinate. The `shell_pitfall_*` checks separately confirm that
the cancellation-free shell forms are the right ones to use, though at these
dyadic level ratios the naive double-precision difference loses only one to two
digits rather than all of them.

Findings and verdicts: the experiment E write-up kept alongside the
manuscripts.

## Experiment E: geometry (`experiment_e_geometry.py`)

**Purpose.** Everything the two-dimensional assembly needs to know about where
the source lives: the three domains (`B` full root box, `L` the re-entrant
polygon `[-0.8, 0.8]^2 \ [0, 0.8]^2`, `S` the smooth star
`r(phi) = 0.55 (1 + 0.3 cos 5 phi)`), the clipping of a quadtree leaf against
them, the plane-wave coefficients of a per-leaf polynomial proxy over a clipped
piece, and the target-centred polar quadrature used for reference values and
for the leaf closure on cut pieces. Data structures come from
`experiment_e_common`; the polar machinery is the leaf-residual study's
(`experiment_c_leaf_residual.ray_coefficients`, `radial_moments`,
`_panel_rule`, `_dyadic_breakpoints`, imported, not re-derived), generalized to
curved edges.

| Entry point | What it gives |
|---|---|
| `make_domain("B"\|"L"\|"S")` | `inside`, `signed_distance` (negative inside; Newton on the curve for `S`), `clip_leaf(box) -> list[Piece]`, `area()`, `leaves(tree, rho, order)` |
| `StarDomain.boundary_point/tangent/normal/curvature/closest_boundary` | the boundary parametrization and its curvature, for the baseline and the Nystrom solver |
| `plane_wave_coeffs(piece, poly, kx, ky)` | `int_piece exp(-i k . y) p(y) dy`, closed form on box and polygon pieces, quadrature on curved ones |
| `polar_reference(piece, poly, targets, kernel)`, `prefix_polar(piece, poly, targets, t)` | target-centred fan quadrature; `prefix_polar` returns a *physical* potential (the `2 pi` of the chi-unit convention is already applied) |
| `polar_gradient(piece, poly, targets, kernel)` | the vector form of the same fan, sharing its angular substitution, panel rule and radial moments; a curved piece's arcs go to the arc rule, never to a chord polygon through them |
| `Domain.piece_membership(box)`, attached to every clipped piece as `inside_exact` | the exact `Omega ∩ box` predicate a consumer needs, because a curved piece's `vertices` are only the chord polygon through its edge endpoints |
| `is_axis_rectangle(piece)`, `loop_area_green(piece)`, `curved_plane_wave_quadrature` | helpers a consumer needs: separable-rule eligibility, an independent area, and a resolved rule for a curved piece |

**Two clipping conventions to know.** A leaf straddling the L corner is
returned as *two axis-aligned rectangles*, not one L-shaped hexagon: the union
is the same, the rectangles are exactly representable and need no ear clipping,
and the corner-at-a-leaf-corner and corner-inside-a-leaf cases then go down one
code path. A clipped rectangle that is not the whole leaf carries
`kind == "polygon"`, never `"box"`, so that `Leaf.is_cut()` keeps its meaning;
call `is_axis_rectangle(piece)` to find the pieces on which a separable rule
still applies. Cut star leaves are assembled by walking the box boundary and
the curve alternately (the curve leaves the box exactly where the box boundary
enters `Omega`), which handles several boundary arcs in one leaf and a curve
crossing one box edge twice.

**Invocation.**

```bash
PYTHONPATH=<worktree> python experiments/windowed_dmk/experiment_e_test_geometry.py \
    --out <dir> --levels 3,4,5,6 --no-plots
# the plots need matplotlib, which the compute venv need not have:
PYTHONPATH=<worktree> python3 experiments/windowed_dmk/experiment_e_test_geometry.py \
    --out <dir> --plots-only --plot-level 4
```

`experiment_e_geometry.py --out <dir>` runs a shorter self-test of the same
kind.

**What the checks establish** (47 rows, all passing). Clipped leaf areas sum to
the closed-form area of `Omega` to `0` for `B`, `5e-15` for `L` and `4e-16` for
`S` at levels 3 to 6, and every curved piece's lune area matches an independent
`(1/2) oint (x dy - y dx)` to `2e-16`. The two L-corner configurations and the
corner-on-a-leaf-edge case reproduce exact rectangle areas to `3e-17` with the
expected piece counts. On shifted grids, leaves whose boundary crosses one box
edge twice clip to two pieces whose area matches both the Green's-theorem value
and the sum over the four children to `1e-16`. Plane-wave coefficients from the
closed forms agree with graded quadrature to `2e-15` relative on box, polygon
and curved pieces over modes that include `k = 0`, modes orthogonal to an edge,
and a full shell grid at `|k| h = 53`; the divergence-theorem route and the
separable box route agree with each other to `8e-16` on the same rectangle. The
polar fan reproduces the leaf-residual study's own polygon routine to `0`, the
separable heat-time closure to `2e-19` on a full box and on the two rectangles
of a cut L leaf, a directly graded quadrature off the piece to `4e-20`, and a
piece's area under a constant kernel to `4e-17` including on curved pieces; on
a cut star leaf with the target `2e-4` from the boundary the fan is
order-converged to `1e-15` relative.

**Two findings worth carrying.** First, sampling the star's parameter uniformly
over the whole period misses grazing incursions: a leaf at level 5 lost a
`2.4e-7` sliver whose arc spanned `1.0e-3` in `phi`, which showed up as a
`4.8e-7` defect in the summed area. Scanning instead the angular window the
leaf subtends at the origin (legitimate because the domain is star-shaped about
it) resolves arcs of chord length about `h / 1000` at the same cost and removes
the defect; arcs far shorter than that can still be missed, and the clipped-area
sums are the check that would report it. Second, the cell size for plane-wave
coefficients of a curved piece was measured, not assumed: at `k d = 2 order / 3`
per cell the coefficients were only accurate to `4e-12` relative, and
`k d = order / 3` (the shipped constant) agrees with a rule two and a half
times finer to `1e-15`.

Findings and verdicts: the experiment E write-up kept alongside the
manuscripts.

## Experiment E: the study driver (`experiment_e_run.py`)

**Purpose.** Run the two-dimensional assembly on the three geometries and
reduce it to the tables the write-up needs. The driver computes nothing of its
own: the stages come from `experiment_e_hierarchy`, the domains, clipping and
polar rules from `experiment_e_geometry`, and the asymptotic local part and the
Nystrom solver from `experiment_e_baseline`. What it contributes is the
experimental design — which probes, which references, which sweeps — and the
incremental, resumable bookkeeping.

| Study key | What it produces |
|---|---|
| `fastpath` | the tensor contraction of curved plane-wave coefficients against the flat route it replaces |
| `b` | case B (full root box), per-stage residuals, mode counts, total against the polar reference, `L = 3, 4, 5` |
| `l` | case L (re-entrant corner), the same plus interface jumps, `L = 3 .. 6` (the extra level under a wall-clock budget) |
| `s` | case S (smooth star), the same, `L = 3, 4, 5` |
| `jumps` | interface jumps alone, at three offsets and per stage (cheap enough to rerun when the estimator changes) |
| `jumps_offsets` | the same at ten times the offsets (`--jump-offset-factor`), the control that decides which extrapolation model each site's column may be read from; run it over the same cases and levels as `jumps` |
| `baseline` | Lemma 4.5 against the untruncated closure, on the star, on a straight L edge and at the L corner |
| `bvp` | the manufactured Dirichlet problem on case S versus `L` and versus the boundary discretization |
| `theta12` | case S at one level count with the secondary declaration `Theta = 12` |
| `volumential` | the optional box-code cross-check, which records itself as skipped with a reason when it is not run |

**Invocation.**

```bash
PYTHONPATH=<worktree>:<worktree>/experiments/windowed_dmk \
  python experiments/windowed_dmk/experiment_e_run.py --out OUT/<study> \
    --studies <key> --levels 3,4,5 --resume
python experiments/windowed_dmk/experiment_e_run.py --out OUT --collect
python3 experiments/windowed_dmk/experiment_e_run.py --out OUT --plots-only
```

Every study writes its CSV rows and its JSON block as it goes, so an
interrupted run is adoptable and `--resume` skips what is already on disk.
Studies are independent processes by design: give each one its own `--out`
subdirectory so that no two append to the same CSV, then `--collect` joins the
subdirectories into one flat set of artifacts and one `experiment_e_summary.json`.
`--plots-only` reads only the CSVs, so the four PNGs can be rendered by a second
interpreter when the compute environment has no matplotlib (which is the case
here).

**Outputs.** `experiment_e_residual.csv` (per stage, per probe family, per
case and level; note that the jump probes are targets too, under the family
label `jump`, so the `all` rows are maxima over them as well — the synthetic
family `all_except_jump` is the same maximum without them),
`experiment_e_modes.csv`, `experiment_e_reference.csv`,
`experiment_e_jumps.csv`, `experiment_e_jumps_offsets.csv` (the same jump
measurement at ten times larger probe offsets, written by the `jumps_offsets`
study key, which is what shows whether an estimator is resolving the scheme or
its own remainder),
`experiment_e_baseline_vs_exact.csv`, `experiment_e_bvp.csv`, one JSON block
per study, `experiment_e_summary.json`, and four PNGs (per-stage residual
versus `L` per case, interface jumps, BVP convergence, baseline error versus
distance and versus `h_L`).

**Two measurement conventions, both forced by the data.**

*The interface jump is the difference of one-sided limits.* The volume
potential is `C^1` across the boundary while its Laplacian jumps, so
`u(b + d n) - u(b - d n)` is `O(d)` however accurate the scheme is. Each side
is evaluated at `d = 10^-3, 10^-4, 10^-5` times `h_L` and extrapolated to the
boundary. For `u` the quadratic extrapolation is the right model and gives
`5e-16` to `1.2e-12`; linear extrapolation from the plan's two offsets alone
leaves `(1/2) rho d_1 d_2`, whose maximum over the probe sites is `2.5e-9` at
`L = 3`, which is geometry and not the scheme. For `grad u` the model depends on the
site, and the `jumps_offsets` control at ten times the offsets is what decides
it: on the straight edges of case L and on the smooth curve the quadratic model
is the right one, while at the re-entrant corner it returns a pure
offset-proportional remainder (`1.85e-6` against `1.85e-5` at ten times the
offsets) and the three-term model `c0 + c1 d + c2 d log d` is the one that is
offset-independent there (`1.96e-10` against `2.69e-10`), because the one-sided
expansion of the gradient carries a `d log d` term at a re-entrant corner.

*The baseline needs the untruncated closure.* `closure_exact` sums
`prefix_potential` over every leaf within `14 sqrt(t_L)` of the target rather
than over the colleague list, so the Lemma 4.5 comparison is not contaminated
by the window truncation that the residual column measures.

**One change to `experiment_e_hierarchy` was needed to make case S run at all**
and is recorded here because it is this driver's author's change, not the
hierarchy module's: `_curved_tensor_coeffs`. A curved piece's plane-wave
coefficients on an `n x n` grid were being formed through the flat route, which
builds an `(n^2, M)` phase matrix; with `n = 147` and `M` of order `10^4` nodes
on a cut star leaf that is `2e8` complex exponentials per piece per level, and
case S at `L = 3` did not finish in ten minutes. Writing the phase as
`e^{-i k_i y_1} e^{-i k_j y_2}` and contracting with one matrix product costs
`2 n M` exponentials and one `zgemm`. The quadrature rule is unchanged — it is
the same rule `experiment_e_geometry.curved_plane_wave_quadrature` picks for the
same grid — so only the summation order differs; the `fastpath` study measures
`1.7e-15` to `2.6e-15` relative agreement with the route it replaces, at
`4.6` to `12` times less wall clock on the reduced grids where the flat route is
still affordable.

Findings and verdicts: the experiment E write-up kept alongside the
manuscripts.

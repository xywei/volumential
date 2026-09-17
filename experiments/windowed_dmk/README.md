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

"""Windowed-DMK Experiment B: parameterized scale-shell test (3D Yukawa/Laplace).

Exploratory campaign code for the windowed RKE / DMK unification study (windowed RKE / DMK
unification).  Not benchmark-grade: no warm-up, no repetition statistics, no
claim of optimality for any quadrature rule used here.

What this script does
---------------------
On a uniform dyadic tree in 3D with box side ``h_l = 2**-l`` and heat-time
windows ``t_l = (h_l / Theta)**2`` it builds the adjacent-window slab (the
"scale shell")

    D_l(r) = chi_0(r; t_l) - chi_0(r; t_{l+1})
           = (1/(2 sqrt(pi))) int_{t_{l+1}}^{t_l} u**-1.5 exp(-zeta u - r^2/(4u)) du

three ways, and measures the mode count of its Fourier (trapezoidal) spectral
representation on the colleague range.

Routes
  (i)   the chi_m channel series, sum_m (-zeta)^m chi_m(.; t) / m!, truncated at
        p_star channels (p_star = 4, 6, 8, 12);
  (ii)  the closed form D4 of the campaign brief, in the naive erfc form and in
        the stable erfcx form of the critic addendum, plus the cancellation-free
        u-integral over [t_{l+1}, t_l];
  (iii) numerical inversion of the D3 Fourier multiplier.

All three are checked against an mpmath reference.

Everything is computed in *box units* (h_l = 1).  In box units a level is fully
described by the pair (Theta, theta_l) with theta_l = mu h_l, because
t_l = Theta^-2, t_{l+1} = (2 Theta)^-2 and zeta = theta_l^2 there.  This is why
the Laplace mode count comes out level-independent, matching DMK (3.36).

Underflow policy
----------------
At coarse levels zeta t_l reaches 4096, so the shell itself is of size
exp(-zeta t_{l+1}) and underflows double precision entirely.  Every quantity in
the mode-count stage is therefore carried with a common factor
exp(+zeta t_{l+1}) divided out ("scaled"); ratios are unaffected and nothing
overflows.  The reference values in the closed-form and series stages are
computed in mpmath, where the exponent range is unbounded.

Mode-count definition (stated so the number is comparable with, not claimed to
reproduce, DMK Table 3.1)
-------------------------------------------------------------------------
The shell is represented by a tensor trapezoidal rule in Fourier space,

    D_l(x) ~ (dk/(2 pi))**3 sum_{m in [-nf, nf]^3} Dhat_l(|dk m|) exp(i dk m . x),
    dk = 2 pi / (nu h_l),

so ``nu h_l`` is the physical period of the resulting periodization.  The
evaluation region is the colleague range: source in the box, target in the box
or a face/edge/corner neighbour, i.e. offsets ``|x_i| <= 2 h_l`` (three boxes
per coordinate).  Two error sources are separated:

  * truncation.  Because Dhat_l >= 0, the max-norm truncation error over any
    evaluation set is attained at x = 0 and equals exactly
        E_trunc(nf) = (dk/(2 pi))**3 sum_{|m|_inf > nf} Dhat_l(|dk m|).
    We take nf minimal with E_trunc(nf) <= tol * D_l(0), and report
    N_1 = 2 nf + 1 together with the dimensionless band limit K h_l = nf dk h_l.
  * aliasing.  By Poisson summation the periodization error is
    sum_{n != 0} D_l(x + n nu h_l); since D_l is positive and decreasing we
    bound it by the sum over 0 < |n|_inf <= 2 of D_l(d_n) with
    d_n = sqrt(sum_i max(0, |n_i| nu h_l - 2 h_l)^2).  A value of nu is called
    admissible at tolerance tol when that bound is <= 0.1 * tol * D_l(0).

N_1 is convention-dependent through nu; the band limit K h_l is not, which is
why it is reported alongside.  DMK's (3.31) corresponds to a smaller period
than the colleague criterion above admits, so N_1 here is expected to exceed
DMK's Table 3.1 columns and is *not* presented as reproducing them.

Invocation
----------
    python experiments/windowed_dmk/experiment_b_shells.py --out <dir>

Optional ``--quick`` shrinks the Fourier sweep for a smoke run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
from scipy.special import erfc, erfcx

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

LEVELS = list(range(7))
"""Uniform dyadic tree levels l = 0 .. 6 (box side h_l = 2**-l)."""

L_FINEST = 6
"""Finest level of the brief's configuration."""

THETAS = (8.0, 16.0)
"""Window declarations Theta swept by the brief."""

P_STARS = (4, 6, 8, 12)
"""Retained channel counts for the chi_m series."""

TOLS = (1e-6, 1e-9, 1e-12)
"""Target tolerances for the Fourier mode count."""

NU_CANDIDATES = (3.0, 4.0, 6.0, 8.0)
"""Periodization factors nu (period = nu h_l) probed for the trapezoidal rule."""

NU_REPORT = 6.0
"""The nu used for the headline N_1 numbers."""

R_OVER_H = (0.05, 0.125, 0.25, 0.5, 1.0)
"""Colleague-range sample radii, in box units, for the series checks."""

ZETA_T_SWEEP = (0.0625, 0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 64.0)
"""Companion conditioning sweep in the single parameter zeta * t_l."""

DPS = 60
"""Working precision for the mpmath references (cancellation-free formulas)."""

SQRT_PI = math.sqrt(math.pi)
LN10 = math.log(10.0)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def output_dir(raw: str) -> Path:
    """Create and return the output directory for a run."""
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` as a CSV, taking the column order from the first row."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    """Read a CSV written by :func:`write_csv`, restoring numbers and booleans."""

    def convert(text: str):
        if text in ("True", "False"):
            return text == "True"
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    with path.open(newline="") as handle:
        return [{k: convert(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as pretty-printed JSON."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def environment_summary() -> dict[str, Any]:
    """Versions and CPU class of the running interpreter.

    Deliberately records no host name, user name or file system path: campaign
    policy forbids those in committed artifacts.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "machine": platform.machine(),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover - scipy is a hard dependency here
        info["scipy"] = None
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = None
    return info


def get_pyplot():
    """Return ``matplotlib.pyplot`` with the Agg backend, or ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def mp_log10(value) -> float:
    """``log10`` of an mpmath value, mapping zero to ``-inf``."""
    if value == 0:
        return -math.inf
    return float(mp.log10(abs(value)))


def safe_log10(value: float) -> float:
    """``log10`` that maps zero and negatives to ``-inf`` instead of raising."""
    if value <= 0.0:
        return -math.inf
    return math.log10(value)


# --------------------------------------------------------------------------
# route (ii): closed-form windowed prefix and shell, in box units
# --------------------------------------------------------------------------


def chi0_yukawa_naive(r: np.ndarray, t: float, mu: float) -> np.ndarray:
    """Brief D4 in its literal ``erfc`` form (overflows at coarse levels).

    Kept so the run can demonstrate *why* the critic addendum asks for the
    stable form; never used for a reported reference value.
    """
    r = np.asarray(r, dtype=np.float64)
    st = math.sqrt(t)
    with np.errstate(over="ignore", invalid="ignore"):
        zm = r / (2.0 * st) - mu * st
        zp = r / (2.0 * st) + mu * st
        return (np.exp(-mu * r) * erfc(zm) + np.exp(mu * r) * erfc(zp)) / (2.0 * r)


def chi0_yukawa(r: np.ndarray, t: float, mu: float) -> np.ndarray:
    """Brief D4 in the stable ``erfcx`` form of the critic addendum.

    Returns ``chi_0^zeta(r; t)``, the windowed (short-range, singular) prefix of
    the 3D Yukawa kernel ``exp(-mu r)/r``.
    """
    r = np.asarray(r, dtype=np.float64)
    st = math.sqrt(t)
    zm = r / (2.0 * st) - mu * st
    zp = r / (2.0 * st) + mu * st
    pref = np.exp(-(r**2) / (4.0 * t) - mu * mu * t)

    term_plus = pref * erfcx(zp)
    term_minus = np.empty_like(r)
    negative = zm < 0.0
    term_minus[negative] = np.exp(-mu * r[negative]) * erfc(zm[negative])
    term_minus[~negative] = pref[~negative] * erfcx(zm[~negative])

    return (term_minus + term_plus) / (2.0 * r)


def shell_difference_double(
    r: np.ndarray, t_coarse: float, t_fine: float, mu: float
) -> np.ndarray:
    """Shell as the double-precision difference of two D4 windowed prefixes."""
    return chi0_yukawa(r, t_coarse, mu) - chi0_yukawa(r, t_fine, mu)


def graded_log_u_rule(
    t_fine: float, t_coarse: float, n_grade: int = 26, n_node: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss rule in ``s = log(u/t_fine)`` graded to both endpoints.

    The shell integrand ``u**-1.5 exp(-zeta(u - t_fine) - r^2/(4u))`` has a
    boundary layer of width ``1/(zeta t_fine)`` in ``s`` at the lower endpoint
    when ``zeta t_fine`` is large, and the analogous behaviour at the upper
    endpoint when the saddle ``u = r/(2 mu)`` sits there.  Dyadic grading toward
    both ends resolves either case without knowing which one occurs.
    """
    span = math.log(t_coarse / t_fine)
    half = 0.5 * span
    left = [0.0] + [half * 2.0**-j for j in range(n_grade, 0, -1)] + [half]
    right = [span - half * 2.0**-j for j in range(1, n_grade + 1)] + [span]
    edges = np.array(left + right)
    nodes, weights = np.polynomial.legendre.leggauss(n_node)
    us, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        h = 0.5 * (hi - lo)
        m = 0.5 * (hi + lo)
        s = m + h * nodes
        u = t_fine * np.exp(s)
        us.append(u)
        ws.append(h * weights * u)
    return np.concatenate(us), np.concatenate(ws)


def shell_u_integral(
    r: np.ndarray,
    t_coarse: float,
    t_fine: float,
    mu: float,
    scaled: bool = False,
) -> tuple[np.ndarray, int]:
    """Shell as the cancellation-free ``u``-integral over ``[t_fine, t_coarse]``.

    The integrand is positive, so nothing cancels; the only failure mode is
    underflow of the exponential, which is the honest statement that the shell
    is below the double-precision range.  With ``scaled`` the common factor
    ``exp(zeta t_fine)`` is divided out, which keeps every coarse level
    representable.

    Returns the values and the number of quadrature nodes used.
    """
    r = np.asarray(r, dtype=np.float64)
    zeta = mu * mu
    shift = t_fine if scaled else 0.0
    u, w = graded_log_u_rule(t_fine, t_coarse)
    expo = -zeta * (u[None, :] - shift) - (r[:, None] ** 2) / (4.0 * u[None, :])
    integrand = u[None, :] ** (-1.5) * np.exp(expo)
    return (integrand * w[None, :]).sum(axis=1) / (2.0 * SQRT_PI), u.size


def _graded_breakpoints(t_fine: float, t_coarse: float, n_grade: int = 20) -> list:
    """Dyadically graded subdivision of ``[t_fine, t_coarse]`` toward both ends."""
    span = math.log(t_coarse / t_fine)
    half = 0.5 * span
    ss = (
        [0.0]
        + [half * 2.0**-j for j in range(n_grade, 0, -1)]
        + [half]
        + [span - half * 2.0**-j for j in range(1, n_grade + 1)]
        + [span]
    )
    out = []
    for s in ss:
        value = mp.mpf(t_fine) * mp.e ** mp.mpf(s)
        if not out or value > out[-1]:
            out.append(value)
    return out


def reference_dps(zeta_t_coarse: float) -> int:
    """Working precision that absorbs the level's own cancellation budget.

    Forming the shell as a difference of two D4 windowed prefixes cancels
    ``zeta t_coarse / ln 10`` digits (this is the D5 statement, applied to the
    closed form rather than to the channel series), so the reference is computed
    with that many guard digits on top of :data:`DPS`.
    """
    return int(DPS + zeta_t_coarse / LN10 + 10)


def chi0_yukawa_mp(r: float, t: float, mu: float):
    """Exact D4 windowed prefix in mpmath (both terms positive, no cancellation).

    Must be called inside an ``mp.workdps`` block.
    """
    st = mp.sqrt(mp.mpf(t))
    mu_mp = mp.mpf(mu)
    rr = mp.mpf(r)
    return (
        mp.e ** (-mu_mp * rr) * mp.erfc(rr / (2 * st) - mu_mp * st)
        + mp.e ** (mu_mp * rr) * mp.erfc(rr / (2 * st) + mu_mp * st)
    ) / (2 * rr)


def shell_reference_mp(r: float, t_coarse: float, t_fine: float, mu: float):
    """Exact shell value: the D4 closed form differenced at guard precision."""
    dps = reference_dps(mu * mu * t_coarse)
    with mp.workdps(dps):
        return chi0_yukawa_mp(r, t_coarse, mu) - chi0_yukawa_mp(r, t_fine, mu)


def shell_quadrature_mp(r: float, t_coarse: float, t_fine: float, mu: float):
    """Cross-check of :func:`shell_reference_mp` by the ``u``-integral.

    The subdivision is graded toward both endpoints: at coarse levels
    ``zeta t_fine`` reaches 1024, so the integrand has a boundary layer of
    relative width ``1/(zeta t_fine)`` at ``u = t_fine`` that an unsubdivided
    ``mp.quad`` silently misses (observed: a 5 per cent error).
    """
    with mp.workdps(DPS):
        zeta = mp.mpf(mu) ** 2
        rr = mp.mpf(r)

        def integrand(u):
            return u ** mp.mpf("-1.5") * mp.e ** (-zeta * u - rr**2 / (4 * u))

        value = mp.quad(integrand, _graded_breakpoints(t_fine, t_coarse))
        return value / (2 * mp.sqrt(mp.pi))


def shell_peak_scaled(t_coarse: float, t_fine: float, mu: float) -> float:
    """``exp(zeta t_fine) D_l(0)``, the scaled peak of the shell.

    For Laplace this is exactly ``Theta / sqrt(pi)`` in box units.
    """
    value, _ = shell_u_integral(np.array([0.0]), t_coarse, t_fine, mu, scaled=True)
    return float(value[0])


# --------------------------------------------------------------------------
# route (iii): the D3 Fourier multiplier
# --------------------------------------------------------------------------


def shell_multiplier_scaled(
    k: np.ndarray, t_coarse: float, t_fine: float, zeta: float
) -> np.ndarray:
    """``exp(zeta t_fine) Dhat_l(k)`` for the convention ``fhat(1/r) = 4 pi/k^2``.

    Written as ``exp(-t_fine k^2) (-expm1(-(t_coarse - t_fine)(k^2 + zeta)))``
    so no difference of nearly equal exponentials is formed, and the common
    ``exp(-zeta t_fine)`` underflow factor is carried outside.
    """
    k2 = np.asarray(k, dtype=np.float64) ** 2
    gap = t_coarse - t_fine
    denom = k2 + zeta
    out = np.empty_like(k2)
    singular = denom <= 0.0
    regular = ~singular
    out[regular] = (
        4.0
        * math.pi
        * np.exp(-t_fine * k2[regular])
        * (-np.expm1(-gap * denom[regular]))
        / denom[regular]
    )
    # limit of the same expression as k -> 0 at zeta = 0
    out[singular] = 4.0 * math.pi * gap
    return out


def shell_by_fourier_inversion(
    r: float, t_coarse: float, t_fine: float, zeta: float
) -> tuple[float, int]:
    """Radial inverse 3D transform of the D3 multiplier, scaled like above.

    Returns ``exp(zeta t_fine) D_l(r)`` and the number of integrand evaluations.
    """
    counter = {"n": 0}
    k_arr = np.empty(1)

    def integrand(k: float) -> float:
        counter["n"] += 1
        k_arr[0] = k
        return k * math.sin(k * r) * float(
            shell_multiplier_scaled(k_arr, t_coarse, t_fine, zeta)[0]
        )

    k_max = math.sqrt(45.0 / t_fine)
    total = 0.0
    edges = np.linspace(0.0, k_max, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for left, right in zip(edges[:-1], edges[1:]):
            piece, _ = quad(
                integrand, left, right, limit=400, epsabs=1e-18, epsrel=1e-11
            )
            total += piece
    return total / (2.0 * math.pi**2 * r), counter["n"]


# --------------------------------------------------------------------------
# route (i): the chi_m channel series
# --------------------------------------------------------------------------


def chi_m_mp(r: float, t: float, m: int):
    """``chi_m(r; t)`` in 3D from the upper incomplete gamma function.

    ``chi_m(r; t) = (1/(2 sqrt(pi))) (r/2)**(2m-1) Gamma(1/2 - m, r^2/(4t))``.
    Must be called inside an ``mp.workdps`` block.
    """
    a = mp.mpf(r) ** 2 / (4 * mp.mpf(t))
    return (mp.mpf(r) / 2) ** (2 * m - 1) * mp.gammainc(mp.mpf(1) / 2 - m, a) / (
        2 * mp.sqrt(mp.pi)
    )


def series_terms_mp(
    r: float, t_coarse: float, t_fine: float, mu: float, n_terms: int, dps: int
) -> list:
    """Terms ``(-zeta)^m (chi_m slab) / m!`` of the channel series for the shell."""
    with mp.workdps(dps):
        zeta = mp.mpf(mu) ** 2
        out = []
        for m in range(n_terms):
            slab = chi_m_mp(r, t_coarse, m) - chi_m_mp(r, t_fine, m)
            out.append((-zeta) ** m * slab / mp.factorial(m))
        return out


def log10_max_series_coeff(zeta_t: float) -> float:
    """``log10 max_m (zeta t)**m / m!`` -- the D5 cancellation budget."""
    if zeta_t <= 0.0:
        return 0.0
    m_star = int(round(zeta_t))
    best = -math.inf
    for m in range(max(0, m_star - 5), m_star + 6):
        best = max(best, m * math.log10(zeta_t) - math.lgamma(m + 1) / LN10)
    return best


# --------------------------------------------------------------------------
# mode counting
# --------------------------------------------------------------------------


def cube_norm2_counts(n_max: int) -> list[np.ndarray]:
    """Counts of lattice points in ``[-n, n]^3`` by squared Euclidean norm.

    ``result[n][s]`` is the number of ``m`` in ``[-n, n]^3`` with ``|m|^2 = s``.
    Built from a triple convolution of the one-dimensional count vector, far
    cheaper than enumerating the cube.  The convolutions are done by FFT and
    rounded back to integers (the counts are integers and the FFT error is many
    orders below one half).
    """
    out = []
    for n in range(n_max + 1):
        a = np.zeros(n * n + 1)
        a[0] = 1.0
        for j in range(1, n + 1):
            a[j * j] += 2.0
        b = np.rint(fftconvolve(a, a))
        c = np.rint(fftconvolve(b, a))
        out.append(c)
    return out


def truncation_profile(
    counts: list[np.ndarray], dk: float, t_coarse: float, t_fine: float, zeta: float
) -> np.ndarray:
    """Cumulative scaled ``sum_{|m|_inf <= n} Dhat(dk |m|)`` for each ``n``."""
    n_max = len(counts) - 1
    s = np.arange(3 * n_max * n_max + 1, dtype=np.float64)
    dhat = shell_multiplier_scaled(dk * np.sqrt(s), t_coarse, t_fine, zeta)
    totals = np.empty(n_max + 1)
    for n, c in enumerate(counts):
        totals[n] = float(np.dot(c, dhat[: c.size]))
    return totals


def aliasing_bound(
    nu: float, t_coarse: float, t_fine: float, mu: float, n_img: int = 2
) -> float:
    """Scaled bound on the periodization error over the colleague range."""
    distances = []
    for n1 in range(-n_img, n_img + 1):
        for n2 in range(-n_img, n_img + 1):
            for n3 in range(-n_img, n_img + 1):
                if n1 == n2 == n3 == 0:
                    continue
                d2 = sum(max(0.0, abs(ni) * nu - 2.0) ** 2 for ni in (n1, n2, n3))
                if d2 <= 0.0:
                    return math.inf
                distances.append(math.sqrt(d2))
    values, _ = shell_u_integral(
        np.array(distances), t_coarse, t_fine, mu, scaled=True
    )
    return float(values.sum())


# --------------------------------------------------------------------------
# experiment stages
# --------------------------------------------------------------------------


def level_parameters(theta_window: float, level: int, theta_finest: float) -> dict:
    """Box-unit parameters of the shell ``D_l`` at ``level``.

    ``theta_finest`` is ``theta_L`` at the finest level ``L_FINEST``; the tree is
    uniform and dyadic, so ``theta_l = 2**(L - l) theta_L``.
    """
    theta_l = theta_finest * 2.0 ** (L_FINEST - level)
    return {
        "level": level,
        "Theta": theta_window,
        "theta_l": theta_l,
        "theta_over_Theta": theta_l / theta_window,
        "t_coarse": 1.0 / theta_window**2,
        "t_fine": 1.0 / (2.0 * theta_window) ** 2,
        "mu": theta_l,
        "zeta_t_coarse": (theta_l / theta_window) ** 2,
        "zeta_t_fine": (theta_l / (2.0 * theta_window)) ** 2,
    }


def stage_closed_form(params_list: list[dict]) -> tuple[list[dict], dict]:
    """Check B1: D4 stable vs naive vs u-integral vs Fourier inversion."""
    rows: list[dict] = []
    worst_stable = 0.0
    worst_fourier = 0.0
    worst_uint = 0.0
    naive_bad = 0
    naive_total = 0
    representable = 0
    meaningful = 0
    for p in params_list:
        peak = shell_peak_scaled(p["t_coarse"], p["t_fine"], p["mu"])
        for rh in R_OVER_H:
            r = np.array([rh])
            naive = float(chi0_yukawa_naive(r, p["t_coarse"], p["mu"])[0])
            stable = float(chi0_yukawa(r, p["t_coarse"], p["mu"])[0])
            naive_total += 1
            naive_ok = math.isfinite(naive)
            if not naive_ok:
                naive_bad += 1

            shell_ref = shell_reference_mp(rh, p["t_coarse"], p["t_fine"], p["mu"])
            u_scaled, n_nodes = shell_u_integral(
                r, p["t_coarse"], p["t_fine"], p["mu"], scaled=True
            )
            diff_double = float(
                shell_difference_double(r, p["t_coarse"], p["t_fine"], p["mu"])[0]
            )
            four_scaled, n_fev = shell_by_fourier_inversion(
                rh, p["t_coarse"], p["t_fine"], p["mu"] ** 2
            )

            # cross-check the closed-form reference against the u-integral in
            # exact arithmetic, but only where mp.quad itself is trustworthy
            quad_check = float("nan")
            if p["zeta_t_coarse"] <= 64.0:
                with mp.workdps(DPS):
                    quad_check = mp_log10(
                        abs(
                            shell_quadrature_mp(
                                rh, p["t_coarse"], p["t_fine"], p["mu"]
                            )
                            - shell_ref
                        )
                        / abs(shell_ref)
                    )

            with mp.workdps(DPS):
                mu_mp = mp.mpf(p["mu"])
                ref_chi0 = chi0_yukawa_mp(rh, p["t_coarse"], p["mu"])
                dev_stable = float(abs(mp.mpf(stable) - ref_chi0) / abs(ref_chi0))
                ref_scaled = shell_ref * mp.e ** (
                    mu_mp**2 * mp.mpf(p["t_fine"])
                )
                rel_four = float(
                    abs(mp.mpf(four_scaled) - ref_scaled) / abs(ref_scaled)
                )
                rel_uint = float(
                    abs(mp.mpf(float(u_scaled[0])) - ref_scaled) / abs(ref_scaled)
                )
                rel_diff = float(
                    abs(mp.mpf(diff_double) - shell_ref) / abs(shell_ref)
                )
                log10_shell = mp_log10(shell_ref)
                log10_shell_scaled = mp_log10(ref_scaled)
                chi0_representable = bool(ref_chi0 > mp.mpf("1e-300"))
                err_four_vs_peak = float(
                    abs(mp.mpf(four_scaled) - ref_scaled) / mp.mpf(peak)
                )
                err_uint_vs_peak = float(
                    abs(mp.mpf(float(u_scaled[0])) - ref_scaled) / mp.mpf(peak)
                )

            # a relative comparison is only informative while the scaled shell is
            # not itself many orders below the scaled peak: the Fourier route is
            # a cancelling oscillatory integral of a peak-scale integrand
            four_meaningful = log10_shell_scaled > safe_log10(peak) - 10.0
            if chi0_representable:
                representable += 1
                worst_stable = max(worst_stable, dev_stable)
            if four_meaningful:
                meaningful += 1
                worst_fourier = max(worst_fourier, rel_four)
                worst_uint = max(worst_uint, rel_uint)

            rows.append(
                {
                    "Theta": p["Theta"],
                    "level": p["level"],
                    "theta_l": p["theta_l"],
                    "zeta_t_coarse": p["zeta_t_coarse"],
                    "r_over_h": rh,
                    "chi0_naive_finite": bool(naive_ok),
                    "chi0_representable_in_double": chi0_representable,
                    "chi0_stable_rel_dev": dev_stable,
                    "log10_shell_exact": log10_shell,
                    "log10_shell_scaled": log10_shell_scaled,
                    "log10_shell_peak_scaled": safe_log10(peak),
                    "log10_closed_form_vs_u_integral_exact": quad_check,
                    "rel_err_u_integral_scaled": rel_uint,
                    "err_u_integral_vs_peak": err_uint_vs_peak,
                    "rel_err_window_difference_double": rel_diff,
                    "rel_err_fourier_inversion_scaled": rel_four,
                    "err_fourier_inversion_vs_peak": err_four_vs_peak,
                    "fourier_comparison_meaningful": bool(four_meaningful),
                    "u_quadrature_nodes": n_nodes,
                    "fourier_integrand_evals": n_fev,
                }
            )
    summary = {
        "max_rel_dev_stable_D4_vs_mpmath_representable_only": worst_stable,
        "chi0_representable_samples": representable,
        "max_rel_err_u_integral_vs_mpmath_meaningful_only": worst_uint,
        "max_rel_err_fourier_inversion_vs_mpmath_meaningful_only": worst_fourier,
        "fourier_meaningful_samples": meaningful,
        "max_err_u_integral_vs_shell_peak": max(
            row["err_u_integral_vs_peak"] for row in rows
        ),
        "max_err_fourier_inversion_vs_shell_peak": max(
            row["err_fourier_inversion_vs_peak"] for row in rows
        ),
        "naive_D4_nonfinite_samples": naive_bad,
        "naive_D4_samples": naive_total,
        "max_log10_closed_form_vs_u_integral_exact": max(
            (
                row["log10_closed_form_vs_u_integral_exact"]
                for row in rows
                if math.isfinite(row["log10_closed_form_vs_u_integral_exact"])
            ),
            default=float("nan"),
        ),
        "note": (
            "relative comparisons are restricted to samples where the quantity is "
            "representable / not many orders below the shell peak; the *_vs_peak "
            "columns are the unrestricted, kernel-scale measure"
        ),
    }
    return rows, summary


def stage_series(params_list: list[dict]) -> tuple[list[dict], dict]:
    """Check B2/B4: chi_m series truncation and conditioning per level."""
    rows: list[dict] = []
    n_terms = max(P_STARS) + 1
    for p in params_list:
        for rh in R_OVER_H:
            terms = series_terms_mp(
                rh, p["t_coarse"], p["t_fine"], p["mu"], n_terms, DPS
            )
            shell_ref = shell_reference_mp(rh, p["t_coarse"], p["t_fine"], p["mu"])
            with mp.workdps(DPS):
                term0 = terms[0]
                shell_vs_laplace = mp_log10(shell_ref / term0)
                for p_star in P_STARS:
                    partial = mp.fsum(terms[:p_star])
                    err = abs(partial - shell_ref)
                    rows.append(
                        {
                            "Theta": p["Theta"],
                            "level": p["level"],
                            "theta_l": p["theta_l"],
                            "theta_over_Theta": p["theta_over_Theta"],
                            "zeta_t_coarse": p["zeta_t_coarse"],
                            "r_over_h": rh,
                            "p_star": p_star,
                            "log10_trunc_err_rel_shell": mp_log10(err / abs(shell_ref)),
                            "log10_trunc_err_rel_laplace_shell": mp_log10(
                                err / abs(term0)
                            ),
                            "log10_max_series_coeff": log10_max_series_coeff(
                                p["zeta_t_coarse"]
                            ),
                            "log10_shell_over_laplace_shell": shell_vs_laplace,
                            "digits_lost_D5_prediction": p["zeta_t_coarse"] / LN10,
                        }
                    )
    summary = {
        "p_stars": list(P_STARS),
        "note": (
            "truncation error is reported both relative to the shell itself and "
            "relative to term 0 (the Laplace shell at the same window pair); the "
            "latter is the kernel-scale measure the error budget uses"
        ),
    }
    return rows, summary


def stage_conditioning_sweep() -> tuple[list[dict], dict]:
    """Companion check B5: digits lost versus ``zeta t``, one parameter."""
    rows: list[dict] = []
    theta_window = 16.0
    t_coarse = 1.0 / theta_window**2
    t_fine = 1.0 / (2.0 * theta_window) ** 2
    rh = 0.25
    for zeta_t in ZETA_T_SWEEP:
        mu = theta_window * math.sqrt(zeta_t)
        dps = int(60 + 2.0 * zeta_t / LN10)
        n_terms = int(6 * zeta_t + 60)
        terms = series_terms_mp(rh, t_coarse, t_fine, mu, n_terms, dps)
        with mp.workdps(dps):
            shell_ref = shell_reference_mp(rh, t_coarse, t_fine, mu)
            full = mp.fsum(terms)
            max_term = max(abs(t) for t in terms)
            series_vs_closed = mp_log10(abs(full - shell_ref) / abs(shell_ref))
            digits_vs_shell = mp_log10(max_term / abs(shell_ref))
            digits_vs_term0 = mp_log10(max_term / abs(terms[0]))
            shell_vs_laplace = mp_log10(shell_ref / terms[0])
            float_terms = []
            overflowed = False
            for term in terms:
                try:
                    float_terms.append(float(term))
                except (OverflowError, ValueError):
                    overflowed = True
                    break
            if overflowed or not all(math.isfinite(x) for x in float_terms):
                double_rel = math.inf
            else:
                acc = 0.0
                for x in float_terms:
                    acc += x
                double_rel = mp_log10(abs(mp.mpf(acc) - shell_ref) / abs(shell_ref))
            trunc = {}
            for p_star in P_STARS:
                err = abs(mp.fsum(terms[:p_star]) - shell_ref)
                trunc[p_star] = mp_log10(err / abs(terms[0]))
        rows.append(
            {
                "zeta_t_coarse": zeta_t,
                "log10_trunc_err_p4": trunc[4],
                "log10_trunc_err_p6": trunc[6],
                "log10_trunc_err_p8": trunc[8],
                "log10_trunc_err_p12": trunc[12],
                "theta_over_Theta": math.sqrt(zeta_t),
                "r_over_h": rh,
                "n_terms": n_terms,
                "log10_series_minus_closed_form": series_vs_closed,
                "digits_lost_measured_vs_shell": digits_vs_shell,
                "digits_lost_measured_vs_term0": digits_vs_term0,
                "digits_lost_D5_prediction": zeta_t / LN10,
                "log10_shell_over_laplace_shell": shell_vs_laplace,
                "log10_double_precision_rel_err": double_rel,
                "mp_dps": dps,
            }
        )
    summary = {
        "sweep_zeta_t": list(ZETA_T_SWEEP),
        "note": (
            "digits lost is log10(max_m |term_m| / |sum|); the D5 prediction "
            "zeta t / ln 10 is the same quantity measured against term 0"
        ),
    }
    return rows, summary


def stage_modes(params_list: list[dict], n_max: int, counts) -> tuple[list[dict], dict]:
    """Check B3: per-coordinate Fourier mode count on the colleague range."""
    rows: list[dict] = []
    dhat_evals = 0
    for p in params_list:
        zeta = p["mu"] ** 2
        d0 = shell_peak_scaled(p["t_coarse"], p["t_fine"], p["mu"])
        d0_laplace = shell_peak_scaled(p["t_coarse"], p["t_fine"], 0.0)
        log10_amp = -p["zeta_t_fine"] / LN10 + safe_log10(d0 / d0_laplace)
        for nu in NU_CANDIDATES:
            dk = 2.0 * math.pi / nu
            totals = truncation_profile(counts, dk, p["t_coarse"], p["t_fine"], zeta)
            dhat_evals += 3 * n_max * n_max + 1
            grand = totals[-1]
            tail = grand - totals
            alias = aliasing_bound(nu, p["t_coarse"], p["t_fine"], p["mu"])
            for tol in TOLS:
                threshold = tol * d0 * nu**3
                hits = np.nonzero(tail <= threshold)[0]
                nf = int(hits[0]) if hits.size else -1
                ok = nf >= 0
                resolved = ok and nf < n_max - 2
                rows.append(
                    {
                        "Theta": p["Theta"],
                        "level": p["level"],
                        "theta_l": p["theta_l"],
                        "theta_over_Theta": p["theta_over_Theta"],
                        "kernel": "yukawa" if p["mu"] > 0 else "laplace",
                        "nu_period_over_h": nu,
                        "tol": tol,
                        "n_f": nf,
                        "N_1": (2 * nf + 1) if ok else -1,
                        "band_limit_K_h": (nf * dk) if ok else float("nan"),
                        "resolved": bool(resolved),
                        "nu_admissible": bool(alias <= 0.1 * tol * d0),
                        "aliasing_bound_rel": alias / d0,
                        "shell_peak_scaled": d0,
                        "log10_shell_amplitude_vs_laplace": log10_amp,
                        "trapezoid_peak_rel_dev": abs(grand / nu**3 - d0) / d0,
                        "n_max_searched": n_max,
                    }
                )
    summary = {
        "n_max_searched": n_max,
        "nu_candidates": list(NU_CANDIDATES),
        "dhat_evaluations": dhat_evals,
        "definition": (
            "n_f minimal with (dk/2pi)^3 sum_{|m|_inf>n_f} Dhat(dk|m|) <= tol * "
            "D_l(0); dk = 2 pi/(nu h_l); N_1 = 2 n_f + 1; band limit K h_l = n_f dk h_l"
        ),
    }
    return rows, summary


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def make_plots(out: Path, series_rows, mode_rows, sweep_rows) -> list[str]:
    """Write the PNG figures if matplotlib is importable; return file names."""
    plt = get_pyplot()
    if plt is None:
        return []
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for level in LEVELS[:-1]:
        xs, ys = [], []
        for row in series_rows:
            if (
                row["Theta"] == 16.0
                and row["level"] == level
                and row["r_over_h"] == 0.25
            ):
                xs.append(row["p_star"])
                ys.append(row["log10_trunc_err_rel_laplace_shell"])
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                label=f"l = {level}, theta/Theta = {2.0 ** (L_FINEST - level):.0f}",
            )
    ax.set_xlabel("retained channels p*")
    ax.set_ylabel("log10 truncation error / Laplace shell")
    ax.set_title("chi_m series truncation for the shell (Theta = 16, r = 0.25 h)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "experiment_b_series_truncation.png", dpi=150)
    plt.close(fig)
    written.append("experiment_b_series_truncation.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels, values = [], []
    for row in mode_rows:
        if (
            row["kernel"] == "laplace"
            and row["nu_period_over_h"] == NU_REPORT
            and row["N_1"] > 0
        ):
            labels.append(f"Theta={row['Theta']:.0f}\ntol={row['tol']:g}")
            values.append(row["N_1"])
    if labels:
        ax.bar(range(len(labels)), values, color="#4c72b0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(f"N_1 = 2 n_f + 1 (nu = {NU_REPORT:.0f})")
    ax.set_title("per-coordinate Fourier modes, Laplace shell, colleague range")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "experiment_b_fourier_modes.png", dpi=150)
    plt.close(fig)
    written.append("experiment_b_fourier_modes.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = [row["zeta_t_coarse"] for row in sweep_rows]
    ax.plot(
        xs,
        [row["digits_lost_measured_vs_term0"] for row in sweep_rows],
        marker="o",
        label="measured digits lost (max term / term 0)",
    )
    ax.plot(
        xs,
        [row["digits_lost_D5_prediction"] for row in sweep_rows],
        linestyle="--",
        label="D5 prediction: zeta t / ln 10",
    )
    ax.plot(
        xs,
        [-row["log10_shell_over_laplace_shell"] for row in sweep_rows],
        marker="^",
        label="shell decay: -log10(shell / Laplace shell)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("zeta t_l = (theta_l / Theta)^2")
    ax.set_ylabel("decimal digits")
    ax.set_title("series conditioning versus shell amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "experiment_b_conditioning.png", dpi=150)
    plt.close(fig)
    written.append("experiment_b_conditioning.png")

    return written


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    """Run every experiment-B stage and write CSV/JSON/PNG artifacts."""
    parser = argparse.ArgumentParser(description="windowed-DMK experiment B")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--quick", action="store_true", help="shrink the Fourier sweep (smoke run)"
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "skip every computation and redraw the PNGs from the CSVs already in "
            "--out (used when the compute interpreter has no matplotlib)"
        ),
    )
    args = parser.parse_args()
    out = output_dir(args.out)
    started = time.time()

    if args.plots_only:
        pngs = make_plots(
            out,
            read_csv(out / "experiment_b_series.csv"),
            read_csv(out / "experiment_b_fourier_modes.csv"),
            read_csv(out / "experiment_b_conditioning.csv"),
        )
        print(f"[B] plots-only wrote {pngs}", flush=True)
        return 0

    params: list[dict] = []
    for theta_window in THETAS:
        for level in LEVELS[:-1]:
            params.append(level_parameters(theta_window, level, theta_window))

    print("[B] stage 1: closed form, u-integral and Fourier inversion", flush=True)
    closed_rows, closed_summary = stage_closed_form(params)
    write_csv(out / "experiment_b_closed_form.csv", closed_rows)

    print("[B] stage 2: chi_m series truncation per level", flush=True)
    series_rows, series_summary = stage_series(params)
    write_csv(out / "experiment_b_series.csv", series_rows)

    print("[B] stage 3: conditioning sweep in zeta t", flush=True)
    sweep_rows, sweep_summary = stage_conditioning_sweep()
    write_csv(out / "experiment_b_conditioning.csv", sweep_rows)

    n_max = 120 if args.quick else 320
    print(f"[B] stage 4: lattice counts up to n = {n_max}", flush=True)
    counts = cube_norm2_counts(n_max)

    print("[B] stage 5: Fourier mode count (Laplace baseline)", flush=True)
    laplace_params = []
    for theta_window in THETAS:
        p = level_parameters(theta_window, 0, 0.0)
        p["level"] = -1
        laplace_params.append(p)
    laplace_rows, _ = stage_modes(laplace_params, n_max, counts)

    print("[B] stage 6: Fourier mode count (Yukawa levels)", flush=True)
    mode_rows, mode_summary = stage_modes(params, n_max, counts)
    all_mode_rows = laplace_rows + mode_rows
    write_csv(out / "experiment_b_fourier_modes.csv", all_mode_rows)

    print("[B] stage 7: regime split", flush=True)
    regime_rows = []
    for p in params:
        zeta_t = p["zeta_t_coarse"]
        best = {}
        for p_star in P_STARS:
            vals = [
                row["log10_trunc_err_rel_laplace_shell"]
                for row in series_rows
                if row["Theta"] == p["Theta"]
                and row["level"] == p["level"]
                and row["p_star"] == p_star
            ]
            best[p_star] = max(vals) if vals else float("nan")
        amp_rows = [
            row["log10_shell_over_laplace_shell"]
            for row in series_rows
            if row["Theta"] == p["Theta"] and row["level"] == p["level"]
        ]
        log10_amp = max(amp_rows) if amp_rows else float("nan")
        if best[12] <= -12.0:
            regime = "series usable at p*=12"
        elif log10_amp < -12.0:
            regime = "shell negligible at 1e-12"
        else:
            regime = "closed form required"
        regime_rows.append(
            {
                "Theta": p["Theta"],
                "level": p["level"],
                "theta_l": p["theta_l"],
                "theta_over_Theta": p["theta_over_Theta"],
                "zeta_t_coarse": zeta_t,
                "log10_shell_over_laplace_shell": log10_amp,
                "digits_lost_D5_prediction": zeta_t / LN10,
                "log10_max_series_coeff": log10_max_series_coeff(zeta_t),
                "log10_trunc_err_p4": best[4],
                "log10_trunc_err_p6": best[6],
                "log10_trunc_err_p8": best[8],
                "log10_trunc_err_p12": best[12],
                "regime": regime,
            }
        )
    write_csv(out / "experiment_b_regime_split.csv", regime_rows)

    print("[B] stage 8: plots", flush=True)
    pngs = make_plots(out, series_rows, all_mode_rows, sweep_rows)

    headline = {}
    for row in all_mode_rows:
        if row["nu_period_over_h"] == NU_REPORT and row["kernel"] == "laplace":
            headline[f"Theta={row['Theta']:.0f},tol={row['tol']:g}"] = {
                "n_f": row["n_f"],
                "N_1": row["N_1"],
                "band_limit_K_h": row["band_limit_K_h"],
                "nu_admissible": row["nu_admissible"],
            }

    summary = {
        "experiment": "B - parameterized shell test (windowed DMK unification)",
        "study": "windowed-rke-dmk-unification",
        "exploratory_not_benchmark_grade": True,
        "environment": environment_summary(),
        "configuration": {
            "dimension": 3,
            "levels": LEVELS,
            "finest_level": L_FINEST,
            "box_side": "h_l = 2**-l",
            "window": "t_l = (h_l/Theta)**2",
            "Theta": list(THETAS),
            "mu_rule": "theta_L = Theta at the finest level, theta_l = 2**(L-l) Theta",
            "p_stars": list(P_STARS),
            "tolerances": list(TOLS),
            "colleague_range": "offsets |x_i| <= 2 h_l (three boxes per coordinate)",
            "units": "box units, h_l = 1",
        },
        "closed_form": closed_summary,
        "series": series_summary,
        "conditioning_sweep": sweep_summary,
        "modes": mode_summary,
        "headline_laplace_mode_counts": headline,
        "dmk_table_3_1_quoted_for_context_only": {
            "note": (
                "quoted from the DMK preprint; the definitions differ (PSWF window, "
                "different periodization), so these are NOT reproduced here"
            ),
            "N_1_pswf": {"1e-3": 13, "1e-6": 25, "1e-9": 39, "1e-12": 53},
            "N_1_gaussian": {"1e-3": 22, "1e-6": 44, "1e-9": 66, "1e-12": 88},
            "fourier_spacing_rule": "h_0 about 4 pi / 3 per DMK (3.31)",
        },
        "plots": pngs,
        "wall_clock_seconds": time.time() - started,
    }
    write_json(out / "experiment_b_summary.json", summary)
    print(f"[B] done in {summary['wall_clock_seconds']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

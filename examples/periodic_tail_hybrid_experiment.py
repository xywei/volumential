"""Hybrid periodic Laplace tail experiment (2D).

This script combines:
1) Gaussian-window eta->0 regularization for low-order lattice sums,
2) direct hard-cutoff shell sums with Richardson acceleration for higher orders,

to build periodic tail coefficients T_{beta,alpha} for a 2D Laplace kernel.

It then evaluates a periodic potential on volumential source modes using:
- central-cell box FMM via drive_volume_fmm,
- near-image sums via direct image-shell evaluation,
- polynomial tail correction from T_{beta,alpha} for remaining images.

For comparison and sanity checking, it reports gauge-aware errors against a
periodic spectral reference and optional direct far-image checks.
"""

from __future__ import annotations

import argparse
import math
import time
from functools import lru_cache, partial
from itertools import product

import numpy as np
import pyopencl as cl
import pyopencl.array
from pytools.obj_array import new_1d as obj_array_1d

import volumential.meshgen as mg
from volumential.expansion_wrangler_fpnd import (
    FPNDExpansionWrangler,
    FPNDTreeIndependentDataForWrangler,
)
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.volume_fmm import drive_volume_fmm, interpolate_volume_potential


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--q-order", type=int, default=4)
    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--fmm-order", type=int, default=16)
    parser.add_argument("--kmax", type=int, default=48)
    parser.add_argument("--near-radius", type=int, default=1)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--tail-exp", type=float, default=40.0)
    parser.add_argument("--eta-fit-points", type=int, default=8)
    parser.add_argument("--high-order-start-R", type=int, default=512)
    parser.add_argument("--high-order-max-R", type=int, default=65536)
    parser.add_argument("--high-order-selftol", type=float, default=1.0e-13)
    parser.add_argument(
        "--high-order-method",
        choices=["hard_richardson", "eisenstein"],
        default="eisenstein",
        help="estimator for even derivative sums with order > 2",
    )
    parser.add_argument(
        "--high-order-mp-dps",
        type=int,
        default=80,
        help="mpmath precision (decimal digits) for high-order eisenstein sums",
    )
    parser.add_argument("--probe-grid-size", type=int, default=45)
    parser.add_argument(
        "--cache-file",
        default="nft-periodic-tail-hybrid.sqlite",
        help="near-field table cache filename",
    )
    parser.add_argument(
        "--verify-low-order-square-pv",
        action="store_true",
        help=(
            "compare eta-regularized m=2 derivative sums against very-large-R "
            "square principal-value extrapolation"
        ),
    )
    parser.add_argument(
        "--direct-far-cutoff-r",
        type=int,
        default=0,
        help=(
            "if >0, compute extrapolated direct far-image reference using "
            "square cutoffs R and 2R, and report tail-vs-direct error"
        ),
    )
    parser.add_argument(
        "--direct-far-second-cutoff-r",
        type=int,
        default=0,
        help=(
            "optional second radius for direct far-image extrapolation; if not "
            "set, uses 2*direct-far-cutoff-r"
        ),
    )
    parser.add_argument(
        "--direct-far-fit-order",
        type=int,
        default=1,
        help=(
            "inverse-even-power extrapolation order for direct far-image limit; "
            "1 matches Richardson with radii [R, 2R]"
        ),
    )
    parser.add_argument(
        "--direct-far-num-radii",
        type=int,
        default=2,
        help=(
            "number of radii used in direct far-image extrapolation; each extra "
            "radius doubles the previous one unless second cutoff is provided"
        ),
    )
    parser.add_argument(
        "--tail-stage-rel-tol",
        type=float,
        default=1.0e-14,
        help=(
            "target relative tolerance for tail-stage validation; requires both "
            "tail-vs-direct and direct-reference self-consistency to pass"
        ),
    )
    parser.add_argument(
        "--direct-far-target-selftol",
        type=float,
        default=0.0,
        help=(
            "if >0, adaptively increase direct-far cutoffs until direct-reference "
            "self-consistency rel_l2 <= this tolerance"
        ),
    )
    parser.add_argument(
        "--direct-far-max-cutoff-r",
        type=int,
        default=0,
        help=(
            "maximum cutoff radius used by adaptive direct-far checks; ignored "
            "if direct-far-target-selftol <= 0"
        ),
    )
    parser.add_argument(
        "--pde-far-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "build a PDE-based non-central-image reference from periodic solution "
            "minus central-cell and near-image contributions"
        ),
    )
    parser.add_argument(
        "--pde-reference-kmax",
        type=int,
        default=0,
        help=(
            "spectral truncation for PDE far reference; if <=0, uses --kmax"
        ),
    )
    parser.add_argument(
        "--pde-reference-method",
        choices=["spectral", "ewald"],
        default="spectral",
        help="backend used to build PDE full-reference potential",
    )
    parser.add_argument(
        "--ewald-xi",
        type=float,
        default=8.0,
        help="Ewald splitting parameter xi for PDE reference",
    )
    parser.add_argument(
        "--ewald-real-cutoff-r",
        type=int,
        default=6,
        help="real-space image cutoff radius for Ewald PDE reference",
    )
    parser.add_argument(
        "--ewald-kmax",
        type=int,
        default=64,
        help="reciprocal-space mode cutoff radius for Ewald PDE reference",
    )
    parser.add_argument(
        "--pde-reference-apply-dipole",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply dipole linear correction when converting spectral PDE "
            "reference to square-sum convention"
        ),
    )
    parser.add_argument(
        "--spectral-chunk-size",
        type=int,
        default=8192,
        help=(
            "number of Fourier modes per chunk in spectral reference evaluation; "
            "use <=0 to process all modes at once"
        ),
    )
    parser.add_argument(
        "--spectral-compensated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use compensated accumulation for spectral periodic reference",
    )
    parser.add_argument(
        "--spectral-source-block-size",
        type=int,
        default=0,
        help=(
            "if >0, accumulate spectral source modes in source blocks; "
            "0 keeps dense source-by-mode matmul"
        ),
    )
    parser.add_argument(
        "--spectral-accum-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accumulate spectral reference in np.longdouble",
    )
    parser.add_argument(
        "--direct-sum-source-block-size",
        type=int,
        default=0,
        help=(
            "if >0, evaluate direct image sums in source blocks to reduce memory "
            "and improve compensated summation"
        ),
    )
    parser.add_argument(
        "--direct-sum-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accumulate direct image sums in np.longdouble",
    )
    parser.add_argument(
        "--tail-eval-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate polynomial tail in np.longdouble before casting to float64",
    )
    parser.add_argument(
        "--calculuspatch-pde-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run -Delta(u_hybrid)-rho residual check on an interior calculus patch",
    )
    parser.add_argument(
        "--patch-order",
        type=int,
        default=11,
        help="calculus patch order for PDE residual diagnostics",
    )
    parser.add_argument(
        "--patch-h",
        type=float,
        default=0.30,
        help="calculus patch half-width for PDE residual diagnostics",
    )
    parser.add_argument(
        "--patch-center-x",
        type=float,
        default=0.5,
        help="x-coordinate of calculus patch center",
    )
    parser.add_argument(
        "--patch-center-y",
        type=float,
        default=0.5,
        help="y-coordinate of calculus patch center",
    )
    parser.add_argument(
        "--periodicity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="measure periodic jumps in value and first derivatives across cell faces",
    )
    parser.add_argument(
        "--periodicity-samples",
        type=int,
        default=48,
        help="number of sample points per face for periodicity diagnostics",
    )
    parser.add_argument(
        "--periodicity-eps",
        type=float,
        default=1.0e-3,
        help="distance from boundary used to sample periodicity jumps",
    )
    parser.add_argument(
        "--periodicity-fd-h",
        type=float,
        default=2.0e-4,
        help="finite-difference spacing for periodic derivative jumps",
    )
    parser.add_argument(
        "--periodicity-evaluator",
        choices=["direct", "fmm"],
        default="direct",
        help=(
            "target evaluator used for periodicity diagnostics; 'direct' supports "
            "translated-face checks outside the base cell"
        ),
    )
    parser.add_argument(
        "--periodicity-gauge",
        choices=["none", "dipole", "affine"],
        default="dipole",
        help="gauge correction applied before periodicity diagnostics",
    )
    parser.add_argument(
        "--periodicity-translated-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compare translated face pairs (x vs x+L, y vs y+L) for true periodic "
            "jump checks"
        ),
    )
    return parser.parse_args()


def _create_non_intel_context() -> cl.Context:
    platforms = cl.get_platforms()
    for platform in platforms:
        if platform.name == "Intel(R) OpenCL":
            continue
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    raise RuntimeError("No OpenCL devices available")


def _spectral_periodic_laplace_reference(
    *,
    src_points: np.ndarray,
    strengths: np.ndarray,
    tgt_points: np.ndarray,
    cell_size: np.ndarray,
    k_max: int,
    chunk_size: int = 8192,
    compensated: bool = False,
    source_block_size: int = 0,
    accum_extended_precision: bool = False,
) -> np.ndarray:
    src_points = np.asarray(src_points, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    tgt_points = np.asarray(tgt_points, dtype=np.float64)
    cell_size = np.asarray(cell_size, dtype=np.float64)

    dim = int(src_points.shape[1])
    if dim != 2:
        raise ValueError("spectral reference helper is implemented for dim=2 only")

    volume = float(np.prod(cell_size))
    n_modes_total = (2 * int(k_max) + 1) ** dim - 1
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = n_modes_total

    source_block_size = int(source_block_size)
    if source_block_size < 0:
        raise ValueError("spectral-source-block-size must be >= 0")

    use_compensated_path = (
        bool(compensated)
        or source_block_size > 0
        or bool(accum_extended_precision)
    )

    def iter_mode_chunks():
        chunk = []
        for mode in product(range(-int(k_max), int(k_max) + 1), repeat=dim):
            if mode[0] == 0 and mode[1] == 0:
                continue
            chunk.append(mode)
            if len(chunk) >= chunk_size:
                yield np.asarray(chunk, dtype=np.float64)
                chunk = []
        if chunk:
            yield np.asarray(chunk, dtype=np.float64)

    if not use_compensated_path:
        acc = np.zeros(tgt_points.shape[0], dtype=np.complex128)
        for modes_chunk in iter_mode_chunks():
            kvec = (2.0 * np.pi) * modes_chunk / cell_size.reshape(1, -1)
            k2 = np.sum(kvec * kvec, axis=1)

            src_phase = np.exp(-1j * (src_points @ kvec.T))
            rho_k = src_phase.T @ strengths

            tgt_phase = np.exp(1j * (tgt_points @ kvec.T))
            acc += tgt_phase @ (rho_k / k2)

        return np.asarray(np.real(acc / volume), dtype=np.float64)

    work_dtype = np.longdouble if bool(accum_extended_precision) else np.float64
    acc_real = np.zeros(tgt_points.shape[0], dtype=work_dtype)
    acc_real_comp = np.zeros_like(acc_real)

    src_block = source_block_size if source_block_size > 0 else src_points.shape[0]
    for modes_chunk in iter_mode_chunks():
        kvec = (2.0 * np.pi) * modes_chunk / cell_size.reshape(1, -1)
        k2 = np.asarray(np.sum(kvec * kvec, axis=1), dtype=work_dtype)

        if source_block_size <= 0:
            src_phase = np.exp(-1j * (src_points @ kvec.T))
            rho_k = src_phase.T @ strengths
            rho_real = np.asarray(np.real(rho_k), dtype=work_dtype)
            rho_imag = np.asarray(np.imag(rho_k), dtype=work_dtype)
        else:
            n_modes = modes_chunk.shape[0]
            rho_real = np.zeros(n_modes, dtype=work_dtype)
            rho_imag = np.zeros(n_modes, dtype=work_dtype)
            rho_real_comp = np.zeros_like(rho_real)
            rho_imag_comp = np.zeros_like(rho_imag)

            for start in range(0, src_points.shape[0], src_block):
                stop = min(src_points.shape[0], start + src_block)
                src_blk = src_points[start:stop]
                str_blk = np.asarray(strengths[start:stop], dtype=work_dtype)

                phase_src = src_blk @ kvec.T
                contrib_real = np.asarray(np.cos(phase_src).T @ str_blk, dtype=work_dtype)
                contrib_imag = np.asarray((-np.sin(phase_src)).T @ str_blk, dtype=work_dtype)
                rho_real, rho_real_comp = _kahan_add_array(
                    rho_real,
                    rho_real_comp,
                    contrib_real,
                )
                rho_imag, rho_imag_comp = _kahan_add_array(
                    rho_imag,
                    rho_imag_comp,
                    contrib_imag,
                )

        w_real = rho_real / k2
        w_imag = rho_imag / k2

        phase_tgt = tgt_points @ kvec.T
        cos_tgt = np.asarray(np.cos(phase_tgt), dtype=work_dtype)
        sin_tgt = np.asarray(np.sin(phase_tgt), dtype=work_dtype)
        chunk_real = cos_tgt @ w_real - sin_tgt @ w_imag
        chunk_real = np.asarray(chunk_real, dtype=work_dtype)
        acc_real, acc_real_comp = _kahan_add_array(acc_real, acc_real_comp, chunk_real)

    return np.asarray(acc_real / work_dtype(volume), dtype=np.float64)


def _ewald_periodic_laplace_reference(
    *,
    src_points: np.ndarray,
    strengths: np.ndarray,
    tgt_points: np.ndarray,
    cell_size: np.ndarray,
    xi: float,
    real_cutoff_r: int,
    k_max: int,
    mode_chunk_size: int = 4096,
) -> np.ndarray:
    try:
        from scipy.special import exp1
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --pde-reference-method=ewald"
        ) from exc

    src_points = np.asarray(src_points, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    tgt_points = np.asarray(tgt_points, dtype=np.float64)
    cell_size = np.asarray(cell_size, dtype=np.float64)

    if src_points.shape[1] != 2 or tgt_points.shape[1] != 2:
        raise ValueError("Ewald PDE reference is implemented for dim=2 only")

    xi = float(xi)
    real_cutoff_r = int(real_cutoff_r)
    k_max = int(k_max)
    mode_chunk_size = int(mode_chunk_size)

    if xi <= 0:
        raise ValueError("--ewald-xi must be > 0")
    if real_cutoff_r < 0:
        raise ValueError("--ewald-real-cutoff-r must be >= 0")
    if k_max < 1:
        raise ValueError("--ewald-kmax must be >= 1")

    lx = float(cell_size[0])
    ly = float(cell_size[1])
    area = lx * ly

    out = np.zeros(tgt_points.shape[0], dtype=np.float64)
    comp = np.zeros_like(out)

    tiny = np.finfo(np.float64).tiny
    xi2 = xi * xi
    for ix in range(-real_cutoff_r, real_cutoff_r + 1):
        shift_x = float(ix) * lx
        for iy in range(-real_cutoff_r, real_cutoff_r + 1):
            shift_y = float(iy) * ly

            dx = tgt_points[:, None, 0] - (src_points[None, :, 0] + shift_x)
            dy = tgt_points[:, None, 1] - (src_points[None, :, 1] + shift_y)
            r2 = dx * dx + dy * dy
            r2 = np.where(r2 == 0.0, tiny, r2)

            increment = (exp1(xi2 * r2) @ strengths) / (4.0 * np.pi)
            y = increment - comp
            t = out + y
            comp = (t - out) - y
            out = t

    n_modes_total = (2 * k_max + 1) ** 2 - 1
    if mode_chunk_size <= 0:
        mode_chunk_size = n_modes_total

    def iter_mode_chunks():
        chunk = []
        for mx in range(-k_max, k_max + 1):
            for my in range(-k_max, k_max + 1):
                if mx == 0 and my == 0:
                    continue
                chunk.append((mx, my))
                if len(chunk) >= mode_chunk_size:
                    yield np.asarray(chunk, dtype=np.float64)
                    chunk = []
        if chunk:
            yield np.asarray(chunk, dtype=np.float64)

    reciprocal = np.zeros(tgt_points.shape[0], dtype=np.complex128)
    for modes_chunk in iter_mode_chunks():
        kvec = np.empty((len(modes_chunk), 2), dtype=np.float64)
        kvec[:, 0] = (2.0 * np.pi) * modes_chunk[:, 0] / lx
        kvec[:, 1] = (2.0 * np.pi) * modes_chunk[:, 1] / ly

        k2 = np.sum(kvec * kvec, axis=1)
        damp = np.exp(-k2 / (4.0 * xi2)) / k2

        src_phase = np.exp(-1j * (src_points @ kvec.T))
        rho_k = src_phase.T @ strengths

        tgt_phase = np.exp(1j * (tgt_points @ kvec.T))
        reciprocal += tgt_phase @ (damp * rho_k)

    out += np.asarray(np.real(reciprocal / area), dtype=np.float64)
    return out


def _sample_smooth_source_terms(
    rng: np.random.Generator,
    *,
    n_random_terms: int = 4,
) -> tuple[tuple[int, int, float, float], ...]:
    terms = []
    for _ in range(int(n_random_terms)):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(-0.2, 0.2))
        terms.append((kx, ky, phase, amp))
    return tuple(terms)


def _evaluate_smooth_source_raw(
    points: np.ndarray,
    random_terms: tuple[tuple[int, int, float, float], ...],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]

    field = (
        0.90 * np.cos(2.0 * np.pi * x)
        - 0.75 * np.sin(4.0 * np.pi * y)
        + 0.35 * np.cos(2.0 * np.pi * (x + y))
        - 0.15 * np.sin(6.0 * np.pi * x - 2.0 * np.pi * y)
    )

    for kx, ky, phase, amp in random_terms:
        field += float(amp) * np.cos(2.0 * np.pi * (int(kx) * x + int(ky) * y) + float(phase))

    return np.asarray(field, dtype=np.float64)


def _build_smooth_neutral_source_from_terms(
    points: np.ndarray,
    weights: np.ndarray | None,
    random_terms: tuple[tuple[int, int, float, float], ...],
    *,
    weighted_mean: float | None = None,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    field = _evaluate_smooth_source_raw(points, random_terms)

    if weighted_mean is None:
        if weights is None:
            raise ValueError("weights are required when weighted_mean is not provided")
        weights = np.asarray(weights, dtype=np.float64)
        weighted_mean = float(np.dot(field, weights) / np.sum(weights))

    return np.asarray(field - float(weighted_mean), dtype=np.float64), float(weighted_mean)


def _build_smooth_neutral_source(
    points: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    random_terms = _sample_smooth_source_terms(rng)
    values, _ = _build_smooth_neutral_source_from_terms(
        points,
        weights,
        random_terms,
    )
    return values


def _build_wrangler(
    *,
    ctx: cl.Context,
    queue: cl.CommandQueue,
    traversal,
    near_field_table,
    fmm_order: int,
    quad_order: int,
):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    dim = int(traversal.tree.dimensions)
    knl = LaplaceKernel(dim)

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        [knl],
        exclude_self=True,
    )

    self_extra_kwargs = {}
    tree = traversal.tree
    if getattr(tree, "sources_are_targets", False):
        self_extra_kwargs = {
            "target_to_source": np.arange(int(tree.ntargets), dtype=np.int32)
        }

    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: int(fmm_order),
        quad_order=int(quad_order),
        self_extra_kwargs=self_extra_kwargs,
    )


def _to_host_array(values, queue: cl.CommandQueue) -> np.ndarray:
    if hasattr(values, "with_queue"):
        return np.asarray(values.with_queue(queue).get())
    if hasattr(values, "get"):
        try:
            return np.asarray(values.get(queue))
        except TypeError:
            return np.asarray(values.get())
    return np.asarray(values)


def _multi_indices_2d(max_order: int):
    for total in range(int(max_order) + 1):
        for nx in range(total + 1):
            yield (nx, total - nx)


def _factorial_multi(alpha: tuple[int, int]) -> int:
    return math.factorial(int(alpha[0])) * math.factorial(int(alpha[1]))


def _laplace_derivative_eval_2d(
    nx: int,
    ny: int,
    x,
    y,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r2 = x * x + y * y

    if int(nx) == 0 and int(ny) == 0:
        return -np.log(r2) / (4.0 * np.pi)

    order = int(nx) + int(ny)
    if order <= 0:
        raise ValueError("derivative order must be non-negative")

    coef = ((-1.0) ** order) * float(math.factorial(order - 1)) / (2.0 * np.pi)
    amp = np.power(r2, -0.5 * order)
    phase = 0.5 * np.pi * int(ny) - order * np.arctan2(y, x)
    return np.asarray(coef * amp * np.cos(phase), dtype=np.float64)


def _kahan_add(total: float, comp: float, increment: float) -> tuple[float, float]:
    y = increment - comp
    t = total + y
    comp = (t - total) - y
    return t, comp


def _kahan_add_array(
    total: np.ndarray,
    comp: np.ndarray,
    increment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = increment - comp
    t = total + y
    comp = (t - total) - y
    return t, comp


def _sum_derivative_hard_square(
    *,
    nu: tuple[int, int],
    cutoff_r: int,
    near_radius: int,
    cell_size: float,
) -> float:
    nx = int(nu[0])
    ny = int(nu[1])

    total = 0.0
    comp = 0.0
    for r in range(int(near_radius) + 1, int(cutoff_r) + 1):
        rr = float(r)

        xs = np.arange(-r, r + 1, dtype=np.float64)
        x_phys = xs * float(cell_size)
        y_top = rr * float(cell_size)
        y_bot = -y_top

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_phys, y_top),
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_phys, y_bot),
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        ys = np.arange(-r + 1, r, dtype=np.float64)
        y_phys = ys * float(cell_size)
        x_right = rr * float(cell_size)
        x_left = -x_right

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_right, y_phys),
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_left, y_phys),
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

    return float(total)


def _sum_derivative_eta_square(
    *,
    nu: tuple[int, int],
    eta: float,
    cutoff_r: int,
    near_radius: int,
    cell_size: float,
) -> float:
    nx = int(nu[0])
    ny = int(nu[1])

    total = 0.0
    comp = 0.0
    for r in range(int(near_radius) + 1, int(cutoff_r) + 1):
        rr = float(r)

        xs = np.arange(-r, r + 1, dtype=np.float64)
        x_phys = xs * float(cell_size)
        y_top = rr * float(cell_size)
        y_bot = -y_top

        w_top = np.exp(-float(eta) * (x_phys * x_phys + y_top * y_top))
        w_bot = np.exp(-float(eta) * (x_phys * x_phys + y_bot * y_bot))

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_phys, y_top) * w_top,
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_phys, y_bot) * w_bot,
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        ys = np.arange(-r + 1, r, dtype=np.float64)
        y_phys = ys * float(cell_size)
        x_right = rr * float(cell_size)
        x_left = -x_right

        w_right = np.exp(-float(eta) * (x_right * x_right + y_phys * y_phys))
        w_left = np.exp(-float(eta) * (x_left * x_left + y_phys * y_phys))

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_right, y_phys) * w_right,
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

        shell_inc = float(
            np.sum(
                _laplace_derivative_eval_2d(nx, ny, x_left, y_phys) * w_left,
                dtype=np.float64,
            )
        )
        total, comp = _kahan_add(total, comp, shell_inc)

    return float(total)


def _estimate_low_order_regularized_sum(
    *,
    nu: tuple[int, int],
    near_radius: int,
    cell_size: float,
    tail_exp: float,
    eta_fit_points: int,
) -> tuple[float, dict]:
    etas = np.asarray(
        [
            1.0e-2,
            5.0e-3,
            2.0e-3,
            1.0e-3,
            5.0e-4,
            2.0e-4,
            1.0e-4,
            5.0e-5,
            2.0e-5,
            1.0e-5,
            5.0e-6,
            2.0e-6,
            1.0e-6,
        ],
        dtype=np.float64,
    )

    values = []
    tail_checks = []
    t_start = time.perf_counter()
    for eta in etas:
        cutoff_r = max(
            int(near_radius) + 8,
            int(np.ceil(np.sqrt(float(tail_exp) / (float(eta) * float(cell_size) ** 2)))),
        )
        cutoff_r2 = int(np.ceil(1.15 * cutoff_r))
        sval_r = _sum_derivative_eta_square(
            nu=nu,
            eta=float(eta),
            cutoff_r=cutoff_r,
            near_radius=near_radius,
            cell_size=cell_size,
        )
        sval_r2 = _sum_derivative_eta_square(
            nu=nu,
            eta=float(eta),
            cutoff_r=cutoff_r2,
            near_radius=near_radius,
            cell_size=cell_size,
        )
        values.append(sval_r2)
        tail_checks.append(abs(sval_r2 - sval_r))

    values = np.asarray(values, dtype=np.float64)
    nfit = min(int(eta_fit_points), len(etas))
    fit_eta = etas[-nfit:]
    fit_values = values[-nfit:]

    design = np.column_stack(
        [
            np.ones_like(fit_eta),
            fit_eta * np.log(fit_eta),
            fit_eta,
            fit_eta * fit_eta * np.log(fit_eta),
            fit_eta * fit_eta,
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, fit_values, rcond=None)
    estimate = float(coeffs[0])

    info = {
        "method": "eta_regularized",
        "fit_points": int(nfit),
        "max_tail_check": float(np.max(np.asarray(tail_checks, dtype=np.float64))),
        "elapsed_s": float(time.perf_counter() - t_start),
        "etas": etas,
        "values": values,
    }
    return estimate, info


def _estimate_high_order_hard_sum(
    *,
    nu: tuple[int, int],
    near_radius: int,
    cell_size: float,
    start_r: int,
    max_r: int,
    self_tol: float,
) -> tuple[float, dict]:
    order = int(sum(nu))
    p = max(1, order - 2)

    eval_cache: dict[int, float] = {}

    def get_sum(cutoff_r: int) -> float:
        if cutoff_r not in eval_cache:
            eval_cache[cutoff_r] = _sum_derivative_hard_square(
                nu=nu,
                cutoff_r=cutoff_r,
                near_radius=near_radius,
                cell_size=cell_size,
            )
        return eval_cache[cutoff_r]

    t_start = time.perf_counter()
    r = max(int(start_r), int(near_radius) + 8)
    prev_estimate = None

    while True:
        s1 = get_sum(r)
        s2 = get_sum(2 * r)
        s4 = get_sum(4 * r)

        rich12 = (2.0**p * s2 - s1) / (2.0**p - 1.0)
        rich24 = (2.0**p * s4 - s2) / (2.0**p - 1.0)

        q = p + 2
        estimate = (2.0**q * rich24 - rich12) / (2.0**q - 1.0)

        if prev_estimate is not None and abs(estimate - prev_estimate) <= float(self_tol):
            break

        if 4 * r >= int(max_r):
            break

        prev_estimate = estimate
        r *= 2

    info = {
        "method": "hard_richardson",
        "p": int(p),
        "final_r": int(r),
        "final_4r": int(4 * r),
        "n_evals": int(len(eval_cache)),
        "self_delta": float(0.0 if prev_estimate is None else abs(estimate - prev_estimate)),
        "elapsed_s": float(time.perf_counter() - t_start),
    }
    return float(estimate), info


@lru_cache(maxsize=None)
def _sigma_power(n: int, power: int) -> int:
    n = int(n)
    power = int(power)

    total = 0
    root = int(math.isqrt(n))
    for divisor in range(1, root + 1):
        if n % divisor != 0:
            continue
        quotient = n // divisor
        total += divisor**power
        if quotient != divisor:
            total += quotient**power
    return int(total)


def _eisenstein_square_lattice_sum_even(
    *,
    order: int,
    mp_dps: int,
) -> float:
    order = int(order)
    if order <= 2 or order % 2 == 1:
        raise ValueError("eisenstein helper requires even order > 2")

    # For the square lattice, rotational symmetry by i forces G_m=0 when
    # m mod 4 = 2.
    if order % 4 == 2:
        return 0.0

    try:
        import mpmath
    except ImportError as exc:
        raise RuntimeError(
            "mpmath is required for --high-order-method=eisenstein"
        ) from exc

    old_dps = mpmath.mp.dps
    mpmath.mp.dps = max(30, int(mp_dps))
    try:
        k = order // 2
        q = mpmath.e ** (-2 * mpmath.pi)
        bernoulli = mpmath.bernoulli(2 * k)
        divisor_power = 2 * k - 1

        sigma_series = mpmath.mpf("0")
        tol = mpmath.power(10, -(int(mpmath.mp.dps) + 10))

        n = 1
        small_term_count = 0
        while True:
            term = mpmath.mpf(_sigma_power(n, divisor_power)) * (q**n)
            sigma_series += term

            if n >= 8 and mpmath.fabs(term) <= tol:
                small_term_count += 1
            else:
                small_term_count = 0

            if small_term_count >= 6:
                break

            if n >= 10000:
                raise RuntimeError(
                    "failed to converge Eisenstein q-series within 10000 terms"
                )

            n += 1

        e_2k = 1 - (4 * k / bernoulli) * sigma_series
        g_2k = 2 * mpmath.zeta(2 * k) * e_2k
        return float(g_2k)
    finally:
        mpmath.mp.dps = old_dps


def _estimate_high_order_eisenstein_sum(
    *,
    nu: tuple[int, int],
    near_radius: int,
    cell_size: float,
    mp_dps: int,
) -> tuple[float, dict]:
    order = int(sum(nu))
    if order <= 2 or order % 2 == 1:
        raise ValueError("eisenstein estimator requires even order > 2")

    t_start = time.perf_counter()
    g_lattice = _eisenstein_square_lattice_sum_even(order=order, mp_dps=int(mp_dps))

    phase = (1j) ** int(nu[1])
    prefactor = ((-1.0) ** order) * float(math.factorial(order - 1)) / (2.0 * np.pi)
    full_lattice_sum = (
        prefactor
        * float(np.real(phase * complex(g_lattice, 0.0)))
        * (float(cell_size) ** (-order))
    )

    near_prefix = 0.0
    if int(near_radius) > 0:
        near_prefix = _sum_derivative_hard_square(
            nu=nu,
            cutoff_r=int(near_radius),
            near_radius=0,
            cell_size=cell_size,
        )

    estimate = float(full_lattice_sum - near_prefix)
    info = {
        "method": "eisenstein",
        "order": int(order),
        "mp_dps": int(mp_dps),
        "full_lattice": float(full_lattice_sum),
        "near_prefix": float(near_prefix),
        "elapsed_s": float(time.perf_counter() - t_start),
    }
    return estimate, info


def build_periodic_tail_coefficients_2d(
    *,
    max_order: int,
    near_radius: int,
    cell_size: float,
    tail_exp: float,
    eta_fit_points: int,
    high_order_start_r: int,
    high_order_max_r: int,
    high_order_selftol: float,
    high_order_method: str = "eisenstein",
    high_order_mp_dps: int = 80,
) -> tuple[dict[tuple[tuple[int, int], tuple[int, int]], float], dict[tuple[int, int], dict]]:
    max_order = int(max_order)
    near_radius = int(near_radius)
    cell_size = float(cell_size)

    derivative_sums: dict[tuple[int, int], float] = {}
    diagnostics: dict[tuple[int, int], dict] = {}

    for nu in _multi_indices_2d(max_order):
        order = int(sum(nu))

        if order % 2 == 1:
            derivative_sums[nu] = 0.0
            diagnostics[nu] = {
                "method": "parity_zero",
                "elapsed_s": 0.0,
            }
            continue

        if order <= 2:
            estimate, info = _estimate_low_order_regularized_sum(
                nu=nu,
                near_radius=near_radius,
                cell_size=cell_size,
                tail_exp=tail_exp,
                eta_fit_points=eta_fit_points,
            )
        else:
            if high_order_method == "hard_richardson":
                estimate, info = _estimate_high_order_hard_sum(
                    nu=nu,
                    near_radius=near_radius,
                    cell_size=cell_size,
                    start_r=high_order_start_r,
                    max_r=high_order_max_r,
                    self_tol=high_order_selftol,
                )
            elif high_order_method == "eisenstein":
                estimate, info = _estimate_high_order_eisenstein_sum(
                    nu=nu,
                    near_radius=near_radius,
                    cell_size=cell_size,
                    mp_dps=high_order_mp_dps,
                )
            else:
                raise ValueError(f"unknown high-order method: {high_order_method}")

        derivative_sums[nu] = float(estimate)
        diagnostics[nu] = info

    coeffs: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for beta in _multi_indices_2d(max_order):
        bsum = int(sum(beta))
        bfac = _factorial_multi(beta)
        for alpha in _multi_indices_2d(max_order):
            asum = int(sum(alpha))
            if asum + bsum > max_order:
                continue

            nu = (int(alpha[0] + beta[0]), int(alpha[1] + beta[1]))
            coeffs[(beta, alpha)] = float(
                ((-1) ** asum) * derivative_sums[nu] / (_factorial_multi(alpha) * bfac)
            )

    return coeffs, diagnostics


def _powers(values: np.ndarray, max_order: int, *, dtype=np.float64) -> np.ndarray:
    values = np.asarray(values, dtype=dtype)
    out = np.empty((int(max_order) + 1, values.size), dtype=dtype)
    out[0, :] = dtype(1.0)
    for iorder in range(1, int(max_order) + 1):
        out[iorder, :] = out[iorder - 1, :] * values
    return out


def evaluate_tail_from_coefficients_2d(
    *,
    source_points: np.ndarray,
    source_strengths: np.ndarray,
    target_points: np.ndarray,
    coeffs: dict[tuple[tuple[int, int], tuple[int, int]], float],
    max_order: int,
    center: np.ndarray,
    use_extended_precision: bool = False,
) -> np.ndarray:
    work_dtype = np.longdouble if bool(use_extended_precision) else np.float64

    source_points = np.asarray(source_points, dtype=work_dtype)
    source_strengths = np.asarray(source_strengths, dtype=work_dtype)
    target_points = np.asarray(target_points, dtype=work_dtype)
    center = np.asarray(center, dtype=work_dtype)

    sx = source_points[:, 0] - center[0]
    sy = source_points[:, 1] - center[1]
    tx = target_points[:, 0] - center[0]
    ty = target_points[:, 1] - center[1]

    sx_pow = _powers(sx, max_order, dtype=work_dtype)
    sy_pow = _powers(sy, max_order, dtype=work_dtype)
    tx_pow = _powers(tx, max_order, dtype=work_dtype)
    ty_pow = _powers(ty, max_order, dtype=work_dtype)

    moments: dict[tuple[int, int], np.longdouble | np.float64] = {}
    for alpha in _multi_indices_2d(max_order):
        ax, ay = alpha
        moments[alpha] = np.dot(source_strengths, sx_pow[ax, :] * sy_pow[ay, :])

    beta_coeff: dict[tuple[int, int], np.longdouble | np.float64] = {}
    for beta in _multi_indices_2d(max_order):
        bsum = int(sum(beta))
        accum = work_dtype(0.0)
        accum_comp = work_dtype(0.0)
        for alpha in _multi_indices_2d(max_order):
            if int(sum(alpha)) + bsum > max_order:
                continue
            term = work_dtype(coeffs[(beta, alpha)]) * moments[alpha]
            accum, accum_comp = _kahan_add(accum, accum_comp, term)
        beta_coeff[beta] = accum

    tail = np.zeros(target_points.shape[0], dtype=work_dtype)
    tail_comp = np.zeros_like(tail)
    for beta in _multi_indices_2d(max_order):
        bx, by = beta
        term = beta_coeff[beta] * tx_pow[bx, :] * ty_pow[by, :]
        y = term - tail_comp
        t = tail + y
        tail_comp = (t - tail) - y
        tail = t

    return np.asarray(tail, dtype=np.float64)


def _dipole_linear_correction_2d(
    *,
    source_points: np.ndarray,
    source_strengths: np.ndarray,
    target_points: np.ndarray,
    center: np.ndarray,
    cell_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float64)
    source_strengths = np.asarray(source_strengths, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    sx = source_points[:, 0] - center[0]
    sy = source_points[:, 1] - center[1]
    dipole = np.asarray(
        [
            np.dot(source_strengths, sx),
            np.dot(source_strengths, sy),
        ],
        dtype=np.float64,
    )

    tx = target_points[:, 0] - center[0]
    ty = target_points[:, 1] - center[1]
    correction = -0.5 * (dipole[0] * tx + dipole[1] * ty) / float(cell_area)
    return correction, dipole


def _a_priori_affine_gauge_coeffs_2d(
    *,
    values: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    source_dipole: np.ndarray,
    cell_area: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    source_dipole = np.asarray(source_dipole, dtype=np.float64)

    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    ax = -0.5 * float(source_dipole[0]) / float(cell_area)
    ay = -0.5 * float(source_dipole[1]) / float(cell_area)
    c0 = -float(np.mean(values + ax * dx + ay * dy))
    return np.asarray([c0, ax, ay], dtype=np.float64)


def _relative_l2_error(values: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(reference))
    if denom <= 0:
        denom = 1.0
    return float(np.linalg.norm(values - reference) / denom)


def _affine_corrected_error_2d(
    *,
    values: np.ndarray,
    reference: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
) -> dict[str, np.ndarray | float]:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    if points.shape[1] != 2:
        raise ValueError("affine correction helper expects 2D points")

    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    design = np.column_stack([np.ones_like(dx), dx, dy])

    coeffs, *_ = np.linalg.lstsq(design, reference - values, rcond=None)
    corrected = values + design @ coeffs

    return {
        "coeffs": np.asarray(coeffs, dtype=np.float64),
        "rel_l2": _relative_l2_error(corrected, reference),
        "linf": float(np.max(np.abs(corrected - reference))),
    }


def _build_probe_points(cell_size: float, grid_size: int) -> np.ndarray:
    grid_size = int(grid_size)
    if grid_size < 8:
        raise ValueError("probe-grid-size must be >= 8")

    x = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * float(cell_size)
    y = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * float(cell_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.ascontiguousarray(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=np.float64)


def _laplace_kernel_2d(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    dtype=np.float64,
) -> np.ndarray:
    return np.asarray(-np.log(dx * dx + dy * dy) / (4.0 * np.pi), dtype=dtype)


def _direct_image_sum_square_2d(
    *,
    src_points: np.ndarray,
    strengths: np.ndarray,
    tgt_points: np.ndarray,
    cell_size: np.ndarray,
    min_radius: int,
    max_radius: int,
    source_block_size: int = 0,
    use_extended_precision: bool = False,
) -> np.ndarray:
    work_dtype = np.longdouble if bool(use_extended_precision) else np.float64

    src_points = np.asarray(src_points, dtype=work_dtype)
    strengths = np.asarray(strengths, dtype=work_dtype)
    tgt_points = np.asarray(tgt_points, dtype=work_dtype)
    cell_size = np.asarray(cell_size, dtype=work_dtype)

    source_block_size = int(source_block_size)
    if source_block_size < 0:
        raise ValueError("direct-sum-source-block-size must be >= 0")

    if src_points.shape[1] != 2 or tgt_points.shape[1] != 2:
        raise ValueError("direct image-sum helper is implemented for dim=2 only")

    min_radius = int(min_radius)
    max_radius = int(max_radius)
    if max_radius < min_radius:
        return np.zeros(tgt_points.shape[0], dtype=np.float64)

    def add_shift(
        accum: np.ndarray,
        accum_comp: np.ndarray,
        ix: int,
        iy: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        shift_x = work_dtype(ix) * cell_size[0]
        shift_y = work_dtype(iy) * cell_size[1]

        if source_block_size <= 0:
            dx = tgt_points[:, None, 0] - (src_points[None, :, 0] + shift_x)
            dy = tgt_points[:, None, 1] - (src_points[None, :, 1] + shift_y)
            increment = _laplace_kernel_2d(dx, dy, dtype=work_dtype) @ strengths
            increment = np.asarray(increment, dtype=work_dtype)
        else:
            increment = np.zeros(tgt_points.shape[0], dtype=work_dtype)
            increment_comp = np.zeros_like(increment)
            for start in range(0, src_points.shape[0], source_block_size):
                stop = min(src_points.shape[0], start + source_block_size)
                src_blk = src_points[start:stop]
                str_blk = strengths[start:stop]

                dx = tgt_points[:, None, 0] - (src_blk[None, :, 0] + shift_x)
                dy = tgt_points[:, None, 1] - (src_blk[None, :, 1] + shift_y)
                inc_blk = _laplace_kernel_2d(dx, dy, dtype=work_dtype) @ str_blk
                inc_blk = np.asarray(inc_blk, dtype=work_dtype)
                increment, increment_comp = _kahan_add_array(
                    increment,
                    increment_comp,
                    inc_blk,
                )

        return _kahan_add_array(accum, accum_comp, increment)

    def sum_shell(radius: int) -> np.ndarray:
        radius = int(radius)
        shell = np.zeros(tgt_points.shape[0], dtype=work_dtype)
        shell_comp = np.zeros_like(shell)

        for ix in range(-radius, radius + 1):
            shell, shell_comp = add_shift(shell, shell_comp, ix, radius)
            if radius > 0:
                shell, shell_comp = add_shift(shell, shell_comp, ix, -radius)

        if radius > 0:
            for iy in range(-radius + 1, radius):
                shell, shell_comp = add_shift(shell, shell_comp, radius, iy)
                shell, shell_comp = add_shift(shell, shell_comp, -radius, iy)

        return shell

    out = np.zeros(tgt_points.shape[0], dtype=work_dtype)
    comp = np.zeros_like(out)
    for radius in range(min_radius, max_radius + 1):
        shell = sum_shell(radius)

        out, comp = _kahan_add_array(out, comp, shell)

    return np.asarray(out, dtype=np.float64)


def _fit_inverse_even_power_limit(
    *,
    radii: list[int],
    values: list[np.ndarray],
    fit_order: int,
) -> np.ndarray:
    if len(radii) != len(values):
        raise ValueError("radii and values must have the same length")
    if len(radii) < 2:
        raise ValueError("need at least two radii for far-limit extrapolation")

    npts = int(np.asarray(values[0]).size)
    stacked = np.empty((len(values), npts), dtype=np.float64)
    for i, vec in enumerate(values):
        vec = np.asarray(vec, dtype=np.float64)
        if vec.size != npts:
            raise ValueError("all extrapolation vectors must have the same length")
        stacked[i, :] = vec.reshape(-1)

    radii_arr = np.asarray(radii, dtype=np.float64)
    max_fit_order = max(1, min(int(fit_order), len(radii) - 1))
    design = [np.ones_like(radii_arr)]
    for iorder in range(1, max_fit_order + 1):
        design.append(np.power(radii_arr, -2 * iorder))
    design_mat = np.column_stack(design)

    coeffs, *_ = np.linalg.lstsq(design_mat, stacked, rcond=None)
    return np.asarray(coeffs[0, :], dtype=np.float64)


def _host_points_to_obj_array(points: np.ndarray, queue: cl.CommandQueue):
    points = np.asarray(points, dtype=np.float64)
    return obj_array_1d(
        [
            cl.array.to_device(queue, np.ascontiguousarray(points[:, 0])),
            cl.array.to_device(queue, np.ascontiguousarray(points[:, 1])),
        ]
    )


def _evaluate_hybrid_fmm_on_points(
    *,
    points: np.ndarray,
    queue: cl.CommandQueue,
    traversal,
    wrangler,
    central_tree_potential,
    source_points: np.ndarray,
    source_strengths_host: np.ndarray,
    near_radius: int,
    cell_size: float,
    coeffs: dict[tuple[tuple[int, int], tuple[int, int]], float],
    max_order: int,
    center: np.ndarray,
    tail_eval_extended_precision: bool,
    direct_sum_source_block_size: int,
    direct_sum_extended_precision: bool,
) -> np.ndarray:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    points_dev = _host_points_to_obj_array(points, queue)

    pot_central = interpolate_volume_potential(
        points_dev,
        traversal,
        wrangler,
        central_tree_potential,
        potential_in_tree_order=True,
        use_mode_to_source_ids=True,
    )
    pot_central_host = _to_host_array(pot_central, queue).astype(np.float64)

    if int(near_radius) > 0:
        pot_near_host = _direct_image_sum_square_2d(
            src_points=source_points,
            strengths=source_strengths_host,
            tgt_points=points,
            cell_size=np.asarray([float(cell_size), float(cell_size)], dtype=np.float64),
            min_radius=1,
            max_radius=int(near_radius),
            source_block_size=int(direct_sum_source_block_size),
            use_extended_precision=bool(direct_sum_extended_precision),
        )
    else:
        pot_near_host = np.zeros(len(points), dtype=np.float64)

    pot_tail_host = evaluate_tail_from_coefficients_2d(
        source_points=source_points,
        source_strengths=source_strengths_host,
        target_points=points,
        coeffs=coeffs,
        max_order=int(max_order),
        center=center,
        use_extended_precision=bool(tail_eval_extended_precision),
    )

    return np.asarray(pot_central_host + pot_near_host + pot_tail_host, dtype=np.float64)


def _evaluate_hybrid_direct_on_points(
    *,
    points: np.ndarray,
    source_points: np.ndarray,
    source_strengths_host: np.ndarray,
    near_radius: int,
    cell_size: float,
    coeffs: dict[tuple[tuple[int, int], tuple[int, int]], float],
    max_order: int,
    center: np.ndarray,
    tail_eval_extended_precision: bool,
    direct_sum_source_block_size: int,
    direct_sum_extended_precision: bool,
) -> np.ndarray:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))

    pot_central_host = _direct_image_sum_square_2d(
        src_points=source_points,
        strengths=source_strengths_host,
        tgt_points=points,
        cell_size=np.asarray([float(cell_size), float(cell_size)], dtype=np.float64),
        min_radius=0,
        max_radius=0,
        source_block_size=int(direct_sum_source_block_size),
        use_extended_precision=bool(direct_sum_extended_precision),
    )

    if int(near_radius) > 0:
        pot_near_host = _direct_image_sum_square_2d(
            src_points=source_points,
            strengths=source_strengths_host,
            tgt_points=points,
            cell_size=np.asarray([float(cell_size), float(cell_size)], dtype=np.float64),
            min_radius=1,
            max_radius=int(near_radius),
            source_block_size=int(direct_sum_source_block_size),
            use_extended_precision=bool(direct_sum_extended_precision),
        )
    else:
        pot_near_host = np.zeros(len(points), dtype=np.float64)

    pot_tail_host = evaluate_tail_from_coefficients_2d(
        source_points=source_points,
        source_strengths=source_strengths_host,
        target_points=points,
        coeffs=coeffs,
        max_order=int(max_order),
        center=center,
        use_extended_precision=bool(tail_eval_extended_precision),
    )

    return np.asarray(pot_central_host + pot_near_host + pot_tail_host, dtype=np.float64)


def _periodicity_pair_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    jump = np.asarray(rhs - lhs, dtype=np.float64)
    scale = max(float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)), 1.0)
    return {
        "rel_l2": float(np.linalg.norm(jump) / scale),
        "abs_l2": float(np.linalg.norm(jump)),
        "linf": float(np.max(np.abs(jump))),
        "mean_abs": float(np.mean(np.abs(jump))),
    }


def _fd_gradient_2d(
    evaluate_potential,
    points: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    h = float(h)

    pxp = points.copy()
    pxm = points.copy()
    pyp = points.copy()
    pym = points.copy()

    pxp[:, 0] += h
    pxm[:, 0] -= h
    pyp[:, 1] += h
    pym[:, 1] -= h

    stacked = np.ascontiguousarray(np.vstack([pxp, pxm, pyp, pym]), dtype=np.float64)
    vals = np.asarray(evaluate_potential(stacked), dtype=np.float64)
    npts = points.shape[0]

    ux = (vals[0:npts] - vals[npts:2 * npts]) / (2.0 * h)
    uy = (vals[2 * npts:3 * npts] - vals[3 * npts:4 * npts]) / (2.0 * h)
    return np.asarray(ux, dtype=np.float64), np.asarray(uy, dtype=np.float64)


def _run_periodicity_jump_check_2d(
    *,
    evaluate_potential,
    cell_size: float,
    n_samples: int,
    eps: float,
    fd_h: float,
    translated_pairs: bool,
) -> dict[str, float | dict[str, float]]:
    n_samples = int(n_samples)
    if n_samples < 4:
        raise ValueError("periodicity-samples must be >= 4")

    cell_size = float(cell_size)
    eps = float(eps)
    fd_h = float(fd_h)

    if eps <= 0.0 or eps >= 0.5 * cell_size:
        raise ValueError("periodicity-eps must lie in (0, 0.5*cell_size)")
    if fd_h <= 0.0:
        raise ValueError("periodicity-fd-h must be > 0")

    if fd_h >= eps:
        fd_h = 0.49 * eps

    samples = (
        (np.arange(n_samples, dtype=np.float64) + 0.5)
        / float(n_samples)
        * cell_size
    )

    left = np.column_stack([np.full(n_samples, eps, dtype=np.float64), samples])
    bottom = np.column_stack([samples, np.full(n_samples, eps, dtype=np.float64)])

    if bool(translated_pairs):
        right = left.copy()
        right[:, 0] += cell_size
        top = bottom.copy()
        top[:, 1] += cell_size
    else:
        right = np.column_stack([
            np.full(n_samples, cell_size - eps, dtype=np.float64),
            samples,
        ])
        top = np.column_stack([
            samples,
            np.full(n_samples, cell_size - eps, dtype=np.float64),
        ])

    face_points = np.ascontiguousarray(
        np.vstack([left, right, bottom, top]),
        dtype=np.float64,
    )
    face_values = np.asarray(evaluate_potential(face_points), dtype=np.float64)
    u_left = face_values[0:n_samples]
    u_right = face_values[n_samples:2 * n_samples]
    u_bottom = face_values[2 * n_samples:3 * n_samples]
    u_top = face_values[3 * n_samples:4 * n_samples]

    val_x = _periodicity_pair_metrics(u_left, u_right)
    val_y = _periodicity_pair_metrics(u_bottom, u_top)

    ux_left, uy_left = _fd_gradient_2d(evaluate_potential, left, fd_h)
    ux_right, uy_right = _fd_gradient_2d(evaluate_potential, right, fd_h)
    ux_bottom, uy_bottom = _fd_gradient_2d(evaluate_potential, bottom, fd_h)
    ux_top, uy_top = _fd_gradient_2d(evaluate_potential, top, fd_h)

    dudx_xfaces = _periodicity_pair_metrics(ux_left, ux_right)
    dudy_xfaces = _periodicity_pair_metrics(uy_left, uy_right)
    dudx_yfaces = _periodicity_pair_metrics(ux_bottom, ux_top)
    dudy_yfaces = _periodicity_pair_metrics(uy_bottom, uy_top)

    rels = [
        val_x["rel_l2"],
        val_y["rel_l2"],
        dudx_xfaces["rel_l2"],
        dudy_xfaces["rel_l2"],
        dudx_yfaces["rel_l2"],
        dudy_yfaces["rel_l2"],
    ]
    linfs = [
        val_x["linf"],
        val_y["linf"],
        dudx_xfaces["linf"],
        dudy_xfaces["linf"],
        dudx_yfaces["linf"],
        dudy_yfaces["linf"],
    ]

    return {
        "samples": int(n_samples),
        "eps": float(eps),
        "fd_h": float(fd_h),
        "translated_pairs": bool(translated_pairs),
        "value_xfaces": val_x,
        "value_yfaces": val_y,
        "dudx_xfaces": dudx_xfaces,
        "dudy_xfaces": dudy_xfaces,
        "dudx_yfaces": dudx_yfaces,
        "dudy_yfaces": dudy_yfaces,
        "worst_rel_l2": float(max(rels)),
        "worst_linf": float(max(linfs)),
    }


def _run_calculuspatch_pde_residual_check_2d(
    *,
    evaluate_potential,
    evaluate_source_density,
    patch_center: np.ndarray,
    patch_h: float,
    patch_order: int,
    cell_size: float,
) -> dict[str, float]:
    from sumpy.point_calculus import CalculusPatch

    patch = CalculusPatch(
        center=[float(patch_center[0]), float(patch_center[1])],
        h=float(patch_h),
        order=int(patch_order),
    )
    patch_points = np.ascontiguousarray(
        np.column_stack([
            np.asarray(patch.x, dtype=np.float64),
            np.asarray(patch.y, dtype=np.float64),
        ]),
        dtype=np.float64,
    )

    cell_size = float(cell_size)
    x = patch_points[:, 0]
    y = patch_points[:, 1]
    margin = min(
        float(np.min(x)),
        float(np.min(y)),
        float(np.min(cell_size - x)),
        float(np.min(cell_size - y)),
    )
    if margin <= 0.0:
        raise ValueError(
            "calculus patch leaves the periodic cell; reduce --patch-h or move --patch-center"
        )

    u_patch = evaluate_potential(patch_points)
    rho_patch = evaluate_source_density(patch_points)
    minus_lap = -np.asarray(patch.laplace(u_patch), dtype=np.float64)
    residual = np.asarray(minus_lap - rho_patch, dtype=np.float64)

    rho_norm = max(float(np.linalg.norm(rho_patch)), 1.0)
    return {
        "rel_l2": float(np.linalg.norm(residual) / rho_norm),
        "abs_l2": float(np.linalg.norm(residual)),
        "linf": float(np.max(np.abs(residual))),
        "rho_l2": float(np.linalg.norm(rho_patch)),
        "n_points": int(rho_patch.size),
        "margin_to_boundary": float(margin),
    }


def _verify_low_order_square_pv(
    *,
    near_radius: int,
    cell_size: float,
    eta_value: float,
) -> None:
    nu = (2, 0)
    reg_est, _ = _estimate_low_order_regularized_sum(
        nu=nu,
        near_radius=near_radius,
        cell_size=cell_size,
        tail_exp=40.0,
        eta_fit_points=8,
    )

    s1 = _sum_derivative_hard_square(
        nu=nu,
        cutoff_r=32768,
        near_radius=near_radius,
        cell_size=cell_size,
    )
    s2 = _sum_derivative_hard_square(
        nu=nu,
        cutoff_r=65536,
        near_radius=near_radius,
        cell_size=cell_size,
    )
    square_pv_est = (4.0 * s2 - s1) / 3.0

    print(
        "low-order check nu=(2,0): "
        f"eta-regularized={reg_est:.18e}, "
        f"square-PV-extrap={square_pv_est:.18e}, "
        f"|diff|={abs(reg_est - square_pv_est):.3e}, "
        f"eta_probe={eta_value:.1e}"
    )


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))

    ctx = _create_non_intel_context()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen2D(int(args.q_order), int(args.nlevels), 0.0, 1.0, queue=queue)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        2,
        int(args.q_order),
        mesh,
        bbox=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
    )

    source_points = np.ascontiguousarray(
        np.stack([q_points[0].get(queue), q_points[1].get(queue)], axis=1),
        dtype=np.float64,
    )
    q_weights_host = np.asarray(q_weights.get(queue), dtype=np.float64)

    source_random_terms = _sample_smooth_source_terms(rng)
    source_values_host, source_weighted_mean = _build_smooth_neutral_source_from_terms(
        source_points,
        q_weights_host,
        source_random_terms,
    )
    source_values_dev = cl.array.to_device(queue, source_values_host)
    source_strengths_dev = source_values_dev * q_weights
    source_strengths_host = np.asarray(source_values_host * q_weights_host, dtype=np.float64)

    probe_points = _build_probe_points(cell_size=float(tree.root_extent), grid_size=int(args.probe_grid_size))
    probe_points_dev = _host_points_to_obj_array(probe_points, queue)

    cell_size = float(tree.root_extent)
    center = np.asarray([0.5 * cell_size, 0.5 * cell_size], dtype=np.float64)

    direct_sum_source_block_size = int(args.direct_sum_source_block_size)
    direct_sum_extended_precision = bool(args.direct_sum_extended_precision)
    spectral_compensated = bool(args.spectral_compensated)
    spectral_source_block_size = int(args.spectral_source_block_size)
    spectral_accum_extended_precision = bool(args.spectral_accum_extended_precision)

    t_coeff_start = time.perf_counter()
    coeffs, diagnostics = build_periodic_tail_coefficients_2d(
        max_order=int(args.max_order),
        near_radius=int(args.near_radius),
        cell_size=cell_size,
        tail_exp=float(args.tail_exp),
        eta_fit_points=int(args.eta_fit_points),
        high_order_start_r=int(args.high_order_start_R),
        high_order_max_r=int(args.high_order_max_R),
        high_order_selftol=float(args.high_order_selftol),
        high_order_method=str(args.high_order_method),
        high_order_mp_dps=int(args.high_order_mp_dps),
    )
    t_coeff = time.perf_counter() - t_coeff_start

    if bool(args.verify_low_order_square_pv):
        _verify_low_order_square_pv(
            near_radius=int(args.near_radius),
            cell_size=cell_size,
            eta_value=1.0e-6,
        )

    with NearFieldInteractionTableManager(
        args.cache_file,
        root_extent=float(tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(2, "Laplace", q_order=int(args.q_order), queue=queue)
        wrangler = _build_wrangler(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            near_field_table=table,
            fmm_order=int(args.fmm_order),
            quad_order=int(args.q_order),
        )

        t_fmm_start = time.perf_counter()
        (pot_central_fmm,) = drive_volume_fmm(
            traversal,
            wrangler,
            source_strengths_dev,
            source_values_dev,
            direct_evaluation=False,
            auto_interpolate_targets=False,
            reorder_potentials=False,
        )
        queue.finish()
        t_fmm = time.perf_counter() - t_fmm_start

        pot_central_fmm_probe = interpolate_volume_potential(
            probe_points_dev,
            traversal,
            wrangler,
            pot_central_fmm,
            potential_in_tree_order=True,
            use_mode_to_source_ids=True,
        )
        pot_central_fmm_probe_host = _to_host_array(
            pot_central_fmm_probe,
            queue,
        ).astype(np.float64)

        t_near_start = time.perf_counter()
        if int(args.near_radius) > 0:
            pot_near_probe_host = _direct_image_sum_square_2d(
                src_points=source_points,
                strengths=source_strengths_host,
                tgt_points=probe_points,
                cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                min_radius=1,
                max_radius=int(args.near_radius),
                source_block_size=direct_sum_source_block_size,
                use_extended_precision=direct_sum_extended_precision,
            )
        else:
            pot_near_probe_host = np.zeros(len(probe_points), dtype=np.float64)
        t_near = time.perf_counter() - t_near_start

    t_tail_start = time.perf_counter()
    pot_tail_probe_host = evaluate_tail_from_coefficients_2d(
        source_points=source_points,
        source_strengths=source_strengths_host,
        target_points=probe_points,
        coeffs=coeffs,
        max_order=int(args.max_order),
        center=center,
        use_extended_precision=bool(args.tail_eval_extended_precision),
    )
    t_tail = time.perf_counter() - t_tail_start

    tail_direct_rel = None
    tail_direct_linf = None
    tail_direct_l2 = None
    tail_ref_l2 = None
    far_ref = None
    hybrid_direct_rel = None
    hybrid_direct_linf = None
    direct_far_ref_rel = None
    direct_far_ref_linf = None
    direct_far_radii_used = None
    direct_far_fit_order_used = None
    direct_far_timing = None
    if int(args.direct_far_cutoff_r) > int(args.near_radius):
        t_direct_far_start = time.perf_counter()

        adaptive_selftol = float(args.direct_far_target_selftol)
        fit_order = max(1, int(args.direct_far_fit_order))
        num_radii = max(2, int(args.direct_far_num_radii))

        r_base = int(args.direct_far_cutoff_r)
        r_second = int(args.direct_far_second_cutoff_r)
        if r_second > 0 and r_second <= r_base:
            raise ValueError("direct-far-second-cutoff-r must be > direct-far-cutoff-r")

        rmax = int(args.direct_far_max_cutoff_r)
        if rmax <= 0:
            rmax = max(2 * r_base, 320)

        radii = [int(r_base)]
        if r_second > 0:
            radii.append(int(r_second))
        else:
            radii.append(int(2 * r_base))

        radii = sorted(set(radii))
        while len(radii) < num_radii:
            next_r = min(rmax, 2 * radii[-1])
            if next_r <= radii[-1]:
                break
            radii.append(int(next_r))

        if radii[-1] > rmax:
            raise ValueError("direct-far radii exceed direct-far-max-cutoff-r")

        min_radius = int(args.near_radius) + 1
        far_cache: dict[int, np.ndarray] = {}
        far_running = np.zeros(len(probe_points), dtype=np.float64)
        far_running_comp = np.zeros_like(far_running)
        far_last_radius = min_radius - 1

        def _direct_shell(radius: int) -> np.ndarray:
            return _direct_image_sum_square_2d(
                src_points=source_points,
                strengths=source_strengths_host,
                tgt_points=probe_points,
                cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                min_radius=radius,
                max_radius=radius,
                source_block_size=direct_sum_source_block_size,
                use_extended_precision=direct_sum_extended_precision,
            )

        def get_far(max_radius: int) -> np.ndarray:
            nonlocal far_last_radius, far_running, far_running_comp
            key = int(max_radius)
            if key not in far_cache:
                if key < min_radius:
                    far_cache[key] = np.zeros(len(probe_points), dtype=np.float64)
                    return far_cache[key]

                if key < far_last_radius:
                    far_cache[key] = _direct_image_sum_square_2d(
                        src_points=source_points,
                        strengths=source_strengths_host,
                        tgt_points=probe_points,
                        cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                        min_radius=min_radius,
                        max_radius=key,
                        source_block_size=direct_sum_source_block_size,
                        use_extended_precision=direct_sum_extended_precision,
                    )
                    return far_cache[key]

                for radius in range(far_last_radius + 1, key + 1):
                    shell = _direct_shell(radius)
                    y = shell - far_running_comp
                    t = far_running + y
                    far_running_comp = (t - far_running) - y
                    far_running = t

                far_last_radius = key
                far_cache[key] = np.asarray(far_running, dtype=np.float64).copy()
            return far_cache[key]

        while True:
            current_values = [get_far(rad) for rad in radii]
            far_ref = _fit_inverse_even_power_limit(
                radii=radii,
                values=current_values,
                fit_order=fit_order,
            )

            if len(radii) >= 3:
                prev_radii = radii[:-1]
                prev_values = [get_far(rad) for rad in prev_radii]
                far_prev = _fit_inverse_even_power_limit(
                    radii=prev_radii,
                    values=prev_values,
                    fit_order=fit_order,
                )
                direct_far_ref_rel = _relative_l2_error(far_prev, far_ref)
                direct_far_ref_linf = float(np.max(np.abs(far_prev - far_ref)))
            else:
                direct_far_ref_rel = float("inf")
                direct_far_ref_linf = float("inf")

            direct_far_radii_used = list(radii)
            direct_far_fit_order_used = min(fit_order, len(radii) - 1)

            if adaptive_selftol <= 0:
                break
            if np.isfinite(direct_far_ref_rel) and direct_far_ref_rel <= adaptive_selftol:
                break
            if radii[-1] >= rmax:
                break

            next_r = min(rmax, 2 * radii[-1])
            if next_r <= radii[-1]:
                break
            radii.append(int(next_r))

        direct_far_timing = float(time.perf_counter() - t_direct_far_start)

        tail_direct_rel = _relative_l2_error(pot_tail_probe_host, far_ref)
        tail_direct_l2 = float(np.linalg.norm(pot_tail_probe_host - far_ref))
        tail_ref_l2 = float(np.linalg.norm(far_ref))
        tail_direct_linf = float(np.max(np.abs(pot_tail_probe_host - far_ref)))

        hybrid_fmm_local = pot_central_fmm_probe_host + pot_near_probe_host + pot_tail_probe_host
        direct_full_ref = pot_central_fmm_probe_host + pot_near_probe_host + far_ref
        hybrid_direct_rel = _relative_l2_error(hybrid_fmm_local, direct_full_ref)
        hybrid_direct_linf = float(np.max(np.abs(hybrid_fmm_local - direct_full_ref)))

    pot_hybrid_fmm = pot_central_fmm_probe_host + pot_near_probe_host + pot_tail_probe_host

    dipole_linear_corr, source_dipole = _dipole_linear_correction_2d(
        source_points=source_points,
        source_strengths=source_strengths_host,
        target_points=probe_points,
        center=center,
        cell_area=cell_size * cell_size,
    )

    pot_hybrid_fmm_dipole = pot_hybrid_fmm + dipole_linear_corr
    pot_hybrid_fmm_dipole -= np.mean(pot_hybrid_fmm_dipole)

    spectral_chunk_size = int(args.spectral_chunk_size)
    t_ref_start = time.perf_counter()
    pot_ref = _spectral_periodic_laplace_reference(
        src_points=source_points,
        strengths=source_strengths_host,
        tgt_points=probe_points,
        cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
        k_max=int(args.kmax),
        chunk_size=spectral_chunk_size,
        compensated=spectral_compensated,
        source_block_size=spectral_source_block_size,
        accum_extended_precision=spectral_accum_extended_precision,
    )
    t_ref = time.perf_counter() - t_ref_start

    err_fmm_rel = _relative_l2_error(pot_hybrid_fmm, pot_ref)
    err_fmm_linf = float(np.max(np.abs(pot_hybrid_fmm - pot_ref)))
    err_fmm_dipole_rel = _relative_l2_error(pot_hybrid_fmm_dipole, pot_ref)
    err_fmm_dipole_linf = float(np.max(np.abs(pot_hybrid_fmm_dipole - pot_ref)))

    tail_pde_rel = None
    tail_pde_linf = None
    tail_pde_l2 = None
    tail_pde_ref_l2 = None
    tail_pde_rel_zeromean = None
    tail_pde_linf_zeromean = None
    tail_pde_affine_rel = None
    tail_pde_affine_linf = None
    tail_pde_affine_coeffs = None
    pde_direct_far_rel = None
    pde_direct_far_linf = None
    pde_direct_far_rel_zeromean = None
    pde_direct_far_linf_zeromean = None
    pde_direct_far_affine_rel = None
    pde_direct_far_affine_linf = None
    pde_direct_far_affine_coeffs = None
    pde_reference_method = str(args.pde_reference_method)
    pde_ref_kmax = int(args.pde_reference_kmax)
    if pde_ref_kmax <= 0:
        pde_ref_kmax = int(args.kmax)
    if bool(args.pde_far_reference):
        if pde_reference_method == "spectral":
            if pde_ref_kmax == int(args.kmax):
                pde_full_ref = pot_ref
            else:
                pde_full_ref = _spectral_periodic_laplace_reference(
                    src_points=source_points,
                    strengths=source_strengths_host,
                    tgt_points=probe_points,
                    cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                    k_max=int(pde_ref_kmax),
                    chunk_size=spectral_chunk_size,
                    compensated=spectral_compensated,
                    source_block_size=spectral_source_block_size,
                    accum_extended_precision=spectral_accum_extended_precision,
                )
        elif pde_reference_method == "ewald":
            pde_full_ref = _ewald_periodic_laplace_reference(
                src_points=source_points,
                strengths=source_strengths_host,
                tgt_points=probe_points,
                cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                xi=float(args.ewald_xi),
                real_cutoff_r=int(args.ewald_real_cutoff_r),
                k_max=int(args.ewald_kmax),
                mode_chunk_size=spectral_chunk_size,
            )
        else:
            raise ValueError(f"unknown pde-reference-method: {pde_reference_method}")

        if bool(args.pde_reference_apply_dipole):
            pde_full_model = pde_full_ref - dipole_linear_corr
        else:
            pde_full_model = pde_full_ref

        pde_far_ref = pde_full_model - pot_central_fmm_probe_host - pot_near_probe_host
        tail_pde_rel = _relative_l2_error(pot_tail_probe_host, pde_far_ref)
        tail_pde_linf = float(np.max(np.abs(pot_tail_probe_host - pde_far_ref)))
        tail_pde_l2 = float(np.linalg.norm(pot_tail_probe_host - pde_far_ref))
        tail_pde_ref_l2 = float(np.linalg.norm(pde_far_ref))

        tail_zm = pot_tail_probe_host - np.mean(pot_tail_probe_host)
        pde_far_zm = pde_far_ref - np.mean(pde_far_ref)
        tail_pde_rel_zeromean = _relative_l2_error(tail_zm, pde_far_zm)
        tail_pde_linf_zeromean = float(np.max(np.abs(tail_zm - pde_far_zm)))

        aff_pde = _affine_corrected_error_2d(
            values=pot_tail_probe_host,
            reference=pde_far_ref,
            points=probe_points,
            center=center,
        )
        tail_pde_affine_rel = float(aff_pde["rel_l2"])
        tail_pde_affine_linf = float(aff_pde["linf"])
        tail_pde_affine_coeffs = np.asarray(aff_pde["coeffs"], dtype=np.float64)

        if far_ref is not None:
            pde_direct_far_rel = _relative_l2_error(pde_far_ref, far_ref)
            pde_direct_far_linf = float(np.max(np.abs(pde_far_ref - far_ref)))

            pde_far_zm2 = pde_far_ref - np.mean(pde_far_ref)
            far_ref_zm = far_ref - np.mean(far_ref)
            pde_direct_far_rel_zeromean = _relative_l2_error(pde_far_zm2, far_ref_zm)
            pde_direct_far_linf_zeromean = float(np.max(np.abs(pde_far_zm2 - far_ref_zm)))

            aff_pde_direct = _affine_corrected_error_2d(
                values=pde_far_ref,
                reference=far_ref,
                points=probe_points,
                center=center,
            )
            pde_direct_far_affine_rel = float(aff_pde_direct["rel_l2"])
            pde_direct_far_affine_linf = float(aff_pde_direct["linf"])
            pde_direct_far_affine_coeffs = np.asarray(
                aff_pde_direct["coeffs"],
                dtype=np.float64,
            )

    affine_fmm = _affine_corrected_error_2d(
        values=pot_hybrid_fmm,
        reference=pot_ref,
        points=probe_points,
        center=center,
    )

    a_priori_fmm_coeffs = _a_priori_affine_gauge_coeffs_2d(
        values=pot_hybrid_fmm,
        points=probe_points,
        center=center,
        source_dipole=source_dipole,
        cell_area=cell_size * cell_size,
    )
    dx_probe = probe_points[:, 0] - center[0]
    dy_probe = probe_points[:, 1] - center[1]
    pot_hybrid_fmm_apriori = (
        pot_hybrid_fmm
        + a_priori_fmm_coeffs[0]
        + a_priori_fmm_coeffs[1] * dx_probe
        + a_priori_fmm_coeffs[2] * dy_probe
    )
    a_priori_fmm_rel = _relative_l2_error(pot_hybrid_fmm_apriori, pot_ref)
    a_priori_fmm_linf = float(np.max(np.abs(pot_hybrid_fmm_apriori - pot_ref)))
    a_priori_vs_fit_coeff_delta = np.asarray(
        a_priori_fmm_coeffs - np.asarray(affine_fmm["coeffs"], dtype=np.float64),
        dtype=np.float64,
    )

    def evaluate_hybrid_on_points(points: np.ndarray) -> np.ndarray:
        return _evaluate_hybrid_fmm_on_points(
            points=points,
            queue=queue,
            traversal=traversal,
            wrangler=wrangler,
            central_tree_potential=pot_central_fmm,
            source_points=source_points,
            source_strengths_host=source_strengths_host,
            near_radius=int(args.near_radius),
            cell_size=cell_size,
            coeffs=coeffs,
            max_order=int(args.max_order),
            center=center,
            tail_eval_extended_precision=bool(args.tail_eval_extended_precision),
            direct_sum_source_block_size=direct_sum_source_block_size,
            direct_sum_extended_precision=direct_sum_extended_precision,
        )

    def evaluate_hybrid_direct_on_points(points: np.ndarray) -> np.ndarray:
        return _evaluate_hybrid_direct_on_points(
            points=points,
            source_points=source_points,
            source_strengths_host=source_strengths_host,
            near_radius=int(args.near_radius),
            cell_size=cell_size,
            coeffs=coeffs,
            max_order=int(args.max_order),
            center=center,
            tail_eval_extended_precision=bool(args.tail_eval_extended_precision),
            direct_sum_source_block_size=direct_sum_source_block_size,
            direct_sum_extended_precision=direct_sum_extended_precision,
        )

    periodicity_gauge_mode = str(args.periodicity_gauge)
    if periodicity_gauge_mode == "none":
        periodicity_gauge_coeffs = None
    elif periodicity_gauge_mode == "dipole":
        periodicity_gauge_coeffs = np.asarray(a_priori_fmm_coeffs, dtype=np.float64)
    elif periodicity_gauge_mode == "affine":
        periodicity_gauge_coeffs = np.asarray(affine_fmm["coeffs"], dtype=np.float64)
    else:
        raise ValueError(f"unknown periodicity-gauge: {periodicity_gauge_mode}")

    periodicity_evaluator_mode = str(args.periodicity_evaluator)

    def evaluate_hybrid_for_periodicity(points: np.ndarray) -> np.ndarray:
        if periodicity_evaluator_mode == "direct":
            values = evaluate_hybrid_direct_on_points(points)
        elif periodicity_evaluator_mode == "fmm":
            values = evaluate_hybrid_on_points(points)
        else:
            raise ValueError(
                f"unknown periodicity-evaluator: {periodicity_evaluator_mode}"
            )

        if periodicity_gauge_coeffs is None:
            return np.asarray(values, dtype=np.float64)

        pts = np.asarray(points, dtype=np.float64)
        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]
        corrected = (
            np.asarray(values, dtype=np.float64)
            + periodicity_gauge_coeffs[0]
            + periodicity_gauge_coeffs[1] * dx
            + periodicity_gauge_coeffs[2] * dy
        )
        return np.asarray(corrected, dtype=np.float64)

    def evaluate_source_density(points: np.ndarray) -> np.ndarray:
        values, _ = _build_smooth_neutral_source_from_terms(
            points,
            None,
            source_random_terms,
            weighted_mean=source_weighted_mean,
        )
        return values

    calculuspatch_check = None
    t_calculuspatch = None
    if bool(args.calculuspatch_pde_check):
        t_patch_start = time.perf_counter()
        calculuspatch_check = _run_calculuspatch_pde_residual_check_2d(
            evaluate_potential=evaluate_hybrid_on_points,
            evaluate_source_density=evaluate_source_density,
            patch_center=np.asarray(
                [float(args.patch_center_x), float(args.patch_center_y)],
                dtype=np.float64,
            ),
            patch_h=float(args.patch_h),
            patch_order=int(args.patch_order),
            cell_size=cell_size,
        )
        t_calculuspatch = float(time.perf_counter() - t_patch_start)

    periodicity_check = None
    t_periodicity = None
    if bool(args.periodicity_check):
        translated_pairs = bool(args.periodicity_translated_pairs)
        if periodicity_evaluator_mode == "fmm" and translated_pairs:
            raise ValueError(
                "--periodicity-evaluator=fmm cannot evaluate translated points; "
                "use --periodicity-evaluator=direct or --no-periodicity-translated-pairs"
            )

        t_periodicity_start = time.perf_counter()
        periodicity_check = _run_periodicity_jump_check_2d(
            evaluate_potential=evaluate_hybrid_for_periodicity,
            cell_size=cell_size,
            n_samples=int(args.periodicity_samples),
            eps=float(args.periodicity_eps),
            fd_h=float(args.periodicity_fd_h),
            translated_pairs=translated_pairs,
        )
        t_periodicity = float(time.perf_counter() - t_periodicity_start)

    print("=== periodic tail hybrid experiment (2D) ===")
    print(
        f"n_modes={len(source_points)} probe_points={len(probe_points)} "
        f"q_order={args.q_order} nlevels={args.nlevels} "
        f"fmm_order={args.fmm_order}"
    )
    print(
        f"near_radius={args.near_radius} max_order={args.max_order} kmax={args.kmax}"
    )
    print("noncentral_images=direct_near_shell + tail (shifted-image P2P disabled)")
    print(f"spectral_chunk_size={int(args.spectral_chunk_size)}")
    print(
        "tail_eval_extended_precision="
        f"{bool(args.tail_eval_extended_precision)}"
    )
    print(
        "direct_image_sum="
        f"block_size={direct_sum_source_block_size} "
        f"extended_precision={direct_sum_extended_precision}"
    )
    print(
        "spectral_reference="
        f"compensated={spectral_compensated} "
        f"source_block_size={spectral_source_block_size} "
        f"accum_extended_precision={spectral_accum_extended_precision}"
    )
    print(
        "calculuspatch_pde_check="
        f"{bool(args.calculuspatch_pde_check)} "
        f"patch_order={int(args.patch_order)} "
        f"patch_h={float(args.patch_h):.3e} "
        f"patch_center=({float(args.patch_center_x):.3e},"
        f"{float(args.patch_center_y):.3e})"
    )
    print(
        "periodicity_check="
        f"{bool(args.periodicity_check)} "
        f"samples={int(args.periodicity_samples)} "
        f"eps={float(args.periodicity_eps):.3e} "
        f"fd_h={float(args.periodicity_fd_h):.3e} "
        f"evaluator={str(args.periodicity_evaluator)} "
        f"gauge={str(args.periodicity_gauge)} "
        f"translated_pairs={bool(args.periodicity_translated_pairs)}"
    )
    if bool(args.pde_far_reference):
        if pde_reference_method == "spectral":
            print(
                "pde_far_reference=True "
                "pde_reference_method=spectral "
                f"pde_reference_kmax={int(pde_ref_kmax)} "
                f"pde_reference_apply_dipole={bool(args.pde_reference_apply_dipole)}"
            )
        else:
            print(
                "pde_far_reference=True "
                "pde_reference_method=ewald "
                f"ewald_xi={float(args.ewald_xi):.6g} "
                f"ewald_real_cutoff_r={int(args.ewald_real_cutoff_r)} "
                f"ewald_kmax={int(args.ewald_kmax)} "
                f"pde_reference_apply_dipole={bool(args.pde_reference_apply_dipole)}"
            )
    else:
        print("pde_far_reference=False")
    print(f"cell_size={cell_size:.12g}")

    print("timings [s]:")
    print(
        f"  coeff_precompute={t_coeff:.3f} central_fmm={t_fmm:.3f} "
        f"near_images={t_near:.3f} tail_eval={t_tail:.3f} spectral_ref={t_ref:.3f}"
    )
    if direct_far_timing is not None:
        if not direct_far_radii_used:
            print(f"  direct_far_reference={direct_far_timing:.3f}")
        else:
            radii_str = ",".join(str(rad) for rad in direct_far_radii_used)
            print(
                f"  direct_far_reference={direct_far_timing:.3f} "
                f"(radii=[{radii_str}], fit_order={direct_far_fit_order_used})"
            )
    if t_calculuspatch is not None:
        print(f"  calculuspatch_pde_check={t_calculuspatch:.3f}")
    if t_periodicity is not None:
        print(f"  periodicity_check={t_periodicity:.3f}")

    print("accuracy vs spectral periodic reference:")
    print(
        f"  hybrid(fmm-central + near + tail): rel_l2={err_fmm_rel:.3e} "
        f"linf={err_fmm_linf:.3e}"
    )
    print("dipole-corrected (no reference fitting) error vs spectral periodic reference:")
    print(
        "  hybrid(fmm-central + near + tail): "
        f"rel_l2={err_fmm_dipole_rel:.3e} "
        f"linf={err_fmm_dipole_linf:.3e}"
    )
    print(
        "  source dipole moments about center: "
        f"mx={source_dipole[0]:+.6e} my={source_dipole[1]:+.6e}"
    )
    print("affine-corrected error vs spectral periodic reference:")
    print(
        "  hybrid(fmm-central + near + tail): "
        f"rel_l2={float(affine_fmm['rel_l2']):.3e} "
        f"linf={float(affine_fmm['linf']):.3e} "
        f"affine=[{float(affine_fmm['coeffs'][0]):+.3e}, "
        f"{float(affine_fmm['coeffs'][1]):+.3e}, "
        f"{float(affine_fmm['coeffs'][2]):+.3e}]"
    )
    print("a-priori gauge correction (dipole slope + zero-mean constant):")
    print(
        "  hybrid(fmm-central + near + tail): "
        f"rel_l2={a_priori_fmm_rel:.3e} "
        f"linf={a_priori_fmm_linf:.3e} "
        f"affine=[{a_priori_fmm_coeffs[0]:+.3e}, "
        f"{a_priori_fmm_coeffs[1]:+.3e}, "
        f"{a_priori_fmm_coeffs[2]:+.3e}]"
    )
    print(
        "  delta vs fitted affine coefficients: "
        f"[{a_priori_vs_fit_coeff_delta[0]:+.3e}, "
        f"{a_priori_vs_fit_coeff_delta[1]:+.3e}, "
        f"{a_priori_vs_fit_coeff_delta[2]:+.3e}]"
    )
    if tail_direct_rel is not None:
        print("hybrid(fmm-central + near + tail) vs direct full-image reference:")
        print(f"  rel_l2={hybrid_direct_rel:.3e} linf={hybrid_direct_linf:.3e}")
        print("tail-only accuracy vs extrapolated direct far-image sum:")
        print(
            f"  rel_l2={tail_direct_rel:.3e} linf={tail_direct_linf:.3e} "
            f"abs_l2={tail_direct_l2:.3e} ref_l2={tail_ref_l2:.3e}"
        )
        print("direct far-image reference extrapolation self-consistency:")
        print(f"  rel_l2={direct_far_ref_rel:.3e} linf={direct_far_ref_linf:.3e}")
        tail_stage_tol = float(args.tail_stage_rel_tol)
        tail_stage_pass = (
            float(tail_direct_rel) <= tail_stage_tol
            and float(direct_far_ref_rel) <= tail_stage_tol
        )
        print(
            "tail-stage machine-precision gate: "
            f"tol={tail_stage_tol:.1e} status={'PASS' if tail_stage_pass else 'FAIL'}"
        )
    if tail_pde_rel is not None:
        print("tail-only accuracy vs PDE-based non-central reference (periodic minus central+near):")
        print(
            f"  rel_l2={tail_pde_rel:.3e} linf={tail_pde_linf:.3e} "
            f"abs_l2={tail_pde_l2:.3e} ref_l2={tail_pde_ref_l2:.3e}"
        )
        print("tail-only vs PDE far after zero-mean gauge alignment:")
        print(
            f"  rel_l2={tail_pde_rel_zeromean:.3e} "
            f"linf={tail_pde_linf_zeromean:.3e}"
        )
        print("tail-only vs PDE far after affine-fit gauge alignment:")
        print(
            f"  rel_l2={tail_pde_affine_rel:.3e} "
            f"linf={tail_pde_affine_linf:.3e} "
            f"affine=[{tail_pde_affine_coeffs[0]:+.3e}, "
            f"{tail_pde_affine_coeffs[1]:+.3e}, "
            f"{tail_pde_affine_coeffs[2]:+.3e}]"
        )
        if pde_direct_far_rel is not None:
            print("PDE-based far reference vs direct far-image extrapolation:")
            print(f"  rel_l2={pde_direct_far_rel:.3e} linf={pde_direct_far_linf:.3e}")
            print("PDE far vs direct far after zero-mean gauge alignment:")
            print(
                f"  rel_l2={pde_direct_far_rel_zeromean:.3e} "
                f"linf={pde_direct_far_linf_zeromean:.3e}"
            )
            print("PDE far vs direct far after affine-fit gauge alignment:")
            print(
                f"  rel_l2={pde_direct_far_affine_rel:.3e} "
                f"linf={pde_direct_far_affine_linf:.3e} "
                f"affine=[{pde_direct_far_affine_coeffs[0]:+.3e}, "
                f"{pde_direct_far_affine_coeffs[1]:+.3e}, "
                f"{pde_direct_far_affine_coeffs[2]:+.3e}]"
            )

    if calculuspatch_check is not None:
        print("calculus-patch PDE residual (-Delta(u_hybrid)-rho):")
        print(
            f"  rel_l2={float(calculuspatch_check['rel_l2']):.3e} "
            f"abs_l2={float(calculuspatch_check['abs_l2']):.3e} "
            f"linf={float(calculuspatch_check['linf']):.3e} "
            f"rho_l2={float(calculuspatch_check['rho_l2']):.3e}"
        )
        print(
            f"  n_points={int(calculuspatch_check['n_points'])} "
            f"margin_to_boundary={float(calculuspatch_check['margin_to_boundary']):.3e}"
        )

    if periodicity_check is not None:
        val_x = periodicity_check["value_xfaces"]
        val_y = periodicity_check["value_yfaces"]
        dudx_x = periodicity_check["dudx_xfaces"]
        dudy_x = periodicity_check["dudy_xfaces"]
        dudx_y = periodicity_check["dudx_yfaces"]
        dudy_y = periodicity_check["dudy_yfaces"]
        print("periodicity jumps across opposite faces:")
        print(
            f"  samples={int(periodicity_check['samples'])} "
            f"eps={float(periodicity_check['eps']):.3e} "
            f"fd_h={float(periodicity_check['fd_h']):.3e} "
            f"translated_pairs={bool(periodicity_check['translated_pairs'])} "
            f"evaluator={periodicity_evaluator_mode} "
            f"gauge={periodicity_gauge_mode}"
        )
        print(
            "  value jump x-faces: "
            f"rel_l2={float(val_x['rel_l2']):.3e} "
            f"linf={float(val_x['linf']):.3e}"
        )
        print(
            "  value jump y-faces: "
            f"rel_l2={float(val_y['rel_l2']):.3e} "
            f"linf={float(val_y['linf']):.3e}"
        )
        print(
            "  d/dx jump x-faces: "
            f"rel_l2={float(dudx_x['rel_l2']):.3e} "
            f"linf={float(dudx_x['linf']):.3e}"
        )
        print(
            "  d/dy jump x-faces: "
            f"rel_l2={float(dudy_x['rel_l2']):.3e} "
            f"linf={float(dudy_x['linf']):.3e}"
        )
        print(
            "  d/dx jump y-faces: "
            f"rel_l2={float(dudx_y['rel_l2']):.3e} "
            f"linf={float(dudx_y['linf']):.3e}"
        )
        print(
            "  d/dy jump y-faces: "
            f"rel_l2={float(dudy_y['rel_l2']):.3e} "
            f"linf={float(dudy_y['linf']):.3e}"
        )
        print(
            "  worst jump metrics: "
            f"rel_l2={float(periodicity_check['worst_rel_l2']):.3e} "
            f"linf={float(periodicity_check['worst_linf']):.3e}"
        )

    print("derivative-sum diagnostics (selected even orders):")
    for nu in [(0, 0), (2, 0), (1, 1), (0, 2), (4, 0), (2, 2), (0, 4), (6, 0)]:
        if nu not in diagnostics:
            continue
        info = diagnostics[nu]
        method = info.get("method", "n/a")
        elapsed = float(info.get("elapsed_s", 0.0))
        if method == "eta_regularized":
            extra = f"fit_points={info.get('fit_points')} max_tail_chk={info.get('max_tail_check'):.3e}"
        elif method == "hard_richardson":
            extra = (
                f"p={info.get('p')} final_r={info.get('final_r')} "
                f"self_delta={info.get('self_delta'):.3e}"
            )
        elif method == "eisenstein":
            extra = (
                f"order={info.get('order')} mp_dps={info.get('mp_dps')} "
                f"near_prefix={info.get('near_prefix'):.3e}"
            )
        else:
            extra = ""
        print(f"  nu={nu}: method={method} elapsed={elapsed:.3f}s {extra}")


if __name__ == "__main__":
    main()

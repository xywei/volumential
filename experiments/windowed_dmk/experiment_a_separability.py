"""Experiment A check 3: the D7 separability proposition against Volumential's
Duffy-built windowed channel entries.

A reduced windowed channel table entry is, by construction of
``_windowed_channel_skeleton`` and ``_duffy_channel_entry_values_*``,

    E(case, target j, mode i) = int_{[0, H]^d} psi_m(|x_j - y|) prod_k L_{a_k}(y_k) dy

with ``H = table.source_box_extent``, ``x_j = table.find_target_point(j, case)``,
``L_k`` the one-dimensional Lagrange basis on the Gauss-Legendre nodes of the
box, ``a = table._get_all_mode_axes()[i]``, and ``psi_m = chi_m / t_w**m`` the
*normalized* channel profile that the skeleton passes to the table (note the
name collision with Appendix E's polynomial-completion channels, also called
``psi_m``; those are unrelated to the code's normalization).

D7 says this d-dimensional singular integral is a one-dimensional integral of a
product of closed-form one-dimensional erf moments, because

    chi_m(r; t_w) = c_d int_0^{t_w} u^{m - d/2} exp(-r^2 / (4u)) du,
    c_2 = 1/2,  c_3 = 1 / (2 sqrt(pi)),

and the Gaussian factorizes over coordinates.  With the substitution u = v^2
(the Critic addendum's correction: this makes the integrand analytic in v for
every target position, including targets on a face, edge or corner),

    E = c_d t_w^{-m} * 2 * int_0^{sqrt(t_w)} v^{2m+1} prod_k Fhat_{a_k}(v) dv,
    Fhat_k(v) = v^{-1} int_0^H exp(-(x - y)^2 / (4 v^2)) L_k(y) dy,

and each ``Fhat_k`` is a finite combination of ``erf`` and Gaussian terms.  The
``v^{-1}`` is folded in so that no factor of the integrand blows up at v = 0.

This script computes both sides at Paper 1's committed window declaration
(Theta = 16, t_w = (H/Theta)^2, p* = 6 channels m = 0..5) for several
self-interaction targets (box centre, near a face, near a corner) in 2D and 3D,
reports the agreement and the number of graded ``v``-quadrature nodes needed for
a self-converged 1e-13.

Exploratory, not benchmark-grade.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from experiment_a_common import (
    P_STAR,
    ROOT_EXTENT,
    WINDOW_THETA,
    environment_summary,
    gauss_legendre,
    get_pyplot,
    output_dir,
    write_csv,
    write_json,
)

REFERENCE_ORDER = 32
REFERENCE_LEVELS = 34
SWEEP_ORDERS = (4, 6, 8, 10, 12, 14, 16, 20, 24)
SWEEP_LEVELS = 26
TARGET_TOLERANCE = 1.0e-13


# {{{ one-dimensional erf moments and the separable entry


def shifted_lagrange_coefficients(nodes: np.ndarray, centre: float) -> np.ndarray:
    """Monomial coefficients of each Lagrange basis function in ``y - centre``.

    Row ``k`` holds the coefficients ``c_{k0}, c_{k1}, ...`` of
    ``L_k(y) = sum_j c_{kj} (y - centre)**j``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    q = nodes.size
    coefficients = np.zeros((q, q), dtype=np.float64)
    for k in range(q):
        others = np.delete(nodes, k)
        # coefficients of prod_{j != k} (y - nodes[j]), ascending in y
        poly = np.polynomial.polynomial.polyfromroots(others) if q > 1 else np.array(
            [1.0]
        )
        denominator = np.prod(nodes[k] - others) if q > 1 else 1.0
        poly = poly / denominator
        # Taylor shift y -> centre + z
        shifted = np.zeros(q, dtype=np.float64)
        for degree, coefficient in enumerate(poly):
            for index in range(degree + 1):
                shifted[index] += (
                    coefficient
                    * math.comb(degree, index)
                    * centre ** (degree - index)
                )
        coefficients[k, : shifted.size] = shifted
    return coefficients


def gaussian_moments(
    lower: float, upper: float, centre: float, v: np.ndarray, max_degree: int
) -> np.ndarray:
    """``M_j(v) = int_lower^upper (y - centre)**j exp(-(y-centre)^2/(4 v^2)) dy``.

    Returned array has shape ``(max_degree + 1, v.size)``.  Uses the
    integration-by-parts recurrence
    ``M_j = -2 u [z^{j-1} e^{-z^2/(4u)}]_alpha^beta + 2 u (j-1) M_{j-2}``
    with ``u = v^2``.
    """
    v = np.asarray(v, dtype=np.float64)
    u = v * v
    alpha = lower - centre
    beta = upper - centre
    import scipy.special as sps

    exp_alpha = np.exp(-(alpha * alpha) / (4.0 * u))
    exp_beta = np.exp(-(beta * beta) / (4.0 * u))

    moments = np.zeros((max_degree + 1, v.size), dtype=np.float64)
    moments[0] = np.sqrt(np.pi) * v * (
        sps.erf(beta / (2.0 * v)) - sps.erf(alpha / (2.0 * v))
    )
    if max_degree >= 1:
        moments[1] = -2.0 * u * (exp_beta - exp_alpha)
    for degree in range(2, max_degree + 1):
        boundary = beta ** (degree - 1) * exp_beta - alpha ** (degree - 1) * exp_alpha
        moments[degree] = -2.0 * u * boundary + 2.0 * u * (degree - 1) * moments[
            degree - 2
        ]
    return moments


def axis_factors(
    lower: float,
    upper: float,
    centre: float,
    nodes: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """``Fhat_k(v) = v^{-1} int_lower^upper exp(-(centre-y)^2/(4v^2)) L_k(y) dy``.

    Shape ``(len(nodes), len(v))``.  The ``v^{-1}`` keeps every factor O(1) as
    ``v -> 0`` (each integral is asymptotically ``2 sqrt(pi) v L_k(centre)``).
    """
    q = int(np.asarray(nodes).size)
    coefficients = shifted_lagrange_coefficients(nodes, centre)
    moments = gaussian_moments(lower, upper, centre, v, q - 1)
    return (coefficients @ moments) / v


def v_quadrature(
    target: np.ndarray,
    extent: float,
    t_w: float,
    order: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Graded composite Gauss-Legendre rule in ``v`` on ``[0, sqrt(t_w)]``.

    Panel breakpoints are placed at the transition scales ``delta / 2`` of every
    target-to-face distance ``delta`` (that is where ``erf(delta / (2 v))``
    turns over), with dyadic grading toward ``v = 0``.
    """
    v_max = math.sqrt(t_w)
    points = {0.0, v_max}
    scales = set()
    for coordinate in np.asarray(target, dtype=np.float64):
        for face in (0.0, float(extent)):
            delta = abs(float(coordinate) - face)
            if delta > 0.0:
                scales.add(0.5 * delta)
    for scale in scales:
        for factor in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            candidate = factor * scale
            if 0.0 < candidate < v_max:
                points.add(candidate)
    smallest = min(point for point in points if point > 0.0)
    for anchor in (smallest, v_max):
        value = anchor
        for _ in range(levels):
            value *= 0.5
            if value < 1.0e-11 * v_max:
                break
            points.add(value)
    breaks = np.array(sorted(points), dtype=np.float64)

    base_nodes, base_weights = gauss_legendre(order)
    all_nodes = []
    all_weights = []
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        all_nodes.append(mid + half * base_nodes)
        all_weights.append(half * base_weights)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


def separable_entries(
    dim: int,
    m: int,
    t_w: float,
    extent: float,
    axis_nodes: np.ndarray,
    target: np.ndarray,
    mode_axes: np.ndarray,
    order: int,
    levels: int,
) -> tuple[np.ndarray, int]:
    """D7 values of ``E(target, mode)`` for every requested mode.

    ``mode_axes`` has shape ``(n_modes, dim)`` and holds the per-axis basis
    index of each tensor-product source mode.  Returns the values and the
    number of ``v``-quadrature nodes used.
    """
    v, weights = v_quadrature(target, extent, t_w, order, levels)
    factors = [
        axis_factors(0.0, float(extent), float(target[axis]), axis_nodes, v)
        for axis in range(dim)
    ]
    prefactor = (0.5 if dim == 2 else 0.5 / math.sqrt(math.pi)) * t_w ** (-m) * 2.0
    radial_weight = weights * v ** (2 * m + 1)
    values = np.empty(mode_axes.shape[0], dtype=np.float64)
    for index, axes in enumerate(mode_axes):
        product = radial_weight.copy()
        for axis in range(dim):
            product = product * factors[axis][int(axes[axis])]
        values[index] = prefactor * float(np.sum(product))
    return values, int(v.size)


# }}}


# {{{ Volumential Duffy reference


def duffy_order_sweep(
    dim: int,
    q_order: int,
    m: int,
    wanted_keys,
    window_theta: float,
    all_mode_axes: np.ndarray,
    axis_nodes: np.ndarray,
    t_w: float,
    extent: float,
    labels: dict[int, str],
    orders,
    radial_orders=(),
) -> list[dict]:
    """Refine the Duffy angular/regular order and watch it approach D7.

    This is the decisive test of which side of a disagreement is wrong: the
    separable value is fixed (self-converged to 1e-13 by the ``v``-sweep), so if
    the Duffy entries march toward it as the module's own angular order is
    raised, the disagreement is the Duffy build's quadrature error.
    """
    import volumential.rke_table_assembly as rta

    rows: list[dict] = []
    default_regular, default_radial = rta._resolve_channel_orders(dim, None, None)
    combinations = [(order, default_radial) for order in orders]
    if radial_orders:
        finest = max(orders) if orders else default_regular
        combinations += [
            (finest, radial)
            for radial in radial_orders
            if radial != default_radial
        ]
    for regular_order, radial_order in combinations:
        table = rta._windowed_channel_skeleton(
            dim, q_order, 0, ROOT_EXTENT, window_theta, m
        )
        started = time.time()
        context = rta._channel_duffy_context(table, regular_order, radial_order)
        narrowed = {
            key: members
            for key, members in context.groups.items()
            if key in wanted_keys
        }
        profile = rta._normalized_windowed_channel_profile(dim, m, t_w)
        filtered = context._replace(groups=narrowed)
        if dim == 2:
            values = rta._duffy_channel_entry_values_2d(filtered, profile)
        else:
            values = rta._duffy_channel_entry_values_3d(filtered, profile)
        seconds = time.time() - started

        for key, members in narrowed.items():
            positions = np.array([position for position, _ in members], dtype=np.int64)
            modes = np.array([mode for _, mode in members], dtype=np.int64)
            target = np.asarray(
                table.find_target_point(key[1], key[0]), dtype=np.float64
            )
            reference, _ = separable_entries(
                dim, m, t_w, extent, axis_nodes, target, all_mode_axes[modes],
                REFERENCE_ORDER, REFERENCE_LEVELS,
            )
            scale = float(np.max(np.abs(reference)))
            deviation = float(np.max(np.abs(values[positions] - reference))) / scale
            rows.append(
                {
                    "dim": dim,
                    "q_order": q_order,
                    "m": m,
                    "target_label": labels[key[1]],
                    "chan_regular_order": regular_order,
                    "is_module_default": (
                        regular_order == default_regular
                        and radial_order == default_radial
                    ),
                    "chan_radial_order": radial_order,
                    "max_rel_dev_vs_separable": deviation,
                    "build_seconds": seconds,
                }
            )
    return rows


def duffy_reference(dim: int, q_order: int, m: int, wanted_keys, window_theta: float):
    """Duffy-built channel entries for the selected ``(case, target)`` groups.

    Runs Volumential's own ``_duffy_channel_entry_values_*`` on a context whose
    group dictionary has been narrowed to the requested targets, so the
    comparison uses the production node sets, basis evaluation and
    normalization rather than a reimplementation.
    """
    import volumential.rke_table_assembly as rta

    table = rta._windowed_channel_skeleton(
        dim, q_order, 0, ROOT_EXTENT, window_theta, m
    )
    regular_order, radial_order = rta._resolve_channel_orders(dim, None, None)
    context = rta._channel_duffy_context(table, regular_order, radial_order)
    narrowed = {
        key: members for key, members in context.groups.items() if key in wanted_keys
    }
    profile = rta._normalized_windowed_channel_profile(
        dim, m, (table.source_box_extent / window_theta) ** 2
    )
    filtered = context._replace(groups=narrowed)
    if dim == 2:
        values = rta._duffy_channel_entry_values_2d(filtered, profile)
    else:
        values = rta._duffy_channel_entry_values_3d(filtered, profile)
    return table, context, values, regular_order, radial_order


def pick_targets(table, context, case_index: int) -> dict[str, int]:
    """Box-centre, near-face and near-corner targets among the reduced set.

    Per axis let ``d_i`` be the distance of the target to the nearer face on
    that axis.  ``centre`` maximizes ``min_i d_i``; ``near_face`` minimizes
    ``min_i d_i`` while keeping ``max_i d_i`` as large as possible (close to one
    face only); ``near_corner`` minimizes ``max_i d_i`` (close to every face at
    once).  These are the three cases whose ``u``-integrand behaviour D7
    distinguishes: integer powers, one ``sqrt(u)`` factor, and ``d`` of them.
    """
    available = sorted(
        target for (case, target) in context.groups if case == case_index
    )
    extent = float(table.source_box_extent)
    scored = []
    for target_index in available:
        point = np.asarray(
            table.find_target_point(target_index, case_index), dtype=np.float64
        )
        axis_distance = np.minimum(point, extent - point)
        scored.append(
            (target_index, float(np.min(axis_distance)), float(np.max(axis_distance)))
        )
    chosen: dict[str, int] = {}
    chosen["centre"] = max(scored, key=lambda row: (row[1], row[2]))[0]
    chosen["near_face"] = min(scored, key=lambda row: (row[1], -row[2]))[0]
    chosen["near_corner"] = min(scored, key=lambda row: (row[2], row[1]))[0]
    # collapse duplicates (possible at very small q) without losing a slot
    seen: dict[int, str] = {}
    unique: dict[str, int] = {}
    for label, index in chosen.items():
        if index in seen:
            continue
        seen[index] = label
        unique[label] = index
    return unique


# }}}


def run_dimension(
    dim: int,
    q_order: int,
    window_theta: float,
    channels: int,
    duffy_orders,
    duffy_radial_orders=(),
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Check 3 for one dimension: convergence sweep plus Duffy comparison."""
    import volumential.rke_table_assembly as rta

    entry_rows: list[dict] = []
    sweep_rows: list[dict] = []

    table = rta._windowed_channel_skeleton(dim, q_order, 0, ROOT_EXTENT, window_theta, 0)
    extent = float(table.source_box_extent)
    t_w = (extent / window_theta) ** 2
    axis_nodes = np.asarray(
        [point[dim - 1] for point in table.q_points[:q_order]], dtype=np.float64
    )
    all_mode_axes = np.asarray(table._get_all_mode_axes(), dtype=np.int64)
    case_vectors = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    self_case = int(np.flatnonzero(np.all(case_vectors == 0, axis=1))[0])

    regular_order, radial_order = rta._resolve_channel_orders(dim, None, None)
    context0 = rta._channel_duffy_context(table, regular_order, radial_order)
    labelled_targets = pick_targets(table, context0, self_case)
    wanted_keys = {(self_case, index) for index in labelled_targets.values()}

    worst_duffy = 0.0
    worst_self = 0.0
    nodes_for_tolerance: dict[str, int] = {}

    for m in range(channels):
        started = time.time()
        table_m, context_m, duffy_values, regular_order, radial_order = duffy_reference(
            dim, q_order, m, wanted_keys, window_theta
        )
        duffy_seconds = time.time() - started

        for label, target_index in labelled_targets.items():
            key = (self_case, target_index)
            members = context_m.groups[key]
            positions = np.array([position for position, _ in members], dtype=np.int64)
            modes = np.array([mode for _, mode in members], dtype=np.int64)
            mode_axes = all_mode_axes[modes]
            target = np.asarray(
                table_m.find_target_point(target_index, self_case), dtype=np.float64
            )
            face_distance = float(np.min(np.minimum(target, extent - target)))

            reference, reference_nodes = separable_entries(
                dim, m, t_w, extent, axis_nodes, target, mode_axes,
                REFERENCE_ORDER, REFERENCE_LEVELS,
            )
            scale = float(np.max(np.abs(reference)))

            best_nodes = None
            for order in SWEEP_ORDERS:
                values, node_count = separable_entries(
                    dim, m, t_w, extent, axis_nodes, target, mode_axes,
                    order, SWEEP_LEVELS,
                )
                error = float(np.max(np.abs(values - reference))) / scale
                sweep_rows.append(
                    {
                        "dim": dim,
                        "q_order": q_order,
                        "m": m,
                        "target_label": label,
                        "target_index": target_index,
                        "face_distance_over_extent": face_distance / extent,
                        "gauss_order_per_panel": order,
                        "n_v_nodes": node_count,
                        "max_rel_self_error": error,
                    }
                )
                if error <= TARGET_TOLERANCE and best_nodes is None:
                    best_nodes = node_count
            worst_self = max(worst_self, 0.0)
            nodes_for_tolerance[f"{dim}d_m{m}_{label}"] = (
                best_nodes if best_nodes is not None else -1
            )

            duffy = duffy_values[positions]
            deviation = np.abs(duffy - reference) / scale
            worst_duffy = max(worst_duffy, float(np.max(deviation)))
            digits = -math.log10(max(float(np.max(deviation)), 1.0e-18))

            for row_index, mode in enumerate(modes):
                entry_rows.append(
                    {
                        "dim": dim,
                        "q_order": q_order,
                        "m": m,
                        "target_label": label,
                        "target_index": int(target_index),
                        "target_coords": ";".join(f"{c:.12g}" for c in target),
                        "face_distance_over_extent": face_distance / extent,
                        "source_mode": int(mode),
                        "mode_axes": ";".join(
                            str(int(a)) for a in mode_axes[row_index]
                        ),
                        "duffy_entry_psi_m": float(duffy[row_index]),
                        "separable_entry_psi_m": float(reference[row_index]),
                        "abs_diff": float(abs(duffy[row_index] - reference[row_index])),
                        "rel_diff_vs_target_max": float(deviation[row_index]),
                    }
                )

            print(
                f"  dim={dim} m={m} target={label:<11s} "
                f"face_dist/H={face_distance / extent:.4f} "
                f"agreement_digits={digits:.2f} "
                f"nodes_for_1e-13={nodes_for_tolerance[f'{dim}d_m{m}_{label}']} "
                f"ref_nodes={reference_nodes}",
                flush=True,
            )
        print(f"  dim={dim} m={m} duffy build {duffy_seconds:.1f}s", flush=True)

    label_by_index = {index: label for label, index in labelled_targets.items()}
    order_rows: list[dict] = []
    if duffy_orders:
        for m in (0, channels - 1):
            order_rows.extend(
                duffy_order_sweep(
                    dim, q_order, m, wanted_keys, window_theta, all_mode_axes,
                    axis_nodes, t_w, extent, label_by_index, duffy_orders,
                    duffy_radial_orders,
                )
            )
        for row in order_rows:
            print(
                f"  dim={dim} m={row['m']} target={row['target_label']:<12s} "
                f"reg={row['chan_regular_order']:<4d} "
                f"rad={row['chan_radial_order']:<4d} "
                f"rel_dev_vs_separable={row['max_rel_dev_vs_separable']:.3e} "
                f"({row['build_seconds']:.1f}s)",
                flush=True,
            )

    summary = {
        "dim": dim,
        "q_order": q_order,
        "window_theta": window_theta,
        "box_extent": extent,
        "t_w": t_w,
        "channels_m": list(range(channels)),
        "self_case_index": self_case,
        "targets": {
            label: {
                "index": int(index),
                "coords": [
                    float(c)
                    for c in table.find_target_point(index, self_case)
                ],
            }
            for label, index in labelled_targets.items()
        },
        "duffy_regular_order": regular_order,
        "duffy_radial_order": radial_order,
        "max_rel_deviation_duffy_vs_separable": worst_duffy,
        "agreement_digits_duffy_vs_separable": -math.log10(
            max(worst_duffy, 1.0e-18)
        ),
        "v_nodes_for_1e-13_self_convergence": nodes_for_tolerance,
        "reference_rule": {
            "gauss_order_per_panel": REFERENCE_ORDER,
            "dyadic_grading_levels": REFERENCE_LEVELS,
        },
        "duffy_order_sweep": {
            f"m{row['m']}_{row['target_label']}"
            f"_reg{row['chan_regular_order']}_rad{row['chan_radial_order']}":
                row["max_rel_dev_vs_separable"]
            for row in order_rows
        },
    }
    return entry_rows, sweep_rows, order_rows, summary


def _plot(out, sweep_rows):
    """Self-convergence of the graded v-rule, per target class."""
    plt = get_pyplot()
    if plt is None:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for axis, dim in zip(axes, (2, 3), strict=True):
        labels = sorted({row["target_label"] for row in sweep_rows if row["dim"] == dim})
        for label in labels:
            selected = [
                row
                for row in sweep_rows
                if row["dim"] == dim and row["target_label"] == label and row["m"] == 0
            ]
            if not selected:
                continue
            axis.loglog(
                [row["n_v_nodes"] for row in selected],
                [max(row["max_rel_self_error"], 1.0e-17) for row in selected],
                marker="o",
                label=label,
            )
        axis.axhline(TARGET_TOLERANCE, color="grey", linestyle=":")
        axis.set_xlabel("graded v-quadrature nodes")
        axis.set_ylabel("max relative self-error")
        axis.set_title(f"{dim}D, m = 0")
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = out / "experiment_a_check3_separability_convergence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path.name]


def main() -> None:
    """Run check 3 in 2D and 3D and write CSV/JSON/PNG outputs."""
    parser = argparse.ArgumentParser(description="Experiment A check 3 (D7)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--q-order-2d", type=int, default=3)
    parser.add_argument("--q-order-3d", type=int, default=3)
    parser.add_argument("--channels", type=int, default=P_STAR)
    parser.add_argument("--theta", type=float, default=WINDOW_THETA)
    parser.add_argument(
        "--dims", default="2,3", help="comma-separated list of dimensions to run"
    )
    parser.add_argument(
        "--duffy-orders",
        default="",
        help=(
            "comma-separated chan_regular_order values for the Duffy refinement "
            "sweep (empty to skip it)"
        ),
    )
    parser.add_argument(
        "--duffy-radial-orders",
        default="",
        help=(
            "comma-separated chan_radial_order values, run at the finest "
            "regular order, to expose the radial rule's own error floor"
        ),
    )
    args = parser.parse_args()
    out = output_dir(args.out)

    started = time.time()
    entry_rows: list[dict] = []
    sweep_rows: list[dict] = []
    order_rows: list[dict] = []
    summaries: dict[str, object] = {}
    failures: dict[str, str] = {}
    duffy_orders = [
        int(token) for token in args.duffy_orders.split(",") if token.strip()
    ]
    duffy_radial_orders = [
        int(token) for token in args.duffy_radial_orders.split(",") if token.strip()
    ]

    for dim in [int(token) for token in args.dims.split(",") if token.strip()]:
        q_order = args.q_order_2d if dim == 2 else args.q_order_3d
        print(f"dimension {dim}, q = {q_order}", flush=True)
        try:
            rows, sweep, orders, summary = run_dimension(
                dim, q_order, args.theta, args.channels, duffy_orders,
                duffy_radial_orders,
            )
        except Exception as exc:  # noqa: BLE001 - a failure here is a reported verdict
            failures[f"{dim}d"] = f"{type(exc).__name__}: {exc}"
            print(f"  dimension {dim} failed: {exc}", flush=True)
            continue
        entry_rows.extend(rows)
        sweep_rows.extend(sweep)
        order_rows.extend(orders)
        summaries[f"{dim}d"] = summary

    if order_rows:
        write_csv(
            out / "experiment_a_check3_duffy_order_sweep.csv",
            list(order_rows[0]),
            order_rows,
        )
    if entry_rows:
        write_csv(
            out / "experiment_a_check3_entries.csv", list(entry_rows[0]), entry_rows
        )
    if sweep_rows:
        write_csv(
            out / "experiment_a_check3_v_convergence.csv",
            list(sweep_rows[0]),
            sweep_rows,
        )
    plots = _plot(out, sweep_rows) if sweep_rows else []

    payload = {
        "experiment": "A (identity and normalization), check 3: D7 separability",
        "exploratory": True,
        "environment": environment_summary(),
        "configuration": {
            "window_theta": args.theta,
            "root_extent": ROOT_EXTENT,
            "channels_p_star": args.channels,
            "q_order_2d": args.q_order_2d,
            "q_order_3d": args.q_order_3d,
            "normalization": "psi_m = chi_m / t_w**m (the code's stored channel)",
            "target_tolerance": TARGET_TOLERANCE,
        },
        "dimensions": summaries,
        "failures": failures,
        "plots": plots,
        "wall_seconds": time.time() - started,
    }
    write_json(out / "experiment_a_check3_summary.json", payload)
    print(json.dumps(payload, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()

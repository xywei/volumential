#!/usr/bin/env python3
"""Empirically validate the Yukawa RKE break-even repeat count.

Section 6 reports a linear amortization model for the repeated-solve tradeoff
between the direct per-parameter-per-level Yukawa table strategy and the RKE
channel strategy, with a modeled ``p = 3`` break-even of roughly 340 repeats
per parameter on the reference host.  This driver measures the crossing
directly:

* cold phase: build the direct fixed-``lambda`` tables and the RKE channel
  family once, both timed through the table-manager timing hooks.  The
  direct provisioning strategy is selectable (``--direct-provisioning``):
  ``eager`` builds every anticipated level per parameter (the historical
  Table 3 policy and the committed-artifact default), while ``lazy`` builds
  only the level the priced workload actually touches — on the uniform
  benchmark tree the leaf level owns all List 1 work, so the lazy baseline
  executes the provisioning policy the cost model previously carried only
  as an arithmetic projection (experiment E3);
* repeat phase: run interleaved warm level-``nlevels`` solves for both
  strategies, recording every solve wall time individually, until past the
  modeled break-even;
* accounting: accumulate the two cumulative cost curves
  ``C(n) = build_total + sum over parameters of the first n solve times``
  and report the measured crossing (first ``n`` with the RKE curve above the
  direct curve, plus a linearly interpolated fractional crossing), next to
  the linear-model prediction recomputed from this run's own measured means;
* operation counters (experiment E3): alongside the timing columns, the
  summary row reports the operation counts of the cost model, computed at
  the driver level from the executed configuration — symmetry-reduced
  entries per table, singular-quadrature node evaluations per entry (from
  the executed node builders and the actual build routing),
  special-function evaluations by function, near-field point pairs per
  solve, split-remainder series lengths, and table-apply FMA counts.  They
  are computed after the timed phases, so timings are unaffected;
* per-phase shares (experiment E6): the ``ops_phase_*`` and ``s_phase_*``
  columns decompose one end-to-end solve, and the run's setup, into the
  phases of :data:`volumential.phase_profile.SOLVE_PHASES`.  See the
  counting rules below.

The repeat phase alternates strategies within each repeat index so slow host
drift affects both curves equally.  This is a timing benchmark: full mode
must run on an otherwise quiet host.

Per-phase counting rules (E6)
-----------------------------

The phases of one solve are the FMM stage graph plus the two near-field
phases, exactly as :func:`volumential.volume_fmm.drive_volume_fmm` runs
them.  Operations and seconds are counted phase by phase under these rules,
and the component shares the manuscript quotes (table / far field /
remainder / recombination) follow by division within one column.

*Far field.*  ``ops_phase_far_*`` are *coefficient touches*: one multiply-add
against one expansion coefficient, summed over the executed traversal's own
interaction lists at the executed wrangler's own per-level expansion sizes
(:func:`volumential.opcounters.fmm_stage_operation_counts`).  Nothing is a
constant and nothing is a machine-instruction count: sumpy may execute a
translation with fewer operations than its dense coefficient count, so these
are a structural decomposition of the stage graph, not a hardware estimate.
Both strategies run the identical far field — same traversal, same kernel,
same FMM order, same expansion classes — so the far-field counts carry no
strategy suffix.

*Near-field table apply.*  ``ops_phase_nearfield_table_apply_*`` is one
fused multiply-add per near-field (target point, source quadrature point)
pair per applied table.  The direct path applies one table; the online split
path applies the base table in this phase and its ``p-1`` retained-channel
tables in the correction phase below, so its total is ``p`` times the
direct path's, which is the identity the ``ops_split_table_fmas_per_solve``
column already reports.  The count is *dtype blind*, matching the
pre-existing ``ops_*`` columns: this driver runs a complex128 source
function, and a retained-channel apply of a real-valued term kernel is
dispatched twice (real part, imaginary part) inside one counted apply, so
the executed List 1 launch count of the correction phase is twice
``ops_phase_split_correction_extra_table_fmas / N_nf``.  Do not read these
columns as launch counts.

*Split correction.*  The online split path's correction phase is *not* the
series remainder alone, and the cost model's ``Delta W`` (kb:
paper1-ops-cost-model, "The online-mode break-even, restated in ops")
understates what the implementation executes in three ways, all recorded
separately here rather than folded away:

1. the retained-channel table applies happen inside this phase, at
   ``ops_phase_split_correction_extra_table_fmas`` FMAs;
2. when the smooth quadrature order exceeds ``q`` the remainder P2P does
   *not* run on the base quadrature nodes.  The wrangler rebuilds an
   interpolated smooth source set of
   ``ops_phase_split_smooth_sources_per_box`` points per active box
   (``_build_helmholtz_split_smooth_correction_sources``), so the executed
   pair count ``ops_phase_split_correction_remainder_pair_evals`` exceeds
   the model's ``N_nf`` by ``(q_smooth/q)**d``.  The rebuild itself costs
   ``ops_phase_split_correction_smooth_interp_fmas`` interpolation FMAs
   *per solve*, priced as the axis-by-axis tensor-product contraction the
   implementation performs and not as a dense ``q_smooth**d`` by ``q**d``
   matrix apply (:func:`_tensor_product_interp_fmas`);
3. each 2D single-table ``power_log`` term runs an additional near-field
   P2P pass for its ``beta`` contribution when the auto-config's
   ``power_log_single_table_beta_mode`` is ``p2p``, counted at
   ``ops_phase_split_correction_beta_p2p_pair_evals``.

``ops_phase_split_correction_remainder_term_evals`` (and therefore
``ops_phase_split_correction_rke`` and ``ops_phase_solve_total_rke``) uses
the *mean* of ``ops_split_series_nmax_per_parameter`` over the run's
parameters, because the profiled seconds it sits beside are likewise means
over the same parameter set.  The per-parameter series lengths stay
available, unaveraged, in the pre-existing column.

The pre-existing ``ops_split_remainder_*`` columns keep their published
meaning (the model's ``N_nf``-based count) and are not touched.

``ops_phase_split_correction_status`` records why a correction count is
blank when the wrangler cannot be interrogated.  A blank there propagates:
``ops_phase_split_correction_rke`` and ``ops_phase_solve_total_rke`` go
blank too, and ``break_even_phases.csv`` then withholds the whole
strategy's ``ops_share`` column rather than dividing the surviving phases
by their own sum, which would report a confident share of a denominator
that is missing the dominant phase.

*Recombination.*  Windowed recombination is a per-*parameter setup* cost
(``p_star`` FMAs per assembled entry), never a per-solve cost, and this
driver provisions no windowed family at all: both
``ops_phase_recombination_per_solve`` and
``ops_phase_setup_recombination_flops`` are therefore 0 here, meaning zero
recombination work was executed, not "unmeasured".

*Setup.*  ``ops_phase_setup_direct_table_build`` and
``ops_phase_setup_channel_family_build`` are the singular-quadrature node
evaluations of the two cold builds; their seconds are the cold-build and
warm cache-load times the table manager's own timing hooks record.  These
are a *different currency* from the per-solve counts above -- whole kernel
or channel evaluations inside a Duffy rule, not multiply-adds -- so setup
and solve operation counts must never be added or shared against each
other.  The long-format ``break_even_phases.csv`` states the currency of
every row in its ``ops_unit`` column, and setup rows carry no
``ops_share`` for exactly this reason.

*Reading the far-field seconds.*  ``s_phase_far_multipole_to_local_*`` is
the one phase whose seconds routinely fail to track its operation count,
and the reason is environmental rather than structural.  When sumpy's M2L
uses its FFT-accelerated translation and pyvkfft is unavailable, the loopy
FFT backend is invoked through an uncached ``TranslationUnit.__call__``
(``sumpy.tools.run_opencl_fft``), which pays a translation-unit
compilation on *every* solve; the run then emits ``VkFFT not found`` and
``DirectCallUncachedWarning`` and the M2L phase can dominate the profiled
solve while its coefficient-touch count does not.  Check the run log
before quoting a far-field time share, and quote the operation share
instead when those warnings are present.  The converse also occurs and is
not a defect: on a uniform tree List 3 and List 4 are empty, so
``ops_phase_far_eval_multipoles`` and ``ops_phase_far_form_locals`` are 0
while their seconds are not, because the wrangler still launches a kernel
over an empty interaction list.  A zero operation count means zero
counted work, never zero elapsed time.

*Seconds.*  ``s_phase_*`` are means over dedicated, phase-instrumented
solves run *after* every timed phase, so no reported timing column is
perturbed by the instrumentation.  Profiling synchronizes the command queue
at every phase boundary, so a profiled solve is slower than an unprofiled
one; ``s_phase_solve_total_*`` reports the profiled total next to the
unprofiled ``*_solve_mean_s`` so the overhead is visible and the shares can
be read as shares of the profiled total.  ``s_phase_other_*`` is the
residual (reordering, finalization, host bookkeeping) and is non-negative
by construction.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from split_parameter_sweep import (  # noqa: E402
    _build_geometry,
    _build_path,
    _coords_host,
    _gaussian_source_host,
    _prepare_direct_tables,
    _prepare_rke_channels,
    _select_opencl_device,
    _split_channel_build_config,
    _split_correction_operation_counts,
    _split_smooth_quad_order,
    # re-exported: named by the counting rule in this module's docstring and
    # exercised through this module's namespace by the driver's tests
    _tensor_product_interp_fmas,  # noqa: F401
    _yukawa_reference_build_config,
)

SOLVE_FIELDS = (
    "mode",
    "kernel",
    "parameter",
    "strategy",
    "split_order",
    "repeat_index",
    "solve_wall_s",
)

SUMMARY_FIELDS = (
    "mode",
    "kernel",
    "q_order",
    "nlevels",
    "fmm_order",
    "split_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "parameter_count",
    "parameters",
    "level_count",
    "direct_levels",
    "repeat_count",
    "n_targets",
    "direct_build_total_s",
    "rke_build_total_s",
    "direct_warmup_total_s",
    "rke_warmup_total_s",
    "direct_solve_mean_s",
    "direct_solve_std_s",
    "rke_solve_mean_s",
    "rke_solve_std_s",
    "measured_break_even_repeat",
    "measured_break_even_repeat_interpolated",
    "measured_crossing_cumulative_cost_s",
    "modeled_break_even_repeat_from_this_run",
    "cumulative_cost_definition",
    "max_rel_l2_rke_vs_direct",
    "benchmark_total_s",
    # provisioning strategy of the direct baseline (E3): "eager" builds all
    # anticipated levels, "lazy" only the level the priced workload touches
    "direct_provisioning",
    # operation counters (E3), computed from the executed configuration
    "ops_reduced_entries_per_table",
    "ops_direct_build_routing",
    "ops_direct_tables_built",
    "ops_direct_entries_built",
    "ops_direct_singular_nodes_per_entry",
    "ops_direct_singular_node_evals",
    "ops_direct_special_function",
    "ops_direct_special_function_evals",
    "ops_rke_channel_tables_built",
    "ops_rke_channel_entries_built",
    "ops_rke_channel_singular_nodes_per_entry",
    "ops_rke_channel_singular_node_evals",
    "ops_rke_channel_special_function_evals",
    "ops_nearfield_point_pairs_per_solve",
    "ops_direct_table_fmas_per_solve",
    "ops_split_table_count",
    "ops_split_table_fmas_per_solve",
    "ops_split_series_nmax_per_parameter",
    "ops_split_remainder_pair_evals_per_solve",
    "ops_split_remainder_term_flops_per_solve_per_parameter",
)

# {{{ per-phase columns (E6)

#: Strategy suffixes of the per-phase columns.  ``rke`` is the online split
#: strategy, matching the existing ``rke_*`` timing columns.
PHASE_STRATEGIES = ("direct", "rke")

#: Far-field stage names, without the ``far_`` phase prefix.
PHASE_FAR_STAGES = (
    "form_multipoles",
    "coarsen_multipoles",
    "multipole_to_local",
    "eval_multipoles",
    "form_locals",
    "refine_locals",
    "eval_locals",
)

#: Phases whose seconds are reported per strategy.
PHASE_SECOND_NAMES = (
    *(f"far_{stage}" for stage in PHASE_FAR_STAGES),
    "far_total",
    "nearfield_table_apply",
    "split_correction",
    "other",
    "solve_total",
)

PHASE_OPS_FIELDS = (
    "phase_counting_rule",
    "phase_profile_repeat_count",
    "phase_profile_solves_per_strategy",
    "phase_profile_nested_phases",
    *(f"ops_phase_far_{stage}" for stage in PHASE_FAR_STAGES),
    "ops_phase_far_total",
    "ops_phase_fmm_multipole_coefficients_by_level",
    "ops_phase_fmm_local_coefficients_by_level",
    "ops_phase_nearfield_table_apply_direct",
    "ops_phase_nearfield_table_apply_rke",
    "ops_phase_split_correction_rke",
    "ops_phase_split_correction_extra_table_fmas",
    "ops_phase_split_correction_remainder_pair_evals",
    "ops_phase_split_correction_remainder_term_evals",
    "ops_phase_split_correction_beta_p2p_pair_evals",
    "ops_phase_split_correction_smooth_interp_fmas",
    "ops_phase_split_smooth_sources_per_box",
    "ops_phase_split_correction_status",
    "ops_phase_recombination_per_solve",
    "ops_phase_solve_total_direct",
    "ops_phase_solve_total_rke",
    "ops_phase_setup_direct_table_build",
    "ops_phase_setup_channel_family_build",
    "ops_phase_setup_recombination_flops",
)

PHASE_SECONDS_FIELDS = (
    *(
        f"s_phase_{name}_{strategy}"
        for strategy in PHASE_STRATEGIES
        for name in PHASE_SECOND_NAMES
    ),
    "s_phase_setup_direct_table_build",
    "s_phase_setup_direct_table_cache_load",
    "s_phase_setup_channel_family_build",
    "s_phase_setup_channel_family_cache_load",
    "s_phase_setup_recombination",
)

SUMMARY_FIELDS = SUMMARY_FIELDS + PHASE_OPS_FIELDS + PHASE_SECONDS_FIELDS

#: Long-format companion CSV: one row per (scope, strategy, phase), so the
#: component shares are a division inside a single row rather than a pivot
#: over the wide summary row.
PHASE_FIELDS = (
    "mode",
    "kernel",
    "direct_provisioning",
    "scope",
    "strategy",
    "phase",
    "unit",
    "ops_unit",
    "ops",
    "seconds",
    "ops_share",
    "seconds_share",
)

#: What the ``ops`` column counts in each scope.  Solve and setup rows are
#: in *different* currencies and must never be added together: a solve row
#: counts coefficient touches, table FMAs, series-term and pair evaluations
#: (all "one multiply-add against one datum"), while a setup row counts
#: singular-quadrature node evaluations of a table build, which are whole
#: kernel or channel evaluations.
PHASE_OPS_UNITS = {
    "solve": "coefficient_touches_fmas_and_pair_evals",
    "setup": "singular_quadrature_node_evals",
}

#: Identifies the counting-rule revision the ``ops_phase_*`` columns follow,
#: so a consumer can tell two executions of different rules apart.
PHASE_COUNTING_RULE = (
    "e6-v2:far=dense_coefficient_touches_from_traversal_and_expansion_sizes;"
    "nearfield=fma_per_nearfield_pair_per_applied_table_dtype_blind;"
    "split_correction=extra_table_fmas+remainder_pair_evals*mean_nmax"
    "+beta_p2p_pair_evals+smooth_interp_fmas;"
    "smooth_interp=tensor_product_axis_by_axis_not_dense;"
    "recombination=0_per_solve_and_no_windowed_family_in_this_driver"
)

# }}}


def _solve_wall_s(queue, traversal, wrangler, weighted_sources, source_vals):
    from volumential.volume_fmm import drive_volume_fmm

    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        weighted_sources,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    queue.finish()
    return potential, time.perf_counter() - start


def _solve_statistics(solve_rows, strategy):
    samples = [
        row["solve_wall_s"] for row in solve_rows
        if row["strategy"] == strategy
    ]
    if not samples:
        raise ValueError(f"no solve samples recorded for strategy {strategy!r}")
    return float(np.mean(samples)), float(np.std(samples))


def _modeled_break_even(
    *,
    parameter_count,
    direct_build_s,
    rke_build_s,
    direct_solve_mean_s,
    rke_solve_mean_s,
):
    setup_advantage = direct_build_s - rke_build_s
    per_repeat_gap = parameter_count * (
        rke_solve_mean_s - direct_solve_mean_s
    )
    if setup_advantage > 0.0 and per_repeat_gap > 0.0:
        return setup_advantage / per_repeat_gap
    return ""


def _resolve_direct_levels(
    *, smoke: bool, provisioning: str, nlevels: int
) -> list[int]:
    """Levels the direct baseline provisions under the chosen strategy.

    ``eager`` reproduces the committed-artifact policy (all anticipated
    levels).  ``lazy`` provisions only what the priced workload touches: the
    warm solves run on a uniform level-``nlevels`` grid whose List 1 work is
    owned entirely by the leaf level, so exactly one table per parameter.
    """
    if provisioning == "lazy":
        return [nlevels]
    if provisioning != "eager":
        raise ValueError(
            f"unknown direct provisioning strategy: {provisioning!r}"
        )
    return [1, 2] if smoke else [0, 1, 2, 3, 4, 5]


def _operation_counters(
    *,
    queue,
    traversal,
    parameters,
    direct_levels,
    direct_tables,
    direct_build_config,
    rke_base_table,
    split_term_tables,
    rke_channel_build_config,
    rke_wranglers,
    split_order,
):
    """Operation counts of the cost model, from the executed configuration.

    Every number is derived from executed objects — the built tables'
    symmetry-reduced entry sets, the node builders at the requested orders,
    the actual build routing predicate, the FMM traversal's List 1, and the
    wrangler's own series-length rule — not from hardcoded constants, so the
    emitted columns confirm (or refute) the analytic counts of the ops cost
    model in situ.
    """
    import volumential.opcounters as opcounters

    sample_direct = direct_tables[parameters[0]]
    n_rep = opcounters.reduced_entry_count(sample_direct)
    routing = (
        "batched"
        if sample_direct._supports_batched_duffy_builder()
        else "scalar"
    )
    if routing == "batched":
        direct_nodes_per_entry = opcounters.batched_duffy_nodes_per_entry(
            int(sample_direct.dim),
            direct_build_config.regular_quad_order,
            direct_build_config.radial_quad_order,
        )
        direct_special_function = "hankel1_imaginary_ray"
    else:
        geometry = opcounters.duffy_block_geometry(sample_direct)
        direct_nodes_per_entry = opcounters.scalar_duffy_singular_nodes(
            sample_direct,
            direct_build_config.regular_quad_order,
            direct_build_config.radial_quad_order,
            geometry=geometry,
        ) / max(geometry["n_reduced_entries"], 1)
        direct_special_function = "kv0"
    direct_tables_built = len(parameters) * len(direct_levels)
    direct_entries_built = direct_tables_built * n_rep
    direct_node_evals = int(
        round(direct_entries_built * direct_nodes_per_entry)
    )

    # split_term_tables maps each term key to a per-level list of tables (the
    # wrangler normalizes a bare table to a one-element list), so flatten
    # before counting: every listed table is separately built and its entries
    # separately quadratured.
    rke_tables = [rke_base_table]
    for term_tables in split_term_tables.values():
        if isinstance(term_tables, list):
            rke_tables.extend(term_tables)
        else:
            rke_tables.append(term_tables)
    rke_entry_counts = [
        opcounters.reduced_entry_count(table) for table in rke_tables
    ]
    rke_nodes_per_entry = opcounters.batched_duffy_nodes_per_entry(
        int(rke_base_table.dim),
        rke_channel_build_config.regular_quad_order,
        rke_channel_build_config.radial_quad_order,
    )
    rke_entries_built = int(np.sum(rke_entry_counts))

    n_nf = opcounters.nearfield_point_pairs(queue, traversal)
    nmax_by_parameter = [
        int(rke_wranglers[parameter]._helmholtz_split_series_nmax(
            split_order
        ))
        for parameter in parameters
    ]
    split_table_count = len(rke_tables)

    return {
        "ops_reduced_entries_per_table": n_rep,
        "ops_direct_build_routing": routing,
        "ops_direct_tables_built": direct_tables_built,
        "ops_direct_entries_built": direct_entries_built,
        "ops_direct_singular_nodes_per_entry": direct_nodes_per_entry,
        "ops_direct_singular_node_evals": direct_node_evals,
        "ops_direct_special_function": direct_special_function,
        "ops_direct_special_function_evals": direct_node_evals,
        "ops_rke_channel_tables_built": len(rke_tables),
        "ops_rke_channel_entries_built": rke_entries_built,
        "ops_rke_channel_singular_nodes_per_entry": rke_nodes_per_entry,
        "ops_rke_channel_singular_node_evals": (
            rke_entries_built * rke_nodes_per_entry
        ),
        # power/power-log channel integrands are elementary (log and radial
        # powers): no special-function quadrature anywhere in the family
        "ops_rke_channel_special_function_evals": 0,
        "ops_nearfield_point_pairs_per_solve": n_nf,
        "ops_direct_table_fmas_per_solve": n_nf,
        "ops_split_table_count": split_table_count,
        "ops_split_table_fmas_per_solve": split_table_count * n_nf,
        "ops_split_series_nmax_per_parameter": ";".join(
            str(nmax) for nmax in nmax_by_parameter
        ),
        "ops_split_remainder_pair_evals_per_solve": n_nf,
        "ops_split_remainder_term_flops_per_solve_per_parameter": ";".join(
            str(n_nf * nmax) for nmax in nmax_by_parameter
        ),
    }


# {{{ per-phase operation counts and timings (E6)

def _phase_operation_counts(
    *,
    queue,
    traversal,
    direct_wrangler,
    rke_wrangler,
    q_order,
    smooth_quad_order,
    nmax_by_parameter,
    split_table_count,
):
    """Per-phase operation counts of one end-to-end solve (E6).

    See the module docstring for the counting rules.  The far-field counts
    are strategy independent because both paths run the identical
    traversal, kernel, FMM order and expansion classes; only the near-field
    phases differ.
    """
    import volumential.opcounters as opcounters

    far = opcounters.fmm_stage_operation_counts_from_traversal(
        queue, traversal, direct_wrangler
    )
    multipole, local = opcounters.expansion_coefficient_counts(direct_wrangler)
    nearfield_pairs = opcounters.nearfield_point_pairs(queue, traversal)

    correction = _split_correction_operation_counts(
        queue=queue,
        traversal=traversal,
        wrangler=rke_wrangler,
        q_order=q_order,
        smooth_quad_order=smooth_quad_order,
    )

    mean_nmax = (
        float(np.mean(nmax_by_parameter)) if nmax_by_parameter else 0.0
    )
    remainder_pairs = correction["remainder_pair_evals"]
    if remainder_pairs == "":
        remainder_term_evals = ""
        correction_total = ""
        rke_solve_total = ""
    else:
        remainder_term_evals = float(remainder_pairs) * mean_nmax
        correction_total = (
            float(correction["extra_table_fmas"])
            + remainder_term_evals
            + float(correction["beta_p2p_pair_evals"])
            + float(correction["smooth_interp_fmas"])
        )
        rke_solve_total = (
            float(far["far_total"]) + float(nearfield_pairs) + correction_total
        )

    columns = {
        "phase_counting_rule": PHASE_COUNTING_RULE,
        "ops_phase_far_total": far["far_total"],
        "ops_phase_fmm_multipole_coefficients_by_level": ";".join(
            str(count) for count in multipole
        ),
        "ops_phase_fmm_local_coefficients_by_level": ";".join(
            str(count) for count in local
        ),
        "ops_phase_nearfield_table_apply_direct": nearfield_pairs,
        # the split path applies the base table in this phase; its retained
        # channels are applied in the correction phase and counted there
        "ops_phase_nearfield_table_apply_rke": nearfield_pairs,
        "ops_phase_split_correction_rke": correction_total,
        "ops_phase_split_correction_extra_table_fmas": (
            correction["extra_table_fmas"]
        ),
        "ops_phase_split_correction_remainder_pair_evals": remainder_pairs,
        "ops_phase_split_correction_remainder_term_evals": (
            remainder_term_evals
        ),
        "ops_phase_split_correction_beta_p2p_pair_evals": (
            correction["beta_p2p_pair_evals"]
        ),
        "ops_phase_split_correction_smooth_interp_fmas": (
            correction["smooth_interp_fmas"]
        ),
        "ops_phase_split_smooth_sources_per_box": (
            correction["smooth_sources_per_box"]
        ),
        "ops_phase_split_correction_status": correction["status"],
        # windowed recombination is a per-parameter setup cost and this
        # driver provisions no windowed family: zero executed, not unknown
        "ops_phase_recombination_per_solve": 0,
        "ops_phase_setup_recombination_flops": 0,
        "ops_phase_solve_total_direct": (
            int(far["far_total"]) + int(nearfield_pairs)
        ),
        "ops_phase_solve_total_rke": rke_solve_total,
    }
    for stage in PHASE_FAR_STAGES:
        columns[f"ops_phase_far_{stage}"] = far[stage]

    # consistency check against the published p-times-direct identity: the
    # base apply plus the correction's extra applies must be p table applies
    if correction["extra_table_fmas"] != "":
        expected = int(split_table_count) * int(nearfield_pairs)
        executed = int(nearfield_pairs) + int(correction["extra_table_fmas"])
        if executed != expected:
            columns["ops_phase_split_correction_status"] = (
                f"{correction['status']};"
                f"table_apply_mismatch:executed={executed},"
                f"ops_split_table_fmas_per_solve={expected}"
            )
    return columns


def _profile_solve_phases(
    *, queue, traversal, paths, parameters, repeat_count
):
    """Run dedicated phase-instrumented solves, one profile per strategy.

    These solves are extra work run after every timed phase; they never
    enter the cumulative cost curves or the ``*_solve_mean_s`` columns.
    """
    from volumential.phase_profile import PhaseProfile, profiling

    profiles = {}
    solve_totals = {}
    solve_counts = {}
    for strategy in PHASE_STRATEGIES:
        profile = PhaseProfile(sync=queue.finish)
        total_s = 0.0
        n_solves = 0
        for parameter in parameters:
            path = paths[parameter]
            for _ in range(repeat_count):
                with profiling(profile):
                    _, wall_s = _solve_wall_s(
                        queue,
                        traversal,
                        path[strategy],
                        path["weighted_sources"],
                        path["source_vals"],
                    )
                total_s += wall_s
                n_solves += 1
        profiles[strategy] = profile
        solve_totals[strategy] = total_s
        solve_counts[strategy] = n_solves
    return profiles, solve_totals, solve_counts


def _phase_second_columns(*, profiles, solve_totals, solve_counts):
    """Mean per-solve seconds per phase and strategy, plus the residual."""
    from volumential.phase_profile import FAR_FIELD_PHASES

    columns = {}
    nested = set()
    for strategy in PHASE_STRATEGIES:
        profile = profiles[strategy]
        nested |= set(profile.nested_names)
        n_solves = max(int(solve_counts[strategy]), 1)
        solve_total = float(solve_totals[strategy]) / n_solves

        far_total = 0.0
        for name in FAR_FIELD_PHASES:
            seconds = profile.seconds(name) / n_solves
            far_total += seconds
            columns[f"s_phase_{name}_{strategy}"] = seconds
        columns[f"s_phase_far_total_{strategy}"] = far_total

        recorded = far_total
        for name in ("nearfield_table_apply", "split_correction"):
            seconds = profile.seconds(name) / n_solves
            recorded += seconds
            columns[f"s_phase_{name}_{strategy}"] = seconds

        columns[f"s_phase_other_{strategy}"] = solve_total - recorded
        columns[f"s_phase_solve_total_{strategy}"] = solve_total

    columns["phase_profile_nested_phases"] = ";".join(sorted(nested))
    columns["phase_profile_solves_per_strategy"] = ";".join(
        f"{strategy}:{solve_counts[strategy]}"
        for strategy in PHASE_STRATEGIES
    )
    return columns


def _phase_rows(summary_row):
    """Long-format phase rows derived from a finished summary row.

    An operation share is emitted only when every phase that is supposed to
    carry a count actually carries one.  ``other`` is unpriced by design (it
    is the seconds residual and has no operation count), but a *priced*
    phase that came back blank -- which is what
    :func:`_split_correction_operation_counts` writes when it cannot
    interrogate the wrangler -- means the remaining counts are not a
    partition of the solve.  Dividing them by their own sum would then
    present a confident share of the wrong denominator, so in that case the
    whole strategy's ``ops_share`` column is left empty instead and the
    blank ``ops`` cell says why.
    """
    rows = []
    shared = {
        "mode": summary_row["mode"],
        "kernel": summary_row["kernel"],
        "direct_provisioning": summary_row["direct_provisioning"],
    }

    def _number(value):
        if value == "" or value is None:
            return None
        return float(value)

    for strategy in PHASE_STRATEGIES:
        # (phase, ops, seconds, priced): "priced" phases must all carry a
        # number for the operation shares to be a partition
        entries = []
        for stage in PHASE_FAR_STAGES:
            entries.append(
                (
                    f"far_{stage}",
                    summary_row[f"ops_phase_far_{stage}"],
                    summary_row[f"s_phase_far_{stage}_{strategy}"],
                    True,
                )
            )
        entries.append(
            (
                "nearfield_table_apply",
                summary_row[f"ops_phase_nearfield_table_apply_{strategy}"],
                summary_row[f"s_phase_nearfield_table_apply_{strategy}"],
                True,
            )
        )
        entries.append(
            (
                "split_correction",
                (
                    summary_row["ops_phase_split_correction_rke"]
                    if strategy == "rke"
                    else 0
                ),
                summary_row[f"s_phase_split_correction_{strategy}"],
                True,
            )
        )
        entries.append(
            (
                "recombination",
                summary_row["ops_phase_recombination_per_solve"],
                0.0,
                True,
            )
        )
        entries.append(
            ("other", "", summary_row[f"s_phase_other_{strategy}"], False)
        )

        ops_values = [_number(ops) for _, ops, _, _ in entries]
        partition_is_complete = all(
            value is not None
            for value, (_, _, _, priced) in zip(
                ops_values, entries, strict=True
            )
            if priced
        )
        ops_total = sum(value for value in ops_values if value is not None)
        seconds_total = float(
            summary_row[f"s_phase_solve_total_{strategy}"]
        )
        for (phase, ops, seconds, _priced), ops_value in zip(
            entries, ops_values, strict=True
        ):
            rows.append(
                {
                    **shared,
                    "scope": "solve",
                    "strategy": strategy,
                    "phase": phase,
                    "unit": "per_solve",
                    "ops_unit": PHASE_OPS_UNITS["solve"],
                    "ops": ops,
                    "seconds": seconds,
                    "ops_share": (
                        ops_value / ops_total
                        if partition_is_complete
                        and ops_value is not None
                        and ops_total > 0.0
                        else ""
                    ),
                    "seconds_share": (
                        float(seconds) / seconds_total
                        if seconds_total > 0.0
                        else ""
                    ),
                }
            )

    setup_entries = (
        (
            "direct",
            "direct_table_build",
            summary_row["ops_phase_setup_direct_table_build"],
            summary_row["s_phase_setup_direct_table_build"],
        ),
        (
            "direct",
            "direct_table_cache_load",
            "",
            summary_row["s_phase_setup_direct_table_cache_load"],
        ),
        (
            "rke",
            "channel_family_build",
            summary_row["ops_phase_setup_channel_family_build"],
            summary_row["s_phase_setup_channel_family_build"],
        ),
        (
            "rke",
            "channel_family_cache_load",
            "",
            summary_row["s_phase_setup_channel_family_cache_load"],
        ),
        (
            "rke",
            "recombination",
            summary_row["ops_phase_setup_recombination_flops"],
            summary_row["s_phase_setup_recombination"],
        ),
    )
    for strategy in PHASE_STRATEGIES:
        entries = [
            entry for entry in setup_entries if entry[0] == strategy
        ]
        seconds_total = sum(float(entry[3]) for entry in entries)
        for _, phase, ops, seconds in entries:
            rows.append(
                {
                    **shared,
                    "scope": "setup",
                    "strategy": strategy,
                    "phase": phase,
                    "unit": "per_run",
                    "ops_unit": PHASE_OPS_UNITS["setup"],
                    "ops": ops,
                    "seconds": seconds,
                    "ops_share": "",
                    "seconds_share": (
                        float(seconds) / seconds_total
                        if seconds_total > 0.0
                        else ""
                    ),
                }
            )
    return rows

# }}}


def run_validation(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    split_order: int,
    parameters: list[float],
    direct_levels: list[int],
    repeat_count: int,
    warmup_count: int,
    direct_provisioning: str = "eager",
    phase_repeat_count: int = 0,
):
    import pyopencl as cl

    benchmark_start = time.perf_counter()
    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    q_points, source_weights, tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    high_accuracy = mode == "full"
    direct_build_config = _yukawa_reference_build_config(
        q_order, high_accuracy=high_accuracy
    )
    rke_channel_build_config = _split_channel_build_config(
        q_order, high_accuracy=high_accuracy
    )
    smooth_quad_order = _split_smooth_quad_order(
        q_order, split_order, high_accuracy=high_accuracy
    )

    # cold phase: direct provisioning per parameter (eager: all anticipated
    # levels; lazy: only the level the priced workload touches)
    direct_tables = {}
    direct_build_total_s = 0.0
    direct_load_total_s = 0.0
    for parameter in parameters:
        table, costs = _prepare_direct_tables(
            kernel="Yukawa",
            queue=queue,
            cache_dir=cache_dir,
            q_order=q_order,
            parameter=parameter,
            direct_levels=direct_levels,
            active_level=nlevels,
            build_config=direct_build_config,
        )
        direct_tables[parameter] = table
        direct_build_total_s += costs["build_s"]
        direct_load_total_s += costs["load_s"]
        print(
            f"direct cold build lam={parameter:g}: {costs['build_s']:.1f} s",
            flush=True,
        )

    # cold phase: one RKE channel family for all parameters
    rke_table, split_term_tables, rke_costs = _prepare_rke_channels(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        q_order=q_order,
        fmm_order=fmm_order,
        kernel="Yukawa",
        parameter=parameters[0],
        split_order=split_order,
        source_weights=source_weights,
        q_points=q_points,
        source_values_host=source_values_host,
        cache_dir=cache_dir,
        build_config=rke_channel_build_config,
        split_smooth_quad_order=smooth_quad_order,
    )
    rke_build_total_s = rke_costs["build_s"]
    rke_load_total_s = rke_costs["load_s"]
    print(f"rke cold build (p={split_order}): {rke_build_total_s:.1f} s",
          flush=True)

    # wranglers built once per parameter and strategy (steady-state reuse)
    paths = {}
    for parameter in parameters:
        direct_wrangler, weighted_sources, source_vals = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=parameter,
            table=direct_tables[parameter],
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=split_order,
        )
        rke_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=parameter,
            table=rke_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_term_tables=split_term_tables,
            split_smooth_quad_order=smooth_quad_order,
        )
        paths[parameter] = {
            "direct": direct_wrangler,
            "rke": rke_wrangler,
            "weighted_sources": weighted_sources,
            "source_vals": source_vals,
        }

    # warmup solves (JIT compilation etc.), excluded from the curves
    warmup_totals = {"direct": 0.0, "rke": 0.0}
    max_rel_l2 = 0.0
    for parameter in parameters:
        path = paths[parameter]
        results = {}
        for _ in range(warmup_count):
            for strategy in ("direct", "rke"):
                potential, wall_s = _solve_wall_s(
                    queue,
                    traversal,
                    path[strategy],
                    path["weighted_sources"],
                    path["source_vals"],
                )
                warmup_totals[strategy] += wall_s
                results[strategy] = potential.get(queue)
        rel_l2 = float(
            np.linalg.norm(results["rke"] - results["direct"])
            / max(np.linalg.norm(results["direct"]), 1e-300)
        )
        max_rel_l2 = max(max_rel_l2, rel_l2)
        print(
            f"warmup lam={parameter:g}: rke vs direct rel_l2={rel_l2:.3e}",
            flush=True,
        )

    # repeat phase: alternate strategies within each repeat index
    solve_rows = []
    per_repeat_cost = {"direct": np.zeros(repeat_count),
                       "rke": np.zeros(repeat_count)}
    for repeat_index in range(repeat_count):
        for parameter in parameters:
            path = paths[parameter]
            for strategy in ("direct", "rke"):
                _, wall_s = _solve_wall_s(
                    queue,
                    traversal,
                    path[strategy],
                    path["weighted_sources"],
                    path["source_vals"],
                )
                per_repeat_cost[strategy][repeat_index] += wall_s
                solve_rows.append(
                    {
                        "mode": mode,
                        "kernel": "Yukawa",
                        "parameter": parameter,
                        "strategy": strategy,
                        "split_order": split_order,
                        "repeat_index": repeat_index,
                        "solve_wall_s": wall_s,
                    }
                )
        if (repeat_index + 1) % 25 == 0 or repeat_index + 1 == repeat_count:
            direct_cum = direct_build_total_s + float(
                np.sum(per_repeat_cost["direct"][: repeat_index + 1])
            )
            rke_cum = rke_build_total_s + float(
                np.sum(per_repeat_cost["rke"][: repeat_index + 1])
            )
            print(
                f"repeat {repeat_index + 1}/{repeat_count}: "
                f"cumulative direct={direct_cum:.1f} s rke={rke_cum:.1f} s",
                flush=True,
            )

    # cumulative curves and measured crossing
    direct_curve = direct_build_total_s + np.cumsum(per_repeat_cost["direct"])
    rke_curve = rke_build_total_s + np.cumsum(per_repeat_cost["rke"])
    above = np.nonzero(rke_curve >= direct_curve)[0]
    if len(above):
        n_cross = int(above[0]) + 1
        crossing_cost = float(direct_curve[above[0]])
        if above[0] == 0:
            interpolated = float(n_cross)
        else:
            i = above[0]
            gap_before = float(direct_curve[i - 1] - rke_curve[i - 1])
            gap_after = float(rke_curve[i] - direct_curve[i])
            interpolated = float(i) + gap_before / max(
                gap_before + gap_after, 1e-300
            )
        measured_break_even = n_cross
    else:
        measured_break_even = ""
        interpolated = ""
        crossing_cost = ""

    direct_mean, direct_std = _solve_statistics(solve_rows, "direct")
    rke_mean, rke_std = _solve_statistics(solve_rows, "rke")
    modeled = _modeled_break_even(
        parameter_count=len(parameters),
        direct_build_s=direct_build_total_s,
        rke_build_s=rke_build_total_s,
        direct_solve_mean_s=direct_mean,
        rke_solve_mean_s=rke_mean,
    )

    # operation counters (E3): computed after every timed phase, so the
    # timing columns are unaffected by the counting itself
    operation_counters = _operation_counters(
        queue=queue,
        traversal=traversal,
        parameters=parameters,
        direct_levels=direct_levels,
        direct_tables=direct_tables,
        direct_build_config=direct_build_config,
        rke_base_table=rke_table,
        split_term_tables=split_term_tables,
        rke_channel_build_config=rke_channel_build_config,
        rke_wranglers={
            parameter: paths[parameter]["rke"] for parameter in parameters
        },
        split_order=split_order,
    )
    print(
        "ops: direct "
        f"{operation_counters['ops_direct_entries_built']} entries x "
        f"{operation_counters['ops_direct_singular_nodes_per_entry']:g} "
        f"{operation_counters['ops_direct_build_routing']} nodes "
        f"({operation_counters['ops_direct_special_function']}); rke "
        f"{operation_counters['ops_rke_channel_entries_built']} entries x "
        f"{operation_counters['ops_rke_channel_singular_nodes_per_entry']} "
        "nodes (elementary); near-field pairs/solve "
        f"{operation_counters['ops_nearfield_point_pairs_per_solve']}, "
        "series nmax "
        f"{operation_counters['ops_split_series_nmax_per_parameter']}",
        flush=True,
    )

    # per-phase shares (E6): analytic phase counts, then dedicated
    # phase-instrumented solves.  Both run after every timed phase, so no
    # reported timing column is perturbed by the instrumentation.
    nmax_by_parameter = [
        int(value)
        for value in str(
            operation_counters["ops_split_series_nmax_per_parameter"]
        ).split(";")
        if value
    ]
    phase_columns = _phase_operation_counts(
        queue=queue,
        traversal=traversal,
        direct_wrangler=paths[parameters[0]]["direct"],
        rke_wrangler=paths[parameters[0]]["rke"],
        q_order=q_order,
        smooth_quad_order=smooth_quad_order,
        nmax_by_parameter=nmax_by_parameter,
        split_table_count=operation_counters["ops_split_table_count"],
    )
    phase_columns.update(
        {
            "phase_profile_repeat_count": phase_repeat_count,
            "ops_phase_setup_direct_table_build": (
                operation_counters["ops_direct_singular_node_evals"]
            ),
            "ops_phase_setup_channel_family_build": (
                operation_counters["ops_rke_channel_singular_node_evals"]
            ),
            "s_phase_setup_direct_table_build": direct_build_total_s,
            "s_phase_setup_direct_table_cache_load": direct_load_total_s,
            "s_phase_setup_channel_family_build": rke_build_total_s,
            "s_phase_setup_channel_family_cache_load": rke_load_total_s,
            # no windowed family is provisioned by this driver
            "s_phase_setup_recombination": 0.0,
        }
    )
    if phase_repeat_count > 0:
        profiles, profile_totals, profile_counts = _profile_solve_phases(
            queue=queue,
            traversal=traversal,
            paths=paths,
            parameters=parameters,
            repeat_count=phase_repeat_count,
        )
        phase_columns.update(
            _phase_second_columns(
                profiles=profiles,
                solve_totals=profile_totals,
                solve_counts=profile_counts,
            )
        )
        print(
            "phase shares (profiled solve, ops / s): "
            + "; ".join(
                f"{strategy} far "
                f"{phase_columns['ops_phase_far_total']} / "
                f"{phase_columns[f's_phase_far_total_{strategy}']:.4f} s, "
                "table "
                f"{phase_columns[f'ops_phase_nearfield_table_apply_{strategy}']}"
                " / "
                f"{phase_columns[f's_phase_nearfield_table_apply_{strategy}']:.4f}"
                " s, correction "
                f"{phase_columns[f's_phase_split_correction_{strategy}']:.4f} s"
                for strategy in PHASE_STRATEGIES
            ),
            flush=True,
        )
    else:
        for field in PHASE_SECONDS_FIELDS:
            phase_columns.setdefault(field, "")
        phase_columns.setdefault("phase_profile_nested_phases", "")
        phase_columns.setdefault("phase_profile_solves_per_strategy", "")

    summary_row = {
        "mode": mode,
        "kernel": "Yukawa",
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "split_order": split_order,
        "direct_regular_quad_order": direct_build_config.regular_quad_order,
        "direct_radial_quad_order": direct_build_config.radial_quad_order,
        "rke_channel_regular_quad_order": (
            rke_channel_build_config.regular_quad_order
        ),
        "rke_channel_radial_quad_order": (
            rke_channel_build_config.radial_quad_order
        ),
        "split_smooth_quad_order": (
            "" if smooth_quad_order is None else smooth_quad_order
        ),
        "parameter_count": len(parameters),
        "parameters": ";".join(f"{p:g}" for p in parameters),
        "level_count": len(direct_levels),
        "direct_levels": ";".join(str(level) for level in direct_levels),
        "repeat_count": repeat_count,
        "n_targets": int(tree.ntargets),
        "direct_build_total_s": direct_build_total_s,
        "rke_build_total_s": rke_build_total_s,
        "direct_warmup_total_s": warmup_totals["direct"],
        "rke_warmup_total_s": warmup_totals["rke"],
        "direct_solve_mean_s": direct_mean,
        "direct_solve_std_s": direct_std,
        "rke_solve_mean_s": rke_mean,
        "rke_solve_std_s": rke_std,
        "measured_break_even_repeat": measured_break_even,
        "measured_break_even_repeat_interpolated": interpolated,
        "measured_crossing_cumulative_cost_s": crossing_cost,
        "modeled_break_even_repeat_from_this_run": modeled,
        "cumulative_cost_definition": (
            "C(n)=strategy_build_total_s+sum_parameters sum_{k<=n} "
            "solve_wall_s;warmup_excluded;interleaved_strategies_per_repeat"
        ),
        "max_rel_l2_rke_vs_direct": max_rel_l2,
        "benchmark_total_s": time.perf_counter() - benchmark_start,
        "direct_provisioning": direct_provisioning,
        **operation_counters,
        **phase_columns,
    }
    return solve_rows, summary_row


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/break-even-validation"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/break-even-cache"),
    )
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--split-order", type=int, default=3)
    parser.add_argument(
        "--direct-provisioning",
        choices=("eager", "lazy"),
        default="eager",
        help="direct-baseline provisioning strategy: 'eager' builds every "
        "anticipated level per parameter (the committed-artifact default); "
        "'lazy' builds only the leaf level the priced workload touches "
        "(the executed lazy baseline of experiment E3)",
    )
    parser.add_argument(
        "--phase-repeat-count",
        type=int,
        help="dedicated phase-instrumented solves per parameter and "
        "strategy for the E6 per-phase shares (default: 2 in smoke mode, 5 "
        "in full mode).  These run after every timed phase and never enter "
        "the cost curves; 0 disables phase timing and leaves the s_phase_* "
        "columns empty while keeping the ops_phase_* counts",
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = 2 if smoke else 4
    nlevels = 2 if smoke else 5
    fmm_order = 8 if smoke else 16
    parameters = [4.0] if smoke else [4.0, 8.0, 12.0]
    direct_levels = _resolve_direct_levels(
        smoke=smoke,
        provisioning=args.direct_provisioning,
        nlevels=nlevels,
    )
    repeat_count = args.repeat_count
    if repeat_count is None:
        repeat_count = 6 if smoke else 400
    warmup_count = 1 if smoke else 2
    phase_repeat_count = args.phase_repeat_count
    if phase_repeat_count is None:
        phase_repeat_count = 2 if smoke else 5
    if phase_repeat_count < 0:
        raise ValueError("--phase-repeat-count must be >= 0")

    solve_rows, summary_row = run_validation(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        fmm_order=fmm_order,
        split_order=args.split_order,
        parameters=parameters,
        direct_levels=direct_levels,
        repeat_count=repeat_count,
        warmup_count=warmup_count,
        direct_provisioning=args.direct_provisioning,
        phase_repeat_count=phase_repeat_count,
    )

    _write_csv(args.out_dir / "break_even_solves.csv", SOLVE_FIELDS, solve_rows)
    _write_csv(
        args.out_dir / "break_even_summary.csv", SUMMARY_FIELDS, [summary_row]
    )
    if phase_repeat_count > 0:
        _write_csv(
            args.out_dir / "break_even_phases.csv",
            PHASE_FIELDS,
            _phase_rows(summary_row),
        )
    print(
        "measured break-even repeat: "
        f"{summary_row['measured_break_even_repeat']} "
        f"(modeled from this run: "
        f"{summary_row['modeled_break_even_repeat_from_this_run']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

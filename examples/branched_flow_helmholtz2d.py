"""Free-space 2D Helmholtz branched flow in a smooth random medium.

The refractive-index perturbation is generated on a rectangular core and
multiplied by a C-infinity window that decays to zero in an outer rectangular
transition layer.  The resulting compactly supported contrast is solved with
the constant-background Lippmann--Schwinger equation

    sigma - m V_k[sigma] = m u_inc,
    u = u_inc + V_k[sigma],

where ``m = k**2 * (n**2 - 1)`` and ``V_k`` is the outgoing free-space
Helmholtz volume potential.  No artificial outer boundary, BIE, or PML is
used.

The optional fixed right preconditioners leave this global operator unchanged.
They apply a target-restricted List-1 correction using either the background
wave number or a Lagrange palette of fixed-wave-number tables that approximates
the source-frozen local kernel.

Use ``--smoke`` for a small local validation.  The default configuration is a
large publication-pilot workload and should be run on approved remote
compute resources.  It uses the optional ``pyfmmlib`` backend; install the
``fmmlib`` project extra before running it.
"""

from __future__ import annotations

__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

import volumential.meshgen as mg


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    q_order: int
    nlevels: int
    fmm_order: int
    fmm_order_mode: str
    fmm_order_padding: int
    fmm_backend: str
    wave_number: float
    root_half_extent: float
    core_x_min: float
    core_x_max: float
    core_y_min: float
    core_y_max: float
    transition_width: float
    correlation_length: float
    refractive_rms: float
    incident_angle_degrees: float
    random_seed: int
    gaussian_bumps: int
    medium_generator: str
    medium_grid_points_per_correlation: int
    medium_filter_truncate: float
    medium_realization: str
    medium_reference_half_extent: float
    medium_normalization_samples: int
    gmres_tolerance: float
    gmres_restart: int
    gmres_maxiter: int
    preconditioner: str
    preconditioner_palette_size: int
    preconditioner_damping: float
    preconditioner_ordering: str
    preconditioner_degree: int
    smoke: bool


def _device_supports_fp64(device):
    has_khr_fp64 = "cl_khr_fp64" in getattr(device, "extensions", "").split()
    has_double_config = bool(getattr(device, "double_fp_config", 0))
    return has_khr_fp64 or has_double_config


def _select_opencl_device(cl_module):
    try:
        platforms = cl_module.get_platforms()
    except cl_module.LogicError as exc:
        raise RuntimeError(f"OpenCL platforms unavailable: {exc}") from exc

    native_gpu_platforms = [
        platform
        for platform in platforms
        if "portable computing language" not in platform.name.lower()
    ]
    for platform in native_gpu_platforms:
        for dev in platform.get_devices():
            if dev.type & cl_module.device_type.GPU and _device_supports_fp64(dev):
                return dev

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl_module.device_type.GPU and _device_supports_fp64(dev):
                return dev

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl_module.device_type.CPU and _device_supports_fp64(dev):
                return dev

    raise RuntimeError("No OpenCL GPU/CPU device with fp64 support found")


def _smooth_ramp(values):
    """Return a C-infinity ramp from zero at 0 to one at 1."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    result[values <= 0.0] = 0.0
    result[values >= 1.0] = 1.0

    interior = (values > 0.0) & (values < 1.0)
    t = values[interior]
    left = np.exp(-1.0 / t)
    right = np.exp(-1.0 / (1.0 - t))
    result[interior] = left / (left + right)
    return result


def _axis_window(values, core_min, core_max, transition_width):
    if not core_min < core_max:
        raise ValueError("core_min must be less than core_max")
    if not transition_width > 0.0:
        raise ValueError("transition_width must be positive")

    left = _smooth_ramp((values - (core_min - transition_width))
                        / transition_width)
    right = _smooth_ramp(((core_max + transition_width) - values)
                         / transition_width)
    return left * right


def _box_window(x, y, config):
    return (
        _axis_window(
            x, config.core_x_min, config.core_x_max, config.transition_width
        )
        * _axis_window(
            y, config.core_y_min, config.core_y_max, config.transition_width
        )
    )


def _filtered_grid_raw_field(x, y, config):
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import gaussian_filter

    spacing = (
        config.correlation_length / config.medium_grid_points_per_correlation
    )
    filter_radius = int(np.ceil(
        config.medium_filter_truncate
        * config.medium_grid_points_per_correlation
    ))
    if config.medium_realization == "nested":
        half_extent = config.medium_reference_half_extent
    else:
        half_extent = config.root_half_extent
    halo_points = filter_radius + 2
    index_min = int(np.floor(-half_extent / spacing)) - halo_points
    index_max = int(np.ceil(half_extent / spacing)) + halo_points
    grid_axis = spacing * np.arange(index_min, index_max + 1)

    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(config.random_seed))
    )
    white = rng.standard_normal((grid_axis.size, grid_axis.size))
    filtered = gaussian_filter(
        white,
        sigma=config.medium_grid_points_per_correlation,
        radius=filter_radius,
        mode="constant",
        cval=0.0,
        output=np.float64,
    )
    interpolator = RectBivariateSpline(
        grid_axis,
        grid_axis,
        filtered,
        kx=3,
        ky=3,
        s=0.0,
    )
    raw = interpolator.ev(y, x)

    if config.medium_realization == "nested":
        reference = config.medium_reference_half_extent
        core_x = (grid_axis >= -0.8 * reference) & (
            grid_axis <= 0.375 * reference
        )
        core_y = (grid_axis >= -0.625 * reference) & (
            grid_axis <= 0.625 * reference
        )
        reference_values = filtered[np.ix_(core_y, core_x)]
        return raw, float(np.mean(reference_values)), float(np.std(reference_values))

    return raw, None, None


def _random_refractive_perturbation(x, y, weights, window, config):
    if config.medium_generator == "filtered-grid":
        raw, fixed_mean, fixed_rms = _filtered_grid_raw_field(x, y, config)
    else:
        raw, fixed_mean, fixed_rms = _gaussian_bump_raw_field(x, y, config)

    core = (
        (x >= config.core_x_min)
        & (x <= config.core_x_max)
        & (y >= config.core_y_min)
        & (y <= config.core_y_max)
    )
    if not np.any(core):
        raise RuntimeError("The discretization has no points in the medium core")

    if fixed_mean is not None and fixed_rms is not None:
        mean = fixed_mean
        rms = fixed_rms
    else:
        core_weights = weights[core]
        core_raw = raw[core]
        weight_sum = np.sum(core_weights)
        mean = np.sum(core_weights * core_raw) / weight_sum
        rms = np.sqrt(
            np.sum(core_weights * (core_raw - mean) ** 2) / weight_sum
        )

    if not rms > 0.0:
        raise RuntimeError("Random refractive field has zero RMS")
    return config.refractive_rms * window * (raw - mean) / rms


def _gaussian_bump_raw_field(x, y, config):
    rng = np.random.default_rng(config.random_seed)
    if config.medium_realization == "nested":
        center_x_min = -0.8 * config.medium_reference_half_extent
        center_x_max = 0.375 * config.medium_reference_half_extent
        center_y_min = -0.625 * config.medium_reference_half_extent
        center_y_max = 0.625 * config.medium_reference_half_extent
    else:
        center_x_min = config.core_x_min
        center_x_max = config.core_x_max
        center_y_min = config.core_y_min
        center_y_max = config.core_y_max

    centers_x = rng.uniform(
        center_x_min,
        center_x_max,
        size=config.gaussian_bumps,
    )
    centers_y = rng.uniform(
        center_y_min,
        center_y_max,
        size=config.gaussian_bumps,
    )
    amplitudes = rng.normal(size=config.gaussian_bumps)

    raw = np.zeros_like(x)
    variance_factor = 0.5 / config.correlation_length**2
    for center_x, center_y, amplitude in zip(
        centers_x, centers_y, amplitudes, strict=True
    ):
        radius_squared = (x - center_x) ** 2 + (y - center_y) ** 2
        raw += amplitude * np.exp(-variance_factor * radius_squared)

    if config.medium_realization == "nested":
        normalization_rng = np.random.default_rng(
            np.random.SeedSequence([config.random_seed, 0x4E455354])
        )
        sample_x = normalization_rng.uniform(
            center_x_min,
            center_x_max,
            size=config.medium_normalization_samples,
        )
        sample_y = normalization_rng.uniform(
            center_y_min,
            center_y_max,
            size=config.medium_normalization_samples,
        )
        sample_raw = np.zeros(config.medium_normalization_samples)
        for center_x, center_y, amplitude in zip(
            centers_x, centers_y, amplitudes, strict=True
        ):
            sample_radius_squared = (
                (sample_x - center_x) ** 2 + (sample_y - center_y) ** 2
            )
            sample_raw += amplitude * np.exp(
                -variance_factor * sample_radius_squared
            )
        return raw, float(np.mean(sample_raw)), float(np.std(sample_raw))

    return raw, None, None


def _get_near_field_table(queue, table_filename, config, wave_number=None):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    regular_quad_order = max(8, 4 * config.q_order)
    radial_quad_order = max(21, 10 * config.q_order)
    manager_kwargs = {
        "root_extent": 2.0 * config.root_half_extent,
        "dtype": np.float64,
        "queue": queue,
    }
    table_kwargs = {}
    kernel_name = "Laplace"
    if config.fmm_backend == "fmmlib":
        from sumpy.kernel import HelmholtzKernel

        kernel = HelmholtzKernel(2)
        if wave_number is None:
            wave_number = config.wave_number
        manager_kwargs["dtype"] = np.complex128
        kernel_name = "Helmholtz-Reference"
        table_kwargs = {
            "source_box_level": config.nlevels - 1,
            "sumpy_knl": kernel,
            kernel.helmholtz_k_name: wave_number,
        }

    with NearFieldInteractionTableManager(
        table_filename,
        **manager_kwargs,
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            kernel_name,
            config.q_order,
            queue=queue,
            build_config=DuffyBuildConfig(
                radial_rule="tanh-sinh-fast",
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
            ),
            **table_kwargs,
        )

    return table


def _device_norm(queue, vector):
    return float(np.sqrt(abs(cla.vdot(vector, vector, queue=queue).get(queue))))


def _fmm_level_order(config, tree, level):
    if config.fmm_order_mode == "fixed":
        return config.fmm_order

    box_extent = tree.root_extent * 2.0 ** (-level)
    box_radius = box_extent / np.sqrt(2.0)
    phase_order = int(np.ceil(abs(config.wave_number) * box_radius))
    return max(config.fmm_order, phase_order + config.fmm_order_padding)


def _chebyshev_lobatto_nodes(lower, upper, count):
    if count < 2:
        raise ValueError("Chebyshev-Lobatto interpolation requires at least 2 nodes")
    if not lower < upper:
        raise ValueError("interpolation interval must have positive width")

    angles = np.linspace(np.pi, 0.0, count)
    reference_nodes = np.cos(angles)
    return 0.5 * (
        lower + upper + (upper - lower) * reference_nodes
    )


def _lagrange_interpolation_weights(values, nodes):
    values = np.asarray(values, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("interpolation nodes must be a nonempty vector")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("interpolation nodes must be distinct")

    weights = []
    for index, node in enumerate(nodes):
        weight = np.ones_like(values)
        for other_index, other_node in enumerate(nodes):
            if other_index != index:
                weight *= (values - other_node) / (node - other_node)
        weights.append(weight)
    return np.asarray(weights)


def _restrict_neighbor_csr(
    target_boxes,
    neighbor_starts,
    neighbor_lists,
    active_box_flags,
):
    target_boxes = np.asarray(target_boxes)
    neighbor_starts = np.asarray(neighbor_starts)
    neighbor_lists = np.asarray(neighbor_lists)
    active_box_flags = np.asarray(active_box_flags, dtype=bool)
    if neighbor_starts.shape != (target_boxes.size + 1,):
        raise ValueError("neighbor CSR starts do not match target boxes")
    if neighbor_starts[-1] != neighbor_lists.size:
        raise ValueError("neighbor CSR terminal offset does not match its lists")
    if target_boxes.size and np.max(target_boxes) >= active_box_flags.size:
        raise ValueError("active-box flags do not cover all target boxes")

    selected_positions = np.flatnonzero(active_box_flags[target_boxes])
    restricted_targets = target_boxes[selected_positions]
    restricted_starts = np.zeros(selected_positions.size + 1, dtype=np.int32)
    chunks = []
    for output_index, input_index in enumerate(selected_positions):
        start = int(neighbor_starts[input_index])
        stop = int(neighbor_starts[input_index + 1])
        chunks.append(neighbor_lists[start:stop])
        restricted_starts[output_index + 1] = (
            restricted_starts[output_index] + stop - start
        )

    if chunks:
        restricted_lists = np.concatenate(chunks)
    else:
        restricted_lists = np.empty(0, dtype=neighbor_lists.dtype)
    return restricted_targets, restricted_starts, restricted_lists


def _extract_single_potential(potentials):
    if isinstance(potentials, np.ndarray) and potentials.dtype == object:
        if len(potentials) != 1:
            raise ValueError("expected one potential output")
        return potentials[0]
    return potentials


def _apply_fixed_polynomial(linear_map, vector, degree, damping):
    result = vector
    term = vector
    for _ in range(degree):
        term = linear_map(term)
        result = result + damping * term
    return result


class _VolumeIntegralOperator:
    def __init__(
        self,
        queue,
        traversal,
        wrangler,
        source_weights,
        contrast,
    ):
        self.queue = queue
        self.traversal = traversal
        self.wrangler = wrangler
        self.source_weights = source_weights
        self.contrast = contrast
        self.shape = (len(contrast), len(contrast))
        self.application_count = 0
        self.application_seconds = 0.0

    def volume_potential(self, density):
        from volumential.volume_fmm import drive_volume_fmm

        self.queue.finish()
        start = time.perf_counter()
        potentials = drive_volume_fmm(
            self.traversal,
            self.wrangler,
            density * self.source_weights,
            density,
            direct_evaluation=False,
            list1_only=False,
        )
        if isinstance(potentials, np.ndarray) and potentials.dtype != object:
            potential = potentials
        else:
            (potential,) = potentials
        if not isinstance(potential, cla.Array):
            potential = cla.to_device(
                self.queue,
                np.ascontiguousarray(potential),
            )
        self.queue.finish()
        self.application_count += 1
        self.application_seconds += time.perf_counter() - start
        return potential

    def __call__(self, density):
        return density - self.contrast * self.volume_potential(density)


class _NearFieldPotential:
    """Apply one fixed-wave-number List-1 table on target-owned boxes."""

    def __init__(
        self,
        queue,
        wrangler,
        target_boxes,
        neighbor_starts,
        neighbor_lists,
    ):
        self.queue = queue
        self.wrangler = wrangler
        self.target_boxes = target_boxes
        self.neighbor_starts = neighbor_starts
        self.neighbor_lists = neighbor_lists

    def __call__(self, source_values):
        tree_values = self.wrangler.reorder_sources(source_values)
        potentials, _ = self.wrangler.eval_direct(
            self.target_boxes,
            self.neighbor_starts,
            self.neighbor_lists,
            tree_values,
        )
        potentials = self.wrangler.reorder_potentials(potentials)
        potentials = self.wrangler.finalize_potentials(potentials)
        potential = _extract_single_potential(potentials)
        if not isinstance(potential, cla.Array):
            potential = cla.to_device(
                self.queue,
                np.ascontiguousarray(potential),
            )
        return potential


class _List1PolynomialPreconditioner:
    """Apply a fixed polynomial in a target- or source-ordered List-1 map."""

    def __init__(
        self,
        queue,
        contrast,
        active_mask,
        near_potentials,
        interpolation_weights,
        damping,
        ordering,
        degree,
    ):
        if len(near_potentials) != len(interpolation_weights):
            raise ValueError("each local wave number needs interpolation weights")
        if not near_potentials:
            raise ValueError("at least one local potential is required")
        self.queue = queue
        self.contrast = contrast
        self.active_mask = active_mask
        self.near_potentials = near_potentials
        self.interpolation_weights = interpolation_weights
        self.damping = damping
        self.ordering = ordering
        self.degree = degree
        self.shape = (len(contrast), len(contrast))
        self.application_count = 0
        self.application_seconds = 0.0

    def _apply_local_potential(self, source_values):
        result = cla.zeros_like(source_values)
        for near_potential, interpolation_weight in zip(
            self.near_potentials,
            self.interpolation_weights,
            strict=True,
        ):
            result = result + near_potential(
                interpolation_weight * source_values
            )
        return result

    def __call__(self, residual):
        self.queue.finish()
        start = time.perf_counter()

        def apply_term(term):
            if self.ordering == "target":
                return self.contrast * self._apply_local_potential(term)
            return self.active_mask * self._apply_local_potential(
                self.contrast * term
            )

        result = _apply_fixed_polynomial(
            apply_term,
            residual,
            self.degree,
            self.damping,
        )
        self.queue.finish()
        self.application_count += 1
        self.application_seconds += time.perf_counter() - start
        return result


class _RightPreconditionedOperator:
    def __init__(self, operator, preconditioner):
        if operator.shape != preconditioner.shape:
            raise ValueError("operator and right preconditioner shapes must match")
        self.operator = operator
        self.preconditioner = preconditioner
        self.shape = operator.shape

    def __call__(self, transformed_density):
        return self.operator(self.preconditioner(transformed_density))

    def recover(self, transformed_density):
        return self.preconditioner(transformed_density)


def _default_config(smoke):
    if smoke:
        return RunConfig(
            q_order=3,
            nlevels=4,
            fmm_order=10,
            fmm_order_mode="fixed",
            fmm_order_padding=0,
            fmm_backend="sumpy",
            wave_number=4.0,
            root_half_extent=1.0,
            core_x_min=-0.55,
            core_x_max=0.20,
            core_y_min=-0.40,
            core_y_max=0.40,
            transition_width=0.12,
            correlation_length=0.28,
            refractive_rms=0.03,
            incident_angle_degrees=0.0,
            random_seed=17,
            gaussian_bumps=12,
            medium_generator="gaussian-bumps",
            medium_grid_points_per_correlation=8,
            medium_filter_truncate=6.0,
            medium_realization="local",
            medium_reference_half_extent=8.0,
            medium_normalization_samples=16384,
            gmres_tolerance=1.0e-6,
            gmres_restart=12,
            gmres_maxiter=40,
            preconditioner="none",
            preconditioner_palette_size=5,
            preconditioner_damping=1.0,
            preconditioner_ordering="target",
            preconditioner_degree=1,
            smoke=True,
        )

    return RunConfig(
        q_order=4,
        nlevels=9,
        fmm_order=40,
        fmm_order_mode="helmholtz",
        fmm_order_padding=20,
        fmm_backend="fmmlib",
        wave_number=12.0,
        root_half_extent=8.0,
        core_x_min=-6.40,
        core_x_max=3.00,
        core_y_min=-5.00,
        core_y_max=5.00,
        transition_width=0.50,
        correlation_length=0.55,
        refractive_rms=0.05,
        incident_angle_degrees=0.0,
        random_seed=17,
        gaussian_bumps=2048,
        medium_generator="gaussian-bumps",
        medium_grid_points_per_correlation=8,
        medium_filter_truncate=6.0,
        medium_realization="local",
        medium_reference_half_extent=8.0,
        medium_normalization_samples=16384,
        gmres_tolerance=1.0e-8,
        gmres_restart=20,
        gmres_maxiter=120,
        preconditioner="none",
        preconditioner_palette_size=5,
        preconditioner_damping=1.0,
        preconditioner_ordering="target",
        preconditioner_degree=1,
        smoke=False,
    )


def _validate_config(config):
    root_min = -config.root_half_extent
    root_max = config.root_half_extent
    support_bounds = (
        config.core_x_min - config.transition_width,
        config.core_x_max + config.transition_width,
        config.core_y_min - config.transition_width,
        config.core_y_max + config.transition_width,
    )
    if min(support_bounds[0], support_bounds[2]) <= root_min:
        raise ValueError("The transition layer must remain inside the root box")
    if max(support_bounds[1], support_bounds[3]) >= root_max:
        raise ValueError("The transition layer must remain inside the root box")
    if config.correlation_length <= 0.0:
        raise ValueError("correlation_length must be positive")
    if config.refractive_rms <= 0.0:
        raise ValueError("refractive_rms must be positive")
    if config.fmm_backend not in {"sumpy", "fmmlib"}:
        raise ValueError("fmm_backend must be 'sumpy' or 'fmmlib'")
    if config.fmm_order_mode not in {"fixed", "helmholtz"}:
        raise ValueError("fmm_order_mode must be 'fixed' or 'helmholtz'")
    if config.fmm_order <= 0:
        raise ValueError("fmm_order must be positive")
    if config.fmm_order_padding < 0:
        raise ValueError("fmm_order_padding must be nonnegative")
    if config.preconditioner not in {"none", "constant", "source-frozen"}:
        raise ValueError(
            "preconditioner must be 'none', 'constant', or 'source-frozen'"
        )
    if config.preconditioner != "none" and config.fmm_backend != "fmmlib":
        raise ValueError("List-1 preconditioning currently requires FMMLib tables")
    if (
        config.preconditioner == "source-frozen"
        and config.preconditioner_palette_size < 2
    ):
        raise ValueError("preconditioner_palette_size must be at least 2")
    if not 0.0 <= config.preconditioner_damping <= 1.0:
        raise ValueError("preconditioner_damping must lie in [0, 1]")
    if config.preconditioner_ordering not in {"source", "target"}:
        raise ValueError("preconditioner_ordering must be 'source' or 'target'")
    if config.preconditioner_degree < 1:
        raise ValueError("preconditioner_degree must be at least 1")
    if config.gmres_restart <= 0:
        raise ValueError("gmres_restart must be positive")
    if config.medium_realization not in {"local", "nested"}:
        raise ValueError("medium_realization must be 'local' or 'nested'")
    if config.medium_reference_half_extent <= 0.0:
        raise ValueError("medium_reference_half_extent must be positive")
    if config.medium_normalization_samples <= 0:
        raise ValueError("medium_normalization_samples must be positive")
    if config.medium_generator not in {"gaussian-bumps", "filtered-grid"}:
        raise ValueError(
            "medium_generator must be 'gaussian-bumps' or 'filtered-grid'"
        )
    if config.medium_grid_points_per_correlation < 4:
        raise ValueError("medium_grid_points_per_correlation must be at least 4")
    if config.medium_filter_truncate < 4.0:
        raise ValueError("medium_filter_truncate must be at least 4")
    if (
        config.medium_generator == "filtered-grid"
        and config.medium_realization == "nested"
        and config.medium_reference_half_extent < config.root_half_extent
    ):
        raise ValueError("nested filtered-grid reference must contain the root box")


def _write_outputs(
    output_dir,
    config,
    arrays,
    metadata,
    residual_norms,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "fields.npz", **arrays)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as outf:
        json.dump(metadata, outf, indent=2, sort_keys=True)
        outf.write("\n")

    with (output_dir / "gmres_residuals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as outf:
        writer = csv.writer(outf)
        writer.writerow(["iteration", "relative_residual"])
        for iteration, residual in enumerate(residual_norms):
            writer.writerow([iteration, f"{residual:.17e}"])


def _write_plot(output_dir, arrays):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib is unavailable; skipping plot generation")
        return

    x = arrays["x"]
    y = arrays["y"]
    perturbation = arrays["refractive_index"] - 1.0
    intensity = arrays["intensity"]
    field_real = arrays["total_field"].real
    x_axis = np.unique(x)
    y_axis = np.unique(y)
    x_indices = np.searchsorted(x_axis, x)
    y_indices = np.searchsorted(y_axis, y)

    def as_grid(field):
        grid = np.empty((y_axis.size, x_axis.size), dtype=field.dtype)
        grid[y_indices, x_indices] = field
        return grid

    figure, axes = plt.subplots(
        3, 1, figsize=(9.0, 15.0), sharex=True, sharey=True
    )
    fields = [perturbation, intensity, field_real]
    titles = ["refractive-index perturbation", "normalized intensity", "Re(u)"]
    color_maps = ["RdBu_r", "magma", "RdBu_r"]

    for axis, field, title, color_map in zip(
        axes, fields, titles, color_maps, strict=True
    ):
        if title == "normalized intensity":
            color_min = 0.0
            color_max = np.quantile(field, 0.995)
        else:
            scale = np.quantile(np.abs(field), 0.995)
            color_min = -scale
            color_max = scale
        image = axis.imshow(
            as_grid(field),
            origin="lower",
            extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
            cmap=color_map,
            vmin=color_min,
            vmax=color_max,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)

    axes[-1].set_xlabel("x")
    figure.tight_layout()
    figure.savefig(output_dir / "branched_flow.png", dpi=220)
    plt.close(figure)


def run(config, output_dir):
    from importlib.util import find_spec

    from pytential.linalg.gmres import gmres
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    _validate_config(config)
    if config.fmm_backend == "fmmlib" and find_spec("pyfmmlib") is None:
        raise RuntimeError(
            "The FMMLib backend requires pyfmmlib; install volumential[fmmlib]"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_opencl_device(cl)
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    root_min = -config.root_half_extent
    root_max = config.root_half_extent
    mesh = mg.MeshGen2D(
        config.q_order,
        config.nlevels,
        root_min,
        root_max,
        queue=queue,
    )
    q_points, source_weights, tree, traversal = mg.build_geometry_info(
        context,
        queue,
        2,
        config.q_order,
        mesh,
        bbox=np.array([[root_min, root_max]] * 2, dtype=np.float64),
    )

    coordinates = np.array([axis.get(queue) for axis in q_points])
    x = coordinates[0]
    y = coordinates[1]
    weights_host = source_weights.get(queue)

    window = _box_window(x, y, config)
    delta_n = _random_refractive_perturbation(
        x, y, weights_host, window, config
    )
    refractive_index = 1.0 + delta_n
    if np.min(refractive_index) <= 0.0:
        raise RuntimeError("The generated refractive index is not positive")

    contrast_host = config.wave_number**2 * (refractive_index**2 - 1.0)
    angle = np.deg2rad(config.incident_angle_degrees)
    incident_host = np.exp(
        1j
        * config.wave_number
        * (np.cos(angle) * x + np.sin(angle) * y)
    )

    contrast = cla.to_device(
        queue, np.ascontiguousarray(contrast_host.astype(np.complex128))
    )
    incident = cla.to_device(
        queue, np.ascontiguousarray(incident_host.astype(np.complex128))
    )

    table = _get_near_field_table(
        queue,
        output_dir / "near_field_table.sqlite",
        config,
    )

    kernel = HelmholtzKernel(2)
    expansion_factory = DefaultExpansionFactory()
    local_expansion_class = expansion_factory.get_local_expansion_class(kernel)
    multipole_expansion_class = (
        expansion_factory.get_multipole_expansion_class(kernel)
    )
    self_extra_kwargs = {}
    if tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            tree.ntargets, dtype=np.int32
        )

    wrangler_kwargs = {
        "queue": queue,
        "near_field_table": table,
        "dtype": np.complex128,
        "fmm_level_to_order": (
            lambda kernel, kernel_args, tree, level: _fmm_level_order(
                config, tree, level
            )
        ),
        "quad_order": config.q_order,
        "kernel_extra_kwargs": {kernel.helmholtz_k_name: config.wave_number},
        "self_extra_kwargs": self_extra_kwargs,
    }
    if config.fmm_backend == "sumpy":
        tree_independent = FPNDTreeIndependentDataForWrangler(
            context,
            partial(multipole_expansion_class, kernel),
            partial(local_expansion_class, kernel),
            [kernel],
            exclude_self=True,
        )
        wrangler = FPNDExpansionWrangler(
            tree_indep=tree_independent,
            traversal=traversal,
            helmholtz_split=True,
            **wrangler_kwargs,
        )
    else:
        from volumential.expansion_wrangler_fpnd import (
            FPNDFMMLibExpansionWrangler,
            FPNDFMMLibTreeIndependentDataForWrangler,
        )

        tree_independent = FPNDFMMLibTreeIndependentDataForWrangler(
            context,
            partial(multipole_expansion_class, kernel),
            partial(local_expansion_class, kernel),
            [kernel],
            exclude_self=True,
        )
        wrangler = FPNDFMMLibExpansionWrangler(
            tree_indep=tree_independent,
            tree=tree,
            traversal=traversal,
            **wrangler_kwargs,
        )

    operator = _VolumeIntegralOperator(
        queue,
        traversal,
        wrangler,
        source_weights,
        contrast,
    )
    preconditioner = None
    preconditioner_metadata = {
        "kind": config.preconditioner,
        "formula": (
            (
                "B_sigma = I + omega sum_(j=1)^p (M_m Q_near)^j"
                if config.preconditioner_ordering == "target"
                else "B_sigma = I + omega sum_(j=1)^p (chi_m Q_near M_m)^j"
            )
            if config.preconditioner != "none" else "identity"
        ),
        "representation": (
            "fixed-k-table-lagrange-list1"
            if config.preconditioner == "source-frozen"
            else "fixed-k-table-list1"
            if config.preconditioner == "constant"
            else "none"
        ),
        "damping": config.preconditioner_damping,
        "ordering": config.preconditioner_ordering,
        "degree": config.preconditioner_degree,
        "palette_wave_numbers": [],
        "active_target_boxes": 0,
        "setup_seconds": 0.0,
    }
    if config.preconditioner != "none":
        preconditioner_setup_start = time.perf_counter()
        active_host = contrast_host != 0.0
        active_mask = cla.to_device(
            queue,
            np.ascontiguousarray(active_host.astype(np.float64)),
        )

        if config.preconditioner == "constant":
            palette_wave_numbers = np.array([config.wave_number])
            interpolation_weights_host = np.ones((1, x.size))
        else:
            local_wave_number = config.wave_number * refractive_index
            active_wave_numbers = local_wave_number[active_host]
            palette_wave_numbers = _chebyshev_lobatto_nodes(
                float(np.min(active_wave_numbers)),
                float(np.max(active_wave_numbers)),
                config.preconditioner_palette_size,
            )
            interpolation_weights_host = _lagrange_interpolation_weights(
                local_wave_number,
                palette_wave_numbers,
            )

        local_tree_independent = FPNDTreeIndependentDataForWrangler(
            context,
            partial(multipole_expansion_class, kernel),
            partial(local_expansion_class, kernel),
            [kernel],
            exclude_self=True,
        )
        local_wranglers = []
        for local_k in palette_wave_numbers:
            if np.isclose(local_k, config.wave_number, rtol=0.0, atol=1.0e-14):
                local_table = table
            else:
                local_table = _get_near_field_table(
                    queue,
                    output_dir / "near_field_table.sqlite",
                    config,
                    wave_number=float(local_k),
                )
            local_wranglers.append(
                FPNDExpansionWrangler(
                    tree_indep=local_tree_independent,
                    queue=queue,
                    traversal=traversal,
                    near_field_table=local_table,
                    dtype=np.complex128,
                    fmm_level_to_order=(
                        lambda kernel, kernel_args, tree, level: _fmm_level_order(
                            config, tree, level
                        )
                    ),
                    quad_order=config.q_order,
                    kernel_extra_kwargs={
                        kernel.helmholtz_k_name: float(local_k),
                    },
                    self_extra_kwargs=self_extra_kwargs,
                    helmholtz_split=False,
                )
            )

        ordered_active = local_wranglers[0].reorder_sources(active_mask)
        if isinstance(ordered_active, cla.Array):
            ordered_active_host = ordered_active.get(queue)
        else:
            ordered_active_host = np.asarray(ordered_active)

        tree = traversal.tree
        target_boxes_host = traversal.target_boxes.get(queue)
        neighbor_starts_host = traversal.neighbor_source_boxes_starts.get(queue)
        neighbor_lists_host = traversal.neighbor_source_boxes_lists.get(queue)
        box_target_starts = tree.box_target_starts.get(queue)
        box_target_counts = tree.box_target_counts_nonchild.get(queue)
        active_box_flags = np.zeros(tree.box_centers.shape[1], dtype=bool)
        for target_box in target_boxes_host:
            start = int(box_target_starts[target_box])
            stop = start + int(box_target_counts[target_box])
            active_box_flags[target_box] = np.any(
                ordered_active_host[start:stop] != 0.0
            )

        (
            restricted_target_boxes,
            restricted_neighbor_starts,
            restricted_neighbor_lists,
        ) = _restrict_neighbor_csr(
            target_boxes_host,
            neighbor_starts_host,
            neighbor_lists_host,
            active_box_flags,
        )
        restricted_target_boxes = cla.to_device(
            queue,
            np.ascontiguousarray(restricted_target_boxes),
        )
        restricted_neighbor_starts = cla.to_device(
            queue,
            np.ascontiguousarray(restricted_neighbor_starts),
        )
        restricted_neighbor_lists = cla.to_device(
            queue,
            np.ascontiguousarray(restricted_neighbor_lists),
        )

        near_potentials = [
            _NearFieldPotential(
                queue,
                local_wrangler,
                restricted_target_boxes,
                restricted_neighbor_starts,
                restricted_neighbor_lists,
            )
            for local_wrangler in local_wranglers
        ]
        interpolation_weights = [
            cla.to_device(queue, np.ascontiguousarray(weight))
            for weight in interpolation_weights_host
        ]
        preconditioner = _List1PolynomialPreconditioner(
            queue,
            contrast,
            active_mask,
            near_potentials,
            interpolation_weights,
            config.preconditioner_damping,
            config.preconditioner_ordering,
            config.preconditioner_degree,
        )
        queue.finish()
        preconditioner_metadata.update({
            "palette_wave_numbers": [
                float(local_k) for local_k in palette_wave_numbers
            ],
            "active_target_boxes": int(restricted_target_boxes.size),
            "setup_seconds": float(
                time.perf_counter() - preconditioner_setup_start
            ),
        })

    right_hand_side = contrast * incident
    rhs_norm = _device_norm(queue, right_hand_side)
    if not rhs_norm > 0.0:
        raise RuntimeError("The Lippmann--Schwinger right-hand side is zero")

    solve_start = time.perf_counter()
    solve_operator = operator
    if preconditioner is not None:
        solve_operator = _RightPreconditionedOperator(operator, preconditioner)
    result = gmres(
        solve_operator,
        right_hand_side,
        restart=config.gmres_restart,
        tol=config.gmres_tolerance,
        maxiter=config.gmres_maxiter,
        hard_failure=False,
        stall_iterations=0,
        require_monotonicity=False,
        inner_product=lambda a, b: cla.vdot(a, b, queue=queue).get(queue).item(),
    )
    queue.finish()
    solve_seconds = time.perf_counter() - solve_start

    if preconditioner is None:
        sigma = result.solution
    else:
        sigma = solve_operator.recover(result.solution)
    true_residual = operator(sigma) - right_hand_side
    relative_true_residual = _device_norm(queue, true_residual) / rhs_norm
    scattered = operator.volume_potential(sigma)
    total = incident + scattered

    sigma_host = sigma.get(queue)
    scattered_host = scattered.get(queue)
    total_host = total.get(queue)
    intensity = np.abs(total_host) ** 2
    incident_intensity = np.mean(np.abs(incident_host) ** 2)
    intensity /= incident_intensity

    transition = (window > 0.0) & (window < 1.0)
    active = window > 0.0
    core = (
        (x >= config.core_x_min)
        & (x <= config.core_x_max)
        & (y >= config.core_y_min)
        & (y <= config.core_y_max)
    )
    core_weight_sum = np.sum(weights_host[core])
    core_perturbation_mean = float(
        np.sum(weights_host[core] * delta_n[core]) / core_weight_sum
    )
    core_perturbation_rms = float(
        np.sqrt(
            np.sum(weights_host[core] * delta_n[core] ** 2)
            / core_weight_sum
        )
    )
    residual_norms = [float(value) / rhs_norm for value in result.residual_norms]
    arrays = {
        "x": x,
        "y": y,
        "quadrature_weights": weights_host,
        "window": window,
        "refractive_index": refractive_index,
        "contrast": contrast_host,
        "incident_field": incident_host,
        "density": sigma_host,
        "scattered_field": scattered_host,
        "total_field": total_host,
        "intensity": intensity,
    }
    metadata = {
        "config": asdict(config),
        "equation": "sigma - m V_k[sigma] = m u_inc",
        "kernel": "outgoing 2D Helmholtz i/4 H_0^(1)(k r)",
        "boundary_treatment": "none; free-space outgoing Green function",
        "support": {
            "core": [
                config.core_x_min,
                config.core_x_max,
                config.core_y_min,
                config.core_y_max,
            ],
            "outer": [
                config.core_x_min - config.transition_width,
                config.core_x_max + config.transition_width,
                config.core_y_min - config.transition_width,
                config.core_y_max + config.transition_width,
            ],
            "transition_width": config.transition_width,
            "window": "C-infinity tensor-product box ramp",
        },
        "discretization": {
            "points": int(x.size),
            "active_points": int(np.count_nonzero(active)),
            "transition_points": int(np.count_nonzero(transition)),
            "leaf_side_length": (
                2.0 * config.root_half_extent / 2 ** (config.nlevels - 1)
            ),
            "local_kh": (
                config.wave_number
                * 2.0
                * config.root_half_extent
                / 2 ** (config.nlevels - 1)
            ),
            "nominal_points_per_wavelength": (
                2.0
                * np.pi
                * config.q_order
                * 2 ** (config.nlevels - 1)
                / (config.wave_number * 2.0 * config.root_half_extent)
            ),
            "fmm_level_orders": [
                _fmm_level_order(config, tree, level)
                for level in range(tree.nlevels)
            ],
        },
        "medium": {
            "generator": config.medium_generator,
            "realization": config.medium_realization,
            "refractive_index_sha256": hashlib.sha256(
                np.ascontiguousarray(refractive_index).view(np.uint8)
            ).hexdigest(),
            "minimum_refractive_index": float(np.min(refractive_index)),
            "maximum_refractive_index": float(np.max(refractive_index)),
            "core_refractive_perturbation_mean": core_perturbation_mean,
            "core_refractive_perturbation_rms": core_perturbation_rms,
            "filtered_grid": (
                {
                    "grid_spacing": (
                        config.correlation_length
                        / config.medium_grid_points_per_correlation
                    ),
                    "filter_sigma_grid_points": (
                        config.medium_grid_points_per_correlation
                    ),
                    "filter_radius_grid_points": int(np.ceil(
                        config.medium_filter_truncate
                        * config.medium_grid_points_per_correlation
                    )),
                    "covariance_model": (
                        "approximately exp(-r^2/(4*correlation_length^2))"
                    ),
                }
                if config.medium_generator == "filtered-grid" else None
            ),
            "core_length_wavelengths": float(
                (config.core_x_max - config.core_x_min)
                * config.wave_number
                / (2.0 * np.pi)
            ),
            "correlation_length_wavelengths": float(
                config.correlation_length * config.wave_number / (2.0 * np.pi)
            ),
            "transition_width_wavelengths": float(
                config.transition_width * config.wave_number / (2.0 * np.pi)
            ),
            "maximum_window_outside_outer_box": float(np.max(window[~active]))
                if np.any(~active) else 0.0,
        },
        "gmres": {
            "success": bool(result.success),
            "state": result.state,
            "iteration_count": int(result.iteration_count),
            "reported_residual_norms": residual_norms,
            "relative_true_residual": float(relative_true_residual),
        },
        "timing": {
            "solve_seconds": float(solve_seconds),
            "volume_applications": int(operator.application_count),
            "volume_application_seconds": float(operator.application_seconds),
        },
        "preconditioner": {
            **preconditioner_metadata,
            "application_count": (
                int(preconditioner.application_count)
                if preconditioner is not None else 0
            ),
            "local_potential_application_count": (
                int(
                    preconditioner.application_count
                    * config.preconditioner_degree
                    * len(preconditioner.near_potentials)
                )
                if preconditioner is not None else 0
            ),
            "application_seconds": (
                float(preconditioner.application_seconds)
                if preconditioner is not None else 0.0
            ),
        },
    }

    _write_outputs(output_dir, config, arrays, metadata, residual_norms)
    _write_plot(output_dir, arrays)

    if config.smoke:
        if not result.success:
            raise RuntimeError(f"Branched-flow smoke GMRES failed: {result.state}")
        if relative_true_residual > 5.0e-5:
            raise RuntimeError(
                "Branched-flow smoke true residual exceeded tolerance band: "
                f"{relative_true_residual:.3e}"
            )
        outer = (
            (x <= config.core_x_min - config.transition_width)
            | (x >= config.core_x_max + config.transition_width)
            | (y <= config.core_y_min - config.transition_width)
            | (y >= config.core_y_max + config.transition_width)
        )
        if np.max(np.abs(contrast_host[outer])) != 0.0:
            raise RuntimeError("Compact-support smoke check failed")

    return metadata


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/branched-flow-helmholtz2d"),
    )
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--fmm-order-mode", choices=("fixed", "helmholtz"))
    parser.add_argument("--fmm-order-padding", type=int)
    parser.add_argument("--fmm-backend", choices=("sumpy", "fmmlib"))
    parser.add_argument("--wave-number", type=float)
    parser.add_argument("--root-half-extent", type=float)
    parser.add_argument("--core-x-min", type=float)
    parser.add_argument("--core-x-max", type=float)
    parser.add_argument("--core-y-min", type=float)
    parser.add_argument("--core-y-max", type=float)
    parser.add_argument("--transition-width", type=float)
    parser.add_argument("--refractive-rms", type=float)
    parser.add_argument("--correlation-length", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--gaussian-bumps", type=int)
    parser.add_argument(
        "--medium-generator",
        choices=("gaussian-bumps", "filtered-grid"),
    )
    parser.add_argument("--medium-grid-points-per-correlation", type=int)
    parser.add_argument("--medium-filter-truncate", type=float)
    parser.add_argument(
        "--medium-realization",
        choices=("local", "nested"),
    )
    parser.add_argument("--medium-reference-half-extent", type=float)
    parser.add_argument("--medium-normalization-samples", type=int)
    parser.add_argument("--gmres-tolerance", type=float)
    parser.add_argument("--gmres-restart", type=int)
    parser.add_argument("--gmres-maxiter", type=int)
    parser.add_argument(
        "--preconditioner",
        choices=("none", "constant", "source-frozen"),
    )
    parser.add_argument("--preconditioner-palette-size", type=int)
    parser.add_argument("--preconditioner-damping", type=float)
    parser.add_argument(
        "--preconditioner-ordering",
        choices=("source", "target"),
    )
    parser.add_argument("--preconditioner-degree", type=int)
    return parser.parse_args()


def _config_from_arguments(arguments):
    config = _default_config(arguments.smoke)
    overrides = {
        "q_order": arguments.q_order,
        "nlevels": arguments.nlevels,
        "fmm_order": arguments.fmm_order,
        "fmm_order_mode": arguments.fmm_order_mode,
        "fmm_order_padding": arguments.fmm_order_padding,
        "fmm_backend": arguments.fmm_backend,
        "wave_number": arguments.wave_number,
        "root_half_extent": arguments.root_half_extent,
        "core_x_min": arguments.core_x_min,
        "core_x_max": arguments.core_x_max,
        "core_y_min": arguments.core_y_min,
        "core_y_max": arguments.core_y_max,
        "transition_width": arguments.transition_width,
        "refractive_rms": arguments.refractive_rms,
        "correlation_length": arguments.correlation_length,
        "random_seed": arguments.random_seed,
        "gaussian_bumps": arguments.gaussian_bumps,
        "medium_generator": arguments.medium_generator,
        "medium_grid_points_per_correlation": (
            arguments.medium_grid_points_per_correlation
        ),
        "medium_filter_truncate": arguments.medium_filter_truncate,
        "medium_realization": arguments.medium_realization,
        "medium_reference_half_extent": arguments.medium_reference_half_extent,
        "medium_normalization_samples": arguments.medium_normalization_samples,
        "gmres_tolerance": arguments.gmres_tolerance,
        "gmres_restart": arguments.gmres_restart,
        "gmres_maxiter": arguments.gmres_maxiter,
        "preconditioner": arguments.preconditioner,
        "preconditioner_palette_size": arguments.preconditioner_palette_size,
        "preconditioner_damping": arguments.preconditioner_damping,
        "preconditioner_ordering": arguments.preconditioner_ordering,
        "preconditioner_degree": arguments.preconditioner_degree,
    }
    values = asdict(config)
    values.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    return RunConfig(**values)


def main():
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments()
    config = _config_from_arguments(arguments)
    metadata = run(config, arguments.output_dir)
    print(json.dumps({
        "output_dir": str(arguments.output_dir),
        "points": metadata["discretization"]["points"],
        "local_kh": metadata["discretization"]["local_kh"],
        "gmres_iterations": metadata["gmres"]["iteration_count"],
        "relative_true_residual": metadata["gmres"]["relative_true_residual"],
        "solve_seconds": metadata["timing"]["solve_seconds"],
        "preconditioner": metadata["preconditioner"],
    }, indent=2))


if __name__ == "__main__":
    main()

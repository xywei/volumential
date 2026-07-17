"""Gaussian source fixtures and exports for free-space volume-potential demos."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf


@dataclass(frozen=True)
class GaussianComponent:
    """One isotropic Gaussian component ``amplitude * exp(-alpha * |x-c|^2)``."""

    amplitude: float
    center: tuple[float, ...]
    alpha: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        if not center:
            raise ValueError("center must have at least one coordinate")
        if not np.all(np.isfinite(center)):
            raise ValueError("center coordinates must be finite")
        if not math.isfinite(float(self.amplitude)):
            raise ValueError("amplitude must be finite")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise ValueError("alpha must be finite and positive")

        object.__setattr__(self, "amplitude", float(self.amplitude))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def dim(self) -> int:
        return len(self.center)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "center": list(self.center),
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class GaussianMixture:
    """A named collection of compatible :class:`GaussianComponent` objects."""

    name: str
    components: tuple[GaussianComponent, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("components must not be empty")
        dim = components[0].dim
        if any(component.dim != dim for component in components):
            raise ValueError("all Gaussian components must have the same dimension")
        object.__setattr__(self, "components", components)

    @property
    def dim(self) -> int:
        return self.components[0].dim

    def as_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "components": [component.as_metadata() for component in self.components],
        }


def default_overlapping_gaussian_mixture(dim: int = 3) -> GaussianMixture:
    """Return a deterministic positive overlapping mixture for smoke demos."""

    if dim < 1 or dim > 3:
        raise ValueError("default mixture is available for dimensions 1, 2, or 3")

    return GaussianMixture(
        name=f"overlapping-gaussian-mixture-{dim}d",
        components=(
            GaussianComponent(1.0, (-0.10, 0.04, -0.05)[:dim], 75.0),
            GaussianComponent(0.70, (0.08, -0.07, 0.06)[:dim], 55.0),
        ),
    )


def _as_points(points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (npoints, dim) or (dim, npoints)")
    if dim is None:
        if 1 <= points.shape[1] <= 3:
            return np.ascontiguousarray(points)
        if 1 <= points.shape[0] <= 3:
            return np.ascontiguousarray(points.T)
        raise ValueError("could not infer point dimension")
    if points.shape[1] == dim:
        return np.ascontiguousarray(points)
    if points.shape[0] == dim:
        return np.ascontiguousarray(points.T)
    raise ValueError(f"points do not match dimension {dim}")


def _as_bbox(bbox: np.ndarray, dim: int | None = None) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float64)
    if bbox.ndim != 2 or bbox.shape[1] != 2:
        raise ValueError("bbox must have shape (dim, 2)")
    if dim is not None and bbox.shape[0] != dim:
        raise ValueError(f"bbox does not match dimension {dim}")
    if not np.all(np.isfinite(bbox)):
        raise ValueError("bbox entries must be finite")
    if np.any(bbox[:, 1] <= bbox[:, 0]):
        raise ValueError("bbox upper bounds must exceed lower bounds")
    return np.ascontiguousarray(bbox)


def evaluate_gaussian_mixture(mixture: GaussianMixture, points: np.ndarray) -> np.ndarray:
    """Evaluate the source density at ``points``."""

    points = _as_points(points, mixture.dim)
    result = np.zeros(points.shape[0], dtype=np.float64)
    for component in mixture.components:
        center = np.asarray(component.center, dtype=np.float64)
        radius_sq = np.sum((points - center) ** 2, axis=1)
        result += component.amplitude * np.exp(-component.alpha * radius_sq)
    return result


def _laplacian_power_polynomial_coefficients(
    *, alpha: float, dim: int, order: int
) -> tuple[float, ...]:
    """Return coefficients for ``Delta**order exp(-alpha*r**2)``."""

    if order == 0:
        return (1.0,)
    if order == 1:
        return (-2.0 * alpha * dim, 4.0 * alpha**2)
    if order == 2:
        return (
            4.0 * alpha**2 * dim * (dim + 2.0),
            -16.0 * alpha**3 * (dim + 2.0),
            16.0 * alpha**4,
        )
    if order == 3:
        return (
            -8.0 * alpha**3 * dim * (dim + 2.0) * (dim + 4.0),
            48.0 * alpha**4 * (dim + 2.0) * (dim + 4.0),
            -96.0 * alpha**5 * (dim + 4.0),
            64.0 * alpha**6,
        )
    raise ValueError("order must be 0, 1, 2, or 3")


def evaluate_gaussian_laplacian_power(
    mixture: GaussianMixture,
    points: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Evaluate ``Delta**order`` of an isotropic Gaussian mixture.

    Orders through three are provided because the fourth-order DMK residual
    effective-density correction needs ``Delta rho``, ``Delta**2 rho``, and
    ``Delta**3 rho`` for a Gaussian source.
    """

    if not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0:
        raise ValueError("order must be non-negative")

    points = _as_points(points, mixture.dim)
    result = np.zeros(points.shape[0], dtype=np.float64)
    for component in mixture.components:
        center = np.asarray(component.center, dtype=np.float64)
        radius_sq = np.sum((points - center) ** 2, axis=1)
        gaussian = component.amplitude * np.exp(-component.alpha * radius_sq)
        coefficients = _laplacian_power_polynomial_coefficients(
            alpha=component.alpha, dim=component.dim, order=order
        )
        coefficient_iterator = reversed(coefficients)
        polynomial = next(coefficient_iterator)
        for coefficient in coefficient_iterator:
            polynomial = polynomial * radius_sq + coefficient
        result += gaussian * polynomial
    return result


def dmk_gaussian_split_sigma(box_side_length: float, epsilon: float) -> float:
    """Return the DMK-style Gaussian split scale for one box level.

    Jiang--Greengard DMK chooses a Gaussian kernel-splitting scale proportional
    to ``r_l / sqrt(log(1/epsilon))`` for box side length ``r_l`` and requested
    precision ``epsilon``. This helper records that normalization for controlled
    diagnostics; it does not include the separate residual sum-of-Gaussians fit.
    """

    box_side_length = float(box_side_length)
    epsilon = float(epsilon)
    if not math.isfinite(box_side_length) or box_side_length <= 0.0:
        raise ValueError("box_side_length must be finite and positive")
    if not math.isfinite(epsilon) or not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be finite and between 0 and 1")
    return box_side_length / math.sqrt(math.log(1.0 / epsilon))


def gaussian_filter_mixture(mixture: GaussianMixture, sigma: float) -> GaussianMixture:
    """Convolve ``mixture`` with a normalized isotropic Gaussian filter.

    The filter is

    ``gamma_sigma(x) = (2*pi*sigma**2)^(-d/2) exp(-|x|**2/(2*sigma**2))``

    and has Fourier multiplier ``exp(-sigma**2 |k|**2 / 2)`` under the
    convention where the 3D Laplace Green's function ``1/(4*pi*r)`` has
    multiplier ``1/|k|**2``. Convolving each ``A exp(-alpha |x-c|^2)`` component
    preserves its mass and maps it to another Gaussian with

    ``alpha_eff = alpha / (1 + 2 alpha sigma**2)``.
    """

    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    components = []
    for component in mixture.components:
        alpha_eff = component.alpha / (1.0 + 2.0 * component.alpha * sigma**2)
        amplitude_eff = component.amplitude * (alpha_eff / component.alpha) ** (
            component.dim / 2.0
        )
        components.append(
            GaussianComponent(
                amplitude=amplitude_eff,
                center=component.center,
                alpha=alpha_eff,
            )
        )

    return GaussianMixture(
        name=f"{mixture.name}-gaussian-filter-sigma-{sigma:.6g}",
        components=tuple(components),
    )


def laplace3d_gaussian_potential(
    mixture: GaussianMixture,
    points: np.ndarray,
    *,
    kernel_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate the full-space 3D Laplace potential of ``mixture``.

    The default ``kernel_scale=1`` gives the unnormalized kernel
    ``K(x, y) = 1 / |x-y|``. Use ``kernel_scale=1/(4*pi)`` for the classical
    Green's function normalization used by Sumpy's 3D Laplace global scaling.
    """

    if mixture.dim != 3:
        raise ValueError("the analytic Laplace Gaussian reference is 3D-only")

    points = _as_points(points, 3)
    result = np.zeros(points.shape[0], dtype=np.float64)
    for component in mixture.components:
        center = np.asarray(component.center, dtype=np.float64)
        radius = np.linalg.norm(points - center, axis=1)
        sqrt_alpha = math.sqrt(component.alpha)
        component_mass = component.amplitude * (math.pi / component.alpha) ** 1.5

        contribution = np.empty_like(radius)
        small = sqrt_alpha * radius < 1.0e-8
        regular = ~small
        contribution[regular] = (
            component_mass * erf(sqrt_alpha * radius[regular]) / radius[regular]
        )
        contribution[small] = (
            2.0
            * math.pi
            * component.amplitude
            / component.alpha
            * (1.0 - component.alpha * radius[small] ** 2 / 3.0)
        )
        result += contribution

    return float(kernel_scale) * result


def _component_shape_mass(component: GaussianComponent) -> float:
    return (math.pi / component.alpha) ** (component.dim / 2.0)


def _component_box_shape_mass(component: GaussianComponent, bbox: np.ndarray) -> float:
    sqrt_alpha = math.sqrt(component.alpha)
    factor = math.sqrt(math.pi) / (2.0 * sqrt_alpha)
    integral = 1.0
    for axis, center in enumerate(component.center):
        lo, hi = bbox[axis]
        integral *= factor * (
            math.erf(sqrt_alpha * (hi - center))
            - math.erf(sqrt_alpha * (lo - center))
        )
    return integral


def gaussian_mixture_tail_report(
    mixture: GaussianMixture,
    bbox: np.ndarray,
) -> dict[str, Any]:
    """Report signed and absolute Gaussian mass omitted outside ``bbox``."""

    bbox = _as_bbox(bbox, mixture.dim)
    total_signed = 0.0
    total_abs = 0.0
    in_box_signed = 0.0
    in_box_abs = 0.0
    component_reports = []

    for component in mixture.components:
        full_shape = _component_shape_mass(component)
        box_shape = _component_box_shape_mass(component, bbox)
        full_signed = component.amplitude * full_shape
        box_signed = component.amplitude * box_shape
        full_abs = abs(component.amplitude) * full_shape
        box_abs = abs(component.amplitude) * box_shape
        omitted_signed = full_signed - box_signed
        omitted_abs = max(full_abs - box_abs, 0.0)

        total_signed += full_signed
        total_abs += full_abs
        in_box_signed += box_signed
        in_box_abs += box_abs
        component_reports.append(
            {
                **component.as_metadata(),
                "full_signed_mass": full_signed,
                "in_box_signed_mass": box_signed,
                "omitted_signed_mass": omitted_signed,
                "full_abs_mass": full_abs,
                "in_box_abs_mass": box_abs,
                "omitted_abs_mass": omitted_abs,
                "omitted_abs_fraction": omitted_abs / full_abs if full_abs else 0.0,
            }
        )

    omitted_signed = total_signed - in_box_signed
    omitted_abs = max(total_abs - in_box_abs, 0.0)
    return {
        "mixture": mixture.as_metadata(),
        "bbox": bbox.tolist(),
        "total_signed_mass": total_signed,
        "in_box_signed_mass": in_box_signed,
        "omitted_signed_mass": omitted_signed,
        "total_abs_component_mass": total_abs,
        "in_box_abs_component_mass": in_box_abs,
        "omitted_abs_component_mass": omitted_abs,
        "omitted_abs_fraction": omitted_abs / total_abs if total_abs else 0.0,
        "components": component_reports,
    }


@dataclass(frozen=True)
class SliceGrid:
    points: np.ndarray
    shape: tuple[int, int]
    axes: tuple[int, int]
    fixed_axis: int
    fixed_value: float
    axis_values: tuple[np.ndarray, np.ndarray]


def axis_aligned_slice_grid(
    bbox: np.ndarray,
    *,
    fixed_axis: int = 2,
    fixed_value: float = 0.0,
    axes: tuple[int, int] | None = None,
    shape: tuple[int, int] = (64, 64),
) -> SliceGrid:
    """Create a 2D Cartesian slice grid embedded in a 3D bounding box."""

    bbox = _as_bbox(bbox, 3)
    if fixed_axis < 0 or fixed_axis >= 3:
        raise ValueError("fixed_axis must be 0, 1, or 2")
    if axes is None:
        computed_axes = tuple(axis for axis in range(3) if axis != fixed_axis)
        axes = (computed_axes[0], computed_axes[1])
    if len(axes) != 2:
        raise ValueError("axes must contain two distinct non-fixed axes")
    axes = (int(axes[0]), int(axes[1]))
    if (
        len(set(axes)) != 2
        or fixed_axis in axes
        or any(axis < 0 or axis >= 3 for axis in axes)
    ):
        raise ValueError("axes must contain two distinct non-fixed axes")
    if not (bbox[fixed_axis, 0] <= fixed_value <= bbox[fixed_axis, 1]):
        raise ValueError("fixed_value must lie within bbox[fixed_axis]")
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise ValueError("shape must contain two entries >= 2")

    axis0 = np.linspace(bbox[axes[0], 0], bbox[axes[0], 1], int(shape[0]))
    axis1 = np.linspace(bbox[axes[1], 0], bbox[axes[1], 1], int(shape[1]))
    grid0, grid1 = np.meshgrid(axis0, axis1, indexing="ij")
    points = np.zeros((grid0.size, 3), dtype=np.float64)
    points[:, fixed_axis] = fixed_value
    points[:, axes[0]] = grid0.ravel()
    points[:, axes[1]] = grid1.ravel()
    return SliceGrid(
        points=points,
        shape=(int(shape[0]), int(shape[1])),
        axes=axes,
        fixed_axis=int(fixed_axis),
        fixed_value=float(fixed_value),
        axis_values=(axis0, axis1),
    )


def nearest_axis_slice(
    points: np.ndarray,
    fields: dict[str, np.ndarray] | None = None,
    *,
    axis: int = 2,
    value: float = 0.0,
    atol: float | None = None,
) -> dict[str, Any]:
    """Extract nodes on the plane nearest to ``points[:, axis] == value``."""

    points = _as_points(points)
    if axis < 0 or axis >= points.shape[1]:
        raise ValueError("axis is outside the point dimension")

    distances = np.abs(points[:, axis] - float(value))
    if atol is None:
        min_distance = float(np.min(distances))
        tolerance = max(1.0e-14, 1.0e-12 * max(abs(min_distance), 1.0))
        mask = np.abs(distances - min_distance) <= tolerance
    else:
        if atol < 0:
            raise ValueError("atol must be non-negative")
        mask = distances <= atol

    indices = np.nonzero(mask)[0]
    result: dict[str, Any] = {
        "coords": points[indices],
        "indices": indices.astype(np.int64),
        "axis_distances": distances[indices],
        "axis": int(axis),
        "value": float(value),
        "max_selected_distance": float(np.max(distances[indices])) if indices.size else math.nan,
    }

    if fields is not None:
        for name, values in fields.items():
            values = np.asarray(values)
            if values.shape[0] != points.shape[0]:
                raise ValueError(f"field {name!r} does not match the point count")
            result[name] = values[indices]

    return result


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def mesh_leaf_box_arrays(mesh: Any) -> dict[str, np.ndarray]:
    """Return host arrays describing the active leaf boxes of a Volumential mesh."""

    boxtree = mesh.boxtree
    leaf_box_ids = _host_array(boxtree.active_boxes).astype(np.int64)
    all_levels = _host_array(boxtree.box_levels).astype(np.int32)
    all_centers = _host_array(boxtree.box_centers).astype(np.float64)
    levels = all_levels[leaf_box_ids]
    centers = all_centers[:, leaf_box_ids].T
    side_lengths = boxtree.root_extent / np.power(2.0, levels)
    return {
        "leaf_box_ids": leaf_box_ids,
        "leaf_centers": centers,
        "leaf_levels": levels,
        "leaf_side_lengths": side_lengths.astype(np.float64),
        "leaf_measures": (side_lengths**mesh.dim).astype(np.float64),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata contains a non-finite float")
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    return value


def write_json_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write JSON metadata with NumPy values converted to plain Python values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(metadata), allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write NumPy arrays, creating the parent directory when needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)

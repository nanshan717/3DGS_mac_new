#!/usr/bin/env python3
"""
Geometry evaluation for 3DGS outputs.

This script is intentionally standalone so you can copy it into an original
gaussian-splatting checkout on the server without modifying the repo.

It computes three proxy geometry metrics from the reconstructed point cloud.
By default, the metrics are measured against a post-fitted Bernstein surface:
  - Floater Ratio: fraction of bottom-slice points that deviate too far from a
    smooth fitted surface.
  - GSD (Geometric Surface Deviation): RMSE of nearest distances to the fitted
    Bernstein surface.
  - Toughness: inverse curvature energy of the fitted surface (higher is better).

These are proxy metrics, not oracle ground-truth geometry metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from plyfile import PlyData

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def find_latest_iteration(point_cloud_root: Path) -> int:
    if not point_cloud_root.exists():
        raise FileNotFoundError(f"Missing point cloud directory: {point_cloud_root}")

    iterations = []
    for child in point_cloud_root.iterdir():
        if child.is_dir():
            m = re.match(r"iteration_(\d+)$", child.name)
            if m:
                iterations.append(int(m.group(1)))
    if not iterations:
        raise FileNotFoundError(f"No iteration_* folder found under {point_cloud_root}")
    return max(iterations)


def resolve_ply_path(model_path: Path, iteration: int) -> Tuple[Path, int]:
    point_cloud_root = model_path / "point_cloud"
    if iteration < 0:
        iteration = find_latest_iteration(point_cloud_root)

    ply_path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing point cloud ply: {ply_path}")
    return ply_path, iteration


def load_xyz_from_ply(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return xyz


def deterministic_sample_indices(num_points: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    return np.linspace(0, num_points - 1, max_points, dtype=np.int64)


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return centroid, normal


def orthogonal_distances(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.abs((points - centroid[None, :]) @ normal)


def bernstein_basis(num_control_points: int, values: np.ndarray) -> np.ndarray:
    if num_control_points < 2:
        raise ValueError("Bernstein fitting needs at least two control points per axis.")

    degree = num_control_points - 1
    values = np.clip(values.astype(np.float64), 0.0, 1.0)
    basis = []
    for i in range(num_control_points):
        coeff = math_comb(degree, i)
        basis.append(coeff * (values ** i) * ((1.0 - values) ** (degree - i)))
    return np.stack(basis, axis=1).astype(np.float32)


def math_comb(n: int, k: int) -> int:
    # Python 3.7 compatibility for older 3DGS environments.
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def horizontal_axes(axis_idx: int) -> Tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(axis_idx)
    return axes[0], axes[1]


def normalize_unit(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    v_min = float(values.min())
    v_max = float(values.max())
    span = max(v_max - v_min, 1e-8)
    return (values - v_min) / span, v_min, v_max


def fit_bernstein_height_surface(
    points: np.ndarray,
    axis_idx: int,
    control_points_u: int,
    control_points_v: int,
    ridge: float,
    max_fit_points: int,
) -> Dict[str, np.ndarray]:
    u_axis, v_axis = horizontal_axes(axis_idx)
    fit_points = points[deterministic_sample_indices(points.shape[0], max_fit_points)]

    u_norm, u_min, u_max = normalize_unit(fit_points[:, u_axis])
    v_norm, v_min, v_max = normalize_unit(fit_points[:, v_axis])
    heights = fit_points[:, axis_idx].astype(np.float32)

    basis_u = bernstein_basis(control_points_u, u_norm)
    basis_v = bernstein_basis(control_points_v, v_norm)
    design = np.einsum("ni,nj->nij", basis_u, basis_v).reshape(fit_points.shape[0], -1)

    lhs = design.T @ design
    rhs = design.T @ heights
    if ridge > 0:
        lhs = lhs + float(ridge) * np.eye(lhs.shape[0], dtype=np.float32)
    try:
        coeff = np.linalg.solve(lhs, rhs).reshape(control_points_u, control_points_v)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(design, heights, rcond=None)[0].reshape(
            control_points_u,
            control_points_v,
        )

    return {
        "coeff": coeff.astype(np.float32),
        "u_axis": np.array(u_axis, dtype=np.int64),
        "v_axis": np.array(v_axis, dtype=np.int64),
        "axis_idx": np.array(axis_idx, dtype=np.int64),
        "u_min": np.array(u_min, dtype=np.float32),
        "u_max": np.array(u_max, dtype=np.float32),
        "v_min": np.array(v_min, dtype=np.float32),
        "v_max": np.array(v_max, dtype=np.float32),
    }


def evaluate_bernstein_surface(surface: Dict[str, np.ndarray], samples_u: int, samples_v: int) -> np.ndarray:
    coeff = surface["coeff"]
    axis_idx = int(surface["axis_idx"])
    u_axis = int(surface["u_axis"])
    v_axis = int(surface["v_axis"])

    u = np.linspace(0.0, 1.0, samples_u, dtype=np.float32)
    v = np.linspace(0.0, 1.0, samples_v, dtype=np.float32)
    basis_u = bernstein_basis(coeff.shape[0], u)
    basis_v = bernstein_basis(coeff.shape[1], v)
    heights = np.einsum("ui,vj,ij->uv", basis_u, basis_v, coeff)

    u_world = float(surface["u_min"]) + u * (float(surface["u_max"]) - float(surface["u_min"]))
    v_world = float(surface["v_min"]) + v * (float(surface["v_max"]) - float(surface["v_min"]))
    uu, vv = np.meshgrid(u_world, v_world, indexing="ij")

    surface_points = np.zeros((samples_u, samples_v, 3), dtype=np.float32)
    surface_points[..., u_axis] = uu
    surface_points[..., v_axis] = vv
    surface_points[..., axis_idx] = heights
    return surface_points.reshape(-1, 3)


def nearest_distances_to_surface(
    points: np.ndarray,
    surface_points: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    if _HAS_SCIPY:
        tree = cKDTree(surface_points)
        dists, _ = tree.query(points, k=1)
        return dists.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    s = torch.from_numpy(surface_points).to(device=device, dtype=torch.float32)
    all_dists = []
    for start in range(0, p.shape[0], max(1, chunk_size)):
        chunk = p[start : start + max(1, chunk_size)]
        dist = torch.cdist(chunk, s)
        all_dists.append(dist.min(dim=1).values.detach().cpu())
    return torch.cat(all_dists, dim=0).numpy()


def bernstein_curvature_energy(coeff: np.ndarray) -> float:
    energies = []
    if coeff.shape[0] >= 3:
        second_u = coeff[2:, :] - 2.0 * coeff[1:-1, :] + coeff[:-2, :]
        energies.append(np.mean(second_u ** 2))
    if coeff.shape[1] >= 3:
        second_v = coeff[:, 2:] - 2.0 * coeff[:, 1:-1] + coeff[:, :-2]
        energies.append(np.mean(second_v ** 2))
    if not energies:
        return 0.0

    height_span = max(float(coeff.max() - coeff.min()), 1e-8)
    return float(np.sqrt(np.mean(energies)) / height_span)


def knn_query(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns sorted distances and indices of k nearest neighbors.
    Uses scipy cKDTree when available, otherwise a torch fallback.
    """
    k = max(1, min(k, reference.shape[0]))

    if _HAS_SCIPY:
        tree = cKDTree(reference)
        dists, idx = tree.query(query, k=k)
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        return dists.astype(np.float32), idx.astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(query).to(device=device, dtype=torch.float32)
    r = torch.from_numpy(reference).to(device=device, dtype=torch.float32)
    chunk_size = 128
    all_dists = []
    all_idx = []
    for start in range(0, q.shape[0], chunk_size):
        chunk = q[start : start + chunk_size]
        dist = torch.cdist(chunk, r)
        vals, inds = torch.topk(dist, k=k, largest=False, dim=1)
        all_dists.append(vals.detach().cpu())
        all_idx.append(inds.detach().cpu())
    return torch.cat(all_dists, dim=0).numpy(), torch.cat(all_idx, dim=0).numpy()


def estimate_spacing(points: np.ndarray, k: int = 2, max_query_points: int = 4096) -> float:
    if points.shape[0] < 2:
        return 0.0

    query = points[deterministic_sample_indices(points.shape[0], max_query_points)]
    dists, _ = knn_query(query, points, k=min(k, points.shape[0]))
    if dists.shape[1] >= 2:
        spacing = dists[:, 1]
    else:
        spacing = dists[:, 0]
    return float(np.median(spacing))


def compute_surface_variation(
    points: np.ndarray,
    k: int = 16,
    max_query_points: int = 2048,
) -> float:
    if points.shape[0] < 3:
        return 0.0

    sample = points[deterministic_sample_indices(points.shape[0], max_query_points)]
    knn_k = min(k + 1, points.shape[0])
    dists, idx = knn_query(sample, points, k=knn_k)

    if idx.shape[1] <= 1:
        return 0.0

    variations = []
    for row in idx:
        nbrs = points[row[1:]]  # drop self-neighbor
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(nbrs.shape[0] - 1, 1)
        evals = np.linalg.eigvalsh(cov)
        denom = float(evals.sum()) + 1e-12
        variations.append(float(evals[0] / denom))

    return float(np.mean(variations))


def evaluate_single_model(
    model_path: Path,
    iteration: int,
    axis: str,
    z_percentile: float,
    floater_tau_factor: float,
    control_points_u: int,
    control_points_v: int,
    surface_samples_u: int,
    surface_samples_v: int,
    fit_ridge: float,
    distance_chunk_size: int,
    roughness_k: int,
    max_query_points: int,
    max_fit_points: int,
) -> Dict[str, float]:
    ply_path, resolved_iteration = resolve_ply_path(model_path, iteration)
    xyz = load_xyz_from_ply(ply_path)

    axis_idx = AXIS_TO_INDEX[axis.lower()]
    support_values = xyz[:, axis_idx]
    support_cutoff = float(np.quantile(support_values, z_percentile))
    bottom_mask = support_values <= support_cutoff
    bottom_points = xyz[bottom_mask]

    if bottom_points.shape[0] < 3:
        raise RuntimeError(
            f"Too few bottom-slice points ({bottom_points.shape[0]}) in {ply_path}"
        )

    centroid, normal = fit_plane_svd(bottom_points)
    plane_dist = orthogonal_distances(bottom_points, centroid, normal)

    surface = fit_bernstein_height_surface(
        bottom_points,
        axis_idx=axis_idx,
        control_points_u=control_points_u,
        control_points_v=control_points_v,
        ridge=fit_ridge,
        max_fit_points=max_fit_points,
    )
    surface_points = evaluate_bernstein_surface(surface, surface_samples_u, surface_samples_v)
    dist = nearest_distances_to_surface(bottom_points, surface_points, distance_chunk_size)

    spacing = estimate_spacing(bottom_points)
    tau = floater_tau_factor * spacing if spacing > 0 else 0.0
    floater_ratio = float(np.mean(dist > tau)) if tau > 0 else 0.0
    gsd = float(np.sqrt(np.mean(dist ** 2)))

    local_variation = compute_surface_variation(
        bottom_points,
        k=roughness_k,
        max_query_points=max_query_points,
    )
    roughness = bernstein_curvature_energy(surface["coeff"])
    toughness = float(1.0 / (1.0 + roughness))
    plane_gsd = float(np.sqrt(np.mean(plane_dist ** 2)))
    plane_floater_ratio = float(np.mean(plane_dist > tau)) if tau > 0 else 0.0

    result = {
        "model_path": str(model_path),
        "iteration": int(resolved_iteration),
        "ply_path": str(ply_path),
        "metric_surface": "postfit_bernstein",
        "axis": axis,
        "z_percentile": float(z_percentile),
        "floater_tau_factor": float(floater_tau_factor),
        "control_points_u": int(control_points_u),
        "control_points_v": int(control_points_v),
        "surface_samples_u": int(surface_samples_u),
        "surface_samples_v": int(surface_samples_v),
        "fit_ridge": float(fit_ridge),
        "num_points": int(xyz.shape[0]),
        "num_bottom_points": int(bottom_points.shape[0]),
        "plane_centroid_x": float(centroid[0]),
        "plane_centroid_y": float(centroid[1]),
        "plane_centroid_z": float(centroid[2]),
        "plane_normal_x": float(normal[0]),
        "plane_normal_y": float(normal[1]),
        "plane_normal_z": float(normal[2]),
        "spacing_median": float(spacing),
        "floater_ratio": floater_ratio,
        "gsd": gsd,
        "roughness": float(roughness),
        "toughness": toughness,
        "local_surface_variation": float(local_variation),
        "plane_floater_ratio": plane_floater_ratio,
        "plane_gsd": plane_gsd,
    }
    return result


def print_result(result: Dict[str, float]) -> None:
    print(f"Model   : {result['model_path']}")
    print(f"Iter    : {result['iteration']}")
    print(f"PLY     : {result['ply_path']}")
    print(f"Surface : {result['metric_surface']}")
    print(f"Points  : {result['num_points']}  |  Bottom slice: {result['num_bottom_points']}")
    print(f"Floater Ratio : {result['floater_ratio']:.6f}")
    print(f"GSD           : {result['gsd']:.6f}")
    print(f"Roughness     : {result['roughness']:.6f}   (Bernstein curvature energy)")
    print(f"Toughness     : {result['toughness']:.6f}   (1 / (1 + roughness))")
    print(f"Plane GSD     : {result['plane_gsd']:.6f}   (diagnostic only)")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate geometry metrics from 3DGS point clouds.")
    parser.add_argument(
        "-m",
        "--model_paths",
        required=True,
        nargs="+",
        help="One or more 3DGS output folders, e.g. output/bonsai_eval",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to evaluate. Use -1 for latest point_cloud/iteration_*.",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis used for the bottom-slice filter.",
    )
    parser.add_argument(
        "--z_percentile",
        type=float,
        default=0.2,
        help="Bottom-slice percentile on the selected axis.",
    )
    parser.add_argument(
        "--floater_tau_factor",
        type=float,
        default=2.5,
        help="Threshold multiplier over the median local point spacing.",
    )
    parser.add_argument(
        "--control_points_u",
        type=int,
        default=5,
        help="Number of Bernstein control points on the first horizontal axis.",
    )
    parser.add_argument(
        "--control_points_v",
        type=int,
        default=5,
        help="Number of Bernstein control points on the second horizontal axis.",
    )
    parser.add_argument(
        "--surface_samples_u",
        type=int,
        default=64,
        help="Number of sampled Bernstein surface points on the first axis.",
    )
    parser.add_argument(
        "--surface_samples_v",
        type=int,
        default=64,
        help="Number of sampled Bernstein surface points on the second axis.",
    )
    parser.add_argument(
        "--fit_ridge",
        type=float,
        default=1e-6,
        help="Small ridge term for stable Bernstein least-squares fitting.",
    )
    parser.add_argument(
        "--max_fit_points",
        type=int,
        default=20000,
        help="Max bottom-slice points used to fit the Bernstein surface.",
    )
    parser.add_argument(
        "--distance_chunk_size",
        type=int,
        default=4096,
        help="Chunk size for point-to-surface nearest-distance evaluation.",
    )
    parser.add_argument(
        "--roughness_k",
        type=int,
        default=16,
        help="Number of neighbors used for diagnostic local PCA variation.",
    )
    parser.add_argument(
        "--max_query_points",
        type=int,
        default=2048,
        help="Max points sampled for roughness estimation.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save geometry_results.json under each model path.",
    )
    args = parser.parse_args()

    for model_path_str in args.model_paths:
        model_path = Path(model_path_str).expanduser().resolve()
        result = evaluate_single_model(
            model_path=model_path,
            iteration=args.iteration,
            axis=args.axis,
            z_percentile=args.z_percentile,
            floater_tau_factor=args.floater_tau_factor,
            control_points_u=args.control_points_u,
            control_points_v=args.control_points_v,
            surface_samples_u=args.surface_samples_u,
            surface_samples_v=args.surface_samples_v,
            fit_ridge=args.fit_ridge,
            distance_chunk_size=args.distance_chunk_size,
            roughness_k=args.roughness_k,
            max_query_points=args.max_query_points,
            max_fit_points=args.max_fit_points,
        )
        print_result(result)

        if args.save_json:
            out_path = model_path / "geometry_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Oracle geometry evaluation for metric CoffeeFabric-Syn triangle meshes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


SCHEMA = "coffee-fabric-gt-geometry-v1"


def latest_iteration(model_path: Path) -> int:
    root = model_path / "point_cloud"
    found = []
    if root.exists():
        for child in root.iterdir():
            match = re.fullmatch(r"iteration_(\d+)", child.name)
            if child.is_dir() and match:
                found.append(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No point_cloud/iteration_* directory under {model_path}")
    return max(found)


def model_ply(model_path: Path, iteration: int) -> Tuple[Path, int]:
    resolved = latest_iteration(model_path) if iteration < 0 else iteration
    path = model_path / "point_cloud" / f"iteration_{resolved}" / "point_cloud.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, resolved


def load_gaussians(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    names = set(vertex.data.dtype.names or ())
    if "opacity" in names:
        raw = np.asarray(vertex["opacity"], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    else:
        opacity = np.ones(xyz.shape[0], dtype=np.float32)
    return xyz, opacity


def triangulate_faces(raw_faces) -> np.ndarray:
    triangles = []
    for raw_face in raw_faces:
        face = list(raw_face)
        if len(face) < 3:
            continue
        # Blender PLY exports may preserve quads/ngons. A deterministic fan is
        # sufficient because the generated fabric faces are planar locally.
        triangles.extend((face[0], face[index], face[index + 1])
                         for index in range(1, len(face) - 1))
    if not triangles:
        raise ValueError("Ground-truth mesh has no valid polygon faces")
    return np.asarray(triangles, dtype=np.int64)


def load_triangle_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    vertices = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    if "face" not in ply:
        raise ValueError(f"Ground truth must contain polygon faces: {path}")
    return vertices, triangulate_faces(ply["face"].data["vertex_indices"])


def sample_mesh_uniform(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministically sample triangles proportional to their surface area."""
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    triangles, areas = triangles[valid], areas[valid]
    if not len(triangles):
        raise ValueError("Ground-truth mesh has no non-degenerate triangles")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    tri = triangles[picked]
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    return (tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0])
            + uv[:, 1:] * (tri[:, 2] - tri[:, 0])).astype(np.float32)


def select_fabric_candidates(
    xyz: np.ndarray, opacity: np.ndarray, gt_samples: np.ndarray,
    opacity_min: float, canopy_margin_m: float, support_margin_m: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Exclude canopy using GT height while retaining below-surface floaters."""
    low = gt_samples[:, :2].min(axis=0) - support_margin_m
    high = gt_samples[:, :2].max(axis=0) + support_margin_m
    preliminary = (np.all((xyz[:, :2] >= low) & (xyz[:, :2] <= high), axis=1)
                   & (opacity >= opacity_min))
    candidates = xyz[preliminary]
    if not len(candidates):
        raise RuntimeError("No Gaussians remain after opacity/support filtering")
    _, nearest = cKDTree(gt_samples[:, :2]).query(candidates[:, :2], k=1)
    selected = candidates[candidates[:, 2] <= gt_samples[nearest, 2] + canopy_margin_m]
    if not len(selected):
        raise RuntimeError("No fabric candidates remain after canopy exclusion")
    return selected, {
        "total_gaussians": int(len(xyz)),
        "opacity_support_candidates": int(len(candidates)),
        "fabric_candidates": int(len(selected)),
    }


def distance_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean_m": float(values.mean()), "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "median_m": float(np.median(values)), "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
    }


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Average centres per metric voxel so density does not dominate geometry scores."""
    if voxel_size_m <= 0:
        return points
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


def evaluate(prediction: np.ndarray, gt_samples: np.ndarray, thresholds_m: Iterable[float]) -> Dict[str, object]:
    pred_to_gt = cKDTree(gt_samples).query(prediction, k=1)[0].astype(np.float32)
    gt_to_pred = cKDTree(prediction).query(gt_samples, k=1)[0].astype(np.float32)
    result: Dict[str, object] = {
        "accuracy_pred_to_gt": distance_summary(pred_to_gt),
        "completeness_gt_to_pred": distance_summary(gt_to_pred),
        "chamfer_l1_m": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        "chamfer_l2_m2": float(0.5 * (np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2))),
        "threshold_metrics": {},
    }
    for threshold in thresholds_m:
        precision = float(np.mean(pred_to_gt <= threshold))
        recall = float(np.mean(gt_to_pred <= threshold))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result["threshold_metrics"][f"{threshold:.3f}m"] = {
            "precision": precision, "recall": recall, "fscore": fscore,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3DGS centres against CoffeeFabric-Syn GT mesh")
    parser.add_argument("-m", "--model_paths", nargs="+", required=True)
    parser.add_argument("-s", "--source_path", required=True, help="Synthetic scene root")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--opacity_min", type=float, default=0.05)
    parser.add_argument("--canopy_margin_m", type=float, default=0.15)
    parser.add_argument("--support_margin_m", type=float, default=0.02)
    parser.add_argument("--voxel_size_m", type=float, default=0.005)
    parser.add_argument("--gt_samples", type=int, default=250000)
    parser.add_argument("--sample_seed", type=int, default=3401)
    parser.add_argument("--thresholds_cm", nargs="+", type=float, default=[1.0, 2.0, 5.0])
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_path).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_type") != "synthetic" or metadata.get("units") != "metres":
        raise ValueError("GT evaluation requires a synthetic scene declared in metres")
    gt_path = source / "ground_truth" / "fabric_mesh.ply"
    vertices, faces = load_triangle_mesh(gt_path)
    gt_samples = sample_mesh_uniform(vertices, faces, args.gt_samples, args.sample_seed)
    thresholds_m = tuple(value / 100.0 for value in args.thresholds_cm)

    for raw_model in args.model_paths:
        model = Path(raw_model).expanduser().resolve()
        ply_path, iteration = model_ply(model, args.iteration)
        xyz, opacity = load_gaussians(ply_path)
        prediction, counts = select_fabric_candidates(
            xyz, opacity, gt_samples, args.opacity_min, args.canopy_margin_m, args.support_margin_m)
        prediction = voxel_downsample(prediction, args.voxel_size_m)
        counts["voxelized_fabric_candidates"] = int(len(prediction))
        metrics = evaluate(prediction, gt_samples, thresholds_m)
        payload = {
            "schema": SCHEMA, "model_path": str(model), "iteration": iteration,
            "ply_path": str(ply_path), "source_path": str(source),
            "scene_id": metadata.get("scene_id"), "ground_truth_mesh": str(gt_path), "units": "metres",
            "protocol": {
                "opacity_min": args.opacity_min, "canopy_margin_m": args.canopy_margin_m,
                "support_margin_m": args.support_margin_m, "gt_samples": args.gt_samples,
                "voxel_size_m": args.voxel_size_m,
                "sample_seed": args.sample_seed, "thresholds_m": thresholds_m,
                "sampling": "triangle-area-weighted deterministic",
                "candidate_rule": "opacity/support and z <= local_gt_z + canopy_margin",
            },
            "counts": counts, "metrics": metrics,
        }
        print(f"\nModel: {model}\nIteration: {iteration}\nFabric candidates: {counts['fabric_candidates']}/{counts['total_gaussians']}"
              f" (voxelized: {counts['voxelized_fabric_candidates']})")
        print(f"Accuracy mean/P90: {metrics['accuracy_pred_to_gt']['mean_m']:.6f} / {metrics['accuracy_pred_to_gt']['p90_m']:.6f} m")
        print(f"Completeness mean/P90: {metrics['completeness_gt_to_pred']['mean_m']:.6f} / {metrics['completeness_gt_to_pred']['p90_m']:.6f} m")
        print(f"Chamfer L1: {metrics['chamfer_l1_m']:.6f} m")
        print("F-score @1/2/5 cm: " + " / ".join(
            f"{entry['fscore']:.6f}" for entry in metrics["threshold_metrics"].values()))
        if args.save_json:
            output = model / f"gt_geometry_results_iter-{iteration}.json"
            output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()

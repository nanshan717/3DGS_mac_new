#!/usr/bin/env python3
"""Read-only, common-protocol comparison for 3DGS/BR-GS model folders."""

import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np

from eval_geometry import (
    evaluate_piecewise_bottom,
    evaluate_single_model,
    load_xyz_from_ply,
    prepare_support_points,
    resolve_ply_path,
)


def parse_models(values):
    models = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected label=/absolute/model/path, got {value!r}")
        label, path = value.split("=", 1)
        models.append((label, Path(path).expanduser().resolve()))
    return models


def load_render_metrics(model_path):
    path = model_path / "results.json"
    if not path.exists():
        return {}, "missing results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return {}, "empty results.json"
    method = max(payload.keys(), key=lambda name: int(re.search(r"(\d+)$", name).group(1)) if re.search(r"(\d+)$", name) else -1)
    return payload[method], None


def recover_config(model_path):
    manifest_path = model_path / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = manifest.get("dataset", {})
        optimization = manifest.get("optimization", {})
        return {
            "source_path": dataset.get("source_path", "unknown"),
            "resolution": dataset.get("resolution", "unknown"),
            "eval": dataset.get("eval", "unknown"),
            "iterations": optimization.get("iterations", "unknown"),
            "seed": dataset.get("seed", "unknown"),
        }
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return {key: "unknown" for key in ("source_path", "resolution", "eval", "iterations", "seed")}
    text = cfg_path.read_text(encoding="utf-8")
    def field(name, default="unknown"):
        match = re.search(rf"(?:^|[, (]){name}=('[^']*'|\"[^\"]*\"|True|False|-?[0-9.]+)", text)
        return match.group(1).strip("'\"") if match else default
    return {key: field(key) for key in ("source_path", "resolution", "eval", "iterations", "seed")}


def count_test_views(model_path):
    test_root = model_path / "test"
    methods = sorted(test_root.glob("ours_*/renders")) if test_root.exists() else []
    return len(list(methods[-1].glob("*.png"))) if methods else 0


def geometry_for_model(model_path, args, support_frame=None, partition_bounds=None, normalization_scale=None):
    global_result = evaluate_single_model(
        model_path, args.iteration, args.axis, args.z_percentile, args.floater_tau_factor,
        args.control_points_u, args.control_points_v, args.surface_samples_u,
        args.surface_samples_v, args.fit_ridge, args.distance_chunk_size,
        args.roughness_k, args.max_query_points, args.max_fit_points,
        support_frame=support_frame,
    )
    xyz = load_xyz_from_ply(Path(global_result["ply_path"]))
    _, bottom, axis_idx, _ = prepare_support_points(xyz, args.axis, args.z_percentile, support_frame)
    piecewise = evaluate_piecewise_bottom(
        bottom, axis_idx, args.eval_patches_u, args.eval_patches_v,
        args.floater_tau_factor, args.control_points_u, args.control_points_v,
        args.surface_samples_u, args.surface_samples_v, args.fit_ridge,
        args.distance_chunk_size, args.max_fit_points, partition_bounds, normalization_scale,
    )
    return global_result, piecewise


def flatten_row(label, model_path, render, global_result, piecewise, config):
    row = {"label": label, "model_path": str(model_path), **config}
    row["iteration"] = global_result["iteration"]
    row["test_views"] = count_test_views(model_path)
    for key in ("PSNR", "SSIM", "LPIPS"):
        row[key.lower()] = render.get(key, "")
    row["points"] = global_result["num_points"]
    for prefix, result in (("global", global_result), ("piecewise", piecewise)):
        for key in ("gsd", "normalized_gsd", "distance_median", "distance_p90", "distance_p95",
                    "floater_ratio_1pct", "floater_ratio_2pct", "floater_ratio_5pct", "roughness", "toughness"):
            row[f"{prefix}_{key}"] = result.get(key, "")
    row["piecewise_seam_position_error"] = piecewise["seam_position_error"]
    row["piecewise_seam_tangent_error"] = piecewise["seam_tangent_error"]
    return row


def validation_warnings(rows):
    warnings = []
    for key in ("source_path", "resolution", "eval", "iteration", "test_views"):
        known = {str(row[key]) for row in rows if str(row[key]) != "unknown"}
        if len(known) > 1:
            warnings.append(f"Configuration mismatch for {key}: {sorted(known)}")
    return warnings


def add_deltas(rows):
    if not rows:
        return
    baseline = rows[0]
    for row in rows:
        for key in ("psnr", "ssim", "lpips", "piecewise_normalized_gsd", "piecewise_distance_p90",
                    "piecewise_floater_ratio_5pct", "piecewise_roughness"):
            if isinstance(row.get(key), (int, float)) and isinstance(baseline.get(key), (int, float)):
                row[f"delta_{key}"] = row[key] - baseline[key]
                row[f"pct_{key}"] = 100.0 * (row[key] - baseline[key]) / max(abs(baseline[key]), 1e-12)


def write_markdown(path, rows, warnings):
    columns = ["label", "psnr", "ssim", "lpips", "points", "piecewise_normalized_gsd",
               "piecewise_distance_p90", "piecewise_floater_ratio_5pct", "piecewise_roughness"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if warnings:
        lines.extend(["", "## Validation warnings", ""] + [f"- {warning}" for warning in warnings])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="label=/absolute/model/path")
    parser.add_argument("--output_dir", default="comparisons")
    parser.add_argument("--name", default="comparison")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--axis", choices=["x", "y", "z", "auto"], default="auto")
    parser.add_argument("--z_percentile", type=float, default=0.2)
    parser.add_argument("--eval_patches_u", type=int, default=2)
    parser.add_argument("--eval_patches_v", type=int, default=2)
    parser.add_argument("--floater_tau_factor", type=float, default=2.5)
    parser.add_argument("--control_points_u", type=int, default=5)
    parser.add_argument("--control_points_v", type=int, default=5)
    parser.add_argument("--surface_samples_u", type=int, default=64)
    parser.add_argument("--surface_samples_v", type=int, default=64)
    parser.add_argument("--fit_ridge", type=float, default=1e-6)
    parser.add_argument("--distance_chunk_size", type=int, default=4096)
    parser.add_argument("--roughness_k", type=int, default=16)
    parser.add_argument("--max_query_points", type=int, default=2048)
    parser.add_argument("--max_fit_points", type=int, default=20000)
    args = parser.parse_args()

    models = parse_models(args.models)
    support_frame = None
    reference_ply, _ = resolve_ply_path(models[0][1], args.iteration)
    reference_xyz = load_xyz_from_ply(reference_ply)
    _, reference_bottom, reference_axis, estimated_frame = prepare_support_points(reference_xyz, args.axis, args.z_percentile)
    if args.axis == "auto":
        support_frame = estimated_frame
    axes = [0, 1, 2]; axes.remove(reference_axis)
    u_min, u_max = np.quantile(reference_bottom[:, axes[0]], [0.01, 0.99])
    v_min, v_max = np.quantile(reference_bottom[:, axes[1]], [0.01, 0.99])
    partition_bounds = (float(u_min), float(u_max), float(v_min), float(v_max))
    reference_span = np.quantile(reference_bottom, 0.99, axis=0) - np.quantile(reference_bottom, 0.01, axis=0)
    normalization_scale = max(float(np.linalg.norm(reference_span)), 1e-8)
    rows, details = [], {}
    for label, model_path in models:
        render, render_warning = load_render_metrics(model_path)
        global_result, piecewise = geometry_for_model(model_path, args, support_frame, partition_bounds, normalization_scale)
        config = recover_config(model_path)
        rows.append(flatten_row(label, model_path, render, global_result, piecewise, config))
        details[label] = {"render": render, "global_1x1": global_result, "piecewise": piecewise,
                          "config": config, "warning": render_warning}
    add_deltas(rows)
    warnings = validation_warnings(rows)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, md_path = output_dir / f"{args.name}.csv", output_dir / f"{args.name}.json", output_dir / f"{args.name}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    protocol = {
        "reference_model": models[0][0], "axis": args.axis,
        "z_percentile": args.z_percentile,
        "patches": [args.eval_patches_u, args.eval_patches_v],
        "partition_bounds": list(partition_bounds),
        "normalization_scale": normalization_scale,
        "common_support_frame": args.axis == "auto",
    }
    json_path.write_text(json.dumps({"protocol": protocol, "rows": rows, "details": details, "warnings": warnings}, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, warnings)
    for warning in warnings:
        print(f"[WARNING] {warning}")
    print(f"Saved: {csv_path}\nSaved: {json_path}\nSaved: {md_path}")


if __name__ == "__main__":
    main()

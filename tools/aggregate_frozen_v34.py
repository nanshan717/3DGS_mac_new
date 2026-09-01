#!/usr/bin/env python3
"""Validate and aggregate the frozen BR-GS v3.4 multi-seed experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


METRICS = (
    "psnr", "ssim", "lpips", "points",
    "accuracy_mean_m", "accuracy_p90_m",
    "completeness_mean_m", "completeness_p90_m",
    "chamfer_l1_m", "fscore_1cm", "fscore_2cm", "fscore_5cm",
    "pruned_fraction",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_render_metrics(path: Path, expected_iteration: int) -> dict:
    payload = read_json(path / "results.json")
    preferred = f"ours_{expected_iteration}"
    if preferred in payload:
        return payload[preferred]
    candidates = []
    for name, metrics in payload.items():
        match = re.search(r"(\d+)$", name)
        if match:
            candidates.append((int(match.group(1)), metrics))
    if not candidates:
        raise ValueError(f"No ours_ITER entry in {path / 'results.json'}")
    iteration, metrics = max(candidates)
    if iteration != expected_iteration:
        raise ValueError(f"Expected render iteration {expected_iteration}, found {iteration} in {path}")
    return metrics


def cfg_field(text: str, name: str):
    match = re.search(rf"(?:^|[, (]){re.escape(name)}=('[^']*'|\"[^\"]*\"|True|False|-?[0-9.]+)", text)
    if not match:
        return None
    value = match.group(1).strip("'\"")
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def run_config(path: Path) -> dict:
    frozen_manifest = path / "frozen_run_manifest.json"
    if frozen_manifest.is_file():
        payload = read_json(frozen_manifest)
        if payload.get("schema") != "brgs-frozen-v34-run-v1" or not payload.get("train_completed"):
            raise ValueError(f"Invalid frozen run manifest in {path}")
        return {
            "source_path": payload.get("source_path"),
            "seed": payload.get("seed"),
            "iterations": payload.get("iterations"),
            "argv": payload.get("train_command", []),
            "seed_provenance": "frozen_run_manifest",
            "scene": payload.get("scene"),
            "method": payload.get("method"),
            "model_path": payload.get("model_path"),
            "resolution": payload.get("resolution"),
        }
    manifest = path / "experiment_manifest.json"
    if manifest.is_file():
        payload = read_json(manifest)
        dataset = payload.get("dataset", {})
        optimization = payload.get("optimization", {})
        return {
            "source_path": dataset.get("source_path"),
            "seed": dataset.get("seed"),
            "iterations": optimization.get("iterations"),
            "argv": payload.get("argv", []),
            "seed_provenance": "experiment_manifest",
            "scene": None,
            "method": None,
            "model_path": None,
            "resolution": None,
        }
    cfg = path / "cfg_args"
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing experiment_manifest.json/cfg_args in {path}")
    text = cfg.read_text(encoding="utf-8")
    return {
        "source_path": cfg_field(text, "source_path"),
        "seed": cfg_field(text, "seed"),
        "iterations": cfg_field(text, "iterations"),
        "argv": [],
        "seed_provenance": "cfg_args",
        "scene": None,
        "method": None,
        "model_path": None,
        "resolution": None,
    }


def load_run(matrix: dict, scene_id: str, method_id: str, seed: int) -> dict:
    method = matrix["methods"][method_id]
    repo = Path(matrix[method["repo"]])
    relative = method["output_pattern"].format(scene=scene_id, seed=seed)
    model = (repo / relative).resolve()
    iteration = int(matrix["iterations"])
    source = str(Path(matrix["scenes"][scene_id]["source"]).resolve())
    config = run_config(model)
    if config["scene"] is not None and config["scene"] != scene_id:
        raise ValueError(f"Scene metadata mismatch for {model}: {config['scene']} != {scene_id}")
    if config["method"] is not None and config["method"] != method_id:
        raise ValueError(f"Method metadata mismatch for {model}: {config['method']} != {method_id}")
    if config["model_path"] is not None and Path(config["model_path"]).resolve() != model:
        raise ValueError(f"Model-path metadata mismatch for {model}")
    if config["resolution"] is not None and int(config["resolution"]) != int(matrix["resolution"]):
        raise ValueError(f"Resolution metadata mismatch for {model}")
    if str(Path(config["source_path"]).resolve()) != source:
        raise ValueError(f"Source mismatch for {model}: {config['source_path']} != {source}")
    # Upstream official 3DGS cfg_args does not serialize train.py's top-level
    # RNG flags.  The compatibility checkout defaults to seed 0, so only the
    # already-completed legacy seed-0 run can be recovered without a runner
    # manifest.  Nonzero seeds always require explicit metadata.
    if config["seed"] is None and method_id == "official_3dgs" and seed == 0:
        patch_audit = repo / "REPRODUCIBILITY_PATCH.json"
        if not patch_audit.is_file():
            raise ValueError(f"Cannot verify legacy official seed 0 without {patch_audit}")
        config["seed"] = 0
        config["seed_provenance"] = "official_compat_default_seed0"
    if config["seed"] is None or int(config["seed"]) != seed:
        raise ValueError(f"Seed metadata mismatch for {model}: {config['seed']} != {seed}")
    if config["iterations"] is not None and int(config["iterations"]) != iteration:
        raise ValueError(f"Iteration metadata mismatch for {model}")
    if method_id == "brgs_v34" and config["argv"] and "--bsr_v34" not in config["argv"]:
        raise ValueError(f"BR-GS run does not record --bsr_v34: {model}")

    render = latest_render_metrics(model, iteration)
    gt = read_json(model / f"gt_geometry_results_iter-{iteration}.json")
    if int(gt.get("iteration", -1)) != iteration or str(Path(gt["source_path"]).resolve()) != source:
        raise ValueError(f"GT geometry protocol mismatch for {model}")
    gm = gt["metrics"]
    thresholds = gm["threshold_metrics"]
    row = {
        "scene": scene_id,
        "role": matrix["scenes"][scene_id]["role"],
        "method": method_id,
        "seed": seed,
        "seed_provenance": config["seed_provenance"],
        "model_path": str(model),
        "psnr": float(render["PSNR"]),
        "ssim": float(render["SSIM"]),
        "lpips": float(render["LPIPS"]),
        "points": int(gt["counts"]["total_gaussians"]),
        "accuracy_mean_m": float(gm["accuracy_pred_to_gt"]["mean_m"]),
        "accuracy_p90_m": float(gm["accuracy_pred_to_gt"]["p90_m"]),
        "completeness_mean_m": float(gm["completeness_gt_to_pred"]["mean_m"]),
        "completeness_p90_m": float(gm["completeness_gt_to_pred"]["p90_m"]),
        "chamfer_l1_m": float(gm["chamfer_l1_m"]),
        "fscore_1cm": float(thresholds["0.010m"]["fscore"]),
        "fscore_2cm": float(thresholds["0.020m"]["fscore"]),
        "fscore_5cm": float(thresholds["0.050m"]["fscore"]),
        "pruned_fraction": 0.0,
    }
    audit_path = model / "bsr_v34_pruning.json"
    if method_id == "brgs_v34":
        audit = read_json(audit_path)
        row["pruned_fraction"] = float(audit["removed_fraction"])
        if int(audit["points_after"]) != row["points"]:
            raise ValueError(f"Final point count does not match v3.4 audit for {model}")
        if row["pruned_fraction"] > 0.0500001:
            raise ValueError(f"Frozen 5% pruning cap exceeded for {model}")
    elif audit_path.exists():
        raise ValueError(f"Official baseline unexpectedly contains a BR-GS pruning audit: {model}")
    return row


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["role"], row["method"]), []).append(row)
    summaries = []
    for (scene, role, method), group in sorted(groups.items()):
        item = {"scene": scene, "role": role, "method": method, "runs": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(item)
    return summaries


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pm(mean: float, std: float, digits: int = 6) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(path: Path, summaries) -> None:
    columns = ("Scene", "Method", "PSNR↑", "SSIM↑", "LPIPS↓", "Points↓",
               "Chamfer↓", "Completeness↓", "F@5cm↑")
    lines = ["# Frozen BR-GS v3.4 multi-seed summary", "",
             "Values are arithmetic mean ± sample standard deviation over declared seeds.", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    by_scene = {}
    for item in summaries:
        by_scene.setdefault(item["scene"], {})[item["method"]] = item
        lines.append("| " + " | ".join([
            item["scene"], item["method"],
            pm(item["psnr_mean"], item["psnr_std"]),
            pm(item["ssim_mean"], item["ssim_std"]),
            pm(item["lpips_mean"], item["lpips_std"]),
            f"{item['points_mean']:.0f} ± {item['points_std']:.0f}",
            pm(item["chamfer_l1_m_mean"], item["chamfer_l1_m_std"]),
            pm(item["completeness_mean_m_mean"], item["completeness_mean_m_std"]),
            pm(item["fscore_5cm_mean"], item["fscore_5cm_std"]),
        ]) + " |")
    lines.extend(["", "## BR-GS v3.4 change relative to official 3DGS", "",
                  "| Scene | ΔPSNR (dB) | ΔLPIPS | Point reduction | ΔChamfer (m) |",
                  "|---|---:|---:|---:|---:|"])
    for scene, methods in sorted(by_scene.items()):
        if not {"official_3dgs", "brgs_v34"}.issubset(methods):
            continue
        base, ours = methods["official_3dgs"], methods["brgs_v34"]
        reduction = 100.0 * (base["points_mean"] - ours["points_mean"]) / base["points_mean"]
        lines.append(
            f"| {scene} | {ours['psnr_mean'] - base['psnr_mean']:+.6f} | "
            f"{ours['lpips_mean'] - base['lpips_mean']:+.6f} | {reduction:.2f}% | "
            f"{ours['chamfer_l1_m_mean'] - base['chamfer_l1_m_mean']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=root / "experiments" / "frozen_v34_matrix.json")
    parser.add_argument("--output_dir", type=Path, default=root / "comparisons" / "frozen_v34_final")
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    matrix = read_json(args.matrix.expanduser().resolve())
    if matrix.get("schema") != "brgs-frozen-v34-matrix-v1" or matrix.get("status") != "frozen":
        raise ValueError("Expected a frozen brgs-frozen-v34-matrix-v1 file")
    if matrix.get("seeds") != [0, 1, 2] or matrix.get("iterations") != 15000:
        raise ValueError("Frozen matrix must declare seeds 0/1/2 at 15000 iterations")
    rows, missing = [], []
    for scene in matrix["scenes"]:
        for method in matrix["methods"]:
            for seed in matrix["seeds"]:
                try:
                    rows.append(load_run(matrix, scene, method, int(seed)))
                except FileNotFoundError as error:
                    missing.append(str(error))
    if missing and not args.allow_incomplete:
        raise SystemExit("Incomplete frozen matrix:\n- " + "\n- ".join(missing))
    summaries = summarize(rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_run.csv", rows)
    write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(json.dumps({
        "schema": "brgs-frozen-v34-summary-v1",
        "matrix": matrix,
        "per_run": rows,
        "summary": summaries,
        "missing": missing,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output / "summary.md", summaries)
    print(f"Validated {len(rows)} runs; missing {len(missing)}")
    print(f"Saved: {output / 'summary.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and aggregate the BR-GS v3.4 no-pruning ablation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("/home/featurize/work/3DGS_mac_new")
RUN_ROOT = ROOT / "baseline_results/brgs_v34_noprune_15k"
FULL_CSV = ROOT / "paper_results/v34/combined_per_run.csv"
OUT_ROOT = ROOT / "paper_results/v34/ablation_noprune"
SCENES = ("P01", "P02", "P03", "P04")
SEEDS = (0, 1, 2)
ROLES = {"P01": "development", "P02": "development", "P03": "held_out", "P04": "held_out"}
METRICS = (
    "psnr", "ssim", "lpips", "points", "accuracy_mean_m",
    "completeness_mean_m", "chamfer_l1_m", "fscore_1cm",
    "fscore_2cm", "fscore_5cm",
)


def load_no_prune() -> list[dict]:
    rows = []
    for scene in SCENES:
        for seed in SEEDS:
            run = RUN_ROOT / f"{scene}_s{seed}"
            required = (
                "COMPLETE", "results.json", "gt_geometry_results_iter-15000.json",
                "experiment_manifest.json", "bsr_v34_pruning.json", "protocol.txt",
            )
            missing = [name for name in required if not (run / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{run}: missing {missing}")

            image = json.loads((run / "results.json").read_text(encoding="utf-8"))
            if set(image) != {"ours_15000"}:
                raise ValueError(f"Unexpected image methods in {run}: {sorted(image)}")
            image = image["ours_15000"]
            geom = json.loads(
                (run / "gt_geometry_results_iter-15000.json").read_text(encoding="utf-8")
            )
            audit = json.loads((run / "bsr_v34_pruning.json").read_text(encoding="utf-8"))
            manifest = json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8"))

            if audit["max_fraction"] != 0.0 or audit["points_removed"] != 0:
                raise ValueError(f"Not a valid no-pruning audit: {run}: {audit}")
            if audit["points_before"] != audit["points_after"]:
                raise ValueError(f"Point count changed during disabled pruning: {run}")
            if manifest["optimization"]["bsr_prune_max_fraction"] != 0.0:
                raise ValueError(f"Resolved manifest is not no-pruning: {run}")
            if int(manifest["dataset"]["seed"]) != seed:
                raise ValueError(f"Seed mismatch: {run}")
            if int(geom["iteration"]) != 15000:
                raise ValueError(f"Iteration mismatch: {run}")

            threshold = geom["metrics"]["threshold_metrics"]
            rows.append({
                "role": ROLES[scene], "scene": scene,
                "method": "brgs_v34_noprune", "seed": seed,
                "model_path": geom["model_path"],
                "psnr": float(image["PSNR"]), "ssim": float(image["SSIM"]),
                "lpips": float(image["LPIPS"]),
                "points": int(geom["counts"]["total_gaussians"]),
                "accuracy_mean_m": float(geom["metrics"]["accuracy_pred_to_gt"]["mean_m"]),
                "completeness_mean_m": float(geom["metrics"]["completeness_gt_to_pred"]["mean_m"]),
                "chamfer_l1_m": float(geom["metrics"]["chamfer_l1_m"]),
                "fscore_1cm": float(threshold["0.010m"]["fscore"]),
                "fscore_2cm": float(threshold["0.020m"]["fscore"]),
                "fscore_5cm": float(threshold["0.050m"]["fscore"]),
            })
    return rows


def load_full() -> list[dict]:
    with FULL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "brgs_v34"]
    rows = [row for row in rows if row["scene"] in SCENES and int(row["seed"]) in SEEDS]
    if len(rows) != 12:
        raise ValueError(f"Expected 12 full BR-GS rows, found {len(rows)}")
    for row in rows:
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def main() -> None:
    no_prune = load_no_prune()
    full = load_full()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_fields = ("role", "scene", "method", "seed", "model_path", *METRICS)
    with (OUT_ROOT / "noprune_per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(no_prune)

    summary = []
    for scene in SCENES:
        for method, source in (("brgs_v34", full), ("brgs_v34_noprune", no_prune)):
            group = [row for row in source if row["scene"] == scene]
            item = {"role": ROLES[scene], "scene": scene, "method": method, "runs": len(group)}
            for metric in METRICS:
                mean, std = mean_std([float(row[metric]) for row in group])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            summary.append(item)

    summary_fields = list(summary[0])
    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    full_lookup = {(row["scene"], row["seed"]): row for row in full}
    no_lookup = {(row["scene"], row["seed"]): row for row in no_prune}
    effects = []
    for scene in SCENES:
        paired = []
        for seed in SEEDS:
            yes = full_lookup[(scene, seed)]
            no = no_lookup[(scene, seed)]
            paired.append({
                "point_reduction_pct": 100.0 * (no["points"] - yes["points"]) / no["points"],
                "delta_psnr": yes["psnr"] - no["psnr"],
                "delta_ssim": yes["ssim"] - no["ssim"],
                "delta_lpips": yes["lpips"] - no["lpips"],
                "delta_accuracy_mean_m": yes["accuracy_mean_m"] - no["accuracy_mean_m"],
                "delta_completeness_mean_m": yes["completeness_mean_m"] - no["completeness_mean_m"],
                "delta_chamfer_l1_m": yes["chamfer_l1_m"] - no["chamfer_l1_m"],
                "delta_fscore_5cm": yes["fscore_5cm"] - no["fscore_5cm"],
            })
        item = {"role": ROLES[scene], "scene": scene}
        for metric in paired[0]:
            mean, std = mean_std([row[metric] for row in paired])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        effects.append(item)

    effect_fields = list(effects[0])
    with (OUT_ROOT / "pruning_effect.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=effect_fields)
        writer.writeheader()
        writer.writerows(effects)

    summary_lookup = {(row["scene"], row["method"]): row for row in summary}
    print("Validated: 12/12 no-pruning runs; all pruning audits removed 0 points.")
    print("Effect = full BR-GS v3.4 minus no-pruning ablation")
    print("Scene | Full points | No-prune points | Reduction | dPSNR | dChamfer | dCompleteness | dF@5cm")
    for effect in effects:
        scene = effect["scene"]
        yes = summary_lookup[(scene, "brgs_v34")]
        no = summary_lookup[(scene, "brgs_v34_noprune")]
        print(
            f"{scene} | {yes['points_mean']:.0f} | {no['points_mean']:.0f} | "
            f"{effect['point_reduction_pct_mean']:.2f}% | {effect['delta_psnr_mean']:+.3f} | "
            f"{effect['delta_chamfer_l1_m_mean']:+.6f} | "
            f"{effect['delta_completeness_mean_m_mean']:+.6f} | "
            f"{effect['delta_fscore_5cm_mean']:+.4f}"
        )
    print(f"Saved: {OUT_ROOT}")


if __name__ == "__main__":
    main()

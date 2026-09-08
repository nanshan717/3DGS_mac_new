#!/usr/bin/env python3
"""Build paper-ready BR-GS v3.4 tables, data, and figures from frozen CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import statistics
from pathlib import Path


EXPECTED = {
    "development": ("P01", "P02"),
    "held_out": ("P03", "P04"),
}
METHODS = ("official_3dgs", "brgs_v34")
SEEDS = (0, 1, 2)
METRICS = (
    "psnr", "ssim", "lpips", "points", "accuracy_mean_m",
    "completeness_mean_m", "chamfer_l1_m", "fscore_5cm",
)
LOWER_IS_BETTER = {
    "lpips", "points", "accuracy_mean_m", "completeness_mean_m", "chamfer_l1_m",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def verify_sha256(root: Path, manifest: Path) -> None:
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        digest, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("*")
        target = root / relative
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA256 mismatch for {target}: {actual} != {digest}")


def validate(rows: list[dict]) -> None:
    expected = {
        (scene, role, method, seed)
        for role, scenes in EXPECTED.items()
        for scene in scenes
        for method in METHODS
        for seed in SEEDS
    }
    observed = {(r["scene"], r["role"], r["method"], r["seed"]) for r in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Unexpected frozen matrix rows; missing={missing}, extra={extra}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for role, scenes in EXPECTED.items():
        for scene in scenes:
            for method in METHODS:
                group = [r for r in rows if r["scene"] == scene and r["method"] == method]
                item = {"role": role, "scene": scene, "method": method, "runs": len(group)}
                for metric in METRICS:
                    mean, std = mean_std([r[metric] for r in group])
                    item[f"{metric}_mean"] = mean
                    item[f"{metric}_std"] = std
                output.append(item)
    return output


def paired_deltas(rows: list[dict]) -> list[dict]:
    lookup = {(r["scene"], r["method"], r["seed"]): r for r in rows}
    output = []
    for role, scenes in EXPECTED.items():
        for scene in scenes:
            for seed in SEEDS:
                base = lookup[(scene, "official_3dgs", seed)]
                ours = lookup[(scene, "brgs_v34", seed)]
                output.append({
                    "role": role,
                    "scene": scene,
                    "seed": seed,
                    "delta_psnr_db": ours["psnr"] - base["psnr"],
                    "delta_ssim": ours["ssim"] - base["ssim"],
                    "delta_lpips": ours["lpips"] - base["lpips"],
                    "point_reduction_pct": 100.0 * (base["points"] - ours["points"]) / base["points"],
                    "delta_accuracy_mean_m": ours["accuracy_mean_m"] - base["accuracy_mean_m"],
                    "delta_completeness_mean_m": (
                        ours["completeness_mean_m"] - base["completeness_mean_m"]
                    ),
                    "delta_chamfer_l1_m": ours["chamfer_l1_m"] - base["chamfer_l1_m"],
                    "delta_fscore_5cm": ours["fscore_5cm"] - base["fscore_5cm"],
                })
    return output


def scene_deltas(summary: list[dict], paired: list[dict]) -> list[dict]:
    lookup = {(r["scene"], r["method"]): r for r in summary}
    output = []
    for role, scenes in EXPECTED.items():
        for scene in scenes:
            base = lookup[(scene, "official_3dgs")]
            ours = lookup[(scene, "brgs_v34")]
            group = [r for r in paired if r["scene"] == scene]
            item = {
                "role": role,
                "scene": scene,
                "delta_psnr_db": ours["psnr_mean"] - base["psnr_mean"],
                "delta_ssim": ours["ssim_mean"] - base["ssim_mean"],
                "delta_lpips": ours["lpips_mean"] - base["lpips_mean"],
                "point_reduction_pct": 100.0 * (
                    base["points_mean"] - ours["points_mean"]
                ) / base["points_mean"],
                "delta_accuracy_mean_m": (
                    ours["accuracy_mean_m_mean"] - base["accuracy_mean_m_mean"]
                ),
                "delta_completeness_mean_m": (
                    ours["completeness_mean_m_mean"] - base["completeness_mean_m_mean"]
                ),
                "delta_chamfer_l1_m": (
                    ours["chamfer_l1_m_mean"] - base["chamfer_l1_m_mean"]
                ),
                "delta_fscore_5cm": ours["fscore_5cm_mean"] - base["fscore_5cm_mean"],
            }
            for metric in (
                "delta_psnr_db", "delta_ssim", "delta_lpips", "point_reduction_pct",
                "delta_accuracy_mean_m", "delta_completeness_mean_m",
                "delta_chamfer_l1_m", "delta_fscore_5cm",
            ):
                item[f"{metric}_paired_std"] = statistics.stdev([r[metric] for r in group])
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path) -> None:
    path.write_text(
        """# BR-GS v3.4 paper-results package

This directory is generated from the checksum-frozen development and pre-registered
held-out matrices. It contains the complete 24-run data, scene summaries, paired-seed
deltas, a paper table, and two vector figures.

Source inputs:

- `comparisons/frozen_v34_final/per_run.csv` (P01--P02 development)
- `comparisons/heldout_v34_final/per_run.csv` (P03--P04 held-out)

Regenerate from the repository root with:

```bash
python3 tools/make_v34_paper_results.py
```

The generator verifies both input `SHA256SUMS` manifests before reading the data and
validates the full 4 scenes x 2 methods x 3 seeds design. `SHA256SUMS` in this directory
covers every generated artifact except itself.

Interpretation boundary: all comparisons are descriptive. Bold table entries mark only
the better per-scene mean and do not indicate statistical significance.
""",
        encoding="utf-8",
    )


def write_sha256(path: Path, artifacts: list[Path]) -> None:
    lines = [f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}" for item in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_pm(item: dict, metric: str) -> str:
    mean, std = item[f"{metric}_mean"], item[f"{metric}_std"]
    if metric == "points":
        return f"{mean / 1000:.0f} ± {std / 1000:.0f}"
    if metric == "psnr":
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.4f} ± {std:.4f}"


def best(summary: list[dict], scene: str, metric: str, method: str) -> bool:
    group = [r for r in summary if r["scene"] == scene]
    values = {r["method"]: r[f"{metric}_mean"] for r in group}
    target = min(values.values()) if metric in LOWER_IS_BETTER else max(values.values())
    return math.isclose(values[method], target)


def write_markdown_table(path: Path, summary: list[dict]) -> None:
    columns = (
        "Split", "Scene", "Method", "PSNR↑", "SSIM↑", "LPIPS↓", "Points (k)↓",
        "Chamfer (m)↓", "Completeness (m)↓", "F@5cm↑",
    )
    metric_columns = (
        "psnr", "ssim", "lpips", "points", "chamfer_l1_m",
        "completeness_mean_m", "fscore_5cm",
    )
    lines = [
        "# BR-GS v3.4 main quantitative table", "",
        "Mean ± sample standard deviation over three paired training seeds. Development and",
        "pre-registered held-out scenes are reported separately.", "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    labels = {"development": "Development", "held_out": "Held-out"}
    names = {"official_3dgs": "Official 3DGS", "brgs_v34": "BR-GS v3.4"}
    for role, scenes in EXPECTED.items():
        for scene in scenes:
            for method in METHODS:
                item = next(r for r in summary if r["scene"] == scene and r["method"] == method)
                values = []
                for metric in metric_columns:
                    value = format_pm(item, metric)
                    values.append(f"**{value}**" if best(summary, scene, metric, method) else value)
                lines.append("| " + " | ".join([
                    labels[role], scene, names[method], *values,
                ]) + " |")
    lines.extend([
        "", "Bold marks the better mean within each scene and metric; it does not denote",
        "statistical significance. Lower is better for LPIPS, points, Chamfer, and completeness.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_value(value: str, bold: bool) -> str:
    escaped = value.replace("±", r"$\pm$")
    return rf"\textbf{{{escaped}}}" if bold else escaped


def write_latex_table(path: Path, summary: list[dict]) -> None:
    metrics = (
        "psnr", "ssim", "lpips", "points", "chamfer_l1_m",
        "completeness_mean_m", "fscore_5cm",
    )
    names = {"official_3dgs": "Official 3DGS", "brgs_v34": "BR-GS v3.4"}
    lines = [
        r"% Requires: \usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{BR-GS v3.4 compared with official 3DGS. Values are mean $\pm$ sample standard deviation over three paired training seeds. P01--P02 are development scenes; P03--P04 are pre-registered held-out scenes. Bold marks the better mean within each scene and does not denote statistical significance.}",
        r"\label{tab:v34-main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Scene & Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & Points (k)$\downarrow$ & Chamfer (m)$\downarrow$ & Completeness (m)$\downarrow$ & F@5cm$\uparrow$ \\",
        r"\midrule",
    ]
    labels = {"development": "Development", "held_out": "Pre-registered held-out"}
    first_role = True
    for role, scenes in EXPECTED.items():
        if not first_role:
            lines.append(r"\midrule")
        first_role = False
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{labels[role]}}}}} \\")
        for scene in scenes:
            for method in METHODS:
                item = next(r for r in summary if r["scene"] == scene and r["method"] == method)
                vals = [
                    latex_value(format_pm(item, metric), best(summary, scene, metric, method))
                    for metric in metrics
                ]
                lines.append(" & ".join([scene, names[method], *vals]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


INK = "#172033"
MUTED = "#617085"
GRID = "#d9e0e8"
DEV = "#2878b5"
HELD = "#d47a22"
BG = "#ffffff"


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
        ]

    def rect(self, x, y, width, height, fill, stroke="none", stroke_width=1, radius=0):
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke, width=1):
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
        )

    def circle(self, cx, cy, radius, fill, stroke="none", stroke_width=1):
        self.items.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polygon(self, points, fill):
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(f'<polygon points="{value}" fill="{fill}"/>')

    def text(self, x, y, value, size=20, fill=INK, weight="normal", anchor="start"):
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'
        )

    def save(self, path: Path):
        path.write_text("\n".join(self.items + ["</svg>"]) + "\n", encoding="utf-8")


def nice_range(values: list[float], include_zero: bool = True) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    pad = span * 0.18 if span else max(abs(hi) * 0.2, 0.1)
    return lo - pad, hi + pad


def draw_delta_panel(svg, box, title, ylabel, scenes, means, seed_values, formatter):
    x0, y0, x1, y1 = box
    svg.rect(x0, y0, x1 - x0, y1 - y0, "#fbfcfe", "#d7dee8", 2, 18)
    svg.text(x0 + 20, y0 + 35, title, 21, INK, "bold")
    left, right, top, bottom = x0 + 88, x1 - 24, y0 + 62, y1 - 55
    all_values = list(means) + [v for values in seed_values for v in values]
    lo, hi = nice_range(all_values)

    def py(value):
        return bottom - (value - lo) * (bottom - top) / (hi - lo)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = lo + fraction * (hi - lo)
        y = py(value)
        svg.line(left, y, right, y, GRID, 1.5)
        svg.text(left - 10, y + 5, formatter(value), 13, MUTED, anchor="end")
    if lo <= 0 <= hi:
        svg.line(left, py(0), right, py(0), "#6b7480", 2)
    step = (right - left) / len(scenes)
    for index, scene in enumerate(scenes):
        x = left + step * (index + 0.5)
        color = DEV if index < 2 else HELD
        for offset, value in zip((-13, 0, 13), seed_values[index]):
            y = py(value)
            svg.circle(x + offset, y, 5, color, BG, 1.5)
        mean_y = py(means[index])
        svg.line(x - 28, mean_y, x + 28, mean_y, INK, 4)
        svg.text(x, bottom + 31, scene, 17, INK, "bold", "middle")
        svg.text(x, mean_y - 13, formatter(means[index]), 13, INK, anchor="middle")
    svg.text(x0 + 20, top - 5, ylabel, 13, MUTED)


def write_tradeoff_figure(path: Path, deltas: list[dict], paired: list[dict]) -> None:
    svg = Svg(1600, 1100)
    svg.text(42, 57, "BR-GS v3.4: change relative to official 3DGS", 32, INK, "bold")
    svg.text(43, 88, "Horizontal marks show scene means; dots show paired training seeds (n=3). P01-P02: development; P03-P04: held-out.", 17, MUTED)
    scenes = ["P01", "P02", "P03", "P04"]
    by_scene = {r["scene"]: r for r in deltas}
    paired_by_scene = {scene: [r for r in paired if r["scene"] == scene] for scene in scenes}
    panels = [
        ("Point-count reduction", "percent; higher is more compact", "point_reduction_pct", lambda x: f"{x:.1f}%"),
        ("PSNR change", "dB; higher is better", "delta_psnr_db", lambda x: f"{x:+.2f}"),
        ("LPIPS change", "lower is better", "delta_lpips", lambda x: f"{x:+.3f}"),
        ("Chamfer L1 change", "meters; lower is better", "delta_chamfer_l1_m", lambda x: f"{x:+.4f}"),
    ]
    boxes = [(35, 120, 785, 590), (815, 120, 1565, 590), (35, 615, 785, 1075), (815, 615, 1565, 1075)]
    for box, (title, ylabel, metric, formatter) in zip(boxes, panels):
        means = [by_scene[scene][metric] for scene in scenes]
        seed_values = [[r[metric] for r in paired_by_scene[scene]] for scene in scenes]
        draw_delta_panel(svg, box, title, ylabel, scenes, means, seed_values, formatter)
    svg.rect(1217, 39, 20, 20, DEV)
    svg.text(1245, 56, "Development", 16, INK)
    svg.rect(1392, 39, 20, 20, HELD)
    svg.text(1420, 56, "Held-out", 16, INK)
    svg.save(path)


def write_consistency_figure(path: Path, summary: list[dict]) -> None:
    svg = Svg(1600, 850)
    svg.text(42, 58, "Mean BR-GS v3.4 compactness-quality trade-off", 32, INK, "bold")
    svg.text(43, 90, "Arrows point from official 3DGS to BR-GS; each marker is a three-seed scene mean.", 17, MUTED)
    lookup = {(r["scene"], r["method"]): r for r in summary}
    scenes = ["P01", "P02", "P03", "P04"]
    colors = {"P01": DEV, "P02": "#55a4d4", "P03": HELD, "P04": "#e6a65e"}
    left, right, top, bottom = 125, 1535, 165, 720
    points_values = [lookup[(s, m)]["points_mean"] / 1e6 for s in scenes for m in METHODS]
    psnr_values = [lookup[(s, m)]["psnr_mean"] for s in scenes for m in METHODS]
    xmin, xmax = min(points_values) - 0.08, max(points_values) + 0.08
    ymin, ymax = min(psnr_values) - 0.6, max(psnr_values) + 0.6

    def px(value):
        return left + (value - xmin) * (right - left) / (xmax - xmin)

    def py(value):
        return bottom - (value - ymin) * (bottom - top) / (ymax - ymin)

    for i in range(6):
        value = xmin + i * (xmax - xmin) / 5
        x = px(value)
        svg.line(x, top, x, bottom, GRID, 1.5)
        svg.text(x, bottom + 28, f"{value:.2f}", 15, MUTED, anchor="middle")
    for i in range(6):
        value = ymin + i * (ymax - ymin) / 5
        y = py(value)
        svg.line(left, y, right, y, GRID, 1.5)
        svg.text(left - 14, y + 5, f"{value:.1f}", 15, MUTED, anchor="end")
    svg.text((left + right) / 2, 790, "Mean Gaussian count (millions; lower is better)", 18, INK, anchor="middle")
    svg.text(42, 146, "Mean PSNR (dB)", 16, INK)
    for scene in scenes:
        base, ours = lookup[(scene, "official_3dgs")], lookup[(scene, "brgs_v34")]
        bx, by = px(base["points_mean"] / 1e6), py(base["psnr_mean"])
        ox, oy = px(ours["points_mean"] / 1e6), py(ours["psnr_mean"])
        color = colors[scene]
        svg.line(bx, by, ox, oy, color, 5)
        angle = math.atan2(oy - by, ox - bx)
        size = 14
        arrow = [
            (ox, oy),
            (ox - size * math.cos(angle - 0.5), oy - size * math.sin(angle - 0.5)),
            (ox - size * math.cos(angle + 0.5), oy - size * math.sin(angle + 0.5)),
        ]
        svg.polygon(arrow, color)
        svg.circle(bx, by, 8, BG, color, 3.5)
        svg.circle(ox, oy, 9, color, BG, 2)
        svg.text(ox + 13, oy - 12, scene, 17, color, "bold")
    legend_x = 1120
    svg.circle(legend_x, 124, 8, BG, INK, 3)
    svg.text(legend_x + 20, 130, "Official 3DGS", 16, INK)
    svg.circle(legend_x + 190, 124, 8, INK)
    svg.text(legend_x + 210, 130, "BR-GS v3.4", 16, INK)
    svg.save(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=root / "comparisons/frozen_v34_final/per_run.csv")
    parser.add_argument("--heldout", type=Path, default=root / "comparisons/heldout_v34_final/per_run.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "paper_results/v34")
    args = parser.parse_args()

    verify_sha256(root, args.development.parent / "SHA256SUMS")
    verify_sha256(root, args.heldout.parent / "SHA256SUMS")
    rows = read_rows(args.development) + read_rows(args.heldout)
    validate(rows)
    summary = summarize(rows)
    paired = paired_deltas(rows)
    deltas = scene_deltas(summary, paired)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "combined_per_run.csv", rows)
    write_csv(output / "scene_summary.csv", summary)
    write_csv(output / "paired_seed_deltas.csv", paired)
    write_csv(output / "scene_deltas.csv", deltas)
    write_markdown_table(output / "table_v34_main.md", summary)
    write_latex_table(output / "table_v34_main.tex", summary)
    write_tradeoff_figure(output / "figure_v34_metric_deltas.svg", deltas, paired)
    write_consistency_figure(output / "figure_v34_compactness_psnr.svg", summary)
    write_readme(output / "README.md")
    artifact_names = (
        "README.md", "combined_per_run.csv", "scene_summary.csv",
        "paired_seed_deltas.csv", "scene_deltas.csv", "table_v34_main.md",
        "table_v34_main.tex", "figure_v34_metric_deltas.svg",
        "figure_v34_compactness_psnr.svg",
    )
    write_sha256(output / "SHA256SUMS", [output / name for name in artifact_names])
    print(f"Validated 24 frozen runs (12 development, 12 held-out)")
    print(f"Saved paper results to: {output}")


if __name__ == "__main__":
    main()

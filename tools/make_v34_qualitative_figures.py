#!/usr/bin/env python3
"""Build reproducible BR-GS v3.4 qualitative figures from exported renders."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import struct
from pathlib import Path


SCENES = ("P03", "P04")
SEEDS = (0, 1, 2)
VIEWS = tuple(f"{index:05d}.png" for index in range(12))
MAIN_VIEW = "00005.png"
OVERVIEW_VIEWS = ("00000.png", "00005.png", "00011.png")
MAIN_SEED = 0
CROP = (180, 180, 440, 440)

INK = "#172033"
MUTED = "#617085"
LINE = "#d7dee8"
DEV_BLUE = "#2878b5"
HELD_ORANGE = "#d47a22"
BG = "#ffffff"


def model_name(scene: str, method: str, seed: int) -> str:
    middle = "official_3dgs" if method == "official" else "brgs_v34"
    return f"{scene}_{middle}_15k_s{seed}"


def render_path(root: Path, scene: str, method: str, seed: int, view: str) -> Path:
    return root / "output" / model_name(scene, method, seed) / "test" / "ours_15000" / "renders" / view


def gt_path(root: Path, scene: str, view: str) -> Path:
    return root / "output" / model_name(scene, "official", 0) / "test" / "ours_15000" / "gt" / view


def metrics_path(root: Path, scene: str, method: str, seed: int) -> Path:
    return root / "output" / model_name(scene, method, seed) / "per_view.json"


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def validate_inputs(root: Path) -> None:
    expected = set(VIEWS)
    for scene in SCENES:
        gt_dir = gt_path(root, scene, MAIN_VIEW).parent
        observed_gt = {item.name for item in gt_dir.glob("*.png")}
        if observed_gt != expected:
            raise ValueError(f"Incomplete GT set for {scene}: {sorted(expected - observed_gt)}")
        for method in ("official", "brgs"):
            for seed in SEEDS:
                renders = render_path(root, scene, method, seed, MAIN_VIEW).parent
                observed = {item.name for item in renders.glob("*.png")}
                if observed != expected:
                    raise ValueError(
                        f"Incomplete render set for {scene}/{method}/s{seed}: "
                        f"{sorted(expected - observed)}"
                    )
                metrics = json.loads(metrics_path(root, scene, method, seed).read_text(encoding="utf-8"))
                values = metrics.get("ours_15000", {})
                for metric in ("PSNR", "SSIM", "LPIPS"):
                    if set(values.get(metric, {})) != expected:
                        raise ValueError(f"Incomplete {metric} entries for {scene}/{method}/s{seed}")
                for image in renders.glob("*.png"):
                    if png_size(image) != (800, 800):
                        raise ValueError(f"Unexpected image size for {image}: {png_size(image)}")
        for image in gt_dir.glob("*.png"):
            if png_size(image) != (800, 800):
                raise ValueError(f"Unexpected image size for {image}: {png_size(image)}")


def read_metrics(root: Path, scene: str, method: str, seed: int, view: str) -> dict[str, float]:
    data = json.loads(metrics_path(root, scene, method, seed).read_text(encoding="utf-8"))["ours_15000"]
    return {metric: float(data[metric][view]) for metric in ("PSNR", "SSIM", "LPIPS")}


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
        self._encoded: dict[Path, str] = {}

    def text(self, x, y, value, size=22, fill=INK, weight="normal", anchor="start"):
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, color=LINE, width=2):
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )

    def rect(self, x, y, width, height, fill="none", stroke=LINE, stroke_width=2):
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def _href(self, path: Path) -> str:
        if path not in self._encoded:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            self._encoded[path] = f"data:image/png;base64,{encoded}"
        return self._encoded[path]

    def image(self, path: Path, x, y, width, height, crop=None):
        href = self._href(path)
        if crop is None:
            self.items.append(
                f'<image x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
                f'preserveAspectRatio="xMidYMid meet" href="{href}"/>'
            )
        else:
            cx, cy, cw, ch = crop
            self.items.append(
                f'<svg x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
                f'viewBox="{cx} {cy} {cw} {ch}" preserveAspectRatio="xMidYMid slice">'
                f'<image x="0" y="0" width="800" height="800" href="{href}"/></svg>'
            )
        self.rect(x, y, width, height, stroke="#c7d0dc", stroke_width=2)

    def save(self, path: Path):
        path.write_text("\n".join(self.items + ["</svg>"]) + "\n", encoding="utf-8")


def draw_full_with_crop_box(svg: Svg, path: Path, x: float, y: float, size: float):
    svg.image(path, x, y, size, size)
    cx, cy, cw, ch = CROP
    scale = size / 800
    svg.rect(
        x + cx * scale, y + cy * scale, cw * scale, ch * scale,
        stroke=HELD_ORANGE, stroke_width=4,
    )


def write_main(path: Path, root: Path) -> None:
    svg = Svg(1890, 850)
    svg.text(42, 52, "Held-out qualitative comparison", 34, INK, "bold")
    svg.text(
        43, 84,
        "Fixed test view 00005 and primary seed s0; the center crop is identical across methods.",
        18, MUTED,
    )
    tile = 250
    scene_x = {"P03": 90, "P04": 1010}
    for scene in SCENES:
        start = scene_x[scene]
        svg.text(start, 126, f"{scene} (pre-registered held-out)", 24, INK, "bold")
        for index, (label, method) in enumerate((("Ground truth", "gt"), ("Official 3DGS", "official"), ("BR-GS v3.4", "brgs"))):
            x = start + index * 270
            svg.text(x + tile / 2, 158, label, 18, INK, "bold", "middle")
            source = gt_path(root, scene, MAIN_VIEW) if method == "gt" else render_path(root, scene, method, MAIN_SEED, MAIN_VIEW)
            draw_full_with_crop_box(svg, source, x, 175, tile)
            if method != "gt":
                metric = read_metrics(root, scene, method, MAIN_SEED, MAIN_VIEW)
                svg.text(
                    x + tile / 2, 450,
                    f'PSNR {metric["PSNR"]:.2f} dB | LPIPS {metric["LPIPS"]:.3f}',
                    14, MUTED, anchor="middle",
                )
            svg.image(source, x, 485, tile, tile, crop=CROP)
        svg.text(start - 20, 755, "Orange boxes indicate the fixed center crop.", 14, MUTED)
    svg.line(945, 110, 945, 775, "#bbc6d3", 2)
    svg.text(42, 806, "Display-only examples; quantitative conclusions use all 12 test views and all three seeds.", 16, MUTED)
    svg.save(path)


def write_overview(path: Path, root: Path) -> None:
    svg = Svg(1900, 1120)
    svg.text(42, 52, "Held-out fixed-view overview", 34, INK, "bold")
    svg.text(43, 84, "Views are fixed to the first, middle, and last test indices; all renders use seed s0.", 18, MUTED)
    tile = 260
    xs = [150 + index * 290 for index in range(6)]
    columns = [(scene, view) for scene in SCENES for view in OVERVIEW_VIEWS]
    for x, (scene, view) in zip(xs, columns):
        svg.text(x + tile / 2, 132, f"{scene} / {view[:-4]}", 18, INK, "bold", "middle")
    rows = (("Ground truth", "gt"), ("Official 3DGS", "official"), ("BR-GS v3.4", "brgs"))
    ys = (165, 465, 765)
    for y, (label, method) in zip(ys, rows):
        svg.text(18, y + tile / 2, label, 17, INK, "bold")
        for x, (scene, view) in zip(xs, columns):
            source = gt_path(root, scene, view) if method == "gt" else render_path(root, scene, method, MAIN_SEED, view)
            svg.image(source, x, y, tile, tile)
    svg.line(995, 110, 995, 1050, "#bbc6d3", 3)
    svg.text(42, 1085, "Selection is index-based and independent of per-view metric values.", 16, MUTED)
    svg.save(path)


def write_multiseed(path: Path, root: Path) -> None:
    svg = Svg(2000, 790)
    svg.text(42, 52, "Held-out multi-seed comparison", 34, INK, "bold")
    svg.text(43, 84, "Fixed test view 00005; all three pre-registered training seeds are shown.", 18, MUTED)
    tile = 245
    labels = ("Ground truth", "Official s0", "BR-GS s0", "Official s1", "BR-GS s1", "Official s2", "BR-GS s2")
    xs = [105 + index * 270 for index in range(7)]
    for x, label in zip(xs, labels):
        svg.text(x + tile / 2, 135, label, 17, INK, "bold", "middle")
    for row, scene in enumerate(SCENES):
        y = 165 + row * 285
        svg.text(48, y + tile / 2, scene, 20, INK, "bold", "middle")
        sources = [gt_path(root, scene, MAIN_VIEW)]
        for seed in SEEDS:
            sources.extend([
                render_path(root, scene, "official", seed, MAIN_VIEW),
                render_path(root, scene, "brgs", seed, MAIN_VIEW),
            ])
        for x, source in zip(xs, sources):
            svg.image(source, x, y, tile, tile)
    svg.text(42, 757, "This panel exposes seed-to-seed appearance variation rather than selecting a favorable run.", 16, MUTED)
    svg.save(path)


def write_readme(path: Path) -> None:
    path.write_text(
        """# BR-GS v3.4 qualitative figures

The figures use only the pre-registered held-out scenes P03 and P04.

Selection protocol:

- Main figure: fixed test view `00005`, fixed primary seed `s0`, and a fixed center
  crop `(x=180, y=180, width=440, height=440)` in the 800 x 800 source image.
- Fixed-view overview: `00000`, `00005`, and `00011` (first, middle, and last
  test indices), all using seed `s0`.
- Multi-seed panel: fixed test view `00005`, showing seeds `s0`, `s1`, and `s2`.

The selections are index-based, not chosen from per-view metric rankings. The main paper
must retain the all-view, three-seed quantitative table; these images are illustrative.
`INPUT_SHA256SUMS` fixes all 180 source PNG/JSON files used by the generator.

Suggested main-figure caption:

> Qualitative comparison on the pre-registered held-out scenes P03 and P04. We show the
> fixed test view 00005 for the primary seed s0, with identical center crops. BR-GS v3.4
> uses fewer Gaussians, while image-space differences relative to official 3DGS are
> subtle and scene-dependent. Quantitative results average all 12 test views and three
> paired training seeds.

Suggested supplementary caption:

> Multi-seed qualitative comparison for fixed held-out test view 00005. All three
> pre-registered seeds are shown to expose stochastic variation and avoid selecting a
> favorable individual run.
""",
        encoding="utf-8",
    )


def write_manifest(path: Path, artifacts: list[Path]) -> None:
    lines = [f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}" for item in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_input_manifest(path: Path, root: Path) -> None:
    inputs = []
    for scene in SCENES:
        inputs.extend(gt_path(root, scene, view) for view in VIEWS)
        for method in ("official", "brgs"):
            for seed in SEEDS:
                inputs.append(metrics_path(root, scene, method, seed))
                inputs.extend(render_path(root, scene, method, seed, view) for view in VIEWS)
    lines = [
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.relative_to(root)}"
        for item in sorted(inputs)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo / "paper_results/v34/qualitative")
    args = parser.parse_args()

    root = args.input_root.resolve()
    output = args.output_dir.resolve()
    validate_inputs(root)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output / "README.md",
        output / "INPUT_SHA256SUMS",
        output / "figure_v34_qualitative_main.svg",
        output / "figure_v34_qualitative_views.svg",
        output / "figure_v34_qualitative_multiseed.svg",
    ]
    write_readme(artifacts[0])
    write_input_manifest(artifacts[1], root)
    write_main(artifacts[2], root)
    write_overview(artifacts[3], root)
    write_multiseed(artifacts[4], root)
    write_manifest(output / "SHA256SUMS", artifacts)
    print("Validated 168 PNGs and 12 per-view metric files")
    print(f"Saved qualitative figures to: {output}")


if __name__ == "__main__":
    main()

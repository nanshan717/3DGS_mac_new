#!/usr/bin/env python3
"""Dependency-free structural validator for a rendered Blender/3DGS scene."""

import argparse
import json
import math
import struct
from pathlib import Path


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    args = parser.parse_args()
    root = Path(args.scene).resolve()
    frames = []
    translations = []
    split_translations = {}
    split_counts = {}
    for split in ("train", "test"):
        payload = json.loads((root / f"transforms_{split}.json").read_text(encoding="utf-8"))
        split_counts[split] = len(payload["frames"])
        split_translations[split] = []
        for frame in payload["frames"]:
            matrix = frame["transform_matrix"]
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise ValueError(f"Invalid 4x4 pose in {split}: {frame['file_path']}")
            if not all(math.isfinite(value) for row in matrix for value in row):
                raise ValueError(f"Non-finite pose in {split}: {frame['file_path']}")
            if any(abs(matrix[3][i] - expected) > 1e-6
                   for i, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
                raise ValueError(f"Invalid homogeneous row in {split}: {frame['file_path']}")
            translation = tuple(float(matrix[i][3]) for i in range(3))
            translations.append(translation)
            split_translations[split].append(translation)
            frames.append(root / (frame["file_path"] + ".png"))
    missing = [str(path) for path in frames if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced images; first: {missing[0]}")
    sizes = {png_size(path) for path in frames}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent image sizes: {sorted(sizes)}")
    if len(set(frames)) != len(frames):
        raise ValueError("Train/test transforms contain duplicate image paths")
    baseline = max(math.dist(a, b) for a in translations for b in translations)
    if baseline < 1e-3:
        raise ValueError(
            f"Degenerate camera poses: maximum camera baseline is {baseline:.6g}; "
            "all cameras appear colocated"
        )
    if len(frames) >= 90:
        if split_counts != {"train": 84, "test": 12}:
            raise ValueError(f"Expected V7 split 84/12, got {split_counts}")
        test_sides = {"left": 0, "right": 0}
        for x, _, _ in split_translations["test"]:
            test_sides["left" if x < 0 else "right"] += 1
        if test_sides != {"left": 6, "right": 6}:
            raise ValueError(f"Unbalanced V7 test sides: {test_sides}")
        if not (root / "points3d.ply").exists():
            raise FileNotFoundError("V7 requires deterministic points3d.ply")
        mask_manifest_path = root / "bsr_masks_manifest.json"
        if mask_manifest_path.exists():
            mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
            masks = [root / relative for relative in mask_manifest["files"]]
            if len(masks) != len(frames):
                raise ValueError(f"Expected {len(frames)} ROI masks, got {len(masks)}")
            expected_masks = {root / "bsr_masks" / (path.stem + ".png") for path in frames}
            if set(masks) != expected_masks:
                raise ValueError("ROI mask names do not exactly match train/test image stems")
            missing_masks = [str(path) for path in masks if not path.exists()]
            if missing_masks:
                raise FileNotFoundError(f"Missing ROI mask: {missing_masks[0]}")
            mask_sizes = {png_size(path) for path in masks}
            if mask_sizes != sizes:
                raise ValueError(f"ROI mask/image size mismatch: masks={mask_sizes}, images={sizes}")
            roi_masks = len(masks)
        else:
            roi_masks = 0
    else:
        test_sides = None
        roi_masks = 0
    print(json.dumps({
        "scene": str(root), "train": split_counts["train"], "test": split_counts["test"],
        "total": len(frames), "resolution": list(next(iter(sizes))),
        "camera_baseline": baseline, "status": "trainable_structure_ok",
        "test_sides": test_sides,
        "roi_masks": roi_masks,
    }, indent=2))


if __name__ == "__main__":
    main()

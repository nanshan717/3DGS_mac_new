#!/usr/bin/env python3
"""Dependency-free structural validator for a rendered Blender/3DGS scene."""

import argparse
import json
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
    split_counts = {}
    for split in ("train", "test"):
        payload = json.loads((root / f"transforms_{split}.json").read_text(encoding="utf-8"))
        split_counts[split] = len(payload["frames"])
        for frame in payload["frames"]:
            matrix = frame["transform_matrix"]
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise ValueError(f"Invalid 4x4 pose in {split}: {frame['file_path']}")
            frames.append(root / (frame["file_path"] + ".png"))
    missing = [str(path) for path in frames if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced images; first: {missing[0]}")
    sizes = {png_size(path) for path in frames}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent image sizes: {sorted(sizes)}")
    if len(set(frames)) != len(frames):
        raise ValueError("Train/test transforms contain duplicate image paths")
    print(json.dumps({
        "scene": str(root), "train": split_counts["train"], "test": split_counts["test"],
        "total": len(frames), "resolution": list(next(iter(sizes))), "status": "trainable_structure_ok",
    }, indent=2))


if __name__ == "__main__":
    main()

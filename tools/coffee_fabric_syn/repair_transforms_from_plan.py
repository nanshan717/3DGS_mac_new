#!/usr/bin/env python3
"""Rebuild NeRF camera transforms from a generated scene's camera plan."""

import argparse
import json
import math
from pathlib import Path


def subtract(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def normalize(value):
    length = math.sqrt(sum(component * component for component in value))
    if length < 1e-12:
        raise ValueError("Cannot normalize a zero-length camera vector")
    return tuple(component / length for component in value)


def camera_to_world(position, target):
    # Matches Blender's direction.to_track_quat('-Z', 'Y'): camera local -Z
    # points at the target and local Y is kept as close as possible to world Z.
    forward = normalize(subtract(target, position))
    z_axis = tuple(-component for component in forward)
    x_axis = normalize(cross((0.0, 0.0, 1.0), z_axis))
    y_axis = normalize(cross(z_axis, x_axis))
    return [
        [x_axis[0], y_axis[0], z_axis[0], float(position[0])],
        [x_axis[1], y_axis[1], z_axis[1], float(position[1])],
        [x_axis[2], y_axis[2], z_axis[2], float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    args = parser.parse_args()
    root = Path(args.scene).resolve()
    specs = json.loads((root / "camera_plan.json").read_text(encoding="utf-8"))
    grouped = {"train": [], "test": []}
    for spec in specs:
        grouped[spec["role"]].append({
            "file_path": spec["file_path"],
            "transform_matrix": camera_to_world(spec["position"], spec["look_at"]),
        })

    lens = float(specs[0]["focal_length_mm"])
    angle_x = 2.0 * math.atan(36.0 / (2.0 * lens))
    for split, frames in grouped.items():
        output = root / f"transforms_{split}.json"
        output.write_text(json.dumps({"camera_angle_x": angle_x, "frames": frames}, indent=2) + "\n",
                          encoding="utf-8")
        print(f"Saved {len(frames)} poses: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a deterministic CoffeeFabric-Syn fabric mesh, point GT, and camera plan.

This asset-independent stage is intentionally runnable without Blender. Import the mesh
and camera plan into Blender/BlenderProc, add licensed soil/coffee assets, and render the
modalities required by docs/COFFEE_FABRIC_SYN.md.
"""

import argparse
import json
import math
import random
from pathlib import Path


def height(x, y, morphology, rng):
    if morphology == "flat":
        return 0.0
    if morphology == "slope":
        return 0.08 * x - 0.03 * y
    if morphology == "sag":
        return -0.22 * math.exp(-1.7 * (x * x + y * y))
    if morphology == "uplift":
        return 0.25 * math.exp(-5.0 * ((x - 0.45) ** 2 + (y + 0.2) ** 2))
    if morphology == "wrinkle":
        return 0.06 * math.sin(5.0 * x + 0.4) * math.sin(3.0 * y - 0.2)
    # Damaged/buried-edge proxy: smooth fabric with a displaced edge zone.
    return -0.13 / (1.0 + math.exp(-10.0 * (y - 0.65))) + 0.02 * rng.uniform(-1, 1)


def write_mesh(path, vertices, faces):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            f.write("{:.8f} {:.8f} {:.8f}\n".format(*vertex))
        for face in faces:
            f.write("3 {} {} {}\n".format(*face))


def write_points(path, vertices):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for vertex in vertices:
            f.write("{:.8f} {:.8f} {:.8f}\n".format(*vertex))


def camera_plan(count, rng):
    cameras = []
    for i in range(count):
        phase = 2.0 * math.pi * i / count
        radius = rng.uniform(2.7, 3.4)
        cameras.append({
            "frame": i,
            "position": [radius * math.cos(phase), radius * math.sin(phase), rng.uniform(0.65, 1.35)],
            "look_at": [rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.0],
            "focal_length_mm": rng.choice([24, 28, 35]),
            "role": "test" if i % 8 == 0 else "train",
        })
    return cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--morphology", choices=["flat", "slope", "sag", "uplift", "wrinkle", "damaged"], default="wrinkle")
    parser.add_argument("--grid", type=int, default=129)
    parser.add_argument("--views", type=int, default=200)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.output).resolve()
    gt = root / "ground_truth"
    gt.mkdir(parents=True, exist_ok=True)
    n = max(3, args.grid)
    vertices = []
    for i in range(n):
        x = -1.5 + 3.0 * i / (n - 1)
        for j in range(n):
            y = -1.5 + 3.0 * j / (n - 1)
            vertices.append((x, y, height(x, y, args.morphology, rng)))
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.extend(((a, a + n, a + 1), (a + 1, a + n, a + n + 1)))

    write_mesh(gt / "fabric_mesh.ply", vertices, faces)
    write_points(gt / "fabric_points.ply", vertices)
    plan = camera_plan(max(16, args.views), rng)
    metadata = {
        "schema": "coffee-fabric-syn-v1",
        "seed": args.seed,
        "units": "metres",
        "morphology": args.morphology,
        "grid_resolution": n,
        "camera_count": len(plan),
        "status": "layout_only_requires_blenderproc_render",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "camera_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Generated deterministic layout at {root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize checksums/metadata when Blender exits after completing Cycles outputs."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--blender-version", default="5.2.0 LTS")
    args = parser.parse_args()
    root = Path(args.scene).resolve()
    train = json.loads((root / "transforms_train.json").read_text(encoding="utf-8"))
    test = json.loads((root / "transforms_test.json").read_text(encoding="utf-8"))
    metadata = {
        "schema": "coffee-fabric-syn-v1",
        "scene_id": root.name,
        "dataset_type": "synthetic",
        "status": "P01_v4_cycles_visual_attempt_not_final_benchmark",
        "seed": args.seed,
        "units": "metres",
        "morphology": "flat_with_independent_multifrequency_micro_relief",
        "occlusion_level": "low",
        "vegetation_label": "procedural_juvenile_coffee_proxy_qualitatively_calibrated_from_project_field_photos",
        "field_reference_use": "morphology/layout reference only; no source pixels or textures included",
        "renderer": {"blender": args.blender_version, "engine": "CYCLES", "samples": 96, "denoising": True},
        "camera_count": len(train["frames"]) + len(test["frames"]),
        "license_status": "candidate_pending_project_review",
        "known_limitations": ["procedural vegetation is not a photogrammetric plant replica", "visual prototype only"],
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = [
        root / f"{root.name}.blend", root / "previews/P01_paper_overview.png",
        root / "previews/P01_root_detail.png", root / "previews/P01_row_level.png",
        root / "ground_truth/fabric_mesh.ply", root / "camera_plan.json",
        root / "transforms_train.json", root / "transforms_test.json", metadata_path,
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs: " + ", ".join(missing))
    (root / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(root)}" for path in files) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

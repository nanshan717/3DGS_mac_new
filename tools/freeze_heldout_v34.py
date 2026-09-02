#!/usr/bin/env python3
"""Validate and cryptographically freeze P03/P04 before any model is trained."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.coffee_fabric_syn.heldout_profiles import HELDOUT_PROFILES
except ModuleNotFoundError:  # Direct execution via python tools/freeze_heldout_v34.py.
    from coffee_fabric_syn.heldout_profiles import HELDOUT_PROFILES


LOCK_NAME = "HELDOUT_FREEZE.json"
CHECKSUM_NAME = "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_paths(root: Path, split: str) -> list[Path]:
    payload = read_json(root / f"transforms_{split}.json")
    return [root / (frame["file_path"] + ".png") for frame in payload["frames"]]


def validate_scene(root: Path, scene_id: str, declaration: dict) -> list[Path]:
    profile = declaration["profile"]
    spec = HELDOUT_PROFILES[profile]
    metadata = read_json(root / "metadata.json")
    if root.name != spec["scene_id"] or metadata.get("scene_id") != spec["scene_id"]:
        raise ValueError(f"Scene ID mismatch for {scene_id}: {root}")
    if metadata.get("scene_role") != "held_out":
        raise ValueError(f"{scene_id} is not labelled held_out")
    if int(metadata.get("seed", -1)) != int(spec["seed"]) or int(spec["seed"]) != int(declaration["dataset_seed"]):
        raise ValueError(f"Pre-registered dataset seed mismatch for {scene_id}")
    if metadata.get("morphology") != spec["morphology"]:
        raise ValueError(f"Pre-registered morphology mismatch for {scene_id}")
    if metadata.get("occlusion_level") != spec["occlusion_level"]:
        raise ValueError(f"Pre-registered occlusion mismatch for {scene_id}")
    if metadata.get("lighting_protocol") != spec["lighting"]:
        raise ValueError(f"Pre-registered lighting mismatch for {scene_id}")
    if metadata.get("camera_count") != 96 or not metadata.get("rgb_dataset_rendered"):
        raise ValueError(f"{scene_id} does not contain the complete rendered V7 protocol")

    train = frame_paths(root, "train")
    test = frame_paths(root, "test")
    if len(train) != 84 or len(test) != 12 or len(set(train + test)) != 96:
        raise ValueError(f"{scene_id} must contain a unique 84/12 frame split")
    render_manifest = read_json(root / "render_manifest.json")
    mask_manifest = read_json(root / "bsr_masks_manifest.json")
    expected_images = {path.relative_to(root).as_posix() for path in train + test}
    expected_masks = {f"bsr_masks/{path.stem}.png" for path in train + test}
    if render_manifest.get("count") != 96 or set(render_manifest.get("files", [])) != expected_images:
        raise ValueError(f"RGB manifest mismatch for {scene_id}")
    if mask_manifest.get("count") != 96 or set(mask_manifest.get("files", [])) != expected_masks:
        raise ValueError(f"ROI mask manifest mismatch for {scene_id}")

    required = [
        root / "metadata.json", root / "camera_plan.json",
        root / "transforms_train.json", root / "transforms_test.json",
        root / "points3d.ply", root / "ground_truth" / "fabric_mesh.ply",
        root / "ASSET_ATTRIBUTION.json", root / "render_manifest.json",
        root / "bsr_masks_manifest.json", root / f"{root.name}.blend",
        *train, *test, *(root / relative for relative in sorted(expected_masks)),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} held-out files; first: {missing[0]}")
    return required


def model_outputs_exist(matrix: dict) -> list[Path]:
    existing = []
    for scene_id in matrix["scenes"]:
        for method in matrix["methods"].values():
            repo = Path(matrix[method["repo"]])
            for seed in matrix["seeds"]:
                output = repo / method["output_pattern"].format(scene=scene_id, seed=seed)
                if output.exists():
                    existing.append(output)
    return existing


def write_scene_lock(root: Path, scene_id: str, declaration: dict, required: list[Path], frozen_at: str) -> str:
    metadata_path = root / "metadata.json"
    metadata = read_json(metadata_path)
    metadata["status"] = "heldout_frozen_before_training"
    metadata["frozen_at_utc"] = frozen_at
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Include every generated artifact, except the checksum and lock files that
    # form the two upper links of the integrity chain.
    generated = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {CHECKSUM_NAME, LOCK_NAME}
    )
    required_set = {path.resolve() for path in required}
    if not required_set.issubset({path.resolve() for path in generated}):
        raise RuntimeError(f"Internal freeze inventory error for {scene_id}")
    checksum_path = root / CHECKSUM_NAME
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in generated),
        encoding="utf-8",
    )
    lock = {
        "schema": "coffee-fabric-syn-heldout-freeze-v1",
        "scene": scene_id,
        "scene_id": root.name,
        "profile": declaration["profile"],
        "dataset_seed": declaration["dataset_seed"],
        "frozen_at_utc": frozen_at,
        "file_count": len(generated),
        "sha256sums_sha256": sha256(checksum_path),
        "metadata_sha256": sha256(metadata_path),
        "training_started_before_freeze": False,
    }
    lock_path = root / LOCK_NAME
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256(lock_path)


def verify_scene_lock(root: Path, expected_lock_hash: str) -> None:
    lock_path = root / LOCK_NAME
    if sha256(lock_path) != expected_lock_hash:
        raise ValueError(f"Held-out lock hash mismatch: {lock_path}")
    lock = read_json(lock_path)
    checksum_path = root / CHECKSUM_NAME
    if sha256(checksum_path) != lock["sha256sums_sha256"]:
        raise ValueError(f"Checksum manifest changed after freeze: {checksum_path}")
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen held-out artifact changed: {path}")
    if len(lines) != int(lock["file_count"]):
        raise ValueError(f"Frozen file count mismatch for {root}")


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=project / "experiments" / "heldout_v34_matrix.json")
    parser.add_argument("--data_root", type=Path, help="Override CoffeeFabric-Syn/heldout root (useful off-server)")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    matrix_path = args.matrix.expanduser().resolve()
    matrix = read_json(matrix_path)
    if matrix.get("schema") != "brgs-frozen-v34-matrix-v1":
        raise ValueError("Unexpected held-out matrix schema")
    if set(matrix.get("scenes", {})) != {"P03", "P04"}:
        raise ValueError("Held-out matrix must contain exactly P03 and P04")

    roots = {}
    for scene_id, declaration in matrix["scenes"].items():
        root = ((args.data_root.expanduser().resolve() / HELDOUT_PROFILES[declaration["profile"]]["scene_id"])
                if args.data_root else Path(declaration["source"]).expanduser().resolve())
        roots[scene_id] = root

    if args.verify_only:
        if matrix.get("status") != "frozen":
            raise ValueError("Cannot verify a held-out matrix that is not frozen")
        for scene_id, root in roots.items():
            verify_scene_lock(root, matrix["scenes"][scene_id]["dataset_lock_sha256"])
        print(f"Verified {len(roots)} frozen held-out datasets")
        return

    if matrix.get("status") != "preregistered":
        raise ValueError("Freeze is allowed exactly once from preregistered status")
    existing = model_outputs_exist(matrix)
    if existing:
        raise RuntimeError(f"Refusing post-training freeze; model output already exists: {existing[0]}")

    inventories = {
        scene_id: validate_scene(root, scene_id, matrix["scenes"][scene_id])
        for scene_id, root in roots.items()
    }
    frozen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for scene_id, root in roots.items():
        matrix["scenes"][scene_id]["dataset_lock_sha256"] = write_scene_lock(
            root, scene_id, matrix["scenes"][scene_id], inventories[scene_id], frozen_at
        )
    matrix["status"] = "frozen"
    matrix["frozen_at_utc"] = frozen_at
    matrix["generator_commit"] = git_commit(project)
    matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen {len(roots)} held-out datasets in {matrix_path}")


if __name__ == "__main__":
    main()

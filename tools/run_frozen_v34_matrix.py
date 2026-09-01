#!/usr/bin/env python3
"""Print or execute the frozen official-3DGS versus BR-GS v3.4 matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def load_matrix(path: Path) -> dict:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema") != "brgs-frozen-v34-matrix-v1" or matrix.get("status") != "frozen":
        raise ValueError("Expected a frozen brgs-frozen-v34-matrix-v1 file")
    if matrix.get("iterations") != 15000:
        raise ValueError("The frozen v3.4 protocol requires 15000 iterations")
    flags = matrix["methods"]["brgs_v34"].get("train_flags", [])
    if flags != ["--bsr_v34"]:
        raise ValueError("Frozen BR-GS flags must contain only --bsr_v34")
    return matrix


def selected(values, requested):
    return values if not requested else [value for value in values if str(value) in requested]


def quote(command) -> str:
    return " ".join(shlex.quote(str(token)) for token in command)


def preflight(matrix: dict) -> None:
    brgs = Path(matrix["brgs_repo"])
    official = Path(matrix["official_repo"])
    required = [brgs / "metrics.py", brgs / "eval_geometry_gt.py", official / "train.py",
                official / "render.py", official / "utils" / "general_utils.py"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    official_train = (official / "train.py").read_text(encoding="utf-8")
    official_utils = (official / "utils" / "general_utils.py").read_text(encoding="utf-8")
    if "--seed" not in official_train or "def safe_state(silent, seed=0" not in official_utils:
        raise RuntimeError(
            "Official compatibility checkout lacks audited seed controls. Run "
            "tools/prepare_official_seed_support.py <official_repo> --apply first."
        )
    for scene in matrix["scenes"].values():
        source = Path(scene["source"])
        for relative in ("metadata.json", "transforms_train.json", "transforms_test.json",
                         "ground_truth/fabric_mesh.ply"):
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)


def commands_for(matrix: dict, scene_id: str, method_id: str, seed: int):
    method = matrix["methods"][method_id]
    repo = Path(matrix[method["repo"]])
    brgs = Path(matrix["brgs_repo"])
    source = Path(matrix["scenes"][scene_id]["source"])
    relative_output = method["output_pattern"].format(scene=scene_id, seed=seed)
    output = repo / relative_output
    python = sys.executable
    train = [python, "train.py", "-s", source, "-m", relative_output, "--eval", "-r",
             str(matrix["resolution"]), "--iterations", str(matrix["iterations"]),
             "--seed", str(seed), "--deterministic", *method.get("train_flags", [])]
    render = [python, "render.py", "-m", relative_output, "--iteration", str(matrix["iterations"])]
    metrics = [python, str(brgs / "metrics.py"), "-m", output]
    geometry = [python, str(brgs / "eval_geometry_gt.py"), "-s", source, "-m", output,
                "--iteration", str(matrix["iterations"]), "--save_json"]
    return repo, output, {"train": train, "render": render, "metrics": metrics, "geometry": geometry}


def main() -> None:
    default_matrix = Path(__file__).resolve().parents[1] / "experiments" / "frozen_v34_matrix.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=default_matrix)
    parser.add_argument("--scene", action="append", choices=["P01", "P02"])
    parser.add_argument("--method", action="append", choices=["official_3dgs", "brgs_v34"])
    parser.add_argument("--seed", action="append", type=int, choices=[0, 1, 2])
    parser.add_argument("--stage", action="append", choices=["train", "render", "metrics", "geometry"])
    parser.add_argument("--execute", action="store_true", help="Execute instead of print")
    parser.add_argument("--skip_preflight", action="store_true", help="Print plans on a non-server host")
    parser.add_argument("--allow_existing_train", action="store_true",
                        help="Allow training into an already existing model directory")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix.expanduser().resolve())
    if not args.skip_preflight:
        preflight(matrix)
    scenes = selected(list(matrix["scenes"]), args.scene)
    methods = selected(list(matrix["methods"]), args.method)
    seeds = selected(matrix["seeds"], {str(value) for value in args.seed} if args.seed else None)
    stages = args.stage or ["train", "render", "metrics", "geometry"]

    for scene in scenes:
        for method in methods:
            for seed in seeds:
                repo, output, commands = commands_for(matrix, scene, method, seed)
                print(f"\n# {scene} | {method} | seed={seed} | output={output}")
                for stage in stages:
                    command = commands[stage]
                    print(f"(cd {shlex.quote(str(repo))} && {quote(command)})")
                    if not args.execute:
                        continue
                    if stage == "train" and output.exists() and not args.allow_existing_train:
                        raise FileExistsError(
                            f"Refusing to train into existing directory {output}; choose a clean matrix output"
                        )
                    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()

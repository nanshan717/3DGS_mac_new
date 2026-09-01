#!/usr/bin/env python3
"""Add auditable RNG controls to an official-3DGS compatibility checkout.

This is infrastructure-only: it changes random-number initialization and CLI plumbing,
not rendering, optimization, losses, densification, or pruning.  The pristine official
checkout should remain untouched; apply this only to a clearly named compatibility copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "official-3dgs-rng-compat-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_general_utils(text: str) -> str:
    text = text.replace("def safe_state(silent):", "def safe_state(silent, seed=0, deterministic=False):")
    text = text.replace("random.seed(0)\n    np.random.seed(0)\n    torch.manual_seed(0)",
                        "random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n"
                        "    torch.cuda.manual_seed_all(seed)\n"
                        "    if hasattr(torch.backends, \"cudnn\"):\n"
                        "        torch.backends.cudnn.deterministic = bool(deterministic)\n"
                        "        torch.backends.cudnn.benchmark = not bool(deterministic)")
    return text


def patch_train(text: str) -> str:
    quiet = '    parser.add_argument("--quiet", action="store_true")'
    controls = (quiet + '\n    parser.add_argument("--seed", type=int, default=0)'
                '\n    parser.add_argument("--deterministic", action="store_true", default=False)')
    if '--seed", type=int' not in text:
        if quiet not in text:
            raise ValueError("Could not locate the official --quiet parser declaration")
        text = text.replace(quiet, controls, 1)
    text = text.replace("safe_state(args.quiet)",
                        "safe_state(args.quiet, args.seed, args.deterministic)")
    return text


def is_patched(train_text: str, utils_text: str) -> bool:
    return ("--seed" in train_text and "--deterministic" in train_text
            and "safe_state(args.quiet, args.seed, args.deterministic)" in train_text
            and "def safe_state(silent, seed=0, deterministic=False):" in utils_text
            and "torch.cuda.manual_seed_all(seed)" in utils_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Official 3DGS compatibility checkout")
    parser.add_argument("--apply", action="store_true", help="Write changes; otherwise only check")
    parser.add_argument("--allow_noncompat_name", action="store_true",
                        help="Override the safety check requiring 'compat' in the directory name")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if "compat" not in repo.name.lower() and not args.allow_noncompat_name:
        raise ValueError("Refusing to patch a checkout not clearly named as a compatibility copy")
    train_path = repo / "train.py"
    utils_path = repo / "utils" / "general_utils.py"
    for path in (train_path, utils_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    before = {str(path.relative_to(repo)): sha256(path) for path in (train_path, utils_path)}
    train_text = train_path.read_text(encoding="utf-8")
    utils_text = utils_path.read_text(encoding="utf-8")
    if is_patched(train_text, utils_text):
        print(f"RNG controls already present in {repo}")
        return
    if not args.apply:
        raise SystemExit("RNG controls are missing; rerun with --apply on the compatibility checkout")

    patched_train = patch_train(train_text)
    patched_utils = patch_general_utils(utils_text)
    if not is_patched(patched_train, patched_utils):
        raise RuntimeError("Patch verification failed; no files were written")
    train_path.write_text(patched_train, encoding="utf-8")
    utils_path.write_text(patched_utils, encoding="utf-8")
    after = {str(path.relative_to(repo)): sha256(path) for path in (train_path, utils_path)}
    audit = {
        "schema": SCHEMA,
        "repo": str(repo),
        "scope": "RNG seeding and deterministic backend controls only",
        "algorithm_changed": False,
        "before_sha256": before,
        "after_sha256": after,
    }
    audit_path = repo / "REPRODUCIBILITY_PATCH.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Applied RNG compatibility controls; audit: {audit_path}")


if __name__ == "__main__":
    main()

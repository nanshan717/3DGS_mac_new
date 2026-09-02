import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.coffee_fabric_syn.heldout_profiles import HELDOUT_PROFILES, heldout_height
from tools.freeze_heldout_v34 import verify_scene_lock, write_scene_lock
from tools.run_frozen_v34_matrix import commands_for, load_matrix


class HeldoutV34Tests(unittest.TestCase):
    def test_profiles_are_distinct_and_fixed(self):
        self.assertEqual(HELDOUT_PROFILES["p03"]["seed"], 303)
        self.assertEqual(HELDOUT_PROFILES["p04"]["seed"], 404)
        self.assertNotEqual(HELDOUT_PROFILES["p03"]["plant_positions"],
                            HELDOUT_PROFILES["p04"]["plant_positions"])
        samples = [heldout_height(profile, 0.7, -1.1) for profile in ("p03", "p04")]
        self.assertTrue(all(math.isfinite(value) for value in samples))
        self.assertNotAlmostEqual(*samples)

    def test_preregistered_matrix_cannot_run(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "matrix.json"
            path.write_text(json.dumps({
                "schema": "brgs-frozen-v34-matrix-v1", "status": "preregistered",
                "iterations": 15000, "methods": {"brgs_v34": {"train_flags": ["--bsr_v34"]}},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_matrix(path)

    def test_runner_accepts_dynamic_heldout_scene(self):
        matrix = {
            "iterations": 15000, "resolution": 1,
            "brgs_repo": "/brgs", "official_repo": "/official",
            "scenes": {"P03": {"source": "/data/P03"}},
            "methods": {"brgs_v34": {
                "repo": "brgs_repo", "output_pattern": "output/{scene}_brgs_v34_15k_s{seed}",
                "train_flags": ["--bsr_v34"],
            }},
        }
        repo, output, commands = commands_for(matrix, "P03", "brgs_v34", 2)
        self.assertEqual(repo, Path("/brgs"))
        self.assertEqual(output, Path("/brgs/output/P03_brgs_v34_15k_s2"))
        self.assertIn("--bsr_v34", commands["train"])
        self.assertIn("2", commands["train"])

    def test_lock_chain_detects_artifact_change(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / HELDOUT_PROFILES["p03"]["scene_id"]
            root.mkdir()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"status": "candidate"}), encoding="utf-8")
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"before")
            declaration = {"profile": "p03", "dataset_seed": 303}
            digest = write_scene_lock(root, "P03", declaration, [metadata, artifact],
                                      "2026-09-02T00:00:00+00:00")
            verify_scene_lock(root, digest)
            artifact.write_bytes(b"after")
            with self.assertRaises(ValueError):
                verify_scene_lock(root, digest)


if __name__ == "__main__":
    unittest.main()

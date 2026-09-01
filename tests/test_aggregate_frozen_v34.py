import json
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_frozen_v34 import load_run, summarize


class AggregateFrozenV34Tests(unittest.TestCase):
    def test_summary_uses_sample_standard_deviation(self):
        rows = []
        for seed, psnr in enumerate((10.0, 11.0, 12.0)):
            row = {"scene": "P01", "role": "development", "method": "brgs_v34", "seed": seed}
            for metric in (
                "ssim", "lpips", "points", "accuracy_mean_m", "accuracy_p90_m",
                "completeness_mean_m", "completeness_p90_m", "chamfer_l1_m",
                "fscore_1cm", "fscore_2cm", "fscore_5cm", "pruned_fraction",
            ):
                row[metric] = 1.0
            row["psnr"] = psnr
            rows.append(row)
        result = summarize(rows)[0]
        self.assertEqual(result["runs"], 3)
        self.assertAlmostEqual(result["psnr_mean"], 11.0)
        self.assertAlmostEqual(result["psnr_std"], 1.0)

    def test_load_run_checks_frozen_audit_and_gt_metrics(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = tmp / "scene"
            source.mkdir()
            repo = tmp / "repo"
            model = repo / "output" / "P01_brgs_v34_15k_s0"
            model.mkdir(parents=True)
            (model / "experiment_manifest.json").write_text(json.dumps({
                "dataset": {"source_path": str(source), "seed": 0},
                "optimization": {"iterations": 15000},
                "argv": ["train.py", "--bsr_v34"],
            }), encoding="utf-8")
            (model / "results.json").write_text(json.dumps({
                "ours_15000": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.3}
            }), encoding="utf-8")
            summary = {"mean_m": 0.1, "p90_m": 0.2}
            (model / "gt_geometry_results_iter-15000.json").write_text(json.dumps({
                "iteration": 15000,
                "source_path": str(source),
                "counts": {"total_gaussians": 95},
                "metrics": {
                    "accuracy_pred_to_gt": summary,
                    "completeness_gt_to_pred": summary,
                    "chamfer_l1_m": 0.1,
                    "threshold_metrics": {
                        "0.010m": {"fscore": 0.1},
                        "0.020m": {"fscore": 0.2},
                        "0.050m": {"fscore": 0.5},
                    },
                },
            }), encoding="utf-8")
            (model / "bsr_v34_pruning.json").write_text(json.dumps({
                "points_after": 95, "removed_fraction": 0.05
            }), encoding="utf-8")
            matrix = {
                "iterations": 15000,
                "brgs_repo": str(repo),
                "scenes": {"P01": {"source": str(source), "role": "development"}},
                "methods": {"brgs_v34": {
                    "repo": "brgs_repo", "output_pattern": "output/{scene}_brgs_v34_15k_s{seed}"
                }},
            }
            row = load_run(matrix, "P01", "brgs_v34", 0)
            self.assertEqual(row["points"], 95)
            self.assertEqual(row["pruned_fraction"], 0.05)
            self.assertEqual(row["fscore_5cm"], 0.5)


if __name__ == "__main__":
    unittest.main()

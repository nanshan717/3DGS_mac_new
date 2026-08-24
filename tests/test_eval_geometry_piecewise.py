import unittest

import numpy as np

from eval_geometry import evaluate_piecewise_bottom


class PiecewiseGeometryTests(unittest.TestCase):
    def test_joined_plane_has_small_error_and_seams(self):
        u = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
        v = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        points = np.stack((uu, vv, 0.1 * uu - 0.05 * vv), axis=-1).reshape(-1, 3)
        result = evaluate_piecewise_bottom(
            points, axis_idx=2, patches_u=2, patches_v=2,
            floater_tau_factor=2.5, control_points_u=5, control_points_v=5,
            surface_samples_u=64, surface_samples_v=64, fit_ridge=1e-6,
            distance_chunk_size=512, max_fit_points=20000,
        )
        self.assertEqual(len(result["patches"]), 4)
        self.assertLess(result["normalized_gsd"], 0.02)
        self.assertLess(result["seam_position_error"], 1e-3)
        self.assertLess(result["roughness"], 1e-3)


if __name__ == "__main__":
    unittest.main()

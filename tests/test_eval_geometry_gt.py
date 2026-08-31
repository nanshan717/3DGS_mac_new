import unittest

import numpy as np

from eval_geometry_gt import (
    evaluate, sample_mesh_uniform, select_fabric_candidates, triangulate_faces,
    voxel_downsample,
)


class GroundTruthGeometryTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.gt = sample_mesh_uniform(self.vertices, self.faces, 20000, 7)

    def test_exact_plane_scores_well(self):
        result = evaluate(self.gt.copy(), self.gt, (0.01,))
        self.assertLess(result["chamfer_l1_m"], 1e-7)
        self.assertAlmostEqual(result["threshold_metrics"]["0.010m"]["fscore"], 1.0)

    def test_candidate_filter_excludes_canopy_but_keeps_below_surface(self):
        xyz = np.array([[0.5, 0.5, 0.01], [0.5, 0.5, 0.40], [0.5, 0.5, -0.30]], dtype=np.float32)
        selected, counts = select_fabric_candidates(
            xyz, np.ones(3, dtype=np.float32), self.gt, 0.05, 0.15, 0.02)
        self.assertEqual(counts["fabric_candidates"], 2)
        self.assertTrue(np.any(selected[:, 2] < -0.2))

    def test_sampling_is_deterministic(self):
        first = sample_mesh_uniform(self.vertices, self.faces, 128, 11)
        second = sample_mesh_uniform(self.vertices, self.faces, 128, 11)
        np.testing.assert_array_equal(first, second)

    def test_quad_is_fan_triangulated(self):
        triangles = triangulate_faces([[0, 1, 2, 3]])
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3]])

    def test_voxel_downsampling_averages_duplicate_density(self):
        points = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.020, 0, 0]], dtype=np.float32)
        reduced = voxel_downsample(points, 0.005)
        self.assertEqual(len(reduced), 2)
        self.assertAlmostEqual(float(reduced[0, 0]), 0.0015, places=6)


if __name__ == "__main__":
    unittest.main()

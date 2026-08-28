import unittest

import torch

from utils.bernstein_utils import (
    bernstein_control_regularization,
    bernstein_surface_distance_loss,
    evaluate_bernstein_surface,
)


class BernsteinV3Tests(unittest.TestCase):
    def test_piecewise_surface_shape_and_gradient(self):
        cp = torch.randn(2, 2, 5, 5, 3, requires_grad=True)
        surface = evaluate_bernstein_surface(cp, 8, 7)
        self.assertEqual(surface.shape, (2 * 2 * 8 * 7, 3))
        surface.square().mean().backward()
        self.assertTrue(torch.isfinite(cp.grad).all())

    def test_flat_joined_patches_have_zero_regularizers(self):
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        patches = []
        for pu in range(2):
            row = []
            for pv in range(2):
                xx, yy = torch.meshgrid((x + pu), (y + pv), indexing="ij")
                row.append(torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1))
            patches.append(torch.stack(row))
        cp = torch.stack(patches)
        smoothness, continuity = bernstein_control_regularization(cp)
        self.assertLess(float(smoothness), 1e-10)
        self.assertLess(float(continuity), 1e-10)

    def test_full_loss_is_finite(self):
        cp = torch.randn(2, 2, 4, 4, 3, requires_grad=True)
        points = torch.randn(128, 3, requires_grad=True)
        loss, debug = bernstein_surface_distance_loss(
            points,
            cp,
            point_weights=torch.rand(128),
            opacities=torch.sigmoid(torch.randn(128, 1, requires_grad=True)),
            samples_u=8,
            samples_v=8,
            max_points=64,
            support_scale=2.0,
            coverage_lambda=0.1,
            control_smoothness_lambda=0.01,
            patch_continuity_lambda=1.0,
            floater_lambda=0.05,
            spatial_sampling=True,
            frame_origin=torch.zeros(3),
            frame_u=torch.tensor([1.0, 0.0, 0.0]),
            frame_v=torch.tensor([0.0, 1.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("coverage_loss", debug)
        loss.backward()
        self.assertTrue(torch.isfinite(cp.grad).all())
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_v32_floater_distance_has_corrective_point_gradient(self):
        x = torch.linspace(-1, 1, 4)
        y = torch.linspace(-1, 1, 4)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        cp = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).requires_grad_(True)
        points = torch.tensor([[0.0, 0.0, 0.40], [0.2, 0.1, 0.01]], requires_grad=True)
        loss, debug = bernstein_surface_distance_loss(
            points,
            cp,
            point_weights=torch.ones(2),
            opacities=torch.full((2, 1), 0.9),
            samples_u=8,
            samples_v=8,
            robust_delta=0.02,
            floater_lambda=1.0,
            floater_margin=0.05,
            floater_points=points,
            floater_weights=torch.ones(2),
            floater_opacities=torch.full((2, 1), 0.9),
            surface_deadzone=0.01,
            surface_one_sided=True,
            surface_normal=torch.tensor([0.0, 0.0, 1.0]),
            floater_distance_loss=True,
            floater_opacity_min=0.05,
        )
        loss.backward()
        self.assertGreater(debug["floater_loss"], 0.0)
        self.assertGreater(float(points.grad[0, 2]), 0.0)
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_v33_opacity_only_floater_does_not_move_points(self):
        x = torch.linspace(-1, 1, 4)
        y = torch.linspace(-1, 1, 4)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        cp = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).requires_grad_(True)
        points = torch.tensor([[0.0, 0.0, 0.40]], requires_grad=True)
        opacity = torch.tensor([[0.9]], requires_grad=True)
        loss, _ = bernstein_surface_distance_loss(
            points,
            cp,
            point_weights=torch.ones(1),
            samples_u=8,
            samples_v=8,
            floater_lambda=1.0,
            floater_margin=0.05,
            floater_points=points,
            floater_weights=torch.ones(1),
            floater_opacities=opacity,
            floater_distance_loss=False,
            floater_opacity_min=0.05,
            surface_loss_lambda=0.0,
        )
        loss.backward()
        self.assertIsNotNone(opacity.grad)
        self.assertGreater(float(opacity.grad), 0.0)
        self.assertTrue(points.grad is None or torch.equal(points.grad, torch.zeros_like(points)))


if __name__ == "__main__":
    unittest.main()

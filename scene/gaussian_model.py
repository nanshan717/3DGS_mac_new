#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.bernstein_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._bernstein_control_points = torch.empty(0)
        self._bsr_frame_origin = torch.empty(0)
        self._bsr_frame_u = torch.empty(0)
        self._bsr_frame_v = torch.empty(0)
        self._bsr_frame_normal = torch.empty(0)
        self._bsr_support_scale = torch.tensor(1.0)
        self._bsr_height_hook = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._bernstein_control_points,
            self.bernstein_optimizer.state_dict() if self.bernstein_optimizer is not None else None,
            self.get_bsr_metadata(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args[:12]
        if len(model_args) > 12 and model_args[12].numel() > 0:
            self._bernstein_control_points = nn.Parameter(model_args[12].detach().cuda().requires_grad_(True))
        if len(model_args) > 14 and model_args[14] is not None:
            self.set_bsr_metadata(model_args[14])
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if len(model_args) > 13 and model_args[13] is not None and self.bernstein_optimizer is not None:
            self.bernstein_optimizer.load_state_dict(model_args[13])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    @property
    def get_bernstein_control_points(self):
        return self._bernstein_control_points

    @property
    def has_bernstein_surface(self):
        return isinstance(self._bernstein_control_points, nn.Parameter) and self._bernstein_control_points.numel() > 0

    def get_bsr_metadata(self):
        if self._bsr_frame_normal.numel() == 0:
            return None
        return {
            "origin": self._bsr_frame_origin.detach().cpu(),
            "u": self._bsr_frame_u.detach().cpu(),
            "v": self._bsr_frame_v.detach().cpu(),
            "normal": self._bsr_frame_normal.detach().cpu(),
            "support_scale": float(self._bsr_support_scale.detach().cpu()),
        }

    def set_bsr_metadata(self, metadata):
        if not metadata:
            return
        device = self.get_xyz.device
        self._bsr_frame_origin = metadata["origin"].to(device)
        self._bsr_frame_u = metadata["u"].to(device)
        self._bsr_frame_v = metadata["v"].to(device)
        self._bsr_frame_normal = metadata["normal"].to(device)
        self._bsr_support_scale = torch.tensor(float(metadata["support_scale"]), device=device)

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

        if training_args.use_bsr:
            self.initialize_bernstein_surface(
                training_args.bsr_control_points_u,
                training_args.bsr_control_points_v,
                training_args.bsr_z_percentile,
                training_args.bsr_axis_mode,
                training_args.bsr_num_patches_u,
                training_args.bsr_num_patches_v,
            )
            if training_args.bsr_height_only and self._bsr_frame_normal.numel() == 3:
                normal = self._bsr_frame_normal.detach().clone()
                self._bsr_height_hook = self._bernstein_control_points.register_hook(
                    lambda grad: (grad * normal).sum(dim=-1, keepdim=True) * normal
                )
            self.bernstein_optimizer = torch.optim.Adam(
                [self._bernstein_control_points],
                lr=training_args.bsr_control_lr,
                eps=1e-15,
            )
        else:
            self.bernstein_optimizer = None

    def initialize_bernstein_surface(self, control_points_u, control_points_v, z_percentile,
                                     axis_mode="z", num_patches_u=1, num_patches_v=1):
        if self.has_bernstein_surface:
            return
        if self.get_xyz.numel() == 0:
            self._bernstein_control_points = nn.Parameter(torch.empty(0, device="cuda"))
            return

        with torch.no_grad():
            xyz = self.get_xyz.detach()
            normal = torch.tensor([0.0, 0.0, 1.0], device=xyz.device, dtype=xyz.dtype)
            if axis_mode == "x":
                normal = torch.tensor([1.0, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype)
            elif axis_mode == "y":
                normal = torch.tensor([0.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
            elif axis_mode == "auto":
                # Robustly initialize from the low-z support candidates. The sign is
                # fixed against +z so that "lower" keeps a deterministic meaning.
                z0 = torch.quantile(xyz[:, 2], min(0.5, max(float(z_percentile) * 2.0, 0.1)))
                candidates = xyz[xyz[:, 2] <= z0]
                if candidates.shape[0] >= 3:
                    centered = candidates - candidates.mean(dim=0, keepdim=True)
                    covariance = centered.T @ centered / max(candidates.shape[0] - 1, 1)
                    _, eigenvectors = torch.linalg.eigh(covariance)
                    normal = eigenvectors[:, 0]
                    if normal[2] < 0:
                        normal = -normal
            normal = normal / normal.norm().clamp_min(1e-8)
            helper = torch.tensor([1.0, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype)
            if torch.abs(torch.dot(helper, normal)) > 0.9:
                helper = torch.tensor([0.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
            axis_u = torch.cross(normal, helper, dim=0)
            axis_u = axis_u / axis_u.norm().clamp_min(1e-8)
            axis_v = torch.cross(normal, axis_u, dim=0)
            axis_v = axis_v / axis_v.norm().clamp_min(1e-8)
            origin = xyz.mean(dim=0)
            height = (xyz - origin) @ normal
            height_base = torch.quantile(height, float(z_percentile))
            lower_mask = height <= height_base
            lower_xyz = xyz[lower_mask] if lower_mask.any() else xyz
            rel = lower_xyz - origin
            coord_u, coord_v = rel @ axis_u, rel @ axis_v
            u_min, u_max = torch.quantile(coord_u, 0.01), torch.quantile(coord_u, 0.99)
            v_min, v_max = torch.quantile(coord_v, 0.01), torch.quantile(coord_v, 0.99)
            patches = []
            for pu in range(max(1, int(num_patches_u))):
                row = []
                ua = u_min + (u_max - u_min) * pu / max(1, int(num_patches_u))
                ub = u_min + (u_max - u_min) * (pu + 1) / max(1, int(num_patches_u))
                for pv in range(max(1, int(num_patches_v))):
                    va = v_min + (v_max - v_min) * pv / max(1, int(num_patches_v))
                    vb = v_min + (v_max - v_min) * (pv + 1) / max(1, int(num_patches_v))
                    gu = torch.linspace(ua, ub, control_points_u, device=xyz.device, dtype=xyz.dtype)
                    gv = torch.linspace(va, vb, control_points_v, device=xyz.device, dtype=xyz.dtype)
                    mesh_u, mesh_v = torch.meshgrid(gu, gv, indexing="ij")
                    cp = origin + mesh_u[..., None] * axis_u + mesh_v[..., None] * axis_v
                    cp = cp + height_base * normal
                    row.append(cp)
                patches.append(torch.stack(row, dim=0))
            control_points = torch.stack(patches, dim=0)
            self._bsr_frame_origin = origin
            self._bsr_frame_u = axis_u
            self._bsr_frame_v = axis_v
            self._bsr_frame_normal = normal
            self._bsr_support_scale = torch.sqrt((u_max-u_min).square() + (v_max-v_min).square()).clamp_min(1e-6)

        self._bernstein_control_points = nn.Parameter(control_points.requires_grad_(True))

    def get_bsr_soft_weights(self, z_percentile=0.2, opacity_threshold=0.05, z_softness=0.04, min_weight=0.02):
        if self.get_xyz.numel() == 0:
            return torch.empty(0, device="cuda")
        with torch.no_grad():
            xyz = self.get_xyz.detach()
            opacity = self.get_opacity.detach().squeeze(-1)
            heights = xyz[:, 2]
            if self._bsr_frame_normal.numel() == 3:
                heights = (xyz - self._bsr_frame_origin) @ self._bsr_frame_normal
            z_cutoff = torch.quantile(heights, float(z_percentile))
            z_range = torch.quantile(heights, 0.95) - torch.quantile(heights, 0.05)
            z_scale = torch.clamp(z_range * float(z_softness), min=1e-6)
            z_weight = torch.sigmoid((z_cutoff - heights) / z_scale)
            op_weight = torch.clamp((opacity - float(opacity_threshold)) / max(1.0 - float(opacity_threshold), 1e-6), min=0.0, max=1.0)
            weights = z_weight * op_weight
            return torch.where(weights >= float(min_weight), weights, torch.zeros_like(weights))

    def get_bsr_height_weights(self, z_percentile=0.2, z_softness=0.04, min_weight=0.02):
        """Near-support confidence without opacity gating, used for floater suppression."""
        if self.get_xyz.numel() == 0:
            return torch.empty(0, device="cuda")
        with torch.no_grad():
            xyz = self.get_xyz.detach()
            heights = xyz[:, 2]
            if self._bsr_frame_normal.numel() == 3:
                heights = (xyz - self._bsr_frame_origin) @ self._bsr_frame_normal
            cutoff = torch.quantile(heights, float(z_percentile))
            height_range = torch.quantile(heights, 0.95) - torch.quantile(heights, 0.05)
            scale = torch.clamp(height_range * float(z_softness), min=1e-6)
            weights = torch.sigmoid((cutoff - heights) / scale)
            return torch.where(weights >= float(min_weight), weights, torch.zeros_like(weights))

    def get_bsr_mask(self, z_percentile=0.2, opacity_threshold=0.05):
        return self.get_bsr_soft_weights(z_percentile, opacity_threshold) > 0.0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        self.save_bernstein_surface(os.path.join(os.path.dirname(path), "bernstein_surface.pt"))

    def save_bernstein_surface(self, path):
        if not self.has_bernstein_surface:
            return
        torch.save(
            {"control_points": self._bernstein_control_points.detach().cpu(),
             "metadata": self.get_bsr_metadata()},
            path,
        )

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, gradients=None):
        """Accumulate image-space gradients, optionally supplied by a selected loss branch."""
        grads = viewspace_point_tensor.grad if gradients is None else gradients
        self.xyz_gradient_accum[update_filter] += torch.norm(grads[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

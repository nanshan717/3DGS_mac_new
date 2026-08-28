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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.bernstein_utils import bernstein_surface_distance_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import json
import torch.nn.functional as F
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def normalize_cli_dashes(argv):
    """Accept common Unicode dash variants pasted from rich text terminals."""
    normalized = []
    replacements = {"—": "--", "–": "--", "−": "-", "﹣": "-", "－": "-"}
    for arg in argv:
        if arg and arg[0] in replacements:
            fixed = replacements[arg[0]] + arg[1:]
            print(f"[CLI] Normalized argument {arg!r} -> {fixed!r}")
            normalized.append(fixed)
        else:
            normalized.append(arg)
    return normalized

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    write_experiment_manifest(dataset.model_path, dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if opt.use_bsr:
        print(
            "[BR-GS] Enabled with "
            f"lambda={opt.bsr_lambda_max}, warmup={opt.bsr_warmup_iters}, ramp={opt.bsr_ramp_iters}, "
            f"z_percentile={opt.bsr_z_percentile}, floater_lambda={opt.bsr_floater_lambda}"
        )
    else:
        print("[BR-GS] Disabled: this run is vanilla 3DGS. Add --use_bsr to train BR-GS.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Lbsr_for_log = 0.0

    def get_bsr_weight(iteration):
        if not opt.use_bsr:
            return 0.0
        if iteration < opt.bsr_warmup_iters:
            return 0.0
        if opt.bsr_ramp_iters <= 0:
            return opt.bsr_lambda_max
        ramp_progress = min(1.0, (iteration - opt.bsr_warmup_iters) / float(opt.bsr_ramp_iters))
        weight = opt.bsr_lambda_max * ramp_progress
        if opt.bsr_refine_end > 0 and iteration > opt.bsr_refine_end:
            tail = max(1, opt.iterations - opt.bsr_refine_end)
            progress = min(1.0, (iteration - opt.bsr_refine_end) / float(tail))
            weight *= (1.0 - 0.5 * progress)
        return weight

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        reconstruction_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss = reconstruction_loss

        Lbsr = torch.tensor(0.0, device="cuda")
        bsr_debug = {"num_bsr_points": 0, "mean_distance": 0.0}
        bsr_weight = get_bsr_weight(iteration)
        if bsr_weight > 0.0 and gaussians.has_bernstein_surface:
            refinement_active = opt.bsr_refine_start < 0 or iteration >= opt.bsr_refine_start
            bsr_weights = gaussians.get_bsr_soft_weights(
                opt.bsr_z_percentile,
                opt.bsr_opacity_threshold,
                opt.bsr_z_softness,
                opt.bsr_min_weight,
            )
            roi_weights = None
            if dataset.bsr_roi_dir:
                if viewpoint_cam.roi_mask is None and dataset.bsr_roi_required:
                    raise FileNotFoundError(
                        f"Missing required BR-GS ROI mask for camera {viewpoint_cam.image_name!r} "
                        f"under {dataset.bsr_roi_dir!r}"
                    )
                if viewpoint_cam.roi_mask is not None:
                    roi_weights = project_roi_weights(
                        gaussians.get_xyz, viewpoint_cam, viewpoint_cam.roi_mask
                    )
                    bsr_weights = bsr_weights * roi_weights
            if opt.bsr_floater_distance_loss or opt.bsr_floater_roi_candidates:
                # v3.2/v3.3: ROI selects valid ground support; distance identifies floaters.
                floater_weights = torch.ones_like(bsr_weights)
            else:
                floater_weights = gaussians.get_bsr_height_weights(
                    opt.bsr_z_percentile, opt.bsr_z_softness, opt.bsr_min_weight
                )
            if roi_weights is not None:
                floater_weights = floater_weights * roi_weights
            if opt.bsr_floater_visible_only:
                # Rasterizer variants expose either a full boolean mask or the [M, 1]
                # indices returned by ``(radii > 0).nonzero()``. Normalize to [N].
                floater_weights = floater_weights.reshape(-1)
                raw_visibility = visibility_filter.detach()
                if raw_visibility.dtype == torch.bool and raw_visibility.numel() == floater_weights.numel():
                    visible_weights = raw_visibility.reshape(-1).to(floater_weights.dtype)
                else:
                    visible_weights = torch.zeros_like(floater_weights)
                    visible_indices = raw_visibility.reshape(-1).long()
                    valid_indices = visible_indices[
                        (visible_indices >= 0) & (visible_indices < floater_weights.numel())
                    ]
                    visible_weights[valid_indices] = 1.0
                floater_weights = floater_weights * visible_weights
            Lbsr, bsr_debug = bernstein_surface_distance_loss(
                gaussians.get_xyz,
                gaussians.get_bernstein_control_points,
                point_weights=bsr_weights,
                opacities=gaussians.get_opacity,
                samples_u=opt.bsr_surface_samples_u,
                samples_v=opt.bsr_surface_samples_v,
                max_points=opt.bsr_max_points,
                robust_delta=opt.bsr_robust_delta,
                density_k=opt.bsr_density_k,
                density_blend=opt.bsr_density_blend,
                floater_lambda=opt.bsr_floater_lambda if refinement_active else 0.0,
                floater_margin=opt.bsr_floater_margin,
                support_scale=gaussians._bsr_support_scale.item() if opt.bsr_normalize_distance else 1.0,
                coverage_lambda=opt.bsr_coverage_lambda if refinement_active else 0.0,
                control_smoothness_lambda=opt.bsr_control_smoothness_lambda,
                patch_continuity_lambda=opt.bsr_patch_continuity_lambda,
                spatial_sampling=opt.bsr_spatial_sampling,
                frame_origin=gaussians._bsr_frame_origin,
                frame_u=gaussians._bsr_frame_u,
                frame_v=gaussians._bsr_frame_v,
                floater_points=gaussians.get_xyz,
                floater_weights=floater_weights,
                floater_opacities=gaussians.get_opacity,
                surface_deadzone=opt.bsr_surface_deadzone,
                surface_one_sided=opt.bsr_surface_one_sided,
                surface_normal=gaussians._bsr_frame_normal,
                floater_distance_loss=opt.bsr_floater_distance_loss,
                floater_opacity_min=opt.bsr_floater_opacity_min,
                surface_loss_lambda=opt.bsr_surface_loss_lambda,
            )
            loss = loss + bsr_weight * Lbsr

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        reconstruction_viewspace_grad = None
        if opt.bsr_isolate_densification and iteration < opt.densify_until_iter:
            reconstruction_viewspace_grad = torch.autograd.grad(
                reconstruction_loss, viewspace_point_tensor, retain_graph=True, allow_unused=True
            )[0]
            if reconstruction_viewspace_grad is not None:
                reconstruction_viewspace_grad = reconstruction_viewspace_grad.detach()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_Lbsr_for_log = 0.4 * Lbsr.item() + 0.6 * ema_Lbsr_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "BSR": f"{ema_Lbsr_for_log:.{7}f}",
                    "Surf": f"{bsr_debug.get('surface_loss', 0.0):.{5}f}",
                    "Float": f"{bsr_debug.get('floater_loss', 0.0):.{5}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if tb_writer:
                tb_writer.add_scalar("train_loss_patches/bsr_loss", Lbsr.item(), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_weight", bsr_weight, iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_num_points", bsr_debug["num_bsr_points"], iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_mean_distance", bsr_debug["mean_distance"], iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_surface_loss", bsr_debug.get("surface_loss", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_floater_loss", bsr_debug.get("floater_loss", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_mean_weight", bsr_debug.get("mean_weight", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_mean_density_weight", bsr_debug.get("mean_density_weight", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_coverage_loss", bsr_debug.get("coverage_loss", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_control_smoothness_loss", bsr_debug.get("control_smoothness_loss", 0.0), iteration)
                tb_writer.add_scalar("train_loss_patches/bsr_patch_continuity_loss", bsr_debug.get("patch_continuity_loss", 0.0), iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, gradients=reconstruction_viewspace_grad
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if gaussians.bernstein_optimizer is not None:
                    gaussians.bernstein_optimizer.step()
                    gaussians.bernstein_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def project_roi_weights(points, camera, roi_mask):
    """Project Gaussian centers into a binary per-view ROI mask without backpropagating through selection."""
    with torch.no_grad():
        homogeneous = torch.cat((points.detach(), torch.ones_like(points[:, :1])), dim=1)
        clip = homogeneous @ camera.full_proj_transform
        w = clip[:, 3]
        ndc = clip[:, :2] / w[:, None].clamp_min(1e-8)
        grid = ndc.view(1, -1, 1, 2)
        sampled = F.grid_sample(
            roi_mask.to(device=points.device, dtype=torch.float32).unsqueeze(0), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).view(-1)
        valid = (w > 0) & (ndc.abs() <= 1).all(dim=1)
        return sampled * valid.to(sampled.dtype)


def write_experiment_manifest(model_path, dataset, opt, pipe):
    """Persist resolved parameters separately from cfg_args for reproducible ablations."""
    manifest = {
        "schema": "brgs-experiment-v1",
        "dataset": vars(dataset),
        "optimization": vars(opt),
        "pipeline": vars(pipe),
        "argv": sys.argv,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with open(os.path.join(model_path, "experiment_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def apply_bsr_v3_preset(args):
    """Recommended target-domain configuration; explicit values are saved in the manifest."""
    if not args.bsr_v3:
        return
    explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}

    def preset(name, value):
        if f"--{name}" not in explicit:
            setattr(args, name, value)

    args.use_bsr = True
    preset("bsr_axis_mode", "auto")
    preset("bsr_num_patches_u", 2)
    preset("bsr_num_patches_v", 2)
    args.bsr_height_only = True
    args.bsr_normalize_distance = True
    preset("bsr_robust_delta", 0.02)
    preset("bsr_density_blend", 0.25)
    preset("bsr_coverage_lambda", 0.10)
    preset("bsr_control_smoothness_lambda", 0.01)
    preset("bsr_patch_continuity_lambda", 1.0)
    preset("bsr_floater_lambda", 0.05)
    args.bsr_spatial_sampling = True
    preset("bsr_warmup_iters", 5000)
    preset("bsr_ramp_iters", 10000)
    preset("bsr_refine_start", 10000)
    preset("bsr_refine_end", 25000)
    print("[BR-GS v3] Applied recommended target-domain preset; resolved values will be saved.")


def apply_bsr_v31_preset(args):
    """ROI-aware conservative preset; v3 remains unchanged for reproducibility."""
    if not args.bsr_v31:
        return
    if args.bsr_v3:
        raise ValueError("Choose only one preset: --bsr_v3 or --bsr_v31")
    explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}

    def preset(name, value):
        if f"--{name}" not in explicit:
            setattr(args, name, value)

    args.use_bsr = True
    preset("bsr_roi_dir", "bsr_masks")
    args.bsr_roi_required = True
    preset("bsr_axis_mode", "auto")
    preset("bsr_num_patches_u", 2)
    preset("bsr_num_patches_v", 2)
    args.bsr_height_only = True
    args.bsr_normalize_distance = True
    preset("bsr_lambda_max", 0.003)
    preset("bsr_robust_delta", 0.02)
    preset("bsr_density_blend", 0.10)
    preset("bsr_coverage_lambda", 0.02)
    preset("bsr_control_smoothness_lambda", 0.005)
    preset("bsr_patch_continuity_lambda", 0.25)
    preset("bsr_floater_lambda", 0.05)
    args.bsr_spatial_sampling = True
    preset("bsr_warmup_iters", 5000)
    preset("bsr_ramp_iters", 10000)
    preset("bsr_refine_start", 10000)
    preset("bsr_refine_end", 20000)
    print("[BR-GS v3.1] Applied ROI-aware conservative preset; ROI masks are required.")


def apply_bsr_v32_preset(args):
    """Geometry-safe ROI preset; older presets remain unchanged for reproducibility."""
    if not args.bsr_v32:
        return
    if args.bsr_v3 or args.bsr_v31:
        raise ValueError("Choose only one preset: --bsr_v3, --bsr_v31, or --bsr_v32")
    explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}

    def preset(name, value):
        if f"--{name}" not in explicit:
            setattr(args, name, value)

    args.use_bsr = True
    preset("bsr_roi_dir", "bsr_masks")
    args.bsr_roi_required = True
    preset("bsr_axis_mode", "auto")
    preset("bsr_num_patches_u", 2)
    preset("bsr_num_patches_v", 2)
    args.bsr_height_only = True
    args.bsr_normalize_distance = True
    preset("bsr_lambda_max", 0.002)
    preset("bsr_robust_delta", 0.02)
    preset("bsr_density_blend", 0.05)
    preset("bsr_coverage_lambda", 0.01)
    preset("bsr_control_smoothness_lambda", 0.003)
    preset("bsr_patch_continuity_lambda", 0.15)
    preset("bsr_floater_lambda", 0.02)
    preset("bsr_floater_margin", 0.02)
    preset("bsr_surface_deadzone", 0.003)
    preset("bsr_floater_opacity_min", 0.05)
    args.bsr_surface_one_sided = True
    args.bsr_floater_distance_loss = True
    args.bsr_isolate_densification = True
    args.bsr_spatial_sampling = True
    preset("bsr_warmup_iters", 5000)
    preset("bsr_ramp_iters", 10000)
    preset("bsr_refine_start", 7000)
    preset("bsr_refine_end", 20000)
    print(
        "[BR-GS v3.2] Applied geometry-safe ROI preset; reconstruction-only gradients drive densification."
    )


def apply_bsr_v33_preset(args):
    """Opacity-only floater suppression with ROI and reconstruction-safe densification."""
    if not args.bsr_v33:
        return
    if args.bsr_v3 or args.bsr_v31 or args.bsr_v32:
        raise ValueError("Choose only one preset: --bsr_v3, --bsr_v31, --bsr_v32, or --bsr_v33")
    explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}

    def preset(name, value):
        if f"--{name}" not in explicit:
            setattr(args, name, value)

    args.use_bsr = True
    preset("bsr_roi_dir", "bsr_masks")
    args.bsr_roi_required = True
    preset("bsr_axis_mode", "auto")
    preset("bsr_num_patches_u", 2)
    preset("bsr_num_patches_v", 2)
    args.bsr_height_only = True
    args.bsr_normalize_distance = True
    preset("bsr_lambda_max", 0.002)
    preset("bsr_surface_loss_lambda", 0.0)
    preset("bsr_density_blend", 0.0)
    preset("bsr_coverage_lambda", 0.0)
    preset("bsr_control_smoothness_lambda", 0.003)
    preset("bsr_patch_continuity_lambda", 0.10)
    preset("bsr_floater_lambda", 0.03)
    preset("bsr_floater_margin", 0.02)
    preset("bsr_floater_opacity_min", 0.05)
    args.bsr_floater_roi_candidates = True
    args.bsr_floater_visible_only = True
    args.bsr_floater_distance_loss = False
    args.bsr_isolate_densification = True
    args.bsr_spatial_sampling = True
    preset("bsr_warmup_iters", 5000)
    preset("bsr_ramp_iters", 5000)
    preset("bsr_refine_start", 7000)
    preset("bsr_refine_end", 20000)
    print(
        "[BR-GS v3.3] Applied ROI-aware opacity-only floater suppression; xyz gradients are disabled."
    )

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(normalize_cli_dashes(sys.argv[1:]))
    apply_bsr_v3_preset(args)
    apply_bsr_v31_preset(args)
    apply_bsr_v32_preset(args)
    apply_bsr_v33_preset(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed, args.deterministic)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    dataset_args.seed = args.seed
    dataset_args.deterministic = args.deterministic
    training(dataset_args, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

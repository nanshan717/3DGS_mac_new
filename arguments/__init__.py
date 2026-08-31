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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.bsr_roi_dir = ""
        self.bsr_roi_required = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.use_bsr = False
        self.bsr_lambda_max = 0.01
        self.bsr_warmup_iters = 1000
        self.bsr_ramp_iters = 3000
        self.bsr_control_points_u = 5
        self.bsr_control_points_v = 5
        self.bsr_surface_samples_u = 32
        self.bsr_surface_samples_v = 32
        self.bsr_max_points = 4096
        self.bsr_z_percentile = 0.2
        self.bsr_opacity_threshold = 0.05
        self.bsr_control_lr = 0.001
        self.bsr_z_softness = 0.01
        self.bsr_min_weight = 0.02
        self.bsr_density_k = 8
        self.bsr_density_blend = 0.0
        self.bsr_robust_delta = 0.0
        self.bsr_floater_lambda = 0.0
        self.bsr_floater_margin = 0.0
        # BR-GS v3 options. Defaults keep legacy BR-GS behavior reproducible;
        # use --bsr_v3 to enable the recommended final configuration in train.py.
        self.bsr_v3 = False
        self.bsr_v31 = False
        self.bsr_v32 = False
        self.bsr_v33 = False
        self.bsr_v34 = False
        self.bsr_v35 = False
        self.bsr_axis_mode = "z"
        self.bsr_num_patches_u = 1
        self.bsr_num_patches_v = 1
        self.bsr_height_only = False
        self.bsr_normalize_distance = False
        self.bsr_coverage_lambda = 0.0
        self.bsr_control_smoothness_lambda = 0.0
        self.bsr_patch_continuity_lambda = 0.0
        self.bsr_spatial_sampling = False
        self.bsr_refine_start = -1
        self.bsr_refine_end = -1
        # v3.2 safety controls. Disabled by default so earlier presets remain exact.
        self.bsr_surface_deadzone = 0.0
        self.bsr_surface_one_sided = False
        self.bsr_floater_distance_loss = False
        self.bsr_floater_opacity_min = 0.0
        self.bsr_isolate_densification = False
        self.bsr_surface_loss_lambda = 1.0
        self.bsr_floater_roi_candidates = False
        self.bsr_floater_visible_only = False
        # v3.4 auditable one-shot geometry pruning. Disabled by default.
        self.bsr_prune_iter = -1
        self.bsr_prune_opacity_max = 0.05
        self.bsr_prune_distance_min = 0.02
        self.bsr_prune_roi_consensus = 0.60
        self.bsr_prune_roi_views = 8
        self.bsr_prune_min_valid_views = 2
        self.bsr_prune_max_fraction = 0.05
        self.bsr_disable_after_prune = False
        # v3.5 bounded, reconstruction-guarded post-pruning recovery.
        self.bsr_recovery_lambda = 0.0002
        self.bsr_recovery_opacity_min = 0.05
        self.bsr_recovery_opacity_max = 0.20
        self.bsr_recovery_max_displacement = 0.005
        self.bsr_recovery_loss_tolerance = 0.03
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

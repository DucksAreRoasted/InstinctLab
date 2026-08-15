# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""Booster K1 adapter for the official Hiking in the Wild parkour task."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.utils import configclass

from instinctlab.assets.booster_k1 import (
    BOOSTER_K1_CFG,
    K1_ACTION_SCALE,
    K1_JOINT_NAMES,
    K1_LINK_NAMES,
    K1_SYMMETRY_JOINT_MAPPING,
    K1_SYMMETRY_JOINT_SIGNS,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
    G1ParkourRoughEnvCfg,
    G1ParkourRoughEnvCfg_PLAY,
)
from instinctlab.utils.noise import RangeBasedGaussianNoiseCfg

K1_PARKOUR_LINKS = [
    "Trunk",
    "Head_2",
    "Left_Arm_3",
    "Right_Arm_3",
    "left_hand_link",
    "right_hand_link",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Shank",
    "Right_Shank",
    "left_foot_link",
    "right_foot_link",
]
K1_PARKOUR_LINK_SYMMETRY = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]

# Live ``/boostercamera/head/depth/camera_info`` calibration.  The simulated
# ray image stays small for training throughput, while its aperture preserves
# the calibrated full-sensor field of view.
K1_DEPTH_WIDTH_PX = 544
K1_DEPTH_HEIGHT_PX = 448
K1_DEPTH_FOCAL_LENGTH_PX = 210.77337743177728
K1_DEPTH_PRINCIPAL_X_PX = 241.36559391021729
K1_DEPTH_PRINCIPAL_Y_PX = 217.1773950913373
K1_DEPTH_UPDATE_PERIOD_S = 0.05
K1_DEPTH_HISTORY_LENGTH = 16
K1_DEPTH_HISTORY_SKIP_FRAMES = 2
K1_MOTION_PACKAGE_DIR = Path(__file__).resolve().parents[7] / "parkour_motion_reference" / "booster_k1"

K1_CFG = copy.deepcopy(BOOSTER_K1_CFG)
K1_CFG.spawn.merge_fixed_joints = True
K1_CFG.init_state.pos = (0.0, 0.0, 0.57)


@configclass
class K1AmassMotionCfg(AmassMotionCfgBase):
    """K1 motions retargeted with GMR's ``smplx_to_k1.json`` constraints."""

    path = os.path.expanduser(os.environ.get("INSTINCTLAB_K1_MOTION_DIR", str(K1_MOTION_PACKAGE_DIR)))
    retargetting_func = None
    filtered_motion_selection_filepath = os.path.expanduser(
        os.environ.get("INSTINCTLAB_K1_MOTION_SELECTION", str(Path(path) / "motions.yaml"))
    )
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


K1_MOTION_REFERENCE_CFG = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Trunk",
    robot_model_path=K1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/Trunk",
    symmetric_augmentation_link_mapping=K1_PARKOUR_LINK_SYMMETRY,
    symmetric_augmentation_joint_mapping=K1_SYMMETRY_JOINT_MAPPING,
    symmetric_augmentation_joint_reverse_buf=K1_SYMMETRY_JOINT_SIGNS,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": K1AmassMotionCfg()},
    link_of_interests=K1_PARKOUR_LINKS,
    mp_split_method="Even",
)


class K1ParkourConfigMixin:
    """Apply K1 morphology at the robot-specific seam of the parkour task."""

    def apply_k1_config(self):
        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.motion_reference = copy.deepcopy(K1_MOTION_REFERENCE_CFG)

        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}

        # Measured on the production K1 from the live ROS TF transform
        # ``head_pitch_link -> head_color_optical_frame``.  ``Head_2`` is the
        # corresponding link in the simulation asset.  Keeping the camera on
        # the articulated head preserves the physical effect of head motion.
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/Head_2"
        self.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(K1_LINK_NAMES)]
        self.scene.camera.offset.pos = (0.05663342989, 0.0462427773, 0.0962657193)
        self.scene.camera.offset.rot = (
            0.5132977331550982,
            -0.5083061254903114,
            0.4877471740434324,
            -0.49015611200872644,
        )
        self.scene.camera.offset.convention = "ros"
        self.scene.camera.pattern_cfg.horizontal_aperture = K1_DEPTH_WIDTH_PX / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.pattern_cfg.vertical_aperture = K1_DEPTH_HEIGHT_PX / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.pattern_cfg.horizontal_aperture_offset = (
            K1_DEPTH_PRINCIPAL_X_PX - K1_DEPTH_WIDTH_PX / 2
        ) / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.pattern_cfg.vertical_aperture_offset = (
            K1_DEPTH_PRINCIPAL_Y_PX - K1_DEPTH_HEIGHT_PX / 2
        ) / K1_DEPTH_FOCAL_LENGTH_PX
        self.scene.camera.update_period = K1_DEPTH_UPDATE_PERIOD_S
        # Force sensor refreshes on physics ticks instead of waiting for the
        # 50 Hz policy read.  Otherwise a 0.05 s lazy sensor updates at 0.06 s.
        self.scene.camera.history_length = 1
        # The 64x36 ray image is the downsampled full sensor.  Taking its lower
        # central half yields the policy's 32x18 terrain-focused observation.
        noise_pipeline = self.scene.camera.noise_pipeline
        noise_pipeline["crop_and_resize"].crop_region = (18, 0, 16, 16)
        self.scene.camera.noise_pipeline = {
            "crop_and_resize": noise_pipeline["crop_and_resize"],
            "gaussian_blur": noise_pipeline["gaussian_blur"],
            # A 20-frame static sample on the robot showed a 12 mm median and
            # 62 mm p90 inter-frame delta inside the policy's 2.5 m range.
            "sensor_noise": RangeBasedGaussianNoiseCfg(
                min_value=0.1,
                max_value=2.5,
                noise_std=0.02,
            ),
            "depth_normalization": noise_pipeline["depth_normalization"],
        }
        self.scene.camera.data_histories["distance_to_image_plane_noised"] = K1_DEPTH_HISTORY_LENGTH
        for observation_group in (self.observations.policy, self.observations.critic):
            observation_group.depth_image.params["history_skip_frames"] = K1_DEPTH_HISTORY_SKIP_FRAMES

        self.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/left_foot_link"
        self.scene.left_height_scanner.offset.pos = (0.014, 0.0, 20.0)
        self.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/right_foot_link"
        self.scene.right_height_scanner.offset.pos = (0.014, 0.0, 20.0)

        self.scene.leg_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_foot_link"
        foot_points = self.scene.leg_volume_points.points_generator
        foot_points.x_min = -0.066
        foot_points.x_max = 0.094
        foot_points.y_min = -0.035
        foot_points.y_max = 0.035
        foot_points.z_min = -0.024
        foot_points.z_max = 0.0

        rewards = self.rewards.rewards
        for term_name in ("feet_air_time",):
            getattr(rewards, term_name).params["sensor_cfg"].body_names = ".*_foot_link"
        for term_name in ("feet_slide", "feet_flat_ori"):
            term = getattr(rewards, term_name)
            term.params["sensor_cfg"].body_names = ".*_foot_link"
            term.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["contact_sensor_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.feet_at_plane.params["height_offset"] = 0.024
        rewards.feet_close_xy.params["asset_cfg"].body_names = ".*_foot_link"
        rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [
            ".*_Hip_Yaw",
            ".*_Hip_Roll",
        ]
        for term_name in ("dof_torques_l2", "energy"):
            getattr(rewards, term_name).params["asset_cfg"].joint_names = [".*_(Hip|Knee|Ankle)_.*"]
        rewards.freeze_upper_body.params["asset_cfg"].joint_names = [
            ".*Head.*",
            ".*_Shoulder_.*",
            ".*_Elbow_.*",
        ]
        rewards.pelvis_orientation_l2.params["asset_cfg"].body_names = "Trunk"
        rewards.undesired_contacts.params["sensor_cfg"].body_names = "(?!.*_foot_link).*"

        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"
        if self.terminations.root_height is not None:
            self.terminations.root_height.params["minimum_height"] = 0.32


@configclass
class K1ParkourEnvCfg(G1ParkourRoughEnvCfg, K1ParkourConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_k1_config()


@configclass
class K1ParkourEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY, K1ParkourConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_k1_config()

        # Checkpoint inspection needs one representative terrain row, not the
        # larger G1 debug scene. Keep all ten terrain columns while avoiding
        # unused replicated meshes and debug geometry in video playback.
        self.scene.num_envs = 1
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.leg_volume_points.debug_vis = False
        self.commands.base_velocity.debug_vis = False

# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""Booster K1 adapter for the official Hiking in the Wild parkour task."""

from __future__ import annotations

import copy
import math
import os

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
from isaaclab.utils import configclass


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

K1_CFG = copy.deepcopy(BOOSTER_K1_CFG)
K1_CFG.spawn.merge_fixed_joints = True
K1_CFG.init_state.pos = (0.0, 0.0, 0.57)


@configclass
class K1AmassMotionCfg(AmassMotionCfgBase):
    """K1 motions retargeted with GMR's ``smplx_to_k1.json`` constraints."""

    path = os.path.expanduser(os.environ.get("INSTINCTLAB_K1_MOTION_DIR", "~/Datasets/K1"))
    retargetting_func = None
    filtered_motion_selection_filepath = os.path.expanduser(
        os.environ.get("INSTINCTLAB_K1_MOTION_SELECTION", "~/Datasets/K1/parkour_motion_without_run.yaml")
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

        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"
        self.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(K1_LINK_NAMES)]
        # Nominal torso-to-ZED pose derived from the K1 head chain. Measure the
        # production camera transform before real-robot deployment.
        self.scene.camera.offset.pos = (0.08, 0.0, 0.34)
        self.scene.camera.offset.rot = (
            math.cos(math.radians(48.0) / 2),
            0.0,
            math.sin(math.radians(48.0) / 2),
            0.0,
        )

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

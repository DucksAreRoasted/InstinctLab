# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""Booster K1 robot assets for Isaac Lab."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from instinctlab.actuators.booster import (
    BoosterDelayedPDActuatorCfg,
    BoosterJointE4310Cfg,
    BoosterJointE4315Cfg,
    BoosterJointE6408Cfg,
    BoosterJointE6416Cfg,
    BoosterJointHT4438Cfg,
    BoosterJointR14Cfg,
    BoosterK1AnkleCfg,
)

_RESOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "booster_k1")

K1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

K1_LINK_NAMES = [
    "Trunk",
    "Head_1",
    "Head_2",
    "Left_Arm_1",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_1",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Shank",
    "Left_Ankle_Cross",
    "left_foot_link",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Shank",
    "Right_Ankle_Cross",
    "right_foot_link",
]

K1_SYMMETRY_JOINT_MAPPING = [
    0,
    1,
    6,
    7,
    8,
    9,
    2,
    3,
    4,
    5,
    16,
    17,
    18,
    19,
    20,
    21,
    10,
    11,
    12,
    13,
    14,
    15,
]
K1_SYMMETRY_JOINT_SIGNS = [
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
]
K1_SYMMETRY_LINK_MAPPING = [
    0,
    1,
    2,
    7,
    8,
    9,
    10,
    3,
    4,
    5,
    6,
    17,
    18,
    19,
    20,
    21,
    22,
    11,
    12,
    13,
    14,
    15,
    16,
]


def _make_k1_actuators() -> dict[str, BoosterDelayedPDActuatorCfg]:
    return {
        "legs": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
            booster_joint_cfgs={
                ".*_Hip_Pitch": BoosterJointE6408Cfg(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Roll": BoosterJointE4315Cfg(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Yaw": BoosterJointE4310Cfg(natural_freq=4.0, damping_ratio=1.5),
                ".*_Knee_Pitch": BoosterJointE6416Cfg(natural_freq=4.0, damping_ratio=1.0),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": BoosterK1AnkleCfg(natural_freq=4.0, damping_ratio=1.5),
                ".*_Ankle_Roll": BoosterK1AnkleCfg(natural_freq=4.0, damping_ratio=1.5),
            },
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=BoosterJointR14Cfg(),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=BoosterJointHT4438Cfg(),
        ),
    }


BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_RESOURCE_DIR, "K1_22dof.urdf"),
        fix_base=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={"Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=_make_k1_actuators(),
)


def _action_scale(cfg: ArticulationCfg) -> dict[str, float]:
    scales = {}
    for actuator in cfg.actuators.values():
        efforts = actuator.effort_limit
        stiffness = actuator.stiffness
        if not isinstance(efforts, dict):
            efforts = {name: efforts for name in actuator.joint_names_expr}
        if not isinstance(stiffness, dict):
            stiffness = {name: stiffness for name in actuator.joint_names_expr}
        for name in actuator.joint_names_expr:
            if efforts[name] is not None and stiffness[name]:
                scales[name] = 0.25 * efforts[name] / stiffness[name]
    return scales


K1_ACTION_SCALE = _action_scale(BOOSTER_K1_CFG)

BOOSTER_K1_LOCOMOTION_CFG = BOOSTER_K1_CFG.copy()
BOOSTER_K1_LOCOMOTION_CFG.spawn.asset_path = os.path.join(_RESOURCE_DIR, "K1_locomotion.urdf")
BOOSTER_K1_LOCOMOTION_CFG.init_state.joint_pos = {}
BOOSTER_K1_LOCOMOTION_CFG.actuators.pop("arms")
BOOSTER_K1_LOCOMOTION_CFG.actuators.pop("head")

K1_LOCOMOTION_ACTION_SCALE = _action_scale(BOOSTER_K1_LOCOMOTION_CFG)

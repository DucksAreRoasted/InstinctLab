# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""Booster motor models used by the K1 robot configurations."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with Booster's piecewise-linear torque-speed limit."""

    cfg: "BoosterDelayedPDActuatorCfg"

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.minimum(
            torch.clamp(self.knee_point_velocity, min=0.0), self.velocity_limit
        )
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._torque_speed_denominator = (self.velocity_limit - self.knee_point_velocity).clamp(min=1.0e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_speed = self._joint_vel.abs()
        linear_limit = self.effort_limit * (self.velocity_limit - joint_speed) / self._torque_speed_denominator
        speed_dependent_limit = linear_limit.clamp(min=0.0)
        speed_dependent_limit = torch.minimum(speed_dependent_limit, self.effort_limit)
        speed_dependent_limit = torch.where(
            torch.isfinite(self.velocity_limit), speed_dependent_limit, self.effort_limit
        )
        speed_dependent_limit = torch.where(
            self.velocity_limit > 0.0, speed_dependent_limit, torch.zeros_like(speed_dependent_limit)
        )
        return torch.clip(effort, min=-speed_dependent_limit, max=speed_dependent_limit)


@configclass
class BoosterJointCfg:
    """Physical parameters for one Booster motor model."""

    joint_model_name: str = MISSING
    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING
    stiffness: float | None = None
    damping: float | None = None
    natural_freq: float = 10.0
    damping_ratio: float = 2.0

    def __post_init__(self):
        angular_frequency = 2.0 * torch.pi * self.natural_freq
        if self.stiffness is None:
            self.stiffness = float(self.armature * angular_frequency**2)
        if self.damping is None:
            self.damping = float(2.0 * self.damping_ratio * self.armature * angular_frequency)


@configclass
class BoosterK1AnkleCfg(BoosterJointCfg):
    """K1 parallel ankle parameters derived from a serial E4310 motor."""

    joint_model_name: str = "K1Ankle(E4310)"
    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0565056


@configclass
class BoosterJointE6408Cfg(BoosterJointCfg):
    joint_model_name: str = "E6408"
    effort_limit: float = 68.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 1.88
    armature: float = 0.0478125


@configclass
class BoosterJointE4315Cfg(BoosterJointCfg):
    joint_model_name: str = "E4315"
    effort_limit: float = 76.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.62
    armature: float = 0.0339552


@configclass
class BoosterJointE4310Cfg(BoosterJointCfg):
    joint_model_name: str = "E4310"
    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


@configclass
class BoosterJointE6416Cfg(BoosterJointCfg):
    joint_model_name: str = "E6416"
    effort_limit: float = 112.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.09
    armature: float = 0.095625


@configclass
class BoosterJointR14Cfg(BoosterJointCfg):
    joint_model_name: str = "R14"
    effort_limit: float = 14.0
    velocity_limit: float = 33.51
    knee_point_velocity: float = 5.24
    armature: float = 0.001


@configclass
class BoosterJointHT4438Cfg(BoosterJointCfg):
    joint_model_name: str = "HT4438"
    effort_limit: float = 6.0
    velocity_limit: float = 7.85
    # Preserve Booster's published calibration. The actuator clamps this value
    # to velocity_limit, so HT4438 keeps full torque throughout its valid range.
    knee_point_velocity: float = 10.47
    armature: float = 0.001


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Build an Isaac Lab actuator group from Booster motor parameters."""

    class_type: type = BoosterDelayedPDActuator
    booster_joint_cfgs: dict[str, BoosterJointCfg] | BoosterJointCfg = MISSING
    knee_point_velocity: dict[str, float] | float | None = None

    def __post_init__(self):
        if isinstance(self.booster_joint_cfgs, dict):
            joint_cfgs = self.booster_joint_cfgs
            self.effort_limit = {name: cfg.effort_limit for name, cfg in joint_cfgs.items()}
            self.velocity_limit = {name: cfg.velocity_limit for name, cfg in joint_cfgs.items()}
            self.knee_point_velocity = {name: cfg.knee_point_velocity for name, cfg in joint_cfgs.items()}
            self.armature = {name: cfg.armature for name, cfg in joint_cfgs.items()}
            self.stiffness = {name: cfg.stiffness for name, cfg in joint_cfgs.items()}
            self.damping = {name: cfg.damping for name, cfg in joint_cfgs.items()}
        else:
            joint_cfg = self.booster_joint_cfgs
            self.effort_limit = joint_cfg.effort_limit
            self.velocity_limit = joint_cfg.velocity_limit
            self.knee_point_velocity = joint_cfg.knee_point_velocity
            self.armature = joint_cfg.armature
            self.stiffness = joint_cfg.stiffness
            self.damping = joint_cfg.damping

        # Keep PhysX safety limits consistent with the explicit motor model.
        self.effort_limit_sim = self.effort_limit
        self.velocity_limit_sim = self.velocity_limit

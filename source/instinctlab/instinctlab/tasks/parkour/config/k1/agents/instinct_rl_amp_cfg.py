# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg, MoEPolicyCfg


@configclass
class K1MoEPolicyCfg(MoEPolicyCfg):
    """K1 parkour policy with bounded deterministic actions."""

    # The environment accepts normalized actions in [-1, 1]. Bounding the MoE
    # mean here prevents deployment-time clipping from erasing depth responses.
    mu_activation = "tanh"


@configclass
class K1ParkourPPORunnerCfg(G1ParkourPPORunnerCfg):
    """Hiking in the Wild AMP/MoE runner adapted to Booster K1."""

    experiment_name = "k1_parkour"
    policy = K1MoEPolicyCfg()

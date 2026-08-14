# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg


@configclass
class K1ParkourPPORunnerCfg(G1ParkourPPORunnerCfg):
    """Hiking in the Wild AMP/MoE runner adapted to Booster K1."""

    experiment_name = "k1_parkour"

# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.k1"


gym.register(
    id="Instinct-Parkour-Target-Amp-K1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.k1_parkour_target_amp_cfg:K1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:K1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-K1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.k1_parkour_target_amp_cfg:K1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:K1ParkourPPORunnerCfg",
    },
)

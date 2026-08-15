# Copyright (c) 2025, BoosterRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""K1 越野跑任务的 Gym 注册表。

本包被 ``instinctlab.tasks`` 的 ``import_packages`` 递归导入
（见 ``scripts/instinct_rl/train.py`` 中的 ``import instinctlab.tasks``），
导入即执行下面的 ``gym.register``，把任务 id 注册进 Gym 环境表。

训练时通过 ``train.py --task <任务id>`` 启动，Hydra 的
``hydra_task_config`` 会从注册表解析出两个入口配置：
- ``env_cfg_entry_point``：环境配置类（本目录的 K1 适配层）
- ``instinct_rl_cfg_entry_point``：训练算法/网络配置类（agents 子包）
"""

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.k1"


# 训练任务：K1 越野跑（Hiking in the Wild）AMP 训练
gym.register(
    id="Instinct-Parkour-Target-Amp-K1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    # 关闭 gym 的环境检查器（Isaac Lab 环境不需要）
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.k1_parkour_target_amp_cfg:K1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:K1ParkourPPORunnerCfg",
    },
)


# 演示/回放任务：单环境、精简地形，用于加载 checkpoint 播放
gym.register(
    id="Instinct-Parkour-Target-Amp-K1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.k1_parkour_target_amp_cfg:K1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:K1ParkourPPORunnerCfg",
    },
)

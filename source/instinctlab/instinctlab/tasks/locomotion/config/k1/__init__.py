import gymnasium as gym

from . import agents
from .flat_env_cfg import K1FlatEnvCfg, K1FlatEnvCfg_PLAY


gym.register(
    id="Instinct-Locomotion-Flat-K1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1FlatEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:K1FlatPPORunnerCfg",
    },
)
gym.register(
    id="Instinct-Locomotion-Flat-K1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1FlatEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:K1FlatPPORunnerCfg",
    },
)

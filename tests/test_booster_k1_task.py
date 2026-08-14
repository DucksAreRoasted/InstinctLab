import importlib.util

import gymnasium as gym
import pytest
import torch

try:
    _ISAAC_ACTUATORS_SPEC = importlib.util.find_spec("isaaclab.actuators")
except ModuleNotFoundError:
    _ISAAC_ACTUATORS_SPEC = None

if _ISAAC_ACTUATORS_SPEC is None:
    pytest.skip("requires pytest to run inside an Isaac Lab AppLauncher process", allow_module_level=True)

from instinctlab.tasks.locomotion.config.k1.flat_env_cfg import K1FlatEnvCfg, K1FlatEnvCfg_PLAY

if importlib.util.find_spec("instinct_rl") is not None:
    from instinctlab.tasks.locomotion.config.k1.agents.instinct_rl_ppo_cfg import K1FlatPPORunnerCfg
else:
    K1FlatPPORunnerCfg = None


def test_k1_train_and_play_tasks_are_registered() -> None:
    train_spec = gym.spec("Instinct-Locomotion-Flat-K1-v0")
    play_spec = gym.spec("Instinct-Locomotion-Flat-K1-Play-v0")

    assert train_spec.kwargs["env_cfg_entry_point"] is K1FlatEnvCfg
    assert play_spec.kwargs["env_cfg_entry_point"] is K1FlatEnvCfg_PLAY
    assert train_spec.kwargs["instinct_rl_cfg_entry_point"].endswith(":K1FlatPPORunnerCfg")


def test_k1_environment_uses_locomotion_asset_and_k1_body_names() -> None:
    cfg = K1FlatEnvCfg_PLAY()

    assert cfg.scene.num_envs == 1
    assert cfg.scene.robot.spawn.asset_path.endswith("K1_locomotion.urdf")
    assert set(cfg.scene.robot.actuators) == {"legs", "feet"}
    assert cfg.actions.joint_pos.clip == {".*": (-1.0, 1.0)}
    assert cfg.rewards.feet_air_time.params["sensor_cfg"].body_names == ".*_foot_link"
    assert cfg.events.randomize_body_mass.params["asset_cfg"].body_names == ".*"
    assert cfg.run_name.startswith("K1Flat")


@pytest.mark.skipif(K1FlatPPORunnerCfg is None, reason="instinct_rl is not installed")
def test_k1_ppo_uses_booster_training_calibration() -> None:
    assert K1FlatPPORunnerCfg is not None
    cfg = K1FlatPPORunnerCfg()

    assert cfg.max_iterations == 50000
    assert cfg.policy.init_noise_std == 0.8
    assert cfg.policy.actor_hidden_dims == [512, 256, 128]
    assert cfg.policy.critic_hidden_dims == [512, 256, 128]
    assert cfg.algorithm.entropy_coef == 0.0


def test_k1_environment_can_reset_and_step() -> None:
    cfg = K1FlatEnvCfg_PLAY()
    cfg.sim.device = "cuda:0"
    env = gym.make("Instinct-Locomotion-Flat-K1-Play-v0", cfg=cfg)
    try:
        env.reset()
        action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        env.step(action)
        assert env.unwrapped.action_space.shape == (1, 12)
    finally:
        env.close()

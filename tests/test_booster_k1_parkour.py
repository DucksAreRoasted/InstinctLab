import importlib.util
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch

try:
    _ISAAC_ACTUATORS_SPEC = importlib.util.find_spec("isaaclab.actuators")
except ModuleNotFoundError:
    _ISAAC_ACTUATORS_SPEC = None

if _ISAAC_ACTUATORS_SPEC is None:
    pytest.skip("requires pytest to run inside an Isaac Lab AppLauncher process", allow_module_level=True)

from instinctlab.tasks.parkour.config.k1.k1_parkour_target_amp_cfg import (
    K1_PARKOUR_LINKS,
    K1ParkourEnvCfg,
    K1ParkourEnvCfg_PLAY,
)
from scripts.gmr.convert_k1_motion import convert_gmr_k1_pickle


def test_k1_hiking_train_and_play_tasks_are_registered() -> None:
    train_spec = gym.spec("Instinct-Parkour-Target-Amp-K1-v0")
    play_spec = gym.spec("Instinct-Parkour-Target-Amp-K1-Play-v0")

    assert train_spec.kwargs["env_cfg_entry_point"].endswith(":K1ParkourEnvCfg")
    assert play_spec.kwargs["env_cfg_entry_point"].endswith(":K1ParkourEnvCfg_PLAY")
    assert train_spec.kwargs["instinct_rl_cfg_entry_point"].endswith(":K1ParkourPPORunnerCfg")


def test_k1_hiking_config_uses_full_body_robot_and_gmr_links() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.robot.spawn.asset_path.endswith("K1_22dof.urdf")
    assert set(cfg.scene.robot.actuators) == {"legs", "feet", "arms", "head"}
    assert cfg.actions.joint_pos.joint_names == [".*"]
    assert cfg.actions.joint_pos.clip == {".*": (-1.0, 1.0)}
    assert cfg.scene.motion_reference.prim_path.endswith("/Robot/Trunk")
    assert cfg.scene.motion_reference.link_of_interests == K1_PARKOUR_LINKS
    assert cfg.scene.motion_reference.symmetric_augmentation_link_mapping == [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]


def test_k1_hiking_config_adapts_perception_and_safety_geometry() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.camera.prim_path.endswith("/Robot/Trunk")
    assert cfg.scene.left_height_scanner.prim_path.endswith("/Robot/left_foot_link")
    assert cfg.scene.right_height_scanner.prim_path.endswith("/Robot/right_foot_link")
    assert cfg.scene.leg_volume_points.prim_path.endswith("/Robot/.*_foot_link")
    assert cfg.scene.leg_volume_points.points_generator.x_min == pytest.approx(-0.066)
    assert cfg.scene.leg_volume_points.points_generator.x_max == pytest.approx(0.094)
    assert cfg.scene.leg_volume_points.points_generator.z_min == pytest.approx(-0.024)
    assert cfg.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"].body_names == "Trunk"
    assert cfg.terminations.base_contact.params["sensor_cfg"].body_names == "Trunk"


def test_k1_hiking_environment_can_reset_and_step_with_retargeted_motion(tmp_path) -> None:
    gmr_path = tmp_path / "neutral.pkl"
    motion_path = tmp_path / "neutral.retargeted.npz"
    selection_path = tmp_path / "motions.yaml"
    joint_pos = np.zeros((40, 22), dtype=np.float32)
    joint_pos[:, 3] = -1.3
    joint_pos[:, 7] = 1.3
    with gmr_path.open("wb") as motion_file:
        pickle.dump(
            {
                "fps": 50.0,
                "root_pos": np.tile(np.asarray([0.0, 0.0, 0.57], dtype=np.float32), (40, 1)),
                "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (40, 1)),
                "dof_pos": joint_pos,
            },
            motion_file,
        )
    convert_gmr_k1_pickle(gmr_path, motion_path)
    selection_path.write_text("selected_files:\n  - neutral.retargeted.npz\n", encoding="utf-8")

    cfg = K1ParkourEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    motion_cfg = cfg.scene.motion_reference.motion_buffers["run_walk"]
    motion_cfg.path = str(tmp_path)
    motion_cfg.filtered_motion_selection_filepath = str(selection_path)
    env = gym.make("Instinct-Parkour-Target-Amp-K1-Play-v0", cfg=cfg)
    try:
        observations, _ = env.reset()
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        assert env.unwrapped.action_space.shape == (1, 22)
        assert observations["policy"]["depth_image"].shape[0] == 1
    finally:
        env.close()

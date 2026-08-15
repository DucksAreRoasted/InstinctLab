import gymnasium as gym
import importlib.util
import numpy as np
import pickle
import torch
from pathlib import Path

import pytest

try:
    _ISAAC_ACTUATORS_SPEC = importlib.util.find_spec("isaaclab.actuators")
except ModuleNotFoundError:
    _ISAAC_ACTUATORS_SPEC = None

if _ISAAC_ACTUATORS_SPEC is None:
    pytest.skip("requires pytest to run inside an Isaac Lab AppLauncher process", allow_module_level=True)

from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import G1ParkourRoughEnvCfg_PLAY
from instinctlab.tasks.parkour.config.k1.k1_parkour_target_amp_cfg import (
    K1_PARKOUR_LINKS,
    K1AmassMotionCfg,
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


def test_k1_motion_defaults_are_portable_with_the_project() -> None:
    motion_cfg = K1AmassMotionCfg()
    package_dir = Path(__file__).resolve().parents[1] / "parkour_motion_reference" / "booster_k1"

    assert Path(motion_cfg.path) == package_dir
    assert Path(motion_cfg.filtered_motion_selection_filepath) == package_dir / "motions.yaml"


def test_k1_hiking_config_uses_full_body_robot_and_gmr_links() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.robot.spawn.asset_path.endswith("K1_22dof.urdf")
    assert set(cfg.scene.robot.actuators) == {"legs", "feet", "arms", "head"}
    assert cfg.actions.joint_pos.joint_names == [".*"]
    assert cfg.actions.joint_pos.clip == {".*": (-1.0, 1.0)}
    assert cfg.scene.motion_reference.prim_path.endswith("/Robot/Trunk")
    assert cfg.scene.motion_reference.link_of_interests == K1_PARKOUR_LINKS
    assert cfg.scene.motion_reference.symmetric_augmentation_link_mapping == [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]


def test_k1_play_scene_is_small_enough_for_checkpoint_inspection() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.num_envs == 1
    assert cfg.scene.terrain.terrain_generator.num_rows == 1
    assert cfg.scene.terrain.terrain_generator.num_cols == 10
    assert cfg.scene.leg_volume_points.debug_vis is False
    assert cfg.commands.base_velocity.debug_vis is False


def test_k1_play_scene_does_not_shrink_training_or_g1_play() -> None:
    K1ParkourEnvCfg_PLAY()
    train_cfg = K1ParkourEnvCfg()
    g1_play_cfg = G1ParkourRoughEnvCfg_PLAY()

    assert train_cfg.scene.num_envs == 4096
    assert train_cfg.scene.terrain.terrain_generator.num_rows == 10
    assert train_cfg.scene.terrain.terrain_generator.num_cols == 20
    assert g1_play_cfg.scene.num_envs == 10
    assert g1_play_cfg.scene.terrain.terrain_generator.num_rows == 4
    assert g1_play_cfg.scene.terrain.terrain_generator.num_cols == 10


def test_k1_hiking_config_adapts_perception_and_safety_geometry() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.camera.prim_path.endswith("/Robot/Head_2")
    assert cfg.scene.camera.offset.pos == pytest.approx(
        (0.05663342989, 0.0462427773, 0.0962657193)
    )
    assert cfg.scene.camera.offset.rot == pytest.approx(
        (
            0.5132977331550982,
            -0.5083061254903114,
            0.4877471740434324,
            -0.49015611200872644,
        )
    )
    assert cfg.scene.camera.offset.convention == "ros"
    assert cfg.scene.left_height_scanner.prim_path.endswith("/Robot/left_foot_link")
    assert cfg.scene.right_height_scanner.prim_path.endswith("/Robot/right_foot_link")
    assert cfg.scene.leg_volume_points.prim_path.endswith("/Robot/.*_foot_link")
    assert cfg.scene.leg_volume_points.points_generator.x_min == pytest.approx(-0.066)
    assert cfg.scene.leg_volume_points.points_generator.x_max == pytest.approx(0.094)
    assert cfg.scene.leg_volume_points.points_generator.z_min == pytest.approx(-0.024)
    assert cfg.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"].body_names == "Trunk"
    assert cfg.terminations.base_contact.params["sensor_cfg"].body_names == "Trunk"


def test_k1_depth_observation_matches_the_real_camera_sampling_contract() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    assert cfg.scene.camera.pattern_cfg.horizontal_aperture == pytest.approx(2.580971119922775)
    assert cfg.scene.camera.pattern_cfg.vertical_aperture == pytest.approx(2.1255056281716973)
    assert cfg.scene.camera.pattern_cfg.horizontal_aperture_offset == pytest.approx(-0.14534286285609482)
    assert cfg.scene.camera.pattern_cfg.vertical_aperture_offset == pytest.approx(-0.03236938645570181)
    assert cfg.scene.camera.pattern_cfg.width == 64
    assert cfg.scene.camera.pattern_cfg.height == 36
    assert cfg.scene.camera.update_period == pytest.approx(0.05)
    assert cfg.scene.camera.history_length == 1
    assert cfg.scene.camera.noise_pipeline["crop_and_resize"].crop_region == (18, 0, 16, 16)
    assert cfg.scene.camera.data_histories["distance_to_image_plane_noised"] == 16
    assert cfg.observations.policy.depth_image.params["history_skip_frames"] == 2
    assert cfg.observations.critic.depth_image.params["history_skip_frames"] == 2
    assert cfg.observations.policy.depth_image.params["num_output_frames"] == 8


def test_k1_depth_noise_matches_observed_real_camera_variation() -> None:
    cfg = K1ParkourEnvCfg_PLAY()

    pipeline = cfg.scene.camera.noise_pipeline
    assert list(pipeline) == ["crop_and_resize", "gaussian_blur", "sensor_noise", "depth_normalization"]
    assert pipeline["sensor_noise"].min_value == pytest.approx(0.1)
    assert pipeline["sensor_noise"].max_value == pytest.approx(2.5)
    assert pipeline["sensor_noise"].noise_std == pytest.approx(0.02)


def test_k1_depth_calibration_does_not_change_the_g1_camera_contract() -> None:
    K1ParkourEnvCfg_PLAY()
    g1_cfg = G1ParkourRoughEnvCfg_PLAY()

    assert g1_cfg.scene.camera.pattern_cfg.horizontal_aperture == pytest.approx(1.9829684971963653)
    assert g1_cfg.scene.camera.pattern_cfg.vertical_aperture == pytest.approx(1.1152440419261633)
    assert g1_cfg.scene.camera.update_period == pytest.approx(0.02)
    assert g1_cfg.scene.camera.data_histories["distance_to_image_plane_noised"] == 37
    assert "sensor_noise" not in g1_cfg.scene.camera.noise_pipeline


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

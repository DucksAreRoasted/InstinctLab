import importlib.util
from pathlib import Path

import pytest

try:
    _ISAAC_ACTUATORS_SPEC = importlib.util.find_spec("isaaclab.actuators")
except ModuleNotFoundError:
    _ISAAC_ACTUATORS_SPEC = None

if _ISAAC_ACTUATORS_SPEC is None:
    pytest.skip("requires pytest to run inside an Isaac Lab AppLauncher process", allow_module_level=True)

from instinctlab.assets.booster_k1 import (
    BOOSTER_K1_CFG,
    BOOSTER_K1_LOCOMOTION_CFG,
    K1_ACTION_SCALE,
    K1_JOINT_NAMES,
    K1_LINK_NAMES,
    K1_SYMMETRY_JOINT_MAPPING,
    K1_SYMMETRY_JOINT_SIGNS,
    K1_SYMMETRY_LINK_MAPPING,
)


def test_k1_configs_publish_full_body_and_locomotion_variants() -> None:
    assert Path(BOOSTER_K1_CFG.spawn.asset_path).name == "K1_22dof.urdf"
    assert Path(BOOSTER_K1_LOCOMOTION_CFG.spawn.asset_path).name == "K1_locomotion.urdf"
    assert set(BOOSTER_K1_CFG.actuators) == {"legs", "feet", "arms", "head"}
    assert set(BOOSTER_K1_LOCOMOTION_CFG.actuators) == {"legs", "feet"}
    assert BOOSTER_K1_CFG.init_state.pos == (0.0, 0.0, 0.57)


def test_k1_action_scale_uses_calibrated_motor_limits() -> None:
    assert set(K1_ACTION_SCALE) == {
        ".*_Hip_Pitch",
        ".*_Hip_Roll",
        ".*_Hip_Yaw",
        ".*_Knee_Pitch",
        ".*_Ankle_Pitch",
        ".*_Ankle_Roll",
        ".*_Shoulder_Pitch",
        ".*_Shoulder_Roll",
        ".*_Elbow_Pitch",
        ".*_Elbow_Yaw",
        ".*Head.*",
    }
    assert K1_ACTION_SCALE[".*_Hip_Pitch"] == pytest.approx(0.56290, rel=1.0e-4)
    assert K1_ACTION_SCALE[".*_Knee_Pitch"] == pytest.approx(0.46354, rel=1.0e-4)


def test_k1_symmetry_metadata_covers_every_full_body_joint() -> None:
    assert len(K1_SYMMETRY_JOINT_MAPPING) == len(K1_JOINT_NAMES)
    assert len(K1_SYMMETRY_JOINT_SIGNS) == len(K1_JOINT_NAMES)
    assert [K1_SYMMETRY_JOINT_MAPPING[index] for index in K1_SYMMETRY_JOINT_MAPPING] == list(
        range(len(K1_JOINT_NAMES))
    )
    assert len(K1_SYMMETRY_LINK_MAPPING) == len(K1_LINK_NAMES)
    assert [K1_SYMMETRY_LINK_MAPPING[index] for index in K1_SYMMETRY_LINK_MAPPING] == list(
        range(len(K1_LINK_NAMES))
    )

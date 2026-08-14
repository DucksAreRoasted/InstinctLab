import numpy as np
import yaml
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "parkour_motion_reference" / "booster_k1"
URDF_PATH = (
    PROJECT_ROOT
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "assets"
    / "resources"
    / "booster_k1"
    / "K1_22dof.urdf"
)
EXPECTED_MOTIONS = [
    "stairs_descent_001.retargeted.npz",
    "stairs_descent_002.retargeted.npz",
    "stairs_descent_003.retargeted.npz",
    "stairs_descent_004.retargeted.npz",
    "stairs_descent_005.retargeted.npz",
    "stairs_descent_006.retargeted.npz",
    "stairs_descent_007.retargeted.npz",
    "stairs_descent_008.retargeted.npz",
    "stairs_descent_009.retargeted.npz",
    "stairs_ascent_010.retargeted.npz",
    "stairs_ascent_011.retargeted.npz",
    "stairs_ascent_012.retargeted.npz",
    "stairs_ascent_013.retargeted.npz",
    "stairs_ascent_014.retargeted.npz",
    "stairs_ascent_015.retargeted.npz",
    "stairs_ascent_016.retargeted.npz",
    "stairs_ascent_017.retargeted.npz",
    "grounded_locomotion_018.retargeted.npz",
    "grounded_locomotion_019.retargeted.npz",
]
EXPECTED_JOINTS = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def _velocity_limit(joint_name: str) -> float:
    if "Head" in joint_name:
        return 7.85
    if "Shoulder" in joint_name or "Elbow" in joint_name:
        return 33.51
    if "Hip_Pitch" in joint_name:
        return 14.66
    if "Hip_Roll" in joint_name or "Knee_Pitch" in joint_name:
        return 12.57
    return 17.59


def test_packaged_k1_motions_are_ready_for_the_official_loader() -> None:
    selection = yaml.safe_load((PACKAGE_DIR / "motions.yaml").read_text(encoding="utf-8"))
    assert selection == {"selected_files": EXPECTED_MOTIONS}

    urdf_root = ElementTree.parse(URDF_PATH).getroot()
    joint_limits = {
        joint.attrib["name"]: (
            float(joint.find("limit").attrib["lower"]),
            float(joint.find("limit").attrib["upper"]),
        )
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous"}
    }

    for motion_name in selection["selected_files"]:
        motion_path = PACKAGE_DIR / motion_name
        assert motion_path.name.endswith(".retargeted.npz")
        assert motion_path.is_file()
        with np.load(motion_path) as motion:
            assert set(motion.files) == {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
            assert motion["joint_names"].tolist() == EXPECTED_JOINTS
            assert float(motion["framerate"]) == 50.0
            frames = motion["joint_pos"].shape[0]
            assert frames >= 2
            assert motion["joint_pos"].shape == (frames, 22)
            assert motion["base_pos_w"].shape == (frames, 3)
            assert motion["base_quat_w"].shape == (frames, 4)
            assert all(np.isfinite(motion[key]).all() for key in ("joint_pos", "base_pos_w", "base_quat_w"))
            np.testing.assert_allclose(np.linalg.norm(motion["base_quat_w"], axis=1), 1.0, atol=1e-5)
            root_step = np.linalg.norm(np.diff(motion["base_pos_w"], axis=0), axis=1)
            assert float(root_step.max()) < 0.15
            assert float(motion["base_pos_w"][:, 2].min()) > 0.4

            lower = np.asarray([joint_limits[name][0] for name in EXPECTED_JOINTS])
            upper = np.asarray([joint_limits[name][1] for name in EXPECTED_JOINTS])
            assert np.all(motion["joint_pos"] >= lower - 1e-5)
            assert np.all(motion["joint_pos"] <= upper + 1e-5)
            velocity_limits = np.asarray([_velocity_limit(name) for name in EXPECTED_JOINTS])
            speed_ratio = np.abs(np.diff(motion["joint_pos"], axis=0)) * 50.0 / velocity_limits
            assert float(speed_ratio.max()) <= 1.0

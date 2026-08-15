import numpy as np
import yaml
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "parkour_motion_reference" / "booster_k1_v2"
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
    "walk_normal_35_01.retargeted.npz",
    "walk_fast_82_12.retargeted.npz",
    "run_steady_111_24.retargeted.npz",
    "start_to_run_143_03.retargeted.npz",
    "stairs_ascent_114_07.retargeted.npz",
    "stairs_descent_114_07.retargeted.npz",
    "stairs_ascent_83_31.retargeted.npz",
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


def test_packaged_k1_motions_are_continuous_and_within_robot_limits() -> None:
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
    lower = np.asarray([joint_limits[name][0] for name in EXPECTED_JOINTS])
    upper = np.asarray([joint_limits[name][1] for name in EXPECTED_JOINTS])
    velocity_limits = np.asarray([_velocity_limit(name) for name in EXPECTED_JOINTS])

    for motion_name in EXPECTED_MOTIONS:
        with np.load(PACKAGE_DIR / motion_name) as motion:
            fps = float(motion["framerate"])
            joint_pos = motion["joint_pos"]
            assert motion["joint_names"].tolist() == EXPECTED_JOINTS
            assert 14.9 <= fps <= 30.0
            assert joint_pos.ndim == 2 and joint_pos.shape[1] == 22
            assert joint_pos.shape[0] >= 60
            assert np.isfinite(joint_pos).all()
            assert np.isfinite(motion["base_pos_w"]).all()
            np.testing.assert_allclose(np.linalg.norm(motion["base_quat_w"], axis=1), 1.0, atol=1e-5)
            assert np.all(joint_pos >= lower - 1e-5)
            assert np.all(joint_pos <= upper + 1e-5)
            speed_ratio = np.abs(np.diff(joint_pos, axis=0)) * fps / velocity_limits
            assert float(speed_ratio.max()) <= 1.0

            # A failed IK solve can remain numerically inside the URDF bounds
            # by pinning many joints exactly at their limits.  Such poses look
            # severely twisted even though the simpler limit check passes.
            distance_to_limit = np.minimum(joint_pos - lower, upper - joint_pos)
            saturated = distance_to_limit < np.deg2rad(1.0)
            assert float(saturated.mean()) < 0.05

            # Reject roots that are on their side or inverted.  For a wxyz
            # quaternion, R[2, 2] = 1 - 2*(x^2 + y^2).
            root_quat = motion["base_quat_w"]
            root_up_z = 1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
            root_tilt = np.arccos(np.clip(root_up_z, -1.0, 1.0))
            assert float(np.rad2deg(root_tilt).max()) < 45.0

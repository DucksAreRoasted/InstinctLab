import xml.etree.ElementTree as ET
from pathlib import Path


ASSET_DIR = (
    Path(__file__).parents[1]
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "assets"
    / "resources"
    / "booster_k1"
)

FULL_BODY_JOINTS = {
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
}
LOCOMOTION_JOINTS = {
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
}


def _urdf_joint_names(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    }


def _urdf_mesh_paths(path: Path) -> set[Path]:
    root = ET.parse(path).getroot()
    return {path.parent / mesh.attrib["filename"] for mesh in root.findall(".//mesh")}


def test_full_body_and_locomotion_assets_expose_expected_joints() -> None:
    assert _urdf_joint_names(ASSET_DIR / "K1_22dof.urdf") == FULL_BODY_JOINTS
    assert _urdf_joint_names(ASSET_DIR / "K1_locomotion.urdf") == LOCOMOTION_JOINTS


def test_all_urdf_mesh_references_are_packaged() -> None:
    for urdf_name in ("K1_22dof.urdf", "K1_locomotion.urdf"):
        missing = {path for path in _urdf_mesh_paths(ASSET_DIR / urdf_name) if not path.is_file()}
        assert not missing


def test_upstream_license_is_preserved() -> None:
    license_text = (ASSET_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "BSD 3-Clause License" in license_text
    assert "Copyright (c) 2025, BoosterRobotics" in license_text

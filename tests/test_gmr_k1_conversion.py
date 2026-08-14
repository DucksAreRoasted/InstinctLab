import importlib.util
import pickle
from pathlib import Path

import numpy as np

CONVERTER_PATH = Path(__file__).parents[1] / "scripts" / "gmr" / "convert_k1_motion.py"
SPEC = importlib.util.spec_from_file_location("convert_k1_motion", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
convert_gmr_k1_pickle = CONVERTER.convert_gmr_k1_pickle


def test_gmr_k1_pickle_converts_to_instinctlab_retargeted_motion(tmp_path) -> None:
    source_path = tmp_path / "walk.pkl"
    output_path = tmp_path / "walk.retargeted.npz"
    root_pos = np.array([[1.0, 2.0, 0.5], [1.1, 2.0, 0.5]], dtype=np.float32)
    root_rot_xyzw = np.array([[0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    dof_pos = np.arange(44, dtype=np.float32).reshape(2, 22)
    with source_path.open("wb") as motion_file:
        pickle.dump(
            {"fps": 30, "root_pos": root_pos, "root_rot": root_rot_xyzw, "dof_pos": dof_pos},
            motion_file,
        )

    result = convert_gmr_k1_pickle(source_path, output_path)

    assert result == output_path
    with np.load(output_path, allow_pickle=True) as motion:
        assert motion["framerate"].item() == 30
        assert motion["joint_names"].tolist() == [
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
        np.testing.assert_array_equal(motion["joint_pos"], dof_pos)
        np.testing.assert_array_equal(motion["base_pos_w"], root_pos)
        np.testing.assert_allclose(motion["base_quat_w"][0], [0.9, 0.1, 0.2, 0.3])

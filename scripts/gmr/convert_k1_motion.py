#!/usr/bin/env python3
"""Convert a GMR Booster K1 pickle into InstinctLab's retargeted NPZ format."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


K1_JOINT_NAMES = [
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


def convert_gmr_k1_pickle(source_path: str | Path, output_path: str | Path) -> Path:
    """Convert one GMR ``booster_k1`` motion and return the written path.

    GMR stores root quaternions as ``xyzw``. InstinctLab's motion reference
    reader expects Isaac Lab's ``wxyz`` convention, so conversion is explicit.
    """

    source_path = Path(source_path).expanduser()
    output_path = Path(output_path).expanduser()
    with source_path.open("rb") as motion_file:
        motion = pickle.load(motion_file)

    required_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing_keys = required_keys.difference(motion)
    if missing_keys:
        raise ValueError(f"GMR motion is missing required keys: {sorted(missing_keys)}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    joint_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (frames, 3), got {root_pos.shape}")
    if root_rot_xyzw.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot must have shape ({root_pos.shape[0]}, 4), got {root_rot_xyzw.shape}")
    if joint_pos.shape != (root_pos.shape[0], len(K1_JOINT_NAMES)):
        raise ValueError(
            f"dof_pos must have shape ({root_pos.shape[0]}, {len(K1_JOINT_NAMES)}), got {joint_pos.shape}"
        )

    root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        framerate=np.asarray(motion["fps"], dtype=np.float32),
        joint_names=np.asarray(K1_JOINT_NAMES),
        joint_pos=joint_pos,
        base_pos_w=root_pos,
        base_quat_w=root_quat_wxyz,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="GMR pickle generated with --robot booster_k1")
    parser.add_argument("output", type=Path, help="Output path ending in .retargeted.npz")
    args = parser.parse_args()
    output_path = convert_gmr_k1_pickle(args.input, args.output)
    print(f"Converted K1 motion: {output_path}")


if __name__ == "__main__":
    main()

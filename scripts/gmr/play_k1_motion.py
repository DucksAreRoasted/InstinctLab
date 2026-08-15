#!/usr/bin/env python3
"""Preview an InstinctLab Booster K1 retargeted motion with GMR/MuJoCo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_k1_viewer(gmr_root: Path):
    """Load the GMR checkout that actually contains the Booster K1 model."""

    package_init = gmr_root / "general_motion_retargeting" / "__init__.py"
    if not package_init.is_file():
        raise RuntimeError(
            f"GMR root does not contain general_motion_retargeting: {gmr_root}. "
            "Pass --gmr-root or set INSTINCTLAB_GMR_ROOT."
        )
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import ROBOT_XML_DICT, RobotMotionViewer

    if "booster_k1" not in ROBOT_XML_DICT:
        imported_from = sys.modules["general_motion_retargeting"].__file__
        raise RuntimeError(
            f"The imported GMR has no booster_k1 model: {imported_from}. "
            f"Requested GMR root: {gmr_root}"
        )
    return RobotMotionViewer, Path(sys.modules["general_motion_retargeting"].__file__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, nargs="?")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record-video", type=Path)
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument(
        "--gmr-root",
        type=Path,
        default=Path(os.environ.get("INSTINCTLAB_GMR_ROOT", Path.home() / "GMR")),
        help="GMR checkout containing the booster_k1 model (default: ~/GMR)",
    )
    parser.add_argument("--check", action="store_true", help="verify the GMR import and exit")
    args = parser.parse_args()

    RobotMotionViewer, imported_from = load_k1_viewer(args.gmr_root.expanduser().resolve())
    if args.check:
        print(f"OK: booster_k1 loaded from {imported_from}")
        return
    if args.motion is None:
        parser.error("motion is required unless --check is used")

    with np.load(args.motion.expanduser(), allow_pickle=False) as archive:
        fps = float(archive["framerate"])
        root_pos = archive["base_pos_w"].copy()
        root_quat_wxyz = archive["base_quat_w"].copy()
        joint_pos = archive["joint_pos"].copy()

    viewer = RobotMotionViewer(
        robot_type="booster_k1",
        motion_fps=fps,
        transparent_robot=0,
        record_video=args.record_video is not None,
        video_path=str(args.record_video.expanduser().resolve()) if args.record_video else None,
    )
    try:
        while True:
            for frame in range(len(joint_pos)):
                viewer.step(
                    root_pos=root_pos[frame],
                    root_rot=root_quat_wxyz[frame],
                    dof_pos=joint_pos[frame],
                    rate_limit=True,
                    follow_camera=args.follow_camera,
                )
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the curated CMU locomotion/stairs motion pack from GMR pickles.

The input pickles are produced by GMR's ``smplx_to_robot_dataset.py`` with
``--robot booster_k1``.  Clip boundaries intentionally live here so the
training package is reproducible and reviewable instead of being assembled by
ad-hoc array edits.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convert_k1_motion import K1_JOINT_NAMES


@dataclass(frozen=True)
class Clip:
    output_name: str
    source_stem: str
    start_s: float
    end_s: float
    output_fps: float | None = None


CLIPS = (
    Clip("walk_normal_35_01", "35_01_stageii", 0.10, 2.90, 24.0),
    Clip("walk_fast_82_12", "82_12_stageii", 0.20, 3.25),
    Clip("run_steady_111_24", "111_24_stageii", 0.35, 6.70),
    # Slow the human start-to-run transition to K1's roughly 0--1.1 m/s
    # locomotion envelope while preserving its continuous acceleration.
    Clip("start_to_run_143_03", "143_03_stageii", 1.50, 3.90, 22.5),
    Clip("stairs_ascent_114_07", "114_07_stageii", 2.70, 7.30),
    Clip("stairs_descent_114_07", "114_07_stageii", 7.30, 11.80),
    Clip("stairs_ascent_83_31", "83_31_stageii", 1.30, 7.50),
)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    missing = {"fps", "root_pos", "root_rot", "dof_pos"}.difference(motion)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return motion


def build_pack(gmr_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for clip in CLIPS:
        source_path = gmr_dir / f"{clip.source_stem}.pkl"
        motion = _load_pickle(source_path)
        source_fps = float(motion["fps"])
        start = round(clip.start_s * source_fps)
        stop = round(clip.end_s * source_fps)

        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)[start:stop].copy()
        root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)[start:stop]
        joint_pos = np.asarray(motion["dof_pos"], dtype=np.float32)[start:stop]
        if len(root_pos) < 2:
            raise ValueError(f"{clip.output_name} produced fewer than two frames")
        if joint_pos.shape != (len(root_pos), len(K1_JOINT_NAMES)):
            raise ValueError(f"Unexpected K1 joint shape for {clip.output_name}: {joint_pos.shape}")

        # Position each independent trajectory at its environment origin while
        # preserving its vertical profile (especially the stair elevation).
        root_pos[:, :2] -= root_pos[0, :2]
        root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        output_path = output_dir / f"{clip.output_name}.retargeted.npz"
        np.savez_compressed(
            output_path,
            framerate=np.asarray(clip.output_fps or source_fps, dtype=np.float32),
            joint_names=np.asarray(K1_JOINT_NAMES),
            joint_pos=joint_pos,
            base_pos_w=root_pos,
            base_quat_w=root_quat_wxyz,
        )
        outputs.append(output_path)

    selection = "selected_files:\n" + "".join(f"  - {path.name}\n" for path in outputs)
    (output_dir / "motions.yaml").write_text(selection, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in build_pack(args.gmr_dir.expanduser(), args.output_dir.expanduser()):
        print(output)


if __name__ == "__main__":
    main()

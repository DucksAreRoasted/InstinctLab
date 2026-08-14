#!/usr/bin/env python3
"""PROTOTYPE: test whether a retargeted G1 motion can drive Booster K1 IK.

Question: after reconstructing semantic link targets with G1 forward
kinematics, can K1 reproduce a representative motion segment without invalid
joint states or catastrophic IK residuals?

Run interactively:
  /home/ducks/miniconda3/envs/gmr/bin/python scripts/gmr/prototype_g1_to_k1.py \
    --input hiking-in-the-wild_Data\&Model.zip

Retarget segment 2 and immediately visualize K1 with semantic targets:
  /home/ducks/miniconda3/envs/gmr/bin/python scripts/gmr/prototype_g1_to_k1.py \
    --input hiking-in-the-wild_Data\&Model.zip --segment 2 --visualize

This is deliberately throwaway code. If the experiment is promising, absorb
the semantic-motion adapter into a production retargeting module and delete
this terminal shell.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d

from _prototype_g1_to_k1_logic import (
    choose_default_segment_index,
    detect_segments,
    level_foot_target,
    recover_semantic_frame,
    sample_frame_ids,
    support_stairs_for_transition,
)


G1_SEMANTIC_BODIES = {
    "pelvis",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_toe_link",
    "right_toe_link",
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
}


def k1_velocity_limit(joint_name: str) -> float:
    if "Head" in joint_name:
        return 7.85
    if any(part in joint_name for part in ("Shoulder", "Elbow")):
        return 33.51
    if "Hip_Pitch" in joint_name:
        return 14.66
    if "Hip_Roll" in joint_name or "Knee_Pitch" in joint_name:
        return 12.57
    if "Hip_Yaw" in joint_name or "Ankle" in joint_name:
        return 17.59
    raise ValueError(f"No prototype velocity limit for {joint_name}")


def load_gmr(gmr_root: Path):
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import ROBOT_XML_DICT

    from general_motion_retargeting import RobotMotionViewer

    return GeneralMotionRetargeting, RobotMotionViewer, ROBOT_XML_DICT


def resolve_input(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path.expanduser()
    candidates = [
        Path("parkour_motion_reference/parkour_motion_without_run_retargetted.npz"),
        Path("hiking-in-the-wild_Data&Model.zip"),
        Path("/tmp/hiking_in_wild_inspect_20260814/parkour_motion_without_run_retargetted.npz"),
        Path("/tmp/hiking-in-the-wild_Data_Model.zip"),
        Path.home() / "Downloads/hiking-in-the-wild_Data&Model.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("No motion found. Pass --input <official ZIP or retargetted NPZ>.")


def load_motion(input_path: Path) -> dict[str, np.ndarray]:
    if input_path.suffix.lower() != ".zip":
        with np.load(input_path, allow_pickle=False) as motion:
            return {name: motion[name] for name in motion.files}

    with zipfile.ZipFile(input_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if not name.startswith("__MACOSX/") and name.endswith("parkour_motion_without_run_retargetted.npz")
        ]
        if len(matches) != 1:
            raise SystemExit(f"Expected one released parkour motion in ZIP, found {len(matches)}")
        with np.load(io.BytesIO(archive.read(matches[0])), allow_pickle=False) as motion:
            return {name: motion[name] for name in motion.files}


def body_poses(model: mj.MjModel, data: mj.MjData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for name in G1_SEMANTIC_BODIES:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        poses[name] = (data.xpos[body_id].copy(), data.xquat[body_id].copy())
    return poses


def apply_g1_frame(
    model: mj.MjModel,
    data: mj.MjData,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_names: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    for name, value in zip(joint_names, joint_pos):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, str(name))
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mj.mj_forward(model, data)


def foot_support_points(model: mj.MjModel, qpos_frames: np.ndarray) -> np.ndarray:
    """Return the lowest world-space box corner for each K1 foot."""

    foot_box_geoms = []
    for foot_name in ("left_foot_link", "right_foot_link"):
        foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, foot_name)
        boxes = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == foot_id and model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_BOX
        ]
        if not boxes:
            raise ValueError(f"No box collision geometry found for {foot_name}")
        foot_box_geoms.append(boxes)

    box_signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    data = mj.MjData(model)
    all_points = []
    for qpos in qpos_frames:
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        frame_points = []
        for geom_ids in foot_box_geoms:
            corners = []
            for geom_id in geom_ids:
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                corners.extend(
                    data.geom_xpos[geom_id]
                    + (box_signs * model.geom_size[geom_id]) @ rotation.T
                )
            corners = np.asarray(corners)
            frame_points.append(corners[np.argmin(corners[:, 2])])
        all_points.append(frame_points)
    return np.asarray(all_points)


def sole_corners_in_foot_body(model: mj.MjModel) -> dict[str, np.ndarray]:
    """Collect K1 box-collision corners in each foot body's local frame."""

    box_signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    result = {}
    for semantic_name, foot_name in (
        ("left_foot", "left_foot_link"),
        ("right_foot", "right_foot_link"),
    ):
        foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, foot_name)
        corners = []
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] != foot_id or model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_BOX:
                continue
            rotation = np.empty(9)
            mj.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            corners.extend(
                model.geom_pos[geom_id]
                + (box_signs * model.geom_size[geom_id]) @ rotation.reshape(3, 3).T
            )
        if not corners:
            raise ValueError(f"No box collision geometry found for {foot_name}")
        result[semantic_name] = np.asarray(corners)
    return result


def foot_tilt_degrees(model: mj.MjModel, qpos_frames: np.ndarray) -> np.ndarray:
    """Measure each foot sole's unsigned angle from a horizontal plane."""

    geom_ids = []
    for foot_name in ("left_foot_link", "right_foot_link"):
        foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, foot_name)
        geom_ids.append(
            next(
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom_bodyid[geom_id] == foot_id
                and model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_BOX
            )
        )
    data = mj.MjData(model)
    result = []
    for qpos in qpos_frames:
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        result.append(
            [
                np.degrees(
                    np.arccos(
                        np.clip(abs(data.geom_xmat[geom_id].reshape(3, 3)[2, 2]), 0.0, 1.0)
                    )
                )
                for geom_id in geom_ids
            ]
        )
    return np.asarray(result)


def draw_support_platform(viewer, platform, label: str | None = None) -> None:
    """Add a translucent box to the MuJoCo user scene."""

    direction_x, direction_y = platform.heading_xy
    rotation = np.array(
        [[direction_x, -direction_y, 0.0], [direction_y, direction_x, 0.0], [0.0, 0.0, 1.0]]
    )
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_BOX,
        size=platform.half_size_xyz,
        pos=platform.center_xyz,
        mat=rotation.flatten(),
        rgba=np.array([0.85, 0.48, 0.12, 0.72]),
    )
    if label is not None:
        geom.label = label
    viewer.user_scn.ngeom += 1


def retarget_segment(args, motion, segment, source_config, GeneralMotionRetargeting, robot_xml_dict):
    g1_model = mj.MjModel.from_xml_path(str(robot_xml_dict["unitree_g1"]))
    g1_data = mj.MjData(g1_model)
    retargeter = GeneralMotionRetargeting(
        actual_human_height=None,
        src_human="smplx",
        tgt_robot="booster_k1",
        verbose=False,
        use_velocity_limit=True,
    )

    frame_ids, rate_multiplier = sample_frame_ids(segment, args.frames)
    effective_fps = float(motion["framerate"]) * rate_multiplier

    semantic_frames = []
    source_table = source_config["ik_match_table2"]
    source_scale = source_config["human_scale_table"]
    for frame_id in frame_ids:
        apply_g1_frame(
            g1_model,
            g1_data,
            motion["base_pos_w"][frame_id],
            motion["base_quat_w"][frame_id],
            motion["joint_names"],
            motion["joint_pos"][frame_id],
        )
        semantic_frames.append(
            recover_semantic_frame(body_poses(g1_model, g1_data), source_table, source_scale)
        )

    foot_names = ("left_foot", "right_foot")
    scaled_foot_positions = np.asarray(
        [
            [
                semantic[name][0] * retargeter.human_scale_table[name]
                for name in foot_names
            ]
            for semantic in semantic_frames
        ]
    )
    if len(scaled_foot_positions) > 1:
        foot_speed = np.linalg.norm(
            np.gradient(scaled_foot_positions, axis=0) * effective_fps,
            axis=2,
        )
        contact_confidence = np.clip(
            gaussian_filter1d(
                np.clip((1.0 - foot_speed) / 0.6, 0.0, 1.0),
                sigma=2.0,
                axis=0,
                mode="nearest",
            ),
            0.0,
            1.0,
        )
    else:
        contact_confidence = np.ones((len(scaled_foot_positions), 2))

    if args.contact_correction:
        sole_corners = sole_corners_in_foot_body(retargeter.model)
        for frame_index, semantic in enumerate(semantic_frames):
            for foot_index, name in enumerate(foot_names):
                semantic[name] = level_foot_target(
                    semantic[name][0],
                    semantic[name][1],
                    retargeter.rot_offsets2[name].as_quat(scalar_first=True),
                    sole_corners[name],
                    float(contact_confidence[frame_index, foot_index]),
                    retargeter.human_scale_table[name],
                )

    output_qpos = []
    solver_error = []
    for output_index, semantic in enumerate(semantic_frames):
        # Warm-start the recurrent IK state on the first target.
        repeats = 8 if output_index == 0 else 1
        for _ in range(repeats):
            qpos = retargeter.retarget(semantic, offset_to_ground=False)
        output_qpos.append(qpos)
        solver_error.append((retargeter.error1(), retargeter.error2()))

    output_qpos = np.asarray(output_qpos, dtype=np.float32)
    if args.contact_correction and len(output_qpos) > 2:
        output_qpos[:, 7:] = gaussian_filter1d(
            output_qpos[:, 7:], sigma=1.25, axis=0, mode="nearest"
        )
    solver_error = np.asarray(solver_error)
    k1_model = retargeter.model
    joint_qpos = output_qpos[:, 7:]
    limited_joint_ids = [joint_id for joint_id in range(k1_model.njnt) if k1_model.jnt_limited[joint_id]]
    violations = 0
    minimum_margin = np.inf
    for joint_id in limited_joint_ids:
        address = k1_model.jnt_qposadr[joint_id]
        values = output_qpos[:, address]
        lower, upper = k1_model.jnt_range[joint_id]
        violations += int(np.count_nonzero((values < lower - 1e-5) | (values > upper + 1e-5)))
        minimum_margin = min(minimum_margin, float(np.min(values - lower)), float(np.min(upper - values)))

    joint_names = [
        mj.mj_id2name(k1_model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(k1_model.njnt)
        if k1_model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE
    ]
    joint_speed = np.abs(np.diff(joint_qpos, axis=0)) * effective_fps
    velocity_limits = np.asarray([k1_velocity_limit(name) for name in joint_names])
    velocity_ratio = joint_speed / velocity_limits if joint_speed.size else np.zeros((0, len(joint_names)))
    worst_velocity_index = np.unravel_index(np.argmax(velocity_ratio), velocity_ratio.shape) if velocity_ratio.size else None
    support_points = foot_support_points(k1_model, output_qpos)
    sole_tilt = foot_tilt_degrees(k1_model, output_qpos)
    foot_clearance = np.min(support_points[:, :, 2], axis=1)
    support_stairs = support_stairs_for_transition(segment, support_points)
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path(f"/tmp/g1_to_k1_segment_{segment.index + 1:03d}.retargeted.npz")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        framerate=np.asarray(effective_fps, dtype=np.float32),
        joint_names=np.asarray(joint_names),
        joint_pos=joint_qpos,
        base_pos_w=output_qpos[:, :3],
        base_quat_w=output_qpos[:, 3:7],
    )

    report = {
        "segment": asdict(segment),
        "motion_kind": segment.motion_kind,
        "sampled_frames": int(len(frame_ids)),
        "effective_fps": effective_fps,
        "contact_correction": bool(args.contact_correction),
        "finite": bool(np.isfinite(output_qpos).all()),
        "joint_limit_violations": violations,
        "minimum_joint_limit_margin_rad": minimum_margin,
        "max_joint_speed_rad_s": float(np.max(joint_speed)) if joint_speed.size else 0.0,
        "max_velocity_limit_ratio": float(np.max(velocity_ratio)) if velocity_ratio.size else 0.0,
        "worst_velocity_joint": joint_names[worst_velocity_index[1]] if worst_velocity_index else None,
        "solver_error1_mean": float(np.mean(solver_error[:, 0])),
        "solver_error1_max": float(np.max(solver_error[:, 0])),
        "solver_error2_mean": float(np.mean(solver_error[:, 1])),
        "solver_error2_max": float(np.max(solver_error[:, 1])),
        "root_height_range_m": [float(output_qpos[:, 2].min()), float(output_qpos[:, 2].max())],
        "lowest_foot_clearance_m": {
            "min": float(np.min(foot_clearance)),
            "median": float(np.median(foot_clearance)),
            "max": float(np.max(foot_clearance)),
        },
        "inferred_support_platform_top_m": (
            support_stairs[0].top_z_m if support_stairs else None
        ),
        "inferred_stair_tops_m": [stair.top_z_m for stair in support_stairs],
        "strong_contact_sole_tilt_deg": {
            "median": (
                float(np.median(sole_tilt[contact_confidence > 0.7]))
                if np.any(contact_confidence > 0.7)
                else None
            ),
            "max": (
                float(np.max(sole_tilt[contact_confidence > 0.7]))
                if np.any(contact_confidence > 0.7)
                else None
            ),
        },
        "output": str(output_path),
    }
    report["verdict"] = (
        "promising"
        if report["finite"]
        and violations == 0
        and report["solver_error2_max"] < 1.0
        and report["max_velocity_limit_ratio"] < 1.25
        and (segment.motion_kind != "grounded locomotion" or np.median(foot_clearance) < 0.05)
        else "needs-tuning"
    )
    return report


def visualize_motion(
    args,
    motion,
    segment,
    source_config,
    output_path,
    GeneralMotionRetargeting,
    RobotMotionViewer,
    robot_xml_dict,
) -> None:
    if segment.motion_kind in {"ascent", "descent"}:
        print(
            f"[INFO] This is a terrain {segment.motion_kind} clip. "
            "The orange stairs are inferred from the foot-contact height sequence."
        )
    elif segment.motion_kind == "airborne/vertical":
        print("[NOTICE] This source segment contains only an airborne phase.")
    else:
        print("[INFO] Visualizing a grounded locomotion segment.")
    output = load_motion(Path(output_path))
    frame_ids, _ = sample_frame_ids(segment, args.frames)
    g1_model = mj.MjModel.from_xml_path(str(robot_xml_dict["unitree_g1"]))
    g1_data = mj.MjData(g1_model)
    target_adapter = GeneralMotionRetargeting(
        actual_human_height=None,
        src_human="smplx",
        tgt_robot="booster_k1",
        verbose=False,
    )
    from general_motion_retargeting.robot_motion_viewer import draw_frame

    viewer = RobotMotionViewer(
        robot_type="booster_k1",
        motion_fps=float(output["framerate"]) * args.playback_speed,
        camera_follow=True,
    )
    source_table = source_config["ik_match_table2"]
    source_scale = source_config["human_scale_table"]
    output_qpos = np.concatenate(
        (output["base_pos_w"], output["base_quat_w"], output["joint_pos"]), axis=1
    )
    support_stairs = support_stairs_for_transition(
        segment, foot_support_points(target_adapter.model, output_qpos)
    )
    try:
        while viewer.viewer.is_running():
            for output_index, source_frame_id in enumerate(frame_ids):
                if not viewer.viewer.is_running():
                    return
                apply_g1_frame(
                    g1_model,
                    g1_data,
                    motion["base_pos_w"][source_frame_id],
                    motion["base_quat_w"][source_frame_id],
                    motion["joint_names"],
                    motion["joint_pos"][source_frame_id],
                )
                semantic = recover_semantic_frame(body_poses(g1_model, g1_data), source_table, source_scale)
                target_adapter.update_targets(semantic, offset_to_ground=False)
                viewer.viewer.user_scn.ngeom = 0
                for body_name, (position, quaternion) in target_adapter.scaled_human_data.items():
                    rotation = np.empty(9)
                    mj.mju_quat2Mat(rotation, quaternion)
                    draw_frame(
                        position,
                        rotation.reshape(3, 3),
                        viewer.viewer,
                        0.06,
                        joint_name=body_name if args.show_labels else None,
                    )
                for stair_index, stair in enumerate(support_stairs):
                    draw_support_platform(
                        viewer.viewer,
                        stair,
                        label="inferred stairs" if stair_index == 0 else None,
                    )
                viewer.step(
                    output["base_pos_w"][output_index],
                    output["base_quat_w"][output_index],
                    output["joint_pos"][output_index],
                    human_motion_data=None,
                )
            if args.play_once:
                break
    finally:
        viewer.close()


def render(input_path, segments, selected, report=None):
    print("\033[2J\033[H", end="")
    print("\033[1mG1 -> Booster K1 retargeting feasibility prototype\033[0m")
    print("\033[2mThrowaway shell; no project data is modified.\033[0m\n")
    print(f"\033[1minput\033[0m: {input_path}")
    segment = segments[selected]
    print(f"\033[1msegment\033[0m: {selected + 1}/{len(segments)}")
    print(f"\033[1mframes\033[0m: {segment.start}:{segment.stop} ({segment.frames})")
    print(f"\033[1mduration\033[0m: {segment.duration_s:.2f} s")
    print(f"\033[1mmedian root speed\033[0m: {segment.median_speed_mps:.3f} m/s")
    print(f"\033[1mroot vertical range\033[0m: {segment.root_vertical_range_m:.3f} m")
    print(f"\033[1mmotion kind\033[0m: {segment.motion_kind}")
    if report is not None:
        print("\n\033[1mlast retarget report\033[0m")
        print(json.dumps(report, indent=2))
    print(
        "\n\033[1m[n]\033[0m next  \033[1m[p]\033[0m previous  "
        "\033[1m[r]\033[0m retarget  \033[1m[v]\033[0m retarget + visualize  \033[1m[q]\033[0m quit"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Official release ZIP or G1 retargetted NPZ; common paths auto-detected")
    parser.add_argument("--gmr-root", type=Path, default=Path("/home/ducks/GMR"))
    parser.add_argument("--segment", type=int, help="1-based segment number; defaults to a short grounded clip")
    parser.add_argument("--frames", type=int, default=0, help="Maximum frames; 0 keeps the original 50 Hz sequence")
    parser.add_argument("--output", help="Output NPZ; defaults to /tmp/g1_to_k1_segment_NNN.retargeted.npz")
    parser.add_argument("--list", action="store_true", help="Print detected segments and exit")
    parser.add_argument("--auto", action="store_true", help="Retarget once without opening the interactive shell")
    parser.add_argument("--visualize", action="store_true", help="Retarget once, then play K1 with semantic target axes")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--play-once", action="store_true", help="Close the viewer after one playback")
    parser.add_argument("--show-labels", action="store_true", help="Label semantic target axes in the viewer")
    parser.add_argument(
        "--contact-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Level likely support feet and preserve sole height (default: enabled)",
    )
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    motion = load_motion(input_path)
    segments = detect_segments(
        motion["base_pos_w"],
        motion["base_quat_w"],
        motion["joint_pos"],
        float(motion["framerate"]),
    )
    source_config_path = args.gmr_root / "general_motion_retargeting/ik_configs/smplx_to_g1.json"
    source_config = json.loads(source_config_path.read_text())
    if args.list:
        print("segment  frames       duration  median-speed  root-dz  kind")
        for segment in segments:
            print(
                f"{segment.index + 1:>7}  {segment.start:>6}:{segment.stop:<6}  "
                f"{segment.duration_s:>7.2f}s  {segment.median_speed_mps:>8.3f} m/s  "
                f"{segment.root_vertical_range_m:>7.3f}  {segment.motion_kind}"
            )
        return
    GeneralMotionRetargeting, RobotMotionViewer, robot_xml_dict = load_gmr(args.gmr_root)
    selected = (
        choose_default_segment_index(segments)
        if args.segment is None
        else max(0, min(args.segment - 1, len(segments) - 1))
    )

    if args.auto or args.visualize:
        report = retarget_segment(
            args, motion, segments[selected], source_config, GeneralMotionRetargeting, robot_xml_dict
        )
        print(json.dumps(report, indent=2))
        if args.visualize:
            visualize_motion(
                args,
                motion,
                segments[selected],
                source_config,
                report["output"],
                GeneralMotionRetargeting,
                RobotMotionViewer,
                robot_xml_dict,
            )
        return

    report = None
    while True:
        render(input_path, segments, selected, report)
        action = input("> ").strip().lower()
        if action == "q":
            break
        if action == "n":
            selected = (selected + 1) % len(segments)
            report = None
        elif action == "p":
            selected = (selected - 1) % len(segments)
            report = None
        elif action == "r":
            report = retarget_segment(
                args, motion, segments[selected], source_config, GeneralMotionRetargeting, robot_xml_dict
            )
        elif action == "v":
            report = retarget_segment(
                args, motion, segments[selected], source_config, GeneralMotionRetargeting, robot_xml_dict
            )
            visualize_motion(
                args,
                motion,
                segments[selected],
                source_config,
                report["output"],
                GeneralMotionRetargeting,
                RobotMotionViewer,
                robot_xml_dict,
            )


if __name__ == "__main__":
    main()

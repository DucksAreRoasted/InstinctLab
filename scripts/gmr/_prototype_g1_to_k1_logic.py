"""Portable logic for the throwaway G1-to-K1 retargeting prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MotionSegment:
    index: int
    start: int
    stop: int
    duration_s: float
    median_speed_mps: float
    root_vertical_range_m: float
    start_height_above_min_m: float = 0.0
    end_height_above_min_m: float = 0.0

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def motion_kind(self) -> str:
        if self.root_vertical_range_m <= 0.20:
            return "grounded locomotion"
        start_high = self.start_height_above_min_m > 0.20
        end_high = self.end_height_above_min_m > 0.20
        if start_high and not end_high:
            return "descent"
        if end_high and not start_high:
            return "ascent"
        if not start_high and not end_high:
            return "jump"
        return "airborne/vertical"


@dataclass(frozen=True)
class SupportPlatform:
    center_xyz: np.ndarray
    heading_xy: np.ndarray
    half_size_xyz: np.ndarray
    top_z_m: float


def support_platform_for_transition(
    segment: MotionSegment,
    foot_support_points: np.ndarray,
) -> SupportPlatform | None:
    """Return the highest tread for callers needing one support surface."""

    stairs = support_stairs_for_transition(segment, foot_support_points)
    return stairs[0] if stairs else None


def support_stairs_for_transition(
    segment: MotionSegment,
    foot_support_points: np.ndarray,
    nominal_riser_height_m: float = 0.11,
) -> list[SupportPlatform]:
    """Infer simple stair treads for a height-changing terrain transition.

    ``foot_support_points`` has shape ``(frames, 2, 3)`` and contains the
    lowest point of each foot. The stairs are visualization aids; they do not
    alter the retargeted root trajectory or participate in simulation.
    """

    if segment.motion_kind not in {"ascent", "descent"}:
        return []
    if foot_support_points.ndim != 3 or foot_support_points.shape[1:] != (2, 3):
        raise ValueError("foot support points must have shape (frames, 2, 3)")
    if nominal_riser_height_m <= 0.0:
        raise ValueError("nominal riser height must be positive")

    high_index = 0 if segment.motion_kind == "descent" else -1
    low_index = -1 if segment.motion_kind == "descent" else 0
    high_center = np.mean(foot_support_points[high_index], axis=0)
    low_center = np.mean(foot_support_points[low_index], axis=0)
    high_to_low = low_center[:2] - high_center[:2]
    travel_distance = float(np.linalg.norm(high_to_low))
    heading = high_to_low / travel_distance if travel_distance > 1e-6 else np.array([1.0, 0.0])

    high_top = float(np.min(foot_support_points[high_index, :, 2]))
    low_top = float(np.min(foot_support_points[:, :, 2]))
    elevation = max(high_top - low_top, 0.0)
    lateral_span = float(
        np.linalg.norm(foot_support_points[high_index, 0, :2] - foot_support_points[high_index, 1, :2])
    )
    half_width = max(0.35, lateral_span / 2.0 + 0.12)

    candidates = []
    if len(foot_support_points) > 1:
        effective_fps = len(foot_support_points) / segment.duration_s
        foot_speed = np.linalg.norm(
            np.gradient(foot_support_points, axis=0) * effective_fps,
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
        for foot_index in range(2):
            mask = contact_confidence[:, foot_index] > 0.65
            edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
            for start, stop in edges.reshape(-1, 2):
                if stop - start >= 3:
                    candidates.append(np.median(foot_support_points[start:stop, foot_index], axis=0))

    # Merge fragments from the same stationary contact, then keep the
    # monotonically descending sequence. This rejects slow swing-foot apexes.
    merged = []
    for candidate in candidates:
        for index, existing in enumerate(merged):
            if np.linalg.norm(candidate[:2] - existing[:2]) < 0.15 and abs(candidate[2] - existing[2]) < 0.05:
                merged[index] = (existing + candidate) / 2.0
                break
        else:
            merged.append(candidate)
    merged.sort(key=lambda point: float(np.dot(point[:2] - high_center[:2], heading)))
    observed_contacts = []
    for candidate in merged:
        if not observed_contacts or candidate[2] < observed_contacts[-1][2] - 0.04:
            observed_contacts.append(candidate)

    if len(observed_contacts) >= 3:
        lateral = np.array([-heading[1], heading[0]])
        projections = np.asarray(
            [np.dot(point[:2] - high_center[:2], heading) for point in observed_contacts]
        )
        lateral_center = float(
            np.median([np.dot(point[:2] - high_center[:2], lateral) for point in observed_contacts])
        )
        drawable_count = len(observed_contacts)
        if observed_contacts[-1][2] <= low_top + 0.05:
            drawable_count -= 1
        stairs = []
        for step_index in range(drawable_count):
            if step_index == 0:
                left_edge = projections[0] - max((projections[1] - projections[0]) / 2.0, 0.08)
            else:
                left_edge = (projections[step_index - 1] + projections[step_index]) / 2.0
            right_edge = (
                (projections[step_index] + projections[step_index + 1]) / 2.0
                if step_index + 1 < len(projections)
                else projections[step_index] + max(projections[step_index] - projections[step_index - 1], 0.16) / 2.0
            )
            center_projection = (left_edge + right_edge) / 2.0
            top_z = float(observed_contacts[step_index][2])
            center_xy = high_center[:2] + heading * center_projection + lateral * lateral_center
            stairs.append(
                SupportPlatform(
                    center_xyz=np.array([center_xy[0], center_xy[1], top_z / 2.0]),
                    heading_xy=heading,
                    half_size_xyz=np.array(
                        [(right_edge - left_edge) / 2.0 + 0.01, half_width, max(top_z / 2.0, 0.01)]
                    ),
                    top_z_m=top_z,
                )
            )
        return stairs

    step_count = max(1, int(round(elevation / nominal_riser_height_m)))
    riser_height = elevation / step_count
    tread_run = max(travel_distance / step_count, 0.16)
    stairs = []
    for step_index in range(step_count):
        top_z = high_top - step_index * riser_height
        # The high-end contact lies on the leading edge of the first tread;
        # subsequent treads extend in the high-to-low direction.
        center_xy = high_center[:2] + heading * tread_run * (step_index - 0.5)
        stairs.append(
            SupportPlatform(
                center_xyz=np.array([center_xy[0], center_xy[1], top_z / 2.0]),
                heading_xy=heading,
                half_size_xyz=np.array(
                    [tread_run / 2.0 + 0.01, half_width, max(top_z / 2.0, 0.01)]
                ),
                top_z_m=top_z,
            )
        )
    return stairs


def detect_segments(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    min_frames: int = 50,
) -> list[MotionSegment]:
    """Split concatenated motion at physically impossible frame transitions."""

    if not (len(root_pos) == len(root_quat_wxyz) == len(joint_pos)):
        raise ValueError("root and joint trajectories must contain the same number of frames")
    root_step = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    quat_dot = np.abs(np.sum(root_quat_wxyz[:-1] * root_quat_wxyz[1:], axis=1))
    root_angle = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))
    # A joint can move quickly during takeoff or landing without indicating a
    # clip boundary. Only discontinuities in the global root trajectory are
    # strong enough evidence that two released motions were concatenated.
    cuts = np.flatnonzero((root_step > 0.15) | (root_angle > np.deg2rad(45.0))) + 1
    edges = np.concatenate(([0], cuts, [len(root_pos)]))

    segments = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < min_frames:
            continue
        speed = np.linalg.norm(np.diff(root_pos[start:stop, :2], axis=0), axis=1) * fps
        root_z = root_pos[start:stop, 2]
        minimum_root_z = float(np.min(root_z))
        segments.append(
            MotionSegment(
                index=len(segments),
                start=int(start),
                stop=int(stop),
                duration_s=(stop - start) / fps,
                median_speed_mps=float(np.median(speed)) if len(speed) else 0.0,
                root_vertical_range_m=float(np.ptp(root_z)),
                start_height_above_min_m=float(root_z[0] - minimum_root_z),
                end_height_above_min_m=float(root_z[-1] - minimum_root_z),
            )
        )
    return segments


def choose_default_segment_index(segments: list[MotionSegment]) -> int:
    """Pick a short, grounded locomotion clip for the first visualization."""

    for segment in segments:
        if (
            segment.motion_kind == "grounded locomotion"
            and 2.0 <= segment.duration_s <= 15.0
            and 0.3 <= segment.median_speed_mps <= 1.0
        ):
            return segment.index
    for segment in segments:
        if segment.motion_kind == "grounded locomotion":
            return segment.index
    return 0


def sample_frame_ids(segment: MotionSegment, maximum_frames: int) -> tuple[np.ndarray, float]:
    """Select prototype frames and return their effective rate multiplier."""

    frame_ids = np.arange(segment.start, segment.stop)
    if maximum_frames > 0 and len(frame_ids) > maximum_frames:
        frame_ids = np.linspace(segment.start, segment.stop - 1, maximum_frames, dtype=np.int64)
    rate_multiplier = len(frame_ids) / segment.frames
    return frame_ids, rate_multiplier


def level_foot_target(
    semantic_position: np.ndarray,
    semantic_quaternion_wxyz: np.ndarray,
    robot_rotation_offset_wxyz: np.ndarray,
    sole_corners_in_body: np.ndarray,
    confidence: float,
    position_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a robot foot target toward level while preserving sole height."""

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("contact confidence must be within [0, 1]")
    if position_scale <= 0.0:
        raise ValueError("position scale must be positive")
    semantic_rotation = Rotation.from_quat(semantic_quaternion_wxyz, scalar_first=True)
    robot_offset = Rotation.from_quat(robot_rotation_offset_wxyz, scalar_first=True)
    source_body_rotation = semantic_rotation * robot_offset
    forward = source_body_rotation.apply([1.0, 0.0, 0.0])
    if np.linalg.norm(forward[:2]) < 1e-8:
        lateral = source_body_rotation.apply([0.0, 1.0, 0.0])
        yaw = np.arctan2(lateral[1], lateral[0]) - np.pi / 2.0
    else:
        yaw = np.arctan2(forward[1], forward[0])
    level_body_rotation = Rotation.from_euler("z", yaw)
    correction = source_body_rotation.inv() * level_body_rotation
    corrected_body_rotation = source_body_rotation * Rotation.from_rotvec(
        correction.as_rotvec() * confidence
    )

    source_lowest_offset = float(np.min(source_body_rotation.apply(sole_corners_in_body)[:, 2]))
    corrected_lowest_offset = float(np.min(corrected_body_rotation.apply(sole_corners_in_body)[:, 2]))
    corrected_position = np.asarray(semantic_position, dtype=np.float64).copy()
    corrected_position[2] += (source_lowest_offset - corrected_lowest_offset) / position_scale
    corrected_semantic_rotation = corrected_body_rotation * robot_offset.inv()
    return corrected_position, corrected_semantic_rotation.as_quat(scalar_first=True)


def recover_semantic_frame(
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    source_match_table: dict[str, list],
    source_scale_table: dict[str, float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Approximately invert the original SMPL-X-to-G1 semantic constraints."""

    scaled = {}
    for body_name, entry in source_match_table.items():
        semantic_name, _, _, pos_offset, rot_offset = entry
        if semantic_name not in source_scale_table or body_name not in body_poses:
            continue
        body_pos, body_quat = body_poses[body_name]
        body_rotation = Rotation.from_quat(body_quat, scalar_first=True)
        semantic_rotation = body_rotation * Rotation.from_quat(rot_offset, scalar_first=True).inv()
        semantic_pos = body_pos - body_rotation.apply(np.asarray(pos_offset, dtype=np.float64))
        scaled[semantic_name] = (semantic_pos, semantic_rotation.as_quat(scalar_first=True))

    root_pos, root_quat = scaled["pelvis"]
    root_scale = source_scale_table["pelvis"]
    unscaled_root = root_pos / root_scale
    semantic = {"pelvis": (unscaled_root, root_quat)}
    for name, (position, quaternion) in scaled.items():
        if name == "pelvis":
            continue
        semantic[name] = (
            unscaled_root + (position - root_pos) / source_scale_table[name],
            quaternion,
        )

    # G1 has no articulated neck in this motion. A torso-aligned head target
    # lets K1 keep a quiet, forward-looking neck while the feasibility of the
    # locomotion transfer is evaluated.
    torso_pos, torso_quat = semantic["spine3"]
    semantic["head"] = (torso_pos + np.array([0.0, 0.0, 0.25]), torso_quat)
    return semantic


def quaternion_angle_wxyz(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angular distance between two scalar-first quaternions."""

    dot = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/gmr"))

from _prototype_g1_to_k1_logic import (  # noqa: E402
    MotionSegment,
    choose_default_segment_index,
    detect_segments,
    level_foot_target,
    support_platform_for_transition,
    support_stairs_for_transition,
)


def test_fast_joint_motion_without_root_discontinuity_stays_in_one_segment() -> None:
    frames = 120
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.arange(frames, dtype=np.float32) * 0.01
    root_quat = np.zeros((frames, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    joint_pos = np.zeros((frames, 22), dtype=np.float32)
    joint_pos[60:, 0] = 0.3

    segments = detect_segments(root_pos, root_quat, joint_pos, fps=50.0)

    assert [(segment.start, segment.stop) for segment in segments] == [(0, frames)]


def test_default_selection_falls_back_to_long_grounded_motion() -> None:
    segments = [
        MotionSegment(0, 0, 100, 2.0, 0.4, 0.5, 0.0, 0.0),
        MotionSegment(1, 100, 3900, 76.0, 0.8, 0.08, 0.0, 0.0),
    ]

    assert choose_default_segment_index(segments) == 1


def test_descent_segment_gets_a_platform_below_its_high_endpoint() -> None:
    segment = MotionSegment(0, 0, 75, 1.5, 0.8, 0.58, 0.58, 0.03)
    foot_centers = np.array(
        [
            [[0.0, -0.1, 0.44], [0.0, 0.1, 0.43]],
            [[0.8, -0.1, 0.02], [0.8, 0.1, 0.02]],
        ],
        dtype=np.float64,
    )

    platform = support_platform_for_transition(segment, foot_centers)

    assert segment.motion_kind == "descent"
    assert platform is not None
    assert platform.top_z_m == 0.43
    assert platform.center_xyz[0] < 0.0


def test_descent_height_is_visualized_as_multiple_stair_treads() -> None:
    segment = MotionSegment(0, 0, 75, 1.5, 0.8, 0.58, 0.58, 0.03)
    foot_centers = np.array(
        [
            [[0.0, -0.1, 0.44], [0.0, 0.1, 0.43]],
            [[0.8, -0.1, 0.02], [0.8, 0.1, 0.02]],
        ],
        dtype=np.float64,
    )

    stairs = support_stairs_for_transition(segment, foot_centers)

    assert len(stairs) == 4
    np.testing.assert_allclose([stair.top_z_m for stair in stairs], [0.43, 0.3275, 0.225, 0.1225])
    assert all(stairs[index].center_xyz[0] < stairs[index + 1].center_xyz[0] for index in range(3))


def test_level_foot_target_preserves_the_lowest_sole_height() -> None:
    source_rotation = Rotation.from_euler("y", 30, degrees=True)
    source_position = np.array([0.0, 0.0, 0.5])
    sole_corners = np.array(
        [[x, y, -0.05] for x in (-0.12, 0.12) for y in (-0.06, 0.06)],
        dtype=np.float64,
    )
    source_lowest = np.min(source_position[2] + source_rotation.apply(sole_corners)[:, 2])

    corrected_position, corrected_quaternion = level_foot_target(
        source_position,
        source_rotation.as_quat(scalar_first=True),
        robot_rotation_offset_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        sole_corners_in_body=sole_corners,
        confidence=1.0,
        position_scale=1.0,
    )

    corrected_rotation = Rotation.from_quat(corrected_quaternion, scalar_first=True)
    corrected_lowest = np.min(corrected_position[2] + corrected_rotation.apply(sole_corners)[:, 2])
    np.testing.assert_allclose(corrected_rotation.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(corrected_lowest, source_lowest, atol=1e-7)


def test_stair_tops_follow_observed_stationary_foot_heights() -> None:
    frames = 50
    points = np.zeros((frames, 2, 3), dtype=np.float64)
    points[:, 0] = np.column_stack(
        (np.linspace(-0.2, 1.0, frames), np.full(frames, -0.1), np.linspace(0.6, 0.02, frames))
    )
    points[:, 1] = np.column_stack(
        (np.linspace(-0.1, 1.1, frames), np.full(frames, 0.1), np.linspace(0.7, 0.03, frames))
    )
    contacts = [
        (0, 0, 8, [0.0, -0.1, 0.43]),
        (1, 10, 18, [0.2, 0.1, 0.32]),
        (0, 20, 28, [0.4, -0.1, 0.21]),
        (1, 30, 38, [0.6, 0.1, 0.10]),
        (0, 40, 48, [0.8, -0.1, 0.02]),
    ]
    for foot, start, stop, position in contacts:
        points[start:stop, foot] = position
    segment = MotionSegment(0, 0, frames, 1.0, 0.8, 0.5, 0.5, 0.0)

    stairs = support_stairs_for_transition(segment, points)

    np.testing.assert_allclose([stair.top_z_m for stair in stairs], [0.43, 0.32, 0.21, 0.10], atol=0.015)

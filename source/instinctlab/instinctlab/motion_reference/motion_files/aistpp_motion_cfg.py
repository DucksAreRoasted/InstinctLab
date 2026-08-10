from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .aistpp_motion import AistppMotion
from .amass_motion_cfg import AmassMotionCfg


@configclass
class AistppMotionCfg(AmassMotionCfg):
    """Configuration for AIST++ motion files."""

    class_type: type | str = "{DIR}.aistpp_motion:AistppMotion"

    assumed_file_framerate: float = 60.0  # refer to https://arxiv.org/pdf/2101.08779

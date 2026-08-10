from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_physx.sensors.ray_caster import RayCasterCamera

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera(NoisyCameraMixin, RayCasterCamera):
    cfg: NoisyRayCasterCameraCfg

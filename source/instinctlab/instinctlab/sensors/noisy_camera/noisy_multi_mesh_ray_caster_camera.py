from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_physx.sensors.ray_caster import MultiMeshRayCasterCamera

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_multi_mesh_ray_caster_camera_cfg import NoisyMultiMeshRayCasterCameraCfg


class NoisyMultiMeshRayCasterCamera(NoisyCameraMixin, MultiMeshRayCasterCamera):
    cfg: NoisyMultiMeshRayCasterCameraCfg

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class GroupedRayCasterCameraCfg(MultiMeshRayCasterCameraCfg):
    """Configuration for the grouped-ray-cast camera sensor."""

    class_type: type | str = "{DIR}.grouped_ray_caster_camera:GroupedRayCasterCamera"

    min_distance: float = 0.0
    """Minimum accepted hit distance from the camera, in meters."""

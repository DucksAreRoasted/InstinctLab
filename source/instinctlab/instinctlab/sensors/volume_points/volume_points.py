from __future__ import annotations

import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    import omni.physics.tensors.api as physx

    from isaaclab.markers import VisualizationMarkersCfg

    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(SensorBase):
    """Volume Points sensor for detecting volume points in a simulation."""

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)

        # Initialize the volume points
        self._volume_points = None

    """
    Properties
    """

    @property
    def data(self) -> VolumePointsData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with volume points sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with volume points sensors attached."""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the rigid bodies captured (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._body_physx_view

    """
    Operations
    """

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        """Record the edges buffer to the sensor so that the penetration data can be updated.

        NOTE: typically this is called by a startup event.

        """
        self._virtual_obstacles.update(virtual_obstacles)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Implementation
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        from isaaclab_physx.physics import PhysxManager

        # create simulation view
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        if self._physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")

        # Resolve the authored articulation once, then select rigid links recursively.
        root_matches = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)
        if not root_matches:
            raise RuntimeError(f"Sensor root at path '{self.cfg.prim_path}' could not be resolved.")
        template_root, destination_root_expr = root_matches[0]
        name_exprs = self.cfg.body_names_expr
        if isinstance(name_exprs, str):
            name_exprs = [name_exprs]
        name_patterns = [re.compile(f"^{expr}$") for expr in name_exprs]
        body_prims = sim_utils.get_all_matching_child_prims(
            template_root.GetPath(),
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and any(pattern.match(prim.GetName()) is not None for pattern in name_patterns),
            traverse_instance_prims=False,
        )
        if not body_prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find rigid bodies matching {name_exprs}."
            )

        template_root_path = template_root.GetPath().pathString
        body_paths_glob = [
            (destination_root_expr + prim.GetPath().pathString[len(template_root_path) :]).replace(".*", "*")
            for prim in body_prims
        ]

        # create a rigid prim view for the sensor
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_paths_glob)

        # resolve the true count of bodies
        self._num_bodies = self.body_physx_view.count // self._num_envs
        # check that volume points sensor succeeded
        if self._num_bodies != len(body_prims):
            raise RuntimeError(
                "Failed to initialize volume points sensor for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_paths_glob}"
            )

        # initialize the volume points data
        self._volume_points_pattern: torch.Tensor = self.cfg.points_generator.func(self.cfg.points_generator).to(
            self.device
        )  # (P, 3)
        self._data: VolumePointsData = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )

        # initialize handlers to access virtual obstacles
        self._virtual_obstacles: dict = dict()

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)

        # update the volume points data
        self._refresh_volume_points(env_ids)
        # update the penetration depth data
        self._refresh_penetration_offset(env_ids)

    def _refresh_volume_points(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh the volume points data. If env_ids is None, refresh all environments."""

        body_poses = wp.to_torch(self.body_physx_view.get_transforms()).view(-1, self.num_bodies, 7)[env_ids]
        body_vels = wp.to_torch(self.body_physx_view.get_velocities()).view(-1, self.num_bodies, 6)[env_ids]
        self._data.pos_w[env_ids] = body_poses[..., :3]  # (N_, B, 3)
        self._data.quat_w[env_ids] = body_poses[..., 3:]  # (N_, B, 4), XYZW
        self._data.vel_w[env_ids] = body_vels[..., :3]  # (N_, B, 3)
        self._data.ang_vel_w[env_ids] = body_vels[..., 3:]  # (N_, B, 3)

        # calculate the volume points positions and velocities in world frame
        N_B = self._data.pos_w[env_ids].shape[0] * self._data.pos_w[env_ids].shape[1]  # (N_*B)
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(N_B, -1, -1),  # (N_*B, P, 3)
            self._data.pos_w[env_ids].flatten(0, 1),  # (N_*B, 3)
            self._data.quat_w[env_ids].flatten(0, 1),  # (N_*B, 4)
        ).reshape(
            *self._data.pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3
        )  # (N_, B, P, 3)
        self._data.points_pos_w[env_ids] = points_pos_w
        points_vel_w = self._data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()  # (N_, B, P, 3)
        points_vel_w += torch.linalg.cross(
            self._data.ang_vel_w[env_ids].unsqueeze(-2),
            (points_pos_w - self._data.pos_w[env_ids].unsqueeze(-2)),
            dim=-1,
        )  # (N_, B, P, 3)
        self._data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: Sequence[int]) -> None:
        """Refresh the penetration depth data. If env_ids is None, refresh all environments."""

        penetration_offset_buf: torch.Tensor = self._data.penetration_offset[env_ids]
        penetration_offset_buf[:] = 0.0
        penetration_depth_buf = torch.zeros_like(penetration_offset_buf[..., 0])  # (N_, B, P)

        for virtual_obstacle in self._virtual_obstacles.values():
            # get the penetration offset for the virtual obstacle
            penetration_offset = virtual_obstacle.get_points_penetration_offset(
                self._data.points_pos_w[env_ids].flatten(0, 2)
            )  # (N_*B*P, 3)
            penetration_offset = penetration_offset.reshape(self._data.points_pos_w[env_ids].shape)  # (N_, B, P, 3)
            penetration_depth = torch.norm(penetration_offset, dim=-1)  # (N_, B, P)
            # update the penetration offset if the depth is greater than the current depth
            mask = penetration_depth > penetration_depth_buf  # (N_, B, P)
            penetration_depth_buf[mask] = penetration_depth[mask]
            penetration_offset_buf[mask] = penetration_offset[mask]

        self._data.penetration_offset[env_ids] = penetration_offset_buf

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.points_visualizer.set_visibility(True)
        else:
            if hasattr(self, "points_visualizer"):
                self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self.body_physx_view is None:
            return

        points = self._data.points_pos_w.view(-1, 3)  # (N_*B*P, 3)
        penetrated = torch.norm(self._data.penetration_offset.view(-1, 3), dim=-1) > 0.0  # (N_*B*P,)

        # add penetrated points if none
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=self.device)], dim=0)

        self.points_visualizer.visualize(
            translations=points,
            marker_indices=penetrated.long(),
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        if hasattr(self, "point_visualizer"):
            delattr(self, "points_visualizer")
        self._physics_sim_view = None
        self._body_physx_view = None

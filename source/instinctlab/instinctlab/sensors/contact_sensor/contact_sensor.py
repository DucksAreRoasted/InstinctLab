from __future__ import annotations

import re

from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.contact_sensor import ContactSensor
from pxr import UsdPhysics

from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source


class HierarchicalContactSensor(ContactSensor):
    """PhysX contact sensor for bodies nested below an asset root.

    Isaac Sim's URDF importer 3.0 authors rigid links along the robot's
    kinematic hierarchy. The upstream PhysX contact sensor discovers those
    links recursively, but its view expression assumes that they are direct
    children of the configured parent. This sensor keeps the upstream data and
    update implementation while constructing the views from the discovered
    bodies' complete relative paths.

    Note:
        This is a temporary workaround for the pinned Isaac Lab 3.0.0-beta2
        contact-sensor bug described above. Remove this class and use the
        upstream contact sensor once it preserves complete descendant paths
        when constructing its PhysX views.
    """

    @property
    def body_names(self) -> list[str]:
        """Ordered rigid-body names in one environment."""
        return self._body_names

    def _initialize_impl(self) -> None:
        SensorBase._initialize_impl(self)

        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        if self._physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")

        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            return (
                bool(name_pattern.fullmatch(prim.GetName()))
                and ("PhysxContactReportAPI" in prim.GetAppliedSchemas())
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )

        matches = resolve_matching_prims_from_source(parent_expr)
        if not matches:
            raise RuntimeError(f"No prim found at '{parent_expr}'.")
        asset_prim, destination_root_expr = matches[0]
        source_root = asset_prim.GetPath().pathString
        body_prims = get_all_matching_child_prims(
            source_root,
            predicate=has_contact_report,
            traverse_instance_prims=False,
        )
        if not body_prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        relative_body_paths = [prim.GetPath().pathString[len(source_root) :] for prim in body_prims]
        if ".*" in destination_root_expr:
            environment_roots = [
                destination_root_expr.replace(".*", str(env_id), 1) for env_id in range(self._num_envs)
            ]
        elif "*" in destination_root_expr:
            environment_roots = [destination_root_expr.replace("*", str(env_id), 1) for env_id in range(self._num_envs)]
        elif self._num_envs == 1:
            environment_roots = [destination_root_expr]
        else:
            raise RuntimeError(
                f"Hierarchical contact sensor cannot expand the environment namespace in '{destination_root_expr}'."
            )

        # PhysX preserves the input-pattern order. Supplying one multi-environment pattern per body
        # groups the view body-major, while the contact kernels require environment-major storage.
        # Expand concrete paths explicitly so each environment occupies one contiguous body block.
        body_paths = [root + relative_path for root in environment_roots for relative_path in relative_body_paths]
        self._body_names = [prim.GetName() for prim in body_prims]
        filter_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]

        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_paths)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_paths,
            filter_patterns=filter_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(body_prims) * self._num_envs,
        )
        self._num_sensors = self.body_physx_view.count // self._num_envs
        actual_body_paths = list(self.body_physx_view.prim_paths)
        if self._num_sensors != len(body_prims) or actual_body_paths != body_paths:
            raise RuntimeError(
                "Failed to initialize contact reporter for hierarchical bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tExpected prim paths: {body_paths}"
                f"\n\tResolved prim paths: {actual_body_paths}"
                f"\n\tExpected per env   : {len(body_prims)}"
                f"\n\tActual per env     : {self._num_sensors}"
            )

        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            tracked_data = "contact points" if self.cfg.track_contact_points else "friction forces"
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track "
                    f"{tracked_data}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    f"Please set it to a value greater than 0 to track {tracked_data}."
                )

        self._create_buffers()

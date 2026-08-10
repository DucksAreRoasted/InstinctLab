# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Spawn functions for mesh files."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import carb
from pxr import Usd, UsdPhysics

from isaaclab.sim import converters, schemas

# Import the private helper from IsaacLab - we cannot add spawn_from_mesh to IsaacLab
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone

if TYPE_CHECKING:
    from . import from_files_cfg


def _activate_hierarchical_contact_sensors(root_prim: Usd.Prim) -> None:
    """Activate contact reporting on every nested Importer 3.0 rigid body."""
    rigid_prims: list[Usd.Prim] = []
    queue = [root_prim]
    while queue:
        prim = queue.pop(0)
        queue.extend(prim.GetChildren())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_prims.append(prim)

    if not rigid_prims:
        raise RuntimeError(f"No rigid bodies found below '{root_prim.GetPath()}'.")
    for prim in rigid_prims:
        schemas.activate_contact_sensors(prim.GetPath().pathString)


def _configure_tensor_leaf_matching(strict: bool) -> None:
    if strict:
        carb.settings.get_settings().set_bool("/physics/tensors/recursiveLeafPatternMatch", False)


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a prebuilt USD asset with Importer 3.0 nested-link support."""
    if cfg.required_asset_digest and not os.path.isfile(cfg.usd_path):
        raise FileNotFoundError(
            f"Required immutable asset '{cfg.required_asset_digest}' is missing at '{cfg.usd_path}'."
            f" Build it first with: {cfg.build_command}"
        )
    _configure_tensor_leaf_matching(cfg.strict_tensor_leaf_pattern_matching)
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    if cfg.activate_contact_sensors:
        _activate_hierarchical_contact_sensors(prim)
    return prim


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a standard Importer 3.0 URDF asset with nested-link contact reporting."""
    _configure_tensor_leaf_matching(cfg.strict_tensor_leaf_pattern_matching)
    urdf_converter = converters.UrdfConverter(cfg)
    prim = _spawn_from_usd_file(prim_path, urdf_converter.usd_path, cfg, translation, orientation, **kwargs)
    if cfg.activate_contact_sensors:
        _activate_hierarchical_contact_sensors(prim)
    return prim


@clone
def spawn_from_mesh(
    prim_path: str,
    cfg: from_files_cfg.MeshFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a mesh file (OBJ, STL, FBX) and override the settings with the given config.

    It uses the :class:`MeshConverter` class to create a USD file from the mesh. This file is then
    imported at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new
    prim or throw an error that the prim already exists. Instead, it just takes the existing prim
    and overrides the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single config and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex
            pattern, then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the translation specified in the generated USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim.
            Defaults to None, in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the mesh file does not exist at the given path.
    """
    mesh_converter = converters.MeshConverter(cfg)
    spawn_cfg = cfg if cfg.apply_collision_props_at_spawn else cfg.replace(collision_props=None)
    return _spawn_from_usd_file(prim_path, mesh_converter.usd_path, spawn_cfg, translation, orientation, **kwargs)

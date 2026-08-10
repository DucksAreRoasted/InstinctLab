"""Helpers for constructing PhysX tensor views from Isaac Lab prim expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source

if TYPE_CHECKING:
    import omni.physics.tensors.api as physx


def get_articulation_view(
    prim_path: str,
    physics_sim_view: physx.SimulationView,
) -> physx.ArticulationView:
    """Create a PhysX articulation view for the single articulation below an asset expression."""
    from pxr import UsdPhysics

    matches = resolve_matching_prims_from_source(prim_path)
    if not matches:
        raise RuntimeError(f"No asset prim found at path expression: {prim_path}")

    asset_prim, asset_expr = matches[0]
    asset_path = asset_prim.GetPath().pathString
    articulation_roots = get_all_matching_child_prims(
        asset_path,
        predicate=lambda prim: bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI)),
        traverse_instance_prims=False,
    )
    if len(articulation_roots) != 1:
        matched_paths = [prim.GetPath().pathString for prim in articulation_roots]
        raise RuntimeError(
            f"Expected exactly one ArticulationRootAPI prim below '{asset_path}' "
            f"(resolved from '{prim_path}'), found {len(articulation_roots)}: {matched_paths}."
        )

    root_path = articulation_roots[0].GetPath().pathString
    root_expr = asset_expr + root_path[len(asset_path) :]
    articulation_view: physx.ArticulationView = physics_sim_view.create_articulation_view(root_expr.replace(".*", "*"))
    if articulation_view._backend is None:
        raise RuntimeError(f"Failed to create a PhysX articulation view at: {root_expr}")
    return articulation_view

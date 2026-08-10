"""Helpers for assets produced by Isaac Sim's URDF importer 3.0."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import cache


@cache
def _urdf_link_lineages(urdf_path: str) -> dict[str, tuple[str, ...]]:
    """Return each URDF link's root-to-link lineage."""
    robot = ET.parse(urdf_path).getroot()
    link_names = {link.attrib["name"] for link in robot.findall("link")}
    parent_by_child: dict[str, str] = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            parent_by_child[child.attrib["link"]] = parent.attrib["link"]

    lineages: dict[str, tuple[str, ...]] = {}
    for link_name in link_names:
        lineage = [link_name]
        seen = {link_name}
        while lineage[-1] in parent_by_child:
            parent_name = parent_by_child[lineage[-1]]
            if parent_name in seen:
                raise ValueError(f"Cycle detected in URDF link tree at '{parent_name}': {urdf_path}")
            if parent_name not in link_names:
                raise ValueError(f"Unknown parent link '{parent_name}' in URDF: {urdf_path}")
            lineage.append(parent_name)
            seen.add(parent_name)
        lineages[link_name] = tuple(reversed(lineage))
    return lineages


def urdf_importer_link_prim_path(
    urdf_path: str,
    link_name: str,
    asset_prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> str:
    """Build a link prim path for the hierarchical output of URDF importer 3.0.

    Isaac Sim 6 authors rigid links below the asset's ``Geometry`` scope following
    the URDF kinematic tree instead of placing every link directly below the asset.
    """
    try:
        lineage = _urdf_link_lineages(urdf_path)[link_name]
    except KeyError as exc:
        raise ValueError(f"Link '{link_name}' does not exist in URDF: {urdf_path}") from exc
    return f"{asset_prim_path}/Geometry/{'/'.join(lineage)}"

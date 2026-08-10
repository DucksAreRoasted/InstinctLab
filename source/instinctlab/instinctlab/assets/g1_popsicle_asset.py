# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Build and resolve the immutable G1 popsicle USD asset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASSET_NAME = "g1_29dof_torsobase_popsicle"
POSTPROCESSOR_VERSION = 1
ISAACLAB_REVISION = "6a7acb0320a0bdc15b13e44e83b575e00797faf4"
CONVERTER_REVISION = "Isaac Sim 6.0.1 / URDF USD Converter v0.1.3"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_RESOURCE_ROOT = _PACKAGE_ROOT / "assets" / "resources" / "unitree_g1"
SOURCE_URDF_PATH = _RESOURCE_ROOT / "urdf" / f"{ASSET_NAME}.urdf"
SHOE_SOURCE_URDF_PATH = _PACKAGE_ROOT / "tasks" / "parkour" / "urdf" / "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
CAPSULE_MANIFEST_PATH = _RESOURCE_ROOT / "manifests" / f"{ASSET_NAME}_capsules.json"
BASELINE_MANIFEST_PATH = _RESOURCE_ROOT / "manifests" / f"{ASSET_NAME}_isaaclab_2_3_2.json"

CONVERTER_CONFIG: dict[str, Any] = {
    "collision_from_visuals": False,
    "collision_type": "Convex Hull",
    "debug_mode": False,
    "fix_base": False,
    "joint_drive": {
        "drive_type": "force",
        "target_type": "position",
        "stiffness": 0.0,
        "damping": 0.0,
    },
    "link_density": 0.0,
    "merge_fixed_joints": True,
    "merge_mesh": False,
    "robot_type": "Default",
    "run_asset_transformer": True,
    "run_multi_physics_conversion": True,
    "self_collision": False,
}


@dataclass(frozen=True)
class G1PopsicleAssetSpec:
    """Inputs that distinguish one immutable G1 popsicle asset variant."""

    cache_namespace: str
    source_urdf_path: Path


G1_POPSICLE_SPEC = G1PopsicleAssetSpec("g1_popsicle", SOURCE_URDF_PATH)
G1_POPSICLE_SHOE_SPEC = G1PopsicleAssetSpec("g1_popsicle_with_shoe", SHOE_SOURCE_URDF_PATH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _referenced_meshes(spec: G1PopsicleAssetSpec) -> list[Path]:
    root = ET.parse(spec.source_urdf_path).getroot()
    mesh_paths = {
        (spec.source_urdf_path.parent / mesh.attrib["filename"]).resolve()
        for mesh in root.findall(".//mesh")
        if not mesh.attrib["filename"].startswith("package://")
    }
    missing = [str(path) for path in sorted(mesh_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G1 popsicle URDF references missing meshes: {missing}")
    return sorted(mesh_paths)


def asset_inputs(spec: G1PopsicleAssetSpec = G1_POPSICLE_SPEC) -> dict[str, Any]:
    """Return the deterministic inputs that identify the generated asset."""
    files = [spec.source_urdf_path, CAPSULE_MANIFEST_PATH, BASELINE_MANIFEST_PATH, *_referenced_meshes(spec)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G1 popsicle asset inputs are missing: {missing}")
    return {
        "asset_name": ASSET_NAME,
        "asset_variant": spec.cache_namespace,
        "postprocessor_version": POSTPROCESSOR_VERSION,
        "isaaclab_revision": ISAACLAB_REVISION,
        "converter_revision": CONVERTER_REVISION,
        "converter_config": CONVERTER_CONFIG,
        "files": {str(path.relative_to(_PACKAGE_ROOT)): _sha256_file(path) for path in sorted(files)},
    }


def asset_digest(spec: G1PopsicleAssetSpec = G1_POPSICLE_SPEC) -> str:
    """Return the SHA-256 digest for the complete asset build inputs."""
    payload = json.dumps(asset_inputs(spec), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def asset_cache_root() -> Path:
    """Return the configured shared asset-cache root."""
    configured = os.environ.get("INSTINCTLAB_ASSET_CACHE")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".cache" / "instinctlab" / "assets"


def resolved_asset_dir(spec: G1PopsicleAssetSpec = G1_POPSICLE_SPEC) -> Path:
    """Return the immutable digest-scoped directory for this asset revision."""
    return asset_cache_root() / spec.cache_namespace / asset_digest(spec)


def resolved_asset_path(spec: G1PopsicleAssetSpec = G1_POPSICLE_SPEC) -> Path:
    """Return the expected final USD path without building it."""
    return resolved_asset_dir(spec) / f"{spec.source_urdf_path.stem}.usda"


def _load_capsule_manifest() -> dict[str, Any]:
    with CAPSULE_MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported capsule manifest schema: {manifest.get('schema_version')}")
    return manifest


def _capsule_extent(radius: float, height: float, axis: str):
    from pxr import Gf

    half_length = 0.5 * height + radius
    half_extents = {
        "X": (half_length, radius, radius),
        "Y": (radius, half_length, radius),
        "Z": (radius, radius, half_length),
    }
    try:
        x, y, z = half_extents[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported capsule axis: {axis}") from exc
    return [Gf.Vec3f(-x, -y, -z), Gf.Vec3f(x, y, z)]


def _postprocess_collision_layer(base_layer_path: Path) -> list[str]:
    """Replace exactly the manifest-selected defining-layer cylinders."""
    from pxr import Usd, UsdGeom

    manifest = _load_capsule_manifest()
    entries = manifest["capsules"]
    expected_paths = {entry["path"] for entry in entries}
    if len(expected_paths) != len(entries):
        raise ValueError("Capsule manifest contains duplicate prim paths.")

    stage = Usd.Stage.Open(str(base_layer_path))
    if stage is None:
        raise RuntimeError(f"Failed to open defining collision layer: {base_layer_path}")
    actual_paths = {prim.GetPath().pathString for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Cylinder)}
    if actual_paths != expected_paths:
        raise RuntimeError(
            "G1 popsicle cylinder manifest mismatch."
            f"\nMissing paths: {sorted(expected_paths - actual_paths)}"
            f"\nUnexpected paths: {sorted(actual_paths - expected_paths)}"
        )

    converted_paths = []
    for entry in entries:
        expected_schemas = entry.get("applied_schemas", manifest["default_applied_schemas"])
        prim = stage.GetPrimAtPath(entry["path"])
        cylinder = UsdGeom.Cylinder(prim)
        radius = float(cylinder.GetRadiusAttr().Get())
        height = float(cylinder.GetHeightAttr().Get())
        axis = str(cylinder.GetAxisAttr().Get())
        expected = (float(entry["radius"]), float(entry["height"]), str(entry["axis"]))
        actual = (radius, height, axis)
        if abs(radius - expected[0]) > 1.0e-8 or abs(height - expected[1]) > 1.0e-8 or axis != expected[2]:
            raise RuntimeError(f"Geometry mismatch at {entry['path']}: expected {expected}, found {actual}.")
        if sorted(prim.GetAppliedSchemas()) != sorted(expected_schemas):
            raise RuntimeError(
                f"Applied-schema mismatch at {entry['path']}:"
                f" expected {expected_schemas}, found {prim.GetAppliedSchemas()}."
            )

        relationships = {
            relationship.GetName(): list(relationship.GetTargets()) for relationship in prim.GetRelationships()
        }
        if not prim.SetTypeName("Capsule"):
            raise RuntimeError(f"Failed to change cylinder type at {entry['path']}.")
        capsule = UsdGeom.Capsule(prim)
        capsule.GetRadiusAttr().Set(radius)
        capsule.GetHeightAttr().Set(height)
        capsule.GetAxisAttr().Set(axis)
        capsule.GetExtentAttr().Set(_capsule_extent(radius, height, axis))
        if sorted(prim.GetAppliedSchemas()) != sorted(expected_schemas):
            raise RuntimeError(f"Postprocessing changed applied schemas at {entry['path']}.")
        for name, targets in relationships.items():
            if list(prim.GetRelationship(name).GetTargets()) != targets:
                raise RuntimeError(f"Postprocessing changed relationship '{name}' at {entry['path']}.")
        converted_paths.append(entry["path"])

    stage.GetRootLayer().Save()
    validation_stage = Usd.Stage.Open(str(base_layer_path))
    capsule_paths = {prim.GetPath().pathString for prim in validation_stage.TraverseAll() if prim.IsA(UsdGeom.Capsule)}
    remaining_cylinders = [
        prim.GetPath().pathString for prim in validation_stage.TraverseAll() if prim.IsA(UsdGeom.Cylinder)
    ]
    if capsule_paths != expected_paths or remaining_cylinders:
        raise RuntimeError(
            f"Capsule validation failed: capsules={sorted(capsule_paths)}, cylinders={remaining_cylinders}."
        )
    return converted_paths


def _validate_existing_asset(final_path: Path, digest: str, spec: G1PopsicleAssetSpec) -> None:
    metadata_path = final_path.parent / "instinctlab_asset.json"
    if not final_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete digest-scoped asset directory: {final_path.parent}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("asset_digest") != digest or metadata.get("inputs") != asset_inputs(spec):
        raise RuntimeError(f"Asset metadata does not match digest-scoped path: {final_path.parent}")


def build_asset(spec: G1PopsicleAssetSpec = G1_POPSICLE_SPEC) -> Path:
    """Build, validate, and atomically promote the digest-scoped final USD."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    digest = asset_digest(spec)
    final_dir = resolved_asset_dir(spec)
    final_path = resolved_asset_path(spec)
    if final_dir.exists():
        _validate_existing_asset(final_path, digest, spec)
        return final_path

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"instinctlab-{spec.cache_namespace}-") as staging_dir_name:
        staging_dir = Path(staging_dir_name)
        cfg = UrdfConverterCfg(
            asset_path=str(spec.source_urdf_path),
            usd_dir=str(staging_dir),
            force_usd_conversion=True,
            fix_base=CONVERTER_CONFIG["fix_base"],
            link_density=CONVERTER_CONFIG["link_density"],
            merge_fixed_joints=CONVERTER_CONFIG["merge_fixed_joints"],
            collision_from_visuals=CONVERTER_CONFIG["collision_from_visuals"],
            collision_type=CONVERTER_CONFIG["collision_type"],
            self_collision=CONVERTER_CONFIG["self_collision"],
            merge_mesh=CONVERTER_CONFIG["merge_mesh"],
            robot_type=CONVERTER_CONFIG["robot_type"],
            run_asset_transformer=CONVERTER_CONFIG["run_asset_transformer"],
            run_multi_physics_conversion=CONVERTER_CONFIG["run_multi_physics_conversion"],
            debug_mode=CONVERTER_CONFIG["debug_mode"],
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type=CONVERTER_CONFIG["joint_drive"]["drive_type"],
                target_type=CONVERTER_CONFIG["joint_drive"]["target_type"],
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=CONVERTER_CONFIG["joint_drive"]["stiffness"],
                    damping=CONVERTER_CONFIG["joint_drive"]["damping"],
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        generated_path = Path(converter.usd_path)
        generated_dir = generated_path.parent
        base_layer_path = generated_dir / "payloads" / "base.usda"
        converted_paths = _postprocess_collision_layer(base_layer_path)

        promotion_dir = final_dir.parent / f".{digest}.tmp-{os.getpid()}"
        if promotion_dir.exists():
            shutil.rmtree(promotion_dir)
        shutil.copytree(generated_dir, promotion_dir)
        metadata = {
            "asset_digest": digest,
            "root_usd": generated_path.name,
            "capsule_count": len(converted_paths),
            "capsule_paths": converted_paths,
            "inputs": asset_inputs(spec),
        }
        with (promotion_dir / "instinctlab_asset.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        try:
            promotion_dir.rename(final_dir)
        except FileExistsError:
            shutil.rmtree(promotion_dir)

    _validate_existing_asset(final_path, digest, spec)
    return final_path


G1_POPSICLE_ASSET_DIGEST = asset_digest(G1_POPSICLE_SPEC)
G1_POPSICLE_USD_PATH = str(resolved_asset_path(G1_POPSICLE_SPEC))
G1_POPSICLE_SHOE_ASSET_DIGEST = asset_digest(G1_POPSICLE_SHOE_SPEC)
G1_POPSICLE_SHOE_USD_PATH = str(resolved_asset_path(G1_POPSICLE_SHOE_SPEC))

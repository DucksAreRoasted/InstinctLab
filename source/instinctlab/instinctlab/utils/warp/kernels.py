from typing import Any

import warp as wp


@wp.kernel(enable_backward=False)
def points_penetrate_cylinder_kernel(
    points: wp.array(dtype=wp.vec3),
    cylinder_start: wp.array(dtype=wp.vec3),
    cylinder_end: wp.array(dtype=wp.vec3),
    cylinder_thinkness: wp.array(dtype=wp.float32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_indices: wp.array(dtype=wp.int32),
    grid_res: wp.vec3i,
    bbox_min: wp.vec3,
    cell_size: wp.vec3,
    penetrate_offset: wp.array(dtype=wp.vec3),
):
    """Compute the penetration depth of points into cylinders in a grid. Return the maximum depth for each point if it
    penetrates any cylinder.
    Args:
        points: Array of points to check for penetration. shape (N, 3) where N is the number of points.
        cylinders: Array of cylinders defined by start and end points and radius. shape (M, 7) where M is the number of cylinders.
        cell_offsets: Offsets for each grid cell in the flattened grid. shape (grid_res^3 + 1,)
        cell_indices: Indices of cylinders in each grid cell. shape (N, 8)
        grid_res: Resolution of the grid.
        bbox_min: Minimum coordinates of the bounding box for the grid. shape (3,)
        cell_size: Size of each grid cell. shape (3,)
        penetrate_offset: Output array to store the penetration offset from the surface of the cylinder to each point.
    """
    tid = wp.tid()
    p = points[tid]

    bbox_min_to_p = p - bbox_min
    ix = int(bbox_min_to_p[0] / cell_size[0])
    iy = int(bbox_min_to_p[1] / cell_size[1])
    iz = int(bbox_min_to_p[2] / cell_size[2])

    depth = float(0.0)
    penetrate_offset_ = wp.vec3(0.0, 0.0, 0.0)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x = ix + dx
                y = iy + dy
                z = iz + dz

                if x < 0 or x >= grid_res.x or y < 0 or y >= grid_res.y or z < 0 or z >= grid_res.z:
                    continue

                flat = x * grid_res.y * grid_res.z
                flat = flat + y * grid_res.z
                flat = flat + z

                start = cell_offsets[flat]
                end = cell_offsets[flat + 1]

                for i in range(start, end):
                    cid = cell_indices[i]
                    a = cylinder_start[cid]
                    b = cylinder_end[cid]
                    r = cylinder_thinkness[cid]

                    ab = b - a
                    ab_len = wp.length(ab)
                    ab_dir = ab / ab_len
                    ap = p - a
                    t = wp.dot(ap, ab_dir)

                    if t < 0.0 or t > ab_len:
                        # points outside the cylinder segment
                        continue

                    # Project point onto the cylinder segment
                    proj = a + t * ab_dir
                    dist = wp.length(p - proj)

                    if dist < r:
                        d = r - dist
                        if d > depth:
                            depth = d
                            offset_ = (proj - p) * (d / dist)
                            # direction from point to projected point
                            penetrate_offset_.x = offset_.x
                            penetrate_offset_.y = offset_.y
                            penetrate_offset_.z = offset_.z

    if depth > 0.0:
        penetrate_offset[tid] = penetrate_offset_


@wp.kernel(enable_backward=False)
def copy_flat_mesh_transforms_kernel(
    source_transforms: wp.array(dtype=wp.transformf),
    entity_indices: wp.array(dtype=wp.int32),
    num_entities: int,
    mesh_positions: wp.array(dtype=wp.vec3f),
    mesh_rotations: wp.array(dtype=wp.quatf),
):
    """Copy a tracked physics view into indexed flat mesh-entity records."""
    view_index = wp.tid()
    entity_index = entity_indices[view_index]
    if entity_index < 0 or entity_index >= num_entities:
        return

    transform = source_transforms[view_index]
    mesh_positions[entity_index] = wp.transform_get_translation(transform)
    mesh_rotations[entity_index] = wp.transform_get_rotation(transform)


@wp.kernel(enable_backward=False)
def raycast_flat_mesh_groups_min_distance_kernel(
    env_mask: wp.array(dtype=wp.bool),
    ray_world_ids: wp.array2d(dtype=wp.int32),
    world_mesh_indices: wp.array(dtype=wp.int32),
    world_mesh_offsets: wp.array(dtype=wp.int32),
    meshes: wp.array(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3f),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    mesh_positions: wp.array(dtype=wp.vec3f),
    mesh_rotations: wp.array(dtype=wp.quatf),
    max_dist: float,
    min_dist: float,
    num_worlds: int,
    num_entities: int,
    num_world_mesh_indices: int,
    num_rays: int,
    return_normal: int,
    return_face_id: int,
    return_mesh_id: int,
):
    """Ray-cast a ray against the precomputed flat mesh set for its world."""
    ray_batch_id, ray_id = wp.tid()
    if ray_batch_id < 0 or ray_batch_id >= num_worlds or ray_id < 0 or ray_id >= num_rays:
        return

    world_id = ray_world_ids[ray_batch_id, ray_id]
    if world_id < 0 or world_id >= num_worlds or not env_mask[world_id]:
        return

    group_start = world_mesh_offsets[world_id]
    group_end = world_mesh_offsets[world_id + 1]
    if group_start < 0 or group_end < group_start or group_end > num_world_mesh_indices:
        return

    ray_start = ray_starts[ray_batch_id, ray_id]
    ray_direction = ray_directions[ray_batch_id, ray_id]
    closest_distance = float(max_dist)
    closest_normal = wp.vec3f()
    closest_face_id = int(-1)
    closest_mesh_id = int(-1)

    # The world membership is fixed for a rollout. One thread owns one ray and
    # deterministically reduces hits over only that world's flat entity indices.
    for group_index in range(group_start, group_end):
        mesh_id = world_mesh_indices[group_index]
        if mesh_id < 0 or mesh_id >= num_entities:
            continue

        mesh_pose = wp.transform(mesh_positions[mesh_id], mesh_rotations[mesh_id])
        mesh_pose_inv = wp.transform_inverse(mesh_pose)
        start_local = wp.transform_point(mesh_pose_inv, ray_start)
        direction_local = wp.transform_vector(mesh_pose_inv, ray_direction)
        query = wp.mesh_query_ray(meshes[mesh_id], start_local, direction_local, max_dist)

        if query.result and query.t > min_dist and query.t < closest_distance:
            closest_distance = query.t
            closest_mesh_id = mesh_id
            if return_normal == 1:
                closest_normal = wp.transform_vector(mesh_pose, query.normal)
            if return_face_id == 1:
                closest_face_id = query.face

    if closest_mesh_id >= 0:
        ray_distance[ray_batch_id, ray_id] = closest_distance
        ray_hits[ray_batch_id, ray_id] = ray_start + closest_distance * ray_direction
        if return_normal == 1:
            ray_normal[ray_batch_id, ray_id] = closest_normal
        if return_face_id == 1:
            ray_face_id[ray_batch_id, ray_id] = closest_face_id
        if return_mesh_id == 1:
            ray_mesh_id[ray_batch_id, ray_id] = wp.int16(closest_mesh_id)
    else:
        if return_face_id == 1:
            ray_face_id[ray_batch_id, ray_id] = -1
        if return_mesh_id == 1:
            ray_mesh_id[ray_batch_id, ray_id] = wp.int16(-1)

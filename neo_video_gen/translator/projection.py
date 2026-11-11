"""Deterministic pinhole-camera projection helpers."""
from __future__ import annotations

import numpy as np


def project_world_points(points_world, world_to_camera, intrinsics):
    """Return pixel coordinates and front-of-camera flags for Nx3 world points."""
    points = np.asarray(points_world, dtype=np.float64)
    extrinsic = np.asarray(world_to_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N, 3]")
    if extrinsic.shape != (4, 4) or k.shape != (3, 3):
        raise ValueError("Expected a 4x4 extrinsic matrix and 3x3 intrinsics matrix")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera_points = (extrinsic @ homogeneous.T).T[:, :3]
    projected = (k @ camera_points.T).T
    depth = projected[:, 2]
    visible = depth > 1e-8
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[visible] = projected[visible, :2] / depth[visible, None]
    return pixels, visible

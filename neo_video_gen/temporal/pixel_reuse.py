"""Forward-warp source RGB with geometry-derived motion and occlusion masks."""
from __future__ import annotations
import numpy as np

def _plane(value, name, height, width, required=True):
    if value is None and not required:
        return None
    array = np.asarray(value)
    if array.shape == (1, height, width):
        array = array[0]
    if array.shape != (height, width):
        raise ValueError(f"{name} must have shape [H,W] or [1,H,W]")
    return array.astype(np.float32, copy=False)

def warp_rgb_forward(
    source_rgb, flow_forward, source_visibility, target_visible_from_source,
    source_depth, depth_delta, target_coverage=None,
):
    """Splat source pixels into target using normalized dense flow and z-buffering.

    Uncovered target pixels are returned as holes; this function does not invent
    or inpaint unseen appearance.
    """
    image = np.asarray(source_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("source_rgb must be HxWx3 or HxWx4")
    height, width = image.shape[:2]
    flow = np.asarray(flow_forward, dtype=np.float32)
    if flow.shape != (2, height, width):
        raise ValueError("flow_forward must have shape [2,H,W]")
    src_vis = _plane(source_visibility, "source_visibility", height, width)
    tgt_vis = _plane(target_visible_from_source, "target_visible_from_source", height, width)
    z0 = _plane(source_depth, "source_depth", height, width)
    dz = _plane(depth_delta, "depth_delta", height, width)
    tgt_cov = _plane(target_coverage, "target_coverage", height, width, required=False)

    valid = (
        (src_vis > 0.5) & (tgt_vis > 0.5)
        & np.isfinite(flow).all(axis=0)
        & np.isfinite(z0) & np.isfinite(dz)
        & ((z0 + dz) > 0)
    )
    if not np.any(valid):
        raise ValueError("No source pixels are valid for forward warping")

    yy, xx = np.mgrid[0:height, 0:width]
    dst_x = xx.astype(np.float32) + 0.5 + flow[0] * (width / 2.0)
    dst_y = yy.astype(np.float32) + 0.5 + flow[1] * (height / 2.0)
    sample_x, sample_y = dst_x - 0.5, dst_y - 0.5
    x0, y0 = np.floor(sample_x).astype(np.int32), np.floor(sample_y).astype(np.int32)
    fx, fy = sample_x - x0, sample_y - y0
    z_target = z0 + dz
    colors = image.astype(np.float32).reshape(-1, image.shape[2])
    depths = z_target.reshape(-1)
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    contributions = []
    neighbors = (
        (0, 0, (1.0-fx)*(1.0-fy)),
        (1, 0, fx*(1.0-fy)),
        (0, 1, (1.0-fx)*fy),
        (1, 1, fx*fy),
    )
    for ox, oy, weight_map in neighbors:
        tx, ty = x0 + ox, y0 + oy
        inside = (
            valid & (tx >= 0) & (tx < width)
            & (ty >= 0) & (ty < height) & (weight_map > 1e-8)
        )
        if not np.any(inside):
            continue
        source_idx = np.flatnonzero(inside.reshape(-1))
        target_idx = ty.reshape(-1)[source_idx] * width + tx.reshape(-1)[source_idx]
        weight = weight_map.reshape(-1)[source_idx].astype(np.float32)
        z = depths[source_idx]
        np.minimum.at(z_buffer, target_idx, z)
        contributions.append((target_idx, source_idx, weight, z))

    color_sum = np.zeros((height * width, image.shape[2]), dtype=np.float32)
    weight_sum = np.zeros(height * width, dtype=np.float32)
    for target_idx, source_idx, weight, z in contributions:
        front_depth = z_buffer[target_idx]
        tolerance = np.maximum(1e-4, front_depth * 1e-4)
        front = np.isfinite(front_depth) & (np.abs(z-front_depth) <= tolerance)
        if not np.any(front):
            continue
        idx = target_idx[front]
        accepted_weight = weight[front]
        np.add.at(color_sum, idx, colors[source_idx[front]] * accepted_weight[:, None])
        np.add.at(weight_sum, idx, accepted_weight)

    coverage = (weight_sum > 1e-8).reshape(height, width)
    warped = np.zeros_like(color_sum)
    nonzero = weight_sum > 1e-8
    warped[nonzero] = color_sum[nonzero] / weight_sum[nonzero, None]
    warped = warped.reshape(height, width, image.shape[2])
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        warped_rgb = np.clip(np.rint(warped), info.min, info.max).astype(image.dtype)
    else:
        warped_rgb = warped.astype(image.dtype, copy=False)
    expected = tgt_cov > 0.5 if tgt_cov is not None else np.ones_like(coverage)
    return {
        "warped_rgb": warped_rgb,
        "warped_coverage": coverage,
        "hole_mask": expected & ~coverage,
    }

"""Dense surface-motion fields from paired evaluated Blender meshes."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def _project(vertices_world, camera_world, view_frame, width, height):
    vertices = np.asarray(vertices_world, dtype=np.float64)
    camera = np.asarray(camera_world, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or camera.shape != (4, 4):
        raise ValueError("Expected vertices [V,3] and a 4x4 camera world matrix")
    points_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    local = (np.linalg.inv(camera) @ points_h.T).T[:, :3]
    depth = -local[:, 2]
    mode = view_frame.get("mode")
    if mode == "perspective_slopes":
        safe_z = np.where(depth > 1e-7, depth, 1.0)
        x_view, y_view = local[:, 0] / safe_z, local[:, 1] / safe_z
    elif mode == "orthographic_plane":
        x_view, y_view = local[:, 0], local[:, 1]
    else:
        raise ValueError(f"Unsupported view-frame mode: {mode!r}")
    left, right = float(view_frame["left"]), float(view_frame["right"])
    bottom, top = float(view_frame["bottom"]), float(view_frame["top"])
    if right <= left or top <= bottom:
        raise ValueError("Invalid camera view-frame bounds")
    u = (x_view - left) / (right - left)
    v = (top - y_view) / (top - bottom)
    pixels = np.stack([u * width, v * height], axis=1)
    valid = (depth > 1e-7) & np.isfinite(pixels).all(axis=1)
    pixels[~valid] = np.nan
    return pixels.astype(np.float32), depth.astype(np.float32), valid


def _samples(screen, depths, tri, width, height, perspective):
    p = screen[tri].astype(np.float64)
    z = depths[tri].astype(np.float64)
    if not np.isfinite(p).all() or not np.isfinite(z).all() or np.any(z <= 1e-7):
        return None
    xmin = max(0, int(np.ceil(float(p[:, 0].min()) - 0.5)))
    xmax = min(width - 1, int(np.floor(float(p[:, 0].max()) - 0.5)))
    ymin = max(0, int(np.ceil(float(p[:, 1].min()) - 0.5)))
    ymax = min(height - 1, int(np.floor(float(p[:, 1].max()) - 0.5)))
    if xmax < xmin or ymax < ymin:
        return None
    a, b, c = p
    den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(den) < 1e-10:
        return None
    gx, gy = np.meshgrid(
        np.arange(xmin, xmax + 1, dtype=np.float64) + 0.5,
        np.arange(ymin, ymax + 1, dtype=np.float64) + 0.5,
    )
    x, y = gx.ravel(), gy.ravel()
    w0 = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / den
    w1 = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / den
    bary = np.stack([w0, w1, 1.0-w0-w1], axis=1)
    inside = np.all(bary >= -1e-7, axis=1)
    if not np.any(inside):
        return None
    grid_w = xmax - xmin + 1
    xi = np.tile(np.arange(xmin, xmax + 1, dtype=np.int32), ymax-ymin+1)[inside]
    yi = np.repeat(np.arange(ymin, ymax + 1, dtype=np.int32), grid_w)[inside]
    bary = np.clip(bary[inside], 0.0, None)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    if perspective:
        invz = bary @ (1.0 / z)
        valid = invz > 1e-12
        if not np.any(valid):
            return None
        xi, yi, bary, invz = xi[valid], yi[valid], bary[valid], invz[valid]
        weights = (bary / z[None, :]) / invz[:, None]
        depth = 1.0 / invz
    else:
        weights = bary
        depth = bary @ z
    return xi, yi, weights, depth.astype(np.float32)


def _zbuffer(screen, depths, triangles, width, height, perspective):
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    for tri in triangles:
        sample = _samples(screen, depths, tri, width, height, perspective)
        if sample is None:
            continue
        xi, yi, _, depth = sample
        nearer = depth < zbuf[yi, xi]
        if np.any(nearer):
            zbuf[yi[nearer], xi[nearer]] = depth[nearer]
    return zbuf


def compute_dense_flow(
    source_vertices_world, target_vertices_world, triangles,
    source_camera_world, target_camera_world,
    source_view_frame, target_view_frame, width=64, height=64,
):
    """Return normalized source-to-target flow and geometry visibility maps."""
    src_v = np.asarray(source_vertices_world, dtype=np.float32)
    tgt_v = np.asarray(target_vertices_world, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    if src_v.shape != tgt_v.shape or src_v.ndim != 2 or src_v.shape[1] != 3:
        raise ValueError("Source and target vertices must have the same [V,3] shape")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape [T,3]")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(src_v)):
        raise ValueError("Triangle index is outside the vertex array")
    src_px, src_z, src_ok = _project(
        src_v, source_camera_world, source_view_frame, width, height
    )
    tgt_px, tgt_z, tgt_ok = _project(
        tgt_v, target_camera_world, target_view_frame, width, height
    )
    src_persp = source_view_frame["mode"] == "perspective_slopes"
    tgt_persp = target_view_frame["mode"] == "perspective_slopes"
    keep = (
        src_ok[triangles].all(axis=1) & tgt_ok[triangles].all(axis=1)
        if len(triangles) else np.zeros(0, dtype=bool)
    )
    triangles = triangles[keep]
    if not len(triangles):
        raise ValueError("No triangles are in front of both cameras")
    src_buf = _zbuffer(src_px, src_z, triangles, width, height, src_persp)
    tgt_buf = _zbuffer(tgt_px, tgt_z, triangles, width, height, tgt_persp)

    flow = np.zeros((2, height, width), dtype=np.float32)
    source_vis = np.zeros((1, height, width), dtype=np.float32)
    target_vis = np.zeros((1, height, width), dtype=np.float32)
    depth_delta = np.zeros((1, height, width), dtype=np.float32)
    source_depth = np.zeros((1, height, width), dtype=np.float32)

    for tri in triangles:
        sample = _samples(src_px, src_z, tri, width, height, src_persp)
        if sample is None:
            continue
        xi, yi, weights, z0 = sample
        zbuf = src_buf[yi, xi]
        tol0 = np.maximum(1e-4, zbuf * 1e-4)
        front = np.isfinite(zbuf) & (np.abs(z0-zbuf) <= tol0)
        if not np.any(front):
            continue
        xi, yi, weights, z0 = xi[front], yi[front], weights[front], z0[front]
        q = weights @ tgt_px[tri].astype(np.float64)
        z1 = (weights @ tgt_z[tri].astype(np.float64)).astype(np.float32)
        qx, qy = q[:, 0], q[:, 1]
        inside = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
        remains_visible = np.zeros(len(xi), dtype=bool)
        if np.any(inside):
            tx = np.floor(qx[inside]).astype(np.int32)
            ty = np.floor(qy[inside]).astype(np.int32)
            target_buffer = tgt_buf[ty, tx]
            tol1 = np.maximum(1e-4, target_buffer * 1e-4)
            remains_visible[inside] = (
                np.isfinite(target_buffer) & (np.abs(z1[inside]-target_buffer) <= tol1)
            )
        flow[0, yi, xi] = ((qx-(xi+0.5)) * (2.0/width)).astype(np.float32)
        flow[1, yi, xi] = ((qy-(yi+0.5)) * (2.0/height)).astype(np.float32)
        source_vis[0, yi, xi] = 1.0
        target_vis[0, yi, xi] = remains_visible.astype(np.float32)
        depth_delta[0, yi, xi] = z1-z0
        source_depth[0, yi, xi] = z0

    coverage = np.isfinite(tgt_buf).astype(np.float32)[None, :, :]
    if not np.any(source_vis):
        raise ValueError("No visible source surface pixels were rasterized")
    return {
        "flow_forward": flow,
        "source_visibility": source_vis,
        "target_visible_from_source": target_vis,
        "target_coverage": coverage,
        "depth_delta": depth_delta,
        "source_depth": source_depth,
    }


def _triangulate(geometry):
    offsets = np.asarray(geometry["face_offsets"], dtype=np.int64)
    indices = np.asarray(geometry["face_vertex_indices"], dtype=np.int64)
    result = []
    for i in range(max(0, len(offsets)-1)):
        poly = indices[offsets[i]:offsets[i+1]]
        for j in range(1, len(poly)-1):
            result.append([poly[0], poly[j], poly[j+1]])
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def _load_geometry(capture, record):
    descriptor = record.get("evaluated_geometry") or {}
    if not descriptor.get("file"):
        raise ValueError(f"Missing evaluated geometry for {record.get('name')!r}")
    path = Path(descriptor["file"])
    if not path.is_absolute():
        path = capture / path
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _mesh_records(state):
    return {
        obj["name"]: obj for obj in state.get("objects", [])
        if obj.get("type") == "MESH" and obj.get("evaluated_geometry")
    }


def _camera(state):
    name = state.get("camera")
    for obj in state.get("objects", []):
        if obj.get("type") == "CAMERA" and obj.get("name") == name:
            camera = obj.get("camera") or {}
            if "view_frame" not in camera:
                raise ValueError(f"Camera {name!r} is missing view_frame metadata")
            return np.asarray(obj["world_matrix"], dtype=np.float32), camera["view_frame"]
    raise ValueError(f"Active camera {name!r} was not captured")


def _combine_meshes(capture, state0, state1):
    r0, r1 = _mesh_records(state0), _mesh_records(state1)
    verts0, verts1, tris, matched, skipped = [], [], [], [], []
    offset = 0
    for name in sorted(set(r0) & set(r1)):
        g0, g1 = _load_geometry(capture, r0[name]), _load_geometry(capture, r1[name])
        v0 = np.asarray(g0["vertices_world"], dtype=np.float32)
        v1 = np.asarray(g1["vertices_world"], dtype=np.float32)
        f0 = np.asarray(g0["face_vertex_indices"], dtype=np.int64)
        f1 = np.asarray(g1["face_vertex_indices"], dtype=np.int64)
        o0 = np.asarray(g0["face_offsets"], dtype=np.int64)
        o1 = np.asarray(g1["face_offsets"], dtype=np.int64)
        if v0.shape != v1.shape or not np.array_equal(f0, f1) or not np.array_equal(o0, o1):
            skipped.append(name)
            continue
        tri = _triangulate(g0)
        if not len(tri):
            skipped.append(name)
            continue
        verts0.append(v0)
        verts1.append(v1)
        tris.append(tri + offset)
        offset += len(v0)
        matched.append(name)
    if not verts0:
        raise ValueError("No matching mesh topology found between the paired frames")
    return np.concatenate(verts0), np.concatenate(verts1), np.concatenate(tris), matched, skipped


def build_dense_capture_dataset(capture: Path, output: Path, size: int = 64):
    capture = Path(capture).expanduser().resolve()
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("engine") != "blender":
        raise ValueError("Expected a Blender capture manifest")
    frames = sorted(manifest.get("frames", []), key=lambda item: item["frame_index"])
    if len(frames) < 2:
        raise ValueError("At least two captured frames are required")
    keys = (
        "flow_forward", "source_visibility", "target_visible_from_source",
        "target_coverage", "depth_delta", "source_depth",
    )
    collected = {key: [] for key in keys}
    frame_pairs, matched_names, skipped_names = [], [], []
    for rec0, rec1 in zip(frames[:-1], frames[1:]):
        state0 = json.loads((capture / rec0["state_file"]).read_text(encoding="utf-8"))
        state1 = json.loads((capture / rec1["state_file"]).read_text(encoding="utf-8"))
        cam0, frame0 = _camera(state0)
        cam1, frame1 = _camera(state1)
        v0, v1, tri, matched, skipped = _combine_meshes(capture, state0, state1)
        result = compute_dense_flow(
            v0, v1, tri, cam0, cam1, frame0, frame1, width=size, height=size
        )
        for key in keys:
            collected[key].append(result[key])
        frame_pairs.append([rec0["frame_index"], rec1["frame_index"]])
        matched_names.append(matched)
        skipped_names.append(skipped)
    arrays = {key: np.asarray(values, dtype=np.float32) for key, values in collected.items()}
    arrays["frame_indices"] = np.asarray(frame_pairs, dtype=np.int32)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dense_flow_pairs.npz", **arrays)
    metadata = {
        "source_capture": str(capture),
        "source_engine": "blender",
        "pair_count": len(frame_pairs),
        "image_size": [size, size],
        "flow_convention": "normalized source-to-target displacement: 2 * pixel_delta / dimension",
        "flow_alignment": "maps are aligned to source-frame pixels",
        "matched_object_names": matched_names,
        "skipped_object_names": skipped_names,
        "label_method": "per-vertex mesh correspondence, triangle rasterization, perspective-correct interpolation and z-buffer visibility",
        "limitations": [
            "Only objects with matching vertex counts and face topology are tracked.",
            "Newly visible target regions without source RGB correspondence need generative repair.",
            "These are geometry-derived labels, not rendered RGB image pairs.",
            "Transparent, refractive, volumetric, hair and particle appearance are not represented by this mesh z-buffer.",
        ],
    }
    (output / "dense_flow_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build dense flow labels from Blender mesh captures")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build_dense_capture_dataset(args.capture, args.output, args.size), indent=2))


if __name__ == "__main__":
    main()

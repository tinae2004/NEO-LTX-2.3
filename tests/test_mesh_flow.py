import numpy as np

from neo_video_gen.dataset.mesh_flow import compute_dense_flow


PERSPECTIVE = {
    "mode": "perspective_slopes",
    "left": -1.0,
    "right": 1.0,
    "bottom": -1.0,
    "top": 1.0,
}


def test_dense_flow_tracks_plane_translation():
    source = np.array([
        [-0.5, -0.5, -2.0],
        [0.5, -0.5, -2.0],
        [0.5, 0.5, -2.0],
        [-0.5, 0.5, -2.0],
    ], dtype=np.float32)
    target = source.copy()
    target[:, 0] += 0.25
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    identity = np.eye(4, dtype=np.float32)

    result = compute_dense_flow(
        source, target, triangles, identity, identity,
        PERSPECTIVE, PERSPECTIVE, width=64, height=64,
    )
    valid = result["source_visibility"][0] > 0
    assert valid.sum() > 100
    assert np.allclose(result["flow_forward"][0][valid].mean(), 0.125, atol=0.01)
    assert np.allclose(result["flow_forward"][1][valid].mean(), 0.0, atol=0.01)
    assert result["target_visible_from_source"][0][valid].mean() > 0.98
    assert np.isfinite(result["depth_delta"]).all()


def test_target_z_buffer_marks_new_occlusion():
    plane_source = np.array([
        [-0.7, -0.7, -3.0], [0.7, -0.7, -3.0],
        [0.7, 0.7, -3.0], [-0.7, 0.7, -3.0],
    ], dtype=np.float32)
    plane_target = plane_source.copy()
    blocker_source = np.array([
        [3.0, -0.2, -1.0], [3.4, -0.2, -1.0], [3.2, 0.2, -1.0],
    ], dtype=np.float32)
    blocker_target = np.array([
        [-0.2, -0.2, -1.0], [0.2, -0.2, -1.0], [0.0, 0.2, -1.0],
    ], dtype=np.float32)
    source = np.concatenate([plane_source, blocker_source], axis=0)
    target = np.concatenate([plane_target, blocker_target], axis=0)
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    identity = np.eye(4, dtype=np.float32)

    result = compute_dense_flow(
        source, target, triangles, identity, identity,
        PERSPECTIVE, PERSPECTIVE, width=64, height=64,
    )
    tracked = result["source_visibility"][0] > 0
    remains_visible = result["target_visible_from_source"][0] > 0
    assert tracked.sum() > 100
    assert np.any(tracked & ~remains_visible)
    assert result["target_coverage"].sum() > 0

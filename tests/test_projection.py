import numpy as np

from neo_video_gen.translator.projection import project_world_points


def test_projection_identity_camera():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    camera = np.eye(4)
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    pixels, visible = project_world_points(points, camera, intrinsics)
    assert np.allclose(pixels, [[50.0, 50.0], [150.0, 150.0]])
    assert visible.tolist() == [True, True]


def test_points_behind_camera():
    pixels, visible = project_world_points(
        np.array([[0.0, 0.0, -1.0]]), np.eye(4), np.eye(3)
    )
    assert visible.tolist() == [False]
    assert np.isnan(pixels).all()

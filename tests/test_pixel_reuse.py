import numpy as np
from neo_video_gen.temporal.pixel_reuse import warp_rgb_forward

def test_forward_warp_translates_pixels_and_reports_exposed_edge():
    h = w = 8
    source = np.zeros((h, w, 3), dtype=np.uint8)
    source[..., 0] = np.arange(w, dtype=np.uint8)[None, :] * 20
    flow = np.zeros((2, h, w), dtype=np.float32)
    flow[0] = 0.25  # one pixel right at width 8
    ones = np.ones((1, h, w), dtype=np.float32)
    depth = np.full((1, h, w), 2.0, dtype=np.float32)
    result = warp_rgb_forward(source, flow, ones, ones, depth, np.zeros_like(depth), ones)
    assert result["warped_coverage"][:, 0].sum() == 0
    assert result["warped_coverage"][:, 1:].all()
    assert result["hole_mask"][:, 0].all()
    assert not result["hole_mask"][:, 1:].any()
    assert result["warped_rgb"][3, 4, 0] == source[3, 3, 0]

def test_occluded_source_pixels_are_not_splatted():
    h = w = 5
    source = np.full((h, w, 3), 127, dtype=np.uint8)
    flow = np.zeros((2, h, w), dtype=np.float32)
    src_vis = np.ones((1, h, w), dtype=np.float32)
    tgt_vis = np.ones((1, h, w), dtype=np.float32)
    tgt_vis[0, 2, 2] = 0
    depth = np.full((1, h, w), 3.0, dtype=np.float32)
    coverage = np.ones((1, h, w), dtype=np.float32)
    result = warp_rgb_forward(source, flow, src_vis, tgt_vis, depth, np.zeros_like(depth), coverage)
    assert not result["warped_coverage"][2, 2]
    assert result["hole_mask"][2, 2]
    assert result["warped_rgb"][2, 2].sum() == 0

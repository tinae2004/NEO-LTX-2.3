import json

import numpy as np

from neo_video_gen.dataset.blender_pairs import BONE_FOR_JOINT, build_dataset


def write_state(path, frame, offset):
    pose_bones = []
    for index, bone_name in enumerate(BONE_FOR_JOINT.values()):
        pose_bones.append({
            "name": bone_name,
            "head_world": [offset + index * 0.005, 1.5 - index * 0.03, -4.5],
            "tail_world": [offset + index * 0.005, 1.45 - index * 0.03, -4.5],
        })
    state = {
        "frame_index": frame,
        "timestamp_seconds": (frame - 1) / 24,
        "objects": [
            {"name": "NEO_Demo_Rig", "type": "ARMATURE", "pose_bones": pose_bones},
            {
                "name": "NEO_Camera",
                "type": "CAMERA",
                "world_matrix": np.eye(4).tolist(),
                "camera": {"lens_mm": 50.0, "sensor_width_mm": 36.0, "sensor_height_mm": 24.0},
            },
        ],
    }
    path.write_text(json.dumps(state), encoding="utf-8")


def test_blender_pair_converter_creates_train_and_val(tmp_path):
    capture = tmp_path / "capture"
    frames = capture / "frames"
    frames.mkdir(parents=True)
    records = []
    for frame, offset in ((1, 0.0), (2, 0.08), (3, 0.16)):
        state_path = frames / f"frame_{frame:06d}.json"
        write_state(state_path, frame, offset)
        records.append({
            "frame_index": frame,
            "timestamp_seconds": (frame - 1) / 24,
            "state_file": f"frames/{state_path.name}",
        })
    (capture / "manifest.json").write_text(
        json.dumps({"engine": "blender", "frames": records}), encoding="utf-8"
    )

    output = tmp_path / "dataset"
    info = build_dataset(capture, output, action_id=0, armature_name="NEO_Demo_Rig",
                         camera_name="NEO_Camera", image_size=32,
                         validation_fraction=0.25, seed=3)
    assert info["pair_count"] == 2
    with np.load(output / "train.npz") as train:
        assert train["joints_3d"].shape[1:] == (2, 17, 3)
        assert train["camera"].shape[1:] == (2, 4, 4)
        assert train["target_flow"].shape[1:] == (2, 32, 32)
    with np.load(output / "val.npz") as val:
        assert len(val["action_id"]) == 1

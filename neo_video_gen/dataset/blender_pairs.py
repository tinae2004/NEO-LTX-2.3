"""Build a standard 17-joint training dataset from a Blender FULL capture.

Run:
python scripts/build_dataset_from_blender.py --capture ./neo_data/scene_001 --output ./datasets/blender_v0 --action-id 0
Use one action ID per capture sequence. Joint order matches JOINT_NAMES in the dataset module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neo_video_gen.dataset.synthetic import ACTION_NAMES, JOINT_NAMES, build_control_targets

BONE_FOR_JOINT = {
    "pelvis": "Pelvis",
    "spine": "Spine",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "Shoulder_L",
    "left_elbow": "Elbow_L",
    "left_wrist": "Wrist_L",
    "right_shoulder": "Shoulder_R",
    "right_elbow": "Elbow_R",
    "right_wrist": "Wrist_R",
    "left_hip": "Hip_L",
    "left_knee": "Knee_L",
    "left_ankle": "Ankle_L",
    "right_hip": "Hip_R",
    "right_knee": "Knee_R",
    "right_ankle": "Ankle_R",
    "nose": "Nose",
}


def read_state(capture: Path, frame_record: dict) -> dict:
    path = capture / frame_record["state_file"]
    return json.loads(path.read_text(encoding="utf-8"))


def find_object(state: dict, name: str) -> dict:
    for obj in state.get("objects", []):
        if obj.get("name") == name:
            return obj
    raise ValueError(f"Object {name!r} not found in captured frame")


def frame_joints_in_camera(state: dict, armature_name: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    armature = find_object(state, armature_name)
    camera = find_object(state, camera_name)
    pose_bones = {bone["name"]: bone for bone in armature.get("pose_bones", [])}
    missing = [name for name in BONE_FOR_JOINT.values() if name not in pose_bones]
    if missing:
        raise ValueError(
            f"Armature {armature_name!r} is missing required pose bones: {', '.join(missing)}. "
            "Use the NEO demo rig or provide a compatible 17-joint rig."
        )
    camera_info = camera.get("camera")
    if not camera_info:
        raise ValueError(f"Camera {camera_name!r} has no camera metadata")

    camera_world = np.asarray(camera["world_matrix"], dtype=np.float64)
    world_to_camera = np.linalg.inv(camera_world)
    points_world = []
    for joint_name in JOINT_NAMES:
        bone_name = BONE_FOR_JOINT[joint_name]
        points_world.append(pose_bones[bone_name]["head_world"])
    points_world = np.asarray(points_world, dtype=np.float64)
    homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    local = (world_to_camera @ homogeneous.T).T[:, :3]
    # Blender camera looks along local -Z; convert depth to positive-forward convention.
    points_camera = np.stack([local[:, 0], local[:, 1], -local[:, 2]], axis=1).astype(np.float32)

    lens = float(camera_info["lens_mm"])
    sensor_width = float(camera_info["sensor_width_mm"])
    sensor_height = float(camera_info["sensor_height_mm"])
    projection = np.eye(4, dtype=np.float32)
    projection[0, 0] = lens / sensor_width
    projection[0, 2] = 0.5
    projection[1, 1] = -lens / sensor_height
    projection[1, 2] = 0.5
    return points_camera, projection


def build_dataset(capture: Path, output: Path, action_id: int, armature_name: str,
                  camera_name: str, image_size: int, validation_fraction: float, seed: int):
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("engine") != "blender":
        raise ValueError("Expected a Blender capture manifest")
    frames = sorted(manifest.get("frames", []), key=lambda item: item["frame_index"])
    if len(frames) < 2:
        raise ValueError("Capture must contain at least two frames")

    samples = {key: [] for key in (
        "joints_3d", "camera", "action_id", "target_flow",
        "target_visibility", "target_depth_delta",
    )}
    for source_record, target_record in zip(frames[:-1], frames[1:]):
        source_state = read_state(capture, source_record)
        target_state = read_state(capture, target_record)
        source_joints, source_camera = frame_joints_in_camera(
            source_state, armature_name, camera_name
        )
        target_joints, target_camera = frame_joints_in_camera(
            target_state, armature_name, camera_name
        )
        flow, visibility, depth = build_control_targets(
            source_joints, target_joints, source_camera, target_camera, image_size
        )
        samples["joints_3d"].append(np.stack([source_joints, target_joints]))
        samples["camera"].append(np.stack([source_camera, target_camera]))
        samples["action_id"].append(action_id)
        samples["target_flow"].append(flow)
        samples["target_visibility"].append(visibility)
        samples["target_depth_delta"].append(depth)

    arrays = {key: np.asarray(value) for key, value in samples.items()}
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(arrays["action_id"]))
    val_count = max(1, int(len(indices) * validation_fraction))
    val_idx, train_idx = indices[:val_count], indices[val_count:]
    if len(train_idx) == 0:
        raise ValueError("Not enough adjacent frame pairs for a train/validation split")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", **{key: value[train_idx] for key, value in arrays.items()})
    np.savez_compressed(output / "val.npz", **{key: value[val_idx] for key, value in arrays.items()})
    metadata = {
        "source_capture": str(capture.resolve()),
        "source_engine": "blender",
        "action_id": action_id,
        "action_name": ACTION_NAMES[action_id],
        "joint_order": list(JOINT_NAMES),
        "image_size": image_size,
        "pair_count": len(indices),
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "label_method": "projected 3D joint correspondence; sparse Gaussian control fields",
        "limitations": [
            "This is skeleton-level supervision, not dense body-surface optical flow.",
            "Use separate captures with accurate action labels.",
            "The camera projection uses lens/sensor intrinsics and the captured camera transform.",
        ],
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-id", type=int, required=True, choices=range(len(ACTION_NAMES)))
    parser.add_argument("--armature", default="NEO_Demo_Rig")
    parser.add_argument("--camera", default="NEO_Camera")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    info = build_dataset(
        args.capture, args.output, args.action_id, args.armature, args.camera,
        args.size, args.validation_fraction, args.seed,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

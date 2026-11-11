"""Deterministic synthetic motion-pair generator for translator smoke training."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ACTION_NAMES = (
    "head_turn",
    "left_arm_raise",
    "right_arm_raise",
    "torso_twist",
    "walk_step",
    "camera_dolly",
)
JOINT_NAMES = (
    "pelvis", "spine", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "nose",
)


def _base_skeleton(rng: np.random.Generator) -> np.ndarray:
    points = np.array([
        [0.00, 0.95, 4.5], [0.00, 1.20, 4.5], [0.00, 1.43, 4.5],
        [0.00, 1.65, 4.5], [-0.23, 1.36, 4.5], [-0.48, 1.14, 4.5],
        [-0.61, 0.91, 4.5], [0.23, 1.36, 4.5], [0.48, 1.14, 4.5],
        [0.61, 0.91, 4.5], [-0.13, 0.88, 4.5], [-0.14, 0.48, 4.5],
        [-0.14, 0.08, 4.5], [0.13, 0.88, 4.5], [0.14, 0.48, 4.5],
        [0.14, 0.08, 4.5], [0.00, 1.66, 4.34],
    ], dtype=np.float32)
    points += rng.normal(0.0, [0.025, 0.025, 0.04], size=points.shape).astype(np.float32)
    points[:, 2] += rng.uniform(-0.8, 1.0)
    points[:, 0] += rng.uniform(-0.3, 0.3)
    return points


def _apply_action(source: np.ndarray, action: int, rng: np.random.Generator) -> np.ndarray:
    target = source.copy()
    amount = float(rng.uniform(0.18, 0.65))
    if action == 0:  # head turn: head/neck/nose shift in depth and horizontal position
        target[[2, 3, 16], 0] += amount * np.array([0.15, 0.35, 0.55])
        target[[2, 3, 16], 2] -= amount * np.array([0.10, 0.20, 0.12])
    elif action == 1:  # left arm raise
        target[[5, 6], 1] += amount * np.array([0.45, 0.95])
        target[[5, 6], 0] -= amount * np.array([0.12, 0.20])
    elif action == 2:  # right arm raise
        target[[8, 9], 1] += amount * np.array([0.45, 0.95])
        target[[8, 9], 0] += amount * np.array([0.12, 0.20])
    elif action == 3:  # torso twist
        target[[1, 2, 3, 4, 5, 6, 7, 8, 9], 0] += (
            amount * np.array([0.05, 0.10, 0.12, 0.13, -0.12, -0.22, 0.12, 0.22, 0.28])
        )
    elif action == 4:  # walking step
        target[[10, 11, 12], 0] += amount * np.array([-0.25, -0.65, -0.85])
        target[[13, 14, 15], 0] += amount * np.array([0.25, 0.65, 0.85])
        target[[5, 6], 0] += amount * np.array([0.10, 0.25])
        target[[8, 9], 0] -= amount * np.array([0.10, 0.25])
        target[:, 1] += rng.uniform(-0.04, 0.04)
    elif action == 5:  # camera dolly represented as coordinated depth change
        target[:, 2] += amount * 0.8
        target[:, 0] *= 1.0 + amount * 0.04
        target[:, 1] *= 1.0 + amount * 0.04
    target += rng.normal(0.0, 0.008, size=target.shape).astype(np.float32)
    return target


def _camera_matrix(rng: np.random.Generator) -> np.ndarray:
    focal = float(rng.uniform(0.36, 0.58))
    cx = float(rng.uniform(0.46, 0.54))
    cy = float(rng.uniform(0.44, 0.56))
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = focal
    matrix[0, 2] = cx
    matrix[1, 1] = -focal
    matrix[1, 2] = cy
    return matrix


def project_camera(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    projected = (matrix @ homogeneous.T).T
    depth = projected[:, 2]
    uv = np.zeros((len(points), 2), dtype=np.float32)
    valid_depth = depth > 1e-5
    uv[valid_depth] = projected[valid_depth, :2] / depth[valid_depth, None]
    visible = valid_depth & (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
    return uv, visible


def _control_targets(
    source: np.ndarray, target: np.ndarray, matrix: np.ndarray, size: int, sigma_px: float = 2.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_uv, source_visible = project_camera(source, matrix)
    target_uv, target_visible = project_camera(target, matrix)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    flow_sum = np.zeros((2, size, size), dtype=np.float32)
    depth_sum = np.zeros((1, size, size), dtype=np.float32)
    weight_sum = np.zeros((1, size, size), dtype=np.float32)
    visibility = np.zeros((1, size, size), dtype=np.float32)
    sigma = sigma_px / size
    for index in range(len(source)):
        if not target_visible[index]:
            continue
        u, v = target_uv[index]
        gaussian = np.exp(-((xx / size - u) ** 2 + (yy / size - v) ** 2) / (2 * sigma * sigma))
        visibility[0] = np.maximum(visibility[0], gaussian)
        if not source_visible[index]:
            continue
        delta = (target_uv[index] - source_uv[index]) * 2.0
        weight_sum[0] += gaussian
        flow_sum[0] += gaussian * delta[0]
        flow_sum[1] += gaussian * delta[1]
        depth_sum[0] += gaussian * (target[index, 2] - source[index, 2])
    denominator = np.maximum(weight_sum, 1e-6)
    flow = flow_sum / denominator
    depth_delta = depth_sum / denominator
    return flow, visibility, depth_delta


def generate_arrays(
    sample_count: int = 512,
    image_size: int = 64,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if image_size < 16 or image_size % 8:
        raise ValueError("image_size must be >=16 and divisible by 8")
    rng = np.random.default_rng(seed)
    joints, cameras, actions, flows, visibility, depths = [], [], [], [], [], []
    for _ in range(sample_count):
        source = _base_skeleton(rng)
        action = int(rng.integers(0, len(ACTION_NAMES)))
        target = _apply_action(source, action, rng)
        camera = _camera_matrix(rng)
        flow, visible, depth_delta = _control_targets(source, target, camera, image_size)
        joints.append(np.stack([source, target]))
        cameras.append(camera)
        actions.append(action)
        flows.append(flow)
        visibility.append(visible)
        depths.append(depth_delta)
    return {
        "joints_3d": np.asarray(joints, dtype=np.float32),
        "camera": np.asarray(cameras, dtype=np.float32),
        "action_id": np.asarray(actions, dtype=np.int64),
        "target_flow": np.asarray(flows, dtype=np.float32),
        "target_visibility": np.asarray(visibility, dtype=np.float32),
        "target_depth_delta": np.asarray(depths, dtype=np.float32),
    }


def save_dataset(output: str | Path, sample_count: int = 1024, validation_fraction: float = 0.2,
                 image_size: int = 64, seed: int = 7) -> tuple[Path, Path]:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    arrays = generate_arrays(sample_count, image_size, seed)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(sample_count)
    val_count = max(1, int(sample_count * validation_fraction))
    val_indices, train_indices = order[:val_count], order[val_count:]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    train_path, val_path = root / "train.npz", root / "val.npz"
    np.savez_compressed(train_path, **{key: value[train_indices] for key, value in arrays.items()})
    np.savez_compressed(val_path, **{key: value[val_indices] for key, value in arrays.items()})
    return train_path, val_path

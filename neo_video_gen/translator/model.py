"""Trainable skeleton-motion-to-2D-control baseline.

Input: two 3D skeleton states plus a 4x4 camera projection matrix and action ID.
Output: sparse joint-motion control fields (flow, visibility, depth delta).

This is intentionally a first control-field baseline, not a complete human-image
renderer or a native FLUX adapter.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Install training dependencies: pip install -e '.[train]'") from exc


class MotionTranslator(nn.Module):
    def __init__(
        self,
        joint_count: int = 17,
        camera_dim: int = 16,
        action_count: int = 6,
        action_dim: int = 16,
        hidden_dim: int = 256,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        self.joint_count = joint_count
        self.output_size = output_size
        self.action_embedding = nn.Embedding(action_count, action_dim)
        input_dim = 2 * joint_count * 3 + camera_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.seed = nn.Linear(hidden_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, 2, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 48, 4, 2, 1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 24, 4, 2, 1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 4, 3, padding=1),
        )

    def forward(
        self,
        joints_3d: torch.Tensor,
        camera: torch.Tensor,
        action_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if joints_3d.ndim != 4 or joints_3d.shape[1] != 2 or joints_3d.shape[-1] != 3:
            raise ValueError("joints_3d must have shape [B, 2, J, 3]")
        batch = joints_3d.shape[0]
        if joints_3d.shape[2] != self.joint_count:
            raise ValueError(f"Expected {self.joint_count} joints, got {joints_3d.shape[2]}")
        camera_flat = camera.reshape(batch, -1)
        if camera_flat.shape[1] != 16:
            raise ValueError("camera must contain a flattened 4x4 projection matrix")
        encoded_input = torch.cat(
            (
                joints_3d.reshape(batch, -1),
                camera_flat,
                self.action_embedding(action_id.long()),
            ),
            dim=1,
        )
        features = self.encoder(encoded_input)
        raw = self.decoder(self.seed(features).reshape(batch, 128, 8, 8))
        return {
            "flow": raw[:, 0:2],
            "visibility_logits": raw[:, 2:3],
            "depth_delta": raw[:, 3:4],
        }

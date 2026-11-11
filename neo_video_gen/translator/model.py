"""Small trainable 3D-state-to-2D-control baseline; not a FLUX adapter."""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Install training dependencies with: pip install -e '.[train]'") from exc


class MotionTranslator(nn.Module):
    def __init__(self, joint_count=52, camera_dim=16, action_count=128, action_dim=32):
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, action_dim)
        input_dim = joint_count * 3 + camera_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
        )
        self.seed = nn.Linear(512, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 4, 3, padding=1),
        )

    def forward(self, joints_3d, camera, action_id):
        batch = joints_3d.shape[0]
        encoded_input = torch.cat(
            (joints_3d.reshape(batch, -1), camera, self.action_embedding(action_id.long())),
            dim=1,
        )
        features = self.encoder(encoded_input)
        raw = self.decoder(self.seed(features).reshape(batch, 256, 8, 8))
        return {
            "flow": raw[:, :2],
            "visibility_logits": raw[:, 2:3],
            "depth_delta": raw[:, 3:4],
        }

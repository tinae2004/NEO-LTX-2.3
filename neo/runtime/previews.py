"""Sparse decoded RGB previews for completed LTX diffusion chunks.

Preview decoding deliberately happens after DiffusionStage returns, so the
transformer has been released before the video VAE is built. This avoids
holding the denoiser and VAE in VRAM at the same time.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class VideoPreviewWriter:
    """Decode first/last frames per completed chunk and bound disk usage.

    The input must be an unpatchified (B, C, T, H, W) video latent from
    DiffusionStage. Sampler callback tokens are not accepted here.
    """

    def __init__(
        self,
        video_decoder: Any,
        directory: str | Path,
        *,
        tiling_config: Any | None = None,
        dtype: torch.dtype | None = None,
        max_previews: int = 8,
    ) -> None:
        if max_previews < 1:
            raise ValueError("max_previews must be >= 1")
        self.video_decoder = video_decoder
        self.directory = Path(directory)
        self.tiling_config = tiling_config
        self.dtype = dtype
        self.max_previews = max_previews
        self._saved: list[Path] = []

    def __call__(self, stage: str, chunk: int, video_latent: torch.Tensor) -> Path:
        """Decode a completed chunk and publish a first/last-frame contact sheet."""
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        if video_latent.ndim != 5:
            raise ValueError(
                "VideoPreviewWriter expects unpatchified (B, C, T, H, W) latents; "
                f"received shape {tuple(video_latent.shape)}"
            )
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("._-")
        if not safe_stage:
            raise ValueError("stage must contain at least one safe filename character")

        decoded = list(
            self.video_decoder(
                video_latent,
                self.tiling_config,
                None,
                dtype=self.dtype,
            )
        )
        if not decoded:
            raise RuntimeError("Video decoder returned no preview frames")
        frames = decoded[0] if len(decoded) == 1 else torch.cat(decoded, dim=0)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
            raise ValueError(
                "Video decoder must return (F, H, W, 3) RGB frames; "
                f"received shape {tuple(frames.shape)}"
            )

        endpoints = torch.stack(
            (frames[0], frames[-1]),
            dim=0,
        ).detach().to(device="cpu")
        if endpoints.is_floating_point():
            endpoints = endpoints.float()
            minimum = float(endpoints.min())
            maximum = float(endpoints.max())
            if minimum < -0.01:
                endpoints = (endpoints + 1.0) / 2.0
            elif maximum > 1.0:
                endpoints = endpoints / 255.0
            endpoints = (
                endpoints.clamp(0.0, 1.0) * 255.0
            ).round().to(torch.uint8)
        else:
            endpoints = endpoints.clamp(0, 255).to(torch.uint8)

        # Left panel is the first decoded frame; right panel is the last.
        contact_sheet = torch.cat((endpoints[0], endpoints[1]), dim=1)

        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_stage}_chunk{chunk:04d}.png"
        destination = self.directory / filename
        fd, temporary = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".png", dir=self.directory
        )
        os.close(fd)
        try:
            Image.fromarray(contact_sheet.contiguous().numpy(), mode="RGB").save(
                temporary, format="PNG"
            )
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        if destination in self._saved:
            self._saved.remove(destination)
        self._saved.append(destination)
        while len(self._saved) > self.max_previews:
            expired = self._saved.pop(0)
            expired.unlink(missing_ok=True)
        return destination

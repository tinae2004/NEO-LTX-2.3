"""Bounded CPU snapshots for sparse LTX sampler checkpoints.

Stores detached CPU copies of patchified sampler latents. These are not VAE-ready
frames; VideoLatentTools must clear conditioning and unpatchify before preview decode.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from ltx_core.types import LatentState


class LatentCheckpointStore:
    """Persist sparse sampler snapshots without retaining GPU allocations.

    Directly usable as the distilled pipeline step_callback. Snapshots are
    written synchronously, so use a sparse interval and a fast local directory.
    Only latent tensors and compact shape/dtype metadata are stored.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        every_n_steps: int = 4,
        max_snapshots: int = 8,
    ) -> None:
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        self.directory = Path(directory)
        self.every_n_steps = every_n_steps
        self.max_snapshots = max_snapshots
        self._saved: list[Path] = []

    def __call__(
        self,
        stage: str,
        chunk: int,
        step: int,
        total_steps: int,
        video_state: LatentState | None,
        audio_state: LatentState | None,
    ) -> Path | None:
        """Capture interval checkpoints and always capture the final step."""
        if step != total_steps and step % self.every_n_steps:
            return None
        if chunk < 1 or step < 1 or total_steps < step:
            raise ValueError("Invalid checkpoint stage/chunk/step coordinates")
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("._-")
        if not safe_stage:
            raise ValueError("stage must contain at least one safe filename character")

        payload: dict[str, Any] = {
            "schema": "neo-latent-checkpoint-v1",
            "stage": stage,
            "chunk": chunk,
            "step": step,
            "total_steps": total_steps,
            "modalities": {},
        }
        for name, state in (("video", video_state), ("audio", audio_state)):
            if state is None:
                continue
            tensor = state.latent.detach().to(device="cpu").contiguous().clone()
            payload["modalities"][name] = {
                "latent": tensor,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }

        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_stage}_chunk{chunk:04d}_step{step:04d}.pt"
        destination = self.directory / filename
        fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        if destination in self._saved:
            self._saved.remove(destination)
        self._saved.append(destination)
        while len(self._saved) > self.max_snapshots:
            expired = self._saved.pop(0)
            expired.unlink(missing_ok=True)
        return destination

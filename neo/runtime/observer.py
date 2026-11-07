"""Safe reasoning checkpoints around, not inside, arbitrary latent mutation.

The host LTX pipeline must explicitly call this observer at a supported
sampler boundary and provide a decoded preview. This module never edits latents.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointReport:
    step: int
    total_steps: int
    preview_path: str | None
    production_state: dict[str, Any]


class ReasoningObserver:
    def __init__(self, director: Any, every_n_steps: int = 4):
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        self.director = director
        self.every_n_steps = every_n_steps

    def should_inspect(self, step: int, total_steps: int) -> bool:
        return step == total_steps or (step > 0 and step % self.every_n_steps == 0)

    def inspect(self, report: CheckpointReport, brief: str) -> dict[str, Any]:
        """Return a new plan as data for a pipeline adapter to consume later.

        DeepSeek is called only at explicit checkpoints. Applying a plan to
        subsequent denoising requires a model-supported conditioning hook.
        """
        state = dict(report.production_state)
        state["denoising_checkpoint"] = {
            "step": report.step, "total_steps": report.total_steps,
            "preview_path": report.preview_path,
        }
        plan = self.director.plan(brief, state)
        return plan.model_dump()


class ReasoningPreviewAdapter:
    """Send sparse decoded stage previews and production state to DeepSeek.

    Pass this object as DistilledPipeline.preview_callback. DeepSeek receives
    the actual preview image through the configured director client. Returned
    plans are recorded as data; this adapter never mutates LTX latents or
    silently applies an unsupported conditioning change.
    """

    def __init__(
        self,
        observer: ReasoningObserver,
        brief: str,
        production_state: Any,
        *,
        every_n_chunks: int = 1,
    ) -> None:
        if every_n_chunks < 1:
            raise ValueError("every_n_chunks must be >= 1")
        state_data = getattr(production_state, "data", production_state)
        if not isinstance(state_data, dict):
            raise TypeError("production_state must be a dict or expose a dict .data")
        self.observer = observer
        self.brief = brief
        self.production_state_owner = production_state
        self.production_state = state_data
        self.every_n_chunks = every_n_chunks
        self.last_plan: dict[str, Any] | None = None

    def __call__(self, stage: str, chunk: int, preview_path: str | Path) -> dict[str, Any] | None:
        """Inspect selected stage-2 chunk previews and persist the returned plan."""
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        if chunk % self.every_n_chunks:
            return None

        preview = Path(preview_path)
        if not preview.is_file():
            raise FileNotFoundError(f"Preview image does not exist: {preview}")

        snapshot = dict(self.production_state)
        snapshot.update({
            "stage": stage,
            "chunk_index": chunk,
            "preview_path": str(preview),
        })
        report = CheckpointReport(
            step=chunk,
            total_steps=chunk,
            preview_path=str(preview),
            production_state=snapshot,
        )
        plan = self.observer.inspect(report, self.brief)
        self.production_state["last_preview_path"] = str(preview)
        self.production_state["last_preview_stage"] = stage
        self.production_state["last_preview_chunk"] = chunk
        self.production_state["last_director_plan"] = plan
        save = getattr(self.production_state_owner, "save", None)
        if callable(save):
            save()
        self.last_plan = plan
        return plan


"""Safe reasoning checkpoints around, not inside, arbitrary latent mutation.

The host LTX pipeline must explicitly call this observer at a supported
sampler boundary and provide a decoded preview. This module never edits latents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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

"""
Stable interface for loading a checkpoint and scoring an image.
Model owns the contract; Product only calls through here so checkpoints
can swap without Product code changes.
"""

from dataclasses import dataclass


@dataclass
class DetectionResult:
    score: float          # P(AI-generated), 0-1
    crop_scores: list[float]  # per-crop scores, averaged for `score`
    backbone: str
    checkpoint_id: str


def load_checkpoint(path: str):
    """Load a checkpoint written by any model/steps/stepN_*.py script."""
    raise NotImplementedError


def score_image(model, image_path: str) -> DetectionResult:
    """Score at native resolution using multiple crops, averaged —
    resizing the whole image to 224px erases the artifacts (see
    docs/plan.md Pitfalls)."""
    raise NotImplementedError

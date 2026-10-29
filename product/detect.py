"""
The one function both the API and the Gradio demo call, so they can never
disagree: provenance first, model only if provenance is inconclusive.

Environment variables (all optional):
    CHECKPOINT_PATH   default model/checkpoints/step2.pt
    AI_THRESHOLD      score at/above which the label is "ai" (default 0.5;
                      replace with the 1%-FPR threshold Evaluation reports)
    USE_DUMMY_MODEL   "1" -> skip the real checkpoint and return a fake,
                      deterministic score so the API/demo can be built and
                      tested before Model has a checkpoint. Results are
                      labelled checkpoint_id="dummy" -- never show them to
                      users as real.
"""

import hashlib
import os
from dataclasses import dataclass, field

from product.provenance.check import check_provenance
from shared import model_interface

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "model/checkpoints/step2.pt")
THRESHOLD = float(os.environ.get("AI_THRESHOLD", "0.5"))

_model = None


@dataclass
class Detection:
    label: str                      # "ai" | "real"
    score: float                    # P(AI), 0-1
    decided_by: str                 # "provenance" | "model"
    signals: list[str] = field(default_factory=list)   # e.g. ["c2pa"] or ["model"]
    crop_scores: list[float] = field(default_factory=list)
    checkpoint_id: str | None = None


def _get_model():
    global _model
    if _model is None:
        _model = model_interface.load_checkpoint(CHECKPOINT_PATH)
    return _model


def _dummy_score(image_path: str) -> float:
    with open(image_path, "rb") as f:
        digest = hashlib.sha256(f.read()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def detect(image_path: str) -> Detection:
    provenance = check_provenance(image_path)
    if provenance.is_conclusive:
        return Detection(
            label="ai" if provenance.is_ai else "real",
            score=1.0 if provenance.is_ai else 0.0,
            decided_by="provenance",
            signals=[provenance.signal],
        )

    if os.environ.get("USE_DUMMY_MODEL") == "1":
        score = _dummy_score(image_path)
        return Detection(
            label="ai" if score >= THRESHOLD else "real",
            score=score,
            decided_by="model",
            signals=["model"],
            crop_scores=[score],
            checkpoint_id="dummy",
        )

    result = model_interface.score_image(_get_model(), image_path)
    return Detection(
        label="ai" if result.score >= THRESHOLD else "real",
        score=result.score,
        decided_by="model",
        signals=["model"],
        crop_scores=list(result.crop_scores),
        checkpoint_id=result.checkpoint_id,
    )

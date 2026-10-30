"""
Gradio demo. Week 2 target: serve a dummy checkpoint. Week 4 target:
show to 3-5 potential users.

Run (from the repo root):
    USE_DUMMY_MODEL=1 python product/demo/app.py      # before a checkpoint exists
    python product/demo/app.py                         # real checkpoint
(PowerShell:  $env:USE_DUMMY_MODEL="1"; python product/demo/app.py)
"""

import sys
from pathlib import Path

import gradio as gr

# Make `product/` and `shared/` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from product.detect import THRESHOLD, detect  # noqa: E402

DISAGREE_GAP = 0.5   # if the highest and lowest crop scores differ by more than this, say so


def headline_for(score: float) -> str:
    """Three bands, tied to the flagging threshold Evaluation reports (AI_THRESHOLD).
    The threshold is strict (about 1% of real photos flagged), so 'not flagged' must
    never be worded as 'real'."""
    if score >= THRESHOLD:
        return "### Flagged: likely AI-generated"
    if THRESHOLD > 0.5 and score >= 0.5:
        return "### Inconclusive: some signs of AI, below our flagging cut-off"
    return "### Not flagged. This does not prove the image is real."


def classify(image_path):
    """Returns (headline in markdown, plain-language explanation)."""
    if image_path is None:
        return "Upload an image first.", ""
    try:
        d = detect(image_path)
    except (NotImplementedError, FileNotFoundError):
        return ("### No model available yet",
                "No model checkpoint is available. Start the demo with USE_DUMMY_MODEL=1, "
                "or point CHECKPOINT_PATH at a checkpoint.")

    if d.decided_by == "provenance":
        # Metadata is a claim, not a measurement of the pixels, so no percentage.
        return ("### This file's metadata says it is AI-generated",
                f"Decided by metadata embedded in the file ({d.signals[0]}), not by the model. "
                "A metadata tag can be planted on a real photo, and a real photo with a small "
                "AI edit can carry an AI tag. Metadata can also be stripped, so a missing tag "
                "would have meant nothing. (We do not yet check C2PA signers against a trust list.)")

    crops = ", ".join(f"{c:.2f}" for c in d.crop_scores)
    why = (f"No usable metadata, so the model scored the pixels. Score {d.score:.2f}; an image is "
           f"flagged at {THRESHOLD:.2f} or higher (that setting wrongly flags about 1% of real photos, "
           f"so many AI images will not be flagged). Per-crop scores: {crops}.")
    if len(d.crop_scores) > 1 and max(d.crop_scores) - min(d.crop_scores) > DISAGREE_GAP:
        why += " The crops disagree strongly, so treat this result as low confidence."
    head = headline_for(d.score)
    if d.checkpoint_id == "dummy":
        head = "### DUMMY MODEL (meaningless score)"
        why = "DUMMY MODEL: this score is meaningless. " + why
    return head, why


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Markdown(), gr.Textbox(label="Why")],
    title="AI Image Detector",
    description="Upload an image to check whether it's AI-generated. This is an early research "
                "model: it only flags images it is very sure about.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

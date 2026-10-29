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
from product.detect import detect  # noqa: E402


def classify(image_path):
    """Returns (label -> confidence dict for gr.Label, plain-language reason)."""
    if image_path is None:
        return {}, "Upload an image first."
    try:
        d = detect(image_path)
    except (NotImplementedError, FileNotFoundError):
        return {}, ("No model checkpoint is available yet. Start the demo with "
                    "USE_DUMMY_MODEL=1, or point CHECKPOINT_PATH at a checkpoint.")

    if d.decided_by == "provenance":
        # Metadata is a claim written into the file, not a measurement of the
        # pixels, so don't show it as a confidence bar. (Week 4 risk 3.)
        scores = {"This file's metadata says it is AI-generated": 1.0}
        why = (f"Decided by metadata embedded in the file ({d.signals[0]}), "
               "not by the model. A metadata tag can be planted on a real photo, "
               "and a real photo with a small AI edit can carry an AI tag. "
               "Metadata can also be stripped, so a missing tag would have "
               "meant nothing. (We do not yet check C2PA signers against a "
               "trust list.)")
    else:
        scores = {"AI-generated": d.score, "Real": 1 - d.score}
        crops = ", ".join(f"{s:.2f}" for s in d.crop_scores)
        why = f"No usable metadata, so the model scored the pixels. Per-crop scores: {crops}."
        if d.checkpoint_id == "dummy":
            why = "DUMMY MODEL: this score is meaningless. " + why
    return scores, why


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Label(num_top_classes=2), gr.Textbox(label="Why")],
    title="AI Image Detector",
    description="Upload an image to check whether it's AI-generated.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

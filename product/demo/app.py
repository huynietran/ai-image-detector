"""
Gradio demo. Week 2 target: serve a dummy checkpoint. Week 4 target:
show to 3-5 potential users.

Run:
    python product/demo/app.py
"""

import gradio as gr


def classify(image):
    """Call product/api or shared/model_interface.py directly for the
    demo. Return a dict of label -> confidence for gr.Label."""
    raise NotImplementedError


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=2),
    title="AI Image Detector",
    description="Upload an image to check whether it's AI-generated.",
)

if __name__ == "__main__":
    demo.launch()

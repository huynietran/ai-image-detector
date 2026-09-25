"""
Build the degradation suite: JPEG 30/50/75, resize 0.5-1.5x, crop,
screenshot. Applied to data/stress_set/ and the held-out test set so
Evaluation can report how AUC/TPR degrade under real-world compression.

Usage:
    python evaluation/scripts/degradation_suite.py \
        --input-dir data/matched/ \
        --output-dir data/degradation_suite/
"""


def apply_jpeg(image, quality: int):
    raise NotImplementedError


def apply_resize(image, scale: float):
    raise NotImplementedError


def apply_crop(image):
    raise NotImplementedError


def apply_screenshot(image):
    """Simulate a screenshot: render + re-capture to strip metadata and
    add compositor artifacts."""
    raise NotImplementedError


def main(input_dir: str, output_dir: str):
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

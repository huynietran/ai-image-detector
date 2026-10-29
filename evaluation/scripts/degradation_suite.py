"""
Build the degradation suite: JPEG 30/50/75, resize 0.5-1.5x, crop,
screenshot. Applied to data/stress_set/ and the held-out test set so
Evaluation can report how AUC/TPR degrade under real-world compression.

Usage:
    python evaluation/scripts/degradation_suite.py \
        --input-dir data/matched/ \
        --output-dir data/degradation_suite/
"""

import io
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size)


def apply_crop(image: Image.Image, keep_fraction: float = 0.8) -> Image.Image:
    """Center crop, keeping `keep_fraction` of each dimension — simulates a
    re-share that's been cropped to a different aspect ratio."""
    w, h = image.size
    new_w, new_h = int(w * keep_fraction), int(h * keep_fraction)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def apply_screenshot(image: Image.Image) -> Image.Image:
    """Simulate a screenshot: render + re-capture to strip metadata and
    add compositor artifacts."""
    w, h = image.size
    # Approximate the resample a screen/compositor does at a non-1:1 pixel
    # ratio (e.g. a browser zoom or a retina-display screenshot).
    scaled = image.resize((max(1, int(w * 0.9)), max(1, int(h * 0.9))))
    scaled = scaled.resize((w, h))
    # Rebuild from raw pixel data so no EXIF/ICC/text metadata survives —
    # a real screenshot never carries the original file's metadata either.
    clean = Image.new("RGB", scaled.size)
    clean.putdata(list(scaled.convert("RGB").getdata()))
    return clean


# (variant folder name) -> function applied to each image
VARIANTS = {
    "jpeg_30": lambda img: apply_jpeg(img, 30),
    "jpeg_50": lambda img: apply_jpeg(img, 50),
    "jpeg_75": lambda img: apply_jpeg(img, 75),
    "resize_0.5x": lambda img: apply_resize(img, 0.5),
    "resize_1.5x": lambda img: apply_resize(img, 1.5),
    "crop": apply_crop,
    "screenshot": apply_screenshot,
}


def main(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    image_paths = sorted(
        p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(
            f"No images found directly inside {input_dir} "
            f"(looked for {sorted(IMAGE_EXTENSIONS)})"
        )

    for variant_name, fn in VARIANTS.items():
        variant_dir = output_path / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"{variant_name}: processing {len(image_paths)} images...")
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            degraded = fn(image)
            degraded.convert("RGB").save(variant_dir / path.name)

    print(
        f"\nWrote {len(VARIANTS)} degraded copies of {len(image_paths)} "
        f"images each, under {output_path}/<variant>/"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

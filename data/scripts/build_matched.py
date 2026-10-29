"""
Turn raw images into the matched set Model trains on.

Why this exists: real photos are usually JPEG and AI images usually PNG, at
different sizes. A classifier will happily learn "PNG = fake" and score 0.99
without learning anything about AI (docs/plan.md, Pitfalls). So every image,
real or AI, goes through ONE identical pipeline:

    open -> fix orientation -> RGB -> center-crop to SIZE x SIZE at native
    resolution (NO resizing, which would smooth away the artifacts) ->
    re-save as JPEG at a random quality from the same range for both classes
    -> metadata stripped.

Expected folders (any depth below the source/generator folder is fine):
    data/raw/real/<source>/...                e.g. real/personal_collection/
    data/raw/ai/<source>/<generator>/...      e.g. ai/genimage/stable-diffusion-v1-5/

Writes data/matched/<real|ai>_<hash>.jpg and appends rows to
data/manifests/manifest.csv (split = train; Evaluation's freeze script moves
the held-out generator's rows to held_out_gen afterwards). Safe to re-run:
images already in the manifest are skipped.

Usage:
    python data/scripts/build_matched.py
    python data/scripts/build_matched.py --max-per-folder 2000 --size 512
"""

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import load_manifest  # noqa: E402

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FIELDS = ["filename", "source", "generator", "split", "is_ai"]


def center_crop(img: Image.Image, size: int) -> Image.Image | None:
    """Square center crop of size x size pixels, or None if the image is too small."""
    w, h = img.size
    if min(w, h) < size:
        return None
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def collect(raw_real: Path, raw_ai: Path, max_per_folder: int | None, seed: int):
    """Yield (path, is_ai, source, generator), at most max_per_folder per folder."""
    rng = random.Random(seed)
    jobs = []
    for root, is_ai in ((raw_real, 0), (raw_ai, 1)):
        if not root.exists():
            continue
        for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if is_ai:
                folders = sorted(p for p in source_dir.iterdir() if p.is_dir())
                labeled = [(f, source_dir.name, f.name) for f in folders]
            else:
                labeled = [(source_dir, source_dir.name, "")]
            for folder, source, generator in labeled:
                files = sorted(p for p in folder.rglob("*")
                               if p.is_file() and p.suffix.lower() in EXTENSIONS)
                if max_per_folder and len(files) > max_per_folder:
                    files = sorted(rng.sample(files, max_per_folder))
                jobs += [(p, is_ai, source, generator) for p in files]
    return jobs


def main(raw_real, raw_ai, out_dir, manifest_path, size, q_min, q_max, max_per_folder, seed):
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {r["filename"] for r in load_manifest(str(manifest_path))} if manifest_path.exists() else set()

    stats = Counter()
    new_rows = []
    for path, is_ai, source, generator in collect(Path(raw_real), Path(raw_ai), max_per_folder, seed):
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        name = f"{'ai' if is_ai else 'real'}_{digest}.jpg"
        if name in existing or any(r["filename"] == name for r in new_rows):
            stats["skipped_duplicate"] += 1
            continue
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                crop = center_crop(img, size)
        except Exception:
            stats["skipped_unreadable"] += 1
            continue
        if crop is None:
            stats["skipped_too_small"] += 1
            continue
        quality = random.Random(digest).randint(q_min, q_max)  # same range for both classes
        crop.save(out_dir / name, "JPEG", quality=quality)       # no exif= -> metadata dropped
        new_rows.append({"filename": name, "source": source, "generator": generator,
                         "split": "train", "is_ai": is_ai})
        stats[f"added_{'ai' if is_ai else 'real'}"] += 1

    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(dict(stats))
    by_group = Counter((r["is_ai"], r["source"], r["generator"]) for r in new_rows)
    for (is_ai, source, generator), n in sorted(by_group.items()):
        print(f"  {'AI  ' if is_ai else 'real'} {source}/{generator or '-'}: {n}")
    ai, real = stats["added_ai"], stats["added_real"]
    if ai and real and max(ai, real) / min(ai, real) > 1.5:
        print(f"NOTE: this run added {real} real vs {ai} AI. Roughly 1:1 is what we want "
              "-- collect more of the smaller class or use --max-per-folder.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-real", default="data/raw/real")
    p.add_argument("--raw-ai", default="data/raw/ai")
    p.add_argument("--out", default="data/matched")
    p.add_argument("--manifest", default="data/manifests/manifest.csv")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--quality-min", type=int, default=70)
    p.add_argument("--quality-max", type=int, default=95)
    p.add_argument("--max-per-folder", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.raw_real, a.raw_ai, a.out, a.manifest, a.size, a.quality_min, a.quality_max,
         a.max_per_folder, a.seed)

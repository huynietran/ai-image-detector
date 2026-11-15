"""
Build the VAE reconstruction set that model/steps/step6_vae_reconstruction.py trains on.

Why: the v0 training set has only 300 AI images, all from one generator
(Stable Diffusion), so the detector partly learns "looks like Stable Diffusion"
and misses most Flux images. A VAE reconstruction is a REAL photo pushed
through a diffusion model's autoencoder (encode, then decode). Content, framing
and subject are identical to the real photo, so the only thing that differs is
the trace the generative model leaves. Labelled AI, these pairs teach
"touched by a generative model" instead of one generator's style or subject
matter.

Rules this script enforces:
  * Only REAL photos from the TRAIN split are reconstructed. Reconstructing a
    held-out photo would leak it into training.
  * Uses the Stable Diffusion 1.x VAE (stabilityai/sd-vae-ft-mse). Flux has its
    own autoencoder; using Flux's would leak the held-out generator.
  * Output goes through the same final step as build_matched.py: JPEG at a
    random quality 70-95 (per file, same range as every other image), no
    metadata, same 512 x 512 size.

Writes data/matched/vae_<hash>.jpg and appends rows to the manifest with
generator = vae_reconstruction, split = train, is_ai = 1. Steps 2-5 ignore
these rows (train.exclude_generators); only step 6 uses them. Safe to re-run:
photos already reconstructed are skipped.

Usage (from the repo root; about 10-20 minutes on a Mac, faster on a GPU):
    python data/scripts/build_vae_set.py
    python data/scripts/build_vae_set.py --limit 20     # quick try
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import load_manifest  # noqa: E402

FIELDS = ["filename", "source", "generator", "split", "is_ai"]
GENERATOR = "vae_reconstruction"
VAE_ID = "stabilityai/sd-vae-ft-mse"


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def reconstruct(vae, img: Image.Image, dev: str) -> Image.Image:
    """Encode then decode one RGB image (sides must be multiples of 8)."""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mean          # mean, not a sample: deterministic
        y = vae.decode(latent).sample
    y = ((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def main(manifest_path, matched_dir, q_min, q_max, limit):
    from diffusers import AutoencoderKL  # imported here so --help works without diffusers

    rows = load_manifest(manifest_path)
    sources = [r for r in rows if r["split"] == "train" and int(r["is_ai"]) == 0]
    done = {r["source"].split(":", 1)[1] for r in rows if r["generator"] == GENERATOR and ":" in r["source"]}
    todo = [r for r in sources if r["filename"] not in done]
    if limit:
        todo = todo[:limit]
    held = {r["filename"] for r in rows if r["split"] != "train"}
    assert not any(r["filename"] in held for r in todo), "refusing to reconstruct a non-train photo"
    print(f"{len(sources)} train real photos, {len(done)} already reconstructed, {len(todo)} to do")
    if not todo:
        return

    dev = device()
    print(f"loading {VAE_ID} on {dev}")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(dev).eval()

    new_rows = []
    for i, r in enumerate(todo, 1):
        with Image.open(Path(matched_dir) / r["filename"]) as im:
            img = im.convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % 8, h - h % 8))
        out = reconstruct(vae, img, dev)
        digest = hashlib.sha1(("vae:" + r["filename"]).encode()).hexdigest()[:12]
        name = f"vae_{digest}.jpg"
        quality = random.Random(digest).randint(q_min, q_max)   # same range as build_matched.py
        out.save(Path(matched_dir) / name, "JPEG", quality=quality)
        # source keeps the link to the original photo, so leakage checks can trace it
        new_rows.append({"filename": name, "source": f"vae_of:{r['filename']}", "generator": GENERATOR,
                         "split": "train", "is_ai": 1})
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")

    with open(manifest_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerows(new_rows)
    print(f"added {len(new_rows)} rows (generator={GENERATOR}, split=train, is_ai=1) to {manifest_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/manifest.csv")
    p.add_argument("--matched-dir", default="data/matched")
    p.add_argument("--quality-min", type=int, default=70)
    p.add_argument("--quality-max", type=int, default=95)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    main(a.manifest, a.matched_dir, a.quality_min, a.quality_max, a.limit)

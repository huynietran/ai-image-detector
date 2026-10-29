"""
Freeze the held-out-generator test set.

Locks in which generator(s) the model must NEVER see during training, by
marking their rows — plus a matched slice of real images — with
split=held_out_gen in data/manifests/manifest.csv.

Run this ONCE per held-out generator, then commit the updated manifest.csv
to git. From that point on:
  - Model's data loader must only read rows where split == "train".
  - Evaluation's score_checkpoint.py reads rows where split == "held_out_gen"
    (that's exactly what --split held_out_gen does).

Usage:
    python evaluation/scripts/freeze_held_out.py \
        --manifest data/manifests/manifest.csv \
        --held-out-generators midjourney flux
"""

import argparse
import csv
import random
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def freeze(rows: list[dict], held_out_generators: list[str], seed: int = 42):
    """Mark AI rows from the held-out generators, and an equal-sized
    matched slice of not-yet-assigned real rows, as split='held_out_gen'.

    Returns the updated rows list and a summary dict.
    """
    held_out_generators = set(held_out_generators)

    ai_held_out = [
        r for r in rows
        if r.get("is_ai") == "1" and r.get("generator") in held_out_generators
    ]
    if not ai_held_out:
        raise SystemExit(
            f"No rows found for generator(s) {sorted(held_out_generators)}. "
            "Check data/manifests/manifest.csv — has Data lead added images "
            "from this generator yet?"
        )

    # Real images not already pinned to a non-train split (held_out_gen/stress)
    real_candidates = [
        r for r in rows
        if r.get("is_ai") == "0" and r.get("split", "") in ("", "train")
    ]
    if len(real_candidates) < len(ai_held_out):
        raise SystemExit(
            f"Only {len(real_candidates)} real images available for "
            f"train/unset split, but {len(ai_held_out)} AI images need a "
            "matched real counterpart. Ask Data lead for more real photos "
            "before freezing this generator."
        )

    rng = random.Random(seed)
    rng.shuffle(real_candidates)
    real_held_out = real_candidates[: len(ai_held_out)]

    held_out_filenames = {r["filename"] for r in ai_held_out} | {
        r["filename"] for r in real_held_out
    }
    for r in rows:
        if r["filename"] in held_out_filenames:
            r["split"] = "held_out_gen"

    return rows, {
        "ai_held_out": len(ai_held_out),
        "real_held_out": len(real_held_out),
        "generators": sorted(held_out_generators),
    }


def main(manifest_path: str, held_out_generators: list[str]) -> None:
    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    rows = load_manifest(path)
    if not rows:
        raise SystemExit(
            f"{path} has no rows yet — Data lead needs to populate it with "
            "matched real/AI images before you can freeze a test set."
        )
    fieldnames = list(rows[0].keys())

    rows, summary = freeze(rows, held_out_generators)
    write_manifest(path, rows, fieldnames)

    print(
        f"Froze {summary['ai_held_out']} AI images from "
        f"{summary['generators']} and {summary['real_held_out']} matched "
        f"real images as split=held_out_gen in {path}"
    )
    print(
        "Commit this updated manifest.csv to git now. From this point on, "
        "nobody — including you — trains on these rows or looks at them "
        "again until it's time to score a checkpoint."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/manifests/manifest.csv",
        help="Path to the manifest CSV (default: data/manifests/manifest.csv)",
    )
    parser.add_argument(
        "--held-out-generators",
        nargs="+",
        required=True,
        help="Generator name(s) exactly as they appear in the manifest's "
        "'generator' column, e.g. --held-out-generators midjourney flux",
    )
    args = parser.parse_args()
    main(args.manifest, args.held_out_generators)

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / "data/manifests/manifest.csv"
OUTPUT = PROJECT / "data/manifests/v0_candidate"

SEED = 42
PER_GROUP = 300


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)

    # Recover original photo IDs from the raw files.
    # build_matched.py used the same content hash for filenames.
    photo_ids = {}

    sources = [
        ("data/raw/real", "real", ""),
        ("data/raw/ai/own/stable_diffusion", "ai", "_sd15"),
        ("data/raw/ai/own/flux", "ai", "_flux"),
    ]

    for folder, prefix, suffix in sources:
        for path in sorted((PROJECT / folder).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".webp",
                ".bmp", ".tif", ".tiff"
            }:
                continue

            digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
            matched_name = f"{prefix}_{digest}.jpg"
            pair_id = path.stem.removesuffix(suffix) if suffix else path.stem
            photo_ids.setdefault(matched_name, pair_id)

    real = sorted(
        [r for r in rows if r["is_ai"] == "0"],
        key=lambda r: r["filename"],
    )
    sd = sorted(
        [r for r in rows if r["generator"] == "stable_diffusion"],
        key=lambda r: r["filename"],
    )
    flux = sorted(
        [r for r in rows if r["generator"] == "flux"],
        key=lambda r: r["filename"],
    )

    if len(real) < PER_GROUP * 2:
        raise RuntimeError("Need at least 600 real photos.")

    for row in real + sd + flux:
        if row["filename"] not in photo_ids:
            raise RuntimeError(
                f"Cannot recover original ID: {row['filename']}"
            )

    rng = random.Random(SEED)

    # PROPOSE 300 real photos for evaluation.
    proposed_test_real = rng.sample(real, PER_GROUP)
    test_names = {r["filename"] for r in proposed_test_real}
    test_photo_ids = {
        photo_ids[r["filename"]] for r in proposed_test_real
    }

    training_real_candidates = [
        r for r in real if r["filename"] not in test_names
    ]
    proposed_train_real = rng.sample(
        training_real_candidates, PER_GROUP
    )

    # Exclude SD versions of proposed evaluation real photos.
    eligible_sd = [
        r for r in sd
        if photo_ids[r["filename"]] not in test_photo_ids
    ]

    if len(eligible_sd) < PER_GROUP or len(flux) < PER_GROUP:
        raise RuntimeError("Not enough eligible AI images.")

    proposed_train_sd = rng.sample(eligible_sd, PER_GROUP)
    proposed_test_flux = rng.sample(flux, PER_GROUP)

    groups = [
        (proposed_train_real, "train"),
        (proposed_train_sd, "train"),
        (proposed_test_real, "held_out_gen"),
        (proposed_test_flux, "held_out_gen"),
    ]

    selected = []
    proposal = []

    for group, proposed_split in groups:
        for row in group:
            selected.append(dict(row))
            proposal.append({
                "filename": row["filename"],
                "pair_id": photo_ids[row["filename"]],
                "generator": row["generator"],
                "is_ai": row["is_ai"],
                "proposed_split": proposed_split,
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Keep the original split values; Evaluation must approve/freeze.
    write_csv(OUTPUT / "manifest.csv", selected, fields)

    write_csv(
        OUTPUT / "split_proposal.csv",
        proposal,
        [
            "filename", "pair_id", "generator",
            "is_ai", "proposed_split"
        ],
    )

    (OUTPUT / "README.txt").write_text(
        "PROPOSAL ONLY: NOT APPROVED FOR TRAINING.\n"
        "Images remain in data/matched; filenames are relative to it.\n"
        "Evaluation must review split_proposal.csv and freeze the split.\n"
        "The candidate manifest retains the original split values.\n"
        "Do not train before the freeze is approved.\n"
        "Flux must remain evaluation-only.\n"
        "If changing the proposed real test IDs, recheck their matching\n"
        "Stable Diffusion versions are excluded from training.\n"
        "Do not blindly freeze all Flux rows in the full 3008-row manifest.\n",
        encoding="utf-8",
    )

    counts = Counter(
        (
            r["proposed_split"],
            r["generator"] or "real"
        )
        for r in proposal
    )

    for group, count in sorted(counts.items()):
        print(group, "| Images:", count)

    print("Total candidate images:", len(selected))
    print("Saved proposal to:", OUTPUT)
    print("Full manifest unchanged.")


if __name__ == "__main__":
    main()
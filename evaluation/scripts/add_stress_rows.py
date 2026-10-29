"""
Add real-photo stress-set images to the manifest (split = stress, is_ai = 0).
Stress images stay in data/stress_set/ exactly as they are (screenshots,
filtered, edited or re-compressed real photos); `filename` is the path relative
to data/stress_set/.

Usage:
    python evaluation/scripts/add_stress_rows.py --folder data/stress_set/screenshots --source stress_screenshot
    python evaluation/scripts/add_stress_rows.py --folder data/stress_set/filters --source stress_filter
"""

import argparse
import csv
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
FIELDS = ["filename", "source", "generator", "split", "is_ai"]


def main(folder: str, source: str, manifest: str, stress_root: str) -> None:
    root, folder = Path(stress_root), Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found under {folder}.")
    manifest_path = Path(manifest)
    existing = set()
    if manifest_path.exists() and manifest_path.stat().st_size:
        with open(manifest_path, newline="") as f:
            existing = {row["filename"] for row in csv.DictReader(f)}
    new = [{"filename": p.relative_to(root).as_posix(), "source": source, "generator": "",
            "split": "stress", "is_ai": 0} for p in files if p.relative_to(root).as_posix() not in existing]
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new)
    print(f"Added {len(new)} stress rows ({len(files) - len(new)} already present) for source '{source}'.")
    print("Tell the Data lead: they own manifest.csv. Commit it so the team sees these rows.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--source", required=True, help="e.g. stress_screenshot, stress_filter, stress_crop")
    p.add_argument("--manifest", default="data/manifests/manifest.csv")
    p.add_argument("--stress-root", default="data/stress_set")
    a = p.parse_args()
    main(a.folder, a.source, a.manifest, a.stress_root)

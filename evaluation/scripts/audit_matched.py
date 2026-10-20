"""
Shortcut audit: can an image's label be predicted WITHOUT looking at what is
in it? If a tiny model that only sees file format, dimensions, file size and
JPEG quantization tables can tell real from AI, the real detector can cheat
the same way (docs/plan.md, Pitfalls: "a near-perfect first score").

Reads data/manifests/manifest.csv, opens every file in data/matched/, builds
metadata-only features, and cross-validates a decision tree on them.
  AUC ~ 0.5  -> good: nothing in the metadata gives the label away
  AUC > 0.6  -> shortcut risk; tell Data which column differs (printed below)

Independent of model/ by design. Usage:
    python evaluation/scripts/audit_matched.py
    python evaluation/scripts/audit_matched.py --split train --output evaluation/results/audit.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import get_split, load_manifest  # noqa: E402

RISK_AUC = 0.6


def file_features(path: Path) -> dict:
    with Image.open(path) as img:
        qtables = getattr(img, "quantization", None) or {}
        mean_q = float(np.mean([v for t in qtables.values() for v in t])) if qtables else -1.0
        return {
            "format": img.format or "?",
            "width": img.width,
            "height": img.height,
            "file_kb": path.stat().st_size / 1024,
            "mean_quant": mean_q,            # lower = higher JPEG quality
            "has_exif": int(bool(img.getexif())),
            "mode": img.mode,
        }


def build_table(rows: list[dict], matched_dir: str) -> pd.DataFrame:
    records = []
    for row in rows:
        path = Path(matched_dir) / row["filename"]
        if not path.exists():
            print(f"missing file listed in manifest: {path}")
            continue
        records.append({"filename": row["filename"], "is_ai": int(row["is_ai"]),
                        "generator": row.get("generator", ""), **file_features(path)})
    return pd.DataFrame(records)


def shortcut_auc(table: pd.DataFrame, seed: int = 42) -> float:
    """5-fold cross-validated AUC of a shallow decision tree on metadata only."""
    X = pd.get_dummies(table[["format", "mode", "width", "height", "file_kb",
                              "mean_quant", "has_exif"]], columns=["format", "mode"])
    y = table["is_ai"].to_numpy()
    folds = max(2, min(5, int(np.bincount(y).min())))
    tree = DecisionTreeClassifier(max_depth=3, random_state=seed)
    proba = cross_val_predict(tree, X, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def main(manifest: str, matched_dir: str, split: str | None, output: str | None) -> None:
    rows = load_manifest(manifest)
    if split:
        rows = get_split(rows, split)
    else:   # stress-set images live in data/stress_set/, not data/matched/
        rows = [r for r in rows if r["split"] != "stress"]
    if not rows:
        raise SystemExit("No rows to audit. Has Data filled in manifest.csv and data/matched/ yet?")
    table = build_table(rows, matched_dir)
    if table.is_ai.nunique() < 2 or table.is_ai.value_counts().min() < 2:
        raise SystemExit("Need at least 2 real and 2 AI images to audit.")

    print(f"Audited {len(table)} images  ({int(table.is_ai.sum())} AI, {int((1 - table.is_ai).sum())} real)\n")
    for col in ("format", "mode", "has_exif"):
        print(pd.crosstab(table[col], table["is_ai"].map({0: "real", 1: "ai"})), "\n")
    summary = table.groupby(table["is_ai"].map({0: "real", 1: "ai"}))[
        ["width", "height", "file_kb", "mean_quant"]].describe().T
    print(summary.to_string(), "\n")

    auc = shortcut_auc(table)
    print(f"Metadata-only shortcut AUC (cross-validated): {auc:.3f}")
    if auc > RISK_AUC:
        print(f"SHORTCUT RISK (> {RISK_AUC}): the label is predictable from file properties alone. "
              "Look at the tables above for the column that differs and tell the Data lead.")
    else:
        print("OK: file properties do not give the label away.")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        print(f"Wrote {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/manifest.csv")
    p.add_argument("--matched-dir", default="data/matched")
    p.add_argument("--split", default=None, help="only audit this split (default: all rows)")
    p.add_argument("--output", default=None, help="optional CSV of per-image features")
    a = p.parse_args()
    main(a.manifest, a.matched_dir, a.split, a.output)

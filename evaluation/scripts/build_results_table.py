"""
The results table Evaluation posts before every weekly meeting.

For one checkpoint, on the frozen held-out-generator split, it reports AUC and
TPR at 1% FPR  (a) per held-out generator and for ALL of them together, and
(b) clean and after every degradation in degradation_suite.py (JPEG 75/50/30,
resize 0.5x/1.5x, crop, screenshot). Degradations are applied in memory.

It also writes:
  * the 1%-FPR score threshold (evaluation/results/threshold.json) -- the
    cut-off Product should use, and the one score_stress_set.py needs;
  * the 20 worst mistakes on clean images (evaluation/results/week4_errors.csv)
    -- "twenty misclassified images teach more than any accuracy number".

Independent of model/: it only uses this folder's own scoring + degradation code.

Usage:
    python evaluation/scripts/build_results_table.py \
        --checkpoint model/checkpoints/step2.pt \
        --output evaluation/results/week4_table.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import get_split, load_manifest  # noqa: E402

from degradation_suite import VARIANTS  # noqa: E402
from score_checkpoint import (  # noqa: E402
    compute_auc_and_tpr_at_fpr,
    load_checkpoint,
    resolve_backbone_id,
    score_pil_image,
)

DEGRADATIONS = {
    "clean": None,
    "jpeg_75": VARIANTS["jpeg_75"],
    "jpeg_50": VARIANTS["jpeg_50"],
    "jpeg_30": VARIANTS["jpeg_30"],
    "resize_0.5x": VARIANTS["resize_0.5x"],
    "resize_1.5x": VARIANTS["resize_1.5x"],
    "crop": VARIANTS["crop"],
    "screenshot": VARIANTS["screenshot"],
}


def threshold_at_fpr(real_scores, fpr: float = 0.01) -> float:
    """Smallest cut-off such that at most `fpr` of REAL images score >= it.
    Pick it on held-out real images only; it is what makes "1% false
    positives" mean something outside this table."""
    r = np.sort(np.asarray(real_scores, dtype=float))[::-1]
    allowed = int(np.floor(fpr * len(r)))
    return float(np.nextafter(r[min(allowed, len(r) - 1)], np.inf))


def summarize(rows: list[dict], scores_by_degradation: dict[str, list[float]]) -> list[dict]:
    """rows and every score list are in the same order. One output row per
    (generator, degradation); generator 'ALL' pools every held-out generator.
    Held-out REAL rows have no generator, so each generator is scored against
    all the real rows."""
    is_ai = np.array([int(r["is_ai"]) for r in rows])
    generators = np.array([r["generator"] for r in rows])
    groups = {"ALL": np.ones(len(rows), dtype=bool)}
    for g in sorted({r["generator"] for r in rows if int(r["is_ai"]) == 1 and r["generator"]}):
        groups[g] = (is_ai == 0) | (generators == g)

    out = []
    for deg, scores in scores_by_degradation.items():
        scores = np.asarray(scores, dtype=float)
        for g, mask in groups.items():
            y = is_ai[mask]
            if y.min() == y.max():
                print(f"  skipping {g}/{deg}: only one class present")
                continue
            m = compute_auc_and_tpr_at_fpr(y, scores[mask])
            out.append({"generator": g, "degradation": deg,
                        "n_ai": int(y.sum()), "n_real": int((1 - y).sum()),
                        "auc": round(m["auc"], 3),
                        "tpr_at_1pct_fpr": round(m["tpr_at_fpr"], 3)})
    return out


def worst_errors(rows: list[dict], scores, k: int = 20) -> list[dict]:
    """The k highest-scored real images (false positives) and the k
    lowest-scored AI images (false negatives)."""
    scores = np.asarray(scores, dtype=float)
    is_ai = np.array([int(r["is_ai"]) for r in rows])
    out = []
    for kind, idx in (("false_positive", np.where(is_ai == 0)[0][np.argsort(-scores[is_ai == 0])][:k]),
                      ("false_negative", np.where(is_ai == 1)[0][np.argsort(scores[is_ai == 1])][:k])):
        for i in idx:
            out.append({"kind": kind, "filename": rows[i]["filename"],
                        "generator": rows[i]["generator"], "score": round(float(scores[i]), 4)})
    return out


def main(checkpoint, test_manifest, matched_dir, split, output, errors_output, threshold_output):
    ckpt = load_checkpoint(checkpoint)
    backbone_id = resolve_backbone_id(ckpt["backbone"])
    clf = ckpt["classifier"]

    rows = get_split(load_manifest(test_manifest), split)
    if not rows:
        raise SystemExit(f"No rows with split='{split}' in {test_manifest}. "
                         "Run evaluation/scripts/freeze_held_out.py first.")

    scores = {name: [] for name in DEGRADATIONS}
    for i, row in enumerate(rows, 1):
        image = Image.open(Path(matched_dir) / row["filename"]).convert("RGB")
        for name, degrade in DEGRADATIONS.items():
            scores[name].append(score_pil_image(degrade(image) if degrade else image, backbone_id, clf))
        if i % 100 == 0:
            print(f"  scored {i}/{len(rows)} images")

    table = pd.DataFrame(summarize(rows, scores))
    print("\n" + table.to_string(index=False))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    pd.DataFrame(worst_errors(rows, scores["clean"])).to_csv(errors_output, index=False)

    real_clean = [s for s, r in zip(scores["clean"], rows) if int(r["is_ai"]) == 0]
    threshold = threshold_at_fpr(real_clean)
    Path(threshold_output).write_text(json.dumps({
        "checkpoint": str(checkpoint), "fpr": 0.01, "threshold": threshold,
        "n_real": len(real_clean)}, indent=2))
    print(f"\n1%-FPR threshold on clean held-out real images: {threshold:.4f} "
          f"(from {len(real_clean)} real images)")
    if len(real_clean) < 300:
        print(f"WARNING: only {len(real_clean)} real held-out images, so '1% FPR' allows "
              f"{len(real_clean) // 100} false positive(s) and is very noisy. Ask Data for more real photos.")
    auc_all = table[(table.generator == "ALL") & (table.degradation == "clean")]["auc"]
    if len(auc_all) and auc_all.iloc[0] > 0.99:
        print("WARNING: clean AUC > 0.99 -- almost certainly a shortcut. Run audit_matched.py.")
    print(f"\nWrote {output}, {errors_output}, {threshold_output}")
    print("Paste the table into evaluation/results/weekly_notes.md and name the biggest problem.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-manifest", default="data/manifests/manifest.csv")
    p.add_argument("--matched-dir", default="data/matched")
    p.add_argument("--split", default="held_out_gen")
    p.add_argument("--output", default="evaluation/results/week4_table.csv")
    p.add_argument("--errors-output", default="evaluation/results/week4_errors.csv")
    p.add_argument("--threshold-output", default="evaluation/results/threshold.json")
    a = p.parse_args()
    main(a.checkpoint, a.test_manifest, a.matched_dir, a.split, a.output,
         a.errors_output, a.threshold_output)

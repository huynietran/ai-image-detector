"""
Score the real-photo stress set: real photos that were edited, filtered,
re-compressed or screenshotted (manifest rows with split == "stress").

Every image here is REAL, so the only number that matters is the false-positive
rate: how many get flagged as AI at the threshold chosen on clean held-out
real images (evaluation/results/threshold.json, written by
build_results_table.py). Reported per `source` so you can say "screenshots
are flagged 22% of the time, filters 4%".

Independent of model/. Usage:
    python evaluation/scripts/score_stress_set.py --checkpoint model/checkpoints/step2.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import get_split, load_manifest  # noqa: E402

from score_checkpoint import load_checkpoint, resolve_backbone_id, score_rows  # noqa: E402


def fpr_by_source(rows: list[dict], scores, threshold: float) -> pd.DataFrame:
    """rows/scores aligned; counts a real image as a false positive when
    score >= threshold. Only real (is_ai == 0) rows are counted."""
    df = pd.DataFrame({"source": [r["source"] for r in rows],
                       "is_ai": [int(r["is_ai"]) for r in rows],
                       "score": np.asarray(scores, dtype=float)})
    df = df[df.is_ai == 0]
    df["flagged"] = df.score >= threshold
    out = df.groupby("source").agg(n=("flagged", "size"), false_positives=("flagged", "sum"),
                                   fpr=("flagged", "mean"), mean_score=("score", "mean")).reset_index()
    total = pd.DataFrame([{"source": "ALL", "n": len(df), "false_positives": int(df.flagged.sum()),
                           "fpr": df.flagged.mean(), "mean_score": df.score.mean()}])
    return pd.concat([out, total], ignore_index=True).round(3)


def main(checkpoint, test_manifest, matched_dir, threshold_file, output):
    info = json.loads(Path(threshold_file).read_text()) if Path(threshold_file).exists() else None
    if info is None:
        raise SystemExit(f"{threshold_file} not found. Run build_results_table.py first -- the "
                         "stress-set false-positive rate needs the 1%-FPR threshold from clean held-out data.")
    if info["checkpoint"] != str(checkpoint):
        raise SystemExit(f"{threshold_file} was computed for {info['checkpoint']}, not {checkpoint}. "
                         "Re-run build_results_table.py for this checkpoint.")

    rows = get_split(load_manifest(test_manifest), "stress")
    if not rows:
        raise SystemExit("No rows with split='stress' yet. Build the stress set first (Week 4, Task 4).")
    if any(int(r["is_ai"]) == 1 for r in rows):
        print("NOTE: AI rows in the stress split are ignored here (this report is false positives on real photos).")

    ckpt = load_checkpoint(checkpoint)
    scores, _ = score_rows(rows, matched_dir, resolve_backbone_id(ckpt["backbone"]), ckpt["classifier"])
    table = fpr_by_source(rows, scores, info["threshold"])
    print(f"threshold = {info['threshold']:.4f}  (1% FPR on {info['n_real']} clean held-out real images)\n")
    print(table.to_string(index=False))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    worst = sorted(zip(scores, (r["filename"] for r in rows)), reverse=True)[:10]
    print("\n10 highest-scored real images (look at these!):")
    for s, f in worst:
        print(f"  {s:.3f}  {f}")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-manifest", default="data/manifests/manifest.csv")
    p.add_argument("--matched-dir", default="data/matched")
    p.add_argument("--threshold-file", default="evaluation/results/threshold.json")
    p.add_argument("--output", default="evaluation/results/stress_fpr.csv")
    a = p.parse_args()
    main(a.checkpoint, a.test_manifest, a.matched_dir, a.threshold_file, a.output)

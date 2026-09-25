"""
Independently score a Model checkpoint on the frozen held-out-generator
test set. Deliberately does NOT import anything from model/ — re-implements
loading + inference so this check stays independent of Model's own code.

Usage:
    python evaluation/scripts/score_checkpoint.py \
        --checkpoint model/checkpoints/step2.pt \
        --test-manifest data/manifests/manifest.csv \
        --split held_out_gen
"""


def compute_auc_and_tpr_at_fpr(y_true, y_scores, target_fpr: float = 0.01):
    """Report AUC and TPR at a fixed FPR (default 1%) — see docs/plan.md
    Pitfalls: accuracy alone hides false positives on real photos."""
    raise NotImplementedError


def main(checkpoint: str, test_manifest: str, split: str):
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--split", default="held_out_gen")
    args = parser.parse_args()
    main(args.checkpoint, args.test_manifest, args.split)

"""
AUC and TPR@FPR. Used by model/steps/ for quick in-loop checks during
training. Evaluation's official numbers should be computed independently
in evaluation/scripts/ — not by importing this file — so the check stays
meaningful (see evaluation/README.md).
"""

from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true, y_scores) -> float:
    return roc_auc_score(y_true, y_scores)


def tpr_at_fpr(y_true, y_scores, target_fpr: float = 0.01) -> float:
    """TPR at a fixed FPR. Report this alongside AUC always — accuracy
    alone hides false positives on real photographers' work
    (see docs/plan.md Pitfalls)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = next((i for i, f in enumerate(fpr) if f > target_fpr), len(fpr) - 1)
    return tpr[max(idx - 1, 0)]

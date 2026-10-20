"""
Independently score a Model checkpoint on the frozen held-out-generator
test set. Deliberately does NOT import anything from model/ or shared/
model_interface.py -- it re-implements loading + inference so this check stays
independent of Model's own code.

Checkpoint format (agreed with the Model lead):

    torch.save({"backbone": "dinov2_vitb14",            # short name or HF hub id
                "classifier": <fitted sklearn estimator with .predict_proba>,
                "crops": "center+4corners@224"}, path)

Preprocessing recipe (written down in docs, implemented separately by Model,
Evaluation and Product -- if the numbers disagree, compare these first):
    5 native-resolution 224x224 crops (center + 4 corners), NO resizing of the
    whole image (images with a short side < 224 px are upscaled to 224 first),
    ImageNet mean/std normalization, DINOv2 CLS token per crop, classifier
    probability per crop, image score = mean of the 5 probabilities.

Usage:
    python evaluation/scripts/score_checkpoint.py \
        --checkpoint model/checkpoints/step2.pt \
        --test-manifest data/manifests/manifest.csv \
        --split held_out_gen
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModel

# Make `shared/` importable when run directly as
# `python evaluation/scripts/score_checkpoint.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.splits import get_split, load_manifest  # noqa: E402

CROP = 224
CROPS_NAME = "center+4corners@224"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# model/configs/*.yaml refer to backbones by short name (e.g. "dinov2_vitb14").
# Map those to the actual Hugging Face Hub id so this script can load them.
BACKBONE_HF_IDS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}


def resolve_backbone_id(name: str) -> str:
    """Accepts either a short config name or an already-valid HF hub id."""
    return BACKBONE_HF_IDS.get(name, name)


def compute_auc_and_tpr_at_fpr(y_true, y_scores, target_fpr: float = 0.01):
    """Report AUC and TPR at a fixed FPR (default 1%) -- see docs/plan.md
    Pitfalls: accuracy alone hides false positives on real photos."""
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr)
    idx = min(idx, len(tpr) - 1)
    return {"auc": auc, "tpr_at_fpr": float(tpr[idx]), "target_fpr": target_fpr}


def load_checkpoint(path: str) -> dict:
    """Independent checkpoint loader -- does not call into model/ or
    shared/model_interface.py (that's the Model -> Product contract, this
    is Model -> Evaluation, and ours must be re-implemented separately)."""
    if not Path(path).exists():
        raise SystemExit(
            f"Checkpoint not found: {path}. Has Model lead saved one from "
            "model/steps/step2_baseline_logreg.py yet?"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "classifier" not in ckpt or "backbone" not in ckpt:
        raise ValueError(
            f"Checkpoint at {path} doesn't have the expected "
            "{'backbone', 'classifier'} keys. If Model lead changed the "
            "checkpoint format, update load_checkpoint() in this file to match."
        )
    if ckpt.get("crops") != CROPS_NAME:
        raise ValueError(
            f"Checkpoint at {path} reports crops={ckpt.get('crops')!r}, but this "
            f"scorer implements {CROPS_NAME!r}. Scores would not be comparable."
        )
    return ckpt


_backbone_cache: dict = {}


def _get_backbone(backbone_id: str):
    """Own copy of the embedding pipeline -- intentionally duplicated from
    model/, not imported, so a bug there can't hide a bug here."""
    if backbone_id not in _backbone_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(backbone_id).to(device).eval()
        _backbone_cache[backbone_id] = (model, device)
    return _backbone_cache[backbone_id]


def native_crops(image: Image.Image) -> list[Image.Image]:
    """Center + 4 corner crops at native resolution (no whole-image resize)."""
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        image = image.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
        w, h = image.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    origins = [(cx, cy), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    return [image.crop((x, y, x + CROP, y + CROP)) for x, y in origins]


def embed_crops(image: Image.Image, backbone_id: str) -> np.ndarray:
    """-> [5, hidden] CLS embeddings, one per crop."""
    model, device = _get_backbone(backbone_id)
    arr = np.stack([(np.asarray(c, dtype=np.float32) / 255.0 - MEAN) / STD for c in native_crops(image)])
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
    with torch.no_grad():
        out = model(pixel_values=batch)
    return out.last_hidden_state[:, 0, :].float().cpu().numpy()


def score_pil_image(image: Image.Image, backbone_id: str, clf) -> float:
    """P(AI) for one image: mean of the per-crop probabilities."""
    return float(np.mean(clf.predict_proba(embed_crops(image, backbone_id))[:, 1]))


def score_rows(rows: list[dict], matched_dir: str, backbone_id: str, clf):
    scores, labels = [], []
    for row in rows:
        image = Image.open(Path(matched_dir) / row["filename"]).convert("RGB")
        scores.append(score_pil_image(image, backbone_id, clf))
        labels.append(int(row["is_ai"]))
    return scores, labels


def main(checkpoint: str, test_manifest: str, split: str, matched_dir: str = "data/matched"):
    ckpt = load_checkpoint(checkpoint)
    backbone_id = resolve_backbone_id(ckpt["backbone"])
    clf = ckpt["classifier"]

    manifest = load_manifest(test_manifest)
    rows = get_split(manifest, split)
    if not rows:
        raise SystemExit(
            f"No rows with split='{split}' in {test_manifest}. Run "
            "evaluation/scripts/freeze_held_out.py first."
        )

    scores, labels = score_rows(rows, matched_dir, backbone_id, clf)
    metrics = compute_auc_and_tpr_at_fpr(labels, scores)

    print(
        f"[{split}] n={len(rows)}  AUC={metrics['auc']:.3f}  "
        f"TPR@{metrics['target_fpr']:.0%}FPR={metrics['tpr_at_fpr']:.3f}"
    )
    if metrics["auc"] > 0.99:
        print(
            "WARNING: AUC > 0.99 on the first run is almost always a "
            "shortcut, not a working detector -- see docs/plan.md Pitfalls."
        )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--split", default="held_out_gen")
    parser.add_argument(
        "--matched-dir",
        default="data/matched",
        help="Folder holding the actual image files the manifest's "
        "'filename' column refers to (default: data/matched)",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.test_manifest, args.split, args.matched_dir)

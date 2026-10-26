"""shared/model_interface.py — the one place Product (and Model's own steps) load and score a checkpoint.

Product only calls load_checkpoint() / score_image() / predict(); checkpoints from any
model/steps/stepN_*.py swap in without Product code changes.

Preprocessing recipe "center+4corners@224" (Evaluation implements it independently):
  * five native-resolution 224 px crops: centre + four corners, NO whole-image resize
    (upscale only if the short side is under 224 px)
  * normalise with the backbone's mean/std (ImageNet for DINOv2; see BACKBONES)
  * one embedding per crop (DINOv2: CLS token of last_hidden_state)
  * image score = mean of the five crop probabilities

Checkpoint contract (torch.save dict):
  required  "backbone":   short name from BACKBONES, e.g. "dinov2_vitb14"
                          (suffix "+lora" means "weight_deltas" must be applied — step 5)
            "classifier": fitted sklearn estimator with .predict_proba
            "crops":      "center+4corners@224"
  optional  "weight_deltas": {param name: tensor} added to the pretrained weights (step 5)
            "lowlevel":      LOWLEVEL_NAME; per-crop features from lowlevel_features() are
                             appended after the embedding before the classifier (step 7)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

CROP = 224
CROPS_NAME = "center+4corners@224"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Short names used in configs and checkpoints. "embedding" is which output is used per crop.
BACKBONES: Dict[str, dict] = {
    "dinov2_vits14": dict(hf_id="facebook/dinov2-small", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "dinov2_vitb14": dict(hf_id="facebook/dinov2-base", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "dinov2_vitl14": dict(hf_id="facebook/dinov2-large", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "clip_vitb16": dict(hf_id="openai/clip-vit-base-patch16", kind="clip", mean=CLIP_MEAN, std=CLIP_STD),
    "clip_vitl14": dict(hf_id="openai/clip-vit-large-patch14", kind="clip", mean=CLIP_MEAN, std=CLIP_STD),
    "siglip_vitb16": dict(hf_id="google/siglip-base-patch16-224", kind="siglip", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
EMBEDDING_OUTPUT = {"dinov2": "last_hidden_state[:, 0] (CLS)",
                    "clip": "pooler_output (post-layernorm CLS)",
                    "siglip": "pooler_output (attention-pooling head)"}


# ---------------------------------------------------------------- results
@dataclass
class DetectionResult:
    score: float                       # P(AI), mean of the crop probabilities
    crop_scores: List[float] = field(default_factory=list)
    backbone: str = ""
    checkpoint_id: str = ""


# ---------------------------------------------------------------- preprocessing
def native_crops(image: Image.Image) -> List[Image.Image]:
    """Centre + four corner 224 px crops at native resolution (exactly the written recipe)."""
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < CROP:  # upscale only when too small
        s = CROP / min(w, h)
        image = image.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
        w, h = image.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    origins = [(cx, cy), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    return [image.crop((x, y, x + CROP, y + CROP)) for x, y in origins]


def crops_to_tensor(crops: List[Image.Image], mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """List of PIL crops -> normalised float tensor [k, 3, H, W]."""
    arr = np.stack([np.asarray(c.convert("RGB"), dtype=np.float32) / 255.0 for c in crops])
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------- low-level branch (step 7)
LOWLEVEL_NAME = "hp_residual_fft32+stats9"
_FFT_BINS = 32


def _box_blur3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    h, w = x.shape[:2]
    return sum(p[i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def lowlevel_features(crop: Image.Image) -> np.ndarray:
    """41 hand-made features of a crop's high-frequency residual (what generators get wrong
    at the pixel level): log radial power spectrum in 32 bins + std / mean|.| / kurtosis per channel."""
    x = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    res = x - _box_blur3(x)                                   # high-pass residual, [H, W, 3]
    flat = res.reshape(-1, 3)
    std = flat.std(0) + 1e-8
    kurt = (((flat - flat.mean(0)) / std) ** 4).mean(0) - 3.0
    stats = np.concatenate([std, np.abs(flat).mean(0), kurt])
    g = res.mean(axis=2)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.indices((h, w))
    r = np.hypot((yy - h // 2) / h, (xx - w // 2) / w)        # cycles/pixel, 0..~0.707
    bins = np.minimum((r / r.max() * _FFT_BINS).astype(int), _FFT_BINS - 1)
    radial = np.bincount(bins.ravel(), power.ravel(), _FFT_BINS) / np.maximum(np.bincount(bins.ravel(), minlength=_FFT_BINS), 1)
    return np.concatenate([np.log10(radial + 1e-12), stats]).astype(np.float32)


# ---------------------------------------------------------------- backbone
def base_name(backbone_name: str) -> str:
    return backbone_name.split("+", 1)[0]


def _hf_model(spec: dict) -> torch.nn.Module:
    """Load the pretrained vision tower. (The smoke test swaps this for tiny random models.)"""
    if spec["kind"] == "dinov2":
        from transformers import AutoModel
        return AutoModel.from_pretrained(spec["hf_id"])
    if spec["kind"] == "clip":
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(spec["hf_id"])
    if spec["kind"] == "siglip":
        from transformers import SiglipVisionModel
        return SiglipVisionModel.from_pretrained(spec["hf_id"])
    raise ValueError(spec["kind"])


# transformers 4.x names DINOv2's attention "attention.attention.query"; 5.x names it "attention.q_proj".
_RENAMES = [("attention.attention.query.", "attention.q_proj."), ("attention.attention.key.", "attention.k_proj."),
            ("attention.attention.value.", "attention.v_proj."), ("attention.output.dense.", "attention.o_proj.")]


def _match_param(name, params):
    """The model's own name for a saved parameter, across transformers versions."""
    if name in params:
        return name
    for old, new in _RENAMES:
        for a, b in ((old, new), (new, old)):
            if a in name and name.replace(a, b) in params:
                return name.replace(a, b)
    return None


class Backbone:
    """A frozen vision tower plus the normalisation and output that go with it."""

    def __init__(self, name: str, weight_deltas: Optional[Dict[str, torch.Tensor]] = None, device=None):
        key = base_name(name)
        if key not in BACKBONES:
            raise ValueError(f"unknown backbone '{name}'; expected one of {sorted(BACKBONES)} (optionally +lora)")
        if name.endswith("+lora") and not weight_deltas:
            raise ValueError(f"backbone '{name}' needs weight_deltas (a step-5 checkpoint)")
        self.name, self.spec = name, BACKBONES[key]
        self.kind, self.mean, self.std = self.spec["kind"], self.spec["mean"], self.spec["std"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _hf_model(self.spec)
        if weight_deltas:
            params = dict(self.model.named_parameters())
            matched = {n: _match_param(n, params) for n in weight_deltas}
            missing = [n for n, m in matched.items() if m is None]
            if missing:
                raise ValueError(f"weight_deltas don't match {key}: e.g. {missing[:2]}")
            with torch.no_grad():
                for n, d in weight_deltas.items():
                    p = params[matched[n]]
                    p.add_(d.to(p.dtype))
        self.model.to(self.device).eval()

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Differentiable per-crop embedding (used by step 5's fine-tuning)."""
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0] if self.kind == "dinov2" else out.pooler_output

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> np.ndarray:
        """[n, 3, 224, 224] normalised -> embeddings [n, dim] float32 numpy."""
        self.model.eval()
        return self.features(batch.to(self.device)).float().cpu().numpy()

    def to_tensor(self, crops: List[Image.Image]) -> torch.Tensor:
        return crops_to_tensor(crops, self.mean, self.std)


def load_backbone(backbone_name: str, weight_deltas=None) -> Backbone:
    return Backbone(backbone_name, weight_deltas)


# ---------------------------------------------------------------- public API
@dataclass
class LoadedCheckpoint:
    backbone_name: str
    classifier: Any
    backbone: Backbone
    lowlevel: Optional[str]
    checkpoint_id: str


def load_checkpoint(path) -> LoadedCheckpoint:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("crops") != CROPS_NAME:
        raise ValueError(f"{path} was not trained with the '{CROPS_NAME}' recipe")
    lowlevel = ckpt.get("lowlevel")
    if lowlevel not in (None, LOWLEVEL_NAME):
        raise ValueError(f"{path} uses low-level features '{lowlevel}', this code implements '{LOWLEVEL_NAME}'")
    backbone = load_backbone(ckpt["backbone"], ckpt.get("weight_deltas"))
    return LoadedCheckpoint(ckpt["backbone"], ckpt["classifier"], backbone, lowlevel, Path(path).stem)


def crop_inputs(model: LoadedCheckpoint, crops: List[Image.Image]) -> np.ndarray:
    """The classifier's input rows for a list of crops: embedding (+ low-level features)."""
    x = model.backbone.embed(model.backbone.to_tensor(crops))
    if model.lowlevel:
        x = np.concatenate([x, np.stack([lowlevel_features(c) for c in crops])], axis=1)
    return x


def score_pil_image(model: LoadedCheckpoint, image: Image.Image) -> DetectionResult:
    probs = model.classifier.predict_proba(crop_inputs(model, native_crops(image)))[:, 1]
    return DetectionResult(score=float(np.mean(probs)), crop_scores=[float(p) for p in probs],
                           backbone=model.backbone_name, checkpoint_id=model.checkpoint_id)


def score_image(model: LoadedCheckpoint, image_path) -> DetectionResult:
    with Image.open(image_path) as img:
        return score_pil_image(model, img)


def predict(model: LoadedCheckpoint, image_path, threshold: float = 0.5):
    """Week 3 handoff: (label, P(AI)). Use Evaluation's threshold.json, not 0.5, once it exists."""
    score = score_image(model, image_path).score
    return ("ai-generated" if score >= threshold else "real"), score

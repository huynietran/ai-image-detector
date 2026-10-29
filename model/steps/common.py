"""
Helpers shared by model/steps/stepN_*.py. Not a step itself.

Everything that embeds an image goes through shared/model_interface.py, so the
training code and the interface Product/Evaluation load can't drift apart.

Embedding caches live in model/embeddings/ (gitignored):
    <backbone_tag>.npz              clean images (what step 1 writes)
    <backbone_tag>__<tag>.npz       transformed images, e.g. aug0_seed42, jpeg50
    lowlevel__<tag>.npz             step 7's low-level features
Each holds emb [n, 5, dim] (float16) plus the filenames, and is extended when new
manifest rows appear instead of being rebuilt.
"""
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared import model_interface as mi  # noqa: E402
from shared.metrics import auc, tpr_at_fpr  # noqa: E402
from shared.splits import get_split, held_out_generators, load_manifest  # noqa: E402

N_CROPS = 5


# ---------------------------------------------------------------- config + rows
def load_config(path):
    cfg = yaml.safe_load(open(path))
    cfg.setdefault("seed", 42)
    cfg.setdefault("train", {})
    cfg.setdefault("run", {})
    cfg["run"].setdefault("batch_size", 16)
    cfg["run"].setdefault("num_workers", 2)
    cfg["data"].setdefault("embeddings_dir", "model/embeddings/")
    return cfg


def check_no_leakage(manifest):
    held_out = {g for g in held_out_generators(manifest) if g}  # drop the blank real entry
    train_generators = {r["generator"] for r in get_split(manifest, "train") if r["generator"]}
    if held_out & train_generators:
        raise SystemExit(f"LEAK: held-out generator(s) {sorted(held_out & train_generators)} appear in train")
    return held_out


def load_rows(cfg, include_generators=()):
    """(train rows, held-out rows, held-out generator names). Train rows skip any generator in
    train.exclude_generators unless it is listed in include_generators (one change at a time)."""
    manifest = load_manifest(cfg["data"]["manifest"])
    held_out = check_no_leakage(manifest)
    exclude = set(cfg["train"].get("exclude_generators") or []) - set(include_generators)
    train = [r for r in get_split(manifest, "train") if r["generator"] not in exclude]
    held = get_split(manifest, "held_out_gen")
    if not train:
        raise SystemExit(f"No train rows in {cfg['data']['manifest']}. Has Data handed over v0?")
    if not held:
        print("WARNING: no held_out_gen rows yet. Evaluation hasn't frozen, so there is no held-out number.")
    return train, held, held_out


def labels(rows):
    return np.array([int(r["is_ai"]) for r in rows])


# ---------------------------------------------------------------- image transforms
def jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_degrade(img, rng):
    """Step 3's training augmentation: random crop (60-100% per side), resize 0.5-1.5x,
    JPEG 30-100, in the order an upload pipeline applies them. Never below 224 px."""
    img = img.convert("RGB")
    w, h = img.size
    if rng.random() < 0.5:                               # crop
        f = rng.uniform(0.6, 1.0)
        cw, ch = max(mi.CROP, int(w * f)), max(mi.CROP, int(h * f))
        cw, ch = min(cw, w), min(ch, h)
        x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        img = img.crop((x, y, x + cw, y + ch))
        w, h = img.size
    if rng.random() < 0.5:                               # resize
        s = max(rng.uniform(0.5, 1.5), mi.CROP / min(w, h))
        img = img.resize((max(mi.CROP, round(w * s)), max(mi.CROP, round(h * s))), Image.BICUBIC)
    if rng.random() < 0.7:                               # JPEG
        img = jpeg(img, rng.integers(30, 101))
    return img


class Augment:
    """Deterministic per (seed, copy, filename): the same augmented image every time it's
    rebuilt, so step 7 can recompute features for exactly the images step 3 embedded."""

    def __init__(self, seed, copy):
        self.seed, self.copy, self.tag = seed, copy, f"aug{copy}_seed{seed}"

    def __call__(self, img, filename):
        return random_degrade(img, np.random.default_rng([self.seed, self.copy, zlib.crc32(filename.encode())]))


class Jpeg:
    def __init__(self, quality):
        self.quality, self.tag = quality, f"jpeg{quality}"

    def __call__(self, img, filename):
        return jpeg(img, self.quality)


# ---------------------------------------------------------------- extraction
class CropDataset(torch.utils.data.Dataset):
    """Yields the five recipe crops of each row's image, optionally transformed first.
    mode="tensor" -> normalised [5, 3, 224, 224]; mode="lowlevel" -> features [5, 41]."""

    def __init__(self, rows, root, mean, std, transform=None, mode="tensor"):
        self.rows, self.root, self.mean, self.std = rows, Path(root), mean, std
        self.transform, self.mode = transform, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name = self.rows[i]["filename"]
        with Image.open(self.root / name) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img, name)
            crops = mi.native_crops(img)
        if self.mode == "lowlevel":
            return torch.from_numpy(np.stack([mi.lowlevel_features(c) for c in crops]))
        return mi.crops_to_tensor(crops, self.mean, self.std)


def _check_files(rows, root):
    missing = [r["filename"] for r in rows if not (Path(root) / r["filename"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest files missing from {root}, e.g. {missing[:3]}")


def extract(rows, root, backbone, transform=None, batch_size=16, num_workers=2, mode="tensor"):
    """-> float16 [n, 5, dim] embeddings (mode="tensor") or float32 [n, 5, 41] (mode="lowlevel")."""
    _check_files(rows, root)
    mean = backbone.mean if backbone is not None else mi.IMAGENET_MEAN
    std = backbone.std if backbone is not None else mi.IMAGENET_STD
    ds = CropDataset(rows, root, mean, std, transform, mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    chunks = []
    for i, batch in enumerate(loader):
        if mode == "lowlevel":
            chunks.append(batch.numpy().astype(np.float32))
        else:
            b, k = batch.shape[:2]
            emb = backbone.embed(batch.view(b * k, *batch.shape[2:]))
            chunks.append(emb.reshape(b, k, -1).astype(np.float16))
        if i % 50 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    return np.concatenate(chunks)


def deltas_tag(weight_deltas):
    h = hashlib.sha1()
    for n in sorted(weight_deltas):
        h.update(n.encode())
        h.update(weight_deltas[n].float().cpu().numpy().tobytes())
    return h.hexdigest()[:8]


def cache_path(cfg, prefix, transform):
    tag = "" if transform is None else f"__{transform.tag}"
    return Path(cfg["data"]["embeddings_dir"]) / f"{prefix}{tag}.npz"


def cached(cfg, rows, backbone=None, backbone_tag=None, transform=None, mode="tensor"):
    """Row-ordered float32 [n, 5, d]; loads the cache, extracts only rows it lacks, saves."""
    prefix = "lowlevel" if mode == "lowlevel" else backbone_tag
    path = cache_path(cfg, prefix, transform)
    tag = "clean" if transform is None else transform.tag
    have_emb, have_names = None, []
    if path.exists():
        z = np.load(path)
        if str(z["crops"]) != mi.CROPS_NAME or str(z["backbone"]) != prefix or str(z["transform"]) != tag:
            raise SystemExit(f"{path} was made with {z['backbone']}/{z['crops']}/{z['transform']}; delete it to rebuild")
        have_emb, have_names = z["emb"], list(z["filenames"])
    index = {f: i for i, f in enumerate(have_names)}
    todo = [r for r in rows if r["filename"] not in index]
    if todo:
        what = "low-level features" if mode == "lowlevel" else f"{prefix} embeddings"
        print(f"  extracting {what} ({tag}) for {len(todo)} images -> {path}")
        if callable(backbone) and not isinstance(backbone, mi.Backbone):
            backbone = backbone()                        # load the backbone only on a cache miss
        new = extract(todo, cfg["data"]["matched_dir"], backbone, transform,
                      cfg["run"]["batch_size"], cfg["run"]["num_workers"], mode)
        have_emb = new if have_emb is None else np.concatenate([have_emb, new.astype(have_emb.dtype)])
        for r in todo:
            index[r["filename"]] = len(have_names)
            have_names.append(r["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, emb=have_emb, filenames=np.array(have_names), backbone=prefix,
                            crops=mi.CROPS_NAME, transform=tag)
    return have_emb[[index[r["filename"]] for r in rows]].astype(np.float32)


# ---------------------------------------------------------------- classifier
def fit_logreg(E, y, seed, C=1.0):
    """One training example per crop. E: [n, 5, d], y: [n]."""
    X = E.reshape(-1, E.shape[-1])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
    clf.fit(X, np.repeat(y, E.shape[1]))
    return clf


def image_scores(clf, E):
    """Mean of the crop probabilities, exactly as score_image() does it."""
    n, k = E.shape[:2]
    return clf.predict_proba(E.reshape(n * k, -1))[:, 1].reshape(n, k).mean(axis=1)


def evaluate(y, scores, label=""):
    """AUC and TPR@1%FPR with the warnings docs/plan.md asks for."""
    out = {"n_ai": int(y.sum()), "n_real": int(len(y) - y.sum())}
    if len(set(y)) < 2:
        print(f"WARNING: {label or 'held-out'} has only one class; AUC undefined.")
        return out
    out.update(auc=float(auc(y, scores)), tpr_at_1pct_fpr=float(tpr_at_fpr(y, scores, 0.01)))
    print(f"  {label:<28} AUC {out['auc']:.4f}   TPR@1%FPR {out['tpr_at_1pct_fpr']:.4f}")
    return out


def warn_results(res):
    if res.get("auc", 0) > 0.99:
        print("WARNING: AUC > 0.99. Suspect a shortcut; tell Evaluation and check with Data (docs/plan.md, Pitfalls).")
    if 0 < res.get("n_real", 0) < 300:
        print(f"WARNING: only {res['n_real']} held-out real images; TPR@1%FPR is noisy below ~300.")


def save(cfg, ckpt, results):
    ckpt_path = Path(cfg["output"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt.setdefault("crops", mi.CROPS_NAME)
    torch.save(ckpt, ckpt_path)
    results["checkpoint"] = str(ckpt_path)
    results["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    log_path = Path(cfg["output"]["results_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {ckpt_path} and {log_path}")
    return results


class LazyBackbone:
    """Loads the backbone the first time an embedding is actually missing from the cache."""

    def __init__(self, name, weight_deltas=None):
        self.name, self.weight_deltas, self._bb = name, weight_deltas, None

    def __call__(self):
        if self._bb is None:
            print(f"  loading backbone {self.name}")
            self._bb = mi.load_backbone(self.name, self.weight_deltas)
        return self._bb


def embedding_features(cfg, backbone_name, weight_deltas=None):
    """feats(rows, transform) -> float32 [n, 5, dim] for one backbone (cached on disk)."""
    lazy = LazyBackbone(backbone_name, weight_deltas)
    tag = backbone_name if not weight_deltas else f"{backbone_name}-{deltas_tag(weight_deltas)}"
    return lambda rows, transform=None: cached(cfg, rows, lazy, tag, transform)


def fit_and_evaluate(cfg, train, held, feats, copies, label=""):
    """Fit the crop-level logreg on clean (+ `copies` augmented) training images and report
    held-out AUC / TPR@1%FPR, clean and after JPEG 50 (Model's quick check; Evaluation's
    independent table is the official number)."""
    seed = cfg["seed"]
    y = labels(train)
    E = [feats(train, None)] + [feats(train, Augment(seed, c)) for c in range(copies)]
    clf = fit_logreg(np.concatenate(E), np.concatenate([y] * len(E)), seed, cfg["train"].get("C", 1.0))
    print(f"  {label or 'fit'}: {len(train)} train images x {len(E)} versions x {N_CROPS} crops")
    res = {"n_train_images": len(train), "train_versions": len(E)}
    if held:
        yh = labels(held)
        res.update(evaluate(yh, image_scores(clf, feats(held, None)), f"{label} held-out clean".strip()))
        res["jpeg50"] = evaluate(yh, image_scores(clf, feats(held, Jpeg(50))), f"{label} held-out JPEG 50".strip())
    return clf, res

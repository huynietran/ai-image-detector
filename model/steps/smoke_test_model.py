"""
Smoke test for every Model step: 1 -> 7, then shared.model_interface on each checkpoint.

Builds ~70 synthetic images (varied sizes, one under 224 px, a fake VAE set) and a
scratch manifest in a temp folder with one generator frozen as held_out_gen, runs
each step with tiny settings, and checks for every checkpoint that
load_checkpoint() + score_image() reproduce the scores the step trained with.
Also checks the guards: leakage, wrong recipe, "+lora" without deltas. If
Evaluation's scorer is in the repo, it cross-checks step 2's scores against it.

Writes nothing into data/, model/ or evaluation/. The AUCs it prints are
meaningless; it proves the plumbing.

    python model/steps/smoke_test_model.py            # real pretrained weights (downloads ~2 GB)
    python model/steps/smoke_test_model.py --tiny     # tiny random backbones, no download, ~1 min on CPU
"""
import csv
import json
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from shared import model_interface as mi  # noqa: E402

FIELDS = ["filename", "source", "generator", "split", "is_ai"]


def tiny_hf_model(spec):
    from transformers import CLIPVisionConfig, CLIPVisionModel, Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
    torch.manual_seed(zlib.crc32(spec["hf_id"].encode()))
    kw = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=224)
    if spec["kind"] == "dinov2":
        return Dinov2Model(Dinov2Config(patch_size=14, **kw))
    if spec["kind"] == "clip":
        return CLIPVisionModel(CLIPVisionConfig(patch_size=16, **kw))
    return SiglipVisionModel(SiglipVisionConfig(patch_size=16, **kw))


def make_data(root, rng):
    rows, sizes, i = [], [(512, 512), (640, 480), (300, 700), (512, 512)], 0
    specs = [("real", "", "train", 0, 16), ("ai", "sd15", "train", 1, 16), ("ai", "vae_reconstruction", "train", 1, 8),
             ("real", "", "held_out_gen", 0, 10), ("ai", "gen_b", "held_out_gen", 1, 10)]
    for kind, gen, split, is_ai, n in specs:
        for _ in range(n):
            w, h = sizes[i % len(sizes)]
            img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            if is_ai:  # smoother pixels, so the classes are separable at all
                img = img.filter(ImageFilter.GaussianBlur(1.2))
            name = f"{gen or kind}_{i:03d}.jpg"
            img.save(root / name, "JPEG", quality=90)
            rows.append({"filename": name, "source": f"smoke_{kind}", "generator": gen, "split": split, "is_ai": is_ai})
            i += 1
    Image.fromarray(rng.integers(0, 256, (150, 200, 3), dtype=np.uint8)).save(root / "tiny.jpg")
    rows.append({"filename": "tiny.jpg", "source": "smoke_real", "generator": "", "split": "train", "is_ai": 0})
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def config(tmp, n, **extra):
    cfg = yaml.safe_load(open(REPO_ROOT / f"model/configs/step{n}.yaml"))
    cfg["data"].update(manifest=str(tmp / "manifest.csv"), matched_dir=str(tmp / "matched"), embeddings_dir=str(tmp / "emb"))
    cfg["output"] = {"checkpoint": str(tmp / f"step{n}.pt"), "results_log": str(tmp / f"step{n}.json")}
    cfg["run"] = {"batch_size": 8, "num_workers": 0}
    for section, values in extra.items():
        cfg.setdefault(section, {}).update(values)
    path = tmp / f"step{n}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path), cfg


def check_interface(cfg, n, held):
    """load_checkpoint + score_image must give the scores the step's own features give."""
    import common
    ckpt = torch.load(cfg["output"]["checkpoint"], weights_only=False)
    feats = common.embedding_features(cfg, ckpt["backbone"], ckpt.get("weight_deltas"))
    E = feats(held)
    if ckpt.get("lowlevel"):
        E = np.concatenate([E, common.cached(cfg, held, None, None, None, mode="lowlevel")], axis=2)
    step_scores = common.image_scores(ckpt["classifier"], E)
    model = mi.load_checkpoint(cfg["output"]["checkpoint"])
    iface = np.array([mi.score_image(model, Path(cfg["data"]["matched_dir"]) / r["filename"]).score for r in held])
    gap = float(np.abs(iface - step_scores).max())
    assert gap < 1e-2, f"step {n}: interface and step disagree by {gap}"
    res = mi.score_image(model, Path(cfg["data"]["matched_dir"]) / held[0]["filename"])
    assert len(res.crop_scores) == 5 and res.checkpoint_id == f"step{n}"
    print(f"  step {n}: interface matches the step's scores (max gap {gap:.1e}), backbone {ckpt['backbone']}")
    return model, iface


def cross_check_evaluation(cfg, held, iface_scores):
    """If Evaluation's independent scorer is present, its scores must match ours (same recipe)."""
    path = REPO_ROOT / "evaluation/scripts/score_checkpoint.py"
    if "score_pil_image" not in path.read_text():
        print("  (Evaluation's scorer not merged yet; skipping the independent cross-check)")
        return
    sys.path.insert(0, str(path.parent))
    import score_checkpoint as sc
    ckpt = sc.load_checkpoint(cfg["output"]["checkpoint"])
    hf_id = sc.resolve_backbone_id(ckpt["backbone"])
    spec = mi.BACKBONES[ckpt["backbone"]]
    sc._backbone_cache[hf_id] = (mi._hf_model(spec).eval(), "cpu")
    theirs = np.array([sc.score_pil_image(Image.open(Path(cfg["data"]["matched_dir"]) / r["filename"]), hf_id, ckpt["classifier"])
                       for r in held])
    gap = float(np.abs(theirs - iface_scores).max())
    assert gap < 1e-4, f"Evaluation's scorer and the interface disagree by {gap}"
    print(f"  Evaluation's independent scorer agrees with the interface (max gap {gap:.1e})")


def expect_exit(fn, text):
    try:
        fn()
    except (SystemExit, ValueError) as e:
        assert text in str(e), e
        return
    raise AssertionError(f"expected a refusal containing {text!r}")


def main(tiny):
    import step1_extract_embeddings as s1, step2_baseline_logreg as s2, step3_add_augmentation as s3  # noqa: E401
    import step4_compare_backbones as s4, step5_lora_finetune as s5, step6_vae_reconstruction as s6  # noqa: E401
    import step7_fuse_lowlevel_branch as s7

    if tiny:
        mi._hf_model = tiny_hf_model
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "matched").mkdir()
        rows = make_data(tmp / "matched", np.random.default_rng(0))
        write_manifest(tmp / "manifest.csv", rows)
        held = [r for r in rows if r["split"] == "held_out_gen"]

        print("\n### step 1")
        p, cfg = config(tmp, 1)
        s1.main(p)
        z = np.load(tmp / "emb" / "dinov2_vitb14.npz")
        assert z["emb"].shape[:2] == (len(rows), 5) and str(z["crops"]) == mi.CROPS_NAME, z["emb"].shape

        print("\n### step 2")
        p, cfg2 = config(tmp, 2)
        r2 = s2.main(p)
        assert r2["n_train_images"] == 33, r2  # 16 real + 16 sd15 + tiny; the VAE rows are excluded
        _, iface2 = check_interface(cfg2, 2, held)
        cross_check_evaluation(cfg2, held, iface2)

        print("\n### step 3")
        p, cfg = config(tmp, 3, train={"augment_copies": 1})
        r3 = s3.main(p)
        assert r3["train_versions"] == 2 and "baseline" in r3
        check_interface(cfg, 3, held)

        print("\n### step 4")
        p, cfg = config(tmp, 4, compare={"backbones": ["dinov2_vitb14", "clip_vitb16", "siglip_vitb16"]})
        r4 = s4.main(p)
        assert len(r4["comparison"]) == 3
        check_interface(cfg, 4, held)

        print("\n### step 5")
        p, cfg = config(tmp, 5, lora={"max_steps": 4, "batch_size": 8, "last_n_blocks": 1})
        s5.main(p)
        ck5 = torch.load(cfg["output"]["checkpoint"], weights_only=False)
        assert ck5["backbone"] == "dinov2_vitb14+lora" and ck5["weight_deltas"], ck5.keys()
        assert all(".layer.1." in k for k in ck5["weight_deltas"]), list(ck5["weight_deltas"])
        check_interface(cfg, 5, held)
        bad = dict(ck5)
        bad.pop("weight_deltas")
        torch.save(bad, tmp / "nodeltas.pt")
        expect_exit(lambda: mi.load_checkpoint(tmp / "nodeltas.pt"), "needs weight_deltas")

        print("\n### step 6")
        p, cfg = config(tmp, 6, backbone={"deltas_from": str(tmp / "step5.pt")})
        r6 = s6.main(p)
        assert r6["n_vae_images"] == 8 and r6["n_train_images"] == 41
        check_interface(cfg, 6, held)

        print("\n### step 7")
        p, cfg = config(tmp, 7, train={"augment_copies": 1})
        s7.main(p)
        assert torch.load(cfg["output"]["checkpoint"], weights_only=False)["lowlevel"] == mi.LOWLEVEL_NAME
        check_interface(cfg, 7, held)
        assert json.loads((tmp / "step7.json").read_text())["lowlevel_only"]

        print("\n### guards")
        assert len(mi.native_crops(Image.open(tmp / "matched" / "tiny.jpg"))) == 5
        ck = torch.load(tmp / "step2.pt", weights_only=False)
        torch.save(dict(ck, crops="resize256"), tmp / "bad.pt")
        expect_exit(lambda: mi.load_checkpoint(tmp / "bad.pt"), "recipe")
        label, prob = mi.predict(mi.load_checkpoint(tmp / "step2.pt"), tmp / "matched" / held[0]["filename"])
        assert label in ("ai-generated", "real") and 0 <= prob <= 1
        leaky = [dict(r, split="train") if r["filename"] == held[-1]["filename"] else r for r in rows]
        write_manifest(tmp / "manifest.csv", leaky)
        expect_exit(lambda: s2.main(str(tmp / "step2.yaml")), "LEAK")
        print("  leakage, wrong recipe and missing-deltas checks all refuse correctly")

    print("\nSMOKE TEST PASSED (steps 1-7 + interface)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="tiny random backbones, no download")
    main(ap.parse_args().tiny)

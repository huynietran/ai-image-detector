"""
HumansRUS end-to-end smoke test: the website, the API, the detector, the
provenance check, the logo stamp's page code, and Evaluation's metric.

    python tests/smoke_test.py            # with the real model (threshold.json's checkpoint)
    python tests/smoke_test.py --dummy    # no model needed: fake, labelled scores

Prints PASS/FAIL per check and exits non-zero if anything failed. It writes
only to a temporary folder. The training pipeline has its own smoke test:
python model/steps/smoke_test_model.py --tiny
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = []


def check(name, fn):
    t = time.time()
    try:
        note = fn() or ""
        RESULTS.append(True)
        print(f"PASS  {name}  {note}  ({time.time() - t:.1f}s)")
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        RESULTS.append(False)
        print(f"FAIL  {name}  {type(e).__name__}: {e}")


def main(dummy):
    if dummy:
        os.environ["USE_DUMMY_MODEL"] = "1"
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    import numpy as np
    from PIL import Image, PngImagePlugin

    tmp = Path(tempfile.mkdtemp(prefix="humansrus_smoke_"))
    rng = np.random.default_rng(0)
    photo = tmp / "photo.jpg"
    Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)).save(photo, quality=90)
    small = tmp / "small.png"
    Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)).save(small)
    tagged = tmp / "tagged.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat, Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Model: sd15")
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tagged, pnginfo=info)
    not_image = tmp / "notes.jpg"
    not_image.write_text("this is not an image")

    # ---- files the site depends on -------------------------------------------------
    def files():
        t = json.loads((REPO / "evaluation/results/threshold.json").read_text())
        assert 0 < t["threshold"] < 1, t
        if not dummy:
            ck = REPO / t.get("checkpoint", "model/checkpoints/step2.pt")
            assert ck.is_file(), f"{ck} missing: download it from Drive (handoff_v0 > model)"
        for f in ["product/web/index.html", "product/web/config.js", "product/web/serve.py"]:
            assert (REPO / f).is_file(), f
        return f"threshold {t['threshold']:.4f}" + ("" if dummy else f", checkpoint {t.get('checkpoint')}")
    check("threshold, checkpoint and page files exist", files)

    # ---- the server (serve.py = existing API + page + /threshold) -------------------
    spec = importlib.util.spec_from_file_location("serve", REPO / "product/web/serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    from fastapi.testclient import TestClient
    client = TestClient(serve.app)

    def health():
        r = client.get("/health"); assert r.status_code == 200 and r.json() == {"ok": True}, r.text
        r = client.get("/threshold"); assert r.status_code == 200 and 0 < r.json()["threshold"] < 1, r.text
    check("GET /health and /threshold", health)

    def page():
        r = client.get("/"); assert r.status_code == 200, r.status_code
        html = r.text
        assert '<meta charset="utf-8">' in html, "no charset: symbols show garbled"
        for needed in ['id="file"', 'id="stamp"', "stampedCopy", "config.js", "HumansRUS"]:
            assert needed in html, f"page is missing {needed}"
        assert client.get("/config.js").status_code == 200
    check("website served at / (charset, upload, stamp button)", page)

    def score_photo():
        r = client.post("/score", files={"file": ("photo.jpg", photo.read_bytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        d = r.json()
        assert 0 <= d["score"] <= 1 and d["decided_by"] == "model", d
        assert len(d["crop_scores"]) == (1 if dummy else 5), d
        if not dummy:
            assert abs(np.mean(d["crop_scores"]) - d["score"]) < 1e-5, "score is not the mean of the crops"
        return f"score {d['score']:.3f} ({d['checkpoint_id']})"
    check("POST /score on a photo", score_photo)

    def score_small():
        r = client.post("/score", files={"file": ("small.png", small.read_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        return f"score {r.json()['score']:.3f}"
    check("POST /score on an image smaller than 224 px", score_small)

    def provenance():
        r = client.post("/score", files={"file": ("tagged.png", tagged.read_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        d = r.json(); assert d["decided_by"] == "provenance" and d["label"] == "ai", d
        return d["signals"][0]
    check("AI metadata is caught before the model runs", provenance)

    def bad_file():
        r = client.post("/score", files={"file": ("notes.jpg", not_image.read_bytes(), "image/jpeg")})
        assert r.status_code == 400, (r.status_code, r.text)
    check("a non-image gets a clear 400, not a crash", bad_file)

    def same_answer_twice():
        a = client.post("/score", files={"file": ("p.jpg", photo.read_bytes(), "image/jpeg")}).json()["score"]
        b = client.post("/score", files={"file": ("p.jpg", photo.read_bytes(), "image/jpeg")}).json()["score"]
        assert abs(a - b) < 1e-6, (a, b)
    check("same image, same score", same_answer_twice)

    def cors():
        r = client.get("/health", headers={"Origin": "https://humansrus.vercel.app"})
        assert r.headers.get("access-control-allow-origin") == "https://humansrus.vercel.app", r.headers
        r = client.get("/health", headers={"Origin": "https://example.com"})
        assert "access-control-allow-origin" not in r.headers
    check("only the HumansRUS site may call the checker from another address", cors)

    # ---- Evaluation's metric -----------------------------------------------------------
    def metric():
        sys.path.insert(0, str(REPO / "evaluation/scripts"))
        from score_checkpoint import compute_auc_and_tpr_at_fpr as compute_auc_and_tpr
        from shared.metrics import tpr_at_fpr
        y = np.r_[np.zeros(300), np.ones(300)]
        for seed in range(50):
            s = np.r_[np.random.default_rng(seed).normal(0, 1, 300), np.random.default_rng(seed + 99).normal(1, 1, 300)]
            assert abs(compute_auc_and_tpr(y, s)["tpr_at_fpr"] - tpr_at_fpr(y, s, 0.01)) < 1e-12
    check("Evaluation's TPR at 1% FPR matches shared/metrics", metric)

    # ---- Model and Evaluation agree on the real checkpoint ------------------------------
    if not dummy:
        def agree():
            from shared import model_interface as mi
            from score_checkpoint import load_checkpoint as ev_load, resolve_backbone_id, score_pil_image
            ck = json.loads((REPO / "evaluation/results/threshold.json").read_text())["checkpoint"]
            m = mi.load_checkpoint(ck)
            ev = ev_load(ck)
            backbone, clf = ev["backbone"], ev["classifier"]
            with Image.open(photo) as im:
                im = im.convert("RGB")
                a = mi.score_image(m, str(photo)).score
                b = score_pil_image(im, resolve_backbone_id(backbone), clf)
            assert abs(a - b) < 1e-3, (a, b)
            return f"{a:.4f} vs {b:.4f}"
        check("Product's and Evaluation's independent scorers agree", agree)

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed" + (" (dummy model)" if dummy else ""))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dummy", action="store_true", help="no checkpoint needed; fake scores")
    sys.exit(main(p.parse_args().dummy))

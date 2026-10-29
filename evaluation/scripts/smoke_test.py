"""
Smoke test: run freeze -> score -> results table on ~60 synthetic images and a
THROWAWAY classifier, in a temp folder. Proves the plumbing (manifest, crops,
degradations, CSV writing) before real data arrives. The numbers are
meaningless. Touches nothing in data/ or evaluation/results/.

Usage (from the repo root):
    python evaluation/scripts/smoke_test.py
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import build_results_table as brt  # noqa: E402
import score_checkpoint as sc  # noqa: E402


def make_image(path: Path, noise: float, rng) -> None:
    blocks = rng.normal(128, 40, (64, 64, 3)).repeat(8, 0).repeat(8, 1)      # 512x512 "content"
    arr = np.clip(blocks + rng.normal(0, noise, blocks.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=90)


def main(backbone: str = "dinov2_vitb14") -> None:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        make_image(tmp / f"real{i}.jpg", 4, rng)
        rows.append({"filename": f"real{i}.jpg", "source": "synthetic", "generator": "", "split": "train", "is_ai": 0})
    for gen in ("gen_a", "gen_b"):                                  # gen_b will be the held-out one
        for i in range(15):
            make_image(tmp / f"{gen}_{i}.jpg", 12, rng)
            rows.append({"filename": f"{gen}_{i}.jpg", "source": "synthetic", "generator": gen, "split": "train", "is_ai": 1})
    manifest = tmp / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run([sys.executable, str(SCRIPTS / "freeze_held_out.py"), "--manifest", str(manifest),
                    "--held-out-generators", "gen_b"], check=True)

    backbone_id = sc.resolve_backbone_id(backbone)
    dim = sc.embed_crops(Image.open(tmp / "real0.jpg"), backbone_id).shape[1]
    clf = LogisticRegression().fit(rng.normal(size=(40, dim)), [0, 1] * 20)      # throwaway
    ckpt = tmp / "throwaway.pt"
    torch.save({"backbone": backbone, "classifier": clf, "crops": sc.CROPS_NAME}, ckpt)

    sc.main(str(ckpt), str(manifest), "held_out_gen", str(tmp))
    out = {name: tmp / name for name in ("table.csv", "errors.csv", "threshold.json")}
    brt.main(str(ckpt), str(manifest), str(tmp), "held_out_gen", str(out["table.csv"]),
             str(out["errors.csv"]), str(out["threshold.json"]))

    table = pd.read_csv(out["table.csv"])
    assert len(table) == 16 and set(table.generator) == {"ALL", "gen_b"}, table
    assert all(p.exists() for p in out.values())
    print(f"\nSMOKE TEST PASSED. Outputs are in {tmp} (the numbers mean nothing).")


if __name__ == "__main__":
    main()

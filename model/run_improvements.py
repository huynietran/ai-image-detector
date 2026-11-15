"""
One command that tries the team's planned model improvements in order, keeps
what helps, and scores the result with Evaluation's independent scorer next to
the current model (step 2).

    python model/run_improvements.py            # everything (~1-2 h on a Mac)
    python model/run_improvements.py --adopt    # afterwards: switch the website to the new model

What it runs (docs/plan.md build order; step 5 needs a rented GPU and step 7 is
optional, so both are left out):

  1. data   build the VAE reconstruction set (data/scripts/build_vae_set.py)
  2. step 3 JPEG / resize / crop augmentation on the current DINOv2 model
  3. step 4 backbones: DINOv2 ViT-B vs CLIP ViT-B/16 vs CLIP ViT-L/14
            (CLIP ViT-L/14 is the backbone best known for spotting images
             from generators it never saw), with step 3's augmentation if it helped
  4. step 6 add the VAE reconstruction set to the best backbone
  5. Evaluation's build_results_table.py on step 2 and on the winner

Every number Model's steps print is on the held-out Flux images, and choosing the
winner on them makes the winner's number a little optimistic. The comparison
that counts is the last one (Evaluation's scorer, same rows, same code for both),
and a second held-out generator should confirm it before anyone relies on a
small gap.

Nothing official is overwritten: Evaluation's tables go to
evaluation/results/improve/, and threshold.json / the website only change with
--adopt.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "model" / "configs"
RES = REPO / "evaluation" / "results"
OUT = RES / "improve"
BACKBONES = ["dinov2_vitb14", "clip_vitb16", "clip_vitl14"]


def run(*args):
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)


def load(path):
    return json.loads(Path(path).read_text())


def write_config(name, step, backbone, copies, extra=""):
    path = CFG / f"improve_{name}.yaml"
    path.write_text(f"""# Written by model/run_improvements.py
seed: 42
data:
  manifest: data/manifests/manifest.csv
  matched_dir: data/matched/
  embeddings_dir: model/embeddings/
backbone:
  name: {backbone}
train:
  exclude_generators: [vae_reconstruction]
  vae_generators: [vae_reconstruction]
  augment_copies: {copies}
run:
  batch_size: 16
  num_workers: 0
{extra}output:
  checkpoint: model/checkpoints/step{step}.pt
  results_log: evaluation/results/step{step}.json
""")
    return path


def check_data():
    rows = list(csv.DictReader(open(REPO / "data/manifests/manifest.csv")))
    base = [r for r in rows if r["generator"] != "vae_reconstruction"]
    missing = [r["filename"] for r in base if not (REPO / "data/matched" / r["filename"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(base)} images are missing from data/matched/ (e.g. {missing[0]}).\n"
                 "Download the folder HumansRUS > HumansRUsDR > handoff_v0 > data > matched from Drive\n"
                 "and put its images in data/matched/ first.")
    if not (REPO / "model/checkpoints/step2.pt").exists():
        sys.exit("model/checkpoints/step2.pt is missing (Drive: handoff_v0 > model > step2.pt).")
    print(f"data ok: {len(base)} images in data/matched/")


def better(a, b):
    """True if result a beats b: higher clean AUC, ties broken by TPR at 1% FPR."""
    return (a.get("auc", 0), a.get("tpr_at_1pct_fpr", 0)) > (b.get("auc", 0), b.get("tpr_at_1pct_fpr", 0))


def evaluate(ckpt, tag):
    run("evaluation/scripts/build_results_table.py", "--checkpoint", ckpt,
        "--output", OUT / f"{tag}_table.csv", "--errors-output", OUT / f"{tag}_errors.csv",
        "--threshold-output", OUT / f"{tag}_threshold.json")
    rows = list(csv.DictReader(open(OUT / f"{tag}_table.csv")))
    return {r["degradation"]: r for r in rows if r["generator"] == "ALL"}


def improve():
    check_data()
    OUT.mkdir(parents=True, exist_ok=True)

    run("data/scripts/build_vae_set.py")

    run("model/steps/step3_add_augmentation.py", "--config", write_config("step3", 3, "dinov2_vitb14", 2))
    s3 = load(RES / "step3.json")
    copies = 2 if better(s3, s3["baseline"]) else 0
    print(f"\nstep 3: augmentation {'kept' if copies else 'dropped'} "
          f"(AUC {s3['baseline']['auc']:.4f} -> {s3['auc']:.4f})")

    extra = f"compare:\n  backbones: [{', '.join(BACKBONES)}]\n  select_on: auc\n"
    run("model/steps/step4_compare_backbones.py", "--config", write_config("step4", 4, "dinov2_vitb14", copies, extra))
    s4 = load(RES / "step4.json")
    print(f"\nstep 4: best backbone {s4['backbone']} (AUC {s4['auc']:.4f})")

    run("model/steps/step6_vae_reconstruction.py", "--config", write_config("step6", 6, s4["backbone"], copies))
    s6 = load(RES / "step6.json")
    final = "step6" if better(s6, s6["without_vae"]) else "step4"
    print(f"\nstep 6: VAE set {'kept' if final == 'step6' else 'dropped'} "
          f"(AUC {s6['without_vae']['auc']:.4f} -> {s6['auc']:.4f})")

    old = evaluate("model/checkpoints/step2.pt", "step2")
    new = evaluate(f"model/checkpoints/{final}.pt", final)

    names = {"clean": "Original", "jpeg_75": "JPEG quality 75", "jpeg_50": "JPEG quality 50",
             "jpeg_30": "JPEG quality 30", "resize_0.5x": "Shrunk to half size",
             "resize_1.5x": "Enlarged 1.5×", "crop": "Cropped", "screenshot": "Screenshot"}
    lines = ["# Model improvement run", "",
             f"Current model: step 2 (DINOv2 ViT-B, no augmentation). New model: {final} "
             f"({s4['backbone']}, augmentation copies {copies}, VAE set {'yes' if final == 'step6' else 'no'}).", "",
             "Evaluation's scorer, held-out Flux + real images, same rows for both:", "",
             "| Test | AUC now | AUC new | TPR@1%FPR now | TPR@1%FPR new |", "|---|---|---|---|---|"]
    for k, label in names.items():
        if k in old and k in new:
            lines.append(f"| {label} | {old[k]['auc']} | {new[k]['auc']} | {old[k]['tpr_at_1pct_fpr']} | "
                         f"{new[k]['tpr_at_1pct_fpr']} |")
    lines += ["", "Model's in-loop numbers (same held-out rows):", "",
              f"- step 3 augmentation: AUC {s3['baseline']['auc']:.4f} -> {s3['auc']:.4f}",
              "- step 4 backbones: " + ", ".join(f"{t['backbone']} {t['auc']:.4f}" for t in s4["comparison"]),
              f"- step 6 VAE set: AUC {s6['without_vae']['auc']:.4f} -> {s6['auc']:.4f}", "",
              "The winner was picked on the same held-out Flux images it is scored on, so its numbers are a",
              "little optimistic. Confirm on a second held-out generator before relying on a small gap.", "",
              "To switch the website to the new model: python model/run_improvements.py --adopt"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "final.json").write_text(json.dumps({"final": final, "backbone": s4["backbone"], "augment_copies": copies}, indent=2))
    print("\n" + "\n".join(lines))


def adopt():
    """Point the product at the new model: its threshold, and the numbers on the website."""
    info = load(OUT / "final.json")
    tag, rows = info["final"], list(csv.DictReader(open(OUT / f"{info['final']}_table.csv")))
    shutil.copy(OUT / f"{tag}_threshold.json", RES / "threshold.json")   # serve.py reads the checkpoint from here
    all_rows = {r["degradation"]: r for r in rows if r["generator"] == "ALL"}
    names = [("clean", "Original"), ("jpeg_75", "JPEG quality 75"), ("jpeg_50", "JPEG quality 50"),
             ("jpeg_30", "JPEG quality 30"), ("resize_0.5x", "Shrunk to half size"),
             ("resize_1.5x", "Enlarged 1.5×"), ("crop", "Cropped"), ("screenshot", "Screenshot")]
    results = {"backbone": info["backbone"],
               "rows": [[label, float(all_rows[k]["auc"])] for k, label in names if k in all_rows],
               "tpr_at_1pct_fpr": float(all_rows["clean"]["tpr_at_1pct_fpr"]),
               "vae_set": tag == "step6", "augmented": info.get("augment_copies", 0) > 0}
    (REPO / "product/web/results.json").write_text(json.dumps(results, indent=2))
    print(f"website now uses model/checkpoints/{tag}.pt (evaluation/results/threshold.json) "
          "and shows its numbers (product/web/results.json). Restart: python product/web/serve.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adopt", action="store_true", help="switch the website to the model from the last run")
    adopt() if p.parse_args().adopt else improve()

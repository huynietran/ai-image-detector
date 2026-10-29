"""
Share a checkpoint with the team (model/checkpoints/ is gitignored).

Uploads the checkpoint and its results JSON as a Weights & Biases artifact and
prints the SHA-256, which Evaluation compares first when cross-checking a number
("same checkpoint file?"). Run `wandb login` once beforehand.

Usage:
    python model/steps/share_checkpoint.py --step 2
    python model/steps/share_checkpoint.py --checkpoint model/checkpoints/step2.pt --results evaluation/results/step2.json

Without W&B, copy the .pt into the shared Drive folder and post the SHA-256 printed by --hash-only.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(checkpoint, results, project, hash_only):
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found; run the step first.")
    digest = sha256(ckpt)
    info = json.loads(Path(results).read_text()) if results and Path(results).exists() else {}
    print(f"{ckpt}  {ckpt.stat().st_size / 1e6:.1f} MB  sha256 {digest}")
    if info:
        print(f"held-out AUC {info.get('auc')}  TPR@1%FPR {info.get('tpr_at_1pct_fpr')}  "
              f"generators {info.get('held_out_generators')}")
        if info.get("checkpoint_sha256") not in (None, digest):
            print("WARNING: this file's hash differs from the one recorded when the step ran.")
    if hash_only:
        return digest

    import wandb
    run = wandb.init(project=project, job_type="checkpoint", config={k: v for k, v in info.items()
                                                                       if not isinstance(v, (dict, list))})
    art = wandb.Artifact(ckpt.stem, type="model", metadata={"sha256": digest, **{k: info[k] for k in
                         ("auc", "tpr_at_1pct_fpr", "backbone", "held_out_generators") if k in info}})
    art.add_file(str(ckpt))
    if info:
        art.add_file(str(results))
    run.log_artifact(art)
    run.finish()
    print(f"Uploaded artifact '{ckpt.stem}'. Post its link and the sha256 above in the team channel.")
    return digest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, help="shorthand for model/checkpoints/stepN.pt + evaluation/results/stepN.json")
    p.add_argument("--checkpoint")
    p.add_argument("--results")
    p.add_argument("--project", default="ai-image-detector")
    p.add_argument("--hash-only", action="store_true", help="just print the size and SHA-256")
    a = p.parse_args()
    if a.step:
        a.checkpoint = a.checkpoint or f"model/checkpoints/step{a.step}.pt"
        a.results = a.results or f"evaluation/results/step{a.step}.json"
    if not a.checkpoint:
        sys.exit("give --step N or --checkpoint PATH")
    main(a.checkpoint, a.results, a.project, a.hash_only)

"""
step6_vae_reconstruction
See model/steps/README.md and docs/plan.md for what this step does and why.

One change: add the VAE reconstruction set to training. These are REAL photos
passed through a diffusion model's VAE and labelled AI, which teaches the
detector "touched by a generative model" rather than one generator's style.
Data builds the set (Weeks 2-3 guide, Data Task 7) and lists it in the manifest
with generator = vae_reconstruction (train.vae_generators), split = train,
is_ai = 1. Steps 2-5 exclude those rows (train.exclude_generators) so they only
enter here.

Reports held-out AUC without and with the set, on the same rows. Uses the frozen
backbone, or a step-5 checkpoint's adapted backbone (backbone.deltas_from).

Data must reconstruct only photos that are in the TRAIN split: a reconstruction
of a held-out real photo leaks that photo into training.

Usage:
    python model/steps/step6_vae_reconstruction.py --config model/configs/step6.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import torch  # noqa: E402


def backbone_from_config(cfg):
    """(name, weight_deltas) from backbone.name, or from a step-5 checkpoint if deltas_from is set."""
    src = cfg["backbone"].get("deltas_from")
    if not src:
        return cfg["backbone"]["name"], None
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "weight_deltas" not in ckpt:
        raise SystemExit(f"{src} has no weight_deltas; it isn't a step-5 checkpoint")
    print(f"  using adapted backbone {ckpt['backbone']} from {src}")
    return ckpt["backbone"], ckpt["weight_deltas"]


def main(config_path: str):
    cfg = common.load_config(config_path)
    vae = list(cfg["train"].get("vae_generators") or ["vae_reconstruction"])
    with_vae, held, held_out = common.load_rows(cfg, include_generators=vae)
    vae_rows = [r for r in with_vae if r["generator"] in vae]
    without = [r for r in with_vae if r["generator"] not in vae]
    if not vae_rows:
        raise SystemExit(f"No train rows with generator in {vae}. Data builds the VAE reconstruction set "
                         "(Weeks 2-3 guide, Data Task 7); step 6 runs once it is in the manifest.")
    if any(int(r["is_ai"]) != 1 for r in vae_rows):
        raise SystemExit(f"Some {vae} rows have is_ai != 1; the reconstructions must be labelled AI.")
    print(f"  {len(vae_rows)} VAE reconstructions; reminder: they must come from TRAIN-split real photos only")

    name, deltas = backbone_from_config(cfg)
    feats = common.embedding_features(cfg, name, deltas)
    copies = int(cfg["train"].get("augment_copies", 0))
    _, base = common.fit_and_evaluate(cfg, without, held, feats, copies, "without VAE set")
    clf, res = common.fit_and_evaluate(cfg, with_vae, held, feats, copies, "with VAE set")
    if "auc" in res and "auc" in base:
        print(f"  change: clean AUC {res['auc'] - base['auc']:+.4f}, "
              f"JPEG 50 AUC {res['jpeg50'].get('auc', 0) - base['jpeg50'].get('auc', 0):+.4f}")
    common.warn_results(res)
    res.update(step=6, backbone=name, vae_generators=vae, n_vae_images=len(vae_rows), augment_copies=copies,
               without_vae=base, held_out_generators=sorted(held_out), n_held_out_images=len(held))
    ckpt = {"backbone": name, "classifier": clf}
    if deltas:
        ckpt["weight_deltas"] = deltas
    return common.save(cfg, ckpt, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

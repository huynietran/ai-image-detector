"""
step7_fuse_lowlevel_branch
See model/steps/README.md and docs/plan.md for what this step does and why.

One change: a low-level branch. For each of the recipe's five crops, 41
hand-made features of the high-pass residual (32-bin log radial power spectrum,
plus std / mean |x| / kurtosis per colour channel; shared/model_interface.py,
lowlevel_features) are appended to the backbone embedding before the logistic
regression. This is where pixel-level generator fingerprints (upsampling grids,
missing sensor noise) show up, which a semantic backbone can smooth over.

Reports held-out AUC for backbone only, low-level only, and fused, on the same
rows. The checkpoint carries "lowlevel": "hp_residual_fft32+stats9", and the
interface computes the same features at scoring time. The features need no GPU.

The VAE set is kept out unless listed in train.include_generators (keep step 6's
change only if it helped).

Usage:
    python model/steps/step7_fuse_lowlevel_branch.py --config model/configs/step7.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import numpy as np  # noqa: E402
from step6_vae_reconstruction import backbone_from_config  # noqa: E402

from shared import model_interface as mi  # noqa: E402


def main(config_path: str):
    cfg = common.load_config(config_path)
    train, held, held_out = common.load_rows(cfg, include_generators=cfg["train"].get("include_generators") or [])
    name, deltas = backbone_from_config(cfg)
    copies = int(cfg["train"].get("augment_copies", 0))

    emb = common.embedding_features(cfg, name, deltas)

    def lowlevel(rows, transform=None):
        return common.cached(cfg, rows, None, None, transform, mode="lowlevel")

    def fused(rows, transform=None):
        return np.concatenate([emb(rows, transform), lowlevel(rows, transform)], axis=2)

    _, base = common.fit_and_evaluate(cfg, train, held, emb, copies, "backbone only")
    _, ll_only = common.fit_and_evaluate(cfg, train, held, lowlevel, copies, "low-level only")
    clf, res = common.fit_and_evaluate(cfg, train, held, fused, copies, "fused")
    if "auc" in res and "auc" in base:
        print(f"  change vs backbone only: clean AUC {res['auc'] - base['auc']:+.4f}, "
              f"JPEG 50 AUC {res['jpeg50'].get('auc', 0) - base['jpeg50'].get('auc', 0):+.4f}")
    common.warn_results(res)
    res.update(step=7, backbone=name, lowlevel=mi.LOWLEVEL_NAME, augment_copies=copies, backbone_only=base,
               lowlevel_only=ll_only, held_out_generators=sorted(held_out), n_held_out_images=len(held))
    ckpt = {"backbone": name, "classifier": clf, "lowlevel": mi.LOWLEVEL_NAME}
    if deltas:
        ckpt["weight_deltas"] = deltas
    return common.save(cfg, ckpt, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

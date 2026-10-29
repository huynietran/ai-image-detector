"""
step3_add_augmentation
See model/steps/README.md and docs/plan.md for what this step does and why.

One change from step 2: each training image is also embedded `train.augment_copies`
times after a random crop (60-100%), resize (0.5-1.5x) and JPEG (quality 30-100),
applied to the image BEFORE the recipe's crops are taken. Held-out images stay
clean; the script reports held-out AUC clean and after JPEG 50, for both the step-2
baseline and the augmented model, on the same rows.

Augmented embeddings are cached (model/embeddings/<backbone>__aug<k>_seed<seed>.npz),
so re-fits are seconds. Expect each copy to cost about as long as step 1.

Usage:
    python model/steps/step3_add_augmentation.py --config model/configs/step3.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(config_path: str):
    cfg = common.load_config(config_path)
    train, held, held_out = common.load_rows(cfg)
    name = cfg["backbone"]["name"]
    copies = int(cfg["train"].get("augment_copies", 2))
    feats = common.embedding_features(cfg, name)

    _, base = common.fit_and_evaluate(cfg, train, held, feats, copies=0, label="step 2 baseline")
    clf, res = common.fit_and_evaluate(cfg, train, held, feats, copies=copies, label="step 3 augmented")
    if held and "auc" in res and "auc" in base:
        print(f"  change vs baseline: clean AUC {res['auc'] - base['auc']:+.4f}, "
              f"JPEG 50 AUC {res['jpeg50'].get('auc', 0) - base['jpeg50'].get('auc', 0):+.4f}")
    common.warn_results(res)
    res.update(step=3, backbone=name, augment_copies=copies, baseline=base,
               held_out_generators=sorted(held_out), n_held_out_images=len(held))
    return common.save(cfg, {"backbone": name, "classifier": clf}, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

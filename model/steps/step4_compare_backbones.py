"""
step4_compare_backbones
See model/steps/README.md and docs/plan.md for what this step does and why.

Same classifier and training setup as the step kept so far (train.augment_copies:
0 = step 2's setup, >0 = step 3's), repeated for every backbone in
compare.backbones: DINOv2 ViT-B vs ViT-L, CLIP, SigLIP. Each backbone uses its own
normalisation and embedding output (shared/model_interface.py, BACKBONES).

Prints a table and saves the backbone with the best held-out score
(compare.select_on: auc = clean, jpeg50_auc = after JPEG 50) as step4.pt, with
every backbone's numbers in step4.json. Picking on the held-out generator is what
docs/plan.md asks for; with a single held-out generator, confirm the winner on a
second one before trusting a small gap.

Each backbone costs one step-1-sized extraction per image version (clean + copies),
on a free Colab T4. ViT-L is ~3x slower than ViT-B.

Usage:
    python model/steps/step4_compare_backbones.py --config model/configs/step4.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(config_path: str):
    cfg = common.load_config(config_path)
    train, held, held_out = common.load_rows(cfg)
    if not held:
        raise SystemExit("Step 4 picks a backbone on held-out scores; wait for Evaluation's freeze.")
    copies = int(cfg["train"].get("augment_copies", 0))
    backbones = cfg["compare"]["backbones"]
    select_on = cfg["compare"].get("select_on", "auc")

    table, fitted = [], {}
    for name in backbones:
        print(f"\n== {name}")
        clf, res = common.fit_and_evaluate(cfg, train, held, common.embedding_features(cfg, name), copies, name)
        fitted[name] = clf
        table.append({"backbone": name, "auc": res.get("auc"), "tpr_at_1pct_fpr": res.get("tpr_at_1pct_fpr"),
                      "jpeg50_auc": res["jpeg50"].get("auc"), "jpeg50_tpr_at_1pct_fpr": res["jpeg50"].get("tpr_at_1pct_fpr"),
                      "n_real": res.get("n_real"), "n_ai": res.get("n_ai")})

    print(f"\n{'backbone':<16}{'AUC':>8}{'TPR@1%':>9}{'AUC jpeg50':>12}{'TPR jpeg50':>12}")
    for t in table:
        print(f"{t['backbone']:<16}{t['auc'] or 0:>8.4f}{t['tpr_at_1pct_fpr'] or 0:>9.4f}"
              f"{t['jpeg50_auc'] or 0:>12.4f}{t['jpeg50_tpr_at_1pct_fpr'] or 0:>12.4f}")
    best = max(table, key=lambda t: t[select_on] if t[select_on] is not None else -1)
    print(f"\nbest on held-out {select_on}: {best['backbone']}")
    common.warn_results(best)

    res = {"step": 4, "backbone": best["backbone"], "select_on": select_on, "augment_copies": copies,
           "auc": best["auc"], "tpr_at_1pct_fpr": best["tpr_at_1pct_fpr"], "comparison": table,
           "held_out_generators": sorted(held_out), "n_train_images": len(train), "n_held_out_images": len(held)}
    return common.save(cfg, {"backbone": best["backbone"], "classifier": fitted[best["backbone"]]}, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

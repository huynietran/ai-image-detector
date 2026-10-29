"""
step2_baseline_logreg
See model/steps/README.md and docs/plan.md for what this step does and why.

Logistic regression on the step-1 embeddings, trained ONLY on split == train,
one training example per crop (five per image). An image's score is the mean of
its five crop probabilities. Reports AUC and TPR at 1% FPR on held_out_gen.
This is the baseline every later step is compared against.

Wait for Evaluation's freeze commit before running: until then the held-out
generator's images are still marked train. The script refuses to run if a
held-out generator appears in train.

Usage:
    python model/steps/step2_baseline_logreg.py --config model/configs/step2.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(config_path: str):
    cfg = common.load_config(config_path)
    train, held, held_out = common.load_rows(cfg)
    name = cfg["backbone"]["name"]
    clf, res = common.fit_and_evaluate(cfg, train, held, common.embedding_features(cfg, name), copies=0)
    common.warn_results(res)
    res.update(step=2, backbone=name, held_out_generators=sorted(held_out), n_held_out_images=len(held))
    return common.save(cfg, {"backbone": name, "classifier": clf}, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

"""
step2_baseline_logreg
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step2_baseline_logreg.py --config model/configs/step2.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step2_baseline_logreg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

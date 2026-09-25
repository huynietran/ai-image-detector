"""
step3_add_augmentation
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step3_add_augmentation.py --config model/configs/step3.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step3_add_augmentation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

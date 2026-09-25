"""
step4_compare_backbones
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step4_compare_backbones.py --config model/configs/step4.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step4_compare_backbones")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

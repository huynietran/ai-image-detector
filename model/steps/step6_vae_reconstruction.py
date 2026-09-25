"""
step6_vae_reconstruction
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step6_vae_reconstruction.py --config model/configs/step6.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step6_vae_reconstruction")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

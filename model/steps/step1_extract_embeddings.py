"""
step1_extract_embeddings
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step1_extract_embeddings.py --config model/configs/step1.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step1_extract_embeddings")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

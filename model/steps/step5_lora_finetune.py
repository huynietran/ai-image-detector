"""
step5_lora_finetune
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step5_lora_finetune.py --config model/configs/step5.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step5_lora_finetune")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

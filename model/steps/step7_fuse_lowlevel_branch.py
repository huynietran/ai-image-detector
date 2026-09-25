"""
step7_fuse_lowlevel_branch
See model/steps/README.md and docs/plan.md for what this step does and why.

Usage:
    python model/steps/step7_fuse_lowlevel_branch.py --config model/configs/step7.yaml
"""

def main(config_path: str):
    raise NotImplementedError("TODO: implement step7_fuse_lowlevel_branch")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

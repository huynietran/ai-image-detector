"""
step1_extract_embeddings
See model/steps/README.md and docs/plan.md for what this step does and why.

Runs the frozen backbone once over every train and held_out_gen row (stress rows
live in data/stress_set/ and are Evaluation's) and saves one embedding per crop
of the "center+4corners@224" recipe to model/embeddings/<backbone>.npz:
    emb [rows, 5, dim] float16, filenames, backbone, crops, transform="clean"
Embeddings don't depend on the split, so this can run before the freeze. Re-running
only embeds rows that are new since last time.

Usage:
    python model/steps/step1_extract_embeddings.py --config model/configs/step1.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from shared.splits import load_manifest  # noqa: E402


def main(config_path: str):
    cfg = common.load_config(config_path)
    rows = [r for r in load_manifest(cfg["data"]["manifest"]) if r["split"] != "stress"]
    if not rows:
        raise SystemExit(f"No train/held_out_gen rows in {cfg['data']['manifest']}. Has Data handed over v0?")
    name = cfg["backbone"]["name"]
    print(f"{len(rows)} images, backbone {name}")
    emb = common.embedding_features(cfg, name)(rows)
    path = common.cache_path(cfg, name, None)
    print(f"embeddings for all {len(rows)} rows are in {path}: shape {emb.shape}")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

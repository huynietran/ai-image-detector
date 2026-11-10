#!/usr/bin/env bash
# Puts together the folder you upload to the Hugging Face Space.
# Run from the repo root:  bash deploy/huggingface/build_space.sh
# The folder is made OUTSIDE the repo (~/humansrus-space) so the model file never ends up in git.
set -euo pipefail

OUT="$HOME/humansrus-space"
CKPT="model/checkpoints/step2.pt"

if [ ! -f "$CKPT" ]; then
  echo "Missing $CKPT. Download step2.pt from the team Drive into model/checkpoints/ first."
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/model/checkpoints" "$OUT/evaluation/results"
cp -R product shared "$OUT/"
cp "$CKPT" "$OUT/model/checkpoints/"
cp evaluation/results/threshold.json "$OUT/evaluation/results/"
cp deploy/huggingface/Dockerfile deploy/huggingface/requirements.txt deploy/huggingface/README.md "$OUT/"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf "$OUT/product/demo"

echo "Ready: $OUT"

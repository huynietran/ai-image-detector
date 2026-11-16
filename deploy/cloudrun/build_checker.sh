#!/usr/bin/env bash
# Puts together the folder that gets deployed to Google Cloud Run.
# Run from the repo root:  bash deploy/cloudrun/build_checker.sh
# The folder is made OUTSIDE the repo (~/humansrus-checker) so the model file never ends up in git.
set -euo pipefail

OUT="$HOME/humansrus-checker"
# the checkpoint Evaluation's threshold was measured on (step2.pt unless a newer model was adopted)
CKPT=$(python3 -c "import json;print(json.load(open('evaluation/results/threshold.json')).get('checkpoint','model/checkpoints/step2.pt'))")

if [ ! -f "$CKPT" ]; then
  echo "Missing $CKPT. Download it from the team Drive (or run model/run_improvements.py) first."
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/model/checkpoints" "$OUT/evaluation/results"
cp -R product shared "$OUT/"
cp "$CKPT" "$OUT/model/checkpoints/"
cp evaluation/results/threshold.json "$OUT/evaluation/results/"
cp deploy/cloudrun/Dockerfile deploy/cloudrun/requirements.txt "$OUT/"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf "$OUT/product/demo"

echo "Ready: $OUT"

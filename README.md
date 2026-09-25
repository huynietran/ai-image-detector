# AI Image Detector

Detect AI-generated images by training a small classifier on top of a frozen
vision foundation model, scored only on generators it has never seen, with a
provenance check (C2PA, EXIF, PNG chunks) running first.

See `docs/plan.md` for the full team plan (method, build order, roles,
pitfalls, first-month schedule).

## Layout

```
data/           # Data lead's domain — raw, matched, manifests, test sets
model/          # Model lead's domain — steps 1-7, configs, checkpoints
evaluation/     # Evaluation lead's domain — scoreboard, degradation/stress tests
product/        # Product lead's domain — API, provenance layer, demo
shared/         # Code all four roles depend on (splits, metrics, io helpers)
notebooks/      # Scratch/exploration notebooks, not production code
docs/           # Plan, decisions, weekly results writeups
```

## The one number that matters

AUC on a generator the model has never seen, measured after JPEG compression.
Every checkpoint is selected on this number — never on in-distribution
validation. See `evaluation/results/`.

## Quickstart

```bash
pip install -r requirements.txt

# Step 1: extract embeddings
python model/steps/step1_extract_embeddings.py --config model/configs/step1.yaml

# Step 2: train baseline logistic regression, measure held-out AUC
python model/steps/step2_baseline_logreg.py --config model/configs/step2.yaml

# Step 3 onward follow the same pattern — see model/steps/README.md
```

## Handoffs (see docs/plan.md for full detail)

- **Data → Model:** a folder per source plus `data/manifests/manifest.csv`
  (filename → generator).
- **Model → Evaluation:** checkpoints in the agreed format in
  `model/checkpoints/`. Evaluation never touches Model's training/scoring
  code — the check must be independent.
- **Evaluation → Data & Model:** findings written to
  `evaluation/results/weekly_notes.md`.
- **Model → Product:** a stable loading interface in `shared/` so
  checkpoints swap without Model's involvement.

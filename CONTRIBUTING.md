# Contributing

## Week 1 — before writing project code

Everyone completes the shared basics first (see `docs/plan.md`):

- Python, PyTorch fundamentals
- git and pull requests
- Linux command line and remote GPUs
- train/validation/test splits with precision, recall, AUC
- Train one toy image classifier end to end

Good starting points: PyTorch's "Learn the Basics" tutorials, or fast.ai's
Practical Deep Learning.

## Branching / PRs

- One branch per step/task, e.g. `model/step3-augmentation`,
  `data/matched-set-v1`, `eval/degradation-suite`.
- Open a PR into `main`; at least one other role reviews before merge.
- Evaluation's scoring code in `evaluation/scripts/` should be reviewed
  with extra scrutiny for independence from `model/` — it must not import
  Model's training or scoring code.

## Where does my code go?

| I'm working on... | Goes in |
|---|---|
| Collecting/generating/matching images | `data/` |
| Anything from the 7-step build order | `model/steps/` + `model/configs/` |
| Scoring, degradation/stress tests, results | `evaluation/` |
| API, provenance checks, demo | `product/` |
| Something both Model and Evaluation (or Product) need | `shared/` — coordinate first, see `shared/README.md` |

## Weekly meeting

Evaluation presents the current results table (`evaluation/results/`)
before each weekly meeting. The team picks the single biggest problem to
fix next and records it at the bottom of that week's entry in
`evaluation/results/weekly_notes.md`.

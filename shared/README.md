# Shared

Code all four roles depend on. Changes here affect everyone — coordinate
before editing.

- `model_interface.py` — the stable checkpoint-loading interface. This is
  the Model → Product handoff: Product only ever calls through here, never
  imports `model/steps/` directly, so checkpoints can swap without
  touching Product code.
- `splits.py` — train / held_out_gen / stress split logic, reading
  `data/manifests/manifest.csv`. Single source of truth so Model and
  Evaluation never disagree about which images are held out.
- `metrics.py` — AUC and TPR@FPR implementations, used by both
  `model/steps/` (for quick in-loop checks) and independently
  re-implemented in `evaluation/scripts/` (Evaluation should NOT just
  import this file for its official numbers — see evaluation/README.md).

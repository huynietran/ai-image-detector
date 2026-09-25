# Evaluation (Evaluation lead)

The scoreboard, and being the skeptic. Freezes test sets, runs the
degradation and stress suites, and reports the numbers the team trusts.

**Never import Model's training or scoring code.** Re-implement scoring
independently — that's the whole point of this role.

```
results/           # Per-step results (AUC, TPR@1%FPR, per-generator, per-degradation)
scripts/           # Independent scoring scripts — no dependency on model/
```

## What to report, every time

- AUC per held-out generator (never in-distribution).
- TPR at 1% FPR per generator — this is what protects real photographers
  from false positives.
- Same two numbers after each degradation (JPEG 30/50/75, resize, crop,
  screenshot).
- A results table posted to `results/weekly_notes.md` before the weekly
  meeting.

## Suspicious signs to flag immediately

- AUC ≈ 0.99 on the very first run (see Pitfalls in `docs/plan.md`) —
  almost certainly a shortcut, not a working detector.
- A checkpoint chosen because it did well on in-distribution validation.

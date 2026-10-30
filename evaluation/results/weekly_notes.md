# Weekly results

One entry per week. Post before the weekly meeting so the team can pick
the single biggest problem to fix next.

## Week 1

- Status: Setup. Evaluation tooling installed (scikit-learn, matplotlib). Metrics chosen: AUC as the main number, accuracy for reference.
- AUC (held-out generator): — (no model or data yet)
- TPR @ 1% FPR: —
- Biggest problem to fix next: get a dataset and a first checkpoint so there is something to evaluate.

## Weeks 2–3

- Status: Scoring code and metrics written independently of Model's code. Added TPR at 1% FPR as the headline metric (accuracy hides false positives on real photos). Freeze, scoring and degradation scripts tested on synthetic data only.
- AUC (held-out generator): — (Data's matched set and Model's checkpoint not delivered yet)
- TPR @ 1% FPR: —
- Issues found: the TPR@1%FPR function in the starter code could report a point above 1% FPR, so it was fixed. The first scorer used Hugging Face's default resize, which blurs away artifacts; it was replaced in Week 4 with the five-crop recipe.
- Biggest problem to fix next: Data's matched set and a filled manifest.

## Week 4

- Status: Evaluation pipeline (freeze, independent scoring, degradation suite, shortcut audit, stress-set scoring, results table) pushed Oct 20, smoke-tested on real DINOv2 weights, and merged to main Oct 29 (PR #1). Data's v0 (1,200 images) is in; split frozen on main; shortcut audit done. Step-2 checkpoint pending.
- Split (frozen Oct 29): train = 300 Stable Diffusion + 300 real; held-out = 300 Flux + 300 real. Flux never in training; no held-out photo has its Stable Diffusion pair in training.
- Audit (metadata-only shortcut AUC): train split 0.488 (clean); held-out Flux split 0.674, from file size only (Flux 51 KB vs real 70 KB). Accepted for v0 as a known limitation; the Flux score may be slightly inflated.
- AUC (held-out generator, Flux): pending
- TPR @ 1% FPR: pending (300 held-out real photos means about 3 false positives allowed, so this number is rough)
- Number of images (train / held-out): 600 / 600
- Not done yet: stress set (real edited and screenshotted photos), independent cross-check of Model's AUC, threshold.json for the demo.
- Biggest problem to fix next: (decided at the meeting)

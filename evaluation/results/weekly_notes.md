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
- Issues found: the first scorer used Hugging Face's default resize, which blurs away artifacts; replaced in Week 4 with the five-crop recipe. The repo scorer's TPR@1%FPR formula (searchsorted) under-reports; found in the Week 4 cross-check, fix pending.
- Biggest problem to fix next: Data's matched set and a filled manifest.

## Week 4

- Status: First real result. Evaluation pipeline merged to main Oct 29 (PR #1). Data's v0 (1,200 images) frozen and audited. Step-2 checkpoint scored independently (SHA-256 9132a8901dc4…11b971e2).
- Split (frozen Oct 29, committed to main): train = 300 Stable Diffusion + 300 real; held-out = 300 Flux + 300 real. Flux never in training; no held-out photo has its Stable Diffusion pair in training.
- Audit (metadata-only shortcut AUC): train split 0.488 (clean); held-out Flux split 0.674, from file size only (Flux 51 KB vs real 70 KB). Accepted for v0 as a known limitation; the Flux score may be slightly inflated.
- AUC (held-out generator, Flux): 0.875 clean; 0.868 / 0.841 / 0.825 at JPEG 75 / 50 / 30; 0.808 at 0.5x resize (biggest drop); 0.876 at 1.5x resize; 0.846 crop; 0.875 screenshot.
- TPR @ 1% FPR: 0.123 clean (Model's run and the 0.9838 threshold agree). The results table shows 0.097 because of the scorer formula issue below. With 300 held-out real photos, 1% is about 3 images, so TPR is noisy.
- Threshold for 1% FPR: 0.9838 (threshold.json), used by Product for the demo.
- Cross-check vs Model: AUC agrees (≈0.87 vs 0.875; same SHA-256). TPR differs (Model 0.123 vs Evaluation table 0.097) because Evaluation's scorer uses searchsorted, which takes the low end of the 1% FPR step: same scores, different formula. Scorer fix to do before the next results table.
- Worst errors (week4_errors.csv): 5 real photos score at or above 0.97 (top one 1.00), and these set the very high threshold. Many Flux images score near 0 (worst 10 below 0.011), so the model is confidently wrong on them. Visual review: pending.
- Number of images (train / held-out): 600 / 600
- Results files: week4_table.csv, week4_errors.csv, threshold.json, audit_v0.csv
- Not done yet: stress set, scorer TPR fix, visual review of worst errors.
- Biggest problem to fix next: low detection at 1% false positives. TPR is about 0.12, so only about 1 in 8 Flux images is caught at a setting that wrongly flags 1 in 100 real photos. Two causes: a few real photos score like AI (0.97–1.00), which pushes the cut-off to 0.9838, and the 300-photo held-out set is too small to set that cut-off reliably. Next steps: more held-out real photos (Data), step 3 augmentation (Model), stress set and the scorer TPR fix (Evaluation).

# Model improvement run

Current model: step 2 (DINOv2 ViT-B, no augmentation). New model: step4 (clip_vitl14, augmentation copies 0, VAE set no).

Evaluation's scorer, held-out Flux + real images, same rows for both:

| Test | AUC now | AUC new | TPR@1%FPR now | TPR@1%FPR new |
|---|---|---|---|---|
| Original | 0.875 | 0.975 | 0.123 | 0.74 |
| JPEG quality 75 | 0.868 | 0.948 | 0.09 | 0.65 |
| JPEG quality 50 | 0.841 | 0.882 | 0.11 | 0.31 |
| JPEG quality 30 | 0.825 | 0.884 | 0.113 | 0.327 |
| Shrunk to half size | 0.808 | 0.921 | 0.153 | 0.213 |
| Enlarged 1.5× | 0.876 | 0.984 | 0.22 | 0.723 |
| Cropped | 0.846 | 0.969 | 0.117 | 0.693 |
| Screenshot | 0.875 | 0.978 | 0.067 | 0.583 |

Model's in-loop numbers (same held-out rows):

- step 3 augmentation: AUC 0.8754 -> 0.8616
- step 4 backbones: dinov2_vitb14 0.8754, clip_vitb16 0.9634, clip_vitl14 0.9750
- step 6 VAE set: AUC 0.9750 -> 0.8921

The winner was picked on the same held-out Flux images it is scored on, so its numbers are a
little optimistic. Confirm on a second held-out generator before relying on a small gap.

To switch the website to the new model: python model/run_improvements.py --adopt

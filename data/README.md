# Data (Data lead)

Every image that goes into training passes through here.

```
raw/real/            # Collected real photos in the target distribution
raw/ai/               # Generated AI images, across many models and settings
matched/              # Real + AI pairs matched in format, size, content
manifests/            # filename -> generator tracking (see manifest.csv)
degradation_suite/    # JPEG 30/50/75, resize, crop, screenshot variants (owned jointly with Evaluation)
stress_set/           # Real-photo stress set: edited, filtered, screenshotted real photos
```

## manifest.csv schema

```
filename,source,generator,split,is_ai
img_000123.png,genimage,stable-diffusion-xl,train,1
img_000124.jpg,personal_collection,,train,0
```

- `split`: `train` | `held_out_gen` (never used in training — Evaluation
  owns this set) | `stress`
- Keep AI and real images matched in format/size/content — mismatches
  teach the model shortcuts instead of the real signal (see Pitfalls in
  `docs/plan.md`).

## Handoff to Model

Model reads only from `matched/` + `manifests/manifest.csv`. Never point
Model at `raw/` directly — matching happens here first.

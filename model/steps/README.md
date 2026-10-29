# Steps

Each file is one step from the build order in `docs/plan.md`. Run in order;
each is compared against the step-2 baseline on the same held-out generator.
Never rename or delete a step file once results depend on it — add a new
step instead.

| File | Step | One change vs. before | Needs |
|---|---|---|---|
| `step1_extract_embeddings.py` | Embed every train/held-out image once (5 crops each), save to `model/embeddings/` | — | Free Colab T4 |
| `step2_baseline_logreg.py` | Logistic regression on the embeddings, held-out AUC — **the baseline** | — | CPU is enough |
| `step3_add_augmentation.py` | Add JPEG 30–100 / resize 0.5–1.5x / crop augmentation to training | augmentation | Free Colab T4 |
| `step4_compare_backbones.py` | DINOv2 ViT-B vs ViT-L vs CLIP vs SigLIP | backbone | Free Colab T4 |
| `step5_lora_finetune.py` | LoRA fine-tuning of the last few blocks | frozen → LoRA | Rented GPU (~$10–50/run) |
| `step6_vae_reconstruction.py` | Add real images passed through a diffusion VAE, labelled AI | training data | Data's VAE set |
| `step7_fuse_lowlevel_branch.py` | Append high-pass / frequency features to the embedding | features | CPU for the features |

Helpers: `common.py` (shared by the steps), `smoke_test_model.py` (runs all
seven on synthetic images), `share_checkpoint.py` (W&B upload + SHA-256).

Every step script:
1. Loads its config from `model/configs/stepN.yaml`.
2. Trains only on `split == train` and refuses to run if a held-out
   generator appears there.
3. Writes `model/checkpoints/stepN.pt` and `evaluation/results/stepN.json`
   (held-out AUC and TPR@1%FPR, clean and after JPEG 50, the image counts,
   and the checkpoint's SHA-256).
4. Prints the previous setup's numbers next to the new ones on the same rows
   (step 3 vs. no augmentation, step 6 vs. no VAE set, step 7 vs. backbone
   only).

Model's numbers are a quick in-loop check. The official numbers are
Evaluation's, from its own scorer.

## Checkpoint contract

Every checkpoint is a `torch.save` dict:

```python
{"backbone": "dinov2_vitb14",          # short name from BACKBONES in shared/model_interface.py
 "classifier": clf,                     # fitted sklearn estimator with .predict_proba
 "crops": "center+4corners@224"}        # names the preprocessing recipe below
```

Two optional keys, added by later steps:

| Key | Step | Meaning |
|---|---|---|
| `weight_deltas` | 5 (and 6–7 if built on step 5) | `{parameter name: tensor}` added to the pretrained backbone weights (merged LoRA). The backbone name gets a `+lora` suffix, e.g. `dinov2_vitb14+lora`, so a loader that ignores the deltas fails instead of scoring with the wrong weights. |
| `lowlevel` | 7 | `"hp_residual_fft32+stats9"`: 41 features per crop from `lowlevel_features()` are appended after the embedding, before the classifier. |

Load and score only through `shared/model_interface.py` (`load_checkpoint`,
`score_image`, `score_pil_image`, `predict`). The loader refuses any
checkpoint whose `crops` key differs.

## Preprocessing recipe `center+4corners@224`

Model, Evaluation and Product each implement this independently, so a
disagreement between our numbers points at a real bug.

1. Five native-resolution 224 px crops: centre plus four corners. **No
   whole-image resize** — upscale only when the short side is under 224 px.
2. Normalise with the backbone's mean and std (table below).
3. One embedding per crop (table below).
4. The image's score is the mean of the five crop probabilities.

| Backbone | Hugging Face id | Mean / std | Embedding per crop |
|---|---|---|---|
| `dinov2_vits14` / `dinov2_vitb14` / `dinov2_vitl14` | `facebook/dinov2-small` / `-base` (768-d) / `-large` | ImageNet `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | `last_hidden_state[:, 0]` (CLS) |
| `clip_vitb16` / `clip_vitl14` | `openai/clip-vit-base-patch16` / `openai/clip-vit-large-patch14` | CLIP `(0.4815, 0.4578, 0.4082)` / `(0.2686, 0.2613, 0.2758)` | `pooler_output` of `CLIPVisionModel` |
| `siglip_vitb16` | `google/siglip-base-patch16-224` | `(0.5, 0.5, 0.5)` / `(0.5, 0.5, 0.5)` | `pooler_output` of `SiglipVisionModel` |

Evaluation's scorer currently handles the DINOv2 rows only (ImageNet
normalisation, CLS token). A CLIP or SigLIP checkpoint from step 4 needs the
other two rows added there before it is scored.

If this recipe changes (more crops, another size), all three
implementations change together and the `crops` key changes with it.

## Running

```bash
python model/steps/smoke_test_model.py --tiny   # all 7 steps on synthetic data, ~1 min on CPU
python model/steps/smoke_test_model.py          # same with real weights (first Colab run)

python model/steps/step1_extract_embeddings.py --config model/configs/step1.yaml
# wait for Evaluation's freeze commit, then:
python model/steps/step2_baseline_logreg.py --config model/configs/step2.yaml
python model/steps/share_checkpoint.py --step 2   # W&B artifact + SHA-256 for Evaluation
```

`notebooks/model_colab.ipynb` does the same on a free Colab T4.

Embeddings are cached in `model/embeddings/` (gitignored): `<backbone>.npz`
for clean images (shape `[rows, 5, 768]`, ~77 MB per 10k images at
float16), `<backbone>__aug<k>_seed<s>.npz` and `<backbone>__jpeg50.npz` for
transformed ones. New manifest rows are added to a cache instead of
rebuilding it. Delete a cache file to force a rebuild.

## Rules the configs enforce

- `train.exclude_generators: [vae_reconstruction]` keeps Data's VAE set out
  of steps 2–5, so it enters only at step 6.
- Step 5's epoch count is fixed in the config. Never choose epochs, C, or a
  backbone by looking at held-out scores more than the plan allows: step 4
  picks on the held-out generator, as `docs/plan.md` says, so confirm a
  small win on a second held-out generator before trusting it.
- VAE reconstructions must come from train-split real photos only. A
  reconstruction of a held-out photo leaks that photo into training.

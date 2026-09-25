# Steps

Each file is one step from the build order in `docs/plan.md`. Run in order;
each is compared against the step-2 baseline on the same held-out generator.
Never rename or delete a step file once results depend on it — add a new
step instead.

| File | Step | Needs |
|---|---|---|
| `step1_extract_embeddings.py` | Extract DINOv2 embeddings for every image, save to disk | Free Colab T4 |
| `step2_baseline_logreg.py` | Logistic regression on embeddings, measure held-out AUC — **the baseline** | Free Colab T4 |
| `step3_add_augmentation.py` | Add JPEG 30–100 / resize 0.5–1.5x / crop augmentation | Free Colab T4 |
| `step4_compare_backbones.py` | DINOv2 vs CLIP vs SigLIP; ViT-L vs ViT-B | Free Colab T4 |
| `step5_lora_finetune.py` | Switch to LoRA fine-tuning of the last few blocks | Rented GPU (~$10–50/run) |
| `step6_vae_reconstruction.py` | Real images through a diffusion VAE, labelled AI | Rented GPU |
| `step7_fuse_lowlevel_branch.py` | Add high-pass/frequency branch, fuse with backbone | Rented GPU |

Every step script should:
1. Load its config from `model/configs/stepN.yaml`.
2. Write its checkpoint to `model/checkpoints/stepN_<name>.pt`.
3. Print (and log to `evaluation/results/`) AUC on the frozen held-out
   generator test set from `shared/`.

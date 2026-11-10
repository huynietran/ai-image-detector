# Putting HumansRUS online

Two parts, because the checker (torch + DINOv2, over 1 GB) is too big for Vercel's
Python functions (500 MB limit):

| Part | Where | What |
|---|---|---|
| The page | Vercel, `humansrus.vercel.app` | `product/web/` (static, no build) |
| The checker | Hugging Face Space (Docker, free CPU) | `product/web/serve.py` + the model |

The page sends images to the address in `product/web/config.js`. Empty means "the
computer serving this page" (local use). On Vercel it must be the Space's address.

## 1. Checker on Hugging Face

1. Make the folder to upload (needs `model/checkpoints/step2.pt` in place):
   `bash deploy/huggingface/build_space.sh` (makes `~/humansrus-space`, outside the repo).
2. huggingface.co: New Space, name `humansrus`, SDK **Docker**, hardware **CPU basic**, **Public**.
3. `pip install -U huggingface_hub`, `hf auth login` (token with write access),
   `hf upload <your-username>/humansrus ~/humansrus-space . --repo-type space`.
4. The Space builds (about 10 minutes). Its address is
   `https://<your-username>-humansrus.hf.space`; `/health` there should show `{"ok":true}`.

A public Space shows its files, so `step2.pt` can be downloaded from it. Agree this with
the Model Lead first. Free Spaces sleep when unused; the page shows "Waking up the
checker" and waits.

## 2. Page on Vercel

1. Put the Space address in `product/web/config.js` and push to `main`.
2. vercel.com: Add New, Project, import `ai-image-detector`.
   Project name `humansrus`, Framework `Other`, Root Directory `product/web`. Deploy.
3. Every push to `main` redeploys the page.

The checker accepts requests from `https://humansrus*.vercel.app`. If Vercel gives the
project another address, add it to the Space under Settings, Variables:
`ALLOWED_ORIGINS=https://that-address.vercel.app`.

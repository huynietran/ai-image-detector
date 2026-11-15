# Putting HumansRUS online

Two parts, because the checker (torch + DINOv2, over 1 GB) is too big for Vercel's
Python functions (500 MB limit):

| Part | Where | What |
|---|---|---|
| The page | Vercel, `humansrus.vercel.app` | `product/web/` (static, no build) |
| The checker | Google Cloud Run | `product/web/serve.py` + the model, in a container |

The page sends images to the address in `product/web/config.js`. Empty means "the
computer serving this page" (local use). On Vercel it must be the Cloud Run address.

## 1. Checker on Google Cloud Run

Needs a Google Cloud project with billing turned on. At student-project traffic it
should stay inside the free monthly allowance; set a small budget alert anyway.

1. Install the Google Cloud CLI and run `gcloud init`.
2. `gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com`
3. From the repo root, with the live checkpoint in place (the one named in `evaluation/results/threshold.json`, currently `model/checkpoints/step4.pt`):
   `bash deploy/cloudrun/build_checker.sh` (makes `~/humansrus-checker`, outside the repo).
4. ```
   cd ~/humansrus-checker
   gcloud run deploy humansrus --source . --region us-central1 --allow-unauthenticated \
     --memory 4Gi --cpu 2 --min-instances 0 --max-instances 2
   ```
5. The command prints the Service URL. `<that URL>/health` should show `{"ok":true}`.

The model file stays inside the container and is not downloadable. With no traffic the
checker scales to zero; the first check afterwards takes about 30 seconds while the page
shows "Waking up the checker".

To update the checker after code or model changes, repeat steps 3 and 4.

## 2. Page on Vercel

1. Put the Cloud Run address in `product/web/config.js` and push to `main`.
2. vercel.com: Add New, Project, import `ai-image-detector`.
   Project name `humansrus`, Framework `Other`, Root Directory `product/web`. Deploy.
3. Every push to `main` redeploys the page.

The checker accepts requests from `https://humansrus*.vercel.app`. If Vercel gives the
project another address, redeploy the checker with
`--set-env-vars ALLOWED_ORIGINS=https://that-address.vercel.app` added to step 4.

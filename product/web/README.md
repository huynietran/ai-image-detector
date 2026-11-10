# HumansRUS website

A one-page site for checking an image: drop, choose or paste it and get the score,
a plain-language result and the score for each of the five crops. It calls the same
`detect()` path as the API and the Gradio demo, so all three always agree.

```bash
# from the repo root, with model/checkpoints/step2.pt in place
python product/web/serve.py
# open http://127.0.0.1:8000

# no checkpoint yet: fake, clearly labelled scores
USE_DUMMY_MODEL=1 python product/web/serve.py
```

PowerShell: `$env:USE_DUMMY_MODEL = "1"` before the command.

- `index.html`: the page (no build step; fonts from Google Fonts).
- `config.js`: where the page sends images. Empty for local use; the hosted checker's
  address for the Vercel site. See `deploy/README.md` for putting it online.
- `serve.py`: serves the page next to the existing API (`/score`, `/health`) and adds
  `/threshold`, read from `evaluation/results/threshold.json`.
- Locally it runs on `127.0.0.1` only, so images go to the API on this computer and nowhere
  else. On the hosted site, images go to the team's Hugging Face Space.
- The results on the page (AUC 0.875, 0.841 at JPEG 50, about 1 in 8 Flux images caught at
  the 0.9838 threshold) are the Week 4 numbers in `evaluation/results/`. Update the page
  when a new results table lands.

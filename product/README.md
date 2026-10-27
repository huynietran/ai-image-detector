# Product (Product lead)

Turns a checkpoint into something usable.

```
detect.py      # the ONE function (provenance first, model second) the API and demo both call
provenance/    # C2PA / EXIF / PNG-chunk checks — runs BEFORE the model
api/           # Image in, score + which signals fired, out
demo/          # Gradio demo
```

## Request flow

1. `provenance/` checks for C2PA credentials, EXIF/XMP fields, PNG text
   chunks. Only a *claim of AI generation* is conclusive (a camera tag proves
   nothing, since it can be forged). If conclusive, return immediately and
   skip the model.
2. Otherwise, fall through to the model (`detect.py` loads a checkpoint via
   the stable interface in `shared/model_interface.py` — never imports
   Model's training code directly).
3. Return a score plus which signals fired (provenance vs. model, and if
   model, per-crop scores).

## Run it (from the repo root)

```bash
pip install -r requirements.txt

# Before Model has a checkpoint: fake, clearly-labelled scores
USE_DUMMY_MODEL=1 uvicorn product.api.main:app --reload   # http://127.0.0.1:8000/docs
USE_DUMMY_MODEL=1 python product/demo/app.py              # http://127.0.0.1:7860

# With a real checkpoint and Evaluation's threshold
CHECKPOINT_PATH=model/checkpoints/step2.pt AI_THRESHOLD=0.62 python product/demo/app.py
```

PowerShell: `$env:USE_DUMMY_MODEL = "1"` before the command.

## Tests

```bash
curl -L -o product/provenance/samples/C.jpg https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/C.jpg   # one-time (mkdir the folder first)
python product/provenance/test_check.py
USE_DUMMY_MODEL=1 python product/api/test_api.py
```

## Known limits (be honest about these in demos)

- Metadata is a claim, not a measurement: tags can be planted on real photos
  and stripped from AI ones. The demo words it as "this file's metadata says…".
- C2PA manifests are accepted if not marked invalid; signers are not checked
  against the C2PA trust list.
- Dummy-model scores are meaningless (`checkpoint_id: "dummy"`).

## Handoff from Model

Model guarantees a stable loading interface in `shared/model_interface.py`
so checkpoints can swap without touching `product/` code.

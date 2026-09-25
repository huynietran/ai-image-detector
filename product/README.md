# Product (Product lead)

Turns a checkpoint into something usable.

```
provenance/    # C2PA / EXIF / PNG-chunk checks — runs BEFORE the model
api/           # Image in, score + which signals fired, out
demo/          # Gradio demo
```

## Request flow

1. `provenance/` checks for C2PA credentials, EXIF/XMP fields, PNG text
   chunks. If present and valid, that's near-proof — return immediately,
   skip the model.
2. Otherwise, fall through to the model (`api/` loads a checkpoint via the
   stable interface in `shared/model_interface.py` — never imports
   Model's training code directly).
3. Return a score plus which signals fired (provenance vs. model, and if
   model, roughly why — e.g. which crops scored highest).

## Handoff from Model

Model guarantees a stable loading interface in `shared/model_interface.py`
so checkpoints can swap without touching `product/` code.

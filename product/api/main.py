"""
FastAPI app: image in, score + which signals fired, out.
Checks provenance first; falls through to the model via
shared/model_interface.py only when provenance is inconclusive.
(The logic lives in product/detect.py so the demo uses the same path.)

Run (from the repo root):
    uvicorn product.api.main:app --reload

Try it:
    curl -F "file=@some_image.jpg" http://127.0.0.1:8000/score
"""

import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from product.detect import detect

app = FastAPI(title="AI Image Detector")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/score")
async def score_image(file: UploadFile):
    """
    1. Run product.provenance.check.check_provenance() first.
    2. If conclusive, return it.
    3. Otherwise, load the current checkpoint via shared/model_interface.py
       and return its score plus which crops/signals contributed.
    """
    suffix = Path(file.filename or "").suffix or ".img"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        try:
            with Image.open(tmp.name) as img:
                img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="That file is not a readable image.")
        try:
            return asdict(detect(tmp.name))
        except (NotImplementedError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=503,
                detail="No model checkpoint is available yet "
                "(set CHECKPOINT_PATH, or USE_DUMMY_MODEL=1 for development). "
                f"({type(exc).__name__})",
            )
    finally:
        os.unlink(tmp.name)

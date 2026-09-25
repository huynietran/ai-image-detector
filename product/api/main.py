"""
FastAPI app: image in, score + which signals fired, out.
Checks provenance first; falls through to the model via
shared/model_interface.py only when provenance is inconclusive.

Run:
    uvicorn product.api.main:app --reload
"""

from fastapi import FastAPI, UploadFile

app = FastAPI(title="AI Image Detector")


@app.post("/score")
async def score_image(file: UploadFile):
    """
    1. Run product.provenance.check.check_provenance() first.
    2. If conclusive, return it.
    3. Otherwise, load the current checkpoint via shared/model_interface.py
       and return its score plus which crops/signals contributed.
    """
    raise NotImplementedError

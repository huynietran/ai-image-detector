"""
API checks that need no checkpoint (they use USE_DUMMY_MODEL). Run from the
repo root:

    pip install httpx
    python product/api/test_api.py
"""

import io
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["USE_DUMMY_MODEL"] = "1"
from product.api.main import app  # noqa: E402

client = TestClient(app)


def png(**save) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(buf, "PNG", **save)
    return buf.getvalue()


def post(data: bytes, name="x.png"):
    return client.post("/score", files={"file": (name, data, "image/png")})


def main():
    assert client.get("/health").json() == {"ok": True}

    r = post(png())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decided_by"] == "model" and body["checkpoint_id"] == "dummy", body
    assert post(png()).json()["score"] == body["score"], "same image must give the same score"

    assert post(b"this is not an image").status_code == 400

    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1")
    r = post(png(pnginfo=meta))
    assert r.json()["decided_by"] == "provenance" and r.json()["signals"] == ["png_chunk"], r.json()

    sample = Path(__file__).resolve().parents[1] / "provenance" / "samples" / "C.jpg"
    if sample.exists():
        r = client.post("/score", files={"file": ("C.jpg", sample.read_bytes(), "image/jpeg")})
        assert r.json()["signals"] == ["c2pa"], r.json()
    print("API checks passed")


if __name__ == "__main__":
    main()

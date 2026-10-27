"""
Provenance checks on small files created here, plus a real signed C2PA sample
if you download one (see below). Run from the repo root:

    python product/provenance/test_check.py

Optional signed sample (declares the image was made by an algorithm):
    curl.exe -L -o product/provenance/samples/C.jpg https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/C.jpg
"""

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from product.provenance.check import check_provenance  # noqa: E402

SAMPLE = Path(__file__).parent / "samples" / "C.jpg"
A1111 = "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1"


def check(name, path, signal):
    r = check_provenance(str(path))
    ok = (r.signal == signal) and (r.is_conclusive == (signal is not None))
    print(("PASS" if ok else "FAIL"), name, "->", r)
    return ok


def main():
    img = Image.new("RGB", (64, 64), "blue")
    d = Path(tempfile.mkdtemp())
    results = []

    img.save(d / "plain.jpg"); img.save(d / "plain.png")
    results += [check("plain jpg (no metadata)", d / "plain.jpg", None),
                check("plain png (no metadata)", d / "plain.png", None)]

    ex = Image.Exif(); ex[0x0131] = "Midjourney V6"
    img.save(d / "software.jpg", exif=ex)
    results.append(check("EXIF Software = Midjourney", d / "software.jpg", "exif"))

    ex = Image.Exif(); ex[0x010F] = "Canon"; ex[0x0110] = "EOS R5"
    img.save(d / "camera.jpg", exif=ex)
    results.append(check("camera EXIF is NOT proof of anything", d / "camera.jpg", None))

    meta = PngImagePlugin.PngInfo(); meta.add_text("parameters", A1111)
    img.save(d / "a1111.png", pnginfo=meta)
    results.append(check("PNG 'parameters' chunk (Automatic1111)", d / "a1111.png", "png_chunk"))

    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", json.dumps({"3": {"class_type": "KSampler", "inputs": {}}}))
    img.save(d / "comfy.png", pnginfo=meta)
    results.append(check("PNG ComfyUI graph", d / "comfy.png", "png_chunk"))

    meta = PngImagePlugin.PngInfo(); meta.add_text("prompt", "Please enter your name")
    img.save(d / "innocent.png", pnginfo=meta)
    results.append(check("PNG chunk named 'prompt' but not a generator", d / "innocent.png", None))

    (d / "junk.jpg").write_bytes(b"not an image")
    results.append(check("corrupt file does not crash", d / "junk.jpg", None))

    if SAMPLE.exists():
        results.append(check("signed C2PA sample", SAMPLE, "c2pa"))
    else:
        print("SKIP signed C2PA sample (download it with the curl.exe line at the top of this file)")

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

from pathlib import Path
from io import BytesIO
import hashlib
import time

from PIL import Image, ImageOps
from datasets import load_dataset

PROJECT = Path(__file__).resolve().parents[2]s[2]
RAW_REAL = PROJECT / "data" / "raw" / "real"
OUTPUT = RAW_REAL / "coco_larger"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = 600
MIN_SIZE = 512


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    # Count qualifying photos already saved.
    seen = set()

    for path in RAW_REAL.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        try:
            with Image.open(path) as image:
                if min(image.size) < MIN_SIZE:
                    continue
            seen.add(digest(path.read_bytes()))
        except Exception:
            print("Could not read:", path.name)

    print("Existing qualifying photos:", len(seen), flush=True)
    print("Target:", TARGET, flush=True)

    if len(seen) >= TARGET:
        print("Target already reached.")
        return

    dataset = load_dataset(
        "jxie/coco_captions",
        split="train",
        streaming=True,
    )

    started = time.monotonic()
    added = 0

    for index, example in enumerate(dataset):
        image = ImageOps.exif_transpose(
            example["image"]
        ).convert("RGB")

        if min(image.size) >= MIN_SIZE:
            with BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=95)
                data = buffer.getvalue()

            fingerprint = digest(data)

            if fingerprint not in seen:
                destination = OUTPUT / f"coco_{fingerprint}.jpg"
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)

                seen.add(fingerprint)
                added += 1

                print(
                    f"Qualifying photos: {len(seen)}/{TARGET} "
                    f"| New: {added} | Size: {image.size}",
                    flush=True,
                )

        if len(seen) >= TARGET:
            break

        if (index + 1) % 500 == 0:
            minutes = (time.monotonic() - started) / 60
            print(
                f"Checked {index + 1} dataset rows "
                f"| Added {added} | {minutes:.1f} minutes",
                flush=True,
            )

    print("Finished. Qualifying photos:", len(seen), flush=True)


if __name__ == "__main__":
    main()
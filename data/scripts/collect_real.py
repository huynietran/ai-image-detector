from pathlib import Path
from io import BytesIO
import hashlib

from datasets import load_dataset

PROJECT = Path(__file__).resolve().parents[2]s[2]
OUTPUT = PROJECT / "data" / "real"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = 5000


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def main():
    # Recognize photos already saved by our earlier script.
    seen = {
        fingerprint(path.read_bytes())
        for path in OUTPUT.glob("*.jpg")
    }

    print(f"Existing unique photos: {len(seen)}", flush=True)

    if len(seen) >= TARGET:
        print("Target already reached.")
        return

    print("Connecting to COCO...", flush=True)

    dataset = load_dataset(
        "jxie/coco_captions",
        split="train",
        streaming=True,
    )

    for index, example in enumerate(dataset):
        with BytesIO() as buffer:
            example["image"].convert("RGB").save(
                buffer, format="JPEG", quality=95
            )
            image_bytes = buffer.getvalue()

        image_hash = fingerprint(image_bytes)

        if image_hash in seen:
            continue

        destination = OUTPUT / f"coco_{index:07d}.jpg"

        # Never overwrite an existing photo.
        if destination.exists():
            destination = OUTPUT / f"coco_{index:07d}_{image_hash[:12]}.jpg"

        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(image_bytes)
        temporary.replace(destination)

        seen.add(image_hash)
        print(
            f"Saved {len(seen)}/{TARGET}: {destination.name}",
            flush=True,
        )

        if len(seen) >= TARGET:
            break

    print(f"Finished: {len(seen)} unique photos")
    print(f"Folder: {OUTPUT}")


if __name__ == "__main__":
    main()
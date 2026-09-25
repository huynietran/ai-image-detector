"""
Provenance layer: C2PA credentials, EXIF/XMP fields, PNG text chunks.
Runs before the model. When present and valid, this is near-proof and
overrides the classifier entirely.

Usage:
    from product.provenance.check import check_provenance
    result = check_provenance(image_path)
    if result.is_conclusive:
        return result  # skip the model
"""

from dataclasses import dataclass


@dataclass
class ProvenanceResult:
    is_conclusive: bool
    is_ai: bool | None
    signal: str | None  # "c2pa" | "exif" | "png_chunk" | None


def check_c2pa(image_path: str) -> ProvenanceResult:
    raise NotImplementedError


def check_exif(image_path: str) -> ProvenanceResult:
    raise NotImplementedError


def check_png_chunks(image_path: str) -> ProvenanceResult:
    raise NotImplementedError


def check_provenance(image_path: str) -> ProvenanceResult:
    """Try C2PA, then EXIF, then PNG chunks; return the first conclusive
    result, else a non-conclusive result so the caller falls through to
    the model."""
    raise NotImplementedError

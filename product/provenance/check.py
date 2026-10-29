"""
Provenance layer: C2PA credentials, EXIF/XMP fields, PNG text chunks.
Runs before the model. When present and valid, this is near-proof and
overrides the classifier entirely.

Usage:
    from product.provenance.check import check_provenance
    result = check_provenance(image_path)
    if result.is_conclusive:
        return result  # skip the model

Only AI *claims* are treated as conclusive. A missing marker proves nothing
(metadata is easy to strip), and a camera-style EXIF block proves nothing
either (it is easy to forge), so those cases return is_conclusive=False and
the caller falls through to the model.
"""

import json
import re
from dataclasses import dataclass

from PIL import Image


@dataclass
class ProvenanceResult:
    is_conclusive: bool
    is_ai: bool | None
    signal: str | None  # "c2pa" | "exif" | "png_chunk" | None


NOT_CONCLUSIVE = ProvenanceResult(is_conclusive=False, is_ai=None, signal=None)

# IPTC "digital source type" values that mean "made by a generative model".
# https://cv.iptc.org/newscodes/digitalsourcetype/
AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)

# Substrings that show up in Software / claim-generator / description fields
# when a generator wrote the file. Lower-case; matched against lower-cased text.
AI_SOFTWARE_HINTS = (
    "midjourney", "dall-e", "dall·e", "openai", "stable diffusion",
    "stablediffusion", "firefly", "flux", "comfyui", "automatic1111",
    "invokeai", "novelai", "leonardo.ai", "ideogram", "imagen", "sdxl",
)

# PNG text-chunk keys written by popular local UIs.
AI_PNG_KEYS = ("parameters", "prompt", "workflow", "invokeai_metadata",
               "sd-metadata", "dream")

# Automatic1111 writes "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: ..."
A1111_PATTERN = re.compile(r"steps:\s*\d+.*(sampler|cfg scale|seed):", re.I | re.S)


def _has_ai_source_type(text: str) -> bool:
    text = text.lower()
    return any(t in text for t in AI_SOURCE_TYPES)


def _has_ai_software(text: str) -> bool:
    text = text.lower()
    return any(h in text for h in AI_SOFTWARE_HINTS)


def check_c2pa(image_path: str) -> ProvenanceResult:
    """Conclusive only if the C2PA manifest validates AND says the image was
    AI-generated (digitalSourceType in an action, or a generator name in the
    claim). A tampered manifest (validation_state 'Invalid') is ignored."""
    try:
        import c2pa  # pip install c2pa-python
    except ImportError:
        return NOT_CONCLUSIVE

    try:
        with c2pa.Reader(image_path) as reader:
            manifest_store = json.loads(reader.json())
    except Exception:  # no manifest ("ManifestNotFound"), unsupported type, corrupt file
        return NOT_CONCLUSIVE

    if manifest_store.get("validation_state") == "Invalid":
        return NOT_CONCLUSIVE

    active = manifest_store.get("manifests", {}).get(
        manifest_store.get("active_manifest"), {}
    )
    # Searching the serialized manifest catches both c2pa.actions and
    # c2pa.actions.v2, and digitalSourceType inside either.
    blob = json.dumps(active)
    claim_generator = json.dumps(
        [active.get("claim_generator"), active.get("claim_generator_info")]
    )
    if _has_ai_source_type(blob) or _has_ai_software(claim_generator):
        return ProvenanceResult(is_conclusive=True, is_ai=True, signal="c2pa")
    return NOT_CONCLUSIVE


def check_exif(image_path: str) -> ProvenanceResult:
    """Looks at EXIF Software / ImageDescription / UserComment and at the XMP
    packet (IPTC DigitalSourceType lives there)."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            fields = [
                str(exif.get(0x0131, "")),                       # Software
                str(exif.get(0x010E, "")),                       # ImageDescription
                str(exif.get(0x013B, "")),                       # Artist
            ]
            exif_ifd = exif.get_ifd(0x8769)                       # Exif sub-IFD
            user_comment = exif_ifd.get(0x9286, b"")              # UserComment
            if isinstance(user_comment, bytes):
                user_comment = user_comment.decode("utf-8", "ignore")
            user_comment = str(user_comment).replace("\x00", "")
            xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp") or b""
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", "ignore")
    except Exception:
        return NOT_CONCLUSIVE

    if _has_ai_source_type(xmp) or _has_ai_software(" ".join(fields)):
        return ProvenanceResult(is_conclusive=True, is_ai=True, signal="exif")
    if A1111_PATTERN.search(user_comment):
        return ProvenanceResult(is_conclusive=True, is_ai=True, signal="exif")
    return NOT_CONCLUSIVE


def check_png_chunks(image_path: str) -> ProvenanceResult:
    """PNG tEXt/iTXt chunks: Automatic1111 'parameters', ComfyUI 'prompt' +
    'workflow', InvokeAI metadata, and a 'Software' chunk naming a generator."""
    try:
        with Image.open(image_path) as img:
            if img.format != "PNG":
                return NOT_CONCLUSIVE
            chunks = {str(k).lower(): v for k, v in img.info.items()
                      if isinstance(v, (str, bytes))}
    except Exception:
        return NOT_CONCLUSIVE

    def as_text(v):
        return v.decode("utf-8", "ignore") if isinstance(v, bytes) else v

    # Key names can collide with innocent files ("prompt" is a common word),
    # so require the value to look like what the tools actually write.
    params = as_text(chunks.get("parameters", ""))
    if params and A1111_PATTERN.search(params):
        return ProvenanceResult(True, True, "png_chunk")
    for key in ("prompt", "workflow"):
        value = as_text(chunks.get(key, ""))
        if value.lstrip().startswith("{") and ("class_type" in value or "nodes" in value):
            return ProvenanceResult(True, True, "png_chunk")  # ComfyUI graph
    if "invokeai_metadata" in chunks or "sd-metadata" in chunks or "dream" in chunks:
        return ProvenanceResult(True, True, "png_chunk")
    software = " ".join(as_text(chunks.get(k, "")) for k in ("software", "comment", "description"))
    if _has_ai_software(software) or _has_ai_source_type(software):
        return ProvenanceResult(True, True, "png_chunk")
    return NOT_CONCLUSIVE


def check_provenance(image_path: str) -> ProvenanceResult:
    """Try C2PA, then EXIF, then PNG chunks; return the first conclusive
    result, else a non-conclusive result so the caller falls through to
    the model."""
    for check in (check_c2pa, check_exif, check_png_chunks):
        result = check(image_path)
        if result.is_conclusive:
            return result
    return NOT_CONCLUSIVE

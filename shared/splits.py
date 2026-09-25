"""
Single source of truth for train / held_out_gen / stress splits, read
from data/manifests/manifest.csv. Model and Evaluation both import this
so they never disagree about which generator is held out.
"""

import csv


def load_manifest(path: str = "data/manifests/manifest.csv") -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_split(manifest: list[dict], split: str) -> list[dict]:
    """split: 'train' | 'held_out_gen' | 'stress'"""
    return [row for row in manifest if row["split"] == split]


def held_out_generators(manifest: list[dict]) -> set[str]:
    """Generators that must NEVER appear in the train split — this is
    what makes held-out-generator AUC meaningful."""
    return {row["generator"] for row in get_split(manifest, "held_out_gen")}

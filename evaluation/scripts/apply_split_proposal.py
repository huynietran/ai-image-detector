"""
Apply Data's proposed split to the manifest, and check it for leaks first.

Data hands over a candidate manifest (every row labelled train) and a
split_proposal.csv that says which rows should be train or held_out_gen. This
script copies the proposed split into the manifest that Model and Evaluation
actually read (data/manifests/manifest.csv) -- after refusing to do so if:

  * the two files do not describe the same images,
  * a held-out generator also appears in train,
  * a train image shares a pair_id with a held-out image (same caption/scene);
    --drop-train-pair-overlap removes those train images instead of failing,
  * a split other than train / held_out_gen is proposed.

It does not choose the held-out real images itself: Data matched them to the
held-out generator's images, and re-picking them at random would break that.
Evaluation reads only csv files here (no model/ imports, no shared/metrics.py).

Usage (from the repo root; --dry-run only prints the report):
    python evaluation/scripts/apply_split_proposal.py --dry-run
    python evaluation/scripts/apply_split_proposal.py --drop-train-pair-overlap --dry-run
    python evaluation/scripts/apply_split_proposal.py
    python evaluation/scripts/apply_split_proposal.py --matched-dir data/matched
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

COLUMNS = ["filename", "source", "generator", "split", "is_ai"]
ALLOWED = {"train", "held_out_gen"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="data/manifests/v0_candidate/manifest.csv")
    p.add_argument("--proposal", default="data/manifests/v0_candidate/split_proposal.csv")
    p.add_argument("--output", default="data/manifests/manifest.csv")
    p.add_argument("--matched-dir", default=None,
                   help="if given, also check that every file exists there")
    p.add_argument("--allow-pair-overlap", action="store_true",
                   help="write the manifest even if train and held-out share pair_ids")
    p.add_argument("--drop-train-pair-overlap", action="store_true",
                   help="leave out of the manifest every train image whose pair_id also "
                        "has a held-out image (held-out images are never touched)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an output manifest that already has rows")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = p.parse_args()

    cand = read_csv(args.candidate)
    prop = read_csv(args.proposal)
    errors, warnings = [], []

    cand_names = [r["filename"] for r in cand]
    prop_names = [r["filename"] for r in prop]
    for label, names in (("candidate", cand_names), ("proposal", prop_names)):
        dupes = [n for n, c in Counter(names).items() if c > 1]
        if dupes:
            errors.append(f"{label} has {len(dupes)} duplicated filenames, e.g. {dupes[:3]}")
    only_c, only_p = set(cand_names) - set(prop_names), set(prop_names) - set(cand_names)
    if only_c:
        errors.append(f"{len(only_c)} images are in the candidate but not the proposal, e.g. {sorted(only_c)[:3]}")
    if only_p:
        errors.append(f"{len(only_p)} images are in the proposal but not the candidate, e.g. {sorted(only_p)[:3]}")

    prop_by_name = {r["filename"]: r for r in prop}
    out_rows = []
    for r in cand:
        q = prop_by_name.get(r["filename"])
        if q is None:
            continue
        if (q["generator"] or "") != (r["generator"] or "") or str(q["is_ai"]) != str(r["is_ai"]):
            errors.append(f"{r['filename']}: generator/is_ai differ between candidate and proposal")
        if q["proposed_split"] not in ALLOWED:
            errors.append(f"{r['filename']}: proposed split '{q['proposed_split']}' is not train or held_out_gen")
        out_rows.append({**{c: r[c] for c in COLUMNS}, "split": q["proposed_split"]})

    # A held-out generator must never be in train.
    gens = defaultdict(set)
    for r in out_rows:
        if r["generator"]:
            gens[r["split"]].add(r["generator"])
    both = gens["train"] & gens["held_out_gen"]
    if both:
        errors.append(f"generator(s) {sorted(both)} appear in BOTH train and held_out_gen")
    if not gens["held_out_gen"]:
        errors.append("no held-out generator: nothing would measure generalisation")

    # Same pair_id on both sides = the model has seen that scene. A scene that is
    # AI-labelled in train and AI-labelled in the held-out set is the harmful case:
    # a frozen backbone encodes content, so it can recognise the scene, not the artifacts.
    dropped = []
    if args.drop_train_pair_overlap:
        held_pairs = {prop_by_name[r["filename"]]["pair_id"] for r in out_rows if r["split"] == "held_out_gen"}
        dropped = [r for r in out_rows
                   if r["split"] == "train" and prop_by_name[r["filename"]]["pair_id"] in held_pairs]
        gone = {r["filename"] for r in dropped}
        out_rows = [r for r in out_rows if r["filename"] not in gone]
        if dropped:
            kinds = Counter(r["generator"] or "real" for r in dropped)
            warnings.append(f"dropped {len(dropped)} train image(s) that share a scene with a held-out image: "
                            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    pairs = defaultdict(set)
    for r in out_rows:
        pairs[prop_by_name[r["filename"]]["pair_id"]].add(r["split"])
    leaked = sorted(k for k, v in pairs.items() if len(v) > 1)
    if leaked:
        msg = (f"{len(leaked)} pair_id(s) have images in both train and held_out_gen "
               f"(same scene on both sides), e.g. {leaked[:3]}. Rerun with "
               f"--drop-train-pair-overlap to remove the train side, or --allow-pair-overlap to keep them")
        (warnings if args.allow_pair_overlap else errors).append(msg)

    if args.matched_dir:
        missing = [r["filename"] for r in out_rows if not (Path(args.matched_dir) / r["filename"]).exists()]
        if missing:
            errors.append(f"{len(missing)} files are not in {args.matched_dir}, e.g. {missing[:3]}")

    print(f"{len(out_rows)} images")
    counts = Counter((r["split"], r["generator"] or "real", r["is_ai"]) for r in out_rows)
    for key, n in sorted(counts.items()):
        print(f"  {n:5d}  split={key[0]:<12} {key[1]:<18} is_ai={key[2]}")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        print("\nNot written. Fix the issues above (or ask Data to), then rerun.")
        sys.exit(1)

    out = Path(args.output)
    if args.dry_run:
        print(f"\nDry run: would write {len(out_rows)} rows to {out}")
        return
    if out.exists() and len(read_csv(out)) > 0 and not args.force:
        sys.exit(f"{out} already has rows; pass --force to overwrite it")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {out}. Commit it and tell Model.")


if __name__ == "__main__":
    main()

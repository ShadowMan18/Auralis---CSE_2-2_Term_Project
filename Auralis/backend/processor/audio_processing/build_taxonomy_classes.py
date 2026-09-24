"""
build_taxonomy_classes.py

Assigns every species in train.json (visipedia/inat_sounds 2024) to a
final training class, without discarding any species. Species with
enough recordings become their own class. Species with too few
recordings get rolled up to the finest ancestor taxon (genus -> family
-> order -> class) that accumulates enough combined recordings --
so a single Jaguar recording isn't thrown away, it just becomes part of
a coarser "Panthera" or "Felidae" class instead of a hopeless one-shot
"Jaguar" class.

This does NOT do the region/season filtering -- that's a separate,
inference-time step (your geo-temporal prior dataset). This only
decides training-time class resolution from recording counts.

Usage:
    python build_taxonomy_classes.py --min-count 10
    python build_taxonomy_classes.py --min-count 10 --supercategory Mammalia
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    with open(path) as f:
        return json.load(f)


def species_label(category):
    """A human-readable *and unique* species class name.

    iNatSounds has repeated common names (and hundreds of empty ones), so a
    bare ``common_name`` can silently merge different species into one CNN
    output.  Keep the scientific name in every species-level label; it is
    stable and gives the region/season prior an unambiguous key as well.
    """
    common = (category.get("common_name") or "").strip()
    scientific = category["name"].strip()
    return f"{common} ({scientific})" if common and common != scientific else scientific


def build_assignments(categories, counts, min_count, only_supercategory=None):
    """Returns {category_id: (final_label, resolution, recordings_in_final_class)}"""
    # group species under each ancestor taxon
    by_genus, by_family, by_order, by_class = (defaultdict(list) for _ in range(4))
    for c in categories:
        if only_supercategory and c["supercategory"] != only_supercategory:
            continue
        by_genus[c["genus"]].append(c)
        by_family[c["family"]].append(c)
        by_order[c["order"]].append(c)
        by_class[c["supercategory"]].append(c)

    def group_count(cats):
        return sum(counts.get(c["id"], 0) for c in cats)

    assignments = {}
    for c in categories:
        if only_supercategory and c["supercategory"] != only_supercategory:
            continue
        n = counts.get(c["id"], 0)
        if n >= min_count:
            assignments[c["id"]] = (species_label(c), "species", n)
            continue
        genus_cats = by_genus[c["genus"]]
        if group_count(genus_cats) >= min_count and len(genus_cats) > 1:
            assignments[c["id"]] = (f"{c['genus']} spp.", "genus", group_count(genus_cats))
            continue
        family_cats = by_family[c["family"]]
        if group_count(family_cats) >= min_count:
            assignments[c["id"]] = (f"{c['family']} (family)", "family", group_count(family_cats))
            continue
        order_cats = by_order[c["order"]]
        if group_count(order_cats) >= min_count:
            assignments[c["id"]] = (f"{c['order']} (order)", "order", group_count(order_cats))
            continue
        assignments[c["id"]] = (None, "excluded_zero_shot_only", 0)
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, default=HERE / "train.json")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--supercategory", default=None,
                         help="Restrict to one taxonomic class, e.g. Mammalia. Omit for all.")
    parser.add_argument("--out", type=Path, default=HERE / "species_class_assignments.json")
    args = parser.parse_args()

    data = load(args.train_json)
    categories = data["categories"]
    counts = Counter(a["category_id"] for a in data["annotations"])

    assignments = build_assignments(categories, counts, args.min_count, args.supercategory)

    by_resolution = Counter(v[1] for v in assignments.values())
    final_labels = {v[0] for v in assignments.values() if v[0] is not None}
    print(f"min_count={args.min_count}  scope={args.supercategory or 'ALL'}")
    print(f"species considered: {len(assignments)}")
    print(f"resolution breakdown: {dict(by_resolution)}")
    print(f"final training classes: {len(final_labels)}")
    print()
    print("sample assignments:")
    shown = 0
    for cid, (label, res, n) in sorted(assignments.items(), key=lambda kv: -kv[1][2]):
        cat = next(c for c in categories if c["id"] == cid)
        if res != "species":
            print(f"  {cat['common_name']:35s} ({res:6s}) -> '{label}'  combined_n={n}")
            shown += 1
        if shown >= 25:
            break

    out = {
        str(cid): {"final_label": label, "resolution": res, "combined_n": n}
        for cid, (label, res, n) in assignments.items()
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote full assignment table to {args.out}")


if __name__ == "__main__":
    main()

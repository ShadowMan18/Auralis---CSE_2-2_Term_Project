"""
inspect_inatsounds.py

Step 1 for using iNatSounds (visipedia/inat_sounds, 2024): check which of
your target species actually exist in the dataset, and how many
recordings each has, BEFORE downloading any of the audio tars (81GB /
25GB / 27GB for train/val/test). The annotation JSONs are tiny (a few MB
to 14MB) and contain everything needed to make that call.

License reminder (from the dataset's own terms): non-commercial research
and educational use only, and you may not redistribute the recordings.
Keep that in mind before this becomes part of anything commercial.

Usage:
    python inspect_inatsounds.py                      # checks train split
    python inspect_inatsounds.py --split val
    python inspect_inatsounds.py --print-extract-cmds  # also print streaming
                                                        # extraction commands
                                                        # for matched species
"""

import argparse
import json
import logging
import tarfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024"
ANNOTATION_URLS = {
    "train": f"{BASE_URL}/train.json.tar.gz",
    "val": f"{BASE_URL}/val.json.tar.gz",
    "test": f"{BASE_URL}/test.json.tar.gz",
}
AUDIO_TAR_URLS = {
    "train": f"{BASE_URL}/train.tar.gz",
    "val": f"{BASE_URL}/val.tar.gz",
    "test": f"{BASE_URL}/test.tar.gz",
}

# label -> substrings matched (case-insensitive) against common_name OR
# scientific name. Same caveat as before: "monkey" is not a species --
# put in real genera/species once you've picked which primate(s) you want.
TARGET_SPECIES = {
    "cat": ["felis catus", "domestic cat"],
    "dog": ["canis familiaris", "canis lupus familiaris", "domestic dog"],
    "cow": ["bos taurus", "domestic cattle", "cattle"],
    "horse": ["equus caballus", "equus ferus caballus"],
    "goat": ["capra hircus", "capra aegagrus hircus"],
    "crow": ["corvus brachyrhynchos", "american crow", "corvus corone", "carrion crow"],
    "rooster": ["gallus gallus", "chicken", "junglefowl"],
    # "monkey": [...],
}


def download_and_extract_json(split: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = cache_dir / f"{split}.json.tar.gz"
    json_path = cache_dir / f"{split}.json"
    if not json_path.exists():
        if not tgz_path.exists():
            logger.info("Downloading %s", ANNOTATION_URLS[split])
            r = requests.get(ANNOTATION_URLS[split], timeout=300, stream=True)
            r.raise_for_status()
            with open(tgz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        logger.info("Extracting %s", tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(cache_dir)
    with open(json_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--cache-dir", type=Path, default=Path(".inatsounds_cache"))
    parser.add_argument("--print-extract-cmds", action="store_true")
    args = parser.parse_args()

    data = download_and_extract_json(args.split, args.cache_dir)
    categories = data["categories"]
    annotations = data["annotations"]

    counts_by_cat_id = {}
    for ann in annotations:
        counts_by_cat_id[ann["category_id"]] = counts_by_cat_id.get(ann["category_id"], 0) + 1

    logger.info("Split '%s': %d categories, %d annotated recordings total",
                args.split, len(categories), len(annotations))

    matches = {}
    for cat in categories:
        blob = f"{cat.get('name', '')} {cat.get('common_name', '')}".lower()
        for label, patterns in TARGET_SPECIES.items():
            if any(p.lower() in blob for p in patterns):
                matches.setdefault(label, []).append(cat)

    print()
    for label in TARGET_SPECIES:
        cats = matches.get(label, [])
        if not cats:
            print(f"{label:10s}  NOT FOUND in {args.split} categories")
            continue
        for cat in cats:
            n = counts_by_cat_id.get(cat["id"], 0)
            print(f"{label:10s}  '{cat['common_name']}' ({cat['name']})  "
                  f"audio_dir_name={cat['audio_dir_name']}  recordings={n}")

    if args.print_extract_cmds:
        print("\n--- streaming extraction commands (still transfers the full")
        print("--- compressed tar over the network -- there's no random")
        print("--- access into a single .tar.gz) ---\n")
        tar_url = AUDIO_TAR_URLS[args.split]
        for label, cats in matches.items():
            for cat in cats:
                pattern = f"{args.split}/{cat['audio_dir_name']}/*"
                print(f"# {label}: {cat['common_name']}")
                print(f'curl -sL "{tar_url}" | tar -xzf - --wildcards "{pattern}"\n')


if __name__ == "__main__":
    main()
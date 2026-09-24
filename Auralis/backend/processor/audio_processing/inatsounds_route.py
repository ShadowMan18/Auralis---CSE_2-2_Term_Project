"""
inatsounds_route.py

Streams an iNatSounds train.tar.gz / val.tar.gz / test.tar.gz (visipedia/
inat_sounds, 2024) exactly once and routes each audio file into
    <output_dir>/<split>/<class_slug>/<original_filename>
where <class_slug> comes from species_class_assignments.json (produced by
build_taxonomy_classes.py): species with enough recordings get their own
folder, species below the threshold land in their rolled-up genus/family/
order folder, and species marked "excluded_zero_shot_only" are skipped
here entirely (they're recorded to a separate manifest for the zero-shot/
embedding-retrieval path instead -- see --excluded-manifest).

WHY STREAMING, NOT RANDOM ACCESS:
train.tar.gz is 81GB and gzip-compressed, so it is not randomly seekable --
selectively pulling out ~half the files inside it still means decompressing
the whole archive start to finish. This script does that ONE pass and
routes every member as it goes, rather than extracting everything to disk
first and sorting it out afterward (which would need >81GB of scratch
space just to get started). Budget for one long-running pass over the
full download.

THE UNDERSCORE GOTCHA (workflow summary, "next steps" #2):
dataset_producer.py's _class_label_for() (phase-1 script) splits a
filename on the FIRST underscore to recover the class from
"<class>_<index>.wav" -- so a rolled-up label like "Rodentia (order)"
would corrupt that scheme both because of the space/parens AND because a
literal underscore inside the label would be mistaken for the
class/index separator. Both are avoided completely here: slugify() strips
every non-alphanumeric character, so slugs never contain spaces, parens,
periods, OR underscores ("Rodentia (order)" -> "rodentiaorder",
"Panthera spp." -> "pantheraspp"). The corresponding real label is looked
up from class_label_map.json, never parsed back out of the slug.

The archive may be supplied as a local ``--tar`` or streamed directly from
the official ``--url``.  Direct streaming saves the additional 81 GB archive
on disk, but it cannot resume: if the connection fails, restart the stream.
Use ``--tar`` when you have enough disk space and want a resumable archive
download (see download_inatsounds.py).

Usage:
    # 1) Validate the metadata only. This is instant and does not download
    #    audio. It is the appropriate "dry run" for a large archive.
    python inatsounds_route.py --split train --annotations-json train.json \
        --assignments species_class_assignments.json --plan

    # 2) Stream the official archive directly to the routed class folders:
    python inatsounds_route.py --split train --annotations-json train.json \
        --assignments species_class_assignments.json --url official

    # 3) Or route an archive that was downloaded locally:
    python inatsounds_route.py \\
        --tar train.tar.gz --split train \\
        --annotations-json train.json \\
        --assignments species_class_assignments.json \\
        --output-dir inat_routed --dry-run

    # 2) Real run:
    python inatsounds_route.py \\
        --tar train.tar.gz --split train \\
        --annotations-json train.json \\
        --assignments species_class_assignments.json \\
        --output-dir inat_routed
"""

import argparse
import json
import logging
import re
import shutil
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROGRESS_EVERY = 5000
BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024"
AUDIO_TAR_URLS = {split: f"{BASE_URL}/{split}.tar.gz" for split in ("train", "val", "test")}


def slugify(label: str) -> str:
    """'Rodentia (order)' -> 'rodentiaorder'; 'Panthera spp.' -> 'pantheraspp'.
    No internal underscores, spaces, or punctuation -- see module docstring."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def build_routing_table(annotations_json: Path, assignments_path: Path):
    """Returns:
        file_name -> (class_slug, display_label, resolution)   -- included
        excluded_file_names -> [(file_name, common_name), ...] -- excluded
    file_name is exactly dataset["audio"][i]["file_name"] from the
    annotation JSON, which is what should appear (relative path) inside
    the tar.
    """
    data = load_json(annotations_json)
    assignments = load_json(assignments_path)  # {category_id(str): {final_label, resolution, combined_n}}

    audio_id2file = {a["id"]: a["file_name"] for a in data["audio"]}
    cat_id2name = {c["id"]: c.get("common_name", str(c["id"])) for c in data["categories"]}

    routing = {}
    excluded = []
    slug_to_label = {}
    unmatched_categories = 0

    for ann in data["annotations"]:
        file_name = audio_id2file.get(ann["audio_id"])
        if file_name is None:
            continue
        cat_id = ann["category_id"]
        info = assignments.get(str(cat_id))
        if info is None:
            unmatched_categories += 1
            continue
        if info["resolution"] == "excluded_zero_shot_only" or info["final_label"] is None:
            excluded.append((file_name, cat_id2name.get(cat_id, str(cat_id))))
            continue
        slug = slugify(info["final_label"])
        label_info = {"final_label": info["final_label"], "resolution": info["resolution"]}
        if slug in slug_to_label and slug_to_label[slug] != label_info:
            raise ValueError(
                f"Class-slug collision for '{slug}': {slug_to_label[slug]} versus {label_info}. "
                "Use an unambiguous slug scheme before routing."
            )
        slug_to_label[slug] = label_info
        routing[file_name] = (slug, info["final_label"], info["resolution"])

    if unmatched_categories:
        logger.warning(
            "%d annotations referenced a category_id not present in the assignments "
            "file -- re-run build_taxonomy_classes.py on the SAME train.json first "
            "if this number looks large.", unmatched_categories,
        )
    return routing, excluded, slug_to_label


def _lookup_for_member(name: str, routing: dict):
    """Constant-time lookup for normal archive paths.

    iNatSounds metadata already stores members as ``train/<taxon>/<file>``.
    A leading ``./`` is the only path variation observed in common tar tools,
    so normalise that rather than doing an O(members * recordings) suffix
    scan, which made a 137k-recording train archive impractical.
    """
    normalized = name.removeprefix("./")
    return routing.get(normalized) or routing.get(name)


@contextmanager
def open_tar_stream(tar_path: Path | None, url: str | None):
    """Yield a forward-only gzip tar stream from a file or HTTP response."""
    if tar_path is not None:
        logger.info("Opening local archive %s in streaming mode...", tar_path)
        with tarfile.open(tar_path, mode="r|gz") as tar:
            yield tar
        return

    assert url is not None
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("HTTP archive streaming requires 'requests'. Install backend requirements first.") from exc
    logger.info("Streaming archive from %s (this is not resumable)", url)
    response = requests.get(url, stream=True, timeout=(30, 300))
    try:
        response.raise_for_status()
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            yield tar
    finally:
        response.close()


def route(tar_path: Path | None, url: str | None, split: str, routing: dict,
          output_dir: Path, limit: int | None, skip_existing: bool):
    """Single streaming pass over the tar. Member names inside the tar are
    matched against file_name values from the annotation JSON; a member is
    considered a match if it ends with that file_name (handles the tar
    typically nesting members under a split/audio_dir_name prefix that the
    JSON's file_name may or may not already include)."""
    counts = Counter()
    matched = 0
    seen = 0
    written = 0
    skipped = 0

    (output_dir / split).mkdir(parents=True, exist_ok=True)

    with open_tar_stream(tar_path, url) as tar:
        for member in tar:
            if not member.isfile():
                continue
            seen += 1
            name = member.name

            slug_info = _lookup_for_member(name, routing)
            if slug_info is None:
                continue

            slug, display_label, resolution = slug_info
            matched += 1
            counts[slug] += 1

            class_dir = output_dir / split / slug
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / Path(name).name
            if out_path.exists() and skip_existing:
                skipped += 1
            else:
                fsrc = tar.extractfile(member)
                if fsrc is None:
                    logger.warning("Could not read matched archive member: %s", name)
                    continue
                with open(out_path, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                written += 1

            if matched % PROGRESS_EVERY == 0:
                logger.info("...%d members seen, %d routed so far", seen, matched)

            if limit is not None and matched >= limit:
                logger.info("Reached --limit=%d routed files, stopping stream early.", limit)
                break

    logger.info("Done: %d members seen, %d matched; %d written, %d already present; %d classes.",
                seen, matched, written, skipped, len(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--tar", type=Path, help="local train.tar.gz / val.tar.gz / test.tar.gz")
    source.add_argument("--url", help="archive URL, or 'official' for the iNatSounds S3 URL matching --split")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--annotations-json", type=Path, required=True,
                         help="the matching train.json/val.json/test.json")
    parser.add_argument("--assignments", type=Path, required=True,
                         help="species_class_assignments.json from build_taxonomy_classes.py "
                              "(built from train.json -- reuse the SAME file for val/test)")
    parser.add_argument("--output-dir", type=Path, default=Path("inat_routed"))
    parser.add_argument("--excluded-manifest", type=Path, default=Path("excluded_zero_shot_only.json"),
                         help="where to record files whose species is excluded_zero_shot_only, "
                              "for the separate zero-shot/embedding-retrieval path")
    parser.add_argument("--plan", action="store_true",
                        help="validate metadata and report expected class coverage without opening/downloading audio")
    parser.add_argument("--limit", type=int, default=None,
                         help="stop after routing this many files (smoke-testing only; "
                              "still streams from the start of the tar)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave matching files already on disk untouched (useful after an interrupted run; "
                             "a streamed archive still has to be read from its beginning)")
    args = parser.parse_args()

    if not args.plan and args.tar is None and args.url is None:
        parser.error("one of --tar or --url is required unless --plan is used")
    if args.tar is not None and not args.tar.is_file():
        parser.error(f"archive not found: {args.tar}")
    if args.url == "official":
        args.url = AUDIO_TAR_URLS[args.split]

    routing, excluded, slug_to_label = build_routing_table(args.annotations_json, args.assignments)
    logger.info("Routing table: %d files mapped to %d final classes; %d files excluded "
                "(zero-shot-only species)", len(routing), len(slug_to_label), len(excluded))

    if args.plan:
        planned = Counter(info[0] for info in routing.values())
        logger.info("Plan only: %d recordings across %d classes will be routed; %d excluded.",
                    len(routing), len(planned), len(excluded))
        return

    counts = route(args.tar, args.url, args.split, routing, args.output_dir, args.limit, args.skip_existing)

    label_map_path = args.output_dir / "class_label_map.json"
    existing = {}
    if label_map_path.exists():
        existing = load_json(label_map_path)
    existing.update(slug_to_label)
    with open(label_map_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Wrote/updated class label map: %s", label_map_path)

    with open(args.excluded_manifest, "w") as f:
        json.dump([{"file_name": fn, "common_name": cn} for fn, cn in excluded], f, indent=2)
    logger.info("Wrote excluded (zero-shot-only) manifest: %s (%d files)",
                 args.excluded_manifest, len(excluded))

    logger.info("Per-class file counts for this split (top 20 smallest, worth checking coverage):")
    for slug, n in sorted(counts.items(), key=lambda kv: kv[1])[:20]:
        logger.info("  %-30s %d", slug, n)


if __name__ == "__main__":
    main()

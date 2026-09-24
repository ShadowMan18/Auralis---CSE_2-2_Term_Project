"""Download official iNatSounds files with safe resume support.

The iNatSounds audio archives are large (approximately train 81 GB, val
25 GB, test 27 GB).  This helper downloads the annotation archive by default
and downloads audio only when ``--audio`` is explicitly supplied.  Partial
audio downloads are kept as ``.part`` files and resumed with HTTP Range on a
subsequent identical command.

Examples (run from this directory):

    # Small metadata download; extracts train.json automatically.
    python download_inatsounds.py --split train

    # Resumable local copy of the 81 GB train audio archive.
    python download_inatsounds.py --split train --audio

    # Do not retain the 81 GB archive: obtain train.json above, then route
    # audio directly to class folders with inatsounds_route.py --url official.

The dataset is for non-commercial research/education under its published
terms. Do not redistribute its recordings.
"""

import argparse
import logging
import os
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024"


def url_for(split: str, kind: str) -> str:
    filename = f"{split}.json.tar.gz" if kind == "annotations" else f"{split}.tar.gz"
    return f"{BASE_URL}/{filename}"


def download(url: str, destination: Path, chunk_size: int = 8 << 20) -> Path:
    """Download to ``destination`` atomically, resuming its .part file."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Downloading requires 'requests'. Install backend requirements first.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Already present: %s", destination)
        return destination

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requests.get(url, stream=True, headers=headers, timeout=(30, 300))
    try:
        # A server that ignores Range returns 200. Restart rather than append
        # a whole file to the partial one and corrupting it.
        if offset and response.status_code != 206:
            logger.warning("Server did not honour resume request; restarting %s", destination.name)
            offset = 0
        response.raise_for_status()
        mode = "ab" if offset else "wb"
        total = response.headers.get("Content-Range", response.headers.get("Content-Length"))
        logger.info("Downloading %s%s", url, f" (starting at {offset:,} bytes)" if offset else "")
        with open(partial, mode) as file_obj:
            received = offset
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_obj.write(chunk)
                received += len(chunk)
                if received and received % (512 << 20) < len(chunk):
                    logger.info("... downloaded %.1f GB%s", received / (1 << 30), f" (server length {total})" if total else "")
    finally:
        response.close()

    os.replace(partial, destination)
    logger.info("Saved %s (%.2f GB)", destination, destination.stat().st_size / (1 << 30))
    return destination


def extract_annotation_json(archive: Path, destination_dir: Path, split: str) -> Path:
    """Extract only the expected JSON member; never use extractall()."""
    destination = destination_dir / f"{split}.json"
    if destination.exists():
        logger.info("Already extracted: %s", destination)
        return destination
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar if member.isfile() and Path(member.name).name == destination.name]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one {destination.name} in {archive}; found {len(members)}")
        source = tar.extractfile(members[0])
        if source is None:
            raise RuntimeError(f"Could not read {members[0].name} from {archive}")
        with source, open(destination, "wb") as target:
            while chunk := source.read(8 << 20):
                target.write(chunk)
    logger.info("Extracted %s", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("inatsounds_downloads"))
    parser.add_argument("--audio", action="store_true", help="also download the very large audio archive")
    args = parser.parse_args()

    annotations = download(url_for(args.split, "annotations"), args.output_dir / f"{args.split}.json.tar.gz")
    extract_annotation_json(annotations, args.output_dir, args.split)
    if args.audio:
        download(url_for(args.split, "audio"), args.output_dir / f"{args.split}.tar.gz")
    else:
        logger.info("Metadata complete. Pass --audio only if you want a local archive; use inatsounds_route.py --url official to stream audio without retaining it.")


if __name__ == "__main__":
    main()

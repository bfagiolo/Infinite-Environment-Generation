"""Download the curated local Kenney asset library for the Godot renderer.

This is a setup-time tool, not a runtime dependency. It fetches official Kenney
CC0 pack zips, extracts them under assets/library/kenney, then refreshes the
semantic asset index used by world exports.
"""

from __future__ import annotations

import argparse
from html import unescape
import re
import shutil
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

from asset_resolver import ASSET_ROOT, build_asset_index


PACK_SLUGS = (
    "simple-space",
    "new-platformer-pack",
    "sports-pack",
    "foliage-pack",
    "background-elements",
    "particle-pack",
)

KENNEY_ASSET_PAGE = "https://kenney.nl/assets/{slug}"
USER_AGENT = "HarnessAlphaAssetSetup/1.0"


def download_curated_assets(
    *,
    root: Path = ASSET_ROOT,
    force: bool = False,
) -> None:
    """Download/extract every curated pack and rebuild the semantic index."""

    root.mkdir(parents=True, exist_ok=True)
    download_dir = root / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    for slug in PACK_SLUGS:
        destination = root / slug
        if destination.exists() and any(destination.iterdir()) and not force:
            print(f"ASSET_PACK: {slug} already present")
            continue
        url = _find_zip_url(slug)
        archive = download_dir / f"{slug}.zip"
        if force or not archive.exists():
            print(f"DOWNLOADING: {slug}")
            _download(url, archive)
        else:
            print(f"ZIP_CACHE: {archive}")
        print(f"EXTRACTING: {slug}")
        _extract_pack(archive, destination, force=force)

    payload = build_asset_index(root=root)
    print(f"ASSET_INDEX: assets/asset_index.json")
    print(f"ASSET_COUNT: {payload.get('asset_count', 0)}")


def _find_zip_url(slug: str) -> str:
    page_url = KENNEY_ASSET_PAGE.format(slug=slug)
    request = Request(page_url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise RuntimeError(f"Could not read Kenney asset page for {slug}: {exc}") from exc

    matches = re.findall(r"""href=['"]([^'"]+\.zip)['"]""", html, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError(f"No zip download link found on {page_url}")
    for match in matches:
        url = unescape(match)
        if f"/assets/{slug}/" in url:
            return url
    return unescape(matches[-1])


def _download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def _extract_pack(archive: Path, destination: Path, *, force: bool) -> None:
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(tmp_dir)
        roots = [path for path in tmp_dir.iterdir() if path.name != "__MACOSX"]
        source = roots[0] if len(roots) == 1 and roots[0].is_dir() else tmp_dir
        for item in source.iterdir():
            if item.name == "__MACOSX":
                continue
            target = destination / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download curated Kenney CC0 packs for renderer props.")
    parser.add_argument("--root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--force", action="store_true", help="redownload/reextract even when packs already exist")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    download_curated_assets(root=args.root, force=args.force)


if __name__ == "__main__":
    main()

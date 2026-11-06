#!/usr/bin/env python3
"""Collect the reviewed paper and portable website assets; never deploy them."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TRACKED_PDF = ROOT / "writeup/thinkorplace-report.pdf"
REPOSITORY = "https://github.com/brmurrell3/macro-placer"
METADATA = {
    "title": "Thinkorplace: Smooth Proposals and Exact Local Search for Chip Macro Placement",
    "author": "Brendan Murrell, Xiaofei Zhang, Andrew Murrell",
    "date": "2026-11-06",
    "version": "Technical report 1.1",
    "description": ("Ninth on the final leaderboard of the Partcl/HRT Macro Placement Challenge 2026 "
                    "(verified mean proxy 0.9974)."),
    "hero_image": "interactive/result-context.svg",
    "social_image": "interactive/result-context.png",
    "repository": REPOSITORY,
}
README = (
    "Thinkorplace website assets\n\n"
    "index.md is the article (Markdown, with a title line and byline). Keep the other files at the\n"
    "same relative paths beside it, or update its links to their deployed paths:\n"
    "  thinkorplace-technical-report.pdf   the technical report\n"
    "  placement-comparison.html           interactive explorer; self-contained, no network requests\n"
    "  interactive/result-context.svg      hero figure (PNG copy for social previews)\n"
    "manifest.json gives the title, author, date and version for the site's post metadata, and a\n"
    "SHA-256 for every file. These files have been packaged locally; no website was changed.\n"
)
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:href|src)=\"([^\"]+)\"")


def sources(pdf):
    """Package path -> source path. index.md is written from the website summary."""
    return {
        "index.md": ROOT / "writeup/website-summary.md",
        "thinkorplace-technical-report.pdf": pdf,
        "placement-comparison.html": ROOT / "writeup/interactive/placement-comparison.html",
        "interactive/result-context.svg": ROOT / "writeup/interactive/result-context.svg",
        "interactive/result-context.png": ROOT / "writeup/interactive/result-context.png",
    }


def broken_links(name, text, packaged):
    """Relative links in a packaged text file that do not resolve to another packaged file."""
    broken = []
    for match in LINK.finditer(text):
        target = (match.group(1) or match.group(2)).split("#")[0]
        if not target or re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        resolved = PurePosixPath(name).parent / target
        if str(resolved) not in packaged:
            broken.append(target)
    return broken


def pdf_title(path):
    """The PDF's document-info title, or None if it is not stored as a plain literal string."""
    match = re.search(rb"/Title\s*\(((?:\\.|[^\\)])*)\)", path.read_bytes())
    if not match:
        return None
    raw = re.sub(rb"\\([0-7]{1,3}|.)", lambda m: bytes([int(m.group(1), 8)]) if m.group(1)[:1].isdigit()
                 else m.group(1), match.group(1), flags=re.S)
    return raw[2:].decode("utf-16-be") if raw.startswith(b"\xfe\xff") else raw.decode("latin-1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/thinkorplace-technical-report.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "output/publication/thinkorplace")
    parser.add_argument("--update-tracked-pdf", action="store_true",
                        help=f"also copy the PDF to {TRACKED_PDF.relative_to(ROOT)}, the copy the README links")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the package layout, its sources and link checks without writing anything")
    args = parser.parse_args()
    files = sources(args.pdf)
    missing = [name for name, source in files.items() if not source.is_file()]
    broken = {}
    for name in ("index.md", "placement-comparison.html"):
        if name not in missing:
            broken[name] = broken_links(name, files[name].read_text(), files)
    title = pdf_title(args.pdf) if args.pdf.is_file() else None
    # A report built from an older version of main.tex carries an older title; never package it.
    wrong_pdf = title is not None and title != METADATA["title"]
    output = args.output.resolve()

    if args.dry_run:
        print(f"Package layout in {output} (zip: {output.with_suffix('.zip')}):")
        for name, source in files.items():
            state = "ok" if source.is_file() else "MISSING"
            print(f"  {name:36} <- {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source} [{state}]")
        print(f"  {'README.txt':36} <- generated\n  {'manifest.json':36} <- generated")
        for name, targets in broken.items():
            print(f"Relative links in {name}: {'all resolve' if not targets else 'BROKEN ' + ', '.join(targets)}")
        if args.pdf.is_file():
            print(f"PDF title: {title!r}" + (" [WRONG REPORT VERSION]" if wrong_pdf else
                                            " [not found; check the PDF by hand]" if title is None else " [ok]"))
        if args.update_tracked_pdf:
            print(f"Would copy {args.pdf} -> {TRACKED_PDF.relative_to(ROOT)}")
        ready = not missing and not any(broken.values()) and not wrong_pdf
        print("Ready to package." if ready else f"Not ready: {len(missing)} missing source(s), "
              f"{sum(map(len, broken.values()))} broken link(s), PDF title {'wrong' if wrong_pdf else 'ok'}.")
        sys.exit(0 if ready else 1)

    if missing:
        raise SystemExit("Required reviewed asset is missing: " + ", ".join(str(files[n]) for n in missing))
    if any(broken.values()):
        raise SystemExit(f"Relative links do not resolve inside the package: {broken}")
    if wrong_pdf:
        raise SystemExit(f"{args.pdf} is titled {title!r}, not {METADATA['title']!r}; rebuild it first")
    if title is None:
        print(f"Warning: could not read the title of {args.pdf}; check that it is the {METADATA['version']} report")
    for name, source in files.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output / name)
    (output / "README.txt").write_text(README)
    names = sorted([*files, "README.txt"])
    manifest = {
        **METADATA,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names},
        "scope": "Website article, technical report, hero figure, and historical placement explorer.",
        "source_directory": "writeup/",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*names, "manifest.json"]:
            bundle.write(output / name, arcname=f"thinkorplace/{name}")
    print(f"Packaged {len(names) + 1} files in {archive}")
    if args.update_tracked_pdf:
        shutil.copyfile(args.pdf, TRACKED_PDF)
        print(f"Updated {TRACKED_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

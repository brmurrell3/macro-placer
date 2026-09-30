#!/usr/bin/env python3
"""Collect the reviewed paper and portable website assets; never deploy them."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/thinkorplace-technical-report.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "output/publication/thinkorplace")
    args = parser.parse_args()
    sources = {
        "thinkorplace-technical-report.pdf": args.pdf,
        "placement-comparison.html": ROOT / "writeup/interactive/placement-comparison.html",
    }
    for source in sources.values():
        if not source.is_file():
            raise SystemExit(f"Required reviewed asset is missing: {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copyfile(source, output / name)
    summary = (ROOT / "writeup/website-summary.md").read_text()
    summary = summary.replace("../output/pdf/thinkorplace-technical-report.pdf",
                              "thinkorplace-technical-report.pdf")
    summary = summary.replace("interactive/placement-comparison.html",
                              "placement-comparison.html")
    (output / "index.md").write_text(summary)
    (output / "README.txt").write_text(
        "Thinkorplace website assets\n\n"
        "Adapt index.md to your website's post format and keep the PDF and HTML "
        "beside it, or update both relative links to their deployed paths.\n"
        "The interactive file embeds its placement data and loads D3 7.9.0 "
        "from jsDelivr; it needs network access for that library. "
        "Host the HTML as a page or embed it in an iframe.\n"
        "The report is a dated working technical report with historical results, "
        "new diagnostics, and explicitly scoped engineering runs. It does not "
        "claim a completed corrected full-suite comparison.\n"
        "These files have been packaged locally; no website was changed.\n"
    )
    names = sorted([*sources, "index.md", "README.txt"])
    manifest = {
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in names},
        "scope": "Website summary, technical report, and historical placement explorer.",
        "source_directory": "writeup/",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*names, "manifest.json"]:
            bundle.write(output / name, arcname=f"thinkorplace/{name}")
    print(f"Packaged {len(names) + 1} files in {archive}")


if __name__ == "__main__":
    main()

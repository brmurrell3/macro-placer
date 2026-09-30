#!/usr/bin/env python3
"""Check mechanical source and release invariants for the technical report."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX_ROOT = ROOT / "writeup/latex"
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
EXTERNAL_LABELS = {"LastPage"}

PLACEHOLDER_PATTERNS = {
    "editorial placeholder": re.compile(
        r"\b(?:TODO|FIXME|TBD|XXX|PLACEHOLDER|LOREM\s+IPSUM|CITATION\s+NEEDED|"
        r"INSERT(?:\s+(?:TEXT|FIGURE|TABLE|CITATION))?\s+HERE)\b",
        re.IGNORECASE,
    ),
    "raw tool citation token": re.compile(
        r"(?:turn\d+(?:search|fetch|view|open)\d+|(?:cite|filecite)|【\d+†[^】]+】)",
        re.IGNORECASE,
    ),
    "raw tool-call token": re.compile(
        r"(?:<(?:tool|function)_call\b|assistant\s+to=|tool_uses\b)",
        re.IGNORECASE,
    ),
}

ADVISORY_PHRASES = (
    "obviously",
    "clearly",
    "undeniably",
    "groundbreaking",
    "revolutionary",
    "remarkably",
    "dramatically",
    "state-of-the-art",
)


class Findings:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.advisories: List[str] = []
        self.passes: List[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def advise(self, message: str) -> None:
        self.advisories.append(message)

    def passed(self, message: str) -> None:
        self.passes.append(message)

    def emit(self) -> int:
        for message in self.passes:
            print("PASS: %s" % message)
        for message in self.advisories:
            print("ADVISORY: %s" % message)
        for message in self.errors:
            print("FAIL: %s" % message)
        print(
            "publication paper check: %d failure(s), %d advisory item(s)"
            % (len(self.errors), len(self.advisories))
        )
        return 1 if self.errors else 0


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        kept = []
        for index, character in enumerate(line):
            if character == "%":
                backslashes = 0
                cursor = index - 1
                while cursor >= 0 and line[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    break
            kept.append(character)
        lines.append("".join(kept))
    return "\n".join(lines)


def braced_argument(text: str, start: int) -> Optional[Tuple[str, int]]:
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None
    depth = 1
    cursor += 1
    content_start = cursor
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            cursor += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:cursor], cursor + 1
        cursor += 1
    return None


def literal_path(value: str) -> bool:
    return bool(value.strip()) and "#" not in value and "\\" not in value


def resolve_tex_path(tex_root: Path, value: str, extension: str = "") -> Path:
    path = Path(value.strip())
    if extension and not path.suffix:
        path = path.with_suffix(extension)
    return path if path.is_absolute() else tex_root / path


def resolve_graphic(tex_root: Path, value: str) -> Optional[Path]:
    path = resolve_tex_path(tex_root, value)
    if path.suffix:
        return path if path.is_file() else None
    for extension in GRAPHIC_EXTENSIONS:
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def select_file_conditionals(
    text: str,
    tex_root: Path,
    source: Path,
    findings: Findings,
    dependencies: Set[Path],
) -> str:
    """Select literal IfFileExists branches and require the referenced artifact."""
    marker = re.compile(r"\\IfFileExists\s*")
    output = []
    cursor = 0
    while True:
        match = marker.search(text, cursor)
        if match is None:
            output.append(text[cursor:])
            break
        output.append(text[cursor:match.start()])
        first = braced_argument(text, match.end())
        if first is None:
            output.append(text[match.start():match.end()])
            cursor = match.end()
            continue
        path_value, after_path = first
        true_branch = braced_argument(text, after_path)
        false_branch = braced_argument(text, true_branch[1]) if true_branch else None
        if true_branch is None or false_branch is None or not literal_path(path_value):
            output.append(text[match.start():after_path])
            cursor = after_path
            continue
        artifact = resolve_tex_path(tex_root, path_value)
        dependencies.add(artifact)
        if not artifact.is_file():
            findings.error(
                "%s conditionally references missing release artifact %s"
                % (display(source), display(artifact))
            )
        output.append(true_branch[0] if artifact.is_file() else false_branch[0])
        cursor = false_branch[1]
    return "".join(output)


def load_active_tex(
    path: Path,
    tex_root: Path,
    findings: Findings,
    dependencies: Set[Path],
    visited: Set[Path],
) -> str:
    path = path.resolve()
    if path in visited:
        findings.error("cyclic or repeated active TeX input: %s" % display(path))
        return ""
    visited.add(path)
    dependencies.add(path)
    try:
        text = strip_comments(path.read_text(encoding="utf-8"))
    except OSError as exc:
        findings.error("cannot read TeX source %s: %s" % (display(path), exc))
        return ""
    text = select_file_conditionals(text, tex_root, path, findings, dependencies)
    pattern = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
    output = []
    cursor = 0
    for match in pattern.finditer(text):
        output.append(text[cursor:match.start()])
        value = match.group(1).strip()
        if not literal_path(value):
            output.append(match.group(0))
        else:
            included = resolve_tex_path(tex_root, value, ".tex")
            if not included.is_file():
                findings.error(
                    "%s references missing TeX input %s"
                    % (display(path), display(included))
                )
            else:
                output.append(
                    load_active_tex(included, tex_root, findings, dependencies, visited)
                )
        cursor = match.end()
    output.append(text[cursor:])
    return "\n".join(output)


def source_files(tex_root: Path) -> List[Path]:
    return sorted(
        path for path in tex_root.rglob("*")
        if path.is_file()
        and path.suffix in (".tex", ".bib")
        and "build" not in path.relative_to(tex_root).parts
    )


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_placeholders(paths: Iterable[Path], findings: Findings) -> None:
    count = 0
    for path in paths:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for name, pattern in PLACEHOLDER_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.error(
                    "%s:%d contains %s %r"
                    % (display(path), line_number(text, match.start()), name, match.group(0))
                )
                count += 1
    if count == 0:
        findings.passed("no editorial placeholders or raw tool tokens in TeX/BibTeX")


def check_graphics(active: str, tex_root: Path, findings: Findings, dependencies: Set[Path]) -> None:
    references = []
    patterns = (
        re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}"),
        re.compile(r"\\(?:maybefigure|reportfigure)\s*\{([^{}]+)\}"),
    )
    for pattern in patterns:
        references.extend(match.group(1).strip() for match in pattern.finditer(active))
    checked = set()
    for value in references:
        if value in checked or not literal_path(value):
            continue
        checked.add(value)
        graphic = resolve_graphic(tex_root, value)
        if graphic is None:
            findings.error("missing referenced figure %s" % value)
        else:
            dependencies.add(graphic.resolve())
    if checked and all(resolve_graphic(tex_root, value) for value in checked):
        findings.passed("all %d literal figure references resolve" % len(checked))


def bibliography_paths(active: str, tex_root: Path) -> List[Path]:
    values = []
    for match in re.finditer(r"\\bibliography\s*\{([^{}]+)\}", active):
        values.extend(part.strip() for part in match.group(1).split(","))
    for match in re.finditer(r"\\addbibresource(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}", active):
        values.append(match.group(1).strip())
    return [resolve_tex_path(tex_root, value, ".bib").resolve() for value in values]


def check_citations_and_references(
    active: str, tex_root: Path, findings: Findings, dependencies: Set[Path]
) -> None:
    citations: Set[str] = set()
    cite_pattern = re.compile(
        r"\\(?:cite[a-zA-Z]*|nocite)\*?"
        r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]+)\}"
    )
    for match in cite_pattern.finditer(active):
        citations.update(
            key.strip() for key in match.group(1).split(",") if key.strip() != "*"
        )
    defined_citations: Set[str] = set()
    bib_paths = bibliography_paths(active, tex_root)
    for path in bib_paths:
        dependencies.add(path)
        if not path.is_file():
            findings.error("missing bibliography file %s" % display(path))
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        if duplicates:
            findings.error(
                "%s contains duplicate BibTeX keys: %s"
                % (display(path), ", ".join(duplicates))
            )
        defined_citations.update(keys)
    missing_citations = sorted(citations - defined_citations)
    if missing_citations:
        findings.error("undefined BibTeX citation keys: %s" % ", ".join(missing_citations))
    elif citations:
        findings.passed("all %d cited BibTeX keys are defined" % len(citations))

    labels = [
        value.strip() for value in re.findall(r"\\label\s*\{([^{}]+)\}", active)
        if literal_path(value)
    ]
    duplicate_labels = sorted(key for key in set(labels) if labels.count(key) > 1)
    if duplicate_labels:
        findings.error("duplicate active labels: %s" % ", ".join(duplicate_labels))
    reference_pattern = re.compile(
        r"\\(?:ref|pageref|eqref|autoref|cref|Cref|vref)\*?\s*\{([^{}]+)\}"
    )
    referenced = {
        key.strip()
        for match in reference_pattern.finditer(active)
        for key in match.group(1).split(",")
        if literal_path(key)
    }
    missing_labels = sorted(referenced - set(labels) - EXTERNAL_LABELS)
    if missing_labels:
        findings.error("undefined active labels: %s" % ", ".join(missing_labels))
    elif referenced:
        findings.passed(
            "%d active labels are unique and all %d references resolve in source"
            % (len(labels), len(referenced))
        )


def check_document_format(active: str, findings: Findings) -> None:
    match = re.search(
        r"\\documentclass\s*(?:\[([^\]]*)\])?\s*\{([^{}]+)\}", active
    )
    if match is None:
        findings.error("active source has no literal document class declaration")
        return
    options = {item.strip() for item in (match.group(1) or "").split(",") if item.strip()}
    document_class = match.group(2).strip()
    if document_class != "article":
        findings.error(
            "document class must be article for this standalone report; found %s"
            % document_class
        )
    if "11pt" not in options:
        findings.error("document class must explicitly select the 11pt body size")
    if "twocolumn" in options or re.search(r"\\twocolumn\b", active):
        findings.error("the standalone technical report must remain one column")
    if (
        document_class == "article"
        and "11pt" in options
        and "twocolumn" not in options
        and not re.search(r"\\twocolumn\b", active)
    ):
        findings.passed("document uses the 11pt one-column article format")


def check_tables(active: str, findings: Findings) -> None:
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", active, re.DOTALL)
    for index, table in enumerate(tables, 1):
        if re.search(r"\\(?:scriptsize|tiny)\b", table):
            findings.error("active table %d uses type smaller than the 9pt minimum" % index)
        if re.search(
            r"\\(?:resizebox|scalebox)\b.*?\\begin\{(?:tabular|tabularx)\}",
            table,
            re.DOTALL,
        ):
            findings.error("active table %d scales a tabular mechanically" % index)
        if "\\caption" not in table:
            findings.error("active table %d has no caption" % index)
        else:
            caption_position = table.find("\\caption")
            tabular_positions = [
                position for position in (
                    table.find("\\begin{tabular}"),
                    table.find("\\begin{tabularx}"),
                )
                if position >= 0
            ]
            if tabular_positions and caption_position > min(tabular_positions):
                findings.error("active table %d places its caption below the tabular" % index)
        if "\\label" not in table:
            findings.error("active table %d has no label" % index)
        for match in re.finditer(r"(?<![A-Za-z0-9])[-+]?\d+\.(\d{7,})(?![A-Za-z0-9])", table):
            findings.error(
                "active table %d reports more than six decimal places: %s"
                % (index, match.group(0))
            )
    if tables and not any("active table" in item for item in findings.errors):
        findings.passed(
            "%d active tables have top captions, labels, readable type, and bounded precision"
            % len(tables)
        )


def advisory_review(active: str, findings: Findings) -> None:
    for phrase in ADVISORY_PHRASES:
        matches = list(re.finditer(r"\b%s\b" % re.escape(phrase), active, re.IGNORECASE))
        if matches:
            findings.advise(
                "review %d use(s) of hand-wavy phrase %r in active prose"
                % (len(matches), phrase)
            )
    long_decimals = re.findall(
        r"(?<![A-Za-z0-9])[-+]?\d+\.\d{7,}(?![A-Za-z0-9])", active
    )
    if long_decimals:
        findings.advise(
            "review %d value(s) outside tables with more than six decimal places"
            % len(long_decimals)
        )
    findings.advise("manual visual review of every final PDF page is required")


def check_release(
    log_path: Path,
    pdf_path: Path,
    dependencies: Set[Path],
    findings: Findings,
) -> None:
    if not log_path.is_file():
        findings.error("final LaTeX log is missing: %s" % display(log_path))
    else:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        bad_patterns = {
            "LaTeX/package warning": r"(?im)^.*(?:LaTeX(?:\s+\w+)?|Package\s+\S+|Class\s+\S+|(?:pdf|Lua|Xe)TeX) Warning:.*$",
            "overfull box": r"(?im)^Overfull \\[hv]box.*$",
            "underfull box": r"(?im)^Underfull \\[hv]box.*$",
            "undefined reference/citation": r"(?im)^.*(?:undefined references?|Citation .* undefined|multiply defined).*$",
            "rerun request": r"(?im)^.*Rerun to get (?:cross-references|outlines) right.*$",
            "fatal TeX error": r"(?im)^.*(?:Fatal error|Emergency stop|No pages of output).*$",
        }
        for name, pattern in bad_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings.error(
                    "%s contains %d %s occurrence(s); first: %s"
                    % (display(log_path), len(matches), name, matches[0].strip())
                )
        if "Output written on" not in text:
            findings.error("final LaTeX log has no successful output record")
        elif not any(display(log_path) in item for item in findings.errors):
            findings.passed("final LaTeX log records a clean successful build")

    if not pdf_path.is_file():
        findings.error("final PDF is missing: %s" % display(pdf_path))
    elif pdf_path.stat().st_size == 0 or pdf_path.read_bytes()[:5] != b"%PDF-":
        findings.error("final PDF is empty or does not have a PDF header: %s" % display(pdf_path))
    else:
        findings.passed("final PDF is present and has a PDF header")

    existing_dependencies = [path for path in dependencies if path.is_file()]
    if log_path.is_file() and pdf_path.is_file() and existing_dependencies:
        newest = max(existing_dependencies, key=lambda path: path.stat().st_mtime_ns)
        oldest_output = min(log_path.stat().st_mtime_ns, pdf_path.stat().st_mtime_ns)
        if oldest_output < newest.stat().st_mtime_ns:
            findings.error(
                "build outputs are older than dependency %s; rebuild before release"
                % display(newest)
            )
        else:
            findings.passed("final log and PDF are no older than checked dependencies")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-root", default=str(DEFAULT_TEX_ROOT))
    parser.add_argument("--main", default="main.tex", help="path relative to --tex-root")
    parser.add_argument("--log", default="build/main.log", help="final log relative to --tex-root")
    parser.add_argument("--pdf", default="build/main.pdf", help="final PDF relative to --tex-root")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="check sources and referenced artifacts without requiring a final log or PDF",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tex_root = Path(args.tex_root).resolve()
    main_path = resolve_tex_path(tex_root, args.main).resolve()
    findings = Findings()
    dependencies: Set[Path] = set()
    if not main_path.is_file():
        findings.error("main TeX source is missing: %s" % display(main_path))
        return findings.emit()

    paths = source_files(tex_root)
    check_placeholders(paths, findings)
    active = load_active_tex(main_path, tex_root, findings, dependencies, set())
    check_graphics(active, tex_root, findings, dependencies)
    check_citations_and_references(active, tex_root, findings, dependencies)
    check_document_format(active, findings)
    check_tables(active, findings)
    advisory_review(active, findings)
    if not args.source_only:
        check_release(
            resolve_tex_path(tex_root, args.log).resolve(),
            resolve_tex_path(tex_root, args.pdf).resolve(),
            dependencies,
            findings,
        )
    else:
        findings.passed("source-only mode skipped final log/PDF and freshness checks")
    return findings.emit()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert Curvenote / JupyterBook chapter markdown into Hugo (lotusdocs) chapters.

Scope: CHAPTERS ONLY. This script reproduces the reference Book12-CPVR chapter
layout (frontmatter + opening quote/abstract alerts + figure divs + katex math).
Scaffolding, Part pages, and non-chapter pages are authored separately.

Usage:
    python3 tools/convert_curvenote.py \
        --src "NCVR Book" --out . --grouping tools/grouping.yml

Idempotent: rewrites content/docs/Part <ROMAN>/chapter-N.md. Never touches theme,
config, images, or non-chapter content.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    m = FM_RE.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, text[m.end():]


def emit_frontmatter(*, weight: int, chapter_no: int, description: str, date: str) -> str:
    # description may contain quotes/colons -> use YAML double-quote escaping.
    desc = (description or "").replace("\\", "\\\\").replace('"', '\\"')
    return (
        "---\n"
        f"weight: {weight}\n"
        f'title: "Chapter {chapter_no}"\n'
        f'description: "{desc}"\n'
        'icon: "article"\n'
        f'date: "{date}"\n'
        f'lastmod: "{date}"\n'
        "katex: true\n"
        "draft: false\n"
        "toc: true\n"
        "---\n"
    )


# ---------------------------------------------------------------------------
# Body transforms
# ---------------------------------------------------------------------------

OXA_RE = re.compile(r"^\+\+\+ \{.*\}\s*$", re.MULTILINE)
LABEL_RE = re.compile(r"^:label:.*$", re.MULTILINE)

# ```{math}\n:label: ID\n\n LATEX \n```  (label line optional)
# NOTE: option lines use [^\n]* (not .*) because re.DOTALL makes . match newlines.
MATH_RE = re.compile(
    r"```\{math\}[ \t]*\n(?::[a-zA-Z_]+:[^\n]*\n)*\s*\n?(.*?)\n```",
    re.DOTALL,
)

# ~~~{list-table} CAPTION\n:opts:\n\n * - cell\n   - cell ... \n~~~
LISTTABLE_RE = re.compile(
    r"~~~\{list-table\}[ \t]*(?P<cap>[^\n]*)\n(?P<body>.*?)\n~~~",
    re.DOTALL,
)

# ```{figure} PATH\n(:opt: val\n)*\n?CAPTION```
FIGURE_RE = re.compile(
    r"```\{figure\}[ \t]*(?P<path>\S+)[ \t]*\n"
    r"(?P<opts>(?::[a-zA-Z_]+:[^\n]*\n)*)"
    r"(?P<rest>.*?)```",
    re.DOTALL,
)

QUOTE_SPLIT_RE = re.compile(r"\s+[—–-]\s+")


def strip_oxa(body: str) -> str:
    body = OXA_RE.sub("", body)
    return body


def convert_math(body: str) -> str:
    def repl(m: re.Match) -> str:
        latex = m.group(1).strip("\n")
        return f"$$\n{latex}\n$$"

    return MATH_RE.sub(repl, body)


def convert_list_tables(body: str) -> str:
    def repl(m: re.Match) -> str:
        caption = m.group("cap").strip()
        inner = m.group("body")
        inner = re.sub(r"^:[a-zA-Z_-]+:[^\n]*\n", "", inner, flags=re.MULTILINE)
        row_blocks = re.split(r"\n(?=\*\s+-\s)", inner.strip())
        rows: list[list[str]] = []
        for rb in row_blocks:
            rb = rb.strip()
            if not rb:
                continue
            cells = re.split(r"(?m)^\s*(?:\*\s+)?-\s+", rb)
            cells = [
                re.sub(r"\s+", " ", c.strip()).replace("|", r"\|")
                for c in cells
                if c.strip()
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        out: list[str] = []
        if caption:
            out.append(f"*{caption}*\n")
        out.append("| " + " | ".join(rows[0]) + " |")
        out.append("| " + " | ".join(["---"] * ncol) + " |")
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    return LISTTABLE_RE.sub(repl, body)


def convert_figures(body: str) -> str:
    def repl(m: re.Match) -> str:
        src = m.group("path").strip()
        # normalise "images/x.png" -> "/images/x.png"
        base = src.split("/")[-1]
        src_url = f"/images/{base}"
        opts = m.group("opts") or ""
        width = "70%"
        wm = re.search(r":width:\s*(\S+)", opts)
        if wm:
            width = wm.group(1).strip()
        caption = m.group("rest").strip()
        caption = re.sub(r"\s*\n\s*", " ", caption).strip()
        cap_html = f"\n        <p>{caption}</p>" if caption else ""
        return (
            '<div class="row justify-content-center">\n'
            '    <div class="rounded p-4 position-relative overflow-hidden '
            f'border-1 text-center" style="width: {width}">\n'
            f'        {{{{< figure src="{src_url}" >}}}}'
            f"{cap_html}\n"
            "    </div>\n"
            "</div>"
        )

    return FIGURE_RE.sub(repl, body)


def build_alerts(preamble: str) -> str:
    """Turn the leading blockquote + italic abstract into two lotusdocs alerts."""
    quote = None
    abstract = None

    # blockquote: consecutive lines starting with '>'
    q_lines: list[str] = []
    for line in preamble.splitlines():
        s = line.strip()
        if s.startswith(">"):
            q_lines.append(s.lstrip(">").strip())
        elif q_lines and not s:
            continue
        elif q_lines:
            break
    if q_lines:
        quote = " ".join(q_lines).strip()

    # abstract: first standalone *...* italic paragraph
    am = re.search(r"^\*(.+?)\*\s*$", preamble, re.DOTALL | re.MULTILINE)
    if am:
        abstract = am.group(1).strip()

    out = []
    if quote:
        parts = QUOTE_SPLIT_RE.split(quote, maxsplit=1)
        if len(parts) == 2:
            body_q = f'"<em>{parts[0].strip().strip(chr(34))}</em>" — {parts[1].strip()}'
        else:
            body_q = f"<em>{quote}</em>"
        out.append(
            '{{% alert icon="💡" context="info" %}}\n'
            f"<strong>{body_q}</strong>\n"
            "{{% /alert %}}"
        )
    if abstract:
        abstract = re.sub(r"\s*\n\s*", " ", abstract).strip()
        out.append(
            '{{% alert icon="📘" context="success" %}}\n'
            f'<p style="text-align: justify;"><em>{abstract}</em></p>\n'
            "{{% /alert %}}"
        )
    return "\n\n".join(out)


def convert_body(body: str) -> str:
    body = strip_oxa(body)

    # split preamble (before first top-level '# ' heading) from the rest
    hm = re.search(r"^# ", body, re.MULTILINE)
    if hm:
        preamble, main = body[: hm.start()], body[hm.start():]
    else:
        preamble, main = body, ""

    alerts = build_alerts(preamble)

    main = convert_list_tables(main)
    main = convert_math(main)
    main = convert_figures(main)
    main = LABEL_RE.sub("", main)

    combined = (alerts + "\n\n" + main) if alerts else main
    # collapse 3+ blank lines
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    return combined + "\n"


# ---------------------------------------------------------------------------
# Warnings: catch any directive we do NOT handle so nothing is silently dropped
# ---------------------------------------------------------------------------

KNOWN = {"figure", "math", "list-table"}
DIRECTIVE_RE = re.compile(r"(?:```|~~~)\{([a-zA-Z-]+)\}")


def warn_unknown(chapter_file: str, raw_body: str) -> None:
    for m in DIRECTIVE_RE.finditer(raw_body):
        name = m.group(1)
        if name not in KNOWN:
            line = raw_body[: m.start()].count("\n") + 1
            print(
                f"  WARNING [{chapter_file}:{line}] unhandled directive "
                f"```{{{name}}}``` left as-is",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Curvenote source dir (has chapter-N.md)")
    ap.add_argument("--out", required=True, help="Hugo repo root (writes content/docs/...)")
    ap.add_argument("--grouping", required=True, help="grouping.yml")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    cfg = yaml.safe_load(Path(args.grouping).read_text())
    date = cfg["date"]

    written = 0
    for i, part in enumerate(cfg["parts"], start=1):
        roman = part["roman"]
        base = 500 + (i - 1) * 600
        part_dir = out / "content" / "docs" / f"Part {roman}"
        part_dir.mkdir(parents=True, exist_ok=True)
        for k, ch in enumerate(part["chapters"]):
            src_file = src / f"chapter-{ch}.md"
            if not src_file.exists():
                print(f"  MISSING source: {src_file}", file=sys.stderr)
                continue
            raw = src_file.read_text()
            fm, body = split_frontmatter(raw)
            warn_unknown(f"chapter-{ch}.md", body)
            weight = base + 100 + k * 100
            description = (fm.get("subtitle") or "").strip()
            doc = emit_frontmatter(
                weight=weight, chapter_no=ch, description=description, date=date
            ) + "\n" + convert_body(body)
            dst = part_dir / f"chapter-{ch}.md"
            dst.write_text(doc)
            written += 1
            print(f"  wrote {dst}  (weight {weight})")

    print(f"\nDone: {written} chapters converted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

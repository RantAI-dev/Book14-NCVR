# NCVR Web Book — Design

**Date:** 2026-07-06
**Status:** Approved (design), pending implementation plan

## Goal

Produce a Hugo web book for **NCVR — Numerical Computing via Rust**, structurally identical to the reference **Book12-CPVR** (Hugo + lotusdocs theme, deployed via GitHub Pages workflow). Source content is raw Curvenote/JupyterBook markdown in `NCVR Book/`. A reusable Python converter transforms Curvenote → Hugo lotusdocs content.

## Key facts established

- Reference `ref/Book12-CPVR` is a Hugo site using theme `lotusdocs` (git module). Content lives in `content/docs/`, images in `static/images/`.
- **No converter script exists** anywhere in the RantAI-dev org (checked all 14 book repos + git history + `book-themes`). Curvenote→Hugo conversion was done off-repo and never saved. We build it ourselves.
- NCVR source: `NCVR Book/` with `_toc.yml` (jb-book, 22 chapters + preface-1/foreword/introduction), `curvenote.yml`, flat `chapter-N.md`, and `images/` (42 files).

## Decisions

- **Part grouping:** 8 parts (approved).
- **Non-chapter pages:** follow CPVR template layouts, filled with NCVR content.
- **Converter scope (refined):** the Python converter handles **chapters only** — its job is to reproduce the CPVR chapter layout exactly. Scaffolding, non-chapter pages, and Part pages are written **manually** (via parallel subagents) for speed and control.
- **Cover:** reuse CPVR `cover.png` as placeholder.

## Target repo structure

```
Book13-NCVR/
├── hugo.toml               # from CPVR; baseURL→ncvr.rantai.dev, titles→NCVR
├── go.mod / go.sum         # same lotusdocs + bootstrap-scss modules
├── themes/lotusdocs/       # copied from CPVR
├── .github/workflows/hugo.yaml
├── archetypes/ dockerfile .gitignore LICENSE README.md  # from CPVR
├── static/images/          # 42 NCVR images + cover.png (CPVR placeholder)
├── content/docs/
│   ├── ncvr.md             # About This Book (weight 100)
│   ├── how-to-use.md       # weight 200
│   ├── table-of-contents.md
│   ├── preface.md  foreword.md            # weight 300/400
│   ├── part-1-main.md … part-8-main.md    # Part intro pages (weight 500,700,…)
│   └── Part I … Part VIII/
│        ├── _index.md      # Part landing (weight 550,750,…)
│        └── chapter-N.md   # converted chapters (weight 600,610,… or 800…)
└── tools/
    ├── convert_curvenote.py
    └── grouping.yml         # part→chapter mapping + part titles
```

## Part grouping (8 parts)

| Part | Title | Chapters |
|---|---|---|
| I | Foundations | 1 |
| II | Linear Systems & Function Approximation | 2, 3, 4, 5, 6 |
| III | Randomness & Data Ordering | 7, 8 |
| IV | Optimization, Roots & Eigensystems | 9, 10, 11 |
| V | Spectral Methods | 12, 13 |
| VI | Data Modeling & Inference | 14, 15, 16 |
| VII | Differential & Integral Equations | 17, 18, 19, 20 |
| VIII | Geometry & Trustworthy Execution | 21, 22 |

## Converter: `tools/convert_curvenote.py`

**Inputs:** `--src "NCVR Book"`, `--out .` (repo root), `--grouping tools/grouping.yml`.
**Behavior:** pure content generation into `content/docs/` and `static/images/`; does not touch theme/config (those are scaffolded separately).

### Transformations (per chapter)

1. **Frontmatter rewrite** — drop Curvenote frontmatter (authors/oxa/tags), emit lotusdocs frontmatter:
   - `weight`: auto-assigned per grouping (part base + chapter offset), matching CPVR's increasing-weight scheme.
   - `title`: `"Chapter N"`; `description`: the Curvenote `subtitle`.
   - `icon: "article"`, `katex: true`, `draft: false`, `toc: true`, `date`/`lastmod` (fixed timestamp passed via arg to keep runs deterministic).
2. **Opening quote + abstract** — the leading `> "..."` blockquote and the following `*italic abstract*` become two lotusdocs alerts:
   - quote → `{{% alert icon="💡" context="info" %}}<strong>...</strong>{{% /alert %}}`
   - abstract → `{{% alert icon="📘" context="success" %}}<p style="text-align: justify;"><em>...</em></p>{{% /alert %}}`
3. **Figures** — ` ```{figure} images/X ``` ...``` ` → `{{< figure src="/images/X" >}}`; discard directive options/caption block (matches CPVR, which renders bare figures).
4. **Math** — ` ```{math}\n:label: ID\n<latex>\n``` ` → `$$\n<latex>\n$$`. Inline `$...$` and existing `$$...$$` left untouched (katex renders them).
5. **Strip Curvenote residue** — remove `+++ {"oxa":...}` fences, stray `:label:` lines, and other MyST directives not handled above (log a warning listing any unknown ` ```{directive} ` encountered so nothing is silently dropped).
6. **Images copy** — copy every file from `NCVR Book/images/` → `static/images/`.

### Part & non-chapter page generation

- For each part: write `Part <roman>/_index.md` and `part-<n>-main.md` from CPVR-derived templates, with the part title from grouping and a placeholder abstract alert (to be hand-edited later).
- Non-chapter pages (`ncvr.md`, `how-to-use.md`, `table-of-contents.md`, `preface.md`, `foreword.md`): scaffold from CPVR layout; `preface.md`/`foreword.md` bodies filled by converting `NCVR Book/preface-1.md` and `NCVR Book/foreword.md`; `ncvr.md`/`how-to-use.md`/`table-of-contents.md` use CPVR structure with NCVR text (title/description/TOC links regenerated from grouping). Content that has no NCVR source is marked with a visible `TODO` alert rather than fabricated.

### Idempotency & safety

- Re-runnable: overwrites generated `content/docs/**` and `static/images/**`, never edits `themes/`, `hugo.toml`, or hand-authored files outside its output set.
- Unknown-directive warnings printed to stderr with file + line so no construct is silently lost.

## Scaffolding (one-time, separate from converter)

Copy from `ref/Book12-CPVR`: `themes/lotusdocs`, `go.mod`, `go.sum`, `archetypes/`, `dockerfile`, `.gitignore`, `LICENSE`, `.github/workflows/hugo.yaml`, and `hugo.toml` (then edit `baseURL`, `title`, language titles, logo/menu text CPVR→NCVR). Copy `static/images/cover.png` as placeholder.

## Verification

1. `hugo` builds locally with no errors/warnings.
2. Spot-check rendered pages in browser: a chapter with alerts + figure + katex math renders correctly; Part navigation and TOC links resolve; sidebar weight ordering matches intended chapter order.
3. Confirm all 42 images resolve (no 404s in build).

## Out of scope

- Final cover art (placeholder now).
- Hand-authored prose for pages lacking NCVR source (left as TODO alerts).
- Multilingual (zh/id) content — keep CPVR's language config but no translated content.

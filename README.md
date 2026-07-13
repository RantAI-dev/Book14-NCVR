# Numerical Computing via Rust (NCVR)

Web edition of **Numerical Computing via Rust (NCVR)**, built with [Hugo](https://gohugo.io/) and the [Lotus Docs](https://github.com/colinwilson/lotusdocs) theme, and deployed to GitHub Pages.

## Development

```bash
hugo mod get ./...     # first run only: fetch theme/bootstrap modules
hugo server            # local preview at http://localhost:1313
```

Requires **Hugo Extended** `0.131.0` and Dart Sass.

## Content pipeline

The book chapters are converted from Curvenote/JupyterBook markdown into Hugo
lotusdocs pages by `tools/convert_curvenote.py`:

```bash
python3 tools/convert_curvenote.py --src "NCVR Book" --out . --grouping tools/grouping.yml
```

- `NCVR Book/` — raw Curvenote source (chapters, images, `_toc.yml`).
- `tools/grouping.yml` — part → chapter grouping and metadata.
- `content/docs/` — generated Hugo content (Parts + chapters) plus hand-authored
  landing, preface, foreword, and table-of-contents pages.

## Deployment

Pushing to `main` triggers `.github/workflows/hugo.yaml`, which builds the site
with Hugo and publishes it to GitHub Pages.

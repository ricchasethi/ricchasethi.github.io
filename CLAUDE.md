# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal portfolio/blog site for ricchasethi.github.io, built with Jekyll and hosted on GitHub Pages. Pushing to `main` triggers automatic deployment via GitHub Pages (no CI pipeline — the GitHub Actions workflow was intentionally removed).

## Commands

```bash
# Serve locally with live reload
jekyll serve

# Build static site to _site/
jekyll build
```

Jekyll 4.4.1 and Bundler 4.0.11 are available. There is no Gemfile; GitHub Pages builds using its own default Jekyll environment.

## Architecture

The site uses Jekyll's default conventions with minimal configuration:

- `_config.yml` — site-wide settings (currently only `title`)
- `about.md` — static About page (no front matter; rendered as a page)
- `_posts/` — blog posts in `YYYY-MM-DD-title.md` format; each requires front matter with `layout`, `title`, and `date`

No custom theme, layouts, or includes are defined — the site relies on GitHub Pages' default Jekyll theme. To add a theme, set `theme:` in `_config.yml` (e.g., `theme: minima`).

# Proposal

## Why

The site uses Quarto's default look, and its homepage gives posts, publications and presentations equal weight as image-card grids. The site should present itself as a blog: posts are the main content, and publications are a plain reference list. Presentations have already been removed from the source, so the homepage needs to stop referring to them.

## What Changes

- Replace the default appearance with a custom modern, minimal sans-serif theme, with one light colour scheme and a single accent colour.
- Turn the homepage into a landing page adapted from an open-source template: an animated hero with the owner's roles, a CV-based summary of their research and open-source work , and three cards advertising Posts, Open source and Publications, all fitting on one screen without scrolling.
- Add a dedicated Publications page, linked from the navbar, listing 16 publications from the CV with authors, venue and year, newest first, with titles linking directly to the paper.
- Add pill-style navbar buttons for Posts, Open source and Publications, each opening its own page with an animated header, keeping search and the GitHub and LinkedIn links and adding Google Scholar. The Open source page shows each project with a relevant thumbnail.
- Remove the Presentations section from the homepage.
- **BREAKING**: Remove the individual publication pages, along with their abstracts and images. Their URLs under `publications/<slug>/` stop working once the site is next deployed.

## Capabilities

### New Capabilities

- `site-appearance`: The visual identity shared by all pages, including typography, colour, and readability of technical content.
- `site-navigation`: The site-wide links and search that lead readers to posts and publications.
- `landing-page`: The homepage hero that introduces the owner.
- `open-source-projects`: The page presenting the owner's open-source projects.
- `post-listing`: How the homepage presents posts as the site's main content.
- `publications-list`: How publications are presented as a plain list linking to each paper.

### Modified Capabilities

None. The existing specs cover the lecture-notes post and are unaffected.

## Impact

- **Site source**: `_quarto.yml`, `index.qmd`, a new theme stylesheet, a new `publications.qmd`, landing page assets including a listing template and a vendored particle script, and a date metadata file beside the published post.
- **Removed source**: the `publications/` directory. The abstracts and images remain in git history.
- **Published post**: `posts/imbalanced-learn-extra/index.qmd` must stay unedited, because any edit would force its code to re-run with packages that are not installed in `.venv`.
- **Deployment**: `docs/` is the committed deploy output and still contains the old site. It is not changed here, so the new design only goes live at the next deploy.
- **Content**: only `imbalanced-learn-extra` is currently visible as a post, because Quarto skips the `posts/_*` folders. Publishing those is out of scope.
- **Dependencies**: no external services at runtime. The site adapts the MIT-licensed Start Bootstrap Grayscale template and vendors the MIT-licensed `particles.js`.

# Design

## Context

The site is a Quarto website with no custom styling. `index.qmd` holds three image-card listings, one of which points at the deleted `presentations/` folder. Each of the 15 publications is a folder holding only front matter (title with embedded link, authors, abstract, categories, image), with no body text and no recorded year or venue.

`freeze: auto` compares a stored hash of the whole `.qmd` file before reusing frozen output. Any edit to `posts/imbalanced-learn-extra/index.qmd` would re-run its 10 code cells, which pin `pandas==2.2.2` and `imbalanced-learn-extra==0.2.6`. Neither is installed in `.venv`, which runs Python 3.14. The post also has no `date`.

## Goals / Non-Goals

**Goals:**

- Keep all styling in one small stylesheet that is easy to adjust.
- Leave the published post's source and frozen output untouched.

**Non-Goals:**

- Publishing the `posts/_*` drafts.
- Updating `docs/` or deploying.
- Changing the `_quarto.yml` render list.
- Editing the CV.

## Decisions

**Theme built on Bootswatch `cosmo` with one custom SCSS file.** `_quarto.yml` uses `theme: [cosmo, theme.scss]`, which gives well-tested components. An earlier light and dark version with a toggle was removed at the owner's request, leaving one light scheme. A from-scratch theme or custom HTML template would mean maintaining what Quarto already provides.

**System sans-serif font stack.** Using the fonts already on the reader's device adds no external requests and no font-loading delay. Inter from Google Fonts was the alternative, but it adds a third-party request for a small visual gain.

**Landing page adapts Start Bootstrap Grayscale.** It is MIT-licensed and built on Bootstrap 5, the framework Quarto already uses, and its dark masthead and featured-project row match what this page needs. Only its masthead and featured-row patterns are ported into `theme.scss`. Its navbar, Bootstrap copy, Google Fonts, form script and photographs are dropped. No maintained animated Quarto landing template exists. Start Bootstrap Creative was rejected for its warm photo-led look. The Quarto `particle-js` recipe was used only as evidence that the approach works, since its repository has no licence.

**Hero differs from the template in three ways.** It is shortened from full-screen height so posts remain visible, it uses a dark gradient with particles instead of a photo, and its title uses a solid gradient instead of fading to transparent, which would fail the contrast requirement. It stays dark in both modes and fades into the page background.

**Extra visual layers are plain CSS and a small script.** Moving colour glows, shimmer, glass panels, staggered entrance, scroll reveal, count-up figures and pointer-following card highlights add depth with no new dependency. Animation libraries were rejected as heavy for this. Hidden-until-revealed styling only applies after the script confirms it can run, so content stays visible without it. Everything continuous is disabled for reduced motion.

**Summary and highlights come only from the CV, and sit in the hero.** Contact details and client names are left out. The publication and project counts are written into the page, so they must be updated by hand with new entries.

**Homepage is a dark hero above three cards on the page background.** Together they fill the viewport below the navbar, the hero taking the spare height, and sizes scale with viewport height so the page fits without scrolling. A full-screen dark hero with glass cards was tried and rejected by the owner, who prefers the contrast of a separate hero. The Posts card comes from a small listing template, so it follows the newest post; the other two cards are written by hand, and their counts need updating with new entries. Card layout replaces the earlier highlight panels, which would have repeated the same numbers. One publication without a verified link is omitted at the owner's request.

**Tags filter the Posts page through the address.** Links use `posts.html#tag=<name>`, so the same link works from the homepage, the Posts page and post pages, and can be shared. A small script applies it, and all posts stay visible without the script. Quarto's built-in category filter was rejected because it depends on its own listing markup, which custom templates replace. On post pages, an include set in `posts/_metadata.yml` turns tags into links, leaving the `.qmd` files and their frozen output alone.

**Section pages share one header pattern and one script include.** Posts, Open source and Publications use `page-layout: custom`, the same glow and particle layers as the homepage, and `assets/landing-scripts.html`. The Open source section moves off the homepage to its own page, leaving the hero and latest post there. The header's markup is repeated in each page because Quarto has no parameterised include for raw HTML, and it is small.

**Project thumbnails are hand-drawn SVGs in `assets/oss/`.** One consistent visual language keeps the grid coherent, the files are small and sharp at any size, and no third-party images or licences are involved. Using logos from each repository was rejected because not every project has one and the styles would clash.

**Particles use a vendored `particles.js`.** At 43 KB unminified with no dependencies, it covers this need. `tsparticles` is actively maintained but much larger, and a fixed animation has little need for updates. The file is served from the site, and the script stops movement when the reader prefers reduced motion. If it fails to load, the hero's CSS background and content still work.

**Posts use a custom listing template with `page-layout: custom`.** An EJS template renders the first sorted item as the feature and any others as the simple list, so the feature updates itself on each render. `sort: "date desc"` and `date-format: "MMM YYYY"` are kept. Hand-writing the feature was rejected, as it would go stale.

**Publications page is Markdown in `publications.qmd`, generated once from the CV.** The CV's `cv.md` is the source for titles, authors, venues, years and links, including two publications the old site lacked. A script writes the page and an independent parser checks it, and it remains ordinary Markdown to edit by hand. Year headings give newest-first order, with the CV's order kept within a year. Three CV entries have no link: two are restored from the old site after confirming them against arXiv, and the third is omitted at the owner's request. One author's missing initial comes from the old site data. The CV itself is not edited.

**Post date comes from a `_metadata.yml` beside the post.** `posts/imbalanced-learn-extra/_metadata.yml` sets `date: 2023-06-22`, the date the post was first added to git. A scratch project confirmed that this date displays and sorts correctly in a listing, and it leaves the `.qmd` hash unchanged. Installing the pinned packages to allow editing the post was rejected, since their compatibility with Python 3.14 is unknown and a re-run could change the output.

**Verify with a full render and browser checks.** There is no test suite, so tasks verify using `quarto render` with `.venv` active, checks of the generated HTML, and viewing pages in both modes.

## Risks / Trade-offs

- [Old publication URLs stop working after the next deploy, and abstracts leave the site] → Accepted by the owner. The source remains in git history.
- [`particles.js` is no longer actively developed] → It is vendored and has no runtime dependencies, and failure leaves the hero intact. Replacing it would only touch one script.
- [Animation can use battery on low-power devices] → The particle count is modest, and it is reduced on narrow screens.
- [The render could fail on an unrelated page or re-run the post] → Compare the post's frozen hash and stored output before and after the render, and stop if they change.
- [Dark-mode styling could make frozen tables or images hard to read] → Check the post's rendered tables and images in both modes.
- [A post date in `_metadata.yml` is less discoverable than front matter] → Only this post needs it. New posts use a normal `date:` field.

## Migration Plan

Changes are made in the working tree and render to `_site/`, which is ignored by git. Deletion of `publications/` happens only after the new page has been checked against its data. Going live remains the owner's existing deploy step, and rollback is a matter of restoring the files from git.


# Tasks

## 1. Publications page

- [x] 1.1 Generate `publications.qmd` from the 15 publication front matter blocks as a plain list of linked titles and authors, then verify by script that all 15 titles, author lists and URLs match the source.
- [x] 1.2 Delete the `publications/` directory only after 1.1 passes, then verify `git status` shows only that directory's deletions and `quarto inspect .` no longer lists any `publications/<slug>/` page.

## 2. Homepage and navigation

- [x] 2.1 Add `posts/imbalanced-learn-extra/_metadata.yml` with `date: 2023-06-22`, then verify `posts/imbalanced-learn-extra/index.qmd` is unchanged in `git status` and its md5 still matches the hash in `_freeze/posts/imbalanced-learn-extra/index/execute-results/html.json`.
- [x] 2.2 Rewrite `index.qmd` as a single posts listing with date, title, description and categories, newest first and without images, then verify the file has no publications or presentations listing. Superseded by 5.2 for layout.
- [x] 2.3 Add Posts and Publications links to the navbar in `_quarto.yml`, keeping search and the GitHub and LinkedIn icons, then verify all four appear in the rendered homepage HTML.

## 3. Theme

- [x] 3.1 Add `theme.scss` and configure light and dark themes in `_quarto.yml` with the system sans-serif stack, neutral palette and one accent colour, then verify the render succeeds and the mode toggle appears on every page.
- [x] 3.2 Calculate contrast for body text and links against the background in both modes, and adjust colours until every ratio is at least 4.5:1.

## 4. Integration checks

- [x] 4.1 Run `quarto render` with `.venv` active, then verify it succeeds, and that the post's frozen hash and stored output are unchanged.
- [x] 4.2 Check `_site/` after a clean render: the homepage lists only published posts, there are no `publications/<slug>/` or presentation pages, and `search.json` points to `publications.html` for publication titles.
- [x] 4.3 View the homepage, Publications page and post in both modes and at 375 pixels wide, then verify the equations, code, tables and images are legible and the page does not scroll sideways.
- [x] 4.4 Confirm no links on the homepage or Publications page point to removed pages, and that every Publications link goes to its recorded external address.

## 5. Landing page

- [x] 5.1 Vendor `particles.js` into `assets/` with its MIT licence, then verify the file came from the 2.0.0 package and contains no references to other sites beyond its licence and project links.
- [x] 5.2 Add a custom listing template and rewrite `index.qmd` with the hero and posts sections, then verify the newest post renders as the feature and that adding a newer scratch post moves the previous one into the list.
- [x] 5.3 Port Grayscale's masthead and featured-row styling into `theme.scss` with its licence notice, then verify the hero text and links reach at least 4.5:1 in both modes and that the top of the posts section is visible at 1200 by 800 pixels.
- [x] 5.4 Add the particle script with reduced-motion handling, then verify in a real browser that particles move by default, do not move under reduced motion, and that the hero remains intact when the script is blocked.
- [x] 5.5 Re-run the integration checks in group 4 against the final site, including 375 pixels wide, both modes, third-party requests, and the unchanged frozen post.

## 6. CV content and richer design

- [x] 6.1 Regenerate `publications.qmd` from `cv.md` with authors, venue and year under year headings, newest first, then verify with an independent parser that all 17 match the CV and the years run from 2026 to 2008.
- [x] 6.2 Add the roles, summary, highlights and open-source section to `index.qmd`, then verify each summary claim against `cv.md`, all 8 project links, and that the phone, email and client names are absent from the page.
- [x] 6.3 Add the glows, shimmer, glass panels, entrance and scroll animations, pointer effects and count-up, then verify in a browser that they run by default, stop under reduced motion, and leave all content visible and correct when scripts are blocked.
- [x] 6.4 Re-run the group 4 and 5.5 checks on the final site, including contrast in both modes, 375 pixels wide, no third-party requests on the homepage, posts visible at 1200 by 800 pixels, and the unchanged frozen post.

## 7. Section pages and navigation buttons

- [x] 7.1 Remove the hero buttons from `index.qmd`, rewrite the summary, and move the open-source section to a new `open-source.qmd`, then verify the homepage has no buttons, each summary claim traces to `cv.md`, and the page lists all 8 projects with correct repository links.
- [x] 7.2 Add a relevant SVG thumbnail for each project, then view them and verify all 8 load from the site with no broken images.
- [x] 7.3 Add `posts.qmd` and give the Posts, Open source and Publications pages the animated header, then verify each renders with its header, that the post appears on the Posts page, and that no section page appears in a listing.
- [x] 7.4 Add Posts, Open source and Publications to the navbar as pill buttons, then verify the links, hover and active states in both modes and at 375 pixels wide.
- [x] 7.5 Re-run all checks on the final site: reduced motion, blocked scripts, pixel contrast, publication data, private details, third-party requests, sideways scrolling and the unchanged frozen post.

## 8. Homepage quick view

- [x] 8.1 Remove the Intraday Trading entry from `publications.qmd`, then verify 16 entries remain, all match the CV, and all have links.
- [x] 8.2 Replace the homepage highlights and latest-post section with three cards, then verify each opens its page and that a newer scratch post updates the Posts card.
- [x] 8.3 Scale the hero to fit, then verify no scrolling at 1200x800, 1366x650, 1440x780, 1920x1000 and 375x812 in both modes.
- [x] 8.4 Re-run all checks on the final site: reduced motion, blocked scripts, card contrast, private details, third-party requests and the unchanged frozen post.

## 9. Separate hero, single colour scheme

- [x] 9.1 Move the cards out of the hero onto the page background and remove the date text from the Posts card, then verify no scrolling at all tested sizes, aligned desktop cards, and card text contrast of at least 4.5:1.
- [x] 9.2 Remove dark mode and the toggle, then verify no page has a colour-mode control or dark stylesheet, and that all pages keep their styling.
- [x] 9.3 Re-run the final checks: reduced motion, blocked scripts, third-party requests, broken images, sideways scrolling, button states and the unchanged frozen post.

## 10. Tags, justified text and title fix

- [x] 10.1 Make tags visible and clickable on the Posts page, homepage card and post pages, then verify in a browser with a scratch two-post site that each tag link filters correctly, All clears it, and all posts stay visible with scripts blocked.
- [x] 10.2 Justify paragraphs and list items site-wide with hyphenation, then verify computed alignment on all pages and that headings, roles and navigation are not justified.
- [x] 10.3 Fix the clipped descender in hero titles, then view the g in the name and verify the homepage still fits at all tested sizes.

## 11. Reliable publication links

- [x] 11.1 Replace publisher addresses with DOI links, confirming each DOI against the title, journal and year on Crossref, then verify every DOI resolves and every link responds to a real click.

## Workflow follow-up

- Review the rendered site, then deploy with the existing `docs/` step when ready.
- Archive the change after deployment.

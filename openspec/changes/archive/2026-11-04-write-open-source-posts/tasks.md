# Tasks

## 1. Writing checks and the shortened post

- [x] 1.1 Add a scratch script, outside the repository, that reports prose words, code example count and length, flagged phrases and em dashes for a post, then verify it flags a sample containing each listed phrase and passes a clean sample.
- [x] 1.2 Rewrite `posts/imbalanced-learn-extra/index.qmd` with plain code fences, keeping its address, image and date, and covering both algorithm families, then verify it passes the script, and run its examples in a scratch environment so any output shown is real.
- [x] 1.3 Remove the post's `requirements.txt`, stale frozen output and `.jupyter_cache`, then verify `quarto render` succeeds with neither package installed and the post loads no third-party scripts.

## 2. Post order

- [x] 2.1 Test `sort: ["date desc", "order"]` in a scratch site, apply it to `posts.qmd` and `index.qmd` if it works or use the design's fallback, then verify the same order appears on the Posts page and homepage card across two renders.

## 3. New posts

- [x] 3.1 Write the `scikit-complexity` post, run its example in a scratch environment, then verify it passes the script, each claim traces to the repository, and any output is real.
- [x] 3.2 Write the `veriphy` post from its README and source, then verify it passes the script, each claim traces to the repository, and the review report marks examples that were not run.
- [x] 3.3 Write the `sports-betting` post, run its example in a scratch environment, then verify it passes the script, each claim traces to the repository, and any output is real.
- [x] 3.4 Write the `scikit-ai` post, then verify it passes the script, each claim traces to the repository, and the review report marks examples needing credentials as not run.
- [x] 3.5 Write the `mathpix-mcp` post, then verify it passes the script, each claim traces to the repository, and the review report marks examples needing credentials as not run.
- [x] 3.6 Write the `copier-modern-python` post, run its command in a scratch folder, then verify it passes the script, each claim traces to the repository, and any output is real.
- [x] 3.7 Write the `agentorg` post from its README and source, then verify it passes the script and each claim traces to the repository.
- [x] 3.8 Give all eight posts a title, description, date, tags and image, then verify they render on the Posts page and that the Open Source tag shows all eight.

## 4. Open source page links

- [x] 4.1 Keep each card on `open-source.qmd` linked to its repository, then verify with real mouse clicks in a browser that each card opens the right repository, including at 375 pixels wide.

## 5. Integration checks

- [x] 5.1 Run `quarto render` with `.venv` active, then verify it succeeds, the homepage still fits at all tested sizes with the new Posts card, posts do not scroll sideways at 375 pixels wide, and there are no broken links or images.
- [x] 5.2 Produce a review report listing each post with its word count, sources checked, and examples run or not run, then give it to the owner with the rendered posts.
- [x] 5.3 Apply the owner's edits to the posts, then verify each edited post still passes the script and record the owner's approval of all eight.

## 6. Plain descriptions and code output

- [x] 6.1 Rewrite each post's subtitle and opening in the terms of the tool's repository or the CV, then verify each subtitle against its source.
- [x] 6.2 Add real output under every example that can run, then verify a fresh run produces identical text for the Python examples, `agentorg --help` and the Copier listing.

## Workflow follow-up

- Deploy with the existing `docs/` step when ready.
- Archive this change after deployment.

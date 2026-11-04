# Proposal

## Why

The Open source page lists eight tools but says little about any of them, and the site has only one published post. A short post per tool gives readers a reason to try each one and gives the blog its main content. The one existing post also runs long, and the site should have a clear standard for short posts that sound like a person wrote them.

## What Changes

- Write one short post for each of the eight open-source tools in the CV: `sports-betting`, `scikit-ai`, `scikit-complexity`, `veriphy`, `copier-modern-python`, `mathpix-mcp`, `agentorg`, and a shortened post for `imbalanced-learn-extra`.
- Shorten the `imbalanced-learn-extra` post. Its prose is only 237 words, so the cut falls on its 10 code cells and 225 lines.
- Set a standard for all posts: short, natural first-person writing, with every claim checked against the tool's repository.
- Show code as plain examples that do not run during site builds. Any output shown comes from a real run of the tool.

## Capabilities

### New Capabilities

- `post-writing-standard`: The length, voice and accuracy rules that every post on the site follows.
- `open-source-posts`: The set of posts covering the owner's open-source tools.

### Modified Capabilities

None. The existing specs cover the lecture-notes post and are unaffected.

## Impact

- **New content**: seven new folders under `posts/`, each with an `index.qmd`.
- **Changed content**: `posts/imbalanced-learn-extra/index.qmd`, which invalidates its frozen output. Without executable cells it no longer needs that output, its pinned packages, or the third-party scripts its output loads.
- **Site source**: the sort order in `posts.qmd` and `index.qmd`.
- **Homepage**: the Posts card will show a new post instead of `imbalanced-learn-extra`, and the Posts page tag filter gains more tags.
- **Dependencies**: none for the site. Checking examples uses temporary environments outside the repository.
- **Order of work**: this builds on the uncommitted `redesign-blog-theme` change, which should be reviewed first.

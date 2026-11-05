# Proposal

## Why

The Kaluza-Klein lecture notes hold about 1.46 million words that nobody can read, because they sit in an unpublished draft folder with no version history. Three subjects are in there: mathematics, theoretical physics, and Kaluza-Klein theory. Publishing them as three tutorials makes the site's largest body of work available.

## What Changes

- Publish three tutorials from the book, each appearing as a post on the Posts page:
  - **Fundamentals of Mathematics for Physicists**: the 13 chapters of Foundations of mathematics.
  - **Fundamentals of Theoretical Physics**: the 12 chapters of Foundations of physics.
  - **Kaluza-Klein Theories**: the preface, conventions and 8 chapters of the abelian, non-abelian, coset space and compactification parts.
- Publish all content in full, including exercises and interactive figures, with citations on each chapter and a glossary and verification registry per tutorial.
- Make each tutorial a self-contained post folder that renders with plain Quarto, with no central book or shared folder.
- Keep cross-references working, including from one tutorial to another.
- Publish the preface as written. The work continues in the open while the three `kk-csdr-*` changes remain unfinished.
- **BREAKING**: exempt tutorials from the short-post limits of the post writing standard. Normal posts stay short.

## Capabilities

### New Capabilities

- `physics-tutorials`: The three published tutorials, how readers reach and move through them, and what the repository keeps.

### Modified Capabilities

- `post-writing-standard`: The length limits now apply to normal posts and not to tutorials.
- `landing-page`: The homepage Posts card summarises all posts instead of showing the newest.
- `post-listing`: Every post appears as an equal list entry with a small thumbnail, with no featured post, and tags no longer appear on the homepage Posts card.
- `open-source-posts`: The deliberate post order now starts with the Kaluza-Klein tutorial, and projects are list entries rather than cards.
- `open-source-projects`: Projects appear as a list with small thumbnails, in the same form as the Posts page.

## Impact

- **New content**: three post folders holding about 1.3 million words of chapters, their figures, recorded Sage outputs, Lean statements, filters, bibliography and styles.
- **Site source**: three sidebars in `_quarto.yml`, three thumbnails, and renumbered `order` values on normal posts.
- **Draft folder**: replaced. Its Lean library, Sage scripts, manifests and checkers move into the post that owns them, and two figures are converted to PNG.
- **Repository size**: chapters, figures and rendered output in `docs/` add several hundred MB. Each rebuild that changes pages adds to permanent history.
- **Render time**: a full `quarto render` takes about 12 minutes or more. `quarto preview` only re-renders what is being edited.
- **Homepage**: the Posts card summarises the posts in general, with a combined thumbnail, instead of showing the latest post. All three cards share one form and show no counts.
- **Third-party requests**: chapter pages load KaTeX from a CDN to display equations, and the Open source page requests star counts from the GitHub API.

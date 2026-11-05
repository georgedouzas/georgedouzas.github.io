# Design

## Context

See proposal.md for motivation. The source is one Quarto book with 35 chapter files in six parts, kept in the ignored draft folder `posts/_kaluza...`. Chapters depend only on their figures, interactive figures, recorded Sage outputs, extracted Lean statements, a bibliography and three small shortcodes. Quarto resolves cross-references across pages only inside a book, and the parts reference each other often: physics to mathematics 378 times, Kaluza-Klein to mathematics 181 and to physics 82.

## Goals / Non-Goals

**Goals:**

- Each tutorial lives entirely in its own post folder, like every other post, and renders with plain `quarto render` or `quarto preview`.
- Cross-references keep working, including from one tutorial to another.

**Non-Goals:**

- Finishing the three `kk-csdr-*` changes, or changing chapter content.
- Running the Lean build or verification checks, or changing what the preface claims.
- A PDF edition.

## Decisions

**Each tutorial is a normal post folder.** `posts/<slug>/` holds the landing `index.qmd`, its chapter `.qmd` files, `figures/`, `sage-exercises/`, `lean-statements.json`, its shortcodes, `references.bib`, `styles.css` and `_metadata.yml`, which applies those to every page in the folder. Nothing outside the folder is needed except the site's theme. A central book, a `tutorials/` folder and a copy step were tried and rejected by the owner.

**References to other pages become ordinary links.** Each post has `cross-references.lua` and `cross-references.json`. Before Quarto resolves references, the filter replaces a reference to a label on another page with a link to that page, such as "Definition 5 of Group theory", including across tutorials. References within a page stay native Quarto references. The JSON is committed data, so no build step is needed to preview or render.

**Numbering restarts on each page.** Outside a book, blocks are numbered per page, as "Definition 5" and not "Definition 3.5". Links between pages therefore name the chapter.

**The label index comes from a rendered pass.** Rendering once and reading the labels from the output means link text always matches what the page shows. The second render uses the index. A one-off script does this at migration, and it fails if any defined label is missing.

**Each tutorial has its own registry and glossary.** They are split by chapter from the book's versions. Bibliography entries appear at the bottom of each chapter that cites them, so no separate references page is needed.

**Sidebars are declared in `_quarto.yml`.** Quarto only reads sidebars from the project configuration, so this site-level navigation is the one thing outside the post folders. It holds chapter order only, not content.

**Each post keeps its own tooling and verification work.** The Lean package, certificates, Sage scripts, verification manifests and checkers move into the post that owns them, replacing the draft folder. Lean packages depend one way: physics and Kaluza-Klein on mathematics. `lake build` compiles each package's root module, which imports the modules the book's build compiled, and modules outside it are unfinished. PDFs and earlier chapter versions stay out of git.

**References live with the tutorial that uses them.** Each tutorial keeps the PDFs and reading notes for its chapters in `posts/<slug>/_references/`, placed by citation and by the chapters each note names. A file used by several tutorials is hard-linked into each. These folders are ignored by git because the PDFs are copyrighted, and Quarto skips them because of the leading underscore.

**Keep KaTeX on a CDN, pinned to a version.** Pinning prevents a future release from changing how the equations render.

**Replace the two oversized figures with PNGs.** The 68 MB and 33 MB SVGs are too heavy for a browser. Their Sage scripts now write PNGs, and re-running them also re-ran their `[PASS]` checks.

**Post order puts Kaluza-Klein first.** Normal posts' `order` values move up by three, so the three tutorials lead the Posts page. The homepage's Posts card summarises all posts instead of featuring one.

## Risks / Trade-offs

- [The preface says every displayed computational result is a Lean theorem, while the published registries list many chapters as not yet formalised, and count oracle tests, which are not proofs] → The owner decided to keep the preface as written, before these figures were known. It is raised again in the review report.
- [`quarto render` takes as long as rendering 1.46 million words, about 12 minutes or more, on every full site build] → Accepted as the cost of keeping posts self-contained. `quarto preview` only re-renders pages being edited.
- [Numbers change from chapter-prefixed to per-page] → Any text hard-coding a number such as "Theorem 3.2" would be wrong. Text uses references, and the source is searched for hard-coded forms.
- [Prose mentions such as "Part I" now span tutorials] → Left as written, since the parts remain a meaningful order. The owner can revise them.
- [The site's search index grows greatly] → Its size is measured, and search is excluded from chapters if it is too large to load.
- [Repository growth from chapters, figures and rendered output in `docs/`] → Accepted by the owner, and reported before pushing. History growth cannot be undone without a rewrite, so the push is the last step.
- [The Lean split has not been fully built, as the owner stopped the build to save time] → The modules are byte-identical to the book's, and building each package remains a follow-up. Three unfinished mathematics modules failed before it stopped.

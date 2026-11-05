# Design

## Context

See proposal.md for motivation. Three ignored draft folders are the source:

- `posts/_ml-tutorials` has six chapter files, an `outline/` folder listing source books per section, and `references/` with six textbook conversions in Markdown. It has 87 labels and no bibliography, and it cites five keys. Its chapters refer to each other, including from Supervised Learning and others to the mathematics chapter, and from the appendices back to the machine learning chapters.
- `posts/_reinforcement-learning` is a 3,400-word post on reinforcement learning and LLM-based agents, with 7 Python cells, 7 images and a one-entry bibliography.
- `posts/_mathematical-foundations-of-reinforcement-learning` is a 240-word summary marked as a draft, with no code cells and one image.

The physics tutorials set the layout: self-contained post folders, a site-level sidebar, and a pre-render filter that turns references to other pages into links.

## Goals / Non-Goals

**Goals:**

- Match the physics tutorials' layout and cross-reference behaviour.
- Keep the cost of a normal edit low, using `quarto preview` on the page being worked on.

**Non-Goals:**

- Rewriting chapters, resolving overlap between chapters, or writing an introduction chapter from `outline/introduction.qmd`.
- Lean or Sage verification, a verification registry, or a glossary.
- Deleting the draft folders.

## Decisions

**The two named reinforcement learning sources are the two draft folders.** Their names match the request exactly. They become their own chapters after the Reinforcement Learning chapter, so all three are in Fundamentals of Machine Learning. Merging them into that chapter was rejected because it would rewrite the owner's text.

**One page per source file.** This matches the existing tutorials, where a file is a chapter. The mathematics tutorial therefore has two long pages with a table of contents, rather than splitting sections into many short pages that would change labels and numbering.

**Each post gets its own filter and label index, covering only these two tutorials.** Some labels, such as `sec-linear-algebra`, also exist in the physics tutorials. A separate index avoids collisions without renaming any label. The physics index stays unchanged, and the first render pass builds this one, as with the physics tutorials.

**No registry or glossary.** The tutorial layout's registry reports verification, which these notes do not have, and they have no glossary source. Generating a glossary would be new content. The sidebar lists the overview and chapters only.

**Create a bibliography per tutorial.** It holds the six books the outlines name: Shalev-Shwartz and Ben-David, Theodoridis, Bishop, Deisenroth and colleagues, Zhao, and James and colleagues, using the five keys the chapters already cite. The Zhao entry in the agents post uses a different key (`Zhao_2025`), and that post's citations move to `zhao2025rl`. Each entry's details come from the book's own front matter in `references/`.

**Add citations from the outlines.** Each outline names the books a section draws on, so these are the owner's recorded sources, unlike the physics notes' table-of-contents mappings. A section whose outline names a book it does not cite gets a "For more on this topic, see" pointer. Sections with no outline entry stay as they are, and are listed for the owner.

**Keep textbook conversions beside the tutorials and out of git.** They move to `posts/<tutorial>/_references/textbooks/`, placed by which tutorial's outline names them. A new ignore rule covers that folder, because the current rule only covers PDFs. Quarto skips underscore folders, so they stay off the site. `outline/` moves to `_references/outline/` and is tracked, as it holds the owner's own planning.

**Run the code once and keep its output.** The agents chapter's 7 Python cells use the site's existing `freeze: auto`, so they re-run only when that chapter changes. NumPy goes in the post's `requirements.txt`. A cell failing to run stops the task, and it is not worked around by pasting output.

**Order after the physics tutorials.** The two take `order` values 4 and 5, and normal posts move up by two. The physics tutorials keep their lead because the homepage and earlier decisions assume it.

**Update the homepage card text only.** "On mathematics, physics and my open-source tools" becomes "On mathematics, physics, machine learning and my open-source tools", provided it still fits at all tested sizes.

## Risks / Trade-offs

- [The Mathematical Foundations of Reinforcement Learning chapter is a 240-word summary marked as a draft, and overlaps the longer chapter] → Published as requested, and raised in the review report so the owner can extend or remove it.
- [The agents post dates from 2025 and its LLM content may be out of date] → Published unchanged, and raised in the review report.
- [The physics change's Tutorial tag scenario says "exactly the three tutorials"] → This change depends on that one. The scenario is updated to say all tutorials before either is archived.
- [Hard-coded labels such as "Lemma A.1" are not numbered by Quarto] → They stay as written, since they are plain text and consistent within the appendix.
- [The two tutorials refer to each other in both directions, so neither can resolve fully alone] → Both render together in one pass, then again with the index, as the physics tutorials do.
- [Chapters may lean closely on the source books] → Out of scope for this change. The owner keeps responsibility for what is published.

## Migration Plan

1. Finish, commit and deploy the pending physics tutorial work first.
2. Render and verify locally, then give the owner the review report.
3. Commit, deploy to `docs/` and push only when the owner asks. Reverting those commits restores the site.

## Open Questions

None.

# Proposal

## Why

The machine learning notes in `posts/_ml-tutorials` hold about 54,000 words that are unpublished, and the two reinforcement learning drafts beside them are hidden as well. Publishing them as two tutorials, in the same layout as the physics tutorials, makes this work available and gives the site's machine learning readers a clear starting point.

## What Changes

- Publish **Fundamentals of Machine Learning** with these chapters:
  - Foundations of learning theory, supervised learning, unsupervised learning and reinforcement learning, from `posts/_ml-tutorials`.
  - Reinforcement Learning and LLM-based Agents, from `posts/_reinforcement-learning`.
  - Mathematical Foundations of Reinforcement Learning, from `posts/_mathematical-foundations-of-reinforcement-learning`.
- Publish **Fundamentals of Mathematics for Machine Learning** with two chapters: mathematical and statistical foundations, and appendices and technical background.
- Use the existing tutorial layout: a landing page with a chapter list, a sidebar, previous and next links, one post folder per tutorial holding everything it needs, a thumbnail, and references at the end of each chapter.
- Keep cross-references working between the two tutorials, which refer to each other in both directions.
- Create the bibliography these chapters lack, and cite the source books each section's outline names.
- Replace the 6 places that call the work a "thesis", since it is now a tutorial. Other chapter text stays as written.
- Place the two tutorials after the physics tutorials in the post order, and mention machine learning in the homepage Posts card.

## Capabilities

### New Capabilities

- `ml-tutorials`: The two published machine learning tutorials, their contents, and how readers reach and move through them.

### Modified Capabilities

None. Tutorials are already exempt from the short-post limits under the pending `publish-physics-tutorials` change, and the homepage and post list requirements hold as written.

## Impact

- **New content**: two post folders, about 58,000 words of chapters, with images and a bibliography.
- **Site source**: two sidebars in `_quarto.yml`, two thumbnails, the homepage Posts card text, renumbered `order` values on normal posts, and a new ignore rule.
- **Build**: one chapter runs Python code, so rendering it needs Jupyter and NumPy. A full render takes somewhat longer than today's 8 to 10 minutes.
- **Copyright**: `posts/_ml-tutorials/references` holds full-text conversions of six textbooks. They move beside the tutorials but stay out of git and off the site.
- **Dependency**: this builds on `publish-physics-tutorials`, including its artifact split, which touches the same files and should be finished first.

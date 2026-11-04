# Tasks

## 1. Undo the central approach

- [x] 1.1 Return the book to its original draft folder, remove the book-specific build changes, and remove `tutorials/` and the site's render exclusion and copy step, then verify the draft folder is ignored by git and the site has no reference to `tutorials/`.

## 2. Create the self-contained posts

- [x] 2.1 Run the migration into the three post folders, then verify each chapter appears exactly once, with 13, 12 and 8 plus the preface and conventions.
- [x] 2.2 Declare the three sidebars and update the landing pages to link to their own chapters, registry and glossary, then verify every landing link resolves.
- [x] 2.3 Render once and build the label index, then verify no label defined in a chapter is missing from it.
- [x] 2.4 Render again with the draft folder moved aside, then verify the render succeeds with no unresolved references.

## 3. Verify the tutorials

- [x] 3.1 Check every internal link and anchor in the rendered site, then verify none is broken.
- [x] 3.2 Follow a reference from a physics chapter to a mathematics definition with real mouse clicks, then verify it opens that definition with the mathematics sidebar.
- [x] 3.3 View chapters in a browser, then verify exercises, solutions, interactive figures, equations and citations work, and no page scrolls sideways at 375 pixels wide.
- [x] 3.4 Compare the preface with its source, search the output for private contact details, and scan the new post folders for credentials, then verify the wording is identical and nothing is found.
- [x] 3.5 Measure the search index and rendered size, and exclude chapters from search if the index is too large, then verify site search still works.
- [x] 3.6 Set the tutorial order, then verify the Posts page shows Kaluza-Klein first, that the homepage Posts card summarises all posts, that the Tutorial tag shows exactly three, and that the homepage still fits at all tested sizes.
- [x] 3.7 Run the writing checker on all posts, then verify normal posts still pass and that landing pages pass its language checks.

## 4. Review

- [x] 4.1 Produce a review report with page counts, render time, repository growth, registry counts and anything unresolved, then give it to the owner before pushing.

## Workflow follow-up

- Deploy to `docs/`, commit and push after the owner approves the report.
- Archive this change after deployment.

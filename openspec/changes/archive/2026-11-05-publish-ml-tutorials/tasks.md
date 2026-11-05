# Tasks

## 1. Create the post folders

- [x] 1.1 Create `posts/fundamentals-of-machine-learning/` and `posts/fundamentals-of-mathematics-for-machine-learning/`, copy the chapters with their images into them, then verify six and two chapters are present and each appears exactly once.
- [x] 1.2 Move the textbook conversions and outlines to each tutorial's `_references/`, add the ignore rule, then verify with `git status` and `git check-ignore` that no textbook conversion would be tracked and the outlines would be.
- [x] 1.3 Replace the 6 "thesis" mentions, then verify by comparing each chapter with its draft that nothing else changed and that the whole word no longer appears.

## 2. Bibliography and citations

- [x] 2.1 Write each tutorial's `references.bib` from the books' front matter and move `Zhao_2025` citations to `zhao2025rl`, then verify a render reports no missing citation.
- [x] 2.2 Add "For more on this topic, see" pointers where an outline names a book its section does not cite, then verify every chapter cites at least one work and list any section left with no source.

## 3. Layout and navigation

- [x] 3.1 Add the shared metadata, styles, filter and sidebars, and write both landing pages with plain descriptions and chapter lists, then verify every landing link opens the right page.
- [x] 3.2 Draw two thumbnails in the existing style, then view them in the Posts list and verify they load and fit.
- [x] 3.3 Render once, build the label index, and render again, then verify no label is missing from the index and no reference is unresolved.
- [x] 3.4 Set up the Python environment for the agents chapter and render it, then verify all 7 cells ran with no errors and that their output appears on the page.
- [x] 3.5 Set the post order and update the homepage card text, then verify the Posts order, and that the homepage fits at all eight tested sizes.
- [x] 3.6 Update the physics change's Tutorial tag scenario to cover all tutorials, then verify `openspec validate --strict` passes for both changes.

## 4. Verify

- [x] 4.1 Run a full render with the draft folders moved aside, then verify it succeeds with no warnings and restore the folders.
- [x] 4.2 Check every internal link and anchor, then verify none is broken apart from the JavaScript tag filter links.
- [x] 4.3 Follow references in both directions between the tutorials with real mouse clicks, then verify each opens the right place with that tutorial's sidebar, clear of the navbar.
- [x] 4.4 View chapters in a browser, then verify equations, images and code output display, each chapter ends with References, no Verification fold appears, and no page scrolls sideways at 375 pixels wide.
- [x] 4.5 Search tracked files and rendered output for textbook conversions, private contact details and credentials, then verify none is found.
- [x] 4.6 Filter by the Tutorial tag and run the writing checker on all posts, then verify all five tutorials appear, normal posts still pass, and the landing pages pass its language checks.

## 5. Review

- [x] 5.1 Produce a review report with page counts, render time, repository growth, sections with no source, and the concerns about the two short or dated chapters, then give it to the owner before pushing.

## Workflow follow-up

- Deploy to `docs/`, commit and push after the owner asks.
- Archive this change after `publish-physics-tutorials`, then deployment.

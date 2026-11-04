# Proposal

## Why

Design D32 makes the textbooks in `references/textbooks/` sources of content, examples and exercises, each mapped to
the chapters it serves in its reading note. The chapters written or extended after that rule use them; the chapters
written before it do not. The largest gap is Nath's *Supersymmetry, Supergravity, and Unification*, mapped to the
supersymmetry and gauge-theory chapters and cited in neither.

## What Changes

- Each older chapter gains content, examples and exercises from the textbook sections mapped to it, cited at the point
  of use with problem numbers, with the reading notes updated.
- New statements are added where a textbook covers a topic the chapter lacks, each with at least three examples and
  three exercises.
- Existing content, identifiers and figures are kept; errors found are corrected and reported.

## Capabilities

### Modified Capabilities

- `tutorial-content`: the textbook rule is applied to every chapter, not only to the new ones.

## Impact

The chapters listed in the tasks, `references/notes/`, `references.bib`, `figures/`, `sage/exercises/`. No
verification work (paused); new blocks carry placeholder Verification boxes.

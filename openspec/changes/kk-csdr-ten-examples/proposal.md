# Proposal

## Why

The book now has 75 textbooks from the author's library, most of them rich in worked problems. The rule of at least
three examples and three exercises per statement leaves most of that material unused. The author raises the minimum
to ten examples and ten exercises per definition, theorem and proposition.

At the time of the spec the book has 1,051 statements, with 3,291 examples and 3,548 exercises. Reaching ten and ten
everywhere needs about 7,219 new examples and 6,963 new exercises.

## What Changes

- **New minimum:** every `def-`, `thm-`, `prp-`, `lem-` and `cor-` block is followed by at least ten examples, of
  increasing complexity, and at least ten exercises, of increasing difficulty, each exercise with its solution. This
  replaces the minimum of three.
- **New content:** the rule applies from now on, including the change kk-csdr-physics-foundations and the new
  electromagnetism chapter.
- **Existing chapters:** they are brought up to ten and ten one part at a time, in the order of the tasks. The
  checkers enforce ten for a chapter once it is listed as raised, and three until then, so the build stays green
  throughout and the backfill can stop at any part boundary.
- **Sources:** new examples and exercises come from the textbooks mapped to each chapter, under the consistency rule
  (rewritten in the book's voice, not copied), with the reading notes updated.

## Capabilities

### Modified Capabilities

- `tutorial-content`: the minimum number of examples and exercises per statement is raised from three to ten, and
  existing chapters are raised one at a time.

## Impact

- **Changed files:**
  - every `tutorial/part-*.qmd`, over the course of the backfill;
  - `scripts/check_format.py` and `scripts/check_definitions.py` (the minimum becomes per chapter);
  - a new list of raised chapters;
  - `references/notes/`, `sage/exercises/` and `figures/`.
- **Verification:** none (paused). New blocks carry placeholder Verification boxes, and plain Sage serves figures and
  about 40% of the new exercise solutions.
- **Cost:** about 14,000 new blocks. The backfill is split into one task per chapter, so the author can stop it at
  any point.

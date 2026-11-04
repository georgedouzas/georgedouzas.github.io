# Design

## Context

Blocks are Quarto divs (`::: {#def-…}`, `#thm-`, `#prp-`, `#lem-`, `#cor-`, `#exm-`, `#exr-`) at fence length three,
followed by collapsed callouts (`.solution`, `.verification`). Titles are coloured by kind (red, green, blue). The
proof of concept on Sets is on branch `poc-folded-sets` (commits 53ef4d9 and 4b0e206).

## Decisions

### P1. Statement boxes

Each statement div and the Verification callout that follows it are wrapped in `:::: {.statement-box}`. The box has a
light red background (`#fbeceb`, a dark red `#3a2321` in dark mode), rounded corners and no vertical bar; inside it
the Verification title is red. Examples and exercises are never put in such a box.

### P2. Folded examples and exercises

The examples that follow a statement, from the prose introducing the first of them to the Verification box of the
last, are wrapped in `:::: {.callout-note .examples-fold collapse="true" title="Examples"}` with a green title; the
exercises, from the prose introducing the first to the Verification box of the last, in an `.exercises-fold` callout
titled **Exercises** with a blue title. Solutions and Verification boxes stay separately collapsible inside. A run
ends at the next statement, heading or figure outside the run. Cross-references to folded blocks keep working
because the identifiers are unchanged.

### P3. One script, applied to every chapter

`scripts/fold_blocks.py` (from the proof of concept) applies P1 and P2 to a chapter idempotently (it skips blocks
already wrapped). It runs once on every chapter; new chapters are written in the folded form or folded by the script
before commit.

### P4. Checkers and PDF

`check_format.py`, `check_definitions.py` and `check_exercises.py` read through the wrappers: the outer fences are
ignored when blocks, prose and solutions are located. In the PDF, folding is not available: the Examples and Exercises
boxes are printed expanded with their titles as small coloured headings, and statement boxes as `tcolorbox`
environments with the same light red.

## Risks

- Readers who print the HTML see collapsed content; the PDF covers printing.
- Agents writing new content must emit the wrappers; the brief and `check_format.py` enforce them.

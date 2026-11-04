# Proposal

## Why

The chapters are long: every statement is followed by at least three examples and three exercises, each with a
solution and a Verification box, so the definitions and theorems that carry the line of the argument are hard to find
while reading. A proof of concept on the Sets chapter (branch `poc-folded-sets`) folded each statement's examples and
exercises into collapsible boxes and put each definition in a light red box; the author approved it and asked that all
statements share the style.

## What Changes

- Every statement block (definition, theorem, proposition, lemma, corollary) is shown in a light red box together
  with its Verification box, whose title is red like the statement's.
- After each statement, its examples (with their connecting prose and Verification boxes) are folded into one
  collapsed green **Examples** box, and its exercises (with their prose, solutions and Verification boxes) into one
  collapsed blue **Exercises** box; the integrative exercises of a part and of a chapter are folded the same way.
- The change applies to every chapter, by a script, with the checkers and the PDF output adapted.

## Capabilities

### Modified Capabilities

- `tutorial-content`: the colour coding of block kinds (statements now have a tinted background), and the folding of
  examples and exercises.

## Impact

`tutorial/*.qmd` (wrapping fences only, no change to the content), `tutorial/styles.css`, the PDF header in
`tutorial/_quarto.yml`, `scripts/check_format.py` and `check_definitions.py` (which must see through the wrappers), the
legend in the conventions chapter.

# Tasks

- [x] 1. Write `scripts/fold_blocks.py` from the proof of concept (statement boxes for every statement kind, Examples and Exercises folds, idempotent); verify it reproduces the Sets result of branch `poc-folded-sets` and extends the box to theorems, propositions, lemmas and corollaries.
- [x] 2. Make `check_format.py`, `check_definitions.py` and `check_exercises.py` read through the wrappers, and make `check_format.py` require them (a statement outside a box, an example or exercise outside a fold); verify the checkers on the folded Sets chapter and on an unfolded one.
- [x] 3. Move the CSS of the proof of concept to `tutorial/styles.css` under the name `.statement-box` (light and dark mode), and add the PDF form (tcolorbox statement boxes, expanded folds with coloured headings) to `tutorial/_quarto.yml`; verify HTML and PDF renders of Sets.
- [x] 4. Update the legend in the conventions chapter (the statement box and the folds); verify a render.
- [x] 5. Apply the script to every chapter; verify every checker reports 0 problems and a full draft render has no unresolved references.
- [x] 6. Update the writing brief so new content is written in the folded form; delete the branch `poc-folded-sets` once merged in substance.

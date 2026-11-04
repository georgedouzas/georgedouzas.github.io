# Design

## Context

The decisions that govern this work were taken in `kk-csdr-tutorial` and stay in force (see its archived design):
D28 (Veriphy verifies every claim, physics as explicit Lean models), D29 (verification shown only in collapsed
boxes with a Veriphy part splitting into Lean and Sage, never in the main text; levels certificate, kernel-checked
test, oracle test), D30 (missing building blocks go into Veriphy itself, on a branch, then a release). Veriphy 0.3.1 is
pinned; its `sage_test` fallback supplies the test levels.

## Decisions

### V1. One chapter at a time

Certification runs one chapter at a time with a single agent and no helper agents, resumed rather than restarted after
an interruption (cost; author's instruction). Lean runs go through `scripts/lean_lock.py` (at most three Lean
processes, a memory guard). The brief and the pitfalls learned so far are recorded in `verification/pilot-report.md`
and `verification/veriphy-gaps.md`.

### V2. Every claim, including asides

Every claim of every block is in the manifest, including parenthetical claims; nothing is left unformalised (a
statement without a mathlib proof becomes a `def … : Prop` that is tested). Tests prefer kernel instances proved by
computation over re-running the general argument.

### V3. Gate

The change is complete when every chapter passes `check_verification.py --strict` and `check_format.py --strict` (no
pending part), the library builds as a whole with only the standard axioms and with Sage absent, and the integration
checks pass.

## Risks

- Cost: about 0.5-1 million agent tokens per chapter at the pace so far.
- Oracle tests dominate the analysis chapters until Veriphy gains recipes for limits, improper integrals, residues and
  series values.

### V4. Every definition has a Verification box: the Lean definition matches the text

Author's instruction (October 2026). A definition is verified too, not only the facts some definitions state. Every
definition block has at least one claim, its **correspondence claim**: the book's wording of the definition, stated
literally in Lean as a `def … : Prop` (or a structure) that follows the text clause by clause, is equivalent to the
Lean or mathlib notion the rest of the library uses. For example the book's three axioms of a topology, written as a
predicate on a family of subsets, hold exactly when the family is a mathlib `TopologicalSpace`; the book's "map with
an inverse on both sides" is equivalent to `Function.Bijective`. The correspondence claim is a theorem (an `iff` or an
equivalence) and is certified or tested like any claim (D29 levels); where the book defines a notion mathlib lacks,
the Lean definition is the book's, and the correspondence claim is replaced by a sanity theorem on the chapter's first
example (the example satisfies the definition). The box's Lean part shows the definition and the correspondence
theorem. The earlier rule that a definition without a stated fact has no box is withdrawn; `none` entries remain only
for blocks that are not definitions and state nothing.


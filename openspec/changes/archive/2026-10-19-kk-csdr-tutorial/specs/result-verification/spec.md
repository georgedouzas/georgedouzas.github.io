# Spec Delta

## Purpose

How the verification of the book's claims is shown to the reader: in collapsed boxes after each block, with levels,
and never in the main text.

## ADDED Requirements

### Requirement: Verification is shown in a hidden callout
No Lean statement SHALL be visible by default. Every block with claims SHALL be followed by a collapsed "Verification"
callout, styled like the solution callouts, consisting of a Veriphy part (recipe and certificate) that decomposes into a
Lean part (the kernel-checked statements) and a Sage part (the oracle computation), always all three (design D29).

#### Scenario: Reading a proposition
- **WHEN** a reader opens a page
- **THEN** the proposition shows no Lean code, and a collapsed "Verification" callout below it expands to the Lean statement, the recipe name and the Sage computation

#### Scenario: The shape of a Verification callout
- **WHEN** a reader expands any Verification callout
- **THEN** it shows a **Veriphy** part naming the recipe and certificate, followed by its **Lean** part (the kernel-checked statements) and its **Sage** part (the oracle computation), in that order, and `scripts/check_format.py` reports a callout missing a part

#### Scenario: The gate
- **WHEN** `scripts/check_format.py --strict` runs at the gate
- **THEN** any Verification part still marked "Pending" is reported

#### Scenario: A stray Lean shortcode
- **WHEN** a chapter contains a `lean` shortcode outside a verification callout
- **THEN** `scripts/check_format.py` reports it

### Requirement: Verification levels
Every Veriphy part SHALL name its level: a recipe and certificate (level `certificate`), a kernel-checked test with its
instance count and script (level `kernel-test`), or an oracle test with its instance count and record (level
`oracle-test`). A test SHALL be used only when no recipe applies, and SHALL be replaced by a certificate when a recipe
for the claim's shape exists. `scripts/check_format.py --strict` SHALL accept the three levels and reject `Pending`.

#### Scenario: A general statement
- **WHEN** a block states rank-nullity for an arbitrary matrix
- **THEN** its Veriphy part reads "Test (kernel-checked, n instances)" with the test script, and its Lean part shows
  the mathlib proof

#### Scenario: A computation
- **WHEN** a block states the dimension of the span of two explicit vectors
- **THEN** its Veriphy part names the `dimension` recipe and its certificate, not a test

#### Scenario: The registry
- **WHEN** the registry appendix is rendered
- **THEN** it reports how many claims are at each level

### Requirement: The main text never refers to verification
Outside the Verification callouts the book SHALL state only mathematics and physics: no Lean names, no mention of Lean,
Veriphy, recipes, certificates, mathlib or the registry, no "checked by exact computation" or "checked in Sage" and no
references to verification scripts. Exercise solutions MAY show Sage computations as teaching material, and figure
captions MAY name their generating script. Chapters SHALL NOT end with a verification summary; the registry appendix
lists every claim.

#### Scenario: A proposition
- **WHEN** a reader reads a proposition block
- **THEN** it states the result only, and its Lean statement, recipe and Sage computation appear only in the collapsed Verification callout below it

#### Scenario: A stray reference
- **WHEN** a block or the connecting prose names a Lean theorem or says that a result was verified
- **THEN** `scripts/check_format.py` reports it

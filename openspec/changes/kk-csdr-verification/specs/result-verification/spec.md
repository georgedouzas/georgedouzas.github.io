# Spec Delta

## Purpose

Guarantees that every claim of the book is backed by a machine-checked Lean theorem, certified or tested through
Veriphy and built without Sage, and that the correspondence between blocks, claims and theorems is explicit and
auditable.

## ADDED Requirements

### Requirement: Every displayed computational result has a Lean theorem
Every displayed equation that states a computed fact (a component, an identity, a spectrum, a group, a minimum)
SHALL correspond to a Lean theorem whose statement expresses that fact under the tutorial's conventions. The
tutorial SHALL display the theorem identifier next to the result.

#### Scenario: Reader traces a result to its theorem
- **WHEN** a reader selects the identifier shown next to a displayed result
- **THEN** a theorem with that identifier exists in the project's Lean library and its statement matches the displayed equation

#### Scenario: A result without a theorem
- **WHEN** a displayed computational result has no theorem identifier and no non-formal verification label
- **THEN** the build's registry check fails and names the result

### Requirement: The Lean library builds completely
The Lean library SHALL build with zero `sorry`, and every theorem SHALL depend only on the three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).

#### Scenario: Axiom audit
- **WHEN** the axioms of every theorem referenced by the tutorial are listed
- **THEN** no axiom beyond the standard three appears and no `sorryAx` appears

### Requirement: Committed proofs are Sage-free
No committed Lean file SHALL contain a live certificate call. Committed proofs SHALL be the Sage-free replacements
produced during authoring, so the library builds on a machine without SageMath.

#### Scenario: Build without Sage
- **WHEN** the Lean library is built with the Sage executable unavailable
- **THEN** the build succeeds

#### Scenario: Live call left in a committed file
- **WHEN** a committed Lean file contains a live certificate tactic call
- **THEN** the pre-render check fails and names the file and line

### Requirement: A verification registry records every result
The book SHALL keep one manifest `verification/<chapter>.yaml` per chapter listing every block, its claims, the Lean
declaration of each and its verification level (design D28-D29), checked by `scripts/check_verification.py`; the
registry appendix SHALL be generated from the manifests. (The former `claims.yaml` registry and `check_claims.py` are
retired.)

#### Scenario: Registry entry references a missing theorem
- **WHEN** a registry entry at level `lean` names a theorem that the library does not declare
- **THEN** validation fails and names the entry

#### Scenario: Document result missing from registry
- **WHEN** a labelled result in the document has no registry entry
- **THEN** validation fails and names the label

### Requirement: Statements are tested numerically before being proved
Each Lean statement SHALL be tested against an independent numerical or exact evaluation before a proof is
attempted, and a statement that fails the test SHALL be reported as suspect and corrected in the tutorial text
rather than forced through.

#### Scenario: A displayed formula is wrong
- **WHEN** the numerical test of a formalised statement disagrees with the formula in the tutorial text
- **THEN** the text is corrected to the verified form and the discrepancy is recorded in the registry's notes

### Requirement: Certificates are reproducible
Every certificate script used to produce a committed proof SHALL be stored in the project with a header stating
its technique and goal shape, and re-running it SHALL reproduce the committed proof.

#### Scenario: Certificate re-run
- **WHEN** a stored certificate script is re-run against its theorem
- **THEN** the emitted Sage-free replacement equals the committed proof

### Requirement: Veriphy is used wherever it applies

Every `lean`-level result SHALL be proved with the Veriphy recipe that fits it when one applies, with the
certificate committed and the recipe named in the result's block; every definition with a formal counterpart SHALL
display its Lean definition; computational exercises SHALL carry a verified Lean statement in their solution; the
registry SHALL record the recipe used for each result.

#### Scenario: Result with an applicable recipe
- **WHEN** a result is a polynomial, matrix, Lie-algebraic, derivative or coordinate-geometry identity
- **THEN** its proof uses the corresponding Veriphy recipe, the block names the recipe, and the certificate is
  under `.veriphy/certificates/`

#### Scenario: No recipe applies
- **WHEN** a result is outside the recipes (an inequality by `nlinarith`, a periodicity by mathlib analysis)
- **THEN** the proof uses mathlib tactics and the registry records "no recipe" with the reason

#### Scenario: Registry summary
- **WHEN** the registry appendix is rendered
- **THEN** it lists, per recipe, the number of results it verified

### Requirement: Veriphy verifies every claim of every block
Every mathematical claim stated in a definition, theorem, proposition, lemma, corollary, example or exercise block SHALL
be a Lean theorem of the library whose computational core is proved by a Veriphy certificate, with mathlib supplying the
remaining steps (design D28). Physical statements SHALL be proved inside an explicit mathematical model formalised in
Lean, not labelled as unproved. The registry level `text` SHALL be used only for historical and bibliographic remarks.

#### Scenario: An example states a value
- **WHEN** an example block states a number, an identity, an inequality or a count
- **THEN** the registry lists a Lean theorem for it with the Veriphy recipe and the committed certificate script, and the theorem builds without Sage

#### Scenario: An exercise has a Sage solution
- **WHEN** an exercise's solution is a Sage script
- **THEN** the script's computation is the oracle of a Veriphy certificate script for the Lean theorem stating the exercise's answer

#### Scenario: A physics postulate
- **WHEN** a block states a physical law (the Born rule, a thermodynamic law, the path integral, a running coupling)
- **THEN** the law is a definition or theorem of an explicit Lean model (Hilbert space, probability space, finite-dimensional Gaussian integral, flow equation), and the book's computed consequences are certified theorems of that model

#### Scenario: Veriphy lacks a recipe
- **WHEN** a claim's computational core matches no Veriphy recipe
- **THEN** the missing recipe is added to Veriphy on its development branch (design D30) and used, rather than replaced by a hand-written proof in the book

### Requirement: Every definition is verified against its Lean counterpart
Every definition block SHALL have a Verification box whose claims include a correspondence claim: the book's wording,
stated literally in Lean, is equivalent to the Lean or mathlib definition the library uses (or, for a notion mathlib
lacks, the book's Lean definition holds on the chapter's first example), certified or tested at a stated level
(design V4); `scripts/check_verification.py` SHALL report a definition whose manifest entry is `none`.

#### Scenario: A pure definition
- **WHEN** the definition of a topological space is verified
- **THEN** its Verification box shows the book's three axioms as a Lean predicate and a theorem that a family of
  subsets satisfies them exactly when it is a mathlib `TopologicalSpace`, with its level

#### Scenario: A definition that also states a fact
- **WHEN** a definition such as "injective" also states a characterisation
- **THEN** its box holds both the correspondence claim and the claim for the stated fact


## MODIFIED Requirements

### Requirement: One definition per block, each with at least three examples
Each definition block SHALL define exactly one notion (no stacked definitions such as "chart, atlas, manifold" in one
block). It SHALL be followed by at least ten worked examples in a chapter listed as raised (design X2), and by at
least three in a chapter not yet raised. Where the notion can be drawn, the examples SHALL include a figure.

#### Scenario: A stacked definition
- **WHEN** a definition block names more than one defined term in its title or defines several notions in its body
- **THEN** the chapter is non-conforming and the block is split, each part with its own examples

#### Scenario: A definition with too few examples
- **WHEN** a definition block in a raised chapter is followed by fewer than ten example blocks before the next definition, or one in another chapter by fewer than three
- **THEN** the chapter is non-conforming and examples are added (a figure counts as an example only together with its explanation)

### Requirement: Every definition, theorem and proposition has graded examples and exercises

Each definition, theorem, proposition, lemma and corollary SHALL be followed by at least ten examples of increasing
complexity and at least ten exercises of increasing difficulty, each exercise with a collapsed solution. Chapters not
yet raised by the backfill (design X2) SHALL keep at least three of each until they are raised. Examples SHALL
include a figure or plot where the object can be drawn, and an interactive figure where feasible. Ten is a minimum,
not a target (design D21, D32, X1).

#### Scenario: Definition inspected
- **WHEN** the checker inspects a statement block of a chapter listed as raised
- **THEN** at least ten `exm-` blocks and at least ten `exr-` blocks with solution callouts follow it before the next statement

#### Scenario: Statement in a chapter not yet raised
- **WHEN** the checker inspects a statement block of a chapter not yet listed as raised
- **THEN** at least three `exm-` and three `exr-` blocks follow it, and the checker's backlog report lists how many each statement lacks for ten

#### Scenario: New content
- **WHEN** a statement is added to any chapter after this change
- **THEN** it has at least ten examples and ten exercises when it is added, whether or not its chapter has been raised

#### Scenario: Interactive example
- **WHEN** an example draws a surface or a vector field
- **THEN** the HTML output embeds an interactive version and the PDF output the static image

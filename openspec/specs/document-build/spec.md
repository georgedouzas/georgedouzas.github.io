# document-build Specification

## Purpose
Makes the tutorial reproducible: it renders to HTML and PDF from a clean checkout with pinned tool versions, and a
failing formal verification stops the render, so a published document never shows an unverified result as verified.

## Requirements

### Requirement: The tutorial renders to HTML and PDF
The document SHALL render to a multi-chapter HTML site and to a single PDF from one source, with numbered sections,
equations, figures and theorem references resolving in both.

#### Scenario: Both formats from one command
- **WHEN** the render command is run on a clean checkout with the pinned tools installed
- **THEN** it produces the HTML site and the PDF, and every cross-reference, equation number and citation resolves in both

### Requirement: Formal verification gates the render
The render SHALL be preceded by the Lean build and the registry, figure and bibliography checks, and SHALL NOT
proceed if any of them fails.

#### Scenario: A theorem breaks
- **WHEN** a Lean theorem referenced by the document fails to build
- **THEN** the render stops with the theorem named and no output document is produced

#### Scenario: All checks pass
- **WHEN** the Lean build and all checks succeed
- **THEN** the render proceeds and the output states the Lean toolchain, mathlib and Veriphy versions used

### Requirement: Theorem statements in the document come from the Lean sources
Where the document shows a Lean statement, the text SHALL be extracted from the Lean source at render time, not
copied by hand.

#### Scenario: Statement edited in Lean
- **WHEN** a theorem statement is changed in the Lean source and the document is re-rendered
- **THEN** the displayed statement reflects the change without any edit to the document source

#### Scenario: Theorem renamed
- **WHEN** a theorem referenced by the document no longer exists under that name
- **THEN** the render fails and names the missing identifier

### Requirement: Builds are reproducible with pinned versions
The Lean toolchain, mathlib, Veriphy, SageMath and Quarto versions SHALL be pinned in the repository, and the Lean
build SHALL NOT require SageMath.

#### Scenario: Fresh machine without Sage
- **WHEN** the Lean library is built on a machine with the pinned Lean toolchain but without SageMath
- **THEN** the build succeeds

#### Scenario: Version drift
- **WHEN** a tool's installed version differs from the pinned one
- **THEN** the build script reports the mismatch before rendering

### Requirement: Repository documents how to reproduce everything
The repository SHALL contain instructions to install the pinned tools, build the Lean library, regenerate figures,
regenerate certificates and render the document.

#### Scenario: New contributor follows the instructions
- **WHEN** a contributor follows the repository instructions on a clean machine
- **THEN** they obtain the rendered HTML and PDF identical in content to the committed outputs

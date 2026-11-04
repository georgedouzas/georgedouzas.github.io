# Design

## Context

The reading notes map each textbook to chapters (from the books' tables of contents). Uses not yet made, by chapter:

| Chapter | Textbook sections not yet used |
|---|---|
| Supersymmetry and supergravity | Nath Ch. 8, 11-12, 18; Zhelnorovich Ch. 1 (Majorana and charge conjugation in any dimension) |
| Gauge theories | Nath Ch. 3-4, 6; Maggiore Ch. 8 |
| Differential geometry | Boskoff-Capozziello Ch. 3-5; Blennow-Ohlsson Ch. 2 (curvature problems) |
| Compactifications (Part VI) | Boskoff-Capozziello Ch. 11-12 (de Sitter, anti-de Sitter) |
| Consistent reductions (Part VI) | Plebański-Krasiński (Bianchi spacetimes) |
| Lie groups and algebras | Boudreau-Swanson Ch. 14; Sinha Ch. 1 (rotations, Euler angles) |
| Quantum field theory | Boudreau-Swanson Ch. 24 (lattice field theory) |
| Quantum mechanics | Izaac-Wang Part III (numerical Schrödinger equation); Svanberg Ch. 2 (atomic structure) |
| Fourier analysis | Maggiore Ch. 6 (multipoles); Izaac-Wang Ch. 8 |
| Differential equations | Cini et al. Ch. 7; Foulaadvand Ch. 3, 5-7; Picard et al. Ch. 2 |
| Probability | Bricmont Ch. 2; Foulaadvand Ch. 10 |
| Thermodynamics and statistical mechanics | Bauerschmidt-Brydges-Slade Part I (spin systems, critical phenomena) |

## Decisions

### T1. One chapter group at a time

One agent per task, no helper agents, committed after each task (cost; author's instruction). The brief is the one
used for task group 12, with the folding script and placeholder Verification boxes.

### T2. Adding, not rewriting

Material is added to the existing structure: new examples and exercises attached to existing statements, and new
statements where a textbook adds a topic, in the chapter's `##`/`###` parts. Existing blocks are changed only to fix
errors.

### T2b. Consistency, not copying (author's instruction)

The textbooks are sources, not text to copy. Every addition is written in the book's voice, notation and conventions
(signature, index placement, units, symbols already introduced) and fits the chapter's line of argument: it is placed
where it belongs in the chapter's order, uses only notions defined earlier (or defines them first), refers back to the
chapter's running examples, and is introduced and followed by connective prose so the chapter reads as one text. A
topic the chapter already treats is deepened, not repeated; a textbook's own notation or argument is translated, and
its problems are restated for the book's setting. Every statement keeps at least three examples and three exercises,
new statements included.

### T3. Format

The rules of the main spec apply (connective prose, counts as a minimum, one notion per definition, at most six
definitions per unit, the main text never refers to verification), and `scripts/fold_blocks.py` regenerates the
statement boxes and folds.

## MODIFIED Requirements

### Requirement: Chapter order builds from geometry to CSDR to the survey
The tutorial SHALL be organised in six parts in this order: mathematics, physics, abelian Kaluza-Klein theory,
non-abelian Kaluza-Klein theory, coset space dimensional reduction, and the survey of compactification methods.
Each part SHALL use only concepts defined in earlier parts or in itself.

#### Scenario: Contents of the two foundation parts
- **WHEN** the table of contents is read
- **THEN** the mathematics part contains, in this order, sets and topology, linear algebra, real analysis, complex analysis, differential equations, Fourier analysis, Hilbert spaces and operator theory, probability and statistics, manifolds, vector fields and tensors, metrics, connections and curvature, flows and Killing vectors, groups, Lie groups and algebras, representations, coset spaces and bundles; and the physics part contains, in this order, classical mechanics, classical electromagnetism, thermodynamics and statistical mechanics, quantum mechanics, classical field theory, the Dirac equation, quantum field theory, gauge theories (with symmetry breaking), renormalisation, special relativity, general relativity, supersymmetry and supergravity

#### Scenario: Table of contents reflects the progression
- **WHEN** a reader opens the rendered tutorial's table of contents
- **THEN** the six parts appear in the stated order, and each later part's chapters cite the earlier definitions they rely on by cross-reference

#### Scenario: A concept is used before its definition
- **WHEN** a chapter uses a notion (for example a Killing vector or a structure constant) that is not defined in that chapter or an earlier one
- **THEN** the tutorial is non-conforming and the definition is added where the progression requires it

#### Scenario: Gauge theories after quantum field theory
- **WHEN** the reader reaches the gauge-theory chapter
- **THEN** the quantum field theory chapter precedes it, the renormalisation chapter follows it, and no block of the quantum field theory chapter uses a gauge-theory definition without a cross-reference to an earlier definition or a forward pointer in prose that names where the notion is defined

## ADDED Requirements

### Requirement: Classical electromagnetism has its own chapter after classical mechanics
The physics part SHALL contain a chapter on classical electromagnetism, placed immediately after classical mechanics.
It SHALL cover, in vector form and with the book's units:
- electrostatics and magnetostatics;
- Maxwell's equations;
- the potentials and gauge invariance;
- energy, momentum and the conservation laws;
- electromagnetic waves;
- the fields of moving charges and radiation;
- fields in material media.

The covariant formulation SHALL remain in the classical field theory and special relativity chapters, which SHALL
cite this chapter for the vector-form results they rewrite (design E3).

#### Scenario: Reading order
- **WHEN** the reader finishes classical mechanics
- **THEN** the next chapter in the sidebar is classical electromagnetism, and the Dirac equation, gauge-theory and Kaluza-Klein chapters cite it for Maxwell's equations, the potentials and gauge invariance

#### Scenario: Gauge invariance prepared
- **WHEN** the electromagnetism chapter introduces the potentials
- **THEN** it defines the gauge transformation and shows that the fields and the Lorentz force are invariant, and the gauge-theory chapter's abelian section starts from this definition

#### Scenario: Sources
- **WHEN** a block of the chapter is adapted from Maggiore's *A Modern Introduction to Classical Electrodynamics*
- **THEN** it cites the section or problem, is rewritten in the book's signature, units and notation, and the reading note lists it with the block that uses it

### Requirement: The gauge-theory chapter teaches gauge fields in depth
The gauge-theory chapter SHALL cover, in addition to abelian gauge invariance, Yang-Mills theory, spontaneous symmetry
breaking and the Higgs mechanism, the following topics:
- gauge fields as connections on principal bundles, with holonomy and Wilson loops;
- the Aharonov-Bohm effect, the Dirac monopole and charge quantisation;
- topological solutions: vortices, the 't Hooft-Polyakov monopole, instantons and the θ vacuum;
- the quantisation of gauge fields: gauge fixing, Faddeev-Popov ghosts and BRST symmetry;
- lattice gauge theory: the Wilson action, the strong-coupling expansion and confinement;
- the Standard Model: its gauge group, its fermion representations with their hypercharges, electroweak symmetry
  breaking, and the masses of the W and Z bosons.

Each topic SHALL have its statements with at least ten examples and ten exercises, and SHALL be connected by
prose to the chapter's line of argument.

#### Scenario: Connections from the mathematics part
- **WHEN** the chapter defines a gauge field as a connection
- **THEN** it cites the bundle and connection definitions of the mathematics part instead of repeating them, and Part IV cites this section when the gauge group arises from isometries

#### Scenario: Quantisation after quantum field theory
- **WHEN** the chapter quantises a gauge field
- **THEN** it uses the path integral and Feynman rules of the quantum field theory chapter by cross-reference

#### Scenario: Topics that need a source the book does not yet have
- **WHEN** a topic has no textbook or arXiv source in `references/`
- **THEN** it is not written until the author has supplied a source (design E5)

### Requirement: The physics foundations cover the topics the later parts rely on
The physics part SHALL include the following foundation topics:

- **General relativity:**
  - geodesics and the classical tests;
  - Killing vectors and conserved quantities;
  - the Einstein-Maxwell equations and the Reissner-Nordström solution;
  - the weak-field approximation and gravitational waves.
- **Quantum mechanics:**
  - scattering theory in one and three dimensions;
  - a charged particle in an electromagnetic field: minimal coupling, Landau levels and the Aharonov-Bohm phase.
- **Classical mechanics:**
  - the rigid body;
  - constrained Hamiltonian systems.
- **Quantum field theory:**
  - the quantisation of the Dirac and Maxwell fields;
  - the S-matrix and cross sections.

Each topic SHALL follow the chapter rules: definitions first, at least ten examples and ten exercises per
statement, and connective prose.

#### Scenario: General relativity prepares Kaluza-Klein theory
- **WHEN** the reader reaches the abelian Kaluza-Klein part
- **THEN** the Einstein-Maxwell equations and Killing vectors are already defined in general relativity, and Part III cites them

#### Scenario: Gauge phases in quantum mechanics
- **WHEN** the quantum mechanics chapter treats a charged particle
- **THEN** it uses the potentials of the electromagnetism chapter, and the gauge-theory chapter cites its Aharonov-Bohm phase when it introduces holonomy

# Proposal

## Why

The physics part has no chapter on classical electromagnetism. Maxwell's theory appears only in covariant form, in the
classical field theory and special relativity chapters, although the Dirac equation, gauge theories and the
Kaluza-Klein parts all start from it. The gauge-theory chapter is also short and comes before quantum field theory.
It covers abelian gauge invariance, Yang-Mills theory and the Higgs mechanism, but none of gauge fields as
connections, monopoles and instantons, the quantisation of gauge fields, lattice gauge theory or the Standard Model.
The Kaluza-Klein and CSDR parts need these, and gauge-field quantisation cannot be taught before quantum field theory.
Other foundation chapters have gaps too: general relativity stops at Schwarzschild and FLRW, and quantum mechanics
has no scattering theory and no charged particle in a field.

## What Changes

- **New chapter, Classical electromagnetism**, placed right after classical mechanics. It covers electrostatics and
  magnetostatics, Maxwell's equations in vector form, potentials and gauge invariance, energy, momentum and
  conservation laws, waves, the fields of moving charges and radiation, and fields in matter. The source is Maggiore,
  *A Modern Introduction to Classical Electrodynamics*. The covariant form stays in the classical field theory and
  special relativity chapters, which now cite the new chapter.
- **Reordered physics part.** The new order is: classical mechanics, classical electromagnetism, thermodynamics and
  statistical mechanics, quantum mechanics, classical field theory, the Dirac equation, quantum field theory, gauge
  theories (with symmetry breaking), renormalisation, special relativity, general relativity, supersymmetry and
  supergravity. Cross-references that now point forward are resolved.
- **Expanded gauge-theory chapter.** New sections:
  - gauge fields as connections, holonomy and Wilson loops;
  - the Aharonov-Bohm effect, the Dirac monopole and charge quantisation;
  - vortices, the 't Hooft-Polyakov monopole and instantons with the θ vacuum;
  - the quantisation of gauge fields (gauge fixing, Faddeev-Popov ghosts, BRST);
  - lattice gauge theory;
  - the Standard Model.
- **Other foundation chapters filled in:**
  - **General relativity:** geodesics and the classical tests, Killing vectors and conserved quantities, the
    Reissner-Nordström solution and Einstein-Maxwell theory, the weak field and gravitational waves.
  - **Quantum mechanics:** scattering theory, and a charged particle in an electromagnetic field.
  - **Classical mechanics:** the rigid body, and constrained Hamiltonian systems.
  - **Quantum field theory:** quantisation of the Dirac and Maxwell fields, the S-matrix and cross sections.
- **Topics without a source here** are written only after the author supplies textbooks. The design lists these
  topics and the books requested for them.

## Capabilities

### Modified Capabilities

- `tutorial-content`:
  - the physics part's chapter order;
  - a classical electromagnetism chapter;
  - the content required of the gauge-theory chapter;
  - the foundation topics required in general relativity, quantum mechanics, classical mechanics and quantum field
    theory.

## Impact

- **Changed files:**
  - `tutorial/_quarto.yml` (chapter order);
  - a new `tutorial/part-ii-01b-electromagnetism.qmd`;
  - the Part II chapters named above;
  - `references/notes/`, `references.bib`, `sage/exercises/` and `figures/`.
- **Verification:** none, since verification is paused. New blocks carry placeholder Verification boxes, as in
  kk-csdr-textbook-pass.
- **Chapter files keep their names.** Only the order in `_quarto.yml` changes, so identifiers and cross-references
  stay valid.

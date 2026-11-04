# Tasks

## 1. Order of the physics part

- [x] 1.1 Reorder `tutorial/_quarto.yml` as in design E1 and add `part-ii-01b-electromagnetism.qmd` after
  classical mechanics. Give the new chapter a stub that names its sections under "To be written", with a
  self-contained introduction.
- [x] 1.2 List every cross-reference in the physics part from a chapter to a block or section of a later chapter.
  Resolve each use-before-definition (design E1, E3 and the Risks section): move the use to an earlier definition, or
  turn the mention into a prose pointer. Update the part and chapter introductions that describe the order.
  Check: the checkers and the HTML draft render.

## 2. Classical electromagnetism (Maggiore)

- [x] 2.1 Electrostatics, magnetostatics and energy (Maggiore Ch. 1-5; Ch. 6 by cross-reference to the Fourier
  chapter's multipoles). Check (checkers and draft render).
- [x] 2.2 Maxwell's equations in vector form, the potentials and gauge invariance, the conservation laws, and minimal
  coupling of a charged particle (Maggiore Ch. 3; design E3). Redirect the Dirac chapter's forward uses of
  `def-covariant-derivative-u1` to the new definition. Check (checkers and draft render).
- [x] 2.3 Waves, the fields of moving charges and radiation in vector form (Maggiore Ch. 9-12). Add back-references
  from the classical field theory and special relativity blocks that restate these results covariantly. Check (checkers and draft render).
- [x] 2.4 Fields in material media, dispersion, waveguides and scattering (Maggiore Ch. 13-16). Check (checkers and draft render).

## 3. Gauge theories from the existing sources

- [x] 3.1 Lattice gauge theory: Z₂, compact U(1) and SU(2), the Wilson action, strong coupling and confinement
  (Boudreau-Swanson Secs. 24.3-24.5). Check (checkers and draft render).
- [ ] 3.2 The Standard Model: the gauge group, the fermion representations and hypercharges, electroweak breaking,
  and the W and Z masses (Nath Ch. 3-4, the parts not yet used). Check (checkers and draft render).

## 4. Other foundations from the existing sources

- [ ] 4.1 General relativity: geodesics and the classical tests; Killing vectors and conserved quantities; the
  Einstein-Maxwell equations and Reissner-Nordström; the weak field and gravitational waves (Plebański-Krasiński
  Ch. 12-14 and later; Blennow-Ohlsson Ch. 2; Boskoff-Capozziello Ch. 10). Check (checkers and draft render).
- [ ] 4.2 Quantum mechanics: scattering in one and three dimensions (Cini et al. Ch. 13-14); a charged particle in an
  electromagnetic field (minimal coupling from 2.2, Landau levels, the Aharonov-Bohm phase), with Guidry-Sun Ch. 27 and Sec. 28.2. Check (checkers and draft render).
- [ ] 4.3 Classical mechanics: the rigid body (Sinha, the chapters not yet used). Check (checkers and draft render).

## 5. Topics sourced from the author's library (design E5)

- [x] 5.1 Ask the author for the textbooks of design E5. Add each supplied book to `references/textbooks/` with a
  reading note and a bib entry, and record each declined topic in the design.
- [ ] 5.2 Gauge fields as connections, holonomy and Wilson loops; the Aharonov-Bohm effect and the Dirac monopole
  (Hamilton Ch. 4-5; Guidry-Sun Ch. 26-27; Năstase Ch. 27). Check (checkers and draft render).
- [ ] 5.3 Vortices, the 't Hooft-Polyakov monopole, instantons and the θ vacuum (Năstase Ch. 24-25, 28-30; Guidry-Sun Ch. 25). Check (checkers and draft render).
- [ ] 5.4 Quantisation of gauge fields: gauge fixing, Faddeev-Popov ghosts and BRST (Williams Ch. 9). Check (checkers and draft render).
- [ ] 5.5 Quantum field theory: quantisation of the Dirac and Maxwell fields, the S-matrix and cross sections (Williams Ch. 6-7). Check (checkers and draft render).
- [ ] 5.6 Classical mechanics: constrained Hamiltonian systems and Dirac brackets (Williams Sec. 2.9). Check (checkers and draft render).

## 6. Close

- [ ] 6.1 Audit the reading notes of every source used (each mapped section used or recorded as not fitting). Run all
  the checkers, the HTML render and the PDF render. Fix what fails.

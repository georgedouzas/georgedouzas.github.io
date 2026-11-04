# Design

## Context

The physics part currently runs in this order (`tutorial/_quarto.yml`): classical mechanics, thermodynamics and
statistical mechanics, quantum mechanics, the Dirac equation, classical field theory, gauge theories, quantum field
theory, renormalisation, special relativity, general relativity, supersymmetry and supergravity.

**Electromagnetism already in the book.**
- The classical field theory chapter has the covariant Maxwell theory, with about 30 blocks adapted from Maggiore
  Ch. 8-11 (prefix `cfb-`), among them Larmor, Liénard and the radiation zones.
- The special relativity chapter has a section "Electrodynamics in covariant form".
- The Fourier chapter has the multipole expansion (Maggiore Ch. 6).
- Nothing yet treats electrostatics, magnetostatics, Maxwell's equations in vector form, energy, media or waves in
  matter.

**The gauge-theory chapter** (3,222 lines) has two sections: "Gauge theories" (abelian gauge invariance and Yang-Mills
theory) and "Symmetry breaking" (spontaneous symmetry breaking and the Higgs mechanism).

**Cross-references into it from earlier chapters, found by search:**
- the classical field theory chapter cites `sec-abelian-gauge`, `sec-yang-mills`, `sec-ssb` and `sec-higgs`;
- the Dirac chapter uses `def-covariant-derivative-u1` and `def-gauged-scalar-model`, which already point forward
  today;
- the quantum field theory chapter cites `sec-yang-mills` and `sec-higgs`.

**Sources already in `references/textbooks/`, with what they cover:**

| Topic | Source available | Status |
|---|---|---|
| Classical electromagnetism | Maggiore Ch. 1-6, 9-16 (vector form, energy, waves, radiation, media, scattering) | available |
| Lattice gauge theory | Boudreau-Swanson Secs. 24.3-24.5 (Z₂, compact U(1), SU(2)) | available, unused |
| Standard Model | Nath Ch. 3-4 (gauge group, electroweak breaking, fermion content), partly used | available |
| General relativity | Plebański-Krasiński Ch. 12-14 and later (Einstein equations, Einstein-Maxwell, Reissner-Nordström, orbits, Kaluza-Klein in Sec. 13.6); Blennow-Ohlsson Ch. 2 problems (Killing vectors, geodesics, Kruskal, lensing, gravitational waves); Boskoff-Capozziello Ch. 10 | available |
| QM scattering | Cini et al. Ch. 13-14 | available |
| Rigid body | Sinha (rigid dynamics) | available, partly used |
| Gauge fields as connections, holonomy, Wilson loops; Aharonov-Bohm; Dirac monopole | Hamilton Ch. 4-5; Guidry-Sun Ch. 26-27; Năstase Ch. 27 | supplied (Zotero) |
| Vortices, 't Hooft-Polyakov monopole, instantons, θ vacuum | Năstase Ch. 24-25, 28-30; Guidry-Sun Ch. 25 | supplied (Zotero) |
| Gauge-field quantisation (Faddeev-Popov, BRST); Dirac and Maxwell field quantisation; S-matrix and cross sections | Williams Ch. 6-7, 9 | supplied (Zotero) |
| Constrained Hamiltonian systems | Williams Sec. 2.9 | supplied (Zotero) |
| Landau levels, electron in a magnetic field | Guidry-Sun Ch. 27, Sec. 28.2 | supplied (Zotero) |

## Goals / Non-Goals

**Goals:**
- Add a classical electromagnetism chapter after classical mechanics.
- Move gauge theories to after quantum field theory.
- Deepen the gauge-theory chapter.
- Fill the foundation gaps that Parts III-VI rely on.
- Keep the reading flow in every chapter.

**Non-Goals:**
- Verification work of any kind (paused).
- Renaming chapter files.
- Rewriting existing blocks, except to fix errors or forward references.
- Topics with no source until the author supplies one.
- Moving special relativity, which the author placed before general relativity.

## Decisions

### E1. The order of the physics part

The new order is: classical mechanics, **classical electromagnetism**, thermodynamics and statistical mechanics,
quantum mechanics, classical field theory, the Dirac equation, quantum field theory, **gauge theories**,
renormalisation, special relativity, general relativity, supersymmetry and supergravity.

- **Gauge theories go immediately after quantum field theory**, not after renormalisation, because the
  renormalisation chapter uses QED Ward identities and asymptotic freedom in Yang-Mills theory.
- **Classical field theory goes before the Dirac equation.** The Dirac chapter cites 29 blocks of the classical field
  theory chapter (the gamma matrices, the Dirac equation, the Lorentz generators), and none points the other way.
- **Electromagnetism goes immediately after classical mechanics** (the author's instruction). The chapter uses
  vector calculus from the mathematics part and Lagrangian mechanics for the charged particle.

### E2. File names are kept

The new chapter is `tutorial/part-ii-01b-electromagnetism.qmd`, following the `08b` precedent. Only `_quarto.yml`
changes order. Identifiers, cross-references, Sage labels and figure names stay valid.

### E3. Vector form here, covariant form later

The electromagnetism chapter is written in vector form: fields **E** and **B**, potentials φ and **A**, with the
book's units (Heaviside-Lorentz, c = 1, and SI where numbers are given, as in the classical field theory chapter).

- **Covariant material stays where it is.** The covariant form remains in the classical field theory and special
  relativity chapters. Their existing blocks are kept and gain cross-references back to the new chapter where they
  rewrite a vector-form result.
- **Results derived in vector form here.** The new chapter derives these itself: the Larmor formula, the dipole
  radiation and the near, induction and far zones. The `cfb-` blocks then read as the covariant restatement.
- **Lorentz invariance is pointed forward.** Where the chapter meets the Lorentz invariance of Maxwell's equations,
  it says in prose that the special relativity chapter develops it. That is a pointer, not a use.
- **Minimal coupling and the gauge transformation of the potentials are defined here.** The Dirac chapter's forward
  uses of `def-covariant-derivative-u1` are redirected to the new definition. The gauge-theory chapter's abelian
  section starts from it.

### E4. Structure of the expanded gauge-theory chapter

The existing two `##` sections are kept. New `##` sections are added in this order:

1. Gauge fields as connections: holonomy and Wilson loops, citing Part I's bundles.
2. The Aharonov-Bohm effect and the Dirac monopole.
3. Topological solutions: vortices, monopoles, instantons and the θ vacuum.
4. Quantisation of gauge fields: gauge fixing, Faddeev-Popov, BRST.
5. Lattice gauge theory: the Wilson action, strong coupling and confinement, from Boudreau-Swanson 24.3-24.5.
6. The Standard Model: Nath Ch. 3-4.

Sections 1-3 come before "Symmetry breaking" and sections 4-6 after it, so that the classical geometry precedes the
quantum theory.

### E5. Sources, and textbooks to request

Every new block is adapted from a source in `references/` (D32), following the consistency rule of
kk-csdr-textbook-pass (T2b): rewritten in the book's voice and conventions, not copied.

**Topics without a source wait for one.** The author supplied the missing sources from their Zotero library. Each
book was copied to `references/textbooks/` and given a reading note and a `keywords = {textbook}` bib entry:

| Book | Topics it sources |
|---|---|
| Hamilton, *Mathematical Gauge Theory* (Springer, 2017; `hamilton2017mathematical`) | gauge fields as connections, holonomy, the Standard Model's geometry |
| Năstase, *Classical Field Theory* (CUP, 2019; `nastase2019classical`) | vortices, the Dirac and 't Hooft-Polyakov monopoles, instantons |
| Williams, *Introduction to Quantum Field Theory* (CUP, 2023; `williams2023introduction`) | constrained Hamiltonian systems, Dirac and Maxwell field quantisation, the S-matrix, Faddeev-Popov and BRST |
| Guidry-Sun, *Symmetry, Broken Symmetry, and Topology in Modern Physics* (CUP, 2022; `guidry2022symmetry`) | geometric phases, Aharonov-Bohm, Landau levels, lattice gauge symmetry, solitons |

**The whole library was then copied.** The author's Zotero library was later copied in full to `references/textbooks/`
(54 further books, Aitchison-Hey and Hosotani among them), for this change and later ones. Each of 51 books was given
a reading note mapped from its table of contents and a `keywords = {textbook}` bib entry. Three books serve no
chapter; they are recorded in `references/notes/README.md`. The books mapped to the chapters this change touches are these:

| Chapter | Books (bib keys) | What they add |
|---|---|---|
| Classical electromagnetism (planned) | `macchi2023problems`, `pierrus2018solved`, `aliev2024mathematical`, `kotkin2020exploring`, `aitchison2024gauge` | exercises for tasks 2.1-2.4; boundary-value problems; charged particles in fields; electromagnetism as a gauge theory |
| Classical mechanics | `moretti2023analytical`, `knauf2018mathematical`, `kotkin2020exploring`, `lee2018global`, `godinho2014introduction`, `rudolph2013differential`, `torresdelcastillo2020differentiable`, `lee2013introduction`, `emam2021covariant` | Lagrangian and Hamiltonian content, the rigid body (task 4.3), symplectic geometry, 350 solved problems |
| Quantum mechanics | `nastase2023quantum`, `zettili2009quantum`, `galitski2013exploring`, `demilio2017problems`, `tornetta2022mathematical`, `talagrand2022what`, `mukunda2022continuous`, `derezinski2022mathematics` | content and problems; Landau levels and scattering (task 4.2); Wigner's theorem; the CCR |
| Quantum field theory | `aitchison2024gauge`, `buchbinder2021introduction`, `talagrand2022what`, `gelis2021problems`, `shankar2017quantum`, `derezinski2022mathematics`, `strocchi2021symmetry`, `paganini2023fundamentals`, `galitski2013exploring`, `nastase2023quantum`, `tornetta2022mathematical` | Dirac and Maxwell field quantisation (task 5.5), the S-matrix, path integrals, Fock spaces, the Goldstone theorem, solved problems |
| Gauge theories | `nastase2023quantum`, `gelis2021problems`, `shankar2017quantum`, `buchbinder2021introduction`, `strocchi2021symmetry`, `pal2015introductory`, `paganini2023fundamentals`, `aitchison2024gauge`, `hosotani2025introduction`, `kivelson2024statistical`, `needham2021visual` | Aharonov-Bohm and the Dirac monopole (task 5.2), Wilson loops and instantons (5.2, 5.3), Faddeev-Popov and BRST (5.4), the Z₂ lattice gauge theory (3.1), the Higgs mechanism and the Standard Model (3.2) |
| Special relativity | `gourgoulhon2013special`, `moretti2023analytical`, `mukunda2022continuous`, `talagrand2022what`, `tornetta2022mathematical`, `knauf2018mathematical`, `godinho2014introduction`, `emam2021covariant`, `gray2019student`, `macchi2023problems`, `pierrus2018solved`, `paganini2023fundamentals`, `pal2015introductory` | observers and frames, the Lorentz and Poincaré groups, covariant electrodynamics, kinematics problems |
| General relativity | `gray2019student`, `godinho2014introduction`, `krasnov2020formulations`, `emam2021covariant`, `buchbinder2021introduction`, `needham2021visual` | Schwarzschild, perihelion and gravitational waves (task 4.1), the tetrad formulation, fields in curved spacetime |

### E6. Writing rules

The rules are those of kk-csdr-textbook-pass:
- **Blocks:** at least ten examples and ten exercises per statement (change kk-csdr-ten-examples), and connective prose before every block.
- **Units and definitions:** at most six definitions per unit, one notion per definition.
- **Folding:** fold with `scripts/fold_blocks.py`.
- **Verification boxes:** placeholder boxes, with plain Sage for figures and about 40% of exercise solutions.
- **Main text:** no reference to verification.
- **Inline formulas:** the PDF-safety rules that `check_format` enforces (no list marker after a line break inside a
  formula, and no space inside the dollars).

### E7. One agent per task, committed after each

One agent per task, with no helper agents. Each task is committed and ticked before the next starts (the author's
cost instruction). The writing brief of kk-csdr-textbook-pass is reused, with this design's additions.

## Risks / Trade-offs

- **Forward references after the reorder.**
  - **Risk:** moving gauge theories after quantum field theory turns cross-references into forward uses.
  - **Mitigation:** task 1 lists every cross-reference from a chapter to a block or section of a later chapter in the
    physics part, and resolves each one. A use is moved to the earlier definition (E3). A mention becomes a prose
    pointer.
- **Overlap between electromagnetism and classical field theory.**
  - **Trade-off:** some results then appear twice, in vector and in covariant form.
  - **Mitigation:** the later block cross-references the earlier one and says what the covariant form adds.
- **Late sources.**
  - **Trade-off:** the four library books were added after the spec, and their reading notes are mapped from tables
    of contents only.
  - **Mitigation:** each task reads its sections before writing and records in the note what it used.
- **Size.** The gauge-theory chapter will roughly triple. It stays one chapter (its `##` sections are single-topic),
  and can be split later if the author wishes.

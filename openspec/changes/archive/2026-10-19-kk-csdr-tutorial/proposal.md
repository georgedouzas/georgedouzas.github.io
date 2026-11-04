# Proposal

## Why

Kaluza-Klein (KK) reduction and coset space dimensional reduction (CSDR) are taught through long tensor
computations that reviews routinely abbreviate ("after a straightforward calculation one finds ..."). Readers
cannot check those steps, and errors in sign conventions and normalisations propagate. This project writes a
pedagogical tutorial in Quarto whose every displayed computational result is a Lean 4 theorem, proved with
Veriphy certificates (SageMath as an untrusted oracle, Lean kernel as the judge), so that no step is omitted
and no result has to be taken on trust. The Lean workspace with Veriphy 0.2.0 already builds in this repository,
so the verification infrastructure is in place.

## What Changes

- New Quarto book project (`tutorial/`) with chapters in this order:
  1. mathematics, assuming only undergraduate mathematics: sets and topology; linear algebra; real analysis,
     complex analysis, differential equations, Fourier analysis, Hilbert spaces and operator theory (spectral
     theory), probability and statistics (added October 2026, design D24, so that the quantum-mechanical and
     quantum-field-theoretic material has its tools); differential geometry (manifolds, charts, embeddings,
     tensors, metric, connection, curvature, Killing vectors) worked on explicit examples (`S^2`, `S^1`, `SU(2)`,
     the torus); group theory (groups, subgroups, cosets and quotient groups, homomorphisms, representations,
     Schur's lemma) with finite and matrix examples; Lie groups and Lie algebras (matrix groups, exponential map,
     structure constants, Killing form, Maurer-Cartan forms); representations of Lie algebras (`su(2)` and `su(3)`,
     roots and weights, tensor products, branching rules under subgroups); an extensive chapter on coset spaces
     (`S^n = SO(n+1)/SO(n)`, `S^3` as `SU(2)` and as `SU(2)×SU(2)/SU(2)`, `CP^n`, the flag manifold
     `SU(3)/U(1)×U(1)`, the nearly-Kähler six-dimensional cosets, reductive and symmetric decompositions, the
     isotropy representation, coset vielbeins and the metric from the Maurer-Cartan forms); fibre bundles and the
     Hopf fibration;
  2. physics: classical mechanics, thermodynamics and statistical mechanics, quantum mechanics (added October
     2026, design D24); classical field theory and quantum field theory (actions, Noether currents, gauge
     invariance, Yang-Mills theory, spontaneous symmetry breaking and the Higgs mechanism); general relativity
     (Einstein-Hilbert action, its variation, the Einstein equations, the cosmological constant); supersymmetry
     (the superalgebra, N=1 multiplets in four dimensions, the Wess-Zumino model, the superpotential);
     supergravity (the gravitino, local supersymmetry, eleven-dimensional supergravity, and what dimensional
     reduction does to supersymmetry);
  3. the simplest KK theory: 5D pure gravity on `M_4 x S^1`, the KK metric ansatz, the reduction of the
     Einstein-Hilbert action to gravity plus Maxwell plus a scalar, the KK tower and charge quantisation;
  4. non-abelian KK: gravity on `M_4 x S^2`, Killing vectors of `S^2` generating `SU(2)`, the gauge fields from
     the metric, structure constants;
  5. CSDR: gauge theory on `M_4 x S/R` with `S/R = SO(3)/SO(2) ≅ SU(2)/U(1) = S^2`, the symmetric-field
     constraints, their solution via embedding `R` in the gauge group, the resulting 4D gauge group, Higgs fields
     and potential;
  6. a survey of every method of compactification and dimensional reduction, in logical order: toroidal and
     Scherk-Schwarz reductions; group-manifold reductions and the consistency problem of coset reductions;
     supergravity sphere reductions (Freund-Rubin, consistent truncations, exceptional field theory); the
     symmetric-field schemes compared (CSDR, Scherk-Schwarz, equivariant, fuzzy); special-holonomy
     compactifications (Calabi-Yau, `G_2`, nearly-Kähler, orbifolds); flux compactifications and moduli
     stabilisation; warped and large extra dimensions as the alternative to compactification.
  Every derivation shows all intermediate steps; the survey part verifies in Lean the algebraic core of one worked
  example per method and labels the rest as argued in the text with citations.
- Pedagogy of every chapter: every term is defined before it is used, each definition is followed by as many
  worked examples as the material allows, and each chapter contains exercises that ask the reader to compute
  something, with the solution folded (collapsed) below the exercise, given theoretically, in Sage, in Lean, or in
  all three.
- New Lean library of verified results (the existing `KaluzaKlein` library): one theorem (or definition plus
  theorem) per displayed result, proved with mathlib tactics or Veriphy certificates, with the committed proofs
  being the Sage-free `Try this:` replacements. No `sorry`.
- New certificate scripts under `.veriphy/certificates/` and the copied `CoordinateGeometry` asset under
  `.veriphy/lean/`, as the Veriphy playbooks require.
- New figure pipeline: Sage scripts (SageManifolds, 2D/3D plots) that generate every figure used in the tutorial
  (`S^1` fibration over `M_4` sketch, `S^2` with Killing vector fields, Hopf fibration `S^3 -> S^2`, coset
  decomposition, KK mass tower, CSDR potential), committed as scripts with generated image outputs.
- New bibliography (`references.bib`) whose entries are arXiv papers that were fetched and read during
  implementation; each citation in the text points at a specific section or equation of the cited paper. The
  arXiv identifiers confirmed so far through the arXiv API: gr-qc/9805018 (Overduin, Wesson, "Kaluza-Klein
  Gravity"), hep-th/9410046 (Duff, "Kaluza-Klein Theory in Perspective"), hep-ph/0106033 (Manousselis,
  Zoupanos, CSDR over non-symmetric cosets), 0905.2398 (Chatzistavrakidis, Zoupanos, heterotic string over
  nearly-Kaehler manifolds), 1002.2606 (Chatzistavrakidis, Steinacker, Zoupanos, orbifolds and fuzzy spheres).
  Further references are added only after their identifier and content are checked.
- New build pipeline: `lake build` must succeed (which runs every Lean check) before `quarto render`; the
  rendered document embeds the theorem statements it relies on, so a result that fails to verify cannot be
  rendered as verified.

## Capabilities

### New Capabilities

- `tutorial-content`: the pedagogical structure and content requirements of the tutorial (chapter order, the survey of
  compactification methods,
  prerequisites, no omitted steps, worked examples, notation and sign conventions fixed once and used
  throughout).
- `result-verification`: the requirement that every displayed computational result corresponds to a Lean
  theorem that builds without `sorry` and without a live Sage call, the document-to-theorem traceability, and
  the handling of results that cannot be formalised (they are labelled, never silently presented as verified).
- `figure-generation`: figures are produced by committed Sage scripts, reproducibly, and the tutorial never
  embeds a hand-drawn or externally sourced image.
- `bibliography`: references are arXiv papers whose identifiers resolve through the arXiv API and whose
  content was read; citations point at specific sections or equations; claims attributed to a paper match that
  paper.
- `document-build`: the Quarto project renders to HTML and PDF from a clean checkout, the Lean build gates the
  render, and the build is reproducible with the pinned toolchain versions.

### Modified Capabilities

None. The project has no existing specs.

## Impact

- New directories: `tutorial/` (Quarto project: `_quarto.yml`, chapter `.qmd` files, `references.bib`),
  `figures/` (Sage scripts and generated images), `.veriphy/lean/` (copied Veriphy asset).
- Modified: `KaluzaKlein/` Lean library gains one module per chapter; `lakefile.toml` gains the `VeriphyAssets`
  library declaration for the copied asset; `.veriphy/certificates/` gains the project's certificate scripts.
- Dependencies (already installed, pinned): Lean `v4.34.1`, mathlib `v4.34.1`, Veriphy `0.2.0` (Lake package and
  Sage daemon), SageMath 10.9, Quarto 1.10.19.
- Risk: mathlib has no coordinate-free Riemannian curvature, so verification happens in coordinates on
  `Fin n -> R` through Veriphy's `CoordinateGeometry` asset, and the design must record which statements are
  verified in that form and which are glue proved directly in Lean.
- Risk: full reduction of the 5D Einstein-Hilbert action is a large symbolic computation; the design must split
  it into kernel-checkable component identities.

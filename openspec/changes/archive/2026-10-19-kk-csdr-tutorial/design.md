# Design

## Context

See proposal.md for motivation. The repository already holds a working Lake workspace (`lakefile.toml` at the root,
Lean `v4.34.1`, mathlib `v4.34.1`, Veriphy `0.2.0`) with a `KaluzaKlein` library and a passing `sage_run` smoke
test, the `veriphy` Claude Code plugin, SageMath 10.9 with the Veriphy daemon, and Quarto 1.10.19. The sibling
project `../coset-space-dimensional-reduction-over-super-coset-spaces` (same author) established conventions worth
reusing: a `claims.yaml` registry with levels `lean` / `sage` / `text` validated by a checker, a certificate driver
that splices Sage-free replacements into committed Lean files, a bibliography of 69 entries (many of them the CSDR
literature this tutorial needs), and a Quarto document rendered with LuaLaTeX.

Constraints that shape the approach:

- mathlib has no usable coordinate-free Riemannian curvature. Veriphy's `CoordinateGeometry` asset defines metric,
  Christoffel symbols, Riemann, Ricci, scalar curvature, Lie derivative, Killing condition and exterior derivative
  in a single chart on `Fin n -> R`, and the `coordinate_geometry` recipe proves component statements about explicit
  metrics. Abstract manifold statements are out of reach and must be labelled `text`.
- The Kaluza-Klein reduction involves arbitrary field functions, not an explicit metric. Its verification must be
  reformulated as polynomial identities in jet variables.
- The CSDR part is finite-dimensional Lie algebra and polynomial algebra, which Veriphy covers well
  (`lie_structure`, `representation`, `matrix_identity`, `ideal_membership`, `witness`, `sos`, `hint_search`,
  `factor_rewrite`).
- Every reference must be an arXiv paper that was fetched and read. Several classical CSDR papers predate arXiv.

## Goals / Non-Goals

**Goals:**

- One repository layout in which text, Lean theorems, certificates, figures, registry and bibliography are linked by
  checkable identifiers, and a single command runs every check and then renders.
- Verification strategies for each part of the tutorial that are feasible within Veriphy's recipe set, with an
  explicit fallback level for anything that is not.
- Reuse of the sibling project's registry, driver and bibliography conventions where they apply, so the two projects
  stay consistent.

**Non-Goals:**

- Formalising manifolds, bundles or coset spaces abstractly in Lean. Verification is of components and algebra.
- Extending Veriphy itself. Gaps found are reported as `blocked-missing-building-block` in the registry notes and
  worked around at level `sage`.
- Fermions, supersymmetry, non-symmetric cosets, fuzzy spaces and Wilson flux. The CSDR part stops at the bosonic
  sector over `SU(2)/U(1)`.

## Decisions

### D1. Repository layout: a Quarto book next to the Lake workspace

```text
lakefile.toml, lean-toolchain, KaluzaKlein.lean    Lake workspace (root), library KaluzaKlein
KaluzaKlein/Conventions.lean                        signature, generators, structure constants shared by all parts
KaluzaKlein/Geometry/*.lean                         Part I results (sphere, circle, SU(2), Killing vectors)
KaluzaKlein/AbelianKK/*.lean                        Part II results (ansatz, inverse metric, Christoffels, reduction, tower)
KaluzaKlein/NonAbelianKK/*.lean                     Part III results (Killing algebra, gauge transformation)
KaluzaKlein/CSDR/*.lean                             Part IV results (constraints, centraliser, scalars, potential)
.veriphy/certificates/*.sage                        certificate scripts (Veriphy layout)
.veriphy/lean/VeriphyAssets/CoordinateGeometry.lean copied asset, declared as lean_lib VeriphyAssets (srcDir .veriphy/lean)
.veriphy/certify.py                                 driver adapted from the sibling project (marker -> replacement)
tutorial/_quarto.yml, tutorial/*.qmd                Quarto book: index, conventions, part-i-*.qmd ... part-v-*.qmd, glossary, appendices
sage/exercises/*.sage, *.out                        exercise solution scripts and their recorded outputs
tutorial/_generated/                                statements extracted from Lean at render time (git-ignored)
tutorial/figures/*.svg|*.png                        generated figures (committed)
figures/*.sage                                      figure scripts, one per figure, named after the figure
claims.yaml                                         result registry (label -> level -> evidence)
references.bib, references/notes/<key>.md           bibliography and one reading note per cited paper
references/papers/                                  downloaded PDFs (git-ignored)
requirements.txt, .venv/                            Python packages for scripts/ (venv git-ignored)
scripts/check_claims.py, check_bib.py, check_figures.py, extract_statements.py, lean_audit.py
Makefile                                            `make check` (lake build + all checks), `make figures`, `make render`
README.md                                           reproduction instructions
```

Alternatives: a single `.qmd` as in the sibling project (rejected: a tutorial of four parts reads better as a book
with per-chapter HTML pages and cross-chapter references); Lean sources under `lean/` (rejected: the workspace with
`KaluzaKlein/` at the root already builds, and Veriphy resolves `.veriphy/` relative to the workspace root).

### D2. Geometry in coordinates through the `CoordinateGeometry` asset

Part I states its results about explicit metrics (`S^1`, round `S^2`, bi-invariant `SU(2)` in Euler angles) as
component theorems about the asset's `christoffel`, `riemann`, `ricci`, `scalarCurvature` and `IsKilling`,
proved with the `coordinate_geometry` recipe (`christoffel_proof`, `killing_proof`) and, for curvature, by stating
the Christoffel symbols first and deriving curvature components from them (the recipe's documented approach for
large components). Vector field brackets are not in the asset; the project defines `lieBracket` for coordinate
vector fields in `Conventions.lean` and proves the `su(2)` closure of the sphere's Killing fields by `simp` with
derivative lemmas and `ring`.

Alternative: mathlib's `Manifold` API (rejected: no curvature, no explicit charts for computation). The tutorial text
explains the coordinate-free notions and states explicitly that the verification is of the component formulas in
the named chart, with chart domains stated as hypotheses (`Real.sin (x 0) ≠ 0`).

### D3. Kaluza-Klein reduction as jet-polynomial identities, in stages

The ansatz `ds^2 = g_{μν}(x) dx^μ dx^ν + φ(x)^2 (dy + A_μ(x) dx^μ)^2` has arbitrary functions. Lean verifies the
reduction as identities in a polynomial ring whose variables are the jet variables: `g_{μν}`, `∂_ρ g_{μν}`,
`∂_ρ ∂_σ g_{μν}`, `A_μ`, `∂_ν A_μ`, `∂_ρ ∂_ν A_μ`, `φ`, `∂_μ φ`, `∂_μ ∂_ν φ`, plus symbols `ginv_{μν}` for the 4D
inverse metric constrained by `g · ginv = 1` (hypotheses). Definitions in `AbelianKK/Jets.lean` give the 5D
Christoffel symbols, Ricci tensor and curvature scalar as the standard polynomial and rational expressions in these
variables, matching the asset's definitions so the tutorial can say the same formula is used.

The chain, each step a theorem:

1. inverse of the 5D ansatz metric: a block matrix identity with `ginv` symbols (`matrix_identity` plus the `g · ginv
   = 1` hypotheses through `ideal_membership`);
2. each 5D Christoffel symbol in terms of the 4D Christoffel symbols (abbreviations defined from the jets), `F_{μν}`
   and `∂φ` (`ideal_membership`, one theorem per index pattern: `Γ^λ_{μν}`, `Γ^λ_{μ5}`, `Γ^5_{μν}`, `Γ^5_{μ5}`, `Γ^λ_{55}`,
   `Γ^5_{55}`);
3. the Ricci components `R_{μν}`, `R_{μ5}`, `R_{55}` in terms of 4D Ricci, `F`, `φ` and covariant derivatives of `φ`
   (`calc_derivation` through the intermediate forms the text displays);
4. the curvature scalar `R_5 = R_4 - (φ^2/4) F_{μν}F^{μν} - (2/φ) □φ` (`linear_combination` from 3);
5. the volume factor `sqrt(-g_5) = φ sqrt(-g_4)` (a determinant identity, `matrix_from_goal`);
6. the Weyl rescaling to the Einstein frame as a separate identity in the jets of the conformal factor.

Sage (SageManifolds, `TensorField` on a 5D product chart with the functions declared as scalar fields) computes the
same reduction independently; its output is the oracle for the certificates and also the `sage` level evidence.

Budget and fallback: the generic 4D metric has 10 + 40 + 100 metric jets. If a stage exceeds a build budget of ten
minutes for its file, the Lean theorems for that stage take the 4D metric diagonal (`g = diag(g_0..g_3)`, still
curved and generic enough for every term of the result to appear) and the registry records the fully generic
statement at level `sage`, labelled in the text. The text derives the general case; the labels say which form was
machine-checked.

Alternatives: orthonormal-frame (Cartan) computation (rejected: the asset and recipes are coordinate based, and the
text's primer is coordinate based); verifying only at a flat 4D metric (rejected: loses the `R_4` term and the
coupling of `φ` to curvature, which the text must show).

### D4. Mode expansion and charge quantisation as analysis on the circle

The Kaluza-Klein tower uses Fourier modes on `S^1`. Lean states: the mode functions `e^{i n y / R}` are eigenfunctions
of `∂_y^2` with eigenvalue `-(n/R)^2` (`derivative` recipe, complex exponential), the 5D massless Klein-Gordon
operator acting on a mode gives the 4D operator with mass `|n|/R` (a `ring`-level identity after the eigenvalue
lemma), the periodicity `y ~ y + 2πR` forces `n ∈ ℤ` (mathlib: `Complex.exp_eq_one_iff`), and the charge relation
`q_n = n κ / R` with `κ` fixed by the normalisation of the Maxwell term, as an algebraic identity. The identification
of the Planck mass and the gauge coupling with `R` is an algebraic consequence displayed and proved by `field_simp`.

### D5. Non-abelian Kaluza-Klein: isometry algebra and gauge transformation

The three Killing fields `K_a` of `S^2` are verified Killing by `killing_proof`. Their brackets `[K_a, K_b] =
ε_{abc} K_c` are component identities proved as in D2. The structure constants `ε_{abc}` are then the `su(2)`
constants in `Conventions.lean`, checked against the Pauli-matrix commutators by `lie_structure`. The gauge
transformation of `A^a_μ` under `y -> y + ε^a(x) K_a` is a jet identity to first order in `ε`: the project defines
the transformed metric to first order (jets of `ε`) and proves the induced `δA^a_μ = ∂_μ ε^a + ε_{abc} A^b_μ ε^c`
with `ideal_membership`. The ansatz metric for `M_4 x S^2` with `A^a_μ K_a` is written out, and its inverse is
verified as in D3 step 1, but the full reduction of its curvature scalar is at level `sage` (text labelled) because
the tutorial's point in Part III is the gauge group, not the action.

### D6. CSDR over `SU(2)/U(1)` with gauge group `SU(3)`

Everything is finite-dimensional algebra, fully at level `lean`:

- `su(2)` and `su(3)` as explicit matrices in `Conventions.lean` (Pauli, Gell-Mann), their structure constants by
  `lie_structure`; the embedding `R = U(1) -> SU(3)` as a chosen diagonal generator;
- the reductive decomposition `su(2) = u(1) ⊕ k` with `k` the tangent space at the pole, and the `R` charges of `k`
  (eigenvalues of `ad` of the `U(1)` generator, `matrix_normal_form`);
- the decomposition of the adjoint of `su(3)` under `R` (`representation`), the centraliser `H = SU(2) x U(1)` as the
  kernel of `ad` of the embedded generator (`matrix_normal_form` or `ideal_membership` on the commutation equations);
- the symmetric-field constraints `[W, φ_i] = f_{i}^{j} φ_j` for the two internal components, their general solution
  as a linear system (`witness` for the solution space basis, `ideal_membership` for completeness), yielding a
  complex `SU(2)` doublet;
- the four-dimensional potential `V(φ)` from the internal field strength
  `F_{ij} = [φ_i, φ_j] - f_{ij}^k φ_k`, expanded as a quartic polynomial (`factor_rewrite` to its displayed form), its
  minimum by `sos` or `hint_search`, the vacuum value, the unbroken group `U(1)_{em}` as the centraliser of the
  vacuum, and `sin^2 θ_W` from the generator normalisations (`field_simp`, `norm_num`).

Constraint on the choice: the specific `U(1)` embedding and the result `SU(3) -> SU(2) x U(1) -> U(1)` with a Higgs
doublet is the classic example (Manton 1979, reviewed in arXiv papers to be read; see D8), which gives the reader a
recognisable end point. The embedding and normalisation used are stated in `Conventions.lean` and the conventions
chapter, and any convention difference from the cited papers is recorded in the reading notes.

### D7. Registry and build gate reuse the sibling's conventions

`claims.yaml` follows the sibling format (labels with parts; levels `lean`, `sage`, `text`; `lean` lists theorem
names; `sage` lists `<script>: <PASS label>`). `scripts/check_claims.py` adapts the sibling checker to a multi-file
Quarto book: it collects result labels from all `.qmd` files (Quarto's `#eq-`, `#thm-`, `#prp-` labels), checks
every registry theorem exists through a generated audit file (`KaluzaKlein/Audit.lean` with `#print axioms` per
theorem, run by `lake env lean`, parsed for `sorryAx` and non-standard axioms), checks `sage` evidence against the
output of `make figures-check` (figure scripts print `[PASS]` lines for the exact values they plot), and rejects
prose that overstates verification for `sage`/`text` results. The Makefile orders `lake build` -> checks -> `quarto
render`; Quarto's `pre-render` hook runs `extract_statements.py`, which fails on a missing identifier.

The checker scripts and the Quarto pre-render hook run in the project's virtual environment (`.venv`, created from
`requirements.txt`, which pins PyYAML); Sage is needed only for the figure scripts, the reduction oracles and
certificate authoring. The Lean build needs neither.

Alternative: embedding Lean in Quarto through a Lean code cell engine (rejected: none exists for Quarto; extraction
from source is simpler and the statement text cannot drift).

### D8. Bibliography: arXiv only for content, classics for attribution

`references.bib` starts from the subset of the sibling bibliography relevant here (Forgacs-Manton, Manton 1979,
Kapetanakis-Zoupanos, Kubyshin et al., Witten 1983, the Manousselis-Zoupanos and Chatzistavrakidis et al. papers,
Lean, mathlib, Sage, Veriphy) plus arXiv papers confirmed through the API during planning: gr-qc/9805018 (Overduin,
Wesson), hep-th/9410046 (Duff), 2502.07710 (Duff, Nilsson, Pope, "Kaluza-Klein Supergravity 2025"), 1602.03673
(Manolakos, Zoupanos review), 0708.3222 and 0712.2717 (Chatzistavrakidis, Manousselis, Prezas), hep-ph/0010141
(Manousselis, Zoupanos), 0901.2491 (Dolan, Szabo, reduction over `S^2`), math-ph/0309059 (Kubyshin, geometrical
formalism). Each paper is downloaded to `references/papers/` and read before it is cited; `references/notes/<key>.md`
records what is used and the convention map. Pre-arXiv entries carry `note = {Not on arXiv; cited for attribution}`
and `check_bib.py` enforces that they are never the only citation attached to a displayed result. `check_bib.py`
resolves every arXiv identifier through `https://export.arxiv.org/api/query` (cached in `references/arxiv_cache.json`
so the build does not depend on the network) and reports dangling and unused keys.

### D9. Figures: Sage scripts, SVG for 2D, PNG for 3D

One script per figure in `figures/`, run through `make figures`. Two-dimensional plots (coordinate charts, the mass
tower, the potential, the circle fibre sketch) are saved as SVG; three-dimensional scenes (`S^2` with Killing fields,
the Hopf fibration) are saved as PNG at 300 dpi because Sage's 3D backends do not emit vector output. Each script also
prints `[PASS]` lines for the quantities it plots (for example the potential's minimum), which the registry can
cite. `check_figures.py` verifies every image referenced by the `.qmd` files is produced by a script and vice versa.
PDF rendering needs an SVG converter; the Makefile's version check reports if `rsvg-convert` is missing.

Per-chapter figure plan (each a Sage script; names are the figure files): manifolds `s2_chart`, `stereographic`;
metric `torus_embedding`, `polar_grid`; connection `parallel_transport` (a vector carried around a loop on the
sphere), `geodesics_sphere`; curvature `torus_curvature` (Gauss curvature colour map), `gauss_bonnet`; Killing
`s2_killing`, `killing_flow`; groups `coset_z6`, `so3_orbits`; Lie groups `su2_one_parameter` (the Euler-angle
image of `e^{tT_3}`), `exp_rotation`; representations `su3_weights` (root and weight diagrams), `branching_su3`;
cosets `hopf`, `coset_tangent_plane`, `flag_manifold_charges`; bundles `mobius_band`, `hopf_connection`; field theory
`su2_field_lines`; Higgs `higgs_potential`; relativity `de_sitter`; supersymmetry `wz_potential`; supergravity
`supercharge_table`; abelian KK `kk_circle`, `kk_tower`; non-abelian KK `s2_gauge_fields`; CSDR `csdr_constraints`
(the surviving components), `csdr_potential`; survey `equivariant_potential`, `warp_factor`, `flux_potential`.
`scripts/check_figures.py` also reports chapters without a figure.

### D10. Writing order follows the verification order

Each chapter is written as: derive in text -> state in Lean with `sorry` -> numeric lie detection (Sage, as the
`prove` playbook requires) -> prove (mathlib tactics first, certificates second) -> register -> write the chapter
summary. Text and Lean for a result are produced together so that a statement found false by the numeric check
corrects the text before it is relied on downstream.

### D11. Part V: the survey of compactification methods, one verified computation per method

Part V surveys the methods in the order of the `tutorial-content` spec. Survey chapters are mostly `text`-level,
cited to arXiv reviews that were read (Graña hep-th/0509003 and Douglas-Kachru hep-th/0610102 for fluxes; Greene
hep-th/9702155 for Calabi-Yau; Acharya-Gukov hep-th/0409191 for `G_2`; Chatzistavrakidis-Zoupanos 0905.2398 for
nearly-Kähler; Chatzistavrakidis-Steinacker-Zoupanos 1002.2606 for orbifolds and fuzzy spheres; Cvetic-Lu-Pope
hep-th/0003286 and Hohm-Samtleben 1410.8145 for consistent sphere truncations; Hull-Reid-Edwards hep-th/0503114 and
Kaloper-Myers hep-th/9901045 for Scherk-Schwarz; Randall-Sundrum hep-ph/9905221, hep-th/9906064 and
Arkani-Hamed-Dimopoulos-Dvali hep-ph/9803315 for warped and large extra dimensions; Duff-Nilsson-Pope 2502.07710 for
Freund-Rubin and the consistency history; Wecht 0708.3984 for non-geometric fluxes). Each chapter carries one
`lean`-level worked computation chosen to be algebraic:

| method | verified computation | recipe |
|---|---|---|
| Scherk-Schwarz on `S^1` | twisted mode expansion: masses `|n/R + ω|` and the charge shift | `derivative`, `ring` |
| group manifolds | Maurer-Cartan equations `dσ_a = −½ ε_{abc} σ_b ∧ σ_c` for the `SU(2)` left-invariant forms in Euler angles | `coordinate_geometry` (`extDeriv`) |
| Freund-Rubin | the 11D field equations on `AdS_4 × S^7` reduce to algebraic identities given `R_{μν} = −12m² g`, `R_{mn} = 6m² g`, `F = 3m ε` | `ring`, `matrix_identity` |
| equivariant (Dolan-Szabo) | the `m = 1` potential's minimum and vacuum value | `sos`, `factor_rewrite` |
| Calabi-Yau | field counting from Hodge numbers (generations `|h^{1,1} − h^{2,1}|`) on a stated example | `norm_num` |
| fluxes | Dirac quantisation of a flux on a two-cycle and the resulting potential's extremum in a one-modulus toy model | `witness`, `derivative` |
| Randall-Sundrum | the warp factor `e^{−2k|y|}` solves the bulk Einstein equations with `k² = −Λ/(24 M³)` away from the branes | `derivative`, `field_simp` |

Supergravity-specific structures (Killing spinors, Lichnerowicz operators, the `S^7` consistency proof) are `text`
and cited; their equations are displayed, not re-derived.

### D12. The foundations part also teaches field theory, relativity, supersymmetry and supergravity

Five chapters follow the geometry chapters in Part I. Their verified computations:

| chapter | verified computation | recipe |
|---|---|---|
| field theory and gauge invariance | Noether current conservation for a complex scalar as a polynomial identity in jets; the field strength and Bianchi identity of an explicit `SU(2)` configuration in components | `ideal_membership`, `coordinate_geometry` (`extDeriv`, `wedge`) |
| Yang-Mills and the Higgs mechanism | the quartic potential's minimum and the mass matrix of the abelian and `SU(2)` Higgs models | `sos`, `factor_rewrite`, `matrix_normal_form` |
| general relativity | Einstein tensor of explicit metrics: it vanishes identically in two dimensions (`S^2`), and the de Sitter metric solves `G_{μν} + Λ g_{μν} = 0` | `coordinate_geometry` through Christoffel lemmas |
| supersymmetry | Clifford algebra and the N=1 superalgebra anticommutators on explicit `4 × 4` gamma matrices; the Wess-Zumino action's invariance up to a total derivative as a jet identity | `operator_identity`, `matrix_identity`, `ideal_membership` |
| supergravity | eleven-dimensional gamma-matrix identities used by the Freund-Rubin analysis; the gravitino variation and the 11D action are displayed and labelled `text` | `operator_identity` |

Sources, to be downloaded and read with reading notes like every other citation, identifiers confirmed through
the arXiv API at apply time: Carroll's general relativity lecture notes (gr-qc/9712019), Martin's supersymmetry
primer (hep-ph/9709356), Tanii's supergravity lectures (hep-th/9802138), and one arXiv set of gauge-theory or
quantum-field-theory lecture notes located by search.

### D13. Pedagogical format: definitions first, many examples, exercises with folded solutions

- Every term is defined at first use (Quarto definition blocks `::: {#def-...}`), and a glossary appendix lists
  every definition with its chapter.
- Examples are Quarto example blocks (`::: {#exm-...}`), several per definition where the material allows.
- Exercises are Quarto exercise blocks (`::: {#exr-...}`); the solution follows in a collapsed callout
  (`::: {.callout-note collapse="true" title="Solution"}`) and may contain a theoretical derivation, Sage code with
  its output, a Lean theorem shown with the `lean` shortcode, or all three.
- Sage solutions are committed scripts `sage/exercises/<exr-label>.sage`; `make exercises` runs them and writes
  `sage/exercises/<exr-label>.out`; the solution callout shows the script and its output through the shortcode
  `{{< sage <exr-label> >}}` (`tutorial/exercise.lua`), which reads both committed files at render time, so the
  shown output is the recorded one (Quarto expands `include` directives before pre-render hooks run, so a
  shortcode is the mechanism that works on a fresh checkout). `scripts/check_exercises.py`
  verifies that every exercise has a solution callout, that every Sage solution script has a current output file,
  and that every Lean solution names an existing theorem.
- Lean solutions are theorems of the library in `KaluzaKlein/Exercises/<Part>.lean`, registered in the claims
  registry under the exercise label.

### D14. Group theory, representations and coset spaces: three chapters, verified algebraically

The mathematics part gains `part-i-groups.qmd` (groups, cosets, quotients, homomorphisms, representations, Schur's
lemma), `part-i-representations.qmd` (`su(2)` and `su(3)` representations, roots, weights, tensor products,
branching rules) and `part-i-cosets.qmd` (the coset chapter required by the spec); the existing Lie groups chapter
stays between them. Their verified computations, in `KaluzaKlein/Groups/*.lean`:

| chapter | verified computation | recipe |
|---|---|---|
| groups | the coset decomposition of a finite group and the quotient's multiplication table (`Z_6/Z_3`, `S_3/A_3`) | `decide` on mathlib's finite groups |
| groups | Schur's lemma on an explicit irreducible `su(2)` representation: a matrix commuting with the spin-1 generators is a scalar | `ideal_membership` / `matrix_normal_form` (kernel of the commutation equations) |
| representations | Cartan matrix and positive roots of `su(3)`; the weights of the adjoint; the branchings `8 → 3₀ + 1₀ + 2₃ + 2₋₃` under `su(2)×u(1)` and `8 → 2·(1,0) + ...` under `u(1)×u(1)` | `lie_structure`, `representation` |
| representations | `so(4) ≅ su(2) ⊕ su(2)`: the two commuting `su(2)` subalgebras from `M_{ij}` | entrywise over `ℚ` |
| cosets | `S³ = SO(4)/SO(3)` decomposition; `SU(3)/U(1)×U(1)` charges and non-symmetry; `S² = SU(2)/U(1)` charges | entrywise over `ℚ` and `ℂ` (done: `Coset.lean`) |
| cosets | the coset vielbein of `S²` reproduces the round metric and `e³ = cos θ dφ` | `sage` (Maurer-Cartan components), `text` for the general construction |

Part numbering: the physics foundations form Part II (`part-ii-*.qmd`), the Kaluza-Klein parts become III and IV,
CSDR V and the survey VI; prose and section identifiers are renumbered accordingly.

### D15. Sage modules, one-definition blocks, and the bottom-up physics part

- **Sage modules.** The shared helper `sage/exercises/_curvature.py` and every script that computes connections or
  curvature use SageManifolds (`Manifold(n, 'M', structure='Riemannian'|'Lorentzian')`, `chart`, `metric`,
  `christoffel_symbols_display`, `riemann`, `ricci`, `ricci_scalar`, vector fields and their brackets, `lie_derivative`),
  so the oracle and the figures come from the same objects the text names. Algebra scripts use `LieAlgebra`,
  `RootSystem`, `WeylCharacterRing` (branching rules via `branch`, tensor products via `*`), `SymmetricGroup`,
  `Zmod`, `MatrixGroup`. Hand-coded formulas remain only where Sage has no module (coset vielbeins, jet identities).
- **One definition per block.** Definition blocks are split so that each `def-` block defines one term; each is
  followed by at least three `exm-` blocks and, where drawable, a figure (per-chapter figure plan of D9 extended to
  definition-level figures: charts, tangent vectors, one-forms, flows, cosets, orbits, weights, potentials).
- **Bottom-up physics part.** Part II becomes nine chapters: `part-ii-mechanics` (Lagrangian and Hamiltonian
  mechanics, Noether's theorem for particles, the harmonic oscillator and the particle on a sphere as examples);
  `part-ii-fields` (the field objects: real scalar, complex scalar, vector field and one-form, spinor fields and
  gamma matrices, fields on curved space, with figures of field configurations); `part-ii-field-theory` (actions,
  Euler-Lagrange equations, Noether currents for fields); `part-ii-gauge` (gauge invariance, covariant derivative,
  Yang-Mills); `part-ii-qft` (quantisation, particles as modes, renormalisation and anomalies in outline);
  `part-ii-higgs`; `part-ii-general-relativity`; `part-ii-supersymmetry`; `part-ii-supergravity`. The mathematics
  part keeps its order (manifolds, metric, connection, curvature, Killing, groups, Lie groups, representations,
  cosets, bundles), each chapter checked for component-before-composite order.
- Verified computations are unchanged in content; `KaluzaKlein/Foundations/Mechanics.lean` adds the Noether
  charge of the free particle and the energy of the harmonic oscillator as polynomial identities, and
  `KaluzaKlein/Foundations/Fields.lean` the Clifford relations of explicit gamma matrices (moved forward from the
  supersymmetry chapter).

### D16. Chapter map: small single-topic chapters built on primer chapters

Mathematics (Part I), in order, one file each (`part-i-NN-slug.qmd`), at most six definitions per chapter; section
identifiers are number-free slugs (`{#sec-topology}`, `{#sec-charts}`), so that inserting a chapter renumbers files
but never breaks a cross-reference:
sets and maps (set, map, injective, surjective, bijection, inverse map); relations, quotients and products
(composition, equivalence relation, equivalence class, quotient set, Cartesian product); topological spaces (open
sets in `ℝ^n`, topology, subspace topology, continuous map, homeomorphism, quotient topology); compact and connected
spaces (Hausdorff, compact, connected, path-connected); vector spaces (vector space, linear combination, span, independence,
basis, dimension); linear maps and dual spaces (linear map, functional, dual space, dual basis); bilinear forms (bilinear,
symmetric, non-degenerate, signature); multilinear maps and tensor products (abstract); calculus on `ℝ^n` (partial derivatives, Jacobian, inverse function theorem, smooth maps); charts and
atlases; manifolds; embeddings and Gram matrices; curves and tangent vectors; vector fields; cotangent vectors and
one-forms; tensor fields; the metric; induced metrics and the running examples; the covariant derivative; the
Levi-Civita connection; geodesics and parallel transport; the Riemann tensor; Ricci tensor and scalar curvature;
flows and the Lie derivative; Killing vectors; groups; cosets and quotient groups; homomorphisms; representations
and Schur's lemma; Lie groups; Lie algebras; the exponential map; Maurer-Cartan forms; weights and `su(2)`
representations; roots and `su(3)`; branching rules; coset spaces; reductive decompositions and isotropy; coset
vielbeins; differential forms; fibre bundles; connections on bundles; the Hopf fibration.
Physics (Part II): Lagrangian mechanics; Hamiltonian mechanics; Noether's theorem for particles; scalar fields;
complex and vector fields; spinor fields and gamma matrices; actions and Euler-Lagrange equations; Noether
currents; abelian gauge invariance; Yang-Mills theory; quantisation in outline; spontaneous symmetry breaking; the
Higgs mechanism; the Einstein-Hilbert action; the Einstein equations and de Sitter space; the superalgebra; the
Wess-Zumino model; the superpotential and gauge multiplets; local supersymmetry and the gravitino;
eleven-dimensional supergravity. `scripts/check_definitions.py` enforces the six-definition limit and reports
chapters by word count; the review table of task 3.20 records the primitives each definition relies on.

## Risks / Trade-offs

- [Generic-metric jet identities too large for `linear_combination` or `ring`] → staged theorems (D3), diagonal
  fallback with the generic case at level `sage`, labelled; build budget of ten minutes per Lean file.
- [Asset definitions (`pd` through `deriv` along `Function.update`, `Matrix.inv`) make curvature proofs slow] →
  state Christoffel symbols as intermediate lemmas and rewrite with them, as the recipe documents; chart-domain
  hypotheses are stated explicitly.
- [Cited reviews use different conventions (signature, generator normalisation, `F` sign)] → conventions chapter
  fixed first, every reading note carries a convention map, and the `bibliography` spec requires the difference to be
  stated next to the citation.
- [Classic CSDR sources not on arXiv] → attribution only; the derivations cite arXiv reviews that were read.
- [Sage 3D figures are raster] → PNG at 300 dpi, accepted; 2D figures vector.
- [Render depends on network for arXiv checks] → cached resolution file committed; the check uses the cache unless
  `--refresh` is passed.
- [Veriphy gaps (for example no vector-field bracket in the asset)] → small project-level definitions in
  `Conventions.lean`, recorded as building-block gaps in the registry notes for upstream.
- [Scope creep into the full non-abelian reduction of the action] → Part III verifies the gauge group and
  transformation in Lean and labels the action reduction `sage` (D5).

### D17. Part names: foundations, not disciplines

The two foundation parts are titled "Foundations of mathematics" and "Foundations of physics" (Quarto part titles
and every "Part I"/"Part II" mention in prose that names them). The titles say what the parts are for: the
mathematics and physics *needed* for the reduction, not surveys of the disciplines.

### D18. Chapter introductions stand on their own

Each chapter introduction motivates its topic on the topic's own terms (what the objects are, why they matter,
what the reader will be able to compute at the end) and names the sources. It does not tie the topic to
Kaluza-Klein theory or CSDR; the connection to the main theme is made once, in the part introduction and in the
chapter where the object is used. Rationale: forty introductions that each end in "and this returns in Part V"
read as repetition (user feedback, October 2026).

### D19. Two levels of organisation inside a part

Quarto books have parts and chapters only, so the second level is realised as follows: each Quarto **chapter
file** is a thematic group (for example "Sets, relations and topology", "Linear algebra", "Charts and manifolds",
"Curvature", "Mechanics", "Gauge theories"), and the former small chapters become its **sections** (`##`, with
their number-free identifiers unchanged, so every cross-reference survives). The group file has a short
introduction of its own (D18), the sections keep the format of D13/D15/D16 (at most six definitions, each with its
examples and exercises, a Sage figure per section), and the group ends with the integrative exercises of D21.
`scripts/check_definitions.py` and `scripts/check_figures.py` count per section (`## ` heading) instead of per
file; `scripts/check_exercises.py` checks the per-definition exercises of D21 and the three end-of-group
exercises. Alternative rejected: making the groups Quarto parts and dropping the six-part level (the reader loses
the mathematics/physics/KK/CSDR/survey structure).

### D20. Colour coding of block types

A stylesheet `tutorial/styles.css` (HTML) and matching PDF environments mark the cross-reference blocks with
**three colours on the block titles only**: no vertical bars, no background highlight, and plain solution callouts;
a legend sits in the conventions chapter (`sec-conventions-blocks`):

| block | identifier prefix | marking |
|---|---|---|
| statement: definition, theorem, lemma, proposition, corollary | `def-`, `thm-`, `lem-`, `prp-`, `cor-` | red title (rgb 192, 57, 43) |
| example | `exm-` | green title (rgb 30, 132, 73) |
| exercise | `exr-` | blue title (rgb 31, 95, 191) |

(Revised from an earlier five-colour scheme with one bar per statement kind and a bar on exercises, at the author's
request: one colour per kind, on the title alone, in the order R, G, B, and no vertical bars anywhere, not even on
solutions.)

Rationale: one colour per kind of block so that a page can be scanned, and nothing heavier than a coloured title so
that a page is not a wall of boxes (author's feedback, October 2026). Definitions are never stacked (D15).

### D21. Examples and exercises per definition, and integrative exercises per chapter

Every definition, theorem and proposition is followed by (i) at least three examples of increasing complexity,
the first a direct instance, the last a case that connects to a later object, and (ii) at least three exercises of
increasing difficulty (compute, derive, generalise) each with a collapsed solution. Examples include a figure or
plot where the object can be drawn, and an **interactive** figure where feasible: Sage's threejs viewer for 3D
scenes (`p.save("name.html")`, interactive in HTML output) and Plotly.js from a CDN fed with JSON data written by
the Sage script for 2D plots, included through a Lua shortcode `{{< interactive name >}}` with the static SVG/PNG as
the PDF fallback. Each chapter (group file of D19) ends with three integrative exercises that use material from
several of its sections, with collapsed solutions. `scripts/check_exercises.py` enforces the counts.

The counts are a **minimum** (author's instruction, October 2026): three examples and three exercises per
definition, theorem and proposition is the floor the checker enforces, not a target. Where a notion is central, has
several standard cases, or a textbook offers good problems (D32), a block carries more, and a chapter may end with
more than three integrative exercises.

### D22. Veriphy throughout

Veriphy is the verification tool of the tutorial and its use is made visible everywhere:

- every definition that has a formal counterpart is mirrored by a Lean definition in `KaluzaKlein/`, displayed
  under the definition block with the `lean` shortcode (the asset definitions `pd`, `christoffel`, `riemann`,
  `extDeriv`, … for the geometric ones);
- every theorem and proposition at level `lean` is proved with the Veriphy recipe that fits it
  (`coordinate_geometry`, `ideal_membership`, `matrix_identity`, `lie_structure`, `witness`, `sos`,
  `factor_rewrite`, `calc_derivation`, `derivative`, `representation`, `matrix_normal_form`, `operator_identity`)
  whenever one applies, with the certificate committed under `.veriphy/certificates/` and the recipe named in the
  block ("verified with Veriphy's `ideal_membership`"); plain mathlib tactics are used only where no recipe
  applies, and the registry records which;
- exercises whose answer is a computation carry a Lean statement verified the same way, displayed in the
  solution; the solution text names the recipe;
- the registry appendix gains a column "Veriphy recipe" and a table of how many results each recipe verified;
  gaps found (D7) stay in `claims.yaml` and are reported upstream.
Rationale: the tutorial doubles as a showcase of Veriphy, so the reader must see the tool at work, not only the
mathlib tactics it drives.

### D23. Metric signature `(+, -, -, -)`

The book uses the particle-physics signature `(+, -, -, -)` throughout (author's decision, October 2026; it is the
signature of Overduin-Wesson, Tanii and the CSDR reviews, while Duff and Martin use `(-, +, +, +)`). Consequences,
recorded in `tutorial/conventions.qmd` and applied to every chapter, Lean statement and Sage script:
`η = diag(1, -1, -1, -1)`, timelike means `η(v, v) > 0`, Lorentzian signature `(1, n - 1)`; internal metrics are
negative definite inside the full metric (`dŝ² = g - φ²(dy + A)²`, `g - R²(dθ² + sin²θ dφ²)`), while Part I's
Riemannian round metrics stay positive and the internal block is `h = -γ`; `□ = ∂_t² - ∇²`, Klein-Gordon
`(□ + m²)φ = 0`, `L = ½ ∂φ·∂φ - V`, `L = -¼ F²` (signature independent), Dirac `(iγ·∂ - m)ψ = 0` with
`{γ^μ, γ^ν} = 2η^{μν}` and the standard Weyl matrices without the factor `i`, Einstein-Hilbert `-(1/16πG) √(-g) R`;
the circle reduction reads `R̂ = R + ¼ φ² F² - 2 φ⁻¹ □φ` and the sphere reduction `R̂ = R - R_γ + ¼ γ_{mn} K^m_a
K^n_b F^a F^b`. Conversion rule for quoted formulas: `g → -g` leaves `Γ`, Riemann, Ricci and `F_{μν}F^{μν}`
unchanged and flips `R`, `□` and every contraction with an odd number of inverse metrics.

### D24. The complete two-level table of contents of the foundation parts

The two foundation parts are organised first and written second (author's instruction, October 2026): their group
chapters (D19) and sections are fixed here, scaffolded as files with section headings and one-line scope statements,
and only then filled in group by group. Existing chapters keep their identifiers and are moved into the groups
below; new groups cover the analysis, probability, mechanics, thermodynamics, quantum mechanics and quantum field
theory that the physics parts and the later quantum-field-theoretic material need. Each section defines at most six
notions (D16), each with examples and exercises (D21), and each group has a Sage module, a figure per section, and
the verified computations listed (recipes of D22 named; `text` where a theorem is quoted).

**Part I, Foundations of mathematics** (group file → sections; `[new]` marks sections to be written):

| # | group file | sections (former chapter identifiers kept) | verified computations and recipes | Sage |
|---|---|---|---|---|
| 1 | `part-i-01-sets-relations-topology` (done) | sets and maps; relations, quotients, products; topological spaces; compact and connected spaces | finite counts (`decide`) | `Set`, `Subsets` |
| 2 | `part-i-02-linear-algebra` (done) | vector spaces; linear maps and dual spaces; bilinear forms; multilinear maps and tensor products | `witness`, `ideal_membership`, `matrix_from_goal`, `sos` | `VectorSpace`, `FiniteRankFreeModule` |
| 3 | `part-i-03-real-analysis` | [new] real numbers, sequences and limits (completeness, sup/inf, Cauchy sequences, limits); [new] series (convergence tests, absolute convergence, power series, radius of convergence); [new] continuity and differentiability in one variable (limits of functions, continuity, derivative, mean value theorem, Taylor's theorem); [new] the Riemann integral (integral, fundamental theorem, improper integrals, Gaussian integral); calculus on `ℝⁿ` (`sec-calculus`: partial derivatives, Jacobian, chain rule, inverse function theorem, smooth maps); [new] multiple integrals and convergence of functions (change of variables, uniform convergence, differentiation under the integral) | `derivative`, `certified_bound` (`e`, `π`, `log 2` bounds), `sum_induction` (partial sums), `transcendental_rewrite`, mathlib `integral_gaussian`; Taylor remainders as `rational_inequality` | `limit`, `taylor`, `integrate`, `sum`, `plot` |
| 4 | `part-i-04-complex-analysis` | [new] complex numbers and functions (`ℂ`, modulus and argument, exponential and logarithm, branches); [new] holomorphic functions (complex derivative, Cauchy-Riemann, harmonic functions, conformal maps); [new] contour integration (contour integral, Cauchy's theorem, Cauchy's integral formula, Liouville); [new] series and residues (Taylor and Laurent series, poles, residue theorem, real integrals by residues); [new] analytic continuation and special functions (Gamma and Beta functions, the zeta function in outline, Wick rotation as a contour deformation) | `transcendental_rewrite` (exp/log identities), `root_isolation` (zeros), `matrix_from_goal` (Cauchy-Riemann as a linear condition), residues of rational functions by `factor_rewrite`/`ideal_membership`, `Complex.exp` identities in mathlib; Gamma recursion `Γ(n+1) = n!` (`sum_induction`) | `residue`, `laurent`, `complex_plot`, `gamma` |
| 5 | `part-i-05-differential-equations` | [new] ordinary differential equations (ODE, initial value problem, existence and uniqueness, separable and linear equations, phase portraits); [new] linear systems and the matrix exponential (linear systems, matrix exponential, stability, the harmonic oscillator, resonance); [new] boundary value and Sturm-Liouville problems (eigenvalue problem, orthogonality, completeness in outline, Legendre and Bessel equations); [new] partial differential equations (classification, wave, heat and Laplace equations, separation of variables, d'Alembert's solution); [new] Green's functions (fundamental solution, Green's function of the Laplacian and of `□ + m²`, retarded and Feynman prescriptions in outline) | `ode_solution`, `matrix_exponential`, `derivative`, `matrix_normal_form` (eigenvalues of linear systems), `calc_derivation`; plane-wave solutions already proved (`planeWave_klein_gordon`) | `desolve`, `desolve_system`, `exp` of matrices, `plot_vector_field` |
| 6 | `part-i-06-fourier-analysis` | [new] Fourier series (periodic functions, Fourier coefficients, orthogonality of modes, Parseval, convergence in outline); [new] the Fourier transform (transform and inverse, Gaussian, convolution, Plancherel, uncertainty inequality); [new] distributions (test functions, the delta function, derivatives of distributions, Fourier transform of `δ`, principal value); [new] harmonic analysis on the sphere and on compact groups (spherical harmonics as eigenfunctions of the Laplacian, `Y_lm` and `SU(2)` representations, Peter-Weyl in outline, the mode expansion on a coset space) | orthogonality integrals and `∫ e^{iny} e^{-imy} = 2π δ_{nm}` (mathlib `integral` lemmas, `transcendental_rewrite`), Parseval for finite sums (`sum_induction`), spherical harmonics eigenvalue equations by `coordinate_geometry` (Laplacian on `S²`), Gaussian transform by `integral_gaussian` | `fourier_series_partial_sum`, `fourier_transform`, `spherical_harmonic` |
| 7 | `part-i-07-hilbert-spaces-operators` | [new] inner product and Hilbert spaces (inner product, norm, orthonormal bases, `ℓ²` and `L²`, completeness, Riesz representation in outline); [new] bounded operators (operator, norm, adjoint, self-adjoint, unitary, projection, commutator); [new] spectral theory (eigenvalues and eigenvectors, spectrum, spectral theorem for Hermitian matrices and compact self-adjoint operators, functional calculus `f(A)`, `exp(iA)`); [new] unbounded operators and the Laplacian (domains, position and momentum, self-adjoint extensions in outline, Laplacian on `S¹`, `S²`, `T²` with its spectrum, the Kaluza-Klein spectrum as a spectral statement); [new] traces, determinants and heat kernels (trace class, `tr e^{-tΔ}`, zeta regularisation and the Casimir energy in outline) | `matrix_normal_form` (spectral decomposition of explicit Hermitian matrices), `matrix_exponential` (`exp(iA)` unitary), `operator_identity` (commutators `[x, p] = i`, ladder operators), `physlib/ladder_operators`, `lie_structure`; Laplacian eigenvalues `n²/R²`, `l(l+1)/R²` by `coordinate_geometry` and `derivative`; `sum_induction` for finite traces | `matrix` methods (`eigenmatrix_right`, `exp`), `spectrum`, `laplacian` on manifolds |
| 8 | `part-i-08-probability` | [new] probability spaces (sample space, events, probability measure, conditional probability, independence, Bayes); [new] random variables and distributions (random variable, distribution, expectation, variance, moments, characteristic function, Gaussian and Poisson distributions); [new] limit theorems (law of large numbers, central limit theorem, Gaussian integrals and Wick's theorem for moments); [new] statistics and estimation (sample mean, estimators, maximum likelihood, confidence intervals, the chi-squared test, Monte Carlo sampling) | finite probability by `decide`/`norm_num`, Gaussian moments `⟨x^{2n}⟩ = (2n-1)!! σ^{2n}` (`sum_induction`, `integral_gaussian`), variance identities (`ideal_membership`), Chebyshev as `rational_inequality`, Wick pairings count `(2n-1)!!` (`decide`) | `RealDistribution`, `random`, `moment`, Monte Carlo scripts |
| 9 | `part-i-09-charts-manifolds` | charts and atlases (`sec-charts`); manifolds (`sec-manifolds`); embeddings and Gram matrices (`sec-embeddings`) | existing (`coordinate_geometry`, `derivative`) | SageManifolds charts |
| 10 | `part-i-10-vector-fields-forms-tensors` | curves and tangent vectors; vector fields; cotangent vectors and one-forms; tensor fields; differential forms (`sec-forms`) | existing (`coordinate_geometry`, `extDeriv`, `wedge`) | SageManifolds |
| 11 | `part-i-11-metrics` | the metric; lengths and volumes; induced metrics | existing | SageManifolds |
| 12 | `part-i-12-connections-curvature` | covariant derivative; Levi-Civita connection; geodesics; Riemann tensor; Ricci tensor and scalar curvature | existing (`coordinate_geometry`) | SageManifolds |
| 13 | `part-i-13-flows-killing` | flows and the Lie derivative; Killing vectors | existing | SageManifolds |
| 14 | `part-i-14-groups` | groups; cosets and quotient groups; homomorphisms; representations and Schur's lemma | existing (`decide`, `ideal_membership`) | `SymmetricGroup`, `Zmod` |
| 15 | `part-i-15-lie-groups-algebras` | Lie groups; Lie algebras; the exponential map; Maurer-Cartan forms | existing (`lie_structure`, `matrix_exponential`) | `LieAlgebra`, `MatrixGroup` |
| 16 | `part-i-16-representations` | weights and `su(2)`; roots and `su(3)`; branching rules | existing (`representation`, `lie_structure`) | `WeylCharacterRing` |
| 17 | `part-i-17-coset-spaces` | coset spaces; reductive decompositions; coset vielbeins | existing | hand-coded vielbeins |
| 18 | `part-i-18-bundles` | fibre bundles; connections on bundles; the Hopf fibration | existing | SageManifolds |

**Part II, Foundations of physics** (same conventions):

| # | group file | sections | verified computations and recipes | Sage |
|---|---|---|---|---|
| 1 | `part-ii-01-classical-mechanics` | [new] Newtonian mechanics (force, momentum, energy, conservation laws, central forces and Kepler's problem); Lagrangian mechanics (`sec-lagrangian`); Hamiltonian mechanics (`sec-hamiltonian`); [new] Poisson brackets and canonical transformations (Poisson bracket, canonical transformation, generating function, Liouville's theorem, Hamilton-Jacobi in outline); Noether's theorem for particles (`sec-noether-particles`); [new] small oscillations and normal modes (equilibrium, normal modes, coupled oscillators, the particle on a sphere and on `SU(2)`) | existing (`ideal_membership` jet identities) plus Poisson bracket identities (`operator_identity` on polynomial brackets, `ideal_membership`), normal-mode frequencies (`matrix_normal_form`), Kepler orbits (`ode_solution`) | `desolve`, `plot`, `matrix` |
| 2 | `part-ii-02-thermodynamics-statistical-mechanics` | [new] thermodynamics (state variables, the laws, entropy, temperature, thermodynamic potentials, Maxwell relations); [new] statistical ensembles (microstates, microcanonical, canonical and grand-canonical ensembles, partition function, Boltzmann distribution); [new] partition functions of simple systems (two-level system, harmonic oscillators, ideal gas, equipartition); [new] quantum statistics (Bose-Einstein and Fermi-Dirac distributions, blackbody radiation, Bose condensation in outline); [new] phase transitions and Landau theory (order parameter, Landau free energy, spontaneous symmetry breaking, mean field, critical exponents in outline); [new] finite temperature as a compact Euclidean time (the thermal circle, Matsubara modes, the analogy with the Kaluza-Klein circle) | Maxwell relations as `ideal_membership` on the jets of a potential, partition sums (`sum_induction`, geometric series), the oscillator free energy (`transcendental_rewrite`), Landau minimum (`sos`, `factor_rewrite`), Stefan-Boltzmann integral `∫ x³/(e^x - 1) = π⁴/15` labelled `text`, Matsubara mode periodicity (`periodic_iff_int` reused) | `sum`, `plot`, `maxima` symbolic integrals |
| 3 | `part-ii-03-quantum-mechanics` | [new] states and observables (state vector, Hilbert space of states, observable as self-adjoint operator, expectation value, uncertainty); [new] measurement and the Schrödinger equation (Born rule, projection, time evolution, Schrödinger equation, stationary states, the free particle and the particle in a box); [new] the harmonic oscillator (ladder operators, spectrum, coherent states); [new] angular momentum and spin (angular momentum algebra, `SU(2)` representations as spin, spin-½ and Pauli matrices, addition of angular momenta); [new] symmetries in quantum mechanics (unitary representations, Wigner's theorem in outline, conserved quantities, the hydrogen atom spectrum from `so(4)` in outline); [new] perturbation theory and the path integral (time-independent perturbation theory, the propagator, Feynman's path integral, the classical limit) | ladder-operator commutators and spectrum (`physlib/ladder_operators`, `operator_identity`), Pauli algebra (`matrix_identity`, `lie_structure`), particle-in-a-box eigenfunctions (`derivative`, `transcendental_rewrite`), uncertainty inequality for Gaussians (`rational_inequality`), free propagator Gaussian (`integral_gaussian`), first-order energy shifts (`ideal_membership`) | `matrix` (ladder operators truncated), `spin` representations, `plot` wavefunctions, `desolve` |
| 4 | `part-ii-04-fields` | scalar fields; complex and vector fields; spinor fields and gamma matrices | existing (`matrix_identity` Clifford, `derivative`) | existing |
| 5 | `part-ii-05-actions-symmetries` | actions and Euler-Lagrange equations; Noether currents | existing (`ideal_membership` jet identities) | existing |
| 6 | `part-ii-06-gauge-theories` | abelian gauge invariance; Yang-Mills theory | existing (`coordinate_geometry`, `ideal_membership`) | existing |
| 7 | `part-ii-07-quantum-field-theory` | quantisation (`sec-quantisation`, extended); [new] canonical quantisation of the free scalar (field operators, canonical commutators, mode expansion, Fock space, particles as modes); [new] propagators and Green's functions (Feynman propagator, Wick rotation, Euclidean correlators, the Klein-Gordon Green's function from Part I); [new] the path integral and Euclidean field theory (generating functional, Gaussian path integrals, Wick's theorem, the thermal circle revisited); [new] perturbation theory and Feynman diagrams (interactions, Feynman rules in outline, the one-loop correction in outline); [new] renormalisation, effective actions and anomalies (regularisation, running couplings, effective field theory, the chiral anomaly, all in outline with citations; what a compact extra dimension does to the spectrum) | ladder/field commutators (`physlib/ladder_operators`, `operator_identity`), Gaussian integrals and Wick pairings (`integral_gaussian`, `decide`), the Feynman propagator residue as a rational-function residue (`factor_rewrite`), the one-loop Gamma-function identities (`transcendental_rewrite`), mode sums (`sum_induction`); renormalisation and anomalies `text` | `matrix`, `residue`, `integrate`, `sum` |
| 8 | `part-ii-08-symmetry-breaking` | spontaneous symmetry breaking; the Higgs mechanism | existing (`sos`, `factor_rewrite`, `matrix_normal_form`) | existing |
| 9 | `part-ii-09-general-relativity` | Einstein-Hilbert action; Einstein equations and de Sitter space | existing (`coordinate_geometry`) | existing |
| 10 | `part-ii-10-supersymmetry` | the superalgebra; the Wess-Zumino model; superpotential and gauge multiplets | existing (`matrix_identity`, `ideal_membership`) | existing |
| 11 | `part-ii-11-supergravity` | local supersymmetry and the gravitino; eleven-dimensional supergravity | existing (`operator_identity`) | existing |

Placement rationale: real and complex analysis, differential equations, Fourier analysis and Hilbert-space theory
form the analysis block of Part I between linear algebra and manifolds, each using only the ones before it
(manifolds need calculus; Fourier analysis needs integrals and series; spectral theory needs Fourier analysis for
its examples; probability needs integrals and the Gaussian). Operator and spectral theory belong to mathematics
(group 7) and are *used* in quantum mechanics (Part II group 3) and in the Kaluza-Klein spectrum (the Laplacian on
the internal space, Part III); Fourier decomposition is mathematics (group 6) and is used for mode expansions
(Parts III-IV) and for Matsubara modes (Part II group 2). Probability precedes statistical mechanics and the path
integral. In Part II, mechanics comes first, thermodynamics second (it needs only mechanics and probability),
quantum mechanics third (it needs Hilbert spaces and spectral theory), fields fourth, and quantum field theory
after gauge theories (it quantises the fields and gauge fields just defined) and before symmetry breaking (which is
discussed classically and quantum-mechanically). Alternatives rejected: putting operator theory inside quantum
mechanics (it is mathematics and is also needed for the Kaluza-Klein spectrum); a single "mathematical methods"
chapter (too many definitions for one group; the six-definition-per-section rule would be violated).
Sources: arXiv lecture notes located and read at apply time (candidates: Tong's lecture notes series is not on
arXiv and is cited only for attribution; arXiv sets on QFT, statistical mechanics and quantum mechanics are to be
found by search and recorded in `references/notes/`).

### D25. Connective prose between blocks

Blocks are never stacked. Every definition, theorem, proposition, lemma, corollary, example and exercise block is
introduced by **two or three sentences of running text** that say what is coming and why: the question a definition
answers and the picture to keep in mind; what an example illustrates and how it goes beyond the previous one; what
an exercise asks the reader to practise or to discover. The prose is part of the exposition, not a label ("Here is an
example"); it carries the argument from one block to the next, so that a reader who skips the blocks still follows
the thread. Solution callouts belong to their exercise and need no introduction; figures and their captions,
shortcodes and `###` headings do not count as prose. `scripts/check_format.py` enforces it: between the end of a
block (after its solution callout, for an exercise) and the start of the next block in the same section there are at
least two sentences and at least twenty words of prose. Rationale: author's feedback (October 2026) that the
block-after-block format "reads too dry". Applies to every chapter already written (groups 1 and 2 of Part I are
revised) and to every chapter to come.

### D26. Group chapters in Parts III–VI and the sidebar

The two-level organisation of D19 applies to every part, not only the foundations. Parts III–VI are merged into
group chapters, the former chapters becoming `##` sections with their identifiers unchanged:

| part | group chapter | sections (former chapters) |
|---|---|---|
| III Abelian Kaluza-Klein theory | `part-iii-01-kaluza-klein-reduction` "The Kaluza-Klein reduction" | ansatz; reduction of the curvature scalar; Einstein frame |
| | `part-iii-02-gauge-tower` "Gauge symmetry and the Kaluza-Klein tower" | gauge invariance from the fifth dimension; the tower |
| IV Non-abelian Kaluza-Klein theory | `part-iv-01-isometries-gauge-fields` "Isometries and gauge fields" | the ansatz; the non-abelian gauge transformation |
| | `part-iv-02-reduced-theory` "The reduced theory" | the reduced theory and the consistency problem |
| V Coset space dimensional reduction | `part-v-01-symmetric-fields` "Symmetric fields and their constraints" | the constraints; the embedding of U(1) in SU(3) |
| | `part-v-02-four-dimensional-theory` "The four-dimensional theory" | solving the constraints; the Higgs potential |
| VI The landscape | `part-vi-01-consistent-reductions` "Consistent reductions" | torus and Scherk-Schwarz; group manifolds; sphere reductions in supergravity; symmetric-field schemes |
| | `part-vi-02-compactifications` "Compactifications" | special holonomy; flux; warped and large extra dimensions |

The HTML sidebar is docked and collapses to the parts (`collapse-level: 1`): each part opens to its group chapters,
and each chapter's sections appear in the page table of contents. Each group chapter ends with integrative
exercises (D21) and the D25 prose, written when the part is brought to the current format.

### D27. Chapters are main themes

A chapter is a main theme at the level of a discipline (real analysis, complex analysis, differential geometry, group
theory, quantum mechanics), never a sub-topic (author's feedback, October 2026: "Charts and manifolds" was too narrow).
Where several group chapters of D24 belong to one theme they are nested: the theme is the chapter (`#`), the former
group chapters are its parts (`##`, identifiers unchanged, each keeping its own `### Exercises` and summary), and
their sections are `###` (`scripts/merge_chapters.py` demotes headings outside `:::` blocks). The single-topic unit of
D16/D19 (at most six definitions, a figure) is then the `###` section; `check_format.py` detects nested parts by
their `### Exercises` subsection. The final table of contents:

Part I, Foundations of mathematics: 1 Sets, relations and topology; 2 Linear algebra; 3 Real analysis; 4 Complex
analysis; 5 Differential equations; 6 Fourier analysis; 7 Hilbert spaces and operator theory; 8 Probability and
statistics; 9 Differential geometry (charts and manifolds; vector fields, forms and tensors; metrics; connections
and curvature; flows and Killing vectors); 10 Group theory; 11 Lie groups and Lie algebras (Lie groups and algebras;
representations); 12 Homogeneous spaces and fibre bundles (coset spaces; fibre bundles).

Part II, Foundations of physics: 1 Classical mechanics; 2 Thermodynamics and statistical mechanics; 3 Quantum
mechanics; 4 Classical field theory (fields; actions and symmetries); 5 Gauge theories (gauge invariance; symmetry
breaking); 6 Quantum field theory; 7 General relativity; 8 Supersymmetry and supergravity.

Parts III–VI as in D26 (two chapters per part, each a theme of the part).

### D28. Veriphy verifies every claim of the book (the strong reading)

Veriphy is the central tool of the book, and "all results" means all (author's decision, October 2026). Every
mathematical claim in every block is a Lean theorem of the library whose computational core is a Veriphy certificate
(Sage finds the certificate, the Lean kernel checks it, only the Sage-free proof is committed); mathlib lemmas supply
the glue. This covers:

| block kind | count (Oct 2026) | what is verified |
|---|---|---|
| definition | 546 | a Lean definition (or an `abbrev` of mathlib's) and a certified instance on the block's running example |
| theorem, proposition, lemma, corollary | 238 | the statement itself, as a Lean theorem |
| example | 2354 | every value, identity, inequality or count the example states |
| exercise | 2464 | the final answer of the solution; the Sage solution script is rewritten as the Veriphy certificate script of that theorem |

**Physics is formalised, not labelled.** Statements that are not mathematics as written (quantum postulates, the path
integral, renormalisation, anomalies, thermodynamic laws) are given an explicit mathematical model in Lean and proved
inside it: quantum states and observables on mathlib inner-product spaces with the Born rule and the Schrödinger
evolution as definitions; the time-sliced path integral as a finite-dimensional Gaussian integral and its limit;
regularised quantities as explicit functions of the cutoff or of ε, running couplings as solutions of their flow
equations, poles from Gamma-function expansions; anomalies through gamma-matrix trace identities and regularised
finite-dimensional traces; ensembles as probability measures on finite or measurable state spaces. A theorem quoted
from the literature (Cauchy, Peter-Weyl, the central limit theorem, Plancherel) is taken from mathlib when it is
there; otherwise the special case the book uses is proved and the general statement is recorded as a library gap.
The registry level `text` is reserved for historical and bibliographic remarks; a mathematical claim may not stay
`text` or `sage`.

**Pipeline.** (1) Claim manifest: one file per chapter, `verification/<chapter>.yaml`, checked by
`scripts/check_verification.py` (`claims.yaml` stays for the result blocks until it is folded into these files); each entry lists the
block, its claims in words, the Lean name of each, the Veriphy recipe and the certificate script; a block with no
checkable claim records why. (2) Formalisation: one Lean module per `###` section under
`KaluzaKlein/<Part>/<Chapter>/`, small enough that each builds alone within the 16 GB memory of the build machine
(large identities, such as the Kaluza-Klein jet identities, are split into lemmas). Statements are tested numerically
before they are proved (Veriphy playbook step 1). (3) Certification: `.veriphy/certify.py` with the recipe templates;
scripts committed under `.veriphy/certificates/`. (4) Gate: `scripts/check_claims.py` fails on any block whose claims
are not all `lean` with a recipe or a mathlib reference, any `sorry`, any non-standard axiom, any committed
`sage_run`. Builds run one at a time (no concurrent `lake build`).

**Order.** A pilot chapter (Linear algebra, which already has nine certificates) measures claims per hour, memory and
missing recipes; then Part I chapter by chapter, Parts II–VI after. Missing recipes go to Veriphy (D30), never to
hand-written workarounds in the book.

### D29. Verification is shown in a hidden callout

No Lean statement is visible by default any more. Each block carries, right after it, a collapsed callout
`::: {.callout-note .verification collapse="true" title="Verification"}` styled like the solution callouts (D20: no
bar, no fill), containing the Lean statement(s) of the block's claims (the `lean` shortcode), the Veriphy recipe and
certificate script name, and the Sage oracle output where one exists (the `sage` shortcode or a certificate excerpt).
For an exercise the verification callout follows the solution callout. **The main text never refers to verification**
(author's instruction, October 2026): definitions, propositions, examples, exercise statements, solutions and the
connecting prose state mathematics and physics only, with no Lean names, no mention of Lean, Veriphy, recipes,
certificates, mathlib or the registry, and no "checked in Sage" or script references; every such fact lives in the
Verification callout. Blocks whose subject was the verification itself are rewritten as mathematical examples and
exercises. Exercise solutions may still show a Sage computation as part of the teaching, and figure captions may name
their generating script. The per-chapter "Summary of verified results" sections are removed; the registry appendix
lists every claim with its verification. Existing visible `{{< lean >}}` shortcodes
(141 as of October 2026) move into these callouts. `scripts/check_format.py` checks that every block with claims has
its verification callout and that no `lean` shortcode is outside one.

**The Verification callout has a fixed shape** (author's instruction, October 2026): a Veriphy part that decomposes
into a Lean part and a Sage part, always all three, in this order:

```markdown
::: {.callout-tip .verification collapse="true" title="Verification"}
**Veriphy.** Recipe `witness`; certificate `linear_finrank_R3`.

**Lean.**

{{< lean KaluzaKlein.Linear.finrank_R3 >}}

**Sage.** The oracle computation that produced the certificate's data (a `sage` code block or the relevant excerpt
of the certificate script).
:::
```

The Veriphy line names the recipe and the certificate (the `.veriphy/certificates/` script) for every claim of the
block; a block with several claims lists them. All `lean` shortcodes sit in the Lean part. A part not yet produced
reads "Pending"; `check_format.py` enforces the three parts and their order, and with `--strict` (the gate, task 11.9)
rejects any pending part. There is no "no recipe" entry: a claim no recipe covers gets a new recipe on the Veriphy
branch (D30).

**Levels (author's instruction, October 2026).** A certificate may also be a test when no certificate is feasible,
and Veriphy provides this fallback (its change `kk-csdr-book-coverage`, command `sage_test`). Every Veriphy part names
its level, in this order of preference:

- `Recipe <name>; certificate <script>`: a recipe found the data and the kernel checked the proof (level
  `certificate`);
- `Test (kernel-checked, <n> instances); script <script>`: the claim's Lean statement was checked by the kernel at
  instances Sage chose, each by computation alone and independently of the claim's proof (level `kernel-test`);
- `Test (oracle, <n> instances); record <record>`: Sage evaluated the claim at instances in exact or interval
  arithmetic (level `oracle-test`), for claims Lean cannot check at instances.

A test is used only when no recipe applies (general statements in arbitrary dimension, structural facts, gaps in
mathlib such as Sylvester's law of inertia), and a certificate replaces it as soon as a recipe for the shape exists.
The Lean part holds the claim's Lean proof when it has one, and the kernel-checked instances otherwise. The registry
appendix reports the number of claims at each level. Definitions get a Verification callout only for the claims they
contain (that the object is well defined, that a stated example satisfies it); a definition with no claim has none.

### D30. Missing building blocks go into Veriphy itself

Veriphy is the author's tool (local repository `~/Projects/Personal/veriphy`, release 0.2.0). Recipes, emitters,
translation-table entries and Lean helper lemmas that the book needs and Veriphy lacks are added there, on a new
branch `feat/kk-csdr-book` cut from `main`, following that repository's own conventions (its OpenSpec workflow,
tests under `tests/` and `VeriphyTest/`, recipe documents under `prompts/recipes/` with templates, the CHANGELOG),
for the next release. The book pins the Veriphy branch (then the release) in `lakefile.toml` and `VERSIONS`. Gaps
already known, to be confirmed and extended by the pilot: rotation exponentials (generators with eigenvalues ±i, now
excluded by `matrix_exponential`); Grassmann-variable jets (supersymmetry); definite and improper integrals over
intervals (Gaussian moments, Gamma and Beta values, Fourier coefficients); contour integrals and residues; limits of
sequences and functions; closed forms of series beyond `sum_induction` (zeta values, Matsubara sums); special-function
identities (Legendre, Bessel, Hermite, spherical harmonics); integrals of differential forms over manifolds (Stokes,
Chern numbers); expectations and moments of probability distributions; finite group and character computations at
scale; documentation gaps already recorded in the registry (`veriphy_gaps`, the `VeriphyAssets` name clash).

### D31. New content: the Dirac operator, functional analysis beyond Hilbert spaces, renormalisation

Author's instruction (October 2026). Each addition follows the format of D17-D22 and D25-D29 (definitions first,
graded examples and exercises, connective prose, figures, hidden verification) and the main-theme rule of D27.

- **Functional analysis (Part I).** `part-i-07` becomes the main-theme chapter *Functional analysis*, with Hilbert
  spaces as one of its parts rather than its subject. New parts before the Hilbert-space material: *Normed and Banach
  spaces* (norms, completeness, `ℓ^p`, `L^p`, `C(K)`, bounded linear maps and their norms, dual spaces and the
  `ℓ^p`-`ℓ^q` and `L^p`-`L^q` dualities); *The four principles* (Hahn-Banach, Baire category, uniform boundedness,
  open mapping and closed graph, with the counterexamples that show each hypothesis is needed); *Weak topologies and
  reflexivity* (weak and weak-* convergence, Banach-Alaoglu, reflexive spaces); *Sobolev spaces* (weak derivatives,
  `H^s` on the torus and on `ℝ^n` through the Fourier transform, embedding and Rellich compactness in the cases the
  book needs); and after the spectral theory, *Compact and Fredholm operators* (compact operators on Banach spaces,
  the Fredholm alternative, the Fredholm index and its stability), which the index of the Dirac operator uses, and
  *Banach and C\*-algebras* in outline (spectrum and spectral radius, the continuous functional calculus), which the
  quantum chapters use.
- **The Dirac operator.** Three places, in order. (i) A new Part I main-theme chapter *Spinors and the Dirac
  operator*, after *Homogeneous spaces and bundles*: Clifford algebras and their representations in every
  dimension and signature, spin groups as double covers, spinor representations (Weyl, Majorana, symplectic
  Majorana) and their dimensions, spin structures and spinor bundles, the spin connection, the Dirac operator on
  `ℝ^n`, tori and spheres with its spectrum on `S^2` and `S^3` computed explicitly, the Lichnerowicz formula and
  vanishing theorems, zero modes, and the index (statement of Atiyah-Singer, the index on `S^2` with a monopole
  background computed by hand and equal to the monopole charge). (ii) A new Part II main-theme chapter *The Dirac
  equation*, after *Quantum mechanics*: derivation, Lorentz covariance, the conserved current, plane-wave solutions
  and the spinors `u`, `v`, completeness and projection operators, the non-relativistic limit (Pauli equation,
  `g = 2`), the Dirac hydrogen atom solved exactly (fine structure), the Foldy-Wouthuysen transformation,
  charge conjugation, Weyl and Majorana fermions, the Klein paradox and the Dirac sea as the motivation for the
  quantised field. (iii) **The higher-dimensional Dirac operator on compact spaces** (author's instruction), in the
  Part I chapter for the analysis and in the reduction parts for the physics: spinors and Clifford algebras in `D`
  dimensions and their decomposition under `SO(1, 3) × SO(n)`, Majorana-Weyl spinors in ten and eleven dimensions;
  the Dirac operator on a product `M^4 × K` split into the four-dimensional operator and the internal one, whose
  eigenvalues are the four-dimensional masses; the internal spectrum and zero modes on each family of compact space
  the book uses: tori `T^n` (spin structures, periodic and antiperiodic spinors, Scherk-Schwarz twists), spheres
  `S^n` in every dimension (the full spectrum and its degeneracies), group manifolds (`S^3 = SU(2)` with
  left-invariant frames), coset spaces `G/H` (harmonic analysis by Frobenius reciprocity, Parthasarathy's formula
  for the square of the Dirac operator on symmetric spaces, the nearly-Kähler cosets of Part VI), complex projective
  spaces and Kähler manifolds (the Dirac operator as the Dolbeault operator, `CP^2` and its lack of a spin structure
  and the spin-`c` fix), Calabi-Yau spaces and `K3` (covariantly constant spinors, the number of generations as half
  the Euler number), orbifolds (projected zero modes); twisted Dirac operators with gauge backgrounds (monopoles on
  `S^2`, instantons, the embedding of `R` in CSDR); index theorems in higher dimensions (the `Â` genus, the twisted
  index, the index density computed on the examples); and the chirality problem: Lichnerowicz and Witten's no-go for
  chiral zero modes on positively curved spaces without gauge fields, and its evasion by gauge backgrounds, which is
  how CSDR (Part V) obtains chiral four-dimensional fermions, with the CSDR fermion constraints solved on `S^2` and
  on the cosets of Part VI.
- **Renormalisation (Part II, extensively).** A new Part II main-theme chapter *Renormalisation*, after *Quantum
  field theory* (whose closing part stays as an introduction): regularisations (cutoff, Pauli-Villars, dimensional,
  zeta-function and heat-kernel, lattice) compared on the same integrals; power counting and superficial degree of
  divergence; renormalisability; counterterms and renormalised perturbation theory for `φ⁴` at one and two loops;
  renormalisation conditions and schemes (on-shell, MS, MS-bar) and scheme dependence; BPHZ and Zimmermann's forest
  formula with worked forests; Ward identities and the renormalisation of QED at one loop; the Callan-Symanzik and
  renormalisation-group equations, beta functions (`φ⁴`, QED, Yang-Mills and asymptotic freedom), running couplings
  and anomalous dimensions; Dyson-Schwinger equations and renormalisation conditions; the Wilsonian renormalisation
  group (block spins, the Gaussian and Wilson-Fisher fixed points, the `ε` expansion, critical exponents compared
  with the statistical-mechanics chapter) and its rigorous formulation in outline; effective field theories and
  decoupling; and in higher dimensions: non-renormalisability of gauge theories above four dimensions, the
  Kaluza-Klein tower in loops, power-law running, threshold corrections and the Casimir energy of compact
  dimensions.

Each new chapter or part is scaffolded first (D24: headings and one-line scopes), then written, then certified
(D28-D30) like every other chapter. Its sources are read before use (D32 and the arXiv reading rule).

### D32. Textbooks as sources of content and exercises

Author's instruction (October 2026): use the textbook references to get content, examples and exercises. The author's
textbooks are kept in `references/textbooks/` (git-ignored PDFs). Every textbook there is read and mapped to the
chapters it serves: a reading note `references/notes/<key>.md` lists the chapters and sections of the book used, with
the book's section, equation and problem numbers. A textbook that serves no chapter is recorded in
`references/notes/README.md` with the reason. Textbooks enter `references.bib` with `keywords = {textbook}` and are
cited at the point of use. Content may follow a textbook's line of argument, rewritten in the book's own voice,
notation and conventions (D23); an example or exercise adapted from a textbook cites it with the problem number
(`[@key, Problem 4.2]`). The arXiv rule for research papers is unchanged; textbooks are the second class of content
sources, read in the same way. Textbooks the author adds later are read before the chapter they serve is written or
revised. The current folder holds 17 books, among them sources for the new material of D31: renormalisation
(Balduf, Dyson-Schwinger equations and renormalisation conditions; Bauerschmidt, Brydges and Slade, a
renormalisation group method), spinors (Theory of Spinors and its Application in Physics and Mechanics),
supersymmetry (Nath), general relativity (Plebański and Krasiński; 300 Problems in Special and General Relativity),
quantum mechanics (Cini et al.; Izaac and Wang), statistical mechanics (Bricmont), mechanics and electrodynamics.

### D33. A special-relativity chapter before general relativity

Author's instruction (October 2026). Special relativity is a main-theme chapter of Part II (D27), placed immediately
before *General relativity* in the sidebar, file `part-ii-08b-special-relativity.qmd`. Until now its material was
scattered (Minkowski space in *Linear algebra*, Lorentz transformations in *Classical field theory*); the chapter
develops it once, in order, and the other chapters keep their uses and cross-reference it. Its parts:

- **Postulates and Lorentz transformations:** the two postulates, the derivation of the Lorentz transformation,
  boosts and rapidity, composition of velocities, length contraction and time dilation, simultaneity.
- **Minkowski spacetime:** the metric in the signature `(+, −, −, −)` (D23), the light cone and causal structure,
  spacetime diagrams, proper time, the twin paradox, hyperbolic geometry of boosts.
- **The Lorentz and Poincaré groups, in detail and formally (author's instruction), built up progressively:** the
  chapter starts physically and becomes formal step by step. (1) Boosts along one axis as `2 × 2` matrices and the
  rapidity group; (2) the Lorentz group `O(1, 3)` defined as the isometry group of the Minkowski form, with full
  proofs that it is a Lie group, that `det Λ = ±1` and `|Λ⁰₀| ≥ 1`, and that it has exactly four connected
  components, with `P`, `T` and `PT` as representatives; (3) the polar (boost × rotation) decomposition of every
  element of `SO^+(1, 3)`, with proof; (4) the Lie algebra `so(1, 3)`: generators `J_i`, `K_i`, their brackets, the
  isomorphism `so(1, 3)_ℂ ≅ sl(2, ℂ) ⊕ sl(2, ℂ)`, the exponential map, which is
  surjective onto `SO^+(1, 3)` (proved), while on the double cover `SL(2, ℂ)` it is not: `−[[1, 1], [0, 1]]` lies on
  no one-parameter subgroup (the counterexample, with its image in `SO^+(1, 3)` a null rotation that is an exponential);
  (5) `SL(2, ℂ)` as the double cover of `SO^+(1, 3)` through Hermitian `2 × 2` matrices, proved, with the kernel
  `{±1}`, and the finite-dimensional representations `(j₁, j₂)`; (6) the non-commutativity of boosts, the
  Thomas-Wigner rotation derived exactly and Thomas precession; (7) the Poincaré group `ISO(1, 3)` as a semidirect
  product, its Lie algebra and its Casimir operators `P²` and `W²` (Pauli-Lubanski vector); (8) Wigner's
  classification of the unitary irreducible representations by the little group (massive: `SU(2)`, spin; massless:
  `ISO(2)`, helicity; tachyonic and zero-momentum cases named), stated formally with the induced-representation
  construction for the massive and massless cases. Each step has its definitions, theorems with proofs, and at least
  three examples and exercises, and is linked to *Lie groups and algebras*, *Homogeneous spaces and bundles* and
  *Spinors and the Dirac operator*. Transformations throughout (passive versus active, the transformation of
  four-vectors, tensors, fields and spinors) are treated with explicit index conventions.
- **Relativistic kinematics:** four-vectors, four-velocity and four-momentum, `E² = p² + m²`, collisions and decays,
  thresholds, invariant mass, Mandelstam variables, Compton scattering.
- **Relativistic optics:** Doppler effect, aberration, the appearance of moving bodies.
- **Relativistic dynamics:** four-force, motion under constant proper acceleration, accelerated observers and the
  Rindler frame (which *General relativity* then uses), the relativistic action of a particle.
- **Electrodynamics in covariant form:** the field-strength tensor, the transformation of fields, the Lorentz force
  and the invariants, cross-referenced to *Classical field theory* rather than repeated.

Sources (D32): Blennow and Ohlsson, *300 Problems in Special and General Relativity*, Chapter 1 (150 problems with
solutions, the main source of examples and exercises); Boskoff and Capozziello, Chapter 9; Maggiore, Chapters 7-8;
Boudreau and Swanson, Chapter 14. The reading notes are extended with the sections and problems used.


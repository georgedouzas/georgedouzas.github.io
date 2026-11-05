# Pilot report: Veriphy for every claim of "Linear algebra"

Task 11.4 of `openspec/changes/kk-csdr-tutorial/tasks.md` (design D28-D30), run on 2026-10-15 against the local
Veriphy branch `feat/kk-csdr-book`. Chapter: `tutorial/part-i-02-linear-algebra.qmd`; manifest:
`verification/part-i-02-linear-algebra.yaml`; Lean: `KaluzaKlein/PartI/LinearAlgebra/*.lean` (25 modules, one per
`###` section plus the closing exercises), all imported from `KaluzaKlein.lean`; certificate scripts:
`.veriphy/certificates/la-*.sage` plus the generic `ideal_membership`, `witness`, `farkas`, `sos`,
`matrix_from_goal` scripts already in the project.

## Numbers

| quantity | value |
|---|---|
| blocks | 185 (20 definitions, 6 propositions, 78 examples, 81 exercises) |
| blocks with claims / with `none` | 183 / 2 (`exr-type-classify`, `exm-contraction-ricci`: notation and recalled definitions, no value stated) |
| claim entries in the manifest | 527 (a theorem may serve several blocks) |
| distinct Lean declarations behind them | 404 (374 new, 30 reused from `KaluzaKlein/Foundations/Linear.lean`) |
| declarations in the 25 new modules | 411 (379 theorems, 32 definitions), 3463 lines |
| proofs that are Veriphy replacements (`-- certificate:` sites) | 186 |
| distinct certificate scripts used | 47 (40 new `la-*` scripts, 7 reused generic ones) |
| claim entries proved by a Veriphy recipe | 205 of 527 (39 %) |
| distinct theorems proved by a Veriphy recipe | 159 of 404 (39 %); 245 mathlib-only |
| blocks whose claims are all Veriphy / mixed / mathlib-only | 30 / 74 / 79 |
| `sorry`, `native_decide`, committed `sage_run` | 0 |
| false claims found in the chapter | 0 |

Recipes, by distinct theorem: `matrix_from_goal` 126, `ideal_membership` 17, `witness` 9,
`transcendental_rewrite` 3, `sum_induction` 2, `farkas` 1, `sos` 1. Every Veriphy proof is a Sage-free replacement
written by `.veriphy/certify.py`; the 20 exercises with a Sage solution have it rerun as the oracle of their
certificate script (the `# oracle` section of `la-<exercise>.sage`), and the oracle's assertions are what the
callout's **Sage** part shows.

Why the 245 mathlib-only theorems have no recipe (the manifest's `why`, by category):

| category | theorems | nature |
|---|---|---|
| general statements | 55 | arbitrary dimension or space (rank-nullity, dual basis, transformation laws, alternating maps) |
| dimension counts | 51 | `Module.finrank … = n`: no recipe certifies a finrank |
| closed evaluations | 39 | finite sums / vector entries / component tables by `simp`, `norm_num` outside the `!![…]` translation |
| glue | 27 | from a certified core (determinant, cofactors, witness) to `LinearIndependent`, spans, kernels |
| structural | 25 | unfolding a mathlib definition or instance |
| definitions | 14 | `abbrev`/`def` backing a definition block |
| definite integrals | 8 | `∫ φ in 0..2π, …` |
| matrix ranks | 7 | `Matrix.rank` of explicit matrices |
| non-linearity refutations | 7 | `¬ ∃ f : V →ₗ W, ∀ x, f x = …` |
| signatures | 7 | `HasSignature S p q z` assembled from certified congruences |
| polynomial-ring identities | 5 | identities and degrees in `ℝ[X]` |

The callouts follow the amended D29 shape: **Veriphy** (recipe and certificate per theorem, then "Pending for …
(proved with mathlib alone; no recipe yet)" for the rest), **Lean** (one `lean` shortcode per theorem), **Sage** (the
oracle code of each certificate, or the generic script's computation; "Pending." when no certificate is involved).
Under `check_format.py --strict` the 153 callouts with a pending part are rejected, as intended: they list exactly the
missing recipes below.

## Cost

Wall clock, one agent, 2026-10-15:

| phase | time |
|---|---|
| reading (design, playbooks, recipes, chapter) | 18:01-18:20 |
| probing recipe shapes, fixing `certify.py` (see below) | 18:20-18:45 |
| formalising and certifying the 25 modules (write, compile-fix, certify, build) | 18:45-20:06 (≈ 1 h 20 min for ≈ 400 theorems) |
| main-text cleanup (D29 amendment), claim table, manifest and callout generation | 20:06-20:30 |
| Fourier-mode independence (`fourierModes_linearIndependent`, via Chebyshev `T_n`, `U_n`), measurements, audit, render | 20:30-20:45 |
| total | 2 h 45 min |

Throughput while formalising: about 300 theorems per hour of Lean work, because most claims of this chapter are
explicit rational matrix computations that one generic script (`la-matrix.sage`) certifies. Of the 80 minutes,
roughly half was waiting for Lean (each `certify.py` run is 40 s to 1 min 50 s, each module build 30-70 s, one Lean
process at a time); compile-fix rounds were 1-3 per module.

Per-module cost (`/usr/bin/time -l lake env lean <file>`, the dominant part is importing Mathlib):

| module | wall clock (s) | peak memory (GB) |
|---|---|---|
| `VectorSpaces` | 66 | 4.6 |
| `LinearCombinations` | 44 | 4.7 |
| `Spans` | 64 | 4.2 |
| `LinearIndependence` | 73 | 3.9 |
| `Bases` | 63 | 4.1 |
| `Dimension` | 61 | 4.4 |
| `VectorSpaceResults` | 49 | 4.6 |
| `LinearMaps` | 40 | 4.7 |
| `LinearFunctionals` | 58 | 4.4 |
| `DualSpace` | 58 | 4.3 |
| `DualBasis` | 53 | 4.6 |
| `DualResults` | 68 | 4.0 |
| `BilinearForms` | 62 | 4.3 |
| `SymmetricForms` | 61 | 4.4 |
| `NondegenerateForms` | 72 | 3.9 |
| `Signature` | 65 | 4.3 |
| `MinkowskiResults` | 58 | 4.5 |
| `MultilinearMaps` | 49 | 4.7 |
| `TensorProducts` | 54 | 4.5 |
| `ElementaryTensors` | 54 | 4.5 |
| `MixedTensors` | 53 | 4.5 |
| `Components` | 56 | 4.4 |
| `Contraction` | 56 | 4.5 |
| `TensorResults` | 53 | 4.6 |
| `Exercises` | 58 | 4.4 |
| total (sequential) | 1447 | max 4.7 |

The measurement ran while nothing else used Lean. A file containing only `import Mathlib` costs 63 s and 4.6 GB
on the same machine (2.8 s of user time: the cost is loading the Mathlib `.olean` files), so the chapter's own proofs
are a negligible part of each figure; the spread between modules is noise of the file system.

`lake build KaluzaKlein.PartI.LinearAlgebra.<M>` took 27-72 s per module once its dependencies were built (the first
build, of `VectorSpaces`, took 171 s, rebuilding `Foundations.Linear`), with peak resident memory 3.4-5.9 GB. No
module needed splitting; nothing approaches the 16 GB limit.

## Missing building blocks

Each item: the Lean goal shape that no recipe closed, what Sage can compute for it, and the recipe or emitter that
would close it. Counts are distinct theorems of this chapter. These feed `openspec/changes/kk-csdr-book-coverage`
in the Veriphy repository.

1. **Dimension (finrank) certificate** (51 theorems; the largest gap). Goals:
   `Module.finrank ℚ ↥(Submodule.span ℚ (Set.range ![![1, 2, 0], ![0, 1, 1]])) = 2`,
   `Module.finrank ℝ ↥(LinearMap.ker sumFun) = 2`, `Module.finrank ℚ ↥(U ⊓ W) = 1`,
   `Module.finrank ℝ ↥(LinearMap.ker (Matrix.traceLinearMap (Fin 3) ℝ ℝ)) = 8`,
   `Module.finrank ℝ ((Fin 2 → ℝ) ⊗[ℝ] (Fin 3 → ℝ)) = 6`. Sage computes the rank, an echelon basis of the span or
   kernel, pivot minors (independence) and coordinates of the generators in the basis (spanning). Needed: a
   `dimension` recipe whose emitter writes `finrank_span_eq_card` + a determinant or cofactor independence proof, or
   rank-nullity (`LinearMap.finrank_range_add_finrank_ker`) + a surjectivity witness, and `Submodule.finrank_mono` /
   `Submodule.eq_of_le_of_finrank_eq` for sums and intersections; plus translation entries so that `finrank` of
   `Matrix`, `Fin n → K`, `TensorProduct`, `Module.Dual` reduce by the product formulas.
2. **Matrix rank** (7). Goals: `Matrix.rank !![(1 : ℚ), 2, 3; 2, 4, 6] = 1`, `KaluzaKlein.Linear.D.rank = 2`,
   `(Matrix.diagonal ![(1 : ℚ), 1, 0]).rank = 2`, `¬ ∃ u v, Matrix.vecMulVec u v = 1`. Sage: `rank()`, a non-singular
   `r × r` minor (lower bound) and a factorisation `A = B C` with inner size `r` (upper bound). Emitter:
   `Matrix.rank_eq_finrank_span_row` + independent rows for `≥`, `Matrix.rank_mul_le_left`/`rank_vecMulVec_le` for `≤`.
3. **Linear independence and span of explicit vectors** (27 glue theorems). Goals:
   `LinearIndependent ℚ ![![1, 2, 0], ![0, 1, 1], ![0, 0, 1]]`, `¬ LinearIndependent ℚ ![…]`,
   `Submodule.span ℚ (Set.range ![…]) = LinearMap.ker A.mulVecLin`, `antisymMatrices 3 = Submodule.span ℝ (Set.range ![L1, L2, L3])`.
   Sage: determinant or pivots, the dependence relation (`linear_dependence`), coordinates. Emitter: package the
   certified core that this pilot proved separately (a `matrix_from_goal` determinant or `ideal_membership` cofactors)
   into `Matrix.linearIndependent_rows_of_det_ne_zero` / `Fintype.linearIndependent_iff`, and dependence by the
   explicit relation.
4. **Vectors and named matrices in the goal translation**. `!![(1 : ℚ), 2, 3; 2, 4, 6] *ᵥ ![1, 1, -1] = ![0, 0]`
   arrives with the two sides as opaque atoms (`x0 - x1`); likewise `⬝ᵥ`, `ᵥ*`, `Matrix.vecMulVec`,
   `Matrix.diagonal ![…]`, `(Pi.basisFun ℝ (Fin 3)).coord 1 ![5, 7, 9]` and matrices given by name (`KaluzaKlein.Linear.eta`).
   Every such claim was restated with column and row matrices (`!![1; 1; -1]`) to reach `matrix_from_goal`; the 39
   "closed evaluation" theorems stayed `simp`/`norm_num`. Needed: translation-table entries for `![…]`, `*ᵥ`, `ᵥ*`,
   `⬝ᵥ`, `vecMulVec`, `diagonal`, and unfolding of named matrix definitions.
5. **`matrix_from_goal` beyond its template** (fixed locally in `la-matrix.sage`; should go into the recipe):
   (a) determinants of `4 × 4` literals (the template asserts `n ∈ {2, 3}`): `norm_num [Matrix.det_succ_row_zero,
   Fin.sum_univ_succ, Fin.succAbove]` closes them; (b) products involving `ᵀ` (`Aᵀ * B = 1`) defeat the default
   entrywise `norm_num` for `3 × 3`; the SIMP set `Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply` is
   needed by default; (c) symbolic entries (`!![0, -c, b; …]`, a free `t`) leave linear identities after `norm_num`:
   append `<;> ring1`; (d) a trace of a `4 × 4` product (`Matrix.trace (F * Fᵀ) = 2 * (B ^ 2 - E ^ 2)`) needs
   `simp [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct]` and `ring1`.
6. **Matrix identities modulo a transcendental relation** (local scripts `la-lorentz-boost`, `la-sqrt-eigen`). Goals:
   `Λᵀ * η * Λ = η` with `cosh u`, `sinh u` entries; `B * !![1; √2; 1] = (2 + √2) • !![1; √2; 1]`. Sage reduces each
   entry of `lhs - rhs` to a constant multiple of `cosh² u - sinh² u - 1` (resp. `(√2)² - 2`). Emitter: per entry
   `linear_combination k * Real.cosh_sq_sub_sinh_sq u` (resp. `Real.sq_sqrt`); a `matrix_from_goal` option "modulo
   side relations" would cover rotations and boosts generally (also the known rotation-exponential gap of D30).
7. **Trigonometric rewrite table entries**. `transcendental_rewrite` reported "no rewrite in the tables applies" to
   `Real.cos x ^ 2 = 1 / 2 + 1 / 2 * Real.cos (2 * x)`, `Real.sin x ^ 3 = 3 / 4 * Real.sin x - 1 / 4 * Real.sin (3 * x)`
   and even `Real.cos x ^ 2 + Real.sin x ^ 2 = 1`; `REWRITES` had to name `Real.cos_sq`, `Real.sin_three_mul`,
   `Real.cos_sq_add_sin_sq`. Needed: power-reduction and multiple-angle entries in the core tables.
8. **Definite integrals** (8; known D30 gap). Goals: `∫ φ in (0 : ℝ)..(2 * Real.pi), Real.cos φ ^ 2 = Real.pi`,
   `∫ φ in (0 : ℝ)..(2 * Real.pi), Real.sin (2 * φ) = 0`, linearity on `3 + 2 cos φ` and `1 - sin 2φ`. Sage
   integrates exactly. Emitter: an antiderivative step (`intervalIntegral.integral_eq_sub_of_hasDerivAt` with the
   `derivative` recipe's `has_deriv_at`) and endpoint evaluation (`Real.cos_nat_mul_two_pi`), or
   `intervalIntegral.integral_comp_mul_left` for rescaled arguments.
9. **Non-linearity refutations** (7). Goals: `¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = x ^ 2`,
   `¬ ∃ B : LinearMap.BilinForm ℝ (Fin 2 → ℝ), ∀ v w, B v w = v ⬝ᵥ w + 1`. Sage finds the test points where
   additivity or homogeneity fails. Emitter: `rintro ⟨f, hf⟩`, specialise `map_smul`/`map_add`/`map_zero` at the
   points, `norm_num`.
10. **Signature (inertia) certificate** (7). Goal: `HasSignature !![(2 : ℚ), 1, 0; 1, 2, 1; 0, 1, 2] 3 0 0` (the
    project's definition: an invertible `P` with `Pᵀ S P` diagonal, and the sign counts). Sage: `LDLᵀ`/
    `QuadraticForm.signature_vector` gives `P` and the diagonal. Emitter: the congruence and `det P` by
    `matrix_from_goal`, the counts by `norm_num [List.ofFn_succ]`. Library gap: Sylvester's law of inertia (that the
    triple is independent of `P`) is not in mathlib, so the definition block's invariance claim is not proved.
11. **Polynomial-ring identities** (5). `(X + 1) ^ 2 = 1 + 2 * X + X ^ 2` in `ℝ[X]`, `derivative (X ^ 2) = 2 * X`,
    `(X ^ 3 : ℝ[X]) ∉ Polynomial.degreeLT ℝ 3`: the translation does not read `Polynomial.X` as a variable. Needed:
    a `Polynomial` translation (X as a generator, `natDegree`, `derivative`, `eval`).
12. **`sage_run` reports a replacement before Lean has checked it**. With a proof that fails (a `3 × 3`
    product with a transpose) `sage_run` still printed `Sage-free replacement: …`, followed by the error. A driver that
    trusts the message writes a broken proof. Needed: print the replacement only after the kernel accepted it (or
    mark it failed). The project driver now guards against it (below).
13. **Inline markers**: a `sage_run` inside a term (`rw [show … by sage_run …]`) cannot be re-indented by the
    driver; the pilot avoided it by separate theorems. Low priority.

Infrastructure fixed in this repository during the pilot (`.veriphy/certify.py`): the marker-line computation used
offsets of the original text in the authoring copy; the `Sage-free replacement` infos carry no position in plain
`lake env lean` output, so partial success never registered (now `lean --json`); a replacement is rejected when an
error lies in its declaration (item 12). With these, one failing certificate no longer blocks the others in the file.

## What remains

- The 245 mathlib-only theorems are "Veriphy. Pending" in their callouts until the recipes of items 1-11 exist on the
  Veriphy branch; `check_format.py --strict` therefore fails on 153 callouts of this chapter (by design).
- Sylvester's law of inertia (invariance of the signature) is not formalised (item 10).
- The two `none` blocks carry no claim by choice; if the gate requires a claim for every block, `exr-type-classify`
  could be given a typing theorem about index contractions.

## Checks at the end of the pilot

- `.venv/bin/python scripts/check_verification.py part-i-02-linear-algebra`: 185/185 blocks covered, 0 problems.
- `.venv/bin/python scripts/check_format.py tutorial/part-i-02-linear-algebra.qmd`: 0 problems (153 problems under
  `--strict`, the pending callouts).
- `lake build KaluzaKlein.PartI.LinearAlgebra.<M>` succeeds for all 25 modules.
- Axiom audit (`scripts/lean_audit.py`'s functions, run on the 25 modules and `Foundations.Linear` because the whole
  library has stale modules such as `AbelianKK.Jets` that the pilot must not rebuild): 413 theorems, all on
  `propext`, `Classical.choice`, `Quot.sound` only; no live `sage_run`. The audit script misread axiom lists that
  Lean wraps over several lines (it kept the newline in the name and reported `\nClassical.choice` as non-standard);
  fixed in `scripts/lean_audit.py`.
- `cd tutorial && KK_DRAFT=1 quarto render part-i-02-linear-algebra.qmd --to html` succeeds: 527 Lean statements
  rendered, all inside the 183 collapsed Verification callouts, none outside, none pending.

## Scaling to the book

This chapter needed 404 Lean declarations for 185 blocks (2.2 per block) and about 2 h 45 min of agent time, of which
the Lean work proper was 1 h 20 min. D28 counts 5602 blocks for the whole book; at the same density that is about
12 000 theorems and, at this chapter's pace, about 80 hours. The pace will not carry over unchanged: 61 % of this
chapter's theorems are mathlib-only, and the chapters ahead (analysis, integrals, special functions, the physics
models of D28) depend on exactly the recipes of items 1, 4, 7 and 8 above, so building those on the Veriphy branch
first is what makes the remaining chapters certifiable rather than merely proved. Build cost is not a constraint:
every module costs about one Mathlib import (≈ 1 min, ≈ 4.5 GB), so a chapter of 25 modules checks in about 25
minutes sequentially.

## Veriphy 0.3.0

Rerun of the chapter on 2026-10-16/17 against the pinned release 0.3.0, whose recipes `dimension`, `matrix_rank`,
`independence`, `nonlinearity`, `signature`, `definite_integral`, the extended `matrix_from_goal` (vectors, named
matrices, any size) and the polynomial translation were written for the gaps listed above. Every pending claim now has
a level; the gate passes (`check_verification.py part-i-02-linear-algebra --strict`: 0 problems; `check_format.py
--strict`: 0 problems).

| level | claim entries (0.2 pilot) | claim entries (0.3.0) |
|---|---|---|
| certificate | 205 | 367 |
| kernel-test | 0 | 142 |
| oracle-test | 0 | 5 |
| pending | 322 | 0 |
| total | 527 | 514 |

The total fell by 13: definitions that state no property (`VectorSpace`, `linComb`, `spanOf`, `fBasis`,
`LinearFunctional`, `doubleDual`, `HasSignature`, `tensorBasis`, `dualTensorEquiv`, `MixedTensor`, `basis33`,
`planeSum0`, and the faithfulness lemma `linearIndependent_iff_coeffs`) are no longer claims (D29); their Lean code is
still shown in the definitions' callouts. "The antisymmetric matrices form a subspace" now points to a closure theorem
(`antisym_add_smul`). The two `none` blocks are unchanged.

Certificates by recipe (claim entries): `matrix_from_goal` 198, `dimension` 41, `independence` 32,
`ideal_membership` 28, `matrix_rank` 14, `witness` 13, `signature` 11, `nonlinearity` 8, `infeasibility` 7,
`definite_integral` 6, `transcendental_rewrite` 4, `sum_induction` 3, `farkas` 1, `sos` 1. New scripts (all copies of
0.3.0 templates, adapted to read their data from the goal): `la-matrix-named`, `la-independence`,
`la-independence-eval`, `la-dependence-functions`, `la-span-member`, `la-span-eq`, `la-span-ker`, `la-span-so3`,
`la-dim-span`, `la-dim-product`, `la-dim-ker`, `la-dim-inter`, `la-rank`, `la-nonlinear`, `la-nonlinear-norm`,
`la-signature`, `la-integral`, `la-polynomial`, `la-infeasible`; 114 `la-*-test` scripts for the tests. Lean: 25
modules plus `KaluzaKlein.PartI.LinearAlgebra` (aggregator), 424 theorems, all on `propext`, `Classical.choice`,
`Quot.sound` only; 127 `veriphy_certified` sites and 114 `veriphy_tested` attributes.

How the old gaps closed: the 51 dimension counts by `dimension` (spans, kernels of functionals, intersections, product
spaces) or, when they quantify over `n`, by tests of the general statement (a count at a fixed `n` such as
`dim so(3) = 3` now points to the tested `finrank_antisym`); ranks by `matrix_rank` (`¬ ∃ u v, u vᵀ = 1` via the rank
of the literal identity); glue to independence and spans by `independence` (spans equal to kernels, membership by
coordinates); closed evaluations with named matrices and vectors by `matrix_from_goal`; integrals by
`definite_integral`; non-linearity by `nonlinearity`; signatures by `signature` (the chapter's `HasSignature` is now an
abbreviation of `Veriphy.Linear.HasSignature`); polynomial identities through the polynomial translation (as zero-ideal
membership, `ideal_membership`); `≠` and `∉` facts by `infeasibility` after extracting the contradictory entry.

Goals that still needed a test, and why (feeds the next release):

- **Quantified over a dimension or size** (99 entries): rank-nullity, `dim Sym(n)`, `dim (Fin n → ℝ)`, transformation
  laws `A⁻¹ T A`, `det diag`, multi-index counts, Fourier truncations. No recipe certifies a statement for every `n`
  (an induction or a parametric-size emitter would).
- **Quantified over fields, spaces or index types** (23): dual bases, `dim (V ⊗ W)`, `V ≅ V**`, alternating maps.
- **Fixed size, opaque entries** (15): identities with a vector variable `v : Fin n → ℝ` or hypotheses on matrices
  (`mem_ker_A`, `polarisation`, `decomposition_unique`); `matrix_from_goal` reads literals, not function-valued
  atoms. Needed: expand a vector atom into its components (`![v 0, v 1, v 2]`) in the translation.
- **Finite index families** (6, tested exhaustively): component tables such as `T_components`, `dual_basis_delta`;
  a `fin_cases` emitter over finite binders would turn them into certificates.
- **Arbitrary functions** (4, two at oracle level): linearity of `∫` and evaluation; Lean cannot compute the
  instances' integrals by kernel evaluation without the FTC step, so Sage evaluates them exactly.
- **Oracle level** (5 entries, 4 claims): `integral_additive`, `integration_linear` (exact integrals by Sage),
  `rank_one_iff_proportional` and `nondegenerate_iff_det` (iff statements whose false side needs a rank or kernel
  computation at each instance; Sage compares both sides).

Infrastructure notes: `.veriphy/certify.py` worked unchanged; the shared lock (`scripts/lean_lock.py`) raised
`FileNotFoundError` when a slot disappeared between `isdir` and `getmtime` (worked around by retrying); with several
agents and heavy `AbelianKK` builds, most of the run's wall clock (about 6 h) was waiting for a Lean slot, the Lean and
Sage work itself about 1 h. Lean/Veriphy pitfalls met: `simp +decide` on determinants recurses too deep over ℝ (use
`norm_num` with the Laplace lemmas); `Finset.card_filter` loops under `simp` (use `simp only`); `infeasibility` needs
the unrelated hypotheses cleared; `Matrix.det ![u, v, w]` needs `Matrix.of` before `det_fin_three`; numeral-only goals
such as `derivative (1 : ℝ[X])` have no symbolic form (`X ^ 0` translates).

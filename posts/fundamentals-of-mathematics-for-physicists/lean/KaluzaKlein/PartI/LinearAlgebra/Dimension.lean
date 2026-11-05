import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.Bases
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Dimension"

The claims of the section "Dimension" of `tutorial/part-i-02-linear-algebra.qmd`: dimensions of coordinate spaces,
matrices, `𝔰𝔬(n)` and the symmetric matrices, of the span of the running triple, of a tangent space, and of spans of
powers of `cos φ`.
-/

namespace KaluzaKlein.LinearAlgebra.Dimension

open Matrix Module KaluzaKlein.LinearAlgebra.VectorSpaces KaluzaKlein.LinearAlgebra.Spans
  KaluzaKlein.LinearAlgebra.LinearIndependence

/-! ## def-dimension -/

/-- All bases of a finite-dimensional space have the same number of elements, the dimension. -/
theorem bases_same_card {K V ι ι' : Type*} [Field K] [AddCommGroup V] [Module K V] [Fintype ι] [Fintype ι']
    (b : Basis ι K V) (b' : Basis ι' K V) : Fintype.card ι = Fintype.card ι' :=
  (Module.finrank_eq_card_basis b).symm.trans (Module.finrank_eq_card_basis b')

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.bases_same_card (ℚ) ((Fin 0 → ℚ)) ((Fin 0)) ((Fin 0)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 0)) (Pi.basisFun ℚ (Fin 0)) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.bases_same_card (ℚ) ((Fin 2 → ℚ)) ((Fin 2)) (Bool) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) ((Pi.basisFun ℚ (Fin 2)).reindex finTwoEquiv) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.bases_same_card (ℚ) ((Fin 3 → ℚ)) ((Fin 3)) ((Fin 3)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 3)) ((Pi.basisFun ℚ (Fin 3)).reindex (Equiv.swap 0 2)) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Dimension.bases_same_card
-- test: la-bases-same-card-test.sage

/-! ## exm-dimension-rn -/

/-- `dim ℝⁿ = n`. -/
theorem finrank_rn (n : ℕ) : finrank ℝ (Fin n → ℝ) = n := Module.finrank_fin_fun ℝ

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_rn (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_rn (4) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_rn (1) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Dimension.finrank_rn
-- test: la-finrank-rn-test.sage

/-! ## exm-dimension-span -/

/-- The span of `(1, 2, 0)`, `(0, 1, 1)`, `(1, 3, 1)` has dimension `2`. -/
theorem finrank_span_v123 :
    finrank ℚ (spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2, KaluzaKlein.Linear.v3]) = 2 := by
  show finrank ℚ ↥(Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1], ![1, 3, 1]])) = 2
  veriphy_certified
    ( have hspan :
        Submodule.span ℚ (Set.range (![![1, 2, 0], ![0, 1, 1], ![1, 3, 1]] : Fin 3 → Fin 3 → ℚ)) =
          Submodule.span ℚ (Set.range (![![1, 0, -2], ![0, 1, 1]] : Fin 2 → Fin 3 → ℚ)) :=
        by
        apply le_antisymm
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i
          ·
            exact
              (Submodule.mem_span_range_iff_exists_fun ℚ).2
                ⟨![1, 2], by
                  ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
          ·
            exact
              (Submodule.mem_span_range_iff_exists_fun ℚ).2
                ⟨![0, 1], by
                  ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
          ·
            exact
              (Submodule.mem_span_range_iff_exists_fun ℚ).2
                ⟨![1, 3], by
                  ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i
          ·
            exact
              (Submodule.mem_span_range_iff_exists_fun ℚ).2
                ⟨![1, -2, 0], by
                  ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
          ·
            exact
              (Submodule.mem_span_range_iff_exists_fun ℚ).2
                ⟨![0, 1, 0], by
                  ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
      rw [hspan]
      exact
        Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-dim-span.sage

/-! ## exm-dimension-tangent -/

/-- The tangent space of an `n`-dimensional manifold modelled on `ℝⁿ` has dimension `n`. -/
theorem finrank_tangentSpace (n : ℕ) (x : EuclideanSpace ℝ (Fin n)) :
    finrank ℝ (TangentSpace (modelWithCornersSelf ℝ (EuclideanSpace ℝ (Fin n))) x) = n :=
  finrank_euclideanSpace_fin

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_tangentSpace (0) (0) := by
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_tangentSpace (2) (0) := by
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_tangentSpace (3) (EuclideanSpace.single 0 1) := by
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Dimension.finrank_tangentSpace
-- test: la-finrank-tangent-test.sage

/-! ## exr-span-dimension -/

/-- Coordinates of `(2, 5, 1)` in the basis `(1, 2, 0)`, `(0, 1, 1)`, `(0, 0, 1)` exist. -/
theorem span_dimension_coords_exist : ∃ a b c : ℚ,
    a * 1 + b * 0 + c * 0 = 2 ∧ a * 2 + b * 1 + c * 0 = 5 ∧ a * 0 + b * 1 + c * 1 = 1 := by
  exact ⟨2, 1, 0, by norm_num⟩
  -- certificate: la-span-dimension.sage

/-- `(2, 5, 1) = 2 v₁ + 1 v₂ + 0 v₄`. -/
theorem span_dimension_coords :
    (2 : ℚ) • !![(1 : ℚ), 2, 0] + (1 : ℚ) • !![(0 : ℚ), 1, 1] + (0 : ℚ) • !![(0 : ℚ), 0, 1] = !![(2 : ℚ), 5, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-dimension-symmetric -/

/-- Every matrix is `½ (A + Aᵀ) + ½ (A - Aᵀ)`, a symmetric plus an antisymmetric matrix. -/
theorem sym_antisym_decomposition {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    A = (1 / 2 : ℝ) • (A + Aᵀ) + (1 / 2 : ℝ) • (A - Aᵀ) ∧ (1 / 2 : ℝ) • (A + Aᵀ) ∈ symMatrices n ∧
      (1 / 2 : ℝ) • (A - Aᵀ) ∈ antisymMatrices n := by
  refine ⟨by module, ?_, ?_⟩
  · show ((1 / 2 : ℝ) • (A + Aᵀ))ᵀ = (1 / 2 : ℝ) • (A + Aᵀ)
    rw [Matrix.transpose_smul, Matrix.transpose_add, Matrix.transpose_transpose, add_comm]
  · show ((1 / 2 : ℝ) • (A - Aᵀ))ᵀ = -((1 / 2 : ℝ) • (A - Aᵀ))
    rw [Matrix.transpose_smul, Matrix.transpose_sub, Matrix.transpose_transpose, ← smul_neg, neg_sub]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_antisym_decomposition (0) (0) := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = _; ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = -_; ext i j; fin_cases i <;> fin_cases j <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_antisym_decomposition (2) (!![1, 2; 3, 4]) := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = _; ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = -_; ext i j; fin_cases i <;> fin_cases j <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_antisym_decomposition (3) (!![1, 0, 2; -1, 3, 5; 4, 4, 0]) := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = _; ext i j; fin_cases i <;> fin_cases j <;> norm_num
  · show Matrix.transpose _ = -_; ext i j; fin_cases i <;> fin_cases j <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Dimension.sym_antisym_decomposition
-- test: la-sym-antisym-decomposition-test.sage

/-- The decomposition is unique: a matrix both symmetric and antisymmetric vanishes. -/
theorem sym_inf_antisym {n : ℕ} : symMatrices n ⊓ antisymMatrices n = ⊥ := by
  rw [eq_bot_iff]
  rintro A ⟨hs, ha⟩
  change Aᵀ = A at hs
  change Aᵀ = -A at ha
  rw [Submodule.mem_bot]
  have : A = -A := hs.symm.trans ha
  ext i j
  have := congrFun (congrFun this i) j
  simp at this ⊢
  linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym (0) := by
  rw [eq_bot_iff]
  rintro A ⟨hs, ha⟩
  change Aᵀ = A at hs
  change Aᵀ = -A at ha
  rw [Submodule.mem_bot]
  ext i j
  have h1 := congrFun (congrFun hs i) j
  have h2 := congrFun (congrFun ha i) j
  simp at h1 h2 ⊢
  linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym (1) := by
  rw [eq_bot_iff]
  rintro A ⟨hs, ha⟩
  change Aᵀ = A at hs
  change Aᵀ = -A at ha
  rw [Submodule.mem_bot]
  ext i j
  have h1 := congrFun (congrFun hs i) j
  have h2 := congrFun (congrFun ha i) j
  simp at h1 h2 ⊢
  linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym (2) := by
  rw [eq_bot_iff]
  rintro A ⟨hs, ha⟩
  change Aᵀ = A at hs
  change Aᵀ = -A at ha
  rw [Submodule.mem_bot]
  ext i j
  have h1 := congrFun (congrFun hs i) j
  have h2 := congrFun (congrFun ha i) j
  simp at h1 h2 ⊢
  linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym (3) := by
  rw [eq_bot_iff]
  rintro A ⟨hs, ha⟩
  change Aᵀ = A at hs
  change Aᵀ = -A at ha
  rw [Submodule.mem_bot]
  ext i j
  have h1 := congrFun (congrFun hs i) j
  have h2 := congrFun (congrFun ha i) j
  simp at h1 h2 ⊢
  linarith

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym
-- test: la-sym-inf-antisym-test.sage

/-! ## exr-dimension-fourier (powers of cos) -/

/-- The powers `cos^k φ`, `k = 0, …, N`, are independent: a vanishing combination is a polynomial in `cos φ` with
infinitely many roots (all of `[-1, 1]`). -/
theorem cos_powers_linearIndependent (N : ℕ) :
    LinearIndependent ℝ (fun k : Fin (N + 1) => fun φ : ℝ => Real.cos φ ^ (k : ℕ)) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  set P : Polynomial ℝ := ∑ k : Fin (N + 1), Polynomial.C (g k) * Polynomial.X ^ (k : ℕ) with hP
  have hroot : ∀ t ∈ Set.Icc (-1 : ℝ) 1, P.IsRoot t := by
    intro t ht
    have := congrFun hg (Real.arccos t)
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply, Real.cos_arccos ht.1 ht.2] at this
    simp [Polynomial.IsRoot, hP, Polynomial.eval_finsetSum, this]
  have hP0 : P = 0 := Polynomial.eq_zero_of_infinite_isRoot P
    ((Set.Icc_infinite (by norm_num : (-1 : ℝ) < 1)).mono hroot)
  intro k
  have := congrArg (Polynomial.coeff · k) hP0
  simp only [hP, Polynomial.finsetSum_coeff, Polynomial.coeff_C_mul_X_pow, Polynomial.coeff_zero] at this
  rw [Finset.sum_eq_single k (fun j _ hj => by simp [Fin.val_ne_of_ne hj.symm]) (by simp)] at this
  simpa using this

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.cos_powers_linearIndependent (0) := by
  refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 1 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0] k)) ?_
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.cos_powers_linearIndependent (1) := by
  refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 2 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2] k)) ?_
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.cos_powers_linearIndependent (2) := by
  refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 3 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2, Real.pi] k)) ?_
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Dimension.cos_powers_linearIndependent
-- test: la-cos-powers-test.sage

/-- The span of `cos^k φ`, `k ≤ N`, has dimension `N + 1`. -/
theorem finrank_span_cos_powers (N : ℕ) :
    finrank ℝ (Submodule.span ℝ (Set.range (fun k : Fin (N + 1) => fun φ : ℝ => Real.cos φ ^ (k : ℕ)))) = N + 1 := by
  rw [finrank_span_eq_card (cos_powers_linearIndependent N)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_cos_powers (0) := by
  rw [finrank_span_eq_card (cos_powers_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_cos_powers (1) := by
  rw [finrank_span_eq_card (cos_powers_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_cos_powers (2) := by
  rw [finrank_span_eq_card (cos_powers_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_cos_powers (3) := by
  rw [finrank_span_eq_card (cos_powers_linearIndependent _)]
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Dimension.finrank_span_cos_powers
-- test: la-finrank-cos-powers-test.sage


/-! ## exr-dimension-fourier (all modes), exm-span-functions -/

open Polynomial in
/-- Polynomials `p n` of degree exactly `n` are independent: a vanishing combination has zero coefficients. -/
theorem coeffs_zero_of_degrees (p : ℕ → ℝ[X]) (hdeg : ∀ n, (p n).natDegree = n) (hlc : ∀ n, (p n).leadingCoeff ≠ 0)
    (s : Finset ℕ) : ∀ c : ℕ → ℝ, ∑ n ∈ s, C (c n) * p n = 0 → ∀ n ∈ s, c n = 0 := by
  induction s using Finset.induction_on_max with
  | empty => simp
  | insert a s hlt ih =>
    intro c hsum
    have ha : a ∉ s := fun h => lt_irrefl a (hlt a h)
    rw [Finset.sum_insert ha] at hsum
    have hcoef := congrArg (Polynomial.coeff · a) hsum
    simp only [Polynomial.coeff_add, Polynomial.finsetSum_coeff, Polynomial.coeff_C_mul, Polynomial.coeff_zero]
      at hcoef
    have hz : ∑ x ∈ s, c x * (p x).coeff a = 0 := Finset.sum_eq_zero fun x hx => by
      rw [Polynomial.coeff_eq_zero_of_natDegree_lt (by rw [hdeg]; exact hlt x hx), mul_zero]
    rw [hz, add_zero] at hcoef
    have hca : c a = 0 := by
      have : (p a).coeff a = (p a).leadingCoeff := by rw [Polynomial.leadingCoeff, hdeg]
      rw [this] at hcoef
      exact (mul_eq_zero.mp hcoef).resolve_right (hlc a)
    intro n hn
    rcases Finset.mem_insert.mp hn with rfl | hn
    · exact hca
    · rw [hca, Polynomial.C_0, zero_mul, zero_add] at hsum
      exact ih c hsum n hn

open Polynomial in
/-- The modes `cos nφ`, `n = 0, 1, 2, …`, are independent (`cos nφ = T_n(cos φ)`, Chebyshev). -/
theorem cos_modes_linearIndependent :
    LinearIndependent ℝ (fun n : ℕ => fun φ : ℝ => Real.cos (n * φ)) := by
  rw [linearIndependent_iff']
  intro s g hg i hi
  set P : ℝ[X] := ∑ n ∈ s, C (g n) * Chebyshev.T ℝ n with hP
  have key : ∀ t ∈ Set.Icc (-1 : ℝ) 1, ∀ n : ℕ, (Chebyshev.T ℝ n).eval t = Real.cos (n * Real.arccos t) := by
    intro t ht n
    conv_lhs => rw [← Real.cos_arccos ht.1 ht.2]
    rw [Chebyshev.T_real_cos]
    simp
  have hroot : ∀ t ∈ Set.Icc (-1 : ℝ) 1, P.IsRoot t := by
    intro t ht
    have := congrFun hg (Real.arccos t)
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at this
    simp [Polynomial.IsRoot, hP, Polynomial.eval_finsetSum, key t ht, this]
  have hP0 : P = 0 := Polynomial.eq_zero_of_infinite_isRoot P
    ((Set.Icc_infinite (by norm_num : (-1 : ℝ) < 1)).mono hroot)
  refine coeffs_zero_of_degrees (fun n => Chebyshev.T ℝ n) (fun n => by simp [Chebyshev.natDegree_T]) (fun n => ?_)
    s g hP0 i hi
  rw [Ne, Polynomial.leadingCoeff_eq_zero]
  intro h
  have hd := Chebyshev.natDegree_T ℝ (n : ℤ)
  rw [h, Polynomial.natDegree_zero] at hd
  have hn : n = 0 := by simpa using hd.symm
  subst hn
  simp at h

open Polynomial in
/-- The modes `sin (n + 1)φ`, `n = 0, 1, 2, …`, are independent (`sin (n + 1)φ = U_n(cos φ) sin φ`). -/
theorem sin_modes_linearIndependent :
    LinearIndependent ℝ (fun n : ℕ => fun φ : ℝ => Real.sin ((n + 1) * φ)) := by
  rw [linearIndependent_iff']
  intro s g hg i hi
  set P : ℝ[X] := ∑ n ∈ s, C (g n) * Chebyshev.U ℝ n with hP
  have hroot : ∀ t ∈ Set.Ioo (-1 : ℝ) 1, P.IsRoot t := by
    intro t ht
    have hsin : 0 < Real.sin (Real.arccos t) := by
      rw [Real.sin_arccos]
      apply Real.sqrt_pos.mpr
      nlinarith [ht.1, ht.2]
    have key : ∀ n : ℕ, (Chebyshev.U ℝ n).eval t * Real.sin (Real.arccos t) =
        Real.sin ((n + 1) * Real.arccos t) := by
      intro n
      have h := Chebyshev.U_real_cos (θ := Real.arccos t) (n := (n : ℤ))
      rw [Real.cos_arccos ht.1.le ht.2.le] at h
      simpa using h
    have := congrFun hg (Real.arccos t)
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at this
    have hsum : P.eval t * Real.sin (Real.arccos t) = 0 := by
      rw [hP, Polynomial.eval_finsetSum, Finset.sum_mul, ← this]
      refine Finset.sum_congr rfl fun n _ => ?_
      rw [Polynomial.eval_mul, Polynomial.eval_C, mul_assoc, key]
    exact (mul_eq_zero.mp hsum).resolve_right hsin.ne'
  have hP0 : P = 0 := Polynomial.eq_zero_of_infinite_isRoot P
    ((Set.Ioo_infinite (by norm_num : (-1 : ℝ) < 1)).mono hroot)
  refine coeffs_zero_of_degrees (fun n => Chebyshev.U ℝ n) (fun n => Chebyshev.natDegree_U_natCast ℝ n) (fun n => ?_)
    s g hP0 i hi
  rw [Ne, Polynomial.leadingCoeff_eq_zero]
  intro h
  have hd := Chebyshev.natDegree_U_natCast ℝ n
  rw [h, Polynomial.natDegree_zero] at hd
  subst hd
  simp at h

/-- Even functions. -/
def evenFunctions : Submodule ℝ (ℝ → ℝ) where
  carrier := {f | ∀ x, f (-x) = f x}
  add_mem' := fun ha hb x => by simp [ha x, hb x]
  zero_mem' := fun x => rfl
  smul_mem' := fun c f hf x => by simp [hf x]

/-- Odd functions. -/
def oddFunctions : Submodule ℝ (ℝ → ℝ) where
  carrier := {f | ∀ x, f (-x) = -f x}
  add_mem' := fun ha hb x => by simp [ha x, hb x]; ring
  zero_mem' := fun x => by simp
  smul_mem' := fun c f hf x => by simp [hf x]

/-- The full family of Fourier modes `cos nφ` (`n ≥ 0`) and `sin (n + 1)φ` (`n ≥ 0`). -/
noncomputable def fourierModes : ℕ ⊕ ℕ → ℝ → ℝ :=
  Sum.elim (fun n φ => Real.cos (n * φ)) (fun n φ => Real.sin ((n + 1) * φ))

/-- All the Fourier modes are independent: the cosine modes span even functions, the sine modes odd ones. -/
theorem fourierModes_linearIndependent : LinearIndependent ℝ fourierModes := by
  apply LinearIndependent.sum_type cos_modes_linearIndependent sin_modes_linearIndependent
  rw [Submodule.disjoint_def]
  intro f hc hs
  have he : f ∈ evenFunctions := (Submodule.span_le.mpr (by
    rintro g ⟨n, rfl⟩ x; simp [Real.cos_neg])) hc
  have ho : f ∈ oddFunctions := (Submodule.span_le.mpr (by
    rintro g ⟨n, rfl⟩ x; simp [Real.sin_neg])) hs
  funext x
  have h1 := he x
  have h2 := ho x
  simp only [Pi.zero_apply]
  linarith

/-- The modes `1, cos φ, …, cos Nφ, sin φ, …, sin Nφ` are independent, for every `N`. -/
theorem fourierModes_truncation_linearIndependent (N : ℕ) :
    LinearIndependent ℝ (fourierModes ∘ Sum.map (fun k : Fin (N + 1) => (k : ℕ)) (fun k : Fin N => (k : ℕ))) :=
  fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent (0) := by
  exact fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent (1) := by
  exact fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent (2) := by
  exact fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent (3) := by
  exact fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent
-- test: la-fourier-truncation-test.sage

/-- `1, cos φ, sin φ, …, cos Nφ, sin Nφ` span a space of dimension `2N + 1`. -/
theorem finrank_span_fourier_modes (N : ℕ) :
    finrank ℝ (Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin (N + 1) => (k : ℕ))
      (fun k : Fin N => (k : ℕ))))) = 2 * N + 1 := by
  have hinj : Function.Injective (Sum.map (fun k : Fin (N + 1) => (k : ℕ)) (fun k : Fin N => (k : ℕ))) :=
    Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩
  rw [finrank_span_eq_card (fourierModes_linearIndependent.comp _ hinj)]
  simp
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_fourier_modes (0) := by
  rw [finrank_span_eq_card (fourierModes_truncation_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_fourier_modes (1) := by
  rw [finrank_span_eq_card (fourierModes_truncation_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_fourier_modes (2) := by
  rw [finrank_span_eq_card (fourierModes_truncation_linearIndependent _)]
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.finrank_span_fourier_modes (3) := by
  rw [finrank_span_eq_card (fourierModes_truncation_linearIndependent _)]
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Dimension.finrank_span_fourier_modes
-- test: la-finrank-fourier-test.sage

/-- The span of all the Fourier modes is infinite-dimensional. -/
theorem fourier_modes_infinite_dimensional :
    ¬ FiniteDimensional ℝ (Submodule.span ℝ (Set.range fourierModes)) := by
  intro h
  set N := finrank ℝ (Submodule.span ℝ (Set.range fourierModes))
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin (N + 1) => (k : ℕ))
      (fun k : Fin N => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes] at this
  omega

/-- The span of all the modes has no finite dimension `N`: it contains `2N + 3` independent modes. -/
theorem fourier_span_finrank_ne (N : ℕ) :
    ¬ (FiniteDimensional ℝ (Submodule.span ℝ (Set.range fourierModes)) ∧
      finrank ℝ (Submodule.span ℝ (Set.range fourierModes)) = N) := by
  rintro ⟨h, hN⟩
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin (N + 2) => (k : ℕ))
      (fun k : Fin (N + 1) => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes, hN] at this
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne (0) := by
  rintro ⟨h, hN⟩
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 2 => (k : ℕ)) (fun k : Fin 1 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes, hN] at this
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne (1) := by
  rintro ⟨h, hN⟩
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 3 => (k : ℕ)) (fun k : Fin 2 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes, hN] at this
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne (2) := by
  rintro ⟨h, hN⟩
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 4 => (k : ℕ)) (fun k : Fin 3 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes, hN] at this
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne (5) := by
  rintro ⟨h, hN⟩
  have hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 7 => (k : ℕ)) (fun k : Fin 6 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=
    Submodule.span_mono (Set.range_comp_subset_range _ _)
  have := Submodule.finrank_mono hle
  rw [finrank_span_fourier_modes, hN] at this
  omega

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne
-- test: la-fourier-span-finrank-test.sage

end KaluzaKlein.LinearAlgebra.Dimension

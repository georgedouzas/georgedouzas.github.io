import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.Spans
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Linear independence"

The claims of the section "Linear independence" of `tutorial/part-i-02-linear-algebra.qmd`: the independent and
dependent triples of `ℚ³`, the pairs of the plane, the functions `1, cos, sin` and `1, cos², sin²`, independence by a
`4 × 4` determinant, and the extension of an independent family.
-/

namespace KaluzaKlein.LinearAlgebra.LinearIndependence

open Matrix Module

/-- Rows of a square matrix with non-zero determinant are independent (mathlib). -/
theorem rows_independent {n : ℕ} {K : Type*} [Field K] (A : Matrix (Fin n) (Fin n) K) (h : A.det ≠ 0) :
    LinearIndependent K (fun i => A i) :=
  Matrix.linearIndependent_rows_of_det_ne_zero h

/-! ## def-linear-independence -/

/-- Linear independence is mathlib's `LinearIndependent`: for a finite family, a vanishing combination has zero
coefficients. -/
theorem linearIndependent_iff_coeffs {ι V : Type*} [Fintype ι] [AddCommGroup V] [Module ℚ V] (v : ι → V) :
    LinearIndependent ℚ v ↔ ∀ g : ι → ℚ, ∑ i, g i • v i = 0 → ∀ i, g i = 0 :=
  Fintype.linearIndependent_iff

/-! ## exm-independent-r3 -/

theorem det_v124 : Matrix.det !![(1 : ℚ), 2, 0; 0, 1, 1; 0, 0, 1] = 1 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-matrix.sage

/-- `(1, 2, 0)`, `(0, 1, 1)`, `(0, 0, 1)` are independent. -/
theorem v124_linearIndependent : LinearIndependent ℚ ![(![1, 2, 0] : Fin 3 → ℚ), ![0, 1, 1], ![0, 0, 1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-- `v₁ + v₂ - v₃ = 0`. -/
theorem v123_relation :
    !![(1 : ℚ), 2, 0] + !![(0 : ℚ), 1, 1] - !![(1 : ℚ), 3, 1] = !![(0 : ℚ), 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- `(1, 2, 0)`, `(0, 1, 1)`, `(1, 3, 1)` are dependent. -/
theorem v123_dependent : ¬ LinearIndependent ℚ ![(![1, 2, 0] : Fin 3 → ℚ), ![0, 1, 1], ![1, 3, 1]] := by
  veriphy_certified
    ( rw [Fintype.not_linearIndependent_iff]
      refine ⟨![1, 1, -1], ?_, 0, by norm_num⟩
      ext j
      fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num)
  -- certificate: la-independence.sage

/-! ## exm-independent-basis-change -/

/-- `2 (2, 1) - (4, 2) = 0`: `(2, 1)` and `(4, 2)` are dependent. -/
theorem pair_relation : (2 : ℚ) • !![(2 : ℚ), 1] - !![(4 : ℚ), 2] = !![(0 : ℚ), 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

theorem pair_dependent : ¬ LinearIndependent ℚ ![(![2, 1] : Fin 2 → ℚ), ![4, 2]] := by
  veriphy_certified
    ( rw [Fintype.not_linearIndependent_iff]
      refine ⟨![2, -1], ?_, 0, by norm_num⟩
      ext j
      fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num)
  -- certificate: la-independence.sage

/-! ## exm-independent-functions, exr-independence-functions -/

/-- Values at `0`, `π`, `π/2` of a vanishing `a + b cos φ + c sin φ`: `a + b = 0`, `a - b = 0`, `a + c = 0` force
`a = b = c = 0`. -/
theorem trig_coefficients (a b c : ℝ) (h0 : a + b = 0) (hpi : a - b = 0) (h2 : a + c = 0) :
    a = 0 ∧ b = 0 ∧ c = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · linear_combination (1 / 2) * h0 + (1 / 2) * hpi
    -- certificate: ideal_membership.sage
  · linear_combination (1 / 2) * h0 + (-1 / 2) * hpi
    -- certificate: ideal_membership.sage
  · linear_combination (-1 / 2) * h0 + (-1 / 2) * hpi + h2
    -- certificate: ideal_membership.sage

/-- `1, cos φ, sin φ` are independent functions. -/
theorem one_cos_sin_linearIndependent :
    LinearIndependent ℝ ![(fun _ => (1 : ℝ)), Real.cos, Real.sin] := by
  veriphy_certified
    ( refine
        LinearIndependent.of_comp
          (LinearMap.pi fun k : Fin 3 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2, Real.pi] k))
          ?_
      exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
          (by
            norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
              Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply,
              Matrix.entryLinearMap_apply, Polynomial.lcoeff_apply, Polynomial.coeff_one, Polynomial.coeff_X,
              Polynomial.coeff_sub, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi,
              Real.sin_pi, KaluzaKlein.LinearAlgebra.LinearCombinations.L1,
              KaluzaKlein.LinearAlgebra.LinearCombinations.L2, KaluzaKlein.LinearAlgebra.LinearCombinations.L3]))
  -- certificate: la-independence-eval.sage

/-- `cos² φ + sin² φ = 1`. -/
theorem cos_sq_add_sin_sq (x : ℝ) : Real.cos x ^ 2 + Real.sin x ^ 2 = 1 := by
  rw [Real.cos_sq_add_sin_sq]
  -- certificate: la-trig-pythagoras.sage

/-- `1, cos², sin²` are dependent: `1 - cos² - sin² = 0`. -/
theorem one_cos_sq_sin_sq_dependent :
    ¬ LinearIndependent ℝ ![(fun _ => (1 : ℝ)), (fun x => Real.cos x ^ 2), (fun x => Real.sin x ^ 2)] := by
  veriphy_certified
    ( rw [Fintype.not_linearIndependent_iff]
      refine ⟨![1, -1, -1], ?_, 0, by norm_num⟩
      funext x
      simp [Fin.sum_univ_succ]
      linear_combination (-1) * Real.cos_sq_add_sin_sq x)
  -- certificate: la-dependence-functions.sage

/-! ## exr-independence-determinant -/

theorem det_family_F :
    Matrix.det !![(1 : ℚ), 0, 1, 0; 0, 1, 1, 0; 1, 0, 0, 1; 0, 1, 0, -1] = 2 := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-independence-determinant.sage

theorem det_family_G :
    Matrix.det !![(1 : ℚ), 0, 1, 0; 0, 1, 1, 0; 1, 0, 0, 1; 0, 1, 0, 1] = 0 := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-independence-determinant.sage

/-- `v₁ + v₂ - v₃ - v₄ = 0` for the second family. -/
theorem family_G_relation :
    !![(1 : ℚ), 0, 1, 0] + !![(0 : ℚ), 1, 0, 1] - !![(1 : ℚ), 1, 0, 0] - !![(0 : ℚ), 0, 1, 1] =
      !![(0 : ℚ), 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-independence-determinant.sage

/-- The determinant of the matrix whose rows are the vectors of the first family (the transpose). -/
theorem det_family_F_rows :
    Matrix.det !![(1 : ℚ), 0, 1, 0; 0, 1, 0, 1; 1, 1, 0, 0; 0, 0, 1, -1] = 2 := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-independence-determinant.sage

/-- The first family is independent (its determinant is `2 ≠ 0`). -/
theorem family_F_independent : LinearIndependent ℚ
    ![(![1, 0, 1, 0] : Fin 4 → ℚ), ![0, 1, 0, 1], ![1, 1, 0, 0], ![0, 0, 1, -1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2, 3]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-- The second family is dependent. -/
theorem family_G_dependent : ¬ LinearIndependent ℚ
    ![(![1, 0, 1, 0] : Fin 4 → ℚ), ![0, 1, 0, 1], ![1, 1, 0, 0], ![0, 0, 1, 1]] := by
  veriphy_certified
    ( rw [Fintype.not_linearIndependent_iff]
      refine ⟨![1, 1, -1, -1], ?_, 0, by norm_num⟩
      ext j
      fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num)
  -- certificate: la-independence.sage

/-! ## exr-independence-extend -/

/-- An independent family extended by a vector outside its span stays independent. -/
theorem extend_independent {K V : Type*} [Field K] [AddCommGroup V] [Module K V] {n : ℕ} (v : Fin n → V) (x : V)
    (hv : LinearIndependent K v) (hx : x ∉ Submodule.span K (Set.range v)) :
    LinearIndependent K (Fin.cons x v : Fin (n + 1) → V) :=
  linearIndependent_finCons.mpr ⟨hv, hx⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearIndependence.extend_independent (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (0) (![]) (![1, 0]) (linearIndependent_empty_type) ((by simp)) := by
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearIndependence.extend_independent (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (1) (![![1, 0]]) (![0, 1]) ((Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]))) ((by rw [Submodule.mem_span_range_iff_exists_fun]; rintro ⟨c, hc⟩; have h0 := congrFun hc 0; have h1 := congrFun hc 1; simp [Fin.sum_univ_succ] at h0 h1 <;> linarith)) := by
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearIndependence.extend_independent (ℚ) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (2) (![![1, 2, 0], ![0, 1, 1]]) (![0, 0, 1]) ((Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]))) ((by rw [Submodule.mem_span_range_iff_exists_fun]; rintro ⟨c, hc⟩; have h0 := congrFun hc 0; have h1 := congrFun hc 1; have h2 := congrFun hc 2; simp [Fin.sum_univ_succ] at h0 h1 h2 <;> linarith)) := by
  exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearIndependence.extend_independent
-- test: la-extend-independent-test.sage

end KaluzaKlein.LinearAlgebra.LinearIndependence

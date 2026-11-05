import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Non-degenerate forms"

The claims of the section "Non-degenerate forms" of `tutorial/part-i-02-linear-algebra.qmd`: the determinant test,
lowering and raising indices, the Minkowski and Euclidean forms, the degenerate form `S = (1, 2; 2, 4)` and
non-degeneracy as an isomorphism `V ≅ V*`.
-/

namespace KaluzaKlein.LinearAlgebra.NondegenerateForms

open Matrix Module

/-! ## def-nondegenerate, exr-nondegenerate-iso -/

/-- Non-degeneracy (`B(v, w) = 0` for all `w` forces `v = 0`) is equivalent to `det B ≠ 0`. -/
theorem nondegenerate_iff_det {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : B.Nondegenerate ↔ B.det ≠ 0 :=
  Matrix.nondegenerate_iff_det_ne_zero

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.LinearAlgebra.NondegenerateForms.nondegenerate_iff_det
-- test: la-nondegenerate-iff-det-test.sage

/-- Raising undoes lowering: `B⁻¹ (B v) = v` for a non-degenerate `B`. -/
theorem raise_lower {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (h : B.det ≠ 0) (v : Fin n → ℝ) : B⁻¹ *ᵥ (B *ᵥ v) = v := by
  rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul B (isUnit_iff_ne_zero.mpr h), Matrix.one_mulVec]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.raise_lower (0) (0) ((by simp)) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.raise_lower (2) (!![5, 4; 4, 5]) ((by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])) (![2, 1]) := by
  rw [Matrix.inv_eq_left_inv (B := !![(5 / 9), (-4 / 9); (-4 / 9), (5 / 9)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.raise_lower (4) (!![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]) ((by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])) (![2, 1, 0, 3]) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.NondegenerateForms.raise_lower
-- test: la-raise-lower-test.sage

/-- `v ↦ B(v, ·)`, in components `v ↦ B v`, is a linear isomorphism `V ≅ V*` when `det B ≠ 0`. -/
theorem lowering_bijective {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (h : B.det ≠ 0) :
    Function.Bijective (Matrix.toLin' B) := by
  have hu : IsUnit B := (Matrix.isUnit_iff_isUnit_det B).mpr (isUnit_iff_ne_zero.mpr h)
  exact ⟨Matrix.mulVec_injective_iff_isUnit.mpr hu, Matrix.mulVec_surjective_iff_isUnit.mpr hu⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.lowering_bijective (0) (0) ((by simp)) := by
  exact ⟨fun v w _ => funext fun i => i.elim0, fun v => ⟨v, funext fun i => i.elim0⟩⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.lowering_bijective (2) (!![5, 4; 4, 5]) ((by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])) := by
  refine Function.bijective_iff_has_inverse.mpr ⟨Matrix.toLin' !![(5 / 9), (-4 / 9); (-4 / 9), (5 / 9)], fun v => ?_, fun v => ?_⟩
  · simp only [Matrix.toLin'_apply, Matrix.mulVec_mulVec]
    rw [show (!![(5 / 9), (-4 / 9); (-4 / 9), (5 / 9)] * !![5, 4; 4, 5] : Matrix (Fin 2) (Fin 2) ℝ) = 1 by
      ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ], Matrix.one_mulVec]
  · simp only [Matrix.toLin'_apply, Matrix.mulVec_mulVec]
    rw [show (!![5, 4; 4, 5] * !![(5 / 9), (-4 / 9); (-4 / 9), (5 / 9)] : Matrix (Fin 2) (Fin 2) ℝ) = 1 by
      ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ], Matrix.one_mulVec]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.NondegenerateForms.lowering_bijective (4) (!![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]) ((by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])) := by
  refine Function.bijective_iff_has_inverse.mpr ⟨Matrix.toLin' !![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1], fun v => ?_, fun v => ?_⟩
  · simp only [Matrix.toLin'_apply, Matrix.mulVec_mulVec]
    rw [show (!![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] : Matrix (Fin 4) (Fin 4) ℝ) = 1 by
      ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ], Matrix.one_mulVec]
  · simp only [Matrix.toLin'_apply, Matrix.mulVec_mulVec]
    rw [show (!![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] : Matrix (Fin 4) (Fin 4) ℝ) = 1 by
      ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ], Matrix.one_mulVec]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.NondegenerateForms.lowering_bijective
-- test: la-lowering-bijective-test.sage

/-! ## exm-nondegenerate-minkowski -/

/-- `det δ = 1` for the Euclidean form on `ℝ³`. -/
theorem det_delta : Matrix.det !![(1 : ℝ), 0, 0; 0, 1, 0; 0, 0, 1] = 1 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-matrix.sage

/-- `det η = -1`. -/
theorem det_eta_literal :
    Matrix.det !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = -1 := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-nondegenerate-test.sage

/-! ## exm-nondegenerate-lowering -/

/-- `v_μ = η_{μν} v^ν = (1, -1, 0, 0)` for `v = (1, 1, 0, 0)`. -/
theorem lower_null :
    !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 1; 0; 0] = !![(1 : ℝ); -1; 0; 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-lower-index -/

/-- Raising `(2, -1, 0, -3)` with `η⁻¹ = η` returns `(2, 1, 0, 3)`. -/
theorem raise_back :
    !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(2 : ℝ); -1; 0; -3] = !![(2 : ℝ); 1; 0; 3] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- `v_μ v^μ = -6 < 0`: spacelike. -/
theorem lower_index_norm : !![(2 : ℝ), -1, 0, -3] * !![(2 : ℝ); 1; 0; 3] = !![(-6 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-nondegenerate-test -/

theorem det_S_test : Matrix.det !![(1 : ℝ), 2; 2, 4] = 0 := by
  norm_num [Matrix.det_fin_two_of]
  -- certificate: la-nondegenerate-test.sage

theorem det_tridiagonal : Matrix.det !![(2 : ℝ), 1, 0; 1, 2, 1; 0, 1, 2] = 4 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-nondegenerate-test.sage

/-- `(2, -1)` pairs to zero with `e₁` and `e₂` under `S`. -/
theorem S_kernel_pairing : !![(2 : ℝ), -1] * !![(1 : ℝ), 2; 2, 4] = !![(0 : ℝ), 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-nondegenerate-test.sage

/-- The vectors pairing to zero with everything under `S` form the line through `(2, -1)`. -/
theorem S_kernel_line :
    Submodule.span ℚ (Set.range ![![(2 : ℚ), -1]]) = LinearMap.ker KaluzaKlein.Linear.S.mulVecLin := by
  rw [show KaluzaKlein.Linear.S = !![(1 : ℚ), 2; 2, 4] from rfl]
  veriphy_certified
    ( apply le_antisymm
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i <;>
          (simp only [SetLike.mem_coe, LinearMap.mem_ker, Matrix.mulVecLin_apply];
            ext j <;> fin_cases j <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> norm_num)
      · intro x hx
        rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx
        have h0 := congrFun hx 0
        have h1 := congrFun hx 1
        simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0 h1
        refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![(1 / 2) * x 0], ?_⟩
        ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
  -- certificate: la-span-ker.sage

end KaluzaKlein.LinearAlgebra.NondegenerateForms

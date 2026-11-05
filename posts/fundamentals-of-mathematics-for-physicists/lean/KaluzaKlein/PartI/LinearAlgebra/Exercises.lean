import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.PartI.LinearAlgebra.Signature
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Exercises"

The claims of the three integrative exercises of `tutorial/part-i-02-linear-algebra.qmd`: the dot product in the
basis `f` (Gram matrix, signature, lowering and raising), a linear map as a type `(1, 1)` tensor, and an antisymmetric
tensor on Minkowski space.
-/

namespace KaluzaKlein.LinearAlgebra.Exercises

open Matrix Module KaluzaKlein.LinearAlgebra.Signature

/-! ## exr-la-gram-f -/

/-- `det G = 9 = (det A)²` for the Gram matrix `G = (5, 4; 4, 5)` of `f`. -/
theorem det_gram : Matrix.det !![(5 : ℚ), 4; 4, 5] = 9 ∧ Matrix.det !![(2 : ℚ), 1; 1, 2] ^ 2 = 9 := by
  refine ⟨?_, ?_⟩
  · norm_num [Matrix.det_fin_two_of]
    -- certificate: la-la-gram-f.sage
  · norm_num [Matrix.det_fin_two_of]
    -- certificate: la-la-gram-f.sage

/-- The congruence `Pᵀ G P = diag(5, 9/5)` with `P = (1, -4/5; 0, 1)`. -/
theorem gram_congruence :
    (!![(1 : ℚ), -4 / 5; 0, 1])ᵀ * !![(5 : ℚ), 4; 4, 5] * !![(1 : ℚ), -4 / 5; 0, 1] = !![(5 : ℚ), 0; 0, 9 / 5] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-la-gram-f.sage

theorem det_gram_P : Matrix.det !![(1 : ℚ), -4 / 5; 0, 1] = 1 := by
  norm_num [Matrix.det_fin_two_of]
  -- certificate: la-matrix.sage

/-- `G` is positive definite: signature `(2, 0, 0)`. -/
theorem gram_signature : HasSignature !![(5 : ℚ), 4; 4, 5] 2 0 0 := by
  veriphy_certified
    ( refine ⟨!![1, (-4 / 5); 0, 1], ![5, (9 / 5)], ?_, ?_, ?_, ?_, ?_⟩
      ·
        norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
          Fin.succAbove_of_le_castSucc, Matrix.diagonal]
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals decide +kernel)
  -- certificate: la-signature.sage

/-- Lowering: `v_i = G_{ij} v^j = (14, 13)`; raising with `G⁻¹ = ⅑ (5, -4; -4, 5)` returns `(2, 1)`. -/
theorem gram_lower_raise :
    !![(5 : ℚ), 4; 4, 5] * !![(2 : ℚ); 1] = !![(14 : ℚ); 13] ∧
      !![(5 / 9 : ℚ), -4 / 9; -4 / 9, 5 / 9] * !![(14 : ℚ); 13] = !![(2 : ℚ); 1] ∧
      !![(5 / 9 : ℚ), -4 / 9; -4 / 9, 5 / 9] * !![(5 : ℚ), 4; 4, 5] = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-gram-f.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-gram-f.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-gram-f.sage

/-- `v_i v^i = 28 + 13 = 41 = 5² + 4²`. -/
theorem gram_invariant :
    !![(14 : ℚ), 13] * !![(2 : ℚ); 1] = !![(41 : ℚ)] ∧ !![(5 : ℚ), 4] * !![(5 : ℚ); 4] = !![(41 : ℚ)] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-gram-f.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-gram-f.sage

/-! ## exr-la-map-tensor -/

/-- `A T' = T A` for `T' = ⅓ (8, 10; -1, 1)`: `T'` is the matrix of `T` in the basis `f`. -/
theorem map_in_f :
    !![(2 : ℚ), 1; 1, 2] * !![(8 / 3 : ℚ), 10 / 3; -1 / 3, 1 / 3] = !![(1 : ℚ), 3; 0, 2] * !![(2 : ℚ), 1; 1, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-la-map-tensor.sage

/-- The traces agree (`3`) and the determinants agree (`2`). -/
theorem map_invariants :
    Matrix.trace !![(1 : ℚ), 3; 0, 2] = 3 ∧ Matrix.trace !![(8 / 3 : ℚ), 10 / 3; -1 / 3, 1 / 3] = 3 ∧
      Matrix.det !![(1 : ℚ), 3; 0, 2] = 2 ∧ Matrix.det !![(8 / 3 : ℚ), 10 / 3; -1 / 3, 1 / 3] = 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [Matrix.trace_fin_two_of]
    -- certificate: la-la-map-tensor.sage
  · norm_num [Matrix.trace_fin_two_of]
    -- certificate: la-la-map-tensor.sage
  · norm_num [Matrix.det_fin_two_of]
    -- certificate: la-la-map-tensor.sage
  · norm_num [Matrix.det_fin_two_of]
    -- certificate: la-la-map-tensor.sage

/-- `T v = (17, 8)`, `T' (2, 1) = (26/3, -1/3)`, and `A (26/3, -1/3) = (17, 8)`. -/
theorem map_on_v :
    !![(1 : ℚ), 3; 0, 2] * !![(5 : ℚ); 4] = !![(17 : ℚ); 8] ∧
      !![(8 / 3 : ℚ), 10 / 3; -1 / 3, 1 / 3] * !![(2 : ℚ); 1] = !![(26 / 3 : ℚ); -1 / 3] ∧
      !![(2 : ℚ), 1; 1, 2] * !![(26 / 3 : ℚ); -1 / 3] = !![(17 : ℚ); 8] := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-map-tensor.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-map-tensor.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-la-map-tensor.sage

/-! ## exr-la-field-strength -/

/-- `F^{μν} = η^{μα} η^{νβ} F_{αβ}` for `F₀₁ = -E`, `F₁₂ = B`. -/
theorem field_strength_raised (E B : ℝ) :
    !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![0, -E, 0, 0; E, 0, B, 0; 0, -B, 0, 0; 0, 0, 0, 0] *
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = !![0, E, 0, 0; -E, 0, B, 0; 0, -B, 0, 0; 0, 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-la-field-strength.sage

/-- `F_{μν} F^{μν} = 2 (B² - E²)`, as the trace of `F (F^{··})ᵀ`. -/
theorem field_strength_invariant (E B : ℝ) :
    Matrix.trace (!![0, -E, 0, 0; E, 0, B, 0; 0, -B, 0, 0; 0, 0, 0, 0] *
      (!![0, E, 0, 0; -E, 0, B, 0; 0, -B, 0, 0; 0, 0, 0, 0])ᵀ) = 2 * (B ^ 2 - E ^ 2) := by
  simp [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.vecMul, dotProduct]
  ring1
  -- certificate: la-la-field-strength.sage

end KaluzaKlein.LinearAlgebra.Exercises

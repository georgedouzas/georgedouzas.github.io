import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Signature"

The claims of the section "Signature" of `tutorial/part-i-02-linear-algebra.qmd`. The signature `(p, q, z)` of a
symmetric matrix is read off a congruence `Pᵀ S P = diag(d)` with `det P ≠ 0`: `p`, `q`, `z` count the positive,
negative and zero entries of `d`. (Sylvester's law of inertia, that the triple does not depend on `P`, is not in
mathlib; mathlib has the existence of such a diagonalisation, `QuadraticForm.equivalent_signType_weighted_sum_squared`.)
-/

namespace KaluzaKlein.LinearAlgebra.Signature

open Matrix Module

/-! ## def-signature -/

/-- `S` has signature `(p, q, z)`: an invertible `P` brings it by congruence to a diagonal matrix with `p` positive,
`q` negative and `z` zero entries (Veriphy's `Veriphy.Linear.HasSignature`, which counts the entries of the diagonal
with `Finset.filter`). -/
abbrev HasSignature {K : Type*} [Field K] [LinearOrder K] {n : ℕ} (S : Matrix (Fin n) (Fin n) K) (p q z : ℕ) : Prop :=
  Veriphy.Linear.HasSignature S p q z

/-- The Minkowski form has signature `(1, 3, 0)` (Lorentzian): it is already diagonal. -/
theorem eta_signature : HasSignature KaluzaKlein.Linear.eta 1 3 0 := by
  unfold KaluzaKlein.Linear.eta
  veriphy_certified
    ( refine ⟨!![1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1], ![1, -1, -1, -1], ?_, ?_, ?_, ?_, ?_⟩
      ·
        norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
          Fin.succAbove_of_le_castSucc, Matrix.diagonal]
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals (simp only [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]; norm_num))
  -- certificate: la-signature.sage

/-! ## exm-signature-euclid -/

/-- The dot product on `ℝ³` has signature `(3, 0, 0)`. -/
theorem euclid_signature : HasSignature (1 : Matrix (Fin 3) (Fin 3) ℝ) 3 0 0 := by
  veriphy_certified
    ( refine ⟨!![1, 0, 0; 0, 1, 0; 0, 0, 1], ![1, 1, 1], ?_, ?_, ?_, ?_, ?_⟩
      ·
        norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
          Fin.succAbove_of_le_castSucc, Matrix.diagonal]
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals (simp only [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]; norm_num))
  -- certificate: la-signature.sage

/-- It is an inner product: `v · v > 0` for `v ≠ 0`. -/
theorem dot_self_pos {n : ℕ} (v : Fin n → ℝ) (hv : v ≠ 0) : 0 < v ⬝ᵥ v := by
  rcases (dotProduct_self_star_nonneg v).lt_or_eq with h | h
  · simpa using h
  · exact absurd (dotProduct_self_star_eq_zero.mp (by simpa using h.symm)) hv

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Signature.dot_self_pos (1) (![1]) ((by intro h; have := congrFun h 0; norm_num at this)) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Signature.dot_self_pos (2) (![1, -2]) ((by intro h; have := congrFun h 0; norm_num at this)) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Signature.dot_self_pos (3) (![1 / 2, 0, 3]) ((by intro h; have := congrFun h 0; norm_num at this)) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Signature.dot_self_pos
-- test: la-dot-self-pos-test.sage

/-! ## exm-signature-minkowski -/

/-- Timelike, spacelike and null examples: `η(w, w) = 1`, `η(e₁, e₁) = -1`, `η(v, v) = 0` for `v = (1, 1, 0, 0)`. -/
theorem minkowski_types :
    !![(1 : ℝ), 0, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 0; 0; 0] = !![(1 : ℝ)] ∧
    !![(0 : ℝ), 1, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(0 : ℝ); 1; 0; 0] = !![(-1 : ℝ)] ∧
    !![(1 : ℝ), 1, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 1; 0; 0] = !![(0 : ℝ)] := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-signature.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-signature.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-signature.sage

/-! ## exm-signature-degenerate -/

/-- `det P = 1` for `P = (1, -2; 0, 1)`. -/
theorem det_P : Matrix.det !![(1 : ℚ), -2; 0, 1] = 1 := by
  norm_num [Matrix.det_fin_two_of]
  -- certificate: la-matrix.sage

/-- `S = (1, 2; 2, 4)` has signature `(1, 0, 1)`, through `Pᵀ S P = diag(1, 0)` (`KaluzaKlein.Linear.S_sylvester`). -/
theorem S_signature : HasSignature KaluzaKlein.Linear.S 1 0 1 := by
  unfold KaluzaKlein.Linear.S
  veriphy_certified
    ( refine ⟨!![1, -2; 0, 1], ![1, 0], ?_, ?_, ?_, ?_, ?_⟩
      ·
        simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
            Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
          norm_num
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals decide +kernel)
  -- certificate: la-signature.sage

/-- The form vanishes identically along the null direction `(-2, 1)`. -/
theorem S_null_direction (t : ℝ) : !![-2 * t, t] * !![(1 : ℝ), 2; 2, 4] * !![-2 * t; t] = !![(0 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-! ## exr-signature-complete-square -/

/-- `Q' = x² + 4xy + 3y²`: `Pᵀ (1, 2; 2, 3) P = diag(1, -1)`. -/
theorem Sprime_congruence :
    (!![(1 : ℚ), -2; 0, 1])ᵀ * !![(1 : ℚ), 2; 2, 3] * !![(1 : ℚ), -2; 0, 1] = !![(1 : ℚ), 0; 0, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-signature-complete-square.sage

/-- `Q'` has signature `(1, 1, 0)`. -/
theorem Sprime_signature : HasSignature !![(1 : ℚ), 2; 2, 3] 1 1 0 := by
  veriphy_certified
    ( refine ⟨!![1, -2; 0, 1], ![1, -1], ?_, ?_, ?_, ?_, ?_⟩
      ·
        simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
            Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
          norm_num
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals decide +kernel)
  -- certificate: la-signature.sage

/-- The two quadratic forms as completed squares. -/
theorem complete_squares (x y : ℝ) :
    !![x, y] * !![(1 : ℝ), 2; 2, 4] * !![x; y] = !![(x + 2 * y) ^ 2] ∧
      !![x, y] * !![(1 : ℝ), 2; 2, 3] * !![x; y] = !![(x + 2 * y) ^ 2 - y ^ 2] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
    -- certificate: la-matrix.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
    -- certificate: la-matrix.sage

/-! ## exr-signature-3x3 -/

/-- Eigenvectors of `B = (2, 1, 0; 1, 2, 1; 0, 1, 2)` for the eigenvalues `2`, `2 + √2`, `2 - √2`. -/
theorem B_eigen_two :
    !![(2 : ℝ), 1, 0; 1, 2, 1; 0, 1, 2] * !![(1 : ℝ); 0; -1] = (2 : ℝ) • !![(1 : ℝ); 0; -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

theorem B_eigen_plus :
    !![(2 : ℝ), 1, 0; 1, 2, 1; 0, 1, 2] * !![(1 : ℝ); Real.sqrt 2; 1] =
      (2 + Real.sqrt 2) • !![(1 : ℝ); Real.sqrt 2; 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
  all_goals
    first
    | ring1
    | linear_combination (-1) * Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  -- certificate: la-sqrt-eigen.sage

theorem B_eigen_minus :
    !![(2 : ℝ), 1, 0; 1, 2, 1; 0, 1, 2] * !![(1 : ℝ); -Real.sqrt 2; 1] =
      (2 - Real.sqrt 2) • !![(1 : ℝ); -Real.sqrt 2; 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
  all_goals
    first
    | ring1
    | linear_combination (-1) * Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  -- certificate: la-sqrt-eigen.sage

/-- All three eigenvalues are positive. -/
theorem B_eigenvalues_pos : 0 < (2 : ℝ) ∧ 0 < 2 + Real.sqrt 2 ∧ 0 < 2 - Real.sqrt 2 := by
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  have hn := Real.sqrt_nonneg 2
  refine ⟨?_, ?_, ?_⟩
  · by_contra h
    push_neg at h
    veriphy_certified
      ( have hκ : (1 : ℝ) ≤ 0 := by linear_combination (1 / 2) * h
        norm_num at hκ)
    -- certificate: la-infeasible.sage
  · by_contra h
    push_neg at h
    veriphy_certified
      ( have hκ : (1 : ℝ) ≤ 0 := by
          linear_combination (2) * h + (1 / 2) * mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr h) + (1 / 2) * hs
        norm_num at hκ)
    -- certificate: la-infeasible.sage
  · by_contra h
    push_neg at h
    veriphy_certified
      ( have hκ : (1 : ℝ) ≤ 0 := by
          linear_combination (2) * h + (1 / 2) * mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr h) + (1 / 2) * hs
        norm_num at hκ)
    -- certificate: la-infeasible.sage

/-- The rational congruence `Pᵀ B P = diag(2, 3/2, 4/3)`. -/
theorem B_congruence :
    (!![(1 : ℚ), -1 / 2, 1 / 3; 0, 1, -2 / 3; 0, 0, 1])ᵀ * !![(2 : ℚ), 1, 0; 1, 2, 1; 0, 1, 2] *
      !![(1 : ℚ), -1 / 2, 1 / 3; 0, 1, -2 / 3; 0, 0, 1] = !![(2 : ℚ), 0, 0; 0, 3 / 2, 0; 0, 0, 4 / 3] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-signature-3x3.sage

theorem det_P3 : Matrix.det !![(1 : ℚ), -1 / 2, 1 / 3; 0, 1, -2 / 3; 0, 0, 1] = 1 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-matrix.sage

/-- `B` is positive definite: signature `(3, 0, 0)`. -/
theorem B_signature : HasSignature !![(2 : ℚ), 1, 0; 1, 2, 1; 0, 1, 2] 3 0 0 := by
  veriphy_certified
    ( refine ⟨!![1, (-1 / 2), (1 / 3); 0, 1, (-2 / 3); 0, 0, 1], ![2, (3 / 2), (4 / 3)], ?_, ?_, ?_, ?_, ?_⟩
      ·
        simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
            Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
          norm_num
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals decide +kernel)
  -- certificate: la-signature.sage

/-- Eigenvectors of `C = (0, 1, 0; 1, 0, 0; 0, 0, -1)`: eigenvalues `1, -1, -1`. -/
theorem C_eigen :
    !![(0 : ℚ), 1, 0; 1, 0, 0; 0, 0, -1] * !![(1 : ℚ), 1, 0; 1, -1, 0; 0, 0, 1] = !![(1 : ℚ), -1, 0; 1, 1, 0; 0, 0, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-signature-3x3.sage

/-- The congruence `Qᵀ C Q = diag(2, -2, -1)`. -/
theorem C_congruence :
    (!![(1 : ℚ), 1, 0; 1, -1, 0; 0, 0, 1])ᵀ * !![(0 : ℚ), 1, 0; 1, 0, 0; 0, 0, -1] * !![(1 : ℚ), 1, 0; 1, -1, 0; 0, 0, 1] =
      !![(2 : ℚ), 0, 0; 0, -2, 0; 0, 0, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-signature-3x3.sage

theorem det_Q3 : Matrix.det !![(1 : ℚ), 1, 0; 1, -1, 0; 0, 0, 1] = -2 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-matrix.sage

/-- `C` has signature `(1, 2, 0)`. -/
theorem C_signature : HasSignature !![(0 : ℚ), 1, 0; 1, 0, 0; 0, 0, -1] 1 2 0 := by
  veriphy_certified
    ( refine ⟨!![1, (-1 / 2), 0; 1, (1 / 2), 0; 0, 0, 1], ![2, (-1 / 2), -1], ?_, ?_, ?_, ?_, ?_⟩
      ·
        simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
            Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
          norm_num
      ·
        ext i j <;> fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
          norm_num
      all_goals decide +kernel)
  -- certificate: la-signature.sage

end KaluzaKlein.LinearAlgebra.Signature

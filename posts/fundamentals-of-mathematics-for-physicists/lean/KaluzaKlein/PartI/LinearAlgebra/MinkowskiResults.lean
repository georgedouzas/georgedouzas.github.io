import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Results" of "Bilinear forms"

The claims of the proposition on the values of the Minkowski form in `tutorial/part-i-02-linear-algebra.qmd` and of its
examples and exercises: the null vector and the determinant by hand, the sign of the determinant, the classification
of vectors, the invariance under a boost and the inverse metric.
-/

namespace KaluzaKlein.LinearAlgebra.MinkowskiResults

open Matrix Module

/-! ## exm-minkowski-det-hand -/

/-- `det diag(1, -1, -1, -1) = 1 · (-1) · (-1) · (-1) = -1`. -/
theorem det_eta_by_hand :
    Matrix.det !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = 1 * (-1) * (-1) * (-1) := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-matrix.sage

/-- The determinant of a diagonal matrix is the product of its diagonal entries. -/
theorem det_diagonal_prod {n : ℕ} (d : Fin n → ℝ) : (Matrix.diagonal d).det = ∏ i, d i :=
  Matrix.det_diagonal

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod (0) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod (2) (![2, 3]) := by
  simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.prod_univ_succ, Matrix.diagonal_apply, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod (3) (![1, -1, 1 / 2]) := by
  simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.prod_univ_succ, Matrix.diagonal_apply, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod (4) (![1, -1, -1, -1]) := by
  simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.prod_univ_succ, Matrix.diagonal_apply, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]
  <;> norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod
-- test: la-det-diagonal-prod-test.sage

/-! ## exm-minkowski-sign-det -/

/-- `det < 0` does not fix `q`: `diag(1, -1, 1, 1)` (one negative direction) has the same determinant as `η`. -/
theorem det_one_negative :
    Matrix.det !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] = -1 := by
  norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove]
  -- certificate: la-matrix.sage

/-- For a diagonal matrix with entries `±1`, the determinant is `(-1)^q`, `q` the number of `-1` entries. -/
theorem det_signs {n : ℕ} (s : Fin n → Bool) :
    (Matrix.diagonal fun i => if s i then (-1 : ℝ) else 1).det = (-1) ^ (Finset.univ.filter fun i => s i = true).card := by
  rw [Matrix.det_diagonal, Finset.prod_ite, Finset.prod_const, Finset.prod_const_one, mul_one]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs (0) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs (2) (![true, false]) := by
  rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![true, false]) i = true).card = 1 from by decide]
  norm_num [Fin.prod_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs (4) (![false, true, true, true]) := by
  rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![false, true, true, true]) i = true).card = 3 from by decide]
  norm_num [Fin.prod_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs (3) (![true, true, false]) := by
  rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![true, true, false]) i = true).card = 2 from by decide]
  norm_num [Fin.prod_univ_succ]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs
-- test: la-det-signs-test.sage

/-! ## exr-minkowski-other-vectors -/

theorem minkowski_classify :
    !![(2 : ℝ), 1, 0, 3] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(2 : ℝ); 1; 0; 3] = !![(-6 : ℝ)] ∧
    !![(1 : ℝ), 0, 1, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 0; 1; 0] = !![(0 : ℝ)] ∧
    !![(0 : ℝ), 1, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(0 : ℝ); 1; 0; 0] = !![(-1 : ℝ)] := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-other-vectors.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-other-vectors.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-minkowski-other-vectors.sage

/-! ## exr-lorentz-boost -/

/-- `Λᵀ η Λ = η` for the boost with rapidity `u` in the `(t, x)` plane. -/
theorem boost_preserves_eta (u : ℝ) :
    (!![Real.cosh u, Real.sinh u, 0, 0; Real.sinh u, Real.cosh u, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1])ᵀ *
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *
      !![Real.cosh u, Real.sinh u, 0, 0; Real.sinh u, Real.cosh u, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] =
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
  all_goals
    first
    | ring1
    | linear_combination (-1) * Real.cosh_sq_sub_sinh_sq u
    | linear_combination (1) * Real.cosh_sq_sub_sinh_sq u
  -- certificate: la-lorentz-boost.sage

/-- A matrix with `Λᵀ η Λ = η` preserves `η(v, v)`, hence the type of every vector. -/
theorem preserves_norm {n : ℕ} (L η : Matrix (Fin n) (Fin n) ℝ) (h : Lᵀ * η * L = η) (v : Fin n → ℝ) :
    (L *ᵥ v) ⬝ᵥ (η *ᵥ (L *ᵥ v)) = v ⬝ᵥ (η *ᵥ v) := by
  rw [Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, ← Matrix.vecMul_transpose, Matrix.vecMul_vecMul,
    Matrix.dotProduct_mulVec, ← Matrix.mul_assoc, h]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.preserves_norm (0) (0) (0) ((by simp)) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.preserves_norm (2) (!![5 / 4, 3 / 4; 3 / 4, 5 / 4]) (!![1, 0; 0, -1]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])) (![1, 2]) := by
  norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MinkowskiResults.preserves_norm (3) (!![5 / 4, 3 / 4, 0; 3 / 4, 5 / 4, 0; 0, 0, 1]) (!![1, 0, 0; 0, -1, 0; 0, 0, -1]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])) (![2, 1, -1]) := by
  norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MinkowskiResults.preserves_norm
-- test: la-preserves-norm-test.sage

/-! ## exr-minkowski-inverse -/

/-- `η η = 1`: `η` is its own inverse, `η^{μν} η_{νρ} = δ^μ_ρ`. -/
theorem eta_mul_eta :
    !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-minkowski-inverse.sage

theorem eta_inv : KaluzaKlein.Linear.eta⁻¹ = KaluzaKlein.Linear.eta := by
  apply Matrix.inv_eq_left_inv
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, KaluzaKlein.Linear.eta,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

end KaluzaKlein.LinearAlgebra.MinkowskiResults

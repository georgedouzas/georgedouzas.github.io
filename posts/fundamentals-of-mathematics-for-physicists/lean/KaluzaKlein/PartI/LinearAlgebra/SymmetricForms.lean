import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Symmetric forms"

The claims of the section "Symmetric forms" of `tutorial/part-i-02-linear-algebra.qmd`: symmetry `B_{ij} = B_{ji}`,
the polarisation identity, the quadratic form `x² + 4xy + 4y²` of `S = (1, 2; 2, 4)`, antisymmetric forms, the
symmetric matrix of a quadratic form in three variables, and the symmetric-antisymmetric decomposition.
-/

namespace KaluzaKlein.LinearAlgebra.SymmetricForms

open Matrix Module

/-! ## def-symmetric-form, exr-polarisation -/

/-- A form is symmetric exactly when its matrix is: `B(v, w) = B(w, v)` for all `v, w` iff `Bᵀ = B`. -/
theorem symmetric_iff {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    (∀ v w, Matrix.toBilin' B v w = Matrix.toBilin' B w v) ↔ Bᵀ = B := by
  constructor
  · intro h
    ext i j
    have := h (Pi.single j 1) (Pi.single i 1)
    simpa [Matrix.toBilin'_single] using this
  · intro h v w
    simp only [Matrix.toBilin'_apply']
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, h, dotProduct_comm]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.symmetric_iff (2) (0) := by
  constructor
  · intro _
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num
  · intro _ v w
    simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.symmetric_iff (2) (!![1, 2; 2, 4]) := by
  constructor
  · intro _
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num
  · intro _ v w
    simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.symmetric_iff (3) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) := by
  constructor
  · intro _
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num
  · intro _ v w
    simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.SymmetricForms.symmetric_iff
-- test: la-symmetric-iff-test.sage

/-- The algebraic core of polarisation: from the expansion `s = B(v,v) + B(v,w) + B(w,v) + B(w,w)` and symmetry. -/
theorem polarisation_core (bvv bvw bwv bww s : ℝ) (hs : s = bvv + bvw + bwv + bww) (h : bvw = bwv) :
    2 * bvw = s - bvv - bww := by
  linear_combination (-1) * hs + h
  -- certificate: ideal_membership.sage

/-- Polarisation: `2 B(v, w) = B(v + w, v + w) - B(v, v) - B(w, w)` for a symmetric form. -/
theorem polarisation {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (hB : Bᵀ = B) (v w : Fin n → ℝ) :
    2 * Matrix.toBilin' B v w =
      Matrix.toBilin' B (v + w) (v + w) - Matrix.toBilin' B v v - Matrix.toBilin' B w w := by
  apply polarisation_core _ _ (Matrix.toBilin' B w v) _ _ (by simp [map_add, LinearMap.add_apply]; ring)
  exact ((symmetric_iff B).mpr hB) v w

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.polarisation (0) (0) ((by simp)) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.polarisation (2) (!![1, 2; 2, 4]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) (![1, 3]) (![2, -1]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.polarisation (3) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) (![1, 2, -1]) (![0, 1, 1 / 2]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.SymmetricForms.polarisation
-- test: la-polarisation-test.sage

/-- For the antisymmetric `J = (0, 1; -1, 0)`, `J(e₁, e₂) = 1` but the polarisation formula gives `0`. -/
theorem polarisation_fails_antisym :
    !![(1 : ℚ), 0] * !![(0 : ℚ), 1; -1, 0] * !![(0 : ℚ); 1] = !![(1 : ℚ)] ∧
      !![(1 : ℚ), 1] * !![(0 : ℚ), 1; -1, 0] * !![(1 : ℚ); 1] - !![(1 : ℚ), 0] * !![(0 : ℚ), 1; -1, 0] * !![(1 : ℚ); 0]
        - !![(0 : ℚ), 1] * !![(0 : ℚ), 1; -1, 0] * !![(0 : ℚ); 1] = !![(0 : ℚ)] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-polarisation.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-polarisation.sage

/-! ## exm-symmetric-dot-minkowski -/

/-- The dot product is symmetric (its matrix is the identity). -/
theorem identity_symm (n : ℕ) : (1 : Matrix (Fin n) (Fin n) ℝ)ᵀ = 1 := Matrix.transpose_one

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.identity_symm (0) := by
  ext i j
  simp [Matrix.one_apply, eq_comm]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.identity_symm (4) := by
  ext i j
  simp [Matrix.one_apply, eq_comm]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.identity_symm (1) := by
  ext i j
  simp [Matrix.one_apply, eq_comm]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.SymmetricForms.identity_symm
-- test: la-identity-symm-test.sage

/-! ## exm-symmetric-quadratic -/

/-- The quadratic form of `S = (1, 2; 2, 4)` is `x² + 4xy + 4y²`. -/
theorem quadratic_form_S (x y : ℝ) : !![x, y] * !![(1 : ℝ), 2; 2, 4] * !![x; y] = !![x ^ 2 + 4 * x * y + 4 * y ^ 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-! ## exm-symmetric-not -/

/-- The quadratic form of the determinant form vanishes identically. -/
theorem det_form_quadratic_zero (x y : ℝ) : !![x, y] * !![(0 : ℝ), 1; -1, 0] * !![x; y] = !![(0 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-! ## exr-symmetric-quadratic -/

/-- The symmetric matrix of `Q = 2x² - 3xy + y² + 4yz`. -/
theorem quadratic_three (x y z : ℝ) :
    !![x, y, z] * !![(2 : ℝ), -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0] * !![x; y; z] =
      !![2 * x ^ 2 - 3 * x * y + y ^ 2 + 4 * y * z] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-- `Q(1, 2, -1) = -8`. -/
theorem quadratic_three_value :
    !![(1 : ℝ), 2, -1] * !![(2 : ℝ), -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0] * !![(1 : ℝ); 2; -1] = !![(-8 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-symmetric-decompose -/

/-- `(1, 2; -1, 0) = (1, ½; ½, 0) + (0, 3/2; -3/2, 0)`. -/
theorem decompose_B :
    !![(1 : ℚ), 2; -1, 0] = !![(1 : ℚ), 1 / 2; 1 / 2, 0] + !![(0 : ℚ), 3 / 2; -3 / 2, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- The parts are symmetric and antisymmetric. -/
theorem decompose_parts :
    (!![(1 : ℚ), 1 / 2; 1 / 2, 0])ᵀ = !![(1 : ℚ), 1 / 2; 1 / 2, 0] ∧
      (!![(0 : ℚ), 3 / 2; -3 / 2, 0])ᵀ = -!![(0 : ℚ), 3 / 2; -3 / 2, 0] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-matrix.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-matrix.sage

/-- Only the symmetric part contributes to the quadratic form. -/
theorem antisym_part_quadratic (x y : ℝ) : !![x, y] * !![(0 : ℝ), 3 / 2; -3 / 2, 0] * !![x; y] = !![(0 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-- The decomposition is unique: `S + A = S' + A'` with `S, S'` symmetric and `A, A'` antisymmetric forces `S = S'`. -/
theorem decomposition_unique {n : ℕ} (S A S' A' : Matrix (Fin n) (Fin n) ℝ) (hS : Sᵀ = S) (hA : Aᵀ = -A)
    (hS' : S'ᵀ = S') (hA' : A'ᵀ = -A') (h : S + A = S' + A') : S = S' := by
  have ht := congrArg Matrix.transpose h
  rw [Matrix.transpose_add, Matrix.transpose_add, hS, hA, hS', hA'] at ht
  have h2 : (2 : ℝ) • S = (2 : ℝ) • S' := by
    rw [two_smul, two_smul]
    calc S + S = (S + A) + (S + -A) := by abel
      _ = (S' + A') + (S' + -A') := by rw [h, ht]
      _ = S' + S' := by abel
  exact smul_right_injective _ (by norm_num : (2 : ℝ) ≠ 0) h2

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.decomposition_unique (0) (0) (0) (0) (0) ((by simp)) ((by simp)) ((by simp)) ((by simp)) (rfl) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.decomposition_unique (2) (!![1, 1 / 2; 1 / 2, 0]) (!![0, 3 / 2; -3 / 2, 0]) (!![1, 1 / 2; 1 / 2, 0]) (!![0, 3 / 2; -3 / 2, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) (rfl) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.SymmetricForms.decomposition_unique (3) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) (!![0, -5, 2; 5, 0, -7; -2, 7, 0]) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) (!![0, -5, 2; 5, 0, -7; -2, 7, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) (rfl) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.SymmetricForms.decomposition_unique
-- test: la-decomposition-unique-test.sage

end KaluzaKlein.LinearAlgebra.SymmetricForms

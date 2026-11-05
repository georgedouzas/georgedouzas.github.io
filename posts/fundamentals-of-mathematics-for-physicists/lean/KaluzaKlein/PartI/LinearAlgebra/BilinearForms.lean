import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Bilinear forms"

The claims of the section "Bilinear forms" of `tutorial/part-i-02-linear-algebra.qmd`: components `B_{ij} = B(e_i, e_j)`
and `B(v, w) = B_{ij} v^i w^j`, the dot product, the Minkowski form, products of functionals, the determinant form,
and the transformation `B' = Aᵀ B A`.
-/

namespace KaluzaKlein.LinearAlgebra.BilinearForms

open Matrix Module

/-! ## def-bilinear-form -/

/-- A bilinear form with component matrix `B`: `B(v, w) = B_{ij} v^i w^j`. -/
theorem bilin_components {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (v w : Fin n → ℝ) :
    Matrix.toBilin' B v w = ∑ i, ∑ j, v i * B i j * w j :=
  Matrix.toBilin'_apply B v w

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_components (0) (0) (![]) (![]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_components (2) (!![1, 2; -1, 0]) (![1, 3]) (![2, -1]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_components (3) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) (![1, 2, -1]) (![0, 1, 1]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.BilinearForms.bilin_components
-- test: la-bilin-components-test.sage

/-- The components are the values on basis vectors: `B(e_i, e_j) = B_{ij}`. -/
theorem bilin_on_basis {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    Matrix.toBilin' B (Pi.single i 1) (Pi.single j 1) = B i j :=
  Matrix.toBilin'_single B i j

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_on_basis (1) (!![5]) (0) (0) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_on_basis (2) (!![1, 2; -1, 0]) (1) (0) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.bilin_on_basis (3) (!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]) (1) (2) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.BilinearForms.bilin_on_basis
-- test: la-bilin-on-basis-test.sage

/-! ## exm-bilinear-dot -/

/-- The dot product is the bilinear form with the identity matrix. -/
theorem dot_product_form {n : ℕ} (v w : Fin n → ℝ) : Matrix.toBilin' (1 : Matrix (Fin n) (Fin n) ℝ) v w = v ⬝ᵥ w := by
  simp [Matrix.toBilin'_apply', dotProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.dot_product_form (0) (![]) (![]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.dot_product_form (2) (![1, 3]) (![2, -1]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.dot_product_form (3) (![1, 2, -1]) (![1 / 2, 0, 4]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.BilinearForms.dot_product_form
-- test: la-dot-product-form-test.sage

/-! ## exm-bilinear-minkowski -/

/-- `η(w, w) = 1` for `w = (1, 0, 0, 0)`. -/
theorem eta_timelike_matrix :
    !![(1 : ℝ), 0, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 0; 0; 0] =
      !![(1 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- `η(v, v) = 0` for `v = (1, 1, 0, 0)`. -/
theorem eta_null_matrix :
    !![(1 : ℝ), 1, 0, 0] * !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] * !![(1 : ℝ); 1; 0; 0] =
      !![(0 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exm-bilinear-general -/

/-- `B(v, w) = α(v) β(w)` has components `α_i β_j` (in two dimensions). -/
theorem product_of_functionals (v1 v2 w1 w2 a1 a2 b1 b2 : ℝ) :
    !![v1, v2] * !![a1 * b1, a1 * b2; a2 * b1, a2 * b2] * !![w1; w2] = !![(v1 * a1 + v2 * a2) * (b1 * w1 + b2 * w2)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-- The determinant form `v¹ w² - v² w¹` changes sign under exchange of its arguments. -/
theorem det_form_antisymm (v1 v2 w1 w2 : ℝ) :
    !![v1, v2] * !![(0 : ℝ), 1; -1, 0] * !![w1; w2] = -(!![w1, w2] * !![(0 : ℝ), 1; -1, 0] * !![v1; v2]) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-! ## exr-bilinear-check -/

/-- `v¹ w²` and `v¹ w¹ - 3 v² w¹` are bilinear: they are the forms of the matrices `(0, 1; 0, 0)` and `(1, 0; -3, 0)`. -/
theorem bilinear_examples (v1 v2 w1 w2 : ℝ) :
    !![v1, v2] * !![(0 : ℝ), 1; 0, 0] * !![w1; w2] = !![v1 * w2] ∧
      !![v1, v2] * !![(1 : ℝ), 0; -3, 0] * !![w1; w2] = !![v1 * w1 - 3 * v2 * w1] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
    -- certificate: la-matrix.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
    -- certificate: la-matrix.sage

/-- `v · w + 1` is not bilinear: a bilinear form vanishes at `v = 0`. -/
theorem dot_plus_one_not_bilinear : ¬ ∃ B : LinearMap.BilinForm ℝ (Fin 2 → ℝ), ∀ v w, B v w = v ⬝ᵥ w + 1 := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := hf 0 0
      simp at h)
  -- certificate: la-nonlinear.sage

/-- `|v| |w|` is not bilinear: `B(-v, w) = |v| |w|`, not `-|v| |w|`. -/
theorem norm_product_not_bilinear :
    ¬ ∃ B : LinearMap.BilinForm ℝ (EuclideanSpace ℝ (Fin 2)), ∀ v w, B v w = ‖v‖ * ‖w‖ := by
  veriphy_certified
    ( rintro ⟨B, hB⟩
      set e : EuclideanSpace ℝ (Fin 2) := EuclideanSpace.single 0 1
      have h1 := hB (-e) e
      rw [map_neg, LinearMap.neg_apply, hB, norm_neg] at h1
      have he : ‖e‖ = 1 := by simp [e]
      rw [he] at h1
      norm_num at h1)
  -- certificate: la-nonlinear-norm.sage

/-! ## exr-bilinear-matrix -/

/-- `B(v, w) = -6` for `B = (1, 2; -1, 0)`, `v = (1, 3)`, `w = (2, -1)`. -/
theorem bilinear_matrix_vw : !![(1 : ℚ), 3] * !![(1 : ℚ), 2; -1, 0] * !![(2 : ℚ); -1] = !![(-6 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-bilinear-matrix.sage

/-- `B(w, v) = 15`. -/
theorem bilinear_matrix_wv : !![(2 : ℚ), -1] * !![(1 : ℚ), 2; -1, 0] * !![(1 : ℚ); 3] = !![(15 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-bilinear-matrix.sage

/-- The components: `B(v, w) = v¹ w¹ + 2 v¹ w² - v² w¹` is the form of `(1, 2; -1, 0)`. -/
theorem bilinear_matrix_formula (v1 v2 w1 w2 : ℝ) :
    !![v1, v2] * !![(1 : ℝ), 2; -1, 0] * !![w1; w2] = !![v1 * w1 + 2 * v1 * w2 - v2 * w1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-- `B` is not symmetric. -/
theorem bilinear_matrix_not_symm : (!![(1 : ℚ), 2; -1, 0])ᵀ ≠ !![(1 : ℚ), 2; -1, 0] := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  change (-1 : ℚ) = 2 at h01
  clear h
  veriphy_certified
    ( have h1 : (1 : ℚ) = 0 := by linear_combination (-1 / 3) * h01
      exact one_ne_zero h1)
  -- certificate: la-infeasible.sage

/-! ## exr-bilinear-change-basis -/

/-- Under `e'_j = A^i_j e_i`, `B(A x, A y) = xᵀ (Aᵀ B A) y`. -/
theorem change_basis_form {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) (x y : Fin n → ℝ) :
    (A *ᵥ x) ⬝ᵥ (B *ᵥ (A *ᵥ y)) = x ⬝ᵥ ((Aᵀ * B * A) *ᵥ y) := by
  rw [Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, ← Matrix.vecMul_transpose, Matrix.vecMul_vecMul,
    Matrix.dotProduct_mulVec, Matrix.mul_assoc]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form (0) (0) (0) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form (2) (!![2, 1; 1, 2]) (!![1, 0; 0, -1]) (![1, 2]) (![3, -1]) := by
  norm_num [Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form (3) (!![1, 0, 1; 1, 1, 0; 0, 1, 1]) (!![1, 2, 0; 2, 0, 1; 0, 1, -1]) (![1, 0, -1]) (![2, 1, 1 / 2]) := by
  norm_num [Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form
-- test: la-change-basis-form-test.sage

/-- The Gram matrix of `f`: `Aᵀ A = (5, 4; 4, 5)`. -/
theorem gram_f : (!![(2 : ℚ), 1; 1, 2])ᵀ * !![(2 : ℚ), 1; 1, 2] = !![(5 : ℚ), 4; 4, 5] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-bilinear-change-basis.sage

/-- `|v|² = 41` from the standard coordinates `(5, 4)` and from the `f`-coordinates `(2, 1)` with the Gram matrix. -/
theorem norm_sq_two_ways :
    !![(5 : ℚ), 4] * !![(5 : ℚ); 4] = !![(41 : ℚ)] ∧
      !![(2 : ℚ), 1] * !![(5 : ℚ), 4; 4, 5] * !![(2 : ℚ); 1] = !![(41 : ℚ)] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-bilinear-change-basis.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-bilinear-change-basis.sage

end KaluzaKlein.LinearAlgebra.BilinearForms

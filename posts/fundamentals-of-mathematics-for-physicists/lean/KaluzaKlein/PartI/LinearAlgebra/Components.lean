import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.BilinearForms
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Components"

The claims of the section "Components" of `tutorial/part-i-02-linear-algebra.qmd`: transformation of components under
`e'_i = A^j_i e_j` (upper indices with `A⁻¹`, lower indices with `A`), the components of `u = (1, 2, 3)` and
`a = (1, 0, -1)` in `f = (e₁ + e₂, e₂, e₃)`, and the determinants of type `(0, 2)` and `(1, 1)` tensors.
-/

namespace KaluzaKlein.LinearAlgebra.Components

open Matrix Module

/-! ## def-tensor-components, exm-components-metric, exr-components-bilinear-transform -/

/-- Upper indices transform with `A⁻¹`, lower indices with `A`: the contraction `a_i u^i` is invariant. -/
theorem upper_lower_invariant {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (h : IsUnit A.det) (u a : Fin n → ℝ) :
    (a ᵥ* A) ⬝ᵥ (A⁻¹ *ᵥ u) = a ⬝ᵥ u := by
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_vecMul, Matrix.mul_nonsing_inv A h, Matrix.vecMul_one]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.upper_lower_invariant (0) (0) ((by simp)) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.upper_lower_invariant (2) (!![1, 0; 1, 1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) (![1, 2]) (![3, -1]) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0; -1, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.upper_lower_invariant (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) (![1, 2, 3]) (![1, 0, -1]) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Components.upper_lower_invariant
-- test: la-upper-lower-invariant-test.sage

/-- Two lower indices: `B'_{ij} = A^k_i A^l_j B_{kl}`, i.e. `B' = Aᵀ B A`, evaluated on new components. -/
theorem two_lower {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) (x y : Fin n → ℝ) :
    x ⬝ᵥ ((Aᵀ * B * A) *ᵥ y) = (A *ᵥ x) ⬝ᵥ (B *ᵥ (A *ᵥ y)) :=
  (KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form A B x y).symm

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.two_lower (0) (0) (0) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.two_lower (2) (!![1, 0; 1, 1]) (!![1, 2; 2, 4]) (![1, 2]) (![3, -1]) := by
  norm_num [Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.two_lower (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) (!![1, 0, 0; 0, -1, 0; 0, 0, 2]) (![1, 0, -1]) (![2, 1, 1 / 2]) := by
  norm_num [Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Components.two_lower
-- test: la-two-lower-test.sage

/-- `det B' = (det A)² det B` for a type `(0, 2)` tensor. -/
theorem det_two_lower {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) : (Aᵀ * B * A).det = A.det ^ 2 * B.det := by
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_two_lower (0) (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_two_lower (2) (!![2, 1; 1, 2]) (!![1, 0; 0, -1]) := by
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_two_lower (3) (!![1, 0, 1; 1, 1, 0; 0, 1, 1]) (!![1, 2, 0; 2, 0, 1; 0, 1, -1]) := by
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Components.det_two_lower
-- test: la-det-two-lower-test.sage

/-- `det T' = det T` for a type `(1, 1)` tensor, `T' = A⁻¹ T A`. -/
theorem det_mixed {n : ℕ} (A T : Matrix (Fin n) (Fin n) ℝ) (h : IsUnit A.det) : (A⁻¹ * T * A).det = T.det := by
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_nonsing_inv, mul_comm, ← mul_assoc,
    Ring.mul_inverse_cancel _ h, one_mul]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_mixed (0) (0) (0) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_mixed (2) (!![1, 0; 1, 1]) (!![1, 3; 0, 2]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0; -1, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.det_mixed (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) (!![1, 2, 0; 0, 3, 1; 0, 0, 5]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]
  <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Components.det_mixed
-- test: la-det-mixed-test.sage

/-! ## exm-components-outer, exr-transform-components -/

/-- `u = (1, 2, 3)` has components `(1, 1, 3)` in the basis `f`: `A (1, 1, 3)ᵀ = u`. -/
theorem u_components_f :
    !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] * !![(1 : ℚ); 1; 3] = !![(1 : ℚ); 2; 3] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-covector-transform.sage

/-- The primed components solve `u'¹ = 1`, `u'¹ + u'² = 2`, `u'³ = 3`. -/
theorem u_components_unique (a b c : ℚ) (h1 : a = 1) (h2 : a + b = 2) (h3 : c = 3) : a = 1 ∧ b = 1 ∧ c = 3 := by
  refine ⟨?_, ?_, ?_⟩
  · linear_combination h1
    -- certificate: ideal_membership.sage
  · linear_combination (-1) * h1 + h2
    -- certificate: ideal_membership.sage
  · linear_combination h3
    -- certificate: ideal_membership.sage

/-! ## exm-components-basis-independent -/

/-- The trace of a type `(1, 1)` tensor is basis independent. -/
theorem trace_mixed {n : ℕ} (A T : Matrix (Fin n) (Fin n) ℝ) (h : IsUnit A.det) : (A⁻¹ * T * A).trace = T.trace := by
  rw [Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc, Matrix.mul_nonsing_inv A h, Matrix.mul_one]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.trace_mixed (0) (0) (0) ((by simp)) := by
  simp [Matrix.trace]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.trace_mixed (2) (!![1, 0; 1, 1]) (!![1, 3; 0, 2]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0; -1, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Components.trace_mixed (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) (!![1, 2, 0; 0, 3, 1; 0, 0, 5]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]
  <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Components.trace_mixed
-- test: la-trace-mixed-test.sage

/-! ## exr-covector-transform -/

/-- `a' = a A = (1, 0, -1)`: unchanged, because `a(e₂) = 0`. -/
theorem a_components_f :
    !![(1 : ℚ), 0, -1] * !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] = !![(1 : ℚ), 0, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-covector-transform.sage

/-- `a'_i u'^i = 1 + 0 - 3 = -2 = a_i u^i`. -/
theorem pairing_both_bases :
    !![(1 : ℚ), 0, -1] * !![(1 : ℚ); 1; 3] = !![(-2 : ℚ)] ∧ !![(1 : ℚ), 0, -1] * !![(1 : ℚ); 2; 3] = !![(-2 : ℚ)] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-covector-transform.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-covector-transform.sage

end KaluzaKlein.LinearAlgebra.Components

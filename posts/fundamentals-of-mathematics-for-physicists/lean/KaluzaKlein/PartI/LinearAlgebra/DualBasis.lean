import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "The dual basis"

The claims of the section "The dual basis" of `tutorial/part-i-02-linear-algebra.qmd`: the dual basis as the rows of
the inverse of the basis matrix, for the standard basis, for `f = ((2, 1), (1, 2))` and for `b = ((1, 1, 0), (0, 1, 1),
(1, 0, 1))`; coordinate differentials; the transformation of components.
-/

namespace KaluzaKlein.LinearAlgebra.DualBasis

open Matrix Module

/-! ## def-dual-basis -/

/-- The dual basis returns coordinates: `e^i(v) = v^i`. -/
theorem dualBasis_coord {K V ι : Type*} [Field K] [AddCommGroup V] [Module K V] [Fintype ι] [DecidableEq ι]
    (e : Basis ι K V) (i : ι) (v : V) : e.dualBasis i v = e.repr v i :=
  e.dualBasis_apply i v

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dualBasis_coord (ℚ) ((Fin 1 → ℚ)) ((Fin 1)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 1)) (0) (0) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dualBasis_coord (ℚ) ((Fin 2 → ℚ)) ((Fin 2)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) (1) (![5, 4]) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dualBasis_coord (ℚ) ((Fin 3 → ℚ)) ((Fin 3)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 3)) (2) (![2, 3, 4]) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualBasis.dualBasis_coord
-- test: la-dualbasis-coord-test.sage

/-! ## exm-dual-f, exr-dual-basis -/

/-- `f¹(5, 4) = 2` and `f²(5, 4) = 1`: the rows of `F⁻¹` applied to `(5, 4)`. -/
theorem dual_f_on_v : !![(2 / 3 : ℚ), -1 / 3; -1 / 3, 2 / 3] * !![(5 : ℚ); 4] = !![(2 : ℚ); 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-basis.sage

/-- `f^i(f_j) = δ^i_j`: the rows of `F⁻¹` times the columns `f_j`. -/
theorem dual_f_delta : !![(2 / 3 : ℚ), -1 / 3; -1 / 3, 2 / 3] * !![(2 : ℚ), 1; 1, 2] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-basis.sage

/-! ## exm-dual-coordinate-differentials -/

/-- In the coordinate basis of `ℝⁿ`, `dx^i(∂_j) = δ^i_j`. -/
theorem coordinate_differentials (n : ℕ) (i j : Fin n) :
    (Pi.basisFun ℝ (Fin n)).coord i (Pi.basisFun ℝ (Fin n) j) = if i = j then 1 else 0 := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials (1) (0) (0) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials (2) (0) (1) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials (3) (2) (2) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials (3) (1) (0) := by
  simp [Pi.single_apply]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials
-- test: la-coordinate-differentials-test.sage

/-! ## exr-dual-basis-r3 -/

/-- The dual basis of `b`: `D B = 1` with rows `½(1, 1, -1)`, `½(-1, 1, 1)`, `½(1, -1, 1)`. -/
theorem dual_b_delta :
    !![(1 / 2 : ℚ), 1 / 2, -1 / 2; -1 / 2, 1 / 2, 1 / 2; 1 / 2, -1 / 2, 1 / 2] * !![(1 : ℚ), 0, 1; 1, 1, 0; 0, 1, 1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-basis-r3.sage

/-- `(b¹(w), b²(w), b³(w)) = (½, 5/2, 3/2)` for `w = (2, 3, 4)`. -/
theorem dual_b_on_w :
    !![(1 / 2 : ℚ), 1 / 2, -1 / 2; -1 / 2, 1 / 2, 1 / 2; 1 / 2, -1 / 2, 1 / 2] * !![(2 : ℚ); 3; 4] =
      !![(1 / 2 : ℚ); 5 / 2; 3 / 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-basis-r3.sage

/-! ## exr-dual-basis-transform -/

/-- New dual basis `e'^i = (A⁻¹)^i_k e^k`: `(A⁻¹)^i_k A^k_j = δ^i_j`. -/
theorem dual_transform_delta {n : ℕ} (A : Matrix (Fin n) (Fin n) ℚ) (h : IsUnit A.det) : A⁻¹ * A = 1 :=
  Matrix.nonsing_inv_mul A h

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dual_transform_delta (0) (0) ((by simp)) := by
  ext i j
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dual_transform_delta (2) (!![1, 1; 1, -1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2); (1 / 2), (-1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.dual_transform_delta (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualBasis.dual_transform_delta
-- test: la-dual-transform-delta-test.sage

/-- Vector components transform with `A⁻¹`, covector components with `A`, and the pairing is invariant:
`α'_j v'^j = α_i v^i` for `v' = A⁻¹ v`, `α' = α A`. -/
theorem pairing_invariant {n : ℕ} (A : Matrix (Fin n) (Fin n) ℚ) (h : IsUnit A.det) (v α : Fin n → ℚ) :
    (α ᵥ* A) ⬝ᵥ (A⁻¹ *ᵥ v) = α ⬝ᵥ v := by
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_vecMul, Matrix.mul_nonsing_inv A h, Matrix.vecMul_one]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.pairing_invariant (0) (0) ((by simp)) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.pairing_invariant (2) (!![1, 1; 1, -1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) (![5, 4]) (![2, -1]) := by
  rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2); (1 / 2), (-1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualBasis.pairing_invariant (3) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) (![1, 2, 3]) (![1, 0, -1]) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  norm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualBasis.pairing_invariant
-- test: la-pairing-invariant-test.sage

end KaluzaKlein.LinearAlgebra.DualBasis

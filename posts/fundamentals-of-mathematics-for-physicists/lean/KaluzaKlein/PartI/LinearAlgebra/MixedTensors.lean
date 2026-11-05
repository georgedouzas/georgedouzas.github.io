import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Tensors of mixed type"

The claims of the section "Tensors of mixed type" of `tutorial/part-i-02-linear-algebra.qmd`. A type `(r, s)` tensor on
an `n`-dimensional space is represented by its array of components, a function of `r` upper and `s` lower indices; it
has `n^(r + s)` components.
-/

namespace KaluzaKlein.LinearAlgebra.MixedTensors

open Matrix Module

/-! ## def-tensor-type -/

/-- The components of a type `(r, s)` tensor in dimension `n`: `r` upper and `s` lower indices. -/
abbrev MixedTensor (n r s : ℕ) := (Fin r → Fin n) → (Fin s → Fin n) → ℝ

/-- A type `(r, s)` tensor in dimension `n` has `n^(r + s)` components. -/
theorem finrank_mixed (n r s : ℕ) : finrank ℝ (MixedTensor n r s) = n ^ (r + s) := by
  simp only [MixedTensor]
  rw [Module.finrank_pi_fintype]
  simp [Fintype.card_fun, pow_add]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (0) (0) (0) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (4) (2) (3) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (0) (0) (1) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (0) (0) (2) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (0) (1) (0) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed (0) (1) (1) := by
  simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, Module.finrank_self]

attribute [veriphy_tested kernel_test 6 0] KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed
-- test: la-finrank-mixed-test.sage

/-! ## exm-type-metric -/

/-- The Kronecker delta, the type `(1, 1)` identity map: `δ^i_j v^j = v^i`. -/
theorem delta_identity {n : ℕ} (v : Fin n → ℝ) : (1 : Matrix (Fin n) (Fin n) ℝ) *ᵥ v = v := Matrix.one_mulVec v

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.delta_identity (0) (![]) := by
  ext i
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.delta_identity (2) (![1, 2]) := by
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.delta_identity (3) (![1 / 2, 0, -3]) := by
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Matrix.one_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MixedTensors.delta_identity
-- test: la-delta-identity-test.sage

/-- The inverse metric `g^{ij}` contracts with `g_{jk}` to `δ^i_k`. -/
theorem inverse_metric {n : ℕ} (g : Matrix (Fin n) (Fin n) ℝ) (h : g.det ≠ 0) : g⁻¹ * g = 1 :=
  Matrix.nonsing_inv_mul g (isUnit_iff_ne_zero.mpr h)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.inverse_metric (0) (0) ((by simp)) := by
  ext i j
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.inverse_metric (2) (!![1, 0; 0, -1]) ((by norm_num [Matrix.det_fin_two_of])) := by
  rw [Matrix.inv_eq_left_inv (B := !![1, 0; 0, -1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.inverse_metric (3) (!![2, 1, 0; 1, 2, 1; 0, 1, 2]) ((by norm_num [Matrix.det_fin_three])) := by
  rw [Matrix.inv_eq_left_inv (B := !![(3 / 4), (-1 / 2), (1 / 4); (-1 / 2), 1, (-1 / 2); (1 / 4), (-1 / 2), (3 / 4)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MixedTensors.inverse_metric
-- test: la-inverse-metric-test.sage

/-! ## exm-type-riemann, exr-type-count-components -/

/-- In four dimensions: the Riemann tensor (type `(1, 3)`) has `256` components, the Ricci tensor and the field
strength (type `(0, 2)`) `16`, the scalar curvature (type `(0, 0)`) one. -/
theorem curvature_counts :
    finrank ℝ (MixedTensor 4 1 3) = 256 ∧ finrank ℝ (MixedTensor 4 0 2) = 16 ∧ finrank ℝ (MixedTensor 4 2 0) = 16 ∧
      finrank ℝ (MixedTensor 4 0 0) = 1 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self, KaluzaKlein.LinearAlgebra.MixedTensors.MixedTensor])
  -- certificate: la-dim-product.sage

/-! ## exm-type-sage, exr-tensor-type-sage -/

/-- The ranks of the tensor modules of a rank-3 free module: `3^(r + s)` for the six types of the exercise. -/
theorem rank_three_counts :
    finrank ℝ (MixedTensor 3 1 0) = 3 ∧ finrank ℝ (MixedTensor 3 0 1) = 3 ∧ finrank ℝ (MixedTensor 3 0 2) = 9 ∧
      finrank ℝ (MixedTensor 3 1 1) = 9 ∧ finrank ℝ (MixedTensor 3 1 2) = 27 ∧ finrank ℝ (MixedTensor 3 2 2) = 81 ∧
      finrank ℝ (MixedTensor 3 2 0) = 9 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self, KaluzaKlein.LinearAlgebra.MixedTensors.MixedTensor])
  -- certificate: la-dim-product.sage

/-- The identity map as a type `(1, 1)` tensor has the identity matrix of components. -/
theorem identity_components (i j : Fin 3) : (1 : Matrix (Fin 3) (Fin 3) ℝ) i j = if i = j then 1 else 0 :=
  Matrix.one_apply

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (0) (0) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (0) (1) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (0) (2) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (1) (0) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (1) (1) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (1) (2) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (2) (0) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (2) (1) := by
  simp [Matrix.one_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MixedTensors.identity_components (2) (2) := by
  simp [Matrix.one_apply]

attribute [veriphy_tested kernel_test 9 0] KaluzaKlein.LinearAlgebra.MixedTensors.identity_components
-- test: la-identity-components-test.sage

end KaluzaKlein.LinearAlgebra.MixedTensors

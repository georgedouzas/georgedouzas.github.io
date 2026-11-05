import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaces
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Results" of "Multilinear maps and tensor products"

The claims of the proposition on the dimension of a tensor product in `tutorial/part-i-02-linear-algebra.qmd` and of its
examples and exercises: the basis count, `V* ⊗ V*`, `ℝ³ ⊗ ℝ`, `ℝ² ⊗ ℝ³`, `dim (V* ⊗ V) = (dim V)²`, and the
dimensions of symmetric and antisymmetric tensors.
-/

namespace KaluzaKlein.LinearAlgebra.TensorResults

open Matrix Module TensorProduct KaluzaKlein.LinearAlgebra.VectorSpaces

/-! ## exm-tensordim-basis-count -/

/-- The nine tensors `e_i ⊗ e_j` form a basis of `ℝ³ ⊗ ℝ³`. -/
noncomputable def basis33 : Basis (Fin 3 × Fin 3) ℝ ((Fin 3 → ℝ) ⊗[ℝ] (Fin 3 → ℝ)) :=
  (Pi.basisFun ℝ (Fin 3)).tensorProduct (Pi.basisFun ℝ (Fin 3))

theorem basis33_card : Fintype.card (Fin 3 × Fin 3) = 9 := by simp

/-! ## exm-tensordim-sage (tensoring with ℝ) -/

/-- `dim (ℝ³ ⊗ ℝ) = 3 · 1 = 3`. -/
theorem finrank_tensor_R : finrank ℝ ((Fin 3 → ℝ) ⊗[ℝ] ℝ) = 3 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exr-tensordim-r2-r3 -/

/-- `dim (ℝ² ⊗ ℝ³) = 6`, with the basis `e_i ⊗ f_a` indexed by `2 × 3` arrays. -/
theorem finrank_tensor_23 : finrank ℝ ((Fin 2 → ℝ) ⊗[ℝ] (Fin 3 → ℝ)) = 6 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

theorem finrank_matrix_23 : finrank ℝ (Matrix (Fin 2) (Fin 3) ℝ) = 6 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exr-tensordim-lean-reading -/

/-- `dim (V* ⊗ V) = (dim V)²` for every finite-dimensional `V`. -/
theorem finrank_dual_tensor_general {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V] :
    finrank K (Module.Dual K V ⊗[K] V) = finrank K V ^ 2 := by
  rw [Module.finrank_tensorProduct, Subspace.dual_finrank_eq, sq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general (ℚ) ((Fin 0 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct, Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general (ℚ) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct, Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct, Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general (ℚ) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct, Subspace.dual_finrank_eq]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general
-- test: la-finrank-dual-tensor-general-test.sage

/-! ## exr-symmetric-tensors-dim -/

/-- Symmetric and antisymmetric tensors in `V ⊗ V`, `dim V = 2, 3, 4`: `3 + 1 = 4`, `6 + 3 = 9`, `10 + 6 = 16`. -/
theorem sym_antisym_small :
    (finrank ℝ (symMatrices 2) = 3 ∧ finrank ℝ (antisymMatrices 2) = 1) ∧
      (finrank ℝ (symMatrices 3) = 6 ∧ finrank ℝ (antisymMatrices 3) = 3) ∧
      (finrank ℝ (symMatrices 4) = 10 ∧ finrank ℝ (antisymMatrices 4) = 6) := by
  simp only [finrank_sym, finrank_antisym]
  norm_num

end KaluzaKlein.LinearAlgebra.TensorResults

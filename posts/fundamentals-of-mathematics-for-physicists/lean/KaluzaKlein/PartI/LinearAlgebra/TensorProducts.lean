import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Tensor products"

The claims of the section "Tensor products" of `tutorial/part-i-02-linear-algebra.qmd`: the tensor product and its
basis `e_i ⊗ f_a`, `dim (V ⊗ W) = dim V · dim W`, `V* ⊗ V*` and bilinear forms, `V* ⊗ V` and linear maps, the
computations of the free-module exercise, iterated tensor products and the basis of `ℝ² ⊗ ℝ²`.
-/

namespace KaluzaKlein.LinearAlgebra.TensorProducts

open Matrix Module TensorProduct

/-! ## def-tensor-product-space -/

/-- `dim (V ⊗ W) = dim V · dim W`. -/
theorem finrank_tensor_general {K V W : Type*} [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W] [Module K W]
    [FiniteDimensional K V] [FiniteDimensional K W] : finrank K (V ⊗[K] W) = finrank K V * finrank K W :=
  Module.finrank_tensorProduct

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.finrank_tensor_general (ℚ) ((Fin 0 → ℚ)) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.finrank_tensor_general (ℚ) ((Fin 2 → ℚ)) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.finrank_tensor_general (ℚ) ((Fin 3 → ℚ)) ((Fin 4 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Module.finrank_tensorProduct]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.TensorProducts.finrank_tensor_general
-- test: la-finrank-tensor-general-test.sage

/-- The products `e_i ⊗ f_a` of basis vectors form a basis of `V ⊗ W` (indexed by pairs `(i, a)`). -/
noncomputable def tensorBasis {K V W ι κ : Type*} [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W]
    [Module K W] (e : Basis ι K V) (f : Basis κ K W) : Basis (ι × κ) K (V ⊗[K] W) :=
  e.tensorProduct f

theorem tensorBasis_apply {K V W ι κ : Type*} [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W]
    [Module K W] (e : Basis ι K V) (f : Basis κ K W) (i : ι) (a : κ) : tensorBasis e f (i, a) = e i ⊗ₜ f a :=
  Basis.tensorProduct_apply e f i a

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.tensorBasis_apply (ℚ) ((Fin 1 → ℚ)) ((Fin 1 → ℚ)) ((Fin 1)) ((Fin 1)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 1)) (Pi.basisFun ℚ (Fin 1)) (0) (0) := by
  simp [tensorBasis]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.tensorBasis_apply (ℚ) ((Fin 2 → ℚ)) ((Fin 3 → ℚ)) ((Fin 2)) ((Fin 3)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) (Pi.basisFun ℚ (Fin 3)) (1) (2) := by
  simp [tensorBasis]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.tensorBasis_apply (ℚ) ((Fin 3 → ℚ)) ((Fin 2 → ℚ)) ((Fin 3)) ((Fin 2)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 3)) (Pi.basisFun ℚ (Fin 2)) (0) (1) := by
  simp [tensorBasis]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.TensorProducts.tensorBasis_apply
-- test: la-tensorbasis-apply-test.sage

/-! ## exm-tensor-space-dim -/

/-- The basis `e_i ⊗ e_j` of `ℝ³ ⊗ ℝ³` has `9` elements. -/
theorem tensor_basis_card : Fintype.card (Fin 3 × Fin 3) = 9 := by simp

/-- The basis `e_i ⊗ f_a` of `V ⊗ W` has `dim V · dim W` elements: `|Fin m × Fin n| = m n`. -/
theorem card_fin_prod (m n : ℕ) : Fintype.card (Fin m × Fin n) = m * n := by simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (4) (2) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (0) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (0) (3) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (1) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod (1) (1) := by
  simp

attribute [veriphy_tested kernel_test 6 0] KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod
-- test: la-card-fin-prod-test.sage

/-! ## exm-tensor-space-dual -/

/-- `V* ⊗ V*` has the dimension `9` of the bilinear forms (`3 × 3` matrices) on `ℝ³`. -/
theorem finrank_dual_dual :
    finrank ℝ (Module.Dual ℝ (Fin 3 → ℝ) ⊗[ℝ] Module.Dual ℝ (Fin 3 → ℝ)) = 9 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exm-tensor-space-mixed -/

/-- `α ⊗ v` acts as the linear map `w ↦ α(w) v`. -/
theorem dualTensor_apply {V : Type*} [AddCommGroup V] [Module ℝ V] (α : Module.Dual ℝ V) (v w : V) :
    dualTensorHom ℝ V V (α ⊗ₜ v) w = α w • v :=
  dualTensorHom_apply α w v

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.dualTensor_apply ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (0) (![1, 2]) (![3, -1]) := by
  simp [dualTensorHom_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.dualTensor_apply ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) (![1, 2]) (![3, -1]) := by
  simp [dualTensorHom_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.dualTensor_apply ((Fin 3 → ℝ)) (inferInstance) (inferInstance) (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 3 => ℝ) 2) (![1, 0, -1]) (![1, 2, 3]) := by
  simp [dualTensorHom_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.TensorProducts.dualTensor_apply
-- test: la-dualtensor-apply-test.sage

/-- `V* ⊗ V ≅ End V` in finite dimension. -/
noncomputable def dualTensorEquiv (V : Type*) [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V] :
    Module.Dual ℝ V ⊗[ℝ] V ≃ₗ[ℝ] (V →ₗ[ℝ] V) :=
  dualTensorHomEquiv ℝ V V

/-! ## exr-tensor-components -/

/-- Contracting `a = (1, 0, -1)` with the first slot of `u ⊗ v` gives `a(u) v = (-2, 0, 2)`. -/
theorem contract_a_uv :
    !![(1 : ℚ), 0, -1] * (!![(1 : ℚ); 2; 3] * !![(1 : ℚ), 0, -1]) = !![(-2 : ℚ), 0, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-tensor-components.sage

/-- The components of `u = (1, 2, 3)` in `f = (e₁ + e₂, e₂, e₃)` are `(1, 1, 3)`: `A (1, 1, 3)ᵀ = u`. -/
theorem u_in_f :
    !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] * !![(1 : ℚ); 1; 3] = !![(1 : ℚ); 2; 3] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-tensor-components.sage

/-- The components of `u ⊗ v` in `f`: `A X Aᵀ = u vᵀ` with `X = (1, -1, -1; 1, -1, -1; 3, -3, -3)`. -/
theorem uv_in_f :
    !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] * !![(1 : ℚ), -1, -1; 1, -1, -1; 3, -3, -3] * (!![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1])ᵀ =
      !![(1 : ℚ), 0, -1; 2, 0, -2; 3, 0, -3] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-tensor-components.sage

/-- `T` in the basis `f`: `A T' = T A` with `T' = (3, 2, 0; 0, 1, 1; 0, 0, 5)`. -/
theorem T_in_f :
    !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] * !![(3 : ℚ), 2, 0; 0, 1, 1; 0, 0, 5] =
      !![(1 : ℚ), 2, 0; 0, 3, 1; 0, 0, 5] * !![(1 : ℚ), 0, 0; 1, 1, 0; 0, 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-tensor-components.sage

/-- The trace is `9` in both bases. -/
theorem trace_T_in_f : Matrix.trace !![(3 : ℚ), 2, 0; 0, 1, 1; 0, 0, 5] = 9 := by
  norm_num [Matrix.trace_fin_three_of]
  -- certificate: la-tensor-components.sage

/-! ## exr-tensor-dim-general -/

/-- `dim (V ⊗ W ⊗ U) = 2 · 3 · 4 = 24`. -/
theorem finrank_triple_tensor :
    finrank ℝ ((Fin 2 → ℝ) ⊗[ℝ] (Fin 3 → ℝ) ⊗[ℝ] (Fin 4 → ℝ)) = 24 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-- A tensor with `k` indices in dimension `n` has `n^k` components (multi-indices `Fin k → Fin n`). -/
theorem multi_index_count (n k : ℕ) : Fintype.card (Fin k → Fin n) = n ^ k := by simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (4) (2) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (0) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (0) (3) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (1) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count (1) (1) := by
  simp

attribute [veriphy_tested kernel_test 6 0] KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count
-- test: la-multi-index-count-test.sage

/-! ## exr-tensor-basis-r2 -/

/-- `T(α, β) = T^{ij} α_i β_j = 3T¹¹ - T¹² + 6T²¹ - 2T²²` for `α = (1, 2)`, `β = (3, -1)`. -/
theorem tensor_eval_r2 (t11 t12 t21 t22 : ℚ) :
    !![(1 : ℚ), 2] * !![t11, t12; t21, t22] * !![(3 : ℚ); -1] = !![3 * t11 - t12 + 6 * t21 - 2 * t22] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-matrix.sage

/-- The basis `e_i ⊗ e_j` of `ℝ² ⊗ ℝ²` has four elements. -/
theorem tensor_basis_r2_card : Fintype.card (Fin 2 × Fin 2) = 4 := by simp

end KaluzaKlein.LinearAlgebra.TensorProducts

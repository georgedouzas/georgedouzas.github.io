import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "The dual space"

The claims of the section "The dual space" of `tutorial/part-i-02-linear-algebra.qmd`: `dim V* = dim V`, rows as the
dual of `ℝⁿ`, cotangent spaces, the double dual, and operations on functionals.
-/

namespace KaluzaKlein.LinearAlgebra.DualSpace

open Matrix Module

/-! ## def-dual-space, exr-dual-dimension -/

/-- `dim V* = dim V` for finite-dimensional `V`. -/
theorem finrank_dual {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V] :
    finrank K (Module.Dual K V) = finrank K V :=
  Subspace.dual_finrank_eq

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual (ℚ) ((Fin 0 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual (ℚ) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual (ℚ) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) := by
  simp [Subspace.dual_finrank_eq]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual
-- test: la-finrank-dual-test.sage

/-- The dual basis of a basis `e`: `e^i(e_j) = δ^i_j`, a basis of `V*`. -/
theorem dualBasis_delta {K V ι : Type*} [Field K] [AddCommGroup V] [Module K V] [Fintype ι] [DecidableEq ι]
    (e : Basis ι K V) (i j : ι) : e.dualBasis i (e j) = if j = i then 1 else 0 :=
  e.dualBasis_apply_self i j

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dualBasis_delta (ℚ) ((Fin 1 → ℚ)) ((Fin 1)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 1)) (0) (0) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dualBasis_delta (ℚ) ((Fin 2 → ℚ)) ((Fin 2)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) (0) (1) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dualBasis_delta (ℚ) ((Fin 3 → ℚ)) ((Fin 3)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 3)) (2) (2) := by
  simp [Basis.dualBasis_apply, Pi.basisFun_repr, Pi.single_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.dualBasis_delta
-- test: la-dualbasis-delta-test.sage

/-- Every functional is `α = α(e_i) e^i`. -/
theorem dual_expansion {K V ι : Type*} [Field K] [AddCommGroup V] [Module K V] [Fintype ι] [DecidableEq ι]
    (e : Basis ι K V) (α : Module.Dual K V) : ∑ i, α (e i) • e.dualBasis i = α := by
  apply e.ext
  intro j
  simp [Basis.dualBasis_apply_self, Finset.sum_apply']

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dual_expansion (ℚ) ((Fin 2 → ℚ)) ((Fin 2)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) (0) := by
  ext v
  simp [Fin.sum_univ_succ, Basis.dualBasis_apply, Pi.basisFun_repr]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dual_expansion (ℚ) ((Fin 2 → ℚ)) ((Fin 2)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 2)) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) := by
  ext v
  simp [Fin.sum_univ_succ, Basis.dualBasis_apply, Pi.basisFun_repr]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.dual_expansion (ℚ) ((Fin 3 → ℚ)) ((Fin 3)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℚ (Fin 3)) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) := by
  ext v
  simp [Fin.sum_univ_succ, Basis.dualBasis_apply, Pi.basisFun_repr]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.dual_expansion
-- test: la-dual-expansion-test.sage

/-! ## exm-dual-space-rows -/

/-- The dual of `ℝⁿ` has dimension `n`. -/
theorem finrank_dual_rn (n : ℕ) : finrank ℝ (Module.Dual ℝ (Fin n → ℝ)) = n := by
  rw [Subspace.dual_finrank_eq, Module.finrank_fin_fun]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual_rn (0) := by
  simp [Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual_rn (4) := by
  simp [Subspace.dual_finrank_eq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual_rn (1) := by
  simp [Subspace.dual_finrank_eq]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual_rn
-- test: la-finrank-dual-rn-test.sage

/-! ## exm-dual-space-cotangent -/

/-- The cotangent space, the dual of the tangent space of an `n`-manifold modelled on `ℝⁿ`, has dimension `n`. -/
theorem finrank_cotangent (n : ℕ) (x : EuclideanSpace ℝ (Fin n)) :
    finrank ℝ (Module.Dual ℝ (TangentSpace (modelWithCornersSelf ℝ (EuclideanSpace ℝ (Fin n))) x)) = n := by
  rw [Subspace.dual_finrank_eq]
  exact finrank_euclideanSpace_fin

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_cotangent (0) (0) := by
  rw [Subspace.dual_finrank_eq]
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_cotangent (2) (0) := by
  rw [Subspace.dual_finrank_eq]
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.finrank_cotangent (3) (EuclideanSpace.single 0 1) := by
  rw [Subspace.dual_finrank_eq]
  show Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.finrank_cotangent
-- test: la-finrank-cotangent-test.sage

/-! ## exm-dual-space-dual-dual, exr-dual-double -/

/-- `v ↦ ev_v` is a linear isomorphism `V ≃ V**` in finite dimension. -/
noncomputable def doubleDual (K V : Type*) [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V] :
    V ≃ₗ[K] Module.Dual K (Module.Dual K V) :=
  Module.evalEquiv K V

theorem doubleDual_apply {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V] (v : V)
    (α : Module.Dual K V) : doubleDual K V v α = α v := rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.doubleDual_apply (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (0) (0) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.doubleDual_apply (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (![1, 2]) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.doubleDual_apply (ℚ) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (![1, 0, -1]) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.doubleDual_apply
-- test: la-double-dual-apply-test.sage

/-- Evaluation is injective. -/
theorem eval_injective {K V : Type*} [Field K] [AddCommGroup V] [Module K V] :
    Function.Injective (Module.Dual.eval K V) :=
  Module.eval_apply_injective K

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.eval_injective (ℚ) ((Fin 0 → ℚ)) (inferInstance) (inferInstance) (inferInstance) := by
  intro v w h
  funext i
  simpa using LinearMap.congr_fun h (LinearMap.proj (R := ℚ) (φ := fun _ => ℚ) i)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.eval_injective (ℚ) ((Fin 2 → ℚ)) (inferInstance) (inferInstance) (inferInstance) := by
  intro v w h
  funext i
  simpa using LinearMap.congr_fun h (LinearMap.proj (R := ℚ) (φ := fun _ => ℚ) i)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualSpace.eval_injective (ℚ) ((Fin 3 → ℚ)) (inferInstance) (inferInstance) (inferInstance) := by
  intro v w h
  funext i
  simpa using LinearMap.congr_fun h (LinearMap.proj (R := ℚ) (φ := fun _ => ℚ) i)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualSpace.eval_injective
-- test: la-eval-injective-test.sage

/-! ## exr-dual-rows -/

/-- `α(v) = 4` for `α = (2, 1, 0)`, `v = (1, 2, 3)`. -/
theorem alpha_v : !![(2 : ℚ), 1, 0] * !![(1 : ℚ); 2; 3] = !![(4 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-rows.sage

/-- `β(v) = 8` for `β = (1, -1, 3)`. -/
theorem beta_v : !![(1 : ℚ), -1, 3] * !![(1 : ℚ); 2; 3] = !![(8 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-rows.sage

/-- `(α + β)(v) = 12`. -/
theorem alpha_add_beta_v : (!![(2 : ℚ), 1, 0] + !![(1 : ℚ), -1, 3]) * !![(1 : ℚ); 2; 3] = !![(12 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-rows.sage

/-- `(4α)(v) = 16`. -/
theorem four_alpha_v : ((4 : ℚ) • !![(2 : ℚ), 1, 0]) * !![(1 : ℚ); 2; 3] = !![(16 : ℚ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-rows.sage

/-- `α = 2 e¹ + e²`: the coordinate functionals span the rows. -/
theorem alpha_in_coordinates :
    (2 : ℚ) • !![(1 : ℚ), 0, 0] + (1 : ℚ) • !![(0 : ℚ), 1, 0] + (0 : ℚ) • !![(0 : ℚ), 0, 1] = !![(2 : ℚ), 1, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-rows.sage

end KaluzaKlein.LinearAlgebra.DualSpace

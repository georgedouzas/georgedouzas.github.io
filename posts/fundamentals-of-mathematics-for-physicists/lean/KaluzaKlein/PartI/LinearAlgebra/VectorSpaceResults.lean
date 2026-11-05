import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.Dimension
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Results" of "Vector spaces"

The claims of the two propositions closing the part "Vector spaces" of `tutorial/part-i-02-linear-algebra.qmd` (the
dimensions of `ℝ³` and of the `3 × 3` matrices; the coordinates of `(5, 4)` in the basis `f`) and of their examples and
exercises: dimensions of function and matrix spaces, the plane `x + y + z = 0`, the dimension formula for a sum and an
intersection, coordinates by the inverse matrix and the change-of-basis matrix.
-/

namespace KaluzaKlein.LinearAlgebra.VectorSpaceResults

open Matrix Module KaluzaKlein.LinearAlgebra.VectorSpaces KaluzaKlein.LinearAlgebra.Spans

/-- Rank-nullity for a surjective functional: its kernel has codimension one. -/
theorem finrank_ker_functional {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V]
    (f : V →ₗ[K] K) (hf : Function.Surjective f) : finrank K (LinearMap.ker f) + 1 = finrank K V := by
  have := LinearMap.finrank_range_add_finrank_ker f
  rw [LinearMap.range_eq_top.mpr hf, finrank_top, Module.finrank_self] at this
  omega

/-! ## exm-dim-lean-pi (functions on a finite set) -/

/-- The functions on a finite set `X` form a space of dimension `|X|`. -/
theorem finrank_functions (X : Type*) [Fintype X] : finrank ℝ (X → ℝ) = Fintype.card X :=
  Module.finrank_fintype_fun_eq_card ℝ

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions ((Fin 0)) (inferInstance) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions (Bool) (inferInstance) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions ((Fin 5)) (inferInstance) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions ((Fin 2 × Fin 3)) (inferInstance) := by
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions
-- test: la-finrank-functions-test.sage

theorem finrank_functions_five : finrank ℝ (Fin 5 → ℝ) = 5 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exm-dim-lean-matrix (matrices as functions of two indices) -/

theorem finrank_matrix_3_3 : finrank ℝ (Matrix (Fin 3) (Fin 3) ℝ) = 3 * 3 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage
theorem finrank_matrix_2_3 : finrank ℝ (Matrix (Fin 2) (Fin 3) ℝ) = 6 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exm-dim-sage (traceless matrices) -/

/-- The traceless `3 × 3` matrices, the kernel of the trace, have dimension `9 - 1 = 8`. -/
theorem finrank_traceless :
    finrank ℝ (LinearMap.ker (Matrix.traceLinearMap (Fin 3) ℝ ℝ)) = 8 := by
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (Matrix.traceLinearMap (Fin 3) ℝ ℝ) (!![1, 0, 0; 0, 0, 0; 0, 0, 0])
          (by simp [Matrix.trace, Fin.sum_univ_succ])
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: la-dim-ker.sage

/-! ## exr-dim-matrix-lean (real and complex dimensions) -/

theorem finrank_complex_matrix : finrank ℂ (Matrix (Fin 2) (Fin 2) ℂ) = 4 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

theorem finrank_real_complex_matrix : finrank ℝ (Matrix (Fin 2) (Fin 2) ℂ) = 8 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-! ## exr-dim-plane -/

/-- The functional `x + y + z` on `ℝ³`. -/
noncomputable def sumFun : (Fin 3 → ℝ) →ₗ[ℝ] ℝ :=
  LinearMap.proj (R := ℝ) (φ := fun _ : Fin 3 => ℝ) 0 + LinearMap.proj (R := ℝ) (φ := fun _ : Fin 3 => ℝ) 1 +
    LinearMap.proj (R := ℝ) (φ := fun _ : Fin 3 => ℝ) 2

theorem ker_sumFun : LinearMap.ker sumFun = planeSum0 := by
  ext v
  simp [sumFun, planeSum0]

/-- The plane `x + y + z = 0` has dimension `2`. -/
theorem finrank_planeSum0 : finrank ℝ planeSum0 = 2 := by
  rw [← ker_sumFun]
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (KaluzaKlein.LinearAlgebra.VectorSpaceResults.sumFun) (![1, 0, 0])
          (by simp [KaluzaKlein.LinearAlgebra.VectorSpaceResults.sumFun])
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: la-dim-ker.sage

/-- `(1, -1, 0)`, `(0, 1, -1)` and Sage's `(1, 0, -1)`, `(0, 1, -1)` lie in the plane. -/
theorem plane_basis_vectors :
    !![(1 : ℝ), 1, 1] * !![(1 : ℝ), 0, 1, 0; -1, 1, 0, 1; 0, -1, -1, -1] = !![(0 : ℝ), 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dim-plane.sage

/-- A vanishing `a (1, -1, 0) + b (0, 1, -1)` has `a = b = 0`; likewise for `(1, 0, -1)`, `(0, 1, -1)`. -/
theorem plane_basis_coefficients (a b : ℝ) (h0 : a = 0) (h1 : -a + b = 0) : a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination h0
    -- certificate: ideal_membership.sage
  · linear_combination h0 + h1
    -- certificate: ideal_membership.sage

theorem plane_basis_independent : LinearIndependent ℝ ![(![1, -1, 0] : Fin 3 → ℝ), ![0, 1, -1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

theorem plane_echelon_independent : LinearIndependent ℝ ![(![1, 0, -1] : Fin 3 → ℝ), ![0, 1, -1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-! ## exr-dim-sum-intersection -/

/-- The plane `z = 0` of `ℚ³`. -/
def U : Submodule ℚ (Fin 3 → ℚ) := LinearMap.ker (LinearMap.proj 2)

theorem finrank_U : finrank ℚ U = 2 := by
  unfold U
  veriphy_certified
    ( have h :=
        Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 2) (![0, 0, 1]) (by simp)
      simp [Module.finrank_matrix, Module.finrank_fin_fun] at h
      omega)
  -- certificate: la-dim-ker.sage

/-- The plane `W = span{(1, 2, 0), (0, 1, 1)}`. -/
abbrev W : Submodule ℚ (Fin 3 → ℚ) := spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2]

/-- `dim (U ∩ W) = 1`: the intersection is the line through `(1, 2, 0)`. -/
theorem finrank_U_inf_W : finrank ℚ (U ⊓ W : Submodule ℚ (Fin 3 → ℚ)) = 1 := by
  have hk : U = LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 0, 1]) := by
    ext v
    simp [U, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, funext_iff, Fin.forall_fin_one]
  have hs : Submodule.span ℚ (Set.range ![![(1 : ℚ), 0, 0], ![0, 1, 0]]) = LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 0, 1]) := by
    veriphy_certified
      ( apply le_antisymm
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i <;>
            (simp only [SetLike.mem_coe, LinearMap.mem_ker, Matrix.mulVecLin_apply];
              ext j <;> fin_cases j <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> norm_num)
        · intro x hx
          rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx
          have h0 := congrFun hx 0
          simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0
          refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![1 * x 0, 1 * x 1], ?_⟩
          ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
    -- certificate: la-span-ker.sage
  rw [hk, ← hs]
  show finrank ℚ ↥(Submodule.span ℚ (Set.range ![![(1 : ℚ), 0, 0], ![0, 1, 0]]) ⊓ Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])) = 1
  veriphy_certified
    ( refine
        Veriphy.Linear.finrank_inf_of_bounds _ _ (Submodule.span ℚ (Set.range (![![1, 2, 0]] : Fin 1 → Fin 3 → ℚ)))
          (Submodule.span ℚ (Set.range (![![1, 0, 0], ![0, 1, 0], ![0, 1, 1]] : Fin 3 → Fin 3 → ℚ)))
          (Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
          (Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
          ?_
          (Veriphy.Linear.finrank_span_of_minor _ ![0]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
          ?_
          (Veriphy.Linear.finrank_span_of_minor _ ![0, 1, 2]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
          (by norm_num)
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        ·
          exact
            Submodule.mem_inf.2
              ⟨(Submodule.mem_span_range_iff_exists_fun ℚ).2
                  ⟨![1, 2], by
                    ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩,
                (Submodule.mem_span_range_iff_exists_fun ℚ).2
                  ⟨![1, 0], by
                    ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩⟩
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        · exact Submodule.mem_sup_left (Submodule.subset_span ⟨0, rfl⟩)
        · exact Submodule.mem_sup_left (Submodule.subset_span ⟨1, rfl⟩)
        · exact Submodule.mem_sup_right (Submodule.subset_span ⟨1, rfl⟩))
  -- certificate: la-dim-inter.sage

/-- `dim (U + W) = 3`, from `dim (U + W) + dim (U ∩ W) = dim U + dim W = 2 + 2` and `dim (U ∩ W) = 1`. -/
theorem finrank_U_sup_W : finrank ℚ (U ⊔ W : Submodule ℚ (Fin 3 → ℚ)) = 3 := by
  have hsum := Submodule.finrank_sup_add_finrank_inf_eq U W
  have hk : U = LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 0, 1]) := by
    ext v
    simp [U, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, funext_iff, Fin.forall_fin_one]
  have hs : Submodule.span ℚ (Set.range ![![(1 : ℚ), 0, 0], ![0, 1, 0]]) = LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 0, 1]) := by
    veriphy_certified
      ( apply le_antisymm
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i <;>
            (simp only [SetLike.mem_coe, LinearMap.mem_ker, Matrix.mulVecLin_apply];
              ext j <;> fin_cases j <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> norm_num)
        · intro x hx
          rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx
          have h0 := congrFun hx 0
          simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0
          refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![1 * x 0, 1 * x 1], ?_⟩
          ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
    -- certificate: la-span-ker.sage
  have hU : finrank ℚ U = 2 := by
    rw [hk, ← hs]
    veriphy_certified
      (exact
          Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
    -- certificate: la-dim-span.sage
  have hW : finrank ℚ W = 2 := by
    show finrank ℚ ↥(Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])) = 2
    veriphy_certified
      (exact
          Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
    -- certificate: la-dim-span.sage
  have hI : finrank ℚ (U ⊓ W : Submodule ℚ (Fin 3 → ℚ)) = 1 := by
    rw [hk, ← hs]
    show finrank ℚ ↥(Submodule.span ℚ (Set.range ![![(1 : ℚ), 0, 0], ![0, 1, 0]]) ⊓ Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])) = 1
    veriphy_certified
      ( refine
          Veriphy.Linear.finrank_inf_of_bounds _ _ (Submodule.span ℚ (Set.range (![![1, 2, 0]] : Fin 1 → Fin 3 → ℚ)))
            (Submodule.span ℚ (Set.range (![![1, 0, 0], ![0, 1, 0], ![0, 1, 1]] : Fin 3 → Fin 3 → ℚ)))
            (Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
            (Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
            ?_
            (Veriphy.Linear.finrank_span_of_minor _ ![0]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
            ?_
            (Veriphy.Linear.finrank_span_of_minor _ ![0, 1, 2]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
            (by norm_num)
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i
          ·
            exact
              Submodule.mem_inf.2
                ⟨(Submodule.mem_span_range_iff_exists_fun ℚ).2
                    ⟨![1, 2], by
                      ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩,
                  (Submodule.mem_span_range_iff_exists_fun ℚ).2
                    ⟨![1, 0], by
                      ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩⟩
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i
          · exact Submodule.mem_sup_left (Submodule.subset_span ⟨0, rfl⟩)
          · exact Submodule.mem_sup_left (Submodule.subset_span ⟨1, rfl⟩)
          · exact Submodule.mem_sup_right (Submodule.subset_span ⟨1, rfl⟩))
    -- certificate: la-dim-inter.sage
  omega

/-- `(1, 2, 0)` spans the intersection. -/
theorem v1_mem_U_inf_W : KaluzaKlein.Linear.v1 ∈ (U ⊓ W : Submodule ℚ (Fin 3 → ℚ)) := by
  have hk : U = LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 0, 1]) := by
    ext v
    simp [U, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, funext_iff, Fin.forall_fin_one]
  refine Submodule.mem_inf.2 ⟨?_, Submodule.subset_span ⟨0, rfl⟩⟩
  rw [hk, LinearMap.mem_ker, Matrix.mulVecLin_apply]
  show !![(0 : ℚ), 0, 1] *ᵥ ![(1 : ℚ), 2, 0] = 0
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-! ## exm-coords-system -/

/-- From `2a + b = 5` and `a + 2b = 4`: `b = 1` and `a = 2`. -/
theorem coords_system (a b : ℚ) (h1 : 2 * a + b = 5) (h2 : a + 2 * b = 4) : a = 2 ∧ b = 1 := by
  refine ⟨?_, ?_⟩
  · linear_combination (2 / 3) * h1 + (-1 / 3) * h2
    -- certificate: ideal_membership.sage
  · linear_combination (-1 / 3) * h1 + (2 / 3) * h2
    -- certificate: ideal_membership.sage

/-! ## exm-coords-witness (checking the coordinates, uniqueness) -/

/-- The matrix with columns `f₁, f₂` has determinant `3 ≠ 0`, so the coordinates are unique. -/
theorem det_F : Matrix.det !![(2 : ℚ), 1; 1, 2] = 3 := by
  norm_num [Matrix.det_fin_two_of]
  -- certificate: la-matrix.sage

/-! ## exr-coords-f-other -/

theorem coords_f_other_1_exist : ∃ a b : ℚ, 2 * a + b = 1 ∧ a + 2 * b = 5 := by
  exact ⟨-1, 3, by norm_num⟩
  -- certificate: la-coords-f-other-1.sage

theorem coords_f_other_2_exist : ∃ a b : ℚ, 2 * a + b = -1 ∧ a + 2 * b = 1 := by
  exact ⟨-1, 1, by norm_num⟩
  -- certificate: la-coords-f-other-2.sage

/-- `(1, 5) = -f₁ + 3 f₂` and `(-1, 1) = -f₁ + f₂`, as `F (−1, 3)ᵀ` and `F (−1, 1)ᵀ`. -/
theorem coords_f_other :
    !![(2 : ℚ), 1; 1, 2] * !![(-1 : ℚ), -1; 3, 1] = !![(1 : ℚ), -1; 5, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-coords-f-other.sage

/-! ## exr-coords-inverse-matrix -/

/-- If `F` is invertible and `F λ = v`, then `λ = F⁻¹ v`. -/
theorem coords_inverse {n : ℕ} (F : Matrix (Fin n) (Fin n) ℚ) (h : IsUnit F.det) (lam v : Fin n → ℚ)
    (hv : F *ᵥ lam = v) : lam = F⁻¹ *ᵥ v := by
  rw [← hv, Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul F h, Matrix.one_mulVec]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.coords_inverse (0) (0) ((by simp)) (![]) (![]) ((by ext i; exact i.elim0)) := by
  ext i
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.coords_inverse (2) (!![2, 1; 1, 2]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))) (![2, 1]) (![5, 4]) ((by ext i; fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ])) := by
  rw [Matrix.inv_eq_left_inv (B := !![(2 / 3), (-1 / 3); (-1 / 3), (2 / 3)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i
  fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaceResults.coords_inverse (3) (!![1, 0, 1; 1, 1, 0; 0, 1, 1]) ((isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))) (![1 / 2, 5 / 2, 3 / 2]) (![2, 3, 4]) ((by ext i; fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ])) := by
  rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2), (-1 / 2); (-1 / 2), (1 / 2), (1 / 2); (1 / 2), (-1 / 2), (1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i
  fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaceResults.coords_inverse
-- test: la-coords-inverse-test.sage

/-- `F · ⅓ (2, -1; -1, 2) = 1`. -/
theorem F_mul_inverse : !![(2 : ℚ), 1; 1, 2] * !![(2 / 3 : ℚ), -1 / 3; -1 / 3, 2 / 3] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-change-of-basis -/

/-- `G M = F` for `M = (3/2, 3/2; 1/2, -1/2)`, so `M = G⁻¹ F`. -/
theorem change_of_basis_M : !![(1 : ℚ), 1; 1, -1] * !![(3 / 2 : ℚ), 3 / 2; 1 / 2, -1 / 2] = !![(2 : ℚ), 1; 1, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-change-of-basis.sage

/-- `M (2, 1)ᵀ = (9/2, 1/2)ᵀ`, and `G (9/2, 1/2)ᵀ = (5, 4)ᵀ`. -/
theorem change_of_basis_check :
    !![(3 / 2 : ℚ), 3 / 2; 1 / 2, -1 / 2] * !![(2 : ℚ); 1] = !![(9 / 2 : ℚ); 1 / 2] ∧
      !![(1 : ℚ), 1; 1, -1] * !![(9 / 2 : ℚ); 1 / 2] = !![(5 : ℚ); 4] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-change-of-basis.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-change-of-basis.sage

end KaluzaKlein.LinearAlgebra.VectorSpaceResults

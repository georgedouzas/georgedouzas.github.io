import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.LinearCombinations
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Spans"

The claims of the section "Spans" of `tutorial/part-i-02-linear-algebra.qmd`: the plane spanned by `v₁ = (1, 2, 0)`
and `v₂ = (0, 1, 1)`, `𝔰𝔬(3)` as the span of `L₁, L₂, L₃`, the first Fourier modes, membership tests and spans of
polynomials.
-/

namespace KaluzaKlein.LinearAlgebra.Spans

open Matrix Module KaluzaKlein.LinearAlgebra.VectorSpaces KaluzaKlein.LinearAlgebra.LinearCombinations

/-! ## def-span -/

/-- The span of a family, mathlib's `Submodule.span` of its range. -/
abbrev spanOf {ι V : Type*} [AddCommGroup V] [Module ℚ V] (v : ι → V) : Submodule ℚ V := Submodule.span ℚ (Set.range v)

/-- The running example: `v₃ = (1, 3, 1)` lies in the span of `v₁, v₂` (`KaluzaKlein.Linear.v3_mem_span`). -/
theorem v3_mem_spanOf : KaluzaKlein.Linear.v3 ∈ spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2] := by
  show ![(1 : ℚ), 3, 1] ∈ Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])
  veriphy_certified
    (exact
        (Submodule.mem_span_range_iff_exists_fun ℚ).2
          ⟨![1, 1], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩)
  -- certificate: la-span-member.sage

/-! ## exm-span-plane -/

/-- A vanishing combination of `v₁, v₂` has zero coefficients (`a = 0`, `2a + b = 0`, `b = 0`). -/
theorem v12_independent_coefficients (a b : ℚ) (h1 : a = 0) (h2 : 2 * a + b = 0) (h3 : b = 0) : a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination (1 / 2) * h2 + (-1 / 2) * h3
    -- certificate: ideal_membership.sage
  · linear_combination h3
    -- certificate: ideal_membership.sage

theorem v12_linearIndependent : LinearIndependent ℚ ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2] := by
  rw [LinearIndependent.pair_iff]
  intro a b h
  have h0 := congrFun h 0; have h1 := congrFun h 1; have h2 := congrFun h 2
  simp [KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2] at h0 h1 h2
  exact v12_independent_coefficients a b h0 (by linarith) h2

/-- The span of `v₁, v₂` is a plane: dimension `2`. -/
theorem finrank_span_v12 : finrank ℚ (spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2]) = 2 := by
  show finrank ℚ ↥(Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])) = 2
  veriphy_certified
    (exact
        Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-dim-span.sage

/-- Adding `v₃` does not enlarge the span. -/
theorem span_v123_eq : spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2, KaluzaKlein.Linear.v3] =
    spanOf ![KaluzaKlein.Linear.v1, KaluzaKlein.Linear.v2] := by
  show Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1], ![1, 3, 1]]) = Submodule.span ℚ (Set.range ![![(1 : ℚ), 2, 0], ![0, 1, 1]])
  veriphy_certified
    ( apply le_antisymm
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![1, 0], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![0, 1], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![1, 1], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![1, 0, 0], by
                ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![0, 1, 0], by
                ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩)
  -- certificate: la-span-eq.sage

/-! ## exm-span-pauli -/

theorem L1_antisymm : (!![(0 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0])ᵀ = -!![(0 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage
theorem L2_antisymm : (!![(0 : ℝ), 0, 1; 0, 0, 0; -1, 0, 0])ᵀ = -!![(0 : ℝ), 0, 1; 0, 0, 0; -1, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage
theorem L3_antisymm : (!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 0])ᵀ = -!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- `𝔰𝔬(3)` is the span of `L₁, L₂, L₃`. -/
theorem so3_eq_span : antisymMatrices 3 = Submodule.span ℝ (Set.range ![L1, L2, L3]) := by
  veriphy_certified
    ( apply le_antisymm
      · intro X hX
        have h : Xᵀ = -X := hX
        have e : ∀ i j, X j i = -X i j := fun i j => by simpa using congrFun (congrFun h i) j
        refine (Submodule.mem_span_range_iff_exists_fun ℝ).2 ⟨![-1 * X 1 2, 1 * X 0 2, -1 * X 0 1], ?_⟩
        ext i j
        fin_cases i <;> fin_cases j <;>
            simp [Fin.sum_univ_succ, KaluzaKlein.LinearAlgebra.LinearCombinations.L1,
              KaluzaKlein.LinearAlgebra.LinearCombinations.L2, KaluzaKlein.LinearAlgebra.LinearCombinations.L3] <;>
          linarith [e 0 0, e 0 1, e 0 2, e 1 1, e 1 2, e 2 2]
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i <;>
          (show Matrix.transpose _ = -_; ext a b;
            fin_cases a <;> fin_cases b <;>
              simp [KaluzaKlein.LinearAlgebra.LinearCombinations.L1, KaluzaKlein.LinearAlgebra.LinearCombinations.L2,
                KaluzaKlein.LinearAlgebra.LinearCombinations.L3]))
  -- certificate: la-span-so3.sage

/-! ## exm-span-functions -/

/-- A vanishing `a cos + b sin` has `a = b = 0` (values at `0` and `π/2`). -/
theorem cos_sin_coefficients (a b : ℝ) (h0 : a * 1 + b * 0 = 0) (h1 : a * 0 + b * 1 = 0) : a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination h0
    -- certificate: ideal_membership.sage
  · linear_combination h1
    -- certificate: ideal_membership.sage

theorem cos_sin_linearIndependent : LinearIndependent ℝ ![Real.cos, Real.sin] := by
  rw [LinearIndependent.pair_iff]
  intro a b h
  have h0 := congrFun h 0
  have h1 := congrFun h (Real.pi / 2)
  simp at h0 h1
  exact cos_sin_coefficients a b (by linarith) (by linarith)

/-- The span of `cos, sin` is two-dimensional. -/
theorem finrank_span_cos_sin : finrank ℝ (Submodule.span ℝ (Set.range ![Real.cos, Real.sin])) = 2 := by
  have hli : LinearIndependent ℝ ![Real.cos, Real.sin] := by
    veriphy_certified
      ( refine
          LinearIndependent.of_comp
            (LinearMap.pi fun k : Fin 2 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2] k)) ?_
        exact
          Veriphy.Linear.linearIndependent_of_minor _ ![0, 1]
            (by
              norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply,
                Matrix.entryLinearMap_apply, Polynomial.lcoeff_apply, Polynomial.coeff_one, Polynomial.coeff_X,
                Polynomial.coeff_sub, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi,
                Real.sin_pi, KaluzaKlein.LinearAlgebra.LinearCombinations.L1,
                KaluzaKlein.LinearAlgebra.LinearCombinations.L2, KaluzaKlein.LinearAlgebra.LinearCombinations.L3]))
    -- certificate: la-independence-eval.sage
  rw [finrank_span_eq_card hli]
  simp

/-- Its elements are the first Fourier modes `a cos φ + b sin φ`. -/
theorem mem_span_cos_sin (f : ℝ → ℝ) :
    f ∈ Submodule.span ℝ (Set.range ![Real.cos, Real.sin]) ↔ ∃ a b : ℝ, f = fun φ => a * Real.cos φ + b * Real.sin φ := by
  rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]
  constructor
  · rintro ⟨a, b, rfl⟩; exact ⟨a, b, by funext φ; simp⟩
  · rintro ⟨a, b, rfl⟩; exact ⟨a, b, by funext φ; simp⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.mem_span_cos_sin (0) := by
  refine ⟨fun _ => ⟨0, 0, ?_⟩, fun _ => ?_⟩
  · funext φ; simp <;> ring
  · rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]
    exact ⟨0, 0, by funext φ; simp <;> ring⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.mem_span_cos_sin (Real.cos) := by
  refine ⟨fun _ => ⟨1, 0, ?_⟩, fun _ => ?_⟩
  · funext φ; simp <;> ring
  · rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]
    exact ⟨1, 0, by funext φ; simp <;> ring⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.mem_span_cos_sin (fun φ => 2 * Real.cos φ - Real.sin φ) := by
  refine ⟨fun _ => ⟨2, -1, ?_⟩, fun _ => ?_⟩
  · funext φ; simp <;> ring
  · rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]
    exact ⟨2, -1, by funext φ; simp <;> ring⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Spans.mem_span_cos_sin
-- test: la-mem-span-cos-sin-test.sage

/-! ## exr-span-membership -/

/-- `(2, 5, 1) = a v₁ + b v₂` has a solution. -/
theorem membership_exists : ∃ a b : ℚ, a * 1 + b * 0 = 2 ∧ a * 2 + b * 1 = 5 ∧ a * 0 + b * 1 = 1 := by
  exact ⟨2, 1, by norm_num⟩
  -- certificate: la-span-membership.sage

/-- `(2, 5, 1) = 2 v₁ + v₂`. -/
theorem membership_check : (2 : ℚ) • !![(1 : ℚ), 2, 0] + (1 : ℚ) • !![(0 : ℚ), 1, 1] = !![(2 : ℚ), 5, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- `(1, 1, 1) = a v₁ + b v₂` is contradictory: `a = 1`, `2a + b = 1`, `b = 1`. -/
theorem nonmember_contradiction (a b : ℚ) (h1 : a = 1) (h2 : 2 * a + b = 1) (h3 : b = 1) : False := by
  have hκ : (1 : ℚ) ≤ 0 := by linear_combination (-1) * h1 + (1 / 2) * h2 + (-1 / 2) * h3
  norm_num at hκ
  -- certificate: farkas.sage

/-- The normal `(2, -1, 1)` of the plane pairs to zero with `v₁` and `v₂`: the plane is `2x - y + z = 0`. -/
theorem plane_normal : !![(2 : ℚ), -1, 1] * !![(1 : ℚ), 0; 2, 1; 0, 1] = !![(0 : ℚ), 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-span-line-plane -/

/-- If `v₃ = a v₁ + b v₂`, every combination of the three is a combination of the first two. -/
theorem span_three_reduce {V : Type*} [AddCommGroup V] [Module ℝ V] (v1 v2 v3 : V) (a b l1 l2 l3 : ℝ)
    (h : v3 = a • v1 + b • v2) :
    l1 • v1 + l2 • v2 + l3 • v3 = (l1 + a * l3) • v1 + (l2 + b * l3) • v2 := by
  rw [h]
  module

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.span_three_reduce ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (0) (0) (0) (0) (0) (0) (0) (0) ((by simp)) := by
  funext i
  fin_cases i <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.span_three_reduce ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (![1, 0]) (![0, 1]) (![2, 3]) (2) (3) (1) (-1) (1 / 2) ((by funext i; fin_cases i <;> norm_num)) := by
  funext i
  fin_cases i <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.span_three_reduce ((Fin 3 → ℝ)) (inferInstance) (inferInstance) (![1, 2, 0]) (![0, 1, 1]) (![1, 3, 1]) (1) (1) (2) (0) (-3) ((by funext i; fin_cases i <;> norm_num)) := by
  funext i
  fin_cases i <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Spans.span_three_reduce
-- test: la-span-three-reduce-test.sage

/-! ## exr-span-polynomials -/

open Polynomial in
/-- `(x + 1)² = 1 + 2x + x²`. -/
theorem shifted_square : ((X + 1) ^ 2 : ℝ[X]) = 1 + 2 * X + X ^ 2 := by
  veriphy_certified (linear_combination)
  -- certificate: la-polynomial.sage

open Polynomial in
/-- `x³` is not in the span of `1, x, x²` (the polynomials of degree `< 3`). -/
theorem X_cube_not_mem : (X ^ 3 : ℝ[X]) ∉ Polynomial.degreeLT ℝ 3 := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  decide

open Polynomial in
/-- `xⁿ` is not in the span of `1, x, …, xⁿ⁻¹` (the polynomials of degree `< n`). -/
theorem X_pow_not_mem (n : ℕ) : (X ^ n : ℝ[X]) ∉ Polynomial.degreeLT ℝ n := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  exact lt_irrefl _

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.X_pow_not_mem (0) := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.X_pow_not_mem (1) := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.X_pow_not_mem (2) := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Spans.X_pow_not_mem (3) := by
  rw [Polynomial.mem_degreeLT, Polynomial.degree_X_pow]
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.Spans.X_pow_not_mem
-- test: la-x-pow-not-mem-test.sage

open Polynomial in
/-- `x = (x + 1) - 1` and `x² = (x + 1)² - 2 (x + 1) + 1`. -/
theorem powers_from_shifted : (X : ℝ[X]) = (X + 1) - 1 ∧ (X ^ 2 : ℝ[X]) = (X + 1) ^ 2 - 2 * (X + 1) + 1 := by
  constructor
  · veriphy_certified (linear_combination)
    -- certificate: la-polynomial.sage
  · veriphy_certified (linear_combination)
    -- certificate: la-polynomial.sage

end KaluzaKlein.LinearAlgebra.Spans

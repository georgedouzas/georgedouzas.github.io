import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.LinearMaps
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Results" of "Linear maps and dual spaces"

The claims of the propositions closing the part "Linear maps and dual spaces" of
`tutorial/part-i-02-linear-algebra.qmd` (a kernel vector of `A = (1, 2, 3; 2, 4, 6)`; dual basis evaluations) and of
their examples and exercises: bases of `ker A`, rank and proportional rows, the dual basis of `f` applied to vectors,
dual bases in general.
-/

namespace KaluzaKlein.LinearAlgebra.DualResults

open Matrix Module KaluzaKlein.LinearAlgebra.LinearMaps

/-! ## exm-kernel-lean-route (more kernel vectors) -/

/-- `(-2, 1, 0)` and every multiple `t (1, 1, -1)` lie in the kernel of `A`. -/
theorem A_mul_more_kernel (t : ℚ) :
    !![(1 : ℚ), 2, 3; 2, 4, 6] * !![(-2 : ℚ), t; 1, t; 0, -t] = !![(0 : ℚ), 0; 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1
  -- certificate: la-kernel-image.sage

/-! ## exr-kernel-basis -/

/-- `(-2, 1, 0)`, `(-3, 0, 1)`, `(1, 1, -1)`, `(3, 0, -1)` are kernel vectors of `A`. -/
theorem A_mul_kernel_vectors :
    !![(1 : ℚ), 2, 3; 2, 4, 6] * !![(-2 : ℚ), -3, 1, 3; 1, 0, 1, 0; 0, 1, -1, -1] = !![(0 : ℚ), 0, 0, 0; 0, 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-kernel-basis.sage

/-- `(1, 1, -1) = 1 · (-2, 1, 0) - 1 · (-3, 0, 1)` and `(3, 0, -1) = 0 · (-2, 1, 0) - 1 · (-3, 0, 1)`. -/
theorem kernel_coordinates :
    !![(-2 : ℚ), -3; 1, 0; 0, 1] * !![(1 : ℚ), 0; -1, -1] = !![(1 : ℚ), 3; 1, 0; -1, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-kernel-basis.sage

/-- A vanishing `a (1, 1, -1) + b (3, 0, -1)` has `a = b = 0` (second component `a = 0`, first `a + 3b = 0`). -/
theorem kernel_pair_coefficients (a b : ℚ) (h1 : a + 3 * b = 0) (h2 : a = 0) : a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination h2
    -- certificate: ideal_membership.sage
  · linear_combination (1 / 3) * h1 + (-1 / 3) * h2
    -- certificate: ideal_membership.sage

theorem kernel_pair_independent : LinearIndependent ℚ ![(![1, 1, -1] : Fin 3 → ℚ), ![3, 0, -1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-- `(1, 1, -1)`, `(3, 0, -1)` is a basis of `ker A`: two independent vectors in a plane. -/
theorem kernel_pair_spans :
    Submodule.span ℚ (Set.range ![(![1, 1, -1] : Fin 3 → ℚ), ![3, 0, -1]]) =
      LinearMap.ker KaluzaKlein.Linear.A.mulVecLin := by
  rw [show KaluzaKlein.Linear.A = !![(1 : ℚ), 2, 3; 2, 4, 6] from rfl]
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
        have h1 := congrFun hx 1
        simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0 h1
        refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![1 * x 1, (1 / 3) * x 0 + (-1 / 3) * x 1], ?_⟩
        ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
  -- certificate: la-span-ker.sage

/-! ## exr-kernel-rows-rank -/

/-- A non-zero `2 × n` matrix has rank `1` exactly when its rows are proportional (stated with a non-zero first
row; the other case is symmetric). -/
theorem rank_one_iff_proportional {K : Type*} [Field K] {n : ℕ} (A : Matrix (Fin 2) (Fin n) K) (h0 : A 0 ≠ 0) :
    A.rank = 1 ↔ ∃ c : K, A 1 = c • A 0 :=
  ⟨proportional_rows_of_rank A h0, fun ⟨c, hc⟩ => rank_of_proportional_rows A h0 c hc⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.LinearAlgebra.DualResults.rank_one_iff_proportional
-- test: la-rank-one-iff-test.sage

/-! ## exr-kernel-second-vector -/

/-- `A (3, 0, -1)ᵀ = 0`. -/
theorem A_mul_second : !![(1 : ℚ), 2, 3; 2, 4, 6] * !![(3 : ℚ); 0; -1] = !![(0 : ℚ); 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-kernel-image.sage

/-! ## exm-dualbasis-coordinates, exr-dual-f-apply -/

/-- The dual basis of `f` applied to `(1, 5)` and `(-1, 1)`: coordinates `(-1, 3)` and `(-1, 1)`. -/
theorem dual_f_apply :
    !![(2 / 3 : ℚ), -1 / 3; -1 / 3, 2 / 3] * !![(1 : ℚ), -1; 5, 1] = !![(-1 : ℚ), -1; 3, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-dual-basis.sage

/-! ## exr-dual-basis-general -/

/-- The dual basis of the standard basis of `ℚ²` is the standard basis. -/
theorem dual_standard_two : (1 : Matrix (Fin 2) (Fin 2) ℚ)⁻¹ = 1 := inv_one

/-- The dual basis of the standard basis of `ℚⁿ` is the standard basis: `1⁻¹ = 1`. -/
theorem dual_standard (n : ℕ) : (1 : Matrix (Fin n) (Fin n) ℚ)⁻¹ = 1 := inv_one

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_standard (0) := by
  exact Matrix.inv_eq_left_inv (by simp)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_standard (4) := by
  exact Matrix.inv_eq_left_inv (by simp)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_standard (1) := by
  exact Matrix.inv_eq_left_inv (by simp)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualResults.dual_standard
-- test: la-dual-standard-test.sage

/-- The dual basis of a basis of `ℚⁿ` is the list of rows of `B⁻¹`: `B⁻¹ B = 1` for invertible `B`. -/
theorem dual_basis_rows {n : ℕ} (B : Matrix (Fin n) (Fin n) ℚ) (h : B.det ≠ 0) : B⁻¹ * B = 1 :=
  Matrix.nonsing_inv_mul B (isUnit_iff_ne_zero.mpr h)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_basis_rows (0) (0) ((by simp)) := by
  ext i j
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_basis_rows (2) (!![2, 1; 1, 2]) ((by norm_num [Matrix.det_fin_two_of])) := by
  rw [Matrix.inv_eq_left_inv (B := !![(2 / 3), (-1 / 3); (-1 / 3), (2 / 3)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.DualResults.dual_basis_rows (3) (!![1, 0, 1; 1, 1, 0; 0, 1, 1]) ((by norm_num [Matrix.det_fin_three])) := by
  rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2), (-1 / 2); (-1 / 2), (1 / 2), (1 / 2); (1 / 2), (-1 / 2), (1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.DualResults.dual_basis_rows
-- test: la-dual-basis-rows-test.sage

end KaluzaKlein.LinearAlgebra.DualResults

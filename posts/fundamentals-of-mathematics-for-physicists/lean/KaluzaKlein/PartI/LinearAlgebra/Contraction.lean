import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Contraction"

The claims of the section "Contraction" of `tutorial/part-i-02-linear-algebra.qmd`: the trace as the contraction of a
type `(1, 1)` tensor and its basis independence, the pairing `α(v) = α_i v^i` as a contraction, contractions of an
elementary type `(2, 2)` tensor, and the contractions of the SageManifolds exercise.
-/

namespace KaluzaKlein.LinearAlgebra.Contraction

open Matrix Module

/-! ## def-contraction -/

/-- Pairing a covector with a vector is the contraction of `v ⊗ α`: `tr (v αᵀ) = α_i v^i`. -/
theorem pairing_is_contraction {n : ℕ} (v α : Fin n → ℝ) : (Matrix.vecMulVec v α).trace = α ⬝ᵥ v := by
  simp [Matrix.trace, Matrix.vecMulVec, dotProduct, mul_comm]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.pairing_is_contraction (0) (![]) (![]) := by
  simp [Matrix.trace]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.pairing_is_contraction (3) (![1, 2, 3]) (![1, 0, -1]) := by
  norm_num [Matrix.trace, Matrix.vecMulVec_apply, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.pairing_is_contraction (2) (![1 / 2, 4]) (![2, -1]) := by
  norm_num [Matrix.trace, Matrix.vecMulVec_apply, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Contraction.pairing_is_contraction
-- test: la-pairing-contraction-test.sage

/-! ## exm-contraction-pairing -/

/-- `a(u) v = -2 (1, 0, -1) = (-2, 0, 2)`. -/
theorem contraction_pairing : (-2 : ℚ) • !![(1 : ℚ), 0, -1] = !![(-2 : ℚ), 0, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-contraction-trace-invariance -/

/-- The cyclic property `tr (X Y) = tr (Y X)`. -/
theorem trace_cyclic {n : ℕ} (X Y : Matrix (Fin n) (Fin n) ℝ) : (X * Y).trace = (Y * X).trace :=
  Matrix.trace_mul_comm X Y

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.trace_cyclic (0) (0) (0) := by
  simp [Matrix.trace]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.trace_cyclic (2) (!![1, 2; 3, 4]) (!![0, 1; -1, 2]) := by
  norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.trace_cyclic (3) (!![1, 2, 0; 0, 3, 1; 0, 0, 5]) (!![1, 0, 0; 1, 1, 0; 0, 2, 1]) := by
  norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Contraction.trace_cyclic
-- test: la-trace-cyclic-test.sage

/-! ## exr-contraction-count -/

/-- Contracting `u^i v^j α_k β_l` over `i = k`, `j = l` gives `α(u) β(v)`. -/
theorem elementary_full_contraction {n : ℕ} (u v α β : Fin n → ℝ) :
    ∑ i, ∑ j, u i * v j * α i * β j = (α ⬝ᵥ u) * (β ⬝ᵥ v) := by
  simp only [dotProduct, Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_full_contraction (0) (![]) (![]) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_full_contraction (2) (![1, 2]) (![3, -1]) (![0, 1]) (![2, 2]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_full_contraction (3) (![1, 0, -1]) (![1, 2, 3]) (![2, 1, 0]) (![0, 1, 1]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Contraction.elementary_full_contraction
-- test: la-full-contraction-test.sage

/-- Contracting over `i = l`, `j = k` gives `β(u) α(v)` instead. -/
theorem elementary_cross_contraction {n : ℕ} (u v α β : Fin n → ℝ) :
    ∑ i, ∑ j, u i * v j * α j * β i = (β ⬝ᵥ u) * (α ⬝ᵥ v) := by
  simp only [dotProduct, Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_cross_contraction (0) (![]) (![]) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_cross_contraction (2) (![1, 2]) (![3, -1]) (![0, 1]) (![2, 2]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.elementary_cross_contraction (3) (![1, 0, -1]) (![1, 2, 3]) (![2, 1, 0]) (![0, 1, 1]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Contraction.elementary_cross_contraction
-- test: la-cross-contraction-test.sage

/-- A type `(2, 2)` tensor has `2 · 2 = 4` single contractions (one of two upper with one of two lower indices). -/
theorem single_contraction_count : Fintype.card (Fin 2 × Fin 2) = 4 := by simp

/-! ## exr-contraction-sage -/

/-- The type `(2, 1)` tensor of the exercise, `S^{11}_1 = 1`, `S^{21}_2 = 2`, `S^{32}_3 = 3`, `S^{13}_1 = 4`. -/
def S (i j k : Fin 3) : ℚ :=
  if (i, j, k) = (0, 0, 0) then 1 else if (i, j, k) = (1, 0, 1) then 2 else if (i, j, k) = (2, 1, 2) then 3
  else if (i, j, k) = (0, 2, 0) then 4 else 0

/-- Contracting `i` with `k` gives the vector `(3, 3, 4)`. -/
theorem S_contraction : (fun j => ∑ i, S i j i) = ![3, 3, 4] := by
  ext j
  fin_cases j <;> simp [S, Fin.sum_univ_three] <;> norm_num

/-- Contracting `i` with `k`, component by component: `S^{i j}_i = (3, 3, 4)_j`. -/
theorem S_contraction_component (j : Fin 3) : ∑ i, S i j i = ![(3 : ℚ), 3, 4] j := by
  fin_cases j <;> simp [S, Fin.sum_univ_three] <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.S_contraction_component (0) := by
  simp [S, Fin.sum_univ_three]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.S_contraction_component (1) := by
  simp [S, Fin.sum_univ_three]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Contraction.S_contraction_component (2) := by
  simp [S, Fin.sum_univ_three]
  <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Contraction.S_contraction_component
-- test: la-s-contraction-test.sage

end KaluzaKlein.LinearAlgebra.Contraction

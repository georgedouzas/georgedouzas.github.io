import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaces
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Linear combinations"

The claims of the section "Linear combinations" of `tutorial/part-i-02-linear-algebra.qmd`: combinations in `ℝ³`, the
generators `L_a` of `𝔰𝔬(3)` with `(L_a)_{bc} = -ε_{abc}`, Fourier modes and the trigonometric identities of the
exercises. Computational cores are Veriphy certificates (`.veriphy/certificates/`).
-/

namespace KaluzaKlein.LinearAlgebra.LinearCombinations

open Matrix

/-! ## def-linear-combination -/

/-- The linear combination `λ^i v_i` of a finite family (mathlib's `Fintype.linearCombination`). -/
noncomputable abbrev linComb {ι V : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V] (v : ι → V) (c : ι → ℝ) : V :=
  Fintype.linearCombination ℝ v c

/-- The running example: `1 · (1, 2, 0) + 1 · (0, 1, 1) = (1, 3, 1)`. -/
theorem combination_r3 : (1 : ℝ) • !![(1 : ℝ), 2, 0] + (1 : ℝ) • !![(0 : ℝ), 1, 1] = !![(1 : ℝ), 3, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- The same combination through the definition. -/
theorem linComb_example : linComb ![(![1, 2, 0] : Fin 3 → ℝ), ![0, 1, 1]] ![1, 1] = ![1, 3, 1] := by
  rw [linComb, Fintype.linearCombination_apply, Fin.sum_univ_two]
  show (1 : ℝ) • ![(1 : ℝ), 2, 0] + (1 : ℝ) • ![(0 : ℝ), 1, 1] = ![1, 3, 1]
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-! ## exm-combination-r3 -/

/-- Every `(x, y, z)` is `x e₁ + y e₂ + z e₃`. -/
theorem standard_combination (x y z : ℝ) :
    x • !![(1 : ℝ), 0, 0] + y • !![(0 : ℝ), 1, 0] + z • !![(0 : ℝ), 0, 1] = !![x, y, z] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exm-combination-matrices -/

/-- The generators of `𝔰𝔬(3)`, `(L_a)_{bc} = -ε_{abc}`. -/
def L1 : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]
def L2 : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 0, 0; -1, 0, 0]
def L3 : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 0]

/-- `a L₁ + b L₂ + c L₃` in components. -/
theorem so3_combination_literal (a b c : ℝ) :
    a • !![(0 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] + b • !![(0 : ℝ), 0, 1; 0, 0, 0; -1, 0, 0] +
      c • !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 0] = !![0, -c, b; c, 0, -a; -b, a, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

theorem so3_combination (a b c : ℝ) : a • L1 + b • L2 + c • L3 = !![0, -c, b; c, 0, -a; -b, a, 0] :=
  so3_combination_literal a b c

/-- The combination is antisymmetric. -/
theorem so3_combination_antisymm (a b c : ℝ) :
    (!![(0 : ℝ), -c, b; c, 0, -a; -b, a, 0])ᵀ = -!![(0 : ℝ), -c, b; c, 0, -a; -b, a, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- Every antisymmetric `3 × 3` matrix is `X₃₂ L₁ + X₁₃ L₂ + X₂₁ L₃`. -/
theorem antisym_eq_combination (X : Matrix (Fin 3) (Fin 3) ℝ) (hX : Xᵀ = -X) :
    X = X 2 1 • L1 + X 0 2 • L2 + X 1 0 • L3 := by
  rw [so3_combination]
  have h : ∀ i j, X j i = -X i j := fun i j => by simpa using congrFun (congrFun hX i) j
  have h00 := h 0 0; have h11 := h 1 1; have h22 := h 2 2
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> linarith [h 0 1, h 0 2, h 1 2]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination (0) ((by simp)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [L1, L2, L3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination (!![0, -5, 2; 5, 0, -7; -2, 7, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [L1, L2, L3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination (!![0, 0, 0; 0, 0, -1; 0, 1, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [L1, L2, L3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination (!![0, 1, -3; -1, 0, 1 / 2; 3, -1 / 2, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [L1, L2, L3]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination
-- test: la-antisym-eq-combination-test.sage

/-! ## exm-combination-fourier -/

/-- `a cos φ + b sin φ` is the linear combination `a • cos + b • sin` of the two functions. -/
theorem fourier_combination (a b : ℝ) :
    (fun φ => a * Real.cos φ + b * Real.sin φ) = a • Real.cos + b • Real.sin := by
  funext φ
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination (0) (0) := by
  funext φ
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination (1) (-5) := by
  funext φ
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination (0) (1) := by
  funext φ
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination (0) (0) := by
  funext φ
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination
-- test: la-fourier-combination-test.sage

/-! ## exr-combination-solve -/

/-- Coefficients exist: `λ¹ v₁ + λ² v₂ + λ⁴ v₄ = (4, 7, 2)` componentwise. -/
theorem combination_solve_exists : ∃ l1 l2 l4 : ℚ,
    l1 * 1 + l2 * 0 + l4 * 0 = 4 ∧ l1 * 2 + l2 * 1 + l4 * 0 = 7 ∧ l1 * 0 + l2 * 1 + l4 * 1 = 2 := by
  exact ⟨4, -1, 3, by norm_num⟩
  -- certificate: la-combination-solve.sage

/-- The coefficients are forced: `λ = (4, -1, 3)`. -/
theorem combination_solve_unique (l1 l2 l4 : ℚ) (h1 : l1 = 4) (h2 : 2 * l1 + l2 = 7) (h3 : l2 + l4 = 2) :
    l1 = 4 ∧ l2 = -1 ∧ l4 = 3 := by
  refine ⟨?_, ?_, ?_⟩
  · linear_combination h1
    -- certificate: ideal_membership.sage
  · linear_combination (-2) * h1 + h2
    -- certificate: ideal_membership.sage
  · linear_combination (2) * h1 + (-1) * h2 + h3
    -- certificate: ideal_membership.sage

/-- `4 v₁ - v₂ + 3 v₄ = (4, 7, 2)`. -/
theorem combination_solve_check :
    (4 : ℚ) • !![(1 : ℚ), 2, 0] + (-1 : ℚ) • !![(0 : ℚ), 1, 1] + (3 : ℚ) • !![(0 : ℚ), 0, 1] = !![(4 : ℚ), 7, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exr-combination-fourier -/

/-- `cos² φ = ½ · 1 + ½ cos 2φ`. -/
theorem cos_sq_combination (x : ℝ) : Real.cos x ^ 2 = 1 / 2 + 1 / 2 * Real.cos (2 * x) := by
  rw [Real.cos_sq]
  ring
  -- certificate: la-trig-cos-sq.sage

/-- `sin³ φ = ¾ sin φ - ¼ sin 3φ`. -/
theorem sin_cube_combination (x : ℝ) : Real.sin x ^ 3 = 3 / 4 * Real.sin x - 1 / 4 * Real.sin (3 * x) := by
  rw [Real.sin_three_mul]
  ring
  -- certificate: la-trig-sin-cube.sage

/-! ## exr-combination-so3 -/

/-- The antisymmetric matrix with `X₁₂ = -5`, `X₁₃ = 2`, `X₂₃ = -7` is `7 L₁ + 2 L₂ + 5 L₃`. -/
theorem combination_so3_literal :
    (7 : ℝ) • !![(0 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] + (2 : ℝ) • !![(0 : ℝ), 0, 1; 0, 0, 0; -1, 0, 0] +
      (5 : ℝ) • !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 0] = !![(0 : ℝ), -5, 2; 5, 0, -7; -2, 7, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-combination-so3.sage

theorem combination_so3 : (7 : ℝ) • L1 + (2 : ℝ) • L2 + (5 : ℝ) • L3 = !![(0 : ℝ), -5, 2; 5, 0, -7; -2, 7, 0] :=
  combination_so3_literal

end KaluzaKlein.LinearAlgebra.LinearCombinations

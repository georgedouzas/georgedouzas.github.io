import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.LinearIndependence
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Bases"

The claims of the section "Bases" of `tutorial/part-i-02-linear-algebra.qmd`: the standard basis, the basis
`f = ((2, 1), (1, 2))` and the coordinates `(2, 1)` of `(5, 4)`, the basis `L₁, L₂, L₃` of `𝔰𝔬(3)`, the basis `b` of
`ℚ³` and the Lagrange basis `1 - x, x`.
-/

namespace KaluzaKlein.LinearAlgebra.Bases

open Matrix Module KaluzaKlein.LinearAlgebra.VectorSpaces KaluzaKlein.LinearAlgebra.LinearCombinations
  KaluzaKlein.LinearAlgebra.LinearIndependence

/-! ## def-basis, exm-basis-f -/

/-- The basis `f₁ = (2, 1)`, `f₂ = (1, 2)` of `ℚ²` (a mathlib `Basis`, from independence and the dimension count). -/
noncomputable def fBasis : Basis (Fin 2) ℚ (Fin 2 → ℚ) :=
  basisOfLinearIndependentOfCardEqFinrank KaluzaKlein.Linear.f_linearIndependent (by simp)

theorem fBasis_apply (i : Fin 2) : fBasis i = ![KaluzaKlein.Linear.f1, KaluzaKlein.Linear.f2] i := by
  simp [fBasis]

/-- The coordinates of `(5, 4)` in the basis `f` are `(2, 1)`. -/
theorem fBasis_coords : fBasis.equivFun ![5, 4] = ![2, 1] := by
  apply fBasis.equivFun.symm.injective
  rw [LinearEquiv.symm_apply_apply, Basis.equivFun_symm_apply, Fin.sum_univ_two, fBasis_apply, fBasis_apply]
  show ![(5 : ℚ), 4] = (![(2 : ℚ), 1] 0) • ![(2 : ℚ), 1] + (![(2 : ℚ), 1] 1) • ![(1 : ℚ), 2]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-! ## exm-basis-standard -/

/-- The coordinates of `x` in the standard basis of `ℝⁿ` are its entries. -/
theorem standard_coords (n : ℕ) (x : Fin n → ℝ) (i : Fin n) : (Pi.basisFun ℝ (Fin n)).repr x i = x i := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Bases.standard_coords (1) (![5]) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Bases.standard_coords (3) (![5, 7, 9]) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.Bases.standard_coords (4) (![1, -2, 1 / 2, 3]) (3) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.Bases.standard_coords
-- test: la-standard-coords-test.sage

/-! ## exm-basis-so3, exr-basis-so3 -/

/-- A vanishing `a L₁ + b L₂ + c L₃` has `a = b = c = 0` (its entries `X₂₃ = -a`, `X₁₃ = b`, `X₁₂ = -c`). -/
theorem so3_coefficients (a b c : ℝ) (h23 : -a = 0) (h13 : b = 0) (h12 : -c = 0) : a = 0 ∧ b = 0 ∧ c = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · linear_combination (-1) * h23
    -- certificate: ideal_membership.sage
  · linear_combination h13
    -- certificate: ideal_membership.sage
  · linear_combination (-1) * h12
    -- certificate: ideal_membership.sage

theorem so3_linearIndependent : LinearIndependent ℝ ![L1, L2, L3] := by
  veriphy_certified
    ( refine
        LinearIndependent.of_comp
          (LinearMap.pi fun k : Fin 3 => Matrix.entryLinearMap ℝ ℝ (![0, 0, 1] k) (![1, 2, 2] k)) ?_
      exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
          (by
            norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
              Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply,
              Matrix.entryLinearMap_apply, Polynomial.lcoeff_apply, Polynomial.coeff_one, Polynomial.coeff_X,
              Polynomial.coeff_sub, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi,
              Real.sin_pi, KaluzaKlein.LinearAlgebra.LinearCombinations.L1,
              KaluzaKlein.LinearAlgebra.LinearCombinations.L2, KaluzaKlein.LinearAlgebra.LinearCombinations.L3]))
  -- certificate: la-independence-eval.sage

/-- The entries above the diagonal of `a L₁ + b L₂ + c L₃` are `(X₁₂, X₁₃, X₂₃) = (-c, b, -a)`. -/
theorem so3_upper_entries (a b c : ℝ) :
    (a • L1 + b • L2 + c • L3) 0 1 = -c ∧ (a • L1 + b • L2 + c • L3) 0 2 = b ∧ (a • L1 + b • L2 + c • L3) 1 2 = -a := by
  have h : a • L1 + b • L2 + c • L3 = !![0, -c, b; c, 0, -a; -b, a, 0] := by
    veriphy_certified
      ((  ext i j
          fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
              KaluzaKlein.LinearAlgebra.LinearCombinations.L1, KaluzaKlein.LinearAlgebra.LinearCombinations.L2,
              KaluzaKlein.LinearAlgebra.LinearCombinations.L3, Matrix.diagonal_apply, Matrix.vecMulVec_apply,
              Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
          all_goals ring1))
    -- certificate: la-matrix-named.sage
  rw [h]
  simp

/-- `L₁, L₂, L₃` is a basis of `𝔰𝔬(3)`: independent and spanning, hence `dim 𝔰𝔬(3) = 3`. -/
theorem so3_basis_card : finrank ℝ (Submodule.span ℝ (Set.range ![L1, L2, L3])) = 3 := by
  have hli : LinearIndependent ℝ ![L1, L2, L3] := by
    veriphy_certified
      ( refine
          LinearIndependent.of_comp
            (LinearMap.pi fun k : Fin 3 => Matrix.entryLinearMap ℝ ℝ (![0, 0, 1] k) (![1, 2, 2] k)) ?_
        exact
          Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
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

/-- The functions `1, cos, sin` form a basis of their span (they are independent): its dimension is `3`. -/
theorem fourier_one_basis_card :
    finrank ℝ (Submodule.span ℝ (Set.range ![(fun _ => (1 : ℝ)), Real.cos, Real.sin])) = 3 := by
  have hli : LinearIndependent ℝ ![(fun _ => (1 : ℝ)), Real.cos, Real.sin] := by
    veriphy_certified
      ( refine
          LinearIndependent.of_comp
            (LinearMap.pi fun k : Fin 3 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2, Real.pi] k))
            ?_
        exact
          Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
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

/-! ## exr-basis-coordinates -/

theorem det_basis_b : Matrix.det !![(1 : ℚ), 0, 1; 1, 1, 0; 0, 1, 1] = 2 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-basis-coordinates-check.sage

/-- `b₁, b₂, b₃` are independent (the matrix with these rows is the transpose of the one with these columns). -/
theorem basis_b_independent : LinearIndependent ℚ ![(![1, 1, 0] : Fin 3 → ℚ), ![0, 1, 1], ![1, 0, 1]] := by
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-- Coordinates of `(2, 3, 4)` in the basis `b` exist. -/
theorem basis_b_coords_exist : ∃ l1 l2 l3 : ℚ,
    l1 * 1 + l2 * 0 + l3 * 1 = 2 ∧ l1 * 1 + l2 * 1 + l3 * 0 = 3 ∧ l1 * 0 + l2 * 1 + l3 * 1 = 4 := by
  exact ⟨1 / 2, 5 / 2, 3 / 2, by norm_num⟩
  -- certificate: la-basis-coordinates.sage

/-- `½ b₁ + 5/2 b₂ + 3/2 b₃ = (2, 3, 4)`. -/
theorem basis_b_coords :
    !![(1 : ℚ), 0, 1; 1, 1, 0; 0, 1, 1] * !![(1 / 2 : ℚ); 5 / 2; 3 / 2] = !![(2 : ℚ); 3; 4] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-basis-coordinates-check.sage

/-! ## exr-basis-lagrange -/

/-- A vanishing `a (1 - x) + b x = a + (b - a) x` has `a = 0` and `b = 0`. -/
theorem lagrange_coefficients (a b : ℝ) (h0 : a = 0) (h1 : b - a = 0) : a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination h0
    -- certificate: ideal_membership.sage
  · linear_combination h0 + h1
    -- certificate: ideal_membership.sage

open Polynomial in
theorem lagrange_linearIndependent : LinearIndependent ℝ ![(1 - X : ℝ[X]), X] := by
  veriphy_certified
    ( refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 2 => Polynomial.lcoeff ℝ (![0, 1] k)) ?_
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

/-- The coordinates of `3x + 2`: the constant and linear coefficients give `a = 2`, `-a + b = 3`. -/
theorem lagrange_coords_exist : ∃ a b : ℝ, a = 2 ∧ -a + b = 3 := by
  exact ⟨2, 5, by norm_num⟩
  -- certificate: la-basis-lagrange.sage

open Polynomial in
/-- `3x + 2 = 2 (1 - x) + 5 x`, and `c + d x = c (1 - x) + (c + d) x` (spanning). -/
theorem lagrange_coords (c d : ℝ) :
    (3 * X + 2 : ℝ[X]) = 2 * (1 - X) + 5 * X ∧ (C c + C d * X : ℝ[X]) = C c * (1 - X) + C (c + d) * X := by
  constructor
  · veriphy_certified (linear_combination)
    -- certificate: la-polynomial.sage
  · veriphy_certified ((simp only [map_add, map_sub, map_mul, map_neg]; ring))
    -- certificate: la-polynomial.sage

open Polynomial in
/-- The coordinates are the values at `0` and `1`: `p(0) = 2`, `p(1) = 5` for `p = 3x + 2`. -/
theorem lagrange_values : (3 * X + 2 : ℝ[X]).eval 0 = 2 ∧ (3 * X + 2 : ℝ[X]).eval 1 = 5 := by
  constructor
  · veriphy_certified ((norm_num; done))
    -- certificate: la-polynomial.sage
  · veriphy_certified ((norm_num; done))
    -- certificate: la-polynomial.sage

end KaluzaKlein.LinearAlgebra.Bases

import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.CosetModes

/-! # Fourier analysis: harmonic computations (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.HarmonicComputations

open Real MeasureTheory Filter Topology Matrix

/-! ## prp-spherical-harmonics, exm-sph-sage-dimension -/

/-- The Laplacian on homogeneous quadratics, from the monomials `x², y², z², xy, xz, yz` to the constants, is
`(2, 2, 2, 0, 0, 0)`, of rank `1`: the harmonic quadratics form a space of dimension `6 - 1 = 5`. -/
theorem harmonic_quadratics_rank : Matrix.rank !![(2 : ℚ), 2, 2, 0, 0, 0] = 1 := by
  veriphy_certified
    ( apply le_antisymm
      · exact Matrix.rank_le_height _
      ·
        exact
          Veriphy.Linear.le_rank_of_minor _ ![0] ![0]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
  -- certificate: fa-rank.sage

/-- The Laplacian from the cubic monomials `x³, y³, z³, x²y, x²z, xy², y²z, xz², yz², xyz` to `x, y, z` has rank `3`:
the harmonic cubics form a space of dimension `10 - 3 = 7`. -/
theorem harmonic_cubics_rank :
    Matrix.rank !![(6 : ℚ), 0, 0, 0, 0, 2, 0, 2, 0, 0; 0, 6, 0, 2, 0, 0, 0, 0, 2, 0; 0, 0, 6, 0, 2, 0, 2, 0, 0, 0] = 3 := by
  veriphy_certified
    ( apply le_antisymm
      · exact Matrix.rank_le_height _
      ·
        exact
          Veriphy.Linear.le_rank_of_minor _ ![0, 1, 2] ![0, 1, 2]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
  -- certificate: fa-rank.sage

/-- The dimension count `C(l+2, 2) - C(l, 2) = 2l + 1`. -/
theorem harmonic_dimension (l : ℕ) : Nat.choose (l + 2) 2 - Nat.choose l 2 = 2 * l + 1 := by
  induction l with
  | zero => simp
  | succ l ih =>
    have h1 : Nat.choose (l + 1 + 2) 2 = (l + 2) + Nat.choose (l + 2) 2 := by
      rw [show l + 1 + 2 = (l + 2) + 1 by ring, Nat.choose_succ_succ', Nat.choose_one_right]
    have h2 : Nat.choose (l + 1) 2 = l + Nat.choose l 2 := by
      rw [Nat.choose_succ_succ', Nat.choose_one_right]
    have h3 : Nat.choose l 2 ≤ Nat.choose (l + 2) 2 := Nat.choose_le_choose 2 (by omega)
    omega

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.harmonic_dimension (0) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.harmonic_dimension (4) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.harmonic_dimension (1) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.HarmonicComputations.harmonic_dimension
-- test: fa-harmonic-dimension-test.sage

/-- The harmonic homogeneous polynomials of degree `l ≤ 6` have dimension `2l + 1`: the Laplacian matrix on the
monomials has full rank `C(l, 2)`, as stated (tested, not proved for `l ≥ 4`: no recipe certifies ranks of
`15 × 28` matrices). -/
def HarmonicDimension : Prop :=
  ∀ l : ℕ, l ≤ 6 → Nat.choose (l + 2) 2 - Nat.choose l 2 = 2 * l + 1 ∧ Nat.choose l 2 ≤ Nat.choose (l + 2) 2

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.HarmonicComputations.HarmonicDimension
-- test: fa-harmonic-dimension-prop-test.sage

/-! ## exm-sph-sage-laplacian, exr-sph-ladder -/

/-- The ladder relation for `l = 1`: `L₊ Y₁₀ = e^{iφ} ∂_θ (√(3/4π) cos θ) = -√(3/4π) sin θ e^{iφ} = √2 Y₁₁`,
with `Y₁₁ = -√(3/8π) sin θ e^{iφ}`. -/
theorem ladder_l1 (θ : ℝ) :
    deriv (fun t : ℝ => Real.sqrt (3 / (4 * π)) * Real.cos t) θ = Real.sqrt 2 * (-(Real.sqrt (3 / (8 * π)) * Real.sin θ)) := by
  have h : Real.sqrt 2 * Real.sqrt (3 / (8 * π)) = Real.sqrt (3 / (4 * π)) := by
    rw [← Real.sqrt_mul (by norm_num)]; congr 1; field_simp; ring
  rw [((Real.hasDerivAt_cos θ).const_mul (Real.sqrt (3 / (4 * π)))).deriv, ← h]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_l1
-- test: fa-ladder-l1-test.sage

/-- The ladder coefficient `‖L₊ Y_{lm}‖² = l(l+1) - m² - m = (l - m)(l + m + 1)`. -/
theorem ladder_coefficient (l m : ℤ) : l * (l + 1) - m ^ 2 - m = (l - m) * (l + m + 1) := by ring

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_coefficient (0) (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_coefficient (-1) (-4) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_coefficient (0) (1) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_coefficient (0) (-2) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.FourierAnalysis.HarmonicComputations.ladder_coefficient
-- test: fa-ladder-coefficient-test.sage

/-! ## exr-sph-l3-check -/

/-- `r³ Y₃₃ ∝ (x + iy)³`: `(sin θ cos φ + i sin θ sin φ)³ = sin³ θ e^{3iφ}`. -/
theorem l3_top (θ φ : ℝ) :
    ((Real.sin θ * Real.cos φ : ℝ) + (Real.sin θ * Real.sin φ : ℝ) * Complex.I) ^ 3 =
      (Real.sin θ : ℂ) ^ 3 * Complex.exp (3 * φ * Complex.I) := by
  rw [show (3 : ℂ) * φ * Complex.I = ((3 : ℕ) : ℂ) * ((φ : ℂ) * Complex.I) by push_cast; ring, Complex.exp_nat_mul,
    Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.HarmonicComputations.l3_top
-- test: fa-l3-top-test.sage

end KaluzaKlein.FourierAnalysis.HarmonicComputations

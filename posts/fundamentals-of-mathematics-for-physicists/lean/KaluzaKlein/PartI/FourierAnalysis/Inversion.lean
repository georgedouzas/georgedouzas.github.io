import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.TheTransform

/-! # Fourier analysis: inversion (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Inversion

open Real MeasureTheory Filter Topology Complex
open scoped FourierTransform

/-! ## thm-fourier-inversion -/

/-- Fourier inversion (mathlib, in the normalisation `𝓕 f(w) = ∫ e^{-2πixw} f`): `𝓕⁻ (𝓕 f) = f` for continuous
integrable `f` with integrable transform. -/
theorem inversion (f : ℝ → ℂ) (hc : Continuous f) (hf : Integrable f) (hf' : Integrable (𝓕 f)) :
    𝓕⁻ (𝓕 f) = f :=
  hc.fourierInv_fourier_eq hf hf'

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.inversion
-- test: fa-inversion-test.sage

/-- Duality on the Gaussian: transforming `√(2π) e^{-k²/2}` again gives `2π e^{-x²/2} = 2π g(-x)`. -/
theorem gaussian_duality (x : ℝ) :
    ∫ k : ℝ, (Real.sqrt (2 * π) * Complex.exp (-(k : ℂ) ^ 2 / 2)) * Complex.exp (-(x * k) * Complex.I) =
      2 * π * Complex.exp (-((-x : ℝ) : ℂ) ^ 2 / 2) := by
  have := KaluzaKlein.FourierAnalysis.TheTransform.gaussian_transform 1 one_pos x
  have hpt : ∀ k : ℝ, (Real.sqrt (2 * π) * Complex.exp (-(k : ℂ) ^ 2 / 2)) * Complex.exp (-(x * k) * Complex.I) =
      (Real.sqrt (2 * π) : ℂ) * (Complex.exp (-(k : ℂ) ^ 2 / (2 * ((1 : ℝ) : ℂ) ^ 2)) *
        Complex.exp (-((x : ℂ) * k) * Complex.I)) := by
    intro k
    rw [show (-(k : ℂ) ^ 2 / (2 * ((1 : ℝ) : ℂ) ^ 2)) = -(k : ℂ) ^ 2 / 2 by push_cast; ring]
    ring
  simp_rw [hpt]
  rw [integral_const_mul, this]
  have hE : Complex.exp (-((1 : ℝ) : ℂ) ^ 2 * (x : ℂ) ^ 2 / 2) = Complex.exp (-((-x : ℝ) : ℂ) ^ 2 / 2) := by
    congr 1; push_cast; ring
  rw [hE]
  have h2 : (Real.sqrt (2 * π) : ℂ) * Real.sqrt (2 * π) = 2 * π := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]; push_cast; ring
  simp only [Complex.ofReal_one, one_mul]
  linear_combination (Complex.exp (-((-x : ℝ) : ℂ) ^ 2 / 2)) * h2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.gaussian_duality
-- test: fa-gaussian-duality-test.sage

/-! ## exm-ft-inversion-lorentzian, exr-ft-lorentzian-inverse -/

/-- The residue at `k = ia`, times `2πi / 2π`, is `e^{-ax}` (closing above for `x > 0`): `i · 2a e^{i(ia)x} = 2(ia) e^{-ax}`,
the residue identity multiplied through by the denominator `2ia`. -/
theorem lorentz_residue_upper (a x : ℝ) :
    Complex.I * ((2 * a) * Complex.exp (Complex.I * (Complex.I * a) * x)) =
      (2 * (Complex.I * a)) * Complex.exp (-(a * x)) := by
  rw [show Complex.I * (Complex.I * a) * x = -(a * x) by linear_combination ((a : ℂ) * x) * Complex.I_sq]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.lorentz_residue_upper
-- test: fa-lorentz-residue-upper-test.sage

/-- The residue at `k = -ia`, with the orientation reversed, is `e^{ax}` (closing below for `x < 0`), multiplied through by
the denominator `-2ia`. -/
theorem lorentz_residue_lower (a x : ℝ) :
    -(Complex.I * ((2 * a) * Complex.exp (Complex.I * (-(Complex.I * a)) * x))) =
      (2 * (-(Complex.I * a))) * Complex.exp (a * x) := by
  rw [show Complex.I * (-(Complex.I * a)) * x = a * x by linear_combination (-((a : ℂ) * x)) * Complex.I_sq]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.lorentz_residue_lower
-- test: fa-lorentz-residue-lower-test.sage

/-! ## exm-ft-inversion-box -/

/-- The Dirichlet integral `∫ sin(ck)/k dk = π sgn c`, as stated (tested, not proved: not in mathlib). -/
def DirichletIntegral : Prop :=
  ∀ c : ℝ, Tendsto (fun L => ∫ k in (-L)..L, Real.sin (c * k) / k) atTop (𝓝 (π * Real.sign c))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.DirichletIntegral
-- test: fa-dirichlet-integral-test.sage

/-- The product-to-sum step `sin(ak) cos(kx) = (sin((a+x)k) + sin((a-x)k))/2`. -/
theorem sin_cos_split (a x k : ℝ) :
    Real.sin (a * k) * Real.cos (x * k) = (Real.sin (a * k + x * k) + Real.sin (a * k - x * k)) / 2 := by
  veriphy_certified
    ( rw [Real.sin_add]
      rw [Real.sin_sub]
      ring)
  -- certificate: fa-trig-addition.sage

/-- The midpoint rule at the jumps: `(sgn(a + x) + sgn(a - x))/2` is `1` inside the box, `1/2` at its ends, `0`
outside. -/
theorem box_midpoint (a x : ℝ) (ha : 0 < a) :
    (Real.sign (a + x) + Real.sign (a - x)) / 2 = if |x| < a then 1 else if |x| = a then 1 / 2 else 0 := by
  split_ifs with h1 h2
  · rw [abs_lt] at h1
    rw [Real.sign_of_pos (by linarith), Real.sign_of_pos (by linarith)]; norm_num
  · rcases abs_eq (by linarith : (0 : ℝ) ≤ a) |>.1 h2 with h | h
    · subst h; rw [sub_self, Real.sign_zero, Real.sign_of_pos (by linarith)]; norm_num
    · rw [h, add_neg_cancel, Real.sign_zero, Real.sign_of_pos (by linarith)]; norm_num
  · push_neg at h1
    rcases le_abs'.1 h1 with h | h
    · have : x < -a := lt_of_le_of_ne h (fun e => h2 (by rw [e, abs_neg, abs_of_pos ha]))
      rw [Real.sign_of_neg (by linarith), Real.sign_of_pos (by linarith)]; norm_num
    · have : a < x := lt_of_le_of_ne h (fun e => h2 (by rw [← e, abs_of_pos ha]))
      rw [Real.sign_of_pos (by linarith), Real.sign_of_neg (by linarith)]; norm_num

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.Inversion.box_midpoint
-- test: fa-box-midpoint-test.sage

/-! ## exm-ft-duality -/

/-- The Gaussian pair at width `1/σ`: `(σ √(2π) e^{-σ²x²/2})^ = 2π e^{-k²/2σ²}`. -/
theorem gaussian_dual_pair (σ : ℝ) (hσ : 0 < σ) (k : ℝ) :
    ∫ x : ℝ, (σ * Real.sqrt (2 * π) * Complex.exp (-(σ : ℂ) ^ 2 * (x : ℂ) ^ 2 / 2)) * Complex.exp (-(k * x) * Complex.I) =
      2 * π * Complex.exp ((-k ^ 2 / (2 * σ ^ 2) : ℝ) : ℂ) := by
  have hτ : 0 < 1 / σ := by positivity
  have := KaluzaKlein.FourierAnalysis.TheTransform.gaussian_transform (1 / σ) hτ k
  have hσ' : (σ : ℂ) ≠ 0 := by exact_mod_cast hσ.ne'
  have hpt : ∀ x : ℝ, (σ * Real.sqrt (2 * π) * Complex.exp (-(σ : ℂ) ^ 2 * (x : ℂ) ^ 2 / 2)) *
      Complex.exp (-(k * x) * Complex.I) = ((σ * Real.sqrt (2 * π) : ℝ) : ℂ) *
      (Complex.exp (-(x : ℂ) ^ 2 / (2 * (((1 / σ : ℝ)) : ℂ) ^ 2)) * Complex.exp (-((k : ℂ) * x) * Complex.I)) := by
    intro x
    have e : -(σ : ℂ) ^ 2 * (x : ℂ) ^ 2 / 2 = -(x : ℂ) ^ 2 / (2 * (((1 / σ : ℝ)) : ℂ) ^ 2) := by
      push_cast; field_simp
    rw [e]; push_cast; ring
  simp_rw [hpt]
  rw [integral_const_mul, this]
  have hE : -(((1 / σ : ℝ)) : ℂ) ^ 2 * (k : ℂ) ^ 2 / 2 = ((-k ^ 2 / (2 * σ ^ 2) : ℝ) : ℂ) := by
    push_cast; field_simp
  rw [hE]
  have hs : σ * Real.sqrt (2 * π) * (1 / σ * Real.sqrt (2 * π)) = 2 * π := by
    rw [show σ * Real.sqrt (2 * π) * (1 / σ * Real.sqrt (2 * π)) = (σ * (1 / σ)) * (Real.sqrt (2 * π) * Real.sqrt (2 * π)) by ring,
      mul_one_div_cancel hσ.ne', Real.mul_self_sqrt (by positivity), one_mul]
  have hsC : ((σ * Real.sqrt (2 * π) : ℝ) : ℂ) * (((1 / σ : ℝ) : ℂ) * (Real.sqrt (2 * π) : ℂ)) = 2 * π := by
    exact_mod_cast hs
  linear_combination (Complex.exp ((-k ^ 2 / (2 * σ ^ 2) : ℝ) : ℂ)) * hsC

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.gaussian_dual_pair
-- test: fa-gaussian-dual-pair-test.sage

/-! ## exr-ft-inversion-gaussian -/

/-- The regularising Gaussians have unit integral: `∫ e^{-x²/2ε} / √(2πε) = 1`. -/
theorem heat_gauss_unit (ε : ℝ) (hε : 0 < ε) :
    ∫ x : ℝ, Real.exp (-x ^ 2 / (2 * ε)) / Real.sqrt (2 * π * ε) = 1 := by
  rw [integral_div]
  have := integral_gaussian (1 / (2 * ε))
  rw [show (fun x : ℝ => Real.exp (-x ^ 2 / (2 * ε))) = fun x : ℝ => Real.exp (-(1 / (2 * ε)) * x ^ 2) by
    funext x; ring_nf, this]
  rw [show π / (1 / (2 * ε)) = 2 * π * ε by field_simp]
  have : 0 < Real.sqrt (2 * π * ε) := Real.sqrt_pos.2 (by positivity)
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.heat_gauss_unit
-- test: fa-heat-gauss-unit-test.sage

/-- The Gaussian approximate identity (mathlib): `∫ (πc)^{1/2} e^{-π²c|w - v|²} f(w) dw → f(v)` as `c → ∞` at a
point of continuity of an integrable `f`. -/
theorem gaussian_approx_identity (f : ℝ → ℂ) (hf : Integrable f) (v : ℝ) (hv : ContinuousAt f v) :
    Tendsto (fun c : ℝ => ∫ w : ℝ, ((π * c : ℂ) ^ (Module.finrank ℝ ℝ / 2 : ℂ) *
      Complex.exp (-π ^ 2 * c * ‖v - w‖ ^ 2)) • f w) atTop (𝓝 (f v)) :=
  Real.tendsto_integral_gaussian_smul' hf hv

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.gaussian_approx_identity
-- test: fa-gaussian-approx-identity-test.sage

/-- The tails of the regularising Gaussians vanish as `ε → 0`, as stated (tested, not proved): `∫_{|y| > δ} G_ε → 0`. -/
def GaussianTail : Prop :=
  ∀ δ : ℝ, 0 < δ → Tendsto (fun ε : ℝ => ∫ y in {y : ℝ | δ < |y|}, Real.exp (-y ^ 2 / (2 * ε)) / Real.sqrt (2 * π * ε))
    (𝓝[>] 0) (𝓝 0)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.GaussianTail
-- test: fa-gaussian-tail-test.sage

/-! ## exr-ft-hermite-eigen -/

/-- An eigenvalue `λ` of a linear map `F` with `F(F g)(x) = 2π g(-x)` satisfies `λ⁴ = (2π)²`. -/
theorem eigenvalue_fourth_power (F : (ℝ → ℂ) →ₗ[ℂ] (ℝ → ℂ)) (hdual : ∀ g x, F (F g) x = 2 * π * g (-x))
    (f : ℝ → ℂ) (lam : ℂ) (hf : F f = lam • f) (hne : f ≠ 0) : lam ^ 4 = (2 * π) ^ 2 := by
  have h2 : ∀ x, lam ^ 2 * f x = 2 * π * f (-x) := by
    intro x
    have := hdual f x
    rw [hf, map_smul, hf] at this
    simp only [Pi.smul_apply, smul_eq_mul] at this
    rw [← this]; ring
  obtain ⟨x, hx⟩ : ∃ x, f x ≠ 0 := by
    by_contra h; push_neg at h; exact hne (funext h)
  have h4 : lam ^ 4 * f x = (2 * π) ^ 2 * f x := by
    have a1 := h2 x
    have a2 := h2 (-x)
    rw [neg_neg] at a2
    calc lam ^ 4 * f x = lam ^ 2 * (lam ^ 2 * f x) := by ring
      _ = lam ^ 2 * (2 * π * f (-x)) := by rw [a1]
      _ = 2 * π * (lam ^ 2 * f (-x)) := by ring
      _ = 2 * π * (2 * π * f x) := by rw [a2]
      _ = (2 * π) ^ 2 * f x := by ring
  exact mul_right_cancel₀ hx h4

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.eigenvalue_fourth_power
-- test: fa-eigenvalue-fourth-power-test.sage

/-- The third Hermite function `(2x² - 1) e^{-x²/2}` is an eigenfunction with eigenvalue `-√(2π)`, as stated
(tested, not proved). -/
def HermiteTwoEigen : Prop :=
  ∀ k : ℝ, ∫ x : ℝ, ((2 * x ^ 2 - 1) * Real.exp (-x ^ 2 / 2) : ℝ) * Complex.exp (-(k * x) * Complex.I) =
    -(Real.sqrt (2 * π)) * ((2 * k ^ 2 - 1) * Real.exp (-k ^ 2 / 2) : ℝ)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Inversion.HermiteTwoEigen
-- test: fa-hermite-two-eigen-test.sage

end KaluzaKlein.FourierAnalysis.Inversion

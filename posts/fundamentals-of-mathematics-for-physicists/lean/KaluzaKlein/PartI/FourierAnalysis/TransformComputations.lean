import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Uncertainty

/-! # Fourier analysis: transform computations (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.TransformComputations

open Real MeasureTheory Filter Topology Complex

/-! ## exm-ft-sage-assume -/

/-- For `a ≤ 0` the two-sided exponential is not integrable: the transform integral diverges. -/
theorem exp_not_integrable (a : ℝ) (ha : a ≤ 0) : ¬ Integrable (fun x : ℝ => Real.exp (-a * |x|)) := by
  intro h
  have h1 : Integrable (fun _ : ℝ => (1 : ℝ)) := by
    refine h.mono' aestronglyMeasurable_const (Eventually.of_forall fun x => ?_)
    rw [Real.norm_eq_abs, abs_one]
    exact Real.one_le_exp (by nlinarith [abs_nonneg x])
  rw [integrable_const_iff] at h1
  rcases h1 with h1 | h1
  · exact one_ne_zero h1
  · have := h1.measure_univ_lt_top
    simp [Real.volume_univ] at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.exp_not_integrable
-- test: fa-exp-not-integrable-test.sage

/-! ## exr-ft-damped-cosine -/

/-- Each Lorentzian `a/(a² + (k - b)²)` drops to half its maximum `1/a` at `|k - b| = a`: the full width at half
maximum is `2a`. -/
theorem lorentzian_half_width (a b : ℝ) (ha : a ≠ 0) :
    a / (a ^ 2 + ((b + a) - b) ^ 2) = (1 / a) / 2 ∧ a / (a ^ 2 + ((b - a) - b) ^ 2) = (1 / a) / 2 := by
  constructor <;> field_simp <;> ring

example : veriphy_instance% KaluzaKlein.FourierAnalysis.TransformComputations.lorentzian_half_width (1) (0) (one_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.TransformComputations.lorentzian_half_width (2) (3) (two_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.TransformComputations.lorentzian_half_width (-1/2) (5) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.TransformComputations.lorentzian_half_width
-- test: fa-lorentzian-half-width-test.sage

/-! ## exr-ft-sech -/

/-- The poles of `sech`: `sinh(iπ(n + 1/2)) = i (-1)^n`, so the residues `1/sinh z_n` are `(-1)^n / i`. -/
theorem sinh_poles (n : ℕ) : Complex.sinh (Complex.I * π * (n + 1 / 2)) = Complex.I * (-1) ^ n := by
  rw [show Complex.I * π * (n + 1 / 2) = ((π * (n + 1 / 2) : ℝ) : ℂ) * Complex.I by push_cast; ring,
    Complex.sinh_mul_I, ← Complex.ofReal_sin]
  have : Real.sin (π * (n + 1 / 2)) = (-1) ^ n := by
    rw [show π * ((n : ℝ) + 1 / 2) = π / 2 + n * π by ring, Real.sin_add, Real.sin_pi_div_two,
      Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.cos_nat_mul_pi]; ring
  rw [this]; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.sinh_poles
-- test: fa-sinh-poles-test.sage

/-- The geometric series of residues sums to `π sech(πk/2)`: `2π e^{πk/2}/(1 + e^{πk}) = π / cosh(πk/2)`. -/
theorem sech_geometric (k : ℝ) : 2 * π * Real.exp (π * k / 2) / (1 + Real.exp (π * k)) = π / Real.cosh (π * k / 2) := by
  rw [Real.cosh_eq]
  have h1 : 0 < Real.exp (π * k / 2) := Real.exp_pos _
  have h2 : Real.exp (π * k) = Real.exp (π * k / 2) ^ 2 := by rw [← Real.exp_nat_mul]; congr 1; push_cast; ring
  have h3 : Real.exp (-(π * k / 2)) = (Real.exp (π * k / 2))⁻¹ := Real.exp_neg _
  rw [h2, h3]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.sech_geometric
-- test: fa-sech-geometric-test.sage

/-- `F̂(sech) = π sech(πk/2)`, as stated (tested, not proved: the residue sum over infinitely many poles is not in
mathlib). -/
def SechTransform : Prop :=
  ∀ k : ℝ, ∫ x : ℝ, (1 / Real.cosh x : ℝ) * Complex.exp (-(k * x) * Complex.I) = ((π / Real.cosh (π * k / 2) : ℝ) : ℂ)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.SechTransform
-- test: fa-sech-transform-test.sage

/-! ## exr-ft-radial -/

/-- The angular integral `∫₀^π cos(q cos θ) sin θ dθ = 2 sin q / q`, the radial reduction of the 3D transform. -/
theorem angular_integral (q : ℝ) (hq : q ≠ 0) :
    ∫ θ in (0 : ℝ)..π, Real.cos (q * Real.cos θ) * Real.sin θ = 2 * Real.sin q / q := by
  have hF : ∀ θ ∈ Set.uIcc 0 π, HasDerivAt (fun θ => -Real.sin (q * Real.cos θ) / q)
      (Real.cos (q * Real.cos θ) * Real.sin θ) θ := by
    intro θ _
    have := (((Real.hasDerivAt_cos θ).const_mul q).sin.neg).div_const q
    convert this using 1
    field_simp
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [Real.cos_pi, Real.cos_zero, mul_neg, mul_one, Real.sin_neg]
  field_simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.angular_integral
-- test: fa-angular-integral-test.sage

/-- The Yukawa function: `(1/k) ∫₀^∞ e^{-mr} sin kr dr = 1/(k² + m²)`. -/
theorem yukawa_radial (m k : ℝ) (hm : 0 < m) (hk : k ≠ 0) :
    1 / k * ∫ r in Set.Ioi (0 : ℝ), Real.exp (-m * r) * Real.sin (k * r) = 1 / (k ^ 2 + m ^ 2) := by
  have hre : (-(m : ℂ) + k * Complex.I).re < 0 := by simp; linarith
  have hc := integral_exp_mul_complex_Ioi hre 0
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero] at hc
  have hint := integrableOn_exp_mul_complex_Ioi hre 0
  have him : ∫ r in Set.Ioi (0 : ℝ), Real.exp (-m * r) * Real.sin (k * r) =
      (∫ r in Set.Ioi (0 : ℝ), Complex.exp ((-(m : ℂ) + k * Complex.I) * r)).im := by
    have hcomm := Complex.imCLM.integral_comp_comm hint
    simp only [Complex.imCLM_apply] at hcomm
    rw [← hcomm]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro r _
    simp only [Complex.exp_im]
    congr 2 <;> simp
  rw [him, hc]
  simp only [Complex.div_im, Complex.normSq_apply]
  simp
  have hden : m ^ 2 + k ^ 2 ≠ 0 := by positivity
  first | (rw [mul_inv_cancel₀ hk, one_mul]) | (field_simp; done) | (field_simp; ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TransformComputations.yukawa_radial
-- test: fa-yukawa-radial-test.sage

end KaluzaKlein.FourierAnalysis.TransformComputations

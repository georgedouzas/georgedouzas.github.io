import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.DistributionDerivatives

/-! # Fourier analysis: principal value (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.PrincipalValue

open Real MeasureTheory Filter Topology

/-! ## def-principal-value -/

/-- The symmetric form: `∫_{-c}^{-ε} φ/x + ∫_ε^c φ/x = ∫_ε^c (φ(x) - φ(-x))/x`. -/
theorem pv_symmetric (φ : ℝ → ℝ) (ε c : ℝ) (h1 : IntervalIntegrable (fun x => φ x / x) volume ε c)
    (h2 : IntervalIntegrable (fun x => φ (-x) / x) volume ε c) :
    (∫ x in (-c)..(-ε), φ x / x) + ∫ x in ε..c, φ x / x = ∫ x in ε..c, (φ x - φ (-x)) / x := by
  have : ∫ x in (-c)..(-ε), φ x / x = ∫ x in ε..c, -(φ (-x) / x) := by
    rw [← intervalIntegral.integral_comp_neg]
    apply intervalIntegral.integral_congr; intro x _; simp only [div_neg]
  rw [this, intervalIntegral.integral_neg, ← sub_eq_neg_add, ← intervalIntegral.integral_sub h1 h2]
  congr 1; funext x; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_symmetric
-- test: fa-pv-symmetric-test.sage

/-- The integrand of the symmetric form is bounded: `|φ(x) - φ(-x)| ≤ 2 M x` when `|φ'| ≤ M`. -/
theorem pv_integrand_bound (φ : ℝ → ℝ) (M : ℝ) (hφ : Differentiable ℝ φ) (hM : ∀ x, |deriv φ x| ≤ M) (x : ℝ)
    (hx : 0 < x) : |φ x - φ (-x)| ≤ 2 * M * x := by
  have := (convex_Icc (-x) x).norm_image_sub_le_of_norm_deriv_le (f := φ) (fun y _ => hφ y)
    (fun y _ => by rw [Real.norm_eq_abs]; exact hM y) (Set.left_mem_Icc.2 (by linarith)) (Set.right_mem_Icc.2 (by linarith))
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (by linarith : (0 : ℝ) < x - -x)] at this
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_integrand_bound
-- test: fa-pv-integrand-bound-test.sage

/-- Sokhotski-Plemelj: `1/(x ∓ iε) = x/(x² + ε²) ± iε/(x² + ε²)` (stated multiplied through by `x ∓ iε`). -/
theorem sokhotski_split (x ε : ℝ) (hε : 0 < ε) :
    ((x : ℂ) - ε * Complex.I) * (((x / (x ^ 2 + ε ^ 2) : ℝ) : ℂ) + Complex.I * ((ε / (x ^ 2 + ε ^ 2) : ℝ) : ℂ)) = 1 ∧
    ((x : ℂ) + ε * Complex.I) * (((x / (x ^ 2 + ε ^ 2) : ℝ) : ℂ) - Complex.I * ((ε / (x ^ 2 + ε ^ 2) : ℝ) : ℂ)) = 1 := by
  have hd : (x : ℝ) ^ 2 + ε ^ 2 ≠ 0 := by positivity
  have hd' : ((x : ℂ) ^ 2 + (ε : ℂ) ^ 2) ≠ 0 := by exact_mod_cast hd
  constructor
  · linear_combination (norm := skip) (-(ε : ℂ) * ((ε / (x ^ 2 + ε ^ 2) : ℝ) : ℂ)) * Complex.I_sq
    push_cast
    field_simp
    ring
  · linear_combination (norm := skip) (-(ε : ℂ) * ((ε / (x ^ 2 + ε ^ 2) : ℝ) : ℂ)) * Complex.I_sq
    push_cast
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.sokhotski_split
-- test: fa-sokhotski-split-test.sage

/-! ## exm-dist-pv-rational, exm-dist-sage-pv, exr-dist-pv -/

/-- `∫_{-1}^{-ε} dx/x + ∫_ε^2 dx/x = log 2` for every `0 < ε < 1`: the divergent logarithms cancel. -/
theorem pv_log_two (ε : ℝ) (h0 : 0 < ε) (h1 : ε < 1) :
    (∫ x in (-1 : ℝ)..(-ε), x⁻¹) + ∫ x in ε..2, x⁻¹ = Real.log 2 := by
  rw [integral_inv_of_neg (by norm_num) (by linarith), integral_inv_of_pos h0 (by norm_num)]
  rw [show -ε / -1 = ε by ring, Real.log_div (by norm_num) h0.ne']
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_log_two
-- test: fa-pv-log-two-test.sage

/-- Removing `(-ε, 2ε)` instead gives `0`: the value depends on the prescription. -/
theorem pv_asymmetric (ε : ℝ) (h0 : 0 < ε) (h1 : ε < 1) :
    (∫ x in (-1 : ℝ)..(-ε), x⁻¹) + ∫ x in (2 * ε)..2, x⁻¹ = 0 := by
  rw [integral_inv_of_neg (by norm_num) (by linarith), integral_inv_of_pos (by linarith) (by norm_num)]
  rw [show -ε / -1 = ε by ring, show (2 : ℝ) / (2 * ε) = ε⁻¹ by field_simp, Real.log_inv]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_asymmetric
-- test: fa-pv-asymmetric-test.sage

/-- `PV ∫_{-1}^{3} dx/x = log 3`. -/
theorem pv_log_three (ε : ℝ) (h0 : 0 < ε) (h1 : ε < 1) :
    (∫ x in (-1 : ℝ)..(-ε), x⁻¹) + ∫ x in ε..3, x⁻¹ = Real.log 3 := by
  rw [integral_inv_of_neg (by norm_num) (by linarith), integral_inv_of_pos h0 (by norm_num)]
  rw [show -ε / -1 = ε by ring, Real.log_div (by norm_num) h0.ne']
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_log_three
-- test: fa-pv-log-three-test.sage

/-- `PV ∫ dx/(x(x² + 1)) = 0`: the symmetric truncations cancel by oddness. -/
theorem pv_odd (ε R : ℝ) :
    (∫ x in (-R)..(-ε), 1 / (x * (x ^ 2 + 1))) + ∫ x in ε..R, 1 / (x * (x ^ 2 + 1)) = 0 := by
  have : ∫ x in (-R)..(-ε), 1 / (x * (x ^ 2 + 1)) = ∫ x in ε..R, -(1 / (x * (x ^ 2 + 1))) := by
    rw [← intervalIntegral.integral_comp_neg]
    apply intervalIntegral.integral_congr; intro x _; simp only [neg_sq, neg_mul, one_div, inv_neg]
  rw [this, intervalIntegral.integral_neg]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.pv_odd
-- test: fa-pv-odd-test.sage

/-- The half-residue: `iπ e^{ia} = iπ cos a - π sin a`, whose real part is `-π sin a` (`a = 1` here). -/
theorem half_residue (a : ℝ) : (Complex.I * π * Complex.exp ((a : ℂ) * Complex.I)).re = -π * Real.sin a := by
  rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  simp only [Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im, Complex.I_re, Complex.I_im,
    Complex.ofReal_re, Complex.ofReal_im]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.half_residue
-- test: fa-half-residue-test.sage

/-- `PV ∫ cos x/(x - 1) dx = -π sin 1`, as stated (tested, not proved). -/
def PVCosShifted : Prop :=
  ∀ c : ℝ, Tendsto (fun R => c * ∫ y in (-R)..R,
    ((Real.cos y * Real.cos 1 - Real.sin y * Real.sin 1) / y - Real.cos y * Real.cos 1 / y)) atTop
    (𝓝 (c * (-π * Real.sin 1)))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.PVCosShifted
-- test: fa-pv-cos-shifted-test.sage

/-! ## exm-dist-pv-plemelj -/

/-- `∫ ε/(x² + ε²) dx = π` for every `ε > 0`. -/
theorem lorentz_pi (ε : ℝ) (hε : 0 < ε) : ∫ x : ℝ, ε / (x ^ 2 + ε ^ 2) = π := by
  have := Measure.integral_comp_div (fun y : ℝ => (1 + y ^ 2)⁻¹) ε
  rw [show (fun x : ℝ => ε / (x ^ 2 + ε ^ 2)) = fun x : ℝ => (1 / ε) * ((fun y : ℝ => (1 + y ^ 2)⁻¹) (x / ε)) by
    funext x; field_simp; ring, integral_const_mul, this, integral_univ_inv_one_add_sq, abs_of_pos hε, smul_eq_mul]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.lorentz_pi
-- test: fa-lorentz-pi-test.sage

/-- The Sokhotski-Plemelj limit `lim ∫ φ/(x - iε) = PV ∫ φ/x + iπ φ(0)`, as stated for `φ = e^{-x²}` (tested, not
proved): the imaginary part tends to `π φ(0) = π`. -/
def PlemeljGaussian : Prop :=
  ∀ c : ℝ, Tendsto (fun ε : ℝ => c * ∫ x : ℝ, ε / (x ^ 2 + ε ^ 2) * Real.exp (-x ^ 2)) (𝓝[>] 0) (𝓝 (c * π))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.PlemeljGaussian
-- test: fa-plemelj-gaussian-test.sage

/-! ## exm-dist-pv-transform, exm-dist-sage-sgn -/

/-- `θ = (1 + sgn)/2` away from `0`. -/
theorem step_sign (x : ℝ) (hx : x ≠ 0) : (1 + Real.sign x) / 2 = if 0 < x then 1 else 0 := by
  rcases lt_or_gt_of_ne hx with h | h
  · rw [Real.sign_of_neg h, if_neg (not_lt.2 h.le)]; norm_num
  · rw [Real.sign_of_pos h, if_pos h]; norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.step_sign
-- test: fa-step-sign-test.sage

/-- For `L > 1`, `(sgn(L + 1) + sgn(L - 1))/2 = 1`. -/
theorem sign_sum_one (L : ℝ) (hL : 1 < L) : (Real.sign (L + 1) + Real.sign (L - 1)) / 2 = 1 := by
  rw [Real.sign_of_pos (by linarith), Real.sign_of_pos (by linarith)]; norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.sign_sum_one
-- test: fa-sign-sum-one-test.sage

/-- `sin Lx cos x = (sin(L+1)x + sin(L-1)x)/2`. -/
theorem sin_cos_product (L x : ℝ) :
    Real.sin (L * x) * Real.cos (1 * x) = (Real.sin (L * x + 1 * x) + Real.sin (L * x - 1 * x)) / 2 := by
  veriphy_certified
    ( rw [Real.sin_add]
      rw [Real.sin_sub]
      ring)
  -- certificate: fa-trig-addition.sage

/-! ## exr-dist-pv-log -/

/-- `∫₀^b log x dx = b log b - b`; in particular `∫₀¹ |log x| = 1`. -/
theorem integral_log_zero (b : ℝ) : ∫ x in (0 : ℝ)..b, Real.log x = b * Real.log b - b := by
  rw [integral_log]; simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.integral_log_zero
-- test: fa-integral-log-zero-test.sage

/-- The boundary term `ε log ε → 0`. -/
theorem eps_log_eps (c : ℝ) : Tendsto (fun ε : ℝ => c * (ε * Real.log ε)) (𝓝[>] 0) (𝓝 0) := by
  have := (tendsto_log_mul_rpow_nhdsGT_zero (r := 1) one_pos).const_mul c
  simp only [Real.rpow_one, mul_zero] at this
  refine this.congr (fun ε => ?_); ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.eps_log_eps
-- test: fa-eps-log-eps-test.sage

/-! ## exr-dist-finite-part -/

/-- `Pf(1/x²)(e^{-x²}) = 2 ∫₀^∞ (e^{-x²} - 1)/x² dx = -2√π`, as stated (tested, not proved). -/
def FinitePartGaussian : Prop :=
  ∀ c : ℝ, c * (2 * ∫ x in Set.Ioi (0 : ℝ), (Real.exp (-x ^ 2) - 1) / x ^ 2) = c * (-2 * Real.sqrt π)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.FinitePartGaussian
-- test: fa-finite-part-gaussian-test.sage

/-- After the integration by parts: `2 ∫₀^∞ 2x e^{-x²}/x dx = 4 ∫₀^∞ e^{-x²} = 2√π`. -/
theorem finite_part_step (c : ℝ) : c * (2 * ∫ x in Set.Ioi (0 : ℝ), 2 * Real.exp (-x ^ 2)) = c * (2 * Real.sqrt π) := by
  congr 1
  rw [integral_const_mul]
  have := integral_gaussian_Ioi 1
  simp only [one_mul, div_one] at this
  rw [show (fun x : ℝ => Real.exp (-x ^ 2)) = fun x : ℝ => Real.exp (-1 * x ^ 2) by funext x; ring_nf, this]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PrincipalValue.finite_part_step
-- test: fa-finite-part-step-test.sage

end KaluzaKlein.FourierAnalysis.PrincipalValue

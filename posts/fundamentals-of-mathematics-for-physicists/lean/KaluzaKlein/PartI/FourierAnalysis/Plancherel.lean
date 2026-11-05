import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.ConvolutionProduct

/-! # Fourier analysis: plancherel (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Plancherel

open Real MeasureTheory Filter Topology Complex
open scoped FourierTransform

/-! ## thm-plancherel -/

/-- Plancherel's theorem (mathlib): the Fourier transform is an isometry of `L²(ℝ)`, `‖𝓕 f‖ = ‖f‖`. -/
theorem plancherel (f : Lp (α := ℝ) ℂ 2) : ‖𝓕 f‖ = ‖f‖ :=
  MeasureTheory.Lp.norm_fourier_eq f

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.plancherel
-- test: fa-plancherel-test.sage

/-- The polarised form (mathlib): `⟪𝓕 f, 𝓕 g⟫ = ⟪f, g⟫`. -/
theorem plancherel_inner (f g : Lp (α := ℝ) ℂ 2) : inner ℂ (𝓕 f) (𝓕 g) = inner ℂ f g :=
  MeasureTheory.Lp.inner_fourier_eq f g

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.plancherel_inner
-- test: fa-plancherel-inner-test.sage

/-! ## exm-ft-plancherel-box, exr-ft-sinc-integrals -/

/-- `∫ sin² ak / k² dk = π a` for `a > 0` (Plancherel for the box), as stated (tested, not proved: no recipe
evaluates improper integrals of the sinc). -/
def SincSquareIntegral : Prop :=
  ∀ a : ℝ, 0 < a → ∫ k : ℝ, Real.sin (a * k) ^ 2 / k ^ 2 = π * a

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.SincSquareIntegral
-- test: fa-sinc-square-integral-test.sage

/-- `∫ sin⁴ x / x⁴ dx = 2π/3` (Plancherel for the triangle), as stated (tested, not proved). -/
def SincFourthIntegral : Prop :=
  ∀ c : ℝ, c * ∫ x : ℝ, Real.sin x ^ 4 / x ^ 4 = c * (2 * π / 3)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.SincFourthIntegral
-- test: fa-sinc-fourth-integral-test.sage

/-- `∫ Λ₁² = 2 ∫₀² (2 - x)² dx = 16/3`. -/
theorem triangle_sq_integral : 2 * ∫ x in (0 : ℝ)..2, (2 - x) ^ 2 = 16 / 3 := by
  have h : ∫ x in (0 : ℝ)..2, (2 - x) ^ 2 = 8 / 3 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 : ℝ),
            HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 - (2 * (x_veriphy ^ 2)) + (4 * x_veriphy) : ℝ))
              ((fun x => (2 - x) ^ 2) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ)).add
                          (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-2 : ℝ))).add
                      ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            ((x_veriphy ^ 3) / 3 - (2 * (x_veriphy ^ 2)) + (4 * x_veriphy) : ℝ)) (f' := fun x => (2 - x) ^ 2) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: fa-integral.sage
  rw [h]; norm_num

/-! ## exm-ft-plancherel-lorentzian -/

/-- `∫ e^{-2a|x|} dx = 1/a`. -/
theorem exp_two_sided_l2 (a : ℝ) (ha : 0 < a) : ∫ x : ℝ, Real.exp (-(2 * a) * |x|) = 1 / a := by
  have h := KaluzaKlein.FourierAnalysis.TheTransform.exp_two_sided (2 * a) (by positivity) 0
  simp only [Complex.ofReal_zero, zero_mul, neg_zero, Complex.exp_zero, mul_one] at h
  rw [integral_complex_ofReal] at h
  rw [Complex.ofReal_injective h]
  have ha' : a ≠ 0 := ha.ne'
  rw [show (2 * a) ^ 2 + 0 ^ 2 = (2 * a) ^ 2 by ring]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.exp_two_sided_l2
-- test: fa-exp-two-sided-l2-test.sage

/-- `∫ dk / (a² + k²)² = π / (2a³)` for `a > 0` (by the antiderivative `k/(2a²(a²+k²)) + arctan(k/a)/(2a³)`), as stated
(tested, not proved: no recipe evaluates improper integrals with arctan limits). -/
def LorentzianSqIntegral : Prop :=
  ∀ a : ℝ, 0 < a → ∫ k : ℝ, 1 / (a ^ 2 + k ^ 2) ^ 2 = π / (2 * a ^ 3)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.LorentzianSqIntegral
-- test: fa-lorentzian-sq-integral-test.sage

/-- The Gaussian case: `∫ e^{-x²/σ²} = σ√π`. -/
theorem gaussian_l2 (σ : ℝ) (hσ : 0 < σ) : ∫ x : ℝ, Real.exp (-x ^ 2 / σ ^ 2) = σ * Real.sqrt π := by
  rw [show (fun x : ℝ => Real.exp (-x ^ 2 / σ ^ 2)) = fun x : ℝ => Real.exp (-(1 / σ ^ 2) * x ^ 2) by
    funext x; ring_nf, integral_gaussian]
  rw [show π / (1 / σ ^ 2) = σ ^ 2 * π by field_simp, Real.sqrt_mul (by positivity), Real.sqrt_sq hσ.le]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.gaussian_l2
-- test: fa-gaussian-l2-test.sage

/-! ## exm-ft-plancherel-hermite -/

/-- `e^{-x²/2}` and `x e^{-x²/2}` are orthogonal: `∫ x e^{-b x²} dx = 0` by oddness (`b = 1` for the pair). -/
theorem hermite_orthogonal (b : ℝ) : ∫ x : ℝ, x * Real.exp (-b * x ^ 2) = 0 := by
  have h := integral_neg_eq_self (fun x : ℝ => x * Real.exp (-b * x ^ 2)) volume
  simp only [neg_mul, even_two, Even.neg_pow, integral_neg] at h ⊢
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.hermite_orthogonal
-- test: fa-hermite-orthogonal-test.sage

/-- An eigenvalue with `λ⁴ = (2π)²` has modulus `√(2π)`: `λ/√(2π)` lies on the unit circle. -/
theorem eigenvalue_modulus (lam : ℂ) (h : lam ^ 4 = (2 * π) ^ 2) : ‖lam‖ = Real.sqrt (2 * π) := by
  have h4 : ‖lam‖ ^ 4 = (2 * π) ^ 2 := by
    rw [← norm_pow, h, norm_pow]
    simp only [norm_mul, Complex.norm_ofNat, Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
  have hu : (‖lam‖ ^ 2) ^ 2 = (2 * π) ^ 2 := by rw [← pow_mul]; exact h4
  have h2 : ‖lam‖ ^ 2 = 2 * π := by
    have hp := Real.pi_pos
    nlinarith [sq_nonneg (‖lam‖ ^ 2 - 2 * π), sq_nonneg (‖lam‖ ^ 2 + 2 * π), sq_nonneg ‖lam‖]
  rw [← h2, Real.sqrt_sq (norm_nonneg _)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.eigenvalue_modulus
-- test: fa-eigenvalue-modulus-test.sage

/-! ## exr-ft-plancherel-proof -/

/-- `h(y) = conj f(-y)` has `ĥ = conj f̂`. -/
theorem ft_reflect_conj (f : ℝ → ℂ) (k : ℝ) :
    ∫ y : ℝ, (starRingEnd ℂ) (f (-y)) * Complex.exp (-(k * y) * Complex.I) =
      (starRingEnd ℂ) (∫ y : ℝ, f y * Complex.exp (-(k * y) * Complex.I)) := by
  rw [← integral_conj, ← integral_neg_eq_self (fun y => (starRingEnd ℂ) (f y * Complex.exp (-(k * y) * Complex.I)))]
  congr 1; funext y
  rw [map_mul, ← Complex.exp_conj]
  congr 2
  first | (simp [Complex.conj_ofReal]; done) | (simp [Complex.conj_ofReal]; ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.ft_reflect_conj
-- test: fa-ft-reflect-conj-test.sage

/-- The polarisation identity `conj u · v = (1/4) Σ_{m=0}^{3} i^m |u + i^m v|²`... in the form
`conj u · v = (1/4) Σ_m conj(i^m) |u + i^m v|²` used for the inner products. -/
theorem polarization (u v : ℂ) :
    (starRingEnd ℂ) u * v = (1 / 4) * ∑ m ∈ Finset.range 4,
      (starRingEnd ℂ) (Complex.I ^ m) * (Complex.normSq (u + Complex.I ^ m * v) : ℂ) := by
  simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, pow_zero, pow_one, Complex.I_sq,
    pow_succ, Complex.normSq_apply, map_mul, Complex.conj_I, map_one, map_neg, one_mul, zero_add]
  apply Complex.ext <;> simp [Complex.mul_re, Complex.mul_im] <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.polarization
-- test: fa-polarization-test.sage

/-- `∫ χ₁ e^{-|x|} = 2 ∫₀¹ e^{-x} = 2(1 - e^{-1})`. -/
theorem box_exp_integral : 2 * ∫ x in (0 : ℝ)..1, Real.exp (-x) = 2 * (1 - Real.exp (-1)) := by
  have h : ∫ x in (0 : ℝ)..1, Real.exp (-x) = 1 - Real.exp (-1) := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => (-(Real.exp (-x_veriphy)) : ℝ)) ((fun x => rexp (-x)) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-1 : ℝ)).exp).const_mul (-1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; done)
          · (field_simp; done)
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.exp (-x_veriphy)) : ℝ)) (f' :=
            fun x => rexp (-x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; ring1))
    -- certificate: fa-integral2.sage
  rw [h]

/-- `∫ sin k / (k (1 + k²)) dk = π (1 - e^{-1})` (Plancherel's polarised form), as stated (tested, not proved). -/
def SincLorentzianIntegral : Prop :=
  ∀ c : ℝ, c * ∫ k : ℝ, Real.sin k / (k * (1 + k ^ 2)) = c * (π * (1 - Real.exp (-1)))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.SincLorentzianIntegral
-- test: fa-sinc-lorentzian-integral-test.sage

/-! ## exr-ft-poisson-summation -/

/-- Poisson summation (mathlib, period 1): `Σ f(x + n) = Σ 𝓕 f(n) e^{2πinx}` for Schwartz `f`. -/
theorem poisson_summation (f : SchwartzMap ℝ ℂ) (x : ℝ) :
    ∑' n : ℤ, f (x + n) = ∑' n : ℤ, 𝓕 f n * fourier n (x : UnitAddCircle) :=
  SchwartzMap.tsum_eq_tsum_fourier f x

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.poisson_summation
-- test: fa-poisson-summation-test.sage

/-- For the Gaussian: `Σ e^{-2π²n²/σ²} = (σ/√(2π)) Σ e^{-σ²n²/2}`, the functional equation of the theta function. -/
theorem poisson_gaussian (σ : ℝ) (hσ : 0 < σ) :
    ∑' n : ℤ, Real.exp (-2 * π ^ 2 * (n : ℝ) ^ 2 / σ ^ 2) =
      σ / Real.sqrt (2 * π) * ∑' n : ℤ, Real.exp (-σ ^ 2 * (n : ℝ) ^ 2 / 2) := by
  have ha : 0 < 2 * π / σ ^ 2 := by positivity
  have hσ' : σ ≠ 0 := hσ.ne'
  have hs2 : Real.sqrt (2 * π) ≠ 0 := (Real.sqrt_pos.2 (by positivity)).ne'
  have := Real.tsum_exp_neg_mul_int_sq ha
  rw [show (fun n : ℤ => Real.exp (-2 * π ^ 2 * (n : ℝ) ^ 2 / σ ^ 2)) =
      fun n : ℤ => Real.exp (-π * (2 * π / σ ^ 2) * (n : ℝ) ^ 2) by funext n; congr 1; ring, this]
  congr 1
  · rw [← Real.sqrt_eq_rpow, Real.sqrt_div' _ (by positivity), Real.sqrt_sq hσ.le, one_div_div]
  · congr 1; funext n; congr 1
    have hπ : π ≠ 0 := Real.pi_ne_zero
    field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Plancherel.poisson_gaussian
-- test: fa-poisson-gaussian-test.sage

end KaluzaKlein.FourierAnalysis.Plancherel

import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.PrincipalValue

/-! # Fourier analysis: distribution transform (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.DistributionTransform

open Real MeasureTheory Filter Topology Complex
open scoped FourierTransform

/-- The book's transform `φ̂(k) = ∫ φ(x) e^{-ikx} dx`. -/
noncomputable def ft (f : ℝ → ℂ) (k : ℝ) : ℂ := ∫ x : ℝ, f x * Complex.exp (-(k * x) * Complex.I)

/-! ## def-distribution-transform -/

/-- The transform is self-adjoint, `∫ f φ̂ = ∫ f̂ φ` (Fubini), as stated for integrable `f, φ` (tested, not proved
in the book's normalisation; mathlib has it as `VectorFourier.integral_fourierIntegral_smul_eq_flip`). -/
def TransformSelfAdjoint : Prop :=
  ∀ f φ : ℝ → ℂ, Integrable f → Integrable φ → ∫ x, f x * ft φ x = ∫ k, ft f k * φ k

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.TransformSelfAdjoint
-- test: fa-transform-self-adjoint-test.sage

/-! ## exm-dist-ft-delta -/

/-- `δ̂ = 1`: `δ(φ̂) = φ̂(0) = ∫ φ`. -/
theorem delta_hat (φ : ℝ → ℂ) : ft φ 0 = ∫ x, φ x := by simp [ft]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.delta_hat
-- test: fa-delta-hat-test.sage

/-- `δ̂_a = e^{-ika}`: `δ_a(φ̂) = φ̂(a) = ∫ φ e^{-iax}`. -/
theorem delta_a_hat (φ : ℝ → ℂ) (a : ℝ) : ft φ a = ∫ x, φ x * Complex.exp (-(a * x) * Complex.I) := rfl

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.delta_a_hat
-- test: fa-delta-a-hat-test.sage

/-- The white spectrum is the limit of the Gaussian transforms: `e^{-ε²k²/4} → 1` as `ε → 0`. -/
theorem white_limit (k : ℝ) : Tendsto (fun ε : ℝ => Real.exp (-ε ^ 2 * k ^ 2 / 4)) (𝓝 0) (𝓝 1) := by
  have : Continuous fun ε : ℝ => Real.exp (-ε ^ 2 * k ^ 2 / 4) := by fun_prop
  simpa using this.tendsto 0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.white_limit
-- test: fa-white-limit-test.sage

/-- `δ̂' = ik`: `-φ̂'(0) = ∫ i x φ(x) dx` (mathlib's derivative of the transform, in its normalisation:
`(𝓕 φ)'(0) = 𝓕(-2πi x φ)(0)`). -/
theorem delta_prime_hat (φ : ℝ → ℂ) (hφ : Integrable φ) (hxφ : Integrable fun x : ℝ => x • φ x) :
    deriv (𝓕 φ) 0 = 𝓕 (fun x : ℝ => (-2 * π * Complex.I * x) • φ x) 0 :=
  (Real.hasDerivAt_fourier hφ hxφ 0).deriv

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.delta_prime_hat
-- test: fa-delta-prime-hat-test.sage

/-! ## exm-dist-ft-constant, exr-dist-ft-constant -/

/-- The transform of `e^{-εx²}` is `√(π/ε) e^{-k²/4ε}` (mathlib's Gaussian transform). -/
theorem damped_constant_transform (ε : ℝ) (hε : 0 < ε) (k : ℝ) :
    ∫ x : ℝ, Complex.exp (Complex.I * (-k) * x) * Complex.exp (-(ε : ℂ) * x ^ 2) =
      ((Real.sqrt (π / ε) : ℝ) : ℂ) * Complex.exp ((-k ^ 2 / (4 * ε) : ℝ) : ℂ) := by
  rw [fourierIntegral_gaussian (by simpa using hε)]
  congr 1
  · rw [show (π : ℂ) / ε = ((π / ε : ℝ) : ℂ) by push_cast; ring, show (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) by push_cast; ring,
      ← Complex.ofReal_cpow (by positivity), ← Real.sqrt_eq_rpow]
  · congr 1; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_constant_transform
-- test: fa-damped-constant-transform-test.sage

/-- Its integral is `2π` for every `ε`: `∫ √(π/ε) e^{-k²/4ε} dk = 2π`, so it tends to `2πδ`. -/
theorem damped_constant_mass (ε : ℝ) (hε : 0 < ε) : ∫ k : ℝ, Real.sqrt (π / ε) * Real.exp (-k ^ 2 / (4 * ε)) = 2 * π := by
  rw [integral_const_mul, show (fun k : ℝ => Real.exp (-k ^ 2 / (4 * ε))) = fun k : ℝ => Real.exp (-(1 / (4 * ε)) * k ^ 2) by
    funext k; ring_nf, integral_gaussian, show π / (1 / (4 * ε)) = 4 * ε * π by field_simp, ← Real.sqrt_mul (by positivity)]
  rw [show π / ε * (4 * ε * π) = (2 * π) ^ 2 by first | (field_simp; done) | (field_simp; ring), Real.sqrt_sq (by positivity)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_constant_mass
-- test: fa-damped-constant-mass-test.sage

/-- Against `e^{-k²/2}`: `∫ √(π/ε) e^{-k²/4ε} e^{-k²/2} dk = 2π/√(1 + 2ε) → 2π`. -/
theorem damped_constant_gauss (ε : ℝ) (hε : 0 < ε) :
    ∫ k : ℝ, Real.sqrt (π / ε) * Real.exp (-k ^ 2 / (4 * ε)) * Real.exp (-k ^ 2 / 2) = 2 * π / Real.sqrt (1 + 2 * ε) := by
  rw [show (fun k : ℝ => Real.sqrt (π / ε) * Real.exp (-k ^ 2 / (4 * ε)) * Real.exp (-k ^ 2 / 2)) =
      fun k : ℝ => Real.sqrt (π / ε) * Real.exp (-((1 + 2 * ε) / (4 * ε)) * k ^ 2) by
    funext k; rw [mul_assoc, ← Real.exp_add]; congr 2; field_simp; ring, integral_const_mul, integral_gaussian,
    ← Real.sqrt_mul (by positivity)]
  rw [show π / ε * (π / ((1 + 2 * ε) / (4 * ε))) = (2 * π) ^ 2 / (1 + 2 * ε) by first | (field_simp; done) | (field_simp; ring),
    Real.sqrt_div' _ (by positivity), Real.sqrt_sq (by positivity)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_constant_gauss
-- test: fa-damped-constant-gauss-test.sage

/-- Against `cos k`: `∫ √(π/ε) e^{-k²/4ε} cos k dk = 2π e^{-ε}`. -/
theorem damped_constant_cos (ε : ℝ) (hε : 0 < ε) :
    ∫ k : ℝ, Real.sqrt (π / ε) * Real.exp (-k ^ 2 / (4 * ε)) * Real.cos k = 2 * π * Real.exp (-ε) := by
  have hσ : 0 < 2 * Real.sqrt ε := by positivity
  have h := KaluzaKlein.FourierAnalysis.Distributions.gauss_cos (2 * Real.sqrt ε) 1 hσ
  have hs : (2 * Real.sqrt ε) ^ 2 = 4 * ε := by rw [mul_pow, Real.sq_sqrt hε.le]; norm_num
  rw [hs, show -(1 : ℝ) ^ 2 * (4 * ε) / 4 = -ε by ring] at h
  simp only [one_mul] at h
  have hc : Real.sqrt (π / ε) = 2 * π * (1 / (2 * Real.sqrt ε * Real.sqrt π)) := by
    rw [Real.sqrt_div' _ hε.le]
    have h1 : 0 < Real.sqrt π := Real.sqrt_pos.2 Real.pi_pos
    have h2 : 0 < Real.sqrt ε := Real.sqrt_pos.2 hε
    field_simp
    rw [Real.sq_sqrt Real.pi_pos.le]
  rw [show (fun k : ℝ => Real.sqrt (π / ε) * Real.exp (-k ^ 2 / (4 * ε)) * Real.cos k) =
      fun k : ℝ => 2 * π * (Real.exp (-k ^ 2 / (4 * ε)) / (2 * Real.sqrt ε * Real.sqrt π) * Real.cos k) by
    funext k; rw [hc]; ring, integral_const_mul, h]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_constant_cos
-- test: fa-damped-constant-cos-test.sage

/-- Against `1/(1 + k²)` for the exponential damping: `∫ 2ε/(ε² + k²) · 1/(1 + k²) dk = 2π/(1 + ε)` (partial
fractions). -/
theorem damped_exp_lorentz (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≠ 1) :
    ∫ k : ℝ, 2 * ε / (ε ^ 2 + k ^ 2) * (1 / (1 + k ^ 2)) = 2 * π / (1 + ε) := by
  have hpf : ∀ k : ℝ, 2 * ε / (ε ^ 2 + k ^ 2) * (1 / (1 + k ^ 2)) =
      (2 / (1 - ε ^ 2)) * (ε / (k ^ 2 + ε ^ 2)) - (2 * ε / (1 - ε ^ 2)) * (1 + k ^ 2)⁻¹ := by
    intro k
    have h1 : 1 - ε ^ 2 ≠ 0 := by
      intro h; have : (ε - 1) * (ε + 1) = 0 := by linarith
      rcases mul_eq_zero.1 this with h | h
      · exact hε1 (by linarith)
      · linarith
    field_simp; ring
  simp_rw [hpf]
  have i1 : Integrable (fun k : ℝ => ε / (k ^ 2 + ε ^ 2)) := by
    have := (integrable_inv_one_add_sq.comp_div hε.ne').const_mul (1 / ε)
    refine this.congr (Eventually.of_forall fun k => ?_)
    simp only [Function.comp]; field_simp; ring
  rw [integral_sub (i1.const_mul _) (integrable_inv_one_add_sq.const_mul _), integral_const_mul, integral_const_mul,
    KaluzaKlein.FourierAnalysis.PrincipalValue.lorentz_pi ε hε, integral_univ_inv_one_add_sq]
  have h1 : (1 - ε ^ 2) = (1 - ε) * (1 + ε) := by ring
  have h2 : 1 - ε ≠ 0 := sub_ne_zero.2 (Ne.symm hε1)
  rw [h1]; field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_exp_lorentz
-- test: fa-damped-exp-lorentz-test.sage

/-- Against `cos k` for the exponential damping: `∫ 2ε/(ε² + k²) cos k dk = 2π e^{-ε}`, as stated (tested, not
proved: inversion of the Lorentzian). -/
def DampedExpCos : Prop :=
  ∀ ε : ℝ, 0 < ε → ∫ k : ℝ, 2 * ε / (ε ^ 2 + k ^ 2) * Real.cos k = 2 * π * Real.exp (-ε)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.DampedExpCos
-- test: fa-damped-exp-cos-test.sage

/-! ## exm-dist-ft-periodic, exr-dist-ft-rules -/

/-- `ê^{inx} = 2πδ(k - n)`: pairing with `φ̂` gives `∫ e^{inx} φ̂(x) dx = 2π φ(n)` (inversion), as stated for the
Gaussian `φ = e^{-x²/2}` (tested, not proved). -/
def PlaneWavePairing : Prop :=
  ∀ n : ℤ, ∫ x : ℝ, Complex.exp (n * x * Complex.I) * (Real.sqrt (2 * π) * Complex.exp (-(x : ℂ) ^ 2 / 2)) =
    2 * π * Complex.exp (-(n : ℂ) ^ 2 / 2)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.PlaneWavePairing
-- test: fa-plane-wave-pairing-test.sage

/-- `δ̂^{(n)} = (ik)^n`: `(-1)^n φ̂^{(n)}(0) = ∫ (ix)^n φ`, as stated for `φ = e^{-x²}` (tested, not proved). -/
def DeltaDerivativeHat : Prop :=
  ∀ n : ℕ, (-1 : ℂ) ^ n * iteratedDeriv n (fun k : ℝ => Real.sqrt π * Complex.exp (-(k : ℂ) ^ 2 / 4)) 0 =
    ∫ x : ℝ, (Complex.I * x) ^ n * Complex.exp (-(x : ℂ) ^ 2)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.DeltaDerivativeHat
-- test: fa-delta-derivative-hat-test.sage

/-! ## exr-dist-ft-sign -/

/-- The damped sign transforms to `1/(ε + ik) - 1/(ε - ik) = -2ik/(ε² + k²)`: the common denominator is
`(ε + ik)(ε - ik) = ε² + k²` and the numerator is `(ε - ik) - (ε + ik) = -2ik`. -/
theorem damped_sign_value (ε k : ℝ) :
    ((ε : ℂ) + k * Complex.I) * ((ε : ℂ) - k * Complex.I) = ((ε ^ 2 + k ^ 2 : ℝ) : ℂ) ∧
      ((ε : ℂ) - k * Complex.I) - ((ε : ℂ) + k * Complex.I) = -2 * k * Complex.I := by
  constructor
  · push_cast
    linear_combination (-(k : ℂ) ^ 2) * Complex.I_sq
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_sign_value
-- test: fa-damped-sign-value-test.sage

/-- The one-sided damping: `1/(ε + ik) = 1/(i(k - iε))`, since `ε + ik = i(k - iε)`. -/
theorem damped_step_value (ε k : ℝ) : (ε : ℂ) + k * Complex.I = Complex.I * (k - ε * Complex.I) := by
  linear_combination (ε : ℂ) * Complex.I_sq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.damped_step_value
-- test: fa-damped-step-value-test.sage

/-- Combining with Sokhotski-Plemelj: `(1/i)(P + iπ d) = P/i + π d` for the principal-value and delta parts, with
`1/i = -i`. -/
theorem sokhotski_combination (P d : ℂ) :
    -Complex.I * (P + Complex.I * π * d) = -Complex.I * P + π * d ∧ Complex.I * -Complex.I = 1 := by
  constructor
  · linear_combination (-(π : ℂ) * d) * Complex.I_sq
  · linear_combination (-1 : ℂ) * Complex.I_sq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionTransform.sokhotski_combination
-- test: fa-sokhotski-combination-test.sage

end KaluzaKlein.FourierAnalysis.DistributionTransform

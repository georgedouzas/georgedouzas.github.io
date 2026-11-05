import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Exercises"

The claims of the integrative exercises of `tutorial/part-i-04-complex-analysis.qmd`: the Fourier transform of a
Gaussian by completing the square and shifting the contour, the keyhole-contour evaluation of
`∫₀^∞ x^{a-1}/(1 + x) dx = π/sin πa` and the reflection formula, and the time-dependent propagator from the
`iε`-displaced poles.
-/

namespace KaluzaKlein.ComplexAnalysis.Exercises

open Complex Real MeasureTheory Filter Topology

/-! ## exr-ca-gaussian-fourier -/

/-- Completing the square: `-x²/2 + ikx = -(x - ik)²/2 - k²/2`. -/
theorem complete_square (x k : ℝ) : -(x : ℂ) ^ 2 / 2 + I * k * x = -((x : ℂ) - I * k) ^ 2 / 2 - (k : ℂ) ^ 2 / 2 := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (1 / 2 * (↑k) ^ 2) * hI)
  -- certificate: ideal_membership.sage

/-- On the vertical sides of the rectangle `[-R, R] × [0, k]` the integrand `e^{-z²/2}` is bounded by
`e^{-R²/2 + k²/2}`. -/
theorem vertical_side_bound (R t k : ℝ) (ht : |t| ≤ |k|) :
    ‖Complex.exp (-((R : ℂ) + t * I) ^ 2 / 2)‖ ≤ Real.exp (-R ^ 2 / 2 + k ^ 2 / 2) := by
  rw [Complex.norm_exp]
  apply Real.exp_le_exp.mpr
  have hre : (-((R : ℂ) + t * I) ^ 2 / 2).re = -(R ^ 2 - t ^ 2) / 2 := by
    simp [sq, Complex.mul_re]
  rw [hre]
  have : t ^ 2 ≤ k ^ 2 := sq_le_sq.mpr ht
  linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Exercises.vertical_side_bound (0) (0) (0) (le_rfl) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Exercises.vertical_side_bound
-- test: ca-ex-vertical-bound-test.sage

/-- The shifted Gaussian integral: `∫ e^{-(x - ik)²/2} dx = √(2π)` (Cauchy's theorem on the rectangle, in mathlib's
form for a Gaussian with complex shift). -/
theorem shifted_gaussian (k : ℝ) : ∫ x : ℝ, Complex.exp (-((x : ℂ) - I * k) ^ 2 / 2) = ((2 * π : ℝ) : ℂ) ^ (1 / 2 : ℂ) := by
  have h := GaussianFourier.integral_cexp_neg_mul_sq_add_real_mul_I (b := (1 / 2 : ℂ)) (by norm_num) (-k)
  have e : ∀ x : ℝ, Complex.exp (-((x : ℂ) - I * k) ^ 2 / 2) =
      Complex.exp (-(1 / 2 : ℂ) * ((x : ℂ) + ((-k : ℝ) : ℂ) * I) ^ 2) := by
    intro x; congr 1; push_cast; ring
  simp_rw [e, h]
  congr 1
  push_cast; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.shifted_gaussian
-- test: ca-ex-shifted-gaussian-test.sage

/-- The Fourier transform of the Gaussian: `∫ e^{-x²/2} e^{ikx} dx = √(2π) e^{-k²/2}`. -/
theorem gaussian_fourier (k : ℝ) :
    ∫ x : ℝ, Complex.exp (-(x : ℂ) ^ 2 / 2) * Complex.exp (I * k * x) =
      ((2 * π : ℝ) : ℂ) ^ (1 / 2 : ℂ) * Complex.exp (-(k : ℂ) ^ 2 / 2) := by
  have h := fourierIntegral_gaussian (b := (1 / 2 : ℂ)) (by norm_num) (k : ℂ)
  have e : ∀ x : ℝ, Complex.exp (-(x : ℂ) ^ 2 / 2) * Complex.exp (I * k * x) =
      Complex.exp (I * k * x) * Complex.exp (-(1 / 2 : ℂ) * x ^ 2) := by
    intro x; rw [mul_comm]; congr 2; ring
  simp_rw [e, h]
  congr 1
  · push_cast; ring_nf
  · congr 1; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.gaussian_fourier
-- test: ca-ex-gaussian-fourier-test.sage

/-- The Gaussian integral `∫ e^{-b u²} du = Γ(½)/√b` for `b > 0` (at `b = ½`: `√(2π) = Γ(½)√2`). -/
theorem gaussian_gamma (b : ℝ) (hb : 0 < b) : ∫ u : ℝ, Real.exp (-b * u ^ 2) = Real.Gamma (1 / 2) / √b := by
  rw [integral_gaussian, Real.Gamma_one_half_eq, Real.sqrt_div' _ hb.le]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.gaussian_gamma
-- test: ca-ex-gaussian-gamma-test.sage

/-! ## exr-ca-keyhole-beta -/

/-- On the lower side of the cut `arg z = 2π`: `x^{a-1} e^{2πi(a-1)} = x^{a-1} e^{2πia}`. -/
theorem lower_side_phase (a : ℝ) : Complex.exp (2 * π * I * (a - 1)) = Complex.exp (2 * π * I * a) := by
  rw [show 2 * π * I * ((a : ℂ) - 1) = 2 * π * I * a + (-1 : ℤ) * (2 * π * I) by push_cast; ring,
    Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, mul_one]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.lower_side_phase
-- test: ca-ex-lower-phase-test.sage

/-- The residue of `z^{a-1}/(1 + z)` at `z = -1 = e^{iπ}` is `e^{iπ(a-1)} = -e^{iπa}`. -/
theorem keyhole_residue (a : ℝ) : Complex.exp (π * I * (a - 1)) = -Complex.exp (π * I * a) := by
  rw [show π * I * ((a : ℂ) - 1) = π * I * a + -(π * I) by ring, Complex.exp_add, Complex.exp_neg,
    Complex.exp_pi_mul_I]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.keyhole_residue
-- test: ca-ex-keyhole-residue-test.sage

/-- Solving the keyhole relation `(1 - e^{2πia}) I = -2πi e^{iπa}`: since `-2πi e^{iπa} sin πa = π(1 - e^{2πia})`,
`I = -2πi e^{iπa}/(1 - e^{2πia}) = π/sin πa` whenever `sin πa ≠ 0`. -/
theorem keyhole_solve (a : ℝ) :
    -2 * π * I * Complex.exp (π * I * a) * (Real.sin (π * a) : ℂ) = π * (1 - Complex.exp (2 * π * I * a)) := by
  have h2 : Complex.exp (2 * π * I * a) = Complex.exp (π * I * a) ^ 2 := by
    rw [← Complex.exp_nat_mul]; congr 1; push_cast; ring
  have e1 : Complex.exp ((π : ℂ) * a * I) = Complex.exp (π * I * a) := by congr 1; ring
  have e2 : Complex.exp (-((π : ℂ) * a) * I) = Complex.exp (-(π * I * a)) := by congr 1; ring
  have h2s := Complex.two_sin ((π : ℂ) * a)
  rw [e1, e2] at h2s
  have huv : Complex.exp (π * I * a) * Complex.exp (-(π * I * a)) = 1 := by
    rw [← Complex.exp_add, add_neg_cancel, Complex.exp_zero]
  have hI : I ^ 2 = -1 := I_sq
  rw [Complex.ofReal_sin, Complex.ofReal_mul, h2]
  generalize Complex.sin (↑π * ↑a) = S at h2s ⊢
  generalize Complex.exp (π * I * a) = u at h2s huv ⊢
  generalize Complex.exp (-(π * I * a)) = v at h2s huv
  generalize (π : ℂ) = p
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (p * u ^ 2 - p * u * v) * hI + (-p * I * u) * h2s + (p) * huv)
  -- certificate: ideal_membership.sage

/-- The reflection formula `Γ(a)Γ(1 - a) = π/sin πa` (mathlib's form, for every complex `a`). -/
theorem reflection (a : ℂ) (ha : ∀ n : ℤ, a ≠ n) :
    Complex.Gamma a * Complex.Gamma (1 - a) * Complex.sin (π * a) = π := by
  have hs : Complex.sin (π * a) ≠ 0 := by
    rw [Ne, Complex.sin_eq_zero_iff]
    rintro ⟨k, hk⟩
    apply ha k
    have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
    apply mul_left_cancel₀ hπ
    rw [hk]; ring
  rw [Complex.Gamma_mul_Gamma_one_sub, div_mul_cancel₀ _ hs]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.reflection
-- test: ca-ex-reflection-test.sage

/-- At `a = ⅓`: `π/sin(π/3) = 2π/√3`. -/
theorem keyhole_third : π / Real.sin (π * (1 / 3)) = 2 * π / √3 := by
  rw [show π * (1 / 3) = π / 3 by ring, Real.sin_pi_div_three]
  have h3 : (√3 : ℝ) ≠ 0 := by positivity
  have h3' : (√3 / 2 : ℝ) ≠ 0 := by positivity
  rw [div_eq_div_iff h3' h3]
  clear h3 h3'
  generalize (√3 : ℝ) = s
  generalize (π : ℝ) = p
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-! ## exr-ca-propagator-time -/

/-- The poles at `±(E - iδ)`: `(k - (E - iδ))(k + (E - iδ)) = k² - E² + iε + δ²` with `ε = 2Eδ`. -/
theorem displaced_poles (k E δ : ℂ) :
    (k - (E - I * δ)) * (k + (E - I * δ)) = k ^ 2 - E ^ 2 + I * (2 * E * δ) + δ ^ 2 := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-δ ^ 2) * hI)
  -- certificate: ideal_membership.sage

/-- For `t > 0` (closing below, the pole `E - iδ`): `(1/2π)(-2πi) e^{-i(E - iδ)t}/(2(E - iδ)) → -(i/2E) e^{-iEt}`. -/
theorem propagator_pos (E t : ℝ) (hE : 0 < E) :
    Tendsto (fun δ : ℝ => 1 / (2 * π) * (-2 * π * I) *
      (Complex.exp (-I * (E - I * δ) * t) / (2 * (E - I * δ)))) (𝓝 0)
      (𝓝 (-(I / (2 * E)) * Complex.exp (-I * E * t))) := by
  have hE' : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  have hcont : ContinuousAt (fun δ : ℝ => 1 / (2 * π) * (-2 * π * I) *
      (Complex.exp (-I * (E - I * δ) * t) / (2 * (E - I * δ)))) 0 := by
    apply ContinuousAt.mul continuousAt_const
    apply ContinuousAt.div (by fun_prop) (by fun_prop)
    simp [hE']
  convert hcont.tendsto using 2
  simp only [Complex.ofReal_zero, mul_zero, sub_zero]
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.propagator_pos
-- test: ca-ex-propagator-pos-test.sage

/-- For `t < 0` (closing above, the pole `-E + iδ`): `(1/2π)(2πi) e^{-i(-E + iδ)t}/(-2(E - iδ)) → -(i/2E) e^{iEt}`,
so that `G(t) = -(i/2E) e^{-iE|t|}` for both signs of `t`. -/
theorem propagator_neg (E t : ℝ) (hE : 0 < E) :
    Tendsto (fun δ : ℝ => 1 / (2 * π) * (2 * π * I) *
      (Complex.exp (-I * (-E + I * δ) * t) / (-2 * (E - I * δ)))) (𝓝 0)
      (𝓝 (-(I / (2 * E)) * Complex.exp (I * E * t))) := by
  have hE' : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  have hcont : ContinuousAt (fun δ : ℝ => 1 / (2 * π) * (2 * π * I) *
      (Complex.exp (-I * (-E + I * δ) * t) / (-2 * (E - I * δ)))) 0 := by
    apply ContinuousAt.mul continuousAt_const
    apply ContinuousAt.div (by fun_prop) (by fun_prop)
    simp [hE']
  convert hcont.tendsto using 2
  simp only [Complex.ofReal_zero, mul_zero, add_zero, sub_zero]
  have hI : I ^ 2 = -1 := I_sq
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Exercises.propagator_neg
-- test: ca-ex-propagator-neg-test.sage

/-- The two limits are `-(i/2E) e^{-iE|t|}`. -/
theorem propagator_abs (E t : ℝ) :
    (0 < t → Complex.exp (-I * E * t) = Complex.exp (-I * E * |t|)) ∧
      (t < 0 → Complex.exp (I * E * t) = Complex.exp (-I * E * |t|)) := by
  constructor
  · intro ht; rw [abs_of_pos ht]
  · intro ht; rw [abs_of_neg ht]; congr 1; push_cast; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Exercises.propagator_abs (0) (0) := by
  constructor <;> intro h <;> simp at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Exercises.propagator_abs (1) (1) := by
  constructor
  · intro _
    simp
  · intro h
    norm_num at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Exercises.propagator_abs (1) ((-1)) := by
  constructor
  · intro h
    norm_num at h
  · intro _
    norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Exercises.propagator_abs
-- test: ca-ex-propagator-abs-test.sage

/-- The Wick-rotated version, `t = -iτ`: `e^{-iE(-iτ)} = e^{-Eτ}`, a decaying Euclidean propagator. -/
theorem wick_rotated_propagator (E τ : ℂ) : Complex.exp (-I * E * (-I * τ)) = Complex.exp (-E * τ) := by
  have hI : I ^ 2 = -1 := I_sq
  congr 1
  veriphy_certified (linear_combination (E * τ) * hI)
  -- certificate: ideal_membership.sage

end KaluzaKlein.ComplexAnalysis.Exercises

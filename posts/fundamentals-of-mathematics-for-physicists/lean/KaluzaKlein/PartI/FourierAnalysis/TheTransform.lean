import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.SeriesComputations

/-- Closes an algebraic side goal: ring, then clearing denominators. -/
macro "fa_fc" : tactic => `(tactic| first | done | ring1 | (field_simp; done) | (field_simp; ring1) | (simp; done) | (norm_num; done) | (ring_nf; done))

/-! # Fourier analysis: the transform (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.TheTransform

open Real MeasureTheory Filter Topology Complex
open scoped FourierTransform

/-- The book's Fourier transform `f̂(k) = ∫ f(x) e^{-ikx} dx`. -/
noncomputable def ft (f : ℝ → ℂ) (k : ℝ) : ℂ := ∫ x : ℝ, f x * Complex.exp (-(k * x) * Complex.I)

/-! ## def-fourier-transform -/

/-- The book's normalisation in terms of mathlib's `𝓕 f (w) = ∫ e^{-2πi x w} f(x) dx`: `f̂(k) = 𝓕 f (k / 2π)`. -/
theorem ft_eq_fourier (f : ℝ → ℂ) (k : ℝ) : ft f k = 𝓕 f (k / (2 * π)) := by
  rw [Real.fourier_real_eq_integral_exp_smul, ft]
  congr 1; funext v
  rw [smul_eq_mul, mul_comm]
  congr 2
  have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  push_cast; field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_eq_fourier
-- test: fa-ft-eq-fourier-test.sage

/-- `|f̂(k)| ≤ ∫ |f|`. -/
theorem ft_norm_le (f : ℝ → ℂ) (k : ℝ) : ‖ft f k‖ ≤ ∫ x, ‖f x‖ := by
  refine (norm_integral_le_integral_norm _).trans (le_of_eq ?_)
  congr 1; funext x
  rw [norm_mul, show -((k : ℂ) * x) * Complex.I = ((-(k * x) : ℝ) : ℂ) * Complex.I by push_cast; ring,
    Complex.norm_exp_ofReal_mul_I, mul_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_norm_le
-- test: fa-ft-norm-le-test.sage

/-- The translation rule `(f(x - a))^ = e^{-ika} f̂`. -/
theorem ft_translate (f : ℝ → ℂ) (a k : ℝ) :
    ft (fun x => f (x - a)) k = Complex.exp (-(k * a) * Complex.I) * ft f k := by
  simp only [ft]
  have := integral_sub_right_eq_self (μ := (volume : Measure ℝ))
    (fun y => f y * Complex.exp (-(k * (y + a)) * Complex.I)) a
  rw [show (fun x : ℝ => f (x - a) * Complex.exp (-(k * x) * Complex.I)) =
      fun x : ℝ => (fun y => f y * Complex.exp (-(k * (y + a)) * Complex.I)) (x - a) by
    funext x; simp, this, ← integral_const_mul]
  congr 1; funext y
  rw [show -((k : ℂ) * ((y : ℂ) + (a : ℂ))) * Complex.I = -(k * a) * Complex.I + -(k * y) * Complex.I by
    ring, Complex.exp_add]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_translate
-- test: fa-ft-translate-test.sage

/-- The modulation rule `(e^{iax} f)^ = f̂(k - a)`. -/
theorem ft_modulate (f : ℝ → ℂ) (a k : ℝ) :
    ft (fun x => Complex.exp (a * x * Complex.I) * f x) k = ft f (k - a) := by
  simp only [ft]
  congr 1; funext x
  rw [mul_comm (Complex.exp _) (f x), mul_assoc, ← Complex.exp_add]
  congr 2; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_modulate
-- test: fa-ft-modulate-test.sage

/-- The scaling rule `(f(λx))^ = f̂(k/λ) / |λ|`. -/
theorem ft_scale (f : ℝ → ℂ) (l : ℝ) (hl : l ≠ 0) (k : ℝ) :
    ft (fun x => f (l * x)) k = (1 / |l| : ℝ) * ft f (k / l) := by
  simp only [ft]
  have hl' : (l : ℂ) ≠ 0 := by exact_mod_cast hl
  have := Measure.integral_comp_mul_left (fun y => f y * Complex.exp (-((k / l : ℝ) * y) * Complex.I)) l
  rw [show (∫ x, f (l * x) * Complex.exp (-(k * x) * Complex.I)) =
      ∫ x, (fun y => f y * Complex.exp (-((k / l : ℝ) * y) * Complex.I)) (l * x) by
    congr 1; funext x; congr 3; push_cast; field_simp]
  rw [this, abs_inv, Complex.real_smul]
  congr 1; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_scale
-- test: fa-ft-scale-test.sage

/-- The derivative rule `(f')^ = i k f̂` (through mathlib's `fourier_deriv`). -/
theorem ft_deriv (f : ℝ → ℂ) (hf : Integrable f) (hd : Differentiable ℝ f) (hf' : Integrable (deriv f))
    (k : ℝ) : ft (deriv f) k = Complex.I * k * ft f k := by
  rw [ft_eq_fourier, ft_eq_fourier, Real.fourier_deriv hf hd hf']
  simp only [smul_eq_mul]
  have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  push_cast; field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.ft_deriv
-- test: fa-ft-deriv-test.sage

/-! ## exm-ft-box -/

/-- The box transform, real part: `∫_{-a}^{a} cos kx dx = 2 sin(ak)/k`. -/
theorem box_transform (a k : ℝ) (hk : k ≠ 0) : ∫ x in (-a)..a, Real.cos (k * x) = 2 * Real.sin (a * k) / k := by
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) hk, integral_cos, smul_eq_mul, mul_neg,
    Real.sin_neg]
  rw [mul_comm k a]; field_simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.box_transform
-- test: fa-box-transform-test.sage

/-- The box transform has no imaginary part: `∫_{-a}^{a} sin kx dx = 0`. -/
theorem box_transform_im (a k : ℝ) : ∫ x in (-a)..a, Real.sin (k * x) = 0 := by
  by_cases hk : k = 0
  · simp [hk]
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hk, integral_sin, smul_eq_mul, mul_neg,
    Real.cos_neg]
  simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.box_transform_im
-- test: fa-box-transform-im-test.sage

/-! ## exm-ft-exponential -/

/-- The two-sided exponential and its two half-line pieces. -/
theorem two_sided_pieces (a : ℝ) (ha : 0 < a) (k : ℝ) :
    IntegrableOn (fun x : ℝ => (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I)) (Set.Iic 0) ∧
    IntegrableOn (fun x : ℝ => (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I)) (Set.Ioi 0) ∧
    ∫ x : ℝ, (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I) =
      1 / (a - k * Complex.I) + 1 / (a + k * Complex.I) := by
  set f : ℝ → ℂ := fun x => (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I)
  have hR : ∀ x ∈ Set.Ioi (0 : ℝ), f x = Complex.exp ((-a - k * Complex.I) * x) := by
    intro x hx
    simp only [f, abs_of_pos (Set.mem_Ioi.1 hx), Complex.ofReal_exp, ← Complex.exp_add]
    congr 1; push_cast; ring
  have hL : ∀ x ∈ Set.Iic (0 : ℝ), f x = Complex.exp ((a - k * Complex.I) * x) := by
    intro x hx
    simp only [f, abs_of_nonpos (Set.mem_Iic.1 hx), Complex.ofReal_exp, ← Complex.exp_add]
    congr 1; push_cast; ring
  have reR : (-a - k * Complex.I : ℂ).re < 0 := by simp; linarith
  have reL : 0 < (a - k * Complex.I : ℂ).re := by simp; linarith
  have iR : IntegrableOn f (Set.Ioi 0) :=
    (integrableOn_congr_fun hR measurableSet_Ioi).2 (integrableOn_exp_mul_complex_Ioi reR 0)
  have iL : IntegrableOn f (Set.Iic 0) :=
    (integrableOn_congr_fun hL measurableSet_Iic).2 (integrableOn_exp_mul_complex_Iic reL 0)
  refine ⟨iL, iR, ?_⟩
  rw [← intervalIntegral.integral_Iic_add_Ioi iL iR, setIntegral_congr_fun measurableSet_Ioi hR,
    setIntegral_congr_fun measurableSet_Iic hL, integral_exp_mul_complex_Ioi reR, integral_exp_mul_complex_Iic reL]
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero]
  have h1 : (-a - k * Complex.I : ℂ) ≠ 0 := fun h => by simpa [h] using reR
  rw [show -(1 : ℂ) / (-a - k * Complex.I) = 1 / (a + k * Complex.I) by
    rw [neg_div, ← div_neg]; congr 1; ring]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.two_sided_pieces
-- test: fa-two-sided-pieces-test.sage

theorem integrable_two_sided (a : ℝ) (ha : 0 < a) (k : ℝ) :
    Integrable (fun x : ℝ => (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I)) := by
  obtain ⟨iL, iR, -⟩ := two_sided_pieces a ha k
  rw [← integrableOn_univ, ← Set.Iic_union_Ioi (a := (0 : ℝ))]
  exact iL.union iR

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.integrable_two_sided
-- test: fa-integrable-two-sided-test.sage

/-- The two-sided exponential transforms to the Lorentzian: `∫ e^{-a|x|} e^{-ikx} dx = 2a / (a² + k²)`. -/
theorem exp_two_sided (a : ℝ) (ha : 0 < a) (k : ℝ) :
    ∫ x : ℝ, (Real.exp (-a * |x|) : ℂ) * Complex.exp (-(k * x) * Complex.I) = ((2 * a / (a ^ 2 + k ^ 2) : ℝ) : ℂ) := by
  rw [(two_sided_pieces a ha k).2.2]
  push_cast
  have h1 : ((a : ℂ) - k * Complex.I) ≠ 0 := fun h => by
    have := congrArg Complex.re h; simp at this; linarith
  have h2 : ((a : ℂ) + k * Complex.I) ≠ 0 := fun h => by
    have := congrArg Complex.re h; simp at this; linarith
  have h3 : ((a : ℂ) ^ 2 + k ^ 2) ≠ 0 := by exact_mod_cast (by positivity : (a : ℝ) ^ 2 + k ^ 2 ≠ 0)
  rw [div_add_div _ _ h1 h2, div_eq_div_iff (mul_ne_zero h1 h2) h3]
  ring_nf
  rw [Complex.I_sq]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.exp_two_sided
-- test: fa-exp-two-sided-test.sage

/-! ## exm-ft-gaussian, exm-ft-sage-assume -/

/-- The Gaussian transform `ĝ_σ(κ) = σ √(2π) e^{-σ²κ²/2}`, for complex `κ` as well. -/
theorem gaussian_transform (σ : ℝ) (hσ : 0 < σ) (κ : ℂ) :
    ∫ x : ℝ, Complex.exp (-(x : ℂ) ^ 2 / (2 * σ ^ 2)) * Complex.exp (-(κ * x) * Complex.I) =
      σ * Real.sqrt (2 * π) * Complex.exp (-(σ : ℂ) ^ 2 * κ ^ 2 / 2) := by
  have hb : 0 < (1 / (2 * (σ : ℂ) ^ 2)).re := by
    rw [show (1 / (2 * (σ : ℂ) ^ 2)) = ((1 / (2 * σ ^ 2) : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re]
    positivity
  have := fourierIntegral_gaussian hb (-κ)
  have hσ' : (σ : ℂ) ≠ 0 := by exact_mod_cast hσ.ne'
  rw [show (∫ x : ℝ, Complex.exp (-(x : ℂ) ^ 2 / (2 * σ ^ 2)) * Complex.exp (-(κ * x) * Complex.I)) =
      ∫ x : ℝ, Complex.exp (Complex.I * (-κ) * x) * Complex.exp (-(1 / (2 * (σ : ℂ) ^ 2)) * x ^ 2) by
    congr 1; funext x; rw [mul_comm]; congr 2 <;> ring, this]
  have e1 : (π : ℂ) / (1 / (2 * (σ : ℂ) ^ 2)) = ((2 * π * σ ^ 2 : ℝ) : ℂ) := by push_cast; field_simp
  have e2 : (((2 * π * σ ^ 2 : ℝ) : ℂ)) ^ (1 / 2 : ℂ) = ((σ * Real.sqrt (2 * π) : ℝ) : ℂ) := by
    rw [show (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) by push_cast; ring, ← Complex.ofReal_cpow (by positivity)]
    congr 1
    rw [← Real.sqrt_eq_rpow, Real.sqrt_mul' _ (by positivity), Real.sqrt_sq hσ.le]; ring
  rw [e1, e2]
  push_cast
  congr 1
  congr 1
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.gaussian_transform
-- test: fa-gaussian-transform-test.sage

/-- For `σ = 1`: `e^{-x²/2}` is an eigenfunction, `ĝ = √(2π) g`. -/
theorem gaussian_eigen (k : ℝ) :
    ft (fun x => Complex.exp (-(x : ℂ) ^ 2 / 2)) k = Real.sqrt (2 * π) * Complex.exp (-(k : ℂ) ^ 2 / 2) := by
  have := gaussian_transform 1 one_pos k
  simp only [ft]
  rw [show (fun x : ℝ => Complex.exp (-(x : ℂ) ^ 2 / 2) * Complex.exp (-(k * x) * Complex.I)) =
      fun x : ℝ => Complex.exp (-(x : ℂ) ^ 2 / (2 * ((1 : ℝ) : ℂ) ^ 2)) * Complex.exp (-((k : ℂ) * x) * Complex.I) by
    funext x; congr 2; push_cast; ring, this]
  push_cast; ring_nf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.gaussian_eigen
-- test: fa-gaussian-eigen-test.sage

/-! ## exr-ft-rules -/

/-- `(e^{-(x-3)²/2})^ = e^{-3ik} √(2π) e^{-k²/2}` by the translation rule. -/
theorem shifted_gaussian (k : ℝ) :
    ft (fun x => Complex.exp (-((x - 3 : ℝ) : ℂ) ^ 2 / 2)) k =
      Complex.exp (-((k : ℂ) * ((3 : ℝ) : ℂ)) * Complex.I) * (Real.sqrt (2 * π) * Complex.exp (-(k : ℂ) ^ 2 / 2)) := by
  rw [← gaussian_eigen k]
  exact ft_translate (fun x => Complex.exp (-(x : ℂ) ^ 2 / 2)) 3 k

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.shifted_gaussian
-- test: fa-shifted-gaussian-test.sage

/-- `(e^{-2|x|} cos 5x)^ = 2/(4 + (k-5)²) + 2/(4 + (k+5)²)` by the modulation rule. -/
theorem damped_cos_transform (k : ℝ) :
    ∫ x : ℝ, (Real.exp (-2 * |x|) * Real.cos (5 * x) : ℝ) * Complex.exp (-(k * x) * Complex.I) =
      ((2 / (4 + (k - 5) ^ 2) + 2 / (4 + (k + 5) ^ 2) : ℝ) : ℂ) := by
  have h1 := exp_two_sided 2 two_pos (k - 5)
  have h2 := exp_two_sided 2 two_pos (k + 5)
  have hint : ∀ c : ℝ, Integrable (fun x : ℝ => (Real.exp (-2 * |x|) : ℂ) * Complex.exp (-(c * x) * Complex.I)) :=
    fun c => integrable_two_sided 2 two_pos c
  have hpt : ∀ x : ℝ, ((Real.exp (-2 * |x|) * Real.cos (5 * x) : ℝ) : ℂ) * Complex.exp (-(k * x) * Complex.I) =
      (1 / 2 : ℂ) * ((Real.exp (-2 * |x|) : ℂ) * Complex.exp (-(((k - 5 : ℝ) : ℂ) * x) * Complex.I)) +
      (1 / 2 : ℂ) * ((Real.exp (-2 * |x|) : ℂ) * Complex.exp (-(((k + 5 : ℝ) : ℂ) * x) * Complex.I)) := by
    intro x
    have hc : ((Real.cos (5 * x) : ℝ) : ℂ) =
        (Complex.exp (((5 * x : ℝ) : ℂ) * Complex.I) + Complex.exp (((-(5 * x) : ℝ) : ℂ) * Complex.I)) / 2 := by
      rw [Complex.exp_mul_I, Complex.exp_mul_I]; push_cast
      simp only [Complex.cos_neg, Complex.sin_neg]; ring
    have e1 : Complex.exp (((5 * x : ℝ) : ℂ) * Complex.I) * Complex.exp (-(k * x) * Complex.I) =
        Complex.exp (-(((k - 5 : ℝ) : ℂ) * x) * Complex.I) := by
      rw [← Complex.exp_add]; congr 1; push_cast; ring
    have e2 : Complex.exp (((-(5 * x) : ℝ) : ℂ) * Complex.I) * Complex.exp (-(k * x) * Complex.I) =
        Complex.exp (-(((k + 5 : ℝ) : ℂ) * x) * Complex.I) := by
      rw [← Complex.exp_add]; congr 1; push_cast; ring
    rw [Complex.ofReal_mul, hc]
    linear_combination ((Real.exp (-2 * |x|) : ℂ) / 2) * e1 + ((Real.exp (-2 * |x|) : ℂ) / 2) * e2
  simp_rw [hpt]
  have i1 := (hint (k - 5)).const_mul (1 / 2 : ℂ)
  have i2 := (hint (k + 5)).const_mul (1 / 2 : ℂ)
  rw [integral_add i1 i2, integral_const_mul, integral_const_mul, h1, h2]
  push_cast
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.damped_cos_transform
-- test: fa-damped-cos-transform-test.sage

/-! ## exr-ft-one-sided -/

/-- The one-sided exponential: `∫₀^∞ e^{-ax} e^{-iκx} dx = 1/(a + iκ)` whenever `Im κ < a` (stated multiplied
through by `a + iκ`). -/
theorem one_sided (a : ℝ) (κ : ℂ) (h : κ.im < a) :
    ((a : ℂ) + Complex.I * κ) * ∫ x in Set.Ioi (0 : ℝ), (Real.exp (-a * x) : ℂ) * Complex.exp (-(κ * x) * Complex.I) = 1 := by
  have hre : (-(a + Complex.I * κ) : ℂ).re < 0 := by simp; linarith
  have hne : (a + Complex.I * κ : ℂ) ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0; simp at this; linarith
  rw [setIntegral_congr_fun measurableSet_Ioi (g := fun x : ℝ => Complex.exp (-(a + Complex.I * κ) * x))
    (fun x _ => by simp only [Complex.ofReal_exp, ← Complex.exp_add]; congr 1; push_cast; ring),
    integral_exp_mul_complex_Ioi hre]
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.one_sided
-- test: fa-one-sided-test.sage

/-- Its real and imaginary parts: `1/(a + ik) = (a - ik)/(a² + k²)`, since `(a + ik)(a - ik) = a² + k²`. -/
theorem one_sided_parts (a k : ℝ) :
    ((a : ℂ) + Complex.I * k) * (a - Complex.I * k) = ((a ^ 2 + k ^ 2 : ℝ) : ℂ) := by
  push_cast; ring_nf; rw [Complex.I_sq]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.one_sided_parts
-- test: fa-one-sided-parts-test.sage

/-! ## exr-ft-gaussian-ode -/

/-- `g = e^{-x²/2σ²}` satisfies `σ² g' + x g = 0`. -/
theorem gaussian_ode (σ x : ℝ) (hσ : σ ≠ 0) :
    deriv (fun x : ℝ => Real.exp (-x ^ 2 / (2 * σ ^ 2))) x = -(x / σ ^ 2) * Real.exp (-x ^ 2 / (2 * σ ^ 2)) := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun x => rexp (-x ^ 2 / (2 * σ ^ 2))) ((-x * (Real.exp (-(x ^ 2) / (2 * (σ ^ 2)))) / (σ ^ 2) : ℝ))
          (x) :=
        by
        convert ((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-1 / (2 * (σ ^ 2)) : ℝ)).exp) using 1
        all_goals
          (simp only [Pi.pow_apply]; first
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (norm_num; done))
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv.sage

/-- `ĝ = C e^{-σ²k²/2}` (with `C = σ √(2π)`) satisfies the transformed equation `ĝ' = -σ² k ĝ`. -/
theorem gaussian_transform_ode (C σ k : ℝ) :
    deriv (fun k : ℝ => C * Real.exp (-σ ^ 2 * k ^ 2 / 2)) k =
      -σ ^ 2 * k * (C * Real.exp (-σ ^ 2 * k ^ 2 / 2)) := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun k => C * rexp (-σ ^ 2 * k ^ 2 / 2))
          ((-(σ ^ 2) * k * C * (Real.exp (-(σ ^ 2) * (k ^ 2) / 2)) : ℝ)) (k) :=
        by
        convert (((((hasDerivAt_id' (k : ℝ)).pow 2).const_mul (-(σ ^ 2) / 2 : ℝ)).exp).const_mul (C : ℝ)) using 1
        all_goals (simp only [Pi.pow_apply]; (ring_nf; done))
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv.sage

/-- The Gaussian integral for complex `a` with `Re a > 0` (mathlib): `∫ e^{-ax²} = (π/a)^{1/2}`. -/
theorem gaussian_complex (a b : ℂ) (ha : 0 < a.re) (hab : a * b = 1) :
    ∫ x : ℝ, Complex.exp (-a * (x : ℂ) ^ 2) = (π * b) ^ (1 / 2 : ℂ) := by
  have hb : b = a⁻¹ := eq_inv_of_mul_eq_one_right hab
  subst hb
  rw [← div_eq_mul_inv]
  exact integral_gaussian_complex ha

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TheTransform.gaussian_complex
-- test: fa-gaussian-complex-test.sage

end KaluzaKlein.FourierAnalysis.TheTransform

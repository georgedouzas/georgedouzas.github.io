import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Plancherel

/-! # Fourier analysis: uncertainty (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Uncertainty

open Real MeasureTheory Filter Topology Complex

/-! ## thm-uncertainty, exr-ft-uncertainty-proof -/

/-- The core of the proof: if `‖f‖² = -2 Re⟪X, D⟫` (the integration by parts, with `X = x f`, `D = f'`), then
Cauchy-Schwarz gives `‖f‖² ≤ 2 ‖X‖ ‖D‖`, which is `Δx Δk ≥ 1/2`. -/
theorem uncertainty_core {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] (f X D : E)
    (h : ‖f‖ ^ 2 = -2 * (inner ℂ X D).re) : ‖f‖ ^ 2 ≤ 2 * ‖X‖ * ‖D‖ := by
  rw [h]
  have h1 : -(inner ℂ X D).re ≤ ‖inner ℂ X D‖ := by
    have := Complex.abs_re_le_norm (inner ℂ X D)
    linarith [neg_abs_le (inner ℂ X D).re]
  have h2 : ‖inner ℂ X D‖ ≤ ‖X‖ * ‖D‖ := norm_inner_le_norm X D
  nlinarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.uncertainty_core
-- test: fa-uncertainty-core-test.sage

/-- The extremal functions: a solution of `f' = λ x f` is `f(x) = f(0) e^{λx²/2}`, a Gaussian for `λ < 0`. -/
theorem extremal_gaussian (lam : ℝ) (f : ℝ → ℝ) (hf : ∀ x, HasDerivAt f (lam * x * f x) x) (x : ℝ) :
    f x = f 0 * Real.exp (lam * x ^ 2 / 2) := by
  set g : ℝ → ℝ := fun x => f x * Real.exp (-(lam * x ^ 2 / 2))
  have hg : ∀ x, HasDerivAt g 0 x := by
    intro x
    have e := ((hasDerivAt_pow 2 x).const_mul lam |>.div_const 2 |>.neg).exp
    convert (hf x).mul e using 1
    simp; ring
  have hc : g x = g 0 := is_const_of_deriv_eq_zero (fun x => (hg x).differentiableAt) (fun x => (hg x).deriv) x 0
  simp only [g, zero_pow two_ne_zero, mul_zero, zero_div, neg_zero, Real.exp_zero, mul_one] at hc
  rw [← hc, mul_assoc, ← Real.exp_add]; simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.extremal_gaussian
-- test: fa-extremal-gaussian-test.sage

/-- The Gaussian solves the equality condition: `(e^{-cx²/2})' = -c x e^{-cx²/2}`. -/
theorem gaussian_equality_case (c x : ℝ) :
    deriv (fun x : ℝ => Real.exp (-c * x ^ 2 / 2)) x = -c * x * Real.exp (-c * x ^ 2 / 2) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => rexp (-c * x ^ 2 / 2)) ((-c * x * (Real.exp (-c * (x ^ 2) / 2)) : ℝ)) (x) :=
        by
        convert ((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-c / 2 : ℝ)).exp) using 1
        all_goals (simp only [Pi.pow_apply]; (ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: fa-deriv.sage

/-- The uncentred case: translating and modulating moves the means, `|e^{ibx} f(x - a)| = |f(x - a)|`. -/
theorem modulation_modulus (b x : ℝ) (z : ℂ) : ‖Complex.exp (b * x * Complex.I) * z‖ = ‖z‖ := by
  rw [norm_mul, show (b : ℂ) * x * Complex.I = ((b * x : ℝ) : ℂ) * Complex.I by push_cast; ring,
    Complex.norm_exp_ofReal_mul_I, one_mul]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.modulation_modulus
-- test: fa-modulation-modulus-test.sage

/-! ## exm-ft-uncertainty-gaussian, exm-ft-sage-widths -/

/-- The second moment `∫ x² e^{-x²/σ²} = σ³√π/2`, from the Gamma integral. -/
theorem gaussian_second_moment (σ : ℝ) (hσ : 0 < σ) : ∫ x : ℝ, x ^ 2 * Real.exp (-x ^ 2 / σ ^ 2) = σ ^ 3 * Real.sqrt π / 2 := by
  have hb : 0 < 1 / σ ^ 2 := by positivity
  have h := integral_comp_abs (f := fun x : ℝ => x ^ 2 * Real.exp (-x ^ 2 / σ ^ 2))
  simp only [sq_abs] at h
  rw [h]
  have hI := _root_.integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := 2) (b := 1 / σ ^ 2) (by norm_num) (by norm_num) hb
  have hpow : (1 / σ ^ 2) ^ (-((2 : ℝ) + 1) / 2) = σ ^ 3 := by
    rw [one_div, Real.inv_rpow (by positivity), ← Real.rpow_neg (by positivity),
      show -(-((2 : ℝ) + 1) / 2) = 3 / 2 by norm_num, ← Real.rpow_natCast σ 2, ← Real.rpow_mul hσ.le]
    norm_num
  rw [show (∫ x in Set.Ioi (0 : ℝ), x ^ 2 * Real.exp (-x ^ 2 / σ ^ 2)) =
      ∫ x in Set.Ioi (0 : ℝ), x ^ (2 : ℝ) * Real.exp (-(1 / σ ^ 2) * x ^ (2 : ℝ)) by
    apply setIntegral_congr_fun measurableSet_Ioi; intro x _
    simp only [Real.rpow_two]; ring_nf, hI, hpow, show ((2 : ℝ) + 1) / 2 = 1 / 2 + 1 by norm_num,
    Real.Gamma_add_one (by norm_num), Real.Gamma_one_half_eq]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.gaussian_second_moment
-- test: fa-gaussian-second-moment-test.sage

/-- The Gaussian's widths: `(Δx)² = σ²/2`, `(Δk)² = 1/(2σ²)`, and `Δx Δk = 1/2`. -/
theorem gaussian_widths (σ : ℝ) (hσ : 0 < σ) :
    (∫ x : ℝ, x ^ 2 * Real.exp (-x ^ 2 / σ ^ 2)) / (∫ x : ℝ, Real.exp (-x ^ 2 / σ ^ 2)) = σ ^ 2 / 2 ∧
    (∫ k : ℝ, k ^ 2 * Real.exp (-k ^ 2 / (1 / σ) ^ 2)) / (∫ k : ℝ, Real.exp (-k ^ 2 / (1 / σ) ^ 2)) = 1 / (2 * σ ^ 2) ∧
    Real.sqrt (σ ^ 2 / 2) * Real.sqrt (1 / (2 * σ ^ 2)) = 1 / 2 := by
  have hs : 0 < Real.sqrt π := Real.sqrt_pos.2 Real.pi_pos
  have hs' : Real.sqrt π ≠ 0 := hs.ne'
  have hσ' : σ ≠ 0 := hσ.ne'
  refine ⟨?_, ?_, ?_⟩
  · rw [gaussian_second_moment σ hσ, KaluzaKlein.FourierAnalysis.Plancherel.gaussian_l2 σ hσ]
    fa_fc
  · rw [gaussian_second_moment (1 / σ) (by positivity), KaluzaKlein.FourierAnalysis.Plancherel.gaussian_l2 (1 / σ) (by positivity)]
    fa_fc
  · rw [← Real.sqrt_mul (by positivity), show σ ^ 2 / 2 * (1 / (2 * σ ^ 2)) = (1 / 2) ^ 2 by fa_fc,
      Real.sqrt_sq (by norm_num)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.gaussian_widths
-- test: fa-gaussian-widths-test.sage

/-! ## exm-ft-uncertainty-exponential -/

/-- `∫ x² e^{-2a|x|} = 1/(2a³)`. -/
theorem exp_second_moment (a : ℝ) (ha : 0 < a) : ∫ x : ℝ, x ^ 2 * Real.exp (-(2 * a) * |x|) = 1 / (2 * a ^ 3) := by
  have h := integral_comp_abs (f := fun x : ℝ => x ^ 2 * Real.exp (-(2 * a) * x))
  simp only [sq_abs] at h
  rw [h]
  have := integral_rpow_mul_exp_neg_mul_Ioi (a := 3) (r := 2 * a) (by norm_num) (by positivity)
  rw [show (∫ x in Set.Ioi (0 : ℝ), x ^ 2 * Real.exp (-(2 * a) * x)) =
      ∫ t in Set.Ioi (0 : ℝ), t ^ ((3 : ℝ) - 1) * Real.exp (-(2 * a * t)) by
    apply setIntegral_congr_fun measurableSet_Ioi; intro x _
    simp only; rw [show (3 : ℝ) - 1 = 2 by norm_num, Real.rpow_two]; ring_nf, this]
  rw [show (3 : ℝ) = 2 + 1 by norm_num, Real.Gamma_add_one (by norm_num), Real.Gamma_two]
  rw [show (2 : ℝ) + 1 = (3 : ℕ) by norm_num, Real.rpow_natCast]
  have ha' : a ≠ 0 := ha.ne'
  fa_fc

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.exp_second_moment
-- test: fa-exp-second-moment-test.sage

/-- The exponential's product: `(Δx)² (Δk)² = (1/(2a²)) a² = 1/2`, so `Δx Δk = 1/√2 > 1/2`. -/
theorem exp_widths (a : ℝ) (ha : 0 < a) :
    Real.sqrt (1 / (2 * a ^ 2)) * Real.sqrt (a ^ 2) = 1 / Real.sqrt 2 ∧ 1 / 2 < 1 / Real.sqrt 2 := by
  constructor
  · rw [← Real.sqrt_mul (by positivity), show 1 / (2 * a ^ 2) * a ^ 2 = 1 / 2 by field_simp, Real.sqrt_div' _ (by norm_num),
      Real.sqrt_one]
  · rw [one_div_lt_one_div (by norm_num) (Real.sqrt_pos.2 (by norm_num))]
    rw [show (2 : ℝ) = Real.sqrt 4 by rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.exp_widths
-- test: fa-exp-widths-test.sage

/-! ## exm-ft-uncertainty-box -/

/-- The box has infinite `Δk`: `k² |χ̂_a|² = 4 sin² ak` is not integrable. -/
theorem box_width_infinite (a : ℝ) (ha : a ≠ 0) : ¬ Integrable (fun k : ℝ => 4 * Real.sin (a * k) ^ 2) := by
  intro h
  have hab : 0 < |a| := abs_pos.2 ha
  have hn : ∀ n : ℕ, ∫ k in (0 : ℝ)..(n * π / |a|), 4 * Real.sin (a * k) ^ 2 = 2 * n * π / |a| := by
    intro n
    rw [intervalIntegral.integral_const_mul,
      intervalIntegral.integral_comp_mul_left (fun x => Real.sin x ^ 2) ha, integral_sin_sq, smul_eq_mul]
    rcases abs_choice a with h | h <;> rw [h]
    · rw [show a * (n * π / a) = n * π by field_simp]
      simp only [mul_zero, Real.sin_zero, Real.sin_nat_mul_pi, zero_mul, sub_zero]
      fa_fc
    · rw [show a * (n * π / -a) = -(n * π) by field_simp]
      simp only [mul_zero, Real.sin_zero, Real.sin_neg, Real.sin_nat_mul_pi, neg_zero, zero_mul, sub_zero]
      fa_fc
  have hbound : ∀ n : ℕ, ∫ k in (0 : ℝ)..(n * π / |a|), 4 * Real.sin (a * k) ^ 2 ≤ ∫ k, 4 * Real.sin (a * k) ^ 2 := by
    intro n
    rw [intervalIntegral.integral_of_le (by positivity)]
    exact setIntegral_le_integral h (Eventually.of_forall fun k => by simp only [Pi.zero_apply]; positivity)
  obtain ⟨n, hn'⟩ := exists_nat_gt ((∫ k, 4 * Real.sin (a * k) ^ 2) * |a| / (2 * π))
  have := hbound n
  rw [hn n, div_le_iff₀ hab] at this
  rw [div_lt_iff₀ (by positivity)] at hn'
  nlinarith [Real.pi_pos]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.box_width_infinite
-- test: fa-box-width-infinite-test.sage

/-! ## exr-ft-uncertainty-hermite -/

/-- The Hermite functions' widths: `Δx = Δk = √(3/2)` for `x e^{-x²/2}` and `√(5/2)` for `(2x² - 1) e^{-x²/2}`, as
stated (tested, not proved). -/
def HermiteWidths : Prop :=
  ∀ c : ℝ, c * ((∫ x : ℝ, x ^ 4 * Real.exp (-x ^ 2)) / (∫ x : ℝ, x ^ 2 * Real.exp (-x ^ 2))) = c * (3 / 2) ∧
    c * ((∫ x : ℝ, x ^ 2 * (2 * x ^ 2 - 1) ^ 2 * Real.exp (-x ^ 2)) /
      (∫ x : ℝ, (2 * x ^ 2 - 1) ^ 2 * Real.exp (-x ^ 2))) = c * (5 / 2)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.HermiteWidths
-- test: fa-hermite-widths-test.sage

/-! ## exr-ft-uncertainty-scaling -/

/-- Scaling: `∫ x² g(λx) dx = (1/|λ|³) ∫ u² g(u) du`, so `(Δx)²` scales by `1/λ²` and `(Δk)²` by `λ²`. -/
theorem moment_scaling (g : ℝ → ℝ) (l : ℝ) (hl : l ≠ 0) :
    ∫ x : ℝ, x ^ 2 * g (l * x) = |l⁻¹| * (1 / l ^ 2) * ∫ u : ℝ, u ^ 2 * g u := by
  have := Measure.integral_comp_mul_left (fun u : ℝ => (u / l) ^ 2 * g u) l
  rw [show (fun x : ℝ => x ^ 2 * g (l * x)) = fun x : ℝ => (fun u : ℝ => (u / l) ^ 2 * g u) (l * x) by
    funext x; simp only; field_simp, this, smul_eq_mul]
  rw [show (fun u : ℝ => (u / l) ^ 2 * g u) = fun u => (1 / l ^ 2) * (u ^ 2 * g u) by funext u; field_simp,
    integral_const_mul]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Uncertainty.moment_scaling
-- test: fa-moment-scaling-test.sage

/-- The product of the widths is scale invariant: `(A/λ²)(λ² B) = A B`. -/
theorem width_product_invariant (A B l : ℝ) (hl : l ≠ 0) : (A / l ^ 2) * (l ^ 2 * B) = A * B := by
  field_simp

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Uncertainty.width_product_invariant (1) (2) (1) (one_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Uncertainty.width_product_invariant (3/2) (-1) (2) (two_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.Uncertainty.width_product_invariant (0) (5) (-3) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.Uncertainty.width_product_invariant
-- test: fa-width-product-invariant-test.sage

end KaluzaKlein.FourierAnalysis.Uncertainty

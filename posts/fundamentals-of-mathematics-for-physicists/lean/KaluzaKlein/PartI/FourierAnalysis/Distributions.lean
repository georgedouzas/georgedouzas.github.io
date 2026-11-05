import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.TestFunctions

/-! # Fourier analysis: distributions (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Distributions

open Real MeasureTheory Filter Topology

/-- The Dirac delta, `δ(φ) = φ(0)`. -/
def delta (φ : ℝ → ℝ) : ℝ := φ 0

/-! ## def-distribution -/

/-- The delta is linear. -/
theorem delta_linear (a b : ℝ) (φ ψ : ℝ → ℝ) : delta (fun x => a * φ x + b * ψ x) = a * delta φ + b * delta ψ := rfl

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.delta_linear
-- test: fa-delta-linear-test.sage

/-- The delta is continuous: `|δφ - δψ| ≤ sup |φ - ψ|`. -/
theorem delta_continuous (φ ψ : ℝ → ℝ) (C : ℝ) (h : ∀ x, |φ x - ψ x| ≤ C) : |delta φ - delta ψ| ≤ C := h 0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.delta_continuous
-- test: fa-delta-continuous-test.sage

/-- `δ_a(φ) = φ(a)` is `δ` composed with the translation. -/
theorem delta_shift (a : ℝ) (φ : ℝ → ℝ) : delta (fun x => φ (x + a)) = φ a := by simp [delta]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.delta_shift
-- test: fa-delta-shift-test.sage

/-! ## exm-dist-regular -/

/-- The tempered bound `∫ |f φ| ≤ sup |(1 + x²)^N φ| · ∫ |f| (1 + x²)^{-N}`. -/
theorem tempered_bound (f φ : ℝ → ℝ) (N : ℕ) (C : ℝ) (hC : ∀ x, |(1 + x ^ 2) ^ N * φ x| ≤ C)
    (hf : Integrable (fun x => |f x| / (1 + x ^ 2) ^ N)) (hfφ : Integrable (fun x => f x * φ x)) :
    ∫ x, |f x * φ x| ≤ C * ∫ x, |f x| / (1 + x ^ 2) ^ N := by
  rw [← integral_const_mul]
  apply integral_mono hfφ.abs (hf.const_mul C)
  intro x
  have hp : 0 < (1 + x ^ 2) ^ N := by positivity
  have h1 : |φ x| ≤ C / (1 + x ^ 2) ^ N := by
    rw [le_div_iff₀ hp, mul_comm]
    calc (1 + x ^ 2) ^ N * |φ x| = |(1 + x ^ 2) ^ N * φ x| := by rw [abs_mul, abs_of_pos hp]
      _ ≤ C := hC x
  simp only
  rw [abs_mul]
  calc |f x| * |φ x| ≤ |f x| * (C / (1 + x ^ 2) ^ N) := by gcongr
    _ = C * (|f x| / (1 + x ^ 2) ^ N) := by ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.tempered_bound
-- test: fa-tempered-bound-test.sage

/-- `e^{x²}` is not tempered: against the Schwartz function `e^{-x²/2}` the integrand is `≥ 1`. -/
theorem exp_sq_not_tempered (c : ℝ) (hc : c ≠ 0) :
    ¬ Integrable (fun x : ℝ => c * (Real.exp (x ^ 2) * Real.exp (-x ^ 2 / 2))) := by
  intro h
  have h1 : Integrable (fun _ : ℝ => |c|) := by
    refine h.abs.mono' aestronglyMeasurable_const (Eventually.of_forall fun x => ?_)
    rw [Real.norm_eq_abs, abs_abs]
    rw [abs_mul]
    have : 1 ≤ |Real.exp (x ^ 2) * Real.exp (-x ^ 2 / 2)| := by
      rw [← Real.exp_add, abs_of_pos (Real.exp_pos _)]
      exact Real.one_le_exp (by nlinarith [sq_nonneg x])
    nlinarith [abs_nonneg c]
  rw [integrable_const_iff] at h1
  rcases h1 with h1 | h1
  · exact hc (abs_eq_zero.1 h1)
  · have := h1.measure_univ_lt_top
    simp at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.exp_sq_not_tempered
-- test: fa-exp-sq-not-tempered-test.sage

/-- `e^{|x|}` is not tempered: against the Schwartz function `e^{-√(1+x²)}` the integrand is `≥ e^{-1}`. -/
theorem exp_abs_not_tempered (c : ℝ) (hc : c ≠ 0) :
    ¬ Integrable (fun x : ℝ => c * (Real.exp |x| * Real.exp (-Real.sqrt (1 + x ^ 2)))) := by
  intro h
  have h1 : Integrable (fun _ : ℝ => |c| * Real.exp (-1)) := by
    refine h.abs.mono' aestronglyMeasurable_const (Eventually.of_forall fun x => ?_)
    rw [Real.norm_eq_abs, abs_of_pos (by positivity), abs_mul]
    have hs : Real.sqrt (1 + x ^ 2) ≤ |x| + 1 := by
      rw [Real.sqrt_le_left (by positivity)]
      nlinarith [abs_mul_abs_self x, abs_nonneg x]
    have : Real.exp (-1) ≤ |Real.exp |x| * Real.exp (-Real.sqrt (1 + x ^ 2))| := by
      rw [← Real.exp_add, abs_of_pos (Real.exp_pos _)]
      exact Real.exp_le_exp.2 (by linarith)
    exact mul_le_mul_of_nonneg_left this (abs_nonneg c)
  rw [integrable_const_iff] at h1
  rcases h1 with h1 | h1
  · rcases mul_eq_zero.1 h1 with h2 | h2
    · exact hc (abs_eq_zero.1 h2)
    · exact (Real.exp_pos _).ne' h2
  · have := h1.measure_univ_lt_top
    simp at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.exp_abs_not_tempered
-- test: fa-exp-abs-not-tempered-test.sage

/-- `1/x` is not locally integrable at `0`: no regular distribution. -/
theorem inv_not_locally_integrable (b : ℝ) (hb : 0 < b) : ¬ IntervalIntegrable (fun x : ℝ => x⁻¹) volume 0 b := by
  rw [intervalIntegrable_inv_iff]
  push_neg
  exact ⟨hb.ne, Set.left_mem_uIcc⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.inv_not_locally_integrable
-- test: fa-inv-not-locally-integrable-test.sage

/-- `x^s` is locally integrable for `s > -1`. -/
theorem rpow_locally_integrable (s b : ℝ) (hs : -1 < s) : IntervalIntegrable (fun x : ℝ => x ^ s) volume 0 b :=
  intervalIntegral.intervalIntegrable_rpow' hs

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.rpow_locally_integrable
-- test: fa-rpow-locally-integrable-test.sage

/-! ## exm-dist-delta-limits, exr-dist-delta-limits -/

/-- The Gaussians `g_ε = e^{-x²/ε²}/(ε√π)` have unit integral. -/
theorem gauss_unit (ε : ℝ) (hε : 0 < ε) : ∫ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) = 1 := by
  rw [integral_div, show (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2)) = fun x : ℝ => Real.exp (-(1 / ε ^ 2) * x ^ 2) by
    funext x; ring_nf, integral_gaussian]
  rw [show π / (1 / ε ^ 2) = ε ^ 2 * π by field_simp, Real.sqrt_mul (by positivity), Real.sqrt_sq hε.le]
  have : 0 < Real.sqrt π := Real.sqrt_pos.2 Real.pi_pos
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.gauss_unit
-- test: fa-gauss-unit-test.sage

/-- `∫ g_ε(x) cos(cx) dx = e^{-c²ε²/4}`: for `c = 1, 2` the limits `e^{-ε²/4}, e^{-ε²} → 1 = cos 0`. -/
theorem gauss_cos (ε c : ℝ) (hε : 0 < ε) :
    ∫ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) * Real.cos (c * x) = Real.exp (-c ^ 2 * ε ^ 2 / 4) := by
  have hb : 0 < (1 / (ε : ℂ) ^ 2).re := by
    rw [show (1 / (ε : ℂ) ^ 2) = ((1 / ε ^ 2 : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re]; positivity
  have hF := fourierIntegral_gaussian hb c
  have hre := congrArg Complex.re hF
  have hint : Integrable (fun x : ℝ => Complex.exp (Complex.I * c * x) * Complex.exp (-(1 / (ε : ℂ) ^ 2) * (x : ℂ) ^ 2)) := by
    have := integrable_cexp_quadratic hb (Complex.I * c) 0
    convert this using 1; funext x; rw [← Complex.exp_add]; congr 1; ring
  have hcomm := Complex.reCLM.integral_comp_comm hint
  simp only [Complex.reCLM_apply] at hcomm
  rw [← hcomm] at hre
  have hpt : ∀ x : ℝ, (Complex.exp (Complex.I * c * x) * Complex.exp (-(1 / (ε : ℂ) ^ 2) * (x : ℂ) ^ 2)).re =
      Real.exp (-x ^ 2 / ε ^ 2) * Real.cos (c * x) := by
    intro x
    rw [show Complex.I * c * x = ((c * x : ℝ) : ℂ) * Complex.I by push_cast; ring,
      show -(1 / (ε : ℂ) ^ 2) * (x : ℂ) ^ 2 = ((-x ^ 2 / ε ^ 2 : ℝ) : ℂ) by push_cast; ring,
      ← Complex.ofReal_exp, Complex.re_mul_ofReal, Complex.exp_ofReal_mul_I_re]
    ring
  simp_rw [hpt] at hre
  have e1 : (π / (1 / (ε : ℂ) ^ 2)) ^ (1 / 2 : ℂ) = ((ε * Real.sqrt π : ℝ) : ℂ) := by
    rw [show (π : ℂ) / (1 / (ε : ℂ) ^ 2) = ((ε ^ 2 * π : ℝ) : ℂ) by push_cast; field_simp,
      show (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) by push_cast; ring, ← Complex.ofReal_cpow (by positivity)]
    congr 1
    rw [← Real.sqrt_eq_rpow, Real.sqrt_mul (by positivity), Real.sqrt_sq hε.le]
  have e2 : -(c : ℂ) ^ 2 / (4 * (1 / (ε : ℂ) ^ 2)) = ((-c ^ 2 * ε ^ 2 / 4 : ℝ) : ℂ) := by
    push_cast; field_simp
  rw [e1, e2, ← Complex.ofReal_exp, ← Complex.ofReal_mul, Complex.ofReal_re] at hre
  have hs : 0 < ε * Real.sqrt π := by positivity
  simp_rw [div_mul_eq_mul_div, integral_div, hre]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.gauss_cos
-- test: fa-gauss-cos-test.sage

/-- `∫ g_ε (x² + 1) = 1 + ε²/2`, from the second moment `∫ x² e^{-x²/ε²} = ε³√π/2`. -/
theorem gauss_poly (ε : ℝ) (hε : 0 < ε) :
    ∫ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) * (x ^ 2 + 1) = 1 + ε ^ 2 / 2 := by
  have hm : ∫ x : ℝ, x ^ 2 * Real.exp (-x ^ 2 / ε ^ 2) = ε ^ 3 * Real.sqrt π / 2 := by
    have hb : 0 < 1 / ε ^ 2 := by positivity
    have h := integral_comp_abs (f := fun x : ℝ => x ^ 2 * Real.exp (-x ^ 2 / ε ^ 2))
    simp only [sq_abs] at h
    rw [h]
    have hI := _root_.integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := 2) (b := 1 / ε ^ 2) (by norm_num) (by norm_num) hb
    have hpow : (1 / ε ^ 2) ^ (-((2 : ℝ) + 1) / 2) = ε ^ 3 := by
      rw [one_div, Real.inv_rpow (by positivity), ← Real.rpow_neg (by positivity),
        show -(-((2 : ℝ) + 1) / 2) = 3 / 2 by norm_num, ← Real.rpow_natCast ε 2, ← Real.rpow_mul hε.le]
      norm_num
    rw [show (∫ x in Set.Ioi (0 : ℝ), x ^ 2 * Real.exp (-x ^ 2 / ε ^ 2)) =
        ∫ x in Set.Ioi (0 : ℝ), x ^ (2 : ℝ) * Real.exp (-(1 / ε ^ 2) * x ^ (2 : ℝ)) by
      apply setIntegral_congr_fun measurableSet_Ioi; intro x _
      simp only [Real.rpow_two]; ring_nf, hI, hpow, show ((2 : ℝ) + 1) / 2 = 1 / 2 + 1 by norm_num,
      Real.Gamma_add_one (by norm_num), Real.Gamma_one_half_eq]
    ring
  have hi1 : Integrable (fun x : ℝ => x ^ 2 * Real.exp (-x ^ 2 / ε ^ 2)) := by
    have := integrable_rpow_mul_exp_neg_mul_sq (b := 1 / ε ^ 2) (by positivity) (s := 2) (by norm_num)
    convert this using 1; funext x; rw [Real.rpow_two]; ring_nf
  have hi2 : Integrable (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2)) := by
    have := integrable_exp_neg_mul_sq (b := 1 / ε ^ 2) (by positivity)
    convert this using 1; funext x; congr 1; ring
  have hs : 0 < ε * Real.sqrt π := by positivity
  rw [show (fun x : ℝ => Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) * (x ^ 2 + 1)) =
      fun x : ℝ => (1 / (ε * Real.sqrt π)) * (x ^ 2 * Real.exp (-x ^ 2 / ε ^ 2) + Real.exp (-x ^ 2 / ε ^ 2)) by
    funext x; field_simp, integral_const_mul, integral_add hi1 hi2, hm]
  have := gauss_unit ε hε
  rw [integral_div] at this
  have hg : ∫ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) = ε * Real.sqrt π := by
    rwa [div_eq_one_iff_eq hs.ne'] at this
  rw [hg]
  have : 0 < Real.sqrt π := Real.sqrt_pos.2 Real.pi_pos
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.gauss_poly
-- test: fa-gauss-poly-test.sage

/-- `∫ g_ε eˣ = e^{ε²/4}`, by completing the square. -/
theorem gauss_exp (ε : ℝ) (hε : 0 < ε) :
    ∫ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) * Real.exp x = Real.exp (ε ^ 2 / 4) := by
  have hpt : ∀ x : ℝ, Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π) * Real.exp x =
      Real.exp (ε ^ 2 / 4) / (ε * Real.sqrt π) * Real.exp (-(1 / ε ^ 2) * (x - ε ^ 2 / 2) ^ 2) := by
    intro x
    rw [div_mul_eq_mul_div, div_mul_eq_mul_div, ← Real.exp_add, ← Real.exp_add]
    congr 2; field_simp; ring
  simp_rw [hpt]
  rw [integral_const_mul]
  have hshift : ∫ x : ℝ, Real.exp (-(1 / ε ^ 2) * (x - ε ^ 2 / 2) ^ 2) = ∫ x : ℝ, Real.exp (-(1 / ε ^ 2) * x ^ 2) :=
    integral_sub_right_eq_self (fun y => Real.exp (-(1 / ε ^ 2) * y ^ 2)) (ε ^ 2 / 2)
  rw [hshift, integral_gaussian, show π / (1 / ε ^ 2) = ε ^ 2 * π by field_simp, Real.sqrt_mul (by positivity),
    Real.sqrt_sq hε.le]
  have : 0 < Real.sqrt π := Real.sqrt_pos.2 Real.pi_pos
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.gauss_exp
-- test: fa-gauss-exp-test.sage

/-- The Lorentzians `ε/(π(x² + ε²))` have unit integral. -/
theorem lorentz_unit (ε : ℝ) (hε : 0 < ε) : ∫ x : ℝ, ε / (π * (x ^ 2 + ε ^ 2)) = 1 := by
  have := Measure.integral_comp_div (fun y : ℝ => (1 / π) * (1 + y ^ 2)⁻¹) ε
  rw [show (fun x : ℝ => ε / (π * (x ^ 2 + ε ^ 2))) = fun x : ℝ => (1 / ε) * ((fun y : ℝ => (1 / π) * (1 + y ^ 2)⁻¹) (x / ε)) by
    funext x; field_simp; ring, integral_const_mul, this, integral_const_mul, integral_univ_inv_one_add_sq,
    abs_of_pos hε, smul_eq_mul]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.lorentz_unit
-- test: fa-lorentz-unit-test.sage

/-- The Fourier kernel `sin(Lx)/(πx)` tested on `cos(cx)` gives `(sgn(L + c) + sgn(L - c))/2`: `1` for `L > c`, as
stated (tested, not proved: the Dirichlet integral is not in mathlib). -/
def FourierKernelCos : Prop :=
  ∀ L c : ℝ, 0 < L → 0 < c → c ≠ L →
    Tendsto (fun R => ∫ x in (-R)..R, Real.sin (L * x) / (π * x) * Real.cos (c * x)) atTop
      (𝓝 ((Real.sign (L + c) + Real.sign (L - c)) / 2))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.FourierKernelCos
-- test: fa-fourier-kernel-cos-test.sage

/-! ## exm-dist-delta-composition -/

/-- Scaling: `∫ g(ax) φ(x) dx = (1/|a|) ∫ g(y) φ(y/a) dy`, hence `δ(ax) = δ(x)/|a|`. -/
theorem delta_scaling (g φ : ℝ → ℝ) (a : ℝ) (ha : a ≠ 0) :
    ∫ x, g (a * x) * φ x = |a⁻¹| * ∫ y, g y * φ (y / a) := by
  have := Measure.integral_comp_mul_left (fun y => g y * φ (y / a)) a
  rw [show (fun x => g (a * x) * φ x) = fun x => (fun y => g y * φ (y / a)) (a * x) by
    funext x; simp only; congr 2; field_simp, this, smul_eq_mul]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.delta_scaling
-- test: fa-delta-scaling-test.sage

/-- `δ(x² - a²)`: the zeros `±a` of `h = x² - a²` have `|h'(±a)| = 2|a|`. -/
theorem sq_sub_sq_slopes (a : ℝ) :
    |deriv (fun x : ℝ => x ^ 2 - a ^ 2) a| = 2 * |a| ∧ |deriv (fun x : ℝ => x ^ 2 - a ^ 2) (-a)| = 2 * |a| := by
  have h : ∀ y, deriv (fun x : ℝ => x ^ 2 - a ^ 2) y = 2 * y := fun y => by
    exact ((hasDerivAt_pow 2 y).sub_const (a ^ 2)).deriv.trans (by norm_num)
  rw [h, h, abs_mul, abs_mul, abs_neg]; norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.sq_sub_sq_slopes
-- test: fa-sq-sub-sq-slopes-test.sage

/-- The polar normalisation `∫₀^{2π} dθ/(2π) = 1` behind `δ²(x) = δ(r)/(2πr)`. -/
theorem polar_normalization : ∫ θ in (0 : ℝ)..(2 * π), 1 / (2 * π) = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy / (2 * Real.pi) : ℝ)) ((fun θ => 1 / (2 * π)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / (2 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy / (2 * Real.pi) : ℝ)) (f' :=
          fun θ => 1 / (2 * π)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: fa-integral2.sage

/-! ## exr-dist-not-function -/

/-- The integral of `|f|` over a shrinking interval tends to `0`, for locally integrable `f`. -/
theorem shrinking_integral (f : ℝ → ℝ) (hf : IntervalIntegrable f volume (-1) 1) :
    Tendsto (fun ε => ∫ x in (-ε)..ε, |f x|) (𝓝 0) (𝓝 0) := by
  have hc : ContinuousOn (fun b => ∫ x in (0 : ℝ)..b, |f x|) (Set.uIcc (-1) 1) :=
    intervalIntegral.continuousOn_primitive_interval' hf.abs (by simp)
  have h0 : ContinuousAt (fun b => ∫ x in (0 : ℝ)..b, |f x|) 0 := by
    apply hc.continuousAt
    rw [Set.uIcc_of_le (by norm_num)]
    exact Icc_mem_nhds (show (-1 : ℝ) < 0 by norm_num) (show (0 : ℝ) < 1 by norm_num)
  have hF0 : (∫ x in (0 : ℝ)..0, |f x|) = 0 := intervalIntegral.integral_same
  have hneg : Tendsto (fun ε : ℝ => ∫ x in (0 : ℝ)..(-ε), |f x|) (𝓝 0) (𝓝 0) := by
    have := h0.tendsto.comp (continuous_neg.tendsto' 0 0 (by simp))
    rw [hF0] at this
    exact this
  have hpos : Tendsto (fun ε : ℝ => ∫ x in (0 : ℝ)..ε, |f x|) (𝓝 0) (𝓝 0) := by
    have := h0.tendsto
    rw [hF0] at this
    exact this
  have hev : (fun ε => ∫ x in (-ε)..ε, |f x|) =ᶠ[𝓝 0]
      fun ε => (∫ x in (0 : ℝ)..ε, |f x|) - ∫ x in (0 : ℝ)..(-ε), |f x| := by
    filter_upwards [Ioo_mem_nhds (by norm_num : (-1 : ℝ) < 0) (by norm_num : (0 : ℝ) < 1)] with ε hε
    have m0 : (0 : ℝ) ∈ Set.uIcc (-1 : ℝ) 1 := Set.mem_uIcc.2 (Or.inl ⟨by norm_num, by norm_num⟩)
    have mε : ε ∈ Set.uIcc (-1 : ℝ) 1 := Set.mem_uIcc.2 (Or.inl ⟨hε.1.le, hε.2.le⟩)
    have mε' : -ε ∈ Set.uIcc (-1 : ℝ) 1 := Set.mem_uIcc.2 (Or.inl ⟨by linarith [hε.2], by linarith [hε.1]⟩)
    rw [intervalIntegral.integral_interval_sub_left (hf.abs.mono_set (Set.uIcc_subset_uIcc m0 mε))
      (hf.abs.mono_set (Set.uIcc_subset_uIcc m0 mε'))]
  have := hpos.sub hneg
  rw [sub_zero] at this
  exact this.congr' hev.symm

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.shrinking_integral
-- test: fa-shrinking-integral-test.sage

/-- A linear functional that vanishes on the functions vanishing at `0` is `c δ` with `c = T ψ`, `ψ(0) = 1`. -/
theorem functional_at_zero (T : (ℝ → ℝ) →ₗ[ℝ] ℝ) (hT : ∀ φ : ℝ → ℝ, φ 0 = 0 → T φ = 0) (ψ : ℝ → ℝ) (hψ : ψ 0 = 1)
    (φ : ℝ → ℝ) : T φ = T ψ * φ 0 := by
  have h := hT (φ - φ 0 • ψ) (by simp [hψ])
  rw [map_sub, map_smul, smul_eq_mul] at h
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.functional_at_zero
-- test: fa-functional-at-zero-test.sage

/-! ## exr-dist-comb -/

/-- The Dirac comb is tempered: `Σ_n c/(1 + 4π²n²) < ∞`. -/
theorem comb_tempered (c : ℝ) : Summable (fun n : ℤ => c * (1 / (1 + 4 * π ^ 2 * (n : ℝ) ^ 2))) := by
  apply Summable.mul_left
  apply (summable_one_div_int_pow.2 one_lt_two).of_norm_bounded_eventually
  filter_upwards [Filter.eventually_cofinite_ne 0] with n hn
  have hn' : (1 : ℝ) ≤ (n : ℝ) ^ 2 := by
    have : (1 : ℤ) ≤ n ^ 2 := by nlinarith [Int.one_le_abs hn, sq_abs n]
    exact_mod_cast this
  rw [Real.norm_eq_abs, abs_of_pos (by positivity)]
  apply one_div_le_one_div_of_le (by positivity)
  have hπ2 : 1 ≤ 4 * π ^ 2 := by nlinarith [Real.pi_gt_three]
  nlinarith [mul_le_mul_of_nonneg_right hπ2 (sq_nonneg (n : ℝ))]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.comb_tempered
-- test: fa-comb-tempered-test.sage

/-- The derivative of the sawtooth as a Fourier series: `1 - Σ_{|n| ≤ N} (-1)^n e^{inx} = 2 Σ_{n=1}^N (-1)^{n+1} cos nx`. -/
theorem comb_series (N : ℕ) (x : ℝ) :
    1 - ∑ n ∈ Finset.Icc (-(N : ℤ)) N, (-1 : ℂ) ^ n * Complex.exp (n * x * Complex.I) =
      ((2 * ∑ n ∈ Finset.range N, (-1 : ℝ) ^ (n + 2) * Real.cos ((n + 1) * x) : ℝ) : ℂ) := by
  induction N with
  | zero => simp
  | succ N ih =>
    have hs : Finset.Icc (-((N + 1 : ℕ) : ℤ)) ((N + 1 : ℕ) : ℤ) =
        insert (-((N : ℤ) + 1)) (insert ((N : ℤ) + 1) (Finset.Icc (-(N : ℤ)) N)) := by
      ext k; simp only [Finset.mem_Icc, Finset.mem_insert]; push_cast; omega
    rw [hs, Finset.sum_insert (by simp; omega), Finset.sum_insert (by simp)]
    rw [show (1 : ℂ) - ((-1 : ℂ) ^ (-((N : ℤ) + 1)) * Complex.exp (((-((N : ℤ) + 1) : ℤ) : ℂ) * x * Complex.I) +
      ((-1 : ℂ) ^ ((N : ℤ) + 1) * Complex.exp ((((N : ℤ) + 1 : ℤ) : ℂ) * x * Complex.I) +
        ∑ n ∈ Finset.Icc (-(N : ℤ)) N, (-1 : ℂ) ^ n * Complex.exp (n * x * Complex.I))) =
      (1 - ∑ n ∈ Finset.Icc (-(N : ℤ)) N, (-1 : ℂ) ^ n * Complex.exp (n * x * Complex.I)) -
        ((-1 : ℂ) ^ (-((N : ℤ) + 1)) * Complex.exp (((-((N : ℤ) + 1) : ℤ) : ℂ) * x * Complex.I) +
          (-1 : ℂ) ^ ((N : ℤ) + 1) * Complex.exp ((((N : ℤ) + 1 : ℤ) : ℂ) * x * Complex.I)) by ring, ih]
    have hsign : (-1 : ℂ) ^ (-((N : ℤ) + 1)) = (-1) ^ (N + 1) := by
      rw [zpow_neg, ← inv_zpow, inv_neg, inv_one, show ((N : ℤ) + 1) = ((N + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
    have hsign2 : (-1 : ℂ) ^ ((N : ℤ) + 1) = (-1) ^ (N + 1) := by
      rw [show ((N : ℤ) + 1) = ((N + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
    have e1 : Complex.exp (((-((N : ℤ) + 1) : ℤ) : ℂ) * x * Complex.I) =
        (Real.cos ((N + 1) * x) : ℂ) - (Real.sin ((N + 1) * x) : ℂ) * Complex.I := by
      rw [show (((-((N : ℤ) + 1) : ℤ) : ℂ) * x * Complex.I) = (((-((N + 1) * x)) : ℝ) : ℂ) * Complex.I by
        push_cast; ring, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_neg, Real.sin_neg]
      push_cast; ring
    have e2 : Complex.exp ((((N : ℤ) + 1 : ℤ) : ℂ) * x * Complex.I) =
        (Real.cos ((N + 1) * x) : ℂ) + (Real.sin ((N + 1) * x) : ℂ) * Complex.I := by
      rw [show ((((N : ℤ) + 1 : ℤ) : ℂ) * x * Complex.I) = ((((N + 1) * x)) : ℝ) * Complex.I by
        push_cast; ring, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
    rw [hsign, hsign2, e1, e2, Finset.sum_range_succ]
    push_cast
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Distributions.comb_series
-- test: fa-comb-series-test.sage

end KaluzaKlein.FourierAnalysis.Distributions

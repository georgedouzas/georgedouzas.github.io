import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.TransformComputations

/-- Closes the goals left by `convert` on a composed `HasDerivAt`: function equality (pointwise) and the derivative. -/
macro "fa_close3" : tactic => `(tactic| first
  | rfl
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; done)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; done)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; done)
  | ring1
  | (field_simp; done)
  | (field_simp; ring1)
  | (norm_num; done)
  | (norm_num; ring1)
  | (norm_num; field_simp; ring1)
  | (simp; done)
  | (simp; ring1)
  | (ring_nf; done)
  | (field_simp; ring_nf; done))

/-! # Fourier analysis: test functions (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.TestFunctions

open Real MeasureTheory Filter Topology

/-- The bump `φ(x) = e^{-1/(1-x²)}` for `|x| < 1`, `0` outside. -/
noncomputable def bump (x : ℝ) : ℝ := expNegInvGlue (1 - x ^ 2)

/-! ## def-test-function, exm-dist-bump, exr-dist-bump -/

/-- The bump vanishes outside `(-1, 1)`. -/
theorem bump_test (x : ℝ) (hx : 1 ≤ |x|) : bump x = 0 := by
  apply expNegInvGlue.zero_of_nonpos
  nlinarith [abs_mul_abs_self x, abs_nonneg x]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.bump_test
-- test: fa-bump-test-test.sage

/-- Scaled and translated bumps `φ((x - a)/ε)` are smooth: with the support bound, test functions. -/
theorem bump_smooth (a ε : ℝ) : ContDiff ℝ (⊤ : ℕ∞) (fun x => bump ((x - a) / ε)) := by
  unfold bump; fun_prop

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.bump_smooth
-- test: fa-bump-smooth-test.sage

/-- Inside `(-1, 1)` the bump is `e^{-1/(1-x²)}`; at the centre it is `e^{-1} ≠ 0`. -/
theorem bump_val (x : ℝ) (hx : |x| < 1) : bump x = Real.exp (-1 / (1 - x ^ 2)) := by
  have h : 0 < 1 - x ^ 2 := by nlinarith [abs_mul_abs_self x, abs_nonneg x]
  simp only [bump, expNegInvGlue, not_le.2 h, if_false]
  congr 1; field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.bump_val
-- test: fa-bump-val-test.sage

/-- The derivatives of a test function are test functions (smooth, same support bound). -/
theorem deriv_test (φ : ℝ → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) (R : ℝ) (hR : ∀ x, R < |x| → φ x = 0) :
    ContDiff ℝ (⊤ : ℕ∞) (deriv φ) ∧ ∀ x, R < |x| → deriv φ x = 0 := by
  constructor
  · exact hφ.iterate_deriv 1
  · intro x hx
    have hev : φ =ᶠ[𝓝 x] fun _ => 0 := by
      have : IsOpen {y : ℝ | R < |y|} := isOpen_lt continuous_const continuous_abs
      filter_upwards [this.mem_nhds hx] with y hy using hR y hy
    rw [hev.deriv_eq]; simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.deriv_test
-- test: fa-deriv-test-test.sage

/-- The bump is not analytic at the edge of its support: `t ↦ e^{-1/(ct)}` is not analytic at `0` for any `c > 0`
(mathlib: `e^{-1/t}` is not analytic at `0`). -/
theorem not_analytic_edge (c : ℝ) (hc : 0 < c) : ¬ AnalyticAt ℝ (fun t => expNegInvGlue (c * t)) 0 := by
  intro h
  apply expNegInvGlue.not_analyticAt_zero
  have hl : AnalyticAt ℝ (fun t : ℝ => c⁻¹ * t) 0 := (analyticAt_const (v := c⁻¹)).mul analyticAt_id
  have h' : AnalyticAt ℝ (fun t => expNegInvGlue (c * t)) (c⁻¹ * 0) := by simpa using h
  have := AnalyticAt.comp h' hl
  refine AnalyticAt.congr this (Eventually.of_forall fun t => ?_)
  simp [Function.comp, mul_inv_cancel_left₀ hc.ne']

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.not_analytic_edge
-- test: fa-not-analytic-edge-test.sage

/-- Scaled and translated bumps `φ((x - a)/ε)` vanish outside `[a - ε, a + ε]`. -/
theorem scaled_bump_support (a ε x : ℝ) (hε : 0 < ε) (hx : ε ≤ |x - a|) : bump ((x - a) / ε) = 0 := by
  apply bump_test
  rw [abs_div, abs_of_pos hε, le_div_iff₀ hε, one_mul]; exact hx

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.scaled_bump_support
-- test: fa-scaled-bump-support-test.sage

/-- `h(t) = e^{-1/t}` has `h' = t⁻² e^{-1/t}`. -/
theorem exp_inv_deriv (t : ℝ) (ht : t ≠ 0) : deriv (fun t : ℝ => Real.exp (-1 / t)) t = Real.exp (-1 / t) / t ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => rexp (-1 / t)) (((Real.exp (-1 / t)) / (t ^ 2) : ℝ)) (t) :=
        by
        convert ((((hasDerivAt_const (t : ℝ) (1 : ℝ)).div (hasDerivAt_id' (t : ℝ)) ht).const_mul (-1 : ℝ)).exp) using 1
        all_goals (simp only [Pi.div_apply]; (ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: fa-deriv.sage

/-- `h'' = (t⁻⁴ - 2t⁻³) e^{-1/t}`, of the form `p₂(1/t) e^{-1/t}`. -/
theorem exp_inv_deriv2 (t : ℝ) (ht : t ≠ 0) :
    deriv (fun t : ℝ => Real.exp (-1 / t) / t ^ 2) t = (1 / t ^ 4 - 2 / t ^ 3) * Real.exp (-1 / t) := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun t => rexp (-1 / t) / t ^ 2)
          ((-2 * (Real.exp (-1 / t)) / (t ^ 3) + ((Real.exp (-1 / t)) / (t ^ 4)) : ℝ)) (t) :=
        by
        convert
          (((((hasDerivAt_const (t : ℝ) (1 : ℝ)).div (hasDerivAt_id' (t : ℝ)) ht).const_mul (-1 : ℝ)).exp).div
            ((hasDerivAt_id' (t : ℝ)).pow 2) (by exact pow_ne_zero 2 ht)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.div_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.div_apply, Pi.pow_apply]; (field_simp; ring1))
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv-pow.sage

/-- `h''' = (t⁻⁶ - 6t⁻⁵ + 6t⁻⁴) e^{-1/t}`. -/
theorem exp_inv_deriv3 (t : ℝ) (ht : t ≠ 0) :
    deriv (fun t : ℝ => (1 / t ^ 4 - 2 / t ^ 3) * Real.exp (-1 / t)) t =
      (1 / t ^ 6 - 6 / t ^ 5 + 6 / t ^ 4) * Real.exp (-1 / t) := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun t => (1 / t ^ 4 - 2 / t ^ 3) * rexp (-1 / t))
          ((2 * (3 / (t ^ 4) - (2 / (t ^ 5))) * (Real.exp (-1 / t)) -
              ((2 / (t ^ 3) - (1 / t ^ 4)) * (Real.exp (-1 / t)) / (t ^ 2)) :
            ℝ))
          (t) :=
        by
        convert
          ((((((hasDerivAt_const (t : ℝ) (1 : ℝ)).div ((hasDerivAt_id' (t : ℝ)).pow 3)
                            (by exact pow_ne_zero 3 ht)).const_mul
                        (2 : ℝ)).add
                    (((hasDerivAt_const (t : ℝ) (1 : ℝ)).div ((hasDerivAt_id' (t : ℝ)).pow 4)
                          (by exact pow_ne_zero 4 ht)).const_mul
                      (-1 : ℝ))).mul
                ((((hasDerivAt_const (t : ℝ) (1 : ℝ)).div (hasDerivAt_id' (t : ℝ)) ht).const_mul
                    (-1 : ℝ)).exp)).const_mul
            (-1 : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.div_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.div_apply, Pi.pow_apply]; (field_simp; ring1))
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv-pow.sage

/-- Every `t^{-k} e^{-1/t}` tends to `0` as `t → 0⁺`: the derivatives of `h` vanish at `0`. -/
theorem inv_pow_mul_exp_tendsto (k : ℕ) :
    Tendsto (fun t : ℝ => t⁻¹ ^ k * Real.exp (-t⁻¹)) (𝓝[>] 0) (𝓝 0) :=
  (tendsto_pow_mul_exp_neg_atTop_nhds_zero k).comp tendsto_inv_nhdsGT_zero

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.inv_pow_mul_exp_tendsto
-- test: fa-inv-pow-mul-exp-tendsto-test.sage

/-! ## exm-dist-schwartz-gaussian -/

/-- `x^m e^{-x²/2}` is bounded for every `m`: the Gaussian is a Schwartz function. -/
theorem gaussian_moment_bounded (m : ℕ) : ∃ C, ∀ x : ℝ, |x| ^ m * Real.exp (-x ^ 2 / 2) ≤ C := by
  refine ⟨max 1 ((m.factorial : ℝ) * 2 ^ m), fun x => ?_⟩
  have he : 0 < Real.exp (x ^ 2 / 2) := Real.exp_pos _
  rw [show Real.exp (-x ^ 2 / 2) = (Real.exp (x ^ 2 / 2))⁻¹ by rw [← Real.exp_neg]; ring_nf]
  rcases le_or_gt |x| 1 with h | h
  · calc |x| ^ m * (Real.exp (x ^ 2 / 2))⁻¹ ≤ 1 * 1 := by
          apply mul_le_mul (pow_le_one₀ (abs_nonneg x) h) (inv_le_one_of_one_le₀ (Real.one_le_exp (by positivity)))
            (by positivity) (by norm_num)
      _ ≤ _ := by rw [one_mul]; exact le_max_left _ _
  · have hb := Real.pow_div_factorial_le_exp (x := x ^ 2 / 2) (by positivity) m
    have hx0 : x ≠ 0 := by rintro rfl; norm_num at h
    have hx2 : 0 < x ^ 2 / 2 := by positivity
    rw [mul_inv_le_iff₀ he]
    calc |x| ^ m ≤ |x| ^ m * |x| ^ m := by
          have : 1 ≤ |x| ^ m := one_le_pow₀ h.le
          nlinarith [pow_nonneg (abs_nonneg x) m]
      _ = ((m.factorial : ℝ) * 2 ^ m) * ((x ^ 2 / 2) ^ m / m.factorial) := by
          rw [← mul_pow, abs_mul_abs_self, div_pow]
          field_simp
      _ ≤ ((m.factorial : ℝ) * 2 ^ m) * Real.exp (x ^ 2 / 2) := by gcongr
      _ ≤ max 1 ((m.factorial : ℝ) * 2 ^ m) * Real.exp (x ^ 2 / 2) := by gcongr; exact le_max_right _ _

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.gaussian_moment_bounded
-- test: fa-gaussian-moment-bounded-test.sage

/-- `e^{-a|x|}` (`a > 0`) is not differentiable at `0`: not a Schwartz function. -/
theorem exp_abs_not_differentiable (a : ℝ) (ha : 0 < a) : ¬ DifferentiableAt ℝ (fun x => Real.exp (-a * |x|)) 0 := by
  intro h
  have hlog : DifferentiableAt ℝ (fun x => Real.log (Real.exp (-a * |x|)) / (-a)) 0 :=
    (h.log (Real.exp_pos _).ne').div_const _
  simp only [Real.log_exp] at hlog
  have : DifferentiableAt ℝ (fun x : ℝ => |x|) 0 := by
    convert hlog using 1; funext x; field_simp [ha.ne']
  exact not_differentiableAt_abs_zero this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.exp_abs_not_differentiable
-- test: fa-exp-abs-not-differentiable-test.sage

/-- `x^n / (1 + x²)` is unbounded for `n ≥ 3`: `1/(1 + x²)` decays only like `x⁻²`. -/
theorem rational_unbounded (n : ℕ) (hn : 3 ≤ n) : ¬ ∃ C, ∀ x : ℝ, |x ^ n / (1 + x ^ 2)| ≤ C := by
  rintro ⟨C, hC⟩
  set x := 2 * |C| + 2 with hx
  have hx1 : 1 ≤ x := by rw [hx]; linarith [abs_nonneg C]
  have h := hC x
  have hd : 0 < 1 + x ^ 2 := by positivity
  rw [abs_div, abs_of_pos hd, div_le_iff₀ hd] at h
  have hxn : x ^ 3 ≤ |x ^ n| := by
    rw [abs_pow, abs_of_pos (by linarith)]; exact pow_le_pow_right₀ hx1 hn
  nlinarith [abs_nonneg C, le_abs_self C, sq_nonneg x]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.rational_unbounded
-- test: fa-rational-unbounded-test.sage

/-- The derivative of `e^{-x²} sin(e^{x²})` is `-2x e^{-x²} sin(e^{x²}) + 2x cos(e^{x²})`, which does not decay. -/
theorem decaying_bad_derivative (x : ℝ) :
    deriv (fun x : ℝ => Real.exp (-x ^ 2) * Real.sin (Real.exp (x ^ 2))) x =
      -2 * x * Real.exp (-x ^ 2) * Real.sin (Real.exp (x ^ 2)) + 2 * x * Real.cos (Real.exp (x ^ 2)) := by
  have h1 := ((hasDerivAt_pow 2 x).neg.exp)
  have h2 := ((hasDerivAt_pow 2 x).exp.sin)
  refine (h1.mul h2).deriv.trans ?_
  have : Real.exp (-x ^ 2) * Real.exp (x ^ 2) = 1 := by rw [← Real.exp_add]; simp
  simp only [Pi.neg_apply, Nat.cast_ofNat, Nat.reduceSub, pow_one]
  linear_combination (2 * x * Real.cos (Real.exp (x ^ 2))) * this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.decaying_bad_derivative
-- test: fa-decaying-bad-derivative-test.sage

/-- Every test function is a Schwartz function: `x^m φ^{(n)}` is bounded. -/
theorem test_is_schwartz (φ : ℝ → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) (hc : HasCompactSupport φ) (m n : ℕ) :
    ∃ C, ∀ x, ‖x ^ m * iteratedDeriv n φ x‖ ≤ C := by
  simp_rw [iteratedDeriv_eq_iterate]
  have hcs : ∀ k : ℕ, HasCompactSupport (deriv^[k] φ) := by
    intro k
    induction k with
    | zero => simpa using hc
    | succ k ih => rw [Function.iterate_succ_apply']; exact HasCompactSupport.deriv ih
  apply Continuous.bounded_above_of_compact_support
  · exact (continuous_pow m).mul (ContDiff.iterate_deriv n hφ).continuous
  · exact HasCompactSupport.mul_left (hcs n)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.test_is_schwartz
-- test: fa-test-is-schwartz-test.sage

/-! ## exr-dist-schwartz-transform -/

/-- `∫ c/(1 + x²) dx = cπ`. -/
theorem lorentz_mass (c : ℝ) : ∫ x : ℝ, c / (1 + x ^ 2) = c * π := by
  simp_rw [div_eq_mul_inv]
  rw [integral_const_mul, integral_univ_inv_one_add_sq]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.lorentz_mass
-- test: fa-lorentz-mass-test.sage

/-- `∫ |g| ≤ π sup |(1 + x²) g|`: the `L¹` norm is bounded by a Schwartz seminorm. -/
theorem l1_le_seminorm (g : ℝ → ℝ) (C : ℝ) (hC : ∀ x, |(1 + x ^ 2) * g x| ≤ C) (hg : Integrable g) :
    ∫ x, |g x| ≤ π * C := by
  have hle : ∀ x, |g x| ≤ C / (1 + x ^ 2) := fun x => by
    have hp : 0 < 1 + x ^ 2 := by positivity
    rw [le_div_iff₀ hp, mul_comm]
    calc (1 + x ^ 2) * |g x| = |(1 + x ^ 2) * g x| := by rw [abs_mul, abs_of_pos hp]
      _ ≤ C := hC x
  have hint : Integrable (fun x : ℝ => C / (1 + x ^ 2)) := by
    simp_rw [div_eq_mul_inv]; exact integrable_inv_one_add_sq.const_mul C
  calc ∫ x, |g x| ≤ ∫ x, C / (1 + x ^ 2) := integral_mono hg.abs hint hle
    _ = π * C := by rw [lorentz_mass]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.l1_le_seminorm
-- test: fa-l1-le-seminorm-test.sage

/-! ## exr-dist-mollifier -/

/-- The convolution of `f` vanishing outside `[-R, R]` with `φ` vanishing outside `[-r, r]` vanishes outside
`[-R - r, R + r]`: a mollified compactly supported function is a test function. -/
theorem mollifier_support (f φ : ℝ → ℝ) (R r : ℝ) (hf : ∀ y, R < |y| → f y = 0) (hφ : ∀ y, r < |y| → φ y = 0)
    (x : ℝ) (hx : R + r < |x|) : ∫ y, f y * φ (x - y) = 0 := by
  have : (fun y => f y * φ (x - y)) = fun _ => 0 := by
    funext y
    by_cases hy : R < |y|
    · rw [hf y hy, zero_mul]
    · push_neg at hy
      have : r < |x - y| := by
        have := abs_sub_abs_le_abs_sub x y
        linarith
      rw [hφ _ this, mul_zero]
  rw [this, integral_zero]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.mollifier_support
-- test: fa-mollifier-support-test.sage

/-- A non-negative kernel of unit mass supported in `[-δ, δ]` moves `f` by at most `sup_{|y| ≤ δ} |f(x-y) - f(x)|`. -/
theorem mollifier_approx (f φ : ℝ → ℝ) (hf : Continuous f) (hφc : Continuous φ) (hpos : ∀ y, 0 ≤ φ y)
    (δ ε x : ℝ) (hδ : 0 < δ) (hsupp : ∀ y, δ < |y| → φ y = 0) (hmass : ∫ y in (-δ)..δ, φ y = 1)
    (hε : ∀ y, |y| ≤ δ → |f (x - y) - f x| ≤ ε) :
    |(∫ y in (-δ)..δ, f (x - y) * φ y) - f x| ≤ ε := by
  have h1 : (∫ y in (-δ)..δ, f (x - y) * φ y) - f x = ∫ y in (-δ)..δ, (f (x - y) - f x) * φ y := by
    have hfx : ∫ y in (-δ)..δ, f x * φ y = f x := by rw [intervalIntegral.integral_const_mul, hmass, mul_one]
    conv_lhs => rw [← hfx]
    rw [← intervalIntegral.integral_sub (by apply Continuous.intervalIntegrable; fun_prop)
        (by apply Continuous.intervalIntegrable; fun_prop)]
    congr 1; funext y; ring
  rw [h1]
  calc |∫ y in (-δ)..δ, (f (x - y) - f x) * φ y| ≤ ∫ y in (-δ)..δ, |(f (x - y) - f x) * φ y| :=
        intervalIntegral.abs_integral_le_integral_abs (by linarith)
    _ ≤ ∫ y in (-δ)..δ, ε * φ y := by
        apply intervalIntegral.integral_mono_on (by linarith)
        · apply Continuous.intervalIntegrable; fun_prop
        · apply Continuous.intervalIntegrable; fun_prop
        · intro y hy
          rw [abs_mul, abs_of_nonneg (hpos y)]
          exact mul_le_mul_of_nonneg_right (hε y (abs_le.2 ⟨hy.1, hy.2⟩)) (hpos y)
    _ = ε := by rw [intervalIntegral.integral_const_mul, hmass, mul_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.TestFunctions.mollifier_approx
-- test: fa-mollifier-approx-test.sage

end KaluzaKlein.FourierAnalysis.TestFunctions

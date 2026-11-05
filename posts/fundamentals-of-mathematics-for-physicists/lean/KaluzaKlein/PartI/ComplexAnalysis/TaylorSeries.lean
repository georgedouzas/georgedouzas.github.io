import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Taylor series"

The claims of the section "Taylor series" of `tutorial/part-i-04-complex-analysis.qmd`: absolute convergence inside
and divergence outside the radius of convergence of `∑ aₙ zⁿ`, the root-test formula for the radius, holomorphy and
term-by-term differentiation of the sum, the Taylor expansion of a function holomorphic on a disc (with radius at
least the radius of the disc), the standard series (`exp`, `cos`, `sin`, `1/(1 - z)`, `Log (1 + z)`, `1/(1 + z²)`,
`1/(z² + 4)`) and their radii, Taylor coefficients as circle integrals, and the Cauchy estimates.
-/

namespace KaluzaKlein.ComplexAnalysis.TaylorSeries

open Complex Filter Topology Metric
open scoped NNReal ENNReal Real Nat

/-- The radius of convergence of the scalar power series `∑ aₙ zⁿ`. -/
noncomputable def radius (a : ℕ → ℂ) : ℝ≥0∞ := (FormalMultilinearSeries.ofScalars ℂ a).radius

lemma norm_ofScalars (a : ℕ → ℂ) (n : ℕ) : ‖FormalMultilinearSeries.ofScalars ℂ a n‖ = ‖a n‖ :=
  FormalMultilinearSeries.ofScalars_norm ℂ a n

/-- A helper: a sequence whose norms are eventually reciprocal to a null sequence is not summable. -/
lemma not_summable_of_eventually_inv {u : ℕ → ℂ} {v : ℕ → ℝ} (hv : Tendsto v atTop (𝓝 0))
    (h : ∀ᶠ n in atTop, ‖u n‖ * v n = 1) : ¬ Summable u := by
  intro hs
  have h1 : Tendsto (fun n => ‖u n‖ * v n) atTop (𝓝 0) := by
    simpa using hs.tendsto_atTop_zero.norm.mul hv
  have h2 : Tendsto (fun n => ‖u n‖ * v n) atTop (𝓝 1) := tendsto_const_nhds.congr' (h.mono fun n hn => hn.symm)
  exact one_ne_zero (tendsto_nhds_unique h2 h1)

/-! ## def-taylor-series -/

/-- Inside the radius of convergence the power series converges absolutely. -/
theorem summable_norm_of_lt_radius (a : ℕ → ℂ) (z : ℂ) (h : (‖z‖₊ : ℝ≥0∞) < radius a) :
    Summable (fun n => ‖a n * z ^ n‖) := by
  have := (FormalMultilinearSeries.ofScalars ℂ a).summable_norm_mul_pow h
  simpa [norm_mul, norm_pow, norm_ofScalars] using this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_norm_of_lt_radius
-- test: ca-res-summable-radius-test.sage

/-- Outside the radius of convergence the power series diverges. -/
theorem not_summable_of_radius_lt (a : ℕ → ℂ) (z : ℂ) (h : radius a < (‖z‖₊ : ℝ≥0∞)) :
    ¬ Summable (fun n => a n * z ^ n) := by
  intro hs
  have ht : Tendsto (fun n => ‖FormalMultilinearSeries.ofScalars ℂ a n‖ * ((‖z‖₊ : ℝ≥0) : ℝ) ^ n) atTop
      (𝓝 0) := by
    simpa [norm_ofScalars, norm_mul, norm_pow] using hs.tendsto_atTop_zero.norm
  exact absurd ((FormalMultilinearSeries.ofScalars ℂ a).le_radius_of_tendsto ht) (not_le.2 h)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.not_summable_of_radius_lt
-- test: ca-res-divergent-radius-test.sage

/-- The root test: `R = liminf 1 / |aₙ|^(1/n)`, i.e. `1/R = limsup |aₙ|^(1/n)`. -/
theorem radius_eq_liminf (a : ℕ → ℂ) :
    radius a = liminf (fun n => 1 / ((‖a n‖₊ ^ (1 / (n : ℝ)) : ℝ≥0) : ℝ≥0∞)) atTop := by
  have h : ∀ n, ‖FormalMultilinearSeries.ofScalars ℂ a n‖₊ = ‖a n‖₊ := fun n => by
    ext; simp
  rw [radius, FormalMultilinearSeries.radius_eq_liminf]
  congr 1
  funext n
  rw [h n]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.radius_eq_liminf
-- test: ca-res-root-test.sage

/-- Inside the disc of convergence the sum is holomorphic. -/
theorem differentiableOn_sum (a : ℕ → ℂ) :
    DifferentiableOn ℂ (fun z => ∑' n, a n * z ^ n) (Metric.eball 0 (radius a)) := by
  rcases eq_zero_or_pos (radius a) with h | h
  · have he : Metric.eball (0 : ℂ) (radius a) = ∅ := by simp [h]
    rw [he]; exact differentiableOn_empty
  have hp := (FormalMultilinearSeries.ofScalars ℂ a).hasFPowerSeriesOnBall h
  have hd := hp.differentiableOn
  refine hd.congr fun z _ => ?_
  have := FormalMultilinearSeries.ofScalars_sum_eq (E := ℂ) a z
  simp only [FormalMultilinearSeries.ofScalarsSum, smul_eq_mul] at this
  exact this.symm

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.differentiableOn_sum
-- test: ca-res-holo-sum-test.sage

/-- Inside the disc of convergence the sum may be differentiated term by term. -/
theorem hasSum_deriv (a : ℕ → ℂ) (z : ℂ) (hz : (‖z‖₊ : ℝ≥0∞) < radius a) :
    HasSum (fun n => a n * (n * z ^ (n - 1))) (deriv (fun w => ∑' n, a n * w ^ n) z) := by
  obtain ⟨r, hzr, hr⟩ := ENNReal.lt_iff_exists_nnreal_btwn.1 hz
  have hs := (FormalMultilinearSeries.ofScalars ℂ a).summable_norm_mul_pow hr
  simp only [norm_ofScalars] at hs
  have hzU : z ∈ Metric.ball (0 : ℂ) r := by
    simpa [mem_ball_zero_iff] using (show (‖z‖₊ : ℝ) < r from mod_cast (ENNReal.coe_lt_coe.1 hzr))
  have key := hasSum_deriv_of_summable_norm (F := fun n w => a n * w ^ n) hs
    (fun n => by fun_prop) isOpen_ball (fun n w hw => by
      rw [mem_ball_zero_iff] at hw
      rw [norm_mul, norm_pow]
      gcongr) hzU
  convert key using 1
  funext n
  simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_deriv
-- test: ca-res-termwise-test.sage

/-- A function holomorphic on the disc `|z - c| < r` equals its Taylor series there. -/
theorem hasSum_taylor (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hf : DifferentiableOn ℂ f (Metric.ball c r)) (z : ℂ)
    (hz : z ∈ Metric.ball c r) :
    HasSum (fun n => iteratedDeriv n f c / (Nat.factorial n : ℂ) * (z - c) ^ n) (f z) := by
  have := Complex.hasSum_taylorSeries_on_ball hf hz
  convert this using 2 with n
  simp only [smul_eq_mul]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_taylor
-- test: ca-res-taylor-test.sage

/-- If the terms `‖aₙ‖ sⁿ` tend to `0`, then `s ≤ R`. -/
lemma le_radius_of_tendsto_norm (a : ℕ → ℂ) (s : ℝ≥0)
    (ht : Tendsto (fun n => ‖a n‖ * (s : ℝ) ^ n) atTop (𝓝 0)) : (s : ℝ≥0∞) ≤ radius a := by
  unfold radius
  refine (FormalMultilinearSeries.ofScalars ℂ a).le_radius_of_tendsto (l := 0) ?_
  simpa only [norm_ofScalars] using ht

/-- The Taylor terms at the point `c + s` of the disc tend to `0`. -/
lemma taylor_term_tendsto (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hf : DifferentiableOn ℂ f (Metric.ball c r))
    (s : ℝ≥0) (hsr : (s : ℝ) < r) :
    Tendsto (fun n => ‖iteratedDeriv n f c / (Nat.factorial n : ℂ)‖ * (s : ℝ) ^ n) atTop (𝓝 0) := by
  have hn : ‖((s : ℝ) : ℂ)‖ = (s : ℝ) := by
    rw [Complex.norm_real, Real.norm_eq_abs, NNReal.abs_eq]
  have hz : c + ((s : ℝ) : ℂ) ∈ Metric.ball c r := by
    rw [mem_ball, Complex.dist_eq, add_sub_cancel_left, hn]
    exact hsr
  have ht := (hasSum_taylor f c r hf _ hz).summable.tendsto_atTop_zero.norm
  rw [norm_zero] at ht
  refine ht.congr fun n => ?_
  rw [norm_mul, norm_pow, add_sub_cancel_left, hn]

/-- The radius of convergence of the Taylor series is at least the radius of any disc of holomorphy. -/
theorem le_radius_taylor (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hf : DifferentiableOn ℂ f (Metric.ball c r)) :
    ENNReal.ofReal r ≤ radius (fun n => iteratedDeriv n f c / (Nat.factorial n : ℂ)) := by
  refine ENNReal.le_of_forall_nnreal_lt fun s hs => ?_
  have hsr : (s : ℝ) < r := by
    rw [← ENNReal.ofReal_coe_nnreal] at hs
    exact (ENNReal.ofReal_lt_ofReal_iff_of_nonneg (NNReal.coe_nonneg s)).1 hs
  exact le_radius_of_tendsto_norm _ s (taylor_term_tendsto f c r hf s hsr)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.le_radius_taylor
-- test: ca-res-radius-taylor-test.sage

/-! ## exm-taylor-standard -/

/-- `e^z = ∑ zⁿ/n!`, absolutely convergent for every `z` (`R = ∞`). -/
theorem hasSum_exp (z : ℂ) :
    HasSum (fun n => z ^ n / (Nat.factorial n : ℂ)) (exp z) ∧
      Summable (fun n => ‖z ^ n / (Nat.factorial n : ℂ)‖) := by
  refine ⟨?_, ?_⟩
  · rw [Complex.exp_eq_exp_ℂ]
    exact NormedSpace.expSeries_div_hasSum_exp z
  · refine (Real.summable_pow_div_factorial ‖z‖).congr fun n => ?_
    rw [norm_div, norm_pow, Complex.norm_natCast]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_exp
-- test: ca-res-exp-series-test.sage

/-- `cos z` and `sin z` equal their series for every `z`. -/
theorem hasSum_cos_sin (z : ℂ) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / (Nat.factorial (2 * n) : ℂ)) (cos z) ∧
      HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n + 1) / (Nat.factorial (2 * n + 1) : ℂ)) (sin z) :=
  ⟨Complex.hasSum_cos z, Complex.hasSum_sin z⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_cos_sin
-- test: ca-res-cos-sin-test.sage

/-- `exp`, `cos` and `sin` are entire. -/
theorem entire (z : ℂ) :
    DifferentiableAt ℂ exp z ∧ DifferentiableAt ℂ cos z ∧ DifferentiableAt ℂ sin z :=
  ⟨differentiableAt_exp, differentiableAt_cos, differentiableAt_sin⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.entire (0) := by
  exact ⟨by fun_prop, by fun_prop, by fun_prop⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.entire (1) := by
  exact ⟨by fun_prop, by fun_prop, by fun_prop⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.entire (Complex.I) := by
  exact ⟨by fun_prop, by fun_prop, by fun_prop⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.entire (-3 + 2 * Complex.I) := by
  exact ⟨by fun_prop, by fun_prop, by fun_prop⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.TaylorSeries.entire
-- test: ca-res-entire-test.sage

/-- `1/(1 - z) = ∑ zⁿ` for `|z| < 1`. -/
theorem hasSum_geometric (z : ℂ) (hz : ‖z‖ < 1) : HasSum (fun n => z ^ n) (1 / (1 - z)) := by
  simpa [one_div] using hasSum_geometric_of_norm_lt_one hz

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_geometric
-- test: ca-res-geometric-test.sage

/-- The geometric series converges exactly on `|z| < 1`: its radius is `1`. -/
theorem summable_geometric_iff (z : ℂ) : Summable (fun n => z ^ n) ↔ ‖z‖ < 1 :=
  summable_geometric_iff_norm_lt_one

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_geometric_iff
-- test: ca-res-geometric-iff-test.sage

/-- `Log (1 + z) = ∑_{n ≥ 1} (-1)^(n+1) zⁿ/n` for `|z| < 1`. -/
theorem hasSum_log (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ => (-1) ^ (n + 1) * z ^ n / n) (log (1 + z)) :=
  Complex.hasSum_taylorSeries_log hz

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_log
-- test: ca-res-log-series-test.sage

/-- The logarithmic series diverges for `|z| > 1`: its radius is `1`. -/
theorem not_summable_log (z : ℂ) (hz : 1 < ‖z‖) :
    ¬ Summable (fun n : ℕ => (-1) ^ (n + 1) * z ^ n / n) := by
  refine not_summable_of_eventually_inv
    (by simpa using tendsto_pow_const_div_const_pow_of_one_lt 1 hz) ?_
  filter_upwards [eventually_ge_atTop 1] with n hn
  have hz0 : ‖z‖ ^ n ≠ 0 := pow_ne_zero _ (by linarith)
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  simp [norm_div, norm_mul, norm_pow]
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.not_summable_log
-- test: ca-res-log-diverges-test.sage

/-! ## exm-taylor-radius-poles -/

/-- `1/(1 + z²) = ∑ (-1)ⁿ z²ⁿ` for `|z| < 1`. -/
theorem hasSum_inv_one_add_sq (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n)) (1 / (1 + z ^ 2)) := by
  have h : ‖-z ^ 2‖ < 1 := by
    rw [norm_neg, norm_pow]; nlinarith [norm_nonneg z]
  have := hasSum_geometric_of_norm_lt_one h
  convert this using 1
  · funext n; rw [neg_pow (z ^ 2), ← pow_mul]
  · rw [sub_neg_eq_add, one_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_inv_one_add_sq
-- test: ca-res-inv-one-add-sq-test.sage

/-- The series of `1/(1 + z²)` converges exactly on `|z| < 1`: its radius is `1`. -/
theorem summable_inv_one_add_sq_iff (z : ℂ) :
    Summable (fun n : ℕ => (-1) ^ n * z ^ (2 * n)) ↔ ‖z‖ < 1 := by
  have h : (fun n : ℕ => (-1) ^ n * z ^ (2 * n)) = fun n => (-z ^ 2) ^ n := by
    funext n; rw [neg_pow (z ^ 2), ← pow_mul]
  rw [h, summable_geometric_iff_norm_lt_one, norm_neg, norm_pow]
  constructor
  · intro h1; nlinarith [norm_nonneg z]
  · intro h1; nlinarith [norm_nonneg z]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_inv_one_add_sq_iff
-- test: ca-res-inv-one-add-sq-iff-test.sage

/-- `1 + z² = (z - i)(z + i)`: the poles of `1/(1 + z²)` are `±i`. -/
theorem one_add_sq_factor (z : ℂ) : 1 + z ^ 2 = (z - I) * (z + I) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  linear_combination hI

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.one_add_sq_factor
-- test: ca-res-one-add-sq-test.sage

/-- The zeros `±ic` of `c² + z²` lie at distance `|c|` from the origin; for `c = 1` the poles `±i` of `1/(1 + z²)`
are at distance `1`. -/
theorem pole_norm (c z : ℂ) (h : c ^ 2 + z ^ 2 = 0) : ‖z‖ = ‖c‖ := by
  have h2 : z ^ 2 = -c ^ 2 := by linear_combination h
  have : ‖z‖ ^ 2 = ‖c‖ ^ 2 := by rw [← norm_pow, h2, norm_neg, norm_pow]
  nlinarith [norm_nonneg z, norm_nonneg c, sq_nonneg (‖z‖ - ‖c‖), sq_nonneg (‖z‖ + ‖c‖)]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm (1) (Complex.I) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm (1) (-Complex.I) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm (2) (2 * Complex.I) ((by rw [mul_pow, Complex.I_sq]; norm_num)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm (0) (0) ((by simp)) := by
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.TaylorSeries.pole_norm
-- test: ca-res-pole-norm-test.sage

/-- On the real line `1/(1 + x²)` is smooth, positive and bounded by `1`. -/
theorem real_smooth_bounded (x : ℝ) :
    ContDiffAt ℝ ⊤ (fun x : ℝ => 1 / (1 + x ^ 2)) x ∧ 0 < 1 / (1 + x ^ 2) ∧ 1 / (1 + x ^ 2) ≤ 1 := by
  refine ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, ?_⟩
  rw [div_le_one (by positivity)]
  nlinarith [sq_nonneg x]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded (0) := by
  exact ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, by norm_num⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded (1) := by
  exact ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, by norm_num⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded (-3) := by
  exact ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, by norm_num⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded (1 / 2) := by
  exact ⟨ContDiffAt.div (by fun_prop) (by fun_prop) (by positivity), by positivity, by norm_num⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.TaylorSeries.real_smooth_bounded
-- test: ca-res-real-bounded-test.sage

/-! ## exm-taylor-coefficients-integral -/

/-- `aₙ = f⁽ⁿ⁾(c)/n! = (2πi)⁻¹ ∮ f(z)/(z - c)^(n+1) dz` for `f` holomorphic on the closed disc. -/
theorem coeff_eq_circleIntegral (f : ℂ → ℂ) (c : ℂ) (R : ℝ) (hR : 0 < R)
    (hf : DifferentiableOn ℂ f (closedBall c R)) (n : ℕ) :
    (2 * π * I)⁻¹ * ∮ z in C(c, R), f z / (z - c) ^ (n + 1) = iteratedDeriv n f c / (Nat.factorial n : ℂ) := by
  have h := hf.circleIntegral_one_div_sub_center_pow_smul hR n
  have h' : (∮ z in C(c, R), f z / (z - c) ^ (n + 1)) = (2 * π * I / (Nat.factorial n : ℂ)) * iteratedDeriv n f c := by
    rw [← smul_eq_mul, ← h]
    congr 1; funext z; rw [smul_eq_mul]; ring
  rw [h']
  have : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have hf : (Nat.factorial n : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
  field_simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.coeff_eq_circleIntegral
-- test: ca-res-coeff-circle-test.sage
end

lemma iteratedDeriv_exp_zero (n : ℕ) : iteratedDeriv n exp 0 = 1 := by
  have := iteratedDeriv_cexp_const_mul n 1
  simp only [one_mul, one_pow] at this
  rw [this]; simp

/-- For `e^z` at `0`: `∮ e^z/z dz = ∮ e^z/z² dz = 2πi`, so `a₀ = a₁ = 1`. -/
theorem exp_circle (R : ℝ) (hR : 0 < R) :
    (∮ z in C(0, R), exp z / z) = 2 * π * I ∧ (∮ z in C(0, R), exp z / z ^ 2) = 2 * π * I := by
  have hd : DifferentiableOn ℂ exp (closedBall 0 R) := differentiable_exp.differentiableOn
  have h0 := coeff_eq_circleIntegral exp 0 R hR hd 0
  have h1 := coeff_eq_circleIntegral exp 0 R hR hd 1
  simp only [sub_zero, zero_add, pow_one, iteratedDeriv_exp_zero, Nat.factorial, Nat.cast_one,
    div_one, Nat.succ_eq_add_one, mul_one] at h0 h1
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  constructor
  · have := congrArg (fun t => (2 * π * I) * t) h0
    simp only [← mul_assoc, mul_inv_cancel₀ hne, one_mul, mul_one] at this
    exact this
  · have := congrArg (fun t => (2 * π * I) * t) h1
    simp only [← mul_assoc, mul_inv_cancel₀ hne, one_mul, mul_one] at this
    norm_num at this
    exact this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.exp_circle
-- test: ca-res-exp-circle-test.sage

/-! ## exr-taylor-exp-log -/

/-- The ratio of successive terms of the exponential series tends to `0` (`R = ∞`). -/
theorem exp_ratio (z : ℂ) (hz : z ≠ 0) :
    Tendsto (fun n : ℕ => ‖z ^ (n + 1) / (Nat.factorial (n + 1) : ℂ)‖ / ‖z ^ n / (Nat.factorial n : ℂ)‖) atTop
      (𝓝 0) := by
  have h : (fun n : ℕ => ‖z ^ (n + 1) / (Nat.factorial (n + 1) : ℂ)‖ / ‖z ^ n / (Nat.factorial n : ℂ)‖) =
      fun n : ℕ => ‖z‖ / ((n : ℝ) + 1) := by
    funext n
    have hz' : ‖z‖ ≠ 0 := norm_ne_zero_iff.2 hz
    have hf : (Nat.factorial n : ℝ) ≠ 0 := by positivity
    simp only [norm_div, norm_pow, Complex.norm_natCast]
    rw [Nat.factorial_succ]
    push_cast
    field_simp
    ring
  rw [h]
  exact tendsto_const_nhds.div_atTop (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.exp_ratio
-- test: ca-res-exp-ratio-test.sage

/-- The `n`-th root of the `n`-th term of the logarithmic series tends to `|z|` (`|aₙ|^(1/n) → 1`, `R = 1`). -/
theorem log_root (z : ℂ) :
    Tendsto (fun n : ℕ => ‖(-1) ^ (n + 1) * z ^ n / n‖ ^ ((1 : ℝ) / n)) atTop (𝓝 ‖z‖) := by
  have hroot : Tendsto (fun n : ℕ => ((n : ℝ) ^ ((1 : ℝ) / n))) atTop (𝓝 1) :=
    tendsto_rpow_div.comp tendsto_natCast_atTop_atTop
  have h2 : Tendsto (fun n : ℕ => ‖z‖ / ((n : ℝ) ^ ((1 : ℝ) / n))) atTop (𝓝 (‖z‖ / 1)) :=
    tendsto_const_nhds.div hroot one_ne_zero
  rw [div_one] at h2
  refine h2.congr' ?_
  filter_upwards [eventually_ge_atTop 1] with n hn
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hnpos : (0 : ℝ) < n := by positivity
  simp only [norm_div, norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul, Complex.norm_natCast]
  rw [Real.div_rpow (by positivity) hnpos.le, ← Real.rpow_natCast, ← Real.rpow_mul (norm_nonneg z),
    mul_one_div_cancel hn0, Real.rpow_one]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.log_root
-- test: ca-res-log-root-test.sage

/-! ## exr-taylor-radius -/

/-- `1/(z² + 4) = ∑ (-1)ⁿ z²ⁿ/4ⁿ⁺¹` for `|z| < 2`. -/
theorem hasSum_inv_sq_add_four (z : ℂ) (hz : ‖z‖ < 2) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / 4 ^ (n + 1)) (1 / (z ^ 2 + 4)) := by
  have h : ‖-(z ^ 2 / 4)‖ < 1 := by
    rw [norm_neg, norm_div, norm_pow]
    have : ‖(4 : ℂ)‖ = 4 := by norm_num
    rw [this, div_lt_one (by norm_num)]; nlinarith [norm_nonneg z]
  have := (hasSum_geometric_of_norm_lt_one h).div_const 4
  convert this using 1
  · funext n
    rw [show (-(z ^ 2 / 4)) = (-1) * (z ^ 2 / 4) by ring, mul_pow, div_pow, ← pow_mul, pow_succ]
    ring
  · have h4 : z ^ 2 + 4 ≠ 0 := by
      intro h0
      have : ‖z‖ ^ 2 = 4 := by
        rw [← norm_pow, show z ^ 2 = -4 by linear_combination h0]; norm_num
      nlinarith [norm_nonneg z]
    have h5 : (1 - -(z ^ 2 / 4) : ℂ) ≠ 0 := by
      intro h0; apply h4; linear_combination 4 * h0
    rw [div_eq_mul_inv ((1 - -(z ^ 2 / 4))⁻¹) 4, ← mul_inv,
      show (1 - -(z ^ 2 / 4)) * 4 = z ^ 2 + 4 by ring, one_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_inv_sq_add_four
-- test: ca-res-inv-sq-add-four-test.sage

/-- The series of `1/(z² + 4)` about `0` converges exactly on `|z| < 2`: its radius is `2`. -/
theorem summable_inv_sq_add_four_iff (z : ℂ) :
    Summable (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / 4 ^ (n + 1)) ↔ ‖z‖ < 2 := by
  have h : (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / 4 ^ (n + 1)) = fun n => (1 / 4 : ℂ) * (-(z ^ 2 / 4)) ^ n := by
    funext n
    rw [show (-(z ^ 2 / 4)) = (-1) * (z ^ 2 / 4) by ring, mul_pow, div_pow, ← pow_mul, pow_succ]
    ring
  rw [h, summable_mul_left_iff (by norm_num), summable_geometric_iff_norm_lt_one, norm_neg, norm_div,
    norm_pow]
  have : ‖(4 : ℂ)‖ = 4 := by norm_num
  rw [this, div_lt_one (by norm_num)]
  constructor
  · intro h1; nlinarith [norm_nonneg z]
  · intro h1; nlinarith [norm_nonneg z]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.TaylorSeries.summable_inv_sq_add_four_iff
-- test: ca-res-inv-sq-add-four-iff-test.sage

/-- `|a + bi| = √(a² + b²)`: the poles `±2i` are at distance `2` from `0` and `√5` from `1`. -/
theorem norm_add_mul_I (a b : ℝ) : ‖(a : ℂ) + b * I‖ = Real.sqrt (a ^ 2 + b ^ 2) := by
  rw [Complex.norm_add_mul_I]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.TaylorSeries.norm_add_mul_I
-- test: ca-res-norm-add-mul-I-test.sage

/-- The Taylor series of `1/(z² + 4)` about `1` converges on `|z - 1| < √5`, the distance to `±2i`. -/
theorem hasSum_taylor_at_one (z : ℂ) (hz : ‖z - 1‖ < Real.sqrt 5) :
    HasSum (fun n => iteratedDeriv n (fun w : ℂ => 1 / (w ^ 2 + 4)) 1 / (Nat.factorial n : ℂ) * (z - 1) ^ n)
      (1 / (z ^ 2 + 4)) := by
  refine hasSum_taylor (fun w : ℂ => 1 / (w ^ 2 + 4)) 1 (Real.sqrt 5) ?_ z
    (by rw [mem_ball, Complex.dist_eq]; exact hz)
  intro w hw
  have hw4 : w ^ 2 + 4 ≠ 0 := by
    intro h0
    have hfac : (w - 2 * I) * (w + 2 * I) = 0 := by
      have hI : I ^ 2 = -1 := Complex.I_sq
      linear_combination h0 - 4 * hI
    rw [mem_ball, Complex.dist_eq] at hw
    rcases mul_eq_zero.1 hfac with h | h
    · have : w = 2 * I := by linear_combination h
      rw [this] at hw
      have : ‖2 * I - 1‖ = Real.sqrt 5 := by
        have := norm_add_mul_I (-1) 2
        rw [show (2 * I - 1 : ℂ) = ((-1 : ℝ) : ℂ) + (2 : ℝ) * I by push_cast; ring, this]; norm_num
      linarith
    · have : w = -2 * I := by linear_combination h
      rw [this] at hw
      have : ‖-2 * I - 1‖ = Real.sqrt 5 := by
        have := norm_add_mul_I (-1) (-2)
        rw [show (-2 * I - 1 : ℂ) = ((-1 : ℝ) : ℂ) + (-2 : ℝ) * I by push_cast; ring, this]; norm_num
      linarith
  exact DifferentiableAt.differentiableWithinAt (by fun_prop (disch := exact hw4))

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_taylor_at_one
-- test: ca-res-taylor-at-one-test.sage

/-- `Log z = ∑ (-1)^(n+1) (z - 1)ⁿ/n` on `|z - 1| < 1`, the distance from `1` to the cut. -/
theorem hasSum_log_at_one (z : ℂ) (hz : ‖z - 1‖ < 1) :
    HasSum (fun n : ℕ => (-1) ^ (n + 1) * (z - 1) ^ n / n) (log z) := by
  simpa using Complex.hasSum_taylorSeries_log hz

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_log_at_one
-- test: ca-res-log-at-one-test.sage

/-! ## exr-taylor-cauchy-estimates -/

/-- The geometric expansion of the Cauchy kernel: `1/(w - z) = ∑ (z - c)ⁿ/(w - c)ⁿ⁺¹` for `|z - c| < |w - c|`. -/
theorem hasSum_cauchy_kernel (c w z : ℂ) (h : ‖z - c‖ < ‖w - c‖) :
    HasSum (fun n : ℕ => (z - c) ^ n / (w - c) ^ (n + 1)) (1 / (w - z)) := by
  have hw : w - c ≠ 0 := by
    intro h0; rw [h0, norm_zero] at h; exact (not_lt.2 (norm_nonneg _)) h
  have hq : ‖(z - c) / (w - c)‖ < 1 := by
    rw [norm_div, div_lt_one (norm_pos_iff.2 hw)]; exact h
  have := (hasSum_geometric_of_norm_lt_one hq).div_const (w - c)
  convert this using 1
  · funext n; rw [div_pow, pow_succ]; field_simp
  · have hwz : w - z ≠ 0 := by
      intro h0
      have : z - c = w - c := by linear_combination -h0
      rw [this] at h; exact lt_irrefl _ h
    have e : 1 - (z - c) / (w - c) = (w - z) / (w - c) := by field_simp; ring
    rw [e, inv_div, div_div, mul_comm, ← div_div, div_self hw, one_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.hasSum_cauchy_kernel
-- test: ca-res-cauchy-kernel-test.sage

/-- The Cauchy estimate: `|aₙ| ≤ M/rⁿ` when `|f| ≤ M` on the circle `|z - c| = r`. -/
theorem cauchy_estimate (f : ℂ → ℂ) (c : ℂ) (r M : ℝ) (hr : 0 < r) (hf : DiffContOnCl ℂ f (ball c r))
    (hM : ∀ z ∈ sphere c r, ‖f z‖ ≤ M) (n : ℕ) :
    ‖iteratedDeriv n f c / (Nat.factorial n : ℂ)‖ ≤ M / r ^ n := by
  have h := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le n hr hf hM
  have hf' : (0 : ℝ) < Nat.factorial n := by positivity
  rw [norm_div, Complex.norm_natCast, div_le_iff₀ hf']
  calc ‖iteratedDeriv n f c‖ ≤ Nat.factorial n * M / r ^ n := h
    _ = M / r ^ n * Nat.factorial n := by ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.TaylorSeries.cauchy_estimate
-- test: ca-res-cauchy-estimate-test.sage

/-- The ML estimate: `(1/2π) (M/rⁿ⁺¹) (2πr) = M/rⁿ`. -/
theorem ml_arith (M r : ℝ) (hr : r ≠ 0) (n : ℕ) :
    1 / (2 * π) * (M / r ^ (n + 1)) * (2 * π * r) = M / r ^ n := by
  have hpi : π ≠ 0 := Real.pi_ne_zero
  have hrn : r ^ n ≠ 0 := pow_ne_zero n hr
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.TaylorSeries.ml_arith
-- test: ca-res-ml-arith-test.sage

end KaluzaKlein.ComplexAnalysis.TaylorSeries

import Mathlib
import Veriphy.Kit.Levels

/-!
# Real analysis, part "Continuity and differentiability"

The claims of the section "Continuity and differentiability" of `tutorial/part-i-03-real-analysis.qmd`: limits of
functions (`ε`-`δ`, sequential, one-sided, at infinity), continuity with the extreme and intermediate value
theorems, the derivative and its rules, the mean value theorem, Taylor's theorem with the Lagrange remainder, and the
derivative and Taylor values of the running examples `sin x`, `e^x`, `x² e^x`.

Derivative values are certified by the `derivative` recipe; general theorems are tested at the chapter's functions.
-/

namespace KaluzaKlein.RealAnalysis.Continuity

open Filter Topology Real Set
open scoped Nat

/-! ## def-function-limit -/

/-- The `ε`-`δ` definition of `lim_{x → a} f(x) = L` is mathlib's `Tendsto f (𝓝[≠] a) (𝓝 L)`. -/
theorem limit_eps_delta (f : ℝ → ℝ) (a L : ℝ) :
    Tendsto f (𝓝[≠] a) (𝓝 L) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| → |x - a| < δ → |f x - L| < ε := by
  rw [Metric.tendsto_nhdsWithin_nhds]
  refine forall₂_congr (fun ε _ => exists_congr (fun δ => and_congr_right (fun _ => ?_)))
  constructor
  · intro h x h0 hx
    have := h (x := x) (by simpa [sub_eq_zero] using (abs_pos.1 h0)) (by rwa [Real.dist_eq])
    rwa [Real.dist_eq] at this
  · intro h x hx hd
    rw [Real.dist_eq] at hd ⊢
    exact h x (abs_pos.2 (sub_ne_zero.2 hx)) hd

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.limit_eps_delta
-- test: ra-limit-eps-delta-test.sage

/-- The sequential characterisation: `lim_{x → a} f(x) = L` if and only if `f(x_n) → L` for every sequence
`x_n → a` with `x_n ≠ a`. -/
theorem limit_sequential (f : ℝ → ℝ) (a L : ℝ) :
    Tendsto f (𝓝[≠] a) (𝓝 L) ↔
      ∀ x : ℕ → ℝ, (∀ n, x n ≠ a) → Tendsto x atTop (𝓝 a) → Tendsto (fun n => f (x n)) atTop (𝓝 L) := by
  rw [tendsto_iff_seq_tendsto]
  constructor
  · intro h x hx hlim
    exact h x (tendsto_nhdsWithin_iff.2 ⟨hlim, Eventually.of_forall hx⟩)
  · intro h x hx
    rw [tendsto_nhdsWithin_iff] at hx
    obtain ⟨hlim, hev⟩ := hx
    set y : ℕ → ℝ := fun n => if x n = a then a + 1 / ((n : ℝ) + 1) else x n with hy
    have hyne : ∀ n, y n ≠ a := fun n => by
      simp only [hy]
      split_ifs with h1
      · intro h2
        have : (0 : ℝ) < 1 / ((n : ℝ) + 1) := by positivity
        linarith
      · exact h1
    have heq : y =ᶠ[atTop] x := by
      filter_upwards [hev] with n hn
      simp only [hy, if_neg (Set.mem_compl_singleton_iff.1 hn)]
    have hylim : Tendsto y atTop (𝓝 a) := hlim.congr' heq.symm
    exact (h y hyne hylim).congr' (heq.fun_comp f)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.limit_sequential
-- test: ra-limit-sequential-test.sage

/-! ## exm-flimit-sinx-x -/

/-- `sin(cx)/x → c` as `x → 0`; at `c = 1`, `sin x / x → 1`. -/
theorem sin_div_limit (c : ℝ) : Tendsto (fun x => Real.sin (c * x) / x) (𝓝[≠] 0) (𝓝 c) := by
  have h : HasDerivAt (fun x => Real.sin (c * x)) (Real.cos (c * 0) * c) 0 :=
    (Real.hasDerivAt_sin (c * 0)).comp 0 ((hasDerivAt_id 0).const_mul c |>.congr_deriv (by simp))
  rw [hasDerivAt_iff_tendsto_slope] at h
  simp only [mul_zero, Real.cos_zero, one_mul] at h
  refine h.congr (fun x => ?_)
  simp [slope_def_field]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.sin_div_limit
-- test: ra-sin-div-limit-test.sage

/-- `(e^(cx) - 1)/x → c` as `x → 0`; at `c = 1`, `(e^x - 1)/x → 1`. -/
theorem exp_sub_one_div_limit (c : ℝ) : Tendsto (fun x => (Real.exp (c * x) - 1) / x) (𝓝[≠] 0) (𝓝 c) := by
  have h : HasDerivAt (fun x => Real.exp (c * x)) (Real.exp (c * 0) * c) 0 :=
    (Real.hasDerivAt_exp (c * 0)).comp 0 ((hasDerivAt_id 0).const_mul c |>.congr_deriv (by simp))
  rw [hasDerivAt_iff_tendsto_slope] at h
  simp only [mul_zero, Real.exp_zero, one_mul] at h
  refine h.congr (fun x => ?_)
  simp [slope_def_field]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.exp_sub_one_div_limit
-- test: ra-exp-sub-one-div-limit-test.sage

/-- `(1 - cos(cx))/x² → c²/2` as `x → 0`; at `c = 1` the limit is `1/2`. -/
theorem one_sub_cos_div_limit (c : ℝ) :
    Tendsto (fun x => (1 - Real.cos (c * x)) / x ^ 2) (𝓝[≠] 0) (𝓝 (c ^ 2 / 2)) := by
  have hc : Tendsto (fun x : ℝ => c ^ 2 / 2 * Real.sinc (c * x / 2) ^ 2) (𝓝 0) (𝓝 (c ^ 2 / 2)) := by
    have h1 : Tendsto (fun x : ℝ => Real.sinc (c * x / 2)) (𝓝 0) (𝓝 1) := by
      have := (Real.continuous_sinc.comp (by fun_prop : Continuous fun x : ℝ => c * x / 2)).tendsto 0
      simpa [Real.sinc_zero, Function.comp_def] using this
    simpa using (h1.pow 2).const_mul (c ^ 2 / 2)
  apply (hc.mono_left nhdsWithin_le_nhds).congr'
  filter_upwards [self_mem_nhdsWithin] with x hx
  have hx' : x ≠ 0 := hx
  by_cases hc0 : c = 0
  · simp [hc0]
  · have hne : c * x / 2 ≠ 0 := div_ne_zero (mul_ne_zero hc0 hx') two_ne_zero
    have hcos : Real.cos (c * x) = 1 - 2 * Real.sin (c * x / 2) ^ 2 := by
      have h := Real.cos_sq_add_sin_sq (c * x / 2)
      have h2 := Real.cos_two_mul (c * x / 2)
      rw [show 2 * (c * x / 2) = c * x by ring] at h2
      linarith
    rw [Real.sinc_of_ne_zero hne, hcos]
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.one_sub_cos_div_limit
-- test: ra-one-sub-cos-div-limit-test.sage

/-! ## exm-flimit-one-sided -/

/-- `|x - a|/(x - a)` has the one-sided limits `1` and `-1` at `a`, and no limit there. -/
theorem sign_one_sided (a : ℝ) :
    Tendsto (fun x => |x - a| / (x - a)) (𝓝[>] a) (𝓝 1) ∧ Tendsto (fun x => |x - a| / (x - a)) (𝓝[<] a) (𝓝 (-1)) ∧
      ¬ ∃ L, Tendsto (fun x => |x - a| / (x - a)) (𝓝[≠] a) (𝓝 L) := by
  have hr : Tendsto (fun x => |x - a| / (x - a)) (𝓝[>] a) (𝓝 1) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [self_mem_nhdsWithin] with x hx
    have : 0 < x - a := sub_pos.2 hx
    rw [abs_of_pos this, div_self this.ne']
  have hl : Tendsto (fun x => |x - a| / (x - a)) (𝓝[<] a) (𝓝 (-1)) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [self_mem_nhdsWithin] with x hx
    have : x - a < 0 := sub_neg.2 hx
    rw [abs_of_neg this, neg_div, div_self this.ne]
  refine ⟨hr, hl, ?_⟩
  rintro ⟨L, hL⟩
  have h1 := tendsto_nhds_unique
    (hL.mono_left (nhdsWithin_mono a (s := Ioi a) (t := {a}ᶜ) (fun x hx => ne_of_gt hx))) hr
  have h2 := tendsto_nhds_unique
    (hL.mono_left (nhdsWithin_mono a (s := Iio a) (t := {a}ᶜ) (fun x hx => ne_of_lt hx))) hl
  rw [h1] at h2
  norm_num at h2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.sign_one_sided
-- test: ra-sign-one-sided-test.sage

/-- `sin(1/x)` has no limit at `0`: for every `L`, `sin(1/x) ↛ L`. -/
theorem sin_inv_no_limit (L : ℝ) : ¬ Tendsto (fun x => Real.sin (1 / x)) (𝓝[≠] 0) (𝓝 L) := by
  intro h
  rw [limit_sequential] at h
  have hx : ∀ n : ℕ, (1 / (2 * π * ((n : ℝ) + 1))) ≠ 0 := fun n => by positivity
  have hy : ∀ n : ℕ, (1 / (2 * π * ((n : ℝ) + 1) + π / 2)) ≠ 0 := fun n => by positivity
  have hxl : Tendsto (fun n : ℕ => 1 / (2 * π * ((n : ℝ) + 1))) atTop (𝓝 0) := by
    apply Tendsto.div_atTop tendsto_const_nhds
    apply Tendsto.const_mul_atTop (by positivity)
    exact tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have hyl : Tendsto (fun n : ℕ => 1 / (2 * π * ((n : ℝ) + 1) + π / 2)) atTop (𝓝 0) := by
    apply Tendsto.div_atTop tendsto_const_nhds
    apply tendsto_atTop_add_const_right
    apply Tendsto.const_mul_atTop (by positivity)
    exact tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have h0 := h _ hx hxl
  have h1 := h _ hy hyl
  have e0 : ∀ n : ℕ, Real.sin (1 / (1 / (2 * π * ((n : ℝ) + 1)))) = 0 := fun n => by
    rw [one_div_one_div]
    have := Real.sin_nat_mul_pi (2 * (n + 1))
    push_cast at this
    rw [show 2 * π * ((n : ℝ) + 1) = 2 * ((n : ℝ) + 1) * π by ring]
    exact this
  have e1 : ∀ n : ℕ, Real.sin (1 / (1 / (2 * π * ((n : ℝ) + 1) + π / 2))) = 1 := fun n => by
    rw [one_div_one_div, add_comm, show 2 * π * ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) * (2 * π) by push_cast; ring,
      Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]
  simp_rw [e0] at h0
  simp_rw [e1] at h1
  have := tendsto_nhds_unique tendsto_const_nhds h0
  have := tendsto_nhds_unique tendsto_const_nhds h1
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.sin_inv_no_limit
-- test: ra-sin-inv-no-limit-test.sage

/-! ## exm-flimit-infinity -/

/-- Exponentials beat powers: `x^n e^(-x) → 0` as `x → ∞` (`x e^(-x)`, `x²/e^x`). -/
theorem pow_mul_exp_neg_limit (n : ℕ) : Tendsto (fun x => x ^ n * Real.exp (-x)) atTop (𝓝 0) :=
  Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero n

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.pow_mul_exp_neg_limit
-- test: ra-pow-mul-exp-neg-limit-test.sage

/-- Powers beat logarithms: `log x / x^r → 0` as `x → ∞` for `r > 0` (`(log x)/x`). -/
theorem log_div_rpow_limit (r : ℝ) (hr : 0 < r) : Tendsto (fun x => Real.log x / x ^ r) atTop (𝓝 0) :=
  (isLittleO_log_rpow_atTop hr).tendsto_div_nhds_zero

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.log_div_rpow_limit
-- test: ra-log-div-rpow-limit-test.sage

/-! ## exr-flimit-epsilon-delta -/

/-- `x² → 4` as `x → 2`: with `δ = min(1, ε/5)`, `|x - 2| < δ` gives `|x² - 4| < ε`. -/
theorem sq_limit_delta (ε : ℝ) (hε : 0 < ε) (x : ℝ) (hx : |x - 2| < min 1 (ε / 5)) : |x ^ 2 - 4| < ε := by
  have h1 : |x - 2| < 1 := lt_of_lt_of_le hx (min_le_left _ _)
  have h2 : |x - 2| < ε / 5 := lt_of_lt_of_le hx (min_le_right _ _)
  have h3 : |x + 2| < 5 := by
    rw [abs_lt] at h1 ⊢; constructor <;> linarith
  rw [show x ^ 2 - 4 = (x - 2) * (x + 2) by ring, abs_mul]
  calc |x - 2| * |x + 2| ≤ |x - 2| * 5 := by gcongr
    _ < ε / 5 * 5 := by gcongr
    _ = ε := by ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta (1) ((by norm_num)) (2) ((by norm_num)) := by
  norm_num [abs_lt]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta (1 / 10) ((by norm_num)) (201 / 100) ((by norm_num)) := by
  norm_num [abs_lt]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta (5) ((by norm_num)) (23 / 10) ((by norm_num)) := by
  norm_num [abs_lt]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta (1 / 2) ((by norm_num)) (99 / 50) ((by norm_num)) := by
  norm_num [abs_lt]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.sq_limit_delta
-- test: ra-sq-limit-delta-test.sage

/-! ## exr-flimit-sequential -/

/-- Along `x_n = 1/(2πn)` the values `sin(1/x_n)` are `0`, along `x'_n = 1/(2πn + π/2)` they are `1`. -/
theorem sin_inv_sequences (n : ℕ) :
    Real.sin (1 / (1 / (2 * π * ((n : ℝ) + 1)))) = 0 ∧ Real.sin (1 / (1 / (2 * π * ((n : ℝ) + 1) + π / 2))) = 1 := by
  constructor
  · rw [one_div_one_div]
    have := Real.sin_nat_mul_pi (2 * (n + 1))
    push_cast at this
    rw [show 2 * π * ((n : ℝ) + 1) = 2 * ((n : ℝ) + 1) * π by ring]
    exact this
  · rw [one_div_one_div, add_comm, show 2 * π * ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) * (2 * π) by push_cast; ring,
      Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_inv_sequences (0) := by
  constructor
  · rw [one_div_one_div, show (2 : ℝ) * Real.pi * (((0 : ℕ) : ℝ) + 1) = ((2 * (0 + 1) : ℕ) : ℝ) * Real.pi by push_cast; ring]
    exact Real.sin_nat_mul_pi _
  · rw [one_div_one_div, add_comm, show (2 : ℝ) * Real.pi * (((0 : ℕ) : ℝ) + 1) = (((0 + 1 : ℕ)) : ℝ) * (2 * Real.pi) by push_cast; ring,
      Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_inv_sequences (1) := by
  constructor
  · rw [one_div_one_div, show (2 : ℝ) * Real.pi * (((1 : ℕ) : ℝ) + 1) = ((2 * (1 + 1) : ℕ) : ℝ) * Real.pi by push_cast; ring]
    exact Real.sin_nat_mul_pi _
  · rw [one_div_one_div, add_comm, show (2 : ℝ) * Real.pi * (((1 : ℕ) : ℝ) + 1) = (((1 + 1 : ℕ)) : ℝ) * (2 * Real.pi) by push_cast; ring,
      Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_inv_sequences (4) := by
  constructor
  · rw [one_div_one_div, show (2 : ℝ) * Real.pi * (((4 : ℕ) : ℝ) + 1) = ((2 * (4 + 1) : ℕ) : ℝ) * Real.pi by push_cast; ring]
    exact Real.sin_nat_mul_pi _
  · rw [one_div_one_div, add_comm, show (2 : ℝ) * Real.pi * (((4 : ℕ) : ℝ) + 1) = (((4 + 1 : ℕ)) : ℝ) * (2 * Real.pi) by push_cast; ring,
      Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.sin_inv_sequences
-- test: ra-sin-inv-sequences-test.sage

/-! ## def-continuity -/

/-- Polynomials, `exp`, `sin`, `cos` are continuous on `ℝ` and `log` on `(0, ∞)`. -/
theorem elementary_continuous (P : Polynomial ℝ) :
    Continuous (fun x => P.eval x) ∧ Continuous Real.exp ∧ Continuous Real.sin ∧ Continuous Real.cos ∧
      ContinuousOn Real.log (Ioi 0) :=
  ⟨P.continuous, Real.continuous_exp, Real.continuous_sin, Real.continuous_cos,
    Real.continuousOn_log.mono (fun x hx => ne_of_gt hx)⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.elementary_continuous ((0 : Polynomial ℝ)) := by
  refine ⟨Polynomial.continuous _, Real.continuous_exp, Real.continuous_sin, Real.continuous_cos, Real.continuousOn_log.mono (fun x hx => ne_of_gt hx)⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.elementary_continuous ((Polynomial.X ^ 2 + 1 : Polynomial ℝ)) := by
  refine ⟨Polynomial.continuous _, Real.continuous_exp, Real.continuous_sin, Real.continuous_cos, Real.continuousOn_log.mono (fun x hx => ne_of_gt hx)⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.elementary_continuous ((Polynomial.C 3 * Polynomial.X ^ 5 : Polynomial ℝ)) := by
  refine ⟨Polynomial.continuous _, Real.continuous_exp, Real.continuous_sin, Real.continuous_cos, Real.continuousOn_log.mono (fun x hx => ne_of_gt hx)⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.elementary_continuous
-- test: ra-elementary-continuous-test.sage

/-- Sums, products, quotients with non-vanishing denominator and compositions of continuous functions are
continuous. -/
theorem continuity_rules (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g) :
    Continuous (fun x => f x + g x) ∧ Continuous (fun x => f x * g x) ∧ Continuous (g ∘ f) ∧
      ((∀ x, g x ≠ 0) → Continuous (fun x => f x / g x)) :=
  ⟨hf.add hg, hf.mul hg, hg.comp hf, fun h => hf.div hg h⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.continuity_rules (Real.sin) (Real.exp) (Real.continuous_sin) (Real.continuous_exp) := by
  refine ⟨by fun_prop, by fun_prop, by fun_prop, fun h => by fun_prop⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.continuity_rules ((fun x : ℝ => x ^ 2)) (Real.cos) ((continuous_pow 2)) (Real.continuous_cos) := by
  refine ⟨by fun_prop, by fun_prop, by fun_prop, fun h => by fun_prop⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.continuity_rules ((fun _ : ℝ => (0 : ℝ))) ((fun _ : ℝ => (1 : ℝ))) (continuous_const) (continuous_const) := by
  refine ⟨by fun_prop, by fun_prop, by fun_prop, fun h => by fun_prop⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.continuity_rules
-- test: ra-continuity-rules-test.sage

/-- The extreme value theorem: a continuous function on `[a, b]` attains its maximum and minimum. -/
theorem extreme_value (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : ContinuousOn f (Icc a b)) :
    (∃ c ∈ Icc a b, ∀ x ∈ Icc a b, f x ≤ f c) ∧ ∃ d ∈ Icc a b, ∀ x ∈ Icc a b, f d ≤ f x := by
  have hne : (Icc a b).Nonempty := nonempty_Icc.2 hab
  obtain ⟨c, hc, hmax⟩ := isCompact_Icc.exists_isMaxOn hne hf
  obtain ⟨d, hd, hmin⟩ := isCompact_Icc.exists_isMinOn hne hf
  exact ⟨⟨c, hc, fun x hx => hmax hx⟩, ⟨d, hd, fun x hx => hmin hx⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.extreme_value ((fun x : ℝ => x ^ 2)) (-1) (2) ((by norm_num)) ((by fun_prop)) := by
  exact ⟨⟨2, by norm_num, fun x hx => by beta_reduce; nlinarith [hx.1, hx.2]⟩, ⟨0, by norm_num, fun x hx => by beta_reduce; nlinarith [sq_nonneg x]⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.extreme_value ((fun x : ℝ => (1 : ℝ))) (0) (1) ((by norm_num)) ((by fun_prop)) := by
  exact ⟨⟨0, by norm_num, fun x hx => le_rfl⟩, ⟨0, by norm_num, fun x hx => le_rfl⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.extreme_value ((fun x : ℝ => x)) (-3) (5) ((by norm_num)) ((by fun_prop)) := by
  exact ⟨⟨5, by norm_num, fun x hx => hx.2⟩, ⟨-3, by norm_num, fun x hx => hx.1⟩⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.extreme_value
-- test: ra-extreme-value-test.sage

/-- The intermediate value theorem: a continuous function on `[a, b]` takes every value between `f(a)` and
`f(b)`. -/
theorem intermediate_value (f : ℝ → ℝ) (a b y : ℝ) (hab : a ≤ b) (hf : ContinuousOn f (Icc a b))
    (hy : y ∈ Icc (f a) (f b)) : ∃ c ∈ Icc a b, f c = y :=
  intermediate_value_Icc hab hf hy

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.intermediate_value ((fun x : ℝ => x ^ 2)) (0) (1) (1 / 4) ((by norm_num)) ((by fun_prop)) ((by norm_num)) := by
  exact ⟨1 / 2, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.intermediate_value ((fun x : ℝ => x)) (0) (5) (3) ((by norm_num)) ((by fun_prop)) ((by norm_num)) := by
  exact ⟨3, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.intermediate_value ((fun x : ℝ => x ^ 3)) (1) (3) (8) ((by norm_num)) ((by fun_prop)) ((by norm_num)) := by
  exact ⟨2, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.intermediate_value ((fun x : ℝ => x)) (0) (0) (0) ((by norm_num)) ((by fun_prop)) ((by norm_num)) := by
  exact ⟨0, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.intermediate_value
-- test: ra-intermediate-value-test.sage

/-! ## exm-continuity-piecewise -/

/-- Gluing `x²` (for `x < a`) to the line `2ax - a² + d` (for `x ≥ a`) is continuous at `a` exactly when `d = 0`;
at `a = 1`: `2x - 1` glues continuously, `2x` jumps. -/
theorem glue_continuous (a d : ℝ) :
    ContinuousAt (fun x => if x < a then x ^ 2 else 2 * a * x - a ^ 2 + d) a ↔ d = 0 := by
  constructor
  · intro h
    have h1 := h.tendsto.mono_left (nhdsWithin_le_nhds (s := Iio a))
    have h2 : Tendsto (fun x => if x < a then x ^ 2 else 2 * a * x - a ^ 2 + d) (𝓝[<] a) (𝓝 (a ^ 2)) := by
      have : Tendsto (fun x : ℝ => x ^ 2) (𝓝[<] a) (𝓝 (a ^ 2)) :=
        ((continuous_pow 2).tendsto a).mono_left nhdsWithin_le_nhds
      apply this.congr'
      filter_upwards [self_mem_nhdsWithin] with x hx
      rw [if_pos (show x < a from hx)]
    have := tendsto_nhds_unique h1 h2
    simp only [lt_irrefl, if_false] at this
    linarith
  · rintro rfl
    have e : (fun x => if x < a then x ^ 2 else 2 * a * x - a ^ 2 + 0) = fun x => x ^ 2 - (max (x - a) 0) ^ 2 := by
      funext x
      split_ifs with hx
      · rw [max_eq_right (by linarith)]; ring
      · rw [max_eq_left (by linarith)]; ring
    rw [e]
    exact (by fun_prop : Continuous fun x : ℝ => x ^ 2 - (max (x - a) 0) ^ 2).continuousAt

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.glue_continuous
-- test: ra-glue-continuous-test.sage

/-! ## exm-continuity-removable -/

/-- `sin(cx)/x`, given the value `c` at `0`, is continuous on `ℝ`: the discontinuity is removable. -/
theorem sin_div_removable (c : ℝ) : Continuous (fun x => if x = 0 then c else Real.sin (c * x) / x) := by
  have h : (fun x => if x = 0 then c else Real.sin (c * x) / x) = fun x => c * Real.sinc (c * x) := by
    funext x
    by_cases hx : x = 0
    · simp [hx]
    · by_cases hc : c = 0
      · simp [hx, hc]
      · rw [if_neg hx, Real.sinc_of_ne_zero (mul_ne_zero hc hx)]
        field_simp
  rw [h]
  exact continuous_const.mul (Real.continuous_sinc.comp (continuous_const.mul continuous_id))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.sin_div_removable
-- test: ra-sin-div-removable-test.sage

/-- The jump of `|x|/x` at `0` is not removable: no value `v` at `0` makes it continuous. -/
theorem sign_not_removable (v : ℝ) : ¬ ContinuousAt (fun x => if x = 0 then v else |x| / x) 0 := by
  intro h
  have h1 := h.tendsto.mono_left (nhdsWithin_le_nhds (s := Ioi 0))
  have h2 := h.tendsto.mono_left (nhdsWithin_le_nhds (s := Iio 0))
  have hr := (sign_one_sided 0).1
  have hl := (sign_one_sided 0).2.1
  simp only [sub_zero] at hr hl
  have e1 : Tendsto (fun x => if x = 0 then v else |x| / x) (𝓝[>] 0) (𝓝 1) := by
    apply hr.congr'
    filter_upwards [self_mem_nhdsWithin] with x hx
    rw [if_neg (ne_of_gt hx)]
  have e2 : Tendsto (fun x => if x = 0 then v else |x| / x) (𝓝[<] 0) (𝓝 (-1)) := by
    apply hl.congr'
    filter_upwards [self_mem_nhdsWithin] with x hx
    rw [if_neg (ne_of_lt hx)]
  have := tendsto_nhds_unique h1 e1
  have := tendsto_nhds_unique h2 e2
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.sign_not_removable
-- test: ra-sign-not-removable-test.sage

/-- `1/x` is unbounded near `0`: for every `M` some `x ∈ (0, 1]` has `1/x > M`. -/
theorem inv_unbounded (M : ℝ) : ∃ x ∈ Ioc (0 : ℝ) 1, M < 1 / x := by
  refine ⟨1 / (|M| + 2), ⟨by positivity, ?_⟩, ?_⟩
  · rw [div_le_one (by positivity)]; linarith [abs_nonneg M]
  · rw [one_div_one_div]; linarith [le_abs_self M]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.inv_unbounded (0) := by
  exact ⟨1 / 2, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.inv_unbounded (10) := by
  exact ⟨1 / 11, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.inv_unbounded (1000) := by
  exact ⟨1 / 1001, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.inv_unbounded (-5) := by
  exact ⟨1, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.inv_unbounded
-- test: ra-inv-unbounded-test.sage

/-! ## exm-continuity-ivt -/

/-- `x³ - y` changes sign on `[1, 2]` for `1 < y < 8`, so it has a root in `(1, 2)`; at `y = 2`: `f(1) = -1 < 0 <
f(2) = 6` and the root is `∛2`. -/
theorem cube_root_exists (y : ℝ) (hy1 : 1 < y) (hy8 : y < 8) :
    (1 : ℝ) ^ 3 - y < 0 ∧ 0 < (2 : ℝ) ^ 3 - y ∧ ∃ c ∈ Ioo (1 : ℝ) 2, c ^ 3 = y := by
  refine ⟨by norm_num; linarith, by norm_num; linarith, ?_⟩
  have hc : ContinuousOn (fun x : ℝ => x ^ 3) (Icc 1 2) := by fun_prop
  obtain ⟨c, hc1, hc2⟩ := intermediate_value_Icc (by norm_num : (1 : ℝ) ≤ 2) hc
    (show y ∈ Icc ((1 : ℝ) ^ 3) ((2 : ℝ) ^ 3) by constructor <;> norm_num <;> linarith)
  refine ⟨c, ⟨?_, ?_⟩, hc2⟩
  · rcases hc1.1.lt_or_eq with h | h
    · exact h
    · rw [← h] at hc2; norm_num at hc2; linarith
  · rcases hc1.2.lt_or_eq with h | h
    · exact h
    · rw [h] at hc2; norm_num at hc2; linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.cube_root_exists (27 / 8) ((by norm_num)) ((by norm_num)) := by
  exact ⟨by norm_num, by norm_num, 3 / 2, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.cube_root_exists (64 / 27) ((by norm_num)) ((by norm_num)) := by
  exact ⟨by norm_num, by norm_num, 4 / 3, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.cube_root_exists (125 / 64) ((by norm_num)) ((by norm_num)) := by
  exact ⟨by norm_num, by norm_num, 5 / 4, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.cube_root_exists
-- test: ra-cube-root-exists-test.sage

/-! ## exr-continuity-check -/

/-- The tangent gluing `x²` / `2ax - a²` is differentiable at `a` with derivative `2a` (both one-sided derivatives
are `2a`); at `a = 1` the derivative is `2`. -/
theorem glue_deriv (a : ℝ) : HasDerivAt (fun x => if x < a then x ^ 2 else 2 * a * x - a ^ 2) (2 * a) a := by
  have e : (fun x => if x < a then x ^ 2 else 2 * a * x - a ^ 2) = fun x => x ^ 2 - (max (x - a) 0) ^ 2 := by
    funext x
    split_ifs with hx
    · rw [max_eq_right (by linarith)]; ring
    · rw [max_eq_left (by linarith)]; ring
  rw [e]
  have hg : HasDerivAt (fun x : ℝ => (max (x - a) 0) ^ 2) 0 a := by
    rw [hasDerivAt_iff_tendsto_slope_zero]
    have hb : ∀ t : ℝ, ‖t⁻¹ • ((max (a + t - a) 0) ^ 2 - (max (a - a) 0) ^ 2)‖ ≤ |t| := by
      intro t
      rw [sub_self, max_self, add_sub_cancel_left, zero_pow two_ne_zero, sub_zero, smul_eq_mul, Real.norm_eq_abs,
        abs_mul, abs_inv]
      rcases eq_or_ne t 0 with rfl | ht
      · simp
      · have h1 : (max t 0) ^ 2 ≤ t ^ 2 := by
          rcases le_total t 0 with h | h
          · rw [max_eq_right h, zero_pow two_ne_zero]; positivity
          · rw [max_eq_left h]
        have h2 : |(max t 0) ^ 2| ≤ |t| ^ 2 := by rw [abs_of_nonneg (by positivity), sq_abs]; exact h1
        have ht' : 0 < |t| := abs_pos.2 ht
        calc |t|⁻¹ * |(max t 0) ^ 2| ≤ |t|⁻¹ * |t| ^ 2 := by gcongr
          _ = |t| := by field_simp
    apply squeeze_zero_norm' (Eventually.of_forall hb)
    have : Tendsto (fun t : ℝ => |t|) (𝓝 0) (𝓝 0) := by simpa using (continuous_abs.tendsto (0 : ℝ))
    exact this.mono_left nhdsWithin_le_nhds
  have := (hasDerivAt_pow 2 a).sub hg
  convert this using 1
  norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.glue_deriv
-- test: ra-glue-deriv-test.sage

/-! ## exr-continuity-composition -/

/-- A continuous `f : [0, 1] → [0, 1]` has a fixed point. -/
theorem fixed_point (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (hmaps : MapsTo f (Icc 0 1) (Icc 0 1)) :
    ∃ x ∈ Icc (0 : ℝ) 1, f x = x := by
  have hg : ContinuousOn (fun x => f x - x) (Icc 0 1) := hf.sub continuousOn_id
  have h0 : 0 ≤ f 0 - 0 := by have := (hmaps (left_mem_Icc.2 zero_le_one)).1; linarith
  have h1 : f 1 - 1 ≤ 0 := by have := (hmaps (right_mem_Icc.2 zero_le_one)).2; linarith
  obtain ⟨c, hc, hc'⟩ := intermediate_value_Icc' (zero_le_one' ℝ) hg (show (0 : ℝ) ∈ Icc (f 1 - 1) (f 0 - 0) from
    ⟨h1, h0⟩)
  exact ⟨c, hc, by simp only at hc'; linarith⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.fixed_point ((fun x : ℝ => 1 - x)) ((by fun_prop)) ((fun x hx => ⟨by simp only [Set.mem_Icc] at hx; linarith [hx.2], by simp only [Set.mem_Icc] at hx; linarith [hx.1]⟩)) := by
  exact ⟨1 / 2, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.fixed_point ((fun x : ℝ => x ^ 2)) ((by fun_prop)) ((fun x hx => ⟨by positivity, by simp only [Set.mem_Icc] at hx; nlinarith [hx.1, hx.2]⟩)) := by
  exact ⟨0, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.fixed_point ((fun x : ℝ => (x + 1) / 2)) ((by fun_prop)) ((fun x hx => ⟨by simp only [Set.mem_Icc] at hx; linarith [hx.1], by simp only [Set.mem_Icc] at hx; linarith [hx.2]⟩)) := by
  exact ⟨1, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.fixed_point
-- test: ra-fixed-point-test.sage

/-! ## exr-continuity-extreme -/

/-- `x/(1 + x)` on `[0, ∞)` is bounded by `1` and never attains it: every value is below `1` and values exceed
`1 - ε` for every `ε > 0`. -/
theorem sat_not_attained (ε : ℝ) (hε : 0 < ε) :
    (∀ x : ℝ, 0 ≤ x → x / (1 + x) < 1) ∧ ∃ x : ℝ, 0 ≤ x ∧ 1 - ε < x / (1 + x) := by
  refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, ?_⟩
  refine ⟨1 / ε, by positivity, ?_⟩
  have h : 1 / ε / (1 + 1 / ε) = 1 / (1 + ε) := by field_simp <;> ring
  rw [h, lt_div_iff₀ (by linarith)]
  nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sat_not_attained (1) ((by norm_num)) := by
  refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 1, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sat_not_attained (1 / 10) ((by norm_num)) := by
  refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 10, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sat_not_attained (1 / 1000) ((by norm_num)) := by
  refine ⟨fun x hx => by rw [div_lt_one (by linarith)]; linarith, 1000, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.sat_not_attained
-- test: ra-sat-not-attained-test.sage

/-- `x` on `(0, 1)` attains neither its supremum nor its infimum: every point has larger and smaller ones. -/
theorem id_no_extremum (x : ℝ) (hx : x ∈ Ioo (0 : ℝ) 1) : (∃ y ∈ Ioo (0 : ℝ) 1, x < y) ∧ ∃ y ∈ Ioo (0 : ℝ) 1, y < x := by
  obtain ⟨h0, h1⟩ := hx
  exact ⟨⟨(x + 1) / 2, ⟨by linarith, by linarith⟩, by linarith⟩, ⟨x / 2, ⟨by linarith, by linarith⟩, by linarith⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.id_no_extremum (1 / 2) ((by norm_num)) := by
  exact ⟨⟨3 / 4, by norm_num, by norm_num⟩, ⟨1 / 4, by norm_num, by norm_num⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.id_no_extremum (1 / 100) ((by norm_num)) := by
  exact ⟨⟨1 / 2, by norm_num, by norm_num⟩, ⟨1 / 200, by norm_num, by norm_num⟩⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.id_no_extremum (99 / 100) ((by norm_num)) := by
  exact ⟨⟨199 / 200, by norm_num, by norm_num⟩, ⟨1 / 2, by norm_num, by norm_num⟩⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.id_no_extremum
-- test: ra-id-no-extremum-test.sage

/-! ## def-derivative -/

/-- A differentiable function is continuous. -/
theorem diff_continuous (f : ℝ → ℝ) (a f' : ℝ) (h : HasDerivAt f f' a) : ContinuousAt f a :=
  h.continuousAt

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.diff_continuous
-- test: ra-diff-continuous-test.sage

/-- The derivative is the limit of the difference quotient, and `f(a + h) = f(a) + f'(a) h + o(h)`. -/
theorem deriv_slope (f : ℝ → ℝ) (a f' : ℝ) :
    HasDerivAt f f' a ↔ Tendsto (fun h => (f (a + h) - f a) / h) (𝓝[≠] 0) (𝓝 f') := by
  rw [hasDerivAt_iff_tendsto_slope_zero]
  simp only [smul_eq_mul]
  constructor <;> intro h <;> refine h.congr (fun t => ?_) <;> field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.deriv_slope
-- test: ra-deriv-slope-test.sage

/-- The product rule. -/
theorem product_rule (f g : ℝ → ℝ) (a f' g' : ℝ) (hf : HasDerivAt f f' a) (hg : HasDerivAt g g' a) :
    HasDerivAt (fun x => f x * g x) (f' * g a + f a * g') a :=
  hf.mul hg

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.product_rule
-- test: ra-product-rule-test.sage

/-- The quotient rule. -/
theorem quotient_rule (f g : ℝ → ℝ) (a f' g' : ℝ) (hf : HasDerivAt f f' a) (hg : HasDerivAt g g' a)
    (hga : g a ≠ 0) : HasDerivAt (fun x => f x / g x) ((f' * g a - f a * g') / g a ^ 2) a :=
  hf.div hg hga

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.quotient_rule
-- test: ra-quotient-rule-test.sage

/-- The chain rule. -/
theorem chain_rule (f g : ℝ → ℝ) (a f' g' : ℝ) (hf : HasDerivAt f f' a) (hg : HasDerivAt g g' (f a)) :
    HasDerivAt (fun x => g (f x)) (g' * f') a :=
  hg.comp a hf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.chain_rule
-- test: ra-chain-rule-test.sage

/-- The inverse function rule: `(f⁻¹)'(y) = 1/f'(x)` at `y = f(x)` when `f'(x) ≠ 0`. -/
theorem inverse_rule (f g : ℝ → ℝ) (x f' : ℝ) (hg : ContinuousAt g (f x)) (hf : HasDerivAt f f' x)
    (hf' : f' ≠ 0) (hgf : ∀ᶠ y in 𝓝 (f x), f (g y) = y) (hgx : g (f x) = x) :
    HasDerivAt g (1 / f') (f x) := by
  rw [one_div]
  have hf2 : HasDerivAt f f' (g (f x)) := by rw [hgx]; exact hf
  exact HasDerivAt.of_local_left_inverse hg hf2 hf' hgf

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.inverse_rule
-- test: ra-inverse-rule-test.sage

/-! ## exm-derivative-power -/

/-- `(x^n)' = n x^(n-1)`. -/
theorem deriv_pow_rule (n : ℕ) (x : ℝ) : deriv (fun x : ℝ => x ^ n) x = n * x ^ (n - 1) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule (0) (5) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule (3) (2) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule (5) (-1) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule (2) (1 / 2) := by
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule
-- test: ra-deriv-pow-test.sage

/-- `(e^x)' = e^x`. -/
theorem deriv_exp_rule (x : ℝ) : deriv (fun x => Real.exp x) x = Real.exp x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => rexp x) ((Real.exp x : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).exp) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-deriv-exp.sage

/-- `(log x)' = 1/x` for `x > 0`. -/
theorem deriv_log_rule (x : ℝ) (hx : 0 < x) : deriv (fun x => Real.log x) x = 1 / x := by
  have hx' : x ≠ 0 := hx.ne'
  clear hx
  show deriv (fun x => Real.log x) x = 1 / x
  veriphy_certified
    ( have hd : HasDerivAt (fun x => log x) ((1 / x ^ 1 : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).log hx') using 1
        all_goals ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-deriv-log.sage

/-- `(sin x)' = cos x`. -/
theorem deriv_sin_rule (x : ℝ) : deriv (fun x => Real.sin x) x = Real.cos x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => sin x) ((Real.cos x : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).sin) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-deriv-sin.sage

/-! ## exm-derivative-chain -/

/-- `(sin x²)' = 2x cos x²`. -/
theorem deriv_sin_sq (x : ℝ) : deriv (fun x => Real.sin (x ^ 2)) x = 2 * x * Real.cos (x ^ 2) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => sin (x ^ 2)) ((2 * x * (Real.cos (x ^ 2)) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).sin) using 1
        all_goals (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-deriv-sin-sq.sage

/-- `(x² e^x)' = (x² + 2x) e^x`. -/
theorem deriv_sq_exp (x : ℝ) : deriv (fun x => x ^ 2 * Real.exp x) x = (x ^ 2 + 2 * x) * Real.exp x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2 * rexp x) (((x ^ 2) * (Real.exp x) + (2 * x * (Real.exp x)) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).mul ((hasDerivAt_id' (x : ℝ)).exp)) using 1
        all_goals (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-deriv-sq-exp.sage

/-- `(tan x)' = (sin x / cos x)' = 1/cos² x` where `cos x ≠ 0`. -/
theorem deriv_tan_rule (x : ℝ) (hx : Real.cos x ≠ 0) :
    deriv (fun x => Real.sin x / Real.cos x) x = 1 / Real.cos x ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => sin x / cos x) ((((Real.sin x) ^ 2) / ((Real.cos x) ^ 2) + 1 : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).sin).div ((hasDerivAt_id' (x : ℝ)).cos) (by exact hx)) using 1
        all_goals (field_simp; ring1)
      exact hd.deriv.trans (by (field_simp; linear_combination Real.sin_sq_add_cos_sq x)))
  -- certificate: ra-deriv-tan.sage

/-! ## exm-derivative-corner -/

/-- `|x|` is continuous at `0` but not differentiable there: the difference quotient `|h|/h` has one-sided limits
`±1`. -/
theorem abs_corner (a : ℝ) :
    Continuous (fun x : ℝ => |x - a|) ∧ ¬ DifferentiableAt ℝ (fun x : ℝ => |x - a|) a := by
  refine ⟨by fun_prop, fun h => not_differentiableAt_abs_zero ?_⟩
  have h1 : DifferentiableAt ℝ (fun x : ℝ => x + a) 0 := by fun_prop
  have h2 := (zero_add a ▸ h).comp 0 h1
  simpa [Function.comp_def] using h2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.abs_corner
-- test: ra-abs-corner-test.sage

/-- Weierstrass's function `W(x) = ∑ aⁿ cos(bⁿ π x)` (`0 < a < 1`) is continuous everywhere. -/
theorem weierstrass_continuous (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) :
    Continuous (fun x => ∑' n : ℕ, a ^ n * Real.cos (b ^ n * π * x)) := by
  apply continuous_tsum (u := fun n => a ^ n)
  · intro n; fun_prop
  · exact summable_geometric_of_lt_one ha.le ha1
  · intro n x
    rw [norm_mul, Real.norm_eq_abs, Real.norm_eq_abs, abs_pow, abs_of_pos ha]
    exact mul_le_of_le_one_right (by positivity) (Real.abs_cos_le_one _)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.weierstrass_continuous
-- test: ra-weierstrass-continuous-test.sage

/-- Weierstrass's function with `a = 1/2`, `b = 13` (`ab > 1 + 3π/2`) is differentiable nowhere (stated, tested
numerically by the growth of its difference quotients, not proved here). -/
def WeierstrassNowhereDifferentiable : Prop :=
  ∀ x : ℝ, ¬ DifferentiableAt ℝ (fun x => ∑' n : ℕ, (1 / 2 : ℝ) ^ n * Real.cos (13 ^ n * π * x)) x

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.WeierstrassNowhereDifferentiable
-- test: ra-weierstrassnowheredifferentiable-test.sage

/-! ## exr-derivative-sage -/

/-- `(log(x + h) - log x)/h → 1/x` as `h → 0`, for `x > 0`: the derivative of `log` from the definition. -/
theorem log_slope_limit (x : ℝ) (hx : 0 < x) :
    Tendsto (fun h => (Real.log (x + h) - Real.log x) / h) (𝓝[≠] 0) (𝓝 (1 / x)) := by
  have h := (Real.hasDerivAt_log hx.ne')
  rw [deriv_slope] at h
  simpa [one_div] using h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.log_slope_limit
-- test: ra-log-slope-limit-test.sage

/-! ## exr-derivative-inverse -/

/-- `(arcsin x)' = 1/√(1 - x²)` and `(arctan x)' = 1/(1 + x²)` for `-1 < x < 1`. -/
theorem arcsin_arctan_deriv (x : ℝ) (hx : -1 < x) (hx1 : x < 1) :
    HasDerivAt Real.arcsin (1 / √(1 - x ^ 2)) x ∧ HasDerivAt Real.arctan (1 / (1 + x ^ 2)) x :=
  ⟨Real.hasDerivAt_arcsin (ne_of_gt hx) (ne_of_lt hx1), Real.hasDerivAt_arctan x⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.arcsin_arctan_deriv
-- test: ra-arcsin-arctan-deriv-test.sage

/-! ## thm-mean-value -/

/-- The mean value theorem. -/
theorem mean_value (f f' : ℝ → ℝ) (a b : ℝ) (hab : a < b) (hf : ContinuousOn f (Icc a b))
    (hd : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) : ∃ c ∈ Ioo a b, f' c = (f b - f a) / (b - a) :=
  exists_hasDerivAt_eq_slope f f' hab hf hd

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.mean_value
-- test: ra-mean-value-test.sage

/-- Rolle's theorem: equal endpoint values give a critical point. -/
theorem rolle (f f' : ℝ → ℝ) (a b : ℝ) (hab : a < b) (hf : ContinuousOn f (Icc a b))
    (hd : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) (he : f a = f b) : ∃ c ∈ Ioo a b, f' c = 0 :=
  exists_hasDerivAt_eq_zero hab hf he hd

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.rolle
-- test: ra-rolle-test.sage

/-- `f' = 0` on `ℝ` forces `f` constant. -/
theorem deriv_zero_const (f : ℝ → ℝ) (hf : Differentiable ℝ f) (h : ∀ x, deriv f x = 0) (x y : ℝ) : f x = f y :=
  is_const_of_deriv_eq_zero hf h x y

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.deriv_zero_const
-- test: ra-deriv-zero-const-test.sage

/-- `f' > 0` forces `f` strictly increasing. -/
theorem deriv_pos_increasing (f : ℝ → ℝ) (h : ∀ x, 0 < deriv f x) : StrictMono f :=
  strictMono_of_deriv_pos h

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.deriv_pos_increasing
-- test: ra-deriv-pos-increasing-test.sage

/-- `|f'| ≤ M` gives `|f(x) - f(y)| ≤ M |x - y|`. -/
theorem deriv_bound_lipschitz (f : ℝ → ℝ) (M : ℝ) (hf : Differentiable ℝ f) (h : ∀ x, |deriv f x| ≤ M)
    (x y : ℝ) : |f x - f y| ≤ M * |x - y| := by
  have := Convex.norm_image_sub_le_of_norm_deriv_le (s := univ) (fun z _ => hf z) (fun z _ => by
    rw [Real.norm_eq_abs]; exact h z) convex_univ (mem_univ y) (mem_univ x)
  simpa [Real.norm_eq_abs] using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.deriv_bound_lipschitz
-- test: ra-deriv-bound-lipschitz-test.sage

/-! ## exm-mvt-cubic -/

/-- For `x³` on `[0, b]` the average slope is `b²` and the mean value point is `c = b/√3 ∈ (0, b)`; at `b = 2`:
slope `4`, `c = 2/√3 ≈ 1.155`. -/
theorem mvt_power (b : ℝ) (hb : 0 < b) :
    3 * (b / √3) ^ 2 = (b ^ 3 - 0 ^ 3) / (b - 0) ∧ 0 < b / √3 ∧ b / √3 < b ∧
      0.577 * b < b / √3 ∧ b / √3 < 0.578 * b := by
  have h3 : √(3 : ℝ) ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 0 < √(3 : ℝ) := by positivity
  have hs1 : 1.732 < √(3 : ℝ) := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √(3 : ℝ) < 1.7321 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hb0 : b ≠ 0 := hb.ne'
  refine ⟨?_, by positivity, ?_, ?_, ?_⟩
  · rw [div_pow, h3]
    field_simp <;> ring
  · rw [div_lt_iff₀ hs]; nlinarith
  · rw [lt_div_iff₀ hs]; nlinarith
  · rw [div_lt_iff₀ hs]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_power (2) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 0 < Real.sqrt 3 := by positivity
  have hs1 : 1.732 < Real.sqrt 3 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : Real.sqrt 3 < 1.7321 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  refine ⟨?_, by positivity, ?_, ?_, ?_⟩
  · field_simp; rw [h3]; ring
  · rw [div_lt_iff₀ hs]; nlinarith
  · rw [lt_div_iff₀ hs]; nlinarith
  · rw [div_lt_iff₀ hs]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_power (1) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 0 < Real.sqrt 3 := by positivity
  have hs1 : 1.732 < Real.sqrt 3 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : Real.sqrt 3 < 1.7321 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  refine ⟨?_, by positivity, ?_, ?_, ?_⟩
  · field_simp; rw [h3]; ring
  · rw [div_lt_iff₀ hs]; nlinarith
  · rw [lt_div_iff₀ hs]; nlinarith
  · rw [div_lt_iff₀ hs]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_power (3) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 0 < Real.sqrt 3 := by positivity
  have hs1 : 1.732 < Real.sqrt 3 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : Real.sqrt 3 < 1.7321 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  refine ⟨?_, by positivity, ?_, ?_, ?_⟩
  · field_simp; rw [h3]; ring
  · rw [div_lt_iff₀ hs]; nlinarith
  · rw [lt_div_iff₀ hs]; nlinarith
  · rw [div_lt_iff₀ hs]; nlinarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.mvt_power
-- test: ra-mvt-power-test.sage

/-- For `sin` on `[0, kπ]` the average slope is `0` and `c = π/2` has `cos c = 0`. -/
theorem mvt_sin (k : ℕ) (hk : 1 ≤ k) :
    Real.cos (π / 2) = (Real.sin (k * π) - Real.sin 0) / (k * π - 0) ∧ π / 2 ∈ Ioo 0 ((k : ℝ) * π) := by
  have hk1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
  refine ⟨by rw [Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.sin_zero]; simp, ⟨by positivity, ?_⟩⟩
  nlinarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_sin (1) ((by norm_num)) := by
  refine ⟨by rw [Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.sin_zero]; simp, ⟨by positivity, ?_⟩⟩
  norm_num
  nlinarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_sin (2) ((by norm_num)) := by
  refine ⟨by rw [Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.sin_zero]; simp, ⟨by positivity, ?_⟩⟩
  norm_num
  nlinarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.mvt_sin (5) ((by norm_num)) := by
  refine ⟨by rw [Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.sin_zero]; simp, ⟨by positivity, ?_⟩⟩
  norm_num
  nlinarith [Real.pi_pos]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.mvt_sin
-- test: ra-mvt-sin-test.sage

/-! ## exm-mvt-constant -/

/-- Functions with the same derivative differ by a constant. -/
theorem same_deriv_const (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
    (h : ∀ x, deriv f x = deriv g x) : ∃ C, ∀ x, f x = g x + C := by
  refine ⟨f 0 - g 0, fun x => ?_⟩
  have := is_const_of_deriv_eq_zero (hf.sub hg) (fun y => by
    rw [deriv_sub (hf y) (hg y), h y, sub_self]) x 0
  simp only [Pi.sub_apply] at this
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.same_deriv_const
-- test: ra-same-deriv-const-test.sage

/-! ## exm-mvt-lipschitz -/

/-- `|sin x - sin y| ≤ |x - y|` and `|e^x - e^y| ≤ e^max(x,y) |x - y|`. -/
theorem sin_exp_lipschitz (x y : ℝ) :
    |Real.sin x - Real.sin y| ≤ |x - y| ∧ |Real.exp x - Real.exp y| ≤ Real.exp (max x y) * |x - y| := by
  refine ⟨Real.abs_sin_sub_sin_le x y, ?_⟩
  have key : ∀ u v : ℝ, v ≤ u → |Real.exp u - Real.exp v| ≤ Real.exp (max u v) * |u - v| := by
    intro u v huv
    rw [max_eq_left huv, abs_of_nonneg (sub_nonneg.2 (Real.exp_le_exp.2 huv)), abs_of_nonneg (sub_nonneg.2 huv)]
    have h1 := Real.add_one_le_exp (v - u)
    have h2 : Real.exp v = Real.exp u * Real.exp (v - u) := by rw [← Real.exp_add]; ring_nf
    rw [h2]
    have h3 : Real.exp u * (1 - Real.exp (v - u)) ≤ Real.exp u * (u - v) :=
      mul_le_mul_of_nonneg_left (by linarith) (Real.exp_pos u).le
    nlinarith [h3]
  rcases le_total y x with h | h
  · exact key x y h
  · have := key y x h
    rwa [abs_sub_comm, max_comm, abs_sub_comm y x] at this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.sin_exp_lipschitz
-- test: ra-sin-exp-lipschitz-test.sage

/-! ## exr-mean-value-sage -/

/-- For `e^x` on `[0, b]` the mean value point is `c = log((e^b - 1)/b) ∈ (0, b)`; at `b = 1`, `c = log(e - 1) ≈
0.541`. -/
theorem mvt_exp (b : ℝ) (hb : 0 < b) :
    Real.exp (Real.log ((Real.exp b - 1) / b)) = (Real.exp b - Real.exp 0) / (b - 0) ∧
      0 < Real.log ((Real.exp b - 1) / b) ∧ Real.log ((Real.exp b - 1) / b) < b := by
  have h1 : b + 1 < Real.exp b := by have := Real.add_one_lt_exp hb.ne'; linarith
  have hq : 1 < (Real.exp b - 1) / b := by rw [one_lt_div hb]; linarith
  refine ⟨by rw [Real.exp_log (by linarith), Real.exp_zero, sub_zero], Real.log_pos hq, ?_⟩
  rw [Real.log_lt_iff_lt_exp (by linarith), div_lt_iff₀ hb]
  -- e^b - 1 < b e^b, i.e. 1 - b < e^(-b)
  have h2 : -b + 1 < Real.exp (-b) := Real.add_one_lt_exp (by linarith)
  have h3 : Real.exp (-b) * Real.exp b = 1 := by rw [← Real.exp_add]; simp
  nlinarith [Real.exp_pos b]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.mvt_exp
-- test: ra-mvt-exp-test.sage

/-! ## exr-mvt-monotone -/

/-- `e^x > 1 + x` for `x ≠ 0`, and `log(1 + 1/n) ≤ 1/n`. -/
theorem exp_gt_one_add (x : ℝ) (hx : x ≠ 0) : 1 + x < Real.exp x := by
  have := Real.add_one_lt_exp hx; linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.exp_gt_one_add
-- test: ra-exp-gt-one-add-test.sage

theorem log_one_add_inv_le (n : ℕ) (hn : 1 ≤ n) : Real.log (1 + 1 / (n : ℝ)) ≤ 1 / n := by
  have := Real.log_le_sub_one_of_pos (show 0 < 1 + 1 / (n : ℝ) by positivity)
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.log_one_add_inv_le
-- test: ra-log-one-add-inv-le-test.sage

/-! ## exr-mvt-rolle -/

/-- `x³ + x - y` has exactly one real root for every `y` (at `y = 1`: `f(0) = -1 < 0 < 1 = f(1)`). -/
theorem cubic_unique_root (y : ℝ) : ∃! x : ℝ, x ^ 3 + x - y = 0 := by
  have hmono : StrictMono (fun x : ℝ => x ^ 3 + x) := by
    intro a b hab
    simp only
    nlinarith [sq_nonneg (a + b), sq_nonneg a, sq_nonneg b, mul_pos (sub_pos.2 hab) (sub_pos.2 hab)]
  have hc : Continuous (fun x : ℝ => x ^ 3 + x) := by fun_prop
  -- surjective: IVT on [-(|y| + 1), |y| + 1]
  obtain ⟨x, -, hx⟩ := intermediate_value_Icc (show -(|y| + 1) ≤ |y| + 1 by linarith [abs_nonneg y]) hc.continuousOn
    (show y ∈ Icc ((-(|y| + 1)) ^ 3 + -(|y| + 1)) ((|y| + 1) ^ 3 + (|y| + 1)) by
      constructor <;> nlinarith [abs_nonneg y, le_abs_self y, neg_abs_le y, sq_nonneg (|y| + 1)])
  refine ⟨x, by simp only at hx; linarith, fun z hz => ?_⟩
  apply hmono.injective
  simp only at hx ⊢
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.cubic_unique_root
-- test: ra-cubic-unique-root-test.sage

/-- A non-zero polynomial of degree `n` has at most `n` real roots. -/
theorem poly_roots_le (P : Polynomial ℝ) (hP : P ≠ 0) : P.roots.toFinset.card ≤ P.natDegree :=
  (Multiset.toFinset_card_le _).trans (Polynomial.card_roots' P)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.poly_roots_le
-- test: ra-poly-roots-le-test.sage

/-! ## thm-taylor -/

/-- Taylor's theorem with the Lagrange remainder. -/
theorem taylor_lagrange (f : ℝ → ℝ) (x₀ x : ℝ) (n : ℕ) (hx : x₀ ≠ x) (hf : ContDiffOn ℝ (n + 1) f (uIcc x₀ x)) :
    ∃ c ∈ uIoo x₀ x, f x - taylorWithinEval f n (uIcc x₀ x) x₀ x =
      iteratedDeriv (n + 1) f c * (x - x₀) ^ (n + 1) / (n + 1)! :=
  taylor_mean_remainder_lagrange_iteratedDeriv hx hf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.taylor_lagrange
-- test: ra-taylor-lagrange-test.sage

/-! ## exm-taylor-sin -/

/-- The Taylor polynomial `T₇(x) = x - x³/6 + x⁵/120 - x⁷/5040` of `sin` at `0` has remainder at most `|x|⁹/9!`. -/
theorem sin_taylor7 (x : ℝ) :
    |Real.sin x - (x - x ^ 3 / 6 + x ^ 5 / 120 - x ^ 7 / 5040)| ≤ |x| ^ 9 / 9 ! := by
  rcases eq_or_ne x 0 with rfl | hx
  · simp
  obtain ⟨c, -, hc⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (f := Real.sin) (n := 8) hx.symm
    Real.contDiff_sin.contDiffOn
  have hd : ∀ k : ℕ, iteratedDeriv k Real.sin 0 = if Even k then 0 else (-1) ^ (k / 2) := by
    intro k
    rcases Nat.even_or_odd' k with ⟨m, rfl | rfl⟩
    · rw [Real.iteratedDeriv_even_sin, if_pos (even_two_mul m)]; simp
    · rw [Real.iteratedDeriv_odd_sin, if_neg (Nat.not_even_iff_odd.2 (odd_two_mul_add_one m))]
      have : (2 * m + 1) / 2 = m := by omega
      rw [this]; simp
  have hs : ∀ k, iteratedDerivWithin k Real.sin (uIcc 0 x) 0 = iteratedDeriv k Real.sin 0 := fun k =>
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hx.symm) Real.contDiff_sin.contDiffAt left_mem_uIcc
  have hT : taylorWithinEval Real.sin 8 (uIcc 0 x) 0 x = x - x ^ 3 / 6 + x ^ 5 / 120 - x ^ 7 / 5040 := by
    simp only [taylorWithinEval_succ, taylor_within_zero_eval, hs, hd, sub_zero, smul_eq_mul]
    norm_num [Nat.factorial]
    ring
  rw [hT] at hc
  rw [hc, sub_zero, abs_div, abs_mul, abs_pow, abs_of_pos (by positivity : (0 : ℝ) < ((8 + 1)! : ℝ))]
  have hb := Real.abs_iteratedDeriv_sin_le_one (8 + 1) c
  calc |iteratedDeriv (8 + 1) Real.sin c| * |x| ^ (8 + 1) / ((8 + 1)! : ℝ) ≤ 1 * |x| ^ (8 + 1) / ((8 + 1)! : ℝ) := by
        gcongr
    _ = |x| ^ 9 / 9 ! := by norm_num

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Continuity.sin_taylor7
-- test: ra-sin-taylor7-test.sage

/-- On `[-π, π]` the remainder of `T₇` is below `0.083` (`π⁹/9! = 0.0821…`), on `[-1, 1]` below `3 · 10⁻⁶`. -/
theorem sin_taylor7_on (x : ℝ) (hx : |x| ≤ π) :
    |Real.sin x - (x - x ^ 3 / 6 + x ^ 5 / 120 - x ^ 7 / 5040)| < 0.083 ∧
      (|x| ≤ 1 → |Real.sin x - (x - x ^ 3 / 6 + x ^ 5 / 120 - x ^ 7 / 5040)| < 3e-6) := by
  have h := sin_taylor7 x
  have hpi := Real.pi_lt_d4
  refine ⟨lt_of_le_of_lt h ?_, fun h1 => lt_of_le_of_lt h ?_⟩
  · have : |x| ^ 9 ≤ (3.1416 : ℝ) ^ 9 := pow_le_pow_left₀ (abs_nonneg x) (by linarith) 9
    norm_num [Nat.factorial] at this ⊢
    linarith
  · have : |x| ^ 9 ≤ 1 := pow_le_one₀ (abs_nonneg x) h1
    norm_num [Nat.factorial]
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.sin_taylor7_on
-- test: ra-sin-taylor7-on-test.sage

/-! ## exm-taylor-exp -/

/-- `T₄(1) = 1 + 1 + 1/2 + 1/6 + 1/24 = 65/24 = 2.7083…`, and for `0 ≤ x ≤ 1` the remainder `e^x - T₄(x)` is between
`0` and `e^x x⁵/5!` (at `x = 1`: at most `e/120 = 0.0227`, actual `0.0099`). -/
theorem exp_taylor4 (x : ℝ) (hx : 0 ≤ x) :
    (1 : ℝ) + 1 + 1 / 2 + 1 / 6 + 1 / 24 = 65 / 24 ∧
      0 ≤ Real.exp x - (1 + x + x ^ 2 / 2 + x ^ 3 / 6 + x ^ 4 / 24) ∧
      Real.exp x - (1 + x + x ^ 2 / 2 + x ^ 3 / 6 + x ^ 4 / 24) ≤ Real.exp x * x ^ 5 / 120 := by
  refine ⟨by norm_num, ?_⟩
  rcases hx.lt_or_eq with hx | rfl
  · obtain ⟨c, hcm, hc⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (f := Real.exp) (n := 4) hx.ne
      Real.contDiff_exp.contDiffOn
    have hi : ∀ k (y : ℝ), iteratedDeriv k Real.exp y = Real.exp y := fun k y => by
      rw [iteratedDeriv_eq_iterate, Real.iter_deriv_exp]
    have hs : ∀ k, iteratedDerivWithin k Real.exp (uIcc 0 x) 0 = 1 := fun k => by
      rw [iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hx.ne) Real.contDiff_exp.contDiffAt left_mem_uIcc,
        hi, Real.exp_zero]
    have hT : taylorWithinEval Real.exp 4 (uIcc 0 x) 0 x = 1 + x + x ^ 2 / 2 + x ^ 3 / 6 + x ^ 4 / 24 := by
      simp only [taylorWithinEval_succ, taylor_within_zero_eval, hs, Real.exp_zero, sub_zero, smul_eq_mul]
      norm_num [Nat.factorial]
      ring
    rw [hT] at hc
    rw [hc, hi, sub_zero]
    rw [uIoo_of_le hx.le] at hcm
    have hle : Real.exp c ≤ Real.exp x := Real.exp_le_exp.2 hcm.2.le
    constructor
    · positivity
    · norm_num [Nat.factorial]
      have : 0 ≤ x ^ 5 := by positivity
      nlinarith
  · simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.exp_taylor4
-- test: ra-exp-taylor4-test.sage

/-! ## exm-taylor-non-analytic -/

/-- The flat function `f(x) = e^(-1/x²)` (`f(0) = 0`): it is smooth, vanishes only at `0`, and every derivative
vanishes at `0`, so its Taylor series is `0` and misses the function. -/
noncomputable def flat (x : ℝ) : ℝ := expNegInvGlue (x ^ 2)

theorem flat_eq (x : ℝ) (hx : x ≠ 0) : flat x = Real.exp (-1 / x ^ 2) := by
  unfold flat
  rw [expNegInvGlue, if_neg (not_le.2 (by positivity))]
  ring_nf

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_eq (1) ((by norm_num)) := by
  unfold KaluzaKlein.RealAnalysis.Continuity.flat
  rw [expNegInvGlue, if_neg (by norm_num)]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_eq (-2) ((by norm_num)) := by
  unfold KaluzaKlein.RealAnalysis.Continuity.flat
  rw [expNegInvGlue, if_neg (by norm_num)]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_eq (1 / 3) ((by norm_num)) := by
  unfold KaluzaKlein.RealAnalysis.Continuity.flat
  rw [expNegInvGlue, if_neg (by norm_num)]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.flat_eq
-- test: ra-flat-eq-test.sage

theorem flat_smooth_positive (x : ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) flat ∧ (x ≠ 0 → 0 < flat x) ∧ flat 0 = 0 := by
  refine ⟨expNegInvGlue.contDiff.comp (contDiff_id.pow 2), fun hx => ?_, ?_⟩
  · unfold flat; exact expNegInvGlue.pos_of_pos (by positivity)
  · unfold flat; simp [expNegInvGlue.zero]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_smooth_positive (0) := by
  refine ⟨expNegInvGlue.contDiff.comp (contDiff_id.pow 2), fun hx => ?_, ?_⟩
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; exact expNegInvGlue.pos_of_pos (by positivity)
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; simp [expNegInvGlue.zero]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_smooth_positive (1) := by
  refine ⟨expNegInvGlue.contDiff.comp (contDiff_id.pow 2), fun hx => ?_, ?_⟩
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; exact expNegInvGlue.pos_of_pos (by positivity)
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; simp [expNegInvGlue.zero]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.flat_smooth_positive (-1 / 2) := by
  refine ⟨expNegInvGlue.contDiff.comp (contDiff_id.pow 2), fun hx => ?_, ?_⟩
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; exact expNegInvGlue.pos_of_pos (by positivity)
  · unfold KaluzaKlein.RealAnalysis.Continuity.flat; simp [expNegInvGlue.zero]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Continuity.flat_smooth_positive
-- test: ra-flat-smooth-positive-test.sage

/-- Every derivative of the flat function vanishes at `0` (stated, tested by Sage's limits of the derivatives, not
proved here). -/
def FlatDerivativesVanish : Prop := ∀ k : ℕ, iteratedDeriv k flat 0 = 0

attribute [veriphy_tested oracle_test 0 6] KaluzaKlein.RealAnalysis.Continuity.FlatDerivativesVanish
-- test: ra-flatderivativesvanish-test.sage

/-! ## exr-taylor-remainder-bound -/

/-- On `[-π, π]` the remainder bound `π^(2m+1)/(2m+1)!` after `T_(2m-1)` is below `10⁻⁶` exactly for `m ≥ 8`
(`π¹⁷/17! = 8.0 · 10⁻⁷`). -/
theorem sin_terms_pi (m : ℕ) : π ^ (2 * m + 1) / ((2 * m + 1)! : ℝ) < 1 / 10 ^ 6 ↔ 8 ≤ m := by
  set t : ℕ → ℝ := fun m => π ^ (2 * m + 1) / ((2 * m + 1)! : ℝ) with ht
  have hrec : ∀ m : ℕ, t (m + 1) = t m * (π ^ 2 / ((2 * m + 2) * (2 * m + 3))) := by
    intro m
    simp only [ht]
    rw [show 2 * (m + 1) + 1 = 2 * m + 1 + 1 + 1 by ring, Nat.factorial_succ, Nat.factorial_succ, pow_succ, pow_succ]
    push_cast
    field_simp
    ring
  have hpi2 : π ^ 2 < 10 := by nlinarith [Real.pi_lt_d2, Real.pi_pos]
  have htpos : ∀ m, 0 < t m := fun m => by simp only [ht]; positivity
  have hstep : ∀ m, 1 ≤ m → t (m + 1) ≤ t m := by
    intro m hm
    rw [hrec]
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
    have hq : π ^ 2 / ((2 * m + 2) * (2 * m + 3)) ≤ 1 := by
      rw [div_le_one (by positivity)]; nlinarith
    have := htpos m
    nlinarith
  have hmono : ∀ a b, 1 ≤ a → a ≤ b → t b ≤ t a := by
    intro a b ha hab
    induction b, hab using Nat.le_induction with
    | base => exact le_rfl
    | succ b hab ih => exact (hstep b (le_trans ha hab)).trans ih
  have h7 : 1 / 10 ^ 6 < t 7 := by
    simp only [ht]
    have h3 : (3 : ℝ) ^ 15 < π ^ 15 := pow_lt_pow_left₀ Real.pi_gt_three (by norm_num) (by norm_num)
    rw [lt_div_iff₀ (by positivity)]
    norm_num [Nat.factorial] at h3 ⊢
    linarith
  have h8 : t 8 < 1 / 10 ^ 6 := by
    simp only [ht]
    have h3 : π ^ 17 < (3.15 : ℝ) ^ 17 := pow_lt_pow_left₀ Real.pi_lt_d2 Real.pi_pos.le (by norm_num)
    rw [div_lt_iff₀ (by positivity)]
    norm_num [Nat.factorial] at h3 ⊢
    linarith
  have ht0 : t 0 = π := by simp only [ht]; norm_num
  show t m < 1 / 10 ^ 6 ↔ 8 ≤ m
  constructor
  · intro h
    by_contra hm
    push_neg at hm
    rcases Nat.eq_zero_or_pos m with rfl | hm1
    · rw [ht0] at h; linarith [Real.pi_gt_three]
    · have := hmono m 7 hm1 (by omega)
      linarith
  · intro hm
    exact lt_of_le_of_lt (hmono 8 m (by norm_num) hm) h8

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.sin_terms_pi
-- test: ra-sin-terms-pi-test.sage

/-- On `[-1, 1]` the bound `1/(2m+1)!` is below `10⁻¹⁰` exactly for `m ≥ 7` (`1/15! = 7.6 · 10⁻¹³`,
`1/13! = 1.6 · 10⁻¹⁰`). -/
theorem sin_terms_one (m : ℕ) : 1 / ((2 * m + 1)! : ℝ) < 1 / 10 ^ 10 ↔ 7 ≤ m := by
  have hf : (0 : ℝ) < (2 * m + 1)! := by positivity
  rw [div_lt_div_iff₀ hf (by positivity), one_mul, one_mul]
  constructor
  · intro h
    by_contra hm
    push_neg at hm
    have : (2 * m + 1)! ≤ 13 ! := Nat.factorial_le (by omega)
    have h13 : (13 ! : ℝ) < 10 ^ 10 := by norm_num [Nat.factorial]
    have : ((2 * m + 1)! : ℝ) ≤ 13 ! := by exact_mod_cast this
    linarith
  · intro hm
    have : 15 ! ≤ (2 * m + 1)! := Nat.factorial_le (by omega)
    have h15 : (10 : ℝ) ^ 10 < 15 ! := by norm_num [Nat.factorial]
    have : (15 ! : ℝ) ≤ (2 * m + 1)! := by exact_mod_cast this
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_terms_one (0) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_terms_one (6) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_terms_one (7) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_terms_one (10) := by
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.sin_terms_one
-- test: ra-sin-terms-one-test.sage

/-! ## exr-taylor-second-derivative -/

/-- The second-derivative test: `f'(a) = 0` and `f''(a) > 0` give a strict local minimum. -/
theorem second_derivative_test (f : ℝ → ℝ) (a : ℝ) (hf : ContDiff ℝ 2 f) (h1 : deriv f a = 0)
    (h2 : 0 < deriv (deriv f) a) : IsLocalMin f a :=
  isLocalMin_of_deriv_deriv_pos h2 h1 hf.continuous.continuousAt

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.second_derivative_test
-- test: ra-second-derivative-test-test.sage

/-- The double well `V(φ) = λ(φ² - v²)²/4` (`λ > 0`, `v ≠ 0`): `V'(φ) = λφ(φ² - v²)`, `V''(φ) = λ(3φ² - v²)`, which is
`-λv² < 0` at `0` (a local maximum) and `2λv² > 0` at `±v` (minima); at `λ = v = 1`: `V'' = -1` and `2`. -/
theorem double_well (lam v : ℝ) (hl : 0 < lam) (hv : v ≠ 0) :
    (∀ p, deriv (fun p => lam * (p ^ 2 - v ^ 2) ^ 2 / 4) p = lam * p * (p ^ 2 - v ^ 2)) ∧
      (∀ p, deriv (fun p => lam * p * (p ^ 2 - v ^ 2)) p = lam * (3 * p ^ 2 - v ^ 2)) ∧
      IsMinOn (fun p => lam * (p ^ 2 - v ^ 2) ^ 2 / 4) univ v ∧
      IsMinOn (fun p => lam * (p ^ 2 - v ^ 2) ^ 2 / 4) univ (-v) ∧
      IsLocalMax (fun p => lam * (p ^ 2 - v ^ 2) ^ 2 / 4) 0 := by
  refine ⟨fun p => ?_, fun p => ?_, fun p _ => ?_, fun p _ => ?_, ?_⟩
  · have h : HasDerivAt (fun p => lam * (p ^ 2 - v ^ 2) ^ 2 / 4) (lam * p * (p ^ 2 - v ^ 2)) p := by
      have h0 : HasDerivAt (fun p : ℝ => p ^ 2 - v ^ 2) (2 * p) p := by
        simpa using (hasDerivAt_pow 2 p).sub_const (v ^ 2)
      convert ((h0.pow 2).const_mul lam).div_const 4 using 1
      push_cast
      ring
    exact h.deriv
  · have h : HasDerivAt (fun p => lam * p * (p ^ 2 - v ^ 2)) (lam * (3 * p ^ 2 - v ^ 2)) p := by
      have h0 : HasDerivAt (fun p : ℝ => p ^ 2 - v ^ 2) (2 * p) p := by
        simpa using (hasDerivAt_pow 2 p).sub_const (v ^ 2)
      have h1 : HasDerivAt (fun p : ℝ => lam * p) lam p := by
        simpa using (hasDerivAt_id p).const_mul lam
      convert h1.mul h0 using 1
      ring
    exact h.deriv
  · show lam * (v ^ 2 - v ^ 2) ^ 2 / 4 ≤ lam * (p ^ 2 - v ^ 2) ^ 2 / 4
    nlinarith [mul_nonneg hl.le (sq_nonneg (p ^ 2 - v ^ 2))]
  · show lam * ((-v) ^ 2 - v ^ 2) ^ 2 / 4 ≤ lam * (p ^ 2 - v ^ 2) ^ 2 / 4
    nlinarith [mul_nonneg hl.le (sq_nonneg (p ^ 2 - v ^ 2))]
  · filter_upwards [Metric.ball_mem_nhds (0 : ℝ) (abs_pos.2 hv)] with y hy
    rw [Metric.mem_ball, Real.dist_eq, sub_zero] at hy
    have hy2 : y ^ 2 < v ^ 2 := sq_lt_sq.2 hy
    have key : (y ^ 2 - v ^ 2) ^ 2 ≤ ((0 : ℝ) ^ 2 - v ^ 2) ^ 2 := by nlinarith [sq_nonneg y]
    show lam * (y ^ 2 - v ^ 2) ^ 2 / 4 ≤ lam * ((0 : ℝ) ^ 2 - v ^ 2) ^ 2 / 4
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left key hl.le) (by norm_num)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Continuity.double_well
-- test: ra-double-well-test.sage

/-! ## exm-derivative-values-sage -/

/-- The coefficients of `taylor(sin(x), x, 0, 7)`: `(-1)^k/(2k+1)! = 1, -1/6, 1/120, -1/5040`. -/
theorem sin_taylor_coefficients (k : Fin 4) :
    (-1 : ℝ) ^ (k : ℕ) / ((2 * (k : ℕ) + 1)! : ℝ) = ![1, -1 / 6, 1 / 120, -1 / 5040] k := by
  fin_cases k <;> norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients (0) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients (1) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients (2) := by
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients (3) := by
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Continuity.sin_taylor_coefficients
-- test: ra-sin-taylor-coefficients-test.sage

/-! ## exm-derivative-values-two-dims -/

/-- `∂ₓ(x² y + sin y) = 2xy`, a one-variable derivative in `x` with `y` fixed. -/
theorem partial_x_poly (x y : ℝ) : deriv (fun x => x ^ 2 * y + Real.sin y) x = 2 * x * y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2 * y + sin y) ((2 * x * y : ℝ)) (x) :=
        by
        convert
          ((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (y : ℝ)).add (hasDerivAt_const (x : ℝ) (Real.sin y : ℝ))) using 1
        all_goals
          first
          | rfl
          | ring1
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (funext y; simp; ring_nf; done)
          | (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-partial-x-poly.sage

/-- `d/dt sin(t²) = cos(t²) · 2t`. -/
theorem chain_sin_sq (t : ℝ) : deriv (fun t => Real.sin (t ^ 2)) t = Real.cos (t ^ 2) * (2 * t) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => sin (t ^ 2)) ((2 * t * (Real.cos (t ^ 2)) : ℝ)) (t) :=
        by
        convert (((hasDerivAt_id' (t : ℝ)).pow 2).sin) using 1
        all_goals (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-chain-sin-sq.sage

/-! ## exr-derivative-higher -/

/-- `(x² e^x)^(n) = (x² + 2nx + n(n - 1)) e^x`. -/
theorem iteratedDeriv_sq_exp (n : ℕ) (x : ℝ) :
    iteratedDeriv n (fun x => x ^ 2 * Real.exp x) x = (x ^ 2 + 2 * n * x + n * (n - 1)) * Real.exp x := by
  induction n generalizing x with
  | zero => simp
  | succ n ih =>
    rw [iteratedDeriv_succ]
    have hf : iteratedDeriv n (fun x => x ^ 2 * Real.exp x) =
        fun x => (x ^ 2 + 2 * (n : ℝ) * x + n * (n - 1)) * Real.exp x := funext ih
    rw [hf]
    have h := ((((hasDerivAt_pow 2 x).add ((hasDerivAt_id x).const_mul (2 * (n : ℝ)))).add_const
      ((n : ℝ) * (n - 1))).mul (Real.hasDerivAt_exp x))
    have h' : HasDerivAt (fun x => (x ^ 2 + 2 * (n : ℝ) * x + n * (n - 1)) * Real.exp x) _ x := h
    rw [h'.deriv]
    push_cast
    simp only [id, mul_one, Pi.add_apply]
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.iteratedDeriv_sq_exp
-- test: ra-iteratedderiv-sq-exp-test.sage

/-! ## exr-taylor-cos-even -/

/-- An even smooth function has vanishing odd derivatives at `0` (and an odd one vanishing even derivatives). -/
theorem even_odd_derivatives (f : ℝ → ℝ) (k : ℕ) :
    ((∀ x, f (-x) = f x) → iteratedDeriv (2 * k + 1) f 0 = 0) ∧
      ((∀ x, f (-x) = -f x) → iteratedDeriv (2 * k) f 0 = 0) := by
  constructor
  · intro he
    have h := iteratedDeriv_comp_neg (2 * k + 1) f 0
    have hf : (fun x => f (-x)) = f := funext he
    rw [hf, neg_zero, pow_succ, pow_mul, neg_one_sq, one_pow, one_mul, neg_one_smul] at h
    linarith
  · intro ho
    have h := iteratedDeriv_comp_neg (2 * k) f 0
    have hf : (fun x => f (-x)) = -f := funext fun x => by rw [Pi.neg_apply, ho]
    rw [hf, iteratedDeriv_neg, neg_zero, pow_mul, neg_one_sq, one_pow, one_smul] at h
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.even_odd_derivatives
-- test: ra-even-odd-derivatives-test.sage

/-- `cos^(k)(0) = 1, 0, -1, 0, …` and `sin^(k)(0) = 0, 1, 0, -1, …`. -/
theorem cos_sin_derivatives_zero (k : ℕ) :
    iteratedDeriv (2 * k) Real.cos 0 = (-1) ^ k ∧ iteratedDeriv (2 * k + 1) Real.cos 0 = 0 ∧
      iteratedDeriv (2 * k) Real.sin 0 = 0 ∧ iteratedDeriv (2 * k + 1) Real.sin 0 = (-1) ^ k := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [Real.iteratedDeriv_even_cos]; simp
  · rw [Real.iteratedDeriv_odd_cos]; simp
  · rw [Real.iteratedDeriv_even_sin]; simp
  · rw [Real.iteratedDeriv_odd_sin]; simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.cos_sin_derivatives_zero
-- test: ra-cos-sin-derivatives-zero-test.sage

/-! ## exr-taylor-pi-estimate -/

/-- Machin's formula `π/4 = 4 arctan(1/5) - arctan(1/239)`. -/
theorem machin : 4 * Real.arctan 5⁻¹ - Real.arctan 239⁻¹ = π / 4 := by
  veriphy_certified (rw [Real.four_mul_arctan_inv_5_sub_arctan_inv_239])
  -- certificate: ra-machin.sage

/-- The term `(-1)^j/(2j + 1)` of Leibniz's series. -/
noncomputable def leibnizTerm (j : ℕ) : ℝ := (-1 : ℝ) ^ j / (2 * (j : ℝ) + 1)

/-- The arctangent series at `x = 1` is Leibniz's series for `π/4`; after `k` terms the error is at most
`1/(2k + 1)`, so three decimals need about `500` terms. -/
theorem leibniz_error (k : ℕ) :
    |∑ j ∈ Finset.range k, leibnizTerm j - π / 4| ≤ 1 / (2 * (k : ℝ) + 1) := by
  simp only [leibnizTerm]
  set f : ℕ → ℝ := fun j => 1 / (2 * (j : ℝ) + 1) with hf
  have hanti : Antitone f := fun a b hab =>
    one_div_le_one_div_of_le (by positivity) (by linarith [(Nat.cast_le (α := ℝ)).2 hab])
  have hS : ∀ n, ∑ j ∈ Finset.range n, (-1 : ℝ) ^ j / (2 * j + 1) = ∑ i ∈ Finset.range n, (-1) ^ i * f i := fun n =>
    Finset.sum_congr rfl (fun j _ => by rw [hf, mul_one_div])
  have hl : Tendsto (fun n => ∑ i ∈ Finset.range n, (-1 : ℝ) ^ i * f i) atTop (𝓝 (π / 4)) :=
    Real.tendsto_sum_pi_div_four.congr (fun n => hS n)
  have lo := hanti.alternating_series_le_tendsto hl
  have hi := hanti.tendsto_le_alternating_series hl
  rw [hS]
  rcases Nat.even_or_odd k with ⟨m, rfl⟩ | ⟨m, rfl⟩
  · rw [← two_mul]
    have h1 := lo m
    have h2 := hi m
    rw [Finset.sum_range_succ] at h2
    have e1 : (-1 : ℝ) ^ (2 * m) = 1 := by rw [pow_mul]; norm_num
    rw [e1, one_mul] at h2
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]
    simp only [hf] at h2 ⊢
    push_cast at h2 ⊢
    linarith
  · have h1 := hi m
    have h2 := lo (m + 1)
    rw [show 2 * (m + 1) = 2 * m + 1 + 1 by ring, Finset.sum_range_succ] at h2
    have e1 : (-1 : ℝ) ^ (2 * m) = 1 := by rw [pow_mul]; norm_num
    have e2 : (-1 : ℝ) ^ (2 * m + 1) = -1 := by rw [pow_succ, e1]; norm_num
    rw [e2] at h2
    rw [abs_of_nonneg (by linarith)]
    simp only [hf] at h2 ⊢
    push_cast at h2 ⊢
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Continuity.leibniz_error
-- test: ra-leibniz-error-test.sage

end KaluzaKlein.RealAnalysis.Continuity

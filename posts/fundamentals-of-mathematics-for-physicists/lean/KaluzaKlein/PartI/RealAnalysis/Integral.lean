import Mathlib
import Veriphy.Kit.Levels

/-!
# Real analysis, part "The Riemann integral"

The claims of the section "The Riemann integral" of `tutorial/part-i-03-real-analysis.qmd`: lower and upper sums
(uniform partitions), integrability of monotone and piecewise continuous functions, the Dirichlet function,
antiderivatives with integration by parts and substitution, the fundamental theorem of calculus, improper integrals
with the `p`-integrals, the exponential and Gaussian integrals, and the values of the running integrals.

The integral is mathlib's interval integral (the Lebesgue integral on an interval, which agrees with the Riemann
integral on Riemann integrable functions). Definite integrals with an elementary antiderivative are certified by the
`definite_integral` recipe; antiderivatives by the `derivative` recipe; finite sums by `sum_induction`.
-/

namespace KaluzaKlein.RealAnalysis.Integral

open Filter Topology Real Set MeasureTheory intervalIntegral
open scoped Nat

/-! ## def-riemann-integral -/

/-- Continuous functions are integrable on `[a, b]`. -/
theorem continuous_integrable (f : ℝ → ℝ) (a b : ℝ) (hf : ContinuousOn f (uIcc a b)) :
    IntervalIntegrable f volume a b :=
  hf.intervalIntegrable

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.continuous_integrable
-- test: ra-continuous-integrable-test.sage

/-- The integral is linear, additive over intervals, and changes sign with the orientation. -/
theorem integral_rules (f g : ℝ → ℝ) (a b c k : ℝ) (hf : Continuous f) (hg : Continuous g) :
    (∫ x in a..b, (f x + k * g x)) = (∫ x in a..b, f x) + k * ∫ x in a..b, g x ∧
      (∫ x in a..b, f x) + (∫ x in b..c, f x) = ∫ x in a..c, f x ∧
      (∫ x in a..b, f x) = -∫ x in b..a, f x := by
  refine ⟨?_, integral_add_adjacent_intervals (hf.intervalIntegrable _ _) (hf.intervalIntegrable _ _),
    integral_symm b a⟩
  rw [integral_add (hf.intervalIntegrable _ _) ((hg.const_mul k).intervalIntegrable _ _), intervalIntegral.integral_const_mul]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.integral_rules
-- test: ra-integral-rules-test.sage

/-- The integral is monotone: `f ≤ g` on `[a, b]` gives `∫ f ≤ ∫ g`. -/
theorem integral_monotone (f g : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : Continuous f) (hg : Continuous g)
    (h : ∀ x ∈ Icc a b, f x ≤ g x) : (∫ x in a..b, f x) ≤ ∫ x in a..b, g x :=
  integral_mono_on hab (hf.intervalIntegrable _ _) (hg.intervalIntegrable _ _) h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.integral_monotone
-- test: ra-integral-monotone-test.sage

/-! ## exm-integral-x-squared -/

/-- The lower and upper sums of `x²` on `[0, 1]` with `n` equal subintervals are `(n-1)n(2n-1)/(6n³)` and
`n(n+1)(2n+1)/(6n³)`, with difference `1/n`. -/
theorem x_sq_sums (n : ℕ) (hn : 1 ≤ n) :
    ∑ k ∈ Finset.range n, ((k : ℝ) / n) ^ 2 * (1 / n) = ((n : ℝ) - 1) * n * (2 * n - 1) / (6 * (n : ℝ) ^ 3) ∧
      ∑ k ∈ Finset.Icc 1 n, ((k : ℝ) / n) ^ 2 * (1 / n) = (n : ℝ) * (n + 1) * (2 * n + 1) / (6 * (n : ℝ) ^ 3) ∧
      ∑ k ∈ Finset.Icc 1 n, ((k : ℝ) / n) ^ 2 * (1 / n) - ∑ k ∈ Finset.range n, ((k : ℝ) / n) ^ 2 * (1 / n) =
        1 / (n : ℝ) := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hlow : ∀ N : ℕ, ∑ k ∈ Finset.range N, (k : ℝ) ^ 2 = ((N : ℝ) - 1) * N * (2 * N - 1) / 6 := by
    intro N
    veriphy_certified
      (induction N with
        | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
        | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
    -- certificate: ra-sum-sq-range.sage
  have hup : ∀ N : ℕ, ∑ k ∈ Finset.Icc 1 N, (k : ℝ) ^ 2 = (N : ℝ) * (N + 1) * (2 * N + 1) / 6 := by
    intro N
    veriphy_certified
      (induction N with
        | zero => simp
        | succ N ih => rw [Finset.sum_Icc_succ_top (by omega), ih]; (try push_cast); ring1)
    -- certificate: ra-sum-sq-icc.sage
  have e1 : ∑ k ∈ Finset.range n, ((k : ℝ) / n) ^ 2 * (1 / n) = (∑ k ∈ Finset.range n, (k : ℝ) ^ 2) / n ^ 3 := by
    rw [Finset.sum_div]; refine Finset.sum_congr rfl (fun k _ => ?_); field_simp
  have e2 : ∑ k ∈ Finset.Icc 1 n, ((k : ℝ) / n) ^ 2 * (1 / n) = (∑ k ∈ Finset.Icc 1 n, (k : ℝ) ^ 2) / n ^ 3 := by
    rw [Finset.sum_div]; refine Finset.sum_congr rfl (fun k _ => ?_); field_simp
  refine ⟨?_, ?_, ?_⟩
  · rw [e1, hlow]; field_simp <;> ring
  · rw [e2, hup]; field_simp <;> ring
  · rw [e1, e2, hlow, hup]; field_simp <;> ring

/-- `∫₀¹ x² dx = 1/3`. -/
theorem integral_x_sq : ∫ x in (0 : ℝ)..1, x ^ 2 = 1 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) ((fun x => x ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) (f' :=
          fun x => x ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-integral-x-sq.sage

/-- Both Riemann sums of `x²` tend to `1/3`: the lower sum is `1/3 - 1/(2n) + 1/(6n²)`, the upper
`1/3 + 1/(2n) + 1/(6n²)`. -/
theorem x_sq_sums_rate (n : ℕ) (hn : 1 ≤ n) :
    ((n : ℝ) - 1) * n * (2 * n - 1) / (6 * (n : ℝ) ^ 3) = 1 / 3 - 1 / (2 * n) + 1 / (6 * (n : ℝ) ^ 2) ∧
      (n : ℝ) * (n + 1) * (2 * n + 1) / (6 * (n : ℝ) ^ 3) = 1 / 3 + 1 / (2 * n) + 1 / (6 * (n : ℝ) ^ 2) := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  constructor <;> field_simp <;> ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.x_sq_sums_rate (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.x_sq_sums_rate (4) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.x_sq_sums_rate (16) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Integral.x_sq_sums_rate
-- test: ra-x-sq-sums-rate-test.sage

/-- The limits of the two closed forms. -/
theorem x_sq_sums_limit :
    Tendsto (fun n : ℕ => ((n : ℝ) - 1) * n * (2 * n - 1) / (6 * (n : ℝ) ^ 3)) atTop (𝓝 (1 / 3)) ∧
      Tendsto (fun n : ℕ => (n : ℝ) * (n + 1) * (2 * n + 1) / (6 * (n : ℝ) ^ 3)) atTop (𝓝 (1 / 3)) := by
  have h1 : Tendsto (fun n : ℕ => (1 : ℝ) / n) atTop (𝓝 0) := tendsto_const_div_atTop_nhds_zero_nat 1
  constructor
  · have := ((((tendsto_const_nhds (x := (1 : ℝ))).sub h1).mul ((tendsto_const_nhds (x := (2 : ℝ))).sub h1)).div_const 6)
    rw [show (1 - 0) * (2 - 0) / 6 = (1 : ℝ) / 3 by norm_num] at this
    apply this.congr'
    filter_upwards [eventually_gt_atTop 0] with n hn
    have : (n : ℝ) ≠ 0 := by positivity
    field_simp
  · have := ((((tendsto_const_nhds (x := (1 : ℝ))).add h1).mul ((tendsto_const_nhds (x := (2 : ℝ))).add h1)).div_const 6)
    rw [show (1 + 0) * (2 + 0) / 6 = (1 : ℝ) / 3 by norm_num] at this
    apply this.congr'
    filter_upwards [eventually_gt_atTop 0] with n hn
    have : (n : ℝ) ≠ 0 := by positivity
    field_simp

/-! ## exm-integral-monotone -/

/-- For `f` increasing on `[a, b]`, the lower and upper sums with `n` equal subintervals bracket the integral and
differ by exactly `(f(b) - f(a))(b - a)/n`, which tends to `0`. -/
theorem monotone_riemann (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : MonotoneOn f (Icc a b)) (n : ℕ) (hn : 1 ≤ n) :
    ∑ k ∈ Finset.range n, f (a + k * ((b - a) / n)) * ((b - a) / n) ≤ ∫ x in a..b, f x ∧
      (∫ x in a..b, f x) ≤ ∑ k ∈ Finset.range n, f (a + (k + 1) * ((b - a) / n)) * ((b - a) / n) ∧
      ∑ k ∈ Finset.range n, f (a + (k + 1) * ((b - a) / n)) * ((b - a) / n) -
        ∑ k ∈ Finset.range n, f (a + k * ((b - a) / n)) * ((b - a) / n) = (f b - f a) * (b - a) / n := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  set h := (b - a) / n with hh
  have h0 : 0 ≤ h := div_nonneg (by linarith) hn0.le
  set x : ℕ → ℝ := fun k => a + k * h with hx
  have hxk : ∀ k : ℕ, x (k + 1) = x k + h := fun k => by simp only [hx]; push_cast; ring
  have hxk' : ∀ k : ℕ, a + ((k : ℝ) + 1) * h = x (k + 1) := fun k => by simp only [hx]; push_cast; ring
  have hxn : x n = b := by simp only [hx, hh]; field_simp; ring
  have hx0 : x 0 = a := by simp [hx]
  have hmem : ∀ k ≤ n, x k ∈ Icc a b := fun k hk => by
    have hk' : (k : ℝ) ≤ n := by exact_mod_cast hk
    have hkh : (k : ℝ) * h ≤ b - a := by
      calc (k : ℝ) * h ≤ n * h := mul_le_mul_of_nonneg_right hk' h0
        _ = b - a := by simp only [hh]; field_simp
    constructor
    · simp only [hx]; nlinarith
    · simp only [hx]; linarith
  have hsub : ∀ k < n, Icc (x k) (x (k + 1)) ⊆ Icc a b := fun k hk =>
    Icc_subset_Icc (hmem k hk.le).1 (hmem (k + 1) hk).2
  have hle : ∀ k, x k ≤ x (k + 1) := fun k => by rw [hxk]; linarith
  have hint : ∀ k < n, IntervalIntegrable f volume (x k) (x (k + 1)) := fun k hk => by
    apply MonotoneOn.intervalIntegrable
    rw [uIcc_of_le (hle k)]
    exact hf.mono (hsub k hk)
  have hsum := sum_integral_adjacent_intervals hint
  rw [hx0, hxn] at hsum
  have hlo : ∀ k < n, f (x k) * h ≤ ∫ t in x k..x (k + 1), f t := fun k hk => by
    have := integral_mono_on (hle k) intervalIntegrable_const (hint k hk)
      (fun t ht => hf (hmem k hk.le) (hsub k hk ht) ht.1)
    rw [intervalIntegral.integral_const, smul_eq_mul, hxk, add_sub_cancel_left] at this
    rw [hxk]
    linarith [mul_comm h (f (x k))]
  have hhi : ∀ k < n, (∫ t in x k..x (k + 1), f t) ≤ f (x (k + 1)) * h := fun k hk => by
    have := integral_mono_on (hle k) (hint k hk) intervalIntegrable_const
      (fun t ht => hf (hsub k hk ht) (hmem (k + 1) hk) ht.2)
    rw [intervalIntegral.integral_const, smul_eq_mul, hxk, add_sub_cancel_left] at this
    rw [hxk]
    linarith [mul_comm h (f (x k + h))]
  have hL : ∑ k ∈ Finset.range n, f (a + k * h) * h = ∑ k ∈ Finset.range n, f (x k) * h := rfl
  have hU : ∑ k ∈ Finset.range n, f (a + (k + 1) * h) * h = ∑ k ∈ Finset.range n, f (x (k + 1)) * h :=
    Finset.sum_congr rfl (fun k _ => by rw [hxk'])
  refine ⟨?_, ?_, ?_⟩
  · rw [hL, ← hsum]
    exact Finset.sum_le_sum (fun k hk => hlo k (Finset.mem_range.1 hk))
  · rw [hU, ← hsum]
    exact Finset.sum_le_sum (fun k hk => hhi k (Finset.mem_range.1 hk))
  · rw [hU, hL, ← Finset.sum_sub_distrib]
    have : ∑ k ∈ Finset.range n, (f (x (k + 1)) * h - f (x k) * h) = (f (x n) - f (x 0)) * h := by
      rw [← Finset.sum_range_sub (fun k => f (x k)) n, Finset.sum_mul]
      exact Finset.sum_congr rfl (fun k _ => by ring)
    rw [this, hxn, hx0, hh]
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.monotone_riemann
-- test: ra-monotone-riemann-test.sage

/-! ## exm-integral-dirichlet -/

open Classical in
/-- The indicator function of the rationals has infimum `0` and supremum `1` on every subinterval, so all its lower
sums are `0` and all its upper sums are `1`: it is not Riemann integrable. Its Lebesgue integral is `0`. -/
theorem dirichlet (c d : ℝ) (hcd : c < d) :
    sInf ((fun x : ℝ => if ∃ q : ℚ, (q : ℝ) = x then (1 : ℝ) else 0) '' Icc c d) = 0 ∧
      sSup ((fun x : ℝ => if ∃ q : ℚ, (q : ℝ) = x then (1 : ℝ) else 0) '' Icc c d) = 1 ∧
      ∫ x in c..d, (if ∃ q : ℚ, (q : ℝ) = x then (1 : ℝ) else 0) = 0 := by
  set χ : ℝ → ℝ := fun x => if ∃ q : ℚ, (q : ℝ) = x then (1 : ℝ) else 0 with hχ
  -- a rational and an irrational point in (c, d)
  obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn hcd
  obtain ⟨r, ⟨hr1, hr2⟩, hrirr⟩ : ∃ r ∈ Ioo c d, Irrational r := by
    obtain ⟨r, hr⟩ := (dense_irrational.inter_open_nonempty (Ioo c d) isOpen_Ioo (nonempty_Ioo.2 hcd))
    exact ⟨r, hr.1, hr.2⟩
  have himg : χ '' Icc c d = {0, 1} := by
    ext y
    simp only [Set.mem_image, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · rintro ⟨x, -, rfl⟩
      simp only [hχ]
      split_ifs <;> simp
    · rintro (rfl | rfl)
      · refine ⟨r, ⟨hr1.le, hr2.le⟩, ?_⟩
        simp only [hχ]
        rw [if_neg]
        rintro ⟨p, hp⟩
        exact hrirr ⟨p, hp⟩
      · exact ⟨q, ⟨hq1.le, hq2.le⟩, by simp [hχ]⟩
  have hae : χ =ᵐ[volume] fun _ => (0 : ℝ) := by
    have hcount : (Set.range ((↑) : ℚ → ℝ)).Countable := Set.countable_range _
    have hnull : volume (Set.range ((↑) : ℚ → ℝ)) = 0 := hcount.measure_zero volume
    filter_upwards [measure_eq_zero_iff_ae_notMem.1 hnull] with x hx
    simp only [hχ]
    rw [if_neg]
    rintro ⟨p, hp⟩
    exact hx ⟨p, hp⟩
  refine ⟨?_, ?_, ?_⟩
  · rw [himg, csInf_pair]; norm_num
  · rw [himg, csSup_pair]; norm_num
  · have h1 : ∀ᵐ x ∂volume, x ∈ Set.uIoc c d → χ x = (fun _ => (0 : ℝ)) x := hae.mono (fun x hx _ => hx)
    rw [intervalIntegral.integral_congr_ae h1]
    simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.dirichlet
-- test: ra-dirichlet-test.sage

/-! ## exr-integral-linear -/

/-- `∫₀¹ x dx = 1/2`; the lower and upper sums are `(n-1)/(2n)` and `(n+1)/(2n)`, and the midpoint sum is exactly
`1/2`. -/
theorem linear_sums (n : ℕ) (hn : 1 ≤ n) :
    ∑ k ∈ Finset.range n, ((k : ℝ) / n) * (1 / n) = ((n : ℝ) - 1) / (2 * n) ∧
      ∑ k ∈ Finset.range n, (((k : ℝ) + 1) / n) * (1 / n) = ((n : ℝ) + 1) / (2 * n) ∧
      ∑ k ∈ Finset.range n, (((k : ℝ) + 1 / 2) / n) * (1 / n) = 1 / 2 := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hs : ∀ N : ℕ, ∑ k ∈ Finset.range N, (k : ℝ) = ((N : ℝ) - 1) * N / 2 := by
    intro N
    veriphy_certified
      (induction N with
        | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
        | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
    -- certificate: ra-sum-range.sage
  have e : ∀ c : ℝ, ∑ k ∈ Finset.range n, (((k : ℝ) + c) / n) * (1 / n) = ((∑ k ∈ Finset.range n, (k : ℝ)) + n * c) / n ^ 2 := by
    intro c
    calc ∑ k ∈ Finset.range n, (((k : ℝ) + c) / n) * (1 / n) = ∑ k ∈ Finset.range n, ((k : ℝ) + c) / n ^ 2 :=
          Finset.sum_congr rfl (fun k _ => by rw [div_mul_div_comm, mul_one, sq])
      _ = _ := by
          rw [← Finset.sum_div, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  refine ⟨?_, ?_, ?_⟩
  · have := e 0; simp only [add_zero, mul_zero] at this; rw [this, hs]; field_simp <;> ring
  · rw [e 1, hs]; field_simp <;> ring
  · rw [e (1 / 2), hs]; field_simp <;> ring

/-- `∫₀¹ x dx = 1/2`. -/
theorem integral_x : ∫ x in (0 : ℝ)..1, x = 1 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) ((fun x => x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) (f' :=
          fun x => x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-integral-x.sage

/-! ## exr-integral-step -/

/-- A step function is integrable on every interval. -/
theorem step_integrable (a b c u v : ℝ) : IntervalIntegrable (fun x => if x < c then u else v) volume a b := by
  rw [intervalIntegrable_iff]
  refine IntegrableOn.of_bound measure_Ioc_lt_top ?_ (|u| + |v|) ?_
  · exact (Measurable.ite measurableSet_Iio measurable_const measurable_const).aestronglyMeasurable
  · refine Eventually.of_forall (fun x => ?_)
    simp only [Real.norm_eq_abs]
    split_ifs <;> linarith [abs_nonneg u, abs_nonneg v]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.step_integrable
-- test: ra-step-integrable-test.sage

/-- A bounded function continuous except at one point `c` is integrable. -/
theorem jump_integrable (f : ℝ → ℝ) (a b c M : ℝ) (hM : ∀ x, |f x| ≤ M) (hf : ContinuousOn f ({c}ᶜ)) :
    IntervalIntegrable f volume a b := by
  have hm : AEStronglyMeasurable f volume := by
    have := hf.aestronglyMeasurable (μ := volume) (measurableSet_singleton c).compl
    rwa [restrict_compl_singleton] at this
  rw [intervalIntegrable_iff]
  exact IntegrableOn.of_bound measure_Ioc_lt_top hm.restrict M
    (Eventually.of_forall (fun x => by rw [Real.norm_eq_abs]; exact hM x))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.jump_integrable
-- test: ra-jump-integrable-test.sage

/-! ## def-antiderivative -/

/-- Integration by parts: `∫ₐᵇ f g' = f g |ₐᵇ - ∫ₐᵇ f' g`. -/
theorem integration_by_parts (f g f' g' : ℝ → ℝ) (a b : ℝ) (hf : ∀ x, HasDerivAt f (f' x) x)
    (hg : ∀ x, HasDerivAt g (g' x) x) (hf' : Continuous f') (hg' : Continuous g') :
    ∫ x in a..b, f x * g' x = f b * g b - f a * g a - ∫ x in a..b, f' x * g x :=
  integral_mul_deriv_eq_deriv_mul (fun x _ => hf x) (fun x _ => hg x) (hf'.intervalIntegrable _ _)
    (hg'.intervalIntegrable _ _)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integration_by_parts
-- test: ra-integration-by-parts-test.sage

/-- Substitution: `∫ₐᵇ f(g(x)) g'(x) dx = ∫_{g(a)}^{g(b)} f(u) du`. -/
theorem substitution (f g g' : ℝ → ℝ) (a b : ℝ) (hg : ∀ x, HasDerivAt g (g' x) x) (hg' : Continuous g')
    (hf : Continuous f) : ∫ x in a..b, f (g x) * g' x = ∫ u in g a..g b, f u := by
  have := integral_comp_mul_deriv (a := a) (b := b) (f := g) (g := f) (fun x _ => hg x) hg'.continuousOn hf
  simpa [Function.comp_def] using this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.substitution
-- test: ra-substitution-test.sage

/-! ## exm-antiderivative-table -/

/-- `(x^(n+1)/(n+1))' = x^n`. -/
theorem antideriv_pow (n : ℕ) (x : ℝ) : HasDerivAt (fun x => x ^ (n + 1) / ((n : ℝ) + 1)) (x ^ n) x := by
  have h := (hasDerivAt_pow (n + 1) x).div_const ((n : ℝ) + 1)
  convert h using 1
  push_cast
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.antideriv_pow (0) (3) := by
  have h := (hasDerivAt_pow (0 + 1) (3 : ℝ)).div_const (((0 : ℕ) : ℝ) + 1)
  convert h using 1
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.antideriv_pow (2) (1) := by
  have h := (hasDerivAt_pow (2 + 1) (1 : ℝ)).div_const (((2 : ℕ) : ℝ) + 1)
  convert h using 1
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.antideriv_pow (4) (-2) := by
  have h := (hasDerivAt_pow (4 + 1) (-2 : ℝ)).div_const (((4 : ℕ) : ℝ) + 1)
  convert h using 1
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Integral.antideriv_pow
-- test: ra-antideriv-pow-test.sage

/-- `(log |x|)' = 1/x` for `x ≠ 0` (mathlib's `log` is `log |·|`). -/
theorem antideriv_log (x : ℝ) (hx : x ≠ 0) : deriv (fun x => Real.log x) x = 1 / x := by
  rw [Real.deriv_log, one_div]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.antideriv_log
-- test: ra-antideriv-log-test.sage

/-- `(-cos x)' = sin x`. -/
theorem antideriv_sin (x : ℝ) : deriv (fun x => -Real.cos x) x = Real.sin x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -cos x) ((Real.sin x : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).cos).const_mul (-1 : ℝ)) using 1
        all_goals ((ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-antideriv-sin.sage

/-! ## exm-antiderivative-parts -/

/-- `((x - 1) e^x)' = x e^x`. -/
theorem antideriv_x_exp (x : ℝ) : deriv (fun x => (x - 1) * Real.exp x) x = x * Real.exp x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => (x - 1) * rexp x) (((x - 1) * (Real.exp x) + (Real.exp x) : ℝ)) (x) :=
        by
        convert
          (((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul ((hasDerivAt_id' (x : ℝ)).exp)) using
          1
        all_goals (simp only [Pi.add_apply]; ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-antideriv-x-exp.sage

/-- `(x log x - x)' = log x` for `x > 0`. -/
theorem antideriv_log_x (x : ℝ) (hx : 0 < x) : deriv (fun x => x * Real.log x - x) x = Real.log x := by
  have hx' : x ≠ 0 := hx.ne'
  clear hx
  show deriv (fun x => x * Real.log x - x) x = Real.log x
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x * log x - x) ((Real.log x : ℝ)) (x) :=
        by
        convert
          (((hasDerivAt_id' (x : ℝ)).mul ((hasDerivAt_id' (x : ℝ)).log hx')).add
            ((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ))) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.mul_apply]; ring_nf; done)
          | ((field_simp; ring1))
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-antideriv-log-x.sage

/-- `∫₀^∞ xⁿ e^(-ax) dx = n!/a^(n+1)` for `a > 0`; at `a = 1` it is `n!`. -/
theorem factorial_integral (n : ℕ) (a : ℝ) (ha : 0 < a) :
    ∫ x in Ioi (0 : ℝ), x ^ n * Real.exp (-(a * x)) = n ! / a ^ (n + 1) := by
  have h := Real.integral_rpow_mul_exp_neg_mul_Ioi (a := (n : ℝ) + 1) (r := a) (by positivity) ha
  rw [show ((n : ℝ) + 1 - 1) = (n : ℝ) by ring, Real.Gamma_nat_eq_factorial] at h
  have e1 : ∫ x in Ioi (0 : ℝ), x ^ n * Real.exp (-(a * x)) = ∫ x in Ioi (0 : ℝ), x ^ (n : ℝ) * Real.exp (-(a * x)) :=
    setIntegral_congr_fun measurableSet_Ioi (fun x _ => by simp only [Real.rpow_natCast])
  rw [e1, h, show ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) by push_cast; ring, Real.rpow_natCast, one_div_pow]
  ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Integral.factorial_integral
-- test: ra-factorial-integral-test.sage

/-! ## exm-antiderivative-none -/

/-- The error-function integral `F(x) = ∫₀ˣ e^(-t²) dt` has `F' = e^(-x²)`: the fundamental theorem produces an
antiderivative although none is elementary. -/
theorem erf_deriv (x : ℝ) : HasDerivAt (fun x => ∫ t in (0 : ℝ)..x, Real.exp (-t ^ 2)) (Real.exp (-x ^ 2)) x := by
  have hc : Continuous fun t : ℝ => Real.exp (-t ^ 2) := by fun_prop
  exact integral_hasDerivAt_right (hc.intervalIntegrable _ _) (hc.stronglyMeasurableAtFilter _ _) hc.continuousAt

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.erf_deriv
-- test: ra-erf-deriv-test.sage

/-! ## exr-antiderivative-table -/

/-- `(x sin x + cos x)' = x cos x`. -/
theorem antideriv_x_cos (x : ℝ) : deriv (fun x => x * Real.sin x + Real.cos x) x = x * Real.cos x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x * sin x + cos x) ((x * (Real.cos x) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).mul ((hasDerivAt_id' (x : ℝ)).sin)).add ((hasDerivAt_id' (x : ℝ)).cos)) using
          1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-antideriv-x-cos.sage

/-- `((log x)²/2)' = log x / x` for `x > 0`. -/
theorem antideriv_log_sq (x : ℝ) (hx : 0 < x) : deriv (fun x => Real.log x ^ 2 / 2) x = Real.log x / x := by
  have hx' : x ≠ 0 := hx.ne'
  clear hx
  show deriv (fun x => Real.log x ^ 2 / 2) x = Real.log x / x
  veriphy_certified
    ( have hd : HasDerivAt (fun x => log x ^ 2 / 2) (((Real.log x) / x : ℝ)) (x) :=
        by
        convert ((((hasDerivAt_id' (x : ℝ)).log hx').pow 2).const_mul (1 / 2 : ℝ)) using 1
        all_goals
          first
          | rfl
          | ring1
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (funext y; ring_nf; done)
          | (simp only [Pi.pow_apply]; (ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-antideriv-log-sq.sage

/-- `(-log |cos x|)' = tan x = sin x / cos x` where `cos x ≠ 0`. -/
theorem antideriv_tan (x : ℝ) (hx : Real.cos x ≠ 0) :
    deriv (fun x => -Real.log (Real.cos x)) x = Real.sin x / Real.cos x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -log (cos x)) (((Real.sin x) / (Real.cos x) : ℝ)) (x) :=
        by
        convert ((((hasDerivAt_id' (x : ℝ)).cos).log (by exact hx)).const_mul (-1 : ℝ)) using 1
        all_goals ((ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-antideriv-tan.sage

/-- `(½ log |(x - 1)/(x + 1)|)' = 1/(x² - 1)` for `x ≠ ±1`. -/
theorem antideriv_partial_fractions (x : ℝ) (h1 : x - 1 ≠ 0) (h2 : x + 1 ≠ 0) :
    deriv (fun x => (Real.log (x - 1) - Real.log (x + 1)) / 2) x = 1 / (x ^ 2 - 1) := by
  have h3 : x + -1 ≠ 0 := by rwa [← sub_eq_add_neg]
  have h4 : x ^ 2 - 1 ≠ 0 := by rw [show x ^ 2 - 1 = (x - 1) * (x + 1) by ring]; exact mul_ne_zero h1 h2
  show deriv (fun x => (Real.log (x - 1) - Real.log (x + 1)) / 2) x = 1 / (x ^ 2 - 1)
  veriphy_certified
    ( have hd :
        HasDerivAt (fun x => (log (x - 1) - log (x + 1)) / 2) ((-1 / (2 * (x + 1)) + (1 / (2 * (x - 1))) : ℝ)) (x) :=
        by
        convert
          (((((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (1 : ℝ))).log h2).const_mul (-(1 / 2) : ℝ)).add
            ((((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).log h1).const_mul (1 / 2 : ℝ))) using
          1
        all_goals
          first
          | rfl
          | ring1
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (funext y; simp; ring_nf; done)
          | (simp only [Pi.add_apply]; (field_simp; ring1))
      exact hd.deriv.trans (by (field_simp; ring1)))
  -- certificate: ra-antideriv-partial-fractions.sage

/-! ## exr-antiderivative-reduction -/

/-- The reduction formula `I_n = (n-1)/n I_(n-2)` for `I_n = ∫₀^(π/2) sinⁿ x dx`. -/
theorem sin_pow_reduction (n : ℕ) :
    ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (n + 2) = ((n : ℝ) + 1) / ((n : ℝ) + 2) * ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ n := by
  rw [integral_sin_pow]
  simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.sin_pow_reduction
-- test: ra-sin-pow-reduction-test.sage

/-- `I_4 = 3π/16`. -/
theorem sin_pow_four : ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ 4 = 3 * π / 16 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π / 2 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (3 * x_veriphy / 8 + ((Real.sin (4 * x_veriphy)) / 32) - ((Real.sin (2 * x_veriphy)) / 4) : ℝ))
            ((fun x => sin x ^ 4) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 / 8 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).sin).const_mul (1 / 32 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                      (-(1 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        ·
          (rw [show (4 : ℝ) * x_veriphy = 2 * (2 * x_veriphy) by ring]; simp only [Real.cos_two_mul, Real.cos_sq'];
            ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (3 * x_veriphy / 8 + ((Real.sin (4 * x_veriphy)) / 32) - ((Real.sin (2 * x_veriphy)) / 4) : ℝ)) (f' :=
          fun x => sin x ^ 4) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((4 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((4 * (π / 2) : ℝ)) = 0 := by
        rw [show ((4 * (π / 2) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      have he2 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((2 * (π / 2) : ℝ)) = 0 := by
        rw [show ((2 * (π / 2) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: ra-sin-pow-four.sage

/-- `I_5 = 8/15`. -/
theorem sin_pow_five : ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ 5 = 8 / 15 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π / 2 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 5) / 5 + (2 * ((Real.cos x_veriphy) ^ 3) / 3) - (Real.cos x_veriphy) : ℝ))
            ((fun x => sin x ^ 5) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(1 / 5) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (2 / 3 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        ·
          ((try simp only [show (4 : ℝ) * x_veriphy = 2 * (2 * x_veriphy) by ring, Real.cos_two_mul, Real.sin_two_mul]);
            linear_combination
              (-Real.sin x_veriphy ^ 3 + Real.sin x_veriphy * Real.cos x_veriphy ^ 2 - Real.sin x_veriphy) *
                Real.sin_sq_add_cos_sq x_veriphy)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 5) / 5 + (2 * ((Real.cos x_veriphy) ^ 3) / 3) - (Real.cos x_veriphy) : ℝ)) (f' :=
          fun x => sin x ^ 5) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      simp only [he0]
      all_goals (norm_num; done))
  -- certificate: ra-sin-pow-five.sage

/-! ## exr-antiderivative-substitution -/

/-- `∫₀ᵃ x √(a² - x²) dx = a³/3` by the substitution `u = a² - x²` (at `a = 1`: `∫₀¹ x √(1 - x²) dx = 1/3`). The
integrand's antiderivative `-(a² - x²)^(3/2)/3` is not differentiable at `x = a`, so the claim is proved by the
substitution rule and tested at instances. -/
theorem integral_x_sqrt (a : ℝ) (ha : 0 ≤ a) : ∫ x in (0 : ℝ)..a, x * √(a ^ 2 - x ^ 2) = a ^ 3 / 3 := by
  have hf : ∀ x ∈ uIcc (0 : ℝ) a, HasDerivAt (fun x => a ^ 2 - x ^ 2) (-(2 * x)) x := fun x _ => by
    simpa using (hasDerivAt_pow 2 x).const_sub (a ^ 2)
  have h := intervalIntegral.integral_comp_mul_deriv' hf (by fun_prop) Real.continuous_sqrt.continuousOn
  simp only [Function.comp_def] at h
  have e : ∫ x in (0 : ℝ)..a, x * √(a ^ 2 - x ^ 2) =
      -(1 / 2) * ∫ x in (0 : ℝ)..a, √(a ^ 2 - x ^ 2) * -(2 * x) := by
    rw [← intervalIntegral.integral_const_mul]
    congr 1; funext x; ring
  have hv : ∫ u in (0 : ℝ)..(a ^ 2), √u = 2 / 3 * a ^ 3 := by
    simp_rw [Real.sqrt_eq_rpow]
    rw [integral_rpow (Or.inl (by norm_num)), Real.zero_rpow (by norm_num)]
    rw [← Real.rpow_natCast a 2, ← Real.rpow_mul ha]
    rw [show ((2 : ℕ) : ℝ) * ((1 : ℝ) / 2 + 1) = ((3 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    ring
  rw [e, h, show (a ^ 2 - 0 ^ 2 : ℝ) = a ^ 2 by ring, sub_self, intervalIntegral.integral_symm, hv]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integral_x_sqrt
-- test: ra-integral-x-sqrt-test.sage

/-- `∫₀^a cos x/(1 + sin² x) dx = arctan(sin a)`; at `a = π/2` it is `arctan 1 = π/4`. -/
theorem integral_cos_div (a : ℝ) :
    ∫ x in (0 : ℝ)..a, Real.cos x / (1 + Real.sin x ^ 2) = Real.arctan (Real.sin a) := by
  have h : ∀ x ∈ uIcc (0 : ℝ) a, HasDerivAt (fun x => Real.arctan (Real.sin x))
      (Real.cos x / (1 + Real.sin x ^ 2)) x := fun x _ => by
    have := (Real.hasDerivAt_arctan (Real.sin x)).comp x (Real.hasDerivAt_sin x)
    exact this.congr_deriv (by ring)
  rw [integral_eq_sub_of_hasDerivAt h (by
    apply Continuous.intervalIntegrable
    exact Continuous.div (by fun_prop) (by fun_prop) (fun x => by positivity))]
  simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integral_cos_div
-- test: ra-integral-cos-div-test.sage

/-! ## thm-fundamental-theorem -/

/-- The fundamental theorem of calculus: `F(x) = ∫ₐˣ f` has `F' = f` for continuous `f`, and `∫ₐᵇ f = G(b) - G(a)`
for every antiderivative `G`. -/
theorem fundamental_theorem (f G : ℝ → ℝ) (a b x : ℝ) (hf : Continuous f) (hG : ∀ y, HasDerivAt G (f y) y) :
    HasDerivAt (fun x => ∫ t in a..x, f t) (f x) x ∧ ∫ t in a..b, f t = G b - G a :=
  ⟨integral_hasDerivAt_right (hf.intervalIntegrable _ _) (hf.stronglyMeasurableAtFilter _ _) hf.continuousAt,
    integral_eq_sub_of_hasDerivAt (fun y _ => hG y) (hf.intervalIntegrable _ _)⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.fundamental_theorem
-- test: ra-fundamental-theorem-test.sage

/-! ## exm-ftc-sin -/

/-- `∫₀^π sin x dx = 2`. -/
theorem integral_sin_pi : ∫ x in (0 : ℝ)..π, Real.sin x = 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) ((fun x => sin x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) (f' :=
          fun x => sin x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: ra-integral-sin-pi.sage

/-! ## exm-ftc-log -/

/-- `log x = ∫₁ˣ dt/t` for `x > 0`. -/
theorem log_integral (x : ℝ) (hx : 0 < x) : Real.log x = ∫ t in (1 : ℝ)..x, 1 / t := by
  rw [integral_one_div_of_pos one_pos hx, div_one]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.log_integral
-- test: ra-log-integral-test.sage

/-! ## exr-fundamental-theorem-sage -/

/-- `∫₀¹ x e^x dx = 1`. -/
theorem integral_x_exp : ∫ x in (0 : ℝ)..1, x * Real.exp x = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy - 1) * (Real.exp x_veriphy) : ℝ)) ((fun x => x * rexp x) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).add (hasDerivAt_const (x_veriphy : ℝ) (-1 : ℝ))).mul
                    ((hasDerivAt_id' (x_veriphy : ℝ)).exp))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy - 1) * (Real.exp x_veriphy) : ℝ)) (f' := fun x => x * rexp x) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-integral-x-exp.sage

/-- `∫₋ₐᵃ dx/(1 + x²) = 2 arctan a`; at `a = 1` it is `π/2`. -/
theorem integral_arctan (a : ℝ) : ∫ x in (-a)..a, 1 / (1 + x ^ 2) = 2 * Real.arctan a := by
  rw [integral_one_div_one_add_sq, Real.arctan_neg]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integral_arctan
-- test: ra-integral-arctan-test.sage

/-- `∫₁⁷ dx/x = log 7`. -/
theorem integral_inv_seven : ∫ x in (1 : ℝ)..7, 1 / x = Real.log 7 := by
  veriphy_certified
    ( have hne : ∀ x ∈ Set.uIcc (1 : ℝ) 7, x ≠ 0 := by
        intro x hx
        rw [Set.uIcc_of_le (by norm_num)] at hx
        exact (lt_of_lt_of_le zero_lt_one hx.1).ne'
      have hF :
        ∀ x_veriphy ∈ Set.uIcc (1 : ℝ) (7 : ℝ),
          HasDerivAt (fun x_veriphy => (Real.log x_veriphy : ℝ)) ((fun x => 1 / x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy hx_veriphy
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).log (by exact hne x_veriphy hx_veriphy))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.log x_veriphy : ℝ)) (f' :=
          fun x => 1 / x) hF
          (ContinuousOn.intervalIntegrable (ContinuousOn.div continuousOn_const continuousOn_id hne))]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-integral-inv-seven.sage

/-! ## exr-ftc-leibniz -/

/-- `d/dx ∫ₓ^(x²) e^(-t²) dt = 2x e^(-x⁴) - e^(-x²)`. -/
theorem leibniz_variable_limits (x : ℝ) :
    HasDerivAt (fun x => ∫ t in x..x ^ 2, Real.exp (-t ^ 2))
      (2 * x * Real.exp (-x ^ 4) - Real.exp (-x ^ 2)) x := by
  have hc : Continuous (fun t : ℝ => Real.exp (-t ^ 2)) := by fun_prop
  set G : ℝ → ℝ := fun y => ∫ t in (0 : ℝ)..y, Real.exp (-t ^ 2) with hG
  have hGd : ∀ y, HasDerivAt G (Real.exp (-y ^ 2)) y := fun y =>
    integral_hasDerivAt_right (hc.intervalIntegrable _ _) (hc.stronglyMeasurableAtFilter _ _) hc.continuousAt
  have he : (fun x => ∫ t in x..x ^ 2, Real.exp (-t ^ 2)) = fun x => G (x ^ 2) - G x := by
    funext y
    simp only [hG]
    rw [integral_interval_sub_left (hc.intervalIntegrable _ _) (hc.intervalIntegrable _ _)]
  rw [he]
  have h1 := (hGd (x ^ 2)).comp x (hasDerivAt_pow 2 x)
  have h2 := h1.sub (hGd x)
  exact h2.congr_deriv (by rw [show (x ^ 2) ^ 2 = x ^ 4 by ring]; norm_num <;> ring)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.leibniz_variable_limits
-- test: ra-leibniz-variable-limits-test.sage

/-- `d/dx ∫₀ˣ (x - t) f(t) dt = ∫₀ˣ f`, so the integral solves `y'' = f`, `y(0) = y'(0) = 0`. -/
theorem leibniz_kernel (f : ℝ → ℝ) (hf : Continuous f) (x : ℝ) :
    HasDerivAt (fun x => ∫ t in (0 : ℝ)..x, (x - t) * f t) (∫ t in (0 : ℝ)..x, f t) x := by
  have hc2 : Continuous (fun t : ℝ => t * f t) := continuous_id.mul hf
  have he : (fun x => ∫ t in (0 : ℝ)..x, (x - t) * f t) =
      fun x => x * (∫ t in (0 : ℝ)..x, f t) - ∫ t in (0 : ℝ)..x, t * f t := by
    funext y
    rw [← intervalIntegral.integral_const_mul, ← integral_sub ((hf.const_mul y).intervalIntegrable _ _) (hc2.intervalIntegrable _ _)]
    congr 1
    funext t
    ring
  rw [he]
  have hF : HasDerivAt (fun x => ∫ t in (0 : ℝ)..x, f t) (f x) x :=
    integral_hasDerivAt_right (hf.intervalIntegrable _ _) (hf.stronglyMeasurableAtFilter _ _) hf.continuousAt
  have hH : HasDerivAt (fun x => ∫ t in (0 : ℝ)..x, t * f t) (x * f x) x :=
    integral_hasDerivAt_right (hc2.intervalIntegrable _ _) (hc2.stronglyMeasurableAtFilter _ _) hc2.continuousAt
  exact (((hasDerivAt_id x).mul hF).sub hH).congr_deriv (by simp only [id_eq, one_mul]; ring)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.leibniz_kernel
-- test: ra-leibniz-kernel-test.sage

/-! ## exr-ftc-average -/

/-- The mean value theorem for integrals: a continuous `f` on `[a, b]` attains its average. -/
theorem integral_mean_value (f : ℝ → ℝ) (a b : ℝ) (hab : a < b) (hf : Continuous f) :
    ∃ c ∈ Icc a b, ∫ x in a..b, f x = f c * (b - a) := by
  have hne : (Icc a b).Nonempty := nonempty_Icc.2 hab.le
  obtain ⟨p, hp, hpmax⟩ := isCompact_Icc.exists_isMaxOn hne hf.continuousOn
  obtain ⟨q, hq, hqmin⟩ := isCompact_Icc.exists_isMinOn hne hf.continuousOn
  have hlo : f q * (b - a) ≤ ∫ x in a..b, f x := by
    have := integral_mono_on hab.le intervalIntegrable_const (hf.intervalIntegrable (μ := volume) a b) (fun x hx => hqmin hx)
    simpa [intervalIntegral.integral_const, mul_comm] using this
  have hhi : (∫ x in a..b, f x) ≤ f p * (b - a) := by
    have := integral_mono_on hab.le (hf.intervalIntegrable (μ := volume) a b) intervalIntegrable_const (fun x hx => hpmax hx)
    simpa [intervalIntegral.integral_const, mul_comm] using this
  have hba : 0 < b - a := by linarith
  set m := (∫ x in a..b, f x) / (b - a) with hm
  have hm1 : f q ≤ m := by rw [hm, le_div_iff₀ hba]; exact hlo
  have hm2 : m ≤ f p := by rw [hm, div_le_iff₀ hba]; exact hhi
  -- the intermediate value theorem between q and p
  obtain ⟨c, hc, hfc⟩ : ∃ c ∈ uIcc q p, f c = m :=
    intermediate_value_uIcc hf.continuousOn ⟨le_trans (min_le_left _ _) hm1, le_trans hm2 (le_max_right _ _)⟩
  have hsub : uIcc q p ⊆ Icc a b := uIcc_subset_Icc hq hp
  refine ⟨c, hsub hc, ?_⟩
  rw [hfc, hm]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.integral_mean_value
-- test: ra-integral-mean-value-test.sage

/-- For `x²` on `[0, b]` the mean value point is `c = b/√3`; at `b = 1`, `c² = 1/3`. -/
theorem integral_mean_value_sq (b : ℝ) (hb : 0 < b) :
    ∫ x in (0 : ℝ)..b, x ^ 2 = (b / √3) ^ 2 * (b - 0) ∧ b / √3 ∈ Icc 0 b := by
  have h3 : √(3 : ℝ) ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 1 ≤ √(3 : ℝ) := by rw [show (1 : ℝ) = √1 by simp]; exact Real.sqrt_le_sqrt (by norm_num)
  refine ⟨?_, by positivity, ?_⟩
  · rw [integral_pow, div_pow, h3]; ring
  · rw [div_le_iff₀ (by positivity)]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.integral_mean_value_sq (1) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 1 ≤ Real.sqrt 3 := by rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_le_sqrt (by norm_num)
  refine ⟨?_, by positivity, ?_⟩
  · rw [integral_pow, div_pow, h3]; norm_num
  · rw [div_le_iff₀ (by positivity)]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.integral_mean_value_sq (2) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 1 ≤ Real.sqrt 3 := by rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_le_sqrt (by norm_num)
  refine ⟨?_, by positivity, ?_⟩
  · rw [integral_pow, div_pow, h3]; norm_num
  · rw [div_le_iff₀ (by positivity)]; nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.integral_mean_value_sq (3) ((by norm_num)) := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs : 1 ≤ Real.sqrt 3 := by rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_le_sqrt (by norm_num)
  refine ⟨?_, by positivity, ?_⟩
  · rw [integral_pow, div_pow, h3]; norm_num
  · rw [div_le_iff₀ (by positivity)]; nlinarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Integral.integral_mean_value_sq
-- test: ra-integral-mean-value-sq-test.sage

/-! ## def-improper-integral -/

/-- `∫₁^∞ x^(-p) dx` converges exactly for `p > 1`. -/
theorem p_integral_infty (p : ℝ) : IntegrableOn (fun x : ℝ => x ^ (-p)) (Ioi 1) ↔ 1 < p := by
  rw [integrableOn_Ioi_rpow_iff zero_lt_one]
  constructor <;> intro h <;> linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.p_integral_infty
-- test: ra-p-integral-infty-test.sage

/-- `∫₀¹ x^(-p) dx` converges exactly for `p < 1`. -/
theorem p_integral_zero (p : ℝ) : IntegrableOn (fun x : ℝ => x ^ (-p)) (Ioo 0 1) ↔ p < 1 := by
  rw [integrableOn_Ioo_rpow_iff zero_lt_one]
  constructor <;> intro h <;> linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.p_integral_zero
-- test: ra-p-integral-zero-test.sage

/-- The comparison test: `0 ≤ f ≤ g` with `∫ g` convergent forces `∫ f` convergent. -/
theorem improper_comparison (f g : ℝ → ℝ) (s : Set ℝ) (hs : MeasurableSet s) (hf : ContinuousOn f s)
    (h0 : ∀ x ∈ s, 0 ≤ f x) (hfg : ∀ x ∈ s, f x ≤ g x) (hg : IntegrableOn g s) : IntegrableOn f s := by
  refine hg.mono' (hf.aestronglyMeasurable hs) ?_
  filter_upwards [ae_restrict_mem hs] with x hx
  rw [Real.norm_eq_abs, abs_of_nonneg (h0 x hx)]
  exact hfg x hx

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.improper_comparison
-- test: ra-improper-comparison-test.sage

/-! ## exm-improper-p -/

/-- `∫₁ᵇ x^(-p) dx = (b^(1-p) - 1)/(1 - p)` for `p ≠ 1`, `b > 0`, and `∫₁ᵇ dx/x = log b`. -/
theorem p_integral_finite (p b : ℝ) (hp : p ≠ 1) (hb : 0 < b) :
    ∫ x in (1 : ℝ)..b, x ^ (-p) = (b ^ (1 - p) - 1) / (1 - p) ∧ ∫ x in (1 : ℝ)..b, x⁻¹ = Real.log b := by
  constructor
  · have h0 : (0 : ℝ) ∉ uIcc 1 b := by
      rw [Set.mem_uIcc]; push_neg; constructor <;> intro h <;> linarith
    rw [integral_rpow (Or.inr ⟨by intro h; apply hp; linarith, h0⟩), Real.one_rpow]
    rw [show -p + 1 = 1 - p by ring]
  · rw [integral_inv_of_pos one_pos hb, div_one]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.p_integral_finite
-- test: ra-p-integral-finite-test.sage

/-! ## exm-improper-exponential -/

/-- `∫₀^∞ e^(-ax) dx = 1/a` for `a > 0`; at `a = 1` it is `1`. -/
theorem exp_integral (a : ℝ) (ha : 0 < a) : ∫ x in Ioi (0 : ℝ), Real.exp (-(a * x)) = 1 / a := by
  have := factorial_integral 0 a ha
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.exp_integral
-- test: ra-exp-integral-test.sage

/-- `∫_ℝ dx/(1 + x²) = π`. -/
theorem integral_cauchy (c : ℝ) (hc : 0 < c) : ∫ x : ℝ, 1 / (c ^ 2 + x ^ 2) = π / c := by
  have h := Measure.integral_comp_mul_left (fun t : ℝ => (1 + t ^ 2)⁻¹) c⁻¹
  rw [integral_univ_inv_one_add_sq, inv_inv, abs_of_pos hc, smul_eq_mul] at h
  have e : ∀ x : ℝ, 1 / (c ^ 2 + x ^ 2) = c⁻¹ ^ 2 * (1 + (c⁻¹ * x) ^ 2)⁻¹ := by
    intro x; field_simp
  simp_rw [e]
  rw [MeasureTheory.integral_const_mul, h]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.integral_cauchy
-- test: ra-integral-cauchy-test.sage

/-- `Γ(n + 1) = n!`: the Gamma function interpolates the factorial. -/
theorem gamma_factorial (n : ℕ) : Real.Gamma (n + 1) = n ! :=
  Real.Gamma_nat_eq_factorial n

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.gamma_factorial (0) := by
  rw [Real.Gamma_nat_eq_factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.gamma_factorial (1) := by
  rw [Real.Gamma_nat_eq_factorial]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Integral.gamma_factorial (4) := by
  rw [Real.Gamma_nat_eq_factorial]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Integral.gamma_factorial
-- test: ra-gamma-factorial-test.sage

/-! ## exm-improper-gaussian -/

/-- The Gaussian integral `∫_ℝ e^(-b x²) dx = √(π/b)`; at `b = 1` it is `√π`, at `b = 1/2` it is `√(2π)`. -/
theorem gaussian (b : ℝ) (hb : 0 < b) : ∫ x : ℝ, Real.exp (-b * x ^ 2) = √(π / b) :=
  integral_gaussian b

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.gaussian
-- test: ra-gaussian-test.sage

/-- `∫_ℝ e^(-ax² + bx) dx = √(π/a) e^(b²/4a)` for `a > 0`. -/
theorem gaussian_linear (a b : ℝ) (ha : 0 < a) :
    ∫ x : ℝ, Real.exp (-a * x ^ 2 + b * x) = √(π / a) * Real.exp (b ^ 2 / (4 * a)) := by
  have e : ∀ x : ℝ, Real.exp (-a * x ^ 2 + b * x) =
      Real.exp (b ^ 2 / (4 * a)) * Real.exp (-a * (x - b / (2 * a)) ^ 2) := by
    intro x
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [e]
  rw [MeasureTheory.integral_const_mul, integral_sub_right_eq_self (fun x => Real.exp (-a * x ^ 2)) (b / (2 * a)),
    integral_gaussian]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.gaussian_linear
-- test: ra-gaussian-linear-test.sage

/-- The even moments `∫_ℝ x^(2n) e^(-a x²) dx = (2n-1)!!/(2a)^n · √(π/a)`; at `a = 1`, `(2n-1)!! √π/2ⁿ`. -/
theorem gaussian_moment (n : ℕ) (a : ℝ) (ha : 0 < a) :
    ∫ x : ℝ, x ^ (2 * n) * Real.exp (-a * x ^ 2) = (2 * n - 1 : ℕ)‼ / (2 * a) ^ n * √(π / a) := by
  have habs : ∫ x : ℝ, x ^ (2 * n) * Real.exp (-a * x ^ 2) =
      2 * ∫ t in Ioi (0 : ℝ), t ^ (2 * n) * Real.exp (-a * t ^ 2) := by
    have := integral_comp_abs (f := fun t : ℝ => t ^ (2 * n) * Real.exp (-a * t ^ 2))
    simp only [pow_mul, sq_abs] at this ⊢
    exact this
  rw [habs]
  have hhalf : ∫ t in Ioi (0 : ℝ), t ^ (2 * n) * Real.exp (-a * t ^ 2) =
      a ^ (-((2 * n : ℝ) + 1) / 2) * (1 / 2) * Real.Gamma (((2 * n : ℝ) + 1) / 2) := by
    rw [← integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := 2 * n) (b := a) (by norm_num)
      (by have : (0 : ℝ) ≤ n := n.cast_nonneg; linarith) ha]
    refine setIntegral_congr_fun measurableSet_Ioi (fun t ht => ?_)
    rw [Real.rpow_two, show (2 * (n : ℝ)) = ((2 * n : ℕ) : ℝ) by push_cast; ring, Real.rpow_natCast]
  rw [hhalf]
  have hg : Real.Gamma (((2 * n : ℝ) + 1) / 2) = (2 * n - 1 : ℕ)‼ * √π / 2 ^ n := by
    rw [show ((2 * n : ℝ) + 1) / 2 = (n : ℝ) + 1 / 2 by ring, Real.Gamma_nat_add_half]
  rw [hg]
  have hpow : a ^ (-((2 * n : ℝ) + 1) / 2) = (a ^ n)⁻¹ * (√a)⁻¹ := by
    rw [show -((2 * n : ℝ) + 1) / 2 = -(n : ℝ) + -(1 / 2) by ring, Real.rpow_add ha, Real.rpow_neg ha.le,
      Real.rpow_neg ha.le, Real.rpow_natCast, Real.sqrt_eq_rpow]
  rw [hpow, Real.sqrt_div' π ha.le, mul_pow]
  have : √a ≠ 0 := (Real.sqrt_pos.2 ha).ne'
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Integral.gaussian_moment
-- test: ra-gaussian-moment-test.sage

/-- The odd moments vanish: `∫_ℝ x^(2n+1) e^(-a x²) dx = 0`. -/
theorem gaussian_odd_moment (n : ℕ) (a : ℝ) : ∫ x : ℝ, x ^ (2 * n + 1) * Real.exp (-a * x ^ 2) = 0 := by
  have h := integral_neg_eq_self (fun x : ℝ => x ^ (2 * n + 1) * Real.exp (-a * x ^ 2)) volume
  have e : (fun x : ℝ => (-x) ^ (2 * n + 1) * Real.exp (-a * (-x) ^ 2)) =
      fun x => -(x ^ (2 * n + 1) * Real.exp (-a * x ^ 2)) := by
    funext x
    rw [Odd.neg_pow (odd_two_mul_add_one n), neg_sq]
    ring
  rw [e, MeasureTheory.integral_neg] at h
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.gaussian_odd_moment
-- test: ra-gaussian-odd-moment-test.sage

/-! ## exr-improper-comparison -/

/-- `∫₀ᵃ dx/√x = 2√a` for `a ≥ 0` (an integrable singularity at `0`); at `a = 1` it is `2`. -/
theorem integral_inv_sqrt (a : ℝ) (ha : 0 ≤ a) : ∫ x in (0 : ℝ)..a, 1 / √x = 2 * √a := by
  have e : ∀ x : ℝ, 0 ≤ x → 1 / √x = x ^ (-(1 / 2 : ℝ)) := fun x hx => by
    rw [Real.sqrt_eq_rpow, Real.rpow_neg hx, one_div]
  have hcongr : ∫ x in (0 : ℝ)..a, 1 / √x = ∫ x in (0 : ℝ)..a, x ^ (-(1 / 2 : ℝ)) := by
    refine integral_congr (fun x hx => ?_)
    rw [uIcc_of_le ha] at hx
    exact e x hx.1
  rw [hcongr, integral_rpow (Or.inl (by norm_num)), Real.zero_rpow (by norm_num), Real.sqrt_eq_rpow]
  norm_num
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integral_inv_sqrt
-- test: ra-integral-inv-sqrt-test.sage

/-! ## exr-integral-values-area -/

/-- `∫₋ᵣʳ √(R² - x²) dx = πR²/2`; at `R = 1`, twice it is the area `π` of the unit disc. -/
theorem half_disc (R : ℝ) (hR : 0 < R) : ∫ x in (-R)..R, √(R ^ 2 - x ^ 2) = π * R ^ 2 / 2 := by
  have e : ∀ x : ℝ, √(R ^ 2 - x ^ 2) = R * √(1 - (R⁻¹ * x) ^ 2) := by
    intro x
    rw [show R ^ 2 - x ^ 2 = R ^ 2 * (1 - (R⁻¹ * x) ^ 2) by field_simp, Real.sqrt_mul (sq_nonneg R),
      Real.sqrt_sq hR.le]
  simp_rw [e]
  rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_comp_mul_left (fun t => √(1 - t ^ 2)) (inv_ne_zero hR.ne')]
  rw [show R⁻¹ * -R = -1 by field_simp, show R⁻¹ * R = 1 by field_simp, integral_sqrt_one_sub_sq]
  rw [inv_inv, smul_eq_mul]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Integral.half_disc
-- test: ra-half-disc-test.sage

/-- `∫_(-π/2)^(π/2) cos² θ dθ = π/2` (the substitution `x = sin θ`). -/
theorem integral_cos_sq : ∫ θ in (-(π / 2))..(π / 2), Real.cos θ ^ 2 = π / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-(π / 2) : ℝ) (π / 2 : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy / 2 + ((Real.sin (2 * x_veriphy)) / 4) : ℝ))
            ((fun θ => cos θ ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                      (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp only [Real.cos_sq, Real.sin_sq_eq_half_sub]; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (x_veriphy / 2 + ((Real.sin (2 * x_veriphy)) / 4) : ℝ)) (f' := fun θ => cos θ ^ 2) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((2 * (π / 2) : ℝ)) = 0 := by
        rw [show ((2 * (π / 2) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0]
      all_goals (norm_num [he0]; ring1))
  -- certificate: ra-integral-cos-sq.sage

/-! ## exr-integral-values-simpson -/

/-- The trapezoidal rule for `x²` on `[0, 1]` with `n` subintervals gives `1/3 + 1/(6n²)`; at `n = 4`, `11/32`. -/
theorem trapezoid_sq (n : ℕ) (hn : 1 ≤ n) :
    (1 / (2 * (n : ℝ))) * (0 ^ 2 + 2 * ∑ k ∈ Finset.range n, ((k : ℝ) / n) ^ 2 + 1 ^ 2) = 1 / 3 + 1 / (6 * (n : ℝ) ^ 2) := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hs : ∀ N : ℕ, ∑ k ∈ Finset.range N, (k : ℝ) ^ 2 = ((N : ℝ) - 1) * N * (2 * N - 1) / 6 := by
    intro N
    veriphy_certified
      (induction N with
        | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
        | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
    -- certificate: ra-sum-sq-range.sage
  have e : ∑ k ∈ Finset.range n, ((k : ℝ) / n) ^ 2 = (∑ k ∈ Finset.range n, (k : ℝ) ^ 2) / n ^ 2 := by
    rw [Finset.sum_div]; refine Finset.sum_congr rfl (fun k _ => ?_); rw [div_pow]
  rw [e, hs]
  field_simp
  ring


/-- Simpson's rule with `n = 4` is exact for cubics: `(h/3)(p₀ + 4p₁ + 2p₂ + 4p₃ + p₄) = ∫₀¹ p` for
`p = c₀ + c₁x + c₂x² + c₃x³`, `h = 1/4`. -/
theorem simpson_cubic (c₀ c₁ c₂ c₃ : ℝ) :
    let p := fun x : ℝ => c₀ + c₁ * x + c₂ * x ^ 2 + c₃ * x ^ 3
    (1 / 4 / 3) * (p 0 + 4 * p (1 / 4) + 2 * p (1 / 2) + 4 * p (3 / 4) + p 1) = c₀ + c₁ / 2 + c₂ / 3 + c₃ / 4 := by
  intro p
  simp only [p]
  veriphy_certified (ring1)
  -- certificate: ra-simpson-cubic.sage

/-- For `sin` on `[0, π]` with `h = π/4`: trapezoidal rule `π(1 + √2)/4 = 1.896…`, Simpson's rule
`π(2√2 + 1)/6 = 2.0046…`. -/
theorem sin_rules :
    π / 4 / 2 * (Real.sin 0 + 2 * Real.sin (π / 4) + 2 * Real.sin (π / 2) + 2 * Real.sin (3 * π / 4) + Real.sin π) =
        π * (1 + √2) / 4 ∧
      π / 4 / 3 * (Real.sin 0 + 4 * Real.sin (π / 4) + 2 * Real.sin (π / 2) + 4 * Real.sin (3 * π / 4) + Real.sin π) =
        π * (2 * √2 + 1) / 6 := by
  rw [show 3 * π / 4 = π - π / 4 by ring, Real.sin_pi_sub]
  constructor
  · veriphy_certified
      ( rw [Real.sin_zero]
        rw [Real.sin_pi]
        rw [Real.sin_pi_div_two]
        rw [Real.sin_pi_div_four]
        ring)
    -- certificate: ra-sin-rules.sage
  · veriphy_certified
      ( rw [Real.sin_zero]
        rw [Real.sin_pi]
        rw [Real.sin_pi_div_two]
        rw [Real.sin_pi_div_four]
        ring)
    -- certificate: ra-sin-rules.sage

/-! ## exr-integral-values-integral-test -/

/-- The integral test bounds: for `f` decreasing on `[1, n]`, `∑_{k=2}^n f(k) ≤ ∫₁ⁿ f ≤ ∑_{k=1}^{n-1} f(k)`. -/
theorem integral_test (f : ℝ → ℝ) (n : ℕ) (hf : AntitoneOn f (Icc 1 (1 + n))) :
    ∑ k ∈ Finset.range n, f (1 + (k + 1 : ℕ)) ≤ ∫ x in (1 : ℝ)..(1 + n), f x ∧
      ∫ x in (1 : ℝ)..(1 + n), f x ≤ ∑ k ∈ Finset.range n, f (1 + k) := by
  constructor
  · have := hf.sum_le_integral
    simpa using this
  · have := hf.integral_le_sum
    simpa using this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Integral.integral_test
-- test: ra-integral-test-test.sage

end KaluzaKlein.RealAnalysis.Integral

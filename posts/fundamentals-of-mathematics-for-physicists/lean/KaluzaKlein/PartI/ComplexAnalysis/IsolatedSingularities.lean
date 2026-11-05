import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.LaurentSeries
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Isolated singularities"

The claims of the section "Isolated singularities" of `tutorial/part-i-04-complex-analysis.qmd`: removable
singularities (Riemann's theorem, `sin z/z`), poles of order `m` (`(z - z₀)ᵐ f` analytic and non-zero at `z₀`,
`|f| → ∞`), the orders of the poles of rational functions, of `1/sin z`, `1/(z² sin z)`, `sin z/z³`,
`(e^z - 1)/z²`, `1/(z² + 1)³`, the series of `(1 - cos z)/z⁴` and `z e^{1/z}`, the behaviour of `e^{1/z}` along
the axes (no limit, finite or infinite) and the Casorati-Weierstrass-Picard property of `e^{1/z}`.

Most examples are stated for a family with a parameter (`1/((z - a)(z - b))`, `sin(az)`, `e^{az}`, `z² + b²`), of
which the text's function is the instance `a = 1`, `b = 2` (or `a = 1`).
-/

namespace KaluzaKlein.ComplexAnalysis.IsolatedSingularities

open Complex Filter Topology Metric Function
open scoped Real Nat

/-- `f` has a pole of order `m` at `c`: `f = g/(z - c)ᵐ` near `c` with `g` analytic at `c` and `g(c) ≠ 0`. -/
def IsPoleOfOrder (f : ℂ → ℂ) (c : ℂ) (m : ℕ) : Prop :=
  ∃ g : ℂ → ℂ, AnalyticAt ℂ g c ∧ g c ≠ 0 ∧ ∀ᶠ z in 𝓝[≠] c, f z = g z / (z - c) ^ m

/-! ## def-isolated-singularity -/

/-- A pole of order `m` means that `(z - c)ᵐ f(z)` has a removable singularity with non-zero value at `c`. -/
theorem isPoleOfOrder_iff (f : ℂ → ℂ) (c : ℂ) (m : ℕ) :
    IsPoleOfOrder f c m ↔
      ∃ g : ℂ → ℂ, AnalyticAt ℂ g c ∧ g c ≠ 0 ∧ ∀ᶠ z in 𝓝[≠] c, (z - c) ^ m * f z = g z := by
  constructor
  · rintro ⟨g, hg, hgc, hev⟩
    refine ⟨g, hg, hgc, ?_⟩
    filter_upwards [hev, self_mem_nhdsWithin] with z hz hne
    have : (z - c) ^ m ≠ 0 := pow_ne_zero _ (sub_ne_zero.2 hne)
    rw [hz]; field_simp
  · rintro ⟨g, hg, hgc, hev⟩
    refine ⟨g, hg, hgc, ?_⟩
    filter_upwards [hev, self_mem_nhdsWithin] with z hz hne
    have : (z - c) ^ m ≠ 0 := pow_ne_zero _ (sub_ne_zero.2 hne)
    rw [← hz]; field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.isPoleOfOrder_iff
-- test: ca-res-pole-iff-test.sage

/-- At a pole of order `m ≥ 1`, `|f(z)| → ∞` as `z → z₀`. -/
theorem tendsto_norm_of_pole (g : ℂ → ℂ) (c : ℂ) (m : ℕ) (hm : 0 < m) (hg : ContinuousAt g c)
    (hgc : g c ≠ 0) : Tendsto (fun z => ‖g z / (z - c) ^ m‖) (𝓝[≠] c) atTop := by
  have h1 : Tendsto (fun z => ‖g z‖) (𝓝[≠] c) (𝓝 ‖g c‖) :=
    (hg.norm.tendsto).mono_left nhdsWithin_le_nhds
  have h2 : Tendsto (fun z => ‖z - c‖ ^ m) (𝓝[≠] c) (𝓝[>] 0) := by
    have hc := tendsto_norm_sub_self_nhdsNE c
    refine tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩
    · have h0 : Tendsto (fun z => ‖z - c‖) (𝓝[≠] c) (𝓝 0) := tendsto_nhds_of_tendsto_nhdsWithin hc
      have := h0.pow m
      rwa [zero_pow hm.ne'] at this
    · filter_upwards [self_mem_nhdsWithin] with z hz
      exact pow_pos (norm_pos_iff.2 (sub_ne_zero.2 hz)) m
  have := h1.pos_mul_atTop (norm_pos_iff.2 hgc) h2.inv_tendsto_nhdsGT_zero
  refine this.congr fun z => ?_
  rw [norm_div, norm_pow, div_eq_mul_inv]
  rfl

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.tendsto_norm_of_pole
-- test: ca-res-pole-norm-infty-test.sage

/-- Riemann's theorem: a function holomorphic and bounded on a punctured neighbourhood of `c` extends
holomorphically to `c` (the singularity is removable). -/
theorem removable_of_bounded (f : ℂ → ℂ) (s : Set ℂ) (c : ℂ) (hs : s ∈ 𝓝 c)
    (hd : DifferentiableOn ℂ f (s \ {c})) (hb : BddAbove (norm ∘ f '' (s \ {c}))) :
    DifferentiableOn ℂ (update f c (limUnder (𝓝[≠] c) f)) s :=
  Complex.differentiableOn_update_limUnder_of_bddAbove hs hd hb

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.removable_of_bounded (fun _ => (0 : ℂ)) (Metric.ball 0 1) (0) ((Metric.ball_mem_nhds 0 one_pos)) ((differentiableOn_const _)) ((⟨0, by rintro _ ⟨x, -, rfl⟩; simp⟩)) := by
  exact Complex.differentiableOn_update_limUnder_of_bddAbove (Metric.ball_mem_nhds 0 one_pos) (differentiableOn_const _) (⟨0, by rintro _ ⟨x, -, rfl⟩; simp⟩)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.removable_of_bounded (fun _ => (1 : ℂ)) (Metric.ball 0 1) (0) ((Metric.ball_mem_nhds 0 one_pos)) ((differentiableOn_const _)) ((⟨1, by rintro _ ⟨x, -, rfl⟩; simp⟩)) := by
  exact Complex.differentiableOn_update_limUnder_of_bddAbove (Metric.ball_mem_nhds 0 one_pos) (differentiableOn_const _) (⟨1, by rintro _ ⟨x, -, rfl⟩; simp⟩)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.removable_of_bounded (fun _ => Complex.I) (Metric.ball 0 1) (0) ((Metric.ball_mem_nhds 0 one_pos)) ((differentiableOn_const _)) ((⟨1, by rintro _ ⟨x, -, rfl⟩; simp⟩)) := by
  exact Complex.differentiableOn_update_limUnder_of_bddAbove (Metric.ball_mem_nhds 0 one_pos) (differentiableOn_const _) (⟨1, by rintro _ ⟨x, -, rfl⟩; simp⟩)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.removable_of_bounded
-- test: ca-res-removable-test.sage

/-! ## exm-singularity-removable -/

/-- `sin z/z = 1 - z²/6 + z⁴/120 - …` on `z ≠ 0`: no principal part. -/
theorem sinc_series (z : ℂ) (hz : z ≠ 0) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / (Nat.factorial (2 * n + 1) : ℂ)) (sin z / z) := by
  have := (Complex.hasSum_sin z).div_const z
  convert this using 1
  funext n
  rw [pow_succ]; field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_series
-- test: ca-res-sinc-series-test.sage

/-- `sin(az)/z → a` as `z → 0`; for `a = 1` the removable singularity of `sin z/z` has the value `1`. -/
theorem sinc_limit (a : ℂ) : Tendsto (fun z => sin (a * z) / z) (𝓝[≠] 0) (𝓝 a) := by
  have hd : HasDerivAt (fun z => sin (a * z)) (cos (a * 0) * a) 0 := by
    simpa using ((hasDerivAt_id (0 : ℂ)).const_mul a).csin
  have := hasDerivAt_iff_tendsto_slope.1 hd
  simp only [mul_zero, Complex.cos_zero, one_mul] at this
  refine this.congr fun z => ?_
  simp [slope_def_field, div_eq_mul_inv]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_limit
-- test: ca-res-sinc-limit-test.sage

/-- Setting the value `1` at `0` makes `sin z/z` entire. -/
theorem sinc_entire (z : ℂ) : DifferentiableAt ℂ (update (fun z => sin z / z) 0 1) z := by
  have h : update (fun z => sin z / z) 0 1 = dslope sin 0 := by
    funext w
    by_cases hw : w = 0
    · subst hw; simp [dslope_same]
    · rw [update_of_ne hw, dslope_of_ne _ hw, slope_def_field]; simp
  rw [h]
  have := (Complex.differentiableOn_dslope (s := Set.univ) (f := sin) (c := 0) Filter.univ_mem).2
    differentiable_sin.differentiableOn
  exact (this z (Set.mem_univ z)).differentiableAt Filter.univ_mem

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_entire (0) := by
  have h : Function.update (fun z : ℂ => Complex.sin z / z) 0 1 = dslope Complex.sin 0 := by
    funext w
    by_cases hw : w = 0
    · subst hw; simp [dslope_same]
    · rw [Function.update_of_ne hw, dslope_of_ne _ hw, slope_def_field]; simp
  rw [h]
  have := (Complex.differentiableOn_dslope (s := Set.univ) (f := Complex.sin) (c := 0) Filter.univ_mem).2
    Complex.differentiable_sin.differentiableOn
  exact (this _ (Set.mem_univ _)).differentiableAt Filter.univ_mem

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_entire (1) := by
  have h : Function.update (fun z : ℂ => Complex.sin z / z) 0 1 = dslope Complex.sin 0 := by
    funext w
    by_cases hw : w = 0
    · subst hw; simp [dslope_same]
    · rw [Function.update_of_ne hw, dslope_of_ne _ hw, slope_def_field]; simp
  rw [h]
  have := (Complex.differentiableOn_dslope (s := Set.univ) (f := Complex.sin) (c := 0) Filter.univ_mem).2
    Complex.differentiable_sin.differentiableOn
  exact (this _ (Set.mem_univ _)).differentiableAt Filter.univ_mem

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_entire (Complex.I) := by
  have h : Function.update (fun z : ℂ => Complex.sin z / z) 0 1 = dslope Complex.sin 0 := by
    funext w
    by_cases hw : w = 0
    · subst hw; simp [dslope_same]
    · rw [Function.update_of_ne hw, dslope_of_ne _ hw, slope_def_field]; simp
  rw [h]
  have := (Complex.differentiableOn_dslope (s := Set.univ) (f := Complex.sin) (c := 0) Filter.univ_mem).2
    Complex.differentiable_sin.differentiableOn
  exact (this _ (Set.mem_univ _)).differentiableAt Filter.univ_mem

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sinc_entire
-- test: ca-res-sinc-entire-test.sage

/-! ## exm-singularity-poles -/

lemma analyticAt_one_div_sub (a b : ℂ) (hab : a ≠ b) : AnalyticAt ℂ (fun z : ℂ => 1 / (z - b)) a :=
  analyticAt_const.div (analyticAt_id.sub analyticAt_const) (sub_ne_zero.2 hab)

/-- `1/((z - a)(z - b))`, `a ≠ b`, has simple poles at `a` and `b` (the text: `a = 1`, `b = 2`). -/
theorem two_simple_poles (a b : ℂ) (hab : a ≠ b) :
    IsPoleOfOrder (fun z => 1 / ((z - a) * (z - b))) a 1 ∧
      IsPoleOfOrder (fun z => 1 / ((z - a) * (z - b))) b 1 := by
  refine ⟨⟨fun z => 1 / (z - b), analyticAt_one_div_sub a b hab, by simp [sub_ne_zero.2 hab],
    Eventually.of_forall fun z => ?_⟩, ⟨fun z => 1 / (z - a), analyticAt_one_div_sub b a hab.symm,
    by simp [sub_ne_zero.2 hab.symm], Eventually.of_forall fun z => ?_⟩⟩
  · rw [pow_one, div_div, mul_comm]
  · rw [pow_one, div_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.two_simple_poles
-- test: ca-res-two-simple-poles-test.sage

/-- `(z² + 1)/(z (z - a)²)`, `a ≠ 0`, `a² ≠ -1`, has a simple pole at `0` and a pole of order `2` at `a`
(the text: `a = 1`). -/
theorem rational_poles (a : ℂ) (ha : a ≠ 0) (ha' : a ^ 2 + 1 ≠ 0) :
    IsPoleOfOrder (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2)) 0 1 ∧
      IsPoleOfOrder (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2)) a 2 := by
  refine ⟨⟨fun z => (z ^ 2 + 1) / (z - a) ^ 2, ?_, ?_, Eventually.of_forall fun z => ?_⟩,
    ⟨fun z => (z ^ 2 + 1) / z, ?_, ?_, Eventually.of_forall fun z => ?_⟩⟩
  · exact (by fun_prop : AnalyticAt ℂ (fun z : ℂ => z ^ 2 + 1) 0).div (by fun_prop)
      (pow_ne_zero _ (by simpa using ha))
  · simp [ha]
  · rw [sub_zero, pow_one, div_div, mul_comm]
  · exact (by fun_prop : AnalyticAt ℂ (fun z : ℂ => z ^ 2 + 1) a).div analyticAt_id ha
  · simp [ha, ha']
  · rw [div_div]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.rational_poles
-- test: ca-res-rational-poles-test.sage

lemma analyticAt_dslope (f : ℂ → ℂ) (hf : Differentiable ℂ f) (c : ℂ) : AnalyticAt ℂ (dslope f c) c := by
  have hd : DifferentiableOn ℂ (dslope f c) Set.univ :=
    (Complex.differentiableOn_dslope Filter.univ_mem).2 hf.differentiableOn
  exact (hd.analyticOnNhd isOpen_univ) c (Set.mem_univ c)

lemma cos_int_mul_pi (k : ℤ) : cos ((k : ℂ) * π) ≠ 0 := by
  intro h
  have h2 := Complex.sin_sq_add_cos_sq ((k : ℂ) * π)
  have hs : sin ((k : ℂ) * π) = 0 := by rw [Complex.sin_eq_zero_iff]; exact ⟨k, rfl⟩
  rw [h, hs] at h2; norm_num at h2

/-- `1/sin z` has a simple pole at every `kπ`. -/
theorem csc_simple_pole (k : ℤ) : IsPoleOfOrder (fun z => 1 / sin z) ((k : ℂ) * π) 1 := by
  set c : ℂ := (k : ℂ) * π
  have hs : sin c = 0 := by rw [Complex.sin_eq_zero_iff]; exact ⟨k, rfl⟩
  have hd0 : dslope sin c c ≠ 0 := by rw [dslope_same, Complex.deriv_sin]; exact cos_int_mul_pi k
  refine ⟨fun z => 1 / dslope sin c z, analyticAt_const.div (analyticAt_dslope sin differentiable_sin c) hd0,
    by simpa using hd0, ?_⟩
  filter_upwards [self_mem_nhdsWithin] with z hz
  rw [dslope_of_ne _ hz, slope_def_field, hs, sub_zero, pow_one]
  have : z - c ≠ 0 := sub_ne_zero.2 hz
  by_cases h : sin z = 0
  · simp [h]
  · field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.csc_simple_pole
-- test: ca-res-csc-pole-test.sage

/-! ## exm-singularity-essential -/

/-- Along the positive real axis `e^{a/x} → ∞` (`a > 0`; the text: `a = 1`). -/
theorem exp_div_pos_axis (a : ℝ) (ha : 0 < a) :
    Tendsto (fun x : ℝ => Real.exp (a / x)) (𝓝[>] 0) atTop :=
  Real.tendsto_exp_atTop.comp (tendsto_inv_nhdsGT_zero.const_mul_atTop ha |>.congr fun x => by ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_div_pos_axis
-- test: ca-res-exp-pos-axis-test.sage

/-- Along the negative real axis `e^{a/x} → 0` (`a > 0`). -/
theorem exp_div_neg_axis (a : ℝ) (ha : 0 < a) :
    Tendsto (fun x : ℝ => Real.exp (a / x)) (𝓝[<] 0) (𝓝 0) :=
  Real.tendsto_exp_atBot.comp (tendsto_inv_nhdsLT_zero.const_mul_atBot ha |>.congr fun x => by ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_div_neg_axis
-- test: ca-res-exp-neg-axis-test.sage

/-- Along the imaginary axis `|e^{1/(iy)}| = |e^{-i/y}| = 1`: the values run around the unit circle. -/
theorem exp_inv_imag_axis (y : ℝ) : ‖exp (1 / (y * I))‖ = 1 := by
  rw [Complex.norm_exp]
  have : (1 / ((y : ℂ) * I)).re = 0 := by
    by_cases hy : y = 0
    · simp [hy]
    · simp [Complex.div_re, Complex.normSq]
  rw [this, Real.exp_zero]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_imag_axis (0) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_imag_axis
-- test: ca-res-exp-imag-axis-test.sage

/-- `e^{a/z}`, `a ≠ 0`, has no limit at `0`, neither finite nor infinite (the text: `a = 1`). -/
theorem exp_inv_no_limit (a : ℂ) (ha : a ≠ 0) :
    (¬ ∃ L, Tendsto (fun z => exp (a / z)) (𝓝[≠] 0) (𝓝 L)) ∧
      ¬ Tendsto (fun z => exp (a / z)) (𝓝[≠] 0) (Bornology.cobounded ℂ) := by
  -- along `z = a x`, `x → 0⁺`, the values are `e^{1/x} → ∞`; along `x → 0⁻` they tend to `0`
  have hmap : ∀ s : Set ℝ, (∀ x ∈ s, x ≠ 0) →
      Tendsto (fun x : ℝ => a * x) (𝓝[s] 0) (𝓝[≠] 0) := by
    intro s hs
    refine tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩
    · have : Tendsto (fun x : ℝ => a * x) (𝓝 0) (𝓝 (a * 0)) :=
        (continuous_const.mul continuous_ofReal).tendsto 0
      simpa using this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with x hx
      exact mul_ne_zero ha (by exact_mod_cast hs x hx)
  have hval : ∀ x : ℝ, x ≠ 0 → exp (a / (a * x)) = ((Real.exp (1 / x) : ℝ) : ℂ) := by
    intro x hx
    rw [show a / (a * x) = ((1 / x : ℝ) : ℂ) by push_cast; field_simp, Complex.ofReal_exp]
  have hpos : Tendsto (fun x : ℝ => ‖exp (a / (a * x))‖) (𝓝[>] 0) atTop := by
    refine (exp_div_pos_axis 1 one_pos).congr' ?_
    filter_upwards [self_mem_nhdsWithin] with x hx
    rw [hval x (ne_of_gt hx), Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  have hneg : Tendsto (fun x : ℝ => ‖exp (a / (a * x))‖) (𝓝[<] 0) (𝓝 0) := by
    refine (exp_div_neg_axis 1 one_pos).congr' ?_
    filter_upwards [self_mem_nhdsWithin] with x hx
    rw [hval x (ne_of_lt hx), Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  constructor
  · rintro ⟨L, hL⟩
    have := (hL.comp (hmap (Set.Ioi 0) fun x hx => ne_of_gt hx)).norm
    exact not_tendsto_nhds_of_tendsto_atTop hpos _ this
  · intro h
    have h1 := (h.comp (hmap (Set.Iio 0) fun x hx => ne_of_lt hx))
    have h2 : Tendsto (fun x : ℝ => ‖exp (a / (a * x))‖) (𝓝[<] 0) atTop :=
      tendsto_norm_cobounded_atTop.comp h1
    exact not_tendsto_nhds_of_tendsto_atTop h2 _ hneg

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_no_limit
-- test: ca-res-exp-no-limit-test.sage

/-! ## exr-singularity-classify -/

/-- `(1 - cos z)/z⁴ = 1/(2z²) - ∑ (-1)ⁿ z²ⁿ/(2n + 4)!` on `z ≠ 0`: the series starts `1/(2z²) - 1/24 + …`. -/
theorem one_sub_cos_series (z : ℂ) (hz : z ≠ 0) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / (Nat.factorial (2 * n + 4) : ℂ))
      (1 / (2 * z ^ 2) - (1 - cos z) / z ^ 4) := by
  have h := (hasSum_nat_add_iff' 2).2 (Complex.hasSum_cos z)
  have h' := h.div_const (z ^ 4)
  convert h' using 1
  · funext n
    rw [show 2 * (n + 2) = 2 * n + 4 by ring, pow_add (-1 : ℂ) n 2, pow_add z (2 * n) 4]
    field_simp
  · simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.one_sub_cos_series
-- test: ca-res-one-sub-cos-series-test.sage

/-- `z² · (1 - cos(az))/z⁴ → a²/2`: a pole of order `2` with leading coefficient `a²/2` (the text: `a = 1`). -/
theorem one_sub_cos_order_two (a : ℂ) :
    Tendsto (fun z => z ^ 2 * ((1 - cos (a * z)) / z ^ 4)) (𝓝[≠] 0) (𝓝 (a ^ 2 / 2)) := by
  have h := ((sinc_limit (a / 2)).pow 2).const_mul 2
  have hv : 2 * (a / 2) ^ 2 = a ^ 2 / 2 := by ring
  rw [hv] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  have hc : 1 - cos (a * z) = 2 * sin (a / 2 * z) ^ 2 := by
    have e : a * z = 2 * (a / 2 * z) := by ring
    rw [e, Complex.cos_two_mul]
    have h2 := Complex.sin_sq_add_cos_sq (a / 2 * z)
    linear_combination (-2) * h2
  rw [hc]
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.one_sub_cos_order_two
-- test: ca-res-one-sub-cos-order-test.sage

/-- `z e^{1/z} = ∑ z^(1-n)/n!` on `z ≠ 0`, with every coefficient non-zero: an essential singularity. -/
theorem mul_exp_inv (z : ℂ) :
    HasSum (fun n : ℕ => z * (z⁻¹ ^ n / (Nat.factorial n : ℂ))) (z * exp z⁻¹) ∧
      ∀ n : ℕ, (Nat.factorial n : ℂ)⁻¹ ≠ 0 :=
  ⟨(LaurentSeries.exp_inv_laurent z).1.mul_left z, (LaurentSeries.exp_inv_laurent z).2.2⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.mul_exp_inv
-- test: ca-res-mul-exp-inv-test.sage

/-! ## exr-singularity-order -/

/-- `1/(z² sin(az))`, `a ≠ 0`, has a pole of order `3` at `0` (the text: `a = 1`). -/
theorem order_csc_sq (a : ℂ) (ha : a ≠ 0) : IsPoleOfOrder (fun z => 1 / (z ^ 2 * sin (a * z))) 0 3 := by
  have hs : dslope (fun z => sin (a * z)) 0 0 = a := by
    rw [dslope_same]
    have := (((hasDerivAt_id (0 : ℂ)).const_mul a).csin).deriv
    simpa using this
  have hd : Differentiable ℂ (fun z => sin (a * z)) := by fun_prop
  refine ⟨fun z => 1 / dslope (fun z => sin (a * z)) 0 z,
    analyticAt_const.div (analyticAt_dslope _ hd 0) (by rw [hs]; exact ha),
    one_div_ne_zero (by rw [hs]; exact ha), ?_⟩
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  rw [dslope_of_ne _ hz, slope_def_field]
  simp only [mul_zero, Complex.sin_zero, sub_zero]
  by_cases h : sin (a * z) = 0
  · simp [h]
  · field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_csc_sq
-- test: ca-res-order-csc-sq-test.sage

/-- `sin(az)/z³ = a z⁻² - …`, `a ≠ 0`, has a pole of order `2` at `0`. -/
theorem order_sin_cube (a : ℂ) (ha : a ≠ 0) : IsPoleOfOrder (fun z => sin (a * z) / z ^ 3) 0 2 := by
  have hs : dslope (fun z => sin (a * z)) 0 0 = a := by
    rw [dslope_same]
    have := (((hasDerivAt_id (0 : ℂ)).const_mul a).csin).deriv
    simpa using this
  have hd : Differentiable ℂ (fun z => sin (a * z)) := by fun_prop
  refine ⟨dslope (fun z => sin (a * z)) 0, analyticAt_dslope _ hd 0, by rw [hs]; exact ha, ?_⟩
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  rw [dslope_of_ne _ hz, slope_def_field]
  simp only [mul_zero, Complex.sin_zero, sub_zero]
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_sin_cube
-- test: ca-res-order-sin-cube-test.sage

/-- `(e^{az} - 1)/z² = a z⁻¹ + …`, `a ≠ 0`, has a simple pole at `0`. -/
theorem order_exp_sub_one (a : ℂ) (ha : a ≠ 0) : IsPoleOfOrder (fun z => (exp (a * z) - 1) / z ^ 2) 0 1 := by
  have hs : dslope (fun z => exp (a * z)) 0 0 = a := by
    rw [dslope_same]
    have := (((hasDerivAt_id (0 : ℂ)).const_mul a).cexp).deriv
    simpa using this
  have hd : Differentiable ℂ (fun z => exp (a * z)) := by fun_prop
  refine ⟨dslope (fun z => exp (a * z)) 0, analyticAt_dslope _ hd 0, by rw [hs]; exact ha, ?_⟩
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  rw [dslope_of_ne _ hz, slope_def_field]
  simp only [mul_zero, Complex.exp_zero, sub_zero]
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_exp_sub_one
-- test: ca-res-order-exp-test.sage

/-- `(z² + b²)³ = (z - ib)³ (z + ib)³`. -/
theorem sq_add_sq_cube (z b : ℂ) : (z ^ 2 + b ^ 2) ^ 3 = (z - I * b) ^ 3 * (z + I * b) ^ 3 := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have h : z ^ 2 + b ^ 2 = (z - I * b) * (z + I * b) := by linear_combination b ^ 2 * hI
  rw [h, mul_pow]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sq_add_sq_cube
-- test: ca-res-sq-add-sq-cube-test.sage

/-- `1/(z² + b²)³`, `b ≠ 0`, has a pole of order `3` at `ib` (the text: `b = 1`, the pole at `i`). -/
theorem order_cube_at_I (b : ℂ) (hb : b ≠ 0) :
    IsPoleOfOrder (fun z => 1 / (z ^ 2 + b ^ 2) ^ 3) (I * b) 3 := by
  have h2 : I * b + I * b ≠ 0 := by
    rw [← two_mul]; exact mul_ne_zero two_ne_zero (mul_ne_zero I_ne_zero hb)
  refine ⟨fun z => 1 / (z + I * b) ^ 3, analyticAt_const.div (by fun_prop) (pow_ne_zero _ h2),
    by simp [h2], Eventually.of_forall fun z => ?_⟩
  show 1 / (z ^ 2 + b ^ 2) ^ 3 = 1 / (z + I * b) ^ 3 / (z - I * b) ^ 3
  rw [sq_add_sq_cube, div_div, mul_comm]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.order_cube_at_I
-- test: ca-res-order-cube-test.sage

/-! ## exr-singularity-casorati -/

/-- `e^{1/z} = w` has infinitely many solutions in every punctured disc `0 < |z| < ε`, for every `w ≠ 0`. -/
theorem casorati (w : ℂ) (hw : w ≠ 0) (ε : ℝ) (hε : 0 < ε) :
    Set.Infinite {z : ℂ | 0 < ‖z‖ ∧ ‖z‖ < ε ∧ exp z⁻¹ = w} := by
  obtain ⟨K, hK⟩ := exists_nat_gt (1 / ε / (2 * π) + 1)
  set u : ℕ → ℂ := fun n => log w + ((n + K : ℕ) : ℂ) * (2 * π * I) with hu
  have hpi := Real.pi_pos
  have him : ∀ n : ℕ, (u n).im = arg w + (n + K : ℕ) * (2 * π) := by
    intro n; simp [hu, Complex.log_im]
  have hlarge : ∀ n : ℕ, 1 / ε < ‖u n‖ := by
    intro n
    have h1 : |(u n).im| ≤ ‖u n‖ := Complex.abs_im_le_norm _
    have h2 : -π < arg w := Complex.neg_pi_lt_arg w
    have h3 : ((K : ℕ) : ℝ) ≤ ((n + K : ℕ) : ℝ) := by exact_mod_cast Nat.le_add_left K n
    have h4 : 1 / ε / (2 * π) + 1 < (K : ℝ) := hK
    have h5 : 1 / ε < ((K : ℝ) - 1) * (2 * π) := by
      have := (div_lt_iff₀ (by positivity : (0 : ℝ) < 2 * π)).1 (by linarith : 1 / ε / (2 * π) < (K : ℝ) - 1)
      linarith
    rw [him] at h1
    have h6 : 1 / ε < arg w + ((n + K : ℕ) : ℝ) * (2 * π) := by nlinarith
    exact lt_of_lt_of_le h6 ((le_abs_self _).trans h1)
  have hu0 : ∀ n, u n ≠ 0 := by
    intro n h; have := hlarge n; rw [h, norm_zero] at this; exact absurd this (not_lt.2 (by positivity))
  refine Set.infinite_of_injective_forall_mem (f := fun n => (u n)⁻¹) ?_ ?_
  · intro m n hmn
    have h0 : u m = u n := inv_injective hmn
    have h := congrArg Complex.im h0
    rw [him, him] at h
    have hpi : (2 * π) ≠ 0 := by positivity
    have h1 : ((m + K : ℕ) : ℝ) = ((n + K : ℕ) : ℝ) :=
      mul_right_cancel₀ hpi (by linarith)
    have h2 : m + K = n + K := by exact_mod_cast h1
    omega
  · intro n
    refine ⟨norm_pos_iff.2 (inv_ne_zero (hu0 n)), ?_, ?_⟩
    · rw [norm_inv]
      have := hlarge n
      rw [inv_lt_comm₀ (norm_pos_iff.2 (hu0 n)) hε, ← one_div]; exact this
    · show exp ((u n)⁻¹)⁻¹ = w
      simp only [inv_inv, hu]
      rw [Complex.exp_add, Complex.exp_log hw, exp_nat_mul_two_pi_mul_I, mul_one]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.casorati
-- test: ca-res-casorati-test.sage

/-- `e^{1/z}` never takes the value `0`: `0` is the one exceptional value. -/
theorem exp_inv_ne_zero (z : ℂ) : exp z⁻¹ ≠ 0 := Complex.exp_ne_zero _

section
open scoped ComplexOrder
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_ne_zero (0) := by
  exact Complex.exp_ne_zero _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_ne_zero (1) := by
  exact Complex.exp_ne_zero _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_ne_zero (Complex.I) := by
  exact Complex.exp_ne_zero _

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_ne_zero
-- test: ca-res-exp-inv-ne-zero-test.sage
end

end KaluzaKlein.ComplexAnalysis.IsolatedSingularities

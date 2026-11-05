import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Laurent series"

The claims of the section "Laurent series" of `tutorial/part-i-04-complex-analysis.qmd`: the orthogonality
`∮ (z - c)ⁿ dz = 2πi δₙ,₋₁` on circles, the coefficient formula `aₘ = (2πi)⁻¹ ∮ f (z - c)^(-m-1) dz` and hence the
uniqueness of the coefficients of an absolutely convergent Laurent series on a circle, the independence of the
coefficient integrals of the radius inside an annulus of holomorphy, the three expansions of `1/((z - 1)(z - 2))`
with their partial-sum errors, the Laurent series of `e^{1/z}` and `z² e^{1/z}`, and the first terms of `1/sin z`.

The existence of the Laurent expansion of a function holomorphic on an arbitrary annulus is not in mathlib; the
claims are stated for the explicit expansions and for series given on a circle.
-/

namespace KaluzaKlein.ComplexAnalysis.LaurentSeries

open Complex Filter Topology Metric
open scoped Real Nat

/-! ## def-laurent-series, exr-laurent-uniqueness -/

/-- Orthogonality on circles: `∮ (z - c)ⁿ dz = 2πi` for `n = -1` and `0` for every other integer `n`. -/
theorem circle_zpow (c : ℂ) (ρ : ℝ) (hρ : ρ ≠ 0) (n : ℤ) :
    (∮ z in C(c, ρ), (z - c) ^ n) = if n = -1 then 2 * π * I else 0 := by
  by_cases h : n = -1
  · subst h
    simp only [zpow_neg_one, if_true]
    exact circleIntegral.integral_sub_center_inv c hρ
  · rw [if_neg h]
    exact circleIntegral.integral_sub_zpow_of_ne h c c ρ

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.LaurentSeries.circle_zpow
-- test: ca-res-circle-zpow-test.sage

lemma circleMap_sub_ne (c : ℂ) {ρ : ℝ} (hρ : 0 < ρ) (θ : ℝ) : circleMap c ρ θ - c ≠ 0 := by
  rw [circleMap_sub_center]; exact circleMap_ne_center hρ.ne'

lemma norm_circleMap_sub (c : ℂ) {ρ : ℝ} (hρ : 0 < ρ) (θ : ℝ) : ‖circleMap c ρ θ - c‖ = ρ := by
  rw [circleMap_sub_center, norm_circleMap_zero, abs_of_pos hρ]

/-- Termwise integration on a circle: if `f = ∑ᵢ bᵢ (z - c)^(eᵢ)` with `∑ |bᵢ| ρ^(eᵢ) < ∞`, then
`(2πi)⁻¹ ∮ (z - c)^(-m-1) f(z) dz` is the sum of the `bᵢ` with `eᵢ = m`. -/
theorem laurent_coeff_gen {ι : Type*} [Countable ι] (c : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (e : ι → ℤ) (b : ι → ℂ)
    (hb : Summable (fun i => ‖b i‖ * ρ ^ e i)) (m : ℤ) :
    (2 * π * I)⁻¹ * (∮ z in C(c, ρ), (z - c) ^ (-m - 1) * ∑' i, b i * (z - c) ^ e i) =
      ∑' i, if e i = m then b i else 0 := by
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  set F : ι → ℝ → ℂ := fun i θ => deriv (circleMap c ρ) θ • (b i * (circleMap c ρ θ - c) ^ (e i - m - 1))
    with hF
  have hterm : ∀ i, (∫ θ in (0 : ℝ)..2 * π, F i θ) = 2 * π * I * (if e i = m then b i else 0) := by
    intro i
    have h1 : (∫ θ in (0 : ℝ)..2 * π, F i θ) = ∮ z in C(c, ρ), b i * (z - c) ^ (e i - m - 1) := rfl
    rw [h1, circleIntegral.integral_const_mul, circle_zpow c ρ hρ.ne' (e i - m - 1)]
    by_cases h : e i = m
    · have h' : e i - m - 1 = -1 := by omega
      simp [h, h']; ring
    · have h' : e i - m - 1 ≠ -1 := by omega
      simp [h, h']
  have hbound : ∀ i θ, ‖F i θ‖ = ρ * (‖b i‖ * ρ ^ (e i - m - 1)) := by
    intro i θ
    simp only [hF, deriv_circleMap, norm_smul, norm_mul, norm_zpow, norm_circleMap_sub c hρ,
      norm_circleMap_zero, abs_of_pos hρ, Complex.norm_I, mul_one]
  have hsumm : Summable (fun i => ρ * (‖b i‖ * ρ ^ (e i - m - 1))) := by
    have : (fun i => ρ * (‖b i‖ * ρ ^ (e i - m - 1))) = fun i => (ρ * ρ ^ (-m - 1)) * (‖b i‖ * ρ ^ e i) := by
      funext i
      rw [show e i - m - 1 = e i + (-m - 1) by ring, zpow_add₀ hρ.ne']
      ring
    rw [this]; exact hb.mul_left _
  have hcont : ∀ i, Continuous (F i) := by
    intro i
    simp only [hF, deriv_circleMap]
    refine Continuous.smul ((continuous_circleMap 0 ρ).mul continuous_const) (Continuous.mul continuous_const ?_)
    exact (continuous_circleMap c ρ |>.sub continuous_const).zpow₀ _ fun θ => Or.inl (circleMap_sub_ne c hρ θ)
  have hsum : HasSum (fun i => ∫ θ in (0 : ℝ)..2 * π, F i θ) (∫ θ in (0 : ℝ)..2 * π, ∑' i, F i θ) := by
    refine intervalIntegral.hasSum_integral_of_dominated_convergence
      (fun i _ => ρ * (‖b i‖ * ρ ^ (e i - m - 1))) (fun i => (hcont i).aestronglyMeasurable)
      (fun i => Eventually.of_forall fun θ _ => (hbound i θ).le)
      (Eventually.of_forall fun θ _ => hsumm) intervalIntegrable_const
      (Eventually.of_forall fun θ _ => ?_)
    exact (Summable.of_norm_bounded hsumm fun i => (hbound i θ).le).hasSum
  have hint : (∮ z in C(c, ρ), (z - c) ^ (-m - 1) * ∑' i, b i * (z - c) ^ e i) =
      ∫ θ in (0 : ℝ)..2 * π, ∑' i, F i θ := by
    simp only [circleIntegral]
    refine intervalIntegral.integral_congr fun θ _ => ?_
    simp only [hF, smul_eq_mul]
    rw [← tsum_mul_left, ← tsum_mul_left]
    refine tsum_congr fun i => ?_
    rw [show e i - m - 1 = (-m - 1) + e i by ring, zpow_add₀ (circleMap_sub_ne c hρ θ)]
    ring
  have hsumm2 : Summable (fun i => if e i = m then b i else 0) := by
    have := (hsum.summable).mul_left (2 * π * I)⁻¹
    simp only [hterm, ← mul_assoc, inv_mul_cancel₀ hne, one_mul] at this
    exact this
  have hval : HasSum (fun i => ∫ θ in (0 : ℝ)..2 * π, F i θ)
      (2 * π * I * ∑' i, if e i = m then b i else 0) := by
    simp only [hterm]
    exact hsumm2.hasSum.mul_left _
  rw [hint, hsum.unique hval, ← mul_assoc, inv_mul_cancel₀ hne, one_mul]

/-- The coefficient formula, hence uniqueness: if `f = ∑ bₙ (z - c)ⁿ` converges absolutely on the circle
`|z - c| = ρ`, then `(2πi)⁻¹ ∮ (z - c)^(-m-1) f(z) dz = bₘ`. -/
theorem laurent_coeff (c : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (b : ℤ → ℂ) (hb : Summable (fun n => ‖b n‖ * ρ ^ n))
    (m : ℤ) :
    (2 * π * I)⁻¹ * (∮ z in C(c, ρ), (z - c) ^ (-m - 1) * ∑' n, b n * (z - c) ^ n) = b m := by
  have h := laurent_coeff_gen c ρ hρ id b hb m
  simp only [id] at h
  rw [h]
  exact tsum_ite_eq m b

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.LaurentSeries.laurent_coeff
-- test: ca-res-laurent-coeff-test.sage
end

/-- The coefficient integrals do not depend on the radius `ρ` of the circle inside an annulus of holomorphy. -/
theorem coeff_indep_radius (f : ℂ → ℂ) (c : ℂ) (r R : ℝ) (hr : 0 < r) (hrR : r ≤ R)
    (hc : ContinuousOn f (closedBall c R \ ball c r))
    (hd : ∀ z ∈ ball c R \ closedBall c r, DifferentiableAt ℂ f z) (n : ℤ) :
    (∮ z in C(c, R), f z * (z - c) ^ (-n - 1)) = ∮ z in C(c, r), f z * (z - c) ^ (-n - 1) := by
  have hz : ∀ z ∈ closedBall c R \ ball c r, z - c ≠ 0 := by
    intro z hz h0
    have : z = c := by linear_combination h0
    rw [this] at hz
    exact hz.2 (mem_ball_self hr)
  refine circleIntegral_eq_of_differentiable_on_annulus_off_countable hr hrR Set.countable_empty ?_ ?_
  · exact hc.mul ((continuousOn_id.sub continuousOn_const).zpow₀ _ fun z hz' => Or.inl (hz z hz'))
  · intro z hz'
    have hz1 : z ∈ ball c R \ closedBall c r := hz'.1
    have hzc : z - c ≠ 0 := by
      intro h0
      have : z = c := by linear_combination h0
      rw [this] at hz1
      exact hz1.2 (mem_closedBall_self hr.le)
    exact (hd z hz1).mul ((differentiableAt_id.sub_const c).zpow (Or.inl hzc))

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.coeff_indep_radius
-- test: ca-res-coeff-indep-test.sage

/-! ## exm-laurent-three-regions -/

/-- Partial fractions: `1/((z - 1)(z - 2)) = 1/(z - 2) - 1/(z - 1)`. -/
theorem partial_fractions (z : ℂ) (h1 : z - 1 ≠ 0) (h2 : z - 2 ≠ 0) :
    1 / ((z - 1) * (z - 2)) = 1 / (z - 2) - 1 / (z - 1) := by
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.partial_fractions
-- test: ca-res-partial-fractions-test.sage
end

lemma f_eq (z : ℂ) (h1 : z - 1 ≠ 0) (h2 : z - 2 ≠ 0) :
    1 / ((z - 1) * (z - 2)) = 1 / (1 - z) - 1 / (2 - z) := by
  have h1' : 1 - z ≠ 0 := by intro h; exact h1 (by linear_combination -h)
  have h2' : 2 - z ≠ 0 := by intro h; exact h2 (by linear_combination -h)
  field_simp
  ring

lemma ne_of_norm_lt {z : ℂ} {a : ℝ} (h : ‖z‖ < a) (ha : a = ‖(a : ℂ)‖) : z - a ≠ 0 := by
  intro h0
  have : z = a := by linear_combination h0
  rw [this, ← ha] at h; exact lt_irrefl _ h

lemma ne_of_lt_norm {z : ℂ} {a : ℝ} (h : a < ‖z‖) (ha : a = ‖(a : ℂ)‖) : z - a ≠ 0 := by
  intro h0
  have : z = a := by linear_combination h0
  rw [this, ← ha] at h; exact lt_irrefl _ h

lemma inv_ne_one_sub {z : ℂ} (h1 : z - 1 ≠ 0) : 1 - z⁻¹ ≠ 0 := by
  intro h; apply h1; rw [inv_eq_one.1 (by linear_combination -h : z⁻¹ = 1)]; ring

lemma inv_ne_two_sub {z : ℂ} (h2 : z - 2 ≠ 0) : 1 - 2 * z⁻¹ ≠ 0 := by
  intro h; apply h2
  have h' : z⁻¹ = 2⁻¹ := by linear_combination -h / 2
  rw [inv_injective h']; ring

lemma half_inv (z : ℂ) : (1 - z / 2)⁻¹ / 2 = (2 - z)⁻¹ := by
  rw [div_eq_mul_inv ((1 - z / 2)⁻¹) 2, ← mul_inv, show (1 - z / 2) * 2 = 2 - z by ring]

lemma inv_mul_geom {z : ℂ} (hz : z ≠ 0) : z⁻¹ * (1 - z⁻¹)⁻¹ = (z - 1)⁻¹ := by
  rw [← mul_inv, mul_sub, mul_one, mul_inv_cancel₀ hz]

lemma sub_one_ne {z : ℂ} (h : ‖z‖ ≠ 1) : z - 1 ≠ 0 := by
  intro h0; apply h; rw [show z = 1 by linear_combination h0]; simp

lemma sub_two_ne {z : ℂ} (h : ‖z‖ ≠ 2) : z - 2 ≠ 0 := by
  intro h0; apply h; rw [show z = 2 by linear_combination h0]; norm_num

/-- On `|z| < 1`: `1/((z - 1)(z - 2)) = ∑ (1 - 2^(-n-1)) zⁿ`, a Taylor series. -/
theorem taylor_region (z : ℂ) (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ => (1 - (1 / 2 : ℂ) ^ (n + 1)) * z ^ n) (1 / ((z - 1) * (z - 2))) := by
  have hA := hasSum_geometric_of_norm_lt_one hz
  have hq : ‖z / 2‖ < 1 := by rw [norm_div]; norm_num; linarith
  have hB := (hasSum_geometric_of_norm_lt_one hq).div_const 2
  have h1 : z - 1 ≠ 0 := sub_one_ne (by linarith)
  have h2 : z - 2 ≠ 0 := sub_two_ne (by linarith)
  rw [f_eq z h1 h2]
  convert hA.sub hB using 1
  · funext n; rw [div_pow z 2 n, one_div_pow]; ring
  · rw [half_inv, one_div, one_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_region
-- test: ca-res-taylor-region-test.sage

/-- On `1 < |z| < 2`: `1/((z - 1)(z - 2)) = -∑_{n ≥ 0} zⁿ/2ⁿ⁺¹ - ∑_{n ≥ 1} z⁻ⁿ`. -/
theorem annulus_region (z : ℂ) (hz1 : 1 < ‖z‖) (hz2 : ‖z‖ < 2) :
    HasSum (fun n : ℕ => -(z ^ n / 2 ^ (n + 1)) - z⁻¹ ^ (n + 1)) (1 / ((z - 1) * (z - 2))) := by
  have hz0 : z ≠ 0 := by intro h; rw [h, norm_zero] at hz1; linarith
  have hq : ‖z / 2‖ < 1 := by rw [norm_div]; norm_num; linarith
  have hA := (hasSum_geometric_of_norm_lt_one hq).div_const 2
  have hp : ‖z⁻¹‖ < 1 := by rw [norm_inv]; exact inv_lt_one_of_one_lt₀ hz1
  have hB := (hasSum_geometric_of_norm_lt_one hp).mul_left z⁻¹
  have h1 : z - 1 ≠ 0 := sub_one_ne (by linarith)
  have h2 : z - 2 ≠ 0 := sub_two_ne (by linarith)
  convert hA.neg.sub hB using 1
  · funext n; rw [div_pow z 2 n, pow_succ']; ring
  · rw [half_inv, inv_mul_geom hz0, partial_fractions z h1 h2]
    have h2' : 2 - z ≠ 0 := by intro h; exact h2 (by linear_combination -h)
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.annulus_region
-- test: ca-res-annulus-region-test.sage

/-- On `|z| > 2`: `1/((z - 1)(z - 2)) = ∑_{n ≥ 2} (2ⁿ⁻¹ - 1) z⁻ⁿ`. -/
theorem outer_region (z : ℂ) (hz : 2 < ‖z‖) :
    HasSum (fun n : ℕ => (2 ^ (n + 1) - 1) * z⁻¹ ^ (n + 2)) (1 / ((z - 1) * (z - 2))) := by
  have hz0 : z ≠ 0 := by intro h; rw [h, norm_zero] at hz; linarith
  have hq : ‖2 * z⁻¹‖ < 1 := by
    rw [norm_mul, norm_inv]; norm_num
    rw [← div_eq_mul_inv, div_lt_one (by linarith)]; exact hz
  have hp : ‖z⁻¹‖ < 1 := by rw [norm_inv]; exact inv_lt_one_of_one_lt₀ (by linarith)
  have hA := (hasSum_geometric_of_norm_lt_one hq).mul_left (2 * z⁻¹ ^ 2)
  have hB := (hasSum_geometric_of_norm_lt_one hp).mul_left (z⁻¹ ^ 2)
  have h1 : z - 1 ≠ 0 := sub_one_ne (by linarith)
  have h2 : z - 2 ≠ 0 := sub_two_ne (by linarith)
  convert hA.sub hB using 1
  · funext n; rw [mul_pow]; ring
  · have h3 := inv_ne_two_sub h2
    have h4 := inv_ne_one_sub h1
    rw [partial_fractions z h1 h2]
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.outer_region
-- test: ca-res-outer-region-test.sage

/-! ## exm-laurent-essential, exr-laurent-exp-1-over-z -/

/-- `e^{1/z} = ∑_{n ≥ 0} z⁻ⁿ/n!` on `0 < |z|`: the coefficient of `z⁻¹` is `1`, and every coefficient of the
principal part is non-zero (the principal part is infinite). -/
theorem exp_inv_laurent (z : ℂ) :
    HasSum (fun n : ℕ => z⁻¹ ^ n / (Nat.factorial n : ℂ)) (exp z⁻¹) ∧ z⁻¹ ^ 1 / (Nat.factorial 1 : ℂ) = z⁻¹ ∧
      ∀ n : ℕ, (Nat.factorial n : ℂ)⁻¹ ≠ 0 := by
  refine ⟨?_, by simp, fun n => inv_ne_zero (by exact_mod_cast (Nat.factorial_ne_zero n))⟩
  rw [Complex.exp_eq_exp_ℂ]
  exact NormedSpace.expSeries_div_hasSum_exp z⁻¹

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.exp_inv_laurent
-- test: ca-res-exp-inv-laurent-test.sage

/-- `z² e^{1/z} = z² + z + 1/2 + 1/(6z) + …`: the coefficient of `1/z` is `1/6 = z² · z⁻³/3!`. -/
theorem sq_mul_exp_inv (z : ℂ) (hz : z ≠ 0) :
    HasSum (fun n : ℕ => z ^ 2 * (z⁻¹ ^ n / (Nat.factorial n : ℂ))) (z ^ 2 * exp z⁻¹) ∧
      z ^ 2 * (z⁻¹ ^ 0 / (Nat.factorial 0 : ℂ)) = z ^ 2 ∧ z ^ 2 * (z⁻¹ ^ 1 / (Nat.factorial 1 : ℂ)) = z ∧
      z ^ 2 * (z⁻¹ ^ 2 / (Nat.factorial 2 : ℂ)) = 1 / 2 ∧
      z ^ 2 * (z⁻¹ ^ 3 / (Nat.factorial 3 : ℂ)) = 1 / 6 * z⁻¹ := by
  refine ⟨(exp_inv_laurent z).1.mul_left _, by simp, ?_, ?_, ?_⟩
  · norm_num [Nat.factorial] <;> field_simp
  · norm_num [Nat.factorial] <;> field_simp
  · norm_num [Nat.factorial] <;> field_simp <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.LaurentSeries.sq_mul_exp_inv
-- test: ca-res-sq-exp-inv-test.sage

/-! ## exm-laurent-cosecant -/

/-- Matching coefficients in `sin z · (1/z + z/6 + 7z³/360) = 1 + O(z⁶)`. -/
theorem cosecant_coeffs (z : ℂ) :
    (1 - z ^ 2 / 6 + z ^ 4 / 120) * (1 + z ^ 2 / 6 + 7 * z ^ 4 / 360) = 1 - z ^ 6 / 540 + 7 * z ^ 8 / 43200 := by
  veriphy_certified
    (calc
        (1 - z ^ 2 / 6 + z ^ 4 / 120) * (1 + z ^ 2 / 6 + 7 * z ^ 4 / 360)
        _ = 1 - z ^ 6 / 540 + 7 * z ^ 8 / 43200 := by ring1)
  -- certificate: ca-res-ring.sage

/-- `sin z ≠ 0` on `0 < |z| < π`: the punctured disc on which the expansion of `1/sin z` holds. -/
theorem sin_ne_zero_of_lt_pi (z : ℂ) (h0 : z ≠ 0) (hπ : ‖z‖ < π) : sin z ≠ 0 := by
  intro hs
  obtain ⟨k, hk⟩ := Complex.sin_eq_zero_iff.1 hs
  have hk0 : k ≠ 0 := by rintro rfl; simp at hk; exact h0 hk
  have : ‖z‖ = |(k : ℝ)| * π := by
    rw [hk, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
    simp
  have hk1 : (1 : ℝ) ≤ |(k : ℝ)| := by
    rw [← Int.cast_abs]; exact_mod_cast Int.one_le_abs hk0
  nlinarith [Real.pi_pos]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.LaurentSeries.sin_ne_zero_of_lt_pi
-- test: ca-res-sin-ne-zero-test.sage

/-- The quotient rule for a limit: if `h(z₀) = 0` and `h'(z₀) = d ≠ 0`, then `(z - z₀) g(z)/h(z) → g(z₀)/d`. -/
theorem tendsto_mul_div_of_hasDerivAt (g h : ℂ → ℂ) (z₀ d : ℂ) (hh : h z₀ = 0) (hd : HasDerivAt h d z₀)
    (hd0 : d ≠ 0) (hg : ContinuousAt g z₀) :
    Tendsto (fun z => (z - z₀) * (g z / h z)) (𝓝[≠] z₀) (𝓝 (g z₀ / d)) := by
  have hs := hasDerivAt_iff_tendsto_slope.1 hd
  have hg' : Tendsto g (𝓝[≠] z₀) (𝓝 (g z₀)) := hg.tendsto.mono_left nhdsWithin_le_nhds
  refine (hg'.div hs hd0).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with z hz
  simp only [Pi.div_apply]
  rw [slope_def_field, hh, sub_zero, div_div_eq_mul_div]
  ring

/-- `1/sin z` has a simple pole at each `kπ` with residue `(-1)ᵏ`; for `k = 0` the residue is `1`. -/
theorem csc_residue (k : ℤ) :
    Tendsto (fun z => (z - k * π) * (1 / sin z)) (𝓝[≠] ((k : ℂ) * π)) (𝓝 ((-1) ^ k)) := by
  have hh : sin ((k : ℂ) * π) = 0 := by
    rw [Complex.sin_eq_zero_iff]; exact ⟨k, rfl⟩
  have hcos : cos ((k : ℂ) * π) = (-1) ^ k := by
    have e : ((k : ℂ) * π) = ((k * π : ℝ) : ℂ) := by push_cast; ring
    rw [e, ← Complex.ofReal_cos, Real.cos_int_mul_pi]; push_cast; rfl
  have hd0 : cos ((k : ℂ) * π) ≠ 0 := by rw [hcos]; exact zpow_ne_zero _ (by norm_num)
  have := tendsto_mul_div_of_hasDerivAt (fun _ => 1) sin _ _ hh (Complex.hasDerivAt_sin _) hd0
    continuousAt_const
  rw [hcos] at this
  have hinv : (1 : ℂ) / (-1) ^ k = (-1) ^ k := by
    rw [one_div, ← inv_zpow, inv_neg, inv_one]
  rw [hinv] at this
  exact this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.csc_residue
-- test: ca-res-csc-residue-test.sage

/-! ## exr-laurent-three-regions -/

/-- The telescoping sum `∑_{n < N} (qⁿ - qⁿ⁺¹) = 1 - q^N`. -/
theorem geom_telescope (q : ℂ) (N : ℕ) : ∑ n ∈ Finset.range N, (q ^ n - q ^ (n + 1)) = 1 - q ^ N := by
  veriphy_certified
    (induction N with
      | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
      | succ N ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1)
  -- certificate: ca-res-sum.sage

/-- Finite geometric sums: `(1 - q) ∑_{n < N} qⁿ = 1 - q^N`. -/
theorem geom_partial_sum (q : ℂ) (N : ℕ) : (1 - q) * ∑ n ∈ Finset.range N, q ^ n = 1 - q ^ N := by
  rw [Finset.mul_sum, ← geom_telescope]
  exact Finset.sum_congr rfl fun n _ => by ring

lemma geom_sum_eq (q : ℂ) (hq : 1 - q ≠ 0) (N : ℕ) : ∑ n ∈ Finset.range N, q ^ n = (1 - q ^ N) / (1 - q) := by
  rw [eq_div_iff hq, mul_comm, geom_partial_sum]

/-- The error of the Taylor partial sums: `f - S_N = z^N/(1 - z) - (z/2)^N/(2 - z)`, geometric on `|z| < 1`. -/
theorem taylor_error (z : ℂ) (N : ℕ) (h1 : z - 1 ≠ 0) (h2 : z - 2 ≠ 0) :
    1 / ((z - 1) * (z - 2)) - ∑ n ∈ Finset.range N, (1 - (1 / 2 : ℂ) ^ (n + 1)) * z ^ n =
      z ^ N / (1 - z) - (z / 2) ^ N / (2 - z) := by
  have h1' : 1 - z ≠ 0 := by intro h; exact h1 (by linear_combination -h)
  have h2' : 2 - z ≠ 0 := by intro h; exact h2 (by linear_combination -h)
  have h3 : 1 - z / 2 ≠ 0 := by intro h; exact h2' (by linear_combination 2 * h)
  have hs : ∑ n ∈ Finset.range N, (1 - (1 / 2 : ℂ) ^ (n + 1)) * z ^ n =
      ∑ n ∈ Finset.range N, z ^ n - (1 / 2) * ∑ n ∈ Finset.range N, (z / 2) ^ n := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    simp only [div_pow, one_div, inv_pow]; ring
  rw [hs, geom_sum_eq z h1', geom_sum_eq (z / 2) h3, f_eq z h1 h2]
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_error
-- test: ca-res-taylor-error-test.sage
end

/-- The error of the annulus partial sums: `f - S_N = -(z/2)^N/(2 - z) - z^(-N)/(z - 1)`, written with
`+ z^(-N)/(1 - z)` for the last term. -/
theorem annulus_error (z : ℂ) (N : ℕ) (h0 : z ≠ 0) (h1 : z - 1 ≠ 0) (h2 : z - 2 ≠ 0) :
    1 / ((z - 1) * (z - 2)) - ∑ n ∈ Finset.range N, (-(z ^ n * (1 / 2) ^ (n + 1)) - z⁻¹ ^ (n + 1)) =
      -(z / 2) ^ N / (2 - z) + z⁻¹ ^ N / (1 - z) := by
  have h2' : 2 - z ≠ 0 := by intro h; exact h2 (by linear_combination -h)
  have h3 : 1 - z / 2 ≠ 0 := by intro h; exact h2' (by linear_combination 2 * h)
  have h4 := inv_ne_one_sub h1
  have hs : ∑ n ∈ Finset.range N, (-(z ^ n * (1 / 2) ^ (n + 1)) - z⁻¹ ^ (n + 1)) =
      -(1 / 2) * ∑ n ∈ Finset.range N, (z / 2) ^ n - z⁻¹ * ∑ n ∈ Finset.range N, z⁻¹ ^ n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    simp only [div_pow, one_div, inv_pow, pow_succ]; ring
  rw [hs, geom_sum_eq _ h3, geom_sum_eq _ h4]
  have h1' : z - 1 ≠ 0 := h1
  have h1'' : 1 - z ≠ 0 := by intro h; exact h1 (by linear_combination -h)
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.annulus_error
-- test: ca-res-annulus-error-test.sage
end

/-- The error of the outer partial sums: `f - S_N = 2 (2/z)^N/(z(z - 2)) - z^(-N)/(z(z - 1))`. -/
theorem outer_error (z : ℂ) (N : ℕ) (h0 : z ≠ 0) (h1 : z - 1 ≠ 0) (h2 : z - 2 ≠ 0) :
    1 / ((z - 1) * (z - 2)) - ∑ n ∈ Finset.range N, (2 ^ (n + 1) - 1) * z⁻¹ ^ (n + 2) =
      2 * (2 * z⁻¹) ^ N / (z * (z - 2)) - z⁻¹ ^ N / (z * (z - 1)) := by
  have h3 := inv_ne_two_sub h2
  have h4 := inv_ne_one_sub h1
  have hs : ∑ n ∈ Finset.range N, (2 ^ (n + 1) - 1) * z⁻¹ ^ (n + 2) =
      2 * z⁻¹ ^ 2 * ∑ n ∈ Finset.range N, (2 * z⁻¹) ^ n - z⁻¹ ^ 2 * ∑ n ∈ Finset.range N, z⁻¹ ^ n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [mul_pow]; ring
  rw [hs, geom_sum_eq _ h3, geom_sum_eq _ h4]
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.outer_error
-- test: ca-res-outer-error-test.sage
end

/-- Used outside its annulus (`|z| ≥ 1`), the Taylor series diverges: its terms do not tend to `0`. -/
theorem taylor_diverges (z : ℂ) (hz : 1 ≤ ‖z‖) :
    ¬ Summable (fun n : ℕ => (1 - (1 / 2 : ℂ) ^ (n + 1)) * z ^ n) := by
  intro hs
  have ht := hs.tendsto_atTop_zero.norm
  rw [norm_zero] at ht
  obtain ⟨n, hn⟩ := (ht.eventually (gt_mem_nhds (show (0 : ℝ) < 1 / 2 by norm_num))).exists
  have hp : ‖(1 / 2 : ℂ) ^ (n + 1)‖ ≤ 1 / 2 := by
    rw [norm_pow, show ‖(1 / 2 : ℂ)‖ = 1 / 2 by norm_num]
    calc (1 / 2 : ℝ) ^ (n + 1) ≤ (1 / 2) ^ 1 :=
          pow_le_pow_of_le_one (by norm_num) (by norm_num) (by omega)
      _ = 1 / 2 := by norm_num
  have h1 : (1 / 2 : ℝ) ≤ ‖1 - (1 / 2 : ℂ) ^ (n + 1)‖ := by
    have h := norm_sub_norm_le (1 : ℂ) ((1 / 2 : ℂ) ^ (n + 1))
    rw [norm_one] at h
    linarith
  have h2 : (1 : ℝ) ≤ ‖z ^ n‖ := by rw [norm_pow]; exact one_le_pow₀ hz
  have h3 := mul_le_mul h1 h2 zero_le_one (norm_nonneg _)
  rw [norm_mul] at hn
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.LaurentSeries.taylor_diverges
-- test: ca-res-taylor-diverges-test.sage

end KaluzaKlein.ComplexAnalysis.LaurentSeries

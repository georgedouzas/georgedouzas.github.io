import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.TaylorSeries
import KaluzaKlein.PartI.ComplexAnalysis.LaurentSeries
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Residues"

The claims of the section "Residues" of `tutorial/part-i-04-complex-analysis.qmd`. The residue of `f` at `c` is
formalised as the circle integral `(2πi)⁻¹ ∮_{|z - c| = ρ} f` over a small circle (`residue f c ρ`); it equals the
coefficient `a₋₁` of a Laurent series, the limit `lim (z - c) f(z)` at a simple pole (which also gives the quotient
rule `g(c)/h'(c)`), and `φ^(m)(c)/m!` at a pole of order `m + 1` with `f = φ/(z - c)^(m+1)`. The examples are stated
for families with a parameter, of which the text's functions are instances: `1/((z - a)(z - b))`, `1/(z² + c²)`,
`e^{az}/z²`, `e^{a/z}`, `(z² + 1)/(z (z - a)²)`, and `1/(1 + z⁴)` at any root `w⁴ = -1`.
-/

namespace KaluzaKlein.ComplexAnalysis.Residues

open Complex Filter Topology Metric
open scoped Real Nat

/-- The residue of `f` at `c`, as the circle integral `(2πi)⁻¹ ∮ f` over the circle `|z - c| = ρ`. -/
noncomputable def residue (f : ℂ → ℂ) (c : ℂ) (ρ : ℝ) : ℂ := (2 * π * I)⁻¹ * ∮ z in C(c, ρ), f z

/-- `∮ c/(z - w)^(k+1) dz = 2πi c` for `k = 0` and `0` for `k ≥ 1`, for `w` inside the circle. -/
theorem circle_pole_term (C w : ℂ) (ρ : ℝ) (hw : w ∈ ball C ρ) (c : ℂ) (k : ℕ) :
    (∮ z in C(C, ρ), c / (z - w) ^ (k + 1)) = if k = 0 then 2 * π * I * c else 0 := by
  have h : (fun z => c / (z - w) ^ (k + 1)) = fun z => c * (z - w) ^ (-((k : ℤ) + 1)) := by
    funext z; rw [zpow_neg, div_eq_mul_inv]; norm_cast
  rw [h, circleIntegral.integral_const_mul]
  by_cases hk : k = 0
  · subst hk
    simp only [Nat.cast_zero, zero_add, zpow_neg_one, ite_true]
    rw [circleIntegral.integral_sub_inv_of_mem_ball hw]; ring
  · rw [if_neg hk, circleIntegral.integral_sub_zpow_of_ne (by omega) C w ρ, mul_zero]

/-- Cauchy's theorem on a closed disc: `∮ g = 0` for `g` holomorphic on the closed disc. -/
theorem circle_holo_zero (g : ℂ → ℂ) (C : ℂ) (ρ : ℝ) (hρ : 0 ≤ ρ) (hg : DifferentiableOn ℂ g (closedBall C ρ)) :
    (∮ z in C(C, ρ), g z) = 0 :=
  ((hg.mono closure_ball_subset_closedBall).diffContOnCl).circleIntegral_eq_zero hρ

lemma circleIntegrable_pole_term (C w : ℂ) (ρ : ℝ) (hρ : 0 ≤ ρ) (hw : w ∈ ball C ρ) (c : ℂ) (k : ℕ) :
    CircleIntegrable (fun z => c / (z - w) ^ (k + 1)) C ρ := by
  refine ContinuousOn.circleIntegrable hρ ?_
  refine continuousOn_const.div (by fun_prop) fun z hz => pow_ne_zero _ (sub_ne_zero.2 ?_)
  rintro rfl
  rw [mem_sphere] at hz; rw [mem_ball] at hw; linarith

lemma circleIntegrable_of_differentiableOn (g : ℂ → ℂ) (C : ℂ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hg : DifferentiableOn ℂ g (closedBall C ρ)) : CircleIntegrable g C ρ :=
  ContinuousOn.circleIntegrable hρ (hg.continuousOn.mono sphere_subset_closedBall)

/-- Principal parts inside a circle: `∮ (∑ᵢ cᵢ/(z - wᵢ)^(kᵢ+1) + g) = 2πi ∑_{kᵢ = 0} cᵢ` for `g` holomorphic on the
closed disc (the residue theorem on circles, proved here and restated in the next section). -/
theorem circle_sum_poles {ι : Type*} (s : Finset ι) (C : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (w c : ι → ℂ) (k : ι → ℕ)
    (hw : ∀ i ∈ s, w i ∈ ball C ρ) (g : ℂ → ℂ) (hg : DifferentiableOn ℂ g (closedBall C ρ)) :
    (∮ z in C(C, ρ), ((∑ i ∈ s, c i / (z - w i) ^ (k i + 1)) + g z)) =
      2 * π * I * ∑ i ∈ s, if k i = 0 then c i else 0 := by
  have hint : ∀ i ∈ s, CircleIntegrable (fun z => c i / (z - w i) ^ (k i + 1)) C ρ :=
    fun i hi => circleIntegrable_pole_term C (w i) ρ hρ.le (hw i hi) (c i) (k i)
  have hsum : CircleIntegrable (fun z => ∑ i ∈ s, c i / (z - w i) ^ (k i + 1)) C ρ := by
    have : (fun z => ∑ i ∈ s, c i / (z - w i) ^ (k i + 1)) = ∑ i ∈ s, fun z => c i / (z - w i) ^ (k i + 1) := by
      funext z; simp
    rw [this]
    exact CircleIntegrable.sum s hint
  rw [circleIntegral.integral_add hsum (circleIntegrable_of_differentiableOn g C ρ hρ.le hg),
    circle_holo_zero g C ρ hρ.le hg, add_zero, circleIntegral.integral_fun_sum hint, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [circle_pole_term C (w i) ρ (hw i hi) (c i) (k i)]
  split_ifs <;> simp

lemma mem_sphere_ne {w z C : ℂ} {ρ : ℝ} (hz : z ∈ sphere C ρ) (hw : w ∈ ball C ρ) : z - w ≠ 0 := by
  intro h; have : z = w := by linear_combination h
  rw [this, mem_sphere] at hz; rw [mem_ball] at hw; linarith

lemma mem_sphere_ne' {w z C : ℂ} {ρ : ℝ} (hz : z ∈ sphere C ρ) (hw : w ∉ closedBall C ρ) : z - w ≠ 0 := by
  intro h; have : z = w := by linear_combination h
  rw [this] at hz; exact hw (sphere_subset_closedBall hz)

lemma ne_of_not_mem {w z C : ℂ} {ρ : ℝ} (hz : z ∈ closedBall C ρ) (hw : w ∉ closedBall C ρ) : z - w ≠ 0 := by
  intro h; have : z = w := by linear_combination h
  rw [this] at hz; exact hw hz

lemma not_mem_closedBall_of_lt {a b : ℂ} {ρ : ℝ} (h : ρ < ‖a - b‖) : b ∉ closedBall a ρ := by
  intro hb
  rw [mem_closedBall, Complex.dist_eq] at hb
  rw [norm_sub_rev] at h; linarith

/-! ## def-residue -/

/-- The residue is the coefficient `a₋₁` of a Laurent series converging absolutely on the circle. -/
theorem residue_eq_coeff (c : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (b : ℤ → ℂ) (hb : Summable (fun n => ‖b n‖ * ρ ^ n)) :
    residue (fun z => ∑' n, b n * (z - c) ^ n) c ρ = b (-1) := by
  have := LaurentSeries.laurent_coeff c ρ hρ b hb (-1)
  simp only [neg_neg, sub_self, zpow_zero, one_mul] at this
  exact this

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.residue_eq_coeff
-- test: ca-res-residue-coeff-test.sage
end

/-- At a simple pole, `f = r/(z - z₀) + g` near `z₀` with `g` holomorphic: the residue is `r = lim (z - z₀) f(z)`. -/
theorem simple_pole (f g : ℂ → ℂ) (z₀ r : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (hg : DifferentiableOn ℂ g (closedBall z₀ ρ))
    (hf : ∀ z ∈ closedBall z₀ ρ, z ≠ z₀ → f z = r / (z - z₀) + g z) :
    residue f z₀ ρ = r ∧ Tendsto (fun z => (z - z₀) * f z) (𝓝[≠] z₀) (𝓝 r) := by
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have hw : z₀ ∈ ball z₀ ρ := mem_ball_self hρ
  constructor
  · have hcongr : (∮ z in C(z₀, ρ), f z) =
        ∮ z in C(z₀, ρ), ((∑ i ∈ ({0} : Finset ℕ), r / (z - z₀) ^ (0 + 1)) + g z) := by
      refine circleIntegral.integral_congr hρ.le fun z hz => ?_
      have hz0 : z ≠ z₀ := fun h => mem_sphere_ne hz hw (by rw [h, sub_self])
      simp only [Finset.sum_singleton, zero_add, pow_one]
      exact hf z (sphere_subset_closedBall hz) hz0
    rw [residue, hcongr, circle_sum_poles {0} z₀ ρ hρ (fun _ => z₀) (fun _ => r) (fun _ => 0)
      (fun _ _ => hw) g hg]
    simp only [Finset.sum_singleton, ite_true]
    rw [← mul_assoc, inv_mul_cancel₀ hne, one_mul]
  · have hgc : ContinuousAt g z₀ :=
      (hg.differentiableAt (closedBall_mem_nhds z₀ hρ)).continuousAt
    have hlim : Tendsto (fun z => r + (z - z₀) * g z) (𝓝[≠] z₀) (𝓝 (r + (z₀ - z₀) * g z₀)) :=
      ((tendsto_const_nhds.add ((continuousAt_id.sub continuousAt_const).mul hgc)).mono_left
        nhdsWithin_le_nhds)
    simp only [sub_self, zero_mul, add_zero] at hlim
    refine hlim.congr' ?_
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (closedBall_mem_nhds z₀ hρ)] with z hz hzB
    have : z - z₀ ≠ 0 := sub_ne_zero.2 hz
    rw [hf z hzB hz]; field_simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.simple_pole
-- test: ca-res-simple-pole-test.sage
end

/-- At a pole of order `m + 1`, `f = φ/(z - c)^(m+1)` with `φ` holomorphic:
`Res f = φ^(m)(c)/m! = (1/m!) lim dᵐ/dzᵐ [(z - c)^(m+1) f(z)]`. -/
theorem pole_order_residue (phi : ℂ → ℂ) (c : ℂ) (ρ : ℝ) (hρ : 0 < ρ)
    (hphi : DifferentiableOn ℂ phi (closedBall c ρ)) (m : ℕ) :
    residue (fun z => phi z / (z - c) ^ (m + 1)) c ρ = iteratedDeriv m phi c / (Nat.factorial m : ℂ) :=
  TaylorSeries.coeff_eq_circleIntegral phi c ρ hρ hphi m

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.pole_order_residue
-- test: ca-res-pole-order-test.sage
end

/-- The quotient rule at a simple pole: `Res g/h = g(z₀)/h'(z₀)` when `h(z₀) = 0 ≠ h'(z₀)`. -/
theorem quotient_rule (g h : ℂ → ℂ) (z₀ d : ℂ) (hh : h z₀ = 0) (hd : HasDerivAt h d z₀) (hd0 : d ≠ 0)
    (hg : ContinuousAt g z₀) : Tendsto (fun z => (z - z₀) * (g z / h z)) (𝓝[≠] z₀) (𝓝 (g z₀ / d)) :=
  LaurentSeries.tendsto_mul_div_of_hasDerivAt g h z₀ d hh hd hd0 hg

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.quotient_rule
-- test: ca-res-quotient-rule-test.sage
end

/-! ## exm-residue-simple -/

/-- Partial fractions with the residues as coefficients: `1/((z - a)(z - b)) = (a - b)⁻¹/(z - a) + (b - a)⁻¹/(z - b)`. -/
theorem partial_fraction_ab (a b z : ℂ) (hab : a - b ≠ 0) (ha : z - a ≠ 0) (hb : z - b ≠ 0) :
    1 / ((z - a) * (z - b)) = 1 / (a - b) / (z - a) + 1 / (b - a) / (z - b) := by
  have hba : b - a ≠ 0 := by intro h; exact hab (by linear_combination -h)
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.partial_fraction_ab
-- test: ca-res-partial-ab-test.sage
end

/-- `1/((z - a)(z - b))` has residue `1/(a - b)` at `a` and `1/(b - a)` at `b` (the text: `-1` at `1`, `1` at `2`),
both as small circle integrals and as limits. -/
theorem residue_two_poles (a b : ℂ) (hab : a ≠ b) (ρ : ℝ) (hρ : 0 < ρ) (hρa : ρ < ‖a - b‖) :
    residue (fun z => 1 / ((z - a) * (z - b))) a ρ = 1 / (a - b) ∧
      Tendsto (fun z => (z - a) * (1 / ((z - a) * (z - b)))) (𝓝[≠] a) (𝓝 (1 / (a - b))) ∧
      residue (fun z => 1 / ((z - a) * (z - b))) b ρ = 1 / (b - a) ∧
      Tendsto (fun z => (z - b) * (1 / ((z - a) * (z - b)))) (𝓝[≠] b) (𝓝 (1 / (b - a))) := by
  have hab' : a - b ≠ 0 := sub_ne_zero.2 hab
  have hρb : ρ < ‖b - a‖ := by rwa [norm_sub_rev]
  have hbA := not_mem_closedBall_of_lt hρa
  have haB := not_mem_closedBall_of_lt hρb
  have hA := simple_pole (fun z => 1 / ((z - a) * (z - b))) (fun z => 1 / (b - a) / (z - b)) a (1 / (a - b)) ρ hρ
    (DifferentiableOn.div (differentiableOn_const _) (by fun_prop) fun z hz => ne_of_not_mem hz hbA)
    (fun z hz hza => partial_fraction_ab a b z hab' (sub_ne_zero.2 hza) (ne_of_not_mem hz hbA))
  have hB := simple_pole (fun z => 1 / ((z - a) * (z - b))) (fun z => 1 / (a - b) / (z - a)) b (1 / (b - a)) ρ hρ
    (DifferentiableOn.div (differentiableOn_const _) (by fun_prop) fun z hz => ne_of_not_mem hz haB)
    (fun z hz hzb => by
      rw [partial_fraction_ab a b z hab' (ne_of_not_mem hz haB) (sub_ne_zero.2 hzb)]; ring)
  exact ⟨hA.1, hA.2, hB.1, hB.2⟩

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.residue_two_poles
-- test: ca-res-residue-two-poles-test.sage
end

/-- `1/(z² + c²)` has residue `1/(2ic)` at `ic` by the quotient rule `g/h' = 1/(2z)` (the text: `c = 1`, `1/(2i)`). -/
theorem residue_inv_sq_add_sq (c : ℂ) (hc : c ≠ 0) :
    Tendsto (fun z => (z - I * c) * (1 / (z ^ 2 + c ^ 2))) (𝓝[≠] (I * c)) (𝓝 (1 / (2 * (I * c)))) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have hh : (I * c) ^ 2 + c ^ 2 = 0 := by linear_combination c ^ 2 * hI
  have hd : HasDerivAt (fun z : ℂ => z ^ 2 + c ^ 2) (2 * (I * c)) (I * c) := by
    simpa using ((hasDerivAt_pow 2 (I * c)).add_const (c ^ 2))
  exact quotient_rule (fun _ => 1) (fun z => z ^ 2 + c ^ 2) (I * c) (2 * (I * c)) hh hd
    (mul_ne_zero two_ne_zero (mul_ne_zero I_ne_zero hc)) continuousAt_const

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.residue_inv_sq_add_sq
-- test: ca-res-inv-sq-residue-test.sage
end

/-- `1/(2i) = -i/2`. -/
theorem inv_two_I : 1 / (2 * I) = -I / 2 := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  field_simp
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage

/-! ## exm-residue-formulas -/

lemma dslope_sin_ne_zero (z : ℂ) (hz : ‖z‖ < π) : dslope sin 0 z ≠ 0 := by
  by_cases h0 : z = 0
  · subst h0; rw [dslope_same, Complex.deriv_sin]; simp
  · rw [dslope_of_ne _ h0, slope_def_field, Complex.sin_zero, sub_zero, sub_zero]
    exact div_ne_zero (LaurentSeries.sin_ne_zero_of_lt_pi z h0 hz) h0

/-- `Res₀ 1/sin z = 1/cos 0 = 1`, as the integral over any circle `|z| = ρ < π`. -/
theorem csc_circle_residue (ρ : ℝ) (hρ : 0 < ρ) (hρπ : ρ < π) : residue (fun z => 1 / sin z) 0 ρ = 1 := by
  set phi : ℂ → ℂ := fun z => 1 / dslope sin 0 z with hphi_def
  have hds : Differentiable ℂ (dslope sin 0) := by
    have := (Complex.differentiableOn_dslope (s := Set.univ) (f := sin) (c := 0) Filter.univ_mem).2
      differentiable_sin.differentiableOn
    exact fun z => (this z (Set.mem_univ z)).differentiableAt Filter.univ_mem
  have hphi : DifferentiableOn ℂ phi (closedBall 0 ρ) := by
    intro z hz
    refine (DifferentiableAt.div (differentiableAt_const _) (hds z) ?_).differentiableWithinAt
    refine dslope_sin_ne_zero z ?_
    rw [mem_closedBall, dist_zero_right] at hz; linarith
  have hcongr : residue (fun z => 1 / sin z) 0 ρ = residue (fun z => phi z / (z - 0) ^ (0 + 1)) 0 ρ := by
    unfold residue
    congr 1
    refine circleIntegral.integral_congr hρ.le fun z hz => ?_
    have hz0 : z ≠ 0 := by
      rintro rfl; rw [mem_sphere, dist_self] at hz; linarith
    simp only [hphi_def, sub_zero, zero_add, pow_one]
    rw [dslope_of_ne _ hz0, slope_def_field, Complex.sin_zero, sub_zero, sub_zero]
    by_cases hs : sin z = 0
    · simp [hs]
    · field_simp
  rw [hcongr, pole_order_residue phi 0 ρ hρ hphi 0]
  simp [hphi_def, dslope_same, Complex.deriv_sin]

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.csc_circle_residue
-- test: ca-res-csc-circle-test.sage
end

/-- `Res₀ e^{az}/z² = (d/dz) e^{az} |₀ = a`, a pole of order `2` (the text: `a = 1`). -/
theorem exp_div_sq_residue (a : ℂ) (ρ : ℝ) (hρ : 0 < ρ) : residue (fun z => exp (a * z) / z ^ 2) 0 ρ = a := by
  have h := pole_order_residue (fun z => exp (a * z)) 0 ρ hρ (by fun_prop) 1
  simp only [sub_zero, Nat.factorial, Nat.cast_one, div_one, iteratedDeriv_one, Nat.succ_eq_add_one,
    zero_add, mul_one] at h
  rw [h]
  have := (((hasDerivAt_id (0 : ℂ)).const_mul a).cexp).deriv
  simpa using this

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.exp_div_sq_residue
-- test: ca-res-exp-sq-residue-test.sage
end

/-- `Res₀ e^{a/z} = a`, read from the series `e^{a/z} = ∑ aⁿ z⁻ⁿ/n!` (the text: `a = 1`). -/
theorem exp_inv_residue (a : ℂ) (ρ : ℝ) (hρ : 0 < ρ) : residue (fun z => exp (a / z)) 0 ρ = a := by
  have hb : Summable (fun k : ℕ => ‖a ^ k / (Nat.factorial k : ℂ)‖ * ρ ^ (-(k : ℤ))) := by
    have := Real.summable_pow_div_factorial (‖a‖ / ρ)
    refine this.congr fun k => ?_
    rw [zpow_neg, zpow_natCast, norm_div, norm_pow, Complex.norm_natCast, div_pow]; ring
  have h := LaurentSeries.laurent_coeff_gen 0 ρ hρ (fun k : ℕ => -(k : ℤ)) (fun k => a ^ k / (Nat.factorial k : ℂ))
    hb (-1)
  simp only [neg_neg, sub_self, zpow_zero, one_mul, sub_zero, neg_inj] at h
  have hcongr : residue (fun z => exp (a / z)) 0 ρ =
      (2 * π * I)⁻¹ * ∮ z in C(0, ρ), ∑' k : ℕ, a ^ k / (Nat.factorial k : ℂ) * z ^ (-(k : ℤ)) := by
    unfold residue
    congr 1
    refine circleIntegral.integral_congr hρ.le fun z hz => ?_
    have hz0 : z ≠ 0 := by
      rintro rfl; rw [mem_sphere, dist_self] at hz; linarith
    have hs : HasSum (fun k : ℕ => (a / z) ^ k / (Nat.factorial k : ℂ)) (exp (a / z)) := by
      rw [Complex.exp_eq_exp_ℂ]; exact NormedSpace.expSeries_div_hasSum_exp (a / z)
    rw [← hs.tsum_eq]
    refine tsum_congr fun k => ?_
    rw [zpow_neg, zpow_natCast, div_pow]; ring
  rw [hcongr, h]
  rw [tsum_eq_single 1 (fun k hk => by
    have : (k : ℤ) ≠ 1 := by exact_mod_cast hk
    simp [this])]
  simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.exp_inv_residue
-- test: ca-res-exp-inv-residue-test.sage
end

/-! ## exm-residue-infinity -/

/-- For `ρ > |a|, |b|` the large circle integral of `1/((z - a)(z - b))` vanishes: the residue at infinity is `0`,
and `1/(a - b) + 1/(b - a) + 0 = 0`. -/
theorem residue_infinity_two_poles (a b : ℂ) (hab : a ≠ b) (ρ : ℝ) (ha : ‖a‖ < ρ) (hb : ‖b‖ < ρ) :
    -residue (fun z => 1 / ((z - a) * (z - b))) 0 ρ = 0 := by
  have hab' : a - b ≠ 0 := sub_ne_zero.2 hab
  have hba' : b - a ≠ 0 := sub_ne_zero.2 hab.symm
  have hρ : 0 < ρ := (norm_nonneg a).trans_lt ha
  have haB : a ∈ ball (0 : ℂ) ρ := by simpa using ha
  have hbB : b ∈ ball (0 : ℂ) ρ := by simpa using hb
  have hcongr : (∮ z in C(0, ρ), 1 / ((z - a) * (z - b))) =
      ∮ z in C(0, ρ), ((∑ i : Fin 2, ![1 / (a - b), 1 / (b - a)] i / (z - ![a, b] i) ^ (0 + 1)) + 0) := by
    refine circleIntegral.integral_congr hρ.le fun z hz => ?_
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, zero_add, pow_one, add_zero]
    exact partial_fraction_ab a b z hab' (mem_sphere_ne hz haB) (mem_sphere_ne hz hbB)
  rw [residue, hcongr, circle_sum_poles Finset.univ 0 ρ hρ ![a, b] ![1 / (a - b), 1 / (b - a)] (fun _ => 0)
    (fun i _ => by fin_cases i <;> simp [haB, hbB]) (fun _ => 0) (differentiableOn_const _)]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, ite_true]
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.residue_infinity_two_poles
-- test: ca-res-residue-infinity-test.sage
end

/-- The Laurent series at infinity `∑_{n ≥ 2} (2ⁿ⁻¹ - 1) z⁻ⁿ` of `1/((z - 1)(z - 2))` has no `1/z` term:
`Res_∞ = -a₋₁ = 0` on every circle `|z| = ρ > 2`. -/
theorem residue_infinity_series (ρ : ℝ) (hρ : 2 < ρ) :
    -residue (fun z => ∑' n : ℕ, (2 ^ (n + 1) - 1) * (z - 0) ^ (-((n : ℤ) + 2))) 0 ρ = 0 := by
  have hρ0 : 0 < ρ := by linarith
  have hb : Summable (fun n : ℕ => ‖(2 : ℂ) ^ (n + 1) - 1‖ * ρ ^ (-((n : ℤ) + 2))) := by
    have hq : (2 / ρ : ℝ) < 1 := by rw [div_lt_one hρ0]; exact hρ
    have hs := (summable_geometric_of_lt_one (by positivity) hq).mul_left (3 / ρ ^ 2)
    refine Summable.of_nonneg_of_le (fun n => by positivity) (fun n => ?_) hs
    have h1 : ‖(2 : ℂ) ^ (n + 1) - 1‖ ≤ 3 * 2 ^ n := by
      have h2n : (1 : ℝ) ≤ 2 ^ n := one_le_pow₀ (by norm_num)
      calc ‖(2 : ℂ) ^ (n + 1) - 1‖ ≤ ‖(2 : ℂ) ^ (n + 1)‖ + ‖(1 : ℂ)‖ := norm_sub_le _ _
        _ = 2 * 2 ^ n + 1 := by rw [norm_pow, norm_one]; norm_num; ring
        _ ≤ 3 * 2 ^ n := by linarith
    rw [show -((n : ℤ) + 2) = -((n + 2 : ℕ) : ℤ) by push_cast; ring, zpow_neg, zpow_natCast]
    calc ‖(2 : ℂ) ^ (n + 1) - 1‖ * (ρ ^ (n + 2))⁻¹ ≤ 3 * 2 ^ n * (ρ ^ (n + 2))⁻¹ := by gcongr
      _ = 3 / ρ ^ 2 * (2 / ρ) ^ n := by rw [div_pow]; field_simp; ring
  have h := LaurentSeries.laurent_coeff_gen 0 ρ hρ0 (fun n : ℕ => -((n : ℤ) + 2))
    (fun n => (2 : ℂ) ^ (n + 1) - 1) hb (-1)
  simp only [neg_neg, sub_self, zpow_zero, one_mul] at h
  rw [residue, h]
  have h0 : ∀ i : ℕ, (if -((i : ℤ) + 2) = -1 then (2 : ℂ) ^ (i + 1) - 1 else 0) = 0 :=
    fun i => if_neg (by omega)
  simp only [h0, tsum_zero, neg_zero]

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.residue_infinity_series
-- test: ca-res-residue-infinity-series-test.sage
end

/-! ## exr-residue-rational -/

/-- Partial fractions of `(z² + 1)/(z (z - a)²)`: residue `1/a²` at `0` and `1 - 1/a²` at `a` (the text: `a = 1`). -/
theorem rational_partial_fractions (a z : ℂ) (ha : a ≠ 0) (hz : z ≠ 0) (hza : z - a ≠ 0) :
    (z ^ 2 + 1) / (z * (z - a) ^ 2) = 1 / a ^ 2 / z + (1 - 1 / a ^ 2) / (z - a) + (a ^ 2 + 1) / a / (z - a) ^ 2 := by
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.rational_partial_fractions
-- test: ca-res-rational-pf-test.sage
end

lemma rational_sum (a z : ℂ) (ha : a ≠ 0) (hz : z ≠ 0) (hza : z - a ≠ 0) :
    (z ^ 2 + 1) / (z * (z - a) ^ 2) =
      (∑ i : Fin 3, ![1 / a ^ 2, 1 - 1 / a ^ 2, (a ^ 2 + 1) / a] i / (z - ![0, a, a] i) ^ (![0, 0, 1] i + 1)) + 0 := by
  simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.head_cons, Matrix.tail_cons, sub_zero, zero_add, pow_one, add_zero, Nat.reduceAdd]
  exact rational_partial_fractions a z ha hz hza

/-- The residues of `(z² + 1)/(z (z - a)²)` on small circles: `1/a²` at `0` (also `lim z f(z)`) and `1 - 1/a²`
at `a` (the text: `a = 1`: `1` and `0`). -/
theorem rational_residues (a : ℂ) (ha : a ≠ 0) (ρ : ℝ) (hρ : 0 < ρ) (hρa : ρ < ‖a‖) :
    residue (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2)) 0 ρ = 1 / a ^ 2 ∧
      Tendsto (fun z => (z - 0) * ((z ^ 2 + 1) / (z * (z - a) ^ 2))) (𝓝[≠] 0) (𝓝 (1 / a ^ 2)) ∧
      residue (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2)) a ρ = 1 - 1 / a ^ 2 := by
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have ha0 : a ∉ closedBall (0 : ℂ) ρ := by
    rw [mem_closedBall, dist_zero_right]; linarith
  have h0a : (0 : ℂ) ∉ closedBall a ρ := by
    rw [mem_closedBall, dist_comm, dist_zero_right]; linarith
  have hS := simple_pole (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2))
    (fun z => (1 - 1 / a ^ 2) / (z - a) + (a ^ 2 + 1) / a / (z - a) ^ 2) 0 (1 / a ^ 2) ρ hρ
    ((DifferentiableOn.div (differentiableOn_const _) (by fun_prop) fun z hz => ne_of_not_mem hz ha0).add
      (DifferentiableOn.div (differentiableOn_const _) (by fun_prop)
        fun z hz => pow_ne_zero _ (ne_of_not_mem hz ha0)))
    (fun z hz hz0 => by
      rw [sub_zero, rational_partial_fractions a z ha hz0 (ne_of_not_mem hz ha0)]; ring)
  refine ⟨hS.1, hS.2, ?_⟩
  have haB : a ∈ ball a ρ := mem_ball_self hρ
  have hcongr : (∮ z in C(a, ρ), (z ^ 2 + 1) / (z * (z - a) ^ 2)) =
      ∮ z in C(a, ρ), ((∑ i : Fin 2, ![1 - 1 / a ^ 2, (a ^ 2 + 1) / a] i / (z - a) ^ (![0, 1] i + 1)) +
        1 / a ^ 2 / z) := by
    refine circleIntegral.integral_congr hρ.le fun z hz => ?_
    have hza := mem_sphere_ne hz haB
    have hz0 : z ≠ 0 := by have := mem_sphere_ne' hz h0a; simpa using this
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, zero_add, pow_one,
      Nat.reduceAdd]
    rw [rational_partial_fractions a z ha hz0 hza]; ring
  have hg : DifferentiableOn ℂ (fun z : ℂ => 1 / a ^ 2 / z) (closedBall a ρ) :=
    DifferentiableOn.div (differentiableOn_const _) differentiableOn_id fun z hz h => h0a (h ▸ hz)
  rw [residue, hcongr, circle_sum_poles Finset.univ a ρ hρ (fun _ => a) ![1 - 1 / a ^ 2, (a ^ 2 + 1) / a] ![0, 1]
    (fun _ _ => haB) _ hg]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, ite_true, one_ne_zero, ite_false,
    add_zero]
  rw [← mul_assoc, inv_mul_cancel₀ hne, one_mul]

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.rational_residues
-- test: ca-res-rational-residues-test.sage
end

/-- `Res_∞ (z² + 1)/(z (z - a)²) = -1` (the function behaves as `1/z`), and the three residues sum to `0`. -/
theorem rational_residue_infinity (a : ℂ) (ha : a ≠ 0) (ρ : ℝ) (hρ : ‖a‖ < ρ) :
    -residue (fun z => (z ^ 2 + 1) / (z * (z - a) ^ 2)) 0 ρ = -1 ∧
      1 / a ^ 2 + (1 - 1 / a ^ 2) + -1 = (0 : ℂ) := by
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have hρ0 : 0 < ρ := (norm_nonneg a).trans_lt hρ
  have haB : a ∈ ball (0 : ℂ) ρ := by simpa using hρ
  have h0B : (0 : ℂ) ∈ ball (0 : ℂ) ρ := mem_ball_self hρ0
  have hcongr : (∮ z in C(0, ρ), (z ^ 2 + 1) / (z * (z - a) ^ 2)) =
      ∮ z in C(0, ρ), ((∑ i : Fin 3, ![1 / a ^ 2, 1 - 1 / a ^ 2, (a ^ 2 + 1) / a] i /
        (z - ![0, a, a] i) ^ (![0, 0, 1] i + 1)) + 0) := by
    refine circleIntegral.integral_congr hρ0.le fun z hz => ?_
    have hza := mem_sphere_ne hz haB
    have hz0 : z ≠ 0 := by have := mem_sphere_ne hz h0B; simpa using this
    exact rational_sum a z ha hz0 hza
  refine ⟨?_, by ring⟩
  rw [residue, hcongr, circle_sum_poles Finset.univ 0 ρ hρ0 ![0, a, a] ![1 / a ^ 2, 1 - 1 / a ^ 2, (a ^ 2 + 1) / a]
    ![0, 0, 1] (fun i _ => by fin_cases i <;> simp [haB, h0B]) (fun _ => 0) (differentiableOn_const _)]
  simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.head_cons, Matrix.tail_cons, ite_true, one_ne_zero, ite_false, add_zero]
  rw [← mul_assoc, inv_mul_cancel₀ hne, one_mul]
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Residues.rational_residue_infinity
-- test: ca-res-rational-infinity-test.sage
end

/-- The order-`2` formula at `a`: `(d/dz) (z² + 1)/z |_a = 1 - 1/a²` (the text: `0` at `a = 1`). -/
theorem rational_deriv (a : ℂ) (ha : a ≠ 0) : deriv (fun z : ℂ => (z ^ 2 + 1) / z) a = 1 - 1 / a ^ 2 := by
  have h1 : HasDerivAt (fun z : ℂ => z ^ 2 + 1) (2 * a) a := by
    simpa using (hasDerivAt_pow 2 a).add_const 1
  have hd : HasDerivAt (fun z : ℂ => (z ^ 2 + 1) / z) ((2 * a * a - (a ^ 2 + 1) * 1) / a ^ 2) a :=
    h1.div (hasDerivAt_id' a) ha
  rw [hd.deriv]
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.rational_deriv
-- test: ca-res-rational-deriv-test.sage
end

/-! ## exr-residue-quotient -/

/-- The poles of `1/(1 + z⁴)`: `z_k = e^{iπ(2k+1)/4}` satisfies `z_k⁴ = -1`. -/
theorem fourth_root (k : ℕ) : exp (((2 * k + 1) * π / 4 : ℝ) * I) ^ 4 = -1 := by
  rw [← Complex.exp_nat_mul]
  have : ((4 : ℕ) : ℂ) * (((2 * k + 1) * π / 4 : ℝ) * I) = (k : ℂ) * (2 * π * I) + π * I := by
    push_cast; ring
  rw [this, Complex.exp_add, exp_nat_mul_two_pi_mul_I, Complex.exp_pi_mul_I]; ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.Residues.fourth_root
-- test: ca-res-fourth-root-test.sage
end

/-- At a root `w⁴ = -1`: `Res = 1/(4w³) = -w/4`. -/
theorem residue_fourth (w : ℂ) (hw : w ^ 4 = -1) : 1 / (4 * w ^ 3) = -w / 4 := by
  have hw0 : w ≠ 0 := by rintro rfl; norm_num at hw
  field_simp
  veriphy_certified (linear_combination hw)
  -- certificate: ideal_membership.sage

/-- The quotient rule at a root `w⁴ = -1`: `(z - w)/(1 + z⁴) → 1/(4w³)`. -/
theorem quotient_fourth (w : ℂ) (hw : w ^ 4 = -1) :
    Tendsto (fun z => (z - w) * (1 / (1 + z ^ 4))) (𝓝[≠] w) (𝓝 (1 / (4 * w ^ 3))) := by
  have hw0 : w ≠ 0 := by rintro rfl; norm_num at hw
  have hd : HasDerivAt (fun z : ℂ => 1 + z ^ 4) (4 * w ^ 3) w := by
    simpa using (hasDerivAt_pow 4 w).const_add 1
  exact quotient_rule (fun _ => 1) (fun z => 1 + z ^ 4) w (4 * w ^ 3) (by rw [hw]; ring) hd
    (by simp [hw0]) continuousAt_const

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.quotient_fourth
-- test: ca-res-quotient-fourth-test.sage
end

/-- `e^{iθ} + e^{i(π - θ)} = 2i sin θ`. -/
theorem exp_reflect (θ : ℝ) : exp (θ * I) + exp (((π - θ : ℝ)) * I) = 2 * Real.sin θ * I := by
  rw [Complex.exp_mul_I, Complex.exp_mul_I]
  push_cast
  rw [Complex.cos_pi_sub, Complex.sin_pi_sub]
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Residues.exp_reflect
-- test: ca-res-exp-reflect-test.sage
end

/-- The two residues in the upper half-plane sum to `-i/(2√2)`. -/
theorem upper_residue_sum :
    -(1 / 4) * (exp ((π / 4 : ℝ) * I) + exp ((3 * π / 4 : ℝ) * I)) = -I / (2 * (Real.sqrt 2 : ℂ)) := by
  have h := exp_reflect (π / 4)
  rw [show (π - π / 4 : ℝ) = 3 * π / 4 by ring, Real.sin_pi_div_four] at h
  rw [h]
  have hs : ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 := by norm_cast; simp
  have hs0 : ((Real.sqrt 2 : ℝ) : ℂ) ≠ 0 := by norm_cast; positivity
  push_cast
  field_simp
  veriphy_certified (linear_combination (-2) * hs)
  -- certificate: ideal_membership.sage

/-- `2πi · (-i/(2√2)) = π/√2`. -/
theorem two_pi_I_upper : 2 * π * I * (-I / (2 * (Real.sqrt 2 : ℂ))) = π / (Real.sqrt 2 : ℂ) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have hs0 : ((Real.sqrt 2 : ℝ) : ℂ) ≠ 0 := by norm_cast; positivity
  field_simp
  veriphy_certified (linear_combination (-1) * hI)
  -- certificate: ideal_membership.sage

end KaluzaKlein.ComplexAnalysis.Residues

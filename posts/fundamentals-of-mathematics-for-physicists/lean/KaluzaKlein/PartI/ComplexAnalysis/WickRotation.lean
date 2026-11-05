import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Wick rotation"

The claims of the section "Wick rotation" of `tutorial/part-i-04-complex-analysis.qmd`. The general rotation of an
integration contour through a pole-free sector (Cauchy's theorem on unbounded sectors) is not in mathlib; it is
proved here for the propagator, where both sides are computed: the key lemma
`∫_ℝ dk/((k - α)(k + α)) = πi/α` for `Im α > 0` (an explicit antiderivative built from `log` and `arctan`) gives the
Minkowski integral with the poles displaced, the rotated integral, the Euclidean integral and the opposite
prescription. Also: the position of the poles, the ML estimate on the arcs, the Gaussian with an `iε` and its limit
(the Fresnel value), the rotation of the exponent, and the factors of `i` in the Euclidean action. The improper
(Riemann) Fresnel integral is formalised as the `iε`-regularised limit; the general rotation identity (Cauchy's
theorem on unbounded sectors with vanishing arcs) is not in mathlib and is not formalised.
-/

namespace KaluzaKlein.ComplexAnalysis.WickRotation

open Complex Filter Topology Set MeasureTheory intervalIntegral
open scoped Real

/-! ## Key lemma: `∫ dk/((k - α)(k + α)) = πi/α` for `Im α > 0` -/

/-- `(x² + b²)⁻¹ = b⁻² (1 + (x/b)²)⁻¹`. -/
theorem inv_sq_add_sq_eq (x b : ℝ) (hb : b ≠ 0) : (x ^ 2 + b ^ 2)⁻¹ = 1 / b ^ 2 * (1 + (x / b) ^ 2)⁻¹ := by
  have hx : x ^ 2 + b ^ 2 ≠ 0 := by positivity
  have hx' : 1 + (x / b) ^ 2 ≠ 0 := by positivity
  field_simp
  try ring

/-- The Cauchy-type density `((k - c)² + b²)⁻¹` is integrable on `ℝ` (`b ≠ 0`). -/
theorem integrable_inv_sq_add_sq (b c : ℝ) (hb : b ≠ 0) : Integrable (fun k : ℝ => ((k - c) ^ 2 + b ^ 2)⁻¹) := by
  have h := ((integrable_inv_one_add_sq.comp_div hb).comp_sub_right c).const_mul (1 / b ^ 2)
  refine h.congr (Eventually.of_forall fun k => ?_)
  simp only
  rw [inv_sq_add_sq_eq (k - c) b hb]

/-- `1/(1 + (x/b)²) · (1/b) = b/(x² + b²)`. -/
theorem arctan_deriv_eq (x b : ℝ) (hb : b ≠ 0) : 1 / (1 + (x / b) ^ 2) * (1 / b) = b / (x ^ 2 + b ^ 2) := by
  have hx : x ^ 2 + b ^ 2 ≠ 0 := by positivity
  have hx' : 1 + (x / b) ^ 2 ≠ 0 := by positivity
  field_simp
  try ring

/-- The key lemma (division form): `∫_ℝ dk/((k - α)(k + α)) = πi/α` for `Im α > 0`. -/
theorem integral_inv_sub_mul_add' (α : ℂ) (hα : 0 < α.im) :
    ∫ k : ℝ, 1 / (((k : ℂ) - α) * ((k : ℂ) + α)) = π * I / α := by
  obtain ⟨a, b, rfl⟩ : ∃ a b : ℝ, α = a + b * I := ⟨α.re, α.im, (Complex.re_add_im α).symm⟩
  have hb : 0 < b := by simpa using hα
  have hα0 : (a : ℂ) + b * I ≠ 0 := fun h => by
    have := congrArg Complex.im h
    simp at this
    linarith
  have hsub : ∀ k : ℝ, (k : ℂ) - (a + b * I) ≠ 0 := fun k h => by
    have := congrArg Complex.im h
    simp at this
    linarith
  have hadd : ∀ k : ℝ, (k : ℂ) + (a + b * I) ≠ 0 := fun k h => by
    have := congrArg Complex.im h
    simp at this
    linarith
  have hD₁ : ∀ k : ℝ, 0 < (k - a) ^ 2 + b ^ 2 := fun k => add_pos_of_nonneg_of_pos (sq_nonneg _) (pow_pos hb 2)
  have hD₂ : ∀ k : ℝ, 0 < (k + a) ^ 2 + b ^ 2 := fun k => add_pos_of_nonneg_of_pos (sq_nonneg _) (pow_pos hb 2)
  -- the antiderivative `F = (1/(2α)) (L/2 + i T)`
  let L : ℝ → ℝ := fun k => Real.log ((k - a) ^ 2 + b ^ 2) - Real.log ((k + a) ^ 2 + b ^ 2)
  let T : ℝ → ℝ := fun k => Real.arctan ((k - a) / b) + Real.arctan ((k + a) / b)
  let F : ℝ → ℂ := fun k => 1 / (2 * ((a : ℂ) + b * I)) * (((L k / 2 : ℝ) : ℂ) + ((T k : ℝ) : ℂ) * I)
  have hL : ∀ k : ℝ, HasDerivAt L (2 * (k - a) / ((k - a) ^ 2 + b ^ 2) - 2 * (k + a) / ((k + a) ^ 2 + b ^ 2)) k := by
    intro k
    have h1 : HasDerivAt (fun k : ℝ => (k - a) ^ 2 + b ^ 2) (2 * (k - a)) k := by
      have := (((hasDerivAt_id k).sub_const a).pow 2).add_const (b ^ 2)
      convert this using 1 <;> simp
    have h2 : HasDerivAt (fun k : ℝ => (k + a) ^ 2 + b ^ 2) (2 * (k + a)) k := by
      have := (((hasDerivAt_id k).add_const a).pow 2).add_const (b ^ 2)
      convert this using 1 <;> simp
    exact (h1.log (hD₁ k).ne').sub (h2.log (hD₂ k).ne')
  have hT : ∀ k : ℝ, HasDerivAt T (b / ((k - a) ^ 2 + b ^ 2) + b / ((k + a) ^ 2 + b ^ 2)) k := by
    intro k
    have h1 : HasDerivAt (fun k : ℝ => Real.arctan ((k - a) / b)) (1 / (1 + ((k - a) / b) ^ 2) * (1 / b)) k :=
      (((hasDerivAt_id k).sub_const a).div_const b).arctan
    have h2 : HasDerivAt (fun k : ℝ => Real.arctan ((k + a) / b)) (1 / (1 + ((k + a) / b) ^ 2) * (1 / b)) k :=
      (((hasDerivAt_id k).add_const a).div_const b).arctan
    rw [arctan_deriv_eq _ _ hb.ne'] at h1 h2
    exact h1.add h2
  have key : ∀ k : ℝ, 1 / (2 * ((a : ℂ) + b * I)) *
      ((((2 * (k - a) / ((k - a) ^ 2 + b ^ 2) - 2 * (k + a) / ((k + a) ^ 2 + b ^ 2)) / 2 : ℝ) : ℂ) +
        (((b / ((k - a) ^ 2 + b ^ 2) + b / ((k + a) ^ 2 + b ^ 2)) : ℝ) : ℂ) * I) =
      1 / (((k : ℂ) - (a + b * I)) * ((k : ℂ) + (a + b * I))) := by
    intro k
    have hD₁' : (((k - a) ^ 2 + b ^ 2 : ℝ) : ℂ) ≠ 0 := by exact_mod_cast (hD₁ k).ne'
    have hD₂' : (((k + a) ^ 2 + b ^ 2 : ℝ) : ℂ) ≠ 0 := by exact_mod_cast (hD₂ k).ne'
    have e1 : (((k - a : ℝ) : ℂ) + (b : ℂ) * I) / (((k - a) ^ 2 + b ^ 2 : ℝ) : ℂ) = 1 / ((k : ℂ) - (a + b * I)) := by
      rw [div_eq_div_iff hD₁' (hsub k)]
      push_cast
      linear_combination (-(b : ℂ) ^ 2) * Complex.I_sq
    have e2 : (((k + a : ℝ) : ℂ) - (b : ℂ) * I) / (((k + a) ^ 2 + b ^ 2 : ℝ) : ℂ) = 1 / ((k : ℂ) + (a + b * I)) := by
      rw [div_eq_div_iff hD₂' (hadd k)]
      push_cast
      linear_combination (-(b : ℂ) ^ 2) * Complex.I_sq
    have h12 : ((((2 * (k - a) / ((k - a) ^ 2 + b ^ 2) - 2 * (k + a) / ((k + a) ^ 2 + b ^ 2)) / 2 : ℝ) : ℂ) +
        (((b / ((k - a) ^ 2 + b ^ 2) + b / ((k + a) ^ 2 + b ^ 2)) : ℝ) : ℂ) * I) =
        (((k - a : ℝ) : ℂ) + (b : ℂ) * I) / (((k - a) ^ 2 + b ^ 2 : ℝ) : ℂ) -
          (((k + a : ℝ) : ℂ) - (b : ℂ) * I) / (((k + a) ^ 2 + b ^ 2 : ℝ) : ℂ) := by
      push_cast <;> ring
    rw [h12, e1, e2]
    have hs := hsub k
    have ha := hadd k
    field_simp <;> ring
  have hF : ∀ k : ℝ, HasDerivAt F (1 / (((k : ℂ) - (a + b * I)) * ((k : ℂ) + (a + b * I)))) k := by
    intro k
    have h := (((hL k).div_const 2).ofReal_comp.add ((hT k).ofReal_comp.mul_const I)).const_mul
      (1 / (2 * ((a : ℂ) + b * I)))
    rwa [key k] at h
  -- integrability: `‖f‖ ≤ (D₁⁻¹ + D₂⁻¹)/2`
  have hnorm : ∀ k : ℝ, ‖1 / (((k : ℂ) - (a + b * I)) * ((k : ℂ) + (a + b * I)))‖ ≤
      (((k - a) ^ 2 + b ^ 2)⁻¹ + ((k + a) ^ 2 + b ^ 2)⁻¹) / 2 := by
    intro k
    have hx : ‖(k : ℂ) - (a + b * I)‖ ^ 2 = (k - a) ^ 2 + b ^ 2 := by
      rw [Complex.sq_norm, Complex.normSq_apply]
      simp
      ring
    have hy : ‖(k : ℂ) + (a + b * I)‖ ^ 2 = (k + a) ^ 2 + b ^ 2 := by
      rw [Complex.sq_norm, Complex.normSq_apply]
      simp
      ring
    have hx0 : 0 < ‖(k : ℂ) - (a + b * I)‖ := norm_pos_iff.mpr (hsub k)
    have hy0 : 0 < ‖(k : ℂ) + (a + b * I)‖ := norm_pos_iff.mpr (hadd k)
    rw [norm_div, norm_one, norm_mul, ← hx, ← hy]
    generalize ‖(k : ℂ) - (a + b * I)‖ = x at hx0 ⊢
    generalize ‖(k : ℂ) + (a + b * I)‖ = y at hy0 ⊢
    have hx1 := hx0.ne'
    have hy1 := hy0.ne'
    have h : ((x ^ 2)⁻¹ + (y ^ 2)⁻¹) / 2 - 1 / (x * y) = (1 / x - 1 / y) ^ 2 / 2 := by
      field_simp <;> ring
    linarith [sq_nonneg (1 / x - 1 / y)]
  have hg : Integrable (fun k : ℝ => (((k - a) ^ 2 + b ^ 2)⁻¹ + ((k + a) ^ 2 + b ^ 2)⁻¹) / 2) := by
    have h1 := integrable_inv_sq_add_sq b a hb.ne'
    have h2 := integrable_inv_sq_add_sq b (-a) hb.ne'
    simp only [sub_neg_eq_add] at h2
    exact (h1.add h2).div_const 2
  have hcont : Continuous (fun k : ℝ => 1 / (((k : ℂ) - (a + b * I)) * ((k : ℂ) + (a + b * I)))) :=
    continuous_const.div (by fun_prop) (fun k => mul_ne_zero (hsub k) (hadd k))
  have hInt : Integrable (fun k : ℝ => 1 / (((k : ℂ) - (a + b * I)) * ((k : ℂ) + (a + b * I)))) :=
    Integrable.mono' hg hcont.aestronglyMeasurable (Eventually.of_forall hnorm)
  -- limits of `L` (through `t = 1/k`) and of `T`
  let φ : ℝ → ℝ := fun t => Real.log ((1 - a * t) ^ 2 + (b * t) ^ 2) - Real.log ((1 + a * t) ^ 2 + (b * t) ^ 2)
  have hφc : ContinuousAt φ 0 :=
    ContinuousAt.sub (ContinuousAt.log (by fun_prop) (by simp)) (ContinuousAt.log (by fun_prop) (by simp))
  have hφ0 : φ 0 = 0 := by simp [φ]
  have hLφ : ∀ k : ℝ, k ≠ 0 → L k = φ k⁻¹ := by
    intro k hk
    have hk2 : k ^ 2 ≠ 0 := pow_ne_zero 2 hk
    have hA : 0 < (1 - a * k⁻¹) ^ 2 + (b * k⁻¹) ^ 2 := by positivity
    have hB : 0 < (1 + a * k⁻¹) ^ 2 + (b * k⁻¹) ^ 2 := by positivity
    have e1 : (k - a) ^ 2 + b ^ 2 = k ^ 2 * ((1 - a * k⁻¹) ^ 2 + (b * k⁻¹) ^ 2) := by
      field_simp
      try ring
    have e2 : (k + a) ^ 2 + b ^ 2 = k ^ 2 * ((1 + a * k⁻¹) ^ 2 + (b * k⁻¹) ^ 2) := by
      field_simp
      try ring
    simp only [L, φ]
    rw [e1, e2, Real.log_mul hk2 hA.ne', Real.log_mul hk2 hB.ne']
    ring
  have hLtop : Tendsto L atTop (𝓝 0) := by
    have h := hφc.tendsto.comp tendsto_inv_atTop_zero
    rw [hφ0] at h
    refine h.congr' ?_
    filter_upwards [eventually_gt_atTop 0] with k hk
    exact (hLφ k hk.ne').symm
  have hLbot : Tendsto L atBot (𝓝 0) := by
    have h := hφc.tendsto.comp tendsto_inv_atBot_zero
    rw [hφ0] at h
    refine h.congr' ?_
    filter_upwards [eventually_lt_atBot 0] with k hk
    exact (hLφ k hk.ne).symm
  have hTtop : Tendsto T atTop (𝓝 (π / 2 + π / 2)) := by
    have h1 : Tendsto (fun k : ℝ => (k - a) / b) atTop atTop :=
      ((tendsto_atTop_add_const_right _ (-a) tendsto_id).atTop_div_const hb).congr
        (fun k => by simp [sub_eq_add_neg])
    have h2 : Tendsto (fun k : ℝ => (k + a) / b) atTop atTop :=
      (tendsto_atTop_add_const_right _ a tendsto_id).atTop_div_const hb
    exact (tendsto_nhds_of_tendsto_nhdsWithin (Real.tendsto_arctan_atTop.comp h1)).add
      (tendsto_nhds_of_tendsto_nhdsWithin (Real.tendsto_arctan_atTop.comp h2))
  have hTbot : Tendsto T atBot (𝓝 (-(π / 2) + -(π / 2))) := by
    have h1 : Tendsto (fun k : ℝ => (k - a) / b) atBot atBot :=
      ((tendsto_atBot_add_const_right _ (-a) tendsto_id).atBot_div_const hb).congr
        (fun k => by simp [sub_eq_add_neg])
    have h2 : Tendsto (fun k : ℝ => (k + a) / b) atBot atBot :=
      (tendsto_atBot_add_const_right _ a tendsto_id).atBot_div_const hb
    exact (tendsto_nhds_of_tendsto_nhdsWithin (Real.tendsto_arctan_atBot.comp h1)).add
      (tendsto_nhds_of_tendsto_nhdsWithin (Real.tendsto_arctan_atBot.comp h2))
  have hFtop : Tendsto F atTop
      (𝓝 (1 / (2 * ((a : ℂ) + b * I)) * ((((0 : ℝ) / 2 : ℝ) : ℂ) + ((π / 2 + π / 2 : ℝ) : ℂ) * I))) := by
    refine Tendsto.const_mul _ (Tendsto.add ?_ (Tendsto.mul_const _ ?_))
    · exact (Complex.continuous_ofReal.tendsto _).comp (hLtop.div_const 2)
    · exact (Complex.continuous_ofReal.tendsto _).comp hTtop
  have hFbot : Tendsto F atBot
      (𝓝 (1 / (2 * ((a : ℂ) + b * I)) * ((((0 : ℝ) / 2 : ℝ) : ℂ) + ((-(π / 2) + -(π / 2) : ℝ) : ℂ) * I))) := by
    refine Tendsto.const_mul _ (Tendsto.add ?_ (Tendsto.mul_const _ ?_))
    · exact (Complex.continuous_ofReal.tendsto _).comp (hLbot.div_const 2)
    · exact (Complex.continuous_ofReal.tendsto _).comp hTbot
  rw [integral_of_hasDerivAt_of_tendsto hF hInt hFbot hFtop]
  push_cast
  field_simp <;> ring


/-- The key lemma: `∫_ℝ dk/((k - α)(k + α)) = πi/α`, i.e. `α ∫ = πi`, for `Im α > 0`. -/
theorem integral_inv_sub_mul_add (α : ℂ) (hα : 0 < α.im) :
    α * ∫ k : ℝ, 1 / (((k : ℂ) - α) * ((k : ℂ) + α)) = π * I := by
  have h0 : α ≠ 0 := fun h => by rw [h] at hα; simp at hα
  rw [integral_inv_sub_mul_add' α hα, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.WickRotation.integral_inv_sub_mul_add
-- test: ca-spec-key-test.sage

/-! ## def-wick-rotation, exm-wick-propagator, exm-wick-euclidean-side -/

/-- The Minkowski propagator integral with displaced poles `E - iδ` (fourth quadrant) and `-E + iδ` (second):
`∫ dk/((k - (E - iδ))(k - (-E + iδ))) = 2πi/(-2E + 2iδ)` (closing above: `2πi Res_{-E+iδ}`). -/
theorem propagator_minkowski' (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    ∫ k : ℝ, 1 / (((k : ℂ) - (E - δ * I)) * ((k : ℂ) - (-E + δ * I))) = 2 * π * I / (-2 * E + 2 * δ * I) := by
  have h := integral_inv_sub_mul_add' (-E + δ * I) (by simpa using hδ)
  have e : ∀ k : ℝ, ((k : ℂ) - (E - δ * I)) * ((k : ℂ) - (-E + δ * I)) =
      ((k : ℂ) - (-E + δ * I)) * ((k : ℂ) + (-E + δ * I)) := fun k => by ring
  simp_rw [e]
  rw [h]
  have h0 : (-(E : ℂ) + δ * I) ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0
    simp at this
    linarith
  have h0' : (-2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => h0 (by linear_combination h1 / 2)
  field_simp <;> ring

/-- The Minkowski integral, multiplied out: `(-2E + 2iδ) ∫ dk/((k - (E - iδ))(k - (-E + iδ))) = 2πi`. -/
theorem propagator_minkowski (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    (-2 * E + 2 * δ * I) * ∫ k : ℝ, 1 / (((k : ℂ) - (E - δ * I)) * ((k : ℂ) - (-E + δ * I))) = 2 * π * I := by
  have h0 : (-2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => by
    have := congrArg Complex.re h1
    simp at this
    linarith
  rw [propagator_minkowski' E δ hE hδ, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.propagator_minkowski
-- test: ca-spec-prop-mink-test.sage

/-- The rotated integral `i∫ dk⁴ h(ik⁴)` has the same value `2πi/(-2E + 2iδ)`: the Wick rotation identity for the
propagator. -/
theorem propagator_rotated' (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    I * ∫ k : ℝ, 1 / ((I * k - (E - δ * I)) * (I * k - (-E + δ * I))) = 2 * π * I / (-2 * E + 2 * δ * I) := by
  have h := integral_inv_sub_mul_add' (δ + E * I) (by simpa using hE)
  have e : ∀ k : ℝ, 1 / ((I * k - (E - δ * I)) * (I * k - (-E + δ * I))) =
      -(1 / (((k : ℂ) - (δ + E * I)) * ((k : ℂ) + (δ + E * I)))) := fun k => by
    rw [← one_div_neg_eq_neg_one_div]
    congr 1
    linear_combination ((k : ℂ) ^ 2 - (E : ℂ) ^ 2 - (δ : ℂ) ^ 2) * Complex.I_sq
  simp_rw [e]
  rw [MeasureTheory.integral_neg, h]
  have h0 : ((δ : ℂ) + E * I) ≠ 0 := fun h0 => by
    have := congrArg Complex.im h0
    simp at this
    linarith
  have h0' : (-2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => by
    have := congrArg Complex.re h1
    simp at this
    linarith
  rw [show (-2 * (E : ℂ) + 2 * δ * I) = 2 * I * (δ + E * I) by linear_combination (-2 * (E : ℂ)) * Complex.I_sq,
    mul_neg, ← mul_div_assoc, ← neg_div,
    div_eq_div_iff h0 (mul_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) h0)]
  linear_combination (-2 * (π : ℂ) * I * ((δ : ℂ) + E * I)) * Complex.I_sq

/-- The rotated integral, multiplied out: `(-2E + 2iδ) · i ∫ dk⁴ h(ik⁴) = 2πi`. -/
theorem propagator_rotated (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    (-2 * E + 2 * δ * I) * (I * ∫ k : ℝ, 1 / ((I * k - (E - δ * I)) * (I * k - (-E + δ * I)))) = 2 * π * I := by
  have h0 : (-2 * (E : ℂ) + 2 * δ * I) ≠ 0 := fun h1 => by
    have := congrArg Complex.re h1
    simp at this
    linarith
  rw [propagator_rotated' E δ hE hδ, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.propagator_rotated
-- test: ca-spec-prop-rot-test.sage

/-- The Wick rotation identity for the propagator: `∫ dk⁰ h(k⁰) = i ∫ dk⁴ h(ik⁴)`. -/
theorem wick_rotation_propagator (E δ : ℝ) (hE : 0 < E) (hδ : 0 < δ) :
    ∫ k : ℝ, 1 / (((k : ℂ) - (E - δ * I)) * ((k : ℂ) - (-E + δ * I))) =
      I * ∫ k : ℝ, 1 / ((I * k - (E - δ * I)) * (I * k - (-E + δ * I))) := by
  rw [propagator_minkowski' E δ hE hδ, propagator_rotated' E δ hE hδ]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.wick_rotation_propagator
-- test: ca-spec-wick-prop-test.sage

/-- As `δ → 0⁺` the common value `2πi/(-2E + 2iδ)` tends to `-iπ/E`. -/
theorem propagator_limit (E : ℝ) (hE : 0 < E) :
    Tendsto (fun δ : ℝ => 2 * π * I / (-2 * E + 2 * δ * I)) (𝓝[>] 0) (𝓝 (-(π * I / E))) := by
  have hE' : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  have hc : ContinuousAt (fun δ : ℝ => 2 * π * I / (-2 * E + 2 * δ * I)) 0 := by
    apply ContinuousAt.div (by fun_prop) (by fun_prop)
    simp [hE']
  have h : Tendsto (fun δ : ℝ => 2 * π * I / (-2 * E + 2 * δ * I)) (𝓝[>] (0 : ℝ))
      (𝓝 (2 * π * I / (-2 * E + 2 * ((0 : ℝ) : ℂ) * I))) := hc.tendsto.mono_left nhdsWithin_le_nhds
  have hv : (2 * π * I / (-2 * E + 2 * ((0 : ℝ) : ℂ) * I) : ℂ) = -(π * I / E) := by
    simp only [Complex.ofReal_zero, mul_zero, zero_mul, add_zero]
    field_simp
  rw [hv] at h
  exact h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.propagator_limit
-- test: ca-spec-prop-limit-test.sage

/-- The Euclidean integral `∫ dk/(k² + E²) = π/E` for `E > 0`. -/
theorem euclidean_integral (E : ℝ) (hE : 0 < E) : ∫ k : ℝ, 1 / (k ^ 2 + E ^ 2) = π / E := by
  have h := integral_univ_inv_one_add_mul_sq (E⁻¹)
  have e : ∀ k : ℝ, 1 / (k ^ 2 + E ^ 2) = E⁻¹ ^ 2 * (1 + (E⁻¹ * k) ^ 2)⁻¹ := fun k => by
    have : k ^ 2 + E ^ 2 ≠ 0 := by positivity
    have : 1 + (E⁻¹ * k) ^ 2 ≠ 0 := by positivity
    field_simp
    try ring
  simp_rw [e]
  rw [MeasureTheory.integral_const_mul, h, abs_of_pos (inv_pos.mpr hE)]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_integral
-- test: ca-spec-eucl-test.sage

/-- The Euclidean side: `i ∫ dk⁴/((ik⁴)² - E²) = -i ∫ dk⁴/((k⁴)² + E²) = -iπ/E`. -/
theorem euclidean_side' (E : ℝ) (hE : 0 < E) :
    I * ∫ k : ℝ, 1 / ((I * k) ^ 2 - (E : ℂ) ^ 2) = -(π * I / E) := by
  have e : ∀ k : ℝ, 1 / ((I * k) ^ 2 - (E : ℂ) ^ 2) = -(((1 / (k ^ 2 + E ^ 2) : ℝ)) : ℂ) := fun k => by
    have : (k : ℂ) ^ 2 + (E : ℂ) ^ 2 ≠ 0 := by exact_mod_cast (by positivity : k ^ 2 + E ^ 2 ≠ 0)
    push_cast
    rw [← one_div_neg_eq_neg_one_div]
    congr 1
    linear_combination (k : ℂ) ^ 2 * Complex.I_sq
  simp_rw [e]
  rw [MeasureTheory.integral_neg, integral_complex_ofReal, euclidean_integral E hE]
  push_cast <;> ring

/-- The Euclidean side, multiplied out: `E · i ∫ dk⁴/((ik⁴)² - E²) = -iπ`. -/
theorem euclidean_side (E : ℝ) (hE : 0 < E) :
    (E : ℂ) * (I * ∫ k : ℝ, 1 / ((I * k) ^ 2 - (E : ℂ) ^ 2)) = -(π * I) := by
  have h0 : (E : ℂ) ≠ 0 := by exact_mod_cast hE.ne'
  rw [euclidean_side' E hE, mul_neg, mul_div_cancel₀ _ h0]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_side
-- test: ca-spec-eucl-side-test.sage

/-- `(k⁴)² + E²` never vanishes for `E ≠ 0`: Euclidean propagators need no `iε`. -/
theorem euclidean_denominator_pos (k E : ℝ) (hE : E ≠ 0) : 0 < k ^ 2 + E ^ 2 := by
  positivity

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_denominator_pos (0) (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_denominator_pos (2) (-3) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_denominator_pos (-1 / 2) (1 / 3) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_denominator_pos
-- test: ca-spec-eucl-pos-test.sage

/-- The poles of `1/((k⁰)² - E² + iε)`: a root `c` of `c² = E² - iε` lies in the fourth quadrant or (as `-c`) in the
second, never in the first or third. -/
theorem pole_quadrants (M ε : ℝ) (hM : 0 < M) (hε : 0 < ε) (c : ℂ) (hc : c ^ 2 = M - ε * I) :
    (0 < c.re ∧ c.im < 0) ∨ (c.re < 0 ∧ 0 < c.im) := by
  have hre : c.re ^ 2 - c.im ^ 2 = M := by
    have := congrArg Complex.re hc
    simp [sq] at this
    nlinarith [this]
  have him : 2 * c.re * c.im = -ε := by
    have := congrArg Complex.im hc
    simp [sq] at this
    linarith
  rcases lt_trichotomy c.re 0 with h | h | h
  · right
    exact ⟨h, by nlinarith⟩
  · exfalso
    rw [h] at hre
    nlinarith [sq_nonneg c.im]
  · left
    exact ⟨h, by nlinarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants (3) (4) ((by norm_num)) ((by norm_num)) (2 - Complex.I) ((by norm_num [sq, Complex.ext_iff])) := by
  first
    | (left; constructor <;> norm_num)
    | (right; constructor <;> norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants (8) (6) ((by norm_num)) ((by norm_num)) (3 - Complex.I) ((by norm_num [sq, Complex.ext_iff])) := by
  first
    | (left; constructor <;> norm_num)
    | (right; constructor <;> norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants (3 / 4) (1) ((by norm_num)) ((by norm_num)) (1 - Complex.I / 2) ((by norm_num [sq, Complex.ext_iff])) := by
  first
    | (left; constructor <;> norm_num)
    | (right; constructor <;> norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants
-- test: ca-spec-poles-test.sage

/-- With the opposite sign `-iε` the roots of `c² = E² + iε` lie in the first and third quadrants. -/
theorem pole_quadrants_conj (M ε : ℝ) (hM : 0 < M) (hε : 0 < ε) (c : ℂ) (hc : c ^ 2 = M + ε * I) :
    (0 < c.re ∧ 0 < c.im) ∨ (c.re < 0 ∧ c.im < 0) := by
  have hre : c.re ^ 2 - c.im ^ 2 = M := by
    have := congrArg Complex.re hc
    simp [sq] at this
    nlinarith [this]
  have him : 2 * c.re * c.im = ε := by
    have := congrArg Complex.im hc
    simp [sq] at this
    linarith
  rcases lt_trichotomy c.re 0 with h | h | h
  · right
    exact ⟨h, by nlinarith⟩
  · exfalso
    rw [h] at hre
    nlinarith [sq_nonneg c.im]
  · left
    exact ⟨h, by nlinarith⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants_conj (3) (4) ((by norm_num)) ((by norm_num)) (2 + Complex.I) ((by norm_num [sq, Complex.ext_iff])) := by
  left
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants_conj (3) (4) ((by norm_num)) ((by norm_num)) (-2 - Complex.I) ((by norm_num [sq, Complex.ext_iff])) := by
  right
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants_conj (8) (6) ((by norm_num)) ((by norm_num)) (3 + Complex.I) ((by norm_num [sq, Complex.ext_iff])) := by
  left
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants_conj (3 / 4) (1) ((by norm_num)) ((by norm_num)) (1 + Complex.I / 2) ((by norm_num [sq, Complex.ext_iff])) := by
  left
  constructor <;> norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.WickRotation.pole_quadrants_conj
-- test: ca-spec-poles-conj-test.sage

/-- The rotation to Euclidean signature: with `k⁰ = ik⁴`, `(k⁰)² - 𝐤² = -((k⁴)² + 𝐤²)` (`ksq` standing for `𝐤²`). -/
theorem euclidean_square (k4 ksq : ℂ) : (I * k4) ^ 2 - ksq = -(k4 ^ 2 + ksq) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  generalize I = j at hI ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (k4 ^ 2) * hI)
  -- certificate: ideal_membership.sage

-- The general rotation identity `∫_ℝ f = i ∫_ℝ f(i·)` for `f` holomorphic on the closed first and third quadrants
-- with `|k f(k)| → 0` (Cauchy's theorem on unbounded sectors) is not in mathlib and is not formalised; the identity is
-- proved for the propagator (`wick_rotation_propagator`) and the Gaussian is handled by `gaussian_i_eps_limit`.

/-! ## exm-wick-fresnel, exr-wick-gaussian, exr-wick-gaussian-action -/

/-- `e^{iπ/4}² = i`. -/
theorem exp_pi_div_four_sq : Complex.exp (π * I / 4) ^ 2 = I := by
  rw [← Complex.exp_nat_mul, show ((2 : ℕ) : ℂ) * (π * I / 4) = π / 2 * I by push_cast <;> ring,
    Complex.exp_pi_div_two_mul_I]

/-- The rotation `t = e^{iπ/4}u` turns the exponent `iat²` into `-au²`. -/
theorem fresnel_rotation_exponent (a u : ℂ) : I * a * (Complex.exp (π * I / 4) * u) ^ 2 = -(a * u ^ 2) := by
  have he := exp_pi_div_four_sq
  have hI : I ^ 2 = -1 := Complex.I_sq
  generalize Complex.exp (π * I / 4) = e at he ⊢
  generalize I = j at he hI ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (j * a * u ^ 2) * he + (a * u ^ 2) * hI)
  -- certificate: ideal_membership.sage

/-- `Re(i a t²) = -a r² sin 2φ` for `t = r(cos φ + i sin φ)`. -/
theorem re_I_mul_sq (a r φ : ℝ) :
    (I * a * ((r : ℂ) * (Real.cos φ + Real.sin φ * I)) ^ 2).re = -a * r ^ 2 * Real.sin (2 * φ) := by
  have h2 : Real.sin (2 * φ) = 2 * Real.sin φ * Real.cos φ := Real.sin_two_mul φ
  simp only [sq, Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.add_re, Complex.add_im]
  generalize Real.sin (2 * φ) = S2 at h2 ⊢
  generalize Real.sin φ = sn at h2 ⊢
  generalize Real.cos φ = cs at h2 ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (a * r ^ 2) * h2)
  -- certificate: ideal_membership.sage

/-- In the sector `0 ≤ arg t ≤ π/4`, `Re(iat²) = -a r² sin 2φ ≤ 0` (`a ≥ 0`): the arcs vanish there. -/
theorem re_I_mul_sq_nonpos (a r φ : ℝ) (ha : 0 ≤ a) (h0 : 0 ≤ φ) (h1 : φ ≤ π / 4) :
    -a * r ^ 2 * Real.sin (2 * φ) ≤ 0 := by
  have hs : 0 ≤ Real.sin (2 * φ) := Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith [Real.pi_pos])
  have : 0 ≤ a * r ^ 2 * Real.sin (2 * φ) := by positivity
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.re_I_mul_sq_nonpos
-- test: ca-spec-sector-test.sage

/-- The Gaussian with complex `λ = p + iq`, `Re λ = p > 0`: `∫ e^{-λx²/2} dx = (2π/λ)^{1/2}` (principal branch), with
`2π/λ = (2π/(p² + q²))(p - iq)`. -/
theorem gaussian_complex (p q : ℝ) (hp : 0 < p) :
    ∫ x : ℝ, Complex.exp (-(1 / 2) * (p + q * I) * (x : ℂ) ^ 2) =
      (((2 * π / (p ^ 2 + q ^ 2) : ℝ) : ℂ) * (p - q * I)) ^ (1 / 2 : ℂ) := by
  have hb : 0 < (((p : ℂ) + q * I) / 2).re := by simp; linarith
  have h := integral_gaussian_complex hb
  have e : ∀ x : ℝ, -(1 / 2) * ((p : ℂ) + q * I) * (x : ℂ) ^ 2 = -(((p : ℂ) + q * I) / 2) * (x : ℂ) ^ 2 :=
    fun x => by ring
  simp_rw [e]
  rw [h]
  congr 1
  have hne : (p : ℂ) + q * I ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0
    simp at this
    linarith
  have hne' : (p : ℂ) - q * I ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0
    simp at this
    linarith
  have hd : p ^ 2 + q ^ 2 ≠ 0 := by positivity
  have hd' : (((p ^ 2 + q ^ 2 : ℝ)) : ℂ) ≠ 0 := by exact_mod_cast hd
  have hprod : ((p : ℂ) + q * I) * (p - q * I) = ((p ^ 2 + q ^ 2 : ℝ) : ℂ) := by
    push_cast
    linear_combination (-(q : ℂ) ^ 2) * Complex.I_sq
  calc (π : ℂ) / (((p : ℂ) + q * I) / 2) = 2 * π * (p - q * I) / (((p : ℂ) + q * I) * (p - q * I)) := by
        field_simp
    _ = (((2 * π / (p ^ 2 + q ^ 2) : ℝ) : ℂ) * (p - q * I)) := by
        rw [hprod]
        push_cast
        ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.WickRotation.gaussian_complex
-- test: ca-spec-gauss-c-test.sage

/-- `λ = -i(a + iε) = ε - ia`. -/
theorem lambda_eq (a ε : ℂ) : -I * (a + ε * I) = ε - a * I := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  generalize I = j at hI ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-ε) * hI)
  -- certificate: ideal_membership.sage

/-- The Gaussian with an `iε`: `∫ e^{(i/2)(a + iε)x²} dx = (2π/(ε - ia))^{1/2}` for `ε > 0` (absolutely convergent),
with `2π/(ε - ia) = (2π/(ε² + a²))(ε + ia)`. -/
theorem gaussian_i_eps (a ε : ℝ) (hε : 0 < ε) :
    ∫ x : ℝ, Complex.exp (I / 2 * (a + ε * I) * (x : ℂ) ^ 2) =
      (((2 * π / (ε ^ 2 + a ^ 2) : ℝ) : ℂ) * (ε + a * I)) ^ (1 / 2 : ℂ) := by
  have h := gaussian_complex ε (-a) hε
  have e : ∀ x : ℝ, I / 2 * (a + ε * I) * (x : ℂ) ^ 2 = -(1 / 2) * (ε + ((-a : ℝ) : ℂ) * I) * (x : ℂ) ^ 2 :=
    fun x => by
      push_cast
      linear_combination ((ε : ℂ) / 2 * (x : ℂ) ^ 2) * Complex.I_sq
  simp_rw [e]
  rw [h]
  congr 2 <;> push_cast <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.gaussian_i_eps
-- test: ca-spec-gauss-ieps-test.sage

/-- `|e^{(i/2)(a + iε)x²}| = e^{-εx²/2}`: the `iε` makes the integral absolutely convergent. -/
theorem norm_gaussian_i_eps (a ε x : ℝ) :
    ‖Complex.exp (I / 2 * (a + ε * I) * (x : ℂ) ^ 2)‖ = Real.exp (-(ε * x ^ 2) / 2) := by
  rw [Complex.norm_exp]
  congr 1
  simp [sq]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.norm_gaussian_i_eps
-- test: ca-spec-gauss-abs-test.sage

/-- `(ci)^{1/2} = √c e^{iπ/4}` for `c > 0`. -/
theorem cpow_half_mul_I (c : ℝ) (hc : 0 < c) : ((c : ℂ) * I) ^ (1 / 2 : ℂ) = (√c : ℝ) * Complex.exp (π * I / 4) := by
  have hne : (c : ℂ) * I ≠ 0 := mul_ne_zero (by exact_mod_cast hc.ne') Complex.I_ne_zero
  have hlog : Complex.log ((c : ℂ) * I) = (Real.log c : ℂ) + (π / 2 : ℝ) * I := by
    rw [Complex.log, Complex.arg_real_mul I hc, Complex.arg_I, norm_mul, Complex.norm_I, mul_one,
      Complex.norm_of_nonneg hc.le]
  rw [Complex.cpow_def_of_ne_zero hne, hlog]
  have hs : ((√c : ℝ) : ℂ) = Complex.exp ((Real.log c : ℂ) / 2) := by
    rw [← Complex.ofReal_ofNat, ← Complex.ofReal_div, ← Complex.ofReal_exp, Real.sqrt_eq_rpow,
      Real.rpow_def_of_pos hc]
    congr 2
    ring
  rw [hs, ← Complex.exp_add]
  congr 1
  push_cast <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.cpow_half_mul_I
-- test: ca-spec-sqrt-i-test.sage

/-- As `ε → 0⁺`, `(2π/(ε - ia))^{1/2} → √(2π/a) e^{iπ/4}` (`a > 0`): the Fresnel value, principal branch. -/
theorem gaussian_i_eps_limit (a : ℝ) (ha : 0 < a) :
    Tendsto (fun ε : ℝ => (2 * π / (ε - a * I)) ^ (1 / 2 : ℂ)) (𝓝[>] 0)
      (𝓝 ((√(2 * π / a) : ℝ) * Complex.exp (π * I / 4))) := by
  have ha' : (a : ℂ) ≠ 0 := by exact_mod_cast ha.ne'
  have hz : (2 * π / ((0 : ℝ) - a * I) : ℂ) = ((2 * π / a : ℝ) : ℂ) * I := by
    push_cast
    field_simp
    linear_combination (a : ℂ) * Complex.I_sq
  have hslit : ((2 * π / a : ℝ) : ℂ) * I ∈ Complex.slitPlane := by
    rw [Complex.mem_slitPlane_iff]
    right
    simp
    positivity
  have hc : ContinuousAt (fun ε : ℝ => (2 * π / (ε - a * I) : ℂ)) 0 := by
    apply ContinuousAt.div (by fun_prop) (by fun_prop)
    simp [ha']
  have h := ((continuousAt_cpow_const (b := (1 / 2 : ℂ)) (hz ▸ hslit)).comp hc).tendsto
  rw [Function.comp_apply, hz, cpow_half_mul_I _ (by positivity)] at h
  exact h.mono_left nhdsWithin_le_nhds

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.gaussian_i_eps_limit
-- test: ca-spec-ieps-limit-test.sage

/-- The real Gaussian after the rotation: `∫ e^{-au²/2} du = √(2π/a)` for `a > 0`. -/
theorem gaussian_real (a : ℝ) (ha : 0 < a) : ∫ u : ℝ, Real.exp (-(a / 2) * u ^ 2) = √(2 * π / a) := by
  rw [integral_gaussian]
  congr 1
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.gaussian_real
-- test: ca-spec-gauss-real-test.sage

/-- `e^{iπ/4}√π = √(π/2)(1 + i)`: real and imaginary parts are the Fresnel integrals `∫ cos t² = ∫ sin t² = √(π/2)`. -/
theorem fresnel_value : Complex.exp (π * I / 4) * (√π : ℝ) = (√(π / 2) : ℝ) * (1 + I) := by
  have he : Complex.exp (π * I / 4) = ((√2 / 2 : ℝ) : ℂ) * (1 + I) := by
    rw [show (π : ℂ) * I / 4 = ((π / 4 : ℝ) : ℂ) * I by push_cast <;> ring, Complex.exp_mul_I]
    rw [← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four]
    ring
  have hs : √(π / 2) * √2 = √π := by
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    ring
  have h2 : √2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs' : ((√(π / 2) : ℝ) : ℂ) * ((√2 : ℝ) : ℂ) = ((√π : ℝ) : ℂ) := by exact_mod_cast hs
  have h2' : ((√2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast h2
  have hh : ((√2 / 2 : ℝ) : ℂ) * 2 = ((√2 : ℝ) : ℂ) := by push_cast; ring
  rw [he]
  generalize ((√2 / 2 : ℝ) : ℂ) = h at hh ⊢
  generalize ((√π : ℝ) : ℂ) = a at hs' ⊢
  generalize ((√(π / 2) : ℝ) : ℂ) = c at hs' ⊢
  generalize ((√2 : ℝ) : ℂ) = b at hs' h2' hh
  generalize I = j
  clear he hs h2
  rw [← sub_eq_zero]
  veriphy_certified
    (linear_combination
        (-1 / 2 * j * b - 1 / 2 * b) * hs' + (1 / 2 * j * c + 1 / 2 * c) * h2' + (1 / 2 * j * a + 1 / 2 * a) * hh)
  -- certificate: ideal_membership.sage

/-- The Fresnel integral as the `iε`-regularised limit: `∫ e^{i(a + iε)t²} dt → e^{iπ/4}√(π/a)` as `ε → 0⁺` (`a > 0`),
from the complex Gaussian `gaussian_i_eps` at `(2a, 2ε)` and its limit `gaussian_i_eps_limit`. (The improper Riemann
integral `lim_{R→∞} ∫_{-R}^{R} e^{iat²} dt` is not in mathlib.) -/
theorem fresnel_regularised (a : ℝ) (ha : 0 < a) :
    Tendsto (fun ε : ℝ => ∫ t : ℝ, Complex.exp (I * (a + ε * I) * (t : ℂ) ^ 2)) (𝓝[>] 0)
      (𝓝 (Complex.exp (π * I / 4) * (√(π / a) : ℝ))) := by
  have h2 : Tendsto (fun ε : ℝ => 2 * ε) (𝓝[>] 0) (𝓝[>] 0) := by
    refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
    · have h : Tendsto (fun ε : ℝ => 2 * ε) (𝓝 0) (𝓝 (2 * 0)) := (continuous_const.mul continuous_id).tendsto 0
      rw [mul_zero] at h
      exact h.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with ε hε
      exact Set.mem_Ioi.mpr (mul_pos two_pos (Set.mem_Ioi.mp hε))
  have h := (gaussian_i_eps_limit (2 * a) (by positivity)).comp h2
  have hl : ((√(2 * π / (2 * a)) : ℝ) : ℂ) * Complex.exp (π * I / 4) =
      Complex.exp (π * I / 4) * ((√(π / a) : ℝ) : ℂ) := by
    rw [mul_div_mul_left π a two_ne_zero, mul_comm]
  rw [hl] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with ε hε
  have hε' : 0 < ε := hε
  have e : ∀ t : ℂ, I * (a + ε * I) * t ^ 2 = I / 2 * (((2 * a : ℝ) : ℂ) + ((2 * ε : ℝ) : ℂ) * I) * t ^ 2 := by
    intro t
    push_cast
    ring
  simp only [Function.comp_apply]
  simp_rw [e]
  rw [gaussian_i_eps (2 * a) (2 * ε) (by positivity)]
  congr 1
  have hD : ((2 * ε) ^ 2 + (2 * a) ^ 2 : ℝ) ≠ 0 := by positivity
  have hD' : (((2 * ε) ^ 2 + (2 * a) ^ 2 : ℝ) : ℂ) ≠ 0 := by exact_mod_cast hD
  have hne : (((2 * ε : ℝ) : ℂ) - ((2 * a : ℝ) : ℂ) * I) ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0
    simp at this
    linarith
  have hm : (((2 * ε : ℝ) : ℂ) + ((2 * a : ℝ) : ℂ) * I) * (((2 * ε : ℝ) : ℂ) - ((2 * a : ℝ) : ℂ) * I) =
      (((2 * ε) ^ 2 + (2 * a) ^ 2 : ℝ) : ℂ) := by
    push_cast
    linear_combination (-(2 * (a : ℂ)) ^ 2) * Complex.I_sq
  rw [div_eq_iff hne, Complex.ofReal_div, mul_assoc, hm, div_mul_cancel₀ _ hD']
  first | (push_cast; done) | (push_cast; rfl) | (push_cast; ring) | (norm_num; done) | (norm_num; ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.fresnel_regularised
-- test: ca-spec-fresnel-test.sage

/-! ## exm-wick-action, exr-wick-euclidean-action -/

/-- `t = -iτ`: if `X(t) = x(it)` then `dX/dt = i x'(it)`, i.e. `ẋ = i dx/dτ`. -/
theorem velocity_rotation (x : ℂ → ℂ) (hx : Differentiable ℂ x) (t : ℂ) :
    HasDerivAt (fun t => x (I * t)) (I * deriv x (I * t)) t := by
  have h := (hx (I * t)).hasDerivAt.comp t ((hasDerivAt_id t).const_mul I)
  have e : I * deriv x (I * t) = deriv x (I * t) * (I * 1) := by ring
  rw [e]
  exact h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.velocity_rotation (id) (differentiable_id) (0) := by
  have h := (differentiable_id (Complex.I * (0 : ℂ))).hasDerivAt.comp (0 : ℂ) ((hasDerivAt_id (0 : ℂ)).const_mul Complex.I)
  have e : Complex.I * deriv (id) (Complex.I * (0 : ℂ)) = deriv (id) (Complex.I * (0 : ℂ)) * (Complex.I * 1) := by ring
  rw [e]
  exact h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.velocity_rotation (Complex.exp) (Complex.differentiable_exp) (1) := by
  have h := (Complex.differentiable_exp (Complex.I * (1 : ℂ))).hasDerivAt.comp (1 : ℂ) ((hasDerivAt_id (1 : ℂ)).const_mul Complex.I)
  have e : Complex.I * deriv (Complex.exp) (Complex.I * (1 : ℂ)) = deriv (Complex.exp) (Complex.I * (1 : ℂ)) * (Complex.I * 1) := by ring
  rw [e]
  exact h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.velocity_rotation (fun z : ℂ => z ^ 2) ((by fun_prop : Differentiable ℂ (fun z : ℂ => z ^ 2))) (Complex.I) := by
  have h := ((by fun_prop : Differentiable ℂ (fun z : ℂ => z ^ 2)) (Complex.I * Complex.I)).hasDerivAt.comp Complex.I ((hasDerivAt_id Complex.I).const_mul Complex.I)
  have e : Complex.I * deriv (fun z : ℂ => z ^ 2) (Complex.I * Complex.I) = deriv (fun z : ℂ => z ^ 2) (Complex.I * Complex.I) * (Complex.I * 1) := by ring
  rw [e]
  exact h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.WickRotation.velocity_rotation
-- test: ca-spec-velocity-test.sage

/-- The particle: `i · (-i dτ) · (½ m (i x')² - V) = -(½ m x'² + V)`, i.e. `iS = -S_E`. -/
theorem particle_rotation (m V xp : ℂ) : I * (-I) * (1 / 2 * m * (I * xp) ^ 2 - V) = -(1 / 2 * m * xp ^ 2 + V) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  generalize I = j at hI ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1 / 2 * j ^ 2 * m * xp ^ 2 + 1 / 2 * m * xp ^ 2 + V) * hI)
  -- certificate: ideal_membership.sage

/-- The field: `i · (-i) · (½(i ∂_τφ)² - ½(∇φ)² - ½m²φ²) = -½((∂_τφ)² + (∇φ)² + m²φ²)`. -/
theorem field_rotation (φτ g m φ : ℂ) :
    I * (-I) * (1 / 2 * (I * φτ) ^ 2 - 1 / 2 * g ^ 2 - 1 / 2 * m ^ 2 * φ ^ 2) =
      -(1 / 2 * (φτ ^ 2 + g ^ 2 + m ^ 2 * φ ^ 2)) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  generalize I = j at hI ⊢
  rw [← sub_eq_zero]
  veriphy_certified
    (linear_combination (-1 / 2 * j ^ 2 * φτ ^ 2 + 1 / 2 * φτ ^ 2 + 1 / 2 * g ^ 2 + 1 / 2 * m ^ 2 * φ ^ 2) * hI)
  -- certificate: ideal_membership.sage

/-- The Euclidean Lagrangian of the particle is non-negative for `m ≥ 0`, `V ≥ 0`. -/
theorem euclidean_particle_nonneg (m xp V : ℝ) (hm : 0 ≤ m) (hV : 0 ≤ V) : 0 ≤ 1 / 2 * m * xp ^ 2 + V := by
  positivity

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_particle_nonneg (0) (0) (0) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_particle_nonneg (1) (2) (3) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_particle_nonneg (1 / 2) (-1) (1 / 4) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_particle_nonneg
-- test: ca-spec-euclid-particle-test.sage

/-- The Euclidean Lagrangian of the field is non-negative: a sum of squares. -/
theorem euclidean_field_nonneg (φτ g m φ : ℝ) : 0 ≤ 1 / 2 * (φτ ^ 2 + g ^ 2 + m ^ 2 * φ ^ 2) := by
  positivity

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_field_nonneg (0) (0) (0) (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_field_nonneg (1) (-2) (3) (1 / 2) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_field_nonneg (-1) (0) (2) (-3) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.WickRotation.euclidean_field_nonneg
-- test: ca-spec-euclid-field-test.sage

/-! ## exr-wick-arcs -/

/-- On `|k| = R`: `|k² - E² + iε| ≥ R² - E² - ε` (`ε ≥ 0`). -/
theorem arc_denominator_bound (E ε : ℝ) (hε : 0 ≤ ε) (k : ℂ) :
    ‖k‖ ^ 2 - E ^ 2 - ε ≤ ‖k ^ 2 - E ^ 2 + ε * I‖ := by
  have h1 : ‖k ^ 2‖ ≤ ‖k ^ 2 - E ^ 2 + ε * I‖ + ‖(E : ℂ) ^ 2 - ε * I‖ := by
    have := norm_add_le (k ^ 2 - E ^ 2 + ε * I) ((E : ℂ) ^ 2 - ε * I)
    rwa [show k ^ 2 - (E : ℂ) ^ 2 + ε * I + ((E : ℂ) ^ 2 - ε * I) = k ^ 2 by ring] at this
  have h2 : ‖(E : ℂ) ^ 2 - ε * I‖ ≤ E ^ 2 + ε := by
    calc ‖(E : ℂ) ^ 2 - ε * I‖ ≤ ‖(E : ℂ) ^ 2‖ + ‖(ε : ℂ) * I‖ := norm_sub_le _ _
      _ = E ^ 2 + ε := by
        rw [norm_pow, norm_mul, Complex.norm_I, mul_one, Complex.norm_real, Complex.norm_real,
          Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, abs_of_nonneg hε]
  rw [norm_pow] at h1
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.WickRotation.arc_denominator_bound
-- test: ca-spec-arc-den-test.sage

/-- The ML estimate on the arc `k = Re^{iθ}`, `0 ≤ θ ≤ π/2`: the integral is at most `πR/(2(R² - E² - ε))`. -/
theorem arc_integral_bound (E ε R : ℝ) (hε : 0 ≤ ε) (hR : E ^ 2 + ε < R ^ 2) (hR0 : 0 < R) :
    ‖∫ θ in (0 : ℝ)..(π / 2), 1 / ((R * Complex.exp (θ * I)) ^ 2 - E ^ 2 + ε * I) * (I * R * Complex.exp (θ * I))‖ ≤
      π * R / (2 * (R ^ 2 - E ^ 2 - ε)) := by
  have hpos : 0 < R ^ 2 - E ^ 2 - ε := by linarith
  have hbound : ∀ θ ∈ Set.uIoc (0 : ℝ) (π / 2),
      ‖1 / ((R * Complex.exp (θ * I)) ^ 2 - E ^ 2 + ε * I) * (I * R * Complex.exp (θ * I))‖ ≤
        R / (R ^ 2 - E ^ 2 - ε) := by
    intro θ _
    have hk : ‖(R : ℂ) * Complex.exp (θ * I)‖ = R := by
      rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hR0]
    have hd := arc_denominator_bound E ε hε ((R : ℂ) * Complex.exp (θ * I))
    rw [hk] at hd
    have hd0 : 0 < ‖((R : ℂ) * Complex.exp (θ * I)) ^ 2 - E ^ 2 + ε * I‖ := by linarith
    rw [norm_mul, norm_div, norm_one, norm_mul, norm_mul, Complex.norm_I, Complex.norm_exp_ofReal_mul_I,
      Complex.norm_real, Real.norm_eq_abs, abs_of_pos hR0, one_mul, mul_one, one_div_mul_eq_div]
    exact div_le_div_of_nonneg_left hR0.le hpos hd
  have h := intervalIntegral.norm_integral_le_of_norm_le_const hbound
  rw [sub_zero, abs_of_pos (by positivity : (0 : ℝ) < π / 2)] at h
  calc _ ≤ R / (R ^ 2 - E ^ 2 - ε) * (π / 2) := h
    _ = π * R / (2 * (R ^ 2 - E ^ 2 - ε)) := by field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.arc_integral_bound
-- test: ca-spec-arc-ml-test.sage

/-- The bound `πR/(2(R² - M))` tends to `0` as `R → ∞`. -/
theorem arc_bound_tendsto (M : ℝ) : Tendsto (fun R : ℝ => π * R / (2 * (R ^ 2 - M))) atTop (𝓝 0) := by
  have h1 : Tendsto (fun R : ℝ => π / 2 * (R⁻¹ / (1 - M * R⁻¹ ^ 2))) atTop (𝓝 (π / 2 * (0 / (1 - M * 0 ^ 2)))) := by
    refine Tendsto.const_mul _ (Tendsto.div tendsto_inv_atTop_zero ?_ (by simp))
    exact tendsto_const_nhds.sub ((tendsto_inv_atTop_zero.pow 2).const_mul M)
  simp only [zero_div, mul_zero] at h1
  refine h1.congr' ?_
  filter_upwards [eventually_gt_atTop (max 0 (|M| + 1)).sqrt] with R hR
  have hR0 : 0 < R := lt_of_le_of_lt (Real.sqrt_nonneg _) hR
  have hR2 : M < R ^ 2 := by
    have := (Real.sqrt_lt' hR0).mp hR
    have : |M| + 1 ≤ max 0 (|M| + 1) := le_max_right _ _
    linarith [le_abs_self M]
  have h3 : R ^ 2 - M ≠ 0 := by linarith
  have h5 : M * R⁻¹ ^ 2 < 1 := by
    rw [inv_pow, ← div_eq_mul_inv, div_lt_one (by positivity)]
    exact hR2
  have h4 : 1 - M * R⁻¹ ^ 2 ≠ 0 := (by linarith : (0 : ℝ) < 1 - M * R⁻¹ ^ 2).ne'
  have hR' : R ≠ 0 := hR0.ne'
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.arc_bound_tendsto
-- test: ca-spec-arc-limit-test.sage

/-- `|e^{-ik⁰t}| = e^{t Im k⁰}`. -/
theorem norm_exp_factor (k : ℂ) (t : ℝ) : ‖Complex.exp (-I * k * t)‖ = Real.exp (t * k.im) := by
  rw [Complex.norm_exp]
  congr 1
  simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.norm_exp_factor
-- test: ca-spec-exp-factor-test.sage

/-- On the upper arc (`Im k ≥ 0`) the factor `e^{-ik⁰t}` is bounded by `1` for `t ≤ 0`. -/
theorem exp_factor_bounded (k : ℂ) (t : ℝ) (ht : t ≤ 0) (hk : 0 ≤ k.im) : ‖Complex.exp (-I * k * t)‖ ≤ 1 := by
  rw [norm_exp_factor, Real.exp_le_one_iff]
  nlinarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.exp_factor_bounded
-- test: ca-spec-exp-bounded-test.sage

/-- For `t > 0` it is unbounded on the upper arc: at `k = iR`, `|e^{-ik t}| = e^{Rt} → ∞`. -/
theorem exp_factor_unbounded (t : ℝ) (ht : 0 < t) :
    Tendsto (fun R : ℝ => ‖Complex.exp (-I * (R * I) * t)‖) atTop atTop := by
  simp_rw [norm_exp_factor]
  simp only [Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im, mul_one,
    zero_mul, add_zero]
  exact Real.tendsto_exp_atTop.comp (tendsto_id.const_mul_atTop ht)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.WickRotation.exp_factor_unbounded
-- test: ca-spec-exp-unbounded-test.sage

end KaluzaKlein.ComplexAnalysis.WickRotation

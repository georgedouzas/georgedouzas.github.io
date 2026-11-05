import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.ResidueTheorem
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Series and residues: worked computations"

The claims of the worked computations of the part "Series and residues" of
`tutorial/part-i-04-complex-analysis.qmd`: the integrals `∫ dx/(a⁴ + x⁴) = π/(√2 a³)` and
`∫ x² dx/(a⁴ + x⁴) = π/(√2 a)` (by explicit antiderivatives and mathlib's improper integrals), the arc estimates of
the rational and Fourier methods (Jordan's inequality `∫₀^π e^{-R sin θ} dθ ≤ π/R`), the residues in the lower
half-plane, the substitution `z = e^{iθ}` and `∫₀^{2π} dθ/(a + cos θ) = 2π/√(a² - 1)` (by the residue theorem on
the unit circle), the residues of `z²/(1 + z⁴)`, and the indentation computations for `e^{iz}/z`.

The Dirichlet integral `∫ sin x/x dx = π` and Cauchy's theorem for the indented contour are not in mathlib and are
not formalised; the small- and large-arc limits are.
-/

namespace KaluzaKlein.ComplexAnalysis.ResidueWorked

open Complex Filter Topology Metric MeasureTheory
open scoped Real Interval

open KaluzaKlein.ComplexAnalysis.Residues KaluzaKlein.ComplexAnalysis.ResidueTheorem

/-! ## prp-residue-integrals: `∫ dx/(1 + x⁴) = π/√2` -/

lemma sqrt_two_sq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)

lemma quad_pos (x : ℝ) : 0 < x ^ 2 + Real.sqrt 2 * x + 1 ∧ 0 < x ^ 2 - Real.sqrt 2 * x + 1 := by
  have hs := sqrt_two_sq
  constructor <;> nlinarith [sq_nonneg (x + Real.sqrt 2 / 2), sq_nonneg (x - Real.sqrt 2 / 2)]

lemma quad_prod (x : ℝ) : (x ^ 2 + Real.sqrt 2 * x + 1) * (x ^ 2 - Real.sqrt 2 * x + 1) = 1 + x ^ 4 := by
  have hs := sqrt_two_sq
  linear_combination (-x ^ 2) * hs

/-- The antiderivative `G = (arctan(√2x + 1) + arctan(√2x - 1))/√2` of `(1 + x²)/(1 + x⁴)`. -/
noncomputable def G (x : ℝ) : ℝ := (Real.arctan (Real.sqrt 2 * x + 1) + Real.arctan (Real.sqrt 2 * x - 1)) / Real.sqrt 2

/-- The antiderivative `H = log((x² + √2x + 1)/(x² - √2x + 1))/(2√2)` of `(1 - x²)/(1 + x⁴)`. -/
noncomputable def H (x : ℝ) : ℝ :=
  Real.log ((x ^ 2 + Real.sqrt 2 * x + 1) / (x ^ 2 - Real.sqrt 2 * x + 1)) / (2 * Real.sqrt 2)

lemma hasDerivAt_G (x : ℝ) : HasDerivAt G ((1 + x ^ 2) / (1 + x ^ 4)) x := by
  have hs := sqrt_two_sq
  have hs0 : Real.sqrt 2 ≠ 0 := by positivity
  have h1 := ((hasDerivAt_id x).const_mul (Real.sqrt 2)).add_const 1 |>.arctan
  have h2 := ((hasDerivAt_id x).const_mul (Real.sqrt 2)).sub_const 1 |>.arctan
  have h : HasDerivAt G ((1 / (1 + (Real.sqrt 2 * x + 1) ^ 2) * (Real.sqrt 2 * 1) +
      1 / (1 + (Real.sqrt 2 * x - 1) ^ 2) * (Real.sqrt 2 * 1)) / Real.sqrt 2) x :=
    (h1.add h2).div_const _
  refine h.congr_deriv ?_
  obtain ⟨hp, hm⟩ := quad_pos x
  have e1 : 1 + (Real.sqrt 2 * x + 1) ^ 2 = 2 * (x ^ 2 + Real.sqrt 2 * x + 1) := by
    linear_combination x ^ 2 * hs
  have e2 : 1 + (Real.sqrt 2 * x - 1) ^ 2 = 2 * (x ^ 2 - Real.sqrt 2 * x + 1) := by
    linear_combination x ^ 2 * hs
  have hp' := hp.ne'
  have hm' := hm.ne'
  rw [e1, e2, ← quad_prod x]
  have hsum : 1 + x ^ 2 = ((x ^ 2 + Real.sqrt 2 * x + 1) + (x ^ 2 - Real.sqrt 2 * x + 1)) / 2 := by ring
  rw [hsum]
  generalize x ^ 2 + Real.sqrt 2 * x + 1 = P at hp' ⊢
  generalize x ^ 2 - Real.sqrt 2 * x + 1 = M at hm' ⊢
  generalize Real.sqrt 2 = t at hs0 ⊢
  field_simp
  ring

lemma hasDerivAt_H (x : ℝ) : HasDerivAt H ((1 - x ^ 2) / (1 + x ^ 4)) x := by
  have hs := sqrt_two_sq
  have hs0 : Real.sqrt 2 ≠ 0 := by positivity
  obtain ⟨hp, hm⟩ := quad_pos x
  have hN : HasDerivAt (fun x => x ^ 2 + Real.sqrt 2 * x + 1) (2 * x + Real.sqrt 2) x :=
    (((hasDerivAt_pow 2 x).add ((hasDerivAt_id x).const_mul (Real.sqrt 2))).add_const 1).congr_deriv
      (by norm_num)
  have hD : HasDerivAt (fun x => x ^ 2 - Real.sqrt 2 * x + 1) (2 * x - Real.sqrt 2) x :=
    (((hasDerivAt_pow 2 x).sub ((hasDerivAt_id x).const_mul (Real.sqrt 2))).add_const 1).congr_deriv
      (by norm_num)
  have h : HasDerivAt H ((((2 * x + Real.sqrt 2) * (x ^ 2 - Real.sqrt 2 * x + 1) -
      (x ^ 2 + Real.sqrt 2 * x + 1) * (2 * x - Real.sqrt 2)) / (x ^ 2 - Real.sqrt 2 * x + 1) ^ 2) /
      ((x ^ 2 + Real.sqrt 2 * x + 1) / (x ^ 2 - Real.sqrt 2 * x + 1)) / (2 * Real.sqrt 2)) x :=
    ((hN.div hD hm.ne').log (div_pos hp hm).ne').div_const (2 * Real.sqrt 2)
  refine h.congr_deriv ?_
  rw [← quad_prod x]
  field_simp
  ring

lemma tendsto_G_atTop : Tendsto G atTop (𝓝 (π / Real.sqrt 2)) := by
  have hs : (0 : ℝ) < Real.sqrt 2 := by positivity
  have h1 : Tendsto (fun x => Real.sqrt 2 * x + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 (tendsto_id.const_mul_atTop hs)
  have h2 : Tendsto (fun x => Real.sqrt 2 * x - 1) atTop atTop :=
    tendsto_atTop_add_const_right _ (-1) (tendsto_id.const_mul_atTop hs) |>.congr fun x => by
      simp only [id]; ring
  have a1 := (tendsto_nhds_of_tendsto_nhdsWithin Real.tendsto_arctan_atTop).comp h1
  have a2 := (tendsto_nhds_of_tendsto_nhdsWithin Real.tendsto_arctan_atTop).comp h2
  have := (a1.add a2).div_const (Real.sqrt 2)
  rw [show (π / 2 + π / 2) / Real.sqrt 2 = π / Real.sqrt 2 by ring] at this
  exact this

lemma tendsto_G_atBot : Tendsto G atBot (𝓝 (-(π / Real.sqrt 2))) := by
  have hs : (0 : ℝ) < Real.sqrt 2 := by positivity
  have h1 : Tendsto (fun x => Real.sqrt 2 * x + 1) atBot atBot :=
    tendsto_atBot_add_const_right _ 1 (tendsto_id.const_mul_atBot hs)
  have h2 : Tendsto (fun x => Real.sqrt 2 * x - 1) atBot atBot :=
    tendsto_atBot_add_const_right _ (-1) (tendsto_id.const_mul_atBot hs) |>.congr fun x => by
      simp only [id]; ring
  have a1 := (tendsto_nhds_of_tendsto_nhdsWithin Real.tendsto_arctan_atBot).comp h1
  have a2 := (tendsto_nhds_of_tendsto_nhdsWithin Real.tendsto_arctan_atBot).comp h2
  have := (a1.add a2).div_const (Real.sqrt 2)
  rw [show (-(π / 2) + -(π / 2)) / Real.sqrt 2 = -(π / Real.sqrt 2) by ring] at this
  exact this

lemma tendsto_ratio {l : Filter ℝ} (hl : Tendsto (fun x : ℝ => x⁻¹) l (𝓝 0)) (hne : ∀ᶠ x in l, x ≠ 0) :
    Tendsto (fun x => (x ^ 2 + Real.sqrt 2 * x + 1) / (x ^ 2 - Real.sqrt 2 * x + 1)) l (𝓝 1) := by
  have hn : Tendsto (fun x : ℝ => 1 + Real.sqrt 2 * x⁻¹ + (x⁻¹) ^ 2) l (𝓝 (1 + Real.sqrt 2 * 0 + 0 ^ 2)) :=
    (tendsto_const_nhds.add (hl.const_mul _)).add (hl.pow 2)
  have hd : Tendsto (fun x : ℝ => 1 - Real.sqrt 2 * x⁻¹ + (x⁻¹) ^ 2) l (𝓝 (1 - Real.sqrt 2 * 0 + 0 ^ 2)) :=
    (tendsto_const_nhds.sub (hl.const_mul _)).add (hl.pow 2)
  simp only [mul_zero, add_zero, sub_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow] at hn hd
  have := hn.div hd one_ne_zero
  rw [div_one] at this
  refine this.congr' ?_
  filter_upwards [hne] with x hx
  obtain ⟨hp, hm⟩ := quad_pos x
  simp only [Pi.div_apply]
  field_simp

lemma tendsto_H (l : Filter ℝ) (hl : Tendsto (fun x : ℝ => x⁻¹) l (𝓝 0)) (hne : ∀ᶠ x in l, x ≠ 0) :
    Tendsto H l (𝓝 0) := by
  have h := (Real.continuousAt_log one_ne_zero).tendsto.comp (tendsto_ratio hl hne)
  rw [Real.log_one] at h
  have := h.div_const (2 * Real.sqrt 2)
  rw [zero_div] at this
  exact this

lemma integrable_bound : Integrable (fun x : ℝ => 2 * (1 + x ^ 2)⁻¹) := integrable_inv_one_add_sq.const_mul 2

lemma bound_aux (x : ℝ) : (1 + x ^ 2) / (1 + x ^ 4) ≤ 2 * (1 + x ^ 2)⁻¹ := by
  have h1 : 0 < 1 + x ^ 2 := by positivity
  have h2 : 0 < 1 + x ^ 4 := by positivity
  rw [← div_eq_mul_inv, div_le_div_iff₀ h2 h1]
  nlinarith [sq_nonneg (1 - x ^ 2)]

lemma integrable_plus : Integrable (fun x : ℝ => (1 + x ^ 2) / (1 + x ^ 4)) := by
  refine Integrable.mono' integrable_bound (Continuous.aestronglyMeasurable
    (Continuous.div (by fun_prop) (by fun_prop) fun x => by positivity)) (Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_pos (by positivity)]; exact bound_aux x

lemma integrable_minus : Integrable (fun x : ℝ => (1 - x ^ 2) / (1 + x ^ 4)) := by
  refine Integrable.mono' integrable_bound (Continuous.aestronglyMeasurable
    (Continuous.div (by fun_prop) (by fun_prop) fun x => by positivity)) (Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 1 + x ^ 4)]
  refine le_trans ?_ (bound_aux x)
  gcongr
  rw [abs_le]; constructor <;> nlinarith [sq_nonneg x]

lemma integral_plus : ∫ x : ℝ, (1 + x ^ 2) / (1 + x ^ 4) = 2 * (π / Real.sqrt 2) := by
  rw [integral_of_hasDerivAt_of_tendsto hasDerivAt_G integrable_plus tendsto_G_atBot tendsto_G_atTop]; ring

lemma integral_minus : ∫ x : ℝ, (1 - x ^ 2) / (1 + x ^ 4) = 0 := by
  rw [integral_of_hasDerivAt_of_tendsto hasDerivAt_H integrable_minus
    (tendsto_H atBot tendsto_inv_atBot_zero (eventually_ne_atBot 0))
    (tendsto_H atTop tendsto_inv_atTop_zero (eventually_ne_atTop 0))]; ring

lemma integral_one_fourth : ∫ x : ℝ, 1 / (1 + x ^ 4) = π / Real.sqrt 2 := by
  have h : (fun x : ℝ => 1 / (1 + x ^ 4)) =
      fun x => ((1 + x ^ 2) / (1 + x ^ 4) + (1 - x ^ 2) / (1 + x ^ 4)) / 2 := by
    funext x; field_simp; ring
  rw [h, integral_div, integral_add integrable_plus integrable_minus, integral_plus, integral_minus]; ring

lemma integral_sq_fourth : ∫ x : ℝ, x ^ 2 / (1 + x ^ 4) = π / Real.sqrt 2 := by
  have h : (fun x : ℝ => x ^ 2 / (1 + x ^ 4)) =
      fun x => ((1 + x ^ 2) / (1 + x ^ 4) - (1 - x ^ 2) / (1 + x ^ 4)) / 2 := by
    funext x; field_simp; ring
  rw [h, integral_div, integral_sub integrable_plus integrable_minus, integral_plus, integral_minus]; ring

/-- `∫ dx/(a⁴ + x⁴) = π/(√2 a³)` for `a > 0` (the text: `a = 1`, `π/√2`). -/
theorem integral_inv_fourth (a : ℝ) (ha : 0 < a) : ∫ x : ℝ, 1 / (a ^ 4 + x ^ 4) = π / (Real.sqrt 2 * a ^ 3) := by
  have h := Measure.integral_comp_div (fun x : ℝ => 1 / (1 + x ^ 4)) a
  rw [integral_one_fourth, abs_of_pos ha, smul_eq_mul] at h
  have hpt : ∀ x : ℝ, 1 / (a ^ 4 + x ^ 4) = (a ^ 4)⁻¹ * (1 / (1 + (x / a) ^ 4)) := by
    intro x; field_simp
  simp_rw [hpt]
  rw [integral_const_mul, h]
  field_simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.integral_inv_fourth
-- test: ca-res-integral-fourth-test.sage
end

/-- `∫ x² dx/(a⁴ + x⁴) = π/(√2 a)` for `a > 0` (the text: `a = 1`, `π/√2`). -/
theorem integral_sq_div_fourth (a : ℝ) (ha : 0 < a) :
    ∫ x : ℝ, x ^ 2 / (a ^ 4 + x ^ 4) = π / (Real.sqrt 2 * a) := by
  have h := Measure.integral_comp_div (fun x : ℝ => x ^ 2 / (1 + x ^ 4)) a
  rw [integral_sq_fourth, abs_of_pos ha, smul_eq_mul] at h
  have hpt : ∀ x : ℝ, x ^ 2 / (a ^ 4 + x ^ 4) = (a ^ 2)⁻¹ * ((x / a) ^ 2 / (1 + (x / a) ^ 4)) := by
    intro x; field_simp
  simp_rw [hpt]
  rw [integral_const_mul, h]
  field_simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.integral_sq_div_fourth
-- test: ca-res-integral-sq-fourth-test.sage
end

/-! ## exm-residue-method-rational -/

/-- If `|f| ≤ C/R²` on the arc `|z| = R`, the arc integral is at most `πC/R = O(1/R)`. -/
theorem arc_bound_general (f : ℂ → ℂ) (R C : ℝ) (hR : 0 < R) (hf : ∀ θ : ℝ, ‖f (R * exp (θ * I))‖ ≤ C / R ^ 2) :
    ‖∫ θ in (0 : ℝ)..π, f (R * exp (θ * I)) * (I * R * exp (θ * I))‖ ≤ π * C / R := by
  have hb : ∀ θ ∈ Ι (0 : ℝ) π, ‖f (R * exp (θ * I)) * (I * R * exp (θ * I))‖ ≤ C / R := by
    intro θ _
    rw [norm_mul, mul_assoc, norm_mul, Complex.norm_I, one_mul, norm_mul, Complex.norm_exp_ofReal_mul_I,
      mul_one, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hR]
    calc ‖f (R * exp (θ * I))‖ * R ≤ C / R ^ 2 * R := by gcongr; exact hf θ
      _ = C / R := by field_simp
  have := intervalIntegral.norm_integral_le_of_norm_le_const hb
  rw [sub_zero, abs_of_pos Real.pi_pos] at this
  calc _ ≤ C / R * π := this
    _ = π * C / R := by ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.arc_bound_general
-- test: ca-res-arc-general-test.sage
end

/-- Closing in the lower half-plane: `Res_{-ic} 1/(z² + c²) = 1/(2(-ic))` and `-2πi` times it is `π/c`, the same
number (the text: `c = 1`). -/
theorem lower_half_residue (c : ℂ) (hc : c ≠ 0) :
    Tendsto (fun z => (z - -(I * c)) * (1 / (z ^ 2 + c ^ 2))) (𝓝[≠] (-(I * c))) (𝓝 (1 / (2 * -(I * c)))) ∧
      -2 * π * I * (1 / (2 * -(I * c))) = π / c := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  constructor
  · have hh : (-(I * c)) ^ 2 + c ^ 2 = 0 := by linear_combination c ^ 2 * hI
    have hd : HasDerivAt (fun z : ℂ => z ^ 2 + c ^ 2) (2 * -(I * c)) (-(I * c)) := by
      simpa using ((hasDerivAt_pow 2 (-(I * c))).add_const (c ^ 2))
    exact quotient_rule (fun _ => 1) (fun z => z ^ 2 + c ^ 2) _ _ hh hd
      (mul_ne_zero two_ne_zero (neg_ne_zero.2 (mul_ne_zero I_ne_zero hc))) continuousAt_const
  · field_simp

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.lower_half_residue
-- test: ca-res-lower-half-test.sage
end

/-! ## exm-residue-method-fourier -/

/-- `|e^{ikz}| = e^{-k Im z}`: at most `1` in the upper half-plane for `k ≥ 0`, in the lower one for `k ≤ 0`. -/
theorem norm_exp_ikz (k : ℝ) (z : ℂ) :
    ‖exp (I * k * z)‖ = Real.exp (-(k * z.im)) ∧ (0 ≤ k → 0 ≤ z.im → ‖exp (I * k * z)‖ ≤ 1) ∧
      (k ≤ 0 → z.im ≤ 0 → ‖exp (I * k * z)‖ ≤ 1) := by
  have h : ‖exp (I * k * z)‖ = Real.exp (-(k * z.im)) := by rw [Complex.norm_exp]; simp
  refine ⟨h, fun hk hz => ?_, fun hk hz => ?_⟩
  · rw [h]; exact Real.exp_le_one_iff.2 (by nlinarith)
  · rw [h]; exact Real.exp_le_one_iff.2 (by nlinarith)

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.norm_exp_ikz
-- test: ca-res-norm-exp-ikz-test.sage
end

/-- Jordan's inequality: `∫₀^π e^{-R sin θ} dθ ≤ π/R` for `R > 0`. -/
theorem jordan (R : ℝ) (hR : 0 < R) : ∫ θ in (0 : ℝ)..π, Real.exp (-(R * Real.sin θ)) ≤ π / R := by
  have hc : Continuous (fun θ : ℝ => Real.exp (-(R * Real.sin θ))) := by fun_prop
  have hsplit := intervalIntegral.integral_add_adjacent_intervals
    (hc.intervalIntegrable (μ := MeasureTheory.volume) 0 (π / 2))
    (hc.intervalIntegrable (μ := MeasureTheory.volume) (π / 2) π)
  have hsym : ∫ θ in (π / 2)..π, Real.exp (-(R * Real.sin θ)) = ∫ θ in (0 : ℝ)..(π / 2), Real.exp (-(R * Real.sin θ)) := by
    have := intervalIntegral.integral_comp_sub_left (fun θ => Real.exp (-(R * Real.sin θ))) (a := 0) (b := π / 2) π
    simp only [Real.sin_pi_sub, sub_zero, show π - π / 2 = π / 2 by ring] at this
    exact this.symm
  have hle : ∫ θ in (0 : ℝ)..(π / 2), Real.exp (-(R * Real.sin θ)) ≤
      ∫ θ in (0 : ℝ)..(π / 2), Real.exp (-(2 * R / π) * θ) := by
    refine intervalIntegral.integral_mono_on (by positivity) (hc.intervalIntegrable (μ := MeasureTheory.volume) _ _)
      ((by fun_prop : Continuous fun θ : ℝ => Real.exp (-(2 * R / π) * θ)).intervalIntegrable
        (μ := MeasureTheory.volume) _ _) fun θ hθ => ?_
    apply Real.exp_le_exp.2
    have := Real.mul_le_sin hθ.1 hθ.2
    have h2 : 2 / π * θ * R ≤ Real.sin θ * R := by nlinarith
    have : -(2 * R / π) * θ = -(2 / π * θ * R) := by ring
    rw [this]; linarith
  have hval : ∫ θ in (0 : ℝ)..(π / 2), Real.exp (-(2 * R / π) * θ) = π / (2 * R) * (1 - Real.exp (-R)) := by
    have hpi := Real.pi_pos
    have hc0 : -(2 * R / π) ≠ 0 := by
      have : 0 < 2 * R / π := by positivity
      linarith
    rw [intervalIntegral.integral_comp_mul_left Real.exp hc0, integral_exp, smul_eq_mul, mul_zero,
      Real.exp_zero, show -(2 * R / π) * (π / 2) = -R by field_simp]
    field_simp
    ring
  have hfin : π / (2 * R) * (1 - Real.exp (-R)) ≤ π / (2 * R) := by
    have := Real.exp_pos (-R)
    have : 0 < π / (2 * R) := by positivity
    nlinarith
  rw [← hsplit, hsym]
  have : π / R = π / (2 * R) + π / (2 * R) := by field_simp; ring
  rw [this]
  linarith

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.jordan
-- test: ca-res-jordan-test.sage
end

/-! ## exm-residue-method-trig -/

/-- With `z = e^{iθ}`: `cos θ = (z + 1/z)/2`, `sin θ = (z - 1/z)/(2i)` and `dz = iz dθ`. -/
theorem subst_trig (θ : ℝ) :
    (Real.cos θ : ℂ) = (exp (θ * I) + (exp (θ * I))⁻¹) / 2 ∧
      (Real.sin θ : ℂ) = (exp (θ * I) - (exp (θ * I))⁻¹) / (2 * I) ∧
      HasDerivAt (fun t : ℝ => exp (t * I)) (I * exp (θ * I)) θ := by
  refine ⟨?_, ?_, ?_⟩
  · rw [← Complex.exp_neg, Complex.ofReal_cos, Complex.cos]; ring_nf
  · rw [← Complex.exp_neg, Complex.ofReal_sin, Complex.sin, div_eq_mul_inv _ (2 * I), mul_inv,
      Complex.inv_I]
    ring
  · have := ((hasDerivAt_id (θ : ℂ)).mul_const I).cexp.comp_ofReal
    simpa [mul_comm] using this

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.subst_trig
-- test: ca-res-subst-trig-test.sage
end

/-- The integrand after the substitution: `1/(a + (z + 1/z)/2) · 1/(iz) = 2/(i(z² + 2az + 1))`
(the text: `a = 2`). -/
theorem trig_integrand (a z : ℂ) (hz : z ≠ 0) (hq : z ^ 2 + 2 * a * z + 1 ≠ 0) :
    1 / (a + (z + z⁻¹) / 2) * (1 / (I * z)) = 2 / (I * (z ^ 2 + 2 * a * z + 1)) := by
  have hden : a + (z + z⁻¹) / 2 ≠ 0 := by
    intro h; apply hq
    have : z * (a + (z + z⁻¹) / 2) = 0 := by rw [h, mul_zero]
    have hzz : z * z⁻¹ = 1 := mul_inv_cancel₀ hz
    linear_combination 2 * this - hzz
  field_simp
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.trig_integrand
-- test: ca-res-trig-integrand-test.sage
end

/-- `∮_{|z|=1} 2 dz/(i(z² + 2az + 1)) = 2π/√(a² - 1)` by the single enclosed pole `-a + √(a² - 1)`, and hence
`∫₀^{2π} dθ/(a + cos θ) = 2π/√(a² - 1)` for `a > 1` (the text: `a = 2`, `2π/√3`). -/
theorem trig_integral (a : ℝ) (ha : 1 < a) :
    (∮ z in C(0, 1), 2 / (I * (z ^ 2 + 2 * a * z + 1))) = ((2 * π / Real.sqrt (a ^ 2 - 1) : ℝ) : ℂ) ∧
      ∫ θ in (0 : ℝ)..(2 * π), 1 / (a + Real.cos θ) = 2 * π / Real.sqrt (a ^ 2 - 1) := by
  set s := Real.sqrt (a ^ 2 - 1) with hs_def
  have hs2 : s ^ 2 = a ^ 2 - 1 := Real.sq_sqrt (by nlinarith)
  have hs0 : 0 < s := Real.sqrt_pos.2 (by nlinarith)
  have hsa : s < a := by nlinarith
  have hsa1 : a - 1 < s := by nlinarith
  set p : ℂ := ((-a + s : ℝ) : ℂ)
  set q : ℂ := ((-a - s : ℝ) : ℂ)
  have hpB : p ∈ ball (0 : ℂ) 1 := by
    rw [mem_ball, dist_zero_right, Complex.norm_real, Real.norm_eq_abs, abs_lt]; constructor <;> linarith
  have hqB : ∀ z ∈ closedBall (0 : ℂ) 1, z - q ≠ 0 := by
    intro z hz h
    have : z = q := by linear_combination h
    rw [this, mem_closedBall, dist_zero_right, Complex.norm_real, Real.norm_eq_abs, abs_le] at hz
    linarith [hz.1]
  have hfac : ∀ z : ℂ, z ^ 2 + 2 * a * z + 1 = (z - p) * (z - q) := by
    intro z
    have : ((s : ℂ)) ^ 2 = a ^ 2 - 1 := by exact_mod_cast hs2
    simp only [p, q]; push_cast
    linear_combination this
  have hc : (2 : ℂ) / (I * (p - q)) = 1 / (I * s) := by
    have : p - q = 2 * s := by simp only [p, q]; push_cast; ring
    rw [this]
    have : (s : ℂ) ≠ 0 := by exact_mod_cast hs0.ne'
    field_simp
  have hpq : p - q ≠ 0 := by
    rw [show p - q = 2 * (s : ℂ) by simp only [p, q]; push_cast; ring]
    exact mul_ne_zero two_ne_zero (by exact_mod_cast hs0.ne')
  have hcircle : (∮ z in C(0, 1), 2 / (I * (z ^ 2 + 2 * a * z + 1))) = ((2 * π / s : ℝ) : ℂ) := by
    have hcongr : (∮ z in C(0, 1), 2 / (I * (z ^ 2 + 2 * a * z + 1))) =
        ∮ z in C(0, 1), ((∑ i ∈ ({0} : Finset ℕ), 2 / (I * (p - q)) / (z - p) ^ (0 + 1)) +
          -(2 / (I * (p - q))) / (z - q)) := by
      refine circleIntegral.integral_congr zero_le_one fun z hz => ?_
      have hzp := mem_sphere_ne hz hpB
      have hzq := hqB z (sphere_subset_closedBall hz)
      simp only [Finset.sum_singleton, zero_add, pow_one]
      rw [hfac z]
      field_simp
      ring
    have hg : DifferentiableOn ℂ (fun z : ℂ => -(2 / (I * (p - q))) / (z - q)) (closedBall 0 1) :=
      DifferentiableOn.div (differentiableOn_const _) (by fun_prop) hqB
    rw [hcongr, residue_theorem_circle {0} 0 1 one_pos (fun _ => p) (fun _ => 2 / (I * (p - q))) (fun _ => 0)
      (fun _ _ => hpB) _ hg]
    simp only [Finset.sum_singleton, if_true]
    rw [hc]
    have hs' : (s : ℂ) ≠ 0 := by exact_mod_cast hs0.ne'
    have hI0 : I ≠ 0 := I_ne_zero
    push_cast
    rw [mul_one_div, div_eq_div_iff (mul_ne_zero hI0 hs') hs']
    ring
  refine ⟨by rw [hcircle], ?_⟩
  have hreal : (∮ z in C(0, 1), 2 / (I * (z ^ 2 + 2 * a * z + 1))) =
      ∫ θ in (0 : ℝ)..(2 * π), ((1 / (a + Real.cos θ) : ℝ) : ℂ) := by
    simp only [circleIntegral]
    refine intervalIntegral.integral_congr fun θ _ => ?_
    simp only [deriv_circleMap, circleMap_zero, Complex.ofReal_one, one_mul, smul_eq_mul]
    have hu : exp (θ * I) ≠ 0 := Complex.exp_ne_zero _
    have hcos := (subst_trig θ).1
    have hpos : 0 < a + Real.cos θ := by have := Real.neg_one_le_cos θ; linarith
    have hden2 : (a : ℂ) + (exp (θ * I) + (exp (θ * I))⁻¹) / 2 ≠ 0 := by
      rw [← hcos]; exact_mod_cast hpos.ne'
    have hden : (exp (θ * I)) ^ 2 + 2 * a * exp (θ * I) + 1 ≠ 0 := by
      have h1 : (exp (θ * I)) ^ 2 + 2 * a * exp (θ * I) + 1 =
          2 * exp (θ * I) * ((a : ℂ) + (exp (θ * I) + (exp (θ * I))⁻¹) / 2) := by
        field_simp; ring
      rw [h1]; exact mul_ne_zero (mul_ne_zero two_ne_zero hu) hden2
    rw [Complex.ofReal_div, Complex.ofReal_one, Complex.ofReal_add, hcos]
    field_simp
    ring
  rw [hreal, intervalIntegral.integral_ofReal] at hcircle
  exact_mod_cast hcircle

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.trig_integral
-- test: ca-res-trig-integral-test.sage
end

/-! ## exr-real-integral-x2-over-x4 -/

/-- At a pole of `z²/(1 + z⁴)`: `Res = z²/(4z³) = 1/(4z)`. -/
theorem residue_sq_fourth (z : ℂ) (hz : z ≠ 0) : z ^ 2 / (4 * z ^ 3) = 1 / (4 * z) := by
  rw [div_eq_div_iff (mul_ne_zero (by norm_num : (4 : ℂ) ≠ 0) (pow_ne_zero 3 hz))
    (mul_ne_zero (by norm_num : (4 : ℂ) ≠ 0) hz)]
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.residue_sq_fourth
-- test: ca-res-residue-sq-fourth-test.sage
end

/-- `e^{-iθ} + e^{-i(π - θ)} = -2i sin θ`. -/
theorem exp_reflect_neg (θ : ℝ) : exp (-(θ * I)) + exp (-(((π - θ : ℝ)) * I)) = -2 * Real.sin θ * I := by
  rw [show -((θ : ℂ) * I) = ((-θ : ℝ) : ℂ) * I by push_cast; ring,
    show -(((π - θ : ℝ) : ℂ) * I) = ((-(π - θ) : ℝ) : ℂ) * I by push_cast; ring,
    Complex.exp_mul_I, Complex.exp_mul_I]
  push_cast
  rw [Complex.cos_neg, Complex.sin_neg, Complex.cos_neg, Complex.sin_neg, Complex.cos_pi_sub,
    Complex.sin_pi_sub]
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.exp_reflect_neg
-- test: ca-res-exp-reflect-neg-test.sage
end

/-- The algebraic step of `upper_sum_sq_fourth`: `(1/4)(-2 (s/2) i) = -i s/4`, with `s = √2`. -/
theorem upper_sum_sq_fourth_alg (J s : ℂ) : 1 / 4 * (-2 * (s / 2) * J) = -J * s / 4 := by
  veriphy_certified
    (calc
        1 / 4 * (-2 * (s / 2) * J)
        _ = -J * s / 4 := by ring1)
  -- certificate: ca-res-ring.sage

/-- The residues `1/(4z_k)` at the two upper poles sum to `-i√2/4`. -/
theorem upper_sum_sq_fourth :
    1 / 4 * (exp (-(((π / 4 : ℝ) : ℂ) * I)) + exp (-(((3 * π / 4 : ℝ) : ℂ) * I))) = -I * (Real.sqrt 2 : ℂ) / 4 := by
  have h := exp_reflect_neg (π / 4)
  rw [show (π - π / 4 : ℝ) = 3 * π / 4 by ring, Real.sin_pi_div_four] at h
  rw [h]
  clear h
  push_cast
  exact upper_sum_sq_fourth_alg I _

/-- `2πi · (-i√2/4) = π/√2`. -/
theorem two_pi_I_sq_fourth : 2 * π * I * (-I * (Real.sqrt 2 : ℂ) / 4) = π / (Real.sqrt 2 : ℂ) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have hs : ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 := by norm_cast; simp
  have hs0 : ((Real.sqrt 2 : ℝ) : ℂ) ≠ 0 := by norm_cast; positivity
  field_simp
  veriphy_certified (linear_combination (-4) * hI + (-2 * I ^ 2) * hs)
  -- certificate: ideal_membership.sage

/-- On `|z| = R > 1`: `|z²/(1 + z⁴)| ≤ R²/(R⁴ - 1)`, so the arc is negligible (`deg q - deg p = 2`). -/
theorem arc_bound_sq_fourth (R : ℝ) (hR : 1 < R) (z : ℂ) (hz : ‖z‖ = R) :
    ‖z ^ 2 / (1 + z ^ 4)‖ ≤ R ^ 2 / (R ^ 4 - 1) := by
  have h1 : R ^ 4 - 1 ≤ ‖1 + z ^ 4‖ := by
    have := norm_sub_norm_le (z ^ 4) (-1)
    rw [norm_pow, hz, norm_neg, norm_one, sub_neg_eq_add, add_comm] at this
    linarith
  have hpos : 0 < R ^ 4 - 1 := by nlinarith [pow_lt_pow_left₀ hR zero_le_one (by norm_num : (4 : ℕ) ≠ 0)]
  rw [norm_div, norm_pow, hz]
  exact div_le_div_of_nonneg_left (by positivity) hpos h1

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.arc_bound_sq_fourth
-- test: ca-res-arc-sq-fourth-test.sage
end

/-! ## exr-real-integral-trig -/

/-- The roots of `z² + 2az + 1` are `-a ± √(a² - 1)` (the text: `a = 2`, `-2 ± √3`). -/
theorem quadratic_roots (a : ℝ) (ha : 1 ≤ a) :
    (-a + Real.sqrt (a ^ 2 - 1)) ^ 2 + 2 * a * (-a + Real.sqrt (a ^ 2 - 1)) + 1 = 0 ∧
      (-a - Real.sqrt (a ^ 2 - 1)) ^ 2 + 2 * a * (-a - Real.sqrt (a ^ 2 - 1)) + 1 = 0 := by
  have hs : Real.sqrt (a ^ 2 - 1) ^ 2 = a ^ 2 - 1 := Real.sq_sqrt (by nlinarith)
  constructor
  · veriphy_certified (linear_combination hs)
    -- certificate: ideal_membership.sage
  · veriphy_certified (linear_combination hs)
    -- certificate: ideal_membership.sage

/-- Only `-a + √(a² - 1)` lies inside the unit circle: `|-a + √(a² - 1)| < 1 < |-a - √(a² - 1)|`
(the text: `-2 + √3 ≈ -0.27`). -/
theorem inside_root (a : ℝ) (ha : 1 < a) :
    |-a + Real.sqrt (a ^ 2 - 1)| < 1 ∧ 1 < |-a - Real.sqrt (a ^ 2 - 1)| := by
  have hs : Real.sqrt (a ^ 2 - 1) ^ 2 = a ^ 2 - 1 := Real.sq_sqrt (by nlinarith)
  have hs0 : 0 ≤ Real.sqrt (a ^ 2 - 1) := Real.sqrt_nonneg _
  have hsa : Real.sqrt (a ^ 2 - 1) < a := by nlinarith
  have hsa1 : a - 1 < Real.sqrt (a ^ 2 - 1) := by nlinarith
  constructor
  · rw [abs_lt]; constructor <;> linarith
  · rw [abs_of_neg (by linarith)]; linarith

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.inside_root
-- test: ca-res-inside-root-test.sage
end

/-- The residue `2/(i · 2√3) = 1/(i√3)` and `2πi/(i√3) = 2π/√3`. -/
theorem trig_residue_value :
    2 / (I * (2 * (Real.sqrt 3 : ℂ))) = 1 / (I * (Real.sqrt 3 : ℂ)) ∧
      2 * π * I * (1 / (I * (Real.sqrt 3 : ℂ))) = 2 * π / (Real.sqrt 3 : ℂ) := by
  have hs0 : ((Real.sqrt 3 : ℝ) : ℂ) ≠ 0 := by norm_cast; positivity
  constructor
  · rw [div_eq_div_iff (mul_ne_zero I_ne_zero (mul_ne_zero two_ne_zero hs0)) (mul_ne_zero I_ne_zero hs0)]
    clear hs0
    generalize I = J
    generalize ((Real.sqrt 3 : ℝ) : ℂ) = s
    rw [← sub_eq_zero]
    veriphy_certified
      (calc
          2 * (J * s) - 1 * (J * (2 * s))
          _ = 0 := by ring1)
    -- certificate: ca-res-ring.sage
  · rw [mul_one_div, div_eq_div_iff (mul_ne_zero I_ne_zero hs0) hs0]
    clear hs0
    generalize I = J
    generalize ((Real.sqrt 3 : ℝ) : ℂ) = s
    rw [← sub_eq_zero]
    veriphy_certified
      (calc
          2 * ↑π * J * s - 2 * ↑π * (J * s)
          _ = 0 := by ring1)
    -- certificate: ca-res-ring.sage

/-! ## exr-real-integral-jordan -/

/-- The small clockwise semicircle: `∫_π^0 e^{iaεe^{iθ}} i dθ → -iπ` as `ε → 0` (the text: `a = 1`). -/
theorem small_arc (a : ℝ) :
    Tendsto (fun ε : ℝ => ∫ θ in π..0, exp (I * a * (ε * exp (θ * I))) * I) (𝓝 0) (𝓝 (-(π * I))) := by
  have hc : Continuous (fun ε : ℝ => ∫ θ in π..0, exp (I * a * (ε * exp (θ * I))) * I) :=
    intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) π 0
  have := hc.tendsto 0
  simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul,
    intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
  convert this using 2
  rw [Complex.real_smul]; push_cast; ring

section
open scoped ComplexOrder
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc (1) := by
  have hc : Continuous (fun ε : ℝ => ∫ θ in Real.pi..0, Complex.exp (Complex.I * (1 : ℝ) * (ε * Complex.exp (θ * Complex.I))) * Complex.I) :=
    intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) Real.pi 0
  have := hc.tendsto 0
  simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul, intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
  convert this using 2
  first
    | (rw [Complex.real_smul]; push_cast; ring)
    | (push_cast; ring)
    | simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc (-2) := by
  have hc : Continuous (fun ε : ℝ => ∫ θ in Real.pi..0, Complex.exp (Complex.I * (-2 : ℝ) * (ε * Complex.exp (θ * Complex.I))) * Complex.I) :=
    intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) Real.pi 0
  have := hc.tendsto 0
  simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul, intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
  convert this using 2
  first
    | (rw [Complex.real_smul]; push_cast; ring)
    | (push_cast; ring)
    | simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc (1 / 2) := by
  have hc : Continuous (fun ε : ℝ => ∫ θ in Real.pi..0, Complex.exp (Complex.I * (1 / 2 : ℝ) * (ε * Complex.exp (θ * Complex.I))) * Complex.I) :=
    intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) Real.pi 0
  have := hc.tendsto 0
  simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul, intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
  convert this using 2
  first
    | (rw [Complex.real_smul]; push_cast; ring)
    | (push_cast; ring)
    | simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc (3) := by
  have hc : Continuous (fun ε : ℝ => ∫ θ in Real.pi..0, Complex.exp (Complex.I * (3 : ℝ) * (ε * Complex.exp (θ * Complex.I))) * Complex.I) :=
    intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' (by fun_prop) Real.pi 0
  have := hc.tendsto 0
  simp only [Complex.ofReal_zero, zero_mul, mul_zero, Complex.exp_zero, one_mul, intervalIntegral.integral_const, smul_eq_mul, zero_sub] at this
  convert this using 2
  first
    | (rw [Complex.real_smul]; push_cast; ring)
    | (push_cast; ring)
    | simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ResidueWorked.small_arc
-- test: ca-res-small-arc-test.sage
end

/-- The large arc: `|∫_{C_R} e^{iz} dz/z| ≤ ∫₀^π e^{-R sin θ} dθ ≤ π/R → 0` (Jordan's lemma). -/
theorem large_arc (R : ℝ) (hR : 0 < R) : ‖∫ θ in (0 : ℝ)..π, exp (I * (R * exp (θ * I))) * I‖ ≤ π / R := by
  refine le_trans (intervalIntegral.norm_integral_le_integral_norm Real.pi_pos.le) (le_trans ?_ (jordan R hR))
  refine le_of_eq (intervalIntegral.integral_congr fun θ _ => ?_)
  rw [norm_mul, Complex.norm_I, mul_one, Complex.norm_exp]
  congr 1
  rw [Complex.exp_mul_I]; simp [Complex.mul_re, Complex.sin_ofReal_re, Complex.cos_ofReal_re]

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueWorked.large_arc
-- test: ca-res-large-arc-test.sage
end

/-- The real part of `∫_{ε < |x| < R} e^{ix}/x dx` vanishes: `cos x/x` is odd. -/
theorem odd_pv (ε R : ℝ) : (∫ x in (-R)..(-ε), Real.cos x / x) + ∫ x in ε..R, Real.cos x / x = 0 := by
  have h := intervalIntegral.integral_comp_neg (fun x => Real.cos x / x) (a := ε) (b := R)
  simp only [Real.cos_neg, div_neg] at h
  rw [intervalIntegral.integral_neg] at h
  linarith

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueWorked.odd_pv
-- test: ca-res-odd-pv-test.sage
end

end KaluzaKlein.ComplexAnalysis.ResidueWorked

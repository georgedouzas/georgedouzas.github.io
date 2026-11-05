import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.Residues
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The residue theorem"

The claims of the section "The residue theorem" of `tutorial/part-i-04-complex-analysis.qmd`. Mathlib has no
residue theorem for general contours; it is proved here on circles for every function of the form
`∑ cᵢ/(z - wᵢ)^(kᵢ+1) + g` with `g` holomorphic on the closed disc (every rational function and every function with
finitely many poles of this form), the residue sum being the sum of the `cᵢ` with `kᵢ = 0` over the poles inside.
The examples use it on circles, together with mathlib's improper integrals and the Fourier inversion theorem, which
evaluates `∫ e^{ikx}/(1 + x²) dx = π e^{-|k|}` (hence `∫ cos x/(1 + x²) dx = π/e` and `∫ sin x/(1 + x²) dx = 0`).
The semicircular and rectangular contours of the text are not formalised as contours (no Jordan curve theorem or
general Cauchy theorem in mathlib); the arc estimates are.
-/

namespace KaluzaKlein.ComplexAnalysis.ResidueTheorem

open Complex Filter Topology Metric MeasureTheory
open scoped Real FourierTransform Interval

open KaluzaKlein.ComplexAnalysis.Residues

/-! ## thm-residue -/

/-- The residue theorem on a circle: for `f = ∑ᵢ cᵢ/(z - wᵢ)^(kᵢ+1) + g` with the poles `wᵢ` inside the circle and
`g` holomorphic on the closed disc, `∮ f = 2πi ∑ Res`, where the residue at `wᵢ` collects the `cᵢ` with `kᵢ = 0`. -/
theorem residue_theorem_circle {ι : Type*} (s : Finset ι) (C : ℂ) (ρ : ℝ) (hρ : 0 < ρ) (w c : ι → ℂ) (k : ι → ℕ)
    (hw : ∀ i ∈ s, w i ∈ ball C ρ) (g : ℂ → ℂ) (hg : DifferentiableOn ℂ g (closedBall C ρ)) :
    (∮ z in C(C, ρ), ((∑ i ∈ s, c i / (z - w i) ^ (k i + 1)) + g z)) =
      2 * π * I * ∑ i ∈ s, if k i = 0 then c i else 0 :=
  circle_sum_poles s C ρ hρ w c k hw g hg

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.residue_theorem_circle
-- test: ca-res-residue-theorem-test.sage

/-! ## exm-residue-theorem-two-poles -/

lemma not_mem_closedBall_zero {b : ℂ} {ρ : ℝ} (hb : ρ < ‖b‖) : ∀ z ∈ closedBall (0 : ℂ) ρ, z - b ≠ 0 := by
  intro z hz h
  have : z = b := by linear_combination h
  rw [this, mem_closedBall, dist_zero_right] at hz; linarith

/-- A circle enclosing both poles of `1/((z - a)(z - b))`: `∮ = 2πi (1/(a - b) + 1/(b - a)) = 0`
(the text: `|z| = 3`, `a = 1`, `b = 2`). -/
theorem two_poles_both (a b : ℂ) (hab : a ≠ b) (ρ : ℝ) (ha : ‖a‖ < ρ) (hb : ‖b‖ < ρ) :
    (∮ z in C(0, ρ), 1 / ((z - a) * (z - b))) = 0 := by
  have h := residue_infinity_two_poles a b hab ρ ha hb
  have hne : (2 * π * I : ℂ)⁻¹ ≠ 0 := inv_ne_zero two_pi_I_ne_zero
  rw [neg_eq_zero, residue, mul_eq_zero] at h
  exact h.resolve_left hne

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_poles_both
-- test: ca-res-two-poles-both-test.sage

/-- A circle enclosing only the pole `a`: `∮ = 2πi/(a - b)` (the text: `|z| = 1.5`, `-2πi`). -/
theorem two_poles_one (a b : ℂ) (hab : a ≠ b) (ρ : ℝ) (ha : ‖a‖ < ρ) (hb : ρ < ‖b‖) :
    (∮ z in C(0, ρ), 1 / ((z - a) * (z - b))) = 2 * π * I / (a - b) := by
  have hρ : 0 < ρ := (norm_nonneg a).trans_lt ha
  have hab' : a - b ≠ 0 := sub_ne_zero.2 hab
  have haB : a ∈ ball (0 : ℂ) ρ := by simpa using ha
  have hg : DifferentiableOn ℂ (fun z : ℂ => 1 / (b - a) / (z - b)) (closedBall 0 ρ) :=
    DifferentiableOn.div (differentiableOn_const _) (by fun_prop) (not_mem_closedBall_zero hb)
  have hcongr : (∮ z in C(0, ρ), 1 / ((z - a) * (z - b))) =
      ∮ z in C(0, ρ), ((∑ i ∈ ({0} : Finset ℕ), 1 / (a - b) / (z - a) ^ (0 + 1)) + 1 / (b - a) / (z - b)) := by
    refine circleIntegral.integral_congr hρ.le fun z hz => ?_
    simp only [Finset.sum_singleton, zero_add, pow_one]
    exact partial_fraction_ab a b z hab' (mem_sphere_ne hz haB)
      (not_mem_closedBall_zero hb z (sphere_subset_closedBall hz))
  rw [hcongr, residue_theorem_circle {0} 0 ρ hρ (fun _ => a) (fun _ => 1 / (a - b)) (fun _ => 0)
    (fun _ _ => haB) _ hg]
  simp only [Finset.sum_singleton, if_true]
  ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_poles_one
-- test: ca-res-two-poles-one-test.sage
end

/-! ## exm-residue-theorem-real-integral -/

/-- `2πi · Res_i c/(1 + z²) = 2πi · (-ic/2) = πc`; for `c = 1`, `2πi · (-i/2) = π`. -/
theorem two_pi_I_residue_I (c : ℂ) : 2 * π * I * (-I * c / 2) = π * c := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  linear_combination (-(π : ℂ) * c) * hI

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I (0) := by
  linear_combination (-(Real.pi : ℂ) * (0)) * Complex.I_sq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I (1) := by
  linear_combination (-(Real.pi : ℂ) * (1)) * Complex.I_sq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I (Complex.I) := by
  linear_combination (-(Real.pi : ℂ) * (Complex.I)) * Complex.I_sq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I (2 - 3 * Complex.I) := by
  linear_combination (-(Real.pi : ℂ) * (2 - 3 * Complex.I)) * Complex.I_sq

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ResidueTheorem.two_pi_I_residue_I
-- test: ca-res-two-pi-residue-test.sage

/-- On the arc `|z| = R > 1`: `|1/(1 + z²)| ≤ 1/(R² - 1)`. -/
theorem arc_pointwise (R : ℝ) (hR : 1 < R) (z : ℂ) (hz : ‖z‖ = R) : ‖1 / (1 + z ^ 2)‖ ≤ 1 / (R ^ 2 - 1) := by
  have h1 : R ^ 2 - 1 ≤ ‖1 + z ^ 2‖ := by
    have := norm_sub_norm_le (z ^ 2) (-1)
    rw [norm_pow, hz, norm_neg, norm_one, sub_neg_eq_add, add_comm] at this
    linarith
  have hpos : 0 < R ^ 2 - 1 := by nlinarith
  rw [norm_div, norm_one]
  exact one_div_le_one_div_of_le hpos h1

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.arc_pointwise
-- test: ca-res-arc-pointwise-test.sage

/-- The arc `C_R` (`z = Re^{iθ}`, `0 ≤ θ ≤ π`) contributes at most `πR/(R² - 1) → 0`. -/
theorem arc_integral_bound (R : ℝ) (hR : 1 < R) :
    ‖∫ θ in (0 : ℝ)..π, 1 / (1 + (R * exp (θ * I)) ^ 2) * (I * R * exp (θ * I))‖ ≤ π * R / (R ^ 2 - 1) := by
  have hpos : 0 < R ^ 2 - 1 := by nlinarith
  have hb : ∀ θ ∈ Ι (0 : ℝ) π, ‖1 / (1 + (R * exp (θ * I)) ^ 2) * (I * R * exp (θ * I))‖ ≤ R / (R ^ 2 - 1) := by
    intro θ _
    have hz : ‖(R : ℂ) * exp (θ * I)‖ = R := by
      rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_real, Real.norm_eq_abs,
        abs_of_pos (by linarith)]
    rw [norm_mul]
    have h2 : ‖I * R * exp (θ * I)‖ = R := by rw [mul_assoc, norm_mul, Complex.norm_I, one_mul, hz]
    rw [h2]
    calc ‖1 / (1 + ((R : ℂ) * exp (θ * I)) ^ 2)‖ * R ≤ 1 / (R ^ 2 - 1) * R := by
          gcongr; exact arc_pointwise R hR _ hz
      _ = R / (R ^ 2 - 1) := by ring
  have := intervalIntegral.norm_integral_le_of_norm_le_const hb
  rw [sub_zero, abs_of_pos Real.pi_pos] at this
  calc _ ≤ R / (R ^ 2 - 1) * π := this
    _ = π * R / (R ^ 2 - 1) := by ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.arc_integral_bound
-- test: ca-res-arc-integral-test.sage

/-- `∫ dx/(a² + x²) = π/a` for `a > 0` (the text: `a = 1`, `∫ dx/(1 + x²) = π`). -/
theorem integral_inv_sq_add_sq (a : ℝ) (ha : 0 < a) : ∫ x : ℝ, 1 / (a ^ 2 + x ^ 2) = π / a := by
  have h := Measure.integral_comp_div (fun x : ℝ => (1 + x ^ 2)⁻¹) a
  rw [integral_univ_inv_one_add_sq, abs_of_pos ha, smul_eq_mul] at h
  have hpt : ∀ x : ℝ, 1 / (a ^ 2 + x ^ 2) = a⁻¹ ^ 2 * (1 + (x / a) ^ 2)⁻¹ := by
    intro x; field_simp
  simp_rw [hpt]
  rw [integral_const_mul, h]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_inv_sq_add_sq
-- test: ca-res-integral-inv-sq-test.sage

/-- The symmetric integrals `∫_{-R}^{R} dx/(1 + x²) = 2 arctan R → π` (arctan confirms). -/
theorem integral_symm_arctan (R : ℝ) : ∫ x in (-R)..R, 1 / (1 + x ^ 2) = 2 * Real.arctan R := by
  rw [integral_one_div_one_add_sq, Real.arctan_neg]; ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_symm_arctan
-- test: ca-res-arctan-test.sage

/-! ## exm-residue-theorem-fourier -/

/-- `|e^{iz}| = e^{-Im z} ≤ 1` on the upper half-plane. -/
theorem norm_exp_I_mul (z : ℂ) : ‖exp (I * z)‖ = Real.exp (-z.im) ∧ (0 ≤ z.im → ‖exp (I * z)‖ ≤ 1) := by
  have h : ‖exp (I * z)‖ = Real.exp (-z.im) := by rw [Complex.norm_exp]; simp
  refine ⟨h, fun hz => ?_⟩
  rw [h]; exact Real.exp_le_one_iff.2 (by linarith)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.norm_exp_I_mul
-- test: ca-res-norm-exp-test.sage

/-- `cos z` grows like `e^{|Im z|}`: `|cos(iy)| = cosh y ≥ e^{|y|}/2`. -/
theorem norm_cos_I_mul (y : ℝ) : ‖cos (y * I)‖ = Real.cosh y ∧ Real.exp |y| / 2 ≤ ‖cos (y * I)‖ := by
  have h : ‖cos (y * I)‖ = Real.cosh y := by
    rw [Complex.cos_mul_I, ← Complex.ofReal_cosh, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.cosh_pos y)]
  refine ⟨h, ?_⟩
  rw [h, Real.cosh_eq]
  rcases le_or_gt 0 y with hy | hy
  · rw [abs_of_nonneg hy]; have := Real.exp_pos (-y); linarith
  · rw [abs_of_neg hy]; have := Real.exp_pos y; linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.norm_cos_I_mul
-- test: ca-res-norm-cos-test.sage

/-- The residue of `e^{ikz}/(1 + z²)` at `i` is `e^{-k}/(2i)` (the text: `k = 1`). -/
theorem residue_fourier (k : ℝ) :
    Tendsto (fun z => (z - I) * (exp (I * k * z) / (1 + z ^ 2))) (𝓝[≠] I)
      (𝓝 (exp (I * k * I) / (2 * I))) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have hd : HasDerivAt (fun z : ℂ => 1 + z ^ 2) (2 * I) I := by
    simpa using (hasDerivAt_pow 2 I).const_add 1
  exact quotient_rule (fun z => exp (I * k * z)) (fun z => 1 + z ^ 2) I (2 * I)
    (by linear_combination hI) hd (mul_ne_zero two_ne_zero I_ne_zero) (by fun_prop)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.residue_fourier
-- test: ca-res-residue-fourier-test.sage

/-- `e^{ik·i} = e^{-k}` and `2πi · e^{-1}/(2i) = π/e`. -/
theorem fourier_contour_value (k : ℝ) :
    exp (I * k * I) = (Real.exp (-k) : ℂ) ∧ 2 * π * I * (exp (I * k * I) / (2 * I)) = π * (Real.exp (-k) : ℂ) := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  have he : exp (I * k * I) = (Real.exp (-k) : ℂ) := by
    rw [Complex.ofReal_exp]; congr 1; push_cast; linear_combination (k : ℂ) * hI
  refine ⟨he, ?_⟩
  rw [he]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourier_contour_value
-- test: ca-res-fourier-value-test.sage

lemma integrable_exp_neg_abs : Integrable (fun t : ℝ => Real.exp (-|t|)) := by
  rw [← integrableOn_univ, ← Set.Iic_union_Ioi (a := 0)]
  refine IntegrableOn.union ?_ ?_
  · have h := integrableOn_exp_mul_Iic (a := (1 : ℝ)) one_pos 0
    refine IntegrableOn.congr_fun h (fun t ht => ?_) measurableSet_Iic
    show Real.exp (1 * t) = Real.exp (-|t|)
    rw [one_mul, abs_of_nonpos ht, neg_neg]
  · have h := exp_neg_integrableOn_Ioi 0 (b := (1 : ℝ)) one_pos
    refine IntegrableOn.congr_fun h (fun t ht => ?_) measurableSet_Ioi
    show Real.exp (-1 * t) = Real.exp (-|t|)
    rw [abs_of_pos ht, neg_one_mul]

lemma fourier_exp_neg_abs (w : ℝ) :
    𝓕 (fun t : ℝ => ((Real.exp (-|t|) : ℝ) : ℂ)) w = ((2 / (1 + (2 * π * w) ^ 2) : ℝ) : ℂ) := by
  rw [Real.fourier_real_eq_integral_exp_smul]
  simp only [smul_eq_mul]
  have hint : Integrable (fun v : ℝ => exp (((-2 * π * v * w : ℝ) : ℂ) * I) * ((Real.exp (-|v|) : ℝ) : ℂ)) := by
    refine Integrable.mono' integrable_exp_neg_abs (by fun_prop) (Eventually.of_forall fun v => ?_)
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _)]
  rw [← setIntegral_univ, ← Set.Iic_union_Ioi (a := 0),
    setIntegral_union (Set.Iic_disjoint_Ioi le_rfl) measurableSet_Ioi hint.integrableOn hint.integrableOn]
  have h1 : ∫ v in Set.Iic (0 : ℝ), exp (((-2 * π * v * w : ℝ) : ℂ) * I) * ((Real.exp (-|v|) : ℝ) : ℂ) =
      ∫ v in Set.Iic (0 : ℝ), exp ((1 - 2 * π * w * I) * v) := by
    refine setIntegral_congr_fun measurableSet_Iic fun v hv => ?_
    rw [abs_of_nonpos hv, neg_neg, Complex.ofReal_exp, ← Complex.exp_add]
    congr 1; push_cast; ring
  have h2 : ∫ v in Set.Ioi (0 : ℝ), exp (((-2 * π * v * w : ℝ) : ℂ) * I) * ((Real.exp (-|v|) : ℝ) : ℂ) =
      ∫ v in Set.Ioi (0 : ℝ), exp ((-1 - 2 * π * w * I) * v) := by
    refine setIntegral_congr_fun measurableSet_Ioi fun v hv => ?_
    rw [abs_of_pos hv, Complex.ofReal_exp, ← Complex.exp_add]
    congr 1; push_cast; ring
  have hre1 : 0 < (1 - 2 * π * w * I : ℂ).re := by simp
  have hre2 : (-1 - 2 * π * w * I : ℂ).re < 0 := by simp
  rw [h1, h2, integral_exp_mul_complex_Iic hre1 0, integral_exp_mul_complex_Ioi hre2 0]
  have hA : (1 - 2 * π * w * I : ℂ) ≠ 0 := fun h => by simpa [h] using hre1
  have hB : (-1 - 2 * π * w * I : ℂ) ≠ 0 := fun h => by simpa [h] using hre2
  have hC : (1 + (2 * π * w) ^ 2 : ℂ) ≠ 0 := by
    have : (0 : ℝ) < 1 + (2 * π * w) ^ 2 := by positivity
    exact_mod_cast this.ne'
  push_cast
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero]
  rw [div_add_div _ _ hA hB, div_eq_div_iff (mul_ne_zero hA hB) hC]
  have hI : I ^ 2 = -1 := Complex.I_sq
  linear_combination (-8 * (π : ℂ) ^ 2 * (w : ℂ) ^ 2) * hI

/-- The Fourier transform of `1/(1 + x²)`: `∫ e^{ikx}/(1 + x²) dx = π e^{-|k|}` (by Fourier inversion of
`e^{-|t|}`, whose transform is `2/(1 + 4π²w²)`). -/
theorem fourier_inv_one_add_sq (k : ℝ) :
    (∫ x : ℝ, exp (((k * x : ℝ) : ℂ) * I) / (1 + (x : ℂ) ^ 2)) = ((π * Real.exp (-|k|) : ℝ) : ℂ) := by
  set f : ℝ → ℂ := fun t => ((Real.exp (-|t|) : ℝ) : ℂ) with hf_def
  have hf : Integrable f := integrable_exp_neg_abs.ofReal
  have hcont : Continuous f := by fun_prop
  have hF : 𝓕 f = fun w : ℝ => ((2 / (1 + (2 * π * w) ^ 2) : ℝ) : ℂ) := funext fourier_exp_neg_abs
  have hFi : Integrable (𝓕 f) := by
    rw [hF]
    refine Integrable.ofReal ?_
    have := (integrable_inv_one_add_sq.comp_mul_left' (by positivity : (2 * π : ℝ) ≠ 0)).const_mul 2
    exact this.congr (Eventually.of_forall fun w => (div_eq_mul_inv _ _).symm)
  have hinv := congrFun (hcont.fourierInv_fourier_eq hf hFi) k
  rw [Real.fourierInv_eq', hF] at hinv
  simp only [hf_def, smul_eq_mul] at hinv
  set G : ℝ → ℂ := fun w => exp (((2 * π * (inner ℝ w k)) : ℝ) * I) * ((2 / (1 + (2 * π * w) ^ 2) : ℝ) : ℂ)
    with hG
  have hcv := Measure.integral_comp_div G (2 * π)
  rw [abs_of_pos (by positivity : (0 : ℝ) < 2 * π)] at hcv
  have hpt : ∀ x : ℝ, G (x / (2 * π)) = 2 * (exp (((k * x : ℝ) : ℂ) * I) / (1 + (x : ℂ) ^ 2)) := by
    intro x
    simp only [hG, RCLike.inner_apply, conj_trivial]
    have hpi : (π : ℝ) ≠ 0 := Real.pi_ne_zero
    have h1 : 2 * π * (k * (x / (2 * π))) = k * x := by field_simp
    have h2 : (2 * π * (x / (2 * π))) = x := by field_simp
    rw [h1, h2]
    push_cast
    ring
  simp_rw [hpt] at hcv
  rw [integral_const_mul] at hcv
  have hGint : (∫ w, G w) = ((Real.exp (-|k|) : ℝ) : ℂ) := by
    rw [← hinv]
  rw [hGint] at hcv
  have h2 : (2 : ℂ) ≠ 0 := two_ne_zero
  have := congrArg (fun t => t / 2) hcv
  rw [mul_div_cancel_left₀ _ h2] at this
  rw [this, Complex.real_smul]
  push_cast
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourier_inv_one_add_sq
-- test: ca-res-fourier-test.sage

lemma integrable_fourier_kernel (k : ℝ) :
    Integrable (fun x : ℝ => exp (((k * x : ℝ) : ℂ) * I) / (1 + (x : ℂ) ^ 2)) := by
  have hne : ∀ x : ℝ, (1 + (x : ℂ) ^ 2) ≠ 0 := by
    intro x; have : (0 : ℝ) < 1 + x ^ 2 := by positivity
    exact_mod_cast this.ne'
  refine Integrable.mono' integrable_inv_one_add_sq (Continuous.aestronglyMeasurable
    (Continuous.div (by fun_prop) (by fun_prop) hne)) (Eventually.of_forall fun x => ?_)
  rw [norm_div, Complex.norm_exp_ofReal_mul_I]
  have : ‖(1 + (x : ℂ) ^ 2)‖ = 1 + x ^ 2 := by
    rw [show (1 + (x : ℂ) ^ 2) = ((1 + x ^ 2 : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
      Real.norm_eq_abs, abs_of_pos (by positivity)]
  rw [this, one_div]

/-- `∫ cos(kx)/(1 + x²) dx = π e^{-|k|}` (the text: `k = 1`, `π/e`). -/
theorem integral_cos_div (k : ℝ) : ∫ x : ℝ, Real.cos (k * x) / (1 + x ^ 2) = π * Real.exp (-|k|) := by
  have h2 := integral_re (𝕜 := ℂ) (integrable_fourier_kernel k)
  simp only [RCLike.re_to_complex] at h2
  rw [fourier_inv_one_add_sq k, Complex.ofReal_re] at h2
  rw [← h2]
  refine integral_congr_ae (Eventually.of_forall fun x => ?_)
  show Real.cos (k * x) / (1 + x ^ 2) = (exp (((k * x : ℝ) : ℂ) * I) / (1 + (x : ℂ) ^ 2)).re
  rw [show (1 + (x : ℂ) ^ 2) = ((1 + x ^ 2 : ℝ) : ℂ) by push_cast; ring, Complex.div_ofReal_re,
    Complex.exp_ofReal_mul_I_re]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_cos_div
-- test: ca-res-cos-integral-test.sage

/-- `∫ sin(kx)/(1 + x²) dx = 0`, consistent with oddness. -/
theorem integral_sin_div (k : ℝ) : ∫ x : ℝ, Real.sin (k * x) / (1 + x ^ 2) = 0 := by
  have h2 := integral_im (𝕜 := ℂ) (integrable_fourier_kernel k)
  simp only [RCLike.im_to_complex] at h2
  rw [fourier_inv_one_add_sq k, Complex.ofReal_im] at h2
  rw [← h2]
  refine integral_congr_ae (Eventually.of_forall fun x => ?_)
  show Real.sin (k * x) / (1 + x ^ 2) = (exp (((k * x : ℝ) : ℂ) * I) / (1 + (x : ℂ) ^ 2)).im
  rw [show (1 + (x : ℂ) ^ 2) = ((1 + x ^ 2 : ℝ) : ℂ) by push_cast; ring, Complex.div_ofReal_im,
    Complex.exp_ofReal_mul_I_im]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.integral_sin_div
-- test: ca-res-sin-integral-test.sage

/-! ## exr-residue-theorem-sum -/

/-- `∮_{|z| = ρ} (z² + 1)/(z (z - a)²) dz = 2πi (1/a² + 1 - 1/a²) = 2πi` for `ρ > |a|` (the text: `a = 1`, `ρ = 3`). -/
theorem rational_contour (a : ℂ) (ha : a ≠ 0) (ρ : ℝ) (hρ : ‖a‖ < ρ) :
    (∮ z in C(0, ρ), (z ^ 2 + 1) / (z * (z - a) ^ 2)) = 2 * π * I := by
  have h := (rational_residue_infinity a ha ρ hρ).1
  rw [neg_inj, residue] at h
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have := congrArg (fun t => 2 * π * I * t) h
  simp only [← mul_assoc, mul_inv_cancel₀ hne, one_mul, mul_one] at this
  exact this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ResidueTheorem.rational_contour
-- test: ca-res-rational-contour-test.sage

/-! ## exr-residue-theorem-outside -/

/-- The circle `|z - b| = r` around `b` only: `∮ = 2πi/(b - a)` (the text: `|z - 2| = 1/2`, `2πi`). -/
theorem circle_around_b (a b : ℂ) (hab : a ≠ b) (r : ℝ) (hr : 0 < r) (hr' : r < ‖a - b‖) :
    (∮ z in C(b, r), 1 / ((z - a) * (z - b))) = 2 * π * I / (b - a) := by
  have h := (residue_two_poles a b hab r hr hr').2.2.1
  rw [residue] at h
  have hne : (2 * π * I : ℂ) ≠ 0 := two_pi_I_ne_zero
  have := congrArg (fun t => 2 * π * I * t) h
  simp only [← mul_assoc, mul_inv_cancel₀ hne, one_mul] at this
  rw [this]; ring

section
open scoped ComplexOrder
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.circle_around_b
-- test: ca-res-circle-b-test.sage
end

/-- A circle enclosing neither pole: `∮ = 0` by Cauchy's theorem (the text: `|z| = 1/2`). -/
theorem circle_neither (a b : ℂ) (ρ : ℝ) (hρ : 0 ≤ ρ) (ha : ρ < ‖a‖) (hb : ρ < ‖b‖) :
    (∮ z in C(0, ρ), 1 / ((z - a) * (z - b))) = 0 :=
  circle_holo_zero _ 0 ρ hρ (DifferentiableOn.div (differentiableOn_const _) (by fun_prop)
    fun z hz => mul_ne_zero (not_mem_closedBall_zero ha z hz) (not_mem_closedBall_zero hb z hz))

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ResidueTheorem.circle_neither (1) (2) (0) ((le_refl 0)) ((by norm_num)) ((by norm_num)) := by
  simp [circleIntegral.integral_radius_zero]

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ResidueTheorem.circle_neither
-- test: ca-res-circle-neither-test.sage

/-- The primitive `log(1 - (z - 1)⁻¹) = log((z - 2)/(z - 1))` of `1/((z - 1)(z - 2))`, holomorphic off `[1, 2]`. -/
noncomputable def primF (z : ℂ) : ℂ := log (1 - (z - 1)⁻¹)

lemma slit_of_im_ne (z : ℂ) (hz : z.im ≠ 0) : 1 - (z - 1)⁻¹ ∈ slitPlane := by
  rw [mem_slitPlane_iff]; right
  have h1 : Complex.normSq (z - 1) ≠ 0 := by
    rw [Ne, Complex.normSq_eq_zero]; intro h; apply hz
    have : z = 1 := by linear_combination h
    rw [this]; simp
  have e : (1 - (z - 1)⁻¹).im = z.im / Complex.normSq (z - 1) := by
    simp [Complex.inv_im, neg_div]
  rw [e]; exact div_ne_zero hz h1

lemma slit_of_real (x : ℝ) (hx : x < 1 ∨ 2 < x) : 1 - ((x : ℂ) - 1)⁻¹ ∈ slitPlane := by
  rw [mem_slitPlane_iff]; left
  have e : 1 - ((x : ℂ) - 1)⁻¹ = ((1 - (x - 1)⁻¹ : ℝ) : ℂ) := by push_cast; ring
  rw [e, Complex.ofReal_re]
  rcases hx with hx | hx
  · have : (x - 1)⁻¹ < 0 := inv_lt_zero.2 (by linarith)
    linarith
  · have : (x - 1)⁻¹ < 1 := inv_lt_one_of_one_lt₀ (by linarith)
    linarith

lemma hasDerivAt_primF (z : ℂ) (hz1 : z - 1 ≠ 0) (hz : 1 - (z - 1)⁻¹ ∈ slitPlane) :
    HasDerivAt primF (1 / ((z - 1) * (z - 2))) z := by
  have hz2 : z - 2 ≠ 0 := by
    intro h
    have : z = 2 := by linear_combination h
    rw [this] at hz; norm_num at hz
  have h1 : HasDerivAt (fun w : ℂ => 1 - (w - 1)⁻¹) (((z - 1) ^ 2)⁻¹) z := by
    have := (((hasDerivAt_id z).sub_const 1).inv hz1).const_sub 1
    refine this.congr_deriv ?_
    simp only [id]
    field_simp
  have h2 := h1.clog hz
  refine h2.congr_deriv ?_
  have h3 : 1 - (z - 1)⁻¹ ≠ 0 := slitPlane_ne_zero hz
  have hz1' : -1 + z ≠ 0 := by intro h; apply hz1; linear_combination h
  have hz2' : -2 + z ≠ 0 := by intro h; apply hz2; linear_combination h
  have hw : (z - 1) * (z - 1)⁻¹ = 1 := mul_inv_cancel₀ hz1
  have e : 1 - (z - 1)⁻¹ = (z - 2) * (z - 1)⁻¹ := by
    rw [eq_comm, mul_inv_eq_iff_eq_mul₀ hz1, sub_mul, one_mul, inv_mul_cancel₀ hz1]; ring
  rw [e, div_eq_div_iff (mul_ne_zero hz2 (inv_ne_zero hz1)) (mul_ne_zero hz1 hz2), pow_two, mul_inv]
  first
    | linear_combination ((z - 1)⁻¹ * (z - 2)) * hw
    | (field_simp; ring)
    | field_simp

/-- The rectangle with vertices `x₀ ± ih`, `x₁ ± ih`, `x₀ < 1 < 2 < x₁`, encloses both poles of
`1/((z - 1)(z - 2))`: its boundary integral is `0` (the text: `x₀ = 0.5`, `x₁ = 2.5`, `h = 1`). -/
theorem rect_both (x₀ x₁ h : ℝ) (hx₀ : x₀ < 1) (hx₁ : 2 < x₁) (hh : 0 < h) :
    (((∫ x in x₀..x₁, 1 / (((x : ℂ) + (-h : ℝ) * I - 1) * ((x : ℂ) + (-h : ℝ) * I - 2))) -
      ∫ x in x₀..x₁, 1 / (((x : ℂ) + h * I - 1) * ((x : ℂ) + h * I - 2))) +
      I * ∫ y in (-h)..h, 1 / (((x₁ : ℂ) + y * I - 1) * ((x₁ : ℂ) + y * I - 2))) -
      I * ∫ y in (-h)..h, 1 / (((x₀ : ℂ) + y * I - 1) * ((x₀ : ℂ) + y * I - 2)) = 0 := by
  have hor : ∀ c : ℝ, c ≠ 0 → (∫ x in x₀..x₁, 1 / (((x : ℂ) + c * I - 1) * ((x : ℂ) + c * I - 2))) =
      primF (x₁ + c * I) - primF (x₀ + c * I) := by
    intro c hc
    have hne : ∀ x : ℝ, ∀ k : ℂ, k.im = 0 → (x : ℂ) + c * I - k ≠ 0 := by
      intro x k hk h
      have := congrArg Complex.im h
      simp [hk] at this
      exact hc this
    have hd : ∀ x : ℝ, HasDerivAt (fun t : ℝ => primF ((t : ℂ) + c * I))
        (1 / (((x : ℂ) + c * I - 1) * ((x : ℂ) + c * I - 2))) x := by
      intro x
      have h3 := HasDerivAt.comp (x : ℂ) (hasDerivAt_primF ((x : ℂ) + c * I)
        (hne x 1 (by simp)) (slit_of_im_ne _ (by simpa using hc)))
        ((hasDerivAt_id (x : ℂ)).add_const (c * I))
      simpa using h3.comp_ofReal
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x)]
    exact (Continuous.div continuous_const (by fun_prop) fun x =>
      mul_ne_zero (hne x 1 (by simp)) (hne x 2 (by simp))).intervalIntegrable _ _
  have ver : ∀ a : ℝ, (a < 1 ∨ 2 < a) → I * (∫ y in (-h)..h, 1 / (((a : ℂ) + y * I - 1) * ((a : ℂ) + y * I - 2))) =
      primF (a + h * I) - primF (a + (-h : ℝ) * I) := by
    intro a ha
    have hs : ∀ y : ℝ, 1 - ((a : ℂ) + y * I - 1)⁻¹ ∈ slitPlane := by
      intro y
      by_cases hy : y = 0
      · subst hy; simpa using slit_of_real a ha
      · exact slit_of_im_ne _ (by simpa using hy)
    have hre : ∀ y : ℝ, ∀ k : ℝ, (a < k) ∨ (k < a) → (a : ℂ) + y * I - k ≠ 0 := by
      intro y k hk h
      have := congrArg Complex.re h
      simp at this
      rcases hk with hk | hk <;> linarith
    have hk1 : a < 1 ∨ 1 < a := by rcases ha with ha | ha; left; exact ha; right; linarith
    have hk2 : a < 2 ∨ 2 < a := by rcases ha with ha | ha; left; linarith; right; exact ha
    have hd : ∀ y : ℝ, HasDerivAt (fun t : ℝ => primF ((a : ℂ) + t * I))
        (I * (1 / (((a : ℂ) + y * I - 1) * ((a : ℂ) + y * I - 2)))) y := by
      intro y
      have h3 := HasDerivAt.comp (y : ℂ) (hasDerivAt_primF ((a : ℂ) + y * I) (by simpa using hre y 1 hk1) (hs y))
        (((hasDerivAt_id (y : ℂ)).mul_const I).const_add (a : ℂ))
      have h4 := h3.comp_ofReal
      convert h4 using 1
      simp; ring
    rw [← intervalIntegral.integral_const_mul]
    refine (intervalIntegral.integral_eq_sub_of_hasDerivAt (fun y _ => hd y) ?_).trans (by push_cast; ring)
    · refine Continuous.intervalIntegrable ?_ _ _
      refine continuous_const.mul (Continuous.div continuous_const (by fun_prop) fun y => mul_ne_zero ?_ ?_)
      · simpa using hre y 1 hk1
      · simpa using hre y 2 hk2
  rw [hor (-h) (by linarith), hor h hh.ne', ver x₁ (Or.inr hx₁), ver x₀ (Or.inl hx₀)]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.rect_both
-- test: ca-res-rect-test.sage

/-! ## exr-residue-infinity -/

/-- The sum of all residues vanishes: for `f = ∑ cᵢ/(z - wᵢ)^(kᵢ+1) + g` with every pole inside `|z| < ρ`,
the finite residues plus `Res_∞ f = -(2πi)⁻¹ ∮_{|z| = ρ} f` sum to `0`. -/
theorem sum_residues_zero {ι : Type*} (s : Finset ι) (ρ : ℝ) (hρ : 0 < ρ) (w c : ι → ℂ) (k : ι → ℕ)
    (hw : ∀ i ∈ s, w i ∈ ball (0 : ℂ) ρ) (g : ℂ → ℂ) (hg : DifferentiableOn ℂ g (closedBall 0 ρ)) :
    (∑ i ∈ s, if k i = 0 then c i else 0) +
      -residue (fun z => (∑ i ∈ s, c i / (z - w i) ^ (k i + 1)) + g z) 0 ρ = 0 := by
  rw [residue, residue_theorem_circle s 0 ρ hρ w c k hw g hg, ← mul_assoc,
    inv_mul_cancel₀ two_pi_I_ne_zero, one_mul]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.sum_residues_zero
-- test: ca-res-sum-residues-test.sage

/-- The poles of `1/(1 + z⁴)` are `z_k = z₀ iᵏ`, `z₀ = e^{iπ/4}`. -/
theorem fourth_roots_rotate (k : ℕ) :
    exp (((2 * k + 1) * π / 4 : ℝ) * I) = exp ((π / 4 : ℝ) * I) * I ^ k := by
  have h : (((2 * k + 1) * π / 4 : ℝ) : ℂ) * I = ((π / 4 : ℝ) : ℂ) * I + k * (π / 2 * I) := by
    push_cast; ring
  rw [h, Complex.exp_add, Complex.exp_nat_mul, Complex.exp_pi_div_two_mul_I]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourth_roots_rotate
-- test: ca-res-fourth-rotate-test.sage

/-- The four residues `-z_k/4 = -z₀ iᵏ/4` sum to `0`: `∑_{k < 4} ζ iᵏ = 0`. -/
theorem fourth_roots_sum (ζ : ℂ) : ∑ k ∈ Finset.range 4, ζ * I ^ k = 0 := by
  have hI : I ^ 2 = -1 := Complex.I_sq
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  linear_combination (ζ + ζ * I) * hI

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourth_roots_sum
-- test: ca-res-fourth-sum-test.sage

/-- At infinity `1/(1 + z⁴) = ∑ (-1)ⁿ z^(-4n-4)` for `|z| > 1`: no `1/z` term, so `Res_∞ = 0` and the four finite
residues sum to `0`. -/
theorem inv_one_add_fourth_infinity (z : ℂ) (hz : 1 < ‖z‖) :
    HasSum (fun n : ℕ => (-1) ^ n * z⁻¹ ^ (4 * n + 4)) (1 / (1 + z ^ 4)) := by
  have hz0 : z ≠ 0 := by rintro rfl; norm_num at hz
  have hq : ‖-(z⁻¹ ^ 4)‖ < 1 := by
    rw [norm_neg, norm_pow, norm_inv]
    exact pow_lt_one₀ (by positivity) (inv_lt_one_of_one_lt₀ hz) (by norm_num)
  have := (hasSum_geometric_of_norm_lt_one hq).mul_left (z⁻¹ ^ 4)
  convert this using 1
  · funext n; rw [neg_pow, pow_add, pow_mul]; ring
  · have h1 : 1 - -(z⁻¹ ^ 4) ≠ 0 := by
      intro h
      have : ‖-(z⁻¹ ^ 4)‖ = 1 := by rw [show -(z⁻¹ ^ 4) = 1 by linear_combination -h]; simp
      linarith
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ResidueTheorem.inv_one_add_fourth_infinity
-- test: ca-res-fourth-infinity-test.sage

end KaluzaKlein.ComplexAnalysis.ResidueTheorem

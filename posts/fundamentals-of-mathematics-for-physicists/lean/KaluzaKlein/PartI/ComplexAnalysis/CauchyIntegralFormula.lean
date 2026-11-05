import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.Contours
import KaluzaKlein.PartI.ComplexAnalysis.ContourIntegrals
import KaluzaKlein.PartI.ComplexAnalysis.CauchyTheorem
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Cauchy's integral formula"

The claims of the section "Cauchy's integral formula" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): Cauchy's integral formula and its derivative form (mathlib, at the centre of a disc), the Cauchy estimates, the mean value property and the worked circle integrals of the section.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula

open KaluzaKlein.ComplexAnalysis.Contours KaluzaKlein.ComplexAnalysis.ContourIntegrals
  KaluzaKlein.ComplexAnalysis.CauchyTheorem

theorem cif_inv (f : ℂ → ℂ) (c w : ℂ) (R : ℝ) (hf : DifferentiableOn ℂ f (closedBall c R)) (hw : w ∈ ball c R) :
    f w = (2 * π * I)⁻¹ * ∮ z in C(c, R), f z / (z - w) := by
  have h := hf.circleIntegral_sub_inv_smul hw
  simp only [smul_eq_mul] at h
  have e : (∮ z in C(c, R), f z / (z - w)) = ∮ z in C(c, R), (z - w)⁻¹ * f z := by
    simp_rw [div_eq_inv_mul]
  rw [e, h, ← mul_assoc, inv_mul_cancel₀ two_pi_I_ne_zero, one_mul]

theorem cif_deriv_div (f : ℂ → ℂ) (c : ℂ) (R : ℝ) (n : ℕ) (hR : 0 < R) (hf : DifferentiableOn ℂ f (closedBall c R)) :
    iteratedDeriv n f c = n.factorial / (2 * π * I) * ∮ z in C(c, R), f z / (z - c) ^ (n + 1) := by
  have h := hf.circleIntegral_one_div_sub_center_pow_smul hR n
  simp only [smul_eq_mul] at h
  have e : (∮ z in C(c, R), f z / (z - c) ^ (n + 1)) = ∮ z in C(c, R), 1 / (z - c) ^ (n + 1) * f z := by
    congr 1; funext z; ring
  rw [e, h, ← mul_assoc]
  have hn : (n.factorial : ℂ) ≠ 0 := by exact_mod_cast n.factorial_ne_zero
  have h1 : (n.factorial : ℂ) / (2 * π * I) * (2 * π * I / n.factorial) = 1 := by
    field_simp
  rw [h1, one_mul]

theorem cif (f : ℂ → ℂ) (c w : ℂ) (R : ℝ) (hf : DifferentiableOn ℂ f (closedBall c R)) (hw : w ∈ ball c R) :
    2 * π * I * f w = ∮ z in C(c, R), f z / (z - w) := by
  rw [cif_inv f c w R hf hw, ← mul_assoc, mul_inv_cancel₀ two_pi_I_ne_zero, one_mul]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cif
-- test: ca-contour-cif-test.sage

theorem cif_deriv (f : ℂ → ℂ) (c : ℂ) (R : ℝ) (n : ℕ) (hR : 0 < R) (hf : DifferentiableOn ℂ f (closedBall c R)) :
    2 * π * I * iteratedDeriv n f c = n.factorial * ∮ z in C(c, R), f z / (z - c) ^ (n + 1) := by
  rw [cif_deriv_div f c R n hR hf, ← mul_assoc]
  congr 1
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cif_deriv
-- test: ca-contour-cif-deriv-test.sage

theorem holo_iteratedDeriv_differentiable (f : ℂ → ℂ) (U : Set ℂ) (hU : IsOpen U)
    (hf : DifferentiableOn ℂ f U) (n : ℕ) : DifferentiableOn ℂ (iteratedDeriv n f) U := by
  induction n with
  | zero => simpa using hf
  | succ n ih => rw [iteratedDeriv_succ]; exact ih.deriv hU

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.holo_iteratedDeriv_differentiable (Complex.exp) (Set.univ) (isOpen_univ) ((Complex.differentiable_exp.differentiableOn)) (0) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp]
  exact Complex.differentiable_exp.differentiableOn

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.holo_iteratedDeriv_differentiable (Complex.exp) (Set.univ) (isOpen_univ) ((Complex.differentiable_exp.differentiableOn)) (3) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp]
  exact Complex.differentiable_exp.differentiableOn

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.holo_iteratedDeriv_differentiable (Complex.exp) ((Metric.ball 0 1)) (Metric.isOpen_ball) ((Complex.differentiable_exp.differentiableOn)) (1) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp]
  exact Complex.differentiable_exp.differentiableOn

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.holo_iteratedDeriv_differentiable
-- test: ca-contour-holo-iteratedderiv-differentiable-test.sage

theorem cauchy_estimate (f : ℂ → ℂ) (c : ℂ) (R M : ℝ) (n : ℕ) (hR : 0 < R)
    (hf : DifferentiableOn ℂ f (closedBall c R)) (hM : ∀ z ∈ sphere c R, ‖f z‖ ≤ M) :
    ‖iteratedDeriv n f c‖ ≤ n.factorial * M / R ^ n :=
  Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le n hR
    (hf.mono closure_ball_subset_closedBall).diffContOnCl hM

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cauchy_estimate
-- test: ca-contour-cauchy-estimate-test.sage

theorem cif_exp (R : ℝ) (hR : 0 < R) :
    (∮ z in C(0, R), exp z / z) = 2 * π * I ∧ (∮ z in C(0, R), exp z / z ^ 2) = 2 * π * I := by
  constructor
  · have h := (differentiable_exp.differentiableOn (s := closedBall (0 : ℂ) R)).circleIntegral_sub_inv_smul
      (w := 0) (mem_ball_self hR)
    simpa [div_eq_inv_mul] using h
  · have h := (differentiable_exp.differentiableOn (s := closedBall (0 : ℂ) R)).circleIntegral_one_div_sub_center_pow_smul
      hR 1
    simpa [div_eq_inv_mul] using h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cif_exp
-- test: ca-contour-cif-exp-test.sage

theorem mean_value (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hr : 0 < r) (hf : DifferentiableOn ℂ f (closedBall c r)) :
    f c = ((2 * π)⁻¹ : ℝ) • ∫ t in (0 : ℝ)..2 * π, f (circleMap c r t) := by
  have h : DiffContOnCl ℂ f (ball c |r|) := by
    rw [abs_of_pos hr]; exact (hf.mono closure_ball_subset_closedBall).diffContOnCl
  rw [← h.circleAverage, Real.circleAverage_def]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.mean_value
-- test: ca-contour-mean-value-test.sage

theorem circle_dz_over (c : ℂ) (r t : ℝ) : deriv (circleMap c r) t = I * (circleMap c r t - c) := by
  rw [deriv_circleMap, circleMap_sub_center]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.circle_dz_over
-- test: ca-contour-circle-dz-over-test.sage

theorem mean_value_re (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hr : 0 < r) (hf : DifferentiableOn ℂ f (closedBall c r)) :
    (f c).re = (2 * π)⁻¹ * ∫ t in (0 : ℝ)..2 * π, (f (circleMap c r t)).re := by
  have hint : IntervalIntegrable (fun t => f (circleMap c r t)) volume 0 (2 * π) :=
    (hf.continuousOn.comp_continuous (continuous_circleMap c r)
      (fun t => sphere_subset_closedBall (circleMap_mem_sphere c hr.le t))).intervalIntegrable _ _
  rw [mean_value f c r hr hf, Complex.real_smul, re_ofReal_mul]
  congr 1
  have := ContinuousLinearMap.intervalIntegral_comp_comm reCLM hint
  simpa using this.symm

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.mean_value_re
-- test: ca-contour-mean-value-re-test.sage

theorem derivs_from_boundary (f g : ℂ → ℂ) (c : ℂ) (R : ℝ) (n : ℕ) (hR : 0 < R)
    (hf : DifferentiableOn ℂ f (closedBall c R)) (hg : DifferentiableOn ℂ g (closedBall c R))
    (hfg : ∀ z ∈ sphere c R, f z = g z) : iteratedDeriv n f c = iteratedDeriv n g c := by
  rw [cif_deriv_div f c R n hR hf, cif_deriv_div g c R n hR hg,
    circleIntegral.integral_congr hR.le (fun z hz => by beta_reduce; rw [hfg z hz])]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.derivs_from_boundary
-- test: ca-contour-derivs-from-boundary-test.sage

theorem cos_div_inside (R : ℝ) (hR : π < R) : (∮ z in C(0, R), cos z / (z - π)) = -(2 * π * I) := by
  have hw : (π : ℂ) ∈ ball (0 : ℂ) R := by
    rw [mem_ball, dist_zero_right, Complex.norm_real, Real.norm_of_nonneg Real.pi_pos.le]; exact hR
  have h := (differentiable_cos.differentiableOn (s := closedBall (0 : ℂ) R)).circleIntegral_sub_inv_smul hw
  simp only [smul_eq_mul, cos_pi] at h
  simp_rw [div_eq_inv_mul]
  rw [h]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cos_div_inside
-- test: ca-contour-cos-div-inside-test.sage

theorem cos_div_outside (R : ℝ) (hR0 : 0 ≤ R) (hR : R < π) : (∮ z in C(0, R), cos z / (z - π)) = 0 := by
  apply cauchy_disc _ 0 R hR0
  apply differentiable_cos.differentiableOn.div (differentiableOn_id.sub_const _)
  intro z hz hzw
  rw [sub_eq_zero] at hzw
  subst hzw
  simp [Real.pi_pos.le, abs_of_pos Real.pi_pos] at hz
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cos_div_outside
-- test: ca-contour-cos-div-outside-test.sage

theorem exp2_div_cube (R : ℝ) (hR : 0 < R) : (∮ z in C(0, R), exp (2 * z) / z ^ 3) = 4 * π * I := by
  have d1 : deriv (fun z : ℂ => exp (2 * z)) = fun z => 2 * exp (2 * z) := by
    funext z
    have := ((hasDerivAt_id' z).const_mul (2 : ℂ)).cexp
    rw [this.deriv]; ring
  have d2 : deriv (fun z : ℂ => 2 * exp (2 * z)) = fun z => 4 * exp (2 * z) := by
    funext z
    have := (((hasDerivAt_id' z).const_mul (2 : ℂ)).cexp).const_mul (2 : ℂ)
    rw [this.deriv]; ring
  have hd2 : iteratedDeriv 2 (fun z : ℂ => exp (2 * z)) 0 = 4 := by
    rw [iteratedDeriv_eq_iterate, Function.iterate_succ_apply', Function.iterate_one, d1, d2]; simp
  have hdiff : DifferentiableOn ℂ (fun z : ℂ => exp (2 * z)) (closedBall 0 R) := by fun_prop
  have h := hdiff.circleIntegral_one_div_sub_center_pow_smul hR 2
  simp only [smul_eq_mul, sub_zero, hd2] at h
  have e : (∮ z in C(0, R), exp (2 * z) / z ^ 3) = ∮ z in C(0, R), 1 / z ^ (2 + 1) * exp (2 * z) := by
    congr 1; funext z; ring
  rw [e, h]
  simp [Nat.factorial]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.exp2_div_cube
-- test: ca-contour-exp2-div-cube-test.sage

theorem sq_div_sq (R : ℝ) (hR : 1 < R) : (∮ z in C(0, R), z ^ 2 / (z - 1) ^ 2) = 4 * π * I := by
  have hR0 : 0 < R := by linarith
  have h : Set.EqOn (fun z : ℂ => z ^ 2 / (z - 1) ^ 2)
      (fun z => ((1 : ℂ) + 2 * (z - 1)⁻¹) + ((z - 1) ^ 2)⁻¹) (sphere 0 R) := by
    intro z hz
    have hz' : ‖z‖ = R := by simpa using hz
    have h1 : z - 1 ≠ 0 := fun h => by
      rw [sub_eq_zero] at h; rw [h] at hz'; simp at hz'; linarith
    simp only
    field_simp
    ring
  have i1 : CircleIntegrable (fun z : ℂ => (1 : ℂ) + 2 * (z - 1)⁻¹) 0 R := by
    apply ContinuousOn.circleIntegrable hR0.le
    apply continuousOn_const.add (continuousOn_const.mul _)
    apply ContinuousOn.inv₀ (by fun_prop)
    intro z hz h0; rw [sub_eq_zero] at h0; subst h0; simp at hz; linarith
  have i2 : CircleIntegrable (fun z : ℂ => ((z - 1) ^ 2)⁻¹) 0 R := by
    apply ContinuousOn.circleIntegrable hR0.le
    apply ContinuousOn.inv₀ (by fun_prop)
    intro z hz h0; rw [pow_eq_zero_iff two_ne_zero, sub_eq_zero] at h0; subst h0; simp at hz; linarith
  have i3 : CircleIntegrable (fun z : ℂ => (z - 1)⁻¹) 0 R := by
    apply ContinuousOn.circleIntegrable hR0.le
    apply ContinuousOn.inv₀ (by fun_prop)
    intro z hz h0; rw [sub_eq_zero] at h0; subst h0; simp at hz; linarith
  have i3' : CircleIntegrable (fun z : ℂ => 2 * (z - 1)⁻¹) 0 R := i3.const_smul (a := (2 : ℂ))
  rw [circleIntegral.integral_congr hR0.le h, circleIntegral.integral_add i1 i2,
    circleIntegral.integral_add (circleIntegrable_const 1 0 R) i3',
    circleIntegral.integral_const_mul]
  have c0 : (∮ z in C(0, R), (1 : ℂ)) = 0 := by
    simpa using circleIntegral.integral_sub_zpow_of_ne (show (0 : ℤ) ≠ -1 by norm_num) 0 1 R
  have c1 : (∮ z in C(0, R), (z - 1)⁻¹) = 2 * π * I :=
    circleIntegral.integral_sub_inv_of_mem_ball (by simpa using hR)
  have c2 : (∮ z in C(0, R), ((z - 1) ^ 2)⁻¹) = 0 := by
    simpa using circleIntegral.integral_sub_zpow_of_ne (show (-2 : ℤ) ≠ -1 by norm_num) 0 1 R
  rw [c0, c1, c2]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.sq_div_sq
-- test: ca-contour-sq-div-sq-test.sage

theorem exp_norm_le_on_circle (z : ℂ) (hz : ‖z‖ = 1) : ‖exp z‖ ≤ Real.exp 1 := by
  rw [norm_exp]; exact Real.exp_le_exp.2 (le_trans (re_le_norm z) hz.le)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.exp_norm_le_on_circle
-- test: ca-contour-exp-norm-le-on-circle-test.sage

theorem iteratedDeriv_exp_zero (n : ℕ) : iteratedDeriv n exp 0 = 1 := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp, exp_zero]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.iteratedDeriv_exp_zero (0) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp, Complex.exp_zero]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.iteratedDeriv_exp_zero (1) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp, Complex.exp_zero]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.iteratedDeriv_exp_zero (10) := by
  rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp, Complex.exp_zero]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.iteratedDeriv_exp_zero
-- test: ca-contour-iteratedderiv-exp-zero-test.sage

theorem exp_estimate (n : ℕ) : ‖iteratedDeriv n exp 0‖ ≤ n.factorial * Real.exp 1 := by
  have := cauchy_estimate exp 0 1 (Real.exp 1) n one_pos differentiable_exp.differentiableOn
    (fun z hz => exp_norm_le_on_circle z (by simpa using hz))
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.exp_estimate
-- test: ca-contour-exp-estimate-test.sage

theorem exp_estimate_value : 9860000 < (Nat.factorial 10 : ℝ) * Real.exp 1 ∧
    (Nat.factorial 10 : ℝ) * Real.exp 1 < 9870000 := by
  have he1 : Real.exp 1 < 2.7183 := by
    veriphy_certified (exact lt_trans Real.exp_one_lt_d9 (by norm_num))
    -- certificate: ca-contour-bound.sage
  have he2 : 2.7182 < Real.exp 1 := by
    veriphy_certified (exact lt_trans (by norm_num) Real.exp_one_gt_d9)
    -- certificate: ca-contour-bound.sage
  norm_num [Nat.factorial]
  constructor <;> linarith

end KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula

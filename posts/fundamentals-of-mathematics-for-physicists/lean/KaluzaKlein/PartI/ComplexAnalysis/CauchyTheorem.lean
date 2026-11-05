import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.Contours
import KaluzaKlein.PartI.ComplexAnalysis.ContourIntegrals
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Cauchy's theorem"

The claims of the section "Cauchy's theorem" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): Cauchy's theorem on discs, rectangles and annuli (mathlib), primitives on discs, closed integrals of entire functions, `1/(z - w)` inside and outside the circle, the square around 0, `1/z²`, and the branch jump of the principal logarithm.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.CauchyTheorem

open KaluzaKlein.ComplexAnalysis.Contours KaluzaKlein.ComplexAnalysis.ContourIntegrals

theorem cauchy_disc (f : ℂ → ℂ) (c : ℂ) (R : ℝ) (hR : 0 ≤ R) (hf : DifferentiableOn ℂ f (closedBall c R)) :
    (∮ z in C(c, R), f z) = 0 :=
  (hf.mono closure_ball_subset_closedBall).diffContOnCl.circleIntegral_eq_zero hR

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.CauchyTheorem.cauchy_disc
-- test: ca-contour-cauchy-disc-test.sage

theorem cauchy_rect (f : ℂ → ℂ) (z w : ℂ)
    (hf : DifferentiableOn ℂ f (Set.uIcc z.re w.re ×ℂ Set.uIcc z.im w.im)) :
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (w.re + y * I)) - I • (∫ y : ℝ in z.im..w.im, f (z.re + y * I)) = 0 :=
  Complex.integral_boundary_rect_eq_zero_of_differentiableOn f z w hf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.cauchy_rect
-- test: ca-contour-cauchy-rect-test.sage

theorem cauchy_annulus (f : ℂ → ℂ) (c : ℂ) (r R : ℝ) (h0 : 0 < r) (hle : r ≤ R)
    (hf : DifferentiableOn ℂ f (closedBall c R \ ball c r)) :
    (∮ z in C(c, R), f z) = ∮ z in C(c, r), f z :=
  Complex.circleIntegral_eq_of_differentiable_on_annulus_off_countable h0 hle Set.countable_empty
    hf.continuousOn (fun z hz => hf.differentiableAt (Filter.mem_of_superset
      ((isOpen_ball.sdiff isClosed_closedBall).mem_nhds hz.1)
      (Set.diff_subset_diff ball_subset_closedBall ball_subset_closedBall)))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.cauchy_annulus
-- test: ca-contour-cauchy-annulus-test.sage

theorem primitive_on_disc (f : ℂ → ℂ) (c : ℂ) (r : ℝ) (hf : DifferentiableOn ℂ f (ball c r)) :
    ∃ F : ℂ → ℂ, ∀ z ∈ ball c r, HasDerivAt F (f z) z :=
  hf.isExactOn_ball

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.primitive_on_disc (Complex.exp) (0) (1) ((Complex.differentiable_exp.differentiableOn)) := by
  exact ⟨Complex.exp, fun z _ => Complex.hasDerivAt_exp z⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.primitive_on_disc (Complex.cos) ((1 + Complex.I)) (2) ((Complex.differentiable_cos.differentiableOn)) := by
  exact ⟨Complex.sin, fun z _ => Complex.hasDerivAt_sin z⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.primitive_on_disc ((fun w : ℂ => 2 * w)) (0) (0) ((by fun_prop)) := by
  exact ⟨fun w => w ^ 2, fun z _ => by simpa using hasDerivAt_pow 2 z⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyTheorem.primitive_on_disc
-- test: ca-contour-primitive-on-disc-test.sage

theorem cint_closed_of_entire (f : ℂ → ℂ) (hf : Differentiable ℂ f) (γ : ℝ → ℂ) (a b : ℝ)
    (hγ : ContDiff ℝ 1 γ) (hclosed : γ a = γ b) : cint f γ a b = 0 := by
  obtain ⟨F, hF⟩ := hf.isExactOn_univ
  rw [cint_entire_primitive f F γ a b hγ (fun z => hF z (Set.mem_univ z)) hf.continuous, hclosed, sub_self]

theorem entire_primitives (z : ℂ) : HasDerivAt exp (exp z) z ∧ HasDerivAt sin (cos z) z :=
  ⟨hasDerivAt_exp z, hasDerivAt_sin z⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.entire_primitives
-- test: ca-contour-entire-primitives-test.sage

theorem entire_closed_integrals (p : Polynomial ℂ) (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ)
    (hclosed : γ a = γ b) :
    cint exp γ a b = 0 ∧ cint cos γ a b = 0 ∧ cint (fun z => p.eval z) γ a b = 0 :=
  ⟨cint_closed_of_entire _ differentiable_exp γ a b hγ hclosed,
    cint_closed_of_entire _ differentiable_cos γ a b hγ hclosed,
    cint_closed_of_entire _ p.differentiable γ a b hγ hclosed⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.entire_closed_integrals
-- test: ca-contour-entire-closed-integrals-test.sage

theorem circle_inv_sub (w : ℂ) (R : ℝ) (hR : 0 ≤ R) (hw : ‖w‖ ≠ R) :
    (∮ z in C(0, R), (z - w)⁻¹) = if ‖w‖ < R then 2 * π * I else 0 := by
  split_ifs with h
  · exact circleIntegral.integral_sub_inv_of_mem_ball (by simpa using h)
  · have h' : R < ‖w‖ := lt_of_le_of_ne (not_lt.1 h) (Ne.symm hw)
    apply cauchy_disc _ 0 R hR
    apply DifferentiableOn.inv (differentiableOn_id.sub_const w)
    intro z hz hzw
    have hzw' : z = w := by simpa [sub_eq_zero] using hzw
    subst hzw'
    simp at hz
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.CauchyTheorem.circle_inv_sub
-- test: ca-contour-circle-inv-sub-test.sage

theorem circle_inv_any (c : ℂ) (R : ℝ) (h : ‖c‖ < R) : (∮ z in C(c, R), z⁻¹) = 2 * π * I := by
  have := circleIntegral.integral_sub_inv_of_mem_ball (c := c) (w := 0) (R := R)
    (by rw [mem_ball, dist_comm, dist_zero_right]; exact h)
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.circle_inv_any
-- test: ca-contour-circle-inv-any-test.sage

/-- The integral of `f` over the counter-clockwise boundary of the square with vertices `s(±1 ± i)`. -/
def squareInt (f : ℂ → ℂ) (s : ℝ) : ℂ :=
  cint f (seg (s * (1 - I)) (s * (1 + I))) 0 1 + cint f (seg (s * (1 + I)) (s * (-1 + I))) 0 1 +
    cint f (seg (s * (-1 + I)) (s * (-1 - I))) 0 1 + cint f (seg (s * (-1 - I)) (s * (1 - I))) 0 1

theorem cint_inv_seg_rot (a b : ℂ) :
    cint (fun z => z⁻¹) (seg (I * a) (I * b)) 0 1 = cint (fun z => z⁻¹) (seg a b) 0 1 := by
  simp only [cint, deriv_seg, seg]
  congr 1; funext t
  rw [show I * a + (t : ℂ) * (I * b - I * a) = I * (a + t * (b - a)) by ring,
    show I * b - I * a = I * (b - a) by ring, mul_inv]
  calc I⁻¹ * (a + t * (b - a))⁻¹ * (I * (b - a)) = (I⁻¹ * I) * ((a + t * (b - a))⁻¹ * (b - a)) := by ring
    _ = _ := by rw [inv_mul_cancel₀ I_ne_zero, one_mul]

theorem side_integral_re : ∫ t in (0 : ℝ)..1, 2 * (2 * t - 1) / (1 + (2 * t - 1) ^ 2) = 0 := by
  have hd : ∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun t : ℝ => Real.log (4 * t ^ 2 - 4 * t + 2) / 2)
      (2 * (2 * t - 1) / (1 + (2 * t - 1) ^ 2)) t := by
    intro t _
    have hq : HasDerivAt (fun t : ℝ => 4 * t ^ 2 - 4 * t + 2) (8 * t - 4) t := by
      have := (((hasDerivAt_pow 2 t).const_mul (4 : ℝ)).sub ((hasDerivAt_id' t).const_mul (4 : ℝ))).add_const 2
      convert this using 1
      push_cast; ring
    have hpos : 0 < 4 * t ^ 2 - 4 * t + 2 := by nlinarith [sq_nonneg (2 * t - 1)]
    have hne : 4 * t ^ 2 - 4 * t + 2 ≠ 0 := hpos.ne'
    convert (hq.log hpos.ne').div_const 2 using 1
    rw [show (1 : ℝ) + (2 * t - 1) ^ 2 = 4 * t ^ 2 - 4 * t + 2 by ring]
    field_simp
    ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd
    ((by fun_prop (disch := intro t; positivity) : Continuous fun t : ℝ =>
      2 * (2 * t - 1) / (1 + (2 * t - 1) ^ 2)).intervalIntegrable _ _)]
  norm_num

theorem side_integral_im : ∫ t in (0 : ℝ)..1, 2 / (1 + (2 * t - 1) ^ 2) = π / 2 := by
  have hd : ∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun t : ℝ => Real.arctan (2 * t - 1))
      (2 / (1 + (2 * t - 1) ^ 2)) t := by
    intro t _
    have hu : HasDerivAt (fun t : ℝ => 2 * t - 1) 2 t := by
      simpa using ((hasDerivAt_id' t).const_mul (2 : ℝ)).sub_const 1
    have := hu.arctan
    convert this using 1
    field_simp
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd
    ((by fun_prop (disch := intro t; positivity) : Continuous fun t : ℝ =>
      2 / (1 + (2 * t - 1) ^ 2)).intervalIntegrable _ _)]
  norm_num
  ring

theorem square_side (s : ℝ) (hs : 0 < s) :
    cint (fun z => z⁻¹) (seg (s * (1 - I)) (s * (1 + I))) 0 1 = π / 2 * I := by
  have h : ∀ t : ℝ, (seg (s * (1 - I)) (s * (1 + I)) t)⁻¹ * deriv (seg (s * (1 - I)) (s * (1 + I))) t =
      ((2 * (2 * t - 1) / (1 + (2 * t - 1) ^ 2) : ℝ) : ℂ) + ((2 / (1 + (2 * t - 1) ^ 2) : ℝ) : ℂ) * I := by
    intro t
    rw [deriv_seg]
    simp only [seg]
    have hD : ((1 + (2 * t - 1) ^ 2 : ℝ) : ℂ) ≠ 0 := by exact_mod_cast (by positivity : (1 + (2 * t - 1) ^ 2 : ℝ) ≠ 0)
    have hx : (s : ℂ) * (1 - I) + (t : ℂ) * (s * (1 + I) - s * (1 - I)) ≠ 0 := by
      intro h0
      have := congrArg Complex.re h0
      simp at this
      linarith
    push_cast at hD ⊢
    have key : ((s : ℂ) * (1 + I) - s * (1 - I)) * (1 + (2 * t - 1) ^ 2) =
        (2 * (2 * t - 1) + 2 * I) * (s * (1 - I) + t * (s * (1 + I) - s * (1 - I))) := by
      linear_combination (-2 * (2 * (t : ℂ) - 1) * s) * I_sq
    calc (s * (1 - I) + t * (s * (1 + I) - s * (1 - I)))⁻¹ * (s * (1 + I) - s * (1 - I))
        = (s * (1 + I) - s * (1 - I)) / (s * (1 - I) + t * (s * (1 + I) - s * (1 - I))) := by ring
      _ = (2 * (2 * t - 1) + 2 * I) / (1 + (2 * t - 1) ^ 2) := (div_eq_div_iff hx hD).2 key
      _ = _ := by ring
  simp only [cint, h]
  rw [integral_re_im _ _ 0 1 (by fun_prop (disch := intro t; positivity))
    (by fun_prop (disch := intro t; positivity)), side_integral_re, side_integral_im]
  push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.square_side
-- test: ca-contour-square-side-test.sage

theorem square_inv (s : ℝ) (hs : 0 < s) : squareInt (fun z => z⁻¹) s = 2 * π * I := by
  have r1 : (s : ℂ) * (1 + I) = I * (s * (1 - I)) := by linear_combination (s : ℂ) * I_sq
  have r2 : (s : ℂ) * (-1 + I) = I * (s * (1 + I)) := by linear_combination (-(s : ℂ)) * I_sq
  have r3 : (s : ℂ) * (-1 - I) = I * (s * (-1 + I)) := by linear_combination (-(s : ℂ)) * I_sq
  have r4 : (s : ℂ) * (1 - I) = I * (s * (-1 - I)) := by linear_combination (s : ℂ) * I_sq
  have e2 : cint (fun z => z⁻¹) (seg (s * (1 + I)) (s * (-1 + I))) 0 1 =
      cint (fun z => z⁻¹) (seg (s * (1 - I)) (s * (1 + I))) 0 1 := by
    rw [r2, r1, cint_inv_seg_rot]
  have e3 : cint (fun z => z⁻¹) (seg (s * (-1 + I)) (s * (-1 - I))) 0 1 =
      cint (fun z => z⁻¹) (seg (s * (1 + I)) (s * (-1 + I))) 0 1 := by
    rw [r3, r2, cint_inv_seg_rot]
  have e4 : cint (fun z => z⁻¹) (seg (s * (-1 - I)) (s * (1 - I))) 0 1 =
      cint (fun z => z⁻¹) (seg (s * (-1 + I)) (s * (-1 - I))) 0 1 := by
    rw [r4, r3, cint_inv_seg_rot]
  unfold squareInt
  rw [e4, e3, e2, square_side s hs]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.square_inv
-- test: ca-contour-square-inv-test.sage

theorem poly5_primitive (z : ℂ) :
    HasDerivAt (fun z : ℂ => z ^ 6 / 6 - 3 * z ^ 2 / 2 + 2 * z) (z ^ 5 - 3 * z + 2) z := by
  have := (((hasDerivAt_pow 6 z).div_const 6).sub (((hasDerivAt_pow 2 z).const_mul 3).div_const 2)).add
    ((hasDerivAt_id' z).const_mul 2)
  convert this using 1
  push_cast
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.poly5_primitive
-- test: ca-contour-poly5-primitive-test.sage

theorem inv_sq_primitive (z : ℂ) (hz : z ≠ 0) : HasDerivAt (fun z : ℂ => -z⁻¹) ((z ^ 2)⁻¹) z := by
  have := (hasDerivAt_inv hz).neg
  rw [neg_neg] at this
  exact this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.inv_sq_primitive
-- test: ca-contour-inv-sq-primitive-test.sage

theorem cint_poly5_closed (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ) (hclosed : γ a = γ b) :
    cint (fun z => z ^ 5 - 3 * z + 2) γ a b = 0 := by
  rw [cint_entire_primitive _ _ γ a b hγ poly5_primitive (by fun_prop), hclosed, sub_self]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.cint_poly5_closed
-- test: ca-contour-cint-poly5-closed-test.sage

theorem cint_inv_sq_closed (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ) (h0 : ∀ t, γ t ≠ 0)
    (hclosed : γ a = γ b) : cint (fun z => (z ^ 2)⁻¹) γ a b = 0 :=
  cint_primitive_closed _ (fun z => -z⁻¹) γ a b (fun t _ => inv_sq_primitive (γ t) (h0 t))
    (fun t _ => ((hγ.differentiable one_ne_zero) t).hasDerivAt)
    (((hγ.continuous.pow 2).inv₀ (fun t => pow_ne_zero 2 (h0 t))).mul (hγ.continuous_deriv le_rfl)).continuousOn
    hclosed

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.CauchyTheorem.cint_inv_sq_closed
-- test: ca-contour-cint-inv-sq-closed-test.sage

theorem log_jump_start (a : ℝ) (ha : 1 < a) :
    Tendsto (fun t : ℝ => log (exp (t * I) - a)) (𝓝[>] 0) (𝓝 (Real.log (a - 1) + π * I)) := by
  have hz : ((1 - a : ℝ) : ℂ) = exp ((0 : ℝ) * I) - a := by push_cast; simp
  have hre : (((1 - a : ℝ)) : ℂ).re < 0 := by simp; linarith
  have him : (((1 - a : ℝ)) : ℂ).im = 0 := by simp
  have hlog := tendsto_log_nhdsWithin_im_nonneg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 - a : ℝ)) : ℂ)‖ = Real.log (a - 1) := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by linarith)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩)
  · rw [hz]
    exact ((continuous_exp.comp (continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left
      nhdsWithin_le_nhds
  · filter_upwards [Ioo_mem_nhdsGT Real.pi_pos] with t ht
    simp only [Set.mem_setOf_eq, sub_im, exp_ofReal_mul_I_im, ofReal_im, sub_zero]
    exact (Real.sin_pos_of_pos_of_lt_pi ht.1 ht.2).le

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_start ((3 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ) = Complex.exp (((0 : ℝ) : ℂ) * Complex.I) - (((3 : ℝ) : ℝ) : ℂ) := by push_cast; simp
  have hre : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_nonneg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ)‖ = Real.log (((3 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsGT Real.pi_pos] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  exact (Real.sin_pos_of_pos_of_lt_pi ht.1 ht.2).le

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_start ((2 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ) = Complex.exp (((0 : ℝ) : ℂ) * Complex.I) - (((2 : ℝ) : ℝ) : ℂ) := by push_cast; simp
  have hre : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_nonneg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ)‖ = Real.log (((2 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsGT Real.pi_pos] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  exact (Real.sin_pos_of_pos_of_lt_pi ht.1 ht.2).le

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_start ((3 / 2 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ) = Complex.exp (((0 : ℝ) : ℂ) * Complex.I) - (((3 / 2 : ℝ) : ℝ) : ℂ) := by push_cast; simp
  have hre : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_nonneg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ)‖ = Real.log (((3 / 2 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsGT Real.pi_pos] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  exact (Real.sin_pos_of_pos_of_lt_pi ht.1 ht.2).le

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_start
-- test: ca-contour-log-jump-start-test.sage

theorem log_jump_end (a : ℝ) (ha : 1 < a) :
    Tendsto (fun t : ℝ => log (exp (t * I) - a)) (𝓝[<] (2 * π)) (𝓝 (Real.log (a - 1) - π * I)) := by
  have hz : ((1 - a : ℝ) : ℂ) = exp (((2 * π : ℝ)) * I) - a := by push_cast; rw [exp_two_pi_mul_I]
  have hre : (((1 - a : ℝ)) : ℂ).re < 0 := by simp; linarith
  have him : (((1 - a : ℝ)) : ℂ).im = 0 := by simp
  have hlog := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 - a : ℝ)) : ℂ)‖ = Real.log (a - 1) := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by linarith)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩)
  · rw [hz]
    exact ((continuous_exp.comp (continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left
      nhdsWithin_le_nhds
  · filter_upwards [Ioo_mem_nhdsLT (show π < 2 * π by linarith [Real.pi_pos])] with t ht
    simp only [Set.mem_setOf_eq, sub_im, exp_ofReal_mul_I_im, ofReal_im, sub_zero]
    have := Real.sin_neg_of_neg_of_neg_pi_lt (show t - 2 * π < 0 by linarith [ht.2])
      (show -π < t - 2 * π by linarith [ht.1])
    rwa [Real.sin_sub_two_pi] at this

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_end ((3 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ) = Complex.exp (((2 * Real.pi : ℝ) : ℂ) * Complex.I) - (((3 : ℝ) : ℝ) : ℂ) := by push_cast; rw [Complex.exp_two_pi_mul_I]
  have hre : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (3 : ℝ) : ℝ) : ℂ)‖ = Real.log (((3 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsLT (show Real.pi < 2 * Real.pi by linarith [Real.pi_pos])] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  have := Real.sin_neg_of_neg_of_neg_pi_lt (show t - 2 * Real.pi < 0 by linarith [ht.2]) (show -Real.pi < t - 2 * Real.pi by linarith [ht.1])
  rwa [Real.sin_sub_two_pi] at this

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_end ((2 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ) = Complex.exp (((2 * Real.pi : ℝ) : ℂ) * Complex.I) - (((2 : ℝ) : ℝ) : ℂ) := by push_cast; rw [Complex.exp_two_pi_mul_I]
  have hre : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (2 : ℝ) : ℝ) : ℂ)‖ = Real.log (((2 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsLT (show Real.pi < 2 * Real.pi by linarith [Real.pi_pos])] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  have := Real.sin_neg_of_neg_of_neg_pi_lt (show t - 2 * Real.pi < 0 by linarith [ht.2]) (show -Real.pi < t - 2 * Real.pi by linarith [ht.1])
  rwa [Real.sin_sub_two_pi] at this

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_end ((3 / 2 : ℝ)) ((by norm_num)) := by
  have hz : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ) = Complex.exp (((2 * Real.pi : ℝ) : ℂ) * Complex.I) - (((3 / 2 : ℝ) : ℝ) : ℂ) := by push_cast; rw [Complex.exp_two_pi_mul_I]
  have hre : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ).re < 0 := by norm_num
  have him : (((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ).im = 0 := by simp
  have hlog := Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him
  have hnorm : Real.log ‖(((1 : ℝ) - (3 / 2 : ℝ) : ℝ) : ℂ)‖ = Real.log (((3 / 2 : ℝ) : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf
  rw [hnorm] at hlog
  refine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)
  filter_upwards [Ioo_mem_nhdsLT (show Real.pi < 2 * Real.pi by linarith [Real.pi_pos])] with t ht
  simp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]
  have := Real.sin_neg_of_neg_of_neg_pi_lt (show t - 2 * Real.pi < 0 by linarith [ht.2]) (show -Real.pi < t - 2 * Real.pi by linarith [ht.1])
  rwa [Real.sin_sub_two_pi] at this

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_end
-- test: ca-contour-log-jump-end-test.sage

end KaluzaKlein.ComplexAnalysis.CauchyTheorem

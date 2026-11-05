import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.Contours
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Contour integrals"

The claims of the section "Contour integrals" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): the contour integral `∫ₐᵇ f(γ t) γ'(t) dt`, its invariance under reparametrisation, reversal, the ML estimate, primitives, the circle integrals of `zⁿ`, path dependence for `z̄`, and the ML estimate of `∮ e^z/(z² + 1)`.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.ContourIntegrals

open KaluzaKlein.ComplexAnalysis.Contours

/-- The contour integral `∫_γ f(z) dz = ∫ₐᵇ f(γ(t)) γ'(t) dt`. -/
def cint (f : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ) : ℂ := ∫ t in a..b, f (γ t) * deriv γ t

theorem cint_circle (f : ℂ → ℂ) (c : ℂ) (R : ℝ) : cint f (circleMap c R) 0 (2 * π) = ∮ z in C(c, R), f z := by
  simp only [cint, circleIntegral, smul_eq_mul, mul_comm]

theorem cint_re_im (f : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ)
    (h : IntervalIntegrable (fun t => f (γ t) * deriv γ t) volume a b) :
    cint f γ a b = ((∫ t in a..b, (f (γ t) * deriv γ t).re : ℝ) : ℂ) +
      ((∫ t in a..b, (f (γ t) * deriv γ t).im : ℝ) : ℂ) * I := by
  have h1 : (cint f γ a b).re = ∫ t in a..b, (f (γ t) * deriv γ t).re := by
    have := ContinuousLinearMap.intervalIntegral_comp_comm reCLM h
    simpa [cint] using this.symm
  have h2 : (cint f γ a b).im = ∫ t in a..b, (f (γ t) * deriv γ t).im := by
    have := ContinuousLinearMap.intervalIntegral_comp_comm imCLM h
    simpa [cint] using this.symm
  apply Complex.ext <;> simp [h1, h2]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_re_im
-- test: ca-contour-cint-re-im-test.sage

theorem cint_comp (f : ℂ → ℂ) (γ : ℝ → ℂ) (φ φ' : ℝ → ℝ) (c d : ℝ) (hf : Continuous f)
    (hγ : Differentiable ℝ γ) (hγ' : Continuous (deriv γ)) (hφ : ∀ s, HasDerivAt φ (φ' s) s)
    (hφ'c : Continuous φ') : cint f (γ ∘ φ) c d = cint f γ (φ c) (φ d) := by
  unfold cint
  have hd : ∀ s, f ((γ ∘ φ) s) * deriv (γ ∘ φ) s = φ' s • (fun x => f (γ x) * deriv γ x) (φ s) := by
    intro s
    rw [((hγ (φ s)).hasDerivAt.scomp s (hφ s)).deriv]
    simp only [Function.comp_apply, real_smul]
    ring
  simp_rw [hd]
  have := intervalIntegral.integral_comp_smul_deriv (a := c) (b := d) (f := φ) (f' := φ')
    (g := fun x => f (γ x) * deriv γ x) (fun x _ => hφ x) hφ'c.continuousOn
    ((hf.comp hγ.continuous).mul hγ')
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_comp
-- test: ca-contour-cint-comp-test.sage

theorem cint_reverse (f : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ) : cint f (reverse γ a b) a b = -cint f γ a b := by
  unfold cint
  have h : ∀ t, f (reverse γ a b t) * deriv (reverse γ a b) t = -(f (γ (a + b - t)) * deriv γ (a + b - t)) := by
    intro t; rw [deriv_reverse]; simp [reverse]
  simp only [h, intervalIntegral.integral_neg]
  have := intervalIntegral.integral_comp_sub_left (fun s => f (γ s) * deriv γ s) (a + b) (a := a) (b := b)
  rw [this]
  congr 2 <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_reverse
-- test: ca-contour-cint-reverse-test.sage

theorem cint_ML (f : ℂ → ℂ) (γ : ℝ → ℂ) (a b M : ℝ) (hab : a ≤ b)
    (hM : ∀ t ∈ Set.Icc a b, ‖f (γ t)‖ ≤ M) (hγ : IntervalIntegrable (fun t => ‖deriv γ t‖) volume a b) :
    ‖cint f γ a b‖ ≤ M * length γ a b := by
  unfold cint length
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.norm_integral_le_of_norm_le hab
  · refine ae_of_all _ (fun t ht => ?_)
    rw [norm_mul]
    exact mul_le_mul_of_nonneg_right (hM t (Set.Ioc_subset_Icc_self ht)) (norm_nonneg _)
  · exact hγ.const_mul M

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_ML
-- test: ca-contour-cint-ml-test.sage

theorem cint_primitive (f F : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ)
    (hF : ∀ t ∈ Set.uIcc a b, HasDerivAt F (f (γ t)) (γ t))
    (hγ : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (deriv γ t) t)
    (hc : ContinuousOn (fun t => f (γ t) * deriv γ t) (Set.uIcc a b)) :
    cint f γ a b = F (γ b) - F (γ a) := by
  unfold cint
  apply intervalIntegral.integral_eq_sub_of_hasDerivAt (f := F ∘ γ)
  · intro t ht; exact (hF t ht).comp t (hγ t ht)
  · exact hc.intervalIntegrable

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_primitive
-- test: ca-contour-cint-primitive-test.sage

theorem cint_primitive_closed (f F : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ)
    (hF : ∀ t ∈ Set.uIcc a b, HasDerivAt F (f (γ t)) (γ t))
    (hγ : ∀ t ∈ Set.uIcc a b, HasDerivAt γ (deriv γ t) t)
    (hc : ContinuousOn (fun t => f (γ t) * deriv γ t) (Set.uIcc a b)) (hclosed : γ a = γ b) :
    cint f γ a b = 0 := by
  rw [cint_primitive f F γ a b hF hγ hc, hclosed, sub_self]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_primitive_closed
-- test: ca-contour-cint-primitive-closed-test.sage

theorem cint_entire_primitive (f F : ℂ → ℂ) (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ)
    (hF : ∀ z, HasDerivAt F (f z) z) (hf : Continuous f) : cint f γ a b = F (γ b) - F (γ a) :=
  cint_primitive f F γ a b (fun t _ => hF (γ t)) (fun t _ => ((hγ.differentiable one_ne_zero) t).hasDerivAt)
    ((hf.comp hγ.continuous).mul (hγ.continuous_deriv le_rfl)).continuousOn

theorem circle_param_zpow (n : ℤ) :
    (∮ z in C(0, 1), z ^ n) = I * ∫ t in (0 : ℝ)..2 * π, exp ((n + 1) * t * I) := by
  rw [circleIntegral, ← intervalIntegral.integral_const_mul]
  congr 1; funext t
  simp only [deriv_circleMap, circleMap_zero, ofReal_one, one_mul, smul_eq_mul]
  rw [← exp_int_mul, show ((n : ℂ) + 1) * t * I = n * (t * I) + t * I by ring, exp_add]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourIntegrals.circle_param_zpow
-- test: ca-contour-circle-param-zpow-test.sage

theorem circle_zpow (n : ℤ) (R : ℝ) (hR : 0 < R) :
    (∮ z in C(0, R), z ^ n) = if n = -1 then 2 * π * I else 0 := by
  split_ifs with h
  · subst h
    simpa using circleIntegral.integral_sub_inv_of_mem_ball (c := 0) (w := 0) (R := R) (by simpa using hR)
  · simpa using circleIntegral.integral_sub_zpow_of_ne h 0 0 R

attribute [veriphy_tested oracle_test 0 6] KaluzaKlein.ComplexAnalysis.ContourIntegrals.circle_zpow
-- test: ca-contour-circle-zpow-test.sage

theorem zpow_primitive (n : ℤ) (hn : n ≠ -1) (z : ℂ) (hz : z ≠ 0) :
    HasDerivAt (fun w : ℂ => w ^ (n + 1) / (n + 1)) (z ^ n) z := by
  have hn' : (n : ℂ) + 1 ≠ 0 := by exact_mod_cast (by omega : n + 1 ≠ 0)
  have := (hasDerivAt_zpow (n + 1) z (Or.inl hz)).div_const ((n : ℂ) + 1)
  convert this using 1
  rw [add_sub_cancel_right]; push_cast; field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourIntegrals.zpow_primitive
-- test: ca-contour-zpow-primitive-test.sage

theorem no_primitive_inv (c : ℂ) (R : ℝ) (hR : 0 < R) :
    ¬ ∃ F : ℂ → ℂ, ∀ z ∈ sphere c R, HasDerivAt F (z - c)⁻¹ z := by
  rintro ⟨F, hF⟩
  have h0 := circleIntegral.integral_eq_zero_of_hasDerivWithinAt (f' := fun z => (z - c)⁻¹) hR.le
    (fun z hz => (hF z hz).hasDerivWithinAt)
  rw [circleIntegral.integral_sub_inv_of_mem_ball (mem_ball_self hR)] at h0
  simp [Real.pi_ne_zero, I_ne_zero] at h0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.no_primitive_inv
-- test: ca-contour-no-primitive-inv-test.sage

theorem cube_one_add (x : ℂ) : (1 + x) ^ 3 = -2 + 2 * x + (3 + x) * (x ^ 2 + 1) := by ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.cube_one_add (0) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.cube_one_add (Complex.I) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.cube_one_add ((2 - Complex.I)) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.cube_one_add ((-1)) := by
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cube_one_add
-- test: ca-contour-cube-one-add-test.sage

theorem sq_one_add (x : ℂ) : (1 + x) ^ 2 = 2 * x + (x ^ 2 + 1) := by ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.sq_one_add (0) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.sq_one_add (Complex.I) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.sq_one_add ((2 - Complex.I)) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourIntegrals.sq_one_add ((-1)) := by
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ContourIntegrals.sq_one_add
-- test: ca-contour-sq-one-add-test.sage

theorem one_add_I_cube : (1 + I) ^ 3 / 3 = (-2 + 2 * I) / 3 := by rw [cube_one_add I, I_sq]; ring

theorem one_add_I_sq : (1 + I) ^ 2 / 2 = I := by rw [sq_one_add I, I_sq]; ring

theorem cint_sq (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ) :
    cint (fun z => z ^ 2) γ a b = (γ b ^ 3 - γ a ^ 3) / 3 := by
  have hF : ∀ z : ℂ, HasDerivAt (fun z : ℂ => z ^ 3 / 3) (z ^ 2) z := by
    intro z
    have := (hasDerivAt_pow 3 z).div_const 3
    convert this using 1
    push_cast; ring
  rw [cint_entire_primitive _ _ γ a b hγ hF (by fun_prop)]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_sq
-- test: ca-contour-cint-sq-test.sage

theorem cint_id (γ : ℝ → ℂ) (a b : ℝ) (hγ : ContDiff ℝ 1 γ) :
    cint (fun z => z) γ a b = (γ b ^ 2 - γ a ^ 2) / 2 := by
  have hF : ∀ z : ℂ, HasDerivAt (fun z : ℂ => z ^ 2 / 2) z z := by
    intro z
    have := (hasDerivAt_pow 2 z).div_const 2
    convert this using 1
    push_cast; ring
  rw [cint_entire_primitive _ _ γ a b hγ hF (by fun_prop)]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_id
-- test: ca-contour-cint-id-test.sage

theorem cint_conj_seg (w : ℂ) : cint (starRingEnd ℂ) (seg 0 w) 0 1 = ((‖w‖ ^ 2 / 2 : ℝ) : ℂ) := by
  have h : ∀ t : ℝ, (starRingEnd ℂ) (seg 0 w t) * deriv (seg 0 w) t = ((t * ‖w‖ ^ 2 : ℝ) : ℂ) := by
    intro t
    rw [deriv_seg]
    simp only [seg, zero_add, sub_zero, map_mul, conj_ofReal]
    rw [mul_assoc, Complex.conj_mul']
    push_cast; ring
  simp only [cint, h, intervalIntegral.integral_ofReal]
  congr 1
  rw [intervalIntegral.integral_mul_const, integral_id]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_conj_seg
-- test: ca-contour-cint-conj-seg-test.sage

/-- The parabola `t ↦ t + i c t²`. -/
def parabolaC (c : ℝ) : ℝ → ℂ := fun s => (s : ℂ) + I * c * (s : ℂ) ^ 2

theorem parabolaC_hasDerivAt (c t : ℝ) : HasDerivAt (parabolaC c) (1 + 2 * I * c * t) t := by
  have h1 := hasDerivAt_ofReal' t
  have h3 := h1.add ((h1.mul h1).const_mul (I * c))
  have e : parabolaC c = fun s : ℝ => (s : ℂ) + I * c * ((s : ℂ) * s) := by
    funext s; simp only [parabolaC]; ring
  rw [e]
  convert h3 using 1
  ring

theorem contDiff_parabolaC (c : ℝ) : ContDiff ℝ 1 (parabolaC c) :=
  contDiff_ofReal'.add (contDiff_const.mul (contDiff_ofReal'.pow 2))

theorem integral_re_im (p q : ℝ → ℝ) (a b : ℝ) (hp : Continuous p) (hq : Continuous q) :
    ∫ t in a..b, (((p t : ℝ) : ℂ) + ((q t : ℝ) : ℂ) * I) =
      ((∫ t in a..b, p t : ℝ) : ℂ) + ((∫ t in a..b, q t : ℝ) : ℂ) * I := by
  have h1 : IntervalIntegrable (fun t => ((p t : ℝ) : ℂ)) volume a b :=
    (continuous_ofReal.comp hp).intervalIntegrable _ _
  have h2 : IntervalIntegrable (fun t => ((q t : ℝ) : ℂ) * I) volume a b :=
    ((continuous_ofReal.comp hq).mul continuous_const).intervalIntegrable _ _
  rw [intervalIntegral.integral_add h1 h2, intervalIntegral.integral_mul_const, intervalIntegral.integral_ofReal, intervalIntegral.integral_ofReal]

theorem cint_conj_parabola (c : ℝ) :
    cint (starRingEnd ℂ) (parabolaC c) 0 1 = (1 + c ^ 2) / 2 + I * c / 3 := by
  have h : ∀ t : ℝ, (starRingEnd ℂ) (parabolaC c t) * deriv (parabolaC c) t
      = ((t + 2 * c ^ 2 * t ^ 3 : ℝ) : ℂ) + ((c * t ^ 2 : ℝ) : ℂ) * I := by
    intro t
    rw [(parabolaC_hasDerivAt c t).deriv]
    simp only [parabolaC, map_add, map_mul, conj_ofReal, conj_I, map_pow]
    push_cast
    linear_combination (-2 * (c : ℂ) ^ 2 * (t : ℂ) ^ 3) * I_sq
  simp only [cint, h]
  rw [integral_re_im _ _ 0 1 (by fun_prop) (by fun_prop)]
  have e1 : ∫ t in (0 : ℝ)..1, (t + 2 * c ^ 2 * t ^ 3) = (1 + c ^ 2) / 2 := by
    rw [intervalIntegral.integral_add (continuous_id'.intervalIntegrable _ _)
      ((by fun_prop : Continuous fun t : ℝ => 2 * c ^ 2 * t ^ 3).intervalIntegrable _ _),
      intervalIntegral.integral_const_mul, integral_id, integral_pow]
    ring
  have e2 : ∫ t in (0 : ℝ)..1, c * t ^ 2 = c / 3 := by
    rw [intervalIntegral.integral_const_mul, integral_pow]; ring
  rw [e1, e2]
  push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_conj_parabola
-- test: ca-contour-cint-conj-parabola-test.sage

theorem conj_no_primitive (F : ℂ → ℂ) : ¬ ∀ z, HasDerivAt F (starRingEnd ℂ z) z := by
  intro hF
  have h1 := cint_entire_primitive (starRingEnd ℂ) F (seg 0 (1 + I)) 0 1 (contDiff_seg 0 (1 + I)) hF
    continuous_conj
  have h2 := cint_entire_primitive (starRingEnd ℂ) F (parabolaC 1) 0 1 (contDiff_parabolaC 1) hF
    continuous_conj
  rw [cint_conj_seg] at h1
  rw [cint_conj_parabola] at h2
  have e1 : seg 0 (1 + I) 1 = parabolaC 1 1 := by simp [seg, parabolaC] <;> ring
  have e0 : seg 0 (1 + I) 0 = parabolaC 1 0 := by simp [seg, parabolaC]
  rw [e1, e0, ← h2] at h1
  have := congrArg Complex.im h1
  rw [ofReal_im] at this
  simp at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.conj_no_primitive
-- test: ca-contour-conj-no-primitive-test.sage

theorem ml_bound (R : ℝ) (hR : 1 < R) :
    ‖∮ z in C(0, R), exp z / (z ^ 2 + 1)‖ ≤ 2 * π * R * (Real.exp R / (R ^ 2 - 1)) := by
  apply circleIntegral.norm_integral_le_of_norm_le_const (by linarith)
  intro z hz
  have hz' : ‖z‖ = R := by simpa using hz
  rw [norm_div]
  have h1 : ‖exp z‖ ≤ Real.exp R := by
    rw [norm_exp]; exact Real.exp_le_exp.2 (le_trans (re_le_norm z) hz'.le)
  have h2 : R ^ 2 - 1 ≤ ‖z ^ 2 + 1‖ := by
    have := norm_sub_norm_le (z ^ 2) (-1)
    rw [sub_neg_eq_add, norm_neg, norm_one, norm_pow, hz'] at this
    linarith
  have hpos : 0 < R ^ 2 - 1 := by nlinarith
  exact div_le_div₀ (Real.exp_pos R).le h1 hpos h2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.ml_bound
-- test: ca-contour-ml-bound-test.sage

theorem ml_bound_value : 30.9 < 4 * π * Real.exp 2 / 3 ∧ 4 * π * Real.exp 2 / 3 < 31 := by
  have he1 : Real.exp 1 < 2.7183 := by
    veriphy_certified (exact lt_trans Real.exp_one_lt_d9 (by norm_num))
    -- certificate: ca-contour-bound.sage
  have he2 : 2.7182 < Real.exp 1 := by
    veriphy_certified (exact lt_trans (by norm_num) Real.exp_one_gt_d9)
    -- certificate: ca-contour-bound.sage
  have hp1 : π < 3.1417 := by
    veriphy_certified (exact lt_trans Real.pi_lt_d4 (by norm_num))
    -- certificate: ca-contour-bound.sage
  have hp2 : 3.1414 < π := by
    veriphy_certified (exact lt_trans (by norm_num) Real.pi_gt_d4)
    -- certificate: ca-contour-bound.sage
  have h2 : Real.exp 2 = Real.exp 1 ^ 2 := by rw [← Real.exp_nat_mul]; norm_num
  rw [h2]
  constructor <;> nlinarith [mul_pos (by linarith : (0:ℝ) < Real.exp 1) (by linarith : (0:ℝ) < Real.exp 1)]

theorem ml_exact_value : 5.17 < 2 * π * Real.sin 1 ∧ 2 * π * Real.sin 1 < 5.3 := by
  have hs1 : Real.sin 1 < 0.8434 := by
    veriphy_certified
      ( have h := Real.sin_bound (x := (1 : ℝ)) (by norm_num [abs_le])
        norm_num [abs_le] at h
        linarith [h.2])
    -- certificate: ca-contour-bound.sage
  have hs2 : 0.8233 < Real.sin 1 := by
    veriphy_certified
      ( have h := Real.sin_bound (x := (1 : ℝ)) (by norm_num [abs_le])
        norm_num [abs_le] at h
        linarith [h.1])
    -- certificate: ca-contour-bound.sage
  have hp1 : π < 3.1417 := by
    veriphy_certified (exact lt_trans Real.pi_lt_d4 (by norm_num))
    -- certificate: ca-contour-bound.sage
  have hp2 : 3.1414 < π := by
    veriphy_certified (exact lt_trans (by norm_num) Real.pi_gt_d4)
    -- certificate: ca-contour-bound.sage
  constructor <;> nlinarith

theorem circleIntegrable_inv_sub_mul (g : ℂ → ℂ) (hg : Continuous g) (w : ℂ) (R : ℝ) (hR : 0 ≤ R)
    (hw : ‖w‖ ≠ R) : CircleIntegrable (fun z => (z - w)⁻¹ * g z) 0 R := by
  apply ContinuousOn.circleIntegrable hR
  apply ContinuousOn.mul _ hg.continuousOn
  apply ContinuousOn.inv₀ (by fun_prop)
  intro z hz h
  rw [sub_eq_zero] at h
  subst h
  simp at hz
  exact hw hz

theorem ml_exact (R : ℝ) (hR : 1 < R) : (∮ z in C(0, R), exp z / (z ^ 2 + 1)) = 2 * π * I * sin 1 := by
  have hR0 : 0 < R := by linarith
  have hI : I ∈ ball (0 : ℂ) R := by simpa using hR
  have hmI : -I ∈ ball (0 : ℂ) R := by simpa using hR
  have h2I : (2 * I) ≠ 0 := by simp [I_ne_zero]
  have hpf : Set.EqOn (fun z => exp z / (z ^ 2 + 1))
      (fun z => (2 * I)⁻¹ * ((z - I)⁻¹ * exp z - (z - -I)⁻¹ * exp z)) (sphere 0 R) := by
    intro z hz
    have hz' : ‖z‖ = R := by simpa using hz
    have h1 : z - I ≠ 0 := fun h => by
      rw [sub_eq_zero] at h; rw [h] at hz'; simp at hz'; linarith
    have h2 : z - -I ≠ 0 := fun h => by
      rw [sub_eq_zero] at h; rw [h] at hz'; simp at hz'; linarith
    have h3 : z ^ 2 + 1 = (z - I) * (z - -I) := by linear_combination I_sq
    simp only
    rw [h3, ← sub_mul, inv_sub_inv h1 h2, show (z - -I) - (z - I) = 2 * I by ring]
    field_simp
  rw [circleIntegral.integral_congr hR0.le hpf, circleIntegral.integral_const_mul,
    circleIntegral.integral_sub (circleIntegrable_inv_sub_mul exp continuous_exp I R hR0.le (by simp; linarith))
      (circleIntegrable_inv_sub_mul exp continuous_exp (-I) R hR0.le (by simp; linarith))]
  have c1 := (differentiable_exp.differentiableOn (s := closedBall (0 : ℂ) R)).circleIntegral_sub_inv_smul hI
  have c2 := (differentiable_exp.differentiableOn (s := closedBall (0 : ℂ) R)).circleIntegral_sub_inv_smul hmI
  simp only [smul_eq_mul] at c1 c2
  rw [c1, c2, Complex.sin]
  simp only [neg_mul, one_mul]
  have k : (2 * I)⁻¹ * (2 * π * I * exp I - 2 * π * I * exp (-I)) = π * (exp I - exp (-I)) := by
    calc _ = ((2 * I)⁻¹ * (2 * I)) * (π * (exp I - exp (-I))) := by ring
      _ = _ := by rw [inv_mul_cancel₀ h2I, one_mul]
  rw [k]
  linear_combination ((π : ℂ) * (exp I - exp (-I))) * I_sq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.ml_exact
-- test: ca-contour-ml-exact-test.sage

theorem circle_conj (R : ℝ) (hR : 0 < R) : (∮ z in C(0, R), (starRingEnd ℂ) z) = 2 * π * I * R ^ 2 := by
  have h : Set.EqOn (fun z => (starRingEnd ℂ) z) (fun z => (R : ℂ) ^ 2 * (z - 0)⁻¹) (sphere 0 R) := by
    intro z hz
    have hz' : ‖z‖ = R := by simpa using hz
    have hz0 : z ≠ 0 := fun h => by rw [h] at hz'; simp at hz'; linarith
    simp only [sub_zero]
    rw [← hz']
    field_simp
    rw [Complex.conj_mul']
  rw [circleIntegral.integral_congr hR.le h, circleIntegral.integral_const_mul,
    circleIntegral.integral_sub_inv_of_mem_ball (mem_ball_self hR)]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.circle_conj
-- test: ca-contour-circle-conj-test.sage

theorem conj_mul_self (z : ℂ) (hz : ‖z‖ = 1) : (starRingEnd ℂ) z * z = 1 := by
  rw [Complex.conj_mul', hz]
  simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourIntegrals.conj_mul_self
-- test: ca-contour-conj-mul-self-test.sage

end KaluzaKlein.ComplexAnalysis.ContourIntegrals

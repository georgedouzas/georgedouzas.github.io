import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Contours"

The claims of the section "Contours" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): contours (paths `γ : [a, b] → ℂ`), their length and reversal, the circle, segments, the square, the semicircular contour, the parabola `t + i t²` and the ellipse.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.Contours

/-- The length `∫ₐᵇ |γ'(t)| dt` of a path `γ` on `[a, b]`. -/
def length (γ : ℝ → ℂ) (a b : ℝ) : ℝ := ∫ t in a..b, ‖deriv γ t‖

/-- The reversed path `t ↦ γ (a + b - t)`. -/
def reverse (γ : ℝ → ℂ) (a b : ℝ) : ℝ → ℂ := fun t => γ (a + b - t)

/-- The segment `t ↦ z₁ + t (z₂ - z₁)`, `t ∈ [0, 1]`. -/
def seg (z₁ z₂ : ℂ) : ℝ → ℂ := fun t => z₁ + (t : ℂ) * (z₂ - z₁)

theorem deriv_reverse (γ : ℝ → ℂ) (a b t : ℝ) : deriv (reverse γ a b) t = -deriv γ (a + b - t) := by
  have := congrFun (iteratedDeriv_comp_const_sub (n := 1) (f := γ) (s := a + b)) t
  unfold reverse
  simpa using this

theorem length_reverse (γ : ℝ → ℂ) (a b : ℝ) : length (reverse γ a b) a b = length γ a b := by
  unfold length
  have h : ∀ t, ‖deriv (reverse γ a b) t‖ = ‖deriv γ (a + b - t)‖ := fun t => by rw [deriv_reverse, norm_neg]
  rw [intervalIntegral.integral_congr (fun t _ => h t)]
  have := intervalIntegral.integral_comp_sub_left (fun t => ‖deriv γ t‖) (a + b) (a := a) (b := b)
  rw [this]
  congr 1 <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Contours.length_reverse
-- test: ca-contour-length-reverse-test.sage

theorem circle_separates (z₀ : ℂ) (r : ℝ) :
    (sphere z₀ r)ᶜ = ball z₀ r ∪ (closedBall z₀ r)ᶜ ∧ Bornology.IsBounded (ball z₀ r) ∧
      ¬ Bornology.IsBounded (closedBall z₀ r)ᶜ := by
  refine ⟨?_, isBounded_ball, fun h => NormedSpace.unbounded_univ ℝ ℂ ?_⟩
  · ext z
    simp only [Set.mem_compl_iff, mem_sphere, Set.mem_union, mem_ball, mem_closedBall, not_le]
    exact ⟨fun h => lt_or_gt_of_ne h, fun h => h.elim ne_of_lt ne_of_gt⟩
  · rw [← Set.union_compl_self (closedBall z₀ r)]
    exact isBounded_closedBall.union h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.circle_separates (0) (0) := by
  refine ⟨?_, Metric.isBounded_ball, fun h => NormedSpace.unbounded_univ ℝ ℂ ?_⟩
  · ext z
    simp only [Set.mem_compl_iff, Metric.mem_sphere, Set.mem_union, Metric.mem_ball, Metric.mem_closedBall, not_le]
    exact ⟨fun h => lt_or_gt_of_ne h, fun h => h.elim ne_of_lt ne_of_gt⟩
  · rw [← Set.union_compl_self (Metric.closedBall _ _)]
    exact Metric.isBounded_closedBall.union h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.circle_separates (0) (1) := by
  refine ⟨?_, Metric.isBounded_ball, fun h => NormedSpace.unbounded_univ ℝ ℂ ?_⟩
  · ext z
    simp only [Set.mem_compl_iff, Metric.mem_sphere, Set.mem_union, Metric.mem_ball, Metric.mem_closedBall, not_le]
    exact ⟨fun h => lt_or_gt_of_ne h, fun h => h.elim ne_of_lt ne_of_gt⟩
  · rw [← Set.union_compl_self (Metric.closedBall _ _)]
    exact Metric.isBounded_closedBall.union h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.circle_separates ((1 + Complex.I)) ((5 / 2)) := by
  refine ⟨?_, Metric.isBounded_ball, fun h => NormedSpace.unbounded_univ ℝ ℂ ?_⟩
  · ext z
    simp only [Set.mem_compl_iff, Metric.mem_sphere, Set.mem_union, Metric.mem_ball, Metric.mem_closedBall, not_le]
    exact ⟨fun h => lt_or_gt_of_ne h, fun h => h.elim ne_of_lt ne_of_gt⟩
  · rw [← Set.union_compl_self (Metric.closedBall _ _)]
    exact Metric.isBounded_closedBall.union h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Contours.circle_separates
-- test: ca-contour-circle-separates-test.sage

theorem hasDerivAt_ofReal' (t : ℝ) : HasDerivAt (fun s : ℝ => (s : ℂ)) 1 t := by
  simpa using (hasDerivAt_id t).ofReal_comp

theorem contDiff_ofReal' : ContDiff ℝ 1 (fun t : ℝ => (t : ℂ)) := by
  have h : (fun t : ℝ => (t : ℂ)) = ⇑ofRealCLM := by funext t; simp
  rw [h]; exact ofRealCLM.contDiff

theorem hasDerivAt_seg (z₁ z₂ : ℂ) (t : ℝ) : HasDerivAt (seg z₁ z₂) (z₂ - z₁) t := by
  have h := ((hasDerivAt_ofReal' t).mul_const (z₂ - z₁)).const_add z₁
  unfold seg
  convert h using 1
  simp

theorem deriv_seg (z₁ z₂ : ℂ) (t : ℝ) : deriv (seg z₁ z₂) t = z₂ - z₁ := (hasDerivAt_seg z₁ z₂ t).deriv

theorem contDiff_seg (z₁ z₂ : ℂ) : ContDiff ℝ 1 (seg z₁ z₂) :=
  contDiff_const.add (contDiff_ofReal'.mul contDiff_const)

theorem seg_length (z₁ z₂ : ℂ) : length (seg z₁ z₂) 0 1 = ‖z₂ - z₁‖ := by
  simp [length, deriv_seg]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.seg_length
-- test: ca-contour-seg-length-test.sage

theorem circle_deriv (z₀ : ℂ) (r t : ℝ) : deriv (circleMap z₀ r) t = I * r * exp (t * I) := by
  rw [deriv_circleMap, circleMap_zero]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.circle_deriv
-- test: ca-contour-circle-deriv-test.sage

theorem circle_norm_deriv (z₀ : ℂ) (r : ℝ) (hr : 0 ≤ r) (t : ℝ) : ‖deriv (circleMap z₀ r) t‖ = r := by
  rw [deriv_circleMap, norm_mul, norm_circleMap_zero, norm_I, mul_one, abs_of_nonneg hr]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.circle_norm_deriv
-- test: ca-contour-circle-norm-deriv-test.sage

theorem circle_length (z₀ : ℂ) (r : ℝ) (hr : 0 ≤ r) : length (circleMap z₀ r) 0 (2 * π) = 2 * π * r := by
  simp only [length, circle_norm_deriv z₀ r hr, intervalIntegral.integral_const, sub_zero, smul_eq_mul]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.circle_length
-- test: ca-contour-circle-length-test.sage

theorem circle_clockwise (z₀ : ℂ) (r t : ℝ) :
    reverse (circleMap z₀ r) 0 (2 * π) t = z₀ + r * exp (-(t * I)) := by
  simp only [reverse, circleMap, zero_add]
  rw [show ((2 * π - t : ℝ) : ℂ) * I = -(t * I) + 2 * π * I by push_cast; ring, exp_add, exp_two_pi_mul_I,
    mul_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.circle_clockwise
-- test: ca-contour-circle-clockwise-test.sage

theorem circle_left_normal (z₀ : ℂ) (r t : ℝ) : I * deriv (circleMap z₀ r) t = -(circleMap z₀ r t - z₀) := by
  rw [deriv_circleMap, circleMap_zero]
  simp only [circleMap]
  linear_combination ((r : ℂ) * exp (t * I)) * I_sq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.circle_left_normal
-- test: ca-contour-circle-left-normal-test.sage

theorem square_closes (h x : ℂ) (hx : x ^ 2 = -1) : h + x * h + x ^ 2 * h + x ^ 3 * h = 0 := by
  veriphy_certified (linear_combination (h * x + h) * hx)
  -- certificate: ca-contour-ideal.sage

theorem semicircle_closed (R : ℝ) : seg (-R) R 1 = circleMap 0 R 0 ∧ circleMap 0 R π = seg (-R) R 0 := by
  constructor
  · simp only [seg, circleMap, ofReal_one, one_mul, ofReal_zero, zero_mul, exp_zero, mul_one, zero_add]
    ring
  · simp only [seg, circleMap, exp_pi_mul_I, ofReal_zero, zero_mul, add_zero, zero_add]
    ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.semicircle_closed
-- test: ca-contour-semicircle-closed-test.sage

theorem semicircle_left (R t : ℝ) : (I * deriv (seg (-R) R) t).im = 2 * R := by
  rw [deriv_seg]
  simp only [mul_im, I_re, I_im, sub_re, sub_im, neg_re, neg_im, ofReal_re, ofReal_im]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.semicircle_left
-- test: ca-contour-semicircle-left-test.sage

theorem upper_half_plane_exhausted (z : ℂ) (hz : 0 < z.im) : ∃ R : ℝ, 0 < z.im ∧ ‖z‖ < R :=
  ⟨‖z‖ + 1, hz, lt_add_one _⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.upper_half_plane_exhausted (Complex.I) ((by norm_num)) := by
  exact ⟨_, by norm_num, lt_add_one _⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.upper_half_plane_exhausted ((2 * Complex.I)) ((by norm_num)) := by
  exact ⟨_, by norm_num, lt_add_one _⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.upper_half_plane_exhausted ((3 + 5 * Complex.I)) ((by norm_num)) := by
  exact ⟨_, by norm_num, lt_add_one _⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Contours.upper_half_plane_exhausted
-- test: ca-contour-upper-half-plane-exhausted-test.sage

theorem square_first_side (t : ℝ) : seg (1 + I) (-1 + I) t = (1 - 2 * t) + I := by
  simp only [seg]; ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.square_first_side (0) := by
  simp only [KaluzaKlein.ComplexAnalysis.Contours.seg]
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.square_first_side ((1 / 2)) := by
  simp only [KaluzaKlein.ComplexAnalysis.Contours.seg]
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.square_first_side (1) := by
  simp only [KaluzaKlein.ComplexAnalysis.Contours.seg]
  push_cast
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Contours.square_first_side ((-3)) := by
  simp only [KaluzaKlein.ComplexAnalysis.Contours.seg]
  push_cast
  ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.Contours.square_first_side
-- test: ca-contour-square-first-side-test.sage

theorem ellipse_identity (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) : (2 * c) ^ 2 + 4 * s ^ 2 = 4 := by
  veriphy_certified (linear_combination (4) * h)
  -- certificate: ca-contour-ideal.sage

theorem ellipse_ccw_identity (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) : 2 * c * c + s * (2 * s) = 2 := by
  veriphy_certified (linear_combination (2) * h)
  -- certificate: ca-contour-ideal.sage

theorem ellipse_on (t : ℝ) : (2 * Real.cos t) ^ 2 / 4 + Real.sin t ^ 2 = 1 := by
  have := ellipse_identity _ _ (Real.cos_sq_add_sin_sq t)
  linarith

theorem ellipse_ccw (t : ℝ) : 2 * Real.cos t * Real.cos t - Real.sin t * (-(2 * Real.sin t)) = 2 := by
  have := ellipse_ccw_identity _ _ (Real.cos_sq_add_sin_sq t)
  linarith

theorem seg_diag_length (a : ℝ) : length (seg 0 (a + a * I)) 0 1 = Real.sqrt 2 * |a| := by
  rw [seg_length, sub_zero, show (a : ℂ) + a * I = (a : ℂ) * (1 + I) by ring, norm_mul, Complex.norm_real,
    Real.norm_eq_abs, Complex.norm_eq_sqrt_sq_add_sq]
  norm_num
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.seg_diag_length
-- test: ca-contour-seg-diag-length-test.sage

/-- The parabola `t ↦ t + i t²`. -/
def parabola : ℝ → ℂ := fun s => (s : ℂ) + I * (s : ℂ) ^ 2

theorem parabola_hasDerivAt (t : ℝ) : HasDerivAt parabola (1 + 2 * I * t) t := by
  have h1 := hasDerivAt_ofReal' t
  have h3 := h1.add ((h1.mul h1).const_mul I)
  have e : parabola = fun s : ℝ => (s : ℂ) + I * ((s : ℂ) * s) := by funext s; simp only [parabola]; ring
  rw [e]
  convert h3 using 1
  ring

theorem parabola_speed (t : ℝ) : ‖deriv parabola t‖ = Real.sqrt (1 + 4 * t ^ 2) := by
  rw [(parabola_hasDerivAt t).deriv, Complex.norm_eq_sqrt_sq_add_sq]
  congr 1
  simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.parabola_speed
-- test: ca-contour-parabola-speed-test.sage

theorem sqrt_integral (b : ℝ) :
    ∫ t in (0 : ℝ)..b, Real.sqrt (1 + 4 * t ^ 2) = b * Real.sqrt (1 + 4 * b ^ 2) / 2 + Real.arsinh (2 * b) / 4 := by
  have hd : ∀ t, HasDerivAt (fun t => t * Real.sqrt (1 + 4 * t ^ 2) / 2 + Real.arsinh (2 * t) / 4)
      (Real.sqrt (1 + 4 * t ^ 2)) t := by
    intro t
    have hpos : 0 < 1 + 4 * t ^ 2 := by positivity
    set s := Real.sqrt (1 + 4 * t ^ 2) with hs_def
    have hs : 0 < s := Real.sqrt_pos.2 hpos
    have hs2 : s ^ 2 = 1 + 4 * t ^ 2 := Real.sq_sqrt hpos.le
    have h1 : HasDerivAt (fun t : ℝ => 1 + 4 * t ^ 2) (8 * t) t := by
      have := ((hasDerivAt_pow 2 t).const_mul 4).const_add 1
      convert this using 1
      push_cast; ring
    have h2 : HasDerivAt (fun t : ℝ => Real.sqrt (1 + 4 * t ^ 2)) (8 * t / (2 * s)) t := h1.sqrt hpos.ne'
    have h4 : HasDerivAt (fun t : ℝ => Real.arsinh (2 * t)) (2 / s) t := by
      have := ((hasDerivAt_id' t).const_mul (2 : ℝ)).arsinh
      convert this using 1
      rw [smul_eq_mul, show (1 : ℝ) + (2 * t) ^ 2 = 1 + 4 * t ^ 2 by ring, ← hs_def]
      field_simp
    have h := (((hasDerivAt_id' t).mul h2).div_const 2).add (h4.div_const 4)
    convert h using 1
    have e : t * (8 * t / (2 * s)) = (s ^ 2 - 1) / s := by rw [hs2]; field_simp; ring
    rw [e]
    field_simp
    ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hd t)
    ((by fun_prop : Continuous fun t : ℝ => Real.sqrt (1 + 4 * t ^ 2)).intervalIntegrable _ _)]
  simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.Contours.sqrt_integral
-- test: ca-contour-sqrt-integral-test.sage

theorem parabola_length : length parabola 0 1 = Real.sqrt 5 / 2 + Real.arsinh 2 / 4 := by
  unfold length
  simp_rw [parabola_speed]
  rw [sqrt_integral]
  norm_num

theorem length_comp (γ : ℝ → ℂ) (φ φ' : ℝ → ℝ) (c d : ℝ) (hγ : Differentiable ℝ γ)
    (hγ' : Continuous (deriv γ)) (hφ : ∀ s, HasDerivAt φ (φ' s) s) (hφ'c : Continuous φ')
    (hpos : ∀ s, 0 < φ' s) : length (γ ∘ φ) c d = length γ (φ c) (φ d) := by
  unfold length
  have hd : ∀ s, ‖deriv (γ ∘ φ) s‖ = ‖deriv γ (φ s)‖ * φ' s := by
    intro s
    rw [((hγ (φ s)).hasDerivAt.scomp s (hφ s)).deriv, norm_smul, Real.norm_of_nonneg (hpos s).le, mul_comm]
  simp_rw [hd]
  have := intervalIntegral.integral_comp_mul_deriv (a := c) (b := d) (f := φ) (f' := φ')
    (g := fun x => ‖deriv γ x‖) (fun x _ => hφ x) hφ'c.continuousOn hγ'.norm
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.length_comp
-- test: ca-contour-length-comp-test.sage

theorem semicircle_length (R : ℝ) (hR : 0 ≤ R) :
    length (seg (-R) R) 0 1 + length (circleMap 0 R) 0 π = 2 * R + π * R := by
  rw [seg_length]
  simp only [length, circle_norm_deriv 0 R hR, intervalIntegral.integral_const, sub_zero, smul_eq_mul]
  rw [show (R : ℂ) - -(R : ℂ) = ((2 * R : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
    Real.norm_of_nonneg (by linarith)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.Contours.semicircle_length
-- test: ca-contour-semicircle-length-test.sage

end KaluzaKlein.ComplexAnalysis.Contours

import Mathlib
import KaluzaKlein.PartI.ComplexAnalysis.Contours
import KaluzaKlein.PartI.ComplexAnalysis.ContourIntegrals
import KaluzaKlein.PartI.ComplexAnalysis.CauchyTheorem
import KaluzaKlein.PartI.ComplexAnalysis.CauchyIntegralFormula
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Worked computations"

The claims of the section "Worked computations" of `tutorial/part-i-04-complex-analysis.qmd` (part "Contour integration"): the circle integrals of `zⁿ`, `1/(z - 3)`, `e^z/z`, `e^z/z²`, the residues of `e^{az}/z` and `e^{az}/z²`, the segment integral of `z²`, Fourier orthogonality and coefficient extraction.
-/

open Complex MeasureTheory Metric Filter Topology
open scoped Real

noncomputable section

namespace KaluzaKlein.ComplexAnalysis.ContourWorked

open KaluzaKlein.ComplexAnalysis.Contours KaluzaKlein.ComplexAnalysis.ContourIntegrals
  KaluzaKlein.ComplexAnalysis.CauchyTheorem KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula

theorem prp_zpow (n : ℤ) (_hn : n ∈ Finset.Icc (-3 : ℤ) 3) :
    (∮ z in C(0, 1), z ^ n) = if n = -1 then 2 * π * I else 0 :=
  circle_zpow n 1 one_pos

attribute [veriphy_tested oracle_test 0 7] KaluzaKlein.ComplexAnalysis.ContourWorked.prp_zpow
-- test: ca-contour-prp-zpow-test.sage

theorem inv_sub_three_series (z : ℂ) (hz : ‖z‖ < 3) :
    HasSum (fun k : ℕ => -(z ^ k / 3 ^ (k + 1))) (1 / (z - 3)) := by
  have hq : ‖z / 3‖ < 1 := by rw [norm_div]; simp; linarith
  have h := (hasSum_geometric_of_norm_lt_one hq).mul_left (-(1 / 3 : ℂ))
  have h3 : z - 3 ≠ 0 := fun h0 => by
    rw [sub_eq_zero] at h0; rw [h0] at hz; norm_num at hz
  convert h using 1
  · funext k; rw [div_pow, pow_succ]; field_simp
  · have h3' : (3 : ℂ) - z ≠ 0 := fun h0 => h3 (by linear_combination -h0)
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourWorked.inv_sub_three_series
-- test: ca-contour-inv-sub-three-series-test.sage

theorem exp_div_residue (a : ℂ) : Tendsto (fun z => z * (exp (a * z) / z)) (𝓝[≠] 0) (𝓝 1) := by
  have hc : Continuous fun z : ℂ => exp (a * z) := by fun_prop
  have h : Tendsto (fun z => exp (a * z)) (𝓝[≠] 0) (𝓝 1) := by
    have := (hc.tendsto 0).mono_left (nhdsWithin_le_nhds (s := {0}ᶜ))
    simpa using this
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.exp_div_residue
-- test: ca-contour-exp-div-residue-test.sage

theorem exp_div_sq_residue (a : ℂ) :
    Tendsto (fun z => (z ^ 2 * (exp (a * z) / z ^ 2) - 1) / z) (𝓝[≠] 0) (𝓝 a) := by
  have hd : HasDerivAt (fun z => exp (a * z)) a 0 := by
    have := ((hasDerivAt_id' (0 : ℂ)).const_mul a).cexp
    simpa using this
  have h := hd.tendsto_slope
  refine (h.congr' ?_)
  filter_upwards [self_mem_nhdsWithin] with z hz
  have hz' : z ≠ 0 := hz
  simp only [slope, vsub_eq_sub, sub_zero, mul_zero, exp_zero, smul_eq_mul]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.exp_div_sq_residue
-- test: ca-contour-exp-div-sq-residue-test.sage

theorem param_exp_div (r : ℝ) (hr : 0 < r) : I * ∫ t in (0 : ℝ)..2 * π, exp (circleMap 0 r t) = 2 * π * I := by
  rw [← (cif_exp r hr).1, circleIntegral, ← intervalIntegral.integral_const_mul]
  congr 1; funext t
  have hne : circleMap 0 r t ≠ 0 := circleMap_ne_center hr.ne'
  rw [deriv_circleMap, smul_eq_mul]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.param_exp_div
-- test: ca-contour-param-exp-div-test.sage

theorem cint_sq_seg (w : ℂ) : cint (fun z => z ^ 2) (seg 0 w) 0 1 = w ^ 3 / 3 := by
  rw [cint_sq _ 0 1 (contDiff_seg 0 w)]
  simp [seg]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.cint_sq_seg
-- test: ca-contour-cint-sq-seg-test.sage

theorem integral_exp_int (k : ℤ) :
    ∫ t in (0 : ℝ)..2 * π, exp (k * t * I) = if k = 0 then (2 * π : ℂ) else 0 := by
  split_ifs with h
  · subst h; simp
  · have hc : (k : ℂ) * I ≠ 0 := mul_ne_zero (by exact_mod_cast h) I_ne_zero
    have e : (fun t : ℝ => exp (k * t * I)) = fun t : ℝ => exp ((k * I) * t) := by funext t; ring_nf
    rw [e, integral_exp_mul_complex hc]
    rw [show (k : ℂ) * I * ((2 * π : ℝ) : ℂ) = k * (2 * π * I) by push_cast; ring, exp_int_mul_two_pi_mul_I]
    simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourWorked.integral_exp_int
-- test: ca-contour-integral-exp-int-test.sage

theorem exp_circle_series (t : ℝ) :
    HasSum (fun k : ℕ => exp (k * t * I) / k.factorial) (exp (exp (t * I))) := by
  have := NormedSpace.expSeries_div_hasSum_exp (exp (t * I))
  rw [← Complex.exp_eq_exp_ℂ] at this
  convert this using 1
  funext k
  rw [← exp_nat_mul]; ring_nf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.exp_circle_series
-- test: ca-contour-exp-circle-series-test.sage

theorem seg_primitive_value : (1 + I) ^ 3 / 3 - 0 ^ 3 / 3 = (-2 + 2 * I) / 3 := by
  rw [cube_one_add I, I_sq]; ring

theorem seg_integrand (x t : ℂ) : ((1 + x) * t) ^ 2 * (1 + x) = (1 + x) ^ 3 * t ^ 2 := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourWorked.seg_integrand (Complex.I) (0) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourWorked.seg_integrand (Complex.I) ((1 / 2)) := by
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ContourWorked.seg_integrand ((3 + Complex.I)) ((-2)) := by
  ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ContourWorked.seg_integrand
-- test: ca-contour-seg-integrand-test.sage

theorem t_sq_integral : ∫ t in (0 : ℝ)..1, t ^ 2 = 1 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) ((fun t => t ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) (f' :=
          fun t => t ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ca-contour-integral.sage

theorem circleIntegrable_const_mul_zpow (a : ℂ) (k : ℤ) (R : ℝ) (hR : 0 < R) :
    CircleIntegrable (fun z : ℂ => a * z ^ k) 0 R := by
  apply ContinuousOn.circleIntegrable hR.le
  apply continuousOn_const.mul
  apply ContinuousOn.zpow₀ continuousOn_id
  intro z hz
  left
  rintro rfl
  simp at hz
  linarith

theorem circle_rational (R : ℝ) (hR : 0 < R) :
    (∮ z in C(0, R), (3 * z ^ 2 + 2 / z + 1 / z ^ 2 + 5 / z ^ 3)) = 4 * π * I := by
  have e : (fun z : ℂ => 3 * z ^ 2 + 2 / z + 1 / z ^ 2 + 5 / z ^ 3) =
      fun z => 3 * z ^ (2 : ℤ) + 2 * z ^ (-1 : ℤ) + 1 * z ^ (-2 : ℤ) + 5 * z ^ (-3 : ℤ) := by
    funext z; simp [zpow_neg, div_eq_mul_inv]
  rw [e]
  have a1 := circleIntegrable_const_mul_zpow 3 2 R hR
  have a2 := circleIntegrable_const_mul_zpow 2 (-1) R hR
  have a3 := circleIntegrable_const_mul_zpow 1 (-2) R hR
  have a4 := circleIntegrable_const_mul_zpow 5 (-3) R hR
  have a12 : CircleIntegrable (fun z : ℂ => 3 * z ^ (2 : ℤ) + 2 * z ^ (-1 : ℤ)) 0 R := a1.add a2
  have a123 : CircleIntegrable (fun z : ℂ => 3 * z ^ (2 : ℤ) + 2 * z ^ (-1 : ℤ) + 1 * z ^ (-2 : ℤ)) 0 R :=
    a12.add a3
  rw [circleIntegral.integral_add a123 a4, circleIntegral.integral_add a12 a3, circleIntegral.integral_add a1 a2]
  simp only [circleIntegral.integral_const_mul, circle_zpow _ R hR]
  norm_num
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.circle_rational
-- test: ca-contour-circle-rational-test.sage

theorem circle_zpow_radius (n : ℤ) (r : ℝ) (hr : 0 < r) :
    (∮ z in C(0, r), z ^ n) = I * (r : ℂ) ^ (n + 1) * ∫ t in (0 : ℝ)..2 * π, exp ((n + 1) * t * I) := by
  rw [circleIntegral, ← intervalIntegral.integral_const_mul]
  congr 1; funext t
  have hr0 : (r : ℂ) ≠ 0 := by exact_mod_cast hr.ne'
  simp only [deriv_circleMap, circleMap_zero, smul_eq_mul]
  rw [mul_zpow, ← exp_int_mul, zpow_add_one₀ hr0,
    show ((n : ℂ) + 1) * t * I = n * (t * I) + t * I by ring, exp_add]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourWorked.circle_zpow_radius
-- test: ca-contour-circle-zpow-radius-test.sage

theorem fourier_orth (m n : ℤ) :
    ∫ t in (0 : ℝ)..2 * π, exp (m * t * I) * exp (-(n * t * I)) = if m = n then (2 * π : ℂ) else 0 := by
  have e : (fun t : ℝ => exp (m * t * I) * exp (-(n * t * I))) = fun t : ℝ => exp (((m - n : ℤ) : ℂ) * t * I) := by
    funext t; rw [← exp_add]; push_cast; ring_nf
  rw [e, integral_exp_int]
  simp [sub_eq_zero]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.ContourWorked.fourier_orth
-- test: ca-contour-fourier-orth-test.sage

theorem fourier_via_circle (m n : ℤ) :
    ∫ t in (0 : ℝ)..2 * π, exp (((m - n : ℤ) : ℂ) * t * I) = -I * ∮ z in C(0, 1), z ^ (m - n - 1) := by
  rw [circle_param_zpow, ← mul_assoc, show -I * I = 1 by rw [neg_mul, I_mul_I, neg_neg], one_mul]
  congr 1; funext t; push_cast; ring_nf

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.ContourWorked.fourier_via_circle
-- test: ca-contour-fourier-via-circle-test.sage

theorem fourier_coeff (N : ℕ) (c : ℤ → ℂ) (n : ℤ) (hn : n ∈ Finset.Icc (-(N : ℤ)) N) :
    ∫ t in (0 : ℝ)..2 * π,
      (∑ k ∈ Finset.Icc (-(N : ℤ)) N, c k * exp (k * t * I)) * exp (-(n * t * I)) = 2 * π * c n := by
  have e : (fun t : ℝ => (∑ k ∈ Finset.Icc (-(N : ℤ)) N, c k * exp (k * t * I)) * exp (-(n * t * I))) =
      fun t : ℝ => ∑ k ∈ Finset.Icc (-(N : ℤ)) N, c k * (exp (k * t * I) * exp (-(n * t * I))) := by
    funext t; rw [Finset.sum_mul]; congr 1; funext k; ring
  rw [e, intervalIntegral.integral_finset_sum (fun k _ => by
    apply Continuous.intervalIntegrable; fun_prop)]
  simp only [intervalIntegral.integral_const_mul, fourier_orth, mul_ite, mul_zero]
  rw [Finset.sum_ite_eq' (Finset.Icc (-(N : ℤ)) N) n (fun k => c k * (2 * π : ℂ))]
  rw [if_pos hn]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ContourWorked.fourier_coeff
-- test: ca-contour-fourier-coeff-test.sage

end KaluzaKlein.ComplexAnalysis.ContourWorked

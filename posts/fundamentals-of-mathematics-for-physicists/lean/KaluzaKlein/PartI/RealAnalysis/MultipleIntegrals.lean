import Mathlib
import KaluzaKlein.PartI.RealAnalysis.Integral
import Veriphy.Kit.Levels

/-!
# Real analysis, part "Multiple integrals and convergence of functions"

The claims of the section "Multiple integrals and convergence of functions" of
`tutorial/part-i-03-real-analysis.qmd`: iterated integrals and Fubini's theorem (and its failure), volumes, the
change of variables formula with the polar computation of the Gaussian integral, uniform convergence, and
differentiation under the integral sign with the Gaussian moments and the Gamma function.

Iterated integrals with elementary antiderivatives are certified by the `definite_integral` recipe, one level at a
time; general theorems are tested at the chapter's functions.
-/

namespace KaluzaKlein.RealAnalysis.MultipleIntegrals

open Filter Topology Real Set MeasureTheory intervalIntegral
open scoped Nat Matrix

/-! ## def-multiple-integral -/

/-- Fubini's theorem on a rectangle: for continuous `f` both iterated integrals equal the double integral. -/
theorem fubini_rectangle (f : ℝ × ℝ → ℝ) (hf : Continuous f) (a b c d : ℝ) (hab : a ≤ b) (hcd : c ≤ d) :
    ∫ p in Icc a b ×ˢ Icc c d, f p = ∫ x in a..b, ∫ y in c..d, f (x, y) ∧
      ∫ p in Icc a b ×ˢ Icc c d, f p = ∫ y in c..d, ∫ x in a..b, f (x, y) := by
  have hint : Integrable f ((volume.restrict (Icc a b)).prod (volume.restrict (Icc c d))) := by
    rw [Measure.prod_restrict]
    exact hf.continuousOn.integrableOn_compact (isCompact_Icc.prod isCompact_Icc)
  have hset : ∫ p in Icc a b ×ˢ Icc c d, f p = ∫ p, f p ∂((volume.restrict (Icc a b)).prod (volume.restrict (Icc c d))) := by
    rw [Measure.prod_restrict, Measure.volume_eq_prod]
  constructor
  · rw [hset, integral_prod _ hint, integral_of_le hab, integral_Icc_eq_integral_Ioc]
    refine setIntegral_congr_fun measurableSet_Ioc (fun x _ => ?_)
    rw [integral_of_le hcd, integral_Icc_eq_integral_Ioc]
  · rw [hset, integral_prod_symm _ hint, integral_of_le hcd, integral_Icc_eq_integral_Ioc]
    refine setIntegral_congr_fun measurableSet_Ioc (fun y _ => ?_)
    rw [integral_of_le hab, integral_Icc_eq_integral_Ioc]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.fubini_rectangle
-- test: ra-fubini-rectangle-test.sage

/-! ## exm-double-rectangle -/

/-- `∫₀² (∫₀¹ x² y dx) dy = 2/3`: the inner integral is `y/3`. -/
theorem rectangle_x_first : ∫ y in (0 : ℝ)..2, (∫ x in (0 : ℝ)..1, x ^ 2 * y) = 2 / 3 := by
  have h : ∀ y : ℝ, ∫ x in (0 : ℝ)..1, x ^ 2 * y = y / 3 := by
    intro y
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => (y * (x_veriphy ^ 3) / 3 : ℝ)) ((fun x => x ^ 2 * y) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (y / 3 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (y * (x_veriphy ^ 3) / 3 : ℝ)) (f' :=
            fun x => x ^ 2 * y) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: ra-rect-inner-x.sage
  simp_rw [h]
  show ∫ y in (0 : ℝ)..2, y / 3 = 2 / 3
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 6 : ℝ)) ((fun y => y / 3) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 6 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        all_goals (beta_reduce; (simp; ring1))
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 6 : ℝ)) (f' :=
          fun y => y / 3) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      try beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-rect-outer-y.sage

/-- `∫₀¹ (∫₀² x² y dy) dx = 2/3`: the inner integral is `2x²`. -/
theorem rectangle_y_first : ∫ x in (0 : ℝ)..1, (∫ y in (0 : ℝ)..2, x ^ 2 * y) = 2 / 3 := by
  have h : ∀ x : ℝ, ∫ y in (0 : ℝ)..2, x ^ 2 * y = 2 * x ^ 2 := by
    intro x
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 : ℝ),
            HasDerivAt (fun x_veriphy => ((x ^ 2) * (x_veriphy ^ 2) / 2 : ℝ)) ((fun y => x ^ 2 * y) x_veriphy)
              x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul ((x ^ 2) / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x ^ 2) * (x_veriphy ^ 2) / 2 : ℝ))
            (f' := fun y => x ^ 2 * y) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals ring1)
    -- certificate: ra-rect-inner-y.sage
  simp_rw [h]
  show ∫ x in (0 : ℝ)..1, 2 * x ^ 2 = 2 / 3
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (2 * (x_veriphy ^ 3) / 3 : ℝ)) ((fun x => 2 * x ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (2 / 3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        all_goals (beta_reduce; (simp; ring1))
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (2 * (x_veriphy ^ 3) / 3 : ℝ)) (f' :=
          fun x => 2 * x ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      try beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-rect-outer-x.sage

/-! ## exm-double-triangle -/

/-- On the triangle `0 ≤ y ≤ x ≤ 1`: `∫₀¹ ∫₀ˣ e^(x+y) dy dx = e²/2 - e + 1/2`. -/
theorem triangle_y_first :
    ∫ x in (0 : ℝ)..1, (∫ y in (0 : ℝ)..x, Real.exp (x + y)) = Real.exp 1 ^ 2 / 2 - Real.exp 1 + 1 / 2 := by
  have h : ∀ x : ℝ, ∫ y in (0 : ℝ)..x, Real.exp (x + y) = Real.exp (2 * x) - Real.exp x := by
    intro x
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
            HasDerivAt (fun x_veriphy => (Real.exp (x + x_veriphy) : ℝ)) ((fun y => rexp (x + y)) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_const (x_veriphy : ℝ) (x : ℝ)).add
                        (hasDerivAt_id' (x_veriphy : ℝ))).exp)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          all_goals (beta_reduce; (simp; done))
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.exp (x + x_veriphy) : ℝ)) (f' :=
            fun y => rexp (x + y)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        try beta_reduce
        all_goals (norm_num; ring1))
    -- certificate: ra-tri-inner-y.sage
  simp_rw [h]
  show ∫ x in (0 : ℝ)..1, (Real.exp (2 * x) - Real.exp x) = Real.exp 1 ^ 2 / 2 - Real.exp 1 + 1 / 2
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.exp (2 * x_veriphy)) / 2 - (Real.exp x_veriphy) : ℝ))
            ((fun x => rexp (2 * x) - rexp x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).exp).const_mul (1 / 2 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).exp).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        all_goals (beta_reduce; (simp; ring1))
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((Real.exp (2 * x_veriphy)) / 2 - (Real.exp x_veriphy) : ℝ)) (f' := fun x => rexp (2 * x) - rexp x) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      try beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-tri-outer-x.sage

/-- The other order: `∫₀¹ ∫_y¹ e^(x+y) dx dy = e²/2 - e + 1/2`. -/
theorem triangle_x_first :
    ∫ y in (0 : ℝ)..1, (∫ x in y..1, Real.exp (x + y)) = Real.exp 1 ^ 2 / 2 - Real.exp 1 + 1 / 2 := by
  have h : ∀ y : ℝ, ∫ x in y..1, Real.exp (x + y) = Real.exp (1 + y) - Real.exp (2 * y) := by
    intro y
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (y : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => (Real.exp (y + x_veriphy) : ℝ)) ((fun x => rexp (x + y)) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_const (x_veriphy : ℝ) (y : ℝ)).add
                        (hasDerivAt_id' (x_veriphy : ℝ))).exp)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          all_goals
            (beta_reduce;
              first
              | rfl
              | ring1
              | (simp; ring1)
              | (simp; done)
              | (field_simp; ring1))
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.exp (y + x_veriphy) : ℝ)) (f' :=
            fun x => rexp (x + y)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        try beta_reduce
        all_goals (norm_num; ring_nf; done))
    -- certificate: ra-tri-inner-x.sage
  simp_rw [h]
  show ∫ y in (0 : ℝ)..1, (Real.exp (1 + y) - Real.exp (2 * y)) = Real.exp 1 ^ 2 / 2 - Real.exp 1 + 1 / 2
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.exp (2 * x_veriphy)) / 2 + (Real.exp (x_veriphy + 1)) : ℝ))
            ((fun y => rexp (1 + y) - rexp (2 * y)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).exp).const_mul (-(1 / 2) : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).add
                        (hasDerivAt_const (x_veriphy : ℝ) (1 : ℝ))).exp))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        all_goals (simp only [Pi.add_apply]; ring_nf; done)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-(Real.exp (2 * x_veriphy)) / 2 + (Real.exp (x_veriphy + 1)) : ℝ)) (f' := fun y =>
          rexp (1 + y) - rexp (2 * y)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      try beta_reduce
      all_goals (norm_num; ring1))
  -- certificate: ra-tri-outer-y.sage

/-- The triangle's area `∫₀¹ ∫₀ˣ dy dx = 1/2`. -/
theorem triangle_area : ∫ x in (0 : ℝ)..1, (∫ y in (0 : ℝ)..x, (1 : ℝ)) = 1 / 2 := by
  simp only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, mul_one]
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) ((fun x => x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) (f' :=
          fun x => x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ra-tri-area.sage

/-! ## exm-double-fubini-fails -/

/-- For `f = (x² - y²)/(x² + y²)²` on `(0, 1]²`: `∫₀¹ f dy = 1/(1 + x²)` for `x > 0`, so `∫₀¹ ∫₀¹ f dy dx = π/4`,
while `∫₀¹ ∫₀¹ f dx dy = -π/4`. -/
theorem fubini_fails :
    ∫ x in (0 : ℝ)..1, (∫ y in (0 : ℝ)..1, (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2) = π / 4 ∧
      ∫ y in (0 : ℝ)..1, (∫ x in (0 : ℝ)..1, (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2) = -(π / 4) := by
  -- the inner integral in `y`, for `x > 0`: the antiderivative is `y/(x² + y²)`
  have hin : ∀ x : ℝ, 0 < x → ∫ y in (0 : ℝ)..1, (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2 = 1 / (1 + x ^ 2) := by
    intro x hx
    veriphy_certified
      ( have hne : ∀ y ∈ Set.uIcc (0 : ℝ) (1 : ℝ), x ^ 2 + y ^ 2 ≠ 0 := fun y _ => by positivity
        have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => (x_veriphy / (x ^ 2 + (x_veriphy ^ 2)) : ℝ))
              ((fun y => (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2) x_veriphy) x_veriphy :=
          by
          intro x_veriphy hx_veriphy
          have hx0 : x ^ 2 + x_veriphy ^ 2 ≠ 0 := hne x_veriphy hx_veriphy
          beta_reduce
          refine
            ((((hasDerivAt_id' (x_veriphy : ℝ)).div
                      ((hasDerivAt_const (x_veriphy : ℝ) (x ^ 2 : ℝ)).add ((hasDerivAt_id' (x_veriphy : ℝ)).pow 2))
                      (by exact hne x_veriphy hx_veriphy))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          all_goals
            (beta_reduce;
              first
              | rfl
              | ring1
              | (simp; ring1)
              | (simp; done)
              | (field_simp; ring1))
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (x_veriphy / (x ^ 2 + (x_veriphy ^ 2)) : ℝ)) (f' := fun y => (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2) hF
            (ContinuousOn.intervalIntegrable
              (ContinuousOn.div (by fun_prop) (by fun_prop) (fun y hy => pow_ne_zero 2 (hne y hy))))]
        try beta_reduce
        all_goals ring1)
    -- certificate: ra-fubini-inner.sage
  have hout : ∫ x in (0 : ℝ)..1, 1 / (1 + x ^ 2) = π / 4 := by
    rw [integral_one_div_one_add_sq, Real.arctan_one, Real.arctan_zero, sub_zero]
  constructor
  · rw [← hout]
    apply intervalIntegral.integral_congr_ae
    exact Eventually.of_forall (fun x hx => hin x (by rw [uIoc_of_le zero_le_one] at hx; exact hx.1))
  · have hsym : ∀ y : ℝ, 0 < y →
        ∫ x in (0 : ℝ)..1, (x ^ 2 - y ^ 2) / (x ^ 2 + y ^ 2) ^ 2 = -(1 / (1 + y ^ 2)) := by
      intro y hy
      rw [← hin y hy, ← intervalIntegral.integral_neg]
      congr 1
      funext x
      ring
    rw [← hout, ← intervalIntegral.integral_neg]
    apply intervalIntegral.integral_congr_ae
    exact Eventually.of_forall (fun y hy => hsym y (by rw [uIoc_of_le zero_le_one] at hy; exact hy.1))

/-! ## exr-volume-tetrahedron -/

/-- The tetrahedron `x, y, z ≥ 0`, `x + y + z ≤ 1` has volume `∫₀¹ ∫₀^(1-x) (1 - x - y) dy dx = 1/6`. -/
theorem tetrahedron :
    (∀ x : ℝ, ∫ y in (0 : ℝ)..(1 - x), (1 - x - y) = (1 - x) ^ 2 / 2) ∧ ∫ x in (0 : ℝ)..1, (1 - x) ^ 2 / 2 = 1 / 6 := by
  constructor
  · intro x
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 - x : ℝ),
            HasDerivAt (fun x_veriphy => (-x * x_veriphy - ((x_veriphy ^ 2) / 2) + x_veriphy : ℝ))
              ((fun y => 1 - x - y) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-x : ℝ)).add
                          (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-(1 / 2) : ℝ))).add
                      (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (-x * x_veriphy - ((x_veriphy ^ 2) / 2) + x_veriphy : ℝ)) (f' := fun y => 1 - x - y) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals ring1)
    -- certificate: ra-tetra-inner.sage
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 6 - ((x_veriphy ^ 2) / 2) + (x_veriphy / 2) : ℝ))
              ((fun x => (1 - x) ^ 2 / 2) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 6 : ℝ)).add
                          (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-(1 / 2) : ℝ))).add
                      ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            ((x_veriphy ^ 3) / 6 - ((x_veriphy ^ 2) / 2) + (x_veriphy / 2) : ℝ)) (f' := fun x => (1 - x) ^ 2 / 2) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: ra-tetra-outer.sage

/-- The iteration behind the simplex volume `1/n!`: `∫₀¹ (1 - x)^k/k! dx = 1/(k+1)!`. -/
theorem simplex_step (k : ℕ) : ∫ x in (0 : ℝ)..1, (1 - x) ^ k / (k ! : ℝ) = 1 / ((k + 1)! : ℝ) := by
  have h := intervalIntegral.integral_comp_sub_left (fun u : ℝ => u ^ k / (k ! : ℝ)) (1 : ℝ) (a := 0) (b := 1)
  simp only [sub_zero, sub_self] at h
  rw [h, intervalIntegral.integral_div, integral_pow]
  rw [Nat.factorial_succ]
  push_cast
  have : (k ! : ℝ) ≠ 0 := by positivity
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.simplex_step
-- test: ra-simplex-step-test.sage

/-! ## exr-swap-order -/

/-- Swapping the order on `0 ≤ x ≤ y ≤ 1`: `∫₀¹ ∫₀ʸ e^(y²) dx dy = ∫₀¹ y e^(y²) dy = (e - 1)/2`. -/
theorem swap_order_value : ∫ y in (0 : ℝ)..1, y * Real.exp (y ^ 2) = (Real.exp 1 - 1) / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.exp (x_veriphy ^ 2)) / 2 : ℝ)) ((fun y => y * rexp (y ^ 2)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).exp).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        all_goals (beta_reduce; (simp; ring1))
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((Real.exp (x_veriphy ^ 2)) / 2 : ℝ))
          (f' := fun y => y * rexp (y ^ 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      try beta_reduce
      all_goals (norm_num; ring1))
  -- certificate: ra-swap-order.sage

/-! ## thm-change-of-variables -/

/-- Change of variables to polar coordinates: `∫_{ℝ²} f = ∫ r f(r cos φ, r sin φ) dr dφ` over `(0, ∞) × (-π, π)`. -/
theorem change_of_variables_polar (f : ℝ × ℝ → ℝ) :
    ∫ p in polarCoord.target, p.1 * f (polarCoord.symm p) = ∫ p, f p := by
  have := integral_comp_polarCoord_symm f
  simpa [smul_eq_mul] using this

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.change_of_variables_polar
-- test: ra-change-of-variables-polar-test.sage

/-! ## exm-change-disc -/

/-- `∫₀^(2π) ∫₀ᴿ r dr dφ = πR²`, and without the Jacobian `∫₀^(2π) ∫₀ᴿ dr dφ = 2πR` (a length). -/
theorem disc_area (R : ℝ) :
    ∫ ψ in (0 : ℝ)..(2 * π), (∫ r in (0 : ℝ)..R, r) = π * R ^ 2 ∧
      ∫ ψ in (0 : ℝ)..(2 * π), (∫ r in (0 : ℝ)..R, (1 : ℝ)) = 2 * π * R := by
  have h1 : ∫ r in (0 : ℝ)..R, r = R ^ 2 / 2 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (R : ℝ),
            HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) ((fun r => r) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) (f' :=
            fun r => r) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: ra-disc-inner.sage
  constructor
  · rw [h1]; simp; ring
  · simp

/-! ## exm-change-gaussian -/

/-- `∫₀^∞ e^(-a r²) r dr = 1/(2a)` for `a > 0`; at `a = 1` it is `1/2`. -/
theorem gaussian_radial (a : ℝ) (ha : 0 < a) : ∫ r in Ioi (0 : ℝ), Real.exp (-(a * r ^ 2)) * r = 1 / (2 * a) := by
  have h := integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := 1) (b := a) (by norm_num) (by norm_num) ha
  have e : ∫ r in Ioi (0 : ℝ), Real.exp (-(a * r ^ 2)) * r = ∫ r in Ioi (0 : ℝ), r ^ (1 : ℝ) * Real.exp (-a * r ^ (2 : ℝ)) := by
    refine setIntegral_congr_fun measurableSet_Ioi (fun r _ => ?_)
    simp only [Real.rpow_one]
    rw [show (r ^ (2 : ℝ)) = r ^ 2 by norm_cast]
    ring_nf
  rw [e, h]
  norm_num [Real.Gamma_one, Real.rpow_neg_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_radial
-- test: ra-gaussian-radial-test.sage

/-- The Gaussian integral squared: `(∫_ℝ e^(-bx²) dx)² = π/b`; at `b = 1` it is `π`, so the integral is `√π`. -/
theorem gaussian_sq (b : ℝ) (hb : 0 < b) :
    (∫ x : ℝ, Real.exp (-b * x ^ 2)) ^ 2 = π / b ∧ ∫ x : ℝ, Real.exp (-b * x ^ 2) = √(π / b) := by
  have h := integral_gaussian b
  refine ⟨by rw [h, Real.sq_sqrt (by positivity)], h⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_sq
-- test: ra-gaussian-sq-test.sage

/-! ## exm-change-ellipse -/

/-- The unit ball of `ℝ³` has volume `4π/3` (mathlib's volume of a Euclidean ball: `π^(3/2)/Γ(5/2)`), and the
spherical integral `∫₀^(2π) ∫₀^π ∫₀¹ r² sin θ dr dθ dφ = 4π/3`. -/
theorem ball_volume :
    ∫ ψ in (0 : ℝ)..(2 * π), (∫ θ in (0 : ℝ)..π, (∫ r in (0 : ℝ)..1, r ^ 2 * Real.sin θ)) = 4 * π / 3 := by
  have h1 : ∀ θ : ℝ, ∫ r in (0 : ℝ)..1, r ^ 2 * Real.sin θ = Real.sin θ / 3 := by
    intro θ
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) * (Real.sin θ) / 3 : ℝ)) ((fun r => r ^ 2 * sin θ) x_veriphy)
              x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul ((Real.sin θ) / 3 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            ((x_veriphy ^ 3) * (Real.sin θ) / 3 : ℝ)) (f' := fun r => r ^ 2 * sin θ) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: ra-ball-r.sage
  have h2 : ∫ θ in (0 : ℝ)..π, Real.sin θ / 3 = 2 / 3 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
            HasDerivAt (fun x_veriphy => (-(Real.cos x_veriphy) / 3 : ℝ)) ((fun θ => sin θ / 3) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(1 / 3) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; ring1
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos x_veriphy) / 3 : ℝ)) (f' :=
            fun θ => sin θ / 3) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
        have he1 : Real.cos (((π) : ℝ)) = -1 := by
          rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
        simp only [he0, he1]
        all_goals (norm_num; done))
    -- certificate: ra-ball-theta.sage
  simp_rw [h1, h2]
  simp
  ring

/-- The ellipse `x²/a² + y²/b² ≤ 1` has area `πab`: `∫₀^(2π) ∫₀¹ abr dr dt = πab`. -/
theorem ellipse_area (a b : ℝ) : ∫ t in (0 : ℝ)..(2 * π), (∫ r in (0 : ℝ)..1, a * b * r) = π * a * b := by
  have h : ∫ r in (0 : ℝ)..1, a * b * r = a * b / 2 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
            HasDerivAt (fun x_veriphy => (a * b * (x_veriphy ^ 2) / 2 : ℝ)) ((fun r => a * b * r) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (a * b / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (a * b * (x_veriphy ^ 2) / 2 : ℝ))
            (f' := fun r => a * b * r) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: ra-ellipse-r.sage
  rw [h]
  simp
  ring

/-! ## exr-gaussian-matrix -/

/-- The multivariate Gaussian integral (stated, tested at `2 × 2` matrices, not proved here): for symmetric
positive definite `A`, `∫_{ℝⁿ} e^(-xᵀAx/2) dⁿx = (2π)^(n/2)/√det A`. -/
def MultivariateGaussian : Prop :=
  ∀ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ), A.PosDef →
    ∫ x : Fin n → ℝ, Real.exp (-(x ⬝ᵥ (A *ᵥ x)) / 2) = (2 * π) ^ ((n : ℝ) / 2) / √A.det

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.MultivariateGaussian
-- test: ra-multivariategaussian-test.sage

/-! ## def-uniform-convergence -/

/-- A uniform limit of continuous functions is continuous. -/
theorem uniform_limit_continuous (f : ℕ → ℝ → ℝ) (g : ℝ → ℝ) (hf : ∀ n, Continuous (f n))
    (h : TendstoUniformly f g atTop) : Continuous g :=
  h.continuous (Frequently.of_forall hf)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.uniform_limit_continuous
-- test: ra-uniform-limit-continuous-test.sage

/-- Uniform convergence on `[a, b]` allows the exchange `lim ∫ f_n = ∫ lim f_n`. -/
theorem uniform_limit_integral (f : ℕ → ℝ → ℝ) (g : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : ∀ n, Continuous (f n))
    (h : TendstoUniformlyOn f g atTop (Icc a b)) :
    Tendsto (fun n => ∫ x in a..b, f n x) atTop (𝓝 (∫ x in a..b, g x)) := by
  have hg : ContinuousOn g (Icc a b) := h.continuousOn (Frequently.of_forall (fun n => (hf n).continuousOn))
  have hgi : IntervalIntegrable g volume a b := (hg.mono (by rw [uIcc_of_le hab])).intervalIntegrable
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hδ : 0 < ε / (b - a + 1) := div_pos hε (by linarith)
  obtain ⟨N, hN⟩ := eventually_atTop.1 (Metric.tendstoUniformlyOn_iff.1 h _ hδ)
  refine ⟨N, fun n hn => ?_⟩
  rw [Real.dist_eq, ← intervalIntegral.integral_sub ((hf n).intervalIntegrable a b) hgi]
  have hb : ∀ x ∈ uIoc a b, ‖f n x - g x‖ ≤ ε / (b - a + 1) := by
    intro x hx
    rw [uIoc_of_le hab] at hx
    have := hN n hn x ⟨hx.1.le, hx.2⟩
    rw [Real.dist_eq, abs_sub_comm] at this
    rw [Real.norm_eq_abs]; exact this.le
  have := intervalIntegral.norm_integral_le_of_norm_le_const hb
  rw [Real.norm_eq_abs, abs_of_nonneg (by linarith : 0 ≤ b - a)] at this
  calc |∫ x in a..b, (f n x - g x)| ≤ ε / (b - a + 1) * (b - a) := by linarith
    _ < ε := by
      rw [div_mul_eq_mul_div, div_lt_iff₀ (by linarith)]
      nlinarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.uniform_limit_integral
-- test: ra-uniform-limit-integral-test.sage

/-- The Weierstrass M-test: if `|f_k(x)| ≤ M_k` with `∑ M_k < ∞`, the partial sums converge uniformly. -/
theorem m_test (f : ℕ → ℝ → ℝ) (M : ℕ → ℝ) (hM : Summable M) (h : ∀ k x, |f k x| ≤ M k) :
    TendstoUniformly (fun n x => ∑ k ∈ Finset.range n, f k x) (fun x => ∑' k, f k x) atTop := by
  have := tendstoUniformly_tsum_nat hM (fun n x => by rw [Real.norm_eq_abs]; exact h n x)
  exact this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.m_test
-- test: ra-m-test-test.sage

/-! ## exm-uniform-fails -/

/-- `xⁿ → 0` for `0 ≤ x < 1` and `1ⁿ = 1`, while `sup_{[0,1)} xⁿ = 1` for every `n ≥ 1`: the convergence is
pointwise, not uniform. -/
theorem pow_not_uniform (n : ℕ) (hn : 1 ≤ n) (x : ℝ) (hx0 : 0 ≤ x) (hx1 : x < 1) :
    Tendsto (fun m : ℕ => x ^ m) atTop (𝓝 0) ∧ IsLUB ((fun y : ℝ => y ^ n) '' Ico 0 1) 1 := by
  refine ⟨tendsto_pow_atTop_nhds_zero_of_lt_one hx0 hx1, ?_, ?_⟩
  · rintro _ ⟨y, ⟨hy0, hy1⟩, rfl⟩
    exact pow_le_one₀ hy0 hy1.le
  · intro M hM
    by_contra hlt
    push_neg at hlt
    have ht : Tendsto (fun y : ℝ => y ^ n) (𝓝[<] 1) (𝓝 1) := by
      simpa using ((continuous_pow n).tendsto (1 : ℝ)).mono_left (nhdsWithin_le_nhds (s := Iio 1))
    have hev := ht.eventually (lt_mem_nhds hlt)
    obtain ⟨y, hyM, hy⟩ := (hev.and (Ioo_mem_nhdsLT (show (0 : ℝ) < 1 by norm_num))).exists
    exact absurd (hM ⟨y, ⟨hy.1.le, hy.2⟩, rfl⟩) (not_le.2 hyM)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.pow_not_uniform
-- test: ra-pow-not-uniform-test.sage

/-! ## exm-uniform-sine -/

/-- `|sin(nx)/n| ≤ 1/n` for all `x`, so `sin(nx)/n → 0` uniformly, and `∫₀^π sin(nx)/n dx = (1 - cos nπ)/n²`. -/
theorem sine_uniform (n : ℕ) (hn : 1 ≤ n) (x : ℝ) :
    |Real.sin (n * x) / n| ≤ 1 / n ∧ ∫ x in (0 : ℝ)..π, Real.sin (n * x) / n = (1 - Real.cos (n * π)) / (n : ℝ) ^ 2 := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  constructor
  · rw [abs_div, abs_of_pos hn0]
    exact div_le_div_of_nonneg_right (Real.abs_sin_le_one _) hn0.le
  · have hc : (n : ℝ) ≠ 0 := hn0.ne'
    rw [intervalIntegral.integral_div, intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hc, integral_sin]
    simp only [mul_zero, Real.cos_zero, smul_eq_mul]
    field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.sine_uniform
-- test: ra-sine-uniform-test.sage

/-- The derivatives `cos(nx)` do not converge at `x = (2k+1)π`, where they are `(-1)ⁿ`. -/
theorem cos_derivatives_diverge (k : ℕ) : ¬ ∃ L, Tendsto (fun n : ℕ => Real.cos (n * ((2 * k + 1) * π))) atTop (𝓝 L) := by
  have hval : ∀ n : ℕ, Real.cos (n * ((2 * k + 1) * π)) = (-1) ^ n := by
    intro n
    have h := Real.cos_nat_mul_pi (n * (2 * k + 1))
    push_cast at h
    rw [show (n : ℝ) * ((2 * k + 1) * π) = (n * (2 * k + 1)) * π by ring, h, pow_mul]
    rcases neg_one_pow_eq_or ℝ n with h' | h' <;> rw [h'] <;> simp [pow_succ, pow_mul]
  rintro ⟨L, hL⟩
  simp_rw [hval] at hL
  have hm1 : Tendsto (fun n : ℕ => 2 * n) atTop atTop :=
    tendsto_atTop_mono (fun n => by show n ≤ 2 * n; omega) tendsto_id
  have hm2 : Tendsto (fun n : ℕ => 2 * n + 1) atTop atTop :=
    tendsto_atTop_mono (fun n => by show n ≤ 2 * n + 1; omega) tendsto_id
  have e1 : ∀ n : ℕ, ((-1 : ℝ) ^ (2 * n)) = 1 := fun n => by rw [pow_mul]; norm_num
  have e2 : ∀ n : ℕ, ((-1 : ℝ) ^ (2 * n + 1)) = -1 := fun n => by rw [pow_succ, e1]; norm_num
  have h1 := hL.comp hm1
  have h2 := hL.comp hm2
  simp only [Function.comp_def, e1] at h1
  simp only [Function.comp_def, e2] at h2
  have := tendsto_nhds_unique tendsto_const_nhds h1
  have := tendsto_nhds_unique tendsto_const_nhds h2
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.cos_derivatives_diverge
-- test: ra-cos-derivatives-diverge-test.sage

/-! ## exr-uniform-convergence-sage -/

/-- `sup |sin(nx)/n| = 1/n` (attained at `x = π/(2n)`), and `h_n = x/(1 + nx²)` has `|h_n| ≤ 1/(2√n)` with equality
at `x = 1/√n`. -/
theorem sup_norms (n : ℕ) (hn : 1 ≤ n) (x : ℝ) :
    Real.sin (n * (π / (2 * n))) / n = 1 / n ∧ |x / (1 + n * x ^ 2)| ≤ 1 / (2 * √n) ∧
      (1 / √n) / (1 + n * (1 / √n) ^ 2) = 1 / (2 * √n) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hs : 0 < √(n : ℝ) := Real.sqrt_pos.2 hn0
  have hs2 : √(n : ℝ) ^ 2 = n := Real.sq_sqrt hn0.le
  refine ⟨?_, ?_, ?_⟩
  · rw [show (n : ℝ) * (π / (2 * n)) = π / 2 by field_simp <;> ring, Real.sin_pi_div_two]
  · rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < 1 + n * x ^ 2), div_le_div_iff₀ (by positivity) (by positivity)]
    have h1 : 0 ≤ (1 - √(n : ℝ) * |x|) ^ 2 := sq_nonneg _
    have h2 : (√(n : ℝ) * |x|) ^ 2 = n * x ^ 2 := by rw [mul_pow, hs2, sq_abs]
    nlinarith
  · rw [div_pow, one_pow, hs2]
    field_simp <;> ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.sup_norms
-- test: ra-sup-norms-test.sage

/-! ## exr-uniform-bump -/

/-- The moving bump `f_n = n` on `(0, 1/n)`: `f_n(x) → 0` for every `x`, but `∫₀¹ f_n = 1` for all `n ≥ 1` and
`sup f_n = n`. -/
theorem moving_bump (n : ℕ) (hn : 1 ≤ n) (x : ℝ) :
    Tendsto (fun m : ℕ => if 0 < x ∧ x < 1 / ((m : ℝ) + 1) then ((m : ℝ) + 1) else 0) atTop (𝓝 0) ∧
      ∫ y in (0 : ℝ)..1, (if 0 < y ∧ y < 1 / (n : ℝ) then (n : ℝ) else 0) = 1 := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  constructor
  · apply tendsto_const_nhds.congr'
    rcases le_or_gt x 0 with hx | hx
    · exact Eventually.of_forall (fun m => by beta_reduce; rw [if_neg (fun h => by linarith [h.1])])
    · obtain ⟨M, hM⟩ := exists_nat_gt (1 / x)
      filter_upwards [eventually_ge_atTop M] with m hm
      beta_reduce
      rw [if_neg]
      rintro ⟨-, h2⟩
      have hm' : (M : ℝ) ≤ m := by exact_mod_cast hm
      have : 1 / ((m : ℝ) + 1) < x := by
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ hx] at hM
        nlinarith
      linarith
  · have h1n : 1 / (n : ℝ) ≤ 1 := by rw [div_le_one hn0]; exact_mod_cast hn
    have hpos : 0 < 1 / (n : ℝ) := by positivity
    set g : ℝ → ℝ := fun y => if 0 < y ∧ y < 1 / (n : ℝ) then (n : ℝ) else 0 with hg
    have hne : ∀ᵐ y ∂volume, y ∉ ({1 / (n : ℝ)} : Set ℝ) := measure_eq_zero_iff_ae_notMem.1 (measure_singleton _)
    have hgi : IntervalIntegrable g volume 0 1 := by
      refine (intervalIntegrable_const (c := (n : ℝ))).mono_fun ?_ ?_
      · exact (Measurable.ite (measurableSet_Ioi.inter measurableSet_Iio) measurable_const
          measurable_const).aestronglyMeasurable
      · filter_upwards with y
        simp only [hg, Real.norm_eq_abs]
        split_ifs <;> simp [abs_of_pos hn0]
    have hgi1 : IntervalIntegrable g volume 0 (1 / n) := hgi.mono_set (by
      rw [uIcc_of_le hpos.le, uIcc_of_le zero_le_one]
      exact Icc_subset_Icc le_rfl h1n)
    have hgi2 : IntervalIntegrable g volume (1 / n) 1 := hgi.mono_set (by
      rw [uIcc_of_le h1n, uIcc_of_le zero_le_one]
      exact Icc_subset_Icc hpos.le le_rfl)
    rw [← integral_add_adjacent_intervals hgi1 hgi2]
    have hA : ∫ y in (0 : ℝ)..(1 / n), g y = ∫ y in (0 : ℝ)..(1 / n), (n : ℝ) := by
      apply intervalIntegral.integral_congr_ae
      filter_upwards [hne] with y hy hyI
      rw [uIoc_of_le hpos.le] at hyI
      simp only [hg]
      rw [if_pos ⟨hyI.1, lt_of_le_of_ne hyI.2 (fun h => hy (by simp [h]))⟩]
    have hB : ∫ y in (1 / (n : ℝ))..1, g y = ∫ y in (1 / (n : ℝ))..1, (0 : ℝ) := by
      apply intervalIntegral.integral_congr_ae
      filter_upwards with y hyI
      rw [uIoc_of_le h1n] at hyI
      simp only [hg]
      rw [if_neg (fun h => by linarith [h.2, hyI.1])]
    rw [hA, hB, intervalIntegral.integral_const, intervalIntegral.integral_zero, smul_eq_mul]
    field_simp
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.moving_bump
-- test: ra-moving-bump-test.sage

/-! ## thm-differentiation-under-integral -/

/-- The Leibniz rule on a finite interval: if `f(x, a)` and `∂ₐf` are continuous, `F(a) = ∫_c^d f(x, a) dx` has
`F'(a) = ∫_c^d ∂ₐ f(x, a) dx`. -/
theorem leibniz_rule (f f' : ℝ → ℝ → ℝ) (c d a : ℝ) (hf : Continuous (fun p : ℝ × ℝ => f p.1 p.2))
    (hf' : Continuous (fun p : ℝ × ℝ => f' p.1 p.2)) (hd : ∀ b x, HasDerivAt (fun b => f b x) (f' b x) b) :
    HasDerivAt (fun b => ∫ x in c..d, f b x) (∫ x in c..d, f' a x) a := by
  -- bound ∂ₐf on the compact set [a - 1, a + 1] × [c, d]
  have hK : IsCompact (Icc (a - 1) (a + 1) ×ˢ uIcc c d) := isCompact_Icc.prod isCompact_uIcc
  obtain ⟨C, hC⟩ := hK.exists_bound_of_continuousOn hf'.continuousOn
  have hcont : ∀ b, Continuous (f b) := fun b => hf.comp (Continuous.prodMk continuous_const continuous_id)
  have hcont' : ∀ b, Continuous (f' b) := fun b => hf'.comp (Continuous.prodMk continuous_const continuous_id)
  have := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := volume) (a := c) (b := d) (F := f) (F' := f')
    (x₀ := a) (s := Ioo (a - 1) (a + 1)) (bound := fun _ => C) (Ioo_mem_nhds (by linarith) (by linarith))
    (Eventually.of_forall (fun b => (hcont b).aestronglyMeasurable)) ((hcont a).intervalIntegrable _ _)
    (hcont' a).aestronglyMeasurable
    (Eventually.of_forall (fun t ht b hb => hC (b, t) ⟨Ioo_subset_Icc_self hb, uIoc_subset_uIcc ht⟩))
    intervalIntegrable_const (Eventually.of_forall (fun t _ b _ => hd b t))
  exact this.2

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.leibniz_rule
-- test: ra-leibniz-rule-test.sage

/-! ## exm-leibniz-gaussian -/

/-- `-d/da √(π/a) = ½ √π a^(-3/2) = ∫_ℝ x² e^(-ax²) dx`. -/
theorem gaussian_second_moment (a : ℝ) (ha : 0 < a) :
    deriv (fun a => √(π / a)) a = -(1 / 2) * √π / (a * √a) ∧
      ∫ x : ℝ, x ^ 2 * Real.exp (-a * x ^ 2) = 1 / 2 * √π / (a * √a) := by
  have hsa : 0 < √a := Real.sqrt_pos.2 ha
  have hsa2 : √a ^ 2 = a := Real.sq_sqrt ha.le
  constructor
  · have h1 : HasDerivAt (fun a => π / a) (-(π / a ^ 2)) a := by
      have := (hasDerivAt_inv ha.ne').const_mul π
      convert this using 1 <;> [funext b; skip] <;> field_simp
    have h2 := (h1.sqrt (by positivity)).deriv
    rw [h2, Real.sqrt_div' π ha.le]
    have hpi : 0 < √π := Real.sqrt_pos.2 Real.pi_pos
    have hπ : √π ^ 2 = π := Real.sq_sqrt Real.pi_pos.le
    generalize √π = s at hpi hπ ⊢
    generalize √a = t at hsa hsa2 ⊢
    rw [← hπ, ← hsa2]
    field_simp
  · have := Integral.gaussian_moment 1 a ha
    simp only [mul_one, Nat.cast_one] at this
    rw [show (2 * 1 - 1 : ℕ) = 1 by rfl] at this
    rw [this, Real.sqrt_div' π ha.le]
    simp [Nat.doubleFactorial]
    field_simp <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_second_moment
-- test: ra-gaussian-second-moment-test.sage

/-! ## exm-leibniz-factorial -/

/-- `(-d/da)ⁿ a⁻¹ = n! a^(-n-1)`. -/
theorem inv_iterated_deriv (n : ℕ) (a : ℝ) (ha : a ≠ 0) :
    (-1 : ℝ) ^ n * iteratedDeriv n (fun a : ℝ => a⁻¹) a = n ! * a⁻¹ ^ (n + 1) := by
  rw [iteratedDeriv_eq_iterate]
  have h := iter_deriv_inv (𝕜 := ℝ) n a
  have hf : (fun a : ℝ => a⁻¹) = Inv.inv := rfl
  rw [hf, h, ← mul_assoc, ← mul_assoc, ← pow_add, ← two_mul, pow_mul, neg_one_sq, one_pow, one_mul]
  congr 1
  rw [show (-1 - n : ℤ) = -((n + 1 : ℕ) : ℤ) by push_cast; ring, zpow_neg, zpow_natCast, inv_pow]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.inv_iterated_deriv
-- test: ra-inv-iterated-deriv-test.sage

/-! ## exm-leibniz-feynman -/

/-- `F'(a) = ∫₀¹ xᵃ dx = 1/(a + 1)` for `a > -1`. -/
theorem feynman_derivative (a : ℝ) (ha : -1 < a) : ∫ x in (0 : ℝ)..1, x ^ a = 1 / (a + 1) := by
  rw [integral_rpow (Or.inl ha)]
  simp [Real.zero_rpow (by linarith : a + 1 ≠ 0)]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.MultipleIntegrals.feynman_derivative
-- test: ra-feynman-derivative-test.sage

/-- Feynman's trick (stated, tested numerically, not proved here): `∫₀¹ (xᵃ - 1)/log x dx = log(1 + a)` for
`a > -1`. -/
def FeynmanTrick : Prop :=
  ∀ a : ℝ, -1 < a → ∫ x in (0 : ℝ)..1, (x ^ a - 1) / Real.log x = Real.log (1 + a)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.FeynmanTrick
-- test: ra-feynmantrick-test.sage

/-! ## exr-leibniz-dirichlet -/

/-- `∫₀^∞ e^(-ax) sin x dx = 1/(1 + a²)` for `a > 0`. -/
theorem laplace_sin (a : ℝ) (ha : 0 < a) : ∫ x in Ioi (0 : ℝ), Real.exp (-(a * x)) * Real.sin x = 1 / (1 + a ^ 2) := by
  set F : ℝ → ℝ := fun x => -(Real.exp (-(a * x)) * (a * Real.sin x + Real.cos x)) / (1 + a ^ 2) with hF
  have h1 : 0 < 1 + a ^ 2 := by positivity
  have hderiv : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt F (Real.exp (-(a * x)) * Real.sin x) x := by
    intro x _
    have h := ((((hasDerivAt_id x).const_mul a).neg.exp).mul
      (((Real.hasDerivAt_sin x).const_mul a).add (Real.hasDerivAt_cos x))).neg.div_const (1 + a ^ 2)
    refine h.congr_deriv ?_
    simp only [id_eq, mul_one, Pi.add_apply, Pi.neg_apply]
    field_simp
    ring
  have hcont : ContinuousWithinAt F (Ici 0) 0 := (by fun_prop : Continuous F).continuousWithinAt
  have hint : IntegrableOn (fun x => Real.exp (-(a * x)) * Real.sin x) (Ioi 0) := by
    have hexp : IntegrableOn (fun x : ℝ => Real.exp (-(a * x))) (Ioi 0) := by
      have := exp_neg_integrableOn_Ioi 0 ha
      simpa using this
    refine hexp.mono' ((by fun_prop : Continuous fun x : ℝ => Real.exp (-(a * x)) * Real.sin x).aestronglyMeasurable) ?_
    filter_upwards with x
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
    exact mul_le_of_le_one_right (Real.exp_pos _).le (Real.abs_sin_le_one x)
  have hlim : Tendsto F atTop (𝓝 0) := by
    have he : Tendsto (fun x : ℝ => Real.exp (-(a * x))) atTop (𝓝 0) := by
      apply Real.tendsto_exp_atBot.comp
      exact tendsto_neg_atTop_atBot.comp (tendsto_id.const_mul_atTop ha)
    have hb : ∀ x, |a * Real.sin x + Real.cos x| ≤ a + 1 := fun x => by
      have := abs_add_le (a * Real.sin x) (Real.cos x)
      rw [abs_mul, abs_of_pos ha] at this
      nlinarith [Real.abs_sin_le_one x, Real.abs_cos_le_one x]
    have hz : Tendsto (fun x => Real.exp (-(a * x)) * (a * Real.sin x + Real.cos x)) atTop (𝓝 0) := by
      apply squeeze_zero_norm' (a := fun x => Real.exp (-(a * x)) * (a + 1))
      · filter_upwards with x
        rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
        exact mul_le_mul_of_nonneg_left (hb x) (Real.exp_pos _).le
      · simpa using he.mul_const (a + 1)
    simpa [hF] using (hz.neg).div_const (1 + a ^ 2)
  rw [integral_Ioi_of_hasDerivAt_of_tendsto hcont hderiv hint hlim]
  simp only [hF, mul_zero, neg_zero, Real.exp_zero, Real.sin_zero, Real.cos_zero, one_mul]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.laplace_sin
-- test: ra-laplace-sin-test.sage

/-- The Dirichlet integral (stated, tested by Sage's exact value, not proved here): `F(a) = ∫₀^∞ e^(-ax) sin x/x dx =
π/2 - arctan a` for `a > 0`, and the improper integral `∫₀^∞ sin x/x dx = π/2`. -/
def DirichletIntegral : Prop :=
  ∀ a : ℝ, 0 < a → ∫ x in Ioi (0 : ℝ), Real.exp (-(a * x)) * (Real.sin x / x) = π / 2 - Real.arctan a

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.DirichletIntegral
-- test: ra-dirichletintegral-test.sage

/-- The improper Dirichlet integral `∫₀^∞ sin x/x dx = π/2`, in the `ε`-`R₀` form of the limit of `∫₀ᴿ sin x/x dx`
(stated, tested by Sage's sine integral, not proved here). -/
def DirichletLimit : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R₀ : ℝ, ∀ R : ℝ, R₀ ≤ R → |(∫ x in (0 : ℝ)..R, Real.sin x / x) - π / 2| < ε

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.DirichletLimit
-- test: ra-dirichletlimit-test.sage

/-! ## exr-leibniz-domination -/

/-- The domination `|∂ₐ e^(-ax²)| = x² e^(-ax²) ≤ x² e^(-a₀x²)` for `a ≥ a₀ > 0`, with `x² e^(-a₀x²)` integrable; and
`F(a) = ½√(π/a) → ∞` as `a → 0⁺`. -/
theorem gaussian_domination (a a₀ x : ℝ) (ha : a₀ ≤ a) :
    |x ^ 2 * Real.exp (-a * x ^ 2)| ≤ x ^ 2 * Real.exp (-a₀ * x ^ 2) := by
  rw [abs_of_nonneg (by positivity)]
  apply mul_le_mul_of_nonneg_left _ (sq_nonneg x)
  apply Real.exp_le_exp.2
  nlinarith [sq_nonneg x]

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_domination (1) (1) (0) ((by norm_num)) := by
  rw [abs_of_nonneg (by positivity)]
  all_goals (apply mul_le_mul_of_nonneg_left _ (sq_nonneg _); all_goals (apply Real.exp_le_exp.2; norm_num))

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_domination (2) (1) (1) ((by norm_num)) := by
  rw [abs_of_nonneg (by positivity)]
  all_goals (apply mul_le_mul_of_nonneg_left _ (sq_nonneg _); all_goals (apply Real.exp_le_exp.2; norm_num))

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_domination (3) (1 / 2) (-2) ((by norm_num)) := by
  rw [abs_of_nonneg (by positivity)]
  all_goals (apply mul_le_mul_of_nonneg_left _ (sq_nonneg _); all_goals (apply Real.exp_le_exp.2; norm_num))

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_domination
-- test: ra-gaussian-domination-test.sage

/-! ## exr-multiple-values-reduce -/

/-- `∫_ℝ e^(-(x-μ)²/(2σ²)) dx = √(2π) σ` for `σ > 0`. -/
theorem gaussian_shifted (μ σ : ℝ) (hσ : 0 < σ) : ∫ x : ℝ, Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) = √(2 * π) * σ := by
  have h := integral_sub_right_eq_self (μ := volume) (fun x : ℝ => Real.exp (-(1 / (2 * σ ^ 2)) * x ^ 2)) μ
  have e : ∀ x : ℝ, Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) = Real.exp (-(1 / (2 * σ ^ 2)) * (x - μ) ^ 2) := by
    intro x; congr 1; ring
  simp_rw [e]
  rw [h, integral_gaussian, show π / (1 / (2 * σ ^ 2)) = (2 * π) * σ ^ 2 by field_simp,
    Real.sqrt_mul (by positivity), Real.sqrt_sq hσ.le]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_shifted
-- test: ra-gaussian-shifted-test.sage

/-- `∫₀^∞ e^(-bx²) dx = √(π/b)/2`; at `b = 1`, `√π/2`. -/
theorem half_gaussian (b : ℝ) (hb : 0 < b) : ∫ x in Ioi (0 : ℝ), Real.exp (-b * x ^ 2) = √(π / b) / 2 :=
  integral_gaussian_Ioi b

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.half_gaussian
-- test: ra-half-gaussian-test.sage

/-- `∫₀^∞ √x e^(-x) dx = Γ(3/2) = √π/2`. -/
theorem sqrt_exp_integral (a : ℝ) (ha : 0 < a) :
    ∫ x in Ioi (0 : ℝ), √x * Real.exp (-(a * x)) = (1 / a) ^ (3 / 2 : ℝ) * Real.Gamma (3 / 2) := by
  rw [← Real.integral_rpow_mul_exp_neg_mul_Ioi (by norm_num) ha]
  refine setIntegral_congr_fun measurableSet_Ioi (fun x _ => ?_)
  rw [Real.sqrt_eq_rpow]
  norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.sqrt_exp_integral
-- test: ra-sqrt-exp-integral-test.sage

/-! ## exr-multiple-values-cone -/

/-- The cone `x² + y² ≤ z²`, `0 ≤ z ≤ h` (radius `R` at height `h`) has volume `πR²h/3`: slicing gives
`∫₀ʰ π (Rz/h)² dz`. -/
theorem cone_volume (R h : ℝ) (hh : h ≠ 0) : ∫ z in (0 : ℝ)..h, π * (R * z / h) ^ 2 = π * R ^ 2 * h / 3 := by
  have e : ∀ z : ℝ, π * (R * z / h) ^ 2 = (π * R ^ 2 / h ^ 2) * z ^ 2 := fun z => by field_simp
  simp_rw [e]
  rw [intervalIntegral.integral_const_mul, integral_pow]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.MultipleIntegrals.cone_volume
-- test: ra-cone-volume-test.sage

/-! ## exr-multiple-values-gamma -/

/-- `Γ(s) = ∫₀^∞ x^(s-1) e^(-x) dx` converges for `s > 0`, `Γ(s + 1) = s Γ(s)`, `Γ(n + 1) = n!`. -/
theorem gamma_props (s : ℝ) (hs : 0 < s) (n : ℕ) :
    IntegrableOn (fun x => Real.exp (-x) * x ^ (s - 1)) (Ioi 0) ∧ Real.Gamma (s + 1) = s * Real.Gamma s ∧
      Real.Gamma (n + 1) = n ! :=
  ⟨Real.GammaIntegral_convergent hs, Real.Gamma_add_one hs.ne', Real.Gamma_nat_eq_factorial n⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gamma_props (1) ((by norm_num)) (0) := by
  exact ⟨Real.GammaIntegral_convergent (by norm_num), Real.Gamma_add_one (by norm_num), Real.Gamma_nat_eq_factorial _⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gamma_props (1 / 2) ((by norm_num)) (3) := by
  exact ⟨Real.GammaIntegral_convergent (by norm_num), Real.Gamma_add_one (by norm_num), Real.Gamma_nat_eq_factorial _⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.MultipleIntegrals.gamma_props (5) ((by norm_num)) (4) := by
  exact ⟨Real.GammaIntegral_convergent (by norm_num), Real.Gamma_add_one (by norm_num), Real.Gamma_nat_eq_factorial _⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.MultipleIntegrals.gamma_props
-- test: ra-gamma-props-test.sage

end KaluzaKlein.RealAnalysis.MultipleIntegrals

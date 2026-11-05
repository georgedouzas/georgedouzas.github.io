import Mathlib
import Veriphy.Kit.Levels

/-!
# Real analysis, part "Calculus on ℝⁿ"

The claims of the section "Calculus on ℝⁿ" of `tutorial/part-i-03-real-analysis.qmd`: partial derivatives and the
gradient, smooth maps, the Jacobian matrix with the chain rule, diffeomorphisms and the inverse function theorem,
and the Jacobian determinant of the polar map.

A partial derivative is written as a one-variable `deriv` with the other variables fixed, and certified by the
`derivative` recipe; Jacobian matrices and determinants are certified by `matrix_from_goal` (with the relation
`cos² + sin² = 1` where needed).
-/

namespace KaluzaKlein.RealAnalysis.Calculus

open Filter Topology Real Set Matrix

/-! ## exm-partial-polynomial -/

/-- `∂ₓ(x² y + sin y) = 2xy`. -/
theorem partial_x_f (x y : ℝ) : deriv (fun x => x ^ 2 * y + Real.sin y) x = 2 * x * y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x ^ 2 * y + sin y) ((2 * x * y : ℝ)) (x) :=
        by
        convert
          ((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (y : ℝ)).add (hasDerivAt_const (x : ℝ) (Real.sin y : ℝ))) using 1
        all_goals
          first
          | rfl
          | ring1
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (funext y; simp; ring_nf; done)
          | (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-partial-x-f.sage

/-- `∂_y(x² y + sin y) = x² + cos y`. -/
theorem partial_y_f (x y : ℝ) : deriv (fun y => x ^ 2 * y + Real.sin y) y = x ^ 2 + Real.cos y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => x ^ 2 * y + sin y) ((x ^ 2 + (Real.cos y) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).const_mul (x ^ 2 : ℝ)).add ((hasDerivAt_id' (y : ℝ)).sin)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-partial-y-f.sage

/-- The mixed partials agree: `∂_y(2xy) = 2x = ∂ₓ(x² + cos y)`. -/
theorem mixed_partials_f (x y : ℝ) :
    deriv (fun y => 2 * x * y) y = 2 * x ∧ deriv (fun x => x ^ 2 + Real.cos y) x = 2 * x := by
  constructor
  · veriphy_certified
      ( have hd : HasDerivAt (fun y => 2 * x * y) ((2 * x : ℝ)) (y) :=
          by
          convert ((hasDerivAt_id' (y : ℝ)).const_mul (2 * x : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-mixed-y.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun x => x ^ 2 + cos y) ((2 * x : ℝ)) (x) :=
          by
          convert (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (Real.cos y : ℝ))) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-mixed-x.sage

/-- Schwarz's theorem in partial-derivative form (stated, tested at the chapter's functions, not proved here): for
a `C²` function of two variables the mixed partial derivatives agree. -/
def SchwarzSymmetry : Prop :=
  ∀ (f : ℝ × ℝ → ℝ), ContDiff ℝ 2 f → ∀ x y : ℝ,
    deriv (fun y => deriv (fun x => f (x, y)) x) y = deriv (fun x => deriv (fun y => f (x, y)) y) x

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Calculus.SchwarzSymmetry
-- test: ra-schwarzsymmetry-test.sage

/-! ## exm-partial-polar -/

/-- The partial derivatives of the polar map: `∂_r x = cos φ`, `∂_φ x = -r sin φ`, `∂_r y = sin φ`,
`∂_φ y = r cos φ`. -/
theorem polar_partials (r φ : ℝ) :
    deriv (fun r => r * Real.cos φ) r = Real.cos φ ∧ deriv (fun φ => r * Real.cos φ) φ = -r * Real.sin φ ∧
      deriv (fun r => r * Real.sin φ) r = Real.sin φ ∧ deriv (fun φ => r * Real.sin φ) φ = r * Real.cos φ := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · veriphy_certified
      ( have hd : HasDerivAt (fun r => r * cos φ) ((Real.cos φ : ℝ)) (r) :=
          by
          convert ((hasDerivAt_id' (r : ℝ)).const_mul (Real.cos φ : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            | (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-polar-dr-x.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun φ => r * cos φ) ((-r * (Real.sin φ) : ℝ)) (φ) :=
          by
          convert (((hasDerivAt_id' (φ : ℝ)).cos).const_mul (r : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-polar-dphi-x.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun r => r * sin φ) ((Real.sin φ : ℝ)) (r) :=
          by
          convert ((hasDerivAt_id' (r : ℝ)).const_mul (Real.sin φ : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            | (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-polar-dr-y.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun φ => r * sin φ) ((r * (Real.cos φ) : ℝ)) (φ) :=
          by
          convert (((hasDerivAt_id' (φ : ℝ)).sin).const_mul (r : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-polar-dphi-y.sage

/-! ## exm-partial-metric -/

/-- `∂_θ sin² θ = 2 sin θ cos θ`, the only non-zero derivative of the sphere metric (`g_θθ = 1` is constant). -/
theorem sphere_metric_derivative (θ : ℝ) :
    deriv (fun θ => Real.sin θ ^ 2) θ = 2 * Real.sin θ * Real.cos θ ∧ deriv (fun _ : ℝ => (1 : ℝ)) θ = 0 := by
  constructor
  · veriphy_certified
      ( have hd : HasDerivAt (fun θ => sin θ ^ 2) ((2 * (Real.cos θ) * (Real.sin θ) : ℝ)) (θ) :=
          by
          convert (((hasDerivAt_id' (θ : ℝ)).sin).pow 2) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by ring1))
    -- certificate: ra-sphere-metric.sage
  · simp

/-! ## exr-partials-chain -/

/-- The directional derivative of `x² y + sin y` at `(1, 0)` along `(3, 4)/5` is `8/5`. -/
theorem directional_f : deriv (fun t => (1 + 3 / 5 * t) ^ 2 * (4 / 5 * t) + Real.sin (4 / 5 * t)) 0 = 8 / 5 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => (1 + 3 / 5 * t) ^ 2 * (4 / 5 * t) + sin (4 / 5 * t)) ((8 / 5 : ℝ)) (0) :=
        by
        convert
          (((((((hasDerivAt_id' (0 : ℝ)).const_mul (3 : ℝ)).add (hasDerivAt_const (0 : ℝ) (5 : ℝ))).pow 2).mul
                    (hasDerivAt_id' (0 : ℝ))).const_mul
                (4 / 125 : ℝ)).add
            (((hasDerivAt_id' (0 : ℝ)).const_mul (4 / 5 : ℝ)).sin)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.pow_apply]; (norm_num; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-directional-f.sage

/-- The third-order expansion of `x² y + sin y` at the origin: `y + x² y - y³/6`, with error at most `|y|⁴ · 5/96`
for `|y| ≤ 1`. -/
theorem third_order_f (x y : ℝ) (hy : |y| ≤ 1) :
    |x ^ 2 * y + Real.sin y - (y + x ^ 2 * y - y ^ 3 / 6)| ≤ |y| ^ 4 * (5 / 96) := by
  have h := Real.sin_bound hy
  rw [show x ^ 2 * y + Real.sin y - (y + x ^ 2 * y - y ^ 3 / 6) = Real.sin y - (y - y ^ 3 / 6) by ring]
  refine h.trans ?_
  have h0 := abs_nonneg y
  have h4 : 0 ≤ |y| ^ 4 := by positivity
  rw [show |y| ^ 5 = |y| ^ 4 * |y| by ring]
  nlinarith [mul_le_mul_of_nonneg_left hy h4]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Calculus.third_order_f
-- test: ra-third-order-f-test.sage

/-! ## exr-gradient-level-sets -/

/-- For `f = x² + 4y²`: `∇f = (2x, 8y)`. -/
theorem gradient_ellipse (x y : ℝ) :
    deriv (fun x => x ^ 2 + 4 * y ^ 2) x = 2 * x ∧ deriv (fun y => x ^ 2 + 4 * y ^ 2) y = 8 * y := by
  constructor
  · veriphy_certified
      ( have hd : HasDerivAt (fun x => x ^ 2 + 4 * y ^ 2) ((2 * x : ℝ)) (x) :=
          by
          convert (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (4 * (y ^ 2) : ℝ))) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-grad-ellipse-x.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun y => x ^ 2 + 4 * y ^ 2) ((8 * y : ℝ)) (y) :=
          by
          convert ((((hasDerivAt_id' (y : ℝ)).pow 2).const_mul (4 : ℝ)).add (hasDerivAt_const (y : ℝ) (x ^ 2 : ℝ))) using
            1
          all_goals
            first
            | rfl
            | ring1
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
            | (funext y; simp; ring_nf; done)
            | (simp only [Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-grad-ellipse-y.sage

/-- The level curve `f = 8` is `c(t) = (2√2 cos t, √2 sin t)`, and `∇f(c(t)) · ċ(t) = 0`: the gradient is
orthogonal to the level set. -/
theorem gradient_orthogonal (t : ℝ) :
    (2 * √2 * Real.cos t) ^ 2 + 4 * (√2 * Real.sin t) ^ 2 = 8 ∧
      2 * (2 * √2 * Real.cos t) * (-(2 * √2) * Real.sin t) + 8 * (√2 * Real.sin t) * (√2 * Real.cos t) = 0 := by
  have h2 : √2 ^ 2 = (2 : ℝ) := Real.sq_sqrt (by norm_num)
  constructor
  · nlinarith [Real.sin_sq_add_cos_sq t]
  · ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Calculus.gradient_orthogonal
-- test: ra-gradient-orthogonal-test.sage

/-- At `(2, 1)`: `|∇f| = √80 = 4√5` (the steepest rate), and the directional derivative along `(3, 4)/5` is
`44/5`. -/
theorem gradient_at_point :
    √((2 * 2 : ℝ) ^ 2 + (8 * 1) ^ 2) = 4 * √5 ∧
      deriv (fun t : ℝ => (2 + 3 / 5 * t) ^ 2 + 4 * (1 + 4 / 5 * t) ^ 2) 0 = 44 / 5 := by
  constructor
  · rw [show ((2 * 2 : ℝ) ^ 2 + (8 * 1) ^ 2) = 4 ^ 2 * 5 by norm_num, Real.sqrt_mul (by norm_num),
      Real.sqrt_sq (by norm_num)]
  · veriphy_certified
      ( have hd : HasDerivAt (fun t => (2 + 3 / 5 * t) ^ 2 + 4 * (1 + 4 / 5 * t) ^ 2) ((44 / 5 : ℝ)) (0) :=
          by
          convert
            ((((((hasDerivAt_id' (0 : ℝ)).const_mul (4 : ℝ)).add (hasDerivAt_const (0 : ℝ) (5 : ℝ))).pow 2).const_mul
                  (4 / 25 : ℝ)).add
              (((((hasDerivAt_id' (0 : ℝ)).const_mul (3 : ℝ)).add (hasDerivAt_const (0 : ℝ) (10 : ℝ))).pow 2).const_mul
                (1 / 25 : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply, Pi.pow_apply, Pi.mul_apply]; ring_nf; done)
            | (funext y; simp; ring_nf; done)
            | (simp only [Pi.add_apply, Pi.pow_apply]; ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-directional-ellipse.sage

/-- Cauchy-Schwarz: the directional derivative `∇f · u` along a unit vector is at most `|∇f|`, with equality for
`u = ∇f/|∇f|`. -/
theorem steepest_direction (g₁ g₂ u₁ u₂ : ℝ) (hu : u₁ ^ 2 + u₂ ^ 2 = 1) :
    g₁ * u₁ + g₂ * u₂ ≤ √(g₁ ^ 2 + g₂ ^ 2) := by
  have h : (g₁ * u₁ + g₂ * u₂) ^ 2 ≤ g₁ ^ 2 + g₂ ^ 2 := by
    nlinarith [sq_nonneg (g₁ * u₂ - g₂ * u₁)]
  calc g₁ * u₁ + g₂ * u₂ ≤ |g₁ * u₁ + g₂ * u₂| := le_abs_self _
    _ = √((g₁ * u₁ + g₂ * u₂) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ √(g₁ ^ 2 + g₂ ^ 2) := Real.sqrt_le_sqrt h

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.steepest_direction (4) (8) (3 / 5) (4 / 5) ((by norm_num)) := by
  have h : (44 / 5 : ℝ) ≤ Real.sqrt 80 := by rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
  norm_num
  calc (44 / 5 : ℝ) ≤ Real.sqrt 80 := h
    _ = _ := by norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.steepest_direction (0) (0) (1) (0) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.steepest_direction (3) (4) (3 / 5) (4 / 5) ((by norm_num)) := by
  norm_num
  try rw [show (25 : ℝ) = 5 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.steepest_direction
-- test: ra-steepest-direction-test.sage

/-! ## exr-partials-not-continuous -/

/-- `f(x, y) = xy/(x² + y²)`, `f(0, 0) = 0`. -/
noncomputable def xyRatio (p : ℝ × ℝ) : ℝ := if p = 0 then 0 else p.1 * p.2 / (p.1 ^ 2 + p.2 ^ 2)

/-- Both partial derivatives of `xy/(x² + y²)` exist at the origin (they vanish: `f` is `0` on the axes), but
`f(x, x) = 1/2` for `x ≠ 0`, so `f` is not continuous at the origin. -/
theorem xyRatio_props (x : ℝ) (hx : x ≠ 0) :
    xyRatio (x, 0) = 0 ∧ xyRatio (0, x) = 0 ∧ xyRatio (x, x) = 1 / 2 ∧
      HasDerivAt (fun t => xyRatio (t, 0)) 0 0 ∧ HasDerivAt (fun t => xyRatio (0, t)) 0 0 ∧
      ¬ ContinuousAt xyRatio 0 := by
  have hax : ∀ t : ℝ, xyRatio (t, 0) = 0 := fun t => by unfold xyRatio; split_ifs <;> simp
  have hay : ∀ t : ℝ, xyRatio (0, t) = 0 := fun t => by unfold xyRatio; split_ifs <;> simp
  have hdiag : ∀ t : ℝ, t ≠ 0 → xyRatio (t, t) = 1 / 2 := fun t ht => by
    unfold xyRatio
    rw [if_neg (by simp [Prod.ext_iff, ht])]
    field_simp
    ring
  refine ⟨hax x, hay x, hdiag x hx, ?_, ?_, ?_⟩
  · have : (fun t => xyRatio (t, 0)) = fun _ => (0 : ℝ) := funext hax
    rw [this]; exact hasDerivAt_const _ _
  · have : (fun t => xyRatio (0, t)) = fun _ => (0 : ℝ) := funext hay
    rw [this]; exact hasDerivAt_const _ _
  · intro hc
    have hpath : Tendsto (fun t : ℝ => ((t, t) : ℝ × ℝ)) (𝓝[≠] 0) (𝓝 ((0 : ℝ), (0 : ℝ))) := by
      have h0 : Continuous (fun t : ℝ => ((t, t) : ℝ × ℝ)) := by fun_prop
      exact (h0.tendsto 0).mono_left nhdsWithin_le_nhds
    have h1 : Tendsto (fun t : ℝ => xyRatio (t, t)) (𝓝[≠] 0) (𝓝 (xyRatio 0)) := hc.tendsto.comp hpath
    have h2 : Tendsto (fun t : ℝ => xyRatio (t, t)) (𝓝[≠] 0) (𝓝 (1 / 2)) := by
      apply tendsto_const_nhds.congr'
      filter_upwards [self_mem_nhdsWithin] with t ht
      exact (hdiag t ht).symm
    have := tendsto_nhds_unique h1 h2
    simp [xyRatio] at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.xyRatio_props
-- test: ra-xyratio-props-test.sage

/-! ## def-smooth-map -/

/-- Sums, products and compositions of smooth maps are smooth. -/
theorem smooth_rules (f g : ℝ → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (hg : ContDiff ℝ (⊤ : ℕ∞) g) :
    ContDiff ℝ (⊤ : ℕ∞) (fun x => f x + g x) ∧ ContDiff ℝ (⊤ : ℕ∞) (fun x => f x * g x) ∧ ContDiff ℝ (⊤ : ℕ∞) (g ∘ f) :=
  ⟨hf.add hg, hf.mul hg, hg.comp hf⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_rules (Real.sin) (Real.exp) (Real.contDiff_sin) (Real.contDiff_exp) := by
  refine ⟨by fun_prop, by fun_prop, ?_⟩
  exact ContDiff.comp (by fun_prop) (by fun_prop)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_rules ((fun x : ℝ => x ^ 2)) (Real.cos) ((contDiff_id.pow 2)) (Real.contDiff_cos) := by
  refine ⟨by fun_prop, by fun_prop, ?_⟩
  exact ContDiff.comp (by fun_prop) (by fun_prop)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_rules ((fun _ : ℝ => (0 : ℝ))) ((fun _ : ℝ => (1 : ℝ))) (contDiff_const) (contDiff_const) := by
  refine ⟨by fun_prop, by fun_prop, ?_⟩
  exact ContDiff.comp (by fun_prop) (by fun_prop)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.smooth_rules
-- test: ra-smooth-rules-test.sage

/-! ## exm-smooth-elementary -/

/-- Polynomials, `exp`, `sin`, `cos` are smooth on `ℝ`; `√x` is smooth on `(0, ∞)`; `|x|` is not differentiable at
`0`, hence not smooth there. -/
theorem smooth_elementary (P : Polynomial ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (fun x => P.eval x) ∧ ContDiff ℝ (⊤ : ℕ∞) Real.exp ∧ ContDiff ℝ (⊤ : ℕ∞) Real.sin ∧ ContDiff ℝ (⊤ : ℕ∞) Real.cos ∧
      ContDiffOn ℝ (⊤ : ℕ∞) Real.sqrt (Ioi 0) ∧ ¬ DifferentiableAt ℝ (fun x : ℝ => |x|) 0 := by
  refine ⟨by simpa using P.contDiff_aeval (𝕜 := ℝ) ((⊤ : ℕ∞) : WithTop ℕ∞), Real.contDiff_exp, Real.contDiff_sin, Real.contDiff_cos, ?_, not_differentiableAt_abs_zero⟩
  intro x hx
  exact (Real.contDiffAt_sqrt (ne_of_gt hx)).contDiffWithinAt

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_elementary ((Polynomial.X : Polynomial ℝ)) := by
  refine ⟨?_, Real.contDiff_exp, Real.contDiff_sin, Real.contDiff_cos, ?_, not_differentiableAt_abs_zero⟩
  · have h := Polynomial.contDiff_aeval (R := ℝ) (𝕜 := ℝ) ((Polynomial.X : Polynomial ℝ)) ((⊤ : ℕ∞) : WithTop ℕ∞)
    first | (simp only [Polynomial.coe_aeval_eq_eval] at h; exact h) | simpa using h | (simp; done)
  · intro x hx
    exact (Real.contDiffAt_sqrt (ne_of_gt hx)).contDiffWithinAt

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_elementary ((Polynomial.X ^ 3 - 2 : Polynomial ℝ)) := by
  refine ⟨?_, Real.contDiff_exp, Real.contDiff_sin, Real.contDiff_cos, ?_, not_differentiableAt_abs_zero⟩
  · have h := Polynomial.contDiff_aeval (R := ℝ) (𝕜 := ℝ) ((Polynomial.X ^ 3 - 2 : Polynomial ℝ)) ((⊤ : ℕ∞) : WithTop ℕ∞)
    first | (simp only [Polynomial.coe_aeval_eq_eval] at h; exact h) | simpa using h | (simp; done)
  · intro x hx
    exact (Real.contDiffAt_sqrt (ne_of_gt hx)).contDiffWithinAt

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.smooth_elementary ((2 * Polynomial.X ^ 2 + 1 : Polynomial ℝ)) := by
  refine ⟨?_, Real.contDiff_exp, Real.contDiff_sin, Real.contDiff_cos, ?_, not_differentiableAt_abs_zero⟩
  · have h := Polynomial.contDiff_aeval (R := ℝ) (𝕜 := ℝ) ((2 * Polynomial.X ^ 2 + 1 : Polynomial ℝ)) ((⊤ : ℕ∞) : WithTop ℕ∞)
    first | (simp only [Polynomial.coe_aeval_eq_eval] at h; exact h) | simpa using h | (simp; done)
  · intro x hx
    exact (Real.contDiffAt_sqrt (ne_of_gt hx)).contDiffWithinAt

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.smooth_elementary
-- test: ra-smooth-elementary-test.sage

/-! ## exm-smooth-polar -/

/-- The polar map is smooth on `ℝ²`; its inverse `(x, y) ↦ (√(x² + y²), atan2(y, x))` is smooth on the plane minus
the non-positive `x`-axis (the target of mathlib's `polarCoord`), where it is mathlib's `polarCoord`. -/
theorem polar_smooth (p : ℝ × ℝ) :
    ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => (p.1 * Real.cos p.2, p.1 * Real.sin p.2)) p ∧
      (p ∈ polarCoord.source → ContDiffAt ℝ (⊤ : ℕ∞) polarCoord p) := by
  have hF : ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × ℝ => (q.1 * Real.cos q.2, q.1 * Real.sin q.2)) := by fun_prop
  refine ⟨hF.contDiffAt, fun hp => ?_⟩
  set q := polarCoord p with hq
  have hqt : q ∈ polarCoord.target := polarCoord.map_source hp
  have hq1 : 0 < q.1 := hqt.1
  have hdet : (fderivPolarCoordSymm q).det ≠ 0 := by rw [det_fderivPolarCoordSymm]; exact hq1.ne'
  have hfun : (polarCoord.symm : ℝ × ℝ → ℝ × ℝ) = fun q => (q.1 * Real.cos q.2, q.1 * Real.sin q.2) := rfl
  have hsymm : ContDiffAt ℝ (⊤ : ℕ∞) polarCoord.symm q := by rw [hfun]; exact hF.contDiffAt
  have hp' : p ∈ polarCoord.symm.target := hp
  have hpq : polarCoord.symm.symm p = q := rfl
  have hderiv : HasFDerivAt polarCoord.symm
      (((fderivPolarCoordSymm q).toContinuousLinearEquivOfDetNeZero hdet : (ℝ × ℝ) ≃L[ℝ] (ℝ × ℝ)) :
        (ℝ × ℝ) →L[ℝ] (ℝ × ℝ))
      (polarCoord.symm.symm p) := by
    rw [ContinuousLinearMap.coe_toContinuousLinearEquivOfDetNeZero, hpq]
    exact hasFDerivAt_polarCoord_symm q
  exact polarCoord.symm.contDiffAt_symm hp' hderiv (by rw [hpq]; exact hsymm)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.polar_smooth
-- test: ra-polar-smooth-test.sage

/-! ## exm-smooth-transition -/

/-- The transition map `(u, v) ↦ (u, v)/(u² + v²)` between the stereographic charts is smooth away from the origin. -/
theorem inversion_smooth (p : ℝ × ℝ) (hp : p ≠ 0) :
    ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => (p.1 / (p.1 ^ 2 + p.2 ^ 2), p.2 / (p.1 ^ 2 + p.2 ^ 2))) p := by
  have hden : p.1 ^ 2 + p.2 ^ 2 ≠ 0 := by
    intro h
    apply hp
    have h1 : p.1 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2]
    have h2 : p.2 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2]
    exact Prod.ext h1 h2
  have hq : ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => p.1 ^ 2 + p.2 ^ 2) p := by fun_prop
  exact (contDiffAt_fst.div hq hden).prodMk (contDiffAt_snd.div hq hden)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.inversion_smooth
-- test: ra-inversion-smooth-test.sage

/-! ## exr-smooth-decide -/

/-- `x|x|` is differentiable everywhere with derivative `2|x|`, which is not differentiable at `0`: `C¹` but not
smooth at `0`. -/
theorem x_abs_x (x : ℝ) :
    HasDerivAt (fun x => x * |x|) (2 * |x|) x ∧ ¬ DifferentiableAt ℝ (fun x : ℝ => 2 * |x|) 0 := by
  refine ⟨?_, fun h => not_differentiableAt_abs_zero ?_⟩
  · rcases lt_trichotomy x 0 with hx | rfl | hx
    · have h1 : HasDerivAt (fun x : ℝ => -x ^ 2) (-(2 * x)) x :=
        (hasDerivAt_pow 2 x).neg.congr_deriv (by norm_num)
      rw [abs_of_neg hx]
      apply (h1.congr_deriv (by ring)).congr_of_eventuallyEq
      filter_upwards [Iio_mem_nhds hx] with y hy
      rw [abs_of_neg hy]; ring
    · rw [hasDerivAt_iff_tendsto_slope_zero]
      simp only [zero_add, abs_zero, mul_zero, sub_zero, smul_eq_mul]
      have h0 : Tendsto (fun t : ℝ => |t|) (𝓝[≠] 0) (𝓝 0) := by
        simpa using (continuous_abs.tendsto (0 : ℝ)).mono_left nhdsWithin_le_nhds
      apply h0.congr'
      filter_upwards [self_mem_nhdsWithin] with t ht
      have ht' : t ≠ 0 := ht
      field_simp
    · have h1 : HasDerivAt (fun x : ℝ => x ^ 2) (2 * x) x := by simpa using (hasDerivAt_pow 2 x)
      rw [abs_of_pos hx]
      apply h1.congr_of_eventuallyEq
      filter_upwards [Ioi_mem_nhds hx] with y hy
      rw [abs_of_pos hy]; ring
  · have := h.const_mul (1 / 2 : ℝ)
    simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.x_abs_x
-- test: ra-x-abs-x-test.sage

/-- `log(x² + y²)` is smooth on `ℝ²` minus the origin, and `(x² - y², 2xy)` on all of `ℝ²`. -/
theorem smooth_two_variables (p : ℝ × ℝ) :
    (p ≠ 0 → ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => Real.log (p.1 ^ 2 + p.2 ^ 2)) p) ∧
      ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => (p.1 ^ 2 - p.2 ^ 2, 2 * p.1 * p.2)) p := by
  refine ⟨fun hp => ?_, by fun_prop⟩
  have hden : p.1 ^ 2 + p.2 ^ 2 ≠ 0 := by
    intro h
    apply hp
    have h1 : p.1 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2]
    have h2 : p.2 = 0 := by nlinarith [sq_nonneg p.1, sq_nonneg p.2]
    exact Prod.ext h1 h2
  have hq : ContDiffAt ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => p.1 ^ 2 + p.2 ^ 2) p := by fun_prop
  exact (Real.contDiffAt_log.2 hden).comp p hq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.smooth_two_variables
-- test: ra-smooth-two-variables-test.sage

/-- An inverse `g` of `x ↦ x^(2k+1)` (`k ≥ 1`; the real cube root `x^(1/3)` for `k = 1`) is not differentiable at `0`:
differentiating `g(x)^(2k+1) = x` at `0` would give `1 = (2k+1) g(0)^(2k) g'(0) = 0`. -/
theorem odd_root_not_differentiable (k : ℕ) (hk : 1 ≤ k) (g : ℝ → ℝ) (hg : ∀ x, g x ^ (2 * k + 1) = x) :
    ¬ DifferentiableAt ℝ g 0 := by
  intro h
  have hd : HasDerivAt (fun x => g x ^ (2 * k + 1)) (((2 * k + 1 : ℕ) : ℝ) * g 0 ^ (2 * k) * deriv g 0) 0 := by
    exact (h.hasDerivAt.pow (2 * k + 1)).congr_deriv (by rw [Nat.add_sub_cancel])
  have h0 : g 0 = 0 := by have := hg 0; exact pow_eq_zero_iff (by omega) |>.1 this
  have hid : HasDerivAt (fun x => g x ^ (2 * k + 1)) 1 0 := by
    have : (fun x => g x ^ (2 * k + 1)) = id := funext hg
    rw [this]; exact hasDerivAt_id 0
  have := hd.unique hid
  rw [h0, zero_pow (by omega)] at this
  norm_num at this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.odd_root_not_differentiable
-- test: ra-odd-root-not-differentiable-test.sage

/-! ## exm-jacobian-polar -/

/-- The Jacobian determinant of the polar map is `r`. -/
theorem polar_jacobian_det (r φ : ℝ) :
    Matrix.det !![Real.cos φ, -r * Real.sin φ; Real.sin φ, r * Real.cos φ] = r := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals linear_combination ((r : ℝ)) * (Real.cos_sq_add_sin_sq (φ : ℝ)))
  -- certificate: ra-polar-det.sage

/-! ## exm-jacobian-linear -/

/-- The derivative of a linear map `x ↦ Ax` is `A` at every point, and that of a translation `x ↦ x + c` is the
identity. -/
theorem jacobian_linear (A : Matrix (Fin 2) (Fin 2) ℝ) (c p : Fin 2 → ℝ) :
    HasFDerivAt (fun x => A *ᵥ x)
        (LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' A)) p ∧
      HasFDerivAt (fun x => x + c) (ContinuousLinearMap.id ℝ (Fin 2 → ℝ)) p := by
  refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' A)).hasFDerivAt, ?_⟩
  simpa using (hasFDerivAt_id (𝕜 := ℝ) p).add_const c

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.jacobian_linear (1) (0) (0) := by
  refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (1 : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩
  simpa using (hasFDerivAt_id (𝕜 := ℝ) (0 : Fin 2 → ℝ)).add_const (0 : Fin 2 → ℝ)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.jacobian_linear (!![(1 : ℝ), 2; 3, 4]) (![1, -1]) (![0, 5]) := by
  refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (!![(1 : ℝ), 2; 3, 4] : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩
  simpa using (hasFDerivAt_id (𝕜 := ℝ) (![0, 5] : Fin 2 → ℝ)).add_const (![1, -1] : Fin 2 → ℝ)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.jacobian_linear (!![(0 : ℝ), -1; 1, 0]) (![2, 2]) (![1, 1]) := by
  refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (!![(0 : ℝ), -1; 1, 0] : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩
  simpa using (hasFDerivAt_id (𝕜 := ℝ) (![1, 1] : Fin 2 → ℝ)).add_const (![2, 2] : Fin 2 → ℝ)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.jacobian_linear
-- test: ra-jacobian-linear-test.sage

/-! ## exr-jacobian-polar -/

/-- The inverse of the polar Jacobian: `J⁻¹ = (cos φ, sin φ; -sin φ/r, cos φ/r)`, `J⁻¹ J = 1` for `r ≠ 0`. -/
theorem polar_jacobian_inverse (r φ : ℝ) (hr : r ≠ 0) :
    !![Real.cos φ, Real.sin φ; -Real.sin φ / r, Real.cos φ / r] *
      !![Real.cos φ, -r * Real.sin φ; Real.sin φ, r * Real.cos φ] = 1 := by
  have h := Real.cos_sq_add_sin_sq φ
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.one_apply] <;> field_simp <;>
    first | linear_combination h | linear_combination r * h | ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.polar_jacobian_inverse
-- test: ra-polar-jacobian-inverse-test.sage

/-- The chain rule against the inverse map `G(x, y) = (√(x² + y²), atan2(y, x))`: at `(x, y) = (r cos φ, r sin φ)`
with `r > 0`, `J_G = (x/ρ, y/ρ; -y/ρ², x/ρ²)` with `ρ = r` is `J_F⁻¹`. -/
theorem polar_inverse_jacobian (r φ : ℝ) (hr : 0 < r) :
    √((r * Real.cos φ) ^ 2 + (r * Real.sin φ) ^ 2) = r ∧
      !![r * Real.cos φ / r, r * Real.sin φ / r; -(r * Real.sin φ) / r ^ 2, r * Real.cos φ / r ^ 2] =
        !![Real.cos φ, Real.sin φ; -Real.sin φ / r, Real.cos φ / r] := by
  constructor
  · rw [show (r * Real.cos φ) ^ 2 + (r * Real.sin φ) ^ 2 = r ^ 2 by nlinarith [Real.sin_sq_add_cos_sq φ],
      Real.sqrt_sq hr.le]
  · ext i j; fin_cases i <;> fin_cases j <;> simp <;> field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.polar_inverse_jacobian
-- test: ra-polar-inverse-jacobian-test.sage

/-! ## exr-spherical-jacobian -/

/-- The Jacobian determinant of spherical coordinates is `r² sin θ`. -/
theorem spherical_jacobian_det (r θ φ : ℝ) :
    Matrix.det !![Real.sin θ * Real.cos φ, r * Real.cos θ * Real.cos φ, -r * Real.sin θ * Real.sin φ;
      Real.sin θ * Real.sin φ, r * Real.cos θ * Real.sin φ, r * Real.sin θ * Real.cos φ;
      Real.cos θ, -r * Real.sin θ, 0] = r ^ 2 * Real.sin θ := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals
        linear_combination
          (((r ^ 2) * (Real.sin θ) : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
            (((r ^ 2) * ((Real.cos θ) ^ 2) * (Real.sin θ) + ((r ^ 2) * ((Real.sin θ) ^ 3)) : ℝ)) *
              (Real.cos_sq_add_sin_sq (φ : ℝ)))
  -- certificate: ra-spherical-det.sage

/-- At the poles (`θ = 0`) the spherical map does not depend on `φ`: it is not injective there. -/
theorem spherical_pole (r φ : ℝ) :
    (r * Real.sin 0 * Real.cos φ, r * Real.sin 0 * Real.sin φ, r * Real.cos 0) = (0, 0, r) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.spherical_pole (1) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.spherical_pole (2) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.spherical_pole (3) (Real.pi) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.spherical_pole
-- test: ra-spherical-pole-test.sage

/-! ## exr-jacobian-spherical-sage -/

/-- The first row of the inverse spherical Jacobian is `(sin θ cos φ, sin θ sin φ, cos θ) = ∇r`: it times `J` is
`(1, 0, 0)`. -/
theorem spherical_inverse_row (r θ φ : ℝ) :
    !![Real.sin θ * Real.cos φ, Real.sin θ * Real.sin φ, Real.cos θ] *
      !![Real.sin θ * Real.cos φ, r * Real.cos θ * Real.cos φ, -r * Real.sin θ * Real.sin φ;
        Real.sin θ * Real.sin φ, r * Real.cos θ * Real.sin φ, r * Real.sin θ * Real.cos φ;
        Real.cos θ, -r * Real.sin θ, 0] = !![1, 0, 0] := by
  have hθ := Real.cos_sq_add_sin_sq θ
  have hφ := Real.cos_sq_add_sin_sq φ
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;>
    first | ring1 | linear_combination (Real.sin θ ^ 2) * hφ + hθ | linear_combination (r * Real.sin θ * Real.cos θ) * hφ

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.spherical_inverse_row
-- test: ra-spherical-inverse-row-test.sage

/-! ## thm-chain-rule -/

/-- The chain rule: the derivative of `G ∘ F` is the composition of the derivatives (the product of the Jacobian
matrices). -/
theorem chain_rule_maps (F G : (Fin 2 → ℝ) → (Fin 2 → ℝ)) (p : Fin 2 → ℝ) (F' G' : (Fin 2 → ℝ) →L[ℝ] (Fin 2 → ℝ))
    (hF : HasFDerivAt F F' p) (hG : HasFDerivAt G G' (F p)) : HasFDerivAt (G ∘ F) (G'.comp F') p :=
  hG.comp p hF

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.chain_rule_maps
-- test: ra-chain-rule-maps-test.sage

/-! ## exm-chain-curve -/

/-- Along the circle: `d/dt f(cos t, sin t) = -2 cos t sin² t + (cos² t + cos(sin t)) cos t`. -/
theorem chain_circle (t : ℝ) :
    deriv (fun t => Real.cos t ^ 2 * Real.sin t + Real.sin (Real.sin t)) t =
      -2 * Real.cos t * Real.sin t ^ 2 + (Real.cos t ^ 2 + Real.cos (Real.sin t)) * Real.cos t := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun t => cos t ^ 2 * sin t + sin (sin t))
          (((Real.cos t) ^ 3 - (2 * (Real.cos t) * ((Real.sin t) ^ 2)) + ((Real.cos t) * (Real.cos (Real.sin t))) : ℝ))
          (t) :=
        by
        convert
          (((((hasDerivAt_id' (t : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (t : ℝ)).sin)).add
            (((hasDerivAt_id' (t : ℝ)).sin).sin)) using
          1
        all_goals (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-chain-circle.sage

/-- `d/dt sin(t²) = cos(t²) · 2t`. -/
theorem chain_sin_sq (t : ℝ) : deriv (fun t => Real.sin (t ^ 2)) t = Real.cos (t ^ 2) * (2 * t) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => sin (t ^ 2)) ((2 * t * (Real.cos (t ^ 2)) : ℝ)) (t) :=
        by
        convert (((hasDerivAt_id' (t : ℝ)).pow 2).sin) using 1
        all_goals (simp only [Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ra-chain-sin-sq-t.sage

/-! ## exr-chain-two-changes -/

/-- The Jacobian of `R_α ∘ F` is `J_F` at `φ + α`, and it equals `R_α J_F(r, φ)`; its determinant is still `r`. -/
theorem rotation_polar (r φ α : ℝ) :
    !![Real.cos α, -Real.sin α; Real.sin α, Real.cos α] * !![Real.cos φ, -r * Real.sin φ; Real.sin φ, r * Real.cos φ] =
      !![Real.cos (φ + α), -r * Real.sin (φ + α); Real.sin (φ + α), r * Real.cos (φ + α)] := by
  rw [Real.cos_add, Real.sin_add]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.rotation_polar
-- test: ra-rotation-polar-test.sage

/-! ## exr-chain-polar-laplacian -/

/-- The Laplacian in polar coordinates (stated, tested on polynomial functions, not proved here): for smooth `u`,
`(∂ₓ² + ∂_y²) u = u_rr + u_r/r + u_φφ/r²` at `(r cos φ, r sin φ)`, `r > 0`. -/
def PolarLaplacian : Prop :=
  ∀ (u : ℝ → ℝ → ℝ), ContDiff ℝ 2 (fun p : ℝ × ℝ => u p.1 p.2) → ∀ r φ : ℝ, 0 < r →
    let v := fun r φ => u (r * Real.cos φ) (r * Real.sin φ)
    deriv (fun x => deriv (fun x => u x (r * Real.sin φ)) x) (r * Real.cos φ) +
      deriv (fun y => deriv (fun y => u (r * Real.cos φ) y) y) (r * Real.sin φ) =
    deriv (fun r => deriv (fun r => v r φ) r) r + deriv (fun r => v r φ) r / r +
      deriv (fun φ => deriv (fun φ => v r φ) φ) φ / r ^ 2

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Calculus.PolarLaplacian
-- test: ra-polarlaplacian-test.sage

/-! ## def-diffeomorphism -/

/-- The inverse function theorem in one variable: a strictly differentiable `f` with `f'(a) ≠ 0` has a local
inverse, differentiable at `f(a)` with derivative `1/f'(a)`. -/
theorem inverse_function_theorem (f : ℝ → ℝ) (a f' : ℝ) (hf : HasStrictDerivAt f f' a) (hf' : f' ≠ 0) :
    HasStrictDerivAt (hf.localInverse f f' a hf') f'⁻¹ (f a) :=
  hf.to_localInverse hf'

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Calculus.inverse_function_theorem
-- test: ra-inverse-function-theorem-test.sage

/-- The Jacobian of a diffeomorphism is invertible: from `G ∘ F = id` the chain rule gives `J_G(F p) J_F(p) = 1`. -/
theorem diffeo_jacobian_invertible (A B : Matrix (Fin 2) (Fin 2) ℝ) (h : B * A = 1) : IsUnit A.det := by
  have := congrArg Matrix.det h
  rw [Matrix.det_mul, Matrix.det_one] at this
  exact isUnit_iff_ne_zero.2 (fun h0 => by rw [h0, mul_zero] at this; exact zero_ne_one this)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.diffeo_jacobian_invertible (1) (1) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.diffeo_jacobian_invertible (!![(2 : ℝ), 1; 1, 1]) (!![(1 : ℝ), -1; -1, 2]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  rw [Matrix.det_fin_two_of]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.diffeo_jacobian_invertible (!![(0 : ℝ), 1; 1, 0]) (!![(0 : ℝ), 1; 1, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  rw [Matrix.det_fin_two_of]; norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.diffeo_jacobian_invertible
-- test: ra-diffeo-jacobian-invertible-test.sage

/-! ## exm-diffeo-polar -/

/-- The polar map is a diffeomorphism from `(0, ∞) × (-π, π)` onto the plane minus the non-positive `x`-axis
(mathlib's `polarCoord`); at `r = 0` all angles give the origin. -/
theorem polar_diffeo (φ : ℝ) :
    polarCoord.target = Ioi (0 : ℝ) ×ˢ Ioo (-π) π ∧ (0 * Real.cos φ, 0 * Real.sin φ) = ((0 : ℝ), (0 : ℝ)) := by
  refine ⟨rfl, by simp⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.polar_diffeo (0) := by
  exact ⟨rfl, by simp⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.polar_diffeo (1) := by
  exact ⟨rfl, by simp⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.polar_diffeo (Real.pi / 2) := by
  exact ⟨rfl, by simp⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.polar_diffeo
-- test: ra-polar-diffeo-test.sage

/-! ## exm-diffeo-cube -/

/-- `x ↦ x³` is a smooth bijection of `ℝ` (strictly increasing and onto); its inverse is not differentiable at `0`
(see `cbrt_not_differentiable`); `tan` has derivative `1 + tan² x > 0` on `(-π/2, π/2)`. -/
theorem cube_bijection (y : ℝ) (x : ℝ) (hx : Real.cos x ≠ 0) :
    (∃! z : ℝ, z ^ 3 = y) ∧ deriv Real.tan x = 1 + Real.tan x ^ 2 := by
  constructor
  · have hmono : StrictMono (fun z : ℝ => z ^ 3) := Odd.strictMono_pow (by decide)
    have hc : Continuous (fun z : ℝ => z ^ 3) := by fun_prop
    obtain ⟨z, -, hz⟩ := intermediate_value_Icc (show -(|y| + 1) ≤ |y| + 1 by linarith [abs_nonneg y]) hc.continuousOn
      (show y ∈ Icc ((-(|y| + 1)) ^ 3) ((|y| + 1) ^ 3) by
        constructor <;> nlinarith [abs_nonneg y, le_abs_self y, neg_abs_le y, sq_nonneg (|y| + 1)])
    exact ⟨z, hz, fun w hw => hmono.injective (hw.trans hz.symm)⟩
  · rw [Real.deriv_tan, Real.tan_eq_sin_div_cos]
    field_simp
    linarith [Real.sin_sq_add_cos_sq x]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.cube_bijection (0) (0) ((by norm_num)) := by
  refine ⟨⟨0, by norm_num, fun z hz => by simpa using hz⟩, ?_⟩
  rw [Real.deriv_tan, Real.tan_zero]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.cube_bijection (8) (0) ((by norm_num)) := by
  refine ⟨⟨2, by norm_num, fun z hz => ?_⟩, ?_⟩
  · nlinarith [sq_nonneg (z - 2), sq_nonneg (z + 1), sq_nonneg z]
  · rw [Real.deriv_tan, Real.tan_zero]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.cube_bijection (-27) (0) ((by norm_num)) := by
  refine ⟨⟨-3, by norm_num, fun z hz => ?_⟩, ?_⟩
  · nlinarith [sq_nonneg (z + 3), sq_nonneg (z - 1), sq_nonneg z]
  · rw [Real.deriv_tan, Real.tan_zero]; norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.cube_bijection
-- test: ra-cube-bijection-test.sage

/-! ## exm-diffeo-transition -/

/-- The inversion `I(u, v) = (u, v)/(u² + v²)` is its own inverse, and its Jacobian determinant is
`-1/(u² + v²)²`. -/
theorem inversion_props (u v : ℝ) (h : u ^ 2 + v ^ 2 ≠ 0) :
    (u / (u ^ 2 + v ^ 2)) / ((u / (u ^ 2 + v ^ 2)) ^ 2 + (v / (u ^ 2 + v ^ 2)) ^ 2) = u ∧
      Matrix.det !![(v ^ 2 - u ^ 2) / (u ^ 2 + v ^ 2) ^ 2, -2 * u * v / (u ^ 2 + v ^ 2) ^ 2;
        -2 * u * v / (u ^ 2 + v ^ 2) ^ 2, (u ^ 2 - v ^ 2) / (u ^ 2 + v ^ 2) ^ 2] = -1 / (u ^ 2 + v ^ 2) ^ 2 := by
  constructor
  · field_simp <;> ring
  · rw [Matrix.det_fin_two_of]; field_simp <;> ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inversion_props (1) (0) ((by norm_num)) := by
  constructor
  · norm_num
  · rw [Matrix.det_fin_two_of]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inversion_props (1) (2) ((by norm_num)) := by
  constructor
  · norm_num
  · rw [Matrix.det_fin_two_of]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inversion_props (-3) (1 / 2) ((by norm_num)) := by
  constructor
  · norm_num
  · rw [Matrix.det_fin_two_of]; norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.inversion_props
-- test: ra-inversion-props-test.sage

/-! ## exr-inverse-function-sage -/

/-- For `F(x, y) = (x² - y², 2xy)`: `det J = 4(x² + y²)`. -/
theorem square_map_det (x y : ℝ) : Matrix.det !![2 * x, -2 * y; 2 * y, 2 * x] = 4 * (x ^ 2 + y ^ 2) := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals ring1)
  -- certificate: ra-square-det.sage

/-- `F(x, y) = F(-x, -y)`: the map is two-to-one. -/
theorem square_map_two_to_one (x y : ℝ) :
    ((-x) ^ 2 - (-y) ^ 2, 2 * (-x) * (-y)) = (x ^ 2 - y ^ 2, 2 * x * y) := by
  ext <;> simp <;> ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.square_map_two_to_one (0) (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.square_map_two_to_one (1) (2) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.square_map_two_to_one (-3) (1 / 2) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.square_map_two_to_one
-- test: ra-square-map-two-to-one-test.sage

/-- The Jacobian of the local inverse at `(1, 1)` is `J⁻¹ = ¼ (1, 1; -1, 1)`. -/
theorem square_map_inverse : !![(1 : ℝ) / 4, 1 / 4; -1 / 4, 1 / 4] * !![2 * 1, -2 * 1; 2 * 1, 2 * 1] = 1 := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals ring1))
  -- certificate: ra-square-inverse.sage

/-! ## exr-diffeo-volume-factor -/

/-- Volume factors multiply: `det(J_G J_F) = det J_G det J_F`. -/
theorem volume_factor_mul (A B : Matrix (Fin 2) (Fin 2) ℚ) : (A * B).det = A.det * B.det :=
  Matrix.det_mul A B

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Calculus.volume_factor_mul
-- test: ra-volume-factor-mul-test.sage

/-! ## exm-polar-jacobian-areas -/

/-- The polar rectangle `[r₀, r₀ + δ] × [φ₀, φ₀ + ε]` maps to a sector of area `½((r₀ + δ)² - r₀²) ε =
(r₀ + δ/2) δ ε`: the Jacobian factor at the midpoint times the rectangle's area, exactly. -/
theorem sector_area (r₀ δ ε : ℝ) : (1 / 2) * ((r₀ + δ) ^ 2 - r₀ ^ 2) * ε = (r₀ + δ / 2) * δ * ε := by
  veriphy_certified (ring1)
  -- certificate: ra-sector-area.sage

/-! ## exm-polar-jacobian-higher -/

/-- The surface area `n π^(n/2)/Γ(n/2 + 1)` of the unit sphere of `ℝⁿ` (`n` times the volume of the unit ball)
equals `2π^(n/2)/Γ(n/2)`. -/
theorem sphere_area (n : ℕ) (hn : 1 ≤ n) :
    (n : ℝ) * π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2 + 1) = 2 * π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2) := by
  have hpos : 0 < (n : ℝ) / 2 := by positivity
  rw [Real.Gamma_add_one hpos.ne']
  have : Real.Gamma ((n : ℝ) / 2) ≠ 0 := (Real.Gamma_pos_of_pos hpos).ne'
  field_simp <;> ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Calculus.sphere_area
-- test: ra-sphere-area-test.sage

/-! ## exr-polar-jacobian-rotation -/

/-- The rotation and the shear have Jacobian determinant `1`. -/
theorem rotation_shear_det (α k : ℝ) :
    Matrix.det !![Real.cos α, -Real.sin α; Real.sin α, Real.cos α] = 1 ∧ Matrix.det !![(1 : ℝ), k; 0, 1] = 1 := by
  constructor
  · veriphy_certified
      ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
          Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
        all_goals linear_combination ((1 : ℝ)) * (Real.cos_sq_add_sin_sq (α : ℝ)))
    -- certificate: ra-rotation-det.sage
  · veriphy_certified
      ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
          Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
        all_goals ring1)
    -- certificate: ra-shear-det.sage

/-! ## exr-polar-jacobian-elliptic -/

/-- Elliptic polar coordinates `(ar cos t, br sin t)` have `det J = abr`. -/
theorem elliptic_det (a b r t : ℝ) :
    Matrix.det !![a * Real.cos t, -a * r * Real.sin t; b * Real.sin t, b * r * Real.cos t] = a * b * r := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals linear_combination ((a * b * r : ℝ)) * (Real.cos_sq_add_sin_sq (t : ℝ)))
  -- certificate: ra-elliptic-det.sage

/-! ## exr-polar-jacobian-inverse -/

/-- `det J_(F⁻¹)(F p) = 1/det J_F(p)`: from `J_(F⁻¹) J_F = 1`; for the inverse polar map, `det J = 1/r`. -/
theorem inverse_det (A B : Matrix (Fin 2) (Fin 2) ℝ) (h : B * A = 1) : B.det = 1 / A.det := by
  have := congrArg Matrix.det h
  rw [Matrix.det_mul, Matrix.det_one] at this
  have hA : A.det ≠ 0 := by intro h0; rw [h0, mul_zero] at this; exact zero_ne_one this
  field_simp
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inverse_det (1) (1) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inverse_det (!![(2 : ℝ), 1; 1, 1]) (!![(1 : ℝ), -1; -1, 2]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  rw [Matrix.det_fin_two_of, Matrix.det_fin_two_of]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Calculus.inverse_det (!![(2 : ℝ), 0; 0, 4]) (!![(1 / 2 : ℝ), 0; 0, 1 / 4]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  rw [Matrix.det_fin_two_of, Matrix.det_fin_two_of]; norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Calculus.inverse_det
-- test: ra-inverse-det-test.sage

theorem inverse_polar_det (r φ : ℝ) (hr : r ≠ 0) :
    Matrix.det !![Real.cos φ, Real.sin φ; -Real.sin φ / r, Real.cos φ / r] = 1 / r := by
  rw [Matrix.det_fin_two_of]
  veriphy_certified
    ( field_simp
      linear_combination Real.cos_sq_add_sin_sq φ)
  -- certificate: ra-inverse-polar-det.sage

/-! ## exm-partial-x-encoding -/

/-- `∂_y(x² y + sin y) = x² + cos y`. -/
theorem partial_y_encoding (x y : ℝ) : deriv (fun y => x ^ 2 * y + Real.sin y) y = x ^ 2 + Real.cos y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun y => x ^ 2 * y + sin y) ((x ^ 2 + (Real.cos y) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).const_mul (x ^ 2 : ℝ)).add ((hasDerivAt_id' (y : ℝ)).sin)) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: ra-partial-y-encoding.sage

/-! ## exr-partial-x-statements -/

/-- `∂ₓ e^(xy) = y e^(xy)`. -/
theorem partial_x_exp (x y : ℝ) : deriv (fun x => Real.exp (x * y)) x = y * Real.exp (x * y) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => rexp (x * y)) ((y * (Real.exp (y * x)) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).const_mul (y : ℝ)).exp) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by (ring_nf; done)))
  -- certificate: ra-partial-x-exp.sage

/-! ## exr-partial-x-hessian -/

/-- The Hessian of `x² y + sin y`: `∂ₓ(2xy) = 2y`, `∂_y(x² + cos y) = -sin y`; it is symmetric. -/
theorem hessian_f (x y : ℝ) :
    deriv (fun x => 2 * x * y) x = 2 * y ∧ deriv (fun y => x ^ 2 + Real.cos y) y = -Real.sin y := by
  constructor
  · veriphy_certified
      ( have hd : HasDerivAt (fun x => 2 * x * y) ((2 * y : ℝ)) (x) :=
          by
          convert ((hasDerivAt_id' (x : ℝ)).const_mul (2 * y : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            | (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-hessian-xx.sage
  · veriphy_certified
      ( have hd : HasDerivAt (fun y => x ^ 2 + cos y) ((-(Real.sin y) : ℝ)) (y) :=
          by
          convert ((hasDerivAt_const (y : ℝ) (x ^ 2 : ℝ)).add ((hasDerivAt_id' (y : ℝ)).cos)) using 1
          all_goals (ring1)
        exact hd.deriv.trans (by rfl))
    -- certificate: ra-hessian-yy.sage

/-- `g = x² + 4y² - 2x` has `∇g = (2x - 2, 8y)`, vanishing at `(1, 0)`, and `g - (-1) = (x - 1)² + 4y² ≥ 0`: a strict
minimum with `g(1, 0) = -1` and Hessian `diag(2, 8)`. -/
theorem critical_point_g (x y : ℝ) :
    x ^ 2 + 4 * y ^ 2 - 2 * x - (-1) = (x - 1) ^ 2 + 4 * y ^ 2 := by
  veriphy_certified (ring1)
  -- certificate: ra-critical-g.sage

end KaluzaKlein.RealAnalysis.Calculus

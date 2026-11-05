import Mathlib
import Veriphy.Kit.Levels

/-!
# Classical mechanics, section "Momentum and force"

The claims of the section "Momentum and force" of `tutorial/part-ii-01-classical-mechanics.qmd`. Newton's second law
is modelled by functions `ℝ → ℝ` (one Cartesian component at a time) and `deriv`; a motion is an explicit function
of time whose derivatives are checked, forces and accelerations enter as jet variables, and the uniqueness of the
motion with given initial data is proved for a constant force.
-/

open Real Filter Topology

namespace KaluzaKlein.ClassicalMechanics.Momentum

/-! ## def-clm-momentum -/

/-- The second law with a constant force is a second-order equation whose solution is fixed by the initial position
and velocity: if `ẍ = a` for every `t`, then `x t = x 0 + ẋ 0 · t + a t²/2`. -/
theorem const_force_unique (x : ℝ → ℝ) (a : ℝ) (hx : Differentiable ℝ x) (hx2 : Differentiable ℝ (deriv x))
    (h : ∀ t, deriv (deriv x) t = a) (t : ℝ) : x t = x 0 + deriv x 0 * t + a * t ^ 2 / 2 := by
  have hg : ∀ s, deriv (fun u => deriv x u - a * u) s = 0 := by
    intro s
    have h1 : HasDerivAt (fun u => deriv x u - a * u) (deriv (deriv x) s - a * 1) s :=
      (hx2 s).hasDerivAt.sub ((hasDerivAt_id' s).const_mul a)
    rw [h1.deriv, h]; ring
  have hgc := is_const_of_deriv_eq_zero (hx2.sub (differentiable_id.const_mul a)) hg
  have hv : ∀ s, deriv x s = deriv x 0 + a * s := by
    intro s; have := hgc s 0; simp only [Pi.sub_apply] at this; linarith
  have hf : ∀ s, deriv (fun u => x u - (x 0 + deriv x 0 * u + a * u ^ 2 / 2)) s = 0 := by
    intro s
    have h1 : HasDerivAt (fun u => x 0 + deriv x 0 * u + a * u ^ 2 / 2)
        (0 + deriv x 0 * 1 + a * (↑(2 : ℕ) * s ^ (2 - 1) * 1) / 2) s :=
      ((hasDerivAt_const s (x 0)).add ((hasDerivAt_id' s).const_mul (deriv x 0))).add
        ((((hasDerivAt_id' s).pow 2).const_mul a).div_const 2)
    have h2 : HasDerivAt (fun u => x u - (x 0 + deriv x 0 * u + a * u ^ 2 / 2))
        (deriv x s - (0 + deriv x 0 * 1 + a * (↑(2 : ℕ) * s ^ (2 - 1) * 1) / 2)) s :=
      (hx s).hasDerivAt.sub h1
    rw [h2.deriv, hv s]; push_cast; ring
  have hfd : Differentiable ℝ (fun u => x u - (x 0 + deriv x 0 * u + a * u ^ 2 / 2)) := by fun_prop
  have := is_const_of_deriv_eq_zero hfd hf t 0
  simp only [mul_zero, add_zero, zero_pow two_ne_zero, zero_div] at this
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ClassicalMechanics.Momentum.const_force_unique
-- test: cm-const-force-unique-test.sage

/-! ## exm-clm-momentum-free -/

/-- With no force, `x = x₀ + v₀ t` has `ẍ = 0`. -/
theorem free_solves (x₀ v₀ : ℝ) (x : ℝ → ℝ) (hx : x = fun s => x₀ + v₀ * s) (t : ℝ) :
    deriv (deriv x) t = 0 := by
  veriphy_certified
    ( subst hx
      have d1 : deriv (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) = fun (s : ℝ) => (v₀ : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) ((v₀ : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (v₀ : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (x₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (v₀ : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (v₀ : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (v₀ : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1, d2]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- Its momentum `m ẋ = m v₀` is constant. -/
theorem free_momentum (m x₀ v₀ : ℝ) (x : ℝ → ℝ) (hx : x = fun s => x₀ + v₀ * s) (t : ℝ) :
    m * deriv x t = m * v₀ := by
  veriphy_certified
    ( subst hx
      have d1 : deriv (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) = fun (s : ℝ) => (v₀ : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) ((v₀ : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (v₀ : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (x₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- Conversely `ṗ = 0` (`ẍ = 0` for every `t`) forces the uniform motion `x t = x 0 + ẋ 0 · t`. -/
theorem free_unique (x : ℝ → ℝ) (hx : Differentiable ℝ x) (hx2 : Differentiable ℝ (deriv x))
    (h : ∀ t, deriv (deriv x) t = 0) (t : ℝ) : x t = x 0 + deriv x 0 * t := by
  have := const_force_unique x 0 hx hx2 h t
  simpa using this
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ClassicalMechanics.Momentum.free_unique
-- test: cm-free-unique-test.sage

/-! ## exm-clm-momentum-gravity -/

/-- The vertical motion `z = z₀ + w₀ t - g t²/2` has `z̈ = -g`. -/
theorem gravity_vertical (g z₀ w₀ : ℝ) (z : ℝ → ℝ) (hz : z = fun s => z₀ + w₀ * s - g * s ^ 2 / 2) (t : ℝ) :
    deriv (deriv z) t = -g := by
  veriphy_certified
    ( subst hz
      have d1 : deriv (fun (s : ℝ) => (z₀ + w₀ * s - g * s ^ 2 / 2 : ℝ)) = fun (s : ℝ) => (-s * g + w₀ : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (z₀ + w₀ * s - g * s ^ 2 / 2 : ℝ)) ((-s * g + w₀ : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-g / 2 : ℝ)).add
                          ((hasDerivAt_id' (s : ℝ)).const_mul (w₀ : ℝ))).add
                      (hasDerivAt_const (s : ℝ) (z₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (-s * g + w₀ : ℝ)) = fun (s : ℝ) => (-g : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (-s * g + w₀ : ℝ)) ((-g : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (-g : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (w₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- The horizontal momentum `m ẋ` of `x = x₀ + v₀ t` is the constant `m v₀`. -/
theorem gravity_horizontal (m x₀ v₀ : ℝ) (x : ℝ → ℝ) (hx : x = fun s => x₀ + v₀ * s) (t : ℝ) :
    m * deriv x t = m * v₀ := by
  veriphy_certified
    ( subst hx
      have d1 : deriv (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) = fun (s : ℝ) => (v₀ : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (x₀ + v₀ * s : ℝ)) ((v₀ : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (v₀ : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (x₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- The vertical momentum `p_z = m ż = m (w₀ - g t)` … -/
theorem gravity_vertical_momentum (m g z₀ w₀ : ℝ) (z : ℝ → ℝ) (hz : z = fun s => z₀ + w₀ * s - g * s ^ 2 / 2)
    (t : ℝ) : m * deriv z t = m * (w₀ - g * t) := by
  veriphy_certified
    ( subst hz
      have d1 : deriv (fun (s : ℝ) => (z₀ + w₀ * s - g * s ^ 2 / 2 : ℝ)) = fun (s : ℝ) => (-s * g + w₀ : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (z₀ + w₀ * s - g * s ^ 2 / 2 : ℝ)) ((-s * g + w₀ : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-g / 2 : ℝ)).add
                          ((hasDerivAt_id' (s : ℝ)).const_mul (w₀ : ℝ))).add
                      (hasDerivAt_const (s : ℝ) (z₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals (ring1))
  -- certificate: cm-ode.sage

/-- … decreases at the rate `m g`. -/
theorem gravity_momentum_rate (m g w₀ t : ℝ) : deriv (fun s => m * (w₀ - g * s)) t = -(m * g) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => m * (w₀ - g * s)) ((-m * g : ℝ)) (t) :=
        by
        refine
          ((((((hasDerivAt_id' (t : ℝ)).const_mul (g : ℝ)).add (hasDerivAt_const (t : ℝ) (-w₀ : ℝ))).const_mul
                    (-m : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: cm-derivative.sage

/-! ## exm-clm-momentum-two-body -/

/-- `Ṗ = F₁₂ + F₂₁ = 0` by the third law `F₂₁ = -F₁₂`. -/
theorem total_momentum_rate (f : ℝ) : f + (-f) = 0 := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- `Ẍ = (m₁ ẍ₁ + m₂ ẍ₂)/(m₁ + m₂) = 0` with `m₁ ẍ₁ = f`, `m₂ ẍ₂ = -f`. -/
theorem centre_of_mass_free (m₁ m₂ f : ℝ) (h₁ : m₁ ≠ 0) (h₂ : m₂ ≠ 0) :
    (m₁ * (f / m₁) + m₂ * (-f / m₂)) / (m₁ + m₂) = 0 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- `X t = X₀ + P t/M` moves uniformly: `Ẋ = P/M`. -/
theorem centre_of_mass_uniform (X₀ P M : ℝ) (hM : M ≠ 0) (X : ℝ → ℝ) (hX : X = fun s => X₀ + P * s / M) (t : ℝ) :
    deriv X t = P / M := by
  veriphy_certified
    ( subst hX
      have d1 : deriv (fun (s : ℝ) => (X₀ + P * s / M : ℝ)) = fun (s : ℝ) => (P / M : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (X₀ + P * s / M : ℝ)) ((P / M : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (P / M : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (X₀ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-! ## exr-clm-projectile -/

/-- `x = v₀ t cos α` solves `ẍ = 0`. -/
theorem projectile_x (v₀ α : ℝ) (x : ℝ → ℝ) (hx : x = fun s => v₀ * s * Real.cos α) (t : ℝ) :
    deriv (deriv x) t = 0 := by
  veriphy_certified
    ( subst hx
      have d1 : deriv (fun (s : ℝ) => (v₀ * s * cos α : ℝ)) = fun (s : ℝ) => (v₀ * (Real.cos α) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (v₀ * s * cos α : ℝ)) ((v₀ * (Real.cos α) : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).const_mul (v₀ * (Real.cos α) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (v₀ * (Real.cos α) : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (v₀ * (Real.cos α) : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (v₀ * (Real.cos α) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1, d2]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- `z = v₀ t sin α - g t²/2` solves `z̈ = -g`. -/
theorem projectile_z (v₀ α g : ℝ) (z : ℝ → ℝ) (hz : z = fun s => v₀ * s * Real.sin α - g * s ^ 2 / 2) (t : ℝ) :
    deriv (deriv z) t = -g := by
  veriphy_certified
    ( subst hz
      have d1 :
        deriv (fun (s : ℝ) => (v₀ * s * sin α - g * s ^ 2 / 2 : ℝ)) = fun (s : ℝ) =>
          (-s * g + (v₀ * (Real.sin α)) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (v₀ * s * sin α - g * s ^ 2 / 2 : ℝ)) ((-s * g + (v₀ * (Real.sin α)) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-g / 2 : ℝ)).add
                      ((hasDerivAt_id' (s : ℝ)).const_mul (v₀ * (Real.sin α) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (-s * g + (v₀ * (Real.sin α)) : ℝ)) = fun (s : ℝ) => (-g : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (-s * g + (v₀ * (Real.sin α)) : ℝ)) ((-g : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (-g : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (v₀ * (Real.sin α) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (rfl))
  -- certificate: cm-ode.sage

/-- `z` returns to `0` at `t = 2 v₀ sin α / g` (with `s = sin α`). -/
theorem projectile_landing (v₀ s g : ℝ) (hg : g ≠ 0) :
    v₀ * (2 * v₀ * s / g) * s - g * (2 * v₀ * s / g) ^ 2 / 2 = 0 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- The range is `x(2 v₀ sin α / g) = 2 v₀² sin α cos α / g` (with `s = sin α`, `c = cos α`) … -/
theorem projectile_range (v₀ s c g : ℝ) (hg : g ≠ 0) :
    v₀ * (2 * v₀ * s / g) * c = v₀ ^ 2 * (2 * s * c) / g := by
  veriphy_certified (ring1)
  -- certificate: cm-poly.sage

/-- … that is `v₀² sin 2α / g`, since `sin 2α = 2 sin α cos α`. -/
theorem projectile_range_double_angle (α : ℝ) : Real.sin (2 * α) = 2 * Real.sin α * Real.cos α := by
  veriphy_certified (exact Real.sin_two_mul α)
  -- certificate: cm-rewrite.sage

/-- The range is maximal at `α = π/4`, where it is `v₀²/g`: `sin 2α ≤ 1 = sin (π/2)`. -/
theorem projectile_range_max (v₀ α g : ℝ) (hg : 0 < g) :
    v₀ ^ 2 * Real.sin (2 * α) / g ≤ v₀ ^ 2 / g ∧ v₀ ^ 2 * Real.sin (2 * (Real.pi / 4)) / g = v₀ ^ 2 / g := by
  constructor
  · apply div_le_div_of_nonneg_right _ hg.le
    calc v₀ ^ 2 * Real.sin (2 * α) ≤ v₀ ^ 2 * 1 := by gcongr; exact Real.sin_le_one _
      _ = v₀ ^ 2 := mul_one _
  · rw [show (2 : ℝ) * (Real.pi / 4) = Real.pi / 2 by ring, Real.sin_pi_div_two, mul_one]
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.projectile_range_max (1) (0) (1) ((by norm_num)) := by
  constructor <;> norm_num [show (2 : ℝ) * (Real.pi / 4) = Real.pi / 2 by ring, show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring, Real.sin_pi_div_two, Real.sin_pi]

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.projectile_range_max (2) (Real.pi / 4) (1) ((by norm_num)) := by
  constructor <;> norm_num [show (2 : ℝ) * (Real.pi / 4) = Real.pi / 2 by ring, show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring, Real.sin_pi_div_two, Real.sin_pi]

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.projectile_range_max (1) (Real.pi / 2) (3) ((by norm_num)) := by
  constructor <;> norm_num [show (2 : ℝ) * (Real.pi / 4) = Real.pi / 2 by ring, show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring, Real.sin_pi_div_two, Real.sin_pi]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Momentum.projectile_range_max
-- test: cm-projectile-range-max-test.sage

/-! ## exr-clm-drag -/

/-- `v = v₀ e^{-γ t/m}` solves `m v̇ = -γ v`. -/
theorem drag_solves (m γ v₀ : ℝ) (hm : m ≠ 0) (v : ℝ → ℝ) (hv : v = fun s => v₀ * Real.exp (-γ * s / m)) (t : ℝ) :
    m * deriv v t = -γ * v t := by
  veriphy_certified
    ( subst hv
      have d1 :
        deriv (fun (s : ℝ) => (v₀ * rexp (-γ * s / m) : ℝ)) = fun (s : ℝ) =>
          (-γ * v₀ * (Real.exp (-s * γ / m)) / m : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (v₀ * rexp (-γ * s / m) : ℝ)) ((-γ * v₀ * (Real.exp (-s * γ / m)) / m : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (-γ / m : ℝ)).exp).const_mul (v₀ : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1]
      all_goals ((field_simp; done)))
  -- certificate: cm-ode.sage

/-- The distance travelled, `∫₀ᵗ v = (m v₀/γ)(1 - e^{-γ t/m})`. -/
theorem drag_distance (m γ v₀ t : ℝ) (hm : m ≠ 0) (hγ : γ ≠ 0) :
    ∫ s in (0 : ℝ)..t, v₀ * Real.exp (-γ * s / m) = m * v₀ / γ * (1 - Real.exp (-γ * t / m)) := by
  veriphy_certified
    ( have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (t : ℝ),
          HasDerivAt (fun i0 => (-m * v₀ * (Real.exp (-i0 * γ / m)) / γ : ℝ)) ((fun s => v₀ * rexp (-γ * s / m)) i0)
            i0 :=
        by
        intro i0 _
        refine
          ((((((hasDerivAt_id' (i0 : ℝ)).const_mul (-γ / m : ℝ)).exp).const_mul
                    (-m * v₀ / γ : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · beta_reduce; (field_simp; done)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      all_goals (try beta_reduce)
      all_goals
        (try
            simp only [mul_zero, zero_mul, zero_div, neg_zero, Real.exp_zero, sub_zero, zero_sub, add_zero, zero_add,
              Real.sin_zero, Real.cos_zero, mul_one, one_mul])
      all_goals ((field_simp; ring1)))
  -- certificate: cm-integral-exp.sage

/-- It tends to `m v₀/γ`: the particle stops after a finite distance … -/
theorem drag_limit (m γ v₀ : ℝ) (hm : 0 < m) (hγ : 0 < γ) :
    Tendsto (fun t => m * v₀ / γ * (1 - Real.exp (-γ * t / m))) atTop (𝓝 (m * v₀ / γ)) := by
  have h1 : Tendsto (fun t : ℝ => -γ * t / m) atTop atBot := by
    have : (fun t : ℝ => -γ * t / m) = fun t => (-γ / m) * t := by funext t; ring
    rw [this]
    exact tendsto_id.const_mul_atTop_of_neg (by apply div_neg_of_neg_of_pos (by linarith) hm)
  have h2 : Tendsto (fun t : ℝ => Real.exp (-γ * t / m)) atTop (𝓝 0) := Real.tendsto_exp_atBot.comp h1
  have h3 := (tendsto_const_nhds (x := (1 : ℝ))).sub h2
  simpa using h3.const_mul (m * v₀ / γ)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ClassicalMechanics.Momentum.drag_limit
-- test: cm-drag-limit-test.sage

/-- … although never in a finite time: `v t > 0` for every `t` when `v₀ > 0`. -/
theorem drag_never_stops (m γ v₀ : ℝ) (hv : 0 < v₀) (t : ℝ) : 0 < v₀ * Real.exp (-γ * t / m) := by
  positivity
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_never_stops (1) (1) (1) ((by norm_num)) (0) := by
  positivity

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_never_stops (2) (1/2) (3) ((by norm_num)) (5) := by
  positivity

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_never_stops (1) (-1) (1/10) ((by norm_num)) (-2) := by
  positivity

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Momentum.drag_never_stops
-- test: cm-drag-never-stops-test.sage

/-- The momentum is not conserved: `ṗ = -γ v ≠ 0`. -/
theorem drag_momentum_changes (γ v₀ : ℝ) (hγ : γ ≠ 0) (hv : v₀ ≠ 0) (m t : ℝ) :
    -γ * (v₀ * Real.exp (-γ * t / m)) ≠ 0 :=
  mul_ne_zero (neg_ne_zero.mpr hγ) (mul_ne_zero hv (Real.exp_pos _).ne')
example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_momentum_changes (1) (1) ((by norm_num)) ((by norm_num)) (1) (0) := by
  norm_num [Real.exp_ne_zero]

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_momentum_changes (1/2) (3) ((by norm_num)) ((by norm_num)) (2) (5) := by
  norm_num [Real.exp_ne_zero]

example : veriphy_instance% KaluzaKlein.ClassicalMechanics.Momentum.drag_momentum_changes (-1) (1/10) ((by norm_num)) ((by norm_num)) (1) (-2) := by
  norm_num [Real.exp_ne_zero]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ClassicalMechanics.Momentum.drag_momentum_changes
-- test: cm-drag-momentum-changes-test.sage

/-! ## exr-clm-two-body -/

/-- `r̈ = ẍ₁ - ẍ₂ = f/m₁ + f/m₂ = f/μ` with `μ = m₁ m₂/(m₁ + m₂)`. -/
theorem relative_acceleration (m₁ m₂ f : ℝ) (h₁ : m₁ ≠ 0) (h₂ : m₂ ≠ 0) (h : m₁ + m₂ ≠ 0) :
    f / m₁ - (-f / m₂) = f / (m₁ * m₂ / (m₁ + m₂)) := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- `x₁ = X + (m₂/M) r` with `X` the centre of mass and `r = x₁ - x₂`. -/
theorem two_body_position_one (m₁ m₂ x₁ x₂ X r : ℝ) (h : m₁ + m₂ ≠ 0)
    (hX : X = (m₁ * x₁ + m₂ * x₂) / (m₁ + m₂)) (hr : r = x₁ - x₂) :
    x₁ = X + m₂ / (m₁ + m₂) * r := by
  rw [hX, hr]
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- `x₂ = X - (m₁/M) r`. -/
theorem two_body_position_two (m₁ m₂ x₁ x₂ X r : ℝ) (h : m₁ + m₂ ≠ 0)
    (hX : X = (m₁ * x₁ + m₂ * x₂) / (m₁ + m₂)) (hr : r = x₁ - x₂) :
    x₂ = X - m₁ / (m₁ + m₂) * r := by
  rw [hX, hr]
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

/-- The reduced mass is close to the lighter mass when the other is large: `m₂ - μ = m₂²/(m₁ + m₂)`. -/
theorem reduced_mass_defect (m₁ m₂ : ℝ) (h : m₁ + m₂ ≠ 0) :
    m₂ - m₁ * m₂ / (m₁ + m₂) = m₂ ^ 2 / (m₁ + m₂) := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: cm-poly.sage

end KaluzaKlein.ClassicalMechanics.Momentum

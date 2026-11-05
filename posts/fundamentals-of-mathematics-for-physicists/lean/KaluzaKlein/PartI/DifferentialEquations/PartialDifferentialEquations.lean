import Mathlib
import Veriphy.Kit.Levels

/-!
# Differential equations, part "Partial differential equations"

The claims of the part "Partial differential equations" of `tutorial/part-i-05-differential-equations.qmd`, one
namespace per section: linearity and superposition, the discriminant classification, d'Alembert's solution and
spherical waves, the heat kernel and the decaying modes, harmonic functions in Cartesian and polar coordinates,
separation of variables and the worked computations.

A function of several variables is `u : ℝ → ℝ → ℝ` (or explicit in its variables), and a partial derivative is the
derivative of a slice: `u_t(t, x) = deriv (fun s => u s x) t`, `u_xx(t, x) = deriv (fun y => deriv (fun z => u t z) y) x`.
Linearity of a field equation is the superposition identity of its jet expression; conservation and energy laws are
pointwise identities between the jet variables (`u_t`, `u_x`, `u_xx`, …) that the equation implies, integrated by the
fundamental theorem of calculus.
-/

open Real Filter Topology

namespace KaluzaKlein.DifferentialEquations

/-! Helper lemmas of the hand proofs below (private: each module carries its own copy). -/

private lemma deriv_eventually_pos {g g' : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (h : ∀ s, 0 < s → HasDerivAt g (g' s) s) : (fun s => deriv g s) =ᶠ[𝓝 r] g' := by
  filter_upwards [Ioi_mem_nhds hr] with s hs
  exact (h s hs).deriv

end KaluzaKlein.DifferentialEquations


namespace KaluzaKlein.DifferentialEquations.PartialDifferentialEquations

/-! ## def-pde -/

/-- The solutions of a homogeneous linear equation `Lu = 0` form a vector space: combinations of solutions are
solutions. -/
theorem pde_superposition {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W] (L : V →ₗ[ℝ] W)
    (u₁ u₂ : V) (α β : ℝ) (h₁ : L u₁ = 0) (h₂ : L u₂ = 0) : L (α • u₁ + β • u₂) = 0 := by
  simp [h₁, h₂]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.pde_superposition
-- test: de-pde-superposition-test.sage

/-- … and `u₁ + u₂` solves `Lu = f₁ + f₂` when `uᵢ` solves `Lu = fᵢ`. -/
theorem superposition_sources {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]
    (L : V →ₗ[ℝ] W) (u₁ u₂ : V) (f₁ f₂ : W) (h₁ : L u₁ = f₁) (h₂ : L u₂ = f₂) : L (u₁ + u₂) = f₁ + f₂ := by
  simp [h₁, h₂]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.superposition_sources
-- test: de-pde-sources-test.sage

/-! ## exm-pde-wave -/

/-- The wave (and Klein-Gordon) operator is linear: superposition in the jet variables `(u_tt, u_xx, u)`. -/
theorem wave_linear (c m a b p q r p' q' r' : ℝ) :
    (a * p + b * p') - c ^ 2 * (a * q + b * q') + m ^ 2 * (a * r + b * r') =
      a * (p - c ^ 2 * q + m ^ 2 * r) + b * (p' - c ^ 2 * q' + m ^ 2 * r') := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-pde-heat -/

/-- Not symmetric under `t → -t`: the reversed mode `e^{κk²t} sin kx` leaves the residual
`u_t - κu_xx = 2κk² e^{κk²t} sin kx ≠ 0`. -/
theorem heat_irreversible (κ k t x : ℝ) :
    deriv (fun s => Real.exp (κ * k ^ 2 * s) * Real.sin (k * x)) t -
      κ * deriv (fun y => deriv (fun z => Real.exp (κ * k ^ 2 * t) * Real.sin (k * z)) y) x =
      2 * κ * k ^ 2 * (Real.exp (κ * k ^ 2 * t) * Real.sin (k * x)) := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun s => rexp (κ * k ^ 2 * s) * sin (k * x))
          ((κ * (k ^ 2) * (Real.exp (κ * (k ^ 2) * t)) * (Real.sin (k * x)) : ℝ)) (t) :=
        by
        convert ((((hasDerivAt_id' (t : ℝ)).const_mul (κ * (k ^ 2) : ℝ)).exp).const_mul (Real.sin (k * x) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      try rw [h0.deriv]
      have i1 :
        deriv (fun z => rexp (κ * k ^ 2 * t) * sin (k * z)) = fun d1 =>
          k * (Real.cos (d1 * k)) * (Real.exp (κ * (k ^ 2) * t)) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun z => rexp (κ * k ^ 2 * t) * sin (k * z))
            ((k * (Real.cos (d1 * k)) * (Real.exp (κ * (k ^ 2) * t)) : ℝ)) (d1) :=
          by
          convert ((((hasDerivAt_id' (d1 : ℝ)).const_mul (k : ℝ)).sin).const_mul (Real.exp (κ * (k ^ 2) * t) : ℝ)) using
            1
          all_goals ((ring_nf; done))
        exact hd.deriv
      rw [i1]
      have h2 :
        HasDerivAt (fun y => (fun d1 => k * (Real.cos (d1 * k)) * (Real.exp (κ * (k ^ 2) * t))) y)
          ((-(k ^ 2) * (Real.exp (κ * (k ^ 2) * t)) * (Real.sin (k * x)) : ℝ)) (x) :=
        by
        convert
          ((((hasDerivAt_id' (x : ℝ)).const_mul (k : ℝ)).cos).const_mul (k * (Real.exp (κ * (k ^ 2) * t)) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      try rw [h2.deriv]
      beta_reduce
      ring1)
  -- certificate: de-partials.sage

/-! ## exm-pde-laplace -/

/-- The real part `x² - y²` of `z²` is harmonic … -/
theorem re_z_sq_harmonic (x y : ℝ) :
    deriv (fun s => deriv (fun r => r ^ 2 - y ^ 2) s) x + deriv (fun s => deriv (fun r => x ^ 2 - r ^ 2) s) y = 0 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => (r ^ 2 - y ^ 2 : ℝ)) = fun (d1 : ℝ) => (2 * d1 : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (r ^ 2 - y ^ 2 : ℝ)) ((2 * d1 : ℝ)) (d1) :=
          by
          refine
            (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-(y ^ 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun s => (fun (d1 : ℝ) => (2 * d1 : ℝ)) s) ((2 : ℝ)) (x) :=
        by
        refine
          ((((hasDerivAt_id' (x : ℝ)).const_mul (2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h1.deriv]
      have i2 : deriv (fun (r : ℝ) => (x ^ 2 - r ^ 2 : ℝ)) = fun (d1 : ℝ) => (-2 * d1 : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (x ^ 2 - r ^ 2 : ℝ)) ((-2 * d1 : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (-1 : ℝ)).add
                      (hasDerivAt_const (d1 : ℝ) (x ^ 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i2]
      have h3 : HasDerivAt (fun s => (fun (d1 : ℝ) => (-2 * d1 : ℝ)) s) ((-2 : ℝ)) (y) :=
        by
        refine
          ((((hasDerivAt_id' (y : ℝ)).const_mul (-2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h3.deriv]
      try beta_reduce
      all_goals
        (first
          | ring1
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
          | (norm_num; done)
          |
            (ext i j;
              fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
                ring_nf)))
  -- certificate: de-partials.sage

/-- … and so is the imaginary part `2xy`. -/
theorem im_z_sq_harmonic (x y : ℝ) :
    deriv (fun s => deriv (fun r => 2 * r * y) s) x + deriv (fun s => deriv (fun r => 2 * x * r) s) y = 0 := by
  veriphy_certified
    ( have i0 : deriv (fun r => 2 * r * y) = fun d1 => 2 * y :=
        by
        funext d1
        have hd : HasDerivAt (fun r => 2 * r * y) ((2 * y : ℝ)) (d1) :=
          by
          convert ((hasDerivAt_id' (d1 : ℝ)).const_mul (2 * y : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun s => (fun d1 => 2 * y) s) ((0 : ℝ)) (x) :=
        by
        convert (hasDerivAt_const (x : ℝ) (2 * y : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 : deriv (fun r => 2 * x * r) = fun d1 => 2 * x :=
        by
        funext d1
        have hd : HasDerivAt (fun r => 2 * x * r) ((2 * x : ℝ)) (d1) :=
          by
          convert ((hasDerivAt_id' (d1 : ℝ)).const_mul (2 * x : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 : HasDerivAt (fun s => (fun d1 => 2 * x) s) ((0 : ℝ)) (y) :=
        by
        convert (hasDerivAt_const (y : ℝ) (2 * x : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

/-! ## exr-pde-classify -/

/-- Poisson's equation `u_xx + u_yy = eˣ` is linear (inhomogeneous): superposition of the left side. -/
theorem poisson_linear (a b p q p' q' : ℝ) : (a * p + b * p') + (a * q + b * q') = a * (p + q) + b * (p' + q') := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `u_t = u_xxxx` is linear and homogeneous: superposition of `u_t - u_xxxx`. -/
theorem fourth_order_linear (a b p q p' q' : ℝ) : (a * p + b * p') - (a * q + b * q') = a * (p - q) + b * (p' - q') := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exr-pde-check-solution -/

/-- `u = e^{-t} sin x` solves `u_t = u_xx`. -/
theorem heat_check (t x : ℝ) :
    deriv (fun s => Real.exp (-s) * Real.sin x) t = deriv (fun y => deriv (fun z => Real.exp (-t) * Real.sin z) y) x := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => rexp (-s) * sin x) ((-(Real.exp (-t)) * (Real.sin x) : ℝ)) (t) :=
        by
        refine
          ((((((hasDerivAt_id' (t : ℝ)).const_mul (-1 : ℝ)).exp).const_mul (Real.sin x : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · (ring_nf; done)
      try rw [h0.deriv]
      have i1 :
        deriv (fun (z : ℝ) => (rexp (-t) * sin z : ℝ)) = fun (d1 : ℝ) => ((Real.cos d1) * (Real.exp (-t)) : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (z : ℝ) => (rexp (-t) * sin z : ℝ)) (((Real.cos d1) * (Real.exp (-t)) : ℝ)) (d1) :=
          by
          refine
            (((((hasDerivAt_id' (d1 : ℝ)).sin).const_mul (Real.exp (-t) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      rw [i1]
      have h2 :
        HasDerivAt (fun y => (fun (d1 : ℝ) => ((Real.cos d1) * (Real.exp (-t)) : ℝ)) y)
          ((-(Real.exp (-t)) * (Real.sin x) : ℝ)) (x) :=
        by
        refine
          (((((hasDerivAt_id' (x : ℝ)).cos).const_mul (Real.exp (-t) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- `u = cos(x - 2t)` solves `u_tt - 4u_xx = 0`. -/
theorem wave_check (t x : ℝ) :
    deriv (fun s => deriv (fun r => Real.cos (x - 2 * r)) s) t -
      4 * deriv (fun s => deriv (fun r => Real.cos (r - 2 * t)) s) x = 0 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => (cos (x - 2 * r) : ℝ)) = fun (d1 : ℝ) => (2 * (Real.sin (-2 * d1 + x)) : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (cos (x - 2 * r) : ℝ)) ((2 * (Real.sin (-2 * d1 + x)) : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (-2 : ℝ)).add
                        (hasDerivAt_const (d1 : ℝ) (x : ℝ))).cos)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun (d1 : ℝ) => (2 * (Real.sin (-2 * d1 + x)) : ℝ)) s)
          ((-4 * (Real.cos (x - (2 * t))) : ℝ)) (t) :=
        by
        refine
          (((((((hasDerivAt_id' (t : ℝ)).const_mul (-2 : ℝ)).add (hasDerivAt_const (t : ℝ) (x : ℝ))).sin).const_mul
                    (2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h1.deriv]
      have i2 : deriv (fun (r : ℝ) => (cos (r - 2 * t) : ℝ)) = fun (d1 : ℝ) => (-(Real.sin (d1 - (2 * t))) : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (cos (r - 2 * t) : ℝ)) ((-(Real.sin (d1 - (2 * t))) : ℝ)) (d1) :=
          by
          refine
            (((((hasDerivAt_id' (d1 : ℝ)).add (hasDerivAt_const (d1 : ℝ) (-2 * t : ℝ))).cos)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt (fun s => (fun (d1 : ℝ) => (-(Real.sin (d1 - (2 * t))) : ℝ)) s) ((-(Real.cos (x - (2 * t))) : ℝ))
          (x) :=
        by
        refine
          ((((((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (-2 * t : ℝ))).sin).const_mul
                    (-1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h3.deriv]
      try beta_reduce
      all_goals
        (first
          | ring1
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
          | (norm_num; done)
          |
            (ext i j;
              fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
                ring_nf)))
  -- certificate: de-partials.sage

/-- `u = arctan(y/x)` (the polar angle) is harmonic for `x ≠ 0`. -/
theorem arctan_harmonic (x y : ℝ) (hx : x ≠ 0) :
    deriv (fun s => deriv (fun r => Real.arctan (y / r)) s) x +
      deriv (fun s => deriv (fun r => Real.arctan (r / x)) s) y = 0 := by
  have hux : ∀ r, r ≠ 0 → HasDerivAt (fun r => Real.arctan (y / r)) (-y / (r ^ 2 + y ^ 2)) r := by
    intro r hr
    have h1 : HasDerivAt (fun r => y / r) (-y / r ^ 2) r := by
      simpa [div_eq_mul_inv] using (hasDerivAt_inv hr).const_mul y
    convert h1.arctan using 1
    first | (field_simp; done) | (field_simp; ring)
  have huy : ∀ r, HasDerivAt (fun r => Real.arctan (r / x)) (x / (x ^ 2 + r ^ 2)) r := by
    intro r
    have h1 : HasDerivAt (fun r => r / x) (1 / x) r := by simpa using (hasDerivAt_id' r).div_const x
    convert h1.arctan using 1
    first | (field_simp; done) | (field_simp; ring)
  have ex : (fun s => deriv (fun r => Real.arctan (y / r)) s) =ᶠ[𝓝 x] fun s => -y / (s ^ 2 + y ^ 2) := by
    filter_upwards [isOpen_ne.mem_nhds hx] with s hs
    exact (hux s hs).deriv
  rw [ex.deriv_eq, show (fun s => deriv (fun r => Real.arctan (r / x)) s) = fun s => x / (x ^ 2 + s ^ 2) from
    funext fun s => (huy s).deriv]
  have hp : 0 < x ^ 2 + y ^ 2 := by positivity
  have d1 : HasDerivAt (fun s => -y / (s ^ 2 + y ^ 2)) (2 * x * y / (x ^ 2 + y ^ 2) ^ 2) x := by
    have := (hasDerivAt_const x (-y)).div ((hasDerivAt_pow 2 x).add_const (y ^ 2)) hp.ne'
    convert this using 1; field_simp; ring
  have hp' : 0 < x ^ 2 + y ^ 2 := hp
  have d2 : HasDerivAt (fun s => x / (x ^ 2 + s ^ 2)) (-2 * x * y / (x ^ 2 + y ^ 2) ^ 2) y := by
    have := (hasDerivAt_const y x).div (((hasDerivAt_pow 2 y).const_add (x ^ 2))) (by positivity)
    convert this using 1; field_simp; ring
  rw [d1.deriv, d2.deriv]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.arctan_harmonic
-- test: de-arctan-harmonic-test.sage

/-! ## exr-pde-superposition -/

/-- `u = 1` solves Burgers' equation `u_t + u u_x = 0`. -/
theorem burgers_one (t x : ℝ) : deriv (fun _ : ℝ => (1 : ℝ)) t + 1 * deriv (fun _ : ℝ => (1 : ℝ)) x = 0 := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun x => 1) ((0 : ℝ)) (t) :=
        by
        convert (hasDerivAt_const (t : ℝ) (1 : ℝ)) using 1
        all_goals rfl
      rw [h0.deriv]
      have h1 : HasDerivAt (fun x => 1) ((0 : ℝ)) (x) :=
        by
        convert (hasDerivAt_const (x : ℝ) (1 : ℝ)) using 1
        all_goals rfl
      rw [h1.deriv]
      ring1)
  -- certificate: de-partials.sage

/-- `u = x/(1 + t)` solves it (for `t ≠ -1`). -/
theorem burgers_ratio (t x : ℝ) (ht : t + 1 ≠ 0) :
    deriv (fun s => x / (1 + s)) t + x / (1 + t) * deriv (fun y => y / (1 + t)) x = 0 := by
  have ht' : 1 + t ≠ 0 := by rwa [add_comm]
  have h1 : HasDerivAt (fun s => x / (1 + s)) ((0 * (1 + t) - x * 1) / (1 + t) ^ 2) t :=
    (hasDerivAt_const t x).div ((hasDerivAt_id' t).const_add 1) ht'
  have h2 : HasDerivAt (fun y => y / (1 + t)) (1 / (1 + t)) x := (hasDerivAt_id' x).div_const (1 + t)
  rw [h1.deriv, h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.burgers_ratio
-- test: de-burgers-ratio-test.sage

/-- … but their sum leaves the residual `1/(1 + t) ≠ 0`. -/
theorem burgers_sum (t x : ℝ) (ht : t + 1 ≠ 0) :
    deriv (fun s => 1 + x / (1 + s)) t + (1 + x / (1 + t)) * deriv (fun y => 1 + y / (1 + t)) x = 1 / (1 + t) := by
  have ht' : 1 + t ≠ 0 := by rwa [add_comm]
  have h1 : HasDerivAt (fun s => 1 + x / (1 + s)) ((0 * (1 + t) - x * 1) / (1 + t) ^ 2) t :=
    ((hasDerivAt_const t x).div ((hasDerivAt_id' t).const_add 1) ht').const_add 1
  have h2 : HasDerivAt (fun y => 1 + y / (1 + t)) (1 / (1 + t)) x := ((hasDerivAt_id' x).div_const (1 + t)).const_add 1
  rw [h1.deriv, h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.burgers_sum
-- test: de-burgers-sum-test.sage

end KaluzaKlein.DifferentialEquations.PartialDifferentialEquations

namespace KaluzaKlein.DifferentialEquations.Classification

/-! ## def-pde-classification -/

/-- The discriminant is minus the determinant of the coefficient matrix: `det (A, B; B, C) = -(B² - AC)`, so the type
is read off the signs of the eigenvalues (whose product is the determinant). -/
theorem discriminant_det (A B C : ℝ) : Matrix.det !![A, B; B, C] = -(B ^ 2 - A * C) := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-! ## exm-class-wave -/

/-- The wave equation: `B² - AC = c²`. -/
theorem wave_discriminant (c : ℝ) : (0 : ℝ) ^ 2 - 1 * (-c ^ 2) = c ^ 2 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- Its coefficient matrix `diag(1, -c²)` has determinant `-c² < 0`: one positive and one negative eigenvalue. -/
theorem wave_matrix_det (c : ℝ) : Matrix.det !![(1 : ℝ), 0; 0, -c ^ 2] = -c ^ 2 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-! ## exm-class-laplace -/

/-- Laplace's equation: `B² - AC = -1`. -/
theorem laplace_discriminant : (0 : ℝ) ^ 2 - 1 * 1 = -1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-class-heat -/

/-- The heat equation: `B² - AC = 0`. -/
theorem heat_discriminant (κ : ℝ) : (0 : ℝ) ^ 2 - (-κ) * 0 = 0 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- … with the degenerate matrix `diag(-κ, 0)`. -/
theorem heat_matrix_det (κ : ℝ) : Matrix.det !![-κ, 0; 0, (0 : ℝ)] = 0 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-! ## exr-class-compute -/

/-- `u_xx + 4u_xy + u_yy`: `B² - AC = 3` (hyperbolic) … -/
theorem compute_one : (2 : ℝ) ^ 2 - 1 * 1 = 3 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage
/-- … eigenvalues `3, -1`. -/
theorem compute_one_eigen (l : ℝ) : Matrix.det !![l - 1, -2; -2, l - 1] = (l - 3) * (l + 1) := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage
/-- `u_xx + 2u_xy + u_yy`: `0` (parabolic) … -/
theorem compute_two : (1 : ℝ) ^ 2 - 1 * 1 = 0 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage
/-- … eigenvalues `0, 2`. -/
theorem compute_two_eigen (l : ℝ) : Matrix.det !![l - 1, -1; -1, l - 1] = l * (l - 2) := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage
/-- `u_xx + 2u_xy + 3u_yy`: `-2` (elliptic) … -/
theorem compute_three : (1 : ℝ) ^ 2 - 1 * 3 = -2 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage
/-- … eigenvalues `2 ± √2` (both positive): `det(λ - M) = λ² - 4λ + 2`. -/
theorem compute_three_eigen (l : ℝ) : Matrix.det !![l - 1, -1; -1, l - 3] = l ^ 2 - 4 * l + 2 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage
/-- Tricomi's equation `yu_xx + u_yy`: `B² - AC = -y`. -/
theorem compute_tricomi (y : ℝ) : (0 : ℝ) ^ 2 - y * 1 = -y := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exr-class-invariance -/

/-- Under a change of variables `M ↦ JMJᵀ`: `det(JMJᵀ) = (det J)² det M`, so the sign of `B² - AC` is unchanged
when `det J ≠ 0`. -/
theorem det_congruence (a b c d A B C : ℝ) :
    Matrix.det (!![a, b; c, d] * !![A, B; B, C] * !![a, c; b, d]) =
      Matrix.det !![a, b; c, d] ^ 2 * Matrix.det !![A, B; B, C] := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-- The sign is preserved: `(det J)² det M` has the sign of `det M` for `det J ≠ 0`. -/
theorem sign_preserved (dJ dM : ℝ) (hJ : dJ ≠ 0) : (0 < dJ ^ 2 * dM ↔ 0 < dM) ∧ (dJ ^ 2 * dM < 0 ↔ dM < 0) := by
  have h : 0 < dJ ^ 2 := by positivity
  exact ⟨⟨fun h' => pos_of_mul_pos_right h' h.le, fun h' => mul_pos h h'⟩,
    ⟨fun h' => by nlinarith, fun h' => mul_neg_of_pos_of_neg h h'⟩⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.Classification.sign_preserved
-- test: de-sign-preserved-test.sage

/-! ## exr-class-characteristics -/

/-- In `ξ = x - ct`, `η = x + ct`: `∂_t = c(∂_η - ∂_ξ)`, `∂_x = ∂_ξ + ∂_η`, and as quadratic forms in the symbols
`(∂_ξ, ∂_η) = (p, q)`, `∂_t² - c²∂_x² = -4c²∂_ξ∂_η`. -/
theorem characteristic_operator (c p q : ℝ) : (c * (q - p)) ^ 2 - c ^ 2 * (p + q) ^ 2 = -4 * c ^ 2 * p * q := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

end KaluzaKlein.DifferentialEquations.Classification

namespace KaluzaKlein.DifferentialEquations.WaveEquation

/-! ## def-wave-equation (with prp-pde-solutions) -/

/-- A travelling profile `F(x + at)`: its slices have `∂_t = aF'`, `∂_t² = a²F''`, `∂_x = F'`, `∂_x² = F''`. -/
theorem travelling_derivs (F F' F'' : ℝ → ℝ) (hF : ∀ y, HasDerivAt F (F' y) y) (hF' : ∀ y, HasDerivAt F' (F'' y) y)
    (a t x : ℝ) :
    deriv (fun s => deriv (fun r => F (x + a * r)) s) t = a ^ 2 * F'' (x + a * t) ∧
      deriv (fun s => deriv (fun r => F (r + a * t)) s) x = F'' (x + a * t) := by
  have h1 : (fun s => deriv (fun r => F (x + a * r)) s) = fun s => a * F' (x + a * s) := by
    funext s
    have h : HasDerivAt (fun r => F (x + a * r)) (a * F' (x + a * s)) s := by
      have := (hF (x + a * s)).comp s (((hasDerivAt_id' s).const_mul a).const_add x)
      convert this using 1 <;> first | rfl | ring | (simp; ring) | (funext s; simp; ring) | simp
    exact h.deriv
  have h2 : (fun s => deriv (fun r => F (r + a * t)) s) = fun s => F' (s + a * t) := by
    funext s
    have h : HasDerivAt (fun r => F (r + a * t)) (F' (s + a * t)) s := by
      have := (hF (s + a * t)).comp s ((hasDerivAt_id' s).add_const (a * t))
      convert this using 1 <;> first | rfl | ring | (simp; ring) | (funext s; simp; ring) | simp
    exact h.deriv
  refine ⟨?_, ?_⟩
  · rw [h1]
    have h : HasDerivAt (fun s => a * F' (x + a * s)) (a ^ 2 * F'' (x + a * t)) t := by
      have := ((hF' (x + a * t)).comp t (((hasDerivAt_id' t).const_mul a).const_add x)).const_mul a
      convert this using 1 <;> first | rfl | ring | (simp; ring) | (funext s; simp; ring) | simp
    exact h.deriv
  · rw [h2]
    have h : HasDerivAt (fun s => F' (s + a * t)) (F'' (x + a * t)) x := by
      have := (hF' (x + a * t)).comp x ((hasDerivAt_id' x).add_const (a * t))
      convert this using 1 <;> first | rfl | ring | (simp; ring) | (funext s; simp; ring) | simp
    exact h.deriv
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.travelling_derivs
-- test: de-travelling-test.sage

/-- `u = f(x - ct) + g(x + ct)` solves `u_tt - c²u_xx = 0` for all twice differentiable `f`, `g`. -/
theorem dalembert_general (f f' f'' g g' g'' : ℝ → ℝ) (hf : ∀ y, HasDerivAt f (f' y) y)
    (hf' : ∀ y, HasDerivAt f' (f'' y) y) (hg : ∀ y, HasDerivAt g (g' y) y) (hg' : ∀ y, HasDerivAt g' (g'' y) y)
    (c t x : ℝ) :
    (c ^ 2 * f'' (x + -c * t) + c ^ 2 * g'' (x + c * t)) - c ^ 2 * (f'' (x + -c * t) + g'' (x + c * t)) = 0 ∧
      deriv (fun s => deriv (fun r => f (x + -c * r)) s) t = (-c) ^ 2 * f'' (x + -c * t) ∧
      deriv (fun s => deriv (fun r => g (x + c * r)) s) t = c ^ 2 * g'' (x + c * t) := by
  refine ⟨by ring, (travelling_derivs f f' f'' hf hf' (-c) t x).1, (travelling_derivs g g' g'' hg hg' c t x).1⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.dalembert_general
-- test: de-dalembert-general-test.sage

/-- d'Alembert's formula meets the initial data: with `Ψ' = ψ`,
`u = ½[φ(x - ct) + φ(x + ct)] + (1/2c)[Ψ(x + ct) - Ψ(x - ct)]` has `u(0, x) = φ(x)` and `u_t(0, x) = ψ(x)`. -/
theorem dalembert_initial (φ φ' Ψ ψ : ℝ → ℝ) (hφ : ∀ y, HasDerivAt φ (φ' y) y) (hΨ : ∀ y, HasDerivAt Ψ (ψ y) y)
    (c x : ℝ) (hc : c ≠ 0) :
    (φ (x - c * 0) + φ (x + c * 0)) / 2 + (Ψ (x + c * 0) - Ψ (x - c * 0)) / (2 * c) = φ x ∧
      HasDerivAt (fun t => (φ (x - c * t) + φ (x + c * t)) / 2 + (Ψ (x + c * t) - Ψ (x - c * t)) / (2 * c))
        (ψ x) 0 := by
  constructor
  · simp
  · have h1 := (hφ (x - c * 0)).comp 0 (((hasDerivAt_id' (0 : ℝ)).const_mul c).const_sub x)
    have h2 := (hφ (x + c * 0)).comp 0 (((hasDerivAt_id' (0 : ℝ)).const_mul c).const_add x)
    have h3 := (hΨ (x + c * 0)).comp 0 (((hasDerivAt_id' (0 : ℝ)).const_mul c).const_add x)
    have h4 := (hΨ (x - c * 0)).comp 0 (((hasDerivAt_id' (0 : ℝ)).const_mul c).const_sub x)
    convert ((h1.add h2).div_const 2).add ((h3.sub h4).div_const (2 * c)) using 1 <;>
      first | rfl | (simp; field_simp; ring) | (field_simp; ring) | (simp; done) | (funext t; simp)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.dalembert_initial
-- test: de-dalembert-initial-test.sage

/-! ## exm-wave-split -/

/-- The bump released from rest: `u = ½[e^{-(x-ct)²} + e^{-(x+ct)²}]` solves the wave equation … -/
theorem bump_solves (c t x : ℝ) :
    deriv (fun s => deriv (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2) s) t -
      c ^ 2 * deriv (fun s => deriv (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2) s) x
      = 0 := by
  veriphy_certified
    ( have i0 :
        deriv (fun (r : ℝ) => ((rexp (-(x - c * r) ^ 2) + rexp (-(x + c * r) ^ 2)) / 2 : ℝ)) = fun (d1 : ℝ) =>
          (-(d1 * c + x) * c * (Real.exp (-((d1 * c + x) ^ 2))) -
              ((d1 * c - x) * c * (Real.exp (-((d1 * c - x) ^ 2)))) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => ((rexp (-(x - c * r) ^ 2) + rexp (-(x + c * r) ^ 2)) / 2 : ℝ))
            ((-(d1 * c + x) * c * (Real.exp (-((d1 * c + x) ^ 2))) -
                ((d1 * c - x) * c * (Real.exp (-((d1 * c - x) ^ 2)))) :
              ℝ))
            (d1) :=
          by
          refine
            ((((((((((hasDerivAt_id' (d1 : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x : ℝ))).pow
                                    2).const_mul
                                (-1 : ℝ)).exp).const_mul
                          (1 / 2 : ℝ)).add
                      (((((((hasDerivAt_id' (d1 : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (d1 : ℝ) (-x : ℝ))).pow
                                  2).const_mul
                              (-1 : ℝ)).exp).const_mul
                        (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (-(d1 * c + x) * c * (Real.exp (-((d1 * c + x) ^ 2))) -
                    ((d1 * c - x) * c * (Real.exp (-((d1 * c - x) ^ 2)))) :
                  ℝ))
              s)
          ((2 * ((c * t + x) ^ 2) * (c ^ 2) * (Real.exp (-((c * t + x) ^ 2))) +
                  (2 * ((c * t - x) ^ 2) * (c ^ 2) * (Real.exp (-((c * t - x) ^ 2)))) -
                ((c ^ 2) * (Real.exp (-((c * t + x) ^ 2)))) -
              ((c ^ 2) * (Real.exp (-((c * t - x) ^ 2)))) :
            ℝ))
          (t) :=
        by
        refine
          ((((((((hasDerivAt_id' (t : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (t : ℝ) (x : ℝ))).mul
                            ((((((hasDerivAt_id' (t : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (t : ℝ) (x : ℝ))).pow
                                    2).const_mul
                                (-1 : ℝ)).exp)).const_mul
                        (-c : ℝ)).add
                    (((((hasDerivAt_id' (t : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (t : ℝ) (-x : ℝ))).mul
                          ((((((hasDerivAt_id' (t : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (t : ℝ) (-x : ℝ))).pow
                                  2).const_mul
                              (-1 : ℝ)).exp)).const_mul
                      (-c : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h1.deriv]
      have i2 :
        deriv (fun (r : ℝ) => ((rexp (-(r - c * t) ^ 2) + rexp (-(r + c * t) ^ 2)) / 2 : ℝ)) = fun (d1 : ℝ) =>
          (-(c * t + d1) * (Real.exp (-((c * t + d1) ^ 2))) + ((c * t - d1) * (Real.exp (-((c * t - d1) ^ 2)))) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => ((rexp (-(r - c * t) ^ 2) + rexp (-(r + c * t) ^ 2)) / 2 : ℝ))
            ((-(c * t + d1) * (Real.exp (-((c * t + d1) ^ 2))) + ((c * t - d1) * (Real.exp (-((c * t - d1) ^ 2)))) : ℝ))
            (d1) :=
          by
          refine
            (((((((((hasDerivAt_const (d1 : ℝ) (c * t : ℝ)).add (hasDerivAt_id' (d1 : ℝ))).pow 2).const_mul
                                (-1 : ℝ)).exp).const_mul
                          (1 / 2 : ℝ)).add
                      ((((((hasDerivAt_const (d1 : ℝ) (c * t : ℝ)).add
                                      ((hasDerivAt_id' (d1 : ℝ)).const_mul (-1 : ℝ))).pow
                                  2).const_mul
                              (-1 : ℝ)).exp).const_mul
                        (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce;
            first
            | rfl
            |
              (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                  Pi.const_apply, Pi.smul_apply, smul_eq_mul];
                ring_nf; done)
            | (ring_nf; done)
            | (simp; ring_nf; done)
            | (simp; done)
            | (field_simp; ring_nf; done)
          ·
            first
            | rfl
            | ring1
            | (ring_nf; done)
            | (field_simp; done)
            | (field_simp; ring1)
            | (field_simp; ring_nf; done)
            | (simp; ring_nf; done)
            | (norm_num; done)
            |
              (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                  Real.exp_neg];
                field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
            | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (-(c * t + d1) * (Real.exp (-((c * t + d1) ^ 2))) + ((c * t - d1) * (Real.exp (-((c * t - d1) ^ 2)))) :
                  ℝ))
              s)
          ((2 * ((c * t + x) ^ 2) * (Real.exp (-((c * t + x) ^ 2))) +
                  (2 * ((c * t - x) ^ 2) * (Real.exp (-((c * t - x) ^ 2)))) -
                (Real.exp (-((c * t + x) ^ 2))) -
              (Real.exp (-((c * t - x) ^ 2))) :
            ℝ))
          (x) :=
        by
        refine
          (((((((hasDerivAt_const (x : ℝ) (c * t : ℝ)).add (hasDerivAt_id' (x : ℝ))).mul
                            (((((hasDerivAt_const (x : ℝ) (c * t : ℝ)).add (hasDerivAt_id' (x : ℝ))).pow 2).const_mul
                                (-1 : ℝ)).exp)).const_mul
                        (-1 : ℝ)).add
                    (((hasDerivAt_const (x : ℝ) (c * t : ℝ)).add ((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ))).mul
                      (((((hasDerivAt_const (x : ℝ) (c * t : ℝ)).add ((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ))).pow
                              2).const_mul
                          (-1 : ℝ)).exp)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce;
          first
          | rfl
          |
            (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply, Pi.div_apply,
                Pi.const_apply, Pi.smul_apply, smul_eq_mul];
              ring_nf; done)
          | (ring_nf; done)
          | (simp; ring_nf; done)
          | (simp; done)
          | (field_simp; ring_nf; done)
        ·
          first
          | rfl
          | ring1
          | (ring_nf; done)
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (simp; ring_nf; done)
          | (norm_num; done)
          |
            (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply,
                Real.exp_neg];
              field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      try rw [h3.deriv]
      try beta_reduce
      all_goals
        (first
          | ring1
          | (field_simp; done)
          | (field_simp; ring1)
          | (field_simp; ring_nf; done)
          | (ring_nf; done)
          | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
          | (norm_num; done)
          |
            (ext i j;
              fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
                ring_nf)))
  -- certificate: de-partials.sage

/-- … with `u(0, x) = e^{-x²}` … -/
theorem bump_initial (c x : ℝ) : (Real.exp (-(x - c * 0) ^ 2) + Real.exp (-(x + c * 0) ^ 2)) / 2 = Real.exp (-x ^ 2) := by
  veriphy_certified (simp)
  -- certificate: de-rewrite.sage

/-- … and `u_t(0, x) = 0`. -/
theorem bump_at_rest (c x : ℝ) :
    deriv (fun s => (Real.exp (-(x - c * s) ^ 2) + Real.exp (-(x + c * s) ^ 2)) / 2) 0 = 0 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => (rexp (-(x - c * s) ^ 2) + rexp (-(x + c * s) ^ 2)) / 2) ((0 : ℝ)) (0) :=
        by
        refine
          ((((((((((hasDerivAt_id' (0 : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (0 : ℝ) (x : ℝ))).pow 2).const_mul
                              (-1 : ℝ)).exp).const_mul
                        (1 / 2 : ℝ)).add
                    (((((((hasDerivAt_id' (0 : ℝ)).const_mul (c : ℝ)).add (hasDerivAt_const (0 : ℝ) (-x : ℝ))).pow
                                2).const_mul
                            (-1 : ℝ)).exp).const_mul
                      (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (norm_num; done)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-! ## exm-wave-velocity -/

/-- `(1/2c)[sin(x + ct) - sin(x - ct)] = (1/c) cos x sin ct`: a standing wave, the superposition of a left- and a
right-moving sine. -/
theorem velocity_standing (c t x : ℝ) (hc : c ≠ 0) :
    (Real.sin (x + c * t) - Real.sin (x - c * t)) / (2 * c) = Real.cos x * Real.sin (c * t) / c := by
  veriphy_certified
    ( rw [Real.sin_add, Real.sin_sub]
      field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- It solves the wave equation … -/
theorem velocity_solves (c t x : ℝ) (hc : c ≠ 0) :
    deriv (fun s => deriv (fun r => Real.cos x * Real.sin (c * r) / c) s) t -
      c ^ 2 * deriv (fun s => deriv (fun r => Real.cos r * Real.sin (c * t) / c) s) x = 0 := by
  veriphy_certified
    ( have i0 : deriv (fun r => cos x * sin (c * r) / c) = fun d1 => (Real.cos (d1 * c)) * (Real.cos x) :=
        by
        funext d1
        have hd : HasDerivAt (fun r => cos x * sin (c * r) / c) (((Real.cos (d1 * c)) * (Real.cos x) : ℝ)) (d1) :=
          by
          convert ((((hasDerivAt_id' (d1 : ℝ)).const_mul (c : ℝ)).sin).const_mul ((Real.cos x) / c : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun d1 => (Real.cos (d1 * c)) * (Real.cos x)) s)
          ((-c * (Real.cos x) * (Real.sin (c * t)) : ℝ)) (t) :=
        by
        convert ((((hasDerivAt_id' (t : ℝ)).const_mul (c : ℝ)).cos).const_mul (Real.cos x : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 : deriv (fun r => cos r * sin (c * t) / c) = fun d1 => -(Real.sin (c * t)) * (Real.sin d1) / c :=
        by
        funext d1
        have hd : HasDerivAt (fun r => cos r * sin (c * t) / c) ((-(Real.sin (c * t)) * (Real.sin d1) / c : ℝ)) (d1) :=
          by
          convert (((hasDerivAt_id' (d1 : ℝ)).cos).const_mul ((Real.sin (c * t)) / c : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt (fun s => (fun d1 => -(Real.sin (c * t)) * (Real.sin d1) / c) s)
          ((-(Real.cos x) * (Real.sin (c * t)) / c : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).sin).const_mul (-(Real.sin (c * t)) / c : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

/-- … with initial velocity `cos x`. -/
theorem velocity_initial (c x : ℝ) (hc : c ≠ 0) :
    deriv (fun s => Real.cos x * Real.sin (c * s) / c) 0 = Real.cos x := by
  have h : HasDerivAt (fun s => Real.cos x * Real.sin (c * s) / c) _ 0 :=
    (((hasDerivAt_id' 0).const_mul c).sin.const_mul (Real.cos x)).div_const c
  rw [h.deriv, mul_zero, Real.cos_zero]
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.velocity_initial
-- test: de-velocity-initial-test.sage

/-! ## exm-wave-spherical and exr-wave-spherical (radial form `∇²u = u_rr + (2/r)u_r`) -/

/-- `u = f(r - t)/r` solves `u_tt - (u_rr + (2/r)u_r) = 0` for `r > 0` and every twice differentiable `f`:
with `v = ru = f(r - t)`, `r∇²(v/r) = v_rr`. -/
theorem spherical_wave (f f' f'' : ℝ → ℝ) (r t : ℝ) (hr : r ≠ 0) :
    f'' (r - t) / r - ((f'' (r - t) / r - 2 * f' (r - t) / r ^ 2 + 2 * f (r - t) / r ^ 3) +
      2 / r * (f' (r - t) / r - f (r - t) / r ^ 2)) = 0 := by
  veriphy_certified ((field_simp; ring))
  -- certificate: de-rewrite.sage

/-- The radial derivatives of `f(r - t)/r` (first and second) that enter the identity above. -/
theorem spherical_wave_derivs (f f' f'' : ℝ → ℝ) (hf : ∀ y, HasDerivAt f (f' y) y)
    (hf' : ∀ y, HasDerivAt f' (f'' y) y) (r t : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun s => f (s - t) / s) (f' (r - t) / r - f (r - t) / r ^ 2) r ∧
      HasDerivAt (fun s => f' (s - t) / s - f (s - t) / s ^ 2)
        (f'' (r - t) / r - 2 * f' (r - t) / r ^ 2 + 2 * f (r - t) / r ^ 3) r ∧
      HasDerivAt (fun s => f (r - s) / r) (-f' (r - t) / r) t ∧
      HasDerivAt (fun s => -f' (r - s) / r) (f'' (r - t) / r) t := by
  have a1 : HasDerivAt (fun s => f (s - t)) (f' (r - t)) r := by
    simpa [Function.comp_def] using (hf (r - t)).comp r ((hasDerivAt_id' r).sub_const t)
  have a2 : HasDerivAt (fun s => f' (s - t)) (f'' (r - t)) r := by
    simpa [Function.comp_def] using (hf' (r - t)).comp r ((hasDerivAt_id' r).sub_const t)
  have b1 : HasDerivAt (fun s => f (r - s)) (-f' (r - t)) t := by
    simpa [Function.comp_def] using (hf (r - t)).comp t ((hasDerivAt_id' t).const_sub r)
  have b2 : HasDerivAt (fun s => f' (r - s)) (-f'' (r - t)) t := by
    simpa [Function.comp_def] using (hf' (r - t)).comp t ((hasDerivAt_id' t).const_sub r)
  have hp : HasDerivAt (fun s : ℝ => s ^ 2) (2 * r) r := by simpa using hasDerivAt_pow 2 r
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h : HasDerivAt (fun s => f (s - t) / s) ((f' (r - t) * r - f (r - t) * 1) / r ^ 2) r :=
      a1.div (hasDerivAt_id' r) hr
    exact h.congr_deriv (by first | (field_simp; done) | (field_simp; ring))
  · have h : HasDerivAt (fun s => f' (s - t) / s - f (s - t) / s ^ 2)
        ((f'' (r - t) * r - f' (r - t) * 1) / r ^ 2 - (f' (r - t) * r ^ 2 - f (r - t) * (2 * r)) / (r ^ 2) ^ 2) r :=
      (a2.div (hasDerivAt_id' r) hr).sub (a1.div hp (pow_ne_zero 2 hr))
    exact h.congr_deriv (by first | (field_simp; done) | (field_simp; ring))
  · exact b1.div_const r
  · have h : HasDerivAt (fun s => -f' (r - s) / r) (- -f'' (r - t) / r) t := b2.neg.div_const r
    exact h.congr_deriv (by first | (simp; done) | ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.spherical_wave_derivs
-- test: de-spherical-derivs-test.sage

/-- Without the `1/r`, `u = f(r - t)` leaves the residual `u_tt - (u_rr + (2/r)u_r) = -2f'(r - t)/r`. -/
theorem spherical_residual (f' f'' : ℝ → ℝ) (r t : ℝ) (hr : r ≠ 0) :
    f'' (r - t) - (f'' (r - t) + 2 / r * f' (r - t)) = -2 * f' (r - t) / r := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- The energy flux through spheres, proportional to `r²u² = f(r - t)²`, does not depend on `r` along the pulse. -/
theorem spherical_flux (F r : ℝ) (hr : r ≠ 0) : r ^ 2 * (F / r) ^ 2 = F ^ 2 := by
  veriphy_certified (field_simp)
  -- certificate: de-rewrite.sage

/-! ## exr-wave-energy -/

/-- The energy density `e = ½(u_t² + c²u_x²)` and flux `c²u_xu_t` obey `∂_te = ∂_x(c²u_xu_t)` on solutions
(`u_tt = c²u_xx`, `u_xt = u_tx`). -/
theorem string_energy_flux (c ut ux utt uxx utx uxt : ℝ) (hw : utt = c ^ 2 * uxx) (hm : uxt = utx) :
    ut * utt + c ^ 2 * ux * uxt = c ^ 2 * (uxx * ut + ux * utx) := by
  veriphy_certified (linear_combination (ut) * hw + (c ^ 2 * ux) * hm)
  -- certificate: de-poly.sage

/-- Integrating, `Ė = [c²u_xu_t]₀^L = 0` when `u_t = 0` at the fixed ends. -/
theorem string_energy_constant (L : ℝ) (flux density : ℝ → ℝ) (hflux : ∀ x, HasDerivAt flux (density x) x)
    (hc : Continuous density) (h0 : flux 0 = 0) (hL : flux L = 0) : ∫ x in (0 : ℝ)..L, density x = 0 := by
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hflux x) (hc.intervalIntegrable 0 L), h0, hL,
    sub_zero]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.string_energy_constant
-- test: de-string-energy-test.sage

/-- Uniqueness: zero energy density forces `w_t = w_x = 0` (`c ≠ 0`). -/
theorem zero_energy (c wt wx : ℝ) (hc : c ≠ 0) (h : wt ^ 2 + c ^ 2 * wx ^ 2 = 0) : wt = 0 ∧ wx = 0 := by
  have h1 : 0 < c ^ 2 := by positivity
  have hw : c ^ 2 * wx ^ 2 = 0 := by nlinarith [sq_nonneg wt, mul_nonneg h1.le (sq_nonneg wx)]
  have hx : wx ^ 2 = 0 := (mul_eq_zero.mp hw).resolve_left h1.ne'
  have ht : wt ^ 2 = 0 := by nlinarith
  exact ⟨pow_eq_zero_iff (two_ne_zero) |>.mp ht, pow_eq_zero_iff (two_ne_zero) |>.mp hx⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.zero_energy
-- test: de-zero-energy-test.sage

/-! ## exr-wave-klein-gordon (with prp-pde-solutions) -/

/-- The plane wave `cos(k·x - ωt)` in `(Box + m²)u` leaves `(-ω² + |k|² + m²)u`, so it solves the Klein-Gordon
equation exactly when `ω² = |k|² + m²` (the real part of `e^{i(k·x - ωt)}`). -/
theorem kg_plane_wave_cos (k₁ k₂ k₃ ω m t x₁ x₂ x₃ : ℝ) :
    deriv (fun s => deriv (fun r => Real.cos (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r)) s) t -
      (deriv (fun s => deriv (fun r => Real.cos (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t)) s) x₁ +
        deriv (fun s => deriv (fun r => Real.cos (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t)) s) x₂ +
        deriv (fun s => deriv (fun r => Real.cos (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t)) s) x₃) +
      m ^ 2 * Real.cos (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * t) =
      (-ω ^ 2 + (k₁ ^ 2 + k₂ ^ 2 + k₃ ^ 2) + m ^ 2) * Real.cos (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * t) := by
  veriphy_certified
    ( have i0 :
        deriv (fun r => cos (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r)) = fun d1 =>
          ω * (Real.sin (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => cos (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r))
            ((ω * (Real.sin (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (-ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).cos) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun d1 => ω * (Real.sin (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃)))) s)
          ((-(ω ^ 2) * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (t) :=
        by
        convert
          (((((((hasDerivAt_id' (t : ℝ)).const_mul (-ω : ℝ)).add (hasDerivAt_const (t : ℝ) (x₁ * k₁ : ℝ))).add
                      (hasDerivAt_const (t : ℝ) (x₂ * k₂ : ℝ))).add
                  (hasDerivAt_const (t : ℝ) (x₃ * k₃ : ℝ))).sin).const_mul
            (ω : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 :
        deriv (fun r => cos (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t)) = fun d1 =>
          -k₁ * (Real.sin (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => cos (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t))
            ((-k₁ * (Real.sin (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).cos) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt (fun s => (fun d1 => -k₁ * (Real.sin (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃)))) s)
          ((-(k₁ ^ 2) * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₁) :=
        by
        convert
          ((((((hasDerivAt_const (x₁ : ℝ) (-t * ω : ℝ)).add ((hasDerivAt_id' (x₁ : ℝ)).const_mul (k₁ : ℝ))).add
                      (hasDerivAt_const (x₁ : ℝ) (x₂ * k₂ : ℝ))).add
                  (hasDerivAt_const (x₁ : ℝ) (x₃ * k₃ : ℝ))).sin).const_mul
            (-k₁ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      have i4 :
        deriv (fun r => cos (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t)) = fun d1 =>
          -k₂ * (Real.sin (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => cos (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t))
            ((-k₂ * (Real.sin (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).cos) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i4]
      have h5 :
        HasDerivAt (fun s => (fun d1 => -k₂ * (Real.sin (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃)))) s)
          ((-(k₂ ^ 2) * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₂) :=
        by
        convert
          ((((((hasDerivAt_const (x₂ : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (x₂ : ℝ) (x₁ * k₁ : ℝ))).add
                      ((hasDerivAt_id' (x₂ : ℝ)).const_mul (k₂ : ℝ))).add
                  (hasDerivAt_const (x₂ : ℝ) (x₃ * k₃ : ℝ))).sin).const_mul
            (-k₂ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h5.deriv]
      have i6 :
        deriv (fun r => cos (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t)) = fun d1 =>
          -k₃ * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => cos (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t))
            ((-k₃ * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₃ : ℝ))).cos) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i6]
      have h7 :
        HasDerivAt (fun s => (fun d1 => -k₃ * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃)))) s)
          ((-(k₃ ^ 2) * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₃) :=
        by
        convert
          ((((((hasDerivAt_const (x₃ : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (x₃ : ℝ) (x₁ * k₁ : ℝ))).add
                      (hasDerivAt_const (x₃ : ℝ) (x₂ * k₂ : ℝ))).add
                  ((hasDerivAt_id' (x₃ : ℝ)).const_mul (k₃ : ℝ))).sin).const_mul
            (-k₃ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h7.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

/-- … and the same for the imaginary part `sin(k·x - ωt)`. -/
theorem kg_plane_wave_sin (k₁ k₂ k₃ ω m t x₁ x₂ x₃ : ℝ) :
    deriv (fun s => deriv (fun r => Real.sin (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r)) s) t -
      (deriv (fun s => deriv (fun r => Real.sin (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t)) s) x₁ +
        deriv (fun s => deriv (fun r => Real.sin (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t)) s) x₂ +
        deriv (fun s => deriv (fun r => Real.sin (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t)) s) x₃) +
      m ^ 2 * Real.sin (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * t) =
      (-ω ^ 2 + (k₁ ^ 2 + k₂ ^ 2 + k₃ ^ 2) + m ^ 2) * Real.sin (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * t) := by
  veriphy_certified
    ( have i0 :
        deriv (fun r => sin (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r)) = fun d1 =>
          -ω * (Real.cos (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k₁ * x₁ + k₂ * x₂ + k₃ * x₃ - ω * r))
            ((-ω * (Real.cos (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (-ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).sin) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun d1 => -ω * (Real.cos (-d1 * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃)))) s)
          ((-(ω ^ 2) * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (t) :=
        by
        convert
          (((((((hasDerivAt_id' (t : ℝ)).const_mul (-ω : ℝ)).add (hasDerivAt_const (t : ℝ) (x₁ * k₁ : ℝ))).add
                      (hasDerivAt_const (t : ℝ) (x₂ * k₂ : ℝ))).add
                  (hasDerivAt_const (t : ℝ) (x₃ * k₃ : ℝ))).cos).const_mul
            (-ω : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 :
        deriv (fun r => sin (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t)) = fun d1 =>
          k₁ * (Real.cos (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k₁ * r + k₂ * x₂ + k₃ * x₃ - ω * t))
            ((k₁ * (Real.cos (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).sin) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt (fun s => (fun d1 => k₁ * (Real.cos (-t * ω + (d1 * k₁) + (x₂ * k₂) + (x₃ * k₃)))) s)
          ((-(k₁ ^ 2) * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₁) :=
        by
        convert
          ((((((hasDerivAt_const (x₁ : ℝ) (-t * ω : ℝ)).add ((hasDerivAt_id' (x₁ : ℝ)).const_mul (k₁ : ℝ))).add
                      (hasDerivAt_const (x₁ : ℝ) (x₂ * k₂ : ℝ))).add
                  (hasDerivAt_const (x₁ : ℝ) (x₃ * k₃ : ℝ))).cos).const_mul
            (k₁ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      have i4 :
        deriv (fun r => sin (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t)) = fun d1 =>
          k₂ * (Real.cos (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k₁ * x₁ + k₂ * r + k₃ * x₃ - ω * t))
            ((k₂ * (Real.cos (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₂ : ℝ))).add
                (hasDerivAt_const (d1 : ℝ) (x₃ * k₃ : ℝ))).sin) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i4]
      have h5 :
        HasDerivAt (fun s => (fun d1 => k₂ * (Real.cos (-t * ω + (x₁ * k₁) + (d1 * k₂) + (x₃ * k₃)))) s)
          ((-(k₂ ^ 2) * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₂) :=
        by
        convert
          ((((((hasDerivAt_const (x₂ : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (x₂ : ℝ) (x₁ * k₁ : ℝ))).add
                      ((hasDerivAt_id' (x₂ : ℝ)).const_mul (k₂ : ℝ))).add
                  (hasDerivAt_const (x₂ : ℝ) (x₃ * k₃ : ℝ))).cos).const_mul
            (k₂ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h5.deriv]
      have i6 :
        deriv (fun r => sin (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t)) = fun d1 =>
          k₃ * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃))) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k₁ * x₁ + k₂ * x₂ + k₃ * r - ω * t))
            ((k₃ * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃))) : ℝ)) (d1) :=
          by
          convert
            (((((hasDerivAt_const (d1 : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (d1 : ℝ) (x₁ * k₁ : ℝ))).add
                    (hasDerivAt_const (d1 : ℝ) (x₂ * k₂ : ℝ))).add
                ((hasDerivAt_id' (d1 : ℝ)).const_mul (k₃ : ℝ))).sin) using
            1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (simp only [Pi.add_apply];
                first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i6]
      have h7 :
        HasDerivAt (fun s => (fun d1 => k₃ * (Real.cos (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (d1 * k₃)))) s)
          ((-(k₃ ^ 2) * (Real.sin (-t * ω + (x₁ * k₁) + (x₂ * k₂) + (x₃ * k₃))) : ℝ)) (x₃) :=
        by
        convert
          ((((((hasDerivAt_const (x₃ : ℝ) (-t * ω : ℝ)).add (hasDerivAt_const (x₃ : ℝ) (x₁ * k₁ : ℝ))).add
                      (hasDerivAt_const (x₃ : ℝ) (x₂ * k₂ : ℝ))).add
                  ((hasDerivAt_id' (x₃ : ℝ)).const_mul (k₃ : ℝ))).cos).const_mul
            (k₃ : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (simp only [Pi.add_apply];
              first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h7.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

/-- The group velocity `dω/dk = k/√(k² + m²)` … -/
theorem kg_group_velocity (k m : ℝ) (h : 0 < k ^ 2 + m ^ 2) :
    deriv (fun s => Real.sqrt (s ^ 2 + m ^ 2)) k = k / Real.sqrt (k ^ 2 + m ^ 2) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => √(s ^ 2 + m ^ 2)) ((k / (Real.sqrt (k ^ 2 + (m ^ 2))) : ℝ)) (k) :=
        by
        refine
          ((((((hasDerivAt_id' (k : ℝ)).pow 2).add (hasDerivAt_const (k : ℝ) (m ^ 2 : ℝ))).sqrt
                    (by exact h.ne'))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · (simp; ring_nf; done)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- … times the phase velocity `√(k² + m²)/k` is `1` … -/
theorem kg_velocities_product (k m : ℝ) (hk : k ≠ 0) (h : 0 < k ^ 2 + m ^ 2) :
    Real.sqrt (k ^ 2 + m ^ 2) / k * (k / Real.sqrt (k ^ 2 + m ^ 2)) = 1 := by
  veriphy_certified
    ( have hs : Real.sqrt (k ^ 2 + m ^ 2) ≠ 0 := (Real.sqrt_pos.mpr h).ne'
      field_simp)
  -- certificate: de-rewrite.sage

/-- … the phase velocity exceeds `1` and the group velocity is below `1` (`k > 0`, `m ≠ 0`). -/
theorem kg_velocities_bounds (k m : ℝ) (hk : 0 < k) (hm : m ≠ 0) :
    1 < Real.sqrt (k ^ 2 + m ^ 2) / k ∧ k / Real.sqrt (k ^ 2 + m ^ 2) < 1 := by
  have hm2 : 0 < m ^ 2 := by positivity
  have hs : k < Real.sqrt (k ^ 2 + m ^ 2) := (Real.lt_sqrt hk.le).2 (by linarith)
  exact ⟨by rw [lt_div_iff₀ hk]; linarith, by rw [div_lt_one (by linarith)]; exact hs⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.WaveEquation.kg_velocities_bounds
-- test: de-kg-velocities-test.sage

end KaluzaKlein.DifferentialEquations.WaveEquation

namespace KaluzaKlein.DifferentialEquations.HeatEquation

/-! ## def-heat-equation (with prp-pde-solutions) -/

/-- The heat kernel `K(t, x) = e^{-x²/(4κt)}/√(4πκt)` solves `u_t = κu_xx` for `κ, t > 0`. -/
theorem kernel_solves (κ t x : ℝ) (hκ : 0 < κ) (ht : 0 < t) (h4 : 4 * κ * t ≠ 0) (hs : 0 < 4 * Real.pi * κ * t) :
    deriv (fun s => Real.exp (-x ^ 2 / (4 * κ * s)) / Real.sqrt (4 * Real.pi * κ * s)) t =
      κ * deriv (fun y => deriv (fun z => Real.exp (-z ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) y) x := by
  have hS2 : Real.sqrt (4 * Real.pi * κ * t) ^ 2 = 4 * Real.pi * κ * t := Real.sq_sqrt hs.le
  have hSne : Real.sqrt (4 * Real.pi * κ * t) ≠ 0 := (Real.sqrt_pos.mpr hs).ne'
  have hE : HasDerivAt (fun s => Real.exp (-x ^ 2 / (4 * κ * s)))
      (Real.exp (-x ^ 2 / (4 * κ * t)) * ((0 * (4 * κ * t) - -x ^ 2 * (4 * κ * 1)) / (4 * κ * t) ^ 2)) t :=
    ((hasDerivAt_const t (-x ^ 2)).div ((hasDerivAt_id' t).const_mul (4 * κ)) h4).exp
  have hS : HasDerivAt (fun s => Real.sqrt (4 * Real.pi * κ * s)) (Real.sqrt (4 * Real.pi * κ * t) / (2 * t)) t := by
    have h0 : HasDerivAt (fun s => Real.sqrt (4 * Real.pi * κ * s))
        ((4 * Real.pi * κ * 1) / (2 * Real.sqrt (4 * Real.pi * κ * t))) t :=
      ((hasDerivAt_id' t).const_mul (4 * Real.pi * κ)).sqrt hs.ne'
    refine h0.congr_deriv ?_
    rw [div_eq_div_iff (by positivity) (by positivity)]
    linear_combination (-2) * hS2
  have hT : HasDerivAt (fun s => Real.exp (-x ^ 2 / (4 * κ * s)) / Real.sqrt (4 * Real.pi * κ * s)) _ t := hE.div hS hSne
  rw [hT.deriv]
  have hz : ∀ z : ℝ, HasDerivAt (fun z => Real.exp (-z ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t))
      (Real.exp (-z ^ 2 / (4 * κ * t)) * (-(2 * z) / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) z := by
    intro z
    have := (((hasDerivAt_pow 2 z).neg.div_const (4 * κ * t)).exp).div_const (Real.sqrt (4 * Real.pi * κ * t))
    convert this using 1
    norm_num
  rw [show (fun y => deriv (fun z => Real.exp (-z ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) y) =
    fun y => Real.exp (-y ^ 2 / (4 * κ * t)) * (-(2 * y) / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) from
    funext fun y => (hz y).deriv]
  have hEy : HasDerivAt (fun y => Real.exp (-y ^ 2 / (4 * κ * t))) (Real.exp (-x ^ 2 / (4 * κ * t)) * (-(2 * x) / (4 * κ * t))) x := by
    have := ((hasDerivAt_pow 2 x).neg.div_const (4 * κ * t)).exp
    convert this using 1
    norm_num
  have hlin : HasDerivAt (fun y => -(2 * y) / (4 * κ * t)) (-(2 * 1) / (4 * κ * t)) x :=
    (((hasDerivAt_id' x).const_mul 2).neg).div_const _
  have hX : HasDerivAt (fun y => Real.exp (-y ^ 2 / (4 * κ * t)) * (-(2 * y) / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) _ x :=
    (hEy.mul hlin).div_const _
  rw [hX.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.kernel_solves
-- test: de-kernel-solves-test.sage

/-- `∫ K dx = 1` (the Gaussian integral `∫ e^{-bx²} = √(π/b)` with `b = 1/(4κt)`). -/
theorem kernel_integral (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    ∫ x, Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) = 1 := by
  have hb : 0 < 1 / (4 * κ * t) := by positivity
  have h := integral_gaussian (1 / (4 * κ * t))
  have e : (fun x : ℝ => Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) =
      fun x => (Real.sqrt (4 * Real.pi * κ * t))⁻¹ * Real.exp (-(1 / (4 * κ * t)) * x ^ 2) := by
    funext x; rw [div_eq_inv_mul]; congr 2; ring
  rw [e, MeasureTheory.integral_const_mul, h, show Real.pi / (1 / (4 * κ * t)) = 4 * Real.pi * κ * t by
    field_simp]
  have : Real.sqrt (4 * Real.pi * κ * t) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.kernel_integral
-- test: de-kernel-integral-test.sage

/-! ## exm-heat-kernel -/

/-- The maximum `K(t, 0) = (4πκt)^{-1/2}`. -/
theorem kernel_maximum (κ t : ℝ) :
    Real.exp (-0 ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) = 1 / Real.sqrt (4 * Real.pi * κ * t) := by
  veriphy_certified (simp)
  -- certificate: de-rewrite.sage

/-- The width `√(2κt)`: four times the time doubles it, `√(2κ·4t) = 2√(2κt)`. -/
theorem kernel_width (κ t : ℝ) (h : 0 ≤ 2 * κ * t) : Real.sqrt (2 * κ * (4 * t)) = 2 * Real.sqrt (2 * κ * t) := by
  veriphy_certified
    (rw [show 2 * κ * (4 * t) = 2 ^ 2 * (2 * κ * t) by ring, Real.sqrt_mul (by norm_num), Real.sqrt_sq (by norm_num)])
  -- certificate: de-rewrite.sage

/-- `K(t, x) ≤ K(t, 0)`: the kernel is largest at the origin. -/
theorem kernel_le_max (κ t x : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    Real.exp (-x ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) ≤ 1 / Real.sqrt (4 * Real.pi * κ * t) := by
  apply div_le_div_of_nonneg_right _ (Real.sqrt_nonneg _)
  rw [Real.exp_le_one_iff]
  apply div_nonpos_of_nonpos_of_nonneg (by nlinarith [sq_nonneg x]) (by positivity)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.kernel_le_max
-- test: de-kernel-le-max-test.sage

/-! ## exm-heat-mode (with prp-pde-solutions) -/

/-- `u = e^{-κk²t} sin kx` solves the heat equation. -/
theorem mode_solves (κ k t x : ℝ) :
    deriv (fun s => Real.exp (-κ * k ^ 2 * s) * Real.sin (k * x)) t =
      κ * deriv (fun y => deriv (fun z => Real.exp (-κ * k ^ 2 * t) * Real.sin (k * z)) y) x := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun s => rexp (-κ * k ^ 2 * s) * sin (k * x))
          ((-κ * (k ^ 2) * (Real.exp (-κ * (k ^ 2) * t)) * (Real.sin (k * x)) : ℝ)) (t) :=
        by
        convert ((((hasDerivAt_id' (t : ℝ)).const_mul (-κ * (k ^ 2) : ℝ)).exp).const_mul (Real.sin (k * x) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      try rw [h0.deriv]
      have i1 :
        deriv (fun z => rexp (-κ * k ^ 2 * t) * sin (k * z)) = fun d1 =>
          k * (Real.cos (d1 * k)) * (Real.exp (-κ * (k ^ 2) * t)) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun z => rexp (-κ * k ^ 2 * t) * sin (k * z))
            ((k * (Real.cos (d1 * k)) * (Real.exp (-κ * (k ^ 2) * t)) : ℝ)) (d1) :=
          by
          convert
            ((((hasDerivAt_id' (d1 : ℝ)).const_mul (k : ℝ)).sin).const_mul (Real.exp (-κ * (k ^ 2) * t) : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      rw [i1]
      have h2 :
        HasDerivAt (fun y => (fun d1 => k * (Real.cos (d1 * k)) * (Real.exp (-κ * (k ^ 2) * t))) y)
          ((-(k ^ 2) * (Real.exp (-κ * (k ^ 2) * t)) * (Real.sin (k * x)) : ℝ)) (x) :=
        by
        convert
          ((((hasDerivAt_id' (x : ℝ)).const_mul (k : ℝ)).cos).const_mul (k * (Real.exp (-κ * (k ^ 2) * t)) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          | (ring1)
      try rw [h2.deriv]
      beta_reduce
      ring1)
  -- certificate: de-partials.sage

/-! ## exm-heat-maximum -/

/-- For a decaying mode the maximum over `t ≥ 0` is attained on the initial line: `|e^{-κk²t} sin kx| ≤ |sin kx|`. -/
theorem mode_maximum (κ k t x : ℝ) (hκ : 0 ≤ κ) (ht : 0 ≤ t) :
    |Real.exp (-κ * k ^ 2 * t) * Real.sin (k * x)| ≤ |Real.sin (k * x)| := by
  rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have : Real.exp (-κ * k ^ 2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by
    have : 0 ≤ κ * k ^ 2 * t := by positivity
    linarith)
  nlinarith [abs_nonneg (Real.sin (k * x))]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.mode_maximum
-- test: de-mode-maximum-test.sage

/-! ## exr-heat-kernel-check -/

/-- The convolution of the kernel with `e^{-y²}` is the Gaussian `e^{-x²/(1 + 4κt)}/√(1 + 4κt)`, of variance
`½ + 2κt`: variances add. -/
theorem kernel_convolution (κ t x : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    ∫ y, Real.exp (-(x - y) ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) * Real.exp (-y ^ 2) =
      Real.exp (-x ^ 2 / (1 + 4 * κ * t)) / Real.sqrt (1 + 4 * κ * t) := by
  have ha0 : 0 < 4 * κ * t := by positivity
  have h1a : 0 < 1 + 4 * κ * t := by positivity
  have ha0' : 4 * κ * t ≠ 0 := ha0.ne'
  have h1a' : 1 + 4 * κ * t ≠ 0 := h1a.ne'
  have e : (fun y => Real.exp (-(x - y) ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) * Real.exp (-y ^ 2)) =
      fun y => (Real.exp (-x ^ 2 / (1 + 4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) *
        Real.exp (-((1 + 4 * κ * t) / (4 * κ * t)) * (y - x / (1 + 4 * κ * t)) ^ 2) := by
    funext y
    rw [div_mul_eq_mul_div, div_mul_eq_mul_div, ← Real.exp_add, ← Real.exp_add]
    congr 2
    field_simp
    ring
  rw [e, MeasureTheory.integral_const_mul,
    MeasureTheory.integral_sub_right_eq_self (fun y => Real.exp (-((1 + 4 * κ * t) / (4 * κ * t)) * y ^ 2))
      (x / (1 + 4 * κ * t)), integral_gaussian,
    show Real.pi / ((1 + 4 * κ * t) / (4 * κ * t)) = 4 * Real.pi * κ * t / (1 + 4 * κ * t) by field_simp,
    Real.sqrt_div' _ h1a.le]
  have h1 : Real.sqrt (4 * Real.pi * κ * t) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  have h2 : Real.sqrt (1 + 4 * κ * t) ≠ 0 := (Real.sqrt_pos.mpr h1a).ne'
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.kernel_convolution
-- test: de-kernel-convolution-test.sage

/-- The variance of `e^{-x²/(1 + 4κt)}` is `(1 + 4κt)/2 = ½ + 2κt`. -/
theorem convolution_variance (κ t : ℝ) : (1 + 4 * κ * t) / 2 = 1 / 2 + 2 * κ * t := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exr-heat-similarity -/

/-- `u = t^{-1/2}F(ξ)`, `ξ = x/√(4κt)`: `u_t - κu_xx = -(t^{-3/2}/4)(F'' + 2ξF' + 2F)`, written with the chain-rule
values `u_t = -½t^{-3/2}F - t^{-3/2}ξF'/2` and `κu_xx = t^{-3/2}F''/4`. -/
theorem similarity_reduction (t3 ξ F F' F'' : ℝ) :
    (-(1 / 2) * t3 * F - t3 * ξ * F' / 2) - t3 * F'' / 4 = -(t3 / 4) * (F'' + 2 * ξ * F' + 2 * F) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `F = e^{-ξ²}` solves `F'' + 2ξF' + 2F = 0`. -/
theorem similarity_gaussian (F : ℝ → ℝ) (hF : F = fun s => Real.exp (-s ^ 2)) (ξ : ℝ) :
    deriv (deriv F) ξ + 2 * ξ * deriv F ξ + 2 * F ξ = 0 := by
  veriphy_certified
    ( subst hF
      have d1 : deriv (fun (s : ℝ) => (rexp (-s ^ 2) : ℝ)) = fun (s : ℝ) => (-2 * s * (Real.exp (-(s ^ 2))) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (rexp (-s ^ 2) : ℝ)) ((-2 * s * (Real.exp (-(s ^ 2))) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-1 : ℝ)).exp)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; done)
          · (simp; ring_nf; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (-2 * s * (Real.exp (-(s ^ 2))) : ℝ)) = fun (s : ℝ) =>
          (4 * (s ^ 2) * (Real.exp (-(s ^ 2))) - (2 * (Real.exp (-(s ^ 2)))) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (-2 * s * (Real.exp (-(s ^ 2))) : ℝ))
            ((4 * (s ^ 2) * (Real.exp (-(s ^ 2))) - (2 * (Real.exp (-(s ^ 2)))) : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).mul ((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-1 : ℝ)).exp)).const_mul
                      (-2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (simp; ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- The equation is `(F' + 2ξF)' = 0`: `F'' + 2ξF' + 2F` is the derivative of `F' + 2ξF`. -/
theorem similarity_exact (F F' F'' : ℝ → ℝ) (ξ : ℝ) (hF : HasDerivAt F (F' ξ) ξ) (hF' : HasDerivAt F' (F'' ξ) ξ) :
    HasDerivAt (fun s => F' s + 2 * s * F s) (F'' ξ + 2 * ξ * F' ξ + 2 * F ξ) ξ := by
  convert hF'.add (((hasDerivAt_id' ξ).const_mul 2).mul hF) using 1
  simp; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.similarity_exact
-- test: de-similarity-exact-test.sage

/-! ## exr-heat-uniqueness -/

/-- `∂_t(u²/2) = uu_t = κ∂_x(uu_x) - κu_x²` on solutions (`u_t = κu_xx`). -/
theorem heat_energy_identity (κ u ut ux uxx : ℝ) (h : ut = κ * uxx) : u * ut = κ * (ux * ux + u * uxx) - κ * ux ^ 2 := by
  veriphy_certified (linear_combination (u) * h)
  -- certificate: de-poly.sage

/-- A non-negative, non-increasing energy that starts at `0` stays `0`. -/
theorem energy_stays_zero (E : ℝ → ℝ) (hanti : AntitoneOn E (Set.Ici 0)) (hnn : ∀ t, 0 ≤ E t) (h0 : E 0 = 0)
    (t : ℝ) (ht : 0 ≤ t) : E t = 0 :=
  le_antisymm (by have := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht) ht; rwa [h0] at this) (hnn t)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.HeatEquation.energy_stays_zero
-- test: de-energy-zero-test.sage

end KaluzaKlein.DifferentialEquations.HeatEquation

namespace KaluzaKlein.DifferentialEquations.LaplaceEquation

/-! ## def-laplace-equation and exr-laplace-uniqueness -/

/-- The energy identity behind uniqueness: `∇·(w∇w) = |∇w|² + w∇²w` (here in two variables, with
`∂ₓ(wwₓ) = wₓ² + wwₓₓ`). -/
theorem divergence_identity (w wx wy wxx wyy : ℝ) :
    (wx * wx + w * wxx) + (wy * wy + w * wyy) = (wx ^ 2 + wy ^ 2) + w * (wxx + wyy) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- Hence for harmonic `w` the divergence of `w∇w` is `|∇w|² ≥ 0`, and `|∇w|² = 0` forces `∇w = 0`. -/
theorem gradient_zero (wx wy : ℝ) (h : wx ^ 2 + wy ^ 2 = 0) : wx = 0 ∧ wy = 0 := by
  constructor <;> nlinarith [sq_nonneg wx, sq_nonneg wy]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.gradient_zero
-- test: de-gradient-zero-test.sage

/-! ## exm-laplace-polynomials and exr-laplace-polar (polar Laplacian `u_rr + u_r/r + u_θθ/r²`, `r > 0`) -/

/-- `rⁿ cos nθ` is harmonic. -/
theorem polar_rn_cos (n r θ : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => ρ ^ n * Real.cos (n * θ)) s) r + 1 / r * deriv (fun ρ => ρ ^ n * Real.cos (n * θ)) r +
      1 / r ^ 2 * deriv (fun s => deriv (fun φ => r ^ n * Real.cos (n * φ)) s) θ = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => ρ ^ n * Real.cos (n * θ)) (n * s ^ (n - 1) * Real.cos (n * θ)) s :=
    fun s hs => (Real.hasDerivAt_rpow_const (Or.inl hs.ne')).mul_const _
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => n * s ^ (n - 1) * Real.cos (n * θ)) (n * ((n - 1) * r ^ (n - 1 - 1)) * Real.cos (n * θ)) r :=
    ((Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul n).mul_const _
  rw [h2.deriv]
  have hθ : ∀ φ : ℝ, HasDerivAt (fun φ => r ^ n * Real.cos (n * φ)) (r ^ n * (-Real.sin (n * φ) * n)) φ := by
    intro φ
    have := ((hasDerivAt_id' φ).const_mul n).cos.const_mul (r ^ n)
    simpa using this
  rw [show (fun s => deriv (fun φ => r ^ n * Real.cos (n * φ)) s) = fun s => r ^ n * (-Real.sin (n * s) * n) from
    funext fun s => (hθ s).deriv]
  have h3 : HasDerivAt (fun s => r ^ n * (-Real.sin (n * s) * n)) (r ^ n * (-(Real.cos (n * θ) * n) * n)) θ := by
    have := (((hasDerivAt_id' θ).const_mul n).sin.neg.mul_const n).const_mul (r ^ n)
    simpa using this
  rw [h3.deriv]
  have e1 : r ^ (n - 1) = r ^ n / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (n - 1 - 1) = r ^ n / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  rw [e1, e2]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.polar_rn_cos
-- test: de-polar-rn-cos-test.sage

/-- `r⁻ⁿ cos nθ` is harmonic. -/
theorem polar_r_neg_n_cos (n r θ : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => ρ ^ (-n) * Real.cos (n * θ)) s) r +
      1 / r * deriv (fun ρ => ρ ^ (-n) * Real.cos (n * θ)) r +
      1 / r ^ 2 * deriv (fun s => deriv (fun φ => r ^ (-n) * Real.cos (n * φ)) s) θ = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => ρ ^ (-n) * Real.cos (n * θ)) (-n * s ^ (-n - 1) * Real.cos (n * θ)) s :=
    fun s hs => (Real.hasDerivAt_rpow_const (Or.inl hs.ne')).mul_const _
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => -n * s ^ (-n - 1) * Real.cos (n * θ))
      (-n * ((-n - 1) * r ^ (-n - 1 - 1)) * Real.cos (n * θ)) r :=
    ((Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul (-n)).mul_const _
  rw [h2.deriv]
  have hθ : ∀ φ : ℝ, HasDerivAt (fun φ => r ^ (-n) * Real.cos (n * φ)) (r ^ (-n) * (-Real.sin (n * φ) * n)) φ := by
    intro φ
    have := ((hasDerivAt_id' φ).const_mul n).cos.const_mul (r ^ (-n))
    simpa using this
  rw [show (fun s => deriv (fun φ => r ^ (-n) * Real.cos (n * φ)) s) = fun s => r ^ (-n) * (-Real.sin (n * s) * n) from
    funext fun s => (hθ s).deriv]
  have h3 : HasDerivAt (fun s => r ^ (-n) * (-Real.sin (n * s) * n)) (r ^ (-n) * (-(Real.cos (n * θ) * n) * n)) θ := by
    have := (((hasDerivAt_id' θ).const_mul n).sin.neg.mul_const n).const_mul (r ^ (-n))
    simpa using this
  rw [h3.deriv]
  have e1 : r ^ (-n - 1) = r ^ (-n) / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (-n - 1 - 1) = r ^ (-n) / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  rw [e1, e2]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.polar_r_neg_n_cos
-- test: de-polar-r-neg-n-cos-test.sage

/-- `log r` is harmonic (away from the origin). -/
theorem polar_log (r : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => Real.log ρ) s) r + 1 / r * deriv (fun ρ => Real.log ρ) r = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => Real.log ρ) s⁻¹ s := fun s hs => Real.hasDerivAt_log hs.ne'
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv, (hasDerivAt_inv hr.ne').deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.polar_log
-- test: de-polar-log-test.sage

/-- `rⁿ cos mθ` leaves the residual `(n² - m²)r^{n-2} cos mθ`: harmonic only for `m = ±n`. -/
theorem polar_residual (n m r θ : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => ρ ^ n * Real.cos (m * θ)) s) r + 1 / r * deriv (fun ρ => ρ ^ n * Real.cos (m * θ)) r +
      1 / r ^ 2 * deriv (fun s => deriv (fun φ => r ^ n * Real.cos (m * φ)) s) θ =
      (n ^ 2 - m ^ 2) * r ^ (n - 2) * Real.cos (m * θ) := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => ρ ^ n * Real.cos (m * θ)) (n * s ^ (n - 1) * Real.cos (m * θ)) s :=
    fun s hs => (Real.hasDerivAt_rpow_const (Or.inl hs.ne')).mul_const _
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => n * s ^ (n - 1) * Real.cos (m * θ)) (n * ((n - 1) * r ^ (n - 1 - 1)) * Real.cos (m * θ)) r :=
    ((Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul n).mul_const _
  rw [h2.deriv]
  have hθ : ∀ φ : ℝ, HasDerivAt (fun φ => r ^ n * Real.cos (m * φ)) (r ^ n * (-Real.sin (m * φ) * m)) φ := by
    intro φ
    have := ((hasDerivAt_id' φ).const_mul m).cos.const_mul (r ^ n)
    simpa using this
  rw [show (fun s => deriv (fun φ => r ^ n * Real.cos (m * φ)) s) = fun s => r ^ n * (-Real.sin (m * s) * m) from
    funext fun s => (hθ s).deriv]
  have h3 : HasDerivAt (fun s => r ^ n * (-Real.sin (m * s) * m)) (r ^ n * (-(Real.cos (m * θ) * m) * m)) θ := by
    have := (((hasDerivAt_id' θ).const_mul m).sin.neg.mul_const m).const_mul (r ^ n)
    simpa using this
  rw [h3.deriv]
  have e1 : r ^ (n - 1) = r ^ n / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (n - 1 - 1) = r ^ n / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  have e3 : r ^ (n - 2) = r ^ n / r ^ 2 := by rw [Real.rpow_sub hr, Real.rpow_two]
  rw [e1, e2, e3]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.polar_residual
-- test: de-polar-residual-test.sage

/-! ## exm-laplace-log (with prp-pde-solutions) -/

/-- `log(x² + y²)` (twice `log r`) is harmonic, here off the axes (`x, y ≠ 0`). -/
theorem log_harmonic (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
    deriv (fun s => deriv (fun r => Real.log (r ^ 2 + y ^ 2)) s) x +
      deriv (fun s => deriv (fun r => Real.log (x ^ 2 + r ^ 2)) s) y = 0 := by
  have hpy : ∀ r : ℝ, r ^ 2 + y ^ 2 ≠ 0 := fun r => by positivity
  have hpx : ∀ r : ℝ, x ^ 2 + r ^ 2 ≠ 0 := fun r => by positivity
  have h1 : ∀ s : ℝ, HasDerivAt (fun r => Real.log (r ^ 2 + y ^ 2)) (2 * s / (s ^ 2 + y ^ 2)) s := by
    intro s
    have hp : HasDerivAt (fun r : ℝ => r ^ 2 + y ^ 2) (2 * s) s := by simpa using (hasDerivAt_pow 2 s).add_const (y ^ 2)
    exact hp.log (hpy s)
  have h1' : ∀ s : ℝ, HasDerivAt (fun r => Real.log (x ^ 2 + r ^ 2)) (2 * s / (x ^ 2 + s ^ 2)) s := by
    intro s
    have hp : HasDerivAt (fun r : ℝ => x ^ 2 + r ^ 2) (2 * s) s := by simpa using (hasDerivAt_pow 2 s).const_add (x ^ 2)
    exact hp.log (hpx s)
  rw [show (fun s => deriv (fun r => Real.log (r ^ 2 + y ^ 2)) s) = fun s => 2 * s / (s ^ 2 + y ^ 2) from
    funext fun s => (h1 s).deriv]
  rw [show (fun s => deriv (fun r => Real.log (x ^ 2 + r ^ 2)) s) = fun s => 2 * s / (x ^ 2 + s ^ 2) from
    funext fun s => (h1' s).deriv]
  have hp : HasDerivAt (fun r : ℝ => r ^ 2 + y ^ 2) (2 * x) x := by simpa using (hasDerivAt_pow 2 x).add_const (y ^ 2)
  have hp' : HasDerivAt (fun r : ℝ => x ^ 2 + r ^ 2) (2 * y) y := by simpa using (hasDerivAt_pow 2 y).const_add (x ^ 2)
  have h2 : HasDerivAt (fun s => 2 * s / (s ^ 2 + y ^ 2)) _ x := ((hasDerivAt_id' x).const_mul 2).div hp (hpy x)
  have h2' : HasDerivAt (fun s => 2 * s / (x ^ 2 + s ^ 2)) _ y := ((hasDerivAt_id' y).const_mul 2).div hp' (hpx y)
  rw [h2.deriv, h2'.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.log_harmonic
-- test: de-log-harmonic-pde-test.sage

/-- `1/r` is harmonic in space away from the origin (radial form `u_rr + (2/r)u_r`). -/
theorem coulomb_harmonic_radial (r : ℝ) (hr : 0 < r) :
    deriv (fun s => deriv (fun ρ => 1 / ρ) s) r + 2 / r * deriv (fun ρ => 1 / ρ) r = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => 1 / ρ) (-1 / s ^ 2) s := by
    intro s hs
    have h : HasDerivAt (fun ρ : ℝ => 1 / ρ) ((0 * s - 1 * 1) / s ^ 2) s := (hasDerivAt_const s (1 : ℝ)).div (hasDerivAt_id' s) hs.ne'
    exact h.congr_deriv (by ring)
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have hp : HasDerivAt (fun s : ℝ => s ^ 2) (2 * r) r := by simpa using hasDerivAt_pow 2 r
  have h2 : HasDerivAt (fun s : ℝ => -1 / s ^ 2) _ r := (hasDerivAt_const r (-1 : ℝ)).div hp (pow_ne_zero 2 hr.ne')
  rw [h2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.coulomb_harmonic_radial
-- test: de-coulomb-harmonic-radial-test.sage

/-! ## exm-laplace-disk -/

/-- The value at the centre is the mean of the boundary values: for a trigonometric polynomial
`g = a₀/2 + Σₙ (aₙ cos nθ + bₙ sin nθ)`, `(1/2π)∫₀^{2π} g = a₀/2`. -/
theorem disk_mean (N : ℕ) (a₀ : ℝ) (a b : ℕ → ℝ) :
    1 / (2 * Real.pi) * ∫ θ in (0 : ℝ)..(2 * Real.pi),
      (a₀ / 2 + ∑ n ∈ Finset.range N, (a n * Real.cos ((n + 1) * θ) + b n * Real.sin ((n + 1) * θ))) = a₀ / 2 := by
  have hc : ∀ n : ℕ, ∫ θ in (0 : ℝ)..(2 * Real.pi), Real.cos ((n + 1) * θ) = 0 := by
    intro n
    have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
    rw [intervalIntegral.integral_comp_mul_left (fun θ => Real.cos θ) hn, integral_cos]
    rw [show ((n : ℝ) + 1) * (2 * Real.pi) = ((2 * (n + 1) : ℕ) : ℝ) * Real.pi by push_cast; ring,
      Real.sin_nat_mul_pi]
    simp
  have hs : ∀ n : ℕ, ∫ θ in (0 : ℝ)..(2 * Real.pi), Real.sin ((n + 1) * θ) = 0 := by
    intro n
    have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
    rw [intervalIntegral.integral_comp_mul_left (fun θ => Real.sin θ) hn, integral_sin]
    rw [show ((n : ℝ) + 1) * (2 * Real.pi) = ((2 * (n + 1) : ℕ) : ℝ) * Real.pi by push_cast; ring,
      Real.cos_nat_mul_pi]
    simp [pow_mul]
  rw [intervalIntegral.integral_add (by apply Continuous.intervalIntegrable; fun_prop)
    (by apply Continuous.intervalIntegrable; fun_prop), intervalIntegral.integral_finset_sum
    (fun n _ => by apply Continuous.intervalIntegrable; fun_prop)]
  rw [Finset.sum_eq_zero fun n _ => by
    rw [intervalIntegral.integral_add (by apply Continuous.intervalIntegrable; fun_prop)
      (by apply Continuous.intervalIntegrable; fun_prop), intervalIntegral.integral_const_mul,
      intervalIntegral.integral_const_mul, hc, hs]; ring]
  simp
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LaplaceEquation.disk_mean
-- test: de-disk-mean-test.sage

/-! ## exr-laplace-mean-value -/

/-- The circle average of `x² - y² + 3x` around `(a, b)` is `u(a, b)`, for every radius `ρ`. -/
theorem mean_value_harmonic (a b ρ : ℝ) :
    1 / (2 * Real.pi) * ∫ θ in (0 : ℝ)..(2 * Real.pi),
      ((a + ρ * Real.cos θ) ^ 2 - (b + ρ * Real.sin θ) ^ 2 + 3 * (a + ρ * Real.cos θ)) = a ^ 2 - b ^ 2 + 3 * a := by
  veriphy_certified
    ( have c2 : ∀ y : ℝ, Real.cos (2 * y) = Real.cos y ^ 2 - Real.sin y ^ 2 :=
        by
        intro y
        rw [show (2 : ℝ) * y = y + y by ring, Real.cos_add]
        ring
      have s2 : ∀ y : ℝ, Real.sin (2 * y) = 2 * Real.cos y * Real.sin y :=
        by
        intro y
        rw [show (2 : ℝ) * y = y + y by ring, Real.sin_add]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun i0 =>
              (i0 * (a ^ 2) - (i0 * (b ^ 2)) + ((2 * i0 + (Real.sin (2 * i0))) * (ρ ^ 2) / 4) -
                          ((2 * i0 - (Real.sin (2 * i0))) * (ρ ^ 2) / 4) +
                        (2 * b * ρ * (Real.cos i0)) +
                      (2 * a * ρ * (Real.sin i0)) +
                    (3 * i0 * a) +
                  (3 * ρ * (Real.sin i0)) :
                ℝ))
            ((fun θ => (a + ρ * cos θ) ^ 2 - (b + ρ * sin θ) ^ 2 + 3 * (a + ρ * cos θ)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((((((hasDerivAt_id' (i0 : ℝ)).const_mul (a ^ 2 : ℝ)).add
                                            ((hasDerivAt_id' (i0 : ℝ)).const_mul (-(b ^ 2) : ℝ))).add
                                        ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).add
                                              (((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin)).const_mul
                                          ((ρ ^ 2) / 4 : ℝ))).add
                                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).add
                                          ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                                            (-1 : ℝ))).const_mul
                                      (-(ρ ^ 2) / 4 : ℝ))).add
                                (((hasDerivAt_id' (i0 : ℝ)).cos).const_mul (2 * b * ρ : ℝ))).add
                            (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (2 * a * ρ : ℝ))).add
                        ((hasDerivAt_id' (i0 : ℝ)).const_mul (3 * a : ℝ))).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (3 * ρ : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, one_mul, mul_one]; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, Real.sin_two_pi,
        Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      (field_simp; ring1))
  -- certificate: de-integral-trig.sage

/-- For `x² + y²` the average is `a² + b² + ρ² > u(a, b)` … -/
theorem mean_value_subharmonic (a b ρ : ℝ) :
    1 / (2 * Real.pi) * ∫ θ in (0 : ℝ)..(2 * Real.pi), ((a + ρ * Real.cos θ) ^ 2 + (b + ρ * Real.sin θ) ^ 2) =
      a ^ 2 + b ^ 2 + ρ ^ 2 := by
  have h : ∫ θ in (0 : ℝ)..(2 * Real.pi), ((a + ρ * Real.cos θ) ^ 2 + (b + ρ * Real.sin θ) ^ 2) =
      2 * Real.pi * (a ^ 2 + b ^ 2 + ρ ^ 2) := by
    rw [intervalIntegral.integral_congr (g := fun θ => a ^ 2 + b ^ 2 + ρ ^ 2 + 2 * a * ρ * Real.cos θ + 2 * b * ρ * Real.sin θ)
      (fun θ _ => by beta_reduce; linear_combination (ρ ^ 2) * Real.sin_sq_add_cos_sq θ)]
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
            HasDerivAt
              (fun x_veriphy =>
                ((a ^ 2) * x_veriphy + ((b ^ 2) * x_veriphy) + ((ρ ^ 2) * x_veriphy) -
                      (2 * b * ρ * (Real.cos x_veriphy)) +
                    (2 * a * ρ * (Real.sin x_veriphy)) :
                  ℝ))
              ((fun x => a ^ 2 + b ^ 2 + ρ ^ 2 + 2 * a * ρ * cos x + 2 * b * ρ * sin x) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            ((((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (a ^ 2 : ℝ)).add
                                  ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (b ^ 2 : ℝ))).add
                              ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (ρ ^ 2 : ℝ))).add
                          (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-2 * b * ρ : ℝ))).add
                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (2 * a * ρ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            ((a ^ 2) * x_veriphy + ((b ^ 2) * x_veriphy) + ((ρ ^ 2) * x_veriphy) - (2 * b * ρ * (Real.cos x_veriphy)) +
                (2 * a * ρ * (Real.sin x_veriphy)) :
              ℝ))
            (f' := fun x => a ^ 2 + b ^ 2 + ρ ^ 2 + 2 * a * ρ * cos x + 2 * b * ρ * sin x) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
        have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
          rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
        have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
        have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
          rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
        simp only [he0, he1, he2, he3]
        all_goals ring1)
    -- certificate: de-integral.sage
  rw [h]
  field_simp

/-- … since `∇²(x² + y²) = 4 > 0`. -/
theorem laplacian_r_sq (x y : ℝ) :
    deriv (fun s => deriv (fun r => r ^ 2 + y ^ 2) s) x + deriv (fun s => deriv (fun r => x ^ 2 + r ^ 2) s) y = 4 := by
  veriphy_certified
    ( have i0 : deriv (fun r => r ^ 2 + y ^ 2) = fun d1 => 2 * d1 :=
        by
        funext d1
        have hd : HasDerivAt (fun r => r ^ 2 + y ^ 2) ((2 * d1 : ℝ)) (d1) :=
          by
          convert (((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (y ^ 2 : ℝ))) using 1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun s => (fun d1 => 2 * d1) s) ((2 : ℝ)) (x) :=
        by
        convert ((hasDerivAt_id' (x : ℝ)).const_mul (2 : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 : deriv (fun r => x ^ 2 + r ^ 2) = fun d1 => 2 * d1 :=
        by
        funext d1
        have hd : HasDerivAt (fun r => x ^ 2 + r ^ 2) ((2 * d1 : ℝ)) (d1) :=
          by
          convert (((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (x ^ 2 : ℝ))) using 1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 : HasDerivAt (fun s => (fun d1 => 2 * d1) s) ((2 : ℝ)) (y) :=
        by
        convert ((hasDerivAt_id' (y : ℝ)).const_mul (2 : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

end KaluzaKlein.DifferentialEquations.LaplaceEquation

namespace KaluzaKlein.DifferentialEquations.SeparationOfVariables

/-! ## def-separation-of-variables -/

/-- Substituting `u = T(t)X(x)` in `u_t = κu_xx` and dividing by `κu`: `T'/(κT) = X''/X`. -/
theorem separate (κ T T' X X'' : ℝ) (hκ : κ ≠ 0) (hT : T ≠ 0) (hX : X ≠ 0) (h : T' * X = κ * T * X'') :
    T' / (κ * T) = X'' / X := by
  rw [div_eq_div_iff (mul_ne_zero hκ hT) hX]
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.separate
-- test: de-separate-test.sage

/-! ## exm-sep-heat -/

/-- `Tₙ = e^{-κn²t}` solves `T' = -κn²T`. -/
theorem heat_time_factor (κ n : ℝ) (T : ℝ → ℝ) (hT : T = fun s => Real.exp (-κ * n ^ 2 * s)) (t : ℝ) :
    deriv T t = -κ * n ^ 2 * T t := by
  veriphy_certified
    ( subst hT
      have d1 : deriv (fun s => rexp (-κ * n ^ 2 * s)) = fun s => -κ * (n ^ 2) * (Real.exp (-s * κ * (n ^ 2))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => rexp (-κ * n ^ 2 * s)) ((-κ * (n ^ 2) * (Real.exp (-s * κ * (n ^ 2))) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).const_mul (-κ * (n ^ 2) : ℝ)).exp) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- A finite sine series `Σ bₙ e^{-κn²t} sin nx` solves the heat equation. -/
theorem heat_series_solves (N : ℕ) (b : ℕ → ℝ) (κ t x : ℝ) :
    deriv (fun s => ∑ n ∈ Finset.range N, b n * (Real.exp (-κ * (n : ℝ) ^ 2 * s) * Real.sin (n * x))) t =
      κ * deriv (fun y => deriv (fun z => ∑ n ∈ Finset.range N,
        b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * Real.sin (n * z))) y) x := by
  have ht : ∀ n : ℕ, HasDerivAt (fun s => b n * (Real.exp (-κ * (n : ℝ) ^ 2 * s) * Real.sin (n * x)))
      (b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * (-κ * (n : ℝ) ^ 2)) * Real.sin (n * x)) t := by
    intro n
    have := ((((hasDerivAt_id' t).const_mul (-κ * (n : ℝ) ^ 2)).exp).mul_const (Real.sin (n * x))).const_mul (b n)
    convert this using 1 <;> simp <;> ring
  have hx1 : ∀ n : ℕ, ∀ z, HasDerivAt (fun z => b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * Real.sin (n * z)))
      (b n * Real.exp (-κ * (n : ℝ) ^ 2 * t) * (Real.cos (n * z) * n)) z := by
    intro n z
    have := ((((hasDerivAt_id' z).const_mul (n : ℝ)).sin).const_mul (Real.exp (-κ * (n : ℝ) ^ 2 * t))).const_mul (b n)
    convert this using 1 <;> simp <;> ring
  have hx2 : ∀ n : ℕ, HasDerivAt (fun z => b n * Real.exp (-κ * (n : ℝ) ^ 2 * t) * (Real.cos (n * z) * n))
      (b n * Real.exp (-κ * (n : ℝ) ^ 2 * t) * (-(Real.sin (n * x)) * n * n)) x := by
    intro n
    have := (((((hasDerivAt_id' x).const_mul (n : ℝ)).cos).mul_const (n : ℝ))).const_mul
      (b n * Real.exp (-κ * (n : ℝ) ^ 2 * t))
    convert this using 1 <;> simp <;> ring
  have e1 : (fun y => deriv (fun z => ∑ n ∈ Finset.range N,
      b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * Real.sin (n * z))) y) =
      fun y => ∑ n ∈ Finset.range N, b n * Real.exp (-κ * (n : ℝ) ^ 2 * t) * (Real.cos (n * y) * n) := by
    funext y
    exact (HasDerivAt.fun_sum (u := Finset.range N) fun n _ => hx1 n y).deriv
  rw [e1, (HasDerivAt.fun_sum (u := Finset.range N) fun n _ => ht n).deriv,
    (HasDerivAt.fun_sum (u := Finset.range N) fun n _ => hx2 n).deriv, Finset.mul_sum]
  exact Finset.sum_congr rfl fun n _ => by ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.heat_series_solves
-- test: de-heat-series-test.sage

/-- … and vanishes at `x = 0` and `x = π`. -/
theorem heat_series_boundary (N : ℕ) (b : ℕ → ℝ) (κ t : ℝ) :
    ∑ n ∈ Finset.range N, b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * Real.sin (n * 0)) = 0 ∧
      ∑ n ∈ Finset.range N, b n * (Real.exp (-κ * (n : ℝ) ^ 2 * t) * Real.sin (n * Real.pi)) = 0 := by
  simp [Real.sin_nat_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.heat_series_boundary
-- test: de-heat-boundary-test.sage

/-! ## exm-sep-wave (with prp-pde-solutions) -/

/-- `Tₙ = aₙ cos nct + bₙ sin nct` solves `T'' = -n²c²T`. -/
theorem wave_time_factor (n c a b : ℝ) (T : ℝ → ℝ) (hT : T = fun s => a * Real.cos (n * c * s) + b * Real.sin (n * c * s))
    (t : ℝ) : deriv (deriv T) t = -(n ^ 2 * c ^ 2) * T t := by
  veriphy_certified
    ( subst hT
      have d1 :
        deriv (fun s => a * cos (n * c * s) + b * sin (n * c * s)) = fun s =>
          n * c * b * (Real.cos (s * n * c)) - (n * c * a * (Real.sin (s * n * c))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => a * cos (n * c * s) + b * sin (n * c * s))
            ((n * c * b * (Real.cos (s * n * c)) - (n * c * a * (Real.sin (s * n * c))) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (n * c : ℝ)).cos).const_mul (a : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (n * c : ℝ)).sin).const_mul (b : ℝ))) using
            1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 :
        deriv (fun s => n * c * b * (Real.cos (s * n * c)) - (n * c * a * (Real.sin (s * n * c)))) = fun s =>
          -(n ^ 2) * (c ^ 2) * a * (Real.cos (s * n * c)) - ((n ^ 2) * (c ^ 2) * b * (Real.sin (s * n * c))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => n * c * b * (Real.cos (s * n * c)) - (n * c * a * (Real.sin (s * n * c))))
            ((-(n ^ 2) * (c ^ 2) * a * (Real.cos (s * n * c)) - ((n ^ 2) * (c ^ 2) * b * (Real.sin (s * n * c))) : ℝ))
            (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (n * c : ℝ)).cos).const_mul (n * c * b : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (n * c : ℝ)).sin).const_mul (-n * c * a : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | ((ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- The standing wave `sin kx cos kct` solves `u_tt = c²u_xx`. -/
theorem standing_wave (k c t x : ℝ) :
    deriv (fun s => deriv (fun r => Real.sin (k * x) * Real.cos (k * c * r)) s) t =
      c ^ 2 * deriv (fun s => deriv (fun r => Real.sin (k * r) * Real.cos (k * c * t)) s) x := by
  veriphy_certified
    ( have i0 :
        deriv (fun r => sin (k * x) * cos (k * c * r)) = fun d1 =>
          -c * k * (Real.sin (d1 * c * k)) * (Real.sin (k * x)) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k * x) * cos (k * c * r))
            ((-c * k * (Real.sin (d1 * c * k)) * (Real.sin (k * x)) : ℝ)) (d1) :=
          by
          convert ((((hasDerivAt_id' (d1 : ℝ)).const_mul (c * k : ℝ)).cos).const_mul (Real.sin (k * x) : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun d1 => -c * k * (Real.sin (d1 * c * k)) * (Real.sin (k * x))) s)
          ((-(c ^ 2) * (k ^ 2) * (Real.cos (c * k * t)) * (Real.sin (k * x)) : ℝ)) (t) :=
        by
        convert
          ((((hasDerivAt_id' (t : ℝ)).const_mul (c * k : ℝ)).sin).const_mul (-c * k * (Real.sin (k * x)) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h1.deriv]
      have i2 :
        deriv (fun r => sin (k * r) * cos (k * c * t)) = fun d1 => k * (Real.cos (c * k * t)) * (Real.cos (d1 * k)) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun r => sin (k * r) * cos (k * c * t)) ((k * (Real.cos (c * k * t)) * (Real.cos (d1 * k)) : ℝ))
            (d1) :=
          by
          convert ((((hasDerivAt_id' (d1 : ℝ)).const_mul (k : ℝ)).sin).const_mul (Real.cos (c * k * t) : ℝ)) using 1
          all_goals
            first
            | rfl
            | (funext y; ring_nf; done)
            |
              (first
                | ring1
                | (ring_nf; done)
                | (field_simp; done)
                | (field_simp; ring1)
                | (field_simp; ring_nf; done)
                | (norm_num; done))
        exact hd.deriv
      rw [i2]
      have h3 :
        HasDerivAt (fun s => (fun d1 => k * (Real.cos (c * k * t)) * (Real.cos (d1 * k))) s)
          ((-(k ^ 2) * (Real.cos (c * k * t)) * (Real.sin (k * x)) : ℝ)) (x) :=
        by
        convert ((((hasDerivAt_id' (x : ℝ)).const_mul (k : ℝ)).cos).const_mul (k * (Real.cos (c * k * t)) : ℝ)) using 1
        all_goals
          first
          | rfl
          | (funext y; ring_nf; done)
          |
            (first
              | ring1
              | (ring_nf; done)
              | (field_simp; done)
              | (field_simp; ring1)
              | (field_simp; ring_nf; done)
              | (norm_num; done))
      try rw [h3.deriv]
      beta_reduce
      first
      | ring1
      | (field_simp; ring1)
      | (field_simp; ring_nf; done)
      | (ring_nf; done)
      | (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)
      | (norm_num; done)
      |
        (ext i j;
          fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, dotProduct] <;>
            ring_nf))
  -- certificate: de-partials.sage

/-! ## exm-sep-laplace -/

/-- The Euler equation `r²R'' + rR' - n²R = 0` has the solution `R = rⁿ` (`r > 0`) … -/
theorem euler_rn (n r : ℝ) (hr : 0 < r) :
    r ^ 2 * deriv (fun s => deriv (fun ρ => ρ ^ n) s) r + r * deriv (fun ρ => ρ ^ n) r - n ^ 2 * r ^ n = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => ρ ^ n) (n * s ^ (n - 1)) s := fun s hs =>
    Real.hasDerivAt_rpow_const (Or.inl hs.ne')
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => n * s ^ (n - 1)) (n * ((n - 1) * r ^ (n - 1 - 1))) r :=
    (Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul n
  rw [h2.deriv]
  have e1 : r ^ (n - 1) = r ^ n / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (n - 1 - 1) = r ^ n / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  rw [e1, e2]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.euler_rn
-- test: de-euler-rn-test.sage

/-- … and `R = r⁻ⁿ` … -/
theorem euler_r_neg_n (n r : ℝ) (hr : 0 < r) :
    r ^ 2 * deriv (fun s => deriv (fun ρ => ρ ^ (-n)) s) r + r * deriv (fun ρ => ρ ^ (-n)) r - n ^ 2 * r ^ (-n) = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => ρ ^ (-n)) (-n * s ^ (-n - 1)) s := fun s hs =>
    Real.hasDerivAt_rpow_const (Or.inl hs.ne')
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv]
  have h2 : HasDerivAt (fun s : ℝ => -n * s ^ (-n - 1)) (-n * ((-n - 1) * r ^ (-n - 1 - 1))) r :=
    (Real.hasDerivAt_rpow_const (Or.inl hr.ne')).const_mul (-n)
  rw [h2.deriv]
  have e1 : r ^ (-n - 1) = r ^ (-n) / r := by rw [Real.rpow_sub hr, Real.rpow_one]
  have e2 : r ^ (-n - 1 - 1) = r ^ (-n) / r / r := by rw [Real.rpow_sub hr, Real.rpow_sub hr, Real.rpow_one]
  rw [e1, e2]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.euler_r_neg_n
-- test: de-euler-r-neg-n-test.sage

/-- … and for `n = 0` also `R = log r`. -/
theorem euler_log (r : ℝ) (hr : 0 < r) :
    r ^ 2 * deriv (fun s => deriv (fun ρ => Real.log ρ) s) r + r * deriv (fun ρ => Real.log ρ) r = 0 := by
  have hin : ∀ s : ℝ, 0 < s → HasDerivAt (fun ρ : ℝ => Real.log ρ) s⁻¹ s := fun s hs => Real.hasDerivAt_log hs.ne'
  rw [(deriv_eventually_pos hr hin).deriv_eq, (hin r hr).deriv, (hasDerivAt_inv hr.ne').deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.euler_log
-- test: de-euler-log-test.sage

/-! ## exr-sep-heat-series -/

/-- The partial sum `N = 5` of the cooling rod, `Σ_{n = 1, 3, 5} 8/(πn³) e^{-n²t} sin nx`, solves `u_t = u_xx`. -/
theorem rod_partial_sum (t x : ℝ) :
    deriv (fun s => 8 / Real.pi * (Real.exp (-s) * Real.sin x + Real.exp (-9 * s) * Real.sin (3 * x) / 27 +
        Real.exp (-25 * s) * Real.sin (5 * x) / 125)) t =
      deriv (fun y => deriv (fun z => 8 / Real.pi * (Real.exp (-t) * Real.sin z + Real.exp (-9 * t) * Real.sin (3 * z) / 27 +
        Real.exp (-25 * t) * Real.sin (5 * z) / 125)) y) x := by
  veriphy_certified
    ( have h0 :
        HasDerivAt
          (fun s => 8 / π * (rexp (-s) * sin x + rexp (-9 * s) * sin (3 * x) / 27 + rexp (-25 * s) * sin (5 * x) / 125))
          ((-8 *
                (3 * (Real.exp (-25 * t)) * (Real.sin (5 * x)) + (5 * (Real.exp (-9 * t)) * (Real.sin (3 * x))) +
                  (15 * (Real.exp (-t)) * (Real.sin x))) /
              (15 * Real.pi) :
            ℝ))
          (t) :=
        by
        refine
          (((((((((hasDerivAt_id' (t : ℝ)).const_mul (-25 : ℝ)).exp).const_mul (27 * (Real.sin (5 * x)) : ℝ)).add
                            ((((hasDerivAt_id' (t : ℝ)).const_mul (-9 : ℝ)).exp).const_mul
                              (125 * (Real.sin (3 * x)) : ℝ))).add
                        ((((hasDerivAt_id' (t : ℝ)).const_mul (-1 : ℝ)).exp).const_mul
                          (3375 * (Real.sin x) : ℝ))).const_mul
                    (8 / (3375 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (ring_nf; done)
      try rw [h0.deriv]
      have i1 :
        deriv
            (fun (z : ℝ) =>
              (8 / π * (rexp (-t) * sin z + rexp (-9 * t) * sin (3 * z) / 27 + rexp (-25 * t) * sin (5 * z) / 125) :
                ℝ)) =
          fun (d1 : ℝ) =>
          (8 *
                (225 * (Real.cos d1) * (Real.exp (-t)) + (25 * (Real.cos (3 * d1)) * (Real.exp (-9 * t))) +
                  (9 * (Real.cos (5 * d1)) * (Real.exp (-25 * t)))) /
              (225 * Real.pi) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (z : ℝ) =>
              (8 / π * (rexp (-t) * sin z + rexp (-9 * t) * sin (3 * z) / 27 + rexp (-25 * t) * sin (5 * z) / 125) : ℝ))
            ((8 *
                  (225 * (Real.cos d1) * (Real.exp (-t)) + (25 * (Real.cos (3 * d1)) * (Real.exp (-9 * t))) +
                    (9 * (Real.cos (5 * d1)) * (Real.exp (-25 * t)))) /
                (225 * Real.pi) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((hasDerivAt_id' (d1 : ℝ)).const_mul (5 : ℝ)).sin).const_mul (27 * (Real.exp (-25 * t)) : ℝ)).add
                              ((((hasDerivAt_id' (d1 : ℝ)).const_mul (3 : ℝ)).sin).const_mul
                                (125 * (Real.exp (-9 * t)) : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).sin).const_mul (3375 * (Real.exp (-t)) : ℝ))).const_mul
                      (8 / (3375 * Real.pi) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i1]
      have h2 :
        HasDerivAt
          (fun y =>
            (fun (d1 : ℝ) =>
                (8 *
                      (225 * (Real.cos d1) * (Real.exp (-t)) + (25 * (Real.cos (3 * d1)) * (Real.exp (-9 * t))) +
                        (9 * (Real.cos (5 * d1)) * (Real.exp (-25 * t)))) /
                    (225 * Real.pi) :
                  ℝ))
              y)
          ((-8 *
                (3 * (Real.exp (-25 * t)) * (Real.sin (5 * x)) + (5 * (Real.exp (-9 * t)) * (Real.sin (3 * x))) +
                  (15 * (Real.exp (-t)) * (Real.sin x))) /
              (15 * Real.pi) :
            ℝ))
          (x) :=
        by
        refine
          ((((((((hasDerivAt_id' (x : ℝ)).cos).const_mul (225 * (Real.exp (-t)) : ℝ)).add
                            ((((hasDerivAt_id' (x : ℝ)).const_mul (3 : ℝ)).cos).const_mul
                              (25 * (Real.exp (-9 * t)) : ℝ))).add
                        ((((hasDerivAt_id' (x : ℝ)).const_mul (5 : ℝ)).cos).const_mul
                          (9 * (Real.exp (-25 * t)) : ℝ))).const_mul
                    (8 / (225 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- … and satisfies the boundary conditions exactly. -/
theorem rod_partial_sum_boundary (t : ℝ) :
    8 / Real.pi * (Real.exp (-t) * Real.sin Real.pi + Real.exp (-9 * t) * Real.sin (3 * Real.pi) / 27 +
      Real.exp (-25 * t) * Real.sin (5 * Real.pi) / 125) = 0 := by
  veriphy_certified
    ( have h3 : Real.sin (3 * Real.pi) = 0 := by exact_mod_cast Real.sin_nat_mul_pi 3
      have h5 : Real.sin (5 * Real.pi) = 0 := by exact_mod_cast Real.sin_nat_mul_pi 5
      simp [h3, h5])
  -- certificate: de-rewrite.sage

/-! ## exr-sep-plucked -/

/-- Each term `sin nx cos nt` has period `2π/n`, which divides `2π`. -/
theorem plucked_periodic (n : ℕ) (t x : ℝ) :
    Real.sin (n * x) * Real.cos (n * (t + 2 * Real.pi)) = Real.sin (n * x) * Real.cos (n * t) := by
  rw [mul_add, show (n : ℝ) * (2 * Real.pi) = (n : ℕ) * (2 * Real.pi) by rfl, Real.cos_add_nat_mul_two_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.plucked_periodic
-- test: de-plucked-periodic-test.sage

/-- d'Alembert's form: `sin nx cos nt = ½[sin n(x - t) + sin n(x + t)]`. -/
theorem plucked_dalembert (n t x : ℝ) :
    Real.sin (n * x) * Real.cos (n * t) = (Real.sin (n * (x - t)) + Real.sin (n * (x + t))) / 2 := by
  veriphy_certified
    ( rw [show n * (x - t) = n * x - n * t by ring, show n * (x + t) = n * x + n * t by ring, Real.sin_sub,
        Real.sin_add]
      ring)
  -- certificate: de-rewrite.sage

/-! ## exr-sep-laplace-disk -/

/-- The Poisson kernel: `1 + 2Σₙ rⁿ cos nα = (1 - r²)/(1 - 2r cos α + r²)` for `0 ≤ r < 1`. -/
theorem poisson_kernel (r α : ℝ) (h0 : 0 ≤ r) (h1 : r < 1) :
    HasSum (fun n : ℕ => 2 * r ^ (n + 1) * Real.cos ((n + 1) * α))
      ((1 - r ^ 2) / (1 - 2 * r * Real.cos α + r ^ 2) - 1) := by
  set z : ℂ := r * Complex.exp (α * Complex.I)
  have hz : ‖z‖ < 1 := by
    simp only [z, norm_mul, Complex.norm_real, Real.norm_eq_abs, Complex.norm_exp_ofReal_mul_I, mul_one,
      abs_of_nonneg h0]
    exact h1
  have hg := hasSum_geometric_of_norm_lt_one hz
  have hg1 : HasSum (fun n : ℕ => z ^ (n + 1)) (z / (1 - z)) := by
    have := (hasSum_nat_add_iff' 1).mpr hg
    simp only [Finset.range_one, Finset.sum_singleton, pow_zero] at this
    convert this using 1
    have : (1 : ℂ) - z ≠ 0 := by
      intro h; rw [sub_eq_zero] at h; rw [← h] at hz; simp at hz
    field_simp
    ring
  have hre := (Complex.hasSum_re hg1).mul_left 2
  have hterm : ∀ n : ℕ, 2 * (z ^ (n + 1)).re = 2 * r ^ (n + 1) * Real.cos ((n + 1) * α) := by
    intro n
    simp only [z, mul_pow, ← Complex.exp_nat_mul]
    rw [show ((n + 1 : ℕ) : ℂ) * (α * Complex.I) = (((n + 1 : ℕ) : ℝ) * α : ℝ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I]
    rw [← Complex.ofReal_pow, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
    simp only [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, Complex.add_re, Complex.mul_im,
      Complex.I_re, Complex.I_im]
    push_cast
    ring
  simp_rw [hterm] at hre
  convert hre using 1
  have hden : 1 - 2 * r * Real.cos α + r ^ 2 ≠ 0 := by
    have : 0 < (1 - r) ^ 2 := by nlinarith
    nlinarith [Real.cos_le_one α, Real.neg_one_le_cos α]
  have hzre : z.re = r * Real.cos α := by simp [z, Complex.exp_ofReal_mul_I_re]
  have hzim : z.im = r * Real.sin α := by simp [z, Complex.exp_ofReal_mul_I_im]
  have hns : Complex.normSq (1 - z) = 1 - 2 * r * Real.cos α + r ^ 2 := by
    rw [Complex.normSq_apply]
    simp only [Complex.sub_re, Complex.one_re, Complex.sub_im, Complex.one_im, hzre, hzim]
    linear_combination r ^ 2 * Real.sin_sq_add_cos_sq α
  have key : r * Real.cos α * (1 - r * Real.cos α) + r * Real.sin α * (0 - r * Real.sin α) =
      r * Real.cos α - r ^ 2 := by
    linear_combination (-r ^ 2) * Real.sin_sq_add_cos_sq α
  rw [Complex.div_re, hns]
  simp only [Complex.sub_re, Complex.one_re, Complex.sub_im, Complex.one_im, hzre, hzim]
  rw [← add_div, key, div_sub_one hden, mul_div_assoc', div_eq_div_iff hden hden]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparationOfVariables.poisson_kernel
-- test: de-poisson-kernel-test.sage

end KaluzaKlein.DifferentialEquations.SeparationOfVariables

namespace KaluzaKlein.DifferentialEquations.PDEComputations

/-! ## prp-pde-solutions -/

/-- d'Alembert's formula with `φ = e^{-x²}`, `ψ = cos x`:
`u = ½[e^{-(x-ct)²} + e^{-(x+ct)²}] + (1/c) cos x sin ct` solves the wave equation … -/
theorem dalembert_example_solves (c t x : ℝ) (hc : c ≠ 0) :
    deriv (fun s => deriv (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2 +
        Real.cos x * Real.sin (c * r) / c) s) t -
      c ^ 2 * deriv (fun s => deriv (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2 +
        Real.cos r * Real.sin (c * t) / c) s) x = 0 := by
  -- the time derivative, at every point
  have ht1 : ∀ s : ℝ, HasDerivAt (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2 +
      Real.cos x * Real.sin (c * r) / c)
      ((Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s)) + Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) / 2 +
        Real.cos x * Real.cos (c * s)) s := by
    intro s
    have hq1 : HasDerivAt (fun r => (x - c * r) ^ 2) _ s := (((hasDerivAt_id' s).const_mul c).const_sub x).pow 2
    have hn1 : HasDerivAt (fun r => -(x - c * r) ^ 2) _ s := hq1.neg
    have hA : HasDerivAt (fun r => Real.exp (-(x - c * r) ^ 2)) _ s := hn1.exp
    have hq2 : HasDerivAt (fun r => (x + c * r) ^ 2) _ s := (((hasDerivAt_id' s).const_mul c).const_add x).pow 2
    have hn2 : HasDerivAt (fun r => -(x + c * r) ^ 2) _ s := hq2.neg
    have hB : HasDerivAt (fun r => Real.exp (-(x + c * r) ^ 2)) _ s := hn2.exp
    have hC : HasDerivAt (fun r => Real.cos x * Real.sin (c * r) / c) _ s :=
      (((hasDerivAt_id' s).const_mul c).sin.const_mul (Real.cos x)).div_const c
    have hAB : HasDerivAt (fun r => Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) _ s := hA.add hB
    have hAB2 : HasDerivAt (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2) _ s := hAB.div_const 2
    have h : HasDerivAt (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2 +
        Real.cos x * Real.sin (c * r) / c) _ s := hAB2.add hC
    refine h.congr_deriv ?_
    first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
  rw [show (fun s => deriv (fun r => (Real.exp (-(x - c * r) ^ 2) + Real.exp (-(x + c * r) ^ 2)) / 2 +
      Real.cos x * Real.sin (c * r) / c) s) = fun s =>
      (Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s)) + Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) / 2 +
        Real.cos x * Real.cos (c * s) from funext fun s => (ht1 s).deriv]
  have hq1 : HasDerivAt (fun s => (x - c * s) ^ 2) _ t := (((hasDerivAt_id' t).const_mul c).const_sub x).pow 2
  have hn1 : HasDerivAt (fun s => -(x - c * s) ^ 2) _ t := hq1.neg
  have hA : HasDerivAt (fun s => Real.exp (-(x - c * s) ^ 2)) _ t := hn1.exp
  have hq2 : HasDerivAt (fun s => (x + c * s) ^ 2) _ t := (((hasDerivAt_id' t).const_mul c).const_add x).pow 2
  have hn2 : HasDerivAt (fun s => -(x + c * s) ^ 2) _ t := hq2.neg
  have hB : HasDerivAt (fun s => Real.exp (-(x + c * s) ^ 2)) _ t := hn2.exp
  have hL1 : HasDerivAt (fun s => 2 * c * (x - c * s)) _ t := (((hasDerivAt_id' t).const_mul c).const_sub x).const_mul (2 * c)
  have hL2' : HasDerivAt (fun s => 2 * c * (x + c * s)) _ t := (((hasDerivAt_id' t).const_mul c).const_add x).const_mul (2 * c)
  have hL2 : HasDerivAt (fun s => -(2 * c * (x + c * s))) _ t := hL2'.neg
  have hP1 : HasDerivAt (fun s => Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s))) _ t := hA.mul hL1
  have hP2 : HasDerivAt (fun s => Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) _ t := hB.mul hL2
  have hS : HasDerivAt (fun s => Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s)) +
      Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) _ t := hP1.add hP2
  have hS2 : HasDerivAt (fun s => (Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s)) +
      Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) / 2) _ t := hS.div_const 2
  have hC : HasDerivAt (fun s => Real.cos x * Real.cos (c * s)) _ t := ((hasDerivAt_id' t).const_mul c).cos.const_mul (Real.cos x)
  have ht2 : HasDerivAt (fun s =>
      (Real.exp (-(x - c * s) ^ 2) * (2 * c * (x - c * s)) + Real.exp (-(x + c * s) ^ 2) * (-(2 * c * (x + c * s)))) / 2 +
        Real.cos x * Real.cos (c * s)) _ t := hS2.add hC
  rw [ht2.deriv]
  -- the space derivative, at every point
  have hx1 : ∀ y : ℝ, HasDerivAt (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2 +
      Real.cos r * Real.sin (c * t) / c)
      ((Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t))) + Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) / 2 +
        -Real.sin y * Real.sin (c * t) / c) y := by
    intro y
    have hq1 : HasDerivAt (fun r => (r - c * t) ^ 2) _ y := ((hasDerivAt_id' y).sub_const (c * t)).pow 2
    have hn1 : HasDerivAt (fun r => -(r - c * t) ^ 2) _ y := hq1.neg
    have hA : HasDerivAt (fun r => Real.exp (-(r - c * t) ^ 2)) _ y := hn1.exp
    have hq2 : HasDerivAt (fun r => (r + c * t) ^ 2) _ y := ((hasDerivAt_id' y).add_const (c * t)).pow 2
    have hn2 : HasDerivAt (fun r => -(r + c * t) ^ 2) _ y := hq2.neg
    have hB : HasDerivAt (fun r => Real.exp (-(r + c * t) ^ 2)) _ y := hn2.exp
    have hC : HasDerivAt (fun r => Real.cos r * Real.sin (c * t) / c) _ y :=
      ((Real.hasDerivAt_cos y).mul_const (Real.sin (c * t))).div_const c
    have hAB : HasDerivAt (fun r => Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) _ y := hA.add hB
    have hAB2 : HasDerivAt (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2) _ y := hAB.div_const 2
    have h : HasDerivAt (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2 +
        Real.cos r * Real.sin (c * t) / c) _ y := hAB2.add hC
    refine h.congr_deriv ?_
    first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
  rw [show (fun s => deriv (fun r => (Real.exp (-(r - c * t) ^ 2) + Real.exp (-(r + c * t) ^ 2)) / 2 +
      Real.cos r * Real.sin (c * t) / c) s) = fun y =>
      (Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t))) + Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) / 2 +
        -Real.sin y * Real.sin (c * t) / c from funext fun y => (hx1 y).deriv]
  have hq1 : HasDerivAt (fun y => (y - c * t) ^ 2) _ x := ((hasDerivAt_id' x).sub_const (c * t)).pow 2
  have hn1 : HasDerivAt (fun y => -(y - c * t) ^ 2) _ x := hq1.neg
  have hA : HasDerivAt (fun y => Real.exp (-(y - c * t) ^ 2)) _ x := hn1.exp
  have hq2 : HasDerivAt (fun y => (y + c * t) ^ 2) _ x := ((hasDerivAt_id' x).add_const (c * t)).pow 2
  have hn2 : HasDerivAt (fun y => -(y + c * t) ^ 2) _ x := hq2.neg
  have hB : HasDerivAt (fun y => Real.exp (-(y + c * t) ^ 2)) _ x := hn2.exp
  have hL1' : HasDerivAt (fun y => 2 * (y - c * t)) _ x := ((hasDerivAt_id' x).sub_const (c * t)).const_mul 2
  have hL1 : HasDerivAt (fun y => -(2 * (y - c * t))) _ x := hL1'.neg
  have hL2' : HasDerivAt (fun y => 2 * (y + c * t)) _ x := ((hasDerivAt_id' x).add_const (c * t)).const_mul 2
  have hL2 : HasDerivAt (fun y => -(2 * (y + c * t))) _ x := hL2'.neg
  have hP1 : HasDerivAt (fun y => Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t)))) _ x := hA.mul hL1
  have hP2 : HasDerivAt (fun y => Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) _ x := hB.mul hL2
  have hS : HasDerivAt (fun y => Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t))) +
      Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) _ x := hP1.add hP2
  have hS2 : HasDerivAt (fun y => (Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t))) +
      Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) / 2) _ x := hS.div_const 2
  have hsin : HasDerivAt (fun y => -Real.sin y) _ x := (Real.hasDerivAt_sin x).neg
  have hC : HasDerivAt (fun y => -Real.sin y * Real.sin (c * t) / c) _ x := (hsin.mul_const (Real.sin (c * t))).div_const c
  have hx2 : HasDerivAt (fun y =>
      (Real.exp (-(y - c * t) ^ 2) * (-(2 * (y - c * t))) + Real.exp (-(y + c * t) ^ 2) * (-(2 * (y + c * t)))) / 2 +
        -Real.sin y * Real.sin (c * t) / c) _ x := hS2.add hC
  rw [hx2.deriv]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PDEComputations.dalembert_example_solves
-- test: de-dalembert-example-solves-test.sage

/-- … with `u(0, x) = e^{-x²}` … -/
theorem dalembert_example_position (c x : ℝ) :
    (Real.exp (-(x - c * 0) ^ 2) + Real.exp (-(x + c * 0) ^ 2)) / 2 + Real.cos x * Real.sin (c * 0) / c =
      Real.exp (-x ^ 2) := by
  veriphy_certified (simp)
  -- certificate: de-rewrite.sage

/-- … and `u_t(0, x) = cos x`. -/
theorem dalembert_example_velocity (c x : ℝ) (hc : c ≠ 0) :
    deriv (fun s => (Real.exp (-(x - c * s) ^ 2) + Real.exp (-(x + c * s) ^ 2)) / 2 +
      Real.cos x * Real.sin (c * s) / c) 0 = Real.cos x := by
  have hq1 : HasDerivAt (fun s => (x - c * s) ^ 2) _ 0 := (((hasDerivAt_id' 0).const_mul c).const_sub x).pow 2
  have hn1 : HasDerivAt (fun s => -(x - c * s) ^ 2) _ 0 := hq1.neg
  have hA : HasDerivAt (fun s => Real.exp (-(x - c * s) ^ 2)) _ 0 := hn1.exp
  have hq2 : HasDerivAt (fun s => (x + c * s) ^ 2) _ 0 := (((hasDerivAt_id' 0).const_mul c).const_add x).pow 2
  have hn2 : HasDerivAt (fun s => -(x + c * s) ^ 2) _ 0 := hq2.neg
  have hB : HasDerivAt (fun s => Real.exp (-(x + c * s) ^ 2)) _ 0 := hn2.exp
  have hC : HasDerivAt (fun s => Real.cos x * Real.sin (c * s) / c) _ 0 :=
    (((hasDerivAt_id' 0).const_mul c).sin.const_mul (Real.cos x)).div_const c
  have hAB : HasDerivAt (fun s => Real.exp (-(x - c * s) ^ 2) + Real.exp (-(x + c * s) ^ 2)) _ 0 := hA.add hB
  have hAB2 : HasDerivAt (fun s => (Real.exp (-(x - c * s) ^ 2) + Real.exp (-(x + c * s) ^ 2)) / 2) _ 0 := hAB.div_const 2
  have h : HasDerivAt (fun s => (Real.exp (-(x - c * s) ^ 2) + Real.exp (-(x + c * s) ^ 2)) / 2 +
      Real.cos x * Real.sin (c * s) / c) _ 0 := hAB2.add hC
  rw [h.deriv]
  simp only [mul_zero, sub_zero, add_zero, Real.cos_zero, mul_one]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PDEComputations.dalembert_example_velocity
-- test: de-dalembert-example-velocity-test.sage

/-! ## exm-ppde-gaussian-integral -/

/-- Sage's raw form of `∫ K dx`: `√π/(√(πκt)√(1/(κt))) = 1` for `κ, t > 0`. -/
theorem kernel_integral_raw (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    Real.sqrt Real.pi / (Real.sqrt (Real.pi * κ * t) * Real.sqrt (1 / (κ * t))) = 1 := by
  veriphy_certified
    (rw [← Real.sqrt_mul (by positivity : (0 : ℝ) ≤ Real.pi * κ * t) (1 / (κ * t)),
        show Real.pi * κ * t * (1 / (κ * t)) = Real.pi by field_simp, div_self (Real.sqrt_pos.mpr Real.pi_pos).ne'])
  -- certificate: de-rewrite.sage

/-! ## exm-ppde-dispersion -/

/-- The residual `m² - ω² + |k|²` vanishes at `ω = √(|k|² + m²)`. -/
theorem dispersion_root (K m : ℝ) (h : 0 ≤ K + m ^ 2) : m ^ 2 - Real.sqrt (K + m ^ 2) ^ 2 + K = 0 := by
  veriphy_certified
    ( rw [Real.sq_sqrt h]
      ring)
  -- certificate: de-rewrite.sage

/-! ## exr-ppde-dalembert-standing -/

/-- `½[sin 2(x - ct) + sin 2(x + ct)] = sin 2x cos 2ct` … -/
theorem standing_from_position (c t x : ℝ) :
    (Real.sin (2 * (x - c * t)) + Real.sin (2 * (x + c * t))) / 2 = Real.sin (2 * x) * Real.cos (2 * (c * t)) := by
  veriphy_certified
    ( rw [show 2 * (x - c * t) = 2 * x - 2 * (c * t) by ring, show 2 * (x + c * t) = 2 * x + 2 * (c * t) by ring,
        Real.sin_sub, Real.sin_add]
      ring)
  -- certificate: de-rewrite.sage

/-- … and `(1/2c)∫_{x-ct}^{x+ct} sin 2s ds = (cos 2(x - ct) - cos 2(x + ct))/(4c)` … -/
theorem standing_velocity_integral (c t x : ℝ) (hc : c ≠ 0) :
    1 / (2 * c) * ∫ s in (x - c * t)..(x + c * t), Real.sin (2 * s) =
      (Real.cos (2 * (x - c * t)) - Real.cos (2 * (x + c * t))) / (4 * c) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (x - c * t : ℝ) (x + c * t : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ)) ((fun s => sin (2 * s)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ))
          (f' := fun s => sin (2 * s)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- … `= (1/2c) sin 2x sin 2ct`. -/
theorem standing_from_velocity (c t x : ℝ) (hc : c ≠ 0) :
    (Real.cos (2 * (x - c * t)) - Real.cos (2 * (x + c * t))) / (4 * c) =
      1 / (2 * c) * (Real.sin (2 * x) * Real.sin (2 * (c * t))) := by
  veriphy_certified
    ( rw [show 2 * (x - c * t) = 2 * x - 2 * (c * t) by ring, show 2 * (x + c * t) = 2 * x + 2 * (c * t) by ring,
        Real.cos_sub, Real.cos_add]
      field_simp
      ring)
  -- certificate: de-rewrite.sage

/-! ## exr-ppde-heat-step -/

/-- The step data give `u(t, 0) = ∫₀^∞ K(t, -y) dy = ½` (half of the Gaussian integral). -/
theorem step_midpoint (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    ∫ y in Set.Ioi (0 : ℝ), Real.exp (-(0 - y) ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t) = 1 / 2 := by
  have e : (fun y : ℝ => Real.exp (-(0 - y) ^ 2 / (4 * κ * t)) / Real.sqrt (4 * Real.pi * κ * t)) =
      fun y => (Real.sqrt (4 * Real.pi * κ * t))⁻¹ * Real.exp (-(1 / (4 * κ * t)) * y ^ 2) := by
    funext y; rw [div_eq_inv_mul]; congr 2; ring
  rw [e, MeasureTheory.integral_const_mul, integral_gaussian_Ioi,
    show Real.pi / (1 / (4 * κ * t)) = 4 * Real.pi * κ * t by field_simp]
  have : Real.sqrt (4 * Real.pi * κ * t) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PDEComputations.step_midpoint
-- test: de-step-midpoint-test.sage

/-! ## exr-ppde-wave-packet -/

/-- Linear dispersion `ω ≈ ω₀ + v(k - k₀)`: `kx - ωt = k(x - vt) - (ω₀ - vk₀)t`. -/
theorem packet_phase (k x t ω₀ v k₀ : ℝ) : k * x - (ω₀ + v * (k - k₀)) * t = k * (x - v * t) - (ω₀ - v * k₀) * t := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The phase factor has modulus one, so `|u(t, x)| = |u(0, x - vt)|`. -/
theorem packet_modulus (φ : ℝ) (w : ℂ) : ‖Complex.exp (φ * Complex.I) * w‖ = ‖w‖ := by
  rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PDEComputations.packet_modulus
-- test: de-packet-modulus-test.sage

/-- The spreading: `ω'' = m²/ω³` for `ω = √(k² + m²)` (`m ≠ 0`). -/
theorem packet_spreading (k m : ℝ) (hm : m ≠ 0) :
    deriv (fun s => deriv (fun r => Real.sqrt (r ^ 2 + m ^ 2)) s) k = m ^ 2 / Real.sqrt (k ^ 2 + m ^ 2) ^ 3 := by
  have hq : ∀ s : ℝ, s ^ 2 + m ^ 2 ≠ 0 := fun s => by positivity
  have hsq : ∀ s : ℝ, HasDerivAt (fun r => Real.sqrt (r ^ 2 + m ^ 2)) (2 * s / (2 * Real.sqrt (s ^ 2 + m ^ 2))) s := by
    intro s
    have hp : HasDerivAt (fun r : ℝ => r ^ 2 + m ^ 2) (2 * s) s := by simpa using (hasDerivAt_pow 2 s).add_const (m ^ 2)
    exact hp.sqrt (hq s)
  rw [show (fun s => deriv (fun r => Real.sqrt (r ^ 2 + m ^ 2)) s) = fun s => 2 * s / (2 * Real.sqrt (s ^ 2 + m ^ 2)) from
    funext fun s => (hsq s).deriv]
  have hS : Real.sqrt (k ^ 2 + m ^ 2) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  have h2 : HasDerivAt (fun s => 2 * s / (2 * Real.sqrt (s ^ 2 + m ^ 2))) _ k :=
    ((hasDerivAt_id' k).const_mul 2).div ((hsq k).const_mul 2) (by positivity)
  rw [h2.deriv]
  have hs2 : Real.sqrt (k ^ 2 + m ^ 2) ^ 2 = k ^ 2 + m ^ 2 := Real.sq_sqrt (by positivity)
  have key : m ^ 2 / Real.sqrt (k ^ 2 + m ^ 2) ^ 3 = (Real.sqrt (k ^ 2 + m ^ 2) ^ 2 - k ^ 2) / Real.sqrt (k ^ 2 + m ^ 2) ^ 3 := by
    rw [hs2]; ring
  rw [key]
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PDEComputations.packet_spreading
-- test: de-packet-spreading-test.sage

end KaluzaKlein.DifferentialEquations.PDEComputations

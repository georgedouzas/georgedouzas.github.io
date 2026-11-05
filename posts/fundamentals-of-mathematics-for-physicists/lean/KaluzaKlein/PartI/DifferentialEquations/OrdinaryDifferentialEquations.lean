import Mathlib
import Veriphy.Kit.Levels

/-!
# Differential equations, part "Ordinary differential equations"

The claims of the part "Ordinary differential equations" of `tutorial/part-i-05-differential-equations.qmd`, one
namespace per section: solutions checked by substitution (exponential decay, the oscillator, the logistic curve,
Euler's equation), initial value problems, Picard-Lindelöf uniqueness and the Picard iterates, separable and linear
first-order equations, phase portraits and conserved quantities, and the closed forms of the worked computations.

An ordinary differential equation is modelled by functions `ℝ → ℝ` and `deriv`; a statement "y solves the equation"
is the pointwise identity at every point (or on the stated interval). Linearity of an equation is the superposition
identity of its jet expression `F(x, y, y', …)` in the jet variables, and non-linearity the failure of a linear law.
-/

open Real Filter Topology

namespace KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations

/-! ## exm-ode-decay -/

/-- `y = C e^{-kx}` solves `y' = -k y`. -/
theorem decay_solves (k C : ℝ) (y : ℝ → ℝ) (hy : y = fun s => C * Real.exp (-k * s)) (x : ℝ) :
    deriv y x = -k * y x := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun s => C * rexp (-k * s)) = fun s => -k * C * (Real.exp (-s * k)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => C * rexp (-k * s)) ((-k * C * (Real.exp (-s * k)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (-k : ℝ)).exp).const_mul (C : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-! ## exm-ode-newton -/

/-- `q = cos ω₀t` solves `q̈ + ω₀² q = 0`. -/
theorem spring_solves (ω₀ : ℝ) (q : ℝ → ℝ) (hq : q = fun s => Real.cos (ω₀ * s)) (t : ℝ) :
    deriv (deriv q) t + ω₀ ^ 2 * q t = 0 := by
  veriphy_certified
    ( subst hq
      have d1 : deriv (fun s => cos (ω₀ * s)) = fun s => -ω₀ * (Real.sin (s * ω₀)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => cos (ω₀ * s)) ((-ω₀ * (Real.sin (s * ω₀)) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).const_mul (ω₀ : ℝ)).cos) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 : deriv (fun s => -ω₀ * (Real.sin (s * ω₀))) = fun s => -(ω₀ ^ 2) * (Real.cos (s * ω₀)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => -ω₀ * (Real.sin (s * ω₀))) ((-(ω₀ ^ 2) * (Real.cos (s * ω₀)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (ω₀ : ℝ)).sin).const_mul (-ω₀ : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- Newton's law `m q̈ = -κ q` with `m ω₀² = κ` is `q̈ + ω₀² q = 0` (times `m`). -/
theorem spring_equation (m κ ω₀ q a : ℝ) (hω : m * ω₀ ^ 2 = κ) (h : m * a = -κ * q) :
    m * (a + ω₀ ^ 2 * q) = 0 := by
  veriphy_certified (linear_combination (q) * hω + h)
  -- certificate: de-poly.sage

/-- The pendulum term `sin θ` is not linear: `sin` is not additive (`sin(π/2 + π/2) ≠ 2 sin(π/2)`). -/
theorem sin_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = Real.sin x := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (2 : ℝ) (Real.pi / 2)
      rw [hf, hf, smul_eq_mul, smul_eq_mul, show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring] at h
      norm_num at h)
  -- certificate: de-nonlinear-sin.sage

/-! ## exm-ode-logistic -/

/-- The constant `y = 0` solves `y' = y(1 - y)`. -/
theorem logistic_zero_solves (y : ℝ → ℝ) (hy : y = fun _ => 0) (x : ℝ) : deriv y x = y x * (1 - y x) := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun (x : ℝ) => (0 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (x : ℝ) => (0 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- The constant `y = 1` solves `y' = y(1 - y)`. -/
theorem logistic_one_solves (y : ℝ → ℝ) (hy : y = fun _ => 1) (x : ℝ) : deriv y x = y x * (1 - y x) := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun (x : ℝ) => (1 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (x : ℝ) => (1 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `y = 1/(1 + e^{-x})` solves `y' = y(1 - y)`. -/
theorem logistic_solves (y : ℝ → ℝ) (hy : y = fun s => 1 / (1 + Real.exp (-s))) (x : ℝ) :
    deriv y x = y x * (1 - y x) := by
  subst hy
  have hpos : 1 + Real.exp (-x) ≠ 0 := by positivity
  simp only [one_div]
  have h : HasDerivAt (fun s => (1 + Real.exp (-s))⁻¹) _ x := (((hasDerivAt_neg x).exp).const_add 1).inv hpos
  rw [h.deriv]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.logistic_solves
-- test: de-logistic-solves-test.sage

/-- The logistic curves `1/(1 + C e^{-x})`, `C > 0`, rise from `0` (at `-∞`) to `1` (at `+∞`). -/
theorem logistic_limits (C : ℝ) (hC : 0 < C) :
    Tendsto (fun x => 1 / (1 + C * Real.exp (-x))) atTop (𝓝 1) ∧
      Tendsto (fun x => 1 / (1 + C * Real.exp (-x))) atBot (𝓝 0) := by
  constructor
  · have h : Tendsto (fun x => 1 + C * Real.exp (-x)) atTop (𝓝 (1 + C * 0)) :=
      tendsto_const_nhds.add (tendsto_const_nhds.mul (Real.tendsto_exp_neg_atTop_nhds_zero))
    have := h.inv₀ (by norm_num : (1 + C * 0 : ℝ) ≠ 0)
    simpa [one_div] using this
  · have h1 : Tendsto (fun x => C * Real.exp (-x)) atBot atTop :=
      (Real.tendsto_exp_atTop.comp tendsto_neg_atBot_atTop).const_mul_atTop hC
    have h2 : Tendsto (fun x => 1 + C * Real.exp (-x)) atBot atTop := tendsto_atTop_add_const_left _ 1 h1
    exact h2.inv_tendsto_atTop.congr (fun x => by simp [one_div])
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.logistic_limits
-- test: de-logistic-limits-test.sage

/-! ## exr-ode-classify -/

/-- `y'' + x y' = sin x` is linear: its left side obeys superposition in the jet variables `(y', y'')`. -/
theorem classify_linear_first (x a b p₁ p₂ q₁ q₂ : ℝ) :
    (a * p₂ + b * q₂) + x * (a * p₁ + b * q₁) = a * (p₂ + x * p₁) + b * (q₂ + x * q₁) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `x² y'' - 2y = 0` is linear and homogeneous: superposition in the jet variables `(y, y'')`. -/
theorem classify_linear_euler (x a b p₀ p₂ q₀ q₂ : ℝ) :
    x ^ 2 * (a * p₂ + b * q₂) - 2 * (a * p₀ + b * q₀) = a * (x ^ 2 * p₂ - 2 * p₀) + b * (x ^ 2 * q₂ - 2 * q₀) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `(y')² + y` and `y y'` are not linear: on the jet line `y = y' = p` their non-linear parts are `p²`, which is
not a linear function. -/
theorem square_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ p, f p = p ^ 2 := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (2 : _) (1 : _)
      rw [hf, hf] at h
      norm_num at h)
  -- certificate: de-nonlinearity.sage

/-- `y = x²` solves `x² y'' - 2y = 0`. -/
theorem euler_sq_solves (y : ℝ → ℝ) (hy : y = fun s => s ^ 2) (x : ℝ) :
    x ^ 2 * deriv (deriv y) x - 2 * y x = 0 := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun (s : ℝ) => (s ^ 2 : ℝ)) = fun (s : ℝ) => (2 * s : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (s ^ 2 : ℝ)) ((2 * s : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).pow 2)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (2 * s : ℝ)) = fun (s : ℝ) => (2 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (2 * s : ℝ)) ((2 : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `y = 1/x` solves `x² y'' - 2y = 0` for `x ≠ 0`. -/
theorem euler_inv_solves (x : ℝ) (hx : x ≠ 0) :
    x ^ 2 * deriv (deriv fun s => s⁻¹) x - 2 * x⁻¹ = 0 := by
  have h1 : deriv (fun s : ℝ => s⁻¹) = fun s => -(s ^ 2)⁻¹ := by
    funext s; exact deriv_inv
  have h2 : HasDerivAt (fun s : ℝ => -(s ^ 2)⁻¹) (2 * x⁻¹ ^ 3) x := by
    have := ((hasDerivAt_pow 2 x).inv (pow_ne_zero 2 hx)).neg
    convert this using 1
    field_simp
    ring
  rw [h1, h2.deriv]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.euler_inv_solves
-- test: de-euler-inv-test.sage

/-! ## exr-ode-verify -/

/-- `y = C₁ cos x + C₂ sin x` solves `y'' + y = 0` for all constants. -/
theorem cos_sin_solves (C₁ C₂ : ℝ) (y : ℝ → ℝ) (hy : y = fun s => C₁ * Real.cos s + C₂ * Real.sin s) (x : ℝ) :
    deriv (deriv y) x + y x = 0 := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun s => C₁ * cos s + C₂ * sin s) = fun s => C₂ * (Real.cos s) - (C₁ * (Real.sin s)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => C₁ * cos s + C₂ * sin s) ((C₂ * (Real.cos s) - (C₁ * (Real.sin s)) : ℝ)) (s) :=
          by
          convert
            ((((hasDerivAt_id' (s : ℝ)).cos).const_mul (C₁ : ℝ)).add
              (((hasDerivAt_id' (s : ℝ)).sin).const_mul (C₂ : ℝ))) using
            1
          all_goals (ring1)
        exact hd.deriv
      have d2 :
        deriv (fun s => C₂ * (Real.cos s) - (C₁ * (Real.sin s))) = fun s => -C₁ * (Real.cos s) - (C₂ * (Real.sin s)) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => C₂ * (Real.cos s) - (C₁ * (Real.sin s))) ((-C₁ * (Real.cos s) - (C₂ * (Real.sin s)) : ℝ))
            (s) :=
          by
          convert
            ((((hasDerivAt_id' (s : ℝ)).cos).const_mul (C₂ : ℝ)).add
              (((hasDerivAt_id' (s : ℝ)).sin).const_mul (-C₁ : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv
      simp only [d1, d2]
      ring1)
  -- certificate: de-ode.sage

/-- `y = e^{x²}` leaves the non-zero residual `y' - 2y = (2x - 2) e^{x²}` in `y' = 2y`. -/
theorem exp_sq_residual (x : ℝ) :
    deriv (fun s => Real.exp (s ^ 2)) x - 2 * Real.exp (x ^ 2) = (2 * x - 2) * Real.exp (x ^ 2) := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => rexp (s ^ 2)) ((2 * x * (Real.exp (x ^ 2)) : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).exp) using 1
        all_goals (simp only [Pi.pow_apply]; ring1)
      try rw [h0.deriv]
      beta_reduce
      ring1)
  -- certificate: de-partials.sage

/-! ## exr-ode-first-order-system -/

/-- An `(n+1)`-th order equation `y^{(n+1)} = f(x, y, …, y^{(n)})` gives the first-order system `uᵢ' = uᵢ₊₁`,
`uₙ' = f(x, u)` for `uᵢ = y^{(i)}`. -/
theorem reduction_to_first_order (n : ℕ) (f : ℝ → (Fin (n + 1) → ℝ) → ℝ) (y : ℝ → ℝ)
    (hy : ∀ x, iteratedDeriv (n + 1) y x = f x (fun i => iteratedDeriv i y x)) (x : ℝ) :
    (∀ i : Fin n, deriv (iteratedDeriv i y) x = iteratedDeriv (i + 1) y x) ∧
      deriv (iteratedDeriv n y) x = f x (fun i => iteratedDeriv i y x) := by
  refine ⟨fun i => by rw [← iteratedDeriv_succ], ?_⟩
  rw [← iteratedDeriv_succ]
  exact hy x
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.reduction_to_first_order
-- test: de-reduction-test.sage

/-- Conversely, the first component of a solution of the system solves the `(n+1)`-th order equation. -/
theorem first_order_to_reduction (n : ℕ) (f : ℝ → (Fin (n + 1) → ℝ) → ℝ) (u : Fin (n + 1) → ℝ → ℝ)
    (hu : ∀ i : Fin n, deriv (u i.castSucc) = u i.succ)
    (hlast : ∀ x, deriv (u (Fin.last n)) x = f x (fun i => u i x)) (x : ℝ) :
    iteratedDeriv (n + 1) (u 0) x = f x (fun i => u i x) := by
  have key : ∀ k (hk : k < n + 1), iteratedDeriv k (u 0) = u ⟨k, hk⟩ := by
    intro k
    induction k with
    | zero => intro hk; rw [iteratedDeriv_zero]; rfl
    | succ k ih =>
      intro hk
      rw [iteratedDeriv_succ, ih (by omega)]
      exact hu ⟨k, by omega⟩
  rw [iteratedDeriv_succ, key n (by omega)]
  exact hlast x
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.first_order_to_reduction
-- test: de-reduction-converse-test.sage

/-- `y''' + y' = 0` is `u' = A u` with `A` the matrix with rows `(0, 1, 0)`, `(0, 0, 1)`, `(0, -1, 0)`:
`A (y, y', y'')ᵀ = (y', y'', -y')ᵀ`. -/
theorem third_order_matrix (a b c : ℝ) :
    !![(0 : ℝ), 1, 0; 0, 0, 1; 0, -1, 0] * !![a; b; c] = !![b; c; -b] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1)
  -- certificate: la-matrix.sage

end KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations

namespace KaluzaKlein.DifferentialEquations.InitialValueProblems

/-! ## exm-ivp-exponential -/

/-- Every `C e^x` solves `y' = y`. -/
theorem exp_family_solves (C : ℝ) (y : ℝ → ℝ) (hy : y = fun s => C * Real.exp s) (x : ℝ) : deriv y x = y x := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun (s : ℝ) => (C * rexp s : ℝ)) = fun (s : ℝ) => (C * (Real.exp s) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (C * rexp s : ℝ)) ((C * (Real.exp s) : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).exp).const_mul (C : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals (rfl))
  -- certificate: de-ode.sage

/-- The initial value of `C e^x` at `0` is `C`, so `y(0) = 1` selects `C = 1`. -/
theorem exp_family_initial (C : ℝ) (y : ℝ → ℝ) (hy : y = fun s => C * Real.exp s) : y 0 = C := by
  veriphy_certified
    ( subst hy
      (norm_num; done))
  -- certificate: de-ode.sage

/-- `y = e^x` solves the initial value problem `y' = y`, `y(0) = 1`. -/
theorem exp_ivp : ∃ y : ℝ → ℝ, (∀ t, deriv y t = y t) ∧ y 0 = 1 := by
  veriphy_certified
    ( have d1 : deriv (fun (s : ℝ) => (Real.exp s : ℝ)) = fun (s : ℝ) => (Real.exp s : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (Real.exp s : ℝ)) ((Real.exp s : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).exp)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      exact
        ⟨fun (s : ℝ) => (Real.exp s : ℝ), fun t => by (simp only [d1]; all_goals (rfl)), by ((norm_num [d1]; done))⟩)
  -- certificate: de-ode.sage

/-! ## exm-ivp-oscillator -/

/-- `q = cos t` solves `q̈ + q = 0`, `q(0) = 1`, `q̇(0) = 0`. -/
theorem osc_cos_ivp : ∃ q : ℝ → ℝ, (∀ t, deriv (deriv q) t + q t = 0) ∧ q 0 = 1 ∧ deriv q 0 = 0 := by
  veriphy_certified
    ( have d1 : deriv (fun s => Real.cos s) = fun s => -(Real.sin s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => Real.cos s) ((-(Real.sin s) : ℝ)) (s) :=
          by
          convert ((hasDerivAt_id' (s : ℝ)).cos) using 1
          all_goals (ring1)
        exact hd.deriv
      have d2 : deriv (fun s => -(Real.sin s)) = fun s => -(Real.cos s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => -(Real.sin s)) ((-(Real.cos s) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).sin).const_mul (-1 : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      exact
        ⟨fun s => Real.cos s, fun t => by (simp only [d1, d2]; ring1), by ((norm_num [d1, d2]; done)), by
          ((norm_num [d1, d2]; done))⟩)
  -- certificate: de-ode.sage

/-- `q = sin t` solves `q̈ + q = 0`, `q(0) = 0`, `q̇(0) = 1`. -/
theorem osc_sin_ivp : ∃ q : ℝ → ℝ, (∀ t, deriv (deriv q) t + q t = 0) ∧ q 0 = 0 ∧ deriv q 0 = 1 := by
  veriphy_certified
    ( have d1 : deriv (fun s => Real.sin s) = fun s => Real.cos s :=
        by
        funext s
        have hd : HasDerivAt (fun s => Real.sin s) ((Real.cos s : ℝ)) (s) :=
          by
          convert ((hasDerivAt_id' (s : ℝ)).sin) using 1
          all_goals (ring1)
        exact hd.deriv
      have d2 : deriv (fun s => Real.cos s) = fun s => -(Real.sin s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => Real.cos s) ((-(Real.sin s) : ℝ)) (s) :=
          by
          convert ((hasDerivAt_id' (s : ℝ)).cos) using 1
          all_goals (ring1)
        exact hd.deriv
      exact
        ⟨fun s => Real.sin s, fun t => by (simp only [d1, d2]; ring1), by ((norm_num [d1, d2]; done)), by
          ((norm_num [d1, d2]; done))⟩)
  -- certificate: de-ode.sage

/-- The two constants of `C₁ cos t + C₂ sin t` are its initial position and velocity: `q̇(0) = C₂`. -/
theorem osc_general_velocity (C₁ C₂ : ℝ) (q : ℝ → ℝ) (hq : q = fun s => C₁ * Real.cos s + C₂ * Real.sin s) :
    deriv q 0 = C₂ := by
  veriphy_certified
    ( subst hq
      have d1 :
        deriv (fun (s : ℝ) => (C₁ * cos s + C₂ * sin s : ℝ)) = fun (s : ℝ) =>
          (C₂ * (Real.cos s) - (C₁ * (Real.sin s)) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (C₁ * cos s + C₂ * sin s : ℝ)) ((C₂ * (Real.cos s) - (C₁ * (Real.sin s)) : ℝ))
            (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).cos).const_mul (C₁ : ℝ)).add
                      (((hasDerivAt_id' (s : ℝ)).sin).const_mul (C₂ : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-- … and `q(0) = C₁`. -/
theorem osc_general_position (C₁ C₂ : ℝ) (q : ℝ → ℝ) (hq : q = fun s => C₁ * Real.cos s + C₂ * Real.sin s) :
    q 0 = C₁ := by
  veriphy_certified
    ( subst hq
      (norm_num; done))
  -- certificate: de-ode.sage

/-! ## exm-ivp-pendulum -/

/-- The pendulum force is Lipschitz with constant `1`: `|sin a - sin b| ≤ |a - b|`, the hypothesis of
Picard-Lindelöf for the pendulum. -/
theorem pendulum_lipschitz (a b : ℝ) : |Real.sin a - Real.sin b| ≤ |a - b| :=
  Real.abs_sin_sub_sin_le a b
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.InitialValueProblems.pendulum_lipschitz
-- test: de-pendulum-lipschitz-test.sage

/-- `θ₀ = π` (balanced upright) gives the constant solution of `θ̈ + sin θ = 0`. -/
theorem pendulum_upright (θ : ℝ → ℝ) (hθ : θ = fun _ => Real.pi) (t : ℝ) :
    deriv (deriv θ) t + Real.sin (θ t) = 0 := by
  veriphy_certified
    ( subst hθ
      have d1 : deriv (fun (x : ℝ) => (π : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (x : ℝ) => (π : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (Real.pi : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (0 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (0 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-- `θ₀ = 0` (hanging) gives the constant solution of `θ̈ + sin θ = 0`. -/
theorem pendulum_hanging (θ : ℝ → ℝ) (hθ : θ = fun _ => 0) (t : ℝ) :
    deriv (deriv θ) t + Real.sin (θ t) = 0 := by
  veriphy_certified
    ( subst hθ
      have d1 : deriv (fun (x : ℝ) => (0 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (x : ℝ) => (0 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (0 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (0 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-! ## exr-ivp-decay -/

/-- `y = 5 e^{-2x}` solves `y' = -2y`, `y(0) = 5`. -/
theorem decay_ivp : ∃ y : ℝ → ℝ, (∀ t, deriv y t = -2 * y t) ∧ y 0 = 5 := by
  veriphy_certified
    ( have d1 : deriv (fun s => 5 * (Real.exp (-2 * s))) = fun s => -10 * (Real.exp (-2 * s)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => 5 * (Real.exp (-2 * s))) ((-10 * (Real.exp (-2 * s)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (-2 : ℝ)).exp).const_mul (5 : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv
      exact ⟨fun s => 5 * (Real.exp (-2 * s)), fun t => by (simp only [d1]; ring1), by ((norm_num [d1]; done))⟩)
  -- certificate: de-ode.sage

/-- The amount has halved at `x = ½ log 2`. -/
theorem decay_half_time : 5 * Real.exp (-2 * (Real.log 2 / 2)) = 5 / 2 := by
  veriphy_certified
    ( rw [show -2 * (Real.log 2 / 2) = -Real.log 2 by ring, Real.exp_neg, Real.exp_log (by norm_num)]
      norm_num)
  -- certificate: de-rewrite.sage

/-! ## exr-ivp-second-order -/

/-- `q = cos 2t + sin 2t` solves `q̈ + 4q = 0`. -/
theorem ivp2_solves (q : ℝ → ℝ) (hq : q = fun s => Real.cos (2 * s) + Real.sin (2 * s)) (t : ℝ) :
    deriv (deriv q) t + 4 * q t = 0 := by
  veriphy_certified
    ( subst hq
      have d1 :
        deriv (fun s => cos (2 * s) + sin (2 * s)) = fun s => 2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => cos (2 * s) + sin (2 * s)) ((2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) : ℝ))
            (s) :=
          by
          convert
            ((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).cos).add
              (((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).sin)) using
            1
          all_goals (ring1)
        exact hd.deriv
      have d2 :
        deriv (fun s => 2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s)))) = fun s =>
          -4 * (Real.cos (2 * s)) - (4 * (Real.sin (2 * s))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => 2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))))
            ((-4 * (Real.cos (2 * s)) - (4 * (Real.sin (2 * s))) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).cos).const_mul (2 : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).sin).const_mul (-2 : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv
      simp only [d1, d2]
      ring1)
  -- certificate: de-ode.sage

/-- … with `q(0) = 1`. -/
theorem ivp2_position (q : ℝ → ℝ) (hq : q = fun s => Real.cos (2 * s) + Real.sin (2 * s)) : q 0 = 1 := by
  veriphy_certified
    ( subst hq
      (norm_num; done))
  -- certificate: de-ode.sage

/-- … and `q̇(0) = 2`. -/
theorem ivp2_velocity (q : ℝ → ℝ) (hq : q = fun s => Real.cos (2 * s) + Real.sin (2 * s)) : deriv q 0 = 2 := by
  veriphy_certified
    ( subst hq
      have d1 :
        deriv (fun (s : ℝ) => (cos (2 * s) + sin (2 * s) : ℝ)) = fun (s : ℝ) =>
          (2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (cos (2 * s) + sin (2 * s) : ℝ))
            ((2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).cos).add
                      (((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).sin))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1]
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-- The initial value problem `q̈ + 4q = 0`, `q(0) = 1`, `q̇(0) = 2` has a solution (Sage's `desolve`). -/
theorem ivp2_exists : ∃ q : ℝ → ℝ, (∀ t, deriv (deriv q) t + 4 * q t = 0) ∧ q 0 = 1 ∧ deriv q 0 = 2 := by
  veriphy_certified
    ( have d1 :
        deriv (fun s => Real.cos (2 * s) + (Real.sin (2 * s))) = fun s =>
          2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => Real.cos (2 * s) + (Real.sin (2 * s)))
            ((2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))) : ℝ)) (s) :=
          by
          convert
            ((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).cos).add
              (((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).sin)) using
            1
          all_goals (ring1)
        exact hd.deriv
      have d2 :
        deriv (fun s => 2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s)))) = fun s =>
          -4 * (Real.cos (2 * s)) - (4 * (Real.sin (2 * s))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => 2 * (Real.cos (2 * s)) - (2 * (Real.sin (2 * s))))
            ((-4 * (Real.cos (2 * s)) - (4 * (Real.sin (2 * s))) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).cos).const_mul (2 : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (2 : ℝ)).sin).const_mul (-2 : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv
      exact
        ⟨fun s => Real.cos (2 * s) + (Real.sin (2 * s)), fun t => by (simp only [d1, d2]; ring1), by
          ((norm_num [d1, d2]; done)), by ((norm_num [d1, d2]; done))⟩)
  -- certificate: de-ode.sage

/-! ## exr-ivp-non-unique -/

/-- `y = 0` solves `y' = 2√|y|` with `y(0) = 0`. -/
theorem nonunique_zero (y : ℝ → ℝ) (hy : y = fun _ => 0) (x : ℝ) : deriv y x = 2 * Real.sqrt |y x| := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun (x : ℝ) => (0 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (x : ℝ) => (0 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `y = x²` solves `y' = 2√|y|` for `x ≥ 0`, with `y(0) = 0`: a second solution. -/
theorem nonunique_sq (x : ℝ) (hx : 0 ≤ x) : deriv (fun s => s ^ 2) x = 2 * Real.sqrt |x ^ 2| := by
  rw [abs_of_nonneg (by positivity), Real.sqrt_sq hx]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.InitialValueProblems.nonunique_sq
-- test: de-nonunique-sq-test.sage

/-- `y = -x²` solves `y' = 2√|y|` for `x ≤ 0`. -/
theorem nonunique_neg_sq (x : ℝ) (hx : x ≤ 0) : deriv (fun s => -s ^ 2) x = 2 * Real.sqrt |-x ^ 2| := by
  rw [abs_neg, abs_of_nonneg (by positivity), Real.sqrt_sq_eq_abs, abs_of_nonpos hx]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.InitialValueProblems.nonunique_neg_sq
-- test: de-nonunique-neg-sq-test.sage

/-- The glued functions `y = (max (x - a) 0)²` solve `y' = 2√|y|` everywhere, for every `a`: infinitely many
solutions with `y(0) = 0` when `a ≥ 0`. -/
theorem nonunique_glued (a x : ℝ) :
    HasDerivAt (fun s => (max (s - a) 0) ^ 2) (2 * Real.sqrt |(max (x - a) 0) ^ 2|) x := by
  rw [abs_of_nonneg (by positivity), Real.sqrt_sq (le_max_right _ _)]
  rcases lt_trichotomy x a with h | h | h
  · have hev : (fun s => (max (s - a) 0) ^ 2) =ᶠ[𝓝 x] fun _ => 0 := by
      filter_upwards [Iio_mem_nhds h] with s hs
      have hs' : s < a := hs
      simp [max_eq_right (by linarith : s - a ≤ 0)]
    rw [max_eq_right (by linarith : x - a ≤ 0)]
    simpa using (hasDerivAt_const x (0 : ℝ)).congr_of_eventuallyEq hev
  · rw [h, sub_self, max_self, mul_zero, hasDerivAt_iff_isLittleO]
    simp only [sub_self, max_self, smul_zero, sub_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
      zero_pow]
    refine Asymptotics.IsBigO.trans_isLittleO (g := fun s => ‖s - a‖ ^ 2) ?_
      (Asymptotics.isLittleO_pow_sub_sub a one_lt_two)
    refine Asymptotics.IsBigO.of_bound 1 (Eventually.of_forall fun s => ?_)
    have h1 : max (s - a) 0 ≤ |s - a| := max_le (le_abs_self _) (abs_nonneg _)
    have h2 : 0 ≤ max (s - a) 0 := le_max_right _ _
    simp only [Real.norm_eq_abs, one_mul, abs_pow, abs_abs, abs_of_nonneg h2]
    exact pow_le_pow_left₀ h2 h1 2
  · have hev : (fun s => (max (s - a) 0) ^ 2) =ᶠ[𝓝 x] fun s => (s - a) ^ 2 := by
      filter_upwards [Ioi_mem_nhds h] with s hs
      have hs' : a < s := hs
      rw [max_eq_left (by linarith : 0 ≤ s - a)]
    rw [max_eq_left (by linarith : 0 ≤ x - a)]
    have := ((hasDerivAt_id' x).sub_const a).pow 2
    exact (this.congr_of_eventuallyEq hev).congr_deriv (by simp)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.InitialValueProblems.nonunique_glued
-- test: de-nonunique-glued-test.sage

/-- `2√|y|` is not Lipschitz at `y = 0`: for every `L` some `0 < y` has `2√y > L y`. -/
theorem sqrt_not_lipschitz (L : ℝ) : ∃ y : ℝ, 0 < y ∧ L * y < 2 * Real.sqrt y := by
  set c := |L| + 1 with hc
  have hc0 : 0 < c := by positivity
  refine ⟨1 / c ^ 2, by positivity, ?_⟩
  have hs : Real.sqrt (1 / c ^ 2) = 1 / c := by
    rw [one_div, Real.sqrt_inv, Real.sqrt_sq hc0.le, one_div]
  rw [hs]
  have hLc : L < 2 * c := by rw [hc]; linarith [le_abs_self L, abs_nonneg L]
  have key : 2 * (1 / c) - L * (1 / c ^ 2) = (2 * c - L) / c ^ 2 := by
    first | (field_simp; done) | (field_simp; ring)
  have : 0 < (2 * c - L) / c ^ 2 := div_pos (by linarith) (by positivity)
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.InitialValueProblems.sqrt_not_lipschitz
-- test: de-sqrt-not-lipschitz-test.sage

end KaluzaKlein.DifferentialEquations.InitialValueProblems

namespace KaluzaKlein.DifferentialEquations.ExistenceUniqueness

/-! ## thm-picard-lindelof -/

/-- Picard-Lindelöf, existence: under the hypotheses `IsPicardLindelof` (Lipschitz in `y` on a ball, continuous in
`x`, bounded by `L`, interval short enough) the initial value problem has a solution on the interval (mathlib). -/
theorem picard_exists {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (f : ℝ → E → E) (tmin tmax : ℝ) (t₀ : Set.Icc tmin tmax) (x₀ : E) (a L K : NNReal)
    (hf : IsPicardLindelof f t₀ x₀ a 0 L K) :
    ∃ y : ℝ → E, y t₀ = x₀ ∧ ∀ t ∈ Set.Icc tmin tmax, HasDerivWithinAt y (f t (y t)) (Set.Icc tmin tmax) t :=
  hf.exists_eq_forall_mem_Icc_hasDerivWithinAt₀
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.picard_exists
-- test: de-picard-exists-test.sage

/-- Picard-Lindelöf, uniqueness: two solutions of `y' = f(x, y)` with `f` Lipschitz in `y` that agree at one point
agree everywhere (mathlib's `ODE_solution_unique_univ`). -/
theorem picard_unique (f : ℝ → ℝ → ℝ) (K : NNReal) (hf : ∀ x, LipschitzWith K (f x)) (y₁ y₂ : ℝ → ℝ)
    (h₁ : ∀ x, HasDerivAt y₁ (f x (y₁ x)) x) (h₂ : ∀ x, HasDerivAt y₂ (f x (y₂ x)) x) (x₀ : ℝ)
    (h : y₁ x₀ = y₂ x₀) : y₁ = y₂ :=
  ODE_solution_unique_univ (s := fun _ => Set.univ) (fun x => (hf x).lipschitzOnWith)
    (fun x => ⟨h₁ x, trivial⟩) (fun x => ⟨h₂ x, trivial⟩) h
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.picard_unique
-- test: de-picard-unique-test.sage

/-! ## exm-picard-exponential -/

/-- The first Picard iterate of `y' = y`, `y(0) = 1`: `1 + ∫₀ˣ 1 = 1 + x`. -/
theorem picard_exp_iterate_one (x : ℝ) : 1 + ∫ s in (0 : ℝ)..x, (1 : ℝ) = 1 + x := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun s => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun s => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- The second Picard iterate: `1 + ∫₀ˣ (1 + s) ds = 1 + x + x²/2`. -/
theorem picard_exp_iterate_two (x : ℝ) : 1 + ∫ s in (0 : ℝ)..x, (1 + s) = 1 + x + x ^ 2 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 + x_veriphy : ℝ)) ((fun s => 1 + s) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ)).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 + x_veriphy : ℝ))
          (f' := fun s => 1 + s) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-! ## exm-picard-blowup -/

/-- `y = 1/(1 - x)` solves `y' = y²` where it is defined (`x ≠ 1`). -/
theorem blowup_solves (x : ℝ) (hx : 1 - x ≠ 0) : deriv (fun s => 1 / (1 - s)) x = (1 / (1 - x)) ^ 2 := by
  simp only [one_div]
  have h : HasDerivAt (fun s => (1 - s)⁻¹) _ x := ((hasDerivAt_id' x).const_sub 1).inv hx
  rw [h.deriv]
  first | (field_simp; done) | (field_simp; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_solves
-- test: de-blowup-solves-test.sage

/-- `f = y²` is Lipschitz on every bounded rectangle: `|a² - b²| ≤ 2M |a - b|` for `|a|, |b| ≤ M`. -/
theorem square_lipschitz_bounded (M a b : ℝ) (ha : |a| ≤ M) (hb : |b| ≤ M) :
    |a ^ 2 - b ^ 2| ≤ 2 * M * |a - b| := by
  rw [show a ^ 2 - b ^ 2 = (a + b) * (a - b) by ring, abs_mul]
  have : |a + b| ≤ 2 * M := (abs_add_le a b).trans (by linarith)
  exact mul_le_mul_of_nonneg_right this (abs_nonneg _)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.square_lipschitz_bounded
-- test: de-square-lipschitz-test.sage

/-- The solution escapes to infinity as `x → 1⁻`: it does not extend past `x = 1`. -/
theorem blowup_tendsto (y₀ : ℝ) (hy₀ : 0 < y₀) :
    Tendsto (fun x => y₀ / (1 - y₀ * x)) (𝓝[<] (1 / y₀)) atTop := by
  have h1 : Tendsto (fun x => 1 - y₀ * x) (𝓝[<] (1 / y₀)) (𝓝[>] 0) := by
    refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
    · have : Tendsto (fun x => 1 - y₀ * x) (𝓝 (1 / y₀)) (𝓝 (1 - y₀ * (1 / y₀))) :=
        (tendsto_const_nhds.sub (tendsto_const_nhds.mul tendsto_id))
      rw [mul_one_div_cancel hy₀.ne', sub_self] at this
      exact this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with x hx
      have : y₀ * x < 1 := by
        have := mul_lt_mul_of_pos_left (Set.mem_Iio.mp hx) hy₀
        rwa [mul_one_div_cancel hy₀.ne'] at this
      simp only [Set.mem_Ioi]; linarith
  exact (tendsto_inv_nhdsGT_zero.comp h1).const_mul_atTop hy₀ |>.congr fun x => by simp [div_eq_mul_inv]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_tendsto
-- test: de-blowup-tendsto-test.sage

/-! ## exr-picard-iterates -/

/-- The Picard iterates of `y' = x + y`, `y(0) = 1`: `y₁ = 1 + x + x²/2`. -/
theorem picard_xy_one (x : ℝ) : 1 + ∫ s in (0 : ℝ)..x, (s + 1) = 1 + x + x ^ 2 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 + x_veriphy : ℝ)) ((fun s => s + 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ)).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 + x_veriphy : ℝ))
          (f' := fun s => s + 1) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- `y₂ = 1 + x + x² + x³/6`. -/
theorem picard_xy_two (x : ℝ) :
    1 + ∫ s in (0 : ℝ)..x, (s + (1 + s + s ^ 2 / 2)) = 1 + x + x ^ 2 + x ^ 3 / 6 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 6 + (x_veriphy ^ 2) + x_veriphy : ℝ))
            ((fun s => s + (1 + s + s ^ 2 / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 6 : ℝ)).add
                        ((hasDerivAt_id' (x_veriphy : ℝ)).pow 2)).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 3) / 6 + (x_veriphy ^ 2) + x_veriphy : ℝ)) (f' := fun s => s + (1 + s + s ^ 2 / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- `y₃ = 1 + x + x² + x³/3 + x⁴/24`. -/
theorem picard_xy_three (x : ℝ) :
    1 + ∫ s in (0 : ℝ)..x, (s + (1 + s + s ^ 2 + s ^ 3 / 6)) = 1 + x + x ^ 2 + x ^ 3 / 3 + x ^ 4 / 24 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 4) / 24 + ((x_veriphy ^ 3) / 3) + (x_veriphy ^ 2) + x_veriphy : ℝ))
            ((fun s => s + (1 + s + s ^ 2 + s ^ 3 / 6)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (1 / 24 : ℝ)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ))).add
                        ((hasDerivAt_id' (x_veriphy : ℝ)).pow 2)).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 4) / 24 + ((x_veriphy ^ 3) / 3) + (x_veriphy ^ 2) + x_veriphy : ℝ)) (f' := fun s =>
          s + (1 + s + s ^ 2 + s ^ 3 / 6)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- `y₄ = 1 + x + x² + x³/3 + x⁴/12 + x⁵/120`, agreeing with the Taylor polynomial of `2eˣ - x - 1` up to degree 4. -/
theorem picard_xy_four (x : ℝ) :
    1 + ∫ s in (0 : ℝ)..x, (s + (1 + s + s ^ 2 + s ^ 3 / 3 + s ^ 4 / 24)) =
      1 + x + x ^ 2 + x ^ 3 / 3 + x ^ 4 / 12 + x ^ 5 / 120 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              ((x_veriphy ^ 5) / 120 + ((x_veriphy ^ 4) / 12) + ((x_veriphy ^ 3) / 3) + (x_veriphy ^ 2) + x_veriphy :
                ℝ))
            ((fun s => s + (1 + s + s ^ 2 + s ^ 3 / 3 + s ^ 4 / 24)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 120 : ℝ)).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (1 / 12 : ℝ))).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ))).add
                        ((hasDerivAt_id' (x_veriphy : ℝ)).pow 2)).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 5) / 120 + ((x_veriphy ^ 4) / 12) + ((x_veriphy ^ 3) / 3) + (x_veriphy ^ 2) + x_veriphy : ℝ))
          (f' := fun s => s + (1 + s + s ^ 2 + s ^ 3 / 3 + s ^ 4 / 24)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-- The exact solution `y = 2eˣ - x - 1` of `y' = x + y`. -/
theorem xy_exact_solves (y : ℝ → ℝ) (hy : y = fun s => 2 * Real.exp s - s - 1) (x : ℝ) : deriv y x = x + y x := by
  veriphy_certified
    ( subst hy
      have d1 : deriv (fun s => 2 * rexp s - s - 1) = fun s => 2 * (Real.exp s) - 1 :=
        by
        funext s
        have hd : HasDerivAt (fun s => 2 * rexp s - s - 1) ((2 * (Real.exp s) - 1 : ℝ)) (s) :=
          by
          convert
            ((((hasDerivAt_id' (s : ℝ)).const_mul (-1 : ℝ)).add (((hasDerivAt_id' (s : ℝ)).exp).const_mul (2 : ℝ))).add
              (hasDerivAt_const (s : ℝ) (-1 : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv
      simp only [d1]
      ring1)
  -- certificate: de-ode.sage

/-- … with `y(0) = 1`. -/
theorem xy_exact_initial (y : ℝ → ℝ) (hy : y = fun s => 2 * Real.exp s - s - 1) : y 0 = 1 := by
  veriphy_certified
    ( subst hy
      (norm_num; done))
  -- certificate: de-ode.sage

/-- The Taylor coefficients of `2eˣ - x - 1` at `0` up to degree 4 are those of `y₄`: `y^{(k)}(0) = 2` for `k ≥ 2`. -/
theorem xy_exact_taylor (k : ℕ) (hk : 2 ≤ k) :
    iteratedDeriv k (fun s => 2 * Real.exp s - s - 1) 0 = 2 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 2 := ⟨k - 2, by omega⟩
  have h1 : deriv (fun s => 2 * Real.exp s - s - 1) = fun s => 2 * Real.exp s - 1 := by
    funext s
    have := (((Real.hasDerivAt_exp s).const_mul 2).sub (hasDerivAt_id' s)).sub_const 1
    simpa using this.deriv
  have h2 : deriv (fun s => 2 * Real.exp s - 1) = fun s => 2 * Real.exp s := by
    funext s
    simpa using (((Real.hasDerivAt_exp s).const_mul 2).sub_const 1).deriv
  have h3 : ∀ i, iteratedDeriv i (fun s => 2 * Real.exp s) = fun s => 2 * Real.exp s := by
    intro i
    induction i with
    | zero => rfl
    | succ i ih =>
      rw [iteratedDeriv_succ, ih]
      funext s
      simpa using ((Real.hasDerivAt_exp s).const_mul 2).deriv
  rw [show j + 2 = j + 1 + 1 by ring, iteratedDeriv_succ', iteratedDeriv_succ', h1, h2, h3]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.xy_exact_taylor
-- test: de-xy-taylor-test.sage

/-! ## exr-picard-blowup-time -/

/-- `y = y₀/(1 - y₀x)` solves `y' = y²` where `1 - y₀x ≠ 0`. -/
theorem blowup_general_solves (y₀ x : ℝ) (hx : 1 - y₀ * x ≠ 0) :
    deriv (fun s => y₀ / (1 - y₀ * s)) x = (y₀ / (1 - y₀ * x)) ^ 2 := by
  have h : HasDerivAt (fun s => y₀ * (1 - y₀ * s)⁻¹) _ x :=
    ((((hasDerivAt_id' x).const_mul y₀).const_sub 1).inv hx).const_mul y₀
  have e : (fun s => y₀ / (1 - y₀ * s)) = fun s => y₀ * (1 - y₀ * s)⁻¹ := by
    funext s
    rw [div_eq_mul_inv]
  rw [e, h.deriv]
  first | (field_simp; done) | (field_simp; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_general_solves
-- test: de-blowup-general-solves-test.sage

/-- `y₀/(1 - y₀x)` has `y(0) = y₀`. -/
theorem blowup_general_initial (y₀ : ℝ) : y₀ / (1 - y₀ * 0) = y₀ := by
  veriphy_certified (simp)
  -- certificate: de-rewrite.sage

/-- For `y₀ < 0` the solution tends to `0` as `x → ∞`. -/
theorem blowup_negative_tendsto (y₀ : ℝ) (hy₀ : y₀ < 0) :
    Tendsto (fun x => y₀ / (1 - y₀ * x)) atTop (𝓝 0) := by
  have h : Tendsto (fun x => 1 - y₀ * x) atTop atTop := by
    have : Tendsto (fun x => (-y₀) * x) atTop atTop := tendsto_id.const_mul_atTop (by linarith)
    exact (tendsto_atTop_add_const_left _ 1 this).congr fun x => by ring
  have h2 := h.inv_tendsto_atTop.const_mul y₀
  rw [mul_zero] at h2
  exact h2.congr fun x => by simp [div_eq_mul_inv]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_negative_tendsto
-- test: de-blowup-negative-test.sage

/-! ## exr-picard-lipschitz-c1 -/

/-- A bounded derivative gives a Lipschitz bound (mean value theorem): `|g a - g b| ≤ L |a - b|`. -/
theorem lipschitz_of_deriv_bound (g g' : ℝ → ℝ) (L : ℝ) (hg : ∀ y, HasDerivAt g (g' y) y)
    (hb : ∀ y, |g' y| ≤ L) (a b : ℝ) : |g a - g b| ≤ L * |a - b| := by
  have := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (𝕜 := ℝ) (f := g) (f' := g') (s := Set.univ) (x := b) (y := a)
    (fun y _ => (hg y).hasDerivWithinAt) (fun y _ => by simpa using hb y) convex_univ trivial trivial
  simpa [Real.norm_eq_abs] using this
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.lipschitz_of_deriv_bound
-- test: de-lipschitz-deriv-test.sage

/-- `∂_y (x cos y + y³) = -x sin y + 3y²`. -/
theorem partial_example (x y : ℝ) : deriv (fun s => x * Real.cos s + s ^ 3) y = -x * Real.sin y + 3 * y ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => x * cos s + s ^ 3) ((3 * (y ^ 2) - (x * (Real.sin y)) : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).pow 3).add (((hasDerivAt_id' (y : ℝ)).cos).const_mul (x : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: de-derivative.sage

/-- … which is bounded on every rectangle: `|-x sin y + 3y²| ≤ |x| + 3y²`. -/
theorem partial_example_bound (x y : ℝ) : |-x * Real.sin y + 3 * y ^ 2| ≤ |x| + 3 * y ^ 2 := by
  calc |-x * Real.sin y + 3 * y ^ 2| ≤ |-x * Real.sin y| + |3 * y ^ 2| := abs_add_le _ _
    _ ≤ |x| + 3 * y ^ 2 := by
      rw [abs_mul, abs_neg, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 3 * y ^ 2)]
      have := Real.abs_sin_le_one y
      nlinarith [abs_nonneg x]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ExistenceUniqueness.partial_example_bound
-- test: de-partial-bound-test.sage

end KaluzaKlein.DifferentialEquations.ExistenceUniqueness

namespace KaluzaKlein.DifferentialEquations.SeparableEquations

/-! ## def-separable-equation -/

/-- Every zero `y⋆` of `g` gives the constant solution `y = y⋆` of `y' = f(x) g(y)`. -/
theorem separable_constant (f g : ℝ → ℝ) (y₀ : ℝ) (h : g y₀ = 0) (x : ℝ) :
    deriv (fun _ : ℝ => y₀) x = f x * g y₀ := by
  simp [h]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.separable_constant
-- test: de-separable-constant-test.sage

/-! ## exm-separable-circles -/

/-- The upper half-circle `y = √(2C - x²)` solves `y' = -x/y` where `x² < 2C`. -/
theorem circle_branch_solves (C x : ℝ) (h : 0 < 2 * C - x ^ 2) :
    deriv (fun s => Real.sqrt (2 * C - s ^ 2)) x = -x / Real.sqrt (2 * C - x ^ 2) := by
  have h1 := ((hasDerivAt_pow 2 x).const_sub (2 * C)).sqrt h.ne'
  rw [h1.deriv]
  have hs : Real.sqrt (2 * C - x ^ 2) ≠ 0 := (Real.sqrt_pos.mpr h).ne'
  norm_num
  first | (field_simp; done) | (field_simp; ring) | ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.circle_branch_solves
-- test: de-circle-branch-solves-test.sage

/-- The half-circle lies on the circle `x² + y² = 2C`. -/
theorem circle_branch_on_circle (C x : ℝ) (h : 0 ≤ 2 * C - x ^ 2) :
    x ^ 2 + Real.sqrt (2 * C - x ^ 2) ^ 2 = 2 * C := by
  veriphy_certified
    ( rw [Real.sq_sqrt h]
      ring)
  -- certificate: de-rewrite.sage

/-! ## exm-separable-decay -/

/-- Every solution of `y' = -ky` is `A e^{-kx}` with `A = y(0)`: the constant solution `0` and the separated
solutions `±e^C e^{-kx}` together. -/
theorem decay_all_solutions (k : ℝ) (y : ℝ → ℝ) (hy : ∀ x, HasDerivAt y (-k * y x) x) (x : ℝ) :
    y x = y 0 * Real.exp (-k * x) := by
  have hd : ∀ s, HasDerivAt (fun s => y s * Real.exp (k * s)) 0 s := by
    intro s
    have h1 := (hy s).mul (((hasDerivAt_id' s).const_mul k).exp)
    convert h1 using 1
    ring
  have hc := is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt) (fun s => (hd s).deriv) x 0
  simp only [mul_zero, Real.exp_zero, mul_one] at hc
  rw [← hc, mul_assoc, ← Real.exp_add]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.decay_all_solutions
-- test: de-decay-all-test.sage

/-! ## exm-separable-logistic -/

/-- Partial fractions: `1/(y(1 - y)) = 1/y + 1/(1 - y)`. -/
theorem logistic_partial_fractions (y : ℝ) (h0 : y ≠ 0) (h1 : 1 - y ≠ 0) :
    1 / (y * (1 - y)) = 1 / y + 1 / (1 - y) := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- `log(y/(1 - y))` is an antiderivative of `1/(y(1 - y))` on `0 < y < 1`. -/
theorem logistic_antiderivative (y : ℝ) (h0 : 0 < y) (h1 : 0 < 1 - y) :
    deriv (fun s => Real.log (s / (1 - s))) y = 1 / (y * (1 - y)) := by
  have hd : HasDerivAt (fun s => s / (1 - s)) _ y := (hasDerivAt_id' y).div ((hasDerivAt_id' y).const_sub 1) h1.ne'
  have hpos : y / (1 - y) ≠ 0 := (div_pos h0 h1).ne'
  have hl : HasDerivAt (fun s => Real.log (s / (1 - s))) _ y := hd.log hpos
  rw [hl.deriv]
  have h0' := h0.ne'
  have h1' := h1.ne'
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.logistic_antiderivative
-- test: de-logistic-antiderivative-test.sage

/-- `y = 1/(1 + e^{-x-C})` satisfies the separated relation `log(y/(1 - y)) = x + C`. -/
theorem logistic_relation (x C : ℝ) :
    Real.log ((1 / (1 + Real.exp (-x - C))) / (1 - 1 / (1 + Real.exp (-x - C)))) = x + C := by
  veriphy_certified
    ( rw [show (1 / (1 + Real.exp (-x - C))) / (1 - 1 / (1 + Real.exp (-x - C))) = Real.exp (x + C)
          by
          have hE : 0 < Real.exp (-x - C) := Real.exp_pos _
          rw [show x + C = -(-x - C) by ring, Real.exp_neg]
          field_simp
          ring]
      exact Real.log_exp _)
  -- certificate: de-rewrite.sage

/-! ## exr-separable-logistic -/

/-- Solving `desolve`'s relation `y/(y - 1) = e^{x+C}` for `y`: with `E = e^{x+C}` and `e = e^{-x-C}`
(`E e = 1`), `y (1 - e) = 1`, that is `y = 1/(1 - e^{-x-C})`. -/
theorem logistic_solve_relation (y E e : ℝ) (hE : E * e = 1) (h : y = E * (y - 1)) : y * (1 - e) = 1 := by
  veriphy_certified (linear_combination (-y + 1) * hE + (-e) * h)
  -- certificate: ideal_membership.sage

/-- `1/(1 - e^{-x-C})` is the family member `1/(1 + C'e^{-x})` with `C' = -e^{-C}`. -/
theorem logistic_reparametrise (x C : ℝ) :
    1 / (1 - Real.exp (-x - C)) = 1 / (1 + -Real.exp (-C) * Real.exp (-x)) := by
  veriphy_certified
    ( rw [show -x - C = -C + -x by ring, Real.exp_add]
      ring)
  -- certificate: de-rewrite.sage

/-- The family `y = 1/(1 + C e^{-x})` solves `y' = y(1 - y)` for every constant `C` (where defined). -/
theorem logistic_family_solves (C x : ℝ) (h : 1 + C * Real.exp (-x) ≠ 0) :
    deriv (fun s => 1 / (1 + C * Real.exp (-s))) x =
      1 / (1 + C * Real.exp (-x)) * (1 - 1 / (1 + C * Real.exp (-x))) := by
  simp only [one_div]
  have hd : HasDerivAt (fun s => (1 + C * Real.exp (-s))⁻¹) _ x := ((((hasDerivAt_neg x).exp).const_mul C).const_add 1).inv h
  rw [hd.deriv]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.logistic_family_solves
-- test: de-logistic-family-solves-test.sage

/-! ## exr-separable-explicit -/

/-- `cos(sin x) > 0` for every `x` (since `|sin x| ≤ 1 < π/2`), so `tan(sin x)` is defined on all of `ℝ`. -/
theorem cos_sin_pos (x : ℝ) : 0 < Real.cos (Real.sin x) :=
  Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.neg_one_le_sin x, Real.pi_gt_three],
    by linarith [Real.sin_le_one x, Real.pi_gt_three]⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.cos_sin_pos
-- test: de-cos-sin-pos-test.sage

/-- `y = tan(sin x) = sin(sin x)/cos(sin x)` solves `y' = (1 + y²) cos x` on all of `ℝ`. -/
theorem tan_sin_solves (y : ℝ → ℝ) (hy : y = fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)) (x : ℝ) :
    deriv y x = (1 + y x ^ 2) * Real.cos x := by
  subst hy
  have hc : Real.cos (Real.sin x) ≠ 0 :=
    (Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.neg_one_le_sin x, Real.pi_gt_three],
      by linarith [Real.sin_le_one x, Real.pi_gt_three]⟩).ne'
  have hd : HasDerivAt (fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)) _ x :=
    ((Real.hasDerivAt_sin x).sin).div ((Real.hasDerivAt_sin x).cos) hc
  rw [hd.deriv]
  beta_reduce
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.tan_sin_solves
-- test: de-tan-sin-solves-test.sage

/-- … with `y(0) = 0`. -/
theorem tan_sin_initial (y : ℝ → ℝ) (hy : y = fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)) :
    y 0 = 0 := by
  veriphy_certified
    ( subst hy
      (norm_num; done))
  -- certificate: de-ode.sage

/-- … and period `2π`. -/
theorem tan_sin_periodic (x : ℝ) :
    Real.sin (Real.sin (x + 2 * Real.pi)) / Real.cos (Real.sin (x + 2 * Real.pi)) =
      Real.sin (Real.sin x) / Real.cos (Real.sin x) := by
  veriphy_certified (rw [Real.sin_add_two_pi])
  -- certificate: de-rewrite.sage

/-! ## exr-separable-general-check -/

/-- Separation is correct: if `H' = 1/g`, `F' = f` and `H(y(x)) = F(x) + C` with `g(y) ≠ 0`, then `y' = f g(y)`. -/
theorem separation_forward (H F y g f : ℝ → ℝ) (y' : ℝ) (C x : ℝ) (hH : HasDerivAt H (1 / g (y x)) (y x))
    (hF : HasDerivAt F (f x) x) (hy : HasDerivAt y y' x) (hg : g (y x) ≠ 0)
    (hHF : ∀ s, H (y s) = F s + C) : y' = f x * g (y x) := by
  have h1 : HasDerivAt (fun s => H (y s)) (1 / g (y x) * y') x := hH.comp x hy
  have h2 : HasDerivAt (fun s => H (y s)) (f x) x := by
    have : (fun s => H (y s)) = fun s => F s + C := funext hHF
    rw [this]; exact hF.add_const C
  have := h1.unique h2
  field_simp at this
  linarith [this]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.separation_forward
-- test: de-separation-forward-test.sage

/-- Conversely, if `y' = f g(y)` then `H(y(x)) - F(x)` has zero derivative (so it is constant on intervals). -/
theorem separation_converse (H F y g f : ℝ → ℝ) (x : ℝ) (hH : HasDerivAt H (1 / g (y x)) (y x))
    (hF : HasDerivAt F (f x) x) (hy : HasDerivAt y (f x * g (y x)) x) (hg : g (y x) ≠ 0) :
    HasDerivAt (fun s => H (y s) - F s) 0 x := by
  have := (hH.comp x hy).sub hF
  refine this.congr_deriv ?_
  rw [div_mul_eq_mul_div, one_mul, mul_div_assoc, div_self hg, mul_one, sub_self]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SeparableEquations.separation_converse
-- test: de-separation-converse-test.sage

end KaluzaKlein.DifferentialEquations.SeparableEquations

namespace KaluzaKlein.DifferentialEquations.LinearFirstOrder

/-! ## def-linear-first-order-equation -/

/-- The integrating factor: if `μ' = p μ` then `(μ y)' = μ (y' + p y)`. -/
theorem integrating_factor (μ y p : ℝ → ℝ) (y' x : ℝ) (hμ : HasDerivAt μ (p x * μ x) x) (hy : HasDerivAt y y' x) :
    HasDerivAt (fun s => μ s * y s) (μ x * (y' + p x * y x)) x := by
  convert hμ.mul hy using 1
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.integrating_factor
-- test: de-integrating-factor-test.sage

/-- The general solution: with `μ' = p μ`, `μ ≠ 0`, and `M' = μ q`, `y = (M + C)/μ` solves `y' + p y = q`. -/
theorem integrating_factor_solution (μ M p q : ℝ → ℝ) (C x : ℝ) (hμ : HasDerivAt μ (p x * μ x) x)
    (hM : HasDerivAt M (μ x * q x) x) (h0 : μ x ≠ 0) :
    HasDerivAt (fun s => (M s + C) / μ s) (q x - p x * ((M x + C) / μ x)) x := by
  convert (hM.add_const C).div hμ h0 using 1
  first | (field_simp; done) | (field_simp; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.integrating_factor_solution
-- test: de-integrating-solution-test.sage

/-! ## exm-linear-constant -/

/-- `y = e^{-x} + C e^{-2x}` solves `y' + 2y = e^{-x}`. -/
theorem linear_const_solves (C : ℝ) (y : ℝ → ℝ) (hy : y = fun s => Real.exp (-s) + C * Real.exp (-2 * s))
    (x : ℝ) : deriv y x + 2 * y x = Real.exp (-x) := by
  veriphy_certified
    ( subst hy
      have d1 :
        deriv (fun s => rexp (-s) + C * rexp (-2 * s)) = fun s => -2 * C * (Real.exp (-2 * s)) - (Real.exp (-s)) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => rexp (-s) + C * rexp (-2 * s)) ((-2 * C * (Real.exp (-2 * s)) - (Real.exp (-s)) : ℝ))
            (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (-2 : ℝ)).exp).const_mul (C : ℝ)).add
              (((hasDerivAt_id' (s : ℝ)).const_mul (-1 : ℝ)).exp)) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | ((ring_nf; done))
        exact hd.deriv
      simp only [d1]
      ring1)
  -- certificate: de-ode.sage

/-- With `μ = e^{2x}`, `(e^{2x} y)' = eˣ`. -/
theorem linear_const_factor (C x : ℝ) :
    deriv (fun s => Real.exp (2 * s) * (Real.exp (-s) + C * Real.exp (-2 * s))) x = Real.exp x := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun s => rexp (2 * s) * (rexp (-s) + C * rexp (-2 * s)))
          ((-(2 * C * (Real.exp (-2 * x)) + (Real.exp (-x))) * (Real.exp (2 * x)) +
              (2 * (C * (Real.exp (-2 * x)) + (Real.exp (-x))) * (Real.exp (2 * x))) :
            ℝ))
          (x) :=
        by
        refine
          ((((((((hasDerivAt_id' (x : ℝ)).const_mul (-2 : ℝ)).exp).const_mul (C : ℝ)).add
                        (((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ)).exp)).mul
                    (((hasDerivAt_id' (x : ℝ)).const_mul (2 : ℝ)).exp))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (simp; ring_nf; done)
      exact hd.deriv.trans (by (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done)))
  -- certificate: de-derivative.sage

/-! ## exm-linear-cooling -/

/-- Newton's law of cooling: `T = T_a + (T₀ - T_a) e^{-kt}` solves `Ṫ = -k (T - T_a)`. -/
theorem cooling_solves (k Ta T₀ : ℝ) (T : ℝ → ℝ) (hT : T = fun s => Ta + (T₀ - Ta) * Real.exp (-k * s)) (t : ℝ) :
    deriv T t = -k * (T t - Ta) := by
  veriphy_certified
    ( subst hT
      have d1 : deriv (fun s => Ta + (T₀ - Ta) * rexp (-k * s)) = fun s => k * (Ta - T₀) * (Real.exp (-s * k)) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => Ta + (T₀ - Ta) * rexp (-k * s)) ((k * (Ta - T₀) * (Real.exp (-s * k)) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (-k : ℝ)).exp).const_mul (-Ta + T₀ : ℝ)).add
              (hasDerivAt_const (s : ℝ) (Ta : ℝ))) using
            1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | ((ring_nf; done))
        exact hd.deriv
      simp only [d1]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- … with `T(0) = T₀`. -/
theorem cooling_initial (k Ta T₀ : ℝ) (T : ℝ → ℝ) (hT : T = fun s => Ta + (T₀ - Ta) * Real.exp (-k * s)) :
    T 0 = T₀ := by
  veriphy_certified
    ( subst hT
      (norm_num; done))
  -- certificate: de-ode.sage

/-- … relaxing to `T_a` for `k > 0`. -/
theorem cooling_limit (k Ta T₀ : ℝ) (hk : 0 < k) :
    Tendsto (fun t => Ta + (T₀ - Ta) * Real.exp (-k * t)) atTop (𝓝 Ta) := by
  have h : Tendsto (fun t => Real.exp (-k * t)) atTop (𝓝 0) := by
    have : Tendsto (fun t => -k * t) atTop atBot := tendsto_id.const_mul_atTop_of_neg (by linarith)
    exact Real.tendsto_exp_atBot.comp this
  simpa using tendsto_const_nhds.add (h.const_mul (T₀ - Ta))
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.cooling_limit
-- test: de-cooling-limit-test.sage

/-! ## exm-linear-variable-coefficient -/

/-- `y = x²/3 + C/x` solves `y' + y/x = x` on `x ≠ 0`. -/
theorem linvar_solves (C x : ℝ) (hx : x ≠ 0) :
    deriv (fun s => s ^ 2 / 3 + C / s) x + (x ^ 2 / 3 + C / x) / x = x := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun s => s ^ 2 / 3 + C / s) ((2 * x / 3 - (C / (x ^ 2)) : ℝ)) (x) :=
        by
        refine
          ((((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (1 / 3 : ℝ)).add
                    (((hasDerivAt_const (x : ℝ) (1 : ℝ)).div (hasDerivAt_id' (x : ℝ)) hx).const_mul
                      (C : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h0.deriv]
      try beta_reduce
      all_goals ((field_simp; ring1)))
  -- certificate: de-partials.sage

/-- The integrating factor is `μ = e^{log x} = x` on `x > 0`. -/
theorem linvar_factor (x : ℝ) (hx : 0 < x) : Real.exp (Real.log x) = x := by
  veriphy_certified (exact Real.exp_log hx)
  -- certificate: de-rewrite.sage

/-- `(x y)' = x²`. -/
theorem linvar_product (C x : ℝ) (hx : x ≠ 0) : deriv (fun s => s * (s ^ 2 / 3 + C / s)) x = x ^ 2 := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun s => s * (s ^ 2 / 3 + C / s))
          (((2 * x - (3 * C / (x ^ 2))) * x / 3 + ((x ^ 2) / 3) + (C / x) : ℝ)) (x) :=
        by
        convert
          (((((hasDerivAt_id' (x : ℝ)).pow 2).add
                    (((hasDerivAt_const (x : ℝ) (1 : ℝ)).div (hasDerivAt_id' (x : ℝ)) hx).const_mul (3 * C : ℝ))).mul
                (hasDerivAt_id' (x : ℝ))).const_mul
            (1 / 3 : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.div_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.div_apply, Pi.pow_apply]; ring1)
      exact hd.deriv.trans (by (field_simp; ring1)))
  -- certificate: de-derivative.sage

/-! ## exr-linear-desolve -/

/-- `y = -1/2 + C e^{x²}` solves `y' - 2xy = x`. -/
theorem linear_gauss_solves (C : ℝ) (y : ℝ → ℝ) (hy : y = fun s => -1 / 2 + C * Real.exp (s ^ 2)) (x : ℝ) :
    deriv y x - 2 * x * y x = x := by
  veriphy_certified
    ( subst hy
      have d1 :
        deriv (fun (s : ℝ) => (-1 / 2 + C * rexp (s ^ 2) : ℝ)) = fun (s : ℝ) => (2 * s * C * (Real.exp (s ^ 2)) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (-1 / 2 + C * rexp (s ^ 2) : ℝ)) ((2 * s * C * (Real.exp (s ^ 2)) : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 2).exp).const_mul (C : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (simp; ring_nf; done)
        exact hd.deriv
      simp only [d1]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- With `μ = e^{-x²}`, `(e^{-x²} y)' = x e^{-x²}`. -/
theorem linear_gauss_factor (C x : ℝ) :
    deriv (fun s => Real.exp (-s ^ 2) * (-1 / 2 + C * Real.exp (s ^ 2))) x = x * Real.exp (-x ^ 2) := by
  have e : (fun s => Real.exp (-s ^ 2) * (-1 / 2 + C * Real.exp (s ^ 2))) = fun s => -1 / 2 * Real.exp (-s ^ 2) + C := by
    funext s
    rw [mul_add, show Real.exp (-s ^ 2) * (C * Real.exp (s ^ 2)) = C * (Real.exp (-s ^ 2) * Real.exp (s ^ 2)) by ring,
      ← Real.exp_add, neg_add_cancel, Real.exp_zero]
    ring
  rw [e]
  have h : HasDerivAt (fun s => -1 / 2 * Real.exp (-s ^ 2) + C) _ x :=
    ((((hasDerivAt_pow 2 x).neg).exp).const_mul (-1 / 2)).add_const C
  rw [h.deriv]
  norm_num
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.linear_gauss_factor
-- test: de-linear-gauss-factor-test.sage

/-! ## exr-linear-superposition -/

/-- The difference of two solutions of `y' + p y = q` solves `y' + p y = 0`. -/
theorem linear_difference (p q y₁ y₂ : ℝ → ℝ) (x a₁ a₂ : ℝ) (h₁ : HasDerivAt y₁ a₁ x) (h₂ : HasDerivAt y₂ a₂ x)
    (e₁ : a₁ + p x * y₁ x = q x) (e₂ : a₂ + p x * y₂ x = q x) :
    HasDerivAt (fun s => y₁ s - y₂ s) (-(p x * (y₁ x - y₂ x))) x := by
  convert h₁.sub h₂ using 1
  linear_combination e₂ - e₁
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.linear_difference
-- test: de-linear-difference-test.sage

/-- Combinations `αy₁ + βy₂` of homogeneous solutions are homogeneous solutions. -/
theorem homogeneous_combination (p y₁ y₂ : ℝ → ℝ) (α β x : ℝ) (h₁ : HasDerivAt y₁ (-(p x * y₁ x)) x)
    (h₂ : HasDerivAt y₂ (-(p x * y₂ x)) x) :
    HasDerivAt (fun s => α * y₁ s + β * y₂ s) (-(p x * (α * y₁ x + β * y₂ x))) x := by
  convert (h₁.const_mul α).add (h₂.const_mul β) using 1
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.homogeneous_combination
-- test: de-homogeneous-comb-test.sage

/-- Every homogeneous solution is `C/μ`: if `μ' = p μ` and `y' = -p y` then `μ y` is constant. -/
theorem homogeneous_is_multiple (p μ y : ℝ → ℝ) (hμ : ∀ x, HasDerivAt μ (p x * μ x) x)
    (hy : ∀ x, HasDerivAt y (-(p x * y x)) x) (x : ℝ) : μ x * y x = μ 0 * y 0 := by
  have hd : ∀ s, HasDerivAt (fun s => μ s * y s) 0 s := by
    intro s; convert (hμ s).mul (hy s) using 1; ring
  exact is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt) (fun s => (hd s).deriv) x 0
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.homogeneous_is_multiple
-- test: de-homogeneous-multiple-test.sage

/-! ## exr-linear-bernoulli -/

/-- Bernoulli's substitution: if `y > 0` solves `y' + p y = q yⁿ` then `u = y^{1-n}` solves
`u' + (1 - n) p u = (1 - n) q`. -/
theorem bernoulli_substitution (n : ℝ) (p q y : ℝ → ℝ) (x y' : ℝ) (hpos : 0 < y x) (hy : HasDerivAt y y' x)
    (he : y' + p x * y x = q x * y x ^ n) :
    HasDerivAt (fun s => y s ^ (1 - n)) ((1 - n) * q x - (1 - n) * p x * y x ^ (1 - n)) x := by
  have h := hy.rpow_const (p := 1 - n) (Or.inl hpos.ne')
  convert h using 1
  have hy' : y' = q x * y x ^ n - p x * y x := by linarith
  have e1 : y x ^ (1 - n - 1) = y x / y x ^ n / y x := by
    rw [Real.rpow_sub hpos, Real.rpow_sub hpos, Real.rpow_one]
  have e2 : y x ^ (1 - n) = y x / y x ^ n := by rw [Real.rpow_sub hpos, Real.rpow_one]
  have hyn : y x ^ n ≠ 0 := (Real.rpow_pos_of_pos hpos n).ne'
  have hy0 : y x ≠ 0 := hpos.ne'
  rw [hy', e1, e2]
  first | (field_simp; done) | (field_simp; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.bernoulli_substitution
-- test: de-bernoulli-test.sage

/-- `y = 1/(1 + eˣ)` solves `y' + y = y²`. -/
theorem bernoulli_example_solves (y : ℝ → ℝ) (hy : y = fun s => 1 / (1 + Real.exp s)) (x : ℝ) :
    deriv y x + y x = y x ^ 2 := by
  subst hy
  have hpos : 1 + Real.exp x ≠ 0 := by positivity
  simp only [one_div]
  have h : HasDerivAt (fun s => (1 + Real.exp s)⁻¹) _ x := ((Real.hasDerivAt_exp x).const_add 1).inv hpos
  rw [h.deriv]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LinearFirstOrder.bernoulli_example_solves
-- test: de-bernoulli-example-solves-test.sage

/-- … with `y(0) = 1/2`. -/
theorem bernoulli_example_initial (y : ℝ → ℝ) (hy : y = fun s => 1 / (1 + Real.exp s)) : y 0 = 1 / 2 := by
  veriphy_certified
    ( subst hy
      (norm_num; done))
  -- certificate: de-ode.sage

/-- `u = 1/y = 1 + eˣ` solves the linear equation `u' - u = -1`. -/
theorem bernoulli_linearised (u : ℝ → ℝ) (hu : u = fun s => 1 + Real.exp s) (x : ℝ) : deriv u x - u x = -1 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => 1 + rexp s) = fun s => Real.exp s :=
        by
        funext s
        have hd : HasDerivAt (fun s => 1 + rexp s) ((Real.exp s : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).exp).add (hasDerivAt_const (s : ℝ) (1 : ℝ))) using 1
          all_goals
            first
            | rfl
            | (funext y; simp only [Pi.add_apply]; ring_nf; done)
            | (ring1)
        exact hd.deriv
      simp only [d1]
      ring1)
  -- certificate: de-ode.sage

end KaluzaKlein.DifferentialEquations.LinearFirstOrder

namespace KaluzaKlein.DifferentialEquations.PhasePortraits

/-! ## def-phase-portrait -/

/-- An equilibrium `F(x⋆) = 0` of an autonomous system `ẋ = F(x)` is an orbit: the constant `x⋆` is a solution. -/
theorem equilibrium_is_orbit {n : ℕ} (F : (Fin n → ℝ) → Fin n → ℝ) (x₀ : Fin n → ℝ) (h : F x₀ = 0) (t : ℝ) :
    HasDerivAt (fun _ : ℝ => x₀) (F x₀) t := by
  rw [h]; exact hasDerivAt_const t x₀
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.equilibrium_is_orbit
-- test: de-equilibrium-orbit-test.sage

/-! ## exm-phase-oscillator -/

/-- Along `(q, p)' = (p, -q)`, `d/dt (q² + p²) = 2qp - 2pq = 0`: the orbits are circles. -/
theorem oscillator_circles (q p q' p' : ℝ) (hq : q' = p) (hp : p' = -q) : 2 * q * q' + 2 * p * p' = 0 := by
  veriphy_certified (linear_combination (2 * q) * hq + (2 * p) * hp)
  -- certificate: de-poly.sage

/-- The angular velocity `q ṗ - p q̇ = -(q² + p²) < 0`: the circles are traversed clockwise. -/
theorem oscillator_clockwise (q p q' p' : ℝ) (hq : q' = p) (hp : p' = -q) : q * p' - p * q' = -(q ^ 2 + p ^ 2) := by
  veriphy_certified (linear_combination (-p) * hq + (q) * hp)
  -- certificate: de-poly.sage

/-- The only equilibrium of `(p, -q)` is the origin. -/
theorem oscillator_equilibrium (q p : ℝ) (h₁ : p = 0) (h₂ : -q = 0) : q = 0 := by
  veriphy_certified (linear_combination (-1) * h₂)
  -- certificate: de-poly.sage

/-! ## exm-phase-pendulum -/

/-- The points `(kπ, 0)` are equilibria of the damped pendulum: `ω = 0` and `-sin(kπ) - γ·0 = 0`. -/
theorem pendulum_equilibria (γ : ℝ) (k : ℤ) : -Real.sin (k * Real.pi) - γ * 0 = 0 := by
  simp [Real.sin_int_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_equilibria
-- test: de-pendulum-equilibria-test.sage

/-- … and they are the only ones: `sin θ = 0` forces `θ = kπ`. -/
theorem pendulum_equilibria_only (θ : ℝ) (h : Real.sin θ = 0) : ∃ k : ℤ, k * Real.pi = θ :=
  Real.sin_eq_zero_iff.mp h
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_equilibria_only
-- test: de-pendulum-only-test.sage

/-- The energy `E = ω²/2 - cos θ` of the damped pendulum decreases at the rate `Ė = -γω²` along the flow. -/
theorem pendulum_energy_rate (γ θ ω θ' ω' sθ : ℝ) (hs : sθ = Real.sin θ) (h₁ : θ' = ω) (h₂ : ω' = -sθ - γ * ω) :
    ω * ω' + sθ * θ' = -γ * ω ^ 2 := by
  veriphy_certified (linear_combination (-ω * γ - ω') * h₁ + (θ') * h₂)
  -- certificate: de-poly.sage

/-- The energy's time derivative: `d/dt (ω²/2 - cos θ) = ω ω̇ + sin θ θ̇` (the chain rule along a trajectory). -/
theorem pendulum_energy_deriv (θ ω : ℝ → ℝ) (θ' ω' t : ℝ) (hθ : HasDerivAt θ θ' t) (hω : HasDerivAt ω ω' t) :
    HasDerivAt (fun s => ω s ^ 2 / 2 - Real.cos (θ s)) (ω t * ω' + Real.sin (θ t) * θ') t := by
  have := ((hω.pow 2).div_const 2).sub (hθ.cos)
  convert this using 1
  simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_energy_deriv
-- test: de-pendulum-energy-deriv-test.sage

/-! ## exm-phase-predator-prey -/

/-- The equilibria of the Lotka-Volterra system are `(0, 0)` and `(1, 1)`. -/
theorem lv_equilibria (x y : ℝ) (h₁ : x - x * y = 0) (h₂ : -y + x * y = 0) :
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := by
  have h₃ : x * (1 - y) = 0 := by linarith
  rcases mul_eq_zero.mp h₃ with hx | hy
  · left; subst hx; constructor <;> linarith
  · right; constructor <;> nlinarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.lv_equilibria
-- test: de-lv-equilibria-test.sage

/-- The quantity `x - log x + y - log y` is constant along orbits: with `x⁻¹ = xi`, `y⁻¹ = yi`, its derivative
`(1 - 1/x) ẋ + (1 - 1/y) ẏ` vanishes on the flow. -/
theorem lv_conserved (x y xi yi x' y' : ℝ) (hx : x * xi = 1) (hy : y * yi = 1) (h₁ : x' = x - x * y)
    (h₂ : y' = -y + x * y) : (1 - xi) * x' + (1 - yi) * y' = 0 := by
  veriphy_certified (linear_combination (y - 1) * hx + (-x + 1) * hy + (-xi + 1) * h₁ + (-yi + 1) * h₂)
  -- certificate: de-poly.sage

/-- `d/dx (x - log x) = 1 - 1/x` for `x > 0`. -/
theorem lv_conserved_deriv (x : ℝ) (hx : 0 < x) : deriv (fun s => s - Real.log s) x = 1 - 1 / x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => s - log s) ((-1 / x + 1 : ℝ)) (x) :=
        by
        refine
          ((((hasDerivAt_id' (x : ℝ)).add
                    (((hasDerivAt_id' (x : ℝ)).log (by exact hx.ne')).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: de-derivative.sage

/-! ## exr-phase-equilibria -/

/-- The Jacobian of `(x - xy, -y + xy)` at `(0, 0)` is `diag(1, -1)`, from the partial derivatives
`(1 - y, -x; y, -1 + x)`. -/
theorem lv_jacobian_origin : !![1 - (0 : ℝ), -0; 0, -1 + 0] = !![1, 0; 0, -1] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])
  -- certificate: la-matrix.sage

/-- At `(1, 1)` it is the rotation generator `(0, -1; 1, 0)`. -/
theorem lv_jacobian_centre : !![1 - (1 : ℝ), -1; 1, -1 + 1] = !![0, -1; 1, 0] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])
  -- certificate: la-matrix.sage

/-- The partial derivatives behind the Jacobian: `∂ₓ(x - xy) = 1 - y`. -/
theorem lv_partial_xx (x y : ℝ) : deriv (fun s => s - s * y) x = 1 - y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => s - s * y) ((-y + 1 : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).const_mul (-y : ℝ)).add (hasDerivAt_id' (x : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: de-derivative.sage

/-- `∂_y(x - xy) = -x`. -/
theorem lv_partial_xy (x y : ℝ) : deriv (fun s => x - x * s) y = -x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => x - x * s) ((-x : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).const_mul (-x : ℝ)).add (hasDerivAt_const (y : ℝ) (x : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- `∂ₓ(-y + xy) = y`. -/
theorem lv_partial_yx (x y : ℝ) : deriv (fun s => -y + s * y) x = y := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => -y + s * y) ((y : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).const_mul (y : ℝ)).add (hasDerivAt_const (x : ℝ) (-y : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- `∂_y(-y + xy) = -1 + x`. -/
theorem lv_partial_yy (x y : ℝ) : deriv (fun s => -s + x * s) y = -1 + x := by
  veriphy_certified
    ( have hd : HasDerivAt (fun s => -s + x * s) ((x - 1 : ℝ)) (y) :=
        by
        convert (((hasDerivAt_id' (y : ℝ)).const_mul (x : ℝ)).add ((hasDerivAt_id' (y : ℝ)).const_mul (-1 : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: de-derivative.sage

/-! ## exr-phase-energy-levels -/

/-- For the undamped pendulum `Ė = ω ω̇ + sin θ θ̇ = 0`. -/
theorem pendulum_energy_conserved (θ ω θ' ω' sθ : ℝ) (hs : sθ = Real.sin θ) (h₁ : θ' = ω) (h₂ : ω' = -sθ) :
    ω * ω' + sθ * θ' = 0 := by
  veriphy_certified (linear_combination (-ω') * h₁ + (θ') * h₂)
  -- certificate: de-poly.sage

/-- The separatrix `E = 1` passes through the upright equilibria: `E(±π, 0) = 0²/2 - cos π = 1`. -/
theorem separatrix_energy : (0 : ℝ) ^ 2 / 2 - Real.cos Real.pi = 1 := by
  veriphy_certified (norm_num)
  -- certificate: de-rewrite.sage

/-- The level curve `E` is `ω² = 2(E + cos θ)`. -/
theorem energy_level_curve (θ ω E c : ℝ) (hc : c = Real.cos θ) (h : ω ^ 2 / 2 - c = E) : ω ^ 2 = 2 * (E + c) := by
  veriphy_certified (linear_combination (2) * h)
  -- certificate: de-poly.sage

/-- For `E > 1` the velocity never vanishes (the pendulum rotates): `ω ≠ 0`. -/
theorem rotation_never_stops (θ ω : ℝ) (h : 1 < ω ^ 2 / 2 - Real.cos θ) : ω ≠ 0 := by
  intro hω
  subst hω
  have := Real.neg_one_le_cos θ
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.rotation_never_stops
-- test: de-rotation-test.sage

/-- For `E < 1` the orbit stays away from the upright position: `cos θ > -1`. -/
theorem libration_bounded (θ ω : ℝ) (h : ω ^ 2 / 2 - Real.cos θ < 1) : -1 < Real.cos θ := by
  nlinarith [sq_nonneg ω]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.libration_bounded
-- test: de-libration-test.sage

/-! ## exr-phase-dissipation -/

/-- For `q̈ + γq̇ + V'(q) = 0`, `E = q̇²/2 + V(q)` has `Ė = q̇ q̈ + V'(q) q̇ = -γ q̇²`. -/
theorem dissipation_rate (γ q' q'' dV : ℝ) (h : q'' + γ * q' + dV = 0) : q' * q'' + dV * q' = -γ * q' ^ 2 := by
  veriphy_certified (linear_combination (q') * h)
  -- certificate: de-poly.sage

/-- … which is `≤ 0` for `γ ≥ 0`. -/
theorem dissipation_nonpos (γ q' : ℝ) (hγ : 0 ≤ γ) : -γ * q' ^ 2 ≤ 0 := by
  nlinarith [sq_nonneg q']
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.dissipation_nonpos
-- test: de-dissipation-nonpos-test.sage

/-- No periodic orbit with damping: if `q̇` is continuous and not zero at some `t₀ ∈ (0, T)`, then
`∫₀ᵀ γ q̇² dt > 0` for `γ > 0`, so `E(T) < E(0)` and the orbit cannot close. -/
theorem no_periodic_orbit (γ T t₀ : ℝ) (qd : ℝ → ℝ) (hγ : 0 < γ) (hc : Continuous qd)
    (ht : t₀ ∈ Set.Ioo 0 T) (hne : qd t₀ ≠ 0) : 0 < ∫ t in (0 : ℝ)..T, γ * qd t ^ 2 := by
  have hcont : Continuous fun t => γ * qd t ^ 2 := by fun_prop
  have hnn : 0 ≤ᵐ[MeasureTheory.volume] fun t => γ * qd t ^ 2 :=
    MeasureTheory.ae_of_all _ fun t => by positivity
  rw [intervalIntegral.integral_pos_iff_support_of_nonneg_ae hnn (hcont.intervalIntegrable 0 T)]
  refine ⟨ht.1.trans ht.2, ?_⟩
  have hopen : IsOpen (Function.support (fun t => γ * qd t ^ 2) ∩ Set.Ioo 0 T) :=
    hcont.isOpen_support.inter isOpen_Ioo
  have hmem : t₀ ∈ Function.support (fun t => γ * qd t ^ 2) ∩ Set.Ioo 0 T :=
    ⟨by simp [hγ.ne', hne], ht⟩
  exact (hopen.measure_pos MeasureTheory.volume ⟨t₀, hmem⟩).trans_le
    (MeasureTheory.measure_mono (Set.inter_subset_inter_right _ Set.Ioo_subset_Ioc_self))
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.PhasePortraits.no_periodic_orbit
-- test: de-no-periodic-test.sage

end KaluzaKlein.DifferentialEquations.PhasePortraits

namespace KaluzaKlein.DifferentialEquations.ODEComputations

/-! ## prp-ode-closed-forms -/

/-- The logistic solution has `y(0) = 1/2`. -/
theorem logistic_initial (y : ℝ → ℝ) (hy : y = fun s => 1 / (1 + Real.exp (-s))) : y 0 = 1 / 2 := by
  veriphy_certified
    ( subst hy
      all_goals ((norm_num; done)))
  -- certificate: de-ode.sage

/-- Implicit solutions: a differentiable `y` with `x² + y(x)² = C` for all `x` satisfies `y' = -x/y` where `y ≠ 0`. -/
theorem circles_solve (C : ℝ) (y : ℝ → ℝ) (y' x : ℝ) (hy : HasDerivAt y y' x) (hC : ∀ s, s ^ 2 + y s ^ 2 = C)
    (h0 : y x ≠ 0) : y' = -x / y x := by
  have h1 : HasDerivAt (fun s => s ^ 2 + y s ^ 2) (2 * x + 2 * y x * y') x := by
    convert (hasDerivAt_pow 2 x).add (hy.pow 2) using 1
    all_goals first | (simp; done) | (simp; ring)
  have h2 : HasDerivAt (fun s => s ^ 2 + y s ^ 2) 0 x := by
    rw [show (fun s => s ^ 2 + y s ^ 2) = fun _ => C from funext hC]
    exact hasDerivAt_const x C
  have := h1.unique h2
  field_simp
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ODEComputations.circles_solve
-- test: de-circles-solve-test.sage

/-- The fourth Picard iterate of `y' = y`, `y(0) = 1`: `1 + ∫₀ˣ y₃ = 1 + x + x²/2 + x³/6 + x⁴/24`. -/
theorem picard_exp_iterate_four (x : ℝ) :
    1 + ∫ s in (0 : ℝ)..x, (1 + s + s ^ 2 / 2 + s ^ 3 / 6) = 1 + x + x ^ 2 / 2 + x ^ 3 / 6 + x ^ 4 / 24 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (x : ℝ),
          HasDerivAt
            (fun x_veriphy => ((x_veriphy ^ 4) / 24 + ((x_veriphy ^ 3) / 6) + ((x_veriphy ^ 2) / 2) + x_veriphy : ℝ))
            ((fun s => 1 + s + s ^ 2 / 2 + s ^ 3 / 6) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (1 / 24 : ℝ)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 6 : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ))).add
                    (hasDerivAt_id' (x_veriphy : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 4) / 24 + ((x_veriphy ^ 3) / 6) + ((x_veriphy ^ 2) / 2) + x_veriphy : ℝ)) (f' := fun s =>
          1 + s + s ^ 2 / 2 + s ^ 3 / 6) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage

/-! ## exm-closed-forms-arbitrary-constant -/

/-- Sage's form `(C + eˣ) e^{-2x}` of the linear solution is `e^{-x} + C e^{-2x}`. -/
theorem desolve_linear_form (C x : ℝ) :
    (C + Real.exp x) * Real.exp (-2 * x) = Real.exp (-x) + C * Real.exp (-2 * x) := by
  veriphy_certified
    ( rw [add_mul, ← Real.exp_add]
      ring_nf)
  -- certificate: de-rewrite.sage

/-! ## exr-closed-forms-logistic-family -/

/-- `C = 1` gives `y(0) = 1/2`. -/
theorem logistic_family_one : 1 / (1 + 1 * Real.exp (-0)) = (1 : ℝ) / 2 := by
  veriphy_certified (norm_num)
  -- certificate: de-rewrite.sage

/-- `C = 0` gives the constant solution `1`. -/
theorem logistic_family_zero (x : ℝ) : 1 / (1 + 0 * Real.exp (-x)) = 1 := by
  veriphy_certified (simp)
  -- certificate: de-rewrite.sage

/-- `C → ∞` gives the constant solution `0`, pointwise. -/
theorem logistic_family_infinity (x : ℝ) : Tendsto (fun C => 1 / (1 + C * Real.exp (-x))) atTop (𝓝 0) := by
  have h : Tendsto (fun C => 1 + C * Real.exp (-x)) atTop atTop :=
    tendsto_atTop_add_const_left _ 1 (tendsto_id.atTop_mul_const (Real.exp_pos _))
  exact h.inv_tendsto_atTop.congr fun C => by simp [one_div]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ODEComputations.logistic_family_infinity
-- test: de-logistic-infinity-test.sage

/-! ## exr-closed-forms-linear-by-hand -/

/-- `y = e^{-x} - e^{-2x}` (the constant `C = -1`) has `y(0) = 0`. -/
theorem linear_zero_initial (y : ℝ → ℝ) (hy : y = fun s => Real.exp (-s) + -1 * Real.exp (-2 * s)) : y 0 = 0 := by
  veriphy_certified
    ( subst hy
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- Its derivative is `-e^{-x} + 2e^{-2x}`. -/
theorem linear_zero_deriv (x : ℝ) :
    deriv (fun s => Real.exp (-s) - Real.exp (-2 * s)) x = -Real.exp (-x) + 2 * Real.exp (-2 * x) := by
  veriphy_certified
    ( have hd :
        HasDerivAt (fun s => rexp (-s) - rexp (-2 * s)) ((-(Real.exp (-x)) + (2 * (Real.exp (-2 * x))) : ℝ)) (x) :=
        by
        convert
          ((((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ)).exp).add
            ((((hasDerivAt_id' (x : ℝ)).const_mul (-2 : ℝ)).exp).const_mul (-1 : ℝ))) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply]; ring_nf; done)
          | ((ring_nf; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: de-derivative.sage

/-- … which vanishes at `x = log 2`. -/
theorem linear_zero_critical : -Real.exp (-Real.log 2) + 2 * Real.exp (-2 * Real.log 2) = 0 := by
  veriphy_certified
    ( rw [show -2 * Real.log 2 = -(Real.log 2 + Real.log 2) by ring, Real.exp_neg, Real.exp_neg, Real.exp_add,
        Real.exp_log (by norm_num)]
      norm_num)
  -- certificate: de-rewrite.sage

/-- The value there is `1/4`. -/
theorem linear_zero_max_value : Real.exp (-Real.log 2) - Real.exp (-2 * Real.log 2) = 1 / 4 := by
  veriphy_certified
    ( rw [show -2 * Real.log 2 = -(Real.log 2 + Real.log 2) by ring, Real.exp_neg, Real.exp_neg, Real.exp_add,
        Real.exp_log (by norm_num)]
      norm_num)
  -- certificate: de-rewrite.sage

/-- `1/4` is the maximum: with `u = e^{-x}` (so `e^{-2x} = u²`), `u - u² ≤ 1/4`. -/
theorem linear_zero_max (u : ℝ) : u - u ^ 2 ≤ 1 / 4 := by
  veriphy_certified (linear_combination (1 / 4) * sq_nonneg (2 * u - 1))
  -- certificate: sos.sage

/-- `e^{-2x} = (e^{-x})²`. -/
theorem exp_neg_two_mul (x : ℝ) : Real.exp (-2 * x) = Real.exp (-x) ^ 2 := by
  veriphy_certified
    ( rw [sq, ← Real.exp_add]
      ring_nf)
  -- certificate: de-rewrite.sage

/-! ## exr-closed-forms-picard-general -/

/-- The Taylor polynomial `Σ_{j≤k} (ax)ʲ/j!` of `e^{ax}`. -/
noncomputable def taylorExp (a x : ℝ) (k : ℕ) : ℝ := ∑ j ∈ Finset.range (k + 1), (a * x) ^ j / (j.factorial : ℝ)

/-- The Picard iterates of `y' = ay`, `y(0) = 1` are the Taylor polynomials of `e^{ax}`:
`1 + ∫₀ˣ a Σ_{j≤k} (as)ʲ/j! ds = Σ_{j≤k+1} (ax)ʲ/j!`. -/
theorem picard_linear_iterate (a x : ℝ) (k : ℕ) :
    1 + ∫ s in (0 : ℝ)..x, a * taylorExp a s k = taylorExp a x (k + 1) := by
  simp only [taylorExp]
  have hF : ∀ s, HasDerivAt (fun s => ∑ j ∈ Finset.range (k + 1), (a * s) ^ (j + 1) / ((j + 1).factorial : ℝ))
      (a * ∑ j ∈ Finset.range (k + 1), (a * s) ^ j / (j.factorial : ℝ)) s := by
    intro s
    rw [Finset.mul_sum]
    apply HasDerivAt.fun_sum
    intro j _
    have h := (((hasDerivAt_id' s).const_mul a).pow (j + 1)).div_const ((j + 1).factorial : ℝ)
    convert h using 1
    simp only [Nat.add_sub_cancel, Nat.factorial_succ]
    push_cast
    have hf : (j.factorial : ℝ) ≠ 0 := by positivity
    first | (field_simp; done) | (field_simp; ring)
  have hint : IntervalIntegrable (fun s => a * ∑ j ∈ Finset.range (k + 1), (a * s) ^ j / (j.factorial : ℝ))
      MeasureTheory.volume 0 x := by
    apply Continuous.intervalIntegrable
    fun_prop
  have key := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s _ => hF s) hint
  rw [key, Finset.sum_range_succ' _ (k + 1)]
  have h0 : ∑ j ∈ Finset.range (k + 1), (a * 0) ^ (j + 1) / ((j + 1).factorial : ℝ) = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    simp
  beta_reduce
  rw [h0]
  simp only [sub_zero, pow_zero, Nat.factorial_zero, Nat.cast_one, div_one]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ODEComputations.picard_linear_iterate
-- test: de-picard-linear-test.sage

/-- The iterates converge to the solution `e^{ax}` for every `x` (the exponential series converges). -/
theorem picard_linear_converges (a x : ℝ) :
    Tendsto (fun k => ∑ j ∈ Finset.range k, (a * x) ^ j / (j.factorial : ℝ)) atTop (𝓝 (Real.exp (a * x))) := by
  have := NormedSpace.expSeries_div_hasSum_exp (a * x)
  rw [← Real.exp_eq_exp_ℝ] at this
  exact this.tendsto_sum_nat
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.ODEComputations.picard_linear_converges
-- test: de-picard-converges-test.sage

end KaluzaKlein.DifferentialEquations.ODEComputations

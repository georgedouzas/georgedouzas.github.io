import Mathlib
import Veriphy.Kit.Levels

/-!
# Differential equations, part "Boundary value and Sturm-Liouville problems"

The claims of the part "Boundary value and Sturm-Liouville problems" of
`tutorial/part-i-05-differential-equations.qmd`, one namespace per section: Dirichlet and Neumann problems on
`[0, π]`, the Sturm-Liouville form and Lagrange's identity, orthogonality of sines, cosines and Legendre polynomials,
the Legendre polynomials `P₀, …, P₆` (explicit polynomials, Rodrigues' formula, the recurrence) and the Bessel
functions `Jₙ` (the series, its truncations, the zero `j₀,₁`).

Eigenfunction statements are pointwise identities checked by substitution; orthogonality relations are definite
integrals; general facts (Lagrange's identity, integration by parts, the energy identities) are theorems about
differentiable functions on an interval.
-/

open Real Filter Topology

namespace KaluzaKlein.DifferentialEquations.BoundaryValueProblems

/-! ## exm-bvp-loaded-string -/

/-- `u = x(1 - x)/2` solves `-u'' = 1` … -/
theorem loaded_string_solves (x : ℝ) : -deriv (fun s => deriv (fun r => r * (1 - r) / 2) s) x = 1 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => (r * (1 - r) / 2 : ℝ)) = fun (d1 : ℝ) => (-d1 + (1 / 2) : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (r * (1 - r) / 2 : ℝ)) ((-d1 + (1 / 2) : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).mul
                          (hasDerivAt_id' (d1 : ℝ))).const_mul
                      (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun s => (fun (d1 : ℝ) => (-d1 + (1 / 2) : ℝ)) s) ((-1 : ℝ)) (x) :=
        by
        refine
          (((((hasDerivAt_id' (x : ℝ)).const_mul (-1 : ℝ)).add
                    (hasDerivAt_const (x : ℝ) (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; done)
        · ring1
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- … with `u(0) = 0` … -/
theorem loaded_string_left (u : ℝ → ℝ) (hu : u = fun s => s * (1 - s) / 2) : u 0 = 0 := by
  veriphy_certified
    ( subst hu
      ring1)
  -- certificate: de-ode.sage

/-- … and `u(1) = 0`. -/
theorem loaded_string_right (u : ℝ → ℝ) (hu : u = fun s => s * (1 - s) / 2) : u 1 = 0 := by
  veriphy_certified
    ( subst hu
      ring1)
  -- certificate: de-ode.sage

/-! ## exm-bvp-eigenvalue -/

/-- `u = sin nx` solves `u'' + n²u = 0` … -/
theorem sine_mode_solves (n : ℝ) (u : ℝ → ℝ) (hu : u = fun s => Real.sin (n * s)) (x : ℝ) :
    deriv (deriv u) x + n ^ 2 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun (s : ℝ) => (sin (n * s) : ℝ)) = fun (s : ℝ) => (n * (Real.cos (s * n)) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (sin (n * s) : ℝ)) ((n * (Real.cos (s * n)) : ℝ)) (s) :=
          by
          refine
            (((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).sin)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · (ring_nf; done)
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (n * (Real.cos (s * n)) : ℝ)) = fun (s : ℝ) => (-(n ^ 2) * (Real.sin (s * n)) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (n * (Real.cos (s * n)) : ℝ)) ((-(n ^ 2) * (Real.sin (s * n)) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).cos).const_mul (n : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (ring_nf; done)
        exact hd.deriv
      simp only [d1, d2]
      all_goals ((ring_nf; done)))
  -- certificate: de-ode.sage

/-- … and vanishes at `0` and at `π` for integer `n`: `sin nπ = 0`. -/
theorem sine_mode_boundary (n : ℕ) : Real.sin (n * 0) = 0 ∧ Real.sin (n * Real.pi) = 0 := by
  simp [Real.sin_nat_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.sine_mode_boundary
-- test: de-sine-boundary-test.sage

/-! ## exm-bvp-no-solution -/

/-- Every solution of `u'' + u = 0` is `u(0) cos x + u'(0) sin x` (uniqueness by the conserved `w² + w'²`). -/
theorem harmonic_general (u u' : ℝ → ℝ) (hu : ∀ x, HasDerivAt u (u' x) x) (hu' : ∀ x, HasDerivAt u' (-u x) x)
    (x : ℝ) : u x = u 0 * Real.cos x + u' 0 * Real.sin x := by
  set w := fun s => u s - (u 0 * Real.cos s + u' 0 * Real.sin s)
  set w' := fun s => u' s - (-(u 0) * Real.sin s + u' 0 * Real.cos s)
  have hw : ∀ s, HasDerivAt w (w' s) s := by
    intro s
    have := (hu s).sub (((Real.hasDerivAt_cos s).const_mul (u 0)).add ((Real.hasDerivAt_sin s).const_mul (u' 0)))
    convert this using 1; simp only [w']; ring
  have hw' : ∀ s, HasDerivAt w' (-w s) s := by
    intro s
    have := (hu' s).sub (((Real.hasDerivAt_sin s).const_mul (-(u 0))).add ((Real.hasDerivAt_cos s).const_mul (u' 0)))
    convert this using 1; simp only [w]; ring
  have hE : ∀ s, HasDerivAt (fun s => w s ^ 2 + w' s ^ 2) 0 s := by
    intro s
    convert ((hw s).pow 2).add ((hw' s).pow 2) using 1
    ring
  have hc := is_const_of_deriv_eq_zero (fun s => (hE s).differentiableAt) (fun s => (hE s).deriv) x 0
  have h0 : w 0 = 0 := by simp [w]
  have h0' : w' 0 = 0 := by simp [w']
  simp only [h0, h0'] at hc
  have : w x = 0 := by nlinarith [sq_nonneg (w x), sq_nonneg (w' x)]
  simp only [w] at this
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.harmonic_general
-- test: de-harmonic-general-test.sage

/-- Hence `u(0) = 0` forces `u = C sin x` and `u(π) = 0`: the problem with `u(π) = 1` has no solution. -/
theorem bvp_no_solution (u u' : ℝ → ℝ) (hu : ∀ x, HasDerivAt u (u' x) x) (hu' : ∀ x, HasDerivAt u' (-u x) x)
    (h0 : u 0 = 0) : u Real.pi = 0 := by
  rw [harmonic_general u u' hu hu' Real.pi, h0]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.bvp_no_solution
-- test: de-bvp-no-solution-test.sage

/-- With `u(π) = 0` instead, every `C sin x` is a solution: infinitely many. -/
theorem bvp_many_solutions (C : ℝ) (u : ℝ → ℝ) (hu : u = fun s => C * Real.sin s) (x : ℝ) :
    deriv (deriv u) x + u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => C * sin s) = fun s => C * (Real.cos s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => C * sin s) ((C * (Real.cos s) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).sin).const_mul (C : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv
      have d2 : deriv (fun s => C * (Real.cos s)) = fun s => -C * (Real.sin s) :=
        by
        funext s
        have hd : HasDerivAt (fun s => C * (Real.cos s)) ((-C * (Real.sin s) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).cos).const_mul (C : ℝ)) using 1
          all_goals (ring1)
        exact hd.deriv
      simp only [d1, d2]
      ring1)
  -- certificate: de-ode.sage

/-! ## exr-bvp-solve -/

/-- `u = sin(πx)/π²` solves `-u'' = sin πx` … -/
theorem dirichlet_sine_solves (x : ℝ) :
    -deriv (fun s => deriv (fun r => Real.sin (Real.pi * r) / Real.pi ^ 2) s) x = Real.sin (Real.pi * x) := by
  veriphy_certified
    ( have i0 :
        deriv (fun (r : ℝ) => (sin (π * r) / π ^ 2 : ℝ)) = fun (d1 : ℝ) => ((Real.cos (Real.pi * d1)) / Real.pi : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (r : ℝ) => (sin (π * r) / π ^ 2 : ℝ)) (((Real.cos (Real.pi * d1)) / Real.pi : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).const_mul (Real.pi : ℝ)).sin).const_mul
                      (1 / Real.pi ^ 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (ring_nf; done)
          · (field_simp; done)
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => (fun (d1 : ℝ) => ((Real.cos (Real.pi * d1)) / Real.pi : ℝ)) s)
          ((-(Real.sin (Real.pi * x)) : ℝ)) (x) :=
        by
        refine
          ((((((hasDerivAt_id' (x : ℝ)).const_mul (Real.pi : ℝ)).cos).const_mul
                    (1 / Real.pi ^ 1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (ring_nf; done)
        · (field_simp; done)
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- … with `u(0) = 0` … -/
theorem dirichlet_sine_left (u : ℝ → ℝ) (hu : u = fun s => Real.sin (Real.pi * s) / Real.pi ^ 2) : u 0 = 0 := by
  veriphy_certified
    ( subst hu
      (norm_num; done))
  -- certificate: de-ode.sage

/-- … and `u(1) = 0`. -/
theorem dirichlet_sine_right (u : ℝ → ℝ) (hu : u = fun s => Real.sin (Real.pi * s) / Real.pi ^ 2) : u 1 = 0 := by
  veriphy_certified
    ( subst hu
      (norm_num; done))
  -- certificate: de-ode.sage

/-! ## exr-sl-dirichlet-eigenvalues -/

/-- `λ = -μ² < 0`: `e^{±μx}` solve `u'' - μ²u = 0` … -/
theorem negative_case_solves (μ C₁ C₂ : ℝ) (u : ℝ → ℝ) (hu : u = fun s => C₁ * Real.exp (μ * s) + C₂ * Real.exp (-μ * s))
    (x : ℝ) : deriv (deriv u) x - μ ^ 2 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 :
        deriv (fun s => C₁ * rexp (μ * s) + C₂ * rexp (-μ * s)) = fun s =>
          μ * C₁ * (Real.exp (s * μ)) - (μ * C₂ * (Real.exp (-s * μ))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => C₁ * rexp (μ * s) + C₂ * rexp (-μ * s))
            ((μ * C₁ * (Real.exp (s * μ)) - (μ * C₂ * (Real.exp (-s * μ))) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (μ : ℝ)).exp).const_mul (C₁ : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (-μ : ℝ)).exp).const_mul (C₂ : ℝ))) using
            1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 :
        deriv (fun s => μ * C₁ * (Real.exp (s * μ)) - (μ * C₂ * (Real.exp (-s * μ)))) = fun s =>
          (μ ^ 2) * C₁ * (Real.exp (s * μ)) + ((μ ^ 2) * C₂ * (Real.exp (-s * μ))) :=
        by
        funext s
        have hd :
          HasDerivAt (fun s => μ * C₁ * (Real.exp (s * μ)) - (μ * C₂ * (Real.exp (-s * μ))))
            (((μ ^ 2) * C₁ * (Real.exp (s * μ)) + ((μ ^ 2) * C₂ * (Real.exp (-s * μ))) : ℝ)) (s) :=
          by
          convert
            (((((hasDerivAt_id' (s : ℝ)).const_mul (μ : ℝ)).exp).const_mul (μ * C₁ : ℝ)).add
              ((((hasDerivAt_id' (s : ℝ)).const_mul (-μ : ℝ)).exp).const_mul (-μ * C₂ : ℝ))) using
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

/-- … and `u(0) = 0` (`C₂ = -C₁`) with `u(π) = 2C₁ sinh μπ = 0` forces `C₁ = 0`. -/
theorem negative_case_trivial (μ C : ℝ) (hμ : 0 < μ) (h : C * Real.exp (μ * Real.pi) - C * Real.exp (-μ * Real.pi) = 0) :
    C = 0 := by
  have hne : Real.exp (μ * Real.pi) - Real.exp (-μ * Real.pi) ≠ 0 := by
    have : Real.exp (-μ * Real.pi) < Real.exp (μ * Real.pi) :=
      Real.exp_lt_exp.mpr (by nlinarith [Real.pi_pos])
    linarith
  have : C * (Real.exp (μ * Real.pi) - Real.exp (-μ * Real.pi)) = 0 := by linarith
  exact (mul_eq_zero.mp this).resolve_right hne
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.negative_case_trivial
-- test: de-negative-case-test.sage

/-- `λ = 0`: `u = C₁ + C₂x` with `u(0) = u(π) = 0` forces `C₁ = C₂ = 0`. -/
theorem zero_case_trivial (C₁ C₂ : ℝ) (h₀ : C₁ + C₂ * 0 = 0) (hπ : C₁ + C₂ * Real.pi = 0) : C₁ = 0 ∧ C₂ = 0 := by
  have h1 : C₁ = 0 := by simpa using h₀
  refine ⟨h1, ?_⟩
  rw [h1, zero_add] at hπ
  exact (mul_eq_zero.mp hπ).resolve_right Real.pi_ne_zero
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.zero_case_trivial
-- test: de-zero-case-test.sage

/-- `λ = k² > 0`: `u(π) = C₂ sin kπ = 0` allows `C₂ ≠ 0` exactly when `k` is an integer `n ≥ 1`. -/
theorem positive_case_integer (k : ℝ) (hk : 0 < k) : Real.sin (k * Real.pi) = 0 ↔ ∃ n : ℕ, 1 ≤ n ∧ k = n := by
  constructor
  · intro h
    obtain ⟨m, hm⟩ := Real.sin_eq_zero_iff.mp h
    have hmk : (m : ℝ) = k := mul_right_cancel₀ Real.pi_ne_zero hm
    have hm0 : 0 < m := by exact_mod_cast hmk ▸ hk
    refine ⟨m.toNat, by omega, ?_⟩
    rw [← hmk]
    exact_mod_cast (Int.toNat_of_nonneg hm0.le).symm
  · rintro ⟨n, -, rfl⟩
    exact Real.sin_nat_mul_pi n
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.positive_case_integer
-- test: de-positive-case-test.sage

/-! ## exr-bvp-alternative -/

/-- Uniqueness: the difference of two solutions of `u'' + λu = f` solves the homogeneous equation. -/
theorem bvp_difference (l : ℝ) (u₁ u₂ d₁ d₂ f : ℝ → ℝ) (x : ℝ) (h₁ : HasDerivAt (deriv u₁) (d₁ x) x)
    (h₂ : HasDerivAt (deriv u₂) (d₂ x) x) (e₁ : d₁ x + l * u₁ x = f x) (e₂ : d₂ x + l * u₂ x = f x) :
    (d₁ x - d₂ x) + l * (u₁ x - u₂ x) = 0 := by
  linear_combination e₁ - e₂
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.bvp_difference
-- test: de-bvp-difference-test.sage

/-- The solvability condition: for `u` with `u(0) = u(π) = 0` and integer `n`,
`∫₀^π (u'' + n²u) sin nx dx = 0` (two integrations by parts: the integrand is the derivative of
`u' sin nx - n u cos nx`). -/
theorem solvability_condition (n : ℕ) (u u' u'' : ℝ → ℝ) (hu : ∀ x, HasDerivAt u (u' x) x)
    (hu' : ∀ x, HasDerivAt u' (u'' x) x) (hc : Continuous u'') (h0 : u 0 = 0) (hπ : u Real.pi = 0) :
    ∫ x in (0 : ℝ)..Real.pi, (u'' x + (n : ℝ) ^ 2 * u x) * Real.sin (n * x) = 0 := by
  have hF : ∀ x, HasDerivAt (fun x => u' x * Real.sin (n * x) - n * u x * Real.cos (n * x))
      ((u'' x + (n : ℝ) ^ 2 * u x) * Real.sin (n * x)) x := by
    intro x
    have hs : HasDerivAt (fun x => Real.sin (n * x)) (Real.cos (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).sin
    have hc' : HasDerivAt (fun x => Real.cos (n * x)) (-Real.sin (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).cos
    convert ((hu' x).mul hs).sub (((hu x).const_mul (n : ℝ)).mul hc') using 1
    ring
  have hcont : Continuous fun x => (u'' x + (n : ℝ) ^ 2 * u x) * Real.sin (n * x) := by
    have : Continuous u := continuous_iff_continuousAt.mpr fun x => (hu x).continuousAt
    fun_prop
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF x) (hcont.intervalIntegrable _ _)]
  simp [h0, hπ, Real.sin_nat_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BoundaryValueProblems.solvability_condition
-- test: de-solvability-test.sage

/-- Non-uniqueness at an eigenvalue: adding `C sin nx` to a solution of `u'' + n²u = f` gives another. -/
theorem eigen_nonunique (n C : ℝ) (v : ℝ → ℝ) (hv : v = fun s => C * Real.sin (n * s)) (x : ℝ) :
    deriv (deriv v) x + n ^ 2 * v x = 0 := by
  veriphy_certified
    ( subst hv
      have d1 : deriv (fun s => C * sin (n * s)) = fun s => n * C * (Real.cos (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => C * sin (n * s)) ((n * C * (Real.cos (s * n)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).sin).const_mul (C : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 : deriv (fun s => n * C * (Real.cos (s * n))) = fun s => -(n ^ 2) * C * (Real.sin (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => n * C * (Real.cos (s * n))) ((-(n ^ 2) * C * (Real.sin (s * n)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).cos).const_mul (n * C : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

end KaluzaKlein.DifferentialEquations.BoundaryValueProblems

namespace KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems

/-! ## exm-sl-dirichlet -/

/-- `-u'' = n²u` for `u = sin nx`: the string's eigenfunctions. -/
theorem string_eigen (n : ℝ) (u : ℝ → ℝ) (hu : u = fun s => Real.sin (n * s)) (x : ℝ) :
    -deriv (deriv u) x = n ^ 2 * u x := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => sin (n * s)) = fun s => n * (Real.cos (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => sin (n * s)) ((n * (Real.cos (s * n)) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).sin) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 : deriv (fun s => n * (Real.cos (s * n))) = fun s => -(n ^ 2) * (Real.sin (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => n * (Real.cos (s * n))) ((-(n ^ 2) * (Real.sin (s * n)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).cos).const_mul (n : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-! ## exm-sl-legendre -/

/-- The Sturm-Liouville form expands: `((1 - x²)u')' = (1 - x²)u'' - 2xu'`. -/
theorem legendre_form (u u' u'' : ℝ → ℝ) (x : ℝ) (hu : HasDerivAt u' (u'' x) x) :
    HasDerivAt (fun s => (1 - s ^ 2) * u' s) ((1 - x ^ 2) * u'' x - 2 * x * u' x) x := by
  convert (((hasDerivAt_pow 2 x).const_sub 1).mul hu) using 1
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.legendre_form
-- test: de-legendre-form-test.sage

/-- `p = 1 - x²` vanishes at `x = ±1`: the problem is singular. -/
theorem legendre_singular (x : ℝ) (h : x ^ 2 = 1) : 1 - x ^ 2 = 0 := by
  veriphy_certified (linear_combination (-1) * h)
  -- certificate: de-poly.sage

/-- From the sphere: with `x = cos θ`, `d/dθ(sin θ dP₂(cos θ)/dθ) = sin θ ((1 - x²)P₂'' - 2xP₂')`. -/
theorem legendre_sphere_P2 (θ : ℝ) :
    deriv (fun t => Real.sin t * deriv (fun φ => (3 * Real.cos φ ^ 2 - 1) / 2) t) θ =
      Real.sin θ * ((1 - Real.cos θ ^ 2) * 3 - 2 * Real.cos θ * (3 * Real.cos θ)) := by
  veriphy_certified
    ( have i0 :
        deriv (fun (φ : ℝ) => ((3 * cos φ ^ 2 - 1) / 2 : ℝ)) = fun (d1 : ℝ) =>
          (-3 * (Real.cos d1) * (Real.sin d1) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (φ : ℝ) => ((3 * cos φ ^ 2 - 1) / 2 : ℝ)) ((-3 * (Real.cos d1) * (Real.sin d1) : ℝ)) (d1) :=
          by
          refine
            (((((((hasDerivAt_id' (d1 : ℝ)).cos).pow 2).const_mul (3 / 2 : ℝ)).add
                      (hasDerivAt_const (d1 : ℝ) (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun t => sin t * (fun (d1 : ℝ) => (-3 * (Real.cos d1) * (Real.sin d1) : ℝ)) t)
          ((-6 * ((Real.cos θ) ^ 2) * (Real.sin θ) + (3 * ((Real.sin θ) ^ 3)) : ℝ)) (θ) :=
        by
        refine
          ((((((hasDerivAt_id' (θ : ℝ)).cos).mul (((hasDerivAt_id' (θ : ℝ)).sin).pow 2)).const_mul
                    (-3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (simp; ring_nf; done)
      try rw [h1.deriv]
      try beta_reduce
      all_goals ((linear_combination (3 * (Real.sin θ)) * Real.sin_sq_add_cos_sq (θ))))
  -- certificate: de-partials.sage

/-! ## exm-sl-bessel -/

/-- The Bessel form expands: `(xu')' = xu'' + u'`, so `-(xu')' + n²u/x = λxu` is `x²u'' + xu' + (λx² - n²)u = 0`
after multiplying by `-x`. -/
theorem bessel_form (u' u'' : ℝ → ℝ) (x : ℝ) (hu : HasDerivAt u' (u'' x) x) :
    HasDerivAt (fun s => s * u' s) (x * u'' x + u' x) x := by
  convert (hasDerivAt_id' x).mul hu using 1
  simp; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.bessel_form
-- test: de-bessel-form-test.sage

/-- … and the two forms agree: `-x(-(xu')' + n²u/x - λxu) = x²u'' + xu' + (λx² - n²)u`. -/
theorem bessel_forms_agree (n l x u u' u'' : ℝ) (hx : x ≠ 0) :
    -x * (-(x * u'' + u') + n ^ 2 / x * u - l * x * u) = x ^ 2 * u'' + x * u' + (l * x ^ 2 - n ^ 2) * u := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-! ## exr-sl-hermite-form -/

/-- If `p' = (b/a)p` then `(pu')' = (p/a)(au'' + bu')`: the multiplier `μ = p/a` brings
`au'' + bu' + cu = λu` into Sturm-Liouville form. -/
theorem sl_multiplier (a b p u' : ℝ → ℝ) (u'' x : ℝ) (ha : a x ≠ 0) (hp : HasDerivAt p (b x / a x * p x) x)
    (hu : HasDerivAt u' u'' x) : HasDerivAt (fun s => p s * u' s) (p x / a x * (a x * u'' + b x * u' x)) x := by
  convert hp.mul hu using 1
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_multiplier
-- test: de-sl-multiplier-test.sage

/-- Hermite: `p = e^{-x²}` (`p' = -2xp`), and `H₂ = 4x² - 2` is an eigenfunction:
`(e^{-x²}H₂')' = -4e^{-x²}H₂` (`λ = 4`). -/
theorem hermite_H2 (x : ℝ) :
    deriv (fun s => Real.exp (-s ^ 2) * deriv (fun r => 4 * r ^ 2 - 2) s) x = -4 * Real.exp (-x ^ 2) * (4 * x ^ 2 - 2) := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => (4 * r ^ 2 - 2 : ℝ)) = fun (d1 : ℝ) => (8 * d1 : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => (4 * r ^ 2 - 2 : ℝ)) ((8 * d1 : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (4 : ℝ)).add
                      (hasDerivAt_const (d1 : ℝ) (-2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt (fun s => rexp (-s ^ 2) * (fun (d1 : ℝ) => (8 * d1 : ℝ)) s)
          ((-16 * (x ^ 2) * (Real.exp (-(x ^ 2))) + (8 * (Real.exp (-(x ^ 2)))) : ℝ)) (x) :=
        by
        refine
          (((((hasDerivAt_id' (x : ℝ)).mul ((((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-1 : ℝ)).exp)).const_mul
                    (8 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (simp; ring_nf; done)
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-! ## exr-sl-neumann -/

/-- `cos nx` solves `-u'' = n²u` … -/
theorem neumann_eigen (n : ℝ) (u : ℝ → ℝ) (hu : u = fun s => Real.cos (n * s)) (x : ℝ) :
    -deriv (deriv u) x = n ^ 2 * u x := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun s => cos (n * s)) = fun s => -n * (Real.sin (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => cos (n * s)) ((-n * (Real.sin (s * n)) : ℝ)) (s) :=
          by
          convert (((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).cos) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      have d2 : deriv (fun s => -n * (Real.sin (s * n))) = fun s => -(n ^ 2) * (Real.cos (s * n)) :=
        by
        funext s
        have hd : HasDerivAt (fun s => -n * (Real.sin (s * n))) ((-(n ^ 2) * (Real.cos (s * n)) : ℝ)) (s) :=
          by
          convert ((((hasDerivAt_id' (s : ℝ)).const_mul (n : ℝ)).sin).const_mul (-n : ℝ)) using 1
          all_goals ((ring_nf; done))
        exact hd.deriv
      simp only [d1, d2]
      (ring_nf; done))
  -- certificate: de-ode.sage

/-- … with `u'(0) = 0` and `u'(π) = -n sin nπ = 0` for integer `n`. -/
theorem neumann_boundary (n : ℕ) :
    deriv (fun s => Real.cos (n * s)) 0 = 0 ∧ deriv (fun s => Real.cos (n * s)) Real.pi = 0 := by
  have h : ∀ x, deriv (fun s => Real.cos (n * s)) x = -Real.sin (n * x) * n := by
    intro x; exact (by simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).cos : HasDerivAt _ _ x).deriv
  simp [h, Real.sin_nat_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.neumann_boundary
-- test: de-neumann-boundary-test.sage

/-- `λ = 0` has the constant eigenfunction: `u = 1` solves `-u'' = 0·u` with `u' = 0`. -/
theorem neumann_constant (x : ℝ) : -deriv (fun s => deriv (fun _ : ℝ => (1 : ℝ)) s) x = 0 * 1 := by
  veriphy_certified
    ( have i0 : deriv (fun (x : ℝ) => (1 : ℝ)) = fun (d1 : ℝ) => (0 : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (x : ℝ) => (1 : ℝ)) ((0 : ℝ)) (d1) :=
          by
          refine
            (((hasDerivAt_const (d1 : ℝ) (1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun s => (fun (d1 : ℝ) => (0 : ℝ)) s) ((0 : ℝ)) (x) :=
        by
        refine
          (((hasDerivAt_const (x : ℝ) (0 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · rfl
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- Negative `λ = -μ²`: `u'(0) = μ(C₁ - C₂) = 0` and `u'(π) = μ(C₁e^{μπ} - C₂e^{-μπ}) = 0` force `C₁ = C₂ = 0`. -/
theorem neumann_negative (μ C₁ C₂ : ℝ) (hμ : 0 < μ) (h₀ : μ * (C₁ - C₂) = 0)
    (hπ : μ * (C₁ * Real.exp (μ * Real.pi) - C₂ * Real.exp (-μ * Real.pi)) = 0) : C₁ = 0 ∧ C₂ = 0 := by
  have h1 : C₁ = C₂ := by
    have := (mul_eq_zero.mp h₀).resolve_left hμ.ne'; linarith
  subst h1
  have h2 := (mul_eq_zero.mp hπ).resolve_left hμ.ne'
  have hne : Real.exp (μ * Real.pi) - Real.exp (-μ * Real.pi) ≠ 0 := by
    have : Real.exp (-μ * Real.pi) < Real.exp (μ * Real.pi) := Real.exp_lt_exp.mpr (by nlinarith [Real.pi_pos])
    linarith
  have : C₁ * (Real.exp (μ * Real.pi) - Real.exp (-μ * Real.pi)) = 0 := by linarith
  have := (mul_eq_zero.mp this).resolve_right hne
  exact ⟨this, this⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.neumann_negative
-- test: de-neumann-negative-test.sage

/-! ## exr-sl-nonnegative -/

/-- Integration by parts: for `u(a) = u(b) = 0` and `P = pu'`, `∫ₐᵇ -P'u = ∫ₐᵇ Pu' = ∫ₐᵇ pu'²`. -/
theorem sl_energy_identity (a b : ℝ) (P P' u u' : ℝ → ℝ) (hP : ∀ x, HasDerivAt P (P' x) x)
    (hu : ∀ x, HasDerivAt u (u' x) x) (hPc : Continuous P') (huc : Continuous u')
    (ha : u a = 0) (hb : u b = 0) :
    ∫ x in a..b, -(P' x) * u x = ∫ x in a..b, P x * u' x := by
  have hd : ∀ x, HasDerivAt (fun x => P x * u x) (P' x * u x + P x * u' x) x := fun x => (hP x).mul (hu x)
  have hcP : Continuous P := continuous_iff_continuousAt.mpr fun x => (hP x).continuousAt
  have hcu : Continuous u := continuous_iff_continuousAt.mpr fun x => (hu x).continuousAt
  have hint := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x)
    ((by fun_prop : Continuous fun x => P' x * u x + P x * u' x).intervalIntegrable a b)
  rw [ha, hb, mul_zero, mul_zero, sub_zero, intervalIntegral.integral_add
    ((by fun_prop : Continuous fun x => P' x * u x).intervalIntegrable a b)
    ((by fun_prop : Continuous fun x => P x * u' x).intervalIntegrable a b)] at hint
  have : ∫ x in a..b, -(P' x) * u x = -∫ x in a..b, P' x * u x := by
    rw [← intervalIntegral.integral_neg]; congr 1; funext x; ring
  rw [this]
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_energy_identity
-- test: de-sl-energy-test.sage

/-- Hence `λ ≥ 0`: `λ ∫wu² = ∫(pu'² + qu²)` with `∫wu² > 0` and the right side `≥ 0`. -/
theorem sl_nonnegative (l W P Q : ℝ) (hW : 0 < W) (hP : 0 ≤ P) (hQ : 0 ≤ Q) (h : l * W = P + Q) : 0 ≤ l := by
  by_contra hl
  push_neg at hl
  nlinarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_nonnegative
-- test: de-sl-nonnegative-test.sage

end KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems

namespace KaluzaKlein.DifferentialEquations.OrthogonalFunctions

/-! ## def-orthogonal-functions -/

/-- The weighted inner product `⟨f, g⟩_w = ∫ₐᵇ f g w` is symmetric … -/
theorem inner_symm (a b : ℝ) (f g w : ℝ → ℝ) : ∫ x in a..b, f x * g x * w x = ∫ x in a..b, g x * f x * w x := by
  congr 1; funext x; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.inner_symm
-- test: de-inner-symm-test.sage

/-- … and linear in each argument (for integrable functions). -/
theorem inner_add_left (a b c : ℝ) (f₁ f₂ g w : ℝ → ℝ) (h₁ : Continuous f₁) (h₂ : Continuous f₂)
    (hg : Continuous g) (hw : Continuous w) :
    ∫ x in a..b, (f₁ x + c * f₂ x) * g x * w x =
      (∫ x in a..b, f₁ x * g x * w x) + c * ∫ x in a..b, f₂ x * g x * w x := by
  rw [← intervalIntegral.integral_const_mul, ← intervalIntegral.integral_add
    ((by fun_prop : Continuous fun x => f₁ x * g x * w x).intervalIntegrable a b)
    ((by fun_prop : Continuous fun x => c * (f₂ x * g x * w x)).intervalIntegrable a b)]
  congr 1; funext x; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.inner_add_left
-- test: de-inner-add-test.sage

/-! ## exm-orth-sines -/

/-- `∫₀^π cos kx dx = 0` for a non-zero integer `k`. -/
theorem integral_cos_int (k : ℤ) (hk : k ≠ 0) : ∫ x in (0 : ℝ)..Real.pi, Real.cos (k * x) = 0 := by
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) (by exact_mod_cast hk), integral_cos]
  simp [Real.sin_int_mul_pi]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.integral_cos_int
-- test: de-integral-cos-int-test.sage

/-- The sines are orthogonal on `[0, π]`: `∫₀^π sin mx sin nx = 0` for `m ≠ n`. -/
theorem sin_orthogonal (m n : ℕ) (h : m ≠ n) :
    ∫ x in (0 : ℝ)..Real.pi, Real.sin (m * x) * Real.sin (n * x) = 0 := by
  have hrw : (fun x : ℝ => Real.sin (m * x) * Real.sin (n * x)) =
      fun x => (Real.cos (((m : ℤ) - n : ℤ) * x) - Real.cos (((m : ℤ) + n : ℤ) * x)) / 2 := by
    funext x
    push_cast
    rw [sub_mul, add_mul, Real.cos_sub, Real.cos_add]
    ring
  have h1 : ((m : ℤ) - n) ≠ 0 := by omega
  have h2 : ((m : ℤ) + n) ≠ 0 := by omega
  rw [hrw, intervalIntegral.integral_div, intervalIntegral.integral_sub
    (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop),
    integral_cos_int _ h1, integral_cos_int _ h2]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.sin_orthogonal
-- test: de-sin-orthogonal-test.sage

/-- … with `∫₀^π sin² nx = π/2` for `n ≥ 1`. -/
theorem sin_norm (n : ℕ) (hn : n ≠ 0) : ∫ x in (0 : ℝ)..Real.pi, Real.sin (n * x) ^ 2 = Real.pi / 2 := by
  have hrw : (fun x : ℝ => Real.sin (n * x) ^ 2) = fun x => (1 - Real.cos (((2 * n : ℕ) : ℤ) * x)) / 2 := by
    funext x
    push_cast
    rw [show 2 * (n : ℝ) * x = 2 * (n * x) by ring, Real.cos_two_mul, Real.sin_sq]
    ring
  have h2 : (((2 * n : ℕ) : ℤ)) ≠ 0 := by omega
  rw [hrw, intervalIntegral.integral_div, intervalIntegral.integral_sub
    (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop),
    integral_cos_int _ h2]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.sin_norm
-- test: de-sin-norm-test.sage

/-- The cosines are orthogonal on `[0, π]` too: `∫₀^π cos mx cos nx = 0` for `m ≠ n`. -/
theorem cos_orthogonal (m n : ℕ) (h : m ≠ n) :
    ∫ x in (0 : ℝ)..Real.pi, Real.cos (m * x) * Real.cos (n * x) = 0 := by
  have hrw : (fun x : ℝ => Real.cos (m * x) * Real.cos (n * x)) =
      fun x => (Real.cos (((m : ℤ) - n : ℤ) * x) + Real.cos (((m : ℤ) + n : ℤ) * x)) / 2 := by
    funext x
    push_cast
    rw [sub_mul, add_mul, Real.cos_sub, Real.cos_add]
    ring
  have h1 : ((m : ℤ) - n) ≠ 0 := by omega
  have h2 : ((m : ℤ) + n) ≠ 0 := by omega
  rw [hrw, intervalIntegral.integral_div, intervalIntegral.integral_add
    (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop),
    integral_cos_int _ h1, integral_cos_int _ h2]
  simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.cos_orthogonal
-- test: de-cos-orthogonal-test.sage

/-- On `[-π, π]`, `cos mx` and `sin nx` are orthogonal for all `m, n` (the integrand is odd). -/
theorem cos_sin_orthogonal (m n : ℝ) : ∫ x in (-Real.pi)..Real.pi, Real.cos (m * x) * Real.sin (n * x) = 0 := by
  have h := intervalIntegral.integral_comp_neg (a := -Real.pi) (b := Real.pi)
    (fun x => Real.cos (m * x) * Real.sin (n * x))
  simp only [mul_neg, Real.cos_neg, Real.sin_neg, neg_neg] at h
  rw [intervalIntegral.integral_neg] at h
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.cos_sin_orthogonal
-- test: de-cos-sin-orth-test.sage

/-- `√(2/π) sin nx` has norm `1`: `(√(2/π))² · π/2 = 1`. -/
theorem sines_orthonormal : Real.sqrt (2 / Real.pi) ^ 2 * (Real.pi / 2) = 1 := by
  veriphy_certified
    ( rw [Real.sq_sqrt (by positivity)]
      field_simp)
  -- certificate: de-rewrite.sage

/-! ## exm-orth-legendre -/

/-- `√((2n+1)/2) Pₙ` has norm `1`: `(2n+1)/2 · 2/(2n+1) = 1`. -/
theorem legendre_orthonormal (n : ℕ) : ((2 * n + 1) / 2 : ℝ) * (2 / (2 * n + 1)) = 1 := by
  have : (2 * (n : ℝ) + 1) ≠ 0 := by positivity
  field_simp
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.legendre_orthonormal
-- test: de-legendre-orthonormal-test.sage

/-! ## exm-orth-chebyshev -/

/-- `T₁ = cos(arccos x) = x` on `[-1, 1]`. -/
theorem chebyshev_T1 (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) : Real.cos (Real.arccos x) = x := by
  veriphy_certified (exact Real.cos_arccos h₁ h₂)
  -- certificate: de-rewrite.sage

/-- `T₂ = cos(2 arccos x) = 2x² - 1` on `[-1, 1]`. -/
theorem chebyshev_T2 (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) : Real.cos (2 * Real.arccos x) = 2 * x ^ 2 - 1 := by
  veriphy_certified (rw [Real.cos_two_mul, Real.cos_arccos h₁ h₂])
  -- certificate: de-rewrite.sage

/-- Without the weight they are not orthogonal: `∫₋₁¹ T₀T₂ dx = -2/3`. -/
theorem chebyshev_unweighted : ∫ x in (-1 : ℝ)..1, (1 : ℝ) * (2 * x ^ 2 - 1) = -2 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (2 * (x_veriphy ^ 3) / 3 - x_veriphy : ℝ))
            ((fun x => 1 * (2 * x ^ 2 - 1)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (2 / 3 : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (2 * (x_veriphy ^ 3) / 3 - x_veriphy : ℝ)) (f' := fun x => 1 * (2 * x ^ 2 - 1)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-! ## exr-orth-sines (the tables; the sine table is also part of prp-legendre-bessel) -/

/-- `⟨sin x, sin x⟩ = π/2`. -/
theorem sin11 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (1 * x) * Real.sin (1 * x) = Real.pi / 2 := by
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
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 - ((Real.sin (2 * i0)) / 4) : ℝ)) ((fun x => sin (1 * x) * sin (1 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                      (-(1 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, one_mul, mul_one]; linear_combination (-1 / 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, Real.sin_two_pi,
        Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin x, sin 2x⟩ = 0`. -/
theorem sin12 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (1 * x) * Real.sin (2 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (3 * i0)) / 6 + ((Real.sin i0) / 2) : ℝ))
            ((fun x => sin (1 * x) * sin (2 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (3 : ℝ)).sin).const_mul (-(1 / 6) : ℝ)).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin x, sin 3x⟩ = 0`. -/
theorem sin13 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (1 * x) * Real.sin (3 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (4 * i0)) / 8 + ((Real.sin (2 * i0)) / 4) : ℝ))
            ((fun x => sin (1 * x) * sin (3 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (4 : ℝ)).sin).const_mul (-(1 / 8) : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0 ^ 2 + 1 / 2 * Real.sin i0 ^ 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin x, sin 4x⟩ = 0`. -/
theorem sin14 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (1 * x) * Real.sin (4 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (5 * i0)) / 10 + ((Real.sin (3 * i0)) / 6) : ℝ))
            ((fun x => sin (1 * x) * sin (4 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (5 : ℝ)).sin).const_mul (-(1 / 10) : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (3 : ℝ)).sin).const_mul (1 / 6 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 3 + 3 / 2 * Real.cos i0 * Real.sin i0 ^ 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 2x, sin 2x⟩ = π/2`. -/
theorem sin22 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (2 * x) * Real.sin (2 * x) = Real.pi / 2 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 - ((Real.sin (4 * i0)) / 8) : ℝ)) ((fun x => sin (2 * x) * sin (2 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (4 : ℝ)).sin).const_mul
                      (-(1 / 8) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0 ^ 2 - 1 / 2 * Real.sin i0 ^ 2 - 1 / 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 2x, sin 3x⟩ = 0`. -/
theorem sin23 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (2 * x) * Real.sin (3 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (5 * i0)) / 10 + ((Real.sin i0) / 2) : ℝ))
            ((fun x => sin (2 * x) * sin (3 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (5 : ℝ)).sin).const_mul (-(1 / 10) : ℝ)).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 3 - 1 / 2 * Real.cos i0 * Real.sin i0 ^ 2 - 1 / 2 * Real.cos i0) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 2x, sin 4x⟩ = 0`. -/
theorem sin24 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (2 * x) * Real.sin (4 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have c6 :
        ∀ y : ℝ,
          Real.cos (6 * y) =
            Real.cos y ^ 6 - 15 * Real.cos y ^ 4 * Real.sin y ^ 2 + 15 * Real.cos y ^ 2 * Real.sin y ^ 4 -
              Real.sin y ^ 6 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.cos_add, c5 y, s5 y]
        ring
      have s6 :
        ∀ y : ℝ,
          Real.sin (6 * y) =
            6 * Real.cos y ^ 5 * Real.sin y - 20 * Real.cos y ^ 3 * Real.sin y ^ 3 + 6 * Real.cos y * Real.sin y ^ 5 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.sin_add, c5 y, s5 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (6 * i0)) / 12 + ((Real.sin (2 * i0)) / 4) : ℝ))
            ((fun x => sin (2 * x) * sin (4 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (6 : ℝ)).sin).const_mul (-(1 / 12) : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 4 + 1 / 2 * Real.sin i0 ^ 4 - 1 / 2 * Real.cos i0 ^ 2 + 1 / 2 * Real.sin i0 ^ 2) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 3x, sin 3x⟩ = π/2`. -/
theorem sin33 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (3 * x) * Real.sin (3 * x) = Real.pi / 2 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have c6 :
        ∀ y : ℝ,
          Real.cos (6 * y) =
            Real.cos y ^ 6 - 15 * Real.cos y ^ 4 * Real.sin y ^ 2 + 15 * Real.cos y ^ 2 * Real.sin y ^ 4 -
              Real.sin y ^ 6 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.cos_add, c5 y, s5 y]
        ring
      have s6 :
        ∀ y : ℝ,
          Real.sin (6 * y) =
            6 * Real.cos y ^ 5 * Real.sin y - 20 * Real.cos y ^ 3 * Real.sin y ^ 3 + 6 * Real.cos y * Real.sin y ^ 5 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.sin_add, c5 y, s5 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 - ((Real.sin (6 * i0)) / 12) : ℝ)) ((fun x => sin (3 * x) * sin (3 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (6 : ℝ)).sin).const_mul
                      (-(1 / 12) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 4 - Real.cos i0 ^ 2 * Real.sin i0 ^ 2 - 1 / 2 * Real.sin i0 ^ 4 -
                      1 / 2 * Real.cos i0 ^ 2 -
                    1 / 2 * Real.sin i0 ^ 2 -
                  1 / 2) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 3x, sin 4x⟩ = 0`. -/
theorem sin34 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (3 * x) * Real.sin (4 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have c6 :
        ∀ y : ℝ,
          Real.cos (6 * y) =
            Real.cos y ^ 6 - 15 * Real.cos y ^ 4 * Real.sin y ^ 2 + 15 * Real.cos y ^ 2 * Real.sin y ^ 4 -
              Real.sin y ^ 6 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.cos_add, c5 y, s5 y]
        ring
      have s6 :
        ∀ y : ℝ,
          Real.sin (6 * y) =
            6 * Real.cos y ^ 5 * Real.sin y - 20 * Real.cos y ^ 3 * Real.sin y ^ 3 + 6 * Real.cos y * Real.sin y ^ 5 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.sin_add, c5 y, s5 y]
        ring
      have c7 :
        ∀ y : ℝ,
          Real.cos (7 * y) =
            Real.cos y ^ 7 - 21 * Real.cos y ^ 5 * Real.sin y ^ 2 + 35 * Real.cos y ^ 3 * Real.sin y ^ 4 -
              7 * Real.cos y * Real.sin y ^ 6 :=
        by
        intro y
        rw [show (7 : ℝ) * y = 6 * y + y by ring, Real.cos_add, c6 y, s6 y]
        ring
      have s7 :
        ∀ y : ℝ,
          Real.sin (7 * y) =
            7 * Real.cos y ^ 6 * Real.sin y - 35 * Real.cos y ^ 4 * Real.sin y ^ 3 +
                21 * Real.cos y ^ 2 * Real.sin y ^ 5 -
              Real.sin y ^ 7 :=
        by
        intro y
        rw [show (7 : ℝ) * y = 6 * y + y by ring, Real.sin_add, c6 y, s6 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (-(Real.sin (7 * i0)) / 14 + ((Real.sin i0) / 2) : ℝ))
            ((fun x => sin (3 * x) * sin (4 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (7 : ℝ)).sin).const_mul (-(1 / 14) : ℝ)).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, c7, s7, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 5 - Real.cos i0 ^ 3 * Real.sin i0 ^ 2 - 1 / 2 * Real.cos i0 * Real.sin i0 ^ 4 -
                      1 / 2 * Real.cos i0 ^ 3 -
                    1 / 2 * Real.cos i0 * Real.sin i0 ^ 2 -
                  1 / 2 * Real.cos i0) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, c7, s7, one_mul, mul_one, Real.sin_pi, Real.cos_pi,
        Real.sin_zero, Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub,
        neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨sin 4x, sin 4x⟩ = π/2`. -/
theorem sin44 : ∫ x in (0 : ℝ)..Real.pi, Real.sin (4 * x) * Real.sin (4 * x) = Real.pi / 2 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have c6 :
        ∀ y : ℝ,
          Real.cos (6 * y) =
            Real.cos y ^ 6 - 15 * Real.cos y ^ 4 * Real.sin y ^ 2 + 15 * Real.cos y ^ 2 * Real.sin y ^ 4 -
              Real.sin y ^ 6 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.cos_add, c5 y, s5 y]
        ring
      have s6 :
        ∀ y : ℝ,
          Real.sin (6 * y) =
            6 * Real.cos y ^ 5 * Real.sin y - 20 * Real.cos y ^ 3 * Real.sin y ^ 3 + 6 * Real.cos y * Real.sin y ^ 5 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.sin_add, c5 y, s5 y]
        ring
      have c7 :
        ∀ y : ℝ,
          Real.cos (7 * y) =
            Real.cos y ^ 7 - 21 * Real.cos y ^ 5 * Real.sin y ^ 2 + 35 * Real.cos y ^ 3 * Real.sin y ^ 4 -
              7 * Real.cos y * Real.sin y ^ 6 :=
        by
        intro y
        rw [show (7 : ℝ) * y = 6 * y + y by ring, Real.cos_add, c6 y, s6 y]
        ring
      have s7 :
        ∀ y : ℝ,
          Real.sin (7 * y) =
            7 * Real.cos y ^ 6 * Real.sin y - 35 * Real.cos y ^ 4 * Real.sin y ^ 3 +
                21 * Real.cos y ^ 2 * Real.sin y ^ 5 -
              Real.sin y ^ 7 :=
        by
        intro y
        rw [show (7 : ℝ) * y = 6 * y + y by ring, Real.sin_add, c6 y, s6 y]
        ring
      have c8 :
        ∀ y : ℝ,
          Real.cos (8 * y) =
            Real.cos y ^ 8 - 28 * Real.cos y ^ 6 * Real.sin y ^ 2 + 70 * Real.cos y ^ 4 * Real.sin y ^ 4 -
                28 * Real.cos y ^ 2 * Real.sin y ^ 6 +
              Real.sin y ^ 8 :=
        by
        intro y
        rw [show (8 : ℝ) * y = 7 * y + y by ring, Real.cos_add, c7 y, s7 y]
        ring
      have s8 :
        ∀ y : ℝ,
          Real.sin (8 * y) =
            8 * Real.cos y ^ 7 * Real.sin y - 56 * Real.cos y ^ 5 * Real.sin y ^ 3 +
                56 * Real.cos y ^ 3 * Real.sin y ^ 5 -
              8 * Real.cos y * Real.sin y ^ 7 :=
        by
        intro y
        rw [show (8 : ℝ) * y = 7 * y + y by ring, Real.sin_add, c7 y, s7 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 - ((Real.sin (8 * i0)) / 16) : ℝ)) ((fun x => sin (4 * x) * sin (4 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (8 : ℝ)).sin).const_mul
                      (-(1 / 16) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, c7, s7, c8, s8, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 6 - 3 / 2 * Real.cos i0 ^ 4 * Real.sin i0 ^ 2 -
                                3 / 2 * Real.cos i0 ^ 2 * Real.sin i0 ^ 4 -
                              1 / 2 * Real.sin i0 ^ 6 -
                            1 / 2 * Real.cos i0 ^ 4 -
                          Real.cos i0 ^ 2 * Real.sin i0 ^ 2 -
                        1 / 2 * Real.sin i0 ^ 4 -
                      1 / 2 * Real.cos i0 ^ 2 -
                    1 / 2 * Real.sin i0 ^ 2 -
                  1 / 2) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, c7, s7, c8, s8, one_mul, mul_one, Real.sin_pi, Real.cos_pi,
        Real.sin_zero, Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub,
        neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage

/-- `⟨1, 1⟩ = π`: the constant mode has twice the norm squared of the others. -/
theorem cos00 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (0 * x) * Real.cos (0 * x) = Real.pi := by
  simp only [zero_mul, Real.cos_zero, one_mul]
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun x => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun x => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨1, cos x⟩ = 0`. -/
theorem cos01 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (0 * x) * Real.cos (1 * x) = 0 := by
  simp only [zero_mul, Real.cos_zero, one_mul]
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (Real.sin x_veriphy : ℝ)) ((fun x => cos x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).sin)).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.sin x_veriphy : ℝ)) (f' :=
          fun x => cos x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨1, cos 2x⟩ = 0`. -/
theorem cos02 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (0 * x) * Real.cos (2 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.sin (2 * x_veriphy)) / 2 : ℝ))
            ((fun x => cos (0 * x) * cos (2 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((Real.sin (2 * x_veriphy)) / 2 : ℝ))
          (f' := fun x => cos (0 * x) * cos (2 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((2 * (π) : ℝ)) = 0 := by
        rw [show ((2 * (π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨1, cos 3x⟩ = 0`. -/
theorem cos03 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (0 * x) * Real.cos (3 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.sin (3 * x_veriphy)) / 3 : ℝ))
            ((fun x => cos (0 * x) * cos (3 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).sin).const_mul (1 / 3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((Real.sin (3 * x_veriphy)) / 3 : ℝ))
          (f' := fun x => cos (0 * x) * cos (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((3 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((3 * (π) : ℝ)) = 0 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 3
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨cos x, cos x⟩ = π/2`. -/
theorem cos11 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (1 * x) * Real.cos (1 * x) = Real.pi / 2 := by
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
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 + ((Real.sin (2 * i0)) / 4) : ℝ)) ((fun x => cos (1 * x) * cos (1 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, one_mul, mul_one]; linear_combination (-1 / 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, Real.sin_two_pi,
        Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨cos x, cos 2x⟩ = 0`. -/
theorem cos12 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (1 * x) * Real.cos (2 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => ((Real.sin (3 * i0)) / 6 + ((Real.sin i0) / 2) : ℝ))
            ((fun x => cos (1 * x) * cos (2 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (3 : ℝ)).sin).const_mul (1 / 6 : ℝ)).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨cos x, cos 3x⟩ = 0`. -/
theorem cos13 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (1 * x) * Real.cos (3 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => ((Real.sin (4 * i0)) / 8 + ((Real.sin (2 * i0)) / 4) : ℝ))
            ((fun x => cos (1 * x) * cos (3 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (4 : ℝ)).sin).const_mul (1 / 8 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (2 : ℝ)).sin).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0 ^ 2 + 1 / 2 * Real.sin i0 ^ 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨cos 2x, cos 2x⟩ = π/2`. -/
theorem cos22 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (2 * x) * Real.cos (2 * x) = Real.pi / 2 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 + ((Real.sin (4 * i0)) / 8) : ℝ)) ((fun x => cos (2 * x) * cos (2 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (4 : ℝ)).sin).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one];
            linear_combination (-1 / 2 * Real.cos i0 ^ 2 - 1 / 2 * Real.sin i0 ^ 2 - 1 / 2) * Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero,
        Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨cos 2x, cos 3x⟩ = 0`. -/
theorem cos23 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (2 * x) * Real.cos (3 * x) = 0 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => ((Real.sin (5 * i0)) / 10 + ((Real.sin i0) / 2) : ℝ))
            ((fun x => cos (2 * x) * cos (3 * x)) i0) i0 :=
        by
        intro i0 _
        refine
          (((((((hasDerivAt_id' (i0 : ℝ)).const_mul (5 : ℝ)).sin).const_mul (1 / 10 : ℝ)).add
                    (((hasDerivAt_id' (i0 : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 3 - 1 / 2 * Real.cos i0 * Real.sin i0 ^ 2 - 1 / 2 * Real.cos i0) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage
/-- `⟨cos 3x, cos 3x⟩ = π/2`. -/
theorem cos33 : ∫ x in (0 : ℝ)..Real.pi, Real.cos (3 * x) * Real.cos (3 * x) = Real.pi / 2 := by
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
      have c3 : ∀ y : ℝ, Real.cos (3 * y) = Real.cos y ^ 3 - 3 * Real.cos y * Real.sin y ^ 2 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.cos_add, c2 y, s2 y]
        ring
      have s3 : ∀ y : ℝ, Real.sin (3 * y) = 3 * Real.cos y ^ 2 * Real.sin y - Real.sin y ^ 3 :=
        by
        intro y
        rw [show (3 : ℝ) * y = 2 * y + y by ring, Real.sin_add, c2 y, s2 y]
        ring
      have c4 : ∀ y : ℝ, Real.cos (4 * y) = Real.cos y ^ 4 - 6 * Real.cos y ^ 2 * Real.sin y ^ 2 + Real.sin y ^ 4 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.cos_add, c3 y, s3 y]
        ring
      have s4 : ∀ y : ℝ, Real.sin (4 * y) = 4 * Real.cos y ^ 3 * Real.sin y - 4 * Real.cos y * Real.sin y ^ 3 :=
        by
        intro y
        rw [show (4 : ℝ) * y = 3 * y + y by ring, Real.sin_add, c3 y, s3 y]
        ring
      have c5 :
        ∀ y : ℝ,
          Real.cos (5 * y) = Real.cos y ^ 5 - 10 * Real.cos y ^ 3 * Real.sin y ^ 2 + 5 * Real.cos y * Real.sin y ^ 4 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.cos_add, c4 y, s4 y]
        ring
      have s5 :
        ∀ y : ℝ,
          Real.sin (5 * y) = 5 * Real.cos y ^ 4 * Real.sin y - 10 * Real.cos y ^ 2 * Real.sin y ^ 3 + Real.sin y ^ 5 :=
        by
        intro y
        rw [show (5 : ℝ) * y = 4 * y + y by ring, Real.sin_add, c4 y, s4 y]
        ring
      have c6 :
        ∀ y : ℝ,
          Real.cos (6 * y) =
            Real.cos y ^ 6 - 15 * Real.cos y ^ 4 * Real.sin y ^ 2 + 15 * Real.cos y ^ 2 * Real.sin y ^ 4 -
              Real.sin y ^ 6 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.cos_add, c5 y, s5 y]
        ring
      have s6 :
        ∀ y : ℝ,
          Real.sin (6 * y) =
            6 * Real.cos y ^ 5 * Real.sin y - 20 * Real.cos y ^ 3 * Real.sin y ^ 3 + 6 * Real.cos y * Real.sin y ^ 5 :=
        by
        intro y
        rw [show (6 : ℝ) * y = 5 * y + y by ring, Real.sin_add, c5 y, s5 y]
        ring
      have hF :
        ∀ i0 ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun i0 => (i0 / 2 + ((Real.sin (6 * i0)) / 12) : ℝ)) ((fun x => cos (3 * x) * cos (3 * x)) i0)
            i0 :=
        by
        intro i0 _
        refine
          (((((hasDerivAt_id' (i0 : ℝ)).const_mul (1 / 2 : ℝ)).add
                    ((((hasDerivAt_id' (i0 : ℝ)).const_mul (6 : ℝ)).sin).const_mul (1 / 12 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · beta_reduce
          (simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one];
            linear_combination
              (-1 / 2 * Real.cos i0 ^ 4 - Real.cos i0 ^ 2 * Real.sin i0 ^ 2 - 1 / 2 * Real.sin i0 ^ 4 -
                      1 / 2 * Real.cos i0 ^ 2 -
                    1 / 2 * Real.sin i0 ^ 2 -
                  1 / 2) *
                Real.cos_sq_add_sin_sq i0)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      simp only [c2, s2, c3, s3, c4, s4, c5, s5, c6, s6, one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero,
        Real.cos_zero, Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero,
        zero_add]
      ring1)
  -- certificate: de-integral-trig.sage

/-! ## exr-orth-gram-schmidt -/

/-- `⟨1, 1⟩ = 2`. -/
theorem gs_one : ∫ x in (-1 : ℝ)..1, (1 : ℝ) = 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun x => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun x => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨x², 1⟩ = 2/3` (so the projection of `x²` on `1` is `1/3`). -/
theorem gs_x2 : ∫ x in (-1 : ℝ)..1, x ^ 2 = 2 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) ((fun x => x ^ 2) x_veriphy) x_veriphy :=
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
          fun x => x ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨x², x⟩ = 0`. -/
theorem gs_x3 : ∫ x in (-1 : ℝ)..1, x ^ 3 = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 4) / 4 : ℝ)) ((fun x => x ^ 3) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (1 / 4 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 4) / 4 : ℝ)) (f' :=
          fun x => x ^ 3) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `⟨x³, x⟩ = 2/5` (so the projection of `x³` on `x` is `(2/5)/(2/3) = 3/5`). -/
theorem gs_x4 : ∫ x in (-1 : ℝ)..1, x ^ 4 = 2 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) ((fun x => x ^ 4) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) (f' :=
          fun x => x ^ 4) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `e₂ = x² - 1/3` is orthogonal to `1` … -/
theorem gs_e2_one : ∫ x in (-1 : ℝ)..1, (x ^ 2 - 1 / 3) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 - (x_veriphy / 3) : ℝ)) ((fun x => x ^ 2 - 1 / 3) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-(1 / 3) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 3) / 3 - (x_veriphy / 3) : ℝ)) (f' := fun x => x ^ 2 - 1 / 3) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- … and `e₃ = x³ - (3/5)x` to `x` … -/
theorem gs_e3_x : ∫ x in (-1 : ℝ)..1, (x ^ 3 - 3 / 5 * x) * x = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 5 - ((x_veriphy ^ 3) / 5) : ℝ))
            ((fun x => (x ^ 3 - 3 / 5 * x) * x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(1 / 5) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 5) / 5 - ((x_veriphy ^ 3) / 5) : ℝ)) (f' := fun x => (x ^ 3 - 3 / 5 * x) * x) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- … and to `e₂`. -/
theorem gs_e3_e2 : ∫ x in (-1 : ℝ)..1, (x ^ 3 - 3 / 5 * x) * (x ^ 2 - 1 / 3) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 6) / 6 - (7 * (x_veriphy ^ 4) / 30) + ((x_veriphy ^ 2) / 10) : ℝ))
            ((fun x => (x ^ 3 - 3 / 5 * x) * (x ^ 2 - 1 / 3)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 6).const_mul (1 / 6 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (-(7 / 30) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 10 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 6) / 6 - (7 * (x_veriphy ^ 4) / 30) + ((x_veriphy ^ 2) / 10) : ℝ)) (f' := fun x =>
          (x ^ 3 - 3 / 5 * x) * (x ^ 2 - 1 / 3)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage
/-- `x² - 1/3 = (2/3)P₂`. -/
theorem gs_e2_P2 (x : ℝ) : x ^ 2 - 1 / 3 = 2 / 3 * ((3 * x ^ 2 - 1) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage
/-- `x³ - (3/5)x = (2/5)P₃`. -/
theorem gs_e3_P3 (x : ℝ) : x ^ 3 - 3 / 5 * x = 2 / 5 * ((5 * x ^ 3 - 3 * x) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exr-orth-expansion -/

/-- Completing the square: `‖f‖² - 2c⟨f, φ⟩ + c²‖φ‖² = ‖f‖² - ⟨f, φ⟩²/‖φ‖² + ‖φ‖²(c - ⟨f, φ⟩/‖φ‖²)²`,
least at `c = ⟨f, φ⟩/‖φ‖²`. -/
theorem complete_square (F I P c : ℝ) (hP : P ≠ 0) :
    F - 2 * c * I + c ^ 2 * P = F - I ^ 2 / P + P * (c - I / P) ^ 2 := by
  veriphy_certified
    ( field_simp
      ring)
  -- certificate: de-rewrite.sage

/-- The expansion `‖f - cφ‖² = ‖f‖² - 2c⟨f, φ⟩ + c²‖φ‖²` in any real inner product space. -/
theorem norm_sub_smul_sq {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (f φ : E) (c : ℝ) :
    ‖f - c • φ‖ ^ 2 = ‖f‖ ^ 2 - 2 * c * inner ℝ f φ + c ^ 2 * ‖φ‖ ^ 2 := by
  rw [@norm_sub_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.norm_sub_smul_sq
-- test: de-norm-sub-smul-test.sage

/-- Bessel's inequality: for an orthonormal family `ψₙ = φₙ/‖φₙ‖`, `Σ ⟨f, ψₙ⟩² ≤ ‖f‖²`, that is
`Σ cₙ² ‖φₙ‖² ≤ ‖f‖²`. -/
theorem bessel_inequality {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {ι : Type*} (ψ : ι → E)
    (hψ : Orthonormal ℝ ψ) (f : E) (s : Finset ι) : ∑ i ∈ s, inner ℝ (ψ i) f ^ 2 ≤ ‖f‖ ^ 2 := by
  have := hψ.sum_inner_products_le (x := f) (s := s)
  simpa [Real.norm_eq_abs, sq_abs] using this
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.OrthogonalFunctions.bessel_inequality
-- test: de-bessel-ineq-test.sage

end KaluzaKlein.DifferentialEquations.OrthogonalFunctions

namespace KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem

/-! ## thm-sturm-liouville (reality and orthogonality are proved; for the Dirichlet string the ordering, the
zeros and the simplicity are proved; completeness is quoted) -/

/-- Lagrange's identity: `d/dx [p(uv' - u'v)] = u (pv')' - v (pu')'`, with `P = pu'`, `Q = pv'`. -/
theorem lagrange_identity (p u v u' v' P' Q' : ℝ → ℝ) (x : ℝ) (hu : HasDerivAt u (u' x) x)
    (hv : HasDerivAt v (v' x) x) (hP : HasDerivAt (fun s => p s * u' s) (P' x) x)
    (hQ : HasDerivAt (fun s => p s * v' s) (Q' x) x) :
    HasDerivAt (fun s => u s * (p s * v' s) - v s * (p s * u' s)) (u x * Q' x - v x * P' x) x := by
  convert (hu.mul hQ).sub (hv.mul hP) using 1
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.lagrange_identity
-- test: de-lagrange-test.sage

/-- Orthogonality: for eigenfunctions (`-(pu')' + qu = λwu`, `-(pv')' + qv = μwv`) the integrand
`(λ - μ) u v w` is `u(pv')' - v(pu')'`, the derivative of `p(uv' - u'v)`. -/
theorem eigen_integrand (l m q w u v P' Q' : ℝ) (hu : -P' + q * u = l * w * u) (hv : -Q' + q * v = m * w * v) :
    (l - m) * u * v * w = u * Q' - v * P' := by
  veriphy_certified (linear_combination (-v) * hu + (u) * hv)
  -- certificate: de-poly.sage

/-- Hence `(λ - μ)∫ₐᵇ uvw = [p(uv' - u'v)]ₐᵇ`, which vanishes for Dirichlet conditions. -/
theorem eigen_orthogonal (a b l m : ℝ) (F g : ℝ → ℝ) (hF : ∀ x, HasDerivAt F (g x) x) (hg : Continuous g)
    (ha : F a = 0) (hb : F b = 0) : ∫ x in a..b, g x = 0 := by
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hF x) (hg.intervalIntegrable a b), ha, hb, sub_zero]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.eigen_orthogonal
-- test: de-eigen-orthogonal-test.sage

/-- For a string of length `L` the eigenvalues `(nπ/L)²` (`n²` for `L = π`) are increasing and unbounded. -/
theorem string_eigenvalues_increasing (L : ℝ) (hL : 0 < L) :
    StrictMono (fun n : ℕ => (((n : ℝ) + 1) * Real.pi / L) ^ 2) ∧
      Tendsto (fun n : ℕ => (((n : ℝ) + 1) * Real.pi / L) ^ 2) atTop atTop := by
  have hc : 0 < Real.pi / L := div_pos Real.pi_pos hL
  have e : (fun n : ℕ => (((n : ℝ) + 1) * Real.pi / L) ^ 2) = fun n : ℕ => (Real.pi / L) ^ 2 * ((n : ℝ) + 1) ^ 2 := by
    funext n; ring
  rw [e]
  constructor
  · intro a b h
    have : (a : ℝ) + 1 < b + 1 := by exact_mod_cast Nat.add_lt_add_right h 1
    exact mul_lt_mul_of_pos_left (pow_lt_pow_left₀ this (by positivity) two_ne_zero) (by positivity)
  · exact Tendsto.const_mul_atTop (by positivity) ((tendsto_pow_atTop two_ne_zero).comp
      (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop))
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.string_eigenvalues_increasing
-- test: de-string-increasing-test.sage

/-- The `n`-th eigenfunction `sin nx` has exactly the `n - 1` interior zeros `kπ/n`, `0 < k < n`. -/
theorem sine_zeros (n : ℕ) (hn : 0 < n) (x : ℝ) (hx : x ∈ Set.Ioo 0 Real.pi) :
    Real.sin (n * x) = 0 ↔ ∃ k : ℕ, 0 < k ∧ k < n ∧ x = k * Real.pi / n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  constructor
  · intro h
    obtain ⟨m, hm⟩ := Real.sin_eq_zero_iff.mp h
    have hx' : x = m * Real.pi / n := by field_simp; linarith
    have hm0 : 0 < m := by
      have : (0 : ℝ) < m * Real.pi := by rw [hm]; exact mul_pos hn' hx.1
      have := pos_of_mul_pos_left this Real.pi_pos.le
      exact_mod_cast this
    have hmn : m < n := by
      have : (m : ℝ) * Real.pi < n * Real.pi := by rw [hm]; exact mul_lt_mul_of_pos_left hx.2 hn'
      have := lt_of_mul_lt_mul_right this Real.pi_pos.le
      exact_mod_cast this
    refine ⟨m.toNat, by omega, by omega, ?_⟩
    rw [hx']
    congr 2
    exact_mod_cast (Int.toNat_of_nonneg hm0.le).symm
  · rintro ⟨k, -, -, rfl⟩
    rw [show (n : ℝ) * (k * Real.pi / n) = k * Real.pi by field_simp]
    exact Real.sin_nat_mul_pi k
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.sine_zeros
-- test: de-sine-zeros-test.sage

/-- Simplicity for the string: a solution of `u'' = -ω²u` (`ω ≠ 0`) with `u(0) = 0` is `(u'(0)/ω) sin ωx`, so each
eigenvalue has one eigenfunction up to a factor. -/
theorem string_simple (ω : ℝ) (hω : ω ≠ 0) (u u' : ℝ → ℝ) (hu : ∀ x, HasDerivAt u (u' x) x)
    (hu' : ∀ x, HasDerivAt u' (-ω ^ 2 * u x) x) (h0 : u 0 = 0) (x : ℝ) :
    u x = u' 0 / ω * Real.sin (ω * x) := by
  set w := fun s => u s - u' 0 / ω * Real.sin (ω * s)
  set w' := fun s => u' s - u' 0 * Real.cos (ω * s)
  have hw : ∀ s, HasDerivAt w (w' s) s := by
    intro s
    have hs : HasDerivAt (fun s => Real.sin (ω * s)) (Real.cos (ω * s) * ω) s := by
      simpa using ((hasDerivAt_id' s).const_mul ω).sin
    convert (hu s).sub (hs.const_mul (u' 0 / ω)) using 1
    simp only [w']; field_simp
  have hw' : ∀ s, HasDerivAt w' (-ω ^ 2 * w s) s := by
    intro s
    have hc : HasDerivAt (fun s => Real.cos (ω * s)) (-Real.sin (ω * s) * ω) s := by
      simpa using ((hasDerivAt_id' s).const_mul ω).cos
    convert (hu' s).sub (hc.const_mul (u' 0)) using 1
    simp only [w]; field_simp; ring
  have hE : ∀ s, HasDerivAt (fun s => ω ^ 2 * w s ^ 2 + w' s ^ 2) 0 s := by
    intro s
    convert (((hw s).pow 2).const_mul (ω ^ 2)).add ((hw' s).pow 2) using 1
    ring
  have hc := is_const_of_deriv_eq_zero (fun s => (hE s).differentiableAt) (fun s => (hE s).deriv) x 0
  have h0w : w 0 = 0 := by simp [w, h0]
  have h0w' : w' 0 = 0 := by simp [w']
  have hc' : ω ^ 2 * w x ^ 2 + w' x ^ 2 = ω ^ 2 * w 0 ^ 2 + w' 0 ^ 2 := hc
  rw [h0w, h0w'] at hc'
  have hω2 : 0 < ω ^ 2 := by positivity
  have h1 : ω ^ 2 * w x ^ 2 = 0 := by
    nlinarith [sq_nonneg (w' x), mul_nonneg hω2.le (sq_nonneg (w x))]
  have : w x = 0 := by
    rcases mul_eq_zero.mp h1 with h | h
    · exact absurd h hω2.ne'
    · exact (pow_eq_zero_iff two_ne_zero).mp h
  simp only [w] at this
  linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.string_simple
-- test: de-string-simple-test.sage

/-! ## exm-slt-sine-series -/

/-- For a finite sine sum `f = Σₖ bₖ sin kx` the coefficient formula `(2/π)∫₀^π f sin nx = bₙ` holds. -/
theorem sine_coefficient (N n : ℕ) (b : ℕ → ℝ) (hn : n ∈ Finset.Icc 1 N) :
    2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, (∑ k ∈ Finset.Icc 1 N, b k * Real.sin (k * x)) * Real.sin (n * x) = b n := by
  have hint : ∀ k, IntervalIntegrable (fun x => b k * Real.sin (k * x) * Real.sin (n * x)) MeasureTheory.volume 0
      Real.pi := fun k => by apply Continuous.intervalIntegrable; fun_prop
  simp_rw [Finset.sum_mul]
  rw [intervalIntegral.integral_finset_sum (fun k _ => hint k), Finset.sum_eq_single n]
  · simp_rw [mul_assoc]
    rw [intervalIntegral.integral_const_mul, show (fun x => Real.sin (n * x) * Real.sin (n * x)) =
      fun x => Real.sin (n * x) ^ 2 by funext x; ring, OrthogonalFunctions.sin_norm n (by
        simp at hn; omega)]
    field_simp
  · intro k _ hk
    simp_rw [mul_assoc]
    rw [intervalIntegral.integral_const_mul, OrthogonalFunctions.sin_orthogonal k n hk, mul_zero]
  · intro h; exact absurd hn h
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.sine_coefficient
-- test: de-sine-coefficient-test.sage

/-! ## exm-slt-legendre-series -/

/-- The Legendre coefficients of `x³`: `c₁ = (3/2)∫₋₁¹ x³·x = 3/5` … -/
theorem legendre_coeff_x3_1 : 3 / 2 * ∫ x in (-1 : ℝ)..1, x ^ 3 * x = 3 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) ((fun x => x ^ 3 * x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) (f' :=
          fun x => x ^ 3 * x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- … and `c₃ = (7/2)∫₋₁¹ x³ P₃ = 2/5`. -/
theorem legendre_coeff_x3_3 : 7 / 2 * ∫ x in (-1 : ℝ)..1, x ^ 3 * ((5 * x ^ 3 - 3 * x) / 2) = 2 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (5 * (x_veriphy ^ 7) / 14 - (3 * (x_veriphy ^ 5) / 10) : ℝ))
            ((fun x => x ^ 3 * ((5 * x ^ 3 - 3 * x) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 7).const_mul (5 / 14 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (-(3 / 10) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (5 * (x_veriphy ^ 7) / 14 - (3 * (x_veriphy ^ 5) / 10) : ℝ)) (f' := fun x =>
          x ^ 3 * ((5 * x ^ 3 - 3 * x) / 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `P₂` has exactly the two zeros `±1/√3` in `(-1, 1)`: `P₂(x) = 0 ↔ 3x² = 1`. -/
theorem P2_zeros (x : ℝ) : (3 * x ^ 2 - 1) / 2 = 0 ↔ 3 * x ^ 2 = 1 := by
  constructor <;> intro h <;> linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.P2_zeros
-- test: de-P2-zeros-test.sage

/-- `P₃ = x(5x² - 3)/2` has exactly the three zeros `0, ±√(3/5)`. -/
theorem P3_zeros (x : ℝ) : (5 * x ^ 3 - 3 * x) / 2 = 0 ↔ x = 0 ∨ 5 * x ^ 2 = 3 := by
  constructor
  · intro h
    have : x * (5 * x ^ 2 - 3) = 0 := by linarith
    rcases mul_eq_zero.mp this with h1 | h1
    · left; exact h1
    · right; linarith
  · rintro (h | h)
    · subst h; ring
    · have : 5 * x ^ 3 = 3 * x := by linear_combination x * h
      linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.P3_zeros
-- test: de-P3-zeros-test.sage

/-! ## exr-slt-real -/

/-- `(λ - λ̄)∫|u|²w = 0` with `∫|u|²w > 0` forces `λ = λ̄`. -/
theorem eigenvalue_real (l : ℂ) (W : ℝ) (hW : 0 < W) (h : (l - (starRingEnd ℂ) l) * W = 0) :
    (starRingEnd ℂ) l = l := by
  have hW' : (W : ℂ) ≠ 0 := by exact_mod_cast hW.ne'
  have := (mul_eq_zero.mp h).resolve_right hW'
  exact (sub_eq_zero.mp this).symm
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.eigenvalue_real
-- test: de-eigenvalue-real-test.sage

/-! ## exr-slt-fourier-sine -/

/-- `b₁ = (2/π)∫₀^π x(π - x) sin x = 8/π`. -/
theorem fs_b1 : 2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, x * (Real.pi - x) * Real.sin (1 * x) = 8 / Real.pi := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (x_veriphy * (Real.cos x_veriphy) - (Real.sin x_veriphy)) +
                    ((x_veriphy ^ 2 - 2) * (Real.cos x_veriphy)) -
                  (2 * x_veriphy * (Real.sin x_veriphy)) :
                ℝ))
            ((fun x => x * (π - x) * sin (1 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (-1 : ℝ))).const_mul
                            (-Real.pi : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).add (hasDerivAt_const (x_veriphy : ℝ) (-2 : ℝ))).mul
                          ((hasDerivAt_id' (x_veriphy : ℝ)).cos))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (-2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (x_veriphy * (Real.cos x_veriphy) - (Real.sin x_veriphy)) +
                ((x_veriphy ^ 2 - 2) * (Real.cos x_veriphy)) -
              (2 * x_veriphy * (Real.sin x_veriphy)) :
            ℝ))
          (f' := fun x => x * (π - x) * sin (1 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: de-integral.sage
/-- `b₂ = 0`. -/
theorem fs_b2 : 2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, x * (Real.pi - x) * Real.sin (2 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (2 * x_veriphy * (Real.cos (2 * x_veriphy)) - (Real.sin (2 * x_veriphy))) / 4 +
                    ((2 * (x_veriphy ^ 2) - 1) * (Real.cos (2 * x_veriphy)) / 4) -
                  (x_veriphy * (Real.sin (2 * x_veriphy)) / 2) :
                ℝ))
            ((fun x => x * (π - x) * sin (2 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos)).const_mul
                                    (2 : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                                  (-1 : ℝ))).const_mul
                            (-Real.pi / 4 : ℝ)).add
                        ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (2 : ℝ)).add
                                  (hasDerivAt_const (x_veriphy : ℝ) (-1 : ℝ))).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos)).const_mul
                          (1 / 4 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                          (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin)).const_mul
                      (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (2 * x_veriphy * (Real.cos (2 * x_veriphy)) - (Real.sin (2 * x_veriphy))) / 4 +
                ((2 * (x_veriphy ^ 2) - 1) * (Real.cos (2 * x_veriphy)) / 4) -
              (x_veriphy * (Real.sin (2 * x_veriphy)) / 2) :
            ℝ))
          (f' := fun x => x * (π - x) * sin (2 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((2 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((2 * (π) : ℝ)) = 1 := by
        rw [show ((2 * (π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((2 * (π) : ℝ)) = 0 := by
        rw [show ((2 * (π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: de-integral.sage
/-- `b₃ = 8/(27π)`. -/
theorem fs_b3 : 2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, x * (Real.pi - x) * Real.sin (3 * x) = 8 / (27 * Real.pi) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (3 * x_veriphy * (Real.cos (3 * x_veriphy)) - (Real.sin (3 * x_veriphy))) / 9 +
                    ((9 * (x_veriphy ^ 2) - 2) * (Real.cos (3 * x_veriphy)) / 27) -
                  (2 * x_veriphy * (Real.sin (3 * x_veriphy)) / 9) :
                ℝ))
            ((fun x => x * (π - x) * sin (3 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).cos)).const_mul
                                    (3 : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).sin).const_mul
                                  (-1 : ℝ))).const_mul
                            (-Real.pi / 9 : ℝ)).add
                        ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (9 : ℝ)).add
                                  (hasDerivAt_const (x_veriphy : ℝ) (-2 : ℝ))).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).cos)).const_mul
                          (1 / 27 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                          (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).sin)).const_mul
                      (-(2 / 9) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (3 * x_veriphy * (Real.cos (3 * x_veriphy)) - (Real.sin (3 * x_veriphy))) / 9 +
                ((9 * (x_veriphy ^ 2) - 2) * (Real.cos (3 * x_veriphy)) / 27) -
              (2 * x_veriphy * (Real.sin (3 * x_veriphy)) / 9) :
            ℝ))
          (f' := fun x => x * (π - x) * sin (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((3 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((3 * (π) : ℝ)) = -1 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin ((3 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((3 * (π) : ℝ)) = 0 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 3
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: de-integral.sage
/-- `b₄ = 0`. -/
theorem fs_b4 : 2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, x * (Real.pi - x) * Real.sin (4 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (4 * x_veriphy * (Real.cos (4 * x_veriphy)) - (Real.sin (4 * x_veriphy))) / 16 +
                    ((8 * (x_veriphy ^ 2) - 1) * (Real.cos (4 * x_veriphy)) / 32) -
                  (x_veriphy * (Real.sin (4 * x_veriphy)) / 8) :
                ℝ))
            ((fun x => x * (π - x) * sin (4 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).cos)).const_mul
                                    (4 : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).sin).const_mul
                                  (-1 : ℝ))).const_mul
                            (-Real.pi / 16 : ℝ)).add
                        ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (8 : ℝ)).add
                                  (hasDerivAt_const (x_veriphy : ℝ) (-1 : ℝ))).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).cos)).const_mul
                          (1 / 32 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                          (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).sin)).const_mul
                      (-(1 / 8) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (4 * x_veriphy * (Real.cos (4 * x_veriphy)) - (Real.sin (4 * x_veriphy))) / 16 +
                ((8 * (x_veriphy ^ 2) - 1) * (Real.cos (4 * x_veriphy)) / 32) -
              (x_veriphy * (Real.sin (4 * x_veriphy)) / 8) :
            ℝ))
          (f' := fun x => x * (π - x) * sin (4 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((4 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((4 * (π) : ℝ)) = 1 := by
        rw [show ((4 * (π) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin ((4 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((4 * (π) : ℝ)) = 0 := by
        rw [show ((4 * (π) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 4
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: de-integral.sage
/-- `b₅ = 8/(125π)`. -/
theorem fs_b5 : 2 / Real.pi * ∫ x in (0 : ℝ)..Real.pi, x * (Real.pi - x) * Real.sin (5 * x) = 8 / (125 * Real.pi) := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (5 * x_veriphy * (Real.cos (5 * x_veriphy)) - (Real.sin (5 * x_veriphy))) / 25 +
                    ((25 * (x_veriphy ^ 2) - 2) * (Real.cos (5 * x_veriphy)) / 125) -
                  (2 * x_veriphy * (Real.sin (5 * x_veriphy)) / 25) :
                ℝ))
            ((fun x => x * (π - x) * sin (5 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 : ℝ)).cos)).const_mul
                                    (5 : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 : ℝ)).sin).const_mul
                                  (-1 : ℝ))).const_mul
                            (-Real.pi / 25 : ℝ)).add
                        ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (25 : ℝ)).add
                                  (hasDerivAt_const (x_veriphy : ℝ) (-2 : ℝ))).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 : ℝ)).cos)).const_mul
                          (1 / 125 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                          (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 : ℝ)).sin)).const_mul
                      (-(2 / 25) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (5 * x_veriphy * (Real.cos (5 * x_veriphy)) - (Real.sin (5 * x_veriphy))) / 25 +
                ((25 * (x_veriphy ^ 2) - 2) * (Real.cos (5 * x_veriphy)) / 125) -
              (2 * x_veriphy * (Real.sin (5 * x_veriphy)) / 25) :
            ℝ))
          (f' := fun x => x * (π - x) * sin (5 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((5 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((5 * (π) : ℝ)) = -1 := by
        rw [show ((5 * (π) : ℝ) : ℝ) = ((5 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin ((5 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((5 * (π) : ℝ)) = 0 := by
        rw [show ((5 * (π) : ℝ) : ℝ) = ((5 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 5
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: de-integral.sage
/-- `‖f‖² = ∫₀^π x²(π - x)² = π⁵/30`. -/
theorem fs_norm : ∫ x in (0 : ℝ)..Real.pi, (x * (Real.pi - x)) ^ 2 = Real.pi ^ 5 / 30 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              ((Real.pi ^ 2) * (x_veriphy ^ 3) / 3 - (Real.pi * (x_veriphy ^ 4) / 2) + ((x_veriphy ^ 5) / 5) : ℝ))
            ((fun x => (x * (π - x)) ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul ((Real.pi ^ 2) / 3 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (-Real.pi / 2 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((Real.pi ^ 2) * (x_veriphy ^ 3) / 3 - (Real.pi * (x_veriphy ^ 4) / 2) + ((x_veriphy ^ 5) / 5) : ℝ)) (f' :=
          fun x => (x * (π - x)) ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: de-integral.sage
/-- After the first term the squared error `‖f‖² - (π/2)b₁² = π⁵/30 - 32/π` is below `0.2%` of `‖f‖²`. -/
theorem fs_error_first : Real.pi ^ 5 / 30 - 32 / Real.pi < 1 / 500 * (Real.pi ^ 5 / 30) := by
  veriphy_certified
    ( have h2 := Real.pi_lt_d6
      have hp := Real.pi_pos
      have h6 : Real.pi ^ 6 < (3.141593 : ℝ) ^ 6 := pow_lt_pow_left₀ h2 hp.le (by norm_num)
      have e :
        Real.pi ^ 5 / 30 - 32 / Real.pi - 1 / 500 * (Real.pi ^ 5 / 30) =
          (499 * Real.pi ^ 6 - 480000) / (15000 * Real.pi) :=
        by
        field_simp
        ring
      have hn : (499 * Real.pi ^ 6 - 480000) / (15000 * Real.pi) < 0 :=
        div_neg_of_neg_of_pos (by norm_num at h6; linarith) (by positivity)
      linarith)
  -- certificate: de-rewrite.sage

end KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem

namespace KaluzaKlein.DifferentialEquations.LegendrePolynomials

/-! The Legendre polynomials `P₀, …, P₆` are written out as polynomials in `x`. -/

/-! ## def-legendre-polynomial (with exr-leg-rodrigues): `Pₙ(1) = 1` -/

/-- `P2(1) = 1`. -/
theorem P2_at_one : ((3 * (1 : ℝ) ^ 2 - 1) / 2) = 1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P3(1) = 1`. -/
theorem P3_at_one : ((5 * (1 : ℝ) ^ 3 - 3 * (1 : ℝ)) / 2) = 1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P4(1) = 1`. -/
theorem P4_at_one : ((35 * (1 : ℝ) ^ 4 - 30 * (1 : ℝ) ^ 2 + 3) / 8) = 1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P5(1) = 1`. -/
theorem P5_at_one : ((63 * (1 : ℝ) ^ 5 - 70 * (1 : ℝ) ^ 3 + 15 * (1 : ℝ)) / 8) = 1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P6(1) = 1`. -/
theorem P6_at_one : ((231 * (1 : ℝ) ^ 6 - 315 * (1 : ℝ) ^ 4 + 105 * (1 : ℝ) ^ 2 - 5) / 16) = 1 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-leg-p2 -/

/-- `P₂` solves Legendre's equation with `n = 2`: `(1 - x²)P₂'' - 2xP₂' + 6P₂ = 0`. -/
theorem P2_equation (u : ℝ → ℝ) (hu : u = fun s => (3 * s ^ 2 - 1) / 2) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 6 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun (s : ℝ) => ((3 * s ^ 2 - 1) / 2 : ℝ)) = fun (s : ℝ) => (3 * s : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => ((3 * s ^ 2 - 1) / 2 : ℝ)) ((3 * s : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (3 / 2 : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (3 * s : ℝ)) = fun (s : ℝ) => (3 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (3 * s : ℝ)) ((3 : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).const_mul (3 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-! ## exr-leg-rodrigues (and exm-leg-p2 for `n = 2`): Rodrigues' formula -/

/-- Rodrigues' formula for `P1`: `1/(2^1·1!) dⁿ/dxⁿ (x² - 1)^1 = P1`. -/
theorem rodrigues_P1 (x : ℝ) : deriv (fun r => (r ^ 2 - 1) ^ 1) x / 2 = x := by
  veriphy_certified
    ( have h0 : HasDerivAt (fun r => (r ^ 2 - 1) ^ 1) ((2 * x : ℝ)) (x) :=
        by
        convert (((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))) using 1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done)
          | (ring1)
      try rw [h0.deriv]
      beta_reduce
      ring1)
  -- certificate: de-partials.sage

/-- Rodrigues' formula for `P2`: `1/(2^2·2!) dⁿ/dxⁿ (x² - 1)^2 = P2`. -/
theorem rodrigues_P2 (x : ℝ) : deriv (fun a => deriv (fun r => (r ^ 2 - 1) ^ 2) a) x / 8 = (3 * x ^ 2 - 1) / 2 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => ((r ^ 2 - 1) ^ 2 : ℝ)) = fun (d1 : ℝ) => (4 * (d1 ^ 2 - 1) * d1 : ℝ) :=
        by
        funext d1
        have hd : HasDerivAt (fun (r : ℝ) => ((r ^ 2 - 1) ^ 2 : ℝ)) ((4 * (d1 ^ 2 - 1) * d1 : ℝ)) (d1) :=
          by
          refine
            ((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow
                      2)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · (simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have h1 : HasDerivAt (fun a => (fun (d1 : ℝ) => (4 * (d1 ^ 2 - 1) * d1 : ℝ)) a) ((12 * (x ^ 2) - 4 : ℝ)) (x) :=
        by
        refine
          (((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul
                        (hasDerivAt_id' (x : ℝ))).const_mul
                    (4 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (simp; ring_nf; done)
      try rw [h1.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- Rodrigues' formula for `P3`: `1/(2^3·3!) dⁿ/dxⁿ (x² - 1)^3 = P3`. -/
theorem rodrigues_P3 (x : ℝ) : deriv (fun a => deriv (fun b => deriv (fun r => (r ^ 2 - 1) ^ 3) b) a) x / 48 = (5 * x ^ 3 - 3 * x) / 2 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => ((r ^ 2 - 1) ^ 3 : ℝ)) = fun (d2 : ℝ) => (6 * ((d2 ^ 2 - 1) ^ 2) * d2 : ℝ) :=
        by
        funext d2
        have hd : HasDerivAt (fun (r : ℝ) => ((r ^ 2 - 1) ^ 3 : ℝ)) ((6 * ((d2 ^ 2 - 1) ^ 2) * d2 : ℝ)) (d2) :=
          by
          refine
            ((((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow
                      3)).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · (simp; ring_nf; done)
        exact hd.deriv
      rw [i0]
      have i1 :
        deriv (fun (b : ℝ) => ((fun (d2 : ℝ) => (6 * ((d2 ^ 2 - 1) ^ 2) * d2 : ℝ)) b : ℝ)) = fun (d1 : ℝ) =>
          (24 * (d1 ^ 2 - 1) * (d1 ^ 2) + (6 * ((d1 ^ 2 - 1) ^ 2)) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt (fun (b : ℝ) => ((fun (d2 : ℝ) => (6 * ((d2 ^ 2 - 1) ^ 2) * d2 : ℝ)) b : ℝ))
            ((24 * (d1 ^ 2 - 1) * (d1 ^ 2) + (6 * ((d1 ^ 2 - 1) ^ 2)) : ℝ)) (d1) :=
          by
          refine
            ((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 2).mul
                          (hasDerivAt_id' (d1 : ℝ))).const_mul
                      (6 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · (simp; ring_nf; done)
        exact hd.deriv
      rw [i1]
      have h2 :
        HasDerivAt (fun a => (fun (d1 : ℝ) => (24 * (d1 ^ 2 - 1) * (d1 ^ 2) + (6 * ((d1 ^ 2 - 1) ^ 2)) : ℝ)) a)
          ((48 * (x ^ 3) + (72 * (x ^ 2 - 1) * x) : ℝ)) (x) :=
        by
        refine
          ((((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul
                            ((hasDerivAt_id' (x : ℝ)).pow 2)).const_mul
                        (24 : ℝ)).add
                    (((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 2).const_mul
                      (6 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · (simp; ring_nf; done)
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- Rodrigues' formula for `P4`: `1/(2^4·4!) dⁿ/dxⁿ (x² - 1)^4 = P4`. -/
theorem rodrigues_P4 (x : ℝ) : deriv (fun a => deriv (fun b => deriv (fun c => deriv (fun r => (r ^ 2 - 1) ^ 4) c) b) a) x / 384 = (35 * x ^ 4 - 30 * x ^ 2 + 3) / 8 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => ((r ^ 2 - 1) ^ 4 : ℝ)) = fun (d3 : ℝ) => (8 * ((d3 ^ 2 - 1) ^ 3) * d3 : ℝ) :=
        by
        funext d3
        have hd : HasDerivAt (fun (r : ℝ) => ((r ^ 2 - 1) ^ 4 : ℝ)) ((8 * ((d3 ^ 2 - 1) ^ 3) * d3 : ℝ)) (d3) :=
          by
          refine
            ((((((hasDerivAt_id' (d3 : ℝ)).pow 2).add (hasDerivAt_const (d3 : ℝ) (-1 : ℝ))).pow
                      4)).congr_of_eventuallyEq
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
      have i1 :
        deriv (fun (c : ℝ) => ((fun (d3 : ℝ) => (8 * ((d3 ^ 2 - 1) ^ 3) * d3 : ℝ)) c : ℝ)) = fun (d2 : ℝ) =>
          (48 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 2) + (8 * ((d2 ^ 2 - 1) ^ 3)) : ℝ) :=
        by
        funext d2
        have hd :
          HasDerivAt (fun (c : ℝ) => ((fun (d3 : ℝ) => (8 * ((d3 ^ 2 - 1) ^ 3) * d3 : ℝ)) c : ℝ))
            ((48 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 2) + (8 * ((d2 ^ 2 - 1) ^ 3)) : ℝ)) (d2) :=
          by
          refine
            ((((((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow 3).mul
                          (hasDerivAt_id' (d2 : ℝ))).const_mul
                      (8 : ℝ))).congr_of_eventuallyEq
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
      rw [i1]
      have i2 :
        deriv
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) => (48 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 2) + (8 * ((d2 ^ 2 - 1) ^ 3)) : ℝ)) b : ℝ)) =
          fun (d1 : ℝ) => (192 * (d1 ^ 2 - 1) * (d1 ^ 3) + (144 * ((d1 ^ 2 - 1) ^ 2) * d1) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) => (48 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 2) + (8 * ((d2 ^ 2 - 1) ^ 3)) : ℝ)) b : ℝ))
            ((192 * (d1 ^ 2 - 1) * (d1 ^ 3) + (144 * ((d1 ^ 2 - 1) ^ 2) * d1) : ℝ)) (d1) :=
          by
          refine
            (((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 2).mul
                              ((hasDerivAt_id' (d1 : ℝ)).pow 2)).const_mul
                          (48 : ℝ)).add
                      (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 3).const_mul
                        (8 : ℝ)))).congr_of_eventuallyEq
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
        HasDerivAt (fun a => (fun (d1 : ℝ) => (192 * (d1 ^ 2 - 1) * (d1 ^ 3) + (144 * ((d1 ^ 2 - 1) ^ 2) * d1) : ℝ)) a)
          ((384 * (x ^ 4) + (1152 * (x ^ 2 - 1) * (x ^ 2)) + (144 * ((x ^ 2 - 1) ^ 2)) : ℝ)) (x) :=
        by
        refine
          ((((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul
                            ((hasDerivAt_id' (x : ℝ)).pow 3)).const_mul
                        (192 : ℝ)).add
                    ((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 2).mul
                          (hasDerivAt_id' (x : ℝ))).const_mul
                      (144 : ℝ)))).congr_of_eventuallyEq
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

/-- Rodrigues' formula for `P5`: `1/(2^5·5!) dⁿ/dxⁿ (x² - 1)^5 = P5`. -/
theorem rodrigues_P5 (x : ℝ) : deriv (fun a => deriv (fun b => deriv (fun c => deriv (fun d => deriv (fun r => (r ^ 2 - 1) ^ 5) d) c) b) a) x / 3840 = (63 * x ^ 5 - 70 * x ^ 3 + 15 * x) / 8 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => ((r ^ 2 - 1) ^ 5 : ℝ)) = fun (d4 : ℝ) => (10 * ((d4 ^ 2 - 1) ^ 4) * d4 : ℝ) :=
        by
        funext d4
        have hd : HasDerivAt (fun (r : ℝ) => ((r ^ 2 - 1) ^ 5 : ℝ)) ((10 * ((d4 ^ 2 - 1) ^ 4) * d4 : ℝ)) (d4) :=
          by
          refine
            ((((((hasDerivAt_id' (d4 : ℝ)).pow 2).add (hasDerivAt_const (d4 : ℝ) (-1 : ℝ))).pow
                      5)).congr_of_eventuallyEq
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
      have i1 :
        deriv (fun (d : ℝ) => ((fun (d4 : ℝ) => (10 * ((d4 ^ 2 - 1) ^ 4) * d4 : ℝ)) d : ℝ)) = fun (d3 : ℝ) =>
          (80 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 2) + (10 * ((d3 ^ 2 - 1) ^ 4)) : ℝ) :=
        by
        funext d3
        have hd :
          HasDerivAt (fun (d : ℝ) => ((fun (d4 : ℝ) => (10 * ((d4 ^ 2 - 1) ^ 4) * d4 : ℝ)) d : ℝ))
            ((80 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 2) + (10 * ((d3 ^ 2 - 1) ^ 4)) : ℝ)) (d3) :=
          by
          refine
            ((((((((hasDerivAt_id' (d3 : ℝ)).pow 2).add (hasDerivAt_const (d3 : ℝ) (-1 : ℝ))).pow 4).mul
                          (hasDerivAt_id' (d3 : ℝ))).const_mul
                      (10 : ℝ))).congr_of_eventuallyEq
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
      rw [i1]
      have i2 :
        deriv
            (fun (c : ℝ) =>
              ((fun (d3 : ℝ) => (80 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 2) + (10 * ((d3 ^ 2 - 1) ^ 4)) : ℝ)) c : ℝ)) =
          fun (d2 : ℝ) => (480 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 3) + (240 * ((d2 ^ 2 - 1) ^ 3) * d2) : ℝ) :=
        by
        funext d2
        have hd :
          HasDerivAt
            (fun (c : ℝ) =>
              ((fun (d3 : ℝ) => (80 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 2) + (10 * ((d3 ^ 2 - 1) ^ 4)) : ℝ)) c : ℝ))
            ((480 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 3) + (240 * ((d2 ^ 2 - 1) ^ 3) * d2) : ℝ)) (d2) :=
          by
          refine
            (((((((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow 3).mul
                              ((hasDerivAt_id' (d2 : ℝ)).pow 2)).const_mul
                          (80 : ℝ)).add
                      (((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow 4).const_mul
                        (10 : ℝ)))).congr_of_eventuallyEq
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
      have i3 :
        deriv
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) => (480 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 3) + (240 * ((d2 ^ 2 - 1) ^ 3) * d2) : ℝ)) b : ℝ)) =
          fun (d1 : ℝ) =>
          (1920 * (d1 ^ 2 - 1) * (d1 ^ 4) + (2880 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 2)) + (240 * ((d1 ^ 2 - 1) ^ 3)) : ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) => (480 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 3) + (240 * ((d2 ^ 2 - 1) ^ 3) * d2) : ℝ)) b : ℝ))
            ((1920 * (d1 ^ 2 - 1) * (d1 ^ 4) + (2880 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 2)) + (240 * ((d1 ^ 2 - 1) ^ 3)) : ℝ))
            (d1) :=
          by
          refine
            (((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 2).mul
                              ((hasDerivAt_id' (d1 : ℝ)).pow 3)).const_mul
                          (480 : ℝ)).add
                      ((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 3).mul
                            (hasDerivAt_id' (d1 : ℝ))).const_mul
                        (240 : ℝ)))).congr_of_eventuallyEq
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
      rw [i3]
      have h4 :
        HasDerivAt
          (fun a =>
            (fun (d1 : ℝ) =>
                (1920 * (d1 ^ 2 - 1) * (d1 ^ 4) + (2880 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 2)) + (240 * ((d1 ^ 2 - 1) ^ 3)) :
                  ℝ))
              a)
          ((3840 * (x ^ 5) + (19200 * (x ^ 2 - 1) * (x ^ 3)) + (7200 * ((x ^ 2 - 1) ^ 2) * x) : ℝ)) (x) :=
        by
        refine
          (((((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul
                                ((hasDerivAt_id' (x : ℝ)).pow 4)).const_mul
                            (1920 : ℝ)).add
                        ((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 2).mul
                              ((hasDerivAt_id' (x : ℝ)).pow 2)).const_mul
                          (2880 : ℝ))).add
                    (((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 3).const_mul
                      (240 : ℝ)))).congr_of_eventuallyEq
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
      try rw [h4.deriv]
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

/-- Rodrigues' formula for `P6`: `1/(2^6·6!) dⁿ/dxⁿ (x² - 1)^6 = P6`. -/
theorem rodrigues_P6 (x : ℝ) : deriv (fun a => deriv (fun b => deriv (fun c => deriv (fun d => deriv (fun e => deriv (fun r => (r ^ 2 - 1) ^ 6) e) d) c) b) a) x / 46080 = (231 * x ^ 6 - 315 * x ^ 4 + 105 * x ^ 2 - 5) / 16 := by
  veriphy_certified
    ( have i0 : deriv (fun (r : ℝ) => ((r ^ 2 - 1) ^ 6 : ℝ)) = fun (d5 : ℝ) => (12 * ((d5 ^ 2 - 1) ^ 5) * d5 : ℝ) :=
        by
        funext d5
        have hd : HasDerivAt (fun (r : ℝ) => ((r ^ 2 - 1) ^ 6 : ℝ)) ((12 * ((d5 ^ 2 - 1) ^ 5) * d5 : ℝ)) (d5) :=
          by
          refine
            ((((((hasDerivAt_id' (d5 : ℝ)).pow 2).add (hasDerivAt_const (d5 : ℝ) (-1 : ℝ))).pow
                      6)).congr_of_eventuallyEq
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
      have i1 :
        deriv (fun (e : ℝ) => ((fun (d5 : ℝ) => (12 * ((d5 ^ 2 - 1) ^ 5) * d5 : ℝ)) e : ℝ)) = fun (d4 : ℝ) =>
          (120 * ((d4 ^ 2 - 1) ^ 4) * (d4 ^ 2) + (12 * ((d4 ^ 2 - 1) ^ 5)) : ℝ) :=
        by
        funext d4
        have hd :
          HasDerivAt (fun (e : ℝ) => ((fun (d5 : ℝ) => (12 * ((d5 ^ 2 - 1) ^ 5) * d5 : ℝ)) e : ℝ))
            ((120 * ((d4 ^ 2 - 1) ^ 4) * (d4 ^ 2) + (12 * ((d4 ^ 2 - 1) ^ 5)) : ℝ)) (d4) :=
          by
          refine
            ((((((((hasDerivAt_id' (d4 : ℝ)).pow 2).add (hasDerivAt_const (d4 : ℝ) (-1 : ℝ))).pow 5).mul
                          (hasDerivAt_id' (d4 : ℝ))).const_mul
                      (12 : ℝ))).congr_of_eventuallyEq
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
      rw [i1]
      have i2 :
        deriv
            (fun (d : ℝ) =>
              ((fun (d4 : ℝ) => (120 * ((d4 ^ 2 - 1) ^ 4) * (d4 ^ 2) + (12 * ((d4 ^ 2 - 1) ^ 5)) : ℝ)) d : ℝ)) =
          fun (d3 : ℝ) => (960 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 3) + (360 * ((d3 ^ 2 - 1) ^ 4) * d3) : ℝ) :=
        by
        funext d3
        have hd :
          HasDerivAt
            (fun (d : ℝ) =>
              ((fun (d4 : ℝ) => (120 * ((d4 ^ 2 - 1) ^ 4) * (d4 ^ 2) + (12 * ((d4 ^ 2 - 1) ^ 5)) : ℝ)) d : ℝ))
            ((960 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 3) + (360 * ((d3 ^ 2 - 1) ^ 4) * d3) : ℝ)) (d3) :=
          by
          refine
            (((((((((hasDerivAt_id' (d3 : ℝ)).pow 2).add (hasDerivAt_const (d3 : ℝ) (-1 : ℝ))).pow 4).mul
                              ((hasDerivAt_id' (d3 : ℝ)).pow 2)).const_mul
                          (120 : ℝ)).add
                      (((((hasDerivAt_id' (d3 : ℝ)).pow 2).add (hasDerivAt_const (d3 : ℝ) (-1 : ℝ))).pow 5).const_mul
                        (12 : ℝ)))).congr_of_eventuallyEq
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
      have i3 :
        deriv
            (fun (c : ℝ) =>
              ((fun (d3 : ℝ) => (960 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 3) + (360 * ((d3 ^ 2 - 1) ^ 4) * d3) : ℝ)) c : ℝ)) =
          fun (d2 : ℝ) =>
          (5760 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 4) + (5760 * ((d2 ^ 2 - 1) ^ 3) * (d2 ^ 2)) + (360 * ((d2 ^ 2 - 1) ^ 4)) :
            ℝ) :=
        by
        funext d2
        have hd :
          HasDerivAt
            (fun (c : ℝ) =>
              ((fun (d3 : ℝ) => (960 * ((d3 ^ 2 - 1) ^ 3) * (d3 ^ 3) + (360 * ((d3 ^ 2 - 1) ^ 4) * d3) : ℝ)) c : ℝ))
            ((5760 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 4) + (5760 * ((d2 ^ 2 - 1) ^ 3) * (d2 ^ 2)) +
                (360 * ((d2 ^ 2 - 1) ^ 4)) :
              ℝ))
            (d2) :=
          by
          refine
            (((((((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow 3).mul
                              ((hasDerivAt_id' (d2 : ℝ)).pow 3)).const_mul
                          (960 : ℝ)).add
                      ((((((hasDerivAt_id' (d2 : ℝ)).pow 2).add (hasDerivAt_const (d2 : ℝ) (-1 : ℝ))).pow 4).mul
                            (hasDerivAt_id' (d2 : ℝ))).const_mul
                        (360 : ℝ)))).congr_of_eventuallyEq
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
      rw [i3]
      have i4 :
        deriv
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) =>
                    (5760 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 4) + (5760 * ((d2 ^ 2 - 1) ^ 3) * (d2 ^ 2)) +
                        (360 * ((d2 ^ 2 - 1) ^ 4)) :
                      ℝ))
                  b :
                ℝ)) =
          fun (d1 : ℝ) =>
          (23040 * (d1 ^ 2 - 1) * (d1 ^ 5) + (57600 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 3)) +
              (14400 * ((d1 ^ 2 - 1) ^ 3) * d1) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (b : ℝ) =>
              ((fun (d2 : ℝ) =>
                    (5760 * ((d2 ^ 2 - 1) ^ 2) * (d2 ^ 4) + (5760 * ((d2 ^ 2 - 1) ^ 3) * (d2 ^ 2)) +
                        (360 * ((d2 ^ 2 - 1) ^ 4)) :
                      ℝ))
                  b :
                ℝ))
            ((23040 * (d1 ^ 2 - 1) * (d1 ^ 5) + (57600 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 3)) +
                (14400 * ((d1 ^ 2 - 1) ^ 3) * d1) :
              ℝ))
            (d1) :=
          by
          refine
            ((((((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 2).mul
                                  ((hasDerivAt_id' (d1 : ℝ)).pow 4)).const_mul
                              (5760 : ℝ)).add
                          ((((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 3).mul
                                ((hasDerivAt_id' (d1 : ℝ)).pow 2)).const_mul
                            (5760 : ℝ))).add
                      (((((hasDerivAt_id' (d1 : ℝ)).pow 2).add (hasDerivAt_const (d1 : ℝ) (-1 : ℝ))).pow 4).const_mul
                        (360 : ℝ)))).congr_of_eventuallyEq
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
      rw [i4]
      have h5 :
        HasDerivAt
          (fun a =>
            (fun (d1 : ℝ) =>
                (23040 * (d1 ^ 2 - 1) * (d1 ^ 5) + (57600 * ((d1 ^ 2 - 1) ^ 2) * (d1 ^ 3)) +
                    (14400 * ((d1 ^ 2 - 1) ^ 3) * d1) :
                  ℝ))
              a)
          ((46080 * (x ^ 6) + (345600 * (x ^ 2 - 1) * (x ^ 4)) + (259200 * ((x ^ 2 - 1) ^ 2) * (x ^ 2)) +
              (14400 * ((x ^ 2 - 1) ^ 3)) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).mul
                                ((hasDerivAt_id' (x : ℝ)).pow 5)).const_mul
                            (23040 : ℝ)).add
                        ((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 2).mul
                              ((hasDerivAt_id' (x : ℝ)).pow 3)).const_mul
                          (57600 : ℝ))).add
                    ((((((hasDerivAt_id' (x : ℝ)).pow 2).add (hasDerivAt_const (x : ℝ) (-1 : ℝ))).pow 3).mul
                          (hasDerivAt_id' (x : ℝ))).const_mul
                      (14400 : ℝ)))).congr_of_eventuallyEq
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
      try rw [h5.deriv]
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

/-! ## exr-leg-rodrigues: parity `Pₙ(-x) = (-1)ⁿ Pₙ(x)` -/

/-- `P2(-x) = P2(x)`. -/
theorem P2_parity (x : ℝ) : ((3 * (-x) ^ 2 - 1) / 2) = ((3 * x ^ 2 - 1) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P3(-x) = -P3(x)`. -/
theorem P3_parity (x : ℝ) : ((5 * (-x) ^ 3 - 3 * (-x)) / 2) = -((5 * x ^ 3 - 3 * x) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P4(-x) = P4(x)`. -/
theorem P4_parity (x : ℝ) : ((35 * (-x) ^ 4 - 30 * (-x) ^ 2 + 3) / 8) = ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P5(-x) = -P5(x)`. -/
theorem P5_parity (x : ℝ) : ((63 * (-x) ^ 5 - 70 * (-x) ^ 3 + 15 * (-x)) / 8) = -((63 * x ^ 5 - 70 * x ^ 3 + 15 * x) / 8) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `P6(-x) = P6(x)`. -/
theorem P6_parity (x : ℝ) : ((231 * (-x) ^ 6 - 315 * (-x) ^ 4 + 105 * (-x) ^ 2 - 5) / 16) = ((231 * x ^ 6 - 315 * x ^ 4 + 105 * x ^ 2 - 5) / 16) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-! ## exm-leg-generating -/

/-- The generating function to order `t²`: `s = 1 + xt + P₂t²` satisfies `(1 - 2xt + t²)s² = 1 + O(t³)`, so
`s = (1 - 2xt + t²)^{-1/2} + O(t³)`. -/
theorem generating_order_two (x t : ℝ) :
    (1 - 2 * x * t + t ^ 2) * (1 + x * t + (3 * x ^ 2 - 1) / 2 * t ^ 2) ^ 2 =
      1 + t ^ 3 * (9 / 4 * t ^ 3 * x ^ 4 - 9 / 2 * t ^ 2 * x ^ 5 - 3 / 2 * t ^ 3 * x ^ 2 + 6 * t ^ 2 * x ^ 3 -
        15 / 4 * t * x ^ 4 + 1 / 4 * t ^ 3 - 3 / 2 * t ^ 2 * x + 9 / 2 * t * x ^ 2 - 5 * x ^ 3 - 3 / 4 * t + 3 * x) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The multipole expansion: `|r - r'|² = r²(1 - 2xt + t²)` with `x = cos θ`, `t = r'/r` (law of cosines). -/
theorem multipole_distance (r r' c : ℝ) (hr : r ≠ 0) :
    r ^ 2 - 2 * r * r' * c + r' ^ 2 = r ^ 2 * (1 - 2 * c * (r' / r) + (r' / r) ^ 2) := by
  veriphy_certified ((field_simp; done))
  -- certificate: de-rewrite.sage

/-! ## exm-leg-recurrence and exr-leg-recurrence: `(n + 1)Pₙ₊₁ = (2n + 1)xPₙ - nPₙ₋₁` for `n ≤ 5` -/

/-- The recurrence for `n = 1`. -/
theorem recurrence_1 (x : ℝ) : 2 * ((3 * x ^ 2 - 1) / 2) = 3 * x * (x) - 1 * (1) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The recurrence for `n = 2`. -/
theorem recurrence_2 (x : ℝ) : 3 * ((5 * x ^ 3 - 3 * x) / 2) = 5 * x * ((3 * x ^ 2 - 1) / 2) - 2 * (x) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The recurrence for `n = 3`. -/
theorem recurrence_3 (x : ℝ) : 4 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 7 * x * ((5 * x ^ 3 - 3 * x) / 2) - 3 * ((3 * x ^ 2 - 1) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The recurrence for `n = 4`. -/
theorem recurrence_4 (x : ℝ) : 5 * ((63 * x ^ 5 - 70 * x ^ 3 + 15 * x) / 8) = 9 * x * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) - 4 * ((5 * x ^ 3 - 3 * x) / 2) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The recurrence for `n = 5`. -/
theorem recurrence_5 (x : ℝ) : 6 * ((231 * x ^ 6 - 315 * x ^ 4 + 105 * x ^ 2 - 5) / 16) = 11 * x * ((63 * x ^ 5 - 70 * x ^ 3 + 15 * x) / 8) - 5 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The generating function's equation: `(1 - 2xt + t²) ∂ₜg = (x - t) g` for `g = (1 - 2xt + t²)^{-1/2}`. -/
theorem generating_equation (x t : ℝ) (h : 0 < 1 - 2 * x * t + t ^ 2) :
    (1 - 2 * x * t + t ^ 2) * deriv (fun s => (1 - 2 * x * s + s ^ 2) ^ (-(1 / 2 : ℝ))) t =
      (x - t) * (1 - 2 * x * t + t ^ 2) ^ (-(1 / 2 : ℝ)) := by
  have hp : HasDerivAt (fun s : ℝ => 1 - 2 * x * s + s ^ 2) (-(2 * x) + 2 * t) t := by
    have := (((hasDerivAt_id' t).const_mul (2 * x)).const_sub 1).add (hasDerivAt_pow 2 t)
    convert this using 1; norm_num
  have hd := hp.rpow_const (p := -(1 / 2 : ℝ)) (Or.inl h.ne')
  rw [hd.deriv, Real.rpow_sub h, Real.rpow_one]
  have hq := h.ne'
  first | (field_simp; done) | (field_simp; ring) | (norm_num; done) | (norm_num; field_simp; done) | (norm_num; field_simp; ring) | (norm_num; ring)
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.LegendrePolynomials.generating_equation
-- test: de-generating-equation-test.sage

/-! ## exr-leg-expansion -/

/-- `x³ = (2/5)P₃ + (3/5)P₁`. -/
theorem x3_expansion (x : ℝ) : x ^ 3 = 2 / 5 * ((5 * x ^ 3 - 3 * x) / 2) + 3 / 5 * x := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- `x⁴ = (8/35)P₄ + (4/7)P₂ + (1/5)P₀`. -/
theorem x4_expansion (x : ℝ) : x ^ 4 = 8 / 35 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) + 4 / 7 * ((3 * x ^ 2 - 1) / 2) + 1 / 5 := by
  veriphy_certified (ring1)
  -- certificate: de-poly.sage

/-- The coefficient formula gives the same: `c₀ = ½∫₋₁¹ x⁴ dx = 1/5`. -/
theorem x4_c0 : 1 / 2 * ∫ x in (-1 : ℝ)..1, x ^ 4 = 1 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) ((fun x => x ^ 4) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) (f' :=
          fun x => x ^ 4) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage


end KaluzaKlein.DifferentialEquations.LegendrePolynomials

namespace KaluzaKlein.DifferentialEquations.BesselFunctions

/-- The `k`-th term `(-1)ᵏ/(k!(k+n)!) (x/2)^{2k+n}` of the series of `Jₙ`. -/
noncomputable def term (n k : ℕ) (x : ℝ) : ℝ :=
  (-1) ^ k / ((k.factorial : ℝ) * ((k + n).factorial : ℝ)) * (x / 2) ^ (2 * k + n)

/-- The Bessel function `Jₙ(x) = Σₖ (-1)ᵏ/(k!(k+n)!) (x/2)^{2k+n}`. -/
noncomputable def besselJ (n : ℕ) (x : ℝ) : ℝ := ∑' k, term n k x

/-! ## def-bessel-function -/

/-- The series converges for every `x` (it is dominated by `(|x|/2)ⁿ e^{x²/4}`). -/
theorem besselJ_summable (n : ℕ) (x : ℝ) : Summable fun k => term n k x := by
  set r := |x / 2| with hr
  have hs : Summable fun k : ℕ => r ^ n * ((r ^ 2) ^ k / (k.factorial : ℝ)) :=
    (Real.summable_pow_div_factorial _).mul_left _
  refine Summable.of_norm_bounded hs fun k => ?_
  have hk : (0 : ℝ) < k.factorial := by exact_mod_cast Nat.factorial_pos k
  have hk' : (k.factorial : ℝ) ≠ 0 := hk.ne'
  have hkn : (1 : ℝ) ≤ (k + n).factorial := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_ne_zero _)
  have hnorm : ‖term n k x‖ = r ^ n * (r ^ 2) ^ k / ((k.factorial : ℝ) * ((k + n).factorial : ℝ)) := by
    simp only [term, Real.norm_eq_abs]
    rw [abs_mul, abs_div, abs_pow, abs_neg, abs_one, one_pow,
      abs_of_pos (by positivity : (0 : ℝ) < (k.factorial : ℝ) * ((k + n).factorial : ℝ)), abs_pow, ← hr]
    ring
  rw [hnorm, div_le_iff₀ (by positivity)]
  have h0 : 0 ≤ r ^ n * ((r ^ 2) ^ k / (k.factorial : ℝ)) := by positivity
  have e : r ^ n * (r ^ 2) ^ k = r ^ n * ((r ^ 2) ^ k / (k.factorial : ℝ)) * k.factorial := by field_simp
  rw [e]
  nlinarith [mul_le_mul_of_nonneg_left hkn (mul_nonneg h0 hk.le)]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_summable
-- test: de-bessel-summable-test.sage

/-! ## exm-bes-j0, exr-bes-series and prp-legendre-bessel: the truncated series in Bessel's equation -/

set_option maxHeartbeats 1000000 in
/-- `J0` truncated at `k = 6`: Bessel's equation leaves the single monomial `1/2123366400 x^14`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J0_K6 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 + 1 / 2123366400 * r ^ 12) s) x +
      x * deriv (fun s => 1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8 - 1 / 14745600 * s ^ 10 + 1 / 2123366400 * s ^ 12) x + (x ^ 2 - 0) * (1 - 1 / 4 * x ^ 2 + 1 / 64 * x ^ 4 - 1 / 2304 * x ^ 6 + 1 / 147456 * x ^ 8 - 1 / 14745600 * x ^ 10 + 1 / 2123366400 * x ^ 12) = 1 / 2123366400 * x ^ 14 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 +
                  1 / 2123366400 * r ^ 12 :
                ℝ)) =
          fun (d1 : ℝ) =>
          ((d1 ^ 11) / 176947200 - ((d1 ^ 9) / 1474560) + ((d1 ^ 7) / 18432) - ((d1 ^ 5) / 384) + ((d1 ^ 3) / 16) -
              (d1 / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 +
                  1 / 2123366400 * r ^ 12 :
                ℝ))
            (((d1 ^ 11) / 176947200 - ((d1 ^ 9) / 1474560) + ((d1 ^ 7) / 18432) - ((d1 ^ 5) / 384) + ((d1 ^ 3) / 16) -
                (d1 / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((hasDerivAt_id' (d1 : ℝ)).pow 12).const_mul (1 / 2123366400 : ℝ)).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 10).const_mul (-(1 / 14745600) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 8).const_mul (1 / 147456 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 6).const_mul (-(1 / 2304) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 4).const_mul (1 / 64 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (-(1 / 4) : ℝ))).add
                      (hasDerivAt_const (d1 : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                ((d1 ^ 11) / 176947200 - ((d1 ^ 9) / 1474560) + ((d1 ^ 7) / 18432) - ((d1 ^ 5) / 384) +
                      ((d1 ^ 3) / 16) -
                    (d1 / 2) :
                  ℝ))
              s)
          ((11 * (x ^ 10) / 176947200 - ((x ^ 8) / 163840) + (7 * (x ^ 6) / 18432) - (5 * (x ^ 4) / 384) +
                (3 * (x ^ 2) / 16) -
              (1 / 2) :
            ℝ))
          (x) :=
        by
        refine
          ((((((((((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (1 / 176947200 : ℝ)).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (-(1 / 1474560) : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (1 / 18432 : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (-(1 / 384) : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (1 / 16 : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8 - 1 / 14745600 * s ^ 10 +
              1 / 2123366400 * s ^ 12)
          (((x ^ 11) / 176947200 - ((x ^ 9) / 1474560) + ((x ^ 7) / 18432) - ((x ^ 5) / 384) + ((x ^ 3) / 16) -
              (x / 2) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (1 / 2123366400 : ℝ)).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (-(1 / 14745600) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (1 / 147456 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (-(1 / 2304) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (1 / 64 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-(1 / 4) : ℝ))).add
                    (hasDerivAt_const (x : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

set_option maxHeartbeats 1000000 in
/-- `J1` truncated at `k = 6`: Bessel's equation leaves the single monomial `1/29727129600 x^15`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J1_K6 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 - 1 / 176947200 * r ^ 11 + 1 / 29727129600 * r ^ 13) s) x +
      x * deriv (fun s => 1 / 2 * s ^ 1 - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5 - 1 / 18432 * s ^ 7 + 1 / 1474560 * s ^ 9 - 1 / 176947200 * s ^ 11 + 1 / 29727129600 * s ^ 13) x + (x ^ 2 - 1) * (1 / 2 * x ^ 1 - 1 / 16 * x ^ 3 + 1 / 384 * x ^ 5 - 1 / 18432 * x ^ 7 + 1 / 1474560 * x ^ 9 - 1 / 176947200 * x ^ 11 + 1 / 29727129600 * x ^ 13) = 1 / 29727129600 * x ^ 15 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 -
                    1 / 176947200 * r ^ 11 +
                  1 / 29727129600 * r ^ 13 :
                ℝ)) =
          fun (d1 : ℝ) =>
          (13 * (d1 ^ 12) / 29727129600 - (11 * (d1 ^ 10) / 176947200) + ((d1 ^ 8) / 163840) - (7 * (d1 ^ 6) / 18432) +
                  (5 * (d1 ^ 4) / 384) -
                (3 * (d1 ^ 2) / 16) +
              (1 / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 -
                    1 / 176947200 * r ^ 11 +
                  1 / 29727129600 * r ^ 13 :
                ℝ))
            ((13 * (d1 ^ 12) / 29727129600 - (11 * (d1 ^ 10) / 176947200) + ((d1 ^ 8) / 163840) -
                      (7 * (d1 ^ 6) / 18432) +
                    (5 * (d1 ^ 4) / 384) -
                  (3 * (d1 ^ 2) / 16) +
                (1 / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((hasDerivAt_id' (d1 : ℝ)).pow 13).const_mul (1 / 29727129600 : ℝ)).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 11).const_mul (-(1 / 176947200) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 9).const_mul (1 / 1474560 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 7).const_mul (-(1 / 18432) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 5).const_mul (1 / 384 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 3).const_mul (-(1 / 16) : ℝ))).add
                      ((hasDerivAt_id' (d1 : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (13 * (d1 ^ 12) / 29727129600 - (11 * (d1 ^ 10) / 176947200) + ((d1 ^ 8) / 163840) -
                          (7 * (d1 ^ 6) / 18432) +
                        (5 * (d1 ^ 4) / 384) -
                      (3 * (d1 ^ 2) / 16) +
                    (1 / 2) :
                  ℝ))
              s)
          ((13 * (x ^ 11) / 2477260800 - (11 * (x ^ 9) / 17694720) + ((x ^ 7) / 20480) - (7 * (x ^ 5) / 3072) +
                (5 * (x ^ 3) / 96) -
              (3 * x / 8) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (13 / 29727129600 : ℝ)).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (-(11 / 176947200) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (1 / 163840 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (-(7 / 18432) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (5 / 384 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-(3 / 16) : ℝ))).add
                    (hasDerivAt_const (x : ℝ) (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 / 2 * s ^ 1 - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5 - 1 / 18432 * s ^ 7 + 1 / 1474560 * s ^ 9 -
                1 / 176947200 * s ^ 11 +
              1 / 29727129600 * s ^ 13)
          ((13 * (x ^ 12) / 29727129600 - (11 * (x ^ 10) / 176947200) + ((x ^ 8) / 163840) - (7 * (x ^ 6) / 18432) +
                  (5 * (x ^ 4) / 384) -
                (3 * (x ^ 2) / 16) +
              (1 / 2) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 13).const_mul (1 / 29727129600 : ℝ)).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (-(1 / 176947200) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (1 / 1474560 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (-(1 / 18432) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (1 / 384 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-(1 / 16) : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

set_option maxHeartbeats 1000000 in
/-- `J2` truncated at `k = 6`: Bessel's equation leaves the single monomial `1/475634073600 x^16`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J2_K6 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 - 1 / 2477260800 * r ^ 12 + 1 / 475634073600 * r ^ 14) s) x +
      x * deriv (fun s => 1 / 8 * s ^ 2 - 1 / 96 * s ^ 4 + 1 / 3072 * s ^ 6 - 1 / 184320 * s ^ 8 + 1 / 17694720 * s ^ 10 - 1 / 2477260800 * s ^ 12 + 1 / 475634073600 * s ^ 14) x + (x ^ 2 - 4) * (1 / 8 * x ^ 2 - 1 / 96 * x ^ 4 + 1 / 3072 * x ^ 6 - 1 / 184320 * x ^ 8 + 1 / 17694720 * x ^ 10 - 1 / 2477260800 * x ^ 12 + 1 / 475634073600 * x ^ 14) = 1 / 475634073600 * x ^ 16 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 -
                    1 / 2477260800 * r ^ 12 +
                  1 / 475634073600 * r ^ 14 :
                ℝ)) =
          fun (d1 : ℝ) =>
          ((d1 ^ 13) / 33973862400 - ((d1 ^ 11) / 206438400) + ((d1 ^ 9) / 1769472) - ((d1 ^ 7) / 23040) +
                  ((d1 ^ 5) / 512) -
                ((d1 ^ 3) / 24) +
              (d1 / 4) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 -
                    1 / 2477260800 * r ^ 12 +
                  1 / 475634073600 * r ^ 14 :
                ℝ))
            (((d1 ^ 13) / 33973862400 - ((d1 ^ 11) / 206438400) + ((d1 ^ 9) / 1769472) - ((d1 ^ 7) / 23040) +
                    ((d1 ^ 5) / 512) -
                  ((d1 ^ 3) / 24) +
                (d1 / 4) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((hasDerivAt_id' (d1 : ℝ)).pow 14).const_mul (1 / 475634073600 : ℝ)).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 12).const_mul (-(1 / 2477260800) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 10).const_mul (1 / 17694720 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 8).const_mul (-(1 / 184320) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 6).const_mul (1 / 3072 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 4).const_mul (-(1 / 96) : ℝ))).add
                      (((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                ((d1 ^ 13) / 33973862400 - ((d1 ^ 11) / 206438400) + ((d1 ^ 9) / 1769472) - ((d1 ^ 7) / 23040) +
                        ((d1 ^ 5) / 512) -
                      ((d1 ^ 3) / 24) +
                    (d1 / 4) :
                  ℝ))
              s)
          ((13 * (x ^ 12) / 33973862400 - (11 * (x ^ 10) / 206438400) + ((x ^ 8) / 196608) - (7 * (x ^ 6) / 23040) +
                  (5 * (x ^ 4) / 512) -
                ((x ^ 2) / 8) +
              (1 / 4) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 13).const_mul (1 / 33973862400 : ℝ)).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (-(1 / 206438400) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (1 / 1769472 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (-(1 / 23040) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (1 / 512 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-(1 / 24) : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 / 8 * s ^ 2 - 1 / 96 * s ^ 4 + 1 / 3072 * s ^ 6 - 1 / 184320 * s ^ 8 + 1 / 17694720 * s ^ 10 -
                1 / 2477260800 * s ^ 12 +
              1 / 475634073600 * s ^ 14)
          (((x ^ 13) / 33973862400 - ((x ^ 11) / 206438400) + ((x ^ 9) / 1769472) - ((x ^ 7) / 23040) +
                  ((x ^ 5) / 512) -
                ((x ^ 3) / 24) +
              (x / 4) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((hasDerivAt_id' (x : ℝ)).pow 14).const_mul (1 / 475634073600 : ℝ)).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (-(1 / 2477260800) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (1 / 17694720 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (-(1 / 184320) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (1 / 3072 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (-(1 / 96) : ℝ))).add
                    (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

set_option maxHeartbeats 1000000 in
/-- `J0` truncated at `k = 8`: Bessel's equation leaves the single monomial `1/106542032486400 x^18`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J0_K8 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 + 1 / 2123366400 * r ^ 12 - 1 / 416179814400 * r ^ 14 + 1 / 106542032486400 * r ^ 16) s) x +
      x * deriv (fun s => 1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8 - 1 / 14745600 * s ^ 10 + 1 / 2123366400 * s ^ 12 - 1 / 416179814400 * s ^ 14 + 1 / 106542032486400 * s ^ 16) x + (x ^ 2 - 0) * (1 - 1 / 4 * x ^ 2 + 1 / 64 * x ^ 4 - 1 / 2304 * x ^ 6 + 1 / 147456 * x ^ 8 - 1 / 14745600 * x ^ 10 + 1 / 2123366400 * x ^ 12 - 1 / 416179814400 * x ^ 14 + 1 / 106542032486400 * x ^ 16) = 1 / 106542032486400 * x ^ 18 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 +
                      1 / 2123366400 * r ^ 12 -
                    1 / 416179814400 * r ^ 14 +
                  1 / 106542032486400 * r ^ 16 :
                ℝ)) =
          fun (d1 : ℝ) =>
          ((d1 ^ 15) / 6658877030400 - ((d1 ^ 13) / 29727129600) + ((d1 ^ 11) / 176947200) - ((d1 ^ 9) / 1474560) +
                    ((d1 ^ 7) / 18432) -
                  ((d1 ^ 5) / 384) +
                ((d1 ^ 3) / 16) -
              (d1 / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 - 1 / 4 * r ^ 2 + 1 / 64 * r ^ 4 - 1 / 2304 * r ^ 6 + 1 / 147456 * r ^ 8 - 1 / 14745600 * r ^ 10 +
                      1 / 2123366400 * r ^ 12 -
                    1 / 416179814400 * r ^ 14 +
                  1 / 106542032486400 * r ^ 16 :
                ℝ))
            (((d1 ^ 15) / 6658877030400 - ((d1 ^ 13) / 29727129600) + ((d1 ^ 11) / 176947200) - ((d1 ^ 9) / 1474560) +
                      ((d1 ^ 7) / 18432) -
                    ((d1 ^ 5) / 384) +
                  ((d1 ^ 3) / 16) -
                (d1 / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((((hasDerivAt_id' (d1 : ℝ)).pow 16).const_mul (1 / 106542032486400 : ℝ)).add
                                                  (((hasDerivAt_id' (d1 : ℝ)).pow 14).const_mul
                                                    (-(1 / 416179814400) : ℝ))).add
                                              (((hasDerivAt_id' (d1 : ℝ)).pow 12).const_mul (1 / 2123366400 : ℝ))).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 10).const_mul (-(1 / 14745600) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 8).const_mul (1 / 147456 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 6).const_mul (-(1 / 2304) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 4).const_mul (1 / 64 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (-(1 / 4) : ℝ))).add
                      (hasDerivAt_const (d1 : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                ((d1 ^ 15) / 6658877030400 - ((d1 ^ 13) / 29727129600) + ((d1 ^ 11) / 176947200) -
                            ((d1 ^ 9) / 1474560) +
                          ((d1 ^ 7) / 18432) -
                        ((d1 ^ 5) / 384) +
                      ((d1 ^ 3) / 16) -
                    (d1 / 2) :
                  ℝ))
              s)
          (((x ^ 14) / 443925135360 - (13 * (x ^ 12) / 29727129600) + (11 * (x ^ 10) / 176947200) - ((x ^ 8) / 163840) +
                    (7 * (x ^ 6) / 18432) -
                  (5 * (x ^ 4) / 384) +
                (3 * (x ^ 2) / 16) -
              (1 / 2) :
            ℝ))
          (x) :=
        by
        refine
          ((((((((((((hasDerivAt_id' (x : ℝ)).pow 15).const_mul (1 / 6658877030400 : ℝ)).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 13).const_mul (-(1 / 29727129600) : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (1 / 176947200 : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (-(1 / 1474560) : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (1 / 18432 : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (-(1 / 384) : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (1 / 16 : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8 - 1 / 14745600 * s ^ 10 +
                  1 / 2123366400 * s ^ 12 -
                1 / 416179814400 * s ^ 14 +
              1 / 106542032486400 * s ^ 16)
          (((x ^ 15) / 6658877030400 - ((x ^ 13) / 29727129600) + ((x ^ 11) / 176947200) - ((x ^ 9) / 1474560) +
                    ((x ^ 7) / 18432) -
                  ((x ^ 5) / 384) +
                ((x ^ 3) / 16) -
              (x / 2) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((((hasDerivAt_id' (x : ℝ)).pow 16).const_mul (1 / 106542032486400 : ℝ)).add
                                                (((hasDerivAt_id' (x : ℝ)).pow 14).const_mul
                                                  (-(1 / 416179814400) : ℝ))).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (1 / 2123366400 : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (-(1 / 14745600) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (1 / 147456 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (-(1 / 2304) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (1 / 64 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-(1 / 4) : ℝ))).add
                    (hasDerivAt_const (x : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

set_option maxHeartbeats 1000000 in
/-- `J1` truncated at `k = 8`: Bessel's equation leaves the single monomial `1/1917756584755200 x^19`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J1_K8 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 - 1 / 176947200 * r ^ 11 + 1 / 29727129600 * r ^ 13 - 1 / 6658877030400 * r ^ 15 + 1 / 1917756584755200 * r ^ 17) s) x +
      x * deriv (fun s => 1 / 2 * s ^ 1 - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5 - 1 / 18432 * s ^ 7 + 1 / 1474560 * s ^ 9 - 1 / 176947200 * s ^ 11 + 1 / 29727129600 * s ^ 13 - 1 / 6658877030400 * s ^ 15 + 1 / 1917756584755200 * s ^ 17) x + (x ^ 2 - 1) * (1 / 2 * x ^ 1 - 1 / 16 * x ^ 3 + 1 / 384 * x ^ 5 - 1 / 18432 * x ^ 7 + 1 / 1474560 * x ^ 9 - 1 / 176947200 * x ^ 11 + 1 / 29727129600 * x ^ 13 - 1 / 6658877030400 * x ^ 15 + 1 / 1917756584755200 * x ^ 17) = 1 / 1917756584755200 * x ^ 19 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 -
                        1 / 176947200 * r ^ 11 +
                      1 / 29727129600 * r ^ 13 -
                    1 / 6658877030400 * r ^ 15 +
                  1 / 1917756584755200 * r ^ 17 :
                ℝ)) =
          fun (d1 : ℝ) =>
          (17 * (d1 ^ 16) / 1917756584755200 - ((d1 ^ 14) / 443925135360) + (13 * (d1 ^ 12) / 29727129600) -
                        (11 * (d1 ^ 10) / 176947200) +
                      ((d1 ^ 8) / 163840) -
                    (7 * (d1 ^ 6) / 18432) +
                  (5 * (d1 ^ 4) / 384) -
                (3 * (d1 ^ 2) / 16) +
              (1 / 2) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 / 2 * r ^ 1 - 1 / 16 * r ^ 3 + 1 / 384 * r ^ 5 - 1 / 18432 * r ^ 7 + 1 / 1474560 * r ^ 9 -
                        1 / 176947200 * r ^ 11 +
                      1 / 29727129600 * r ^ 13 -
                    1 / 6658877030400 * r ^ 15 +
                  1 / 1917756584755200 * r ^ 17 :
                ℝ))
            ((17 * (d1 ^ 16) / 1917756584755200 - ((d1 ^ 14) / 443925135360) + (13 * (d1 ^ 12) / 29727129600) -
                          (11 * (d1 ^ 10) / 176947200) +
                        ((d1 ^ 8) / 163840) -
                      (7 * (d1 ^ 6) / 18432) +
                    (5 * (d1 ^ 4) / 384) -
                  (3 * (d1 ^ 2) / 16) +
                (1 / 2) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((((hasDerivAt_id' (d1 : ℝ)).pow 17).const_mul (1 / 1917756584755200 : ℝ)).add
                                                  (((hasDerivAt_id' (d1 : ℝ)).pow 15).const_mul
                                                    (-(1 / 6658877030400) : ℝ))).add
                                              (((hasDerivAt_id' (d1 : ℝ)).pow 13).const_mul (1 / 29727129600 : ℝ))).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 11).const_mul (-(1 / 176947200) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 9).const_mul (1 / 1474560 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 7).const_mul (-(1 / 18432) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 5).const_mul (1 / 384 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 3).const_mul (-(1 / 16) : ℝ))).add
                      ((hasDerivAt_id' (d1 : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                (17 * (d1 ^ 16) / 1917756584755200 - ((d1 ^ 14) / 443925135360) + (13 * (d1 ^ 12) / 29727129600) -
                              (11 * (d1 ^ 10) / 176947200) +
                            ((d1 ^ 8) / 163840) -
                          (7 * (d1 ^ 6) / 18432) +
                        (5 * (d1 ^ 4) / 384) -
                      (3 * (d1 ^ 2) / 16) +
                    (1 / 2) :
                  ℝ))
              s)
          ((17 * (x ^ 15) / 119859786547200 - ((x ^ 13) / 31708938240) + (13 * (x ^ 11) / 2477260800) -
                      (11 * (x ^ 9) / 17694720) +
                    ((x ^ 7) / 20480) -
                  (7 * (x ^ 5) / 3072) +
                (5 * (x ^ 3) / 96) -
              (3 * x / 8) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((((hasDerivAt_id' (x : ℝ)).pow 16).const_mul (17 / 1917756584755200 : ℝ)).add
                                                (((hasDerivAt_id' (x : ℝ)).pow 14).const_mul
                                                  (-(1 / 443925135360) : ℝ))).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (13 / 29727129600 : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (-(11 / 176947200) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (1 / 163840 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (-(7 / 18432) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (5 / 384 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-(3 / 16) : ℝ))).add
                    (hasDerivAt_const (x : ℝ) (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 / 2 * s ^ 1 - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5 - 1 / 18432 * s ^ 7 + 1 / 1474560 * s ^ 9 -
                    1 / 176947200 * s ^ 11 +
                  1 / 29727129600 * s ^ 13 -
                1 / 6658877030400 * s ^ 15 +
              1 / 1917756584755200 * s ^ 17)
          ((17 * (x ^ 16) / 1917756584755200 - ((x ^ 14) / 443925135360) + (13 * (x ^ 12) / 29727129600) -
                        (11 * (x ^ 10) / 176947200) +
                      ((x ^ 8) / 163840) -
                    (7 * (x ^ 6) / 18432) +
                  (5 * (x ^ 4) / 384) -
                (3 * (x ^ 2) / 16) +
              (1 / 2) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((((hasDerivAt_id' (x : ℝ)).pow 17).const_mul (1 / 1917756584755200 : ℝ)).add
                                                (((hasDerivAt_id' (x : ℝ)).pow 15).const_mul
                                                  (-(1 / 6658877030400) : ℝ))).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 13).const_mul (1 / 29727129600 : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (-(1 / 176947200) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (1 / 1474560 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (-(1 / 18432) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (1 / 384 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-(1 / 16) : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

set_option maxHeartbeats 1000000 in
/-- `J2` truncated at `k = 8`: Bessel's equation leaves the single monomial `1/38355131695104000 x^20`
(`(-1)^K/(K!(K+n)!) x^{2K+n+2}/2^{2K+n}`). -/
theorem truncated_J2_K8 (x : ℝ) :
    x ^ 2 * deriv (fun s => deriv (fun r => 1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 - 1 / 2477260800 * r ^ 12 + 1 / 475634073600 * r ^ 14 - 1 / 119859786547200 * r ^ 16 + 1 / 38355131695104000 * r ^ 18) s) x +
      x * deriv (fun s => 1 / 8 * s ^ 2 - 1 / 96 * s ^ 4 + 1 / 3072 * s ^ 6 - 1 / 184320 * s ^ 8 + 1 / 17694720 * s ^ 10 - 1 / 2477260800 * s ^ 12 + 1 / 475634073600 * s ^ 14 - 1 / 119859786547200 * s ^ 16 + 1 / 38355131695104000 * s ^ 18) x + (x ^ 2 - 4) * (1 / 8 * x ^ 2 - 1 / 96 * x ^ 4 + 1 / 3072 * x ^ 6 - 1 / 184320 * x ^ 8 + 1 / 17694720 * x ^ 10 - 1 / 2477260800 * x ^ 12 + 1 / 475634073600 * x ^ 14 - 1 / 119859786547200 * x ^ 16 + 1 / 38355131695104000 * x ^ 18) = 1 / 38355131695104000 * x ^ 20 := by
  veriphy_certified
    ( have i0 :
        deriv
            (fun (r : ℝ) =>
              (1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 -
                        1 / 2477260800 * r ^ 12 +
                      1 / 475634073600 * r ^ 14 -
                    1 / 119859786547200 * r ^ 16 +
                  1 / 38355131695104000 * r ^ 18 :
                ℝ)) =
          fun (d1 : ℝ) =>
          ((d1 ^ 17) / 2130840649728000 - ((d1 ^ 15) / 7491236659200) + ((d1 ^ 13) / 33973862400) -
                        ((d1 ^ 11) / 206438400) +
                      ((d1 ^ 9) / 1769472) -
                    ((d1 ^ 7) / 23040) +
                  ((d1 ^ 5) / 512) -
                ((d1 ^ 3) / 24) +
              (d1 / 4) :
            ℝ) :=
        by
        funext d1
        have hd :
          HasDerivAt
            (fun (r : ℝ) =>
              (1 / 8 * r ^ 2 - 1 / 96 * r ^ 4 + 1 / 3072 * r ^ 6 - 1 / 184320 * r ^ 8 + 1 / 17694720 * r ^ 10 -
                        1 / 2477260800 * r ^ 12 +
                      1 / 475634073600 * r ^ 14 -
                    1 / 119859786547200 * r ^ 16 +
                  1 / 38355131695104000 * r ^ 18 :
                ℝ))
            (((d1 ^ 17) / 2130840649728000 - ((d1 ^ 15) / 7491236659200) + ((d1 ^ 13) / 33973862400) -
                          ((d1 ^ 11) / 206438400) +
                        ((d1 ^ 9) / 1769472) -
                      ((d1 ^ 7) / 23040) +
                    ((d1 ^ 5) / 512) -
                  ((d1 ^ 3) / 24) +
                (d1 / 4) :
              ℝ))
            (d1) :=
          by
          refine
            (((((((((((((hasDerivAt_id' (d1 : ℝ)).pow 18).const_mul (1 / 38355131695104000 : ℝ)).add
                                                  (((hasDerivAt_id' (d1 : ℝ)).pow 16).const_mul
                                                    (-(1 / 119859786547200) : ℝ))).add
                                              (((hasDerivAt_id' (d1 : ℝ)).pow 14).const_mul (1 / 475634073600 : ℝ))).add
                                          (((hasDerivAt_id' (d1 : ℝ)).pow 12).const_mul (-(1 / 2477260800) : ℝ))).add
                                      (((hasDerivAt_id' (d1 : ℝ)).pow 10).const_mul (1 / 17694720 : ℝ))).add
                                  (((hasDerivAt_id' (d1 : ℝ)).pow 8).const_mul (-(1 / 184320) : ℝ))).add
                              (((hasDerivAt_id' (d1 : ℝ)).pow 6).const_mul (1 / 3072 : ℝ))).add
                          (((hasDerivAt_id' (d1 : ℝ)).pow 4).const_mul (-(1 / 96) : ℝ))).add
                      (((hasDerivAt_id' (d1 : ℝ)).pow 2).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      rw [i0]
      have h1 :
        HasDerivAt
          (fun s =>
            (fun (d1 : ℝ) =>
                ((d1 ^ 17) / 2130840649728000 - ((d1 ^ 15) / 7491236659200) + ((d1 ^ 13) / 33973862400) -
                              ((d1 ^ 11) / 206438400) +
                            ((d1 ^ 9) / 1769472) -
                          ((d1 ^ 7) / 23040) +
                        ((d1 ^ 5) / 512) -
                      ((d1 ^ 3) / 24) +
                    (d1 / 4) :
                  ℝ))
              s)
          ((17 * (x ^ 16) / 2130840649728000 - ((x ^ 14) / 499415777280) + (13 * (x ^ 12) / 33973862400) -
                        (11 * (x ^ 10) / 206438400) +
                      ((x ^ 8) / 196608) -
                    (7 * (x ^ 6) / 23040) +
                  (5 * (x ^ 4) / 512) -
                ((x ^ 2) / 8) +
              (1 / 4) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((((hasDerivAt_id' (x : ℝ)).pow 17).const_mul (1 / 2130840649728000 : ℝ)).add
                                                (((hasDerivAt_id' (x : ℝ)).pow 15).const_mul
                                                  (-(1 / 7491236659200) : ℝ))).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 13).const_mul (1 / 33973862400 : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 11).const_mul (-(1 / 206438400) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 9).const_mul (1 / 1769472 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 7).const_mul (-(1 / 23040) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 5).const_mul (1 / 512 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-(1 / 24) : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h1.deriv]
      have h2 :
        HasDerivAt
          (fun s =>
            1 / 8 * s ^ 2 - 1 / 96 * s ^ 4 + 1 / 3072 * s ^ 6 - 1 / 184320 * s ^ 8 + 1 / 17694720 * s ^ 10 -
                    1 / 2477260800 * s ^ 12 +
                  1 / 475634073600 * s ^ 14 -
                1 / 119859786547200 * s ^ 16 +
              1 / 38355131695104000 * s ^ 18)
          (((x ^ 17) / 2130840649728000 - ((x ^ 15) / 7491236659200) + ((x ^ 13) / 33973862400) -
                        ((x ^ 11) / 206438400) +
                      ((x ^ 9) / 1769472) -
                    ((x ^ 7) / 23040) +
                  ((x ^ 5) / 512) -
                ((x ^ 3) / 24) +
              (x / 4) :
            ℝ))
          (x) :=
        by
        refine
          (((((((((((((hasDerivAt_id' (x : ℝ)).pow 18).const_mul (1 / 38355131695104000 : ℝ)).add
                                                (((hasDerivAt_id' (x : ℝ)).pow 16).const_mul
                                                  (-(1 / 119859786547200) : ℝ))).add
                                            (((hasDerivAt_id' (x : ℝ)).pow 14).const_mul (1 / 475634073600 : ℝ))).add
                                        (((hasDerivAt_id' (x : ℝ)).pow 12).const_mul (-(1 / 2477260800) : ℝ))).add
                                    (((hasDerivAt_id' (x : ℝ)).pow 10).const_mul (1 / 17694720 : ℝ))).add
                                (((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (-(1 / 184320) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (1 / 3072 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (-(1 / 96) : ℝ))).add
                    (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h2.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-! ## exm-bes-j0 -/

/-- `J₀(0) = 1` and `Jₙ(0) = 0` for `n ≥ 1`. -/
theorem besselJ_at_zero (n : ℕ) : besselJ n 0 = if n = 0 then 1 else 0 := by
  rw [besselJ, tsum_eq_single 0]
  · split_ifs with h
    · subst h; simp [term]
    · simp [term, h]
  · intro k hk
    simp [term, hk]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_at_zero
-- test: de-bessel-at-zero-test.sage


/-! ## exm-bes-derivative and exr-bes-recurrence: the derivative relations, termwise on the truncations -/

/-- `J₀' = -J₁` term by term: `d/dx` of `J₀` truncated at `k = 4` is minus `J₁` truncated at `k = 3`. -/
theorem truncated_J0_derivative (x : ℝ) :
    deriv (fun s => 1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8) x =
      -(1 / 2 * x ^ 1 - 1 / 16 * x ^ 3 + 1 / 384 * x ^ 5 - 1 / 18432 * x ^ 7) := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun s => 1 - 1 / 4 * s ^ 2 + 1 / 64 * s ^ 4 - 1 / 2304 * s ^ 6 + 1 / 147456 * s ^ 8)
          (((x ^ 7) / 18432 - ((x ^ 5) / 384) + ((x ^ 3) / 16) - (x / 2) : ℝ)) (x) :=
        by
        refine
          (((((((((hasDerivAt_id' (x : ℝ)).pow 8).const_mul (1 / 147456 : ℝ)).add
                                (((hasDerivAt_id' (x : ℝ)).pow 6).const_mul (-(1 / 2304) : ℝ))).add
                            (((hasDerivAt_id' (x : ℝ)).pow 4).const_mul (1 / 64 : ℝ))).add
                        (((hasDerivAt_id' (x : ℝ)).pow 2).const_mul (-(1 / 4) : ℝ))).add
                    (hasDerivAt_const (x : ℝ) (1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring_nf; done)
        · ring1
      try rw [h0.deriv]
      try beta_reduce
      all_goals (ring1))
  -- certificate: de-partials.sage

/-- `d/dx[xⁿJₙ] = xⁿJₙ₋₁` for `n = 1`, on the truncations at `k = 2`: `d/dx[x(x/2 - x³/16 + x⁵/384)]
= x(1 - x²/4 + x⁴/64)`. -/
theorem truncated_recurrence_up (x : ℝ) :
    deriv (fun s => s * (1 / 2 * s - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5)) x = x * (1 - 1 / 4 * x ^ 2 + 1 / 64 * x ^ 4) := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun s => s * (1 / 2 * s - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5))
          (((x ^ 5) / 384 - ((x ^ 3) / 16) + ((5 * (x ^ 4) - (72 * (x ^ 2)) + 192) * x / 384) + (x / 2) : ℝ)) (x) :=
        by
        convert
          ((((((hasDerivAt_id' (x : ℝ)).pow 5).add (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-24 : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (192 : ℝ))).mul
                (hasDerivAt_id' (x : ℝ))).const_mul
            (1 / 384 : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.pow_apply]; ring1)
      try rw [h0.deriv]
      beta_reduce
      ring1)
  -- certificate: de-partials.sage

/-- `d/dx[x⁻ⁿJₙ] = -x⁻ⁿJₙ₊₁` for `n = 1`, on the truncations (`k ≤ 2` for `J₁`, `k ≤ 1` for `J₂`), `x ≠ 0`. -/
theorem truncated_recurrence_down (x : ℝ) (hx : x ≠ 0) :
    deriv (fun s => (1 / 2 * s - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5) / s) x = -((1 / 8 * x ^ 2 - 1 / 96 * x ^ 4) / x) := by
  veriphy_certified
    ( have h0 :
        HasDerivAt (fun s => (1 / 2 * s - 1 / 16 * s ^ 3 + 1 / 384 * s ^ 5) / s)
          (((5 * (x ^ 4) - (72 * (x ^ 2)) + 192) / (384 * x) -
              ((x ^ 5 - (24 * (x ^ 3)) + (192 * x)) / (384 * (x ^ 2))) :
            ℝ))
          (x) :=
        by
        convert
          ((((((hasDerivAt_id' (x : ℝ)).pow 5).add (((hasDerivAt_id' (x : ℝ)).pow 3).const_mul (-24 : ℝ))).add
                    ((hasDerivAt_id' (x : ℝ)).const_mul (192 : ℝ))).div
                (hasDerivAt_id' (x : ℝ)) hx).const_mul
            (1 / 384 : ℝ)) using
          1
        all_goals
          first
          | rfl
          | (funext y; simp only [Pi.add_apply, Pi.div_apply, Pi.pow_apply]; ring_nf; done)
          | (simp only [Pi.add_apply, Pi.pow_apply]; (field_simp; ring1))
      try rw [h0.deriv]
      beta_reduce
      (field_simp; ring1))
  -- certificate: de-partials.sage

/-- Termwise identity behind `Jₙ₋₁ + Jₙ₊₁ = (2n/x)Jₙ`: with `m = n + 1`,
`term n (k+1) + term (n+2) k = (2(n+1)/x) term (n+1) (k+1)`. -/
theorem term_recurrence (n k : ℕ) (x : ℝ) (hx : x ≠ 0) :
    term n (k + 1) x + term (n + 2) k x = 2 * (n + 1) / x * term (n + 1) (k + 1) x := by
  simp only [term]
  rw [show k + 1 + n = k + n + 1 by ring, show k + (n + 2) = k + n + 1 + 1 by ring,
    show k + 1 + (n + 1) = k + n + 1 + 1 by ring, show 2 * (k + 1) + n = 2 * k + n + 2 by ring,
    show 2 * k + (n + 2) = 2 * k + n + 2 by ring, show 2 * (k + 1) + (n + 1) = 2 * k + n + 2 + 1 by ring]
  simp only [Nat.factorial_succ]
  push_cast
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.term_recurrence
-- test: de-term-recurrence-test.sage

/-- The first terms: `term n 0 = (2(n+1)/x) term (n+1) 0`. -/
theorem term_recurrence_zero (n : ℕ) (x : ℝ) (hx : x ≠ 0) : term n 0 x = 2 * (n + 1) / x * term (n + 1) 0 x := by
  simp only [term, Nat.factorial_zero, Nat.cast_one, one_mul, pow_zero, zero_add, mul_zero]
  rw [show (n + 1).factorial = (n + 1) * n.factorial from Nat.factorial_succ n]
  push_cast
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.term_recurrence_zero
-- test: de-term-recurrence-zero-test.sage

/-- `Jₙ₋₁ + Jₙ₊₁ = (2n/x)Jₙ` (stated for `n + 1`, `x ≠ 0`). -/
theorem besselJ_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
    besselJ n x + besselJ (n + 2) x = 2 * (n + 1) / x * besselJ (n + 1) x := by
  have h₁ := besselJ_summable n x
  have h₂ := besselJ_summable (n + 2) x
  have h₃ := besselJ_summable (n + 1) x
  have h₁' : Summable fun k => term n (k + 1) x := (summable_nat_add_iff 1).mpr h₁
  rw [besselJ, besselJ, besselJ, h₁.tsum_eq_zero_add, h₃.tsum_eq_zero_add, add_assoc,
    ← h₁'.tsum_add h₂, mul_add, ← tsum_mul_left, term_recurrence_zero n x hx]
  congr 1
  exact tsum_congr fun k => term_recurrence n k x hx
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_recurrence
-- test: de-bessel-recurrence-test.sage

/-! ## def-bessel-function, exr-bes-zeros, exm-plb-numerical-zero: the first zero of `J₀` -/

/-- The terms `gᵢ = (x/2)^{2i+2}/((i+1)!)²` of `1 - J₀(x)`. -/
noncomputable def g0 (x : ℝ) (i : ℕ) : ℝ := (x / 2) ^ (2 * i + 2) / ((i + 1).factorial : ℝ) ^ 2

/-- `J₀(x) = 1 - Σᵢ (-1)ⁱ gᵢ`. -/
theorem besselJ0_eq (x : ℝ) : besselJ 0 x = 1 - ∑' i, (-1) ^ i * g0 x i := by
  rw [besselJ, (besselJ_summable 0 x).tsum_eq_zero_add]
  rw [show (∑' i, term 0 (i + 1) x) = -∑' i, (-1) ^ i * g0 x i by
    rw [← tsum_neg]; congr 1; funext i; simp [term, g0, pow_succ]; ring]
  simp [term]; ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_eq
-- test: de-bessel-j0-eq-test.sage

/-- For `0 ≤ x ≤ 4` the `gᵢ` decrease (`(x/2)² ≤ (i+2)²`) … -/
theorem g0_antitone (x : ℝ) (h0 : 0 ≤ x) (h4 : x ≤ 4) : Antitone (g0 x) := by
  refine antitone_nat_of_succ_le fun i => ?_
  simp only [g0]
  rw [show 2 * (i + 1) + 2 = 2 * i + 2 + 2 by ring, pow_add, Nat.factorial_succ (i + 1)]
  push_cast
  have hf : (0 : ℝ) < (i + 1).factorial := by exact_mod_cast Nat.factorial_pos _
  have hq : (x / 2) ^ 2 ≤ ((i : ℝ) + 1 + 1) ^ 2 := by
    apply pow_le_pow_left₀ (by positivity); linarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)]
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  have hp : 0 ≤ (x / 2) ^ (2 * i + 2) := by positivity
  nlinarith [mul_le_mul_of_nonneg_left hq (mul_nonneg hp (sq_nonneg ((i + 1).factorial : ℝ)))]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.g0_antitone
-- test: de-g0-antitone-test.sage

/-- … and they are summable. -/
theorem g0_summable (x : ℝ) : Summable (g0 x) := by
  have h := ((summable_nat_add_iff 1).mpr (besselJ_summable 0 x)).abs
  refine h.congr fun i => ?_
  have he : Even (2 * i + 2) := ⟨i + 1, by ring⟩
  simp only [term, g0, add_zero, show 2 * (i + 1) = 2 * i + 2 by ring]
  rw [abs_mul, abs_div, abs_pow, abs_neg, abs_one, one_pow,
    abs_of_pos (by positivity : (0 : ℝ) < ((i + 1).factorial : ℝ) * ((i + 1).factorial : ℝ)), abs_pow, he.pow_abs]
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.g0_summable
-- test: de-g0-summable-test.sage

/-- Alternating-series bounds: for `0 ≤ x ≤ 4`, `1 - Σ_{i<2k+1}(-1)ⁱgᵢ ≤ J₀(x) ≤ 1 - Σ_{i<2k}(-1)ⁱgᵢ`. -/
theorem besselJ0_bounds (x : ℝ) (h0 : 0 ≤ x) (h4 : x ≤ 4) (k : ℕ) :
    1 - ∑ i ∈ Finset.range (2 * k + 1), (-1) ^ i * g0 x i ≤ besselJ 0 x ∧
      besselJ 0 x ≤ 1 - ∑ i ∈ Finset.range (2 * k), (-1) ^ i * g0 x i := by
  have hl := (g0_summable x).tendsto_alternating_series_tsum
  have ha := g0_antitone x h0 h4
  rw [besselJ0_eq]
  exact ⟨by linarith [ha.tendsto_le_alternating_series hl k], by linarith [ha.alternating_series_le_tendsto hl k]⟩
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_bounds
-- test: de-bessel-bounds-test.sage

/-- `J₀` is continuous on every `[-R, R]` (the series converges uniformly there). -/
theorem besselJ0_continuousOn (R : ℝ) : ContinuousOn (besselJ 0) (Set.Icc (-R) R) := by
  have hs : Summable fun k : ℕ => (R ^ 2 / 4) ^ k / (k.factorial : ℝ) := Real.summable_pow_div_factorial _
  refine continuousOn_tsum (fun k => by unfold term; fun_prop) hs fun k x hx => ?_
  have hk : (0 : ℝ) < k.factorial := by exact_mod_cast Nat.factorial_pos k
  have hk1 : (1 : ℝ) ≤ k.factorial := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_ne_zero k)
  have hx2 : (x / 2) ^ 2 ≤ R ^ 2 / 4 := by
    have : x ^ 2 ≤ R ^ 2 := sq_le_sq' (by linarith [hx.1]) hx.2
    rw [div_pow]; linarith
  have hp : ((x / 2) ^ 2) ^ k ≤ (R ^ 2 / 4) ^ k := pow_le_pow_left₀ (by positivity) hx2 k
  have hnorm : ‖term 0 k x‖ = ((x / 2) ^ 2) ^ k / ((k.factorial : ℝ) * (k.factorial : ℝ)) := by
    have he : Even (2 * k) := ⟨k, by ring⟩
    simp only [term, add_zero, Real.norm_eq_abs]
    rw [abs_mul, abs_div, abs_pow, abs_neg, abs_one, one_pow,
      abs_of_pos (by positivity : (0 : ℝ) < (k.factorial : ℝ) * (k.factorial : ℝ)), abs_pow, he.pow_abs, pow_mul]
    ring
  rw [hnorm, div_le_div_iff₀ (by positivity) hk]
  have h0 : 0 ≤ ((x / 2) ^ 2) ^ k := by positivity
  nlinarith [mul_le_mul_of_nonneg_right hp hk.le,
    mul_le_mul_of_nonneg_left hk1 (by positivity : (0 : ℝ) ≤ (R ^ 2 / 4) ^ k * k.factorial)]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_continuousOn
-- test: de-bessel-continuous-test.sage

/-- A sign change of the bounds brackets a zero: if the lower bound (seven terms) at `a` is positive and the upper
bound (eight terms) at `b` is negative (`0 ≤ a < b ≤ 4`), then `J₀` vanishes somewhere in `(a, b)`. -/
theorem besselJ0_zero_of_bracket (a b : ℝ) (ha : 0 ≤ a) (hab : a < b) (hb : b ≤ 4)
    (hA : 0 < 1 - (a / 2) ^ 2 + (a / 2) ^ 4 / 4 - (a / 2) ^ 6 / 36 + (a / 2) ^ 8 / 576 - (a / 2) ^ 10 / 14400 + (a / 2) ^ 12 / 518400 - (a / 2) ^ 14 / 25401600)
    (hB : 1 - (b / 2) ^ 2 + (b / 2) ^ 4 / 4 - (b / 2) ^ 6 / 36 + (b / 2) ^ 8 / 576 - (b / 2) ^ 10 / 14400 + (b / 2) ^ 12 / 518400 - (b / 2) ^ 14 / 25401600 + (b / 2) ^ 16 / 1625702400 < 0) :
    ∃ x ∈ Set.Ioo a b, besselJ 0 x = 0 := by
  have eA : 1 - ∑ i ∈ Finset.range (2 * 3 + 1), (-1) ^ i * g0 a i = 1 - (a / 2) ^ 2 + (a / 2) ^ 4 / 4 - (a / 2) ^ 6 / 36 + (a / 2) ^ 8 / 576 - (a / 2) ^ 10 / 14400 + (a / 2) ^ 12 / 518400 - (a / 2) ^ 14 / 25401600 := by
    norm_num [g0, Finset.sum_range_succ, Nat.factorial]; ring
  have eB : 1 - ∑ i ∈ Finset.range (2 * 4), (-1) ^ i * g0 b i = 1 - (b / 2) ^ 2 + (b / 2) ^ 4 / 4 - (b / 2) ^ 6 / 36 + (b / 2) ^ 8 / 576 - (b / 2) ^ 10 / 14400 + (b / 2) ^ 12 / 518400 - (b / 2) ^ 14 / 25401600 + (b / 2) ^ 16 / 1625702400 := by
    norm_num [g0, Finset.sum_range_succ, Nat.factorial]; ring
  have hJa : 0 < besselJ 0 a := by
    have := (besselJ0_bounds a ha (by linarith) 3).1; rw [eA] at this; linarith
  have hJb : besselJ 0 b < 0 := by
    have := (besselJ0_bounds b (by linarith) hb 4).2; rw [eB] at this; linarith
  have hcont : ContinuousOn (besselJ 0) (Set.Icc a b) :=
    (besselJ0_continuousOn 4).mono (Set.Icc_subset_Icc (by linarith) hb)
  obtain ⟨x, hx, hx0⟩ := intermediate_value_Icc' hab.le hcont ⟨hJb.le, hJa.le⟩
  refine ⟨x, ⟨lt_of_le_of_ne hx.1 ?_, lt_of_le_of_ne hx.2 ?_⟩, hx0⟩
  · rintro rfl; linarith
  · rintro rfl; linarith
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_zero_of_bracket
-- test: de-bessel-bracket-test.sage

/-- The bracket at `a = 2.404825`: the lower bound is positive … -/
theorem j01_lower : (0 : ℝ) < 1 - (480965 / 200000 / 2) ^ 2 + (480965 / 200000 / 2) ^ 4 / 4 - (480965 / 200000 / 2) ^ 6 / 36 + (480965 / 200000 / 2) ^ 8 / 576 - (480965 / 200000 / 2) ^ 10 / 14400 + (480965 / 200000 / 2) ^ 12 / 518400 - (480965 / 200000 / 2) ^ 14 / 25401600 := by
  veriphy_certified (norm_num)
  -- certificate: de-rewrite.sage

/-- … and at `b = 2.404826` the upper bound is negative: `j₀,₁ ∈ (2.404825, 2.404826)`. -/
theorem j01_upper : (1 - (1202413 / 500000 / 2) ^ 2 + (1202413 / 500000 / 2) ^ 4 / 4 - (1202413 / 500000 / 2) ^ 6 / 36 + (1202413 / 500000 / 2) ^ 8 / 576 - (1202413 / 500000 / 2) ^ 10 / 14400 + (1202413 / 500000 / 2) ^ 12 / 518400 - (1202413 / 500000 / 2) ^ 14 / 25401600 + (1202413 / 500000 / 2) ^ 16 / 1625702400 : ℝ) < 0 := by
  veriphy_certified (norm_num)
  -- certificate: de-rewrite.sage

/-! ## exr-plb-bessel-orthogonality -/

/-- If `F` satisfies Bessel's equation of order `0`, `y²F'' + yF' + y²F = 0`, then `u(x) = F(jx)` has
`u' = jF'(jx)`, `u'' = j²F''(jx)` and `x²u'' + xu' + j²x²u = 0`, that is `-(xu')' = j²xu`: the drum modes are
Sturm-Liouville eigenfunctions with weight `x`. -/
theorem drum_mode (j x : ℝ) (F F' F'' : ℝ → ℝ) (hF : ∀ y, HasDerivAt F (F' y) y) (hF' : ∀ y, HasDerivAt F' (F'' y) y)
    (hB : ∀ y, y ^ 2 * F'' y + y * F' y + y ^ 2 * F y = 0) :
    HasDerivAt (fun s => F (j * s)) (j * F' (j * x)) x ∧
      HasDerivAt (fun s => j * F' (j * s)) (j ^ 2 * F'' (j * x)) x ∧
      x ^ 2 * (j ^ 2 * F'' (j * x)) + x * (j * F' (j * x)) + j ^ 2 * x ^ 2 * F (j * x) = 0 := by
  refine ⟨?_, ?_, by linear_combination hB (j * x)⟩
  · have := (hF (j * x)).comp x ((hasDerivAt_id' x).const_mul j)
    convert this using 1 <;> first | rfl | ring | (simp; ring) | simp [Function.comp_def]
  · have := ((hF' (j * x)).comp x ((hasDerivAt_id' x).const_mul j)).const_mul j
    convert this using 1 <;> first | rfl | ring | (simp; ring) | simp [Function.comp_def]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.DifferentialEquations.BesselFunctions.drum_mode
-- test: de-drum-mode-test.sage

end KaluzaKlein.DifferentialEquations.BesselFunctions

namespace KaluzaKlein.DifferentialEquations.SturmLiouvilleComputations

/-! ## prp-legendre-bessel: Legendre's equation for `n ≤ 5` -/

/-- `P0` solves `(1 - x²)u'' - 2xu' + 0u = 0`, i.e. `((1 - x²)P0')' + 0P0 = 0`. -/
theorem legendre_equation_0 (u : ℝ → ℝ) (hu : u = fun s => 1) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 0 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun (s : ℝ) => (1 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (1 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (1 : ℝ))).congr_of_eventuallyEq
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
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `P1` solves `(1 - x²)u'' - 2xu' + 2u = 0`, i.e. `((1 - x²)P1')' + 2P1 = 0`. -/
theorem legendre_equation_1 (u : ℝ → ℝ) (hu : u = fun s => s) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 2 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun (s : ℝ) => (s : ℝ)) = fun (s : ℝ) => (1 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (s : ℝ)) ((1 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_id' (s : ℝ))).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (1 : ℝ)) = fun (s : ℝ) => (0 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (1 : ℝ)) ((0 : ℝ)) (s) :=
          by
          refine
            (((hasDerivAt_const (s : ℝ) (1 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · rfl
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `P2` solves `(1 - x²)u'' - 2xu' + 6u = 0`, i.e. `((1 - x²)P2')' + 6P2 = 0`. -/
theorem legendre_equation_2 (u : ℝ → ℝ) (hu : u = fun s => (3 * s ^ 2 - 1) / 2) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 6 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 : deriv (fun (s : ℝ) => ((3 * s ^ 2 - 1) / 2 : ℝ)) = fun (s : ℝ) => (3 * s : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => ((3 * s ^ 2 - 1) / 2 : ℝ)) ((3 * s : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (3 / 2 : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (3 * s : ℝ)) = fun (s : ℝ) => (3 : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (3 * s : ℝ)) ((3 : ℝ)) (s) :=
          by
          refine
            ((((hasDerivAt_id' (s : ℝ)).const_mul (3 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; rfl
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `P3` solves `(1 - x²)u'' - 2xu' + 12u = 0`, i.e. `((1 - x²)P3')' + 12P3 = 0`. -/
theorem legendre_equation_3 (u : ℝ → ℝ) (hu : u = fun s => (5 * s ^ 3 - 3 * s) / 2) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 12 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 :
        deriv (fun (s : ℝ) => ((5 * s ^ 3 - 3 * s) / 2 : ℝ)) = fun (s : ℝ) => (15 * (s ^ 2) / 2 - (3 / 2) : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => ((5 * s ^ 3 - 3 * s) / 2 : ℝ)) ((15 * (s ^ 2) / 2 - (3 / 2) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 3).const_mul (5 / 2 : ℝ)).add
                      ((hasDerivAt_id' (s : ℝ)).const_mul (-(3 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 : deriv (fun (s : ℝ) => (15 * (s ^ 2) / 2 - (3 / 2) : ℝ)) = fun (s : ℝ) => (15 * s : ℝ) :=
        by
        funext s
        have hd : HasDerivAt (fun (s : ℝ) => (15 * (s ^ 2) / 2 - (3 / 2) : ℝ)) ((15 * s : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (15 / 2 : ℝ)).add
                      (hasDerivAt_const (s : ℝ) (-(3 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-- `P4` solves `(1 - x²)u'' - 2xu' + 20u = 0`, i.e. `((1 - x²)P4')' + 20P4 = 0`. -/
theorem legendre_equation_4 (u : ℝ → ℝ) (hu : u = fun s => (35 * s ^ 4 - 30 * s ^ 2 + 3) / 8) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 20 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 :
        deriv (fun (s : ℝ) => ((35 * s ^ 4 - 30 * s ^ 2 + 3) / 8 : ℝ)) = fun (s : ℝ) =>
          (35 * (s ^ 3) / 2 - (15 * s / 2) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => ((35 * s ^ 4 - 30 * s ^ 2 + 3) / 8 : ℝ)) ((35 * (s ^ 3) / 2 - (15 * s / 2) : ℝ))
            (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 4).const_mul (35 / 8 : ℝ)).add
                          (((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-(15 / 4) : ℝ))).add
                      (hasDerivAt_const (s : ℝ) (3 / 8 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (35 * (s ^ 3) / 2 - (15 * s / 2) : ℝ)) = fun (s : ℝ) =>
          (105 * (s ^ 2) / 2 - (15 / 2) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (35 * (s ^ 3) / 2 - (15 * s / 2) : ℝ)) ((105 * (s ^ 2) / 2 - (15 / 2) : ℝ)) (s) :=
          by
          refine
            ((((((hasDerivAt_id' (s : ℝ)).pow 3).const_mul (35 / 2 : ℝ)).add
                      ((hasDerivAt_id' (s : ℝ)).const_mul (-(15 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

set_option maxHeartbeats 1000000 in
/-- `P5` solves `(1 - x²)u'' - 2xu' + 30u = 0`, i.e. `((1 - x²)P5')' + 30P5 = 0`. -/
theorem legendre_equation_5 (u : ℝ → ℝ) (hu : u = fun s => (63 * s ^ 5 - 70 * s ^ 3 + 15 * s) / 8) (x : ℝ) :
    (1 - x ^ 2) * deriv (deriv u) x - 2 * x * deriv u x + 30 * u x = 0 := by
  veriphy_certified
    ( subst hu
      have d1 :
        deriv (fun (s : ℝ) => ((63 * s ^ 5 - 70 * s ^ 3 + 15 * s) / 8 : ℝ)) = fun (s : ℝ) =>
          (315 * (s ^ 4) / 8 - (105 * (s ^ 2) / 4) + (15 / 8) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => ((63 * s ^ 5 - 70 * s ^ 3 + 15 * s) / 8 : ℝ))
            ((315 * (s ^ 4) / 8 - (105 * (s ^ 2) / 4) + (15 / 8) : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 5).const_mul (63 / 8 : ℝ)).add
                          (((hasDerivAt_id' (s : ℝ)).pow 3).const_mul (-(35 / 4) : ℝ))).add
                      ((hasDerivAt_id' (s : ℝ)).const_mul (15 / 8 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      have d2 :
        deriv (fun (s : ℝ) => (315 * (s ^ 4) / 8 - (105 * (s ^ 2) / 4) + (15 / 8) : ℝ)) = fun (s : ℝ) =>
          (315 * (s ^ 3) / 2 - (105 * s / 2) : ℝ) :=
        by
        funext s
        have hd :
          HasDerivAt (fun (s : ℝ) => (315 * (s ^ 4) / 8 - (105 * (s ^ 2) / 4) + (15 / 8) : ℝ))
            ((315 * (s ^ 3) / 2 - (105 * s / 2) : ℝ)) (s) :=
          by
          refine
            (((((((hasDerivAt_id' (s : ℝ)).pow 4).const_mul (315 / 8 : ℝ)).add
                          (((hasDerivAt_id' (s : ℝ)).pow 2).const_mul (-(105 / 4) : ℝ))).add
                      (hasDerivAt_const (s : ℝ) (15 / 8 : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring_nf; done)
          · ring1
        exact hd.deriv
      simp only [d1, d2]
      all_goals (ring1))
  -- certificate: de-ode.sage

/-! ## prp-legendre-bessel, exm-orth-legendre, exm-plb-orthogonality-table, exr-plb-legendre-norm: `∫₋₁¹ PₘPₙ = 2/(2n+1) δₘₙ` for `m, n ≤ 4` -/

/-- `∫₋₁¹ P0P0 dx = 2 / 1`. -/
theorem leg00 : ∫ x in (-1 : ℝ)..1, (1 : ℝ) * 1 = 2 / 1 := by
  rw [show (2 : ℝ) / 1 = 2 by norm_num]
  simp only [mul_one]
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy : ℝ)) ((fun x => 1) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((hasDerivAt_id' (x_veriphy : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (x_veriphy : ℝ)) (f' := fun x => 1) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P0P1 dx = 0`. -/
theorem leg01 : ∫ x in (-1 : ℝ)..1, (1) * (x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) ((fun x => 1 * x) x_veriphy) x_veriphy :=
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
          fun x => 1 * x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P0P2 dx = 0`. -/
theorem leg02 : ∫ x in (-1 : ℝ)..1, (1) * ((3 * x ^ 2 - 1) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 2 - (x_veriphy / 2) : ℝ))
            ((fun x => 1 * ((3 * x ^ 2 - 1) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 2 : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 3) / 2 - (x_veriphy / 2) : ℝ)) (f' := fun x => 1 * ((3 * x ^ 2 - 1) / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P0P3 dx = 0`. -/
theorem leg03 : ∫ x in (-1 : ℝ)..1, (1) * ((5 * x ^ 3 - 3 * x) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (5 * (x_veriphy ^ 4) / 8 - (3 * (x_veriphy ^ 2) / 4) : ℝ))
            ((fun x => 1 * ((5 * x ^ 3 - 3 * x) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (5 / 8 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-(3 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (5 * (x_veriphy ^ 4) / 8 - (3 * (x_veriphy ^ 2) / 4) : ℝ)) (f' := fun x => 1 * ((5 * x ^ 3 - 3 * x) / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P0P4 dx = 0`. -/
theorem leg04 : ∫ x in (-1 : ℝ)..1, (1) * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (7 * (x_veriphy ^ 5) / 8 - (5 * (x_veriphy ^ 3) / 4) + (3 * x_veriphy / 8) : ℝ))
            ((fun x => 1 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (7 / 8 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(5 / 4) : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (7 * (x_veriphy ^ 5) / 8 - (5 * (x_veriphy ^ 3) / 4) + (3 * x_veriphy / 8) : ℝ)) (f' := fun x =>
          1 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P1P1 dx = 2 / 3`. -/
theorem leg11 : ∫ x in (-1 : ℝ)..1, (x) * (x) = 2 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) ((fun x => x * x) x_veriphy) x_veriphy :=
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
          fun x => x * x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P1P2 dx = 0`. -/
theorem leg12 : ∫ x in (-1 : ℝ)..1, (x) * ((3 * x ^ 2 - 1) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (((3 * (x_veriphy ^ 2) - 1) ^ 2) / 24 : ℝ))
            ((fun x => x * ((3 * x ^ 2 - 1) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (3 : ℝ)).add
                            (hasDerivAt_const (x_veriphy : ℝ) (-1 : ℝ))).pow
                        2).const_mul
                    (1 / 24 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((3 * (x_veriphy ^ 2) - 1) ^ 2) / 24 : ℝ)) (f' := fun x => x * ((3 * x ^ 2 - 1) / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P1P3 dx = 0`. -/
theorem leg13 : ∫ x in (-1 : ℝ)..1, (x) * ((5 * x ^ 3 - 3 * x) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 2 - ((x_veriphy ^ 3) / 2) : ℝ))
            ((fun x => x * ((5 * x ^ 3 - 3 * x) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 2 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 5) / 2 - ((x_veriphy ^ 3) / 2) : ℝ)) (f' := fun x => x * ((5 * x ^ 3 - 3 * x) / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P1P4 dx = 0`. -/
theorem leg14 : ∫ x in (-1 : ℝ)..1, (x) * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (35 * (x_veriphy ^ 6) / 48 - (15 * (x_veriphy ^ 4) / 16) + (3 * (x_veriphy ^ 2) / 16) : ℝ))
            ((fun x => x * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 6).const_mul (35 / 48 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (-(15 / 16) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (3 / 16 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (35 * (x_veriphy ^ 6) / 48 - (15 * (x_veriphy ^ 4) / 16) + (3 * (x_veriphy ^ 2) / 16) : ℝ)) (f' := fun x =>
          x * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P2P2 dx = 2 / 5`. -/
theorem leg22 : ∫ x in (-1 : ℝ)..1, ((3 * x ^ 2 - 1) / 2) * ((3 * x ^ 2 - 1) / 2) = 2 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (9 * (x_veriphy ^ 5) / 20 - ((x_veriphy ^ 3) / 2) + (x_veriphy / 4) : ℝ))
            ((fun x => (3 * x ^ 2 - 1) / 2 * ((3 * x ^ 2 - 1) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (9 / 20 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(1 / 2) : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (9 * (x_veriphy ^ 5) / 20 - ((x_veriphy ^ 3) / 2) + (x_veriphy / 4) : ℝ)) (f' := fun x =>
          (3 * x ^ 2 - 1) / 2 * ((3 * x ^ 2 - 1) / 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P2P3 dx = 0`. -/
theorem leg23 : ∫ x in (-1 : ℝ)..1, ((3 * x ^ 2 - 1) / 2) * ((5 * x ^ 3 - 3 * x) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy => (5 * (x_veriphy ^ 6) / 8 - (7 * (x_veriphy ^ 4) / 8) + (3 * (x_veriphy ^ 2) / 8) : ℝ))
            ((fun x => (3 * x ^ 2 - 1) / 2 * ((5 * x ^ 3 - 3 * x) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 6).const_mul (5 / 8 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (-(7 / 8) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (3 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (5 * (x_veriphy ^ 6) / 8 - (7 * (x_veriphy ^ 4) / 8) + (3 * (x_veriphy ^ 2) / 8) : ℝ)) (f' := fun x =>
          (3 * x ^ 2 - 1) / 2 * ((5 * x ^ 3 - 3 * x) / 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P2P4 dx = 0`. -/
theorem leg24 : ∫ x in (-1 : ℝ)..1, ((3 * x ^ 2 - 1) / 2) * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (15 * (x_veriphy ^ 7) / 16 - (25 * (x_veriphy ^ 5) / 16) + (13 * (x_veriphy ^ 3) / 16) -
                  (3 * x_veriphy / 16) :
                ℝ))
            ((fun x => (3 * x ^ 2 - 1) / 2 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 7).const_mul (15 / 16 : ℝ)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (-(25 / 16) : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (13 / 16 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-(3 / 16) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (15 * (x_veriphy ^ 7) / 16 - (25 * (x_veriphy ^ 5) / 16) + (13 * (x_veriphy ^ 3) / 16) -
              (3 * x_veriphy / 16) :
            ℝ))
          (f' := fun x => (3 * x ^ 2 - 1) / 2 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P3P3 dx = 2 / 7`. -/
theorem leg33 : ∫ x in (-1 : ℝ)..1, ((5 * x ^ 3 - 3 * x) / 2) * ((5 * x ^ 3 - 3 * x) / 2) = 2 / 7 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy => (25 * (x_veriphy ^ 7) / 28 - (3 * (x_veriphy ^ 5) / 2) + (3 * (x_veriphy ^ 3) / 4) : ℝ))
            ((fun x => (5 * x ^ 3 - 3 * x) / 2 * ((5 * x ^ 3 - 3 * x) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 7).const_mul (25 / 28 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (-(3 / 2) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (3 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (25 * (x_veriphy ^ 7) / 28 - (3 * (x_veriphy ^ 5) / 2) + (3 * (x_veriphy ^ 3) / 4) : ℝ)) (f' := fun x =>
          (5 * x ^ 3 - 3 * x) / 2 * ((5 * x ^ 3 - 3 * x) / 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P3P4 dx = 0`. -/
theorem leg34 : ∫ x in (-1 : ℝ)..1, ((5 * x ^ 3 - 3 * x) / 2) * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (175 * (x_veriphy ^ 8) / 128 - (85 * (x_veriphy ^ 6) / 32) + (105 * (x_veriphy ^ 4) / 64) -
                  (9 * (x_veriphy ^ 2) / 32) :
                ℝ))
            ((fun x => (5 * x ^ 3 - 3 * x) / 2 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 8).const_mul (175 / 128 : ℝ)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 6).const_mul (-(85 / 32) : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (105 / 64 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-(9 / 32) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (175 * (x_veriphy ^ 8) / 128 - (85 * (x_veriphy ^ 6) / 32) + (105 * (x_veriphy ^ 4) / 64) -
              (9 * (x_veriphy ^ 2) / 32) :
            ℝ))
          (f' := fun x => (5 * x ^ 3 - 3 * x) / 2 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-- `∫₋₁¹ P4P4 dx = 2 / 9`. -/
theorem leg44 : ∫ x in (-1 : ℝ)..1, ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8) = 2 / 9 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (1225 * (x_veriphy ^ 9) / 576 - (75 * (x_veriphy ^ 7) / 16) + (111 * (x_veriphy ^ 5) / 32) -
                    (15 * (x_veriphy ^ 3) / 16) +
                  (9 * x_veriphy / 64) :
                ℝ))
            ((fun x => (35 * x ^ 4 - 30 * x ^ 2 + 3) / 8 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 9).const_mul (1225 / 576 : ℝ)).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).pow 7).const_mul (-(75 / 16) : ℝ))).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (111 / 32 : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(15 / 16) : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (9 / 64 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (1225 * (x_veriphy ^ 9) / 576 - (75 * (x_veriphy ^ 7) / 16) + (111 * (x_veriphy ^ 5) / 32) -
                (15 * (x_veriphy ^ 3) / 16) +
              (9 * x_veriphy / 64) :
            ℝ))
          (f' := fun x => (35 * x ^ 4 - 30 * x ^ 2 + 3) / 8 * ((35 * x ^ 4 - 30 * x ^ 2 + 3) / 8)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage

/-! ## exr-plb-legendre-norm: `∫₋₁¹ xP₂ dx = 0` (the integrand is odd) -/

/-- `∫₋₁¹ x P₂ dx = 0`. -/
theorem x_P2 : ∫ x in (-1 : ℝ)..1, x * ((3 * x ^ 2 - 1) / 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-1 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => (((3 * (x_veriphy ^ 2) - 1) ^ 2) / 24 : ℝ))
            ((fun x => x * ((3 * x ^ 2 - 1) / 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (3 : ℝ)).add
                            (hasDerivAt_const (x_veriphy : ℝ) (-1 : ℝ))).pow
                        2).const_mul
                    (1 / 24 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((3 * (x_veriphy ^ 2) - 1) ^ 2) / 24 : ℝ)) (f' := fun x => x * ((3 * x ^ 2 - 1) / 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: de-integral.sage


end KaluzaKlein.DifferentialEquations.SturmLiouvilleComputations

import Mathlib

/-!
# Classical mechanics: displayed computations

The harmonic oscillator `x(t) = A cos (ω t)` solves its Euler-Lagrange equation, its energy is conserved, the
action of a one-parameter family of varied paths is stationary at the classical path (a polynomial identity in
the variation), Hamilton's equations hold on the oscillator solution, the basic Poisson brackets, and the
conservation of angular momentum for a central force as an identity in the jet variables.
-/

namespace KaluzaKlein.Mechanics

/-- The oscillator solution `x(t) = A cos (ω t)`. -/
noncomputable def osc (A ω t : ℝ) : ℝ := A * Real.cos (ω * t)

/-- `ẋ = −A ω sin (ω t)`. -/
theorem osc_deriv (A ω t : ℝ) : deriv (osc A ω) t = -(A * ω * Real.sin (ω * t)) := by
  unfold osc
  have h : HasDerivAt (fun t => A * Real.cos (ω * t)) (A * (-(Real.sin (ω * t)) * ω)) t := by
    have := ((hasDerivAt_id t).const_mul ω).cos
    simpa using this.const_mul A
  rw [h.deriv]; ring

/-- The Euler-Lagrange equation `ẍ + ω² x = 0` holds for `x = A cos (ω t)`. -/
theorem osc_euler_lagrange (A ω t : ℝ) : deriv (deriv (osc A ω)) t + ω ^ 2 * osc A ω t = 0 := by
  have hd : deriv (osc A ω) = fun t => -(A * ω * Real.sin (ω * t)) := funext (osc_deriv A ω)
  rw [hd]
  have h : HasDerivAt (fun t => -(A * ω) * Real.sin (ω * t)) (-(A * ω) * (Real.cos (ω * t) * ω)) t := by
    have := ((hasDerivAt_id t).const_mul ω).sin
    simpa using this.const_mul (-(A * ω))
  have hf : (fun t => -(A * ω * Real.sin (ω * t))) = fun t => -(A * ω) * Real.sin (ω * t) := by ext; ring
  rw [hf, h.deriv]; unfold osc; ring

/-- The energy `½ ẋ² + ½ ω² x²` of the oscillator solution is the constant `½ A² ω²`. -/
theorem osc_energy (A ω t : ℝ) :
    (1 / 2) * (deriv (osc A ω) t) ^ 2 + (1 / 2) * ω ^ 2 * (osc A ω t) ^ 2 = (1 / 2) * A ^ 2 * ω ^ 2 := by
  rw [osc_deriv]; unfold osc
  have := Real.sin_sq_add_cos_sq (ω * t)
  linear_combination (1 / 2) * A ^ 2 * ω ^ 2 * this

/-- Stationary action for the free particle: for the varied path `x(t) = v t + ε η(t)` with `η(0) = η(T) = 0`,
the action `∫₀ᵀ ½ ẋ² dt = ½ v² T + ε v (η(T) − η(0)) + ε² ∫ ½ η̇²` has no term linear in `ε`.
Stated as the vanishing of the linear coefficient, with the integral of `η̇` evaluated by the boundary values. -/
theorem free_action_stationary (v T ε : ℝ) (η : ℝ → ℝ) (hη : ∀ t, HasDerivAt η (deriv η t) t)
    (hint : IntervalIntegrable (deriv η) MeasureTheory.volume 0 T) (h0 : η 0 = 0) (hT : η T = 0) :
    ∫ t in (0 : ℝ)..T, v * (ε * deriv η t) = 0 := by
  have : ∫ t in (0 : ℝ)..T, v * (ε * deriv η t) = v * ε * ∫ t in (0 : ℝ)..T, deriv η t := by
    rw [← intervalIntegral.integral_const_mul]; congr 1; ext t; ring
  rw [this, intervalIntegral.integral_deriv_eq_sub (fun t _ => (hη t).differentiableAt) hint, hT, h0]; ring

/-- Hamilton's equations for the oscillator: `q̇ = p` and `ṗ = −ω² q` on `q = A cos (ω t)`, `p = q̇`. -/
theorem osc_hamilton (A ω t : ℝ) :
    deriv (osc A ω) t = deriv (osc A ω) t ∧ deriv (deriv (osc A ω)) t = -(ω ^ 2 * osc A ω t) := by
  refine ⟨rfl, ?_⟩
  have := osc_euler_lagrange A ω t
  linarith

/-- The Poisson bracket of two functions of `(q, p)`, `{f, g} = ∂_q f ∂_p g − ∂_p f ∂_q g`, as a formula in the
partial derivatives (jet variables). -/
def poisson (fq fp gq gp : ℝ) : ℝ := fq * gp - fp * gq

/-- `{q, p} = 1`, `{p, q} = −1`, `{q, q} = 0`. -/
theorem poisson_canonical : poisson 1 0 0 1 = 1 ∧ poisson 0 1 1 0 = -1 ∧ poisson 1 0 1 0 = 0 := by
  simp [poisson]

/-- The Poisson bracket is antisymmetric. -/
theorem poisson_antisymm (fq fp gq gp : ℝ) : poisson gq gp fq fp = -poisson fq fp gq gp := by
  simp [poisson]; ring

/-- Angular momentum `L = x p_y − y p_x` is conserved for a central force: with `ẋ = p_x`, `ẏ = p_y`,
`ṗ_x = −x f`, `ṗ_y = −y f` (force along the radius), `dL/dt = ẋ p_y + x ṗ_y − ẏ p_x − y ṗ_x = 0`. -/
theorem angular_momentum_conserved (x y px py f : ℝ) :
    px * py + x * (-(y * f)) - py * px - y * (-(x * f)) = 0 := by ring

end KaluzaKlein.Mechanics

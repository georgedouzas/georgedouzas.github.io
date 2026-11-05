import Mathlib
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaces

/-!
# Classical mechanics, section "Configuration space and the Lagrangian"

The claims of the section "Configuration space and the Lagrangian" of `tutorial/part-ii-01-classical-mechanics.qmd`.
Degrees of freedom are dimensions of explicit spaces (`Fin n → ℝ`, the antisymmetric `3 × 3` matrices for `SO(3)`);
constrained configuration spaces are parametrised by generalised coordinates (the circle and the sphere by angles,
`SO(3)` by the Euler-angle product of rotations); Lagrangians in generalised coordinates are identities in the jet
variables (coordinates, velocities, `cos`, `sin` with `cos² + sin² = 1`).
-/

open Real Matrix Module

namespace KaluzaKlein.ClassicalMechanics.ConfigurationSpace

/-! ## exm-configuration-particle -/

/-- A particle in `ℝ³` has three degrees of freedom. -/
theorem particle_dof : finrank ℝ (Fin 3 → ℝ) = 3 := by
  VERIPHY[cm-dim]

/-- `N` particles have `Q = ℝ^{3N}`, with `3N` degrees of freedom. -/
theorem particles_dof (N : ℕ) : finrank ℝ (Fin (3 * N) → ℝ) = 3 * N := by
  simp [Module.finrank_fin_fun]
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.ConfigurationSpace.particles_dof, cm-particles-dof-test]

/-! ## exm-configuration-pendulum -/

/-- The pendulum's configuration space is the circle, parametrised by the angle: `(l cos q, l sin q)` lies on the
circle of radius `l`. -/
theorem pendulum_circle (l q : ℝ) : (l * Real.cos q) ^ 2 + (l * Real.sin q) ^ 2 = l ^ 2 := by
  VERIPHY[cm-rewrite]

/-- The particle on the sphere is parametrised by `(θ, φ)`: `(sin θ cos φ, sin θ sin φ, cos θ)` lies on the unit
sphere (with `c² + s² = 1` for both angles). -/
theorem sphere_coordinates (cθ sθ cφ sφ : ℝ) (h1 : cθ ^ 2 + sθ ^ 2 = 1) (h2 : cφ ^ 2 + sφ ^ 2 = 1) :
    (sθ * cφ) ^ 2 + (sθ * sφ) ^ 2 + cθ ^ 2 = 1 := by
  VERIPHY[cm-poly]

/-! ## exm-configuration-rigid-body -/

/-- `SO(3)` has three degrees of freedom: its Lie algebra, the antisymmetric `3 × 3` matrices, has dimension `3`. -/
theorem rigid_body_dof : finrank ℝ (KaluzaKlein.LinearAlgebra.VectorSpaces.antisymMatrices 3) = 3 := by
  VERIPHY[cm-dim]

/-- A free rigid body, `Q = ℝ³ × SO(3)`, has `3 + 3 = 6` degrees of freedom. -/
theorem free_rigid_body_dof : finrank ℝ (Fin 3 → ℝ) + finrank ℝ (KaluzaKlein.LinearAlgebra.VectorSpaces.antisymMatrices 3) = 6 := by
  VERIPHY[cm-dim]

/-! ## exr-clm-config-count -/

/-- The double pendulum (`Q = S¹ × S¹`) and the particle on the torus have `1 + 1 = 2` degrees of freedom. -/
theorem torus_dof : (1 + 1 : ℝ) = 2 := by
  VERIPHY[cm-poly]

/-- The rod: `6` coordinates minus one constraint, `6 - 1 = 5` degrees of freedom (`Q = ℝ³ × S²`). -/
theorem rod_dof : (6 - 1 : ℝ) = 5 := by
  VERIPHY[cm-poly]

/-! ## exr-clm-config-rod -/

/-- `x₁ = X + (m₂ d/M) n` with `X` the centre of mass and `n = (x₁ - x₂)/d` (one component). -/
theorem rod_position_one (m₁ m₂ d x₁ x₂ : ℝ) (hM : m₁ + m₂ ≠ 0) (hd : d ≠ 0) :
    x₁ = (m₁ * x₁ + m₂ * x₂) / (m₁ + m₂) + m₂ * d / (m₁ + m₂) * ((x₁ - x₂) / d) := by
  VERIPHY[cm-poly]

/-- `x₂ = X - (m₁ d/M) n`. -/
theorem rod_position_two (m₁ m₂ d x₁ x₂ : ℝ) (hM : m₁ + m₂ ≠ 0) (hd : d ≠ 0) :
    x₂ = (m₁ * x₁ + m₂ * x₂) / (m₁ + m₂) - m₁ * d / (m₁ + m₂) * ((x₁ - x₂) / d) := by
  VERIPHY[cm-poly]

/-- `n = (x₁ - x₂)/d` is a unit vector when `|x₁ - x₂| = d`: `(x₁ - x₂)²/d² = 1` (so `(X, n) ∈ ℝ³ × S²`). -/
theorem rod_direction_unit (d x₁ x₂ : ℝ) (hd : d ≠ 0) (h : (x₁ - x₂) ^ 2 = d ^ 2) : ((x₁ - x₂) / d) ^ 2 = 1 := by
  rw [div_pow, h]
  VERIPHY[cm-poly]

/-- The kinetic energy `m₁ ẋ₁²/2 + m₂ ẋ₂²/2 = M Ẋ²/2 + μ d² ṅ²/2` (`μ = m₁m₂/M`; the cross terms cancel). -/
theorem rod_kinetic (m₁ m₂ d Ẋ ṅ : ℝ) (hM : m₁ + m₂ ≠ 0) :
    m₁ / 2 * (Ẋ + m₂ * d / (m₁ + m₂) * ṅ) ^ 2 + m₂ / 2 * (Ẋ - m₁ * d / (m₁ + m₂) * ṅ) ^ 2 =
      (m₁ + m₂) / 2 * Ẋ ^ 2 + m₁ * m₂ / (m₁ + m₂) / 2 * d ^ 2 * ṅ ^ 2 := by
  VERIPHY[cm-poly]

/-! ## exr-clm-config-rotations -/

/-- The rotation about the `z`-axis. -/
noncomputable def Rz (α : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos α, -Real.sin α, 0; Real.sin α, Real.cos α, 0; 0, 0, 1]

/-- The rotation about the `x`-axis. -/
noncomputable def Rx (α : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1, 0, 0; 0, Real.cos α, -Real.sin α; 0, Real.sin α, Real.cos α]

/-- `R = R_z(φ) R_x(θ) R_z(ψ)` is orthogonal: `Rᵀ R = 1`. -/
theorem euler_orthogonal (φ θ ψ : ℝ) : (Rz φ * Rx θ * Rz ψ)ᵀ * (Rz φ * Rx θ * Rz ψ) = 1 := by
  VERIPHY[cm-matrix-euler]

/-- `det R = 1`. -/
theorem euler_det (φ θ ψ : ℝ) : (Rz φ * Rx θ * Rz ψ).det = 1 := by
  VERIPHY[cm-matrix-euler]

/-- At `θ = 0`, `R_x(0) = 1` … -/
theorem Rx_zero : Rx 0 = 1 := by
  VERIPHY[cm-matrix-values]

/-- … so `R = R_z(φ) R_z(ψ) = R_z(φ + ψ)` depends on the sum `φ + ψ` only: the chart degenerates. -/
theorem Rz_add (φ ψ : ℝ) : Rz φ * Rz ψ = Rz (φ + ψ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Rz, Matrix.mul_apply, Fin.sum_univ_succ, Real.cos_add, Real.sin_add] <;> ring
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.ConfigurationSpace.Rz_add, cm-Rz-add-test]

/-- For `θ = 0` and `ψ = -φ`, `R = R_z(φ) R_z(-φ) = 1`. -/
theorem euler_identity (φ : ℝ) : Rz φ * Rz (-φ) = 1 := by
  VERIPHY[cm-matrix-euler]

/-! ## exm-lagrangian-oscillator -/

/-- `L = m q̇²/2 - m ω² q²/2` is the kinetic energy minus the potential `k q²/2` of a spring with `k = m ω²`. -/
theorem oscillator_lagrangian (m ω k q v : ℝ) (hk : k = m * ω ^ 2) :
    m * v ^ 2 / 2 - m * ω ^ 2 * q ^ 2 / 2 = m * v ^ 2 / 2 - k * q ^ 2 / 2 := by
  VERIPHY[cm-poly]

/-! ## exm-lagrangian-pendulum -/

/-- The kinetic energy of a point on a circle of radius `l`, `(l cos q, l sin q)`, is `m l² q̇²/2`. -/
theorem pendulum_kinetic (m l c s q̇ : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    m / 2 * ((-(l * s) * q̇) ^ 2 + (l * c * q̇) ^ 2) = m * l ^ 2 * q̇ ^ 2 / 2 := by
  VERIPHY[cm-poly]

/-- The potential of its height `z = -l cos q` is `m g z = -m g l cos q`. -/
theorem pendulum_potential (m g l c : ℝ) : m * g * (-(l * c)) = -(m * g * l * c) := by
  VERIPHY[cm-poly]

/-! ## exm-lagrangian-sphere -/

/-- The squared velocity of `(sin θ cos φ, sin θ sin φ, cos θ)` is `θ̇² + sin² θ φ̇²`: the round metric, and
`L = (θ̇² + sin² θ φ̇²)/2` is half the squared length of the velocity. -/
theorem sphere_velocity_sq (cθ sθ cφ sφ θ̇ φ̇ : ℝ) (h1 : cθ ^ 2 + sθ ^ 2 = 1) (h2 : cφ ^ 2 + sφ ^ 2 = 1) :
    (cθ * cφ * θ̇ - sθ * sφ * φ̇) ^ 2 + (cθ * sφ * θ̇ + sθ * cφ * φ̇) ^ 2 + (-(sθ * θ̇)) ^ 2 = θ̇ ^ 2 + sθ ^ 2 * φ̇ ^ 2 := by
  VERIPHY[cm-poly]

/-! ## exr-clm-lagrangian-charged -/

/-- `∇ × A` for `A = (B/2)(-y, x, 0)`: `∂_x A_y = B/2` … -/
theorem vector_potential_dx (B x : ℝ) : deriv (fun x => B * x / 2) x = B / 2 := by
  VERIPHY[cm-derivative]

/-- … `∂_y A_x = -B/2` … -/
theorem vector_potential_dy (B y : ℝ) : deriv (fun y => -(B * y) / 2) y = -(B / 2) := by
  VERIPHY[cm-derivative]

/-- … so `(∇ × A)_z = B/2 + B/2 = B`. -/
theorem vector_potential_curl (B : ℝ) : B / 2 - (-(B / 2)) = B := by
  VERIPHY[cm-poly]

/-- `L = m|ẋ|²/2 + q ẋ · A = m(ẋ² + ẏ² + ż²)/2 + (qB/2)(x ẏ - y ẋ)`. -/
theorem magnetic_lagrangian (m q B x y vx vy vz : ℝ) :
    m / 2 * (vx ^ 2 + vy ^ 2 + vz ^ 2) + q * (vx * (-(B * y) / 2) + vy * (B * x / 2) + vz * 0) =
      m / 2 * (vx ^ 2 + vy ^ 2 + vz ^ 2) + q * B / 2 * (x * vy - y * vx) := by
  VERIPHY[cm-poly]

/-- The magnetic term is linear in the velocities: it is not a function `-V(x)` of the position alone, so the
Lagrangian is not kinetic minus potential. -/
theorem magnetic_not_potential (q B : ℝ) (h : q * B ≠ 0) :
    ¬ ∃ V : ℝ → ℝ → ℝ, ∀ x y vx vy : ℝ, q * B / 2 * (x * vy - y * vx) = -V x y := by
  rintro ⟨V, hV⟩
  have h0 := hV 1 0 0 0
  have h1 := hV 1 0 0 1
  simp only [mul_zero, mul_one, sub_zero, zero_mul, sub_self] at h0 h1
  exact h (by linarith)
VERIPHY_TEST[KaluzaKlein.ClassicalMechanics.ConfigurationSpace.magnetic_not_potential, cm-magnetic-not-potential-test]

/-! ## exr-clm-lagrangian-double-pendulum -/

/-- The kinetic energy of the double pendulum, from `x₁ = l₁ sin θ₁`, `y₁ = -l₁ cos θ₁`, `x₂ = x₁ + l₂ sin θ₂`,
`y₂ = y₁ - l₂ cos θ₂`: `(m₁ + m₂) l₁² θ̇₁²/2 + m₂ l₂² θ̇₂²/2 + m₂ l₁ l₂ cos (θ₁ - θ₂) θ̇₁ θ̇₂`, with
`cos (θ₁ - θ₂) = c₁ c₂ + s₁ s₂`: the metric has the configuration-dependent off-diagonal entry
`m₂ l₁ l₂ cos (θ₁ - θ₂)`. -/
theorem double_pendulum_kinetic (m₁ m₂ l₁ l₂ c₁ s₁ c₂ s₂ w₁ w₂ : ℝ) (h1 : c₁ ^ 2 + s₁ ^ 2 = 1) (h2 : c₂ ^ 2 + s₂ ^ 2 = 1) :
    m₁ / 2 * ((l₁ * c₁ * w₁) ^ 2 + (l₁ * s₁ * w₁) ^ 2) +
      m₂ / 2 * ((l₁ * c₁ * w₁ + l₂ * c₂ * w₂) ^ 2 + (l₁ * s₁ * w₁ + l₂ * s₂ * w₂) ^ 2) =
      (m₁ + m₂) / 2 * l₁ ^ 2 * w₁ ^ 2 + m₂ / 2 * l₂ ^ 2 * w₂ ^ 2 + m₂ * l₁ * l₂ * (c₁ * c₂ + s₁ * s₂) * w₁ * w₂ := by
  VERIPHY[cm-poly]

/-- The potential energy `m₁ g y₁ + m₂ g y₂ = -(m₁ + m₂) g l₁ cos θ₁ - m₂ g l₂ cos θ₂`, so
`-V = (m₁ + m₂) g l₁ cos θ₁ + m₂ g l₂ cos θ₂`. -/
theorem double_pendulum_potential (m₁ m₂ g l₁ l₂ c₁ c₂ : ℝ) :
    -(m₁ * g * (-(l₁ * c₁)) + m₂ * g * (-(l₁ * c₁) - l₂ * c₂)) = (m₁ + m₂) * g * l₁ * c₁ + m₂ * g * l₂ * c₂ := by
  VERIPHY[cm-poly]

end KaluzaKlein.ClassicalMechanics.ConfigurationSpace

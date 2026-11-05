import Mathlib

/-!
# Fields: displayed computations

A plane wave solves the Klein-Gordon equation exactly when its frequency satisfies the dispersion relation, the
phase rotation preserves the modulus of a complex field, and the explicit Dirac gamma matrices satisfy the
Clifford relations `{γ^μ, γ^ν} = 2 η^{μν}` with `η = diag(1, −1, −1, −1)` (signature `(+, −, −, −)`).
-/

namespace KaluzaKlein.Fields

/-- The plane wave `cos (k x − ω t)` in one space dimension. -/
noncomputable def planeWave (k ω : ℝ) (t x : ℝ) : ℝ := Real.cos (k * x - ω * t)

/-- `∂²_t cos (k x − ω t) = −ω² cos (k x − ω t)`. -/
theorem planeWave_tt (k ω t x : ℝ) :
    deriv (deriv (fun t => planeWave k ω t x)) t = -ω ^ 2 * planeWave k ω t x := by
  have h1 : ∀ s, HasDerivAt (fun t => planeWave k ω t x) (-(Real.sin (k * x - ω * s)) * (0 - ω)) s := by
    intro s
    have := ((hasDerivAt_const s (k * x)).sub ((hasDerivAt_id s).const_mul ω)).cos
    simpa [planeWave] using this
  have hd : deriv (fun t => planeWave k ω t x) = fun s => -(Real.sin (k * x - ω * s)) * (0 - ω) := funext fun s => (h1 s).deriv
  rw [hd]
  have h2 : HasDerivAt (fun s => -(Real.sin (k * x - ω * s)) * (0 - ω)) (-(Real.cos (k * x - ω * t) * (0 - ω)) * (0 - ω)) t := by
    have := ((hasDerivAt_const t (k * x)).sub ((hasDerivAt_id t).const_mul ω)).sin
    simpa using (this.neg).mul_const (0 - ω)
  rw [h2.deriv]; unfold planeWave; ring

/-- `∂²_x cos (k x − ω t) = −k² cos (k x − ω t)`. -/
theorem planeWave_xx (k ω t x : ℝ) :
    deriv (deriv (fun x => planeWave k ω t x)) x = -k ^ 2 * planeWave k ω t x := by
  have h1 : ∀ s, HasDerivAt (fun x => planeWave k ω t x) (-(Real.sin (k * s - ω * t)) * (k - 0)) s := by
    intro s
    have := (((hasDerivAt_id s).const_mul k).sub (hasDerivAt_const s (ω * t))).cos
    simpa [planeWave] using this
  have hd : deriv (fun x => planeWave k ω t x) = fun s => -(Real.sin (k * s - ω * t)) * (k - 0) := funext fun s => (h1 s).deriv
  rw [hd]
  have h2 : HasDerivAt (fun s => -(Real.sin (k * s - ω * t)) * (k - 0)) (-(Real.cos (k * x - ω * t) * (k - 0)) * (k - 0)) x := by
    have := (((hasDerivAt_id x).const_mul k).sub (hasDerivAt_const x (ω * t))).sin
    simpa using (this.neg).mul_const (k - 0)
  rw [h2.deriv]; unfold planeWave; ring

/-- The Klein-Gordon operator `□ + m² = ∂²_t − ∂²_x + m²` (signature `(+, −)`) annihilates the plane wave exactly
when `ω² = k² + m²`. -/
theorem planeWave_klein_gordon (k ω m t x : ℝ) (h : ω ^ 2 = k ^ 2 + m ^ 2) :
    deriv (deriv (fun t => planeWave k ω t x)) t - deriv (deriv (fun x => planeWave k ω t x)) x
      + m ^ 2 * planeWave k ω t x = 0 := by
  rw [planeWave_tt, planeWave_xx]; rw [h]; ring

/-- The phase rotation `φ ↦ e^{iα} φ` preserves `|φ|²`. -/
theorem phase_rotation_normSq (φ : ℂ) (α : ℝ) :
    Complex.normSq (Complex.exp (α * Complex.I) * φ) = Complex.normSq φ := by
  rw [Complex.normSq_mul, Complex.normSq_eq_norm_sq, Complex.norm_exp_ofReal_mul_I]; norm_num

/-- The Dirac gamma matrices in the standard Weyl (chiral) representation, `γ^0 = [[0, 1], [1, 0]]` and
`γ^i = [[0, σ_i], [−σ_i, 0]]` in `2 × 2` blocks, that is `γ^μ = [[0, σ^μ], [σ̄^μ, 0]]` with `σ^μ = (1, σ_i)` and
`σ̄^μ = (1, −σ_i)`, so that `{γ^μ, γ^ν} = 2 η^{μν}` with the mostly-minus metric `η = diag(1, −1, −1, −1)`,
`(γ^0)² = 1`, `(γ^i)² = −1`. -/
noncomputable def gamma : Fin 4 → Matrix (Fin 4) (Fin 4) ℂ :=
  ![!![0, 0, 1, 0; 0, 0, 0, 1; 1, 0, 0, 0; 0, 1, 0, 0],
    !![0, 0, 0, 1; 0, 0, 1, 0; 0, -1, 0, 0; -1, 0, 0, 0],
    !![0, 0, 0, -Complex.I; 0, 0, Complex.I, 0; 0, Complex.I, 0, 0; -Complex.I, 0, 0, 0],
    !![0, 0, 1, 0; 0, 0, 0, -1; -1, 0, 0, 0; 0, 1, 0, 0]]

/-- The Minkowski metric `η = diag(1, −1, −1, −1)` as a matrix of complex numbers. -/
def etaC : Fin 4 → Fin 4 → ℂ := fun μ ν => if μ = ν then (if μ = 0 then 1 else -1) else 0

set_option maxHeartbeats 4000000 in
/-- The Clifford relations `γ^μ γ^ν + γ^ν γ^μ = 2 η^{μν} 1`, checked entry by entry. -/
theorem clifford (μ ν : Fin 4) : gamma μ * gamma ν + gamma ν * gamma μ = (2 * etaC μ ν) • (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
  fin_cases μ <;> fin_cases ν <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [gamma, etaC, Matrix.mul_apply, Fin.sum_univ_four, Matrix.one_apply, Complex.ext_iff] <;> norm_num

end KaluzaKlein.Fields

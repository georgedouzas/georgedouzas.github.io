import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.Metrics

/-!
# Christoffel symbols of the running examples

Every component `Γᵏᵢⱼ` with `i ≤ j` (the symbols are symmetric, `christoffel_symm`). Each proof is the Sage-free
replacement of a `coordinate_geometry` certificate: SageManifolds computes the symbol, Lean proves it from the
definition (inverse metric, partial derivatives, `field_simp`, `ring`). The chart hypothesis `Real.sin (x 0) ≠ 0`
excludes the coordinate singularities.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry

/-! ## The circle: a constant metric has vanishing symbols -/

/-- `Γ⁰₀₀ = 0` on the circle. -/
theorem circle_christoffel (x : Fin 1 → ℝ) : christoffel circle 0 0 0 x = 0 := by
  have : circle = fun i j _ => (1 : Matrix (Fin 1) (Fin 1) ℝ) i j := by
    ext i j y; fin_cases i; fin_cases j; simp [circle]
  rw [this, christoffel_const]

/-! ## The sphere -/

/-- `Γ^θ_θθ = 0`. -/
theorem sphere_christoffel_0_0_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 0 0 0 x = 0 := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_christoffel.sage

/-- `Γ^θ_θφ = 0`. -/
theorem sphere_christoffel_0_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 0 0 1 x = 0 := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_christoffel.sage

/-- `Γ^θ_φφ = −sin θ cos θ`. -/
theorem sphere_christoffel_0_1_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 0 1 1 x = -(Real.sin (x 0) * Real.cos (x 0)) := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_christoffel.sage

/-- `Γ^φ_θθ = 0`. -/
theorem sphere_christoffel_1_0_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 1 0 0 x = 0 := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_christoffel.sage

/-- `Γ^φ_θφ = cos θ / sin θ`. -/
theorem sphere_christoffel_1_0_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 1 0 1 x = Real.cos (x 0) / Real.sin (x 0) := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals (field_simp; done)
  -- certificate: sphere_christoffel.sage

/-- `Γ^φ_φφ = 0`. -/
theorem sphere_christoffel_1_1_1 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 1 1 1 x = 0 := by
  have hinv :
    CoordinateGeometry.invMetric KaluzaKlein.Geometry.sphere x =
      !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / (Real.sin (x 0)) ^ 2 : ℝ)] :=
    by
    refine Matrix.inv_eq_right_inv ?_
    ext a b
    fin_cases a <;> fin_cases b <;>
        simp [KaluzaKlein.Geometry.sphere, CoordinateGeometry.metricAt, Matrix.mul_apply, Fin.sum_univ_succ] <;>
      field_simp
  rw [CoordinateGeometry.christoffel, hinv]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.sphere, Fin.sum_univ_succ,
    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, Real.deriv_sin, Real.deriv_cos,
    Real.deriv_exp]
  all_goals ring1
  -- certificate: sphere_christoffel.sage

/-! ## `SU(2)` in Euler angles `(θ, φ, ψ)` -/

/-- `Γ^θ_θθ = 0`. -/
theorem su2_christoffel_0_0_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 0 0 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^θ_θφ = 0`. -/
theorem su2_christoffel_0_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 0 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^θ_θψ = 0`. -/
theorem su2_christoffel_0_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 0 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^θ_φφ = 0`. -/
theorem su2_christoffel_0_1_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 1 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^θ_φψ = sin θ / 2`. -/
theorem su2_christoffel_0_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 1 2 x = Real.sin (x 0) / 2 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^θ_ψψ = 0`. -/
theorem su2_christoffel_0_2_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 0 2 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_θθ = 0`. -/
theorem su2_christoffel_1_0_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 0 0 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_θφ = cos θ / (2 sin θ)`. -/
theorem su2_christoffel_1_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 0 1 x = Real.cos (x 0) / (2 * Real.sin (x 0)) := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals (field_simp; done)
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_θψ = −1 / (2 sin θ)`. -/
theorem su2_christoffel_1_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 0 2 x = -1 / (2 * Real.sin (x 0)) := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals (field_simp; done)
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_φφ = 0`. -/
theorem su2_christoffel_1_1_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 1 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_φψ = 0`. -/
theorem su2_christoffel_1_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 1 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^φ_ψψ = 0`. -/
theorem su2_christoffel_1_2_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 2 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_θθ = 0`. -/
theorem su2_christoffel_2_0_0 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 0 0 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_θφ = −1 / (2 sin θ)`. -/
theorem su2_christoffel_2_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 0 1 x = -1 / (2 * Real.sin (x 0)) := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals (field_simp; done)
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_θψ = cos θ / (2 sin θ)`. -/
theorem su2_christoffel_2_0_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 0 2 x = Real.cos (x 0) / (2 * Real.sin (x 0)) := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals (field_simp; done)
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_φφ = 0`. -/
theorem su2_christoffel_2_1_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 1 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_φψ = 0`. -/
theorem su2_christoffel_2_1_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 1 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

/-- `Γ^ψ_ψψ = 0`. -/
theorem su2_christoffel_2_2_2 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 2 2 2 x = 0 := by
  rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,
    deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]
  all_goals ring1
  -- certificate: su2_christoffel.sage

end KaluzaKlein.Geometry

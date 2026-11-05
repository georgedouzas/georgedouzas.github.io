import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.Metrics

/-!
# Metric examples displayed in the metric chapters

The polar metric of the plane (components, determinant, inverse), the Minkowski metric as a `metric` family,
and index lowering and raising with the round metric of the sphere.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry Matrix

/-- The polar metric `ds² = dr² + r² dφ²` on the chart `x = (r, φ)`. -/
noncomputable def polar : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![![fun _ => 1, fun _ => 0], ![fun _ => 0, fun x => x 0 ^ 2]]

/-- The matrix of the polar metric at a point. -/
theorem polar_metricAt (x : Fin 2 → ℝ) : metricAt polar x = !![1, 0; 0, x 0 ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, polar]

/-- `det g = r²` for the polar metric. -/
theorem polar_det (x : Fin 2 → ℝ) : (metricAt polar x).det = x 0 ^ 2 := by
  rw [polar_metricAt, Matrix.det_fin_two_of]; ring

/-- The inverse of the polar metric is `diag(1, 1/r²)` away from the origin. -/
theorem polar_invMetric (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    (metricAt polar x)⁻¹ = !![1, 0; 0, 1 / x 0 ^ 2] := by
  refine Matrix.inv_eq_right_inv ?_
  rw [polar_metricAt]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp

/-- The Minkowski metric `η = diag(1, −1, −1, −1)` as a constant metric on `ℝ⁴`. -/
noncomputable def minkowski : Fin 4 → Fin 4 → (Fin 4 → ℝ) → ℝ :=
  fun i j _ => if i = j then (if i = 0 then 1 else -1) else 0

/-- `det η = −1`. -/
theorem minkowski_det (x : Fin 4 → ℝ) : (metricAt minkowski x).det = -1 := by
  have : metricAt minkowski x = Matrix.diagonal ![1, -1, -1, -1] := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, minkowski, Matrix.diagonal]
  rw [this, Matrix.det_diagonal]; simp [Fin.prod_univ_succ]

/-- Lowering the index of `∂_φ = (0, 1)` with the round metric gives the one-form `sin²θ dφ`. -/
theorem lower_dphi (x : Fin 2 → ℝ) : (metricAt sphere x).mulVec ![0, 1] = ![0, Real.sin (x 0) ^ 2] := by
  rw [sphere_metricAt]
  ext i; fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- Raising the index of `dZ = (−sin θ, 0)` with the round metric gives the gradient `−sin θ ∂_θ`. -/
theorem raise_dZ (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (invMetric sphere x).mulVec ![-Real.sin (x 0), 0] = ![-Real.sin (x 0), 0] := by
  rw [sphere_invMetric x h]
  ext i; fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- Lowering then raising returns the original components: `g⁻¹ (g v) = v` for the round metric. -/
theorem raise_lower (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) (v : Fin 2 → ℝ) :
    (invMetric sphere x).mulVec ((metricAt sphere x).mulVec v) = v := by
  unfold invMetric
  rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ (by rw [sphere_det]; exact isUnit_iff_ne_zero.mpr (pow_ne_zero 2 h)), Matrix.one_mulVec]

end KaluzaKlein.Geometry

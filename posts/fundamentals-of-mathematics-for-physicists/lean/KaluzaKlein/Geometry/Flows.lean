import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.MetricExamples

/-!
# Lie derivatives of the polar metric

The Lie derivative of the polar metric along the rotation field vanishes (the rotation is an isometry) and along
the radial field `r ∂_r` it is twice the metric (the dilation is a conformal transformation).
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry

/-- The rotation field `∂_φ` of the plane in polar coordinates, components `(0, 1)`. -/
noncomputable def rotPolar : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun _ => 1]

/-- The radial field `r ∂_r`, components `(r, 0)`. -/
noncomputable def radPolar : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => x 0, fun _ => 0]

/-- `L_{∂_φ} g = 0` for the polar metric: the rotation is an isometry. -/
theorem rotPolar_lieMetric (i j : Fin 2) (x : Fin 2 → ℝ) : lieMetric polar rotPolar i j x = 0 := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [lieMetric, polar, rotPolar, pd, Fin.sum_univ_succ, Function.update_apply]

/-- `L_{r ∂_r} g = 2 g` for the polar metric: the dilation rescales the metric. -/
theorem radPolar_lieMetric (i j : Fin 2) (x : Fin 2 → ℝ) : lieMetric polar radPolar i j x = 2 * polar i j x := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [lieMetric, polar, radPolar, pd, Fin.sum_univ_succ, Function.update_apply] <;> ring

end KaluzaKlein.Geometry

import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.Metrics

/-!
# Tangent vectors, vector fields and one-forms on the sphere: displayed computations

Components of `∂_φ` in the stereographic chart, two Lie brackets of vector fields on the sphere, and the
differential of the height function, all in the spherical chart `x = (θ, φ)`.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry

/-- The stereographic coordinate `u` as a function of `(θ, φ)`. -/
noncomputable def stereoU (x : Fin 2 → ℝ) : ℝ := Real.sin (x 0) * Real.cos (x 1) / (1 - Real.cos (x 0))

/-- The stereographic coordinate `v` as a function of `(θ, φ)`. -/
noncomputable def stereoV (x : Fin 2 → ℝ) : ℝ := Real.sin (x 0) * Real.sin (x 1) / (1 - Real.cos (x 0))

/-- `∂u/∂φ = −v`: the first component of `∂_φ` in the stereographic chart. -/
theorem dphi_u (x : Fin 2 → ℝ) : pd 1 stereoU x = -stereoV x := by
  simp (disch := fun_prop) [pd, stereoU, stereoV, Function.update_apply, deriv_div_const, deriv_const_mul,
    Real.deriv_cos]
  ring

/-- `∂v/∂φ = u`: the second component of `∂_φ` in the stereographic chart. -/
theorem dphi_v (x : Fin 2 → ℝ) : pd 1 stereoV x = stereoU x := by
  simp (disch := fun_prop) [pd, stereoU, stereoV, Function.update_apply, deriv_div_const, deriv_const_mul,
    Real.deriv_sin]

/-- The rotation field `∂_φ`, components `(0, 1)`. -/
noncomputable def dphi : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun _ => 1]

/-- The gradient of the height, `−sin θ ∂_θ`, components `(−sin θ, 0)`. -/
noncomputable def gradZ : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => -Real.sin (x 0), fun _ => 0]

/-- The field `sin θ ∂_φ`, components `(0, sin θ)`. -/
noncomputable def sinDphi : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun x => Real.sin (x 0)]

/-- `∂_θ`, components `(1, 0)`. -/
noncomputable def dtheta : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 1, fun _ => 0]

/-- `[∂_φ, −sin θ ∂_θ] = 0`: the rotation field commutes with the gradient of the height. -/
theorem bracket_dphi_gradZ (k : Fin 2) (x : Fin 2 → ℝ) : lieBracket dphi gradZ k x = 0 := by
  fin_cases k <;>
    simp (disch := fun_prop) [lieBracket, dphi, gradZ, pd, Fin.sum_univ_succ, Function.update_apply]

/-- `[∂_θ, sin θ ∂_φ] = cos θ ∂_φ`. -/
theorem bracket_dtheta_sinDphi (x : Fin 2 → ℝ) :
    lieBracket dtheta sinDphi 0 x = 0 ∧ lieBracket dtheta sinDphi 1 x = Real.cos (x 0) := by
  constructor <;>
    simp (disch := fun_prop) [lieBracket, dtheta, sinDphi, pd, Fin.sum_univ_succ, Function.update_apply,
      Real.deriv_sin]

/-- The height function `Z = cos θ` in the spherical chart. -/
noncomputable def heightZ (x : Fin 2 → ℝ) : ℝ := Real.cos (x 0)

/-- The components of `dZ` are `(−sin θ, 0)`. -/
theorem dZ_components (x : Fin 2 → ℝ) : pd 0 heightZ x = -Real.sin (x 0) ∧ pd 1 heightZ x = 0 := by
  constructor <;> simp (disch := fun_prop) [pd, heightZ, Function.update_apply, Real.deriv_cos]

/-- The velocity `∂_θ + 2 ∂_φ` applied to the height gives `−sin θ`. -/
theorem velocity_on_height (x : Fin 2 → ℝ) : (∑ i, pd i heightZ x * (![1, 2] : Fin 2 → ℝ) i) = -Real.sin (x 0) := by
  have h := dZ_components x
  simp [Fin.sum_univ_succ, h.1, h.2]

/-- `dZ(∂_φ) = 0` and `dZ(∂_θ) = −sin θ`: pairing the differential with the coordinate fields. -/
theorem dZ_on_fields (x : Fin 2 → ℝ) :
    (∑ i, pd i heightZ x * dphi i x) = 0 ∧ (∑ i, pd i heightZ x * dtheta i x) = -Real.sin (x 0) := by
  have h := dZ_components x
  constructor <;> simp [Fin.sum_univ_succ, dphi, dtheta, h.1, h.2]

end KaluzaKlein.Geometry

import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.VectorFields

/-!
# Differential forms on the sphere: displayed computations

The exterior derivative of the one-form `cos θ dφ` is `−sin θ dθ ∧ dφ`, the exterior derivative of `dZ` vanishes
(`d² = 0` on the height function), and the wedge product `dθ ∧ dφ` has the component `1`.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry

/-- The one-form `cos θ dφ` on the sphere, as a function of an index list of length one. -/
noncomputable def hopfForm : (Fin 1 → Fin 2) → (Fin 2 → ℝ) → ℝ :=
  fun I x => if I 0 = 1 then Real.cos (x 0) else 0

/-- The one-form `dZ = −sin θ dθ`. -/
noncomputable def dZform : (Fin 1 → Fin 2) → (Fin 2 → ℝ) → ℝ :=
  fun I x => if I 0 = 0 then -Real.sin (x 0) else 0

/-- `d(cos θ dφ)` has the `(θ, φ)` component `−sin θ`: it is minus the area form. -/
theorem extDeriv_hopfForm (x : Fin 2 → ℝ) : extDeriv hopfForm ![0, 1] x = -Real.sin (x 0) := by
  unfold CoordinateGeometry.extDeriv
  simp (disch := fun_prop) [hopfForm, pd, Fin.sum_univ_succ, Fin.removeNth, Function.update_apply, Real.deriv_cos]

/-- `d(dZ) = 0`: the exterior derivative of an exact form vanishes, on its only component. -/
theorem extDeriv_dZform (x : Fin 2 → ℝ) : extDeriv dZform ![0, 1] x = 0 := by
  unfold CoordinateGeometry.extDeriv
  simp (disch := fun_prop) [dZform, pd, Fin.sum_univ_succ, Fin.removeNth, Function.update_apply]

/-- The coordinate one-form `dθ`. -/
noncomputable def dtheta1 : (Fin 1 → Fin 2) → (Fin 2 → ℝ) → ℝ := fun I _ => if I 0 = 0 then 1 else 0

/-- The coordinate one-form `dφ`. -/
noncomputable def dphi1 : (Fin 1 → Fin 2) → (Fin 2 → ℝ) → ℝ := fun I _ => if I 0 = 1 then 1 else 0

/-- `(dθ ∧ dφ)_{θφ} = 1` and `(dθ ∧ dφ)_{φθ} = −1`: the wedge product is antisymmetric. -/
theorem wedge_dtheta_dphi (x : Fin 2 → ℝ) :
    wedge dtheta1 dphi1 ![0, 1] x = 1 ∧ wedge dtheta1 dphi1 ![1, 0] x = -1 := by
  have hu : (Finset.univ : Finset (Equiv.Perm (Fin 2))) = {1, Equiv.swap 0 1} := by decide
  unfold CoordinateGeometry.wedge
  rw [hu, Finset.sum_pair (by decide), Finset.sum_pair (by decide)]
  simp [dtheta1, dphi1, Equiv.swap_apply_left, Equiv.swap_apply_right]

end KaluzaKlein.Geometry

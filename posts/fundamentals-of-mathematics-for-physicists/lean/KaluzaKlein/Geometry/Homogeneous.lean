import KaluzaKlein.Geometry.LieGroups

/-!
# The sphere as a homogeneous space: displayed facts

Rotations about the `Z`-axis fix the north pole (they form its isotropy group), and the rotation by `t` about
the `Z`-axis moves the point `(1, 0, 0)` of the equator to `(cos t, sin t, 0)`: the orbit of a point of the
equator under the isotropy group of the pole is the whole equator.
-/

namespace KaluzaKlein

open Matrix

/-- The rotations about the `Z`-axis fix the north pole `(0, 0, 1)`. -/
theorem rotation_fixes_pole (t : ℝ) : (rotation t).mulVec ![0, 0, 1] = ![0, 0, 1] := by
  ext i; fin_cases i <;> simp [rotation, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- The rotation by `t` about the `Z`-axis moves `(1, 0, 0)` to `(cos t, sin t, 0)`. -/
theorem rotation_moves_equator (t : ℝ) : (rotation t).mulVec ![1, 0, 0] = ![Real.cos t, Real.sin t, 0] := by
  ext i; fin_cases i <;> simp [rotation, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- Rotations preserve the unit sphere: the image of a unit vector is a unit vector. -/
theorem rotation_preserves_norm (t : ℝ) (v : Fin 3 → ℝ) :
    ((rotation t).mulVec v) ⬝ᵥ ((rotation t).mulVec v) = v ⬝ᵥ v := by
  simp only [rotation, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Fin.sum_univ_zero]
  simp
  nlinarith [Real.sin_sq_add_cos_sq t]

end KaluzaKlein

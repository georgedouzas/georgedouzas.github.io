import Mathlib

/-!
# Charts: algebraic facts displayed in the charts and manifolds chapters

The stereographic inverse chart lands on the sphere, the transition map between the two stereographic charts is
the inversion in the unit circle and is its own inverse, and the `SU(2)` matrix form has determinant
`|a|² + |b|²`.
-/

namespace KaluzaKlein.Geometry

/-- The inverse of the north stereographic chart: `(u, v) ↦ (X, Y, Z)`. -/
noncomputable def stereoInv (u v : ℝ) : ℝ × ℝ × ℝ :=
  (2 * u / (1 + u ^ 2 + v ^ 2), 2 * v / (1 + u ^ 2 + v ^ 2), (u ^ 2 + v ^ 2 - 1) / (1 + u ^ 2 + v ^ 2))

/-- The point `stereoInv u v` lies on the unit sphere. -/
theorem stereoInv_on_sphere (u v : ℝ) :
    (stereoInv u v).1 ^ 2 + (stereoInv u v).2.1 ^ 2 + (stereoInv u v).2.2 ^ 2 = 1 := by
  have h : 1 + u ^ 2 + v ^ 2 ≠ 0 := by positivity
  simp only [stereoInv]
  field_simp
  ring

/-- The inversion in the unit circle, the transition map between the two stereographic charts. -/
noncomputable def inversion (p : ℝ × ℝ) : ℝ × ℝ := (p.1 / (p.1 ^ 2 + p.2 ^ 2), p.2 / (p.1 ^ 2 + p.2 ^ 2))

/-- The inversion is its own inverse away from the origin. -/
theorem inversion_inversion (u v : ℝ) (h : u ^ 2 + v ^ 2 ≠ 0) : inversion (inversion (u, v)) = (u, v) := by
  simp only [inversion, Prod.mk.injEq]
  constructor <;> field_simp <;> ring

/-- The south stereographic coordinates `(X, Y)/(1 + Z)` of the point `stereoInv u v` are the inversion of `(u, v)`. -/
theorem south_of_north (u v : ℝ) (h : u ^ 2 + v ^ 2 ≠ 0) :
    ((stereoInv u v).1 / (1 + (stereoInv u v).2.2), (stereoInv u v).2.1 / (1 + (stereoInv u v).2.2)) =
      inversion (u, v) := by
  have h1 : 1 + u ^ 2 + v ^ 2 ≠ 0 := by positivity
  have h2 : 1 + (u ^ 2 + v ^ 2 - 1) / (1 + u ^ 2 + v ^ 2) = 2 * (u ^ 2 + v ^ 2) / (1 + u ^ 2 + v ^ 2) := by
    field_simp; ring
  simp only [stereoInv, inversion, Prod.mk.injEq, h2]
  constructor <;> field_simp <;> ring

/-- The `SU(2)` matrix form built from two complex numbers. -/
def su2Matrix (a b : ℂ) : Matrix (Fin 2) (Fin 2) ℂ := !![a, -(starRingEnd ℂ) b; b, (starRingEnd ℂ) a]

/-- Its determinant is `|a|² + |b|²`, so the matrix is in `SU(2)` exactly when `(a, b)` lies on the three-sphere. -/
theorem su2Matrix_det (a b : ℂ) : (su2Matrix a b).det = ((Complex.normSq a + Complex.normSq b : ℝ) : ℂ) := by
  simp [su2Matrix, Matrix.det_fin_two_of, Complex.normSq_eq_conj_mul_self]
  ring

end KaluzaKlein.Geometry

import KaluzaKlein.Conventions
import KaluzaKlein.Geometry.MetricExamples
import KaluzaKlein.Geometry.Christoffel

/-!
# The polar connection and two geodesic checks

Christoffel symbols of the polar metric `ds² = dr² + r² dφ²`, the vanishing covariant derivative of the constant
field `∂_x` in polar components, and the geodesic equation on the equator and the meridians of the sphere.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry Matrix

/-- The inverse of the polar metric as a matrix identity, used by the Christoffel computations. -/
theorem polar_inv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    CoordinateGeometry.invMetric polar x = !![(1 : ℝ), (0 : ℝ); (0 : ℝ), (1 / x 0 ^ 2 : ℝ)] := by
  unfold CoordinateGeometry.invMetric
  exact polar_invMetric x h

/-- `Γ^r_{rr} = 0`. -/
theorem polar_christoffel_0_0_0 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 0 0 0 x = 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]

/-- `Γ^r_{rφ} = 0`. -/
theorem polar_christoffel_0_0_1 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 0 0 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]

/-- `Γ^r_{φφ} = −r`. -/
theorem polar_christoffel_0_1_1 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 0 1 1 x = -x 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]
  ring

/-- `Γ^φ_{rr} = 0`. -/
theorem polar_christoffel_1_0_0 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 1 0 0 x = 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]

/-- `Γ^φ_{rφ} = 1/r`. -/
theorem polar_christoffel_1_0_1 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 1 0 1 x = 1 / x 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]
  field_simp

/-- `Γ^φ_{φφ} = 0`. -/
theorem polar_christoffel_1_1_1 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 1 1 1 x = 0 := by
  rw [CoordinateGeometry.christoffel, polar_inv x h]
  simp (disch := fun_prop) [CoordinateGeometry.pd, polar, Fin.sum_univ_succ, Function.update_apply]

/-- The constant field `∂_x` in polar components, `(cos φ, −sin φ / r)`. -/
noncomputable def dxPolar : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => Real.cos (x 1), fun x => -(Real.sin (x 1) / x 0)]

/-- The covariant derivative `∇_i X^k = ∂_i X^k + Γ^k_{ij} X^j` of a vector field. -/
noncomputable def covDeriv {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ)
    (i k : Fin n) (x : Fin n → ℝ) : ℝ :=
  pd i (X k) x + ∑ j, christoffel g k i j x * X j x

/-- `∇ ∂_x = 0` in polar coordinates: the components vary but the covariant derivative vanishes. -/
theorem dxPolar_parallel (i k : Fin 2) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : covDeriv polar dxPolar i k x = 0 := by
  have h2 := polar_christoffel_0_1_1 x h
  have h3 := polar_christoffel_1_0_1 x h
  have h1 := polar_christoffel_0_0_0 x h
  have h4 := polar_christoffel_0_0_1 x h
  have h5 := polar_christoffel_1_0_0 x h
  have h6 := polar_christoffel_1_1_1 x h
  have hs : ∀ k i j, christoffel polar k i j x = christoffel polar k j i x := fun k i j =>
    christoffel_symm polar (by intro i j; ext y; fin_cases i <;> fin_cases j <;> rfl) k i j x
  have h4' : christoffel polar 0 1 0 x = 0 := by rw [hs]; exact h4
  have h3' : christoffel polar 1 1 0 x = 1 / x 0 := by rw [hs]; exact h3
  unfold covDeriv
  fin_cases i <;> fin_cases k <;>
    simp (disch := first | assumption | fun_prop (disch := assumption)) [dxPolar, CoordinateGeometry.pd, Fin.sum_univ_succ,
      Function.update_apply, h1, h2, h3, h4, h5, h6, h3', h4', Real.deriv_sin, Real.deriv_cos, deriv_div_const,
      deriv.fun_neg, deriv_fun_inv'', deriv_mul_const, deriv_fun_div] <;>
    field_simp <;> ring

/-- On the equator `θ = π/2` the only Christoffel symbol in the geodesic equation of `φ = t` vanishes, so the
equator is a geodesic. -/
theorem equator_geodesic (t : ℝ) : christoffel sphere 0 1 1 ![Real.pi / 2, t] = 0 := by
  rw [sphere_christoffel_0_1_1 _ (by simp)]
  simp [Real.cos_pi_div_two]

/-- Along a meridian `θ = t`, `φ = φ₀` the geodesic equation needs only `Γ^k_{θθ}`, which vanish. -/
theorem meridian_geodesic (t φ₀ : ℝ) (h : Real.sin t ≠ 0) :
    christoffel sphere 0 0 0 ![t, φ₀] = 0 ∧ christoffel sphere 1 0 0 ![t, φ₀] = 0 :=
  ⟨sphere_christoffel_0_0_0 _ (by simpa using h), sphere_christoffel_1_0_0 _ (by simpa using h)⟩

end KaluzaKlein.Geometry

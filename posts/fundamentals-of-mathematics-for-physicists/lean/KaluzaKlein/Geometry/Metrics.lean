import Mathlib
import TutorialAssets.CoordinateGeometry

/-!
# The running examples: `S¹`, `S²` and `SU(2) ≅ S³`

Metrics in coordinates, as families of component functions on `Fin n → ℝ` (the `CoordinateGeometry` asset).

* `circle` on `S¹` with the angle coordinate `x 0 = φ ∈ (0, 2π)`: `ds² = dφ²`.
* `sphere` on the unit `S²` in spherical coordinates `x 0 = θ ∈ (0, π)`, `x 1 = φ ∈ (0, 2π)`:
  `ds² = dθ² + sin²θ dφ²`.
* `su2` on `SU(2) ≅ S³` (unit radius) in Euler angles `x 0 = θ`, `x 1 = φ`, `x 2 = ψ`, the bi-invariant metric
  `ds² = ¼ (dθ² + dφ² + dψ² + 2 cos θ dφ dψ)` written with the left-invariant forms
  `σ₁ = cos ψ dθ + sin θ sin ψ dφ`, `σ₂ = −sin ψ dθ + sin θ cos ψ dφ`, `σ₃ = dψ + cos θ dφ`
  as `ds² = ¼ (σ₁² + σ₂² + σ₃²)`.

The chart hypotheses (`Real.sin (x 0) ≠ 0`) exclude the coordinate singularities of the charts.
-/

namespace KaluzaKlein.Geometry

open CoordinateGeometry Matrix

/-- The metric of the unit circle in the angle coordinate: `ds² = dφ²`. -/
noncomputable def circle : Fin 1 → Fin 1 → (Fin 1 → ℝ) → ℝ :=
  ![![fun _ => 1]]

/-- The round metric of the unit sphere in spherical coordinates `(θ, φ)`: `ds² = dθ² + sin²θ dφ²`. -/
noncomputable def sphere : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![![fun _ => 1, fun _ => 0], ![fun _ => 0, fun x => Real.sin (x 0) ^ 2]]

/-- The bi-invariant metric of `SU(2) ≅ S³` in Euler angles `(θ, φ, ψ)`:
`ds² = ¼ (dθ² + dφ² + dψ² + 2 cos θ dφ dψ)`. -/
noncomputable def su2 : Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ :=
  ![![fun _ => 1 / 4, fun _ => 0, fun _ => 0],
    ![fun _ => 0, fun _ => 1 / 4, fun x => Real.cos (x 0) / 4],
    ![fun _ => 0, fun x => Real.cos (x 0) / 4, fun _ => 1 / 4]]

/-! ## Matrices, determinants and inverses -/

/-- The circle's metric matrix is the `1 × 1` identity. -/
theorem circle_metricAt (x : Fin 1 → ℝ) : metricAt circle x = 1 := by
  ext i j; fin_cases i; fin_cases j; simp [metricAt, circle]

/-- The sphere's metric matrix: `diag(1, sin²θ)`. -/
theorem sphere_metricAt (x : Fin 2 → ℝ) :
    metricAt sphere x = !![(1 : ℝ), 0; 0, Real.sin (x 0) ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, sphere]

/-- `det g = sin²θ` on the sphere. -/
theorem sphere_det (x : Fin 2 → ℝ) : (metricAt sphere x).det = Real.sin (x 0) ^ 2 := by
  rw [sphere_metricAt, Matrix.det_fin_two_of]; ring

/-- The inverse metric of the sphere: `diag(1, 1/sin²θ)`, away from the poles. -/
theorem sphere_invMetric (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    invMetric sphere x = !![(1 : ℝ), 0; 0, 1 / Real.sin (x 0) ^ 2] := by
  unfold invMetric
  refine Matrix.inv_eq_right_inv ?_
  rw [sphere_metricAt]
  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp

/-- The metric matrix of `SU(2)` in Euler angles. -/
theorem su2_metricAt (x : Fin 3 → ℝ) :
    metricAt su2 x = !![(1 / 4 : ℝ), 0, 0; 0, 1 / 4, Real.cos (x 0) / 4; 0, Real.cos (x 0) / 4, 1 / 4] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, su2]

/-- `det g = sin²θ / 64` on `SU(2)`: the Euler-angle chart degenerates at `θ ∈ {0, π}`. -/
theorem su2_det (x : Fin 3 → ℝ) : (metricAt su2 x).det = Real.sin (x 0) ^ 2 / 64 := by
  rw [su2_metricAt, Matrix.det_fin_three]
  simp
  nlinarith [Real.sin_sq_add_cos_sq (x 0)]

/-- The inverse metric of `SU(2)` in Euler angles, away from `θ ∈ {0, π}`. -/
theorem su2_invMetric (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    invMetric su2 x =
      !![(4 : ℝ), 0, 0;
         0, 4 / Real.sin (x 0) ^ 2, -4 * Real.cos (x 0) / Real.sin (x 0) ^ 2;
         0, -4 * Real.cos (x 0) / Real.sin (x 0) ^ 2, 4 / Real.sin (x 0) ^ 2] := by
  unfold invMetric
  refine Matrix.inv_eq_right_inv ?_
  rw [su2_metricAt]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp <;>
    nlinarith [Real.sin_sq_add_cos_sq (x 0)]

/-! ## Embeddings and their Gram matrices

The spherical chart is the inverse of an embedding `S² → ℝ³`, and the Euler angles invert an embedding `S³ → ℝ⁴`
(the entries of the `SU(2)` matrix). The Gram matrix `G_ij = Σ_k ∂_i E^k ∂_j E^k` of an embedding is the induced
metric; the theorems below identify it with the metrics defined above. -/

/-- The Gram matrix of an embedding `E : (Fin n → ℝ) → ℝ^m`: `G_ij = Σ_k ∂_i E^k ∂_j E^k`. -/
noncomputable def gram {m n : ℕ} (E : Fin m → (Fin n → ℝ) → ℝ) (i j : Fin n) (x : Fin n → ℝ) : ℝ :=
  ∑ k, pd i (E k) x * pd j (E k) x

/-- The spherical chart's inverse, `(θ, φ) ↦ (sin θ cos φ, sin θ sin φ, cos θ)`. -/
noncomputable def sphereEmbed : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => Real.sin (x 0) * Real.cos (x 1), fun x => Real.sin (x 0) * Real.sin (x 1), fun x => Real.cos (x 0)]

/-- The Euler-angle parametrisation of `SU(2) ≅ S³ ⊂ ℝ⁴`: the real and imaginary parts of the matrix entries
`a = cos(θ/2) e^{−i(φ+ψ)/2}`, `b = −i sin(θ/2) e^{i(φ−ψ)/2}`. -/
noncomputable def su2Embed : Fin 4 → (Fin 3 → ℝ) → ℝ :=
  ![fun x => Real.cos (x 0 / 2) * Real.cos ((x 1 + x 2) / 2),
    fun x => -(Real.cos (x 0 / 2) * Real.sin ((x 1 + x 2) / 2)),
    fun x => Real.sin (x 0 / 2) * Real.sin ((x 1 - x 2) / 2),
    fun x => -(Real.sin (x 0 / 2) * Real.cos ((x 1 - x 2) / 2))]

/-- The Gram matrix of the spherical chart is the round metric: `G = diag(1, sin²θ)`. -/
theorem sphere_gram (i j : Fin 2) (x : Fin 2 → ℝ) : gram sphereEmbed i j x = sphere i j x := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [gram, pd, sphereEmbed, sphere, Fin.sum_univ_succ, Function.update_apply,
      deriv_fun_mul, Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | ((try ring_nf); (try simp only [Real.cos_sq']); ring1)

/-- The Gram determinant of the spherical chart is `sin²θ`: the chart is regular away from the poles. -/
theorem sphere_gram_det (x : Fin 2 → ℝ) :
    (Matrix.of fun i j => gram sphereEmbed i j x).det = Real.sin (x 0) ^ 2 := by
  have : (Matrix.of fun i j => gram sphereEmbed i j x) = metricAt sphere x := by
    ext i j; simp [metricAt, sphere_gram]
  rw [this, sphere_det]

/-- The Gram matrix of the Euler-angle embedding is the bi-invariant metric of `SU(2)`. -/
theorem su2_gram (i j : Fin 3) (x : Fin 3 → ℝ) : gram su2Embed i j x = su2 i j x := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [gram, pd, su2Embed, su2, Fin.sum_univ_succ, Function.update_apply,
      deriv_fun_mul, deriv.fun_neg, Real.deriv_sin, Real.deriv_cos, deriv_div_const, deriv_fun_add, deriv_fun_sub]
  all_goals first
    | ring1
    | ((try rw [show Real.cos (x 0) = 2 * Real.cos (x 0 / 2) ^ 2 - 1 by rw [← Real.cos_two_mul]; ring_nf])
       (try ring_nf)
       (try simp only [Real.cos_sq'])
       ring1)

end KaluzaKlein.Geometry

import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Geometry.Metrics
import KaluzaKlein.Geometry.MetricExamples

/-!
# Differential geometry, section "The metric"

The claims of the section "The metric" of the part "Metrics" of `tutorial/part-i-09-differential-geometry.qmd`
(design D28: geometry in coordinates). A metric is a family of component functions
`g : Fin n → Fin n → (Fin n → ℝ) → ℝ` (the `CoordinateGeometry` asset); its matrix at a chart point `x` is
`metricAt g x`, the inner product of two tangent vectors is `gInner g x X Y = X ⬝ᵥ (metricAt g x *ᵥ Y)`, and a
change of chart acts on the component matrix by the congruence `Jᵀ g J` with the Jacobian `J = ∂x/∂x'`, written as
an explicit `!![…]` matrix. Determinants, inverses and line elements are matrix identities of explicit matrices
(certified), the general facts (transformation law of the determinant, sign of the determinant, positive
definiteness, polarisation) are theorems with universal binders, tested at instances. Signatures are Veriphy's
`HasSignature` (an explicit congruence to a diagonal matrix with the counts of signs); the signature convention is
`(+, −, −, −)`.
-/

namespace KaluzaKlein.DifferentialGeometry.Metric

open Matrix CoordinateGeometry KaluzaKlein.Geometry Filter Topology

/-! ## def-metric -/

/-- The inner product `g(X, Y) = g_ij X^i Y^j` of two tangent vectors at the chart point `x`. -/
noncomputable def gInner {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x X Y : Fin n → ℝ) : ℝ :=
  X ⬝ᵥ (metricAt g x *ᵥ Y)

/-- A metric: symmetric components, non-degenerate (`det g ≠ 0`) at every point of the chart. -/
def IsMetric {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) : Prop :=
  (∀ i j x, g i j x = g j i x) ∧ ∀ x, (metricAt g x).det ≠ 0

/-- A Riemannian metric: `g(X, X) > 0` for every non-zero tangent vector at every point. -/
def IsRiemannian {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) : Prop :=
  ∀ x X, X ≠ 0 → 0 < gInner g x X X

/-- The Euclidean metric `g_ij = δ_ij` on `ℝⁿ` in Cartesian coordinates. -/
def euclid (n : ℕ) : Fin n → Fin n → (Fin n → ℝ) → ℝ := fun i j _ => if i = j then 1 else 0

/-- A metric with symmetric components defines a symmetric bilinear form on each tangent space:
`g(X, Y) = g(Y, X)`. -/
theorem gInner_symm {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (hs : ∀ i j x, g i j x = g j i x)
    (x X Y : Fin n → ℝ) : gInner g x X Y = gInner g x Y X := by
  have ht : (metricAt g x)ᵀ = metricAt g x := by
    ext i j; simp [metricAt, hs i j x]
  unfold gInner
  rw [dotProduct_mulVec, ← mulVec_transpose, ht, dotProduct_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.gInner_symm, dg-me-ginner-symm-test]

/-! ## exm-metric-euclidean -/

/-- The Euclidean metric is the dot product: `g(X, Y) = X · Y` in Cartesian coordinates. -/
theorem euclid_inner {n : ℕ} (x X Y : Fin n → ℝ) : gInner (euclid n) x X Y = X ⬝ᵥ Y := by
  have h1 : metricAt (euclid n) x = 1 := by
    ext i j; simp [metricAt, euclid, Matrix.one_apply]
  simp [gInner, h1]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.euclid_inner, dg-me-euclid-inner-test]

/-- The same inner product in polar coordinates: with the Jacobian `J = ∂(x, y)/∂(r, φ)`, `Jᵀ 1 J = diag(1, r²)`,
so `g_rr = 1`, `g_φφ = r²`, `g_rφ = 0`: the components depend on the chart, the metric does not. -/
theorem polar_from_cartesian (r φ : ℝ) :
    (!![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ])ᵀ * !![(1 : ℝ), 0; 0, 1] *
      !![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ] = !![1, 0; 0, r ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- The matrix of the polar metric at the chart point `x = (r, φ)` is `diag(1, r²)`. -/
theorem polar_components (x : Fin 2 → ℝ) : metricAt polar x = !![1, 0; 0, x 0 ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, polar]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.polar_components, dg-me-polar-components-test]

/-! ## exm-metric-minkowski -/

/-- The matrix of the Minkowski metric is `η = diag(1, −1, −1, −1)` at every point. -/
theorem minkowski_components (x : Fin 4 → ℝ) :
    metricAt minkowski x = !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, minkowski]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.minkowski_components, dg-me-minkowski-components-test]

/-- `η` is non-degenerate: `det η = −1`. -/
theorem minkowski_det (x : Fin 4 → ℝ) : (metricAt minkowski x).det = -1 := by
  rw [minkowski_components]
  VERIPHY[dg-me-matrix]

/-- `η(X, X)` takes all three signs: `+1` on `∂_t`, `0` on `∂_t + ∂_x`, `−1` on `∂_x`. -/
theorem minkowski_three_signs :
    ![(1 : ℝ), 0, 0, 0] ⬝ᵥ (!![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *ᵥ ![1, 0, 0, 0]) = 1 ∧
    ![(1 : ℝ), 1, 0, 0] ⬝ᵥ (!![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *ᵥ ![1, 1, 0, 0]) = 0 ∧
    ![(0 : ℝ), 1, 0, 0] ⬝ᵥ (!![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *ᵥ ![0, 1, 0, 0]) = -1 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-! ## exm-metric-sphere -/

/-- The matrix of the round metric at `(θ, φ)` is `diag(1, sin²θ)`. -/
theorem sphere_components (x : Fin 2 → ℝ) : metricAt sphere x = !![(1 : ℝ), 0; 0, Real.sin (x 0) ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sphere_components, dg-me-sphere-components-test]

/-- The round metric is the Gram matrix of the spherical chart: `G_ij = Σ_k ∂_i E^k ∂_j E^k = g_ij`. -/
theorem sphere_gram_components (i j : Fin 2) (x : Fin 2 → ℝ) : gram sphereEmbed i j x = sphere i j x :=
  sphere_gram i j x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sphere_gram_components, dg-me-sphere-gram-test]

/-- `det g = sin²θ` on the sphere, non-zero on the chart domain `sin θ ≠ 0`. -/
theorem sphere_det (x : Fin 2 → ℝ) : (metricAt sphere x).det = Real.sin (x 0) ^ 2 := by
  rw [sphere_metricAt]
  VERIPHY[dg-me-matrix]

/-! ## exr-polar-metric -/

/-- `det g = r²` for the polar metric. -/
theorem polar_det (x : Fin 2 → ℝ) : (metricAt polar x).det = x 0 ^ 2 := by
  rw [polar_metricAt]
  VERIPHY[dg-me-matrix]

/-- The inverse of the polar metric is `diag(1, 1/r²)` for `r ≠ 0`: `g · diag(1, 1/r²) = 1`. -/
theorem polar_inv_mul (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : metricAt polar x * !![1, 0; 0, 1 / x 0 ^ 2] = 1 := by
  rw [polar_metricAt]
  VERIPHY[dg-me-matrix]

/-- The volume element `√(det g) = √(r²) = r` of polar coordinates (`r ≥ 0`). -/
theorem polar_volume_element (r : ℝ) (h : 0 ≤ r) : Real.sqrt (r ^ 2) = r := by
  VERIPHY[dg-me-trig]

/-! ## exr-metric-which-are-metrics -/

/-- The determinants of the four candidate metrics: `1`, `−1`, `x²`, `−1`. -/
theorem four_determinants (x : ℝ) :
    (!![(1 : ℝ), 0; 0, 1]).det = 1 ∧ (!![(0 : ℝ), 1; 1, 0]).det = -1 ∧
      (!![x ^ 2, 0; 0, 1]).det = x ^ 2 ∧ (!![(1 : ℝ), 0; 0, -1]).det = -1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `x² dx ⊗ dx + dy ⊗ dy` is degenerate exactly on the line `x = 0`. -/
theorem third_degenerate_iff (x : ℝ) : (!![x ^ 2, 0; 0, 1]).det = 0 ↔ x = 0 := by
  rw [Matrix.det_fin_two_of]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.third_degenerate_iff, dg-me-third-degenerate-test]

/-- The Euclidean form is positive: `g(X, X) = (Xˣ)² + (Xʸ)² > 0` for `X ≠ 0`. -/
theorem euclid2_posdef (a b : ℝ) (hab : a ≠ 0 ∨ b ≠ 0) :
    0 < ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, 1] *ᵥ ![a, b]) := by
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  rcases hab with ha | hb
  · have := pow_pos (abs_pos.mpr ha) 2; rw [sq_abs] at this; nlinarith [sq_nonneg b]
  · have := pow_pos (abs_pos.mpr hb) 2; rw [sq_abs] at this; nlinarith [sq_nonneg a]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.euclid2_posdef, dg-me-euclid2-posdef-test]

/-- `dx ⊗ dy + dy ⊗ dx` gives `g(X, X) = 2 Xˣ Xʸ`, negative on `∂_x − ∂_y`. -/
theorem hyperbolic_form (a b : ℝ) :
    ![a, b] ⬝ᵥ (!![(0 : ℝ), 1; 1, 0] *ᵥ ![a, b]) = 2 * a * b ∧
      ![(1 : ℝ), -1] ⬝ᵥ (!![(0 : ℝ), 1; 1, 0] *ᵥ ![1, -1]) = -2 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `dx ⊗ dx − dy ⊗ dy` gives `g(X, X) = (Xˣ)² − (Xʸ)²`, negative on `∂_y`. -/
theorem lorentz2_form (a b : ℝ) :
    ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, -1] *ᵥ ![a, b]) = a ^ 2 - b ^ 2 ∧
      ![(0 : ℝ), 1] ⬝ᵥ (!![(1 : ℝ), 0; 0, -1] *ᵥ ![0, 1]) = -1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- The second and fourth forms are the two-dimensional Minkowski metric in different coordinates: both have
signature `(1, 1, 0)`. -/
theorem hyperbolic_signature : Veriphy.Linear.HasSignature !![(0 : ℚ), 1; 1, 0] 1 1 0 := by
  VERIPHY[la-signature]

/-- `diag(1, −1)` has signature `(1, 1, 0)`. -/
theorem lorentz2_signature : Veriphy.Linear.HasSignature !![(1 : ℚ), 0; 0, -1] 1 1 0 := by
  VERIPHY[la-signature]

/-! ## exr-metric-chart-independence -/

/-- Under a change of chart `g' = Jᵀ g J`, so `det g' = (det J)² det g`. -/
theorem det_congr {n : ℕ} (J g : Matrix (Fin n) (Fin n) ℝ) : (Jᵀ * g * J).det = J.det ^ 2 * g.det := by
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.det_congr, dg-me-det-congr-test]

/-- Non-degeneracy is chart independent: `det J ≠ 0` and `det g ≠ 0` give `det (Jᵀ g J) ≠ 0`. -/
theorem det_congr_ne_zero {n : ℕ} (J g : Matrix (Fin n) (Fin n) ℝ) (hJ : J.det ≠ 0) (hg : g.det ≠ 0) :
    (Jᵀ * g * J).det ≠ 0 := by
  rw [det_congr]; exact mul_ne_zero (pow_ne_zero 2 hJ) hg
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.det_congr_ne_zero, dg-me-det-congr-ne-test]

/-- The polar Jacobian `J = ∂(x, y)/∂(r, φ)` has `det J = r`, so `det g_polar = r² · 1`. -/
theorem polar_jacobian_det (r φ : ℝ) : (!![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ]).det = r := by
  VERIPHY[dg-me-matrix]

/-! ## def-signature-metric -/

/-- Lorentzian signature `(+, −, −, −)`: `η` has signature `(1, 3, 0)`. -/
theorem eta_signature :
    Veriphy.Linear.HasSignature !![(1 : ℚ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] 1 3 0 := by
  VERIPHY[la-signature]

/-- The Euclidean metric is Riemannian: `g(X, X) = X · X > 0` for `X ≠ 0`. -/
theorem euclid_posdef {n : ℕ} (x X : Fin n → ℝ) (hX : X ≠ 0) : 0 < gInner (euclid n) x X X := by
  rw [euclid_inner]
  rcases (dotProduct_self_star_nonneg X).lt_or_eq with h | h
  · simpa using h
  · exact absurd ((dotProduct_self_star_eq_zero X).mp (by simpa using h.symm)) hX
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.euclid_posdef, dg-me-euclid-riemannian-test]

/-! ## exm-signature-riemannian -/

/-- The Gram matrix of an embedding is positive: `vᵀ (Jᵀ J) v = |J v|²`. -/
theorem gram_quadratic {m n : ℕ} (J : Matrix (Fin m) (Fin n) ℝ) (v : Fin n → ℝ) :
    v ⬝ᵥ ((Jᵀ * J) *ᵥ v) = (J *ᵥ v) ⬝ᵥ (J *ᵥ v) := by
  rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.gram_quadratic, dg-me-gram-quadratic-test]

/-- `vᵀ (Jᵀ J) v = |J v|² ≥ 0`. -/
theorem gram_nonneg {m n : ℕ} (J : Matrix (Fin m) (Fin n) ℝ) (v : Fin n → ℝ) : 0 ≤ v ⬝ᵥ ((Jᵀ * J) *ᵥ v) := by
  rw [gram_quadratic]; simpa using dotProduct_self_star_nonneg (J *ᵥ v)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.gram_nonneg, dg-me-gram-nonneg-test]

/-- The round metric is Riemannian on the chart domain: `a² + sin²θ b² > 0` for `(a, b) ≠ 0`. -/
theorem sphere_posdef (θ a b : ℝ) (h : Real.sin θ ≠ 0) (hab : a ≠ 0 ∨ b ≠ 0) :
    0 < a ^ 2 + Real.sin θ ^ 2 * b ^ 2 := by
  rcases hab with ha | hb
  · have := pow_pos (abs_pos.mpr ha) 2; positivity
  · have h1 : 0 < Real.sin θ ^ 2 * b ^ 2 := by positivity
    positivity
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sphere_posdef, dg-me-sphere-posdef-test]

/-- The torus metric `r² da² + (R + r cos a)² db²` is Riemannian for `0 < r < R`. -/
theorem torus_posdef (R r a u v : ℝ) (hr : 0 < r) (hR : r < R) (huv : u ≠ 0 ∨ v ≠ 0) :
    0 < r ^ 2 * u ^ 2 + (R + r * Real.cos a) ^ 2 * v ^ 2 := by
  have hpos : 0 < R + r * Real.cos a := by nlinarith [Real.neg_one_le_cos a]
  rcases huv with hu | hv
  · have := pow_pos (abs_pos.mpr hu) 2; positivity
  · have h1 : 0 < (R + r * Real.cos a) ^ 2 * v ^ 2 := by positivity
    positivity
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.torus_posdef, dg-me-torus-posdef-test]

/-- The `SU(2)` metric is Riemannian on the Euler chart: `¼(a² + b² + c² + 2 cos θ b c) > 0` for `(a, b, c) ≠ 0`,
since `b² + c² + 2 cos θ b c = (b + cos θ c)² + sin²θ c²`. -/
theorem su2_posdef (θ a b c : ℝ) (h : Real.sin θ ≠ 0) (habc : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
    0 < (a ^ 2 + b ^ 2 + c ^ 2 + 2 * Real.cos θ * b * c) / 4 := by
  have key : a ^ 2 + b ^ 2 + c ^ 2 + 2 * Real.cos θ * b * c =
      a ^ 2 + (b + Real.cos θ * c) ^ 2 + Real.sin θ ^ 2 * c ^ 2 := by
    linear_combination c ^ 2 * (Real.sin_sq_add_cos_sq θ)
  rw [key]
  have ha2 : 0 ≤ a ^ 2 := sq_nonneg a
  have hb2 : 0 ≤ (b + Real.cos θ * c) ^ 2 := sq_nonneg _
  have hs : 0 ≤ Real.sin θ ^ 2 * c ^ 2 := by positivity
  rcases habc with ha | hb | hc
  · have := pow_pos (abs_pos.mpr ha) 2; rw [sq_abs] at this; linarith
  · by_cases hc : c = 0
    · subst hc; have := pow_pos (abs_pos.mpr hb) 2; rw [sq_abs] at this; nlinarith
    · have h1 : 0 < Real.sin θ ^ 2 * c ^ 2 := by positivity
      linarith
  · have h1 : 0 < Real.sin θ ^ 2 * c ^ 2 := by positivity
    linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.su2_posdef, dg-me-su2-posdef-test]

/-! ## exm-signature-lorentzian -/

/-- The Kaluza-Klein metric `diag(1, −1, −1, −1, −R²)` on `M₄ × S¹` has signature `(1, 4, 0)`: five eigenvalues of
signs `(+, −, −, −, −)`, Lorentzian in five dimensions. -/
theorem kk_circle_signature (R : ℝ) (hR : R ≠ 0) :
    Veriphy.Linear.HasSignature
      !![(1 : ℝ), 0, 0, 0, 0; 0, -1, 0, 0, 0; 0, 0, -1, 0, 0; 0, 0, 0, -1, 0; 0, 0, 0, 0, -R ^ 2] 1 4 0 := by
  have hR2 : -R ^ 2 < 0 := by have := pow_pos (abs_pos.mpr hR) 2; rw [sq_abs] at this; linarith
  refine ⟨1, ![1, -1, -1, -1, -R ^ 2], by simp, ?_, ?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.diagonal]
  have hRsq : 0 < R ^ 2 := by linarith
  all_goals
    simp only [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]
    norm_num [hR2, hR2.ne, not_lt.mpr hR2.le, hRsq, hRsq.ne', not_lt.mpr hRsq.le]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.kk_circle_signature, dg-me-kk-circle-signature-test]

/-! ## exm-metric-signature-degenerate -/

/-- At `θ = 0` the spherical chart's matrix `diag(1, sin²0) = diag(1, 0)` has a zero eigenvalue: `det = 0`. -/
theorem sphere_pole_degenerate : (!![(1 : ℝ), 0; 0, Real.sin 0 ^ 2]).det = 0 := by
  rw [Real.sin_zero]
  VERIPHY[dg-me-matrix]

/-- The stereographic components `4 δ_ij / (1 + u² + v²)²` are positive definite everywhere. -/
theorem stereo_posdef (u v a b : ℝ) (hab : a ≠ 0 ∨ b ≠ 0) :
    0 < 4 / (1 + u ^ 2 + v ^ 2) ^ 2 * (a ^ 2 + b ^ 2) := by
  have : 0 < a ^ 2 + b ^ 2 := by
    rcases hab with ha | hb
    · have := pow_pos (abs_pos.mpr ha) 2; rw [sq_abs] at this; positivity
    · have := pow_pos (abs_pos.mpr hb) 2; rw [sq_abs] at this; positivity
  positivity
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.stereo_posdef, dg-me-stereo-posdef-test]

/-- The inverse stereographic projection `E(u, v) = (2u, 2v, u² + v² − 1)/(1 + u² + v²)` as an embedding of the
chart plane into `ℝ³`. -/
noncomputable def stereoEmbed : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => 2 * x 0 / (1 + x 0 ^ 2 + x 1 ^ 2), fun x => 2 * x 1 / (1 + x 0 ^ 2 + x 1 ^ 2),
    fun x => (x 0 ^ 2 + x 1 ^ 2 - 1) / (1 + x 0 ^ 2 + x 1 ^ 2)]

/-- The Gram matrix of the stereographic chart is `4 δ_ij / (1 + u² + v²)²`. -/
def stereoGramClaim : Prop :=
  ∀ (i j : Fin 2) (x : Fin 2 → ℝ), gram stereoEmbed i j x = if i = j then 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 else 0
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.stereoGramClaim, dg-me-stereo-gram-test]

/-- The Jacobian of `E` satisfies `Jᵀ J = 4 δ_ij / (1 + u² + v²)²`: the stereographic chart is conformal. -/
theorem stereo_jacobian_gram (u v : ℝ) :
    (!![2 * (1 - u ^ 2 + v ^ 2) / (1 + u ^ 2 + v ^ 2) ^ 2, -(4 * u * v) / (1 + u ^ 2 + v ^ 2) ^ 2;
        -(4 * u * v) / (1 + u ^ 2 + v ^ 2) ^ 2, 2 * (1 + u ^ 2 - v ^ 2) / (1 + u ^ 2 + v ^ 2) ^ 2;
        4 * u / (1 + u ^ 2 + v ^ 2) ^ 2, 4 * v / (1 + u ^ 2 + v ^ 2) ^ 2])ᵀ *
      !![2 * (1 - u ^ 2 + v ^ 2) / (1 + u ^ 2 + v ^ 2) ^ 2, -(4 * u * v) / (1 + u ^ 2 + v ^ 2) ^ 2;
        -(4 * u * v) / (1 + u ^ 2 + v ^ 2) ^ 2, 2 * (1 + u ^ 2 - v ^ 2) / (1 + u ^ 2 + v ^ 2) ^ 2;
        4 * u / (1 + u ^ 2 + v ^ 2) ^ 2, 4 * v / (1 + u ^ 2 + v ^ 2) ^ 2] =
      !![4 / (1 + u ^ 2 + v ^ 2) ^ 2, 0; 0, 4 / (1 + u ^ 2 + v ^ 2) ^ 2] := by
  have hD : 1 + u ^ 2 + v ^ 2 ≠ 0 := by positivity
  rw [← sub_eq_zero]
  VERIPHY[dg-me-matrix]

/-! ## exr-null-vectors -/

/-- The null vectors of the `(t, x)` Minkowski plane: `a² − b² = 0` iff `b = ±a`, the two null lines. -/
theorem null_iff (a b : ℝ) : ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, -1] *ᵥ ![a, b]) = 0 ↔ b = a ∨ b = -a := by
  have e : ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, -1] *ᵥ ![a, b]) = a ^ 2 - b ^ 2 := by
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]; ring
  rw [e]
  constructor
  · intro h
    have : (b - a) * (b + a) = 0 := by linear_combination -h
    rcases mul_eq_zero.mp this with h1 | h1
    · left; linarith
    · right; linarith
  · rintro (h | h) <;> subst h <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.null_iff, dg-me-null-iff-test]

/-! ## exr-signature-sage -/

/-- `diag(1, −1, −1, −1, −1)` has signature `(1, 4, 0)`: Lorentzian in five dimensions. -/
theorem five_signature :
    Veriphy.Linear.HasSignature
      !![(1 : ℚ), 0, 0, 0, 0; 0, -1, 0, 0, 0; 0, 0, -1, 0, 0; 0, 0, 0, -1, 0; 0, 0, 0, 0, -1] 1 4 0 := by
  VERIPHY[la-signature]

/-- `[[1, 2], [2, 1]]` has signature `(1, 1, 0)`: a two-dimensional Lorentzian metric in disguise. -/
theorem cross12_signature : Veriphy.Linear.HasSignature !![(1 : ℚ), 2; 2, 1] 1 1 0 := by
  VERIPHY[la-signature]

/-- Its eigenvalues are `3` and `−1`, with eigenvectors `(1, 1)` and `(1, −1)`. -/
theorem cross12_eigen :
    !![(1 : ℝ), 2; 2, 1] *ᵥ ![1, 1] = (3 : ℝ) • ![1, 1] ∧ !![(1 : ℝ), 2; 2, 1] *ᵥ ![1, -1] = (-1 : ℝ) • ![1, -1] := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `[[2, 1], [1, 2]]` has signature `(2, 0, 0)`: Riemannian. -/
theorem cross21_signature : Veriphy.Linear.HasSignature !![(2 : ℚ), 1; 1, 2] 2 0 0 := by
  VERIPHY[la-signature]

/-- Its eigenvalues are `1` and `3`. -/
theorem cross21_eigen :
    !![(2 : ℝ), 1; 1, 2] *ᵥ ![1, -1] = (1 : ℝ) • ![1, -1] ∧ !![(2 : ℝ), 1; 1, 2] *ᵥ ![1, 1] = (3 : ℝ) • ![1, 1] := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `[[1, 2], [2, 4]]` has determinant `0` and signature `(1, 0, 1)`: degenerate, not a metric. -/
theorem degenerate_matrix : (!![(1 : ℝ), 2; 2, 4]).det = 0 := by
  VERIPHY[dg-me-matrix]

/-- Signature `(1, 0, 1)` of `[[1, 2], [2, 4]]`. -/
theorem degenerate_signature : Veriphy.Linear.HasSignature !![(1 : ℚ), 2; 2, 4] 1 0 1 := by
  VERIPHY[la-signature]

/-! ## exr-signature-constant -/

/-- A continuous function that never vanishes on a connected set keeps its sign there. -/
theorem sign_constant_of_ne_zero {s : Set ℝ} (hs : IsPreconnected s) {f : ℝ → ℝ} (hf : ContinuousOn f s)
    (h0 : ∀ x ∈ s, f x ≠ 0) : (∀ x ∈ s, 0 < f x) ∨ (∀ x ∈ s, f x < 0) := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨⟨a, ha, hfa⟩, ⟨b, hb, hfb⟩⟩ := hcon
  have hfa' : f a < 0 := lt_of_le_of_ne hfa (h0 a ha)
  have hfb' : 0 < f b := lt_of_le_of_ne hfb (Ne.symm (h0 b hb))
  have himg := (hs.image f hf).ordConnected
  obtain ⟨c, hc, hfc⟩ := himg.out ⟨a, ha, rfl⟩ ⟨b, hb, rfl⟩ ⟨hfa'.le, hfb'.le⟩
  exact h0 c hc hfc
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sign_constant_of_ne_zero, dg-me-sign-constant-test]

/-- The round metric has signature `(2, 0, 0)` at every point of the chart domain: the count of positive
eigenvalues does not change with the point. -/
theorem sphere_signature_constant (θ : ℝ) (h : Real.sin θ ≠ 0) :
    Veriphy.Linear.HasSignature !![(1 : ℝ), 0; 0, Real.sin θ ^ 2] 2 0 0 := by
  have h2 : 0 < Real.sin θ ^ 2 := by positivity
  refine ⟨1, ![1, Real.sin θ ^ 2], by simp, ?_, ?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.diagonal]
  all_goals
    simp only [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]
    norm_num [h2, h2.ne', not_lt.mpr h2.le, h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sphere_signature_constant, dg-me-sphere-signature-test]

/-! ## def-line-element -/

/-- The symmetrised product `dx^i dx^j = ½(dx^i ⊗ dx^j + dx^j ⊗ dx^i)` evaluates on `X` to `X^i X^j`. -/
theorem symmetrised_product (a b : ℝ) : (a * b + b * a) / 2 = a * b := by
  VERIPHY[de-poly]

/-- A cross term `2 g_ij dx^i dx^j` (`i ≠ j`) counts both orderings: `g_ij X^i X^j + g_ji X^j X^i = 2 g_ij X^i X^j`
for symmetric components. -/
theorem cross_term_both_orderings (g a b : ℝ) : g * a * b + g * b * a = 2 * g * a * b := by
  VERIPHY[de-poly]

/-! ## exm-line-element-polar -/

/-- `ds² = dr² + r² dφ²` evaluates on `X = (a, b)` to `a² + r² b²`. -/
theorem polar_line_element (r a b : ℝ) : ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, r ^ 2] *ᵥ ![a, b]) = a ^ 2 + r ^ 2 * b ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- `ds² = dx² + dy²` evaluates on `X = (a, b)` to `a² + b²`. -/
theorem cartesian_line_element (a b : ℝ) : ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, 1] *ᵥ ![a, b]) = a ^ 2 + b ^ 2 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-line-element-sphere -/

/-- `ds² = dθ² + sin²θ dφ²` evaluates on `X = (a, b)` to `a² + sin²θ b²`. -/
theorem sphere_line_element (θ a b : ℝ) :
    ![a, b] ⬝ᵥ (!![(1 : ℝ), 0; 0, Real.sin θ ^ 2] *ᵥ ![a, b]) = a ^ 2 + Real.sin θ ^ 2 * b ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The sphere of radius `R`: `R² (dθ² + sin²θ dφ²)` has matrix `diag(R², R² sin²θ)`. -/
theorem sphere_radius_matrix (R θ : ℝ) :
    R ^ 2 • !![(1 : ℝ), 0; 0, Real.sin θ ^ 2] = !![R ^ 2, 0; 0, R ^ 2 * Real.sin θ ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- `ds² = dt² − dx² − dy² − dz²` evaluates on `(t, x, y, z)` to `t² − x² − y² − z²`. -/
theorem minkowski_line_element (t x y z : ℝ) :
    ![t, x, y, z] ⬝ᵥ (!![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *ᵥ ![t, x, y, z]) =
      t ^ 2 - x ^ 2 - y ^ 2 - z ^ 2 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-line-element-cross -/

/-- The matrix of the `SU(2)` metric in Euler angles: `g_φφ = g_ψψ = ¼`, `g_φψ = g_ψφ = ¼ cos θ`. -/
theorem su2_components (x : Fin 3 → ℝ) :
    metricAt su2 x = !![(1 / 4 : ℝ), 0, 0; 0, 1 / 4, Real.cos (x 0) / 4; 0, Real.cos (x 0) / 4, 1 / 4] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, su2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.su2_components, dg-me-su2-components-test]

/-- `ds² = ¼(dθ² + dφ² + dψ² + 2 cos θ dφ dψ)`: the cross term counts both orderings. -/
theorem su2_line_element (θ a b c : ℝ) :
    ![a, b, c] ⬝ᵥ (!![(1 / 4 : ℝ), 0, 0; 0, 1 / 4, Real.cos θ / 4; 0, Real.cos θ / 4, 1 / 4] *ᵥ ![a, b, c]) =
      (a ^ 2 + b ^ 2 + c ^ 2 + 2 * Real.cos θ * b * c) / 4 := by
  VERIPHY[dg-me-matrix]

/-! ## exr-line-element-read-off -/

/-- `4 du² + 6 du dv + 9 dv²` has matrix `[[4, 3], [3, 9]]`. -/
theorem readoff_first (a b : ℝ) :
    ![a, b] ⬝ᵥ (!![(4 : ℝ), 3; 3, 9] *ᵥ ![a, b]) = 4 * a ^ 2 + 6 * a * b + 9 * b ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- Its determinant is `27` and its trace `13`: both eigenvalues positive. -/
theorem readoff_first_det_trace : (!![(4 : ℝ), 3; 3, 9]).det = 27 ∧ (!![(4 : ℝ), 3; 3, 9]).trace = 13 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `[[4, 3], [3, 9]]` is Riemannian: signature `(2, 0, 0)`. -/
theorem readoff_first_signature : Veriphy.Linear.HasSignature !![(4 : ℚ), 3; 3, 9] 2 0 0 := by
  VERIPHY[la-signature]

/-- `dt² − 2 dt dx` has matrix `[[1, −1], [−1, 0]]` with determinant `−1`. -/
theorem readoff_second (t x : ℝ) :
    ![t, x] ⬝ᵥ (!![(1 : ℝ), -1; -1, 0] *ᵥ ![t, x]) = t ^ 2 - 2 * t * x ∧ (!![(1 : ℝ), -1; -1, 0]).det = -1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- `[[1, −1], [−1, 0]]` is Lorentzian: signature `(1, 1, 0)`, although no `dx²` term appears. -/
theorem readoff_second_signature : Veriphy.Linear.HasSignature !![(1 : ℚ), -1; -1, 0] 1 1 0 := by
  VERIPHY[la-signature]

/-! ## exr-line-element-parabolic -/

/-- Parabolic coordinates `x = uv`, `y = ½(u² − v²)`: the Jacobian `J = [[v, u], [u, −v]]` gives
`Jᵀ J = (u² + v²) δ_ij`: an orthogonal, conformal chart. -/
theorem parabolic_metric (u v : ℝ) :
    (!![v, u; u, -v])ᵀ * !![v, u; u, -v] = !![u ^ 2 + v ^ 2, 0; 0, u ^ 2 + v ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- `det g = (u² + v²)²` in parabolic coordinates. -/
theorem parabolic_det (u v : ℝ) : (!![u ^ 2 + v ^ 2, 0; 0, u ^ 2 + v ^ 2]).det = (u ^ 2 + v ^ 2) ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The inverse is `(u² + v²)⁻¹ δ^ij` away from the origin. -/
theorem parabolic_inverse (u v : ℝ) (h : u ^ 2 + v ^ 2 ≠ 0) :
    !![u ^ 2 + v ^ 2, 0; 0, u ^ 2 + v ^ 2] * !![1 / (u ^ 2 + v ^ 2), 0; 0, 1 / (u ^ 2 + v ^ 2)] = 1 := by
  VERIPHY[dg-me-matrix]

/-- The parabolic chart degenerates at the origin only: `u² + v² = 0 ↔ u = 0 ∧ v = 0`. -/
theorem parabolic_degenerate_iff (u v : ℝ) : u ^ 2 + v ^ 2 = 0 ↔ u = 0 ∧ v = 0 := by
  constructor
  · intro h
    have hu : u ^ 2 = 0 := by nlinarith [sq_nonneg u, sq_nonneg v]
    have hv : v ^ 2 = 0 := by nlinarith [sq_nonneg u, sq_nonneg v]
    exact ⟨pow_eq_zero_iff (by norm_num) |>.mp hu, pow_eq_zero_iff (by norm_num) |>.mp hv⟩
  · rintro ⟨rfl, rfl⟩; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.parabolic_degenerate_iff, dg-me-parabolic-degenerate-test]

/-! ## exr-line-element-symmetrised -/

/-- Polarisation: a symmetric matrix is determined by its quadratic form,
`g(X, Y) = ½ (g(X + Y, X + Y) − g(X, X) − g(Y, Y))`. -/
theorem polarisation {n : ℕ} (g : Matrix (Fin n) (Fin n) ℝ) (hg : gᵀ = g) (X Y : Fin n → ℝ) :
    X ⬝ᵥ (g *ᵥ Y) = ((X + Y) ⬝ᵥ (g *ᵥ (X + Y)) - X ⬝ᵥ (g *ᵥ X) - Y ⬝ᵥ (g *ᵥ Y)) / 2 := by
  have hYX : Y ⬝ᵥ (g *ᵥ X) = X ⬝ᵥ (g *ᵥ Y) := by
    rw [dotProduct_mulVec, ← mulVec_transpose, hg, dotProduct_comm]
  rw [Matrix.mulVec_add, dotProduct_add, add_dotProduct, add_dotProduct, hYX]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.polarisation, dg-me-polarisation-test]

/-- An antisymmetric part is invisible in `ds²`: `a_ij X^i X^j = 0` for `aᵀ = −a`. -/
theorem antisymmetric_invisible {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (hA : Aᵀ = -A) (X : Fin n → ℝ) :
    X ⬝ᵥ (A *ᵥ X) = 0 := by
  have h : X ⬝ᵥ (A *ᵥ X) = -(X ⬝ᵥ (A *ᵥ X)) := by
    conv_lhs => rw [dotProduct_mulVec, ← mulVec_transpose, hA, Matrix.neg_mulVec, neg_dotProduct, dotProduct_comm]
  linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.antisymmetric_invisible, dg-me-antisymmetric-test]

/-! ## def-inverse-metric -/

/-- The inverse metric exists because the metric is non-degenerate: `g⁻¹ g = 1` when `det g ≠ 0`. -/
theorem inverse_mul_self {n : ℕ} (g : Matrix (Fin n) (Fin n) ℝ) (hg : g.det ≠ 0) : g⁻¹ * g = 1 :=
  Matrix.nonsing_inv_mul g (isUnit_iff_ne_zero.mpr hg)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.inverse_mul_self, dg-me-inverse-mul-test]

/-- The inverse of a symmetric matrix is symmetric: `g^ij = g^ji`. -/
theorem inverse_symm {n : ℕ} (g : Matrix (Fin n) (Fin n) ℝ) (hg : gᵀ = g) : (g⁻¹)ᵀ = g⁻¹ := by
  rw [← Matrix.transpose_nonsing_inv, hg]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.inverse_symm, dg-me-inverse-symm-test]

/-! ## exm-inverse-diagonal -/

/-- The inverse of the round metric is `diag(1, 1/sin²θ)` on the chart domain. -/
theorem sphere_inv_mul (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    metricAt sphere x * !![1, 0; 0, 1 / Real.sin (x 0) ^ 2] = 1 := by
  rw [sphere_metricAt]
  VERIPHY[dg-me-matrix]

/-- The Minkowski metric is its own inverse: `η η = 1`. -/
theorem eta_mul_eta :
    !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = 1 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-inverse-cross -/

/-- The block `[[1, cos θ], [cos θ, 1]]` has inverse `(1/sin²θ) [[1, −cos θ], [−cos θ, 1]]` where `sin θ ≠ 0`. -/
theorem cross_block_inverse (θ : ℝ) (h : Real.sin θ ≠ 0) :
    !![(1 : ℝ), Real.cos θ; Real.cos θ, 1] * ((1 / Real.sin θ ^ 2) • !![(1 : ℝ), -Real.cos θ; -Real.cos θ, 1]) = 1 := by
  VERIPHY[dg-me-matrix]

/-- Its determinant is `1 − cos²θ = sin²θ`, so the inverse exists exactly where `sin θ ≠ 0`. -/
theorem cross_block_det (θ : ℝ) : (!![(1 : ℝ), Real.cos θ; Real.cos θ, 1]).det = Real.sin θ ^ 2 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-inverse-conformal -/

/-- The inverse of a conformal metric `Ω² δ_ij` is `Ω⁻² δ^ij`. -/
theorem conformal_inverse (Ω : ℝ) (h : Ω ≠ 0) :
    !![Ω ^ 2, 0; 0, Ω ^ 2] * !![1 / Ω ^ 2, 0; 0, 1 / Ω ^ 2] = 1 := by
  VERIPHY[dg-me-matrix]

/-- The stereographic conformal factor `Ω = 2/(1 + u² + v²)` is positive, so the inverse exists everywhere. -/
theorem stereo_factor_pos (u v : ℝ) : 0 < 2 / (1 + u ^ 2 + v ^ 2) := by positivity
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.stereo_factor_pos, dg-me-stereo-factor-test]

/-! ## exr-cross-term-inverse -/

/-- `det (¼ [[1, cos θ], [cos θ, 1]]) = (1 − cos²θ)/16 = sin²θ/16`. -/
theorem quarter_block_det (θ : ℝ) :
    (!![(1 / 4 : ℝ), Real.cos θ / 4; Real.cos θ / 4, 1 / 4]).det = Real.sin θ ^ 2 / 16 := by
  VERIPHY[dg-me-matrix]

/-- The inverse is `(4/sin²θ) [[1, −cos θ], [−cos θ, 1]]` where `sin θ ≠ 0`. -/
theorem quarter_block_inverse (θ : ℝ) (h : Real.sin θ ≠ 0) :
    !![(1 / 4 : ℝ), Real.cos θ / 4; Real.cos θ / 4, 1 / 4] *
      ((4 / Real.sin θ ^ 2) • !![(1 : ℝ), -Real.cos θ; -Real.cos θ, 1]) = 1 := by
  VERIPHY[dg-me-matrix]

/-- At `θ = 0` (`cos 0 = 1`) the two rows coincide and the determinant vanishes: the Euler chart degenerates. -/
theorem quarter_block_degenerate : (!![(1 / 4 : ℝ), 1 / 4; 1 / 4, 1 / 4]).det = 0 := by
  VERIPHY[dg-me-matrix]

/-! ## exr-inverse-metric-check -/

/-- `[[4, 3], [3, 9]]⁻¹ = (1/27) [[9, −3], [−3, 4]]`. -/
theorem readoff_first_inverse : !![(4 : ℝ), 3; 3, 9] * ((1 / 27 : ℝ) • !![(9 : ℝ), -3; -3, 4]) = 1 := by
  VERIPHY[dg-me-matrix]

/-- `[[1, −1], [−1, 0]]⁻¹ = [[0, −1], [−1, −1]]`: the inverse acquires an `xx` entry. -/
theorem readoff_second_inverse : !![(1 : ℝ), -1; -1, 0] * !![(0 : ℝ), -1; -1, -1] = 1 := by
  VERIPHY[dg-me-matrix]

/-- The characteristic polynomial of `[[1, −1], [−1, 0]]` is `λ² − λ − 1`, whose roots `(1 ± √5)/2` have
product `−1`: eigenvalues of opposite signs, Lorentzian. -/
theorem readoff_second_charpoly (l : ℝ) : (!![(1 : ℝ) - l, -1; -1, -l]).det = l ^ 2 - l - 1 := by
  VERIPHY[dg-me-matrix]

/-- `(1 + √5)/2` and `(1 − √5)/2` are the roots of `λ² − λ − 1`. -/
theorem readoff_second_eigenvalues :
    ((1 + Real.sqrt 5) / 2) ^ 2 - (1 + Real.sqrt 5) / 2 - 1 = 0 ∧
      ((1 - Real.sqrt 5) / 2) ^ 2 - (1 - Real.sqrt 5) / 2 - 1 = 0 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-trig]
  · VERIPHY[dg-me-trig]

/-! ## exr-inverse-metric-tensor -/

/-- The inverse metric transforms with two inverse Jacobians: `(Jᵀ g J)⁻¹ = J⁻¹ g⁻¹ J⁻ᵀ`. -/
theorem inverse_congr {n : ℕ} (J g : Matrix (Fin n) (Fin n) ℝ) : (Jᵀ * g * J)⁻¹ = J⁻¹ * g⁻¹ * (Jᵀ)⁻¹ := by
  rw [Matrix.mul_inv_rev, Matrix.mul_inv_rev, Matrix.mul_assoc]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.inverse_congr, dg-me-inverse-congr-test]

/-- The inverse of a block-diagonal metric is block diagonal with the inverses of the blocks (a `2 + 1` block
example: `A = [[a, b], [c, d]]`, `B = (e)`). -/
theorem block_inverse (a b c d e : ℝ) (hA : a * d - b * c ≠ 0) (he : e ≠ 0) :
    !![a, b, 0; c, d, 0; 0, 0, e] *
      !![d / (a * d - b * c), -b / (a * d - b * c), 0; -c / (a * d - b * c), a / (a * d - b * c), 0; 0, 0, 1 / e] =
      1 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-polar-metric-points -/

/-- At `r = 1` the polar matrix is the identity; at `r = 2` it is `diag(1, 4)`, with `|∂_φ|² = 4`. -/
theorem polar_at_points :
    !![(1 : ℝ), 0; 0, (1 : ℝ) ^ 2] = 1 ∧ !![(1 : ℝ), 0; 0, (2 : ℝ) ^ 2] = !![1, 0; 0, 4] ∧
      ![(0 : ℝ), 1] ⬝ᵥ (!![(1 : ℝ), 0; 0, (2 : ℝ) ^ 2] *ᵥ ![0, 1]) = 2 ^ 2 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- At `r = ½` the `φ` component is `¼ < 1`: the unit ellipse is elongated along `φ`. -/
theorem polar_at_half : ![(0 : ℝ), 1] ⬝ᵥ (!![(1 : ℝ), 0; 0, (1 / 2 : ℝ) ^ 2] *ᵥ ![0, 1]) = 1 / 4 := by
  VERIPHY[dg-me-matrix]

/-- The determinant `r²` is continuous and tends to `a²` as `r → a`: to zero at the origin, where the chart stops
being one-to-one. -/
theorem polar_det_tendsto (a : ℝ) : Tendsto (fun r : ℝ => r ^ 2) (𝓝 a) (𝓝 (a ^ 2)) :=
  (continuous_pow 2).tendsto a
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.polar_det_tendsto, dg-me-polar-det-tendsto-test]

/-! ## exm-polar-metric-lean -/

/-- `det diag(1, r²) = 1 · r²`: the determinant of a diagonal matrix is the product of its entries. -/
theorem polar_det_product (r : ℝ) : (!![(1 : ℝ), 0; 0, r ^ 2]).det = 1 * r ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- At `r = 0` no inverse exists: for every `M`, `M diag(1, 0)` has a zero second column, so it is never `1`. -/
theorem no_inverse_at_origin (M : Matrix (Fin 2) (Fin 2) ℝ) : M * !![(1 : ℝ), 0; 0, 0] ≠ 1 := by
  intro h
  have := congrFun (congrFun h 1) 1
  simp [Matrix.mul_apply, Fin.sum_univ_succ] at this
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.no_inverse_at_origin, dg-me-no-inverse-origin-test]

/-- `|∂_φ|² = r²` vanishes at the origin. -/
theorem dphi_norm_sq_polar (r : ℝ) : ![(0 : ℝ), 1] ⬝ᵥ (!![(1 : ℝ), 0; 0, r ^ 2] *ᵥ ![0, 1]) = r ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The inverse component `c/r²` (`c > 0`; `c = 1` is the squared length of `dφ`) blows up at the origin. -/
theorem inverse_component_blowup (c : ℝ) (hc : 0 < c) : Tendsto (fun r : ℝ => c / r ^ 2) (𝓝[≠] 0) atTop := by
  have h0 : Tendsto (fun r : ℝ => r ^ 2) (𝓝 0) (𝓝 0) := by
    simpa using (continuous_pow 2).tendsto (0 : ℝ)
  have h1 : Tendsto (fun r : ℝ => r ^ 2) (𝓝[≠] 0) (𝓝[>] 0) := by
    refine tendsto_nhdsWithin_iff.mpr ⟨h0.mono_left nhdsWithin_le_nhds, ?_⟩
    exact eventually_nhdsWithin_of_forall fun r hr => by
      have : (r : ℝ) ≠ 0 := hr
      exact lt_of_le_of_ne (sq_nonneg r) (Ne.symm (pow_ne_zero 2 this))
  have h2 : Tendsto (fun r : ℝ => (r ^ 2)⁻¹) (𝓝[≠] 0) atTop := tendsto_inv_nhdsGT_zero.comp h1
  simpa [div_eq_mul_inv] using h2.const_mul_atTop hc
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.inverse_component_blowup, dg-me-blowup-test]

/-! ## exm-polar-metric-cylindrical -/

/-- Cylindrical coordinates: the Jacobian of `(r cos φ, r sin φ, z)` gives `Jᵀ J = diag(1, r², 1)`. -/
theorem cylindrical_metric (r φ : ℝ) :
    (!![Real.cos φ, -(r * Real.sin φ), 0; Real.sin φ, r * Real.cos φ, 0; 0, 0, 1])ᵀ *
      !![Real.cos φ, -(r * Real.sin φ), 0; Real.sin φ, r * Real.cos φ, 0; 0, 0, 1] = !![1, 0, 0; 0, r ^ 2, 0; 0, 0, 1] := by
  VERIPHY[dg-me-matrix]

/-- `det diag(1, r², 1) = r²`. -/
theorem cylindrical_det (r : ℝ) : (!![(1 : ℝ), 0, 0; 0, r ^ 2, 0; 0, 0, 1]).det = r ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The inverse is `diag(1, 1/r², 1)` for `r ≠ 0`. -/
theorem cylindrical_inverse (r : ℝ) (h : r ≠ 0) :
    !![(1 : ℝ), 0, 0; 0, r ^ 2, 0; 0, 0, 1] * !![1, 0, 0; 0, 1 / r ^ 2, 0; 0, 0, 1] = 1 := by
  VERIPHY[dg-me-matrix]

/-! ## exr-polar-metric-spherical -/

/-- Spherical coordinates on `ℝ³`: `Jᵀ J = diag(1, ρ², ρ² sin²θ)`. -/
theorem spherical_metric (ρ θ φ : ℝ) :
    (!![Real.sin θ * Real.cos φ, ρ * Real.cos θ * Real.cos φ, -(ρ * Real.sin θ * Real.sin φ);
        Real.sin θ * Real.sin φ, ρ * Real.cos θ * Real.sin φ, ρ * Real.sin θ * Real.cos φ;
        Real.cos θ, -(ρ * Real.sin θ), 0])ᵀ *
      !![Real.sin θ * Real.cos φ, ρ * Real.cos θ * Real.cos φ, -(ρ * Real.sin θ * Real.sin φ);
        Real.sin θ * Real.sin φ, ρ * Real.cos θ * Real.sin φ, ρ * Real.sin θ * Real.cos φ;
        Real.cos θ, -(ρ * Real.sin θ), 0] =
      !![1, 0, 0; 0, ρ ^ 2, 0; 0, 0, ρ ^ 2 * Real.sin θ ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- `det g = ρ⁴ sin²θ` in spherical coordinates. -/
theorem spherical_det (ρ θ : ℝ) :
    (!![(1 : ℝ), 0, 0; 0, ρ ^ 2, 0; 0, 0, ρ ^ 2 * Real.sin θ ^ 2]).det = ρ ^ 4 * Real.sin θ ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The inverse is `diag(1, ρ⁻², ρ⁻² sin⁻²θ)` on the chart domain. -/
theorem spherical_inverse (ρ θ : ℝ) (hρ : ρ ≠ 0) (hθ : Real.sin θ ≠ 0) :
    !![(1 : ℝ), 0, 0; 0, ρ ^ 2, 0; 0, 0, ρ ^ 2 * Real.sin θ ^ 2] *
      !![1, 0, 0; 0, 1 / ρ ^ 2, 0; 0, 0, 1 / (ρ ^ 2 * Real.sin θ ^ 2)] = 1 := by
  VERIPHY[dg-me-matrix]

/-- The angular block is `ρ²` times the round metric. -/
theorem spherical_angular_block (ρ θ : ℝ) :
    !![ρ ^ 2, 0; 0, ρ ^ 2 * Real.sin θ ^ 2] = ρ ^ 2 • !![(1 : ℝ), 0; 0, Real.sin θ ^ 2] := by
  VERIPHY[dg-me-matrix]

/-! ## exr-log-polar -/

/-- `r = e^u` gives `dr = e^u du`. -/
theorem log_polar_dr (u : ℝ) : deriv (fun u : ℝ => Real.exp u) u = Real.exp u := by
  VERIPHY[de-derivative]

/-- In the coordinates `(u, φ)` the polar metric becomes `e^{2u}(du² + dφ²)`, conformal to the flat cylinder. -/
theorem log_polar_metric (u : ℝ) :
    (!![Real.exp u, 0; 0, 1])ᵀ * !![(1 : ℝ), 0; 0, Real.exp u ^ 2] * !![Real.exp u, 0; 0, 1] =
      !![Real.exp u ^ 2, 0; 0, Real.exp u ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- `e^{2u} = (e^u)²`. -/
theorem exp_two_mul_sq (u : ℝ) : Real.exp (2 * u) = Real.exp u ^ 2 := by
  VERIPHY[dg-me-trig]

/-- `det g = e^{4u}` in log-polar coordinates. -/
theorem log_polar_det (u : ℝ) : (!![Real.exp u ^ 2, 0; 0, Real.exp u ^ 2]).det = Real.exp u ^ 4 := by
  VERIPHY[dg-me-matrix]

/-- The inverse `e^{−2u} δ^ij` exists for every `u`, and dividing by `e^{2u}` gives the flat cylinder metric. -/
theorem log_polar_inverse (u : ℝ) :
    !![Real.exp u ^ 2, 0; 0, Real.exp u ^ 2] * !![1 / Real.exp u ^ 2, 0; 0, 1 / Real.exp u ^ 2] = 1 ∧
      (1 / Real.exp u ^ 2) • !![Real.exp u ^ 2, 0; 0, Real.exp u ^ 2] = 1 := by
  have h : Real.exp u ≠ 0 := Real.exp_ne_zero u
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-! ## exr-orthogonal-chart-scale-factors -/

/-- For a diagonal metric `det g = h₁² ⋯ hₙ²`. -/
theorem diagonal_det {n : ℕ} (h : Fin n → ℝ) : (Matrix.diagonal fun i => h i ^ 2).det = ∏ i, h i ^ 2 := by
  rw [Matrix.det_diagonal]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.diagonal_det, dg-me-diagonal-det-test]

/-- The volume element is `√(det g) = h₁ ⋯ hₙ` for positive scale factors. -/
theorem diagonal_volume {n : ℕ} (h : Fin n → ℝ) (hh : ∀ i, 0 ≤ h i) : Real.sqrt (∏ i, h i ^ 2) = ∏ i, h i := by
  rw [← Finset.prod_pow, Real.sqrt_sq (Finset.prod_nonneg fun i _ => hh i)]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.diagonal_volume, dg-me-diagonal-volume-test]

/-- The inverse of a diagonal metric is the diagonal of reciprocals `diag(h₁⁻², …, hₙ⁻²)`. -/
theorem diagonal_inverse {n : ℕ} (h : Fin n → ℝ) (hh : ∀ i, h i ≠ 0) :
    (Matrix.diagonal fun i => h i ^ 2) * Matrix.diagonal (fun i => 1 / h i ^ 2) = 1 := by
  rw [Matrix.diagonal_mul_diagonal, ← Matrix.diagonal_one]
  congr 1; ext i; field_simp [hh i]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.diagonal_inverse, dg-me-diagonal-inverse-test]

/-- The scale factors of the polar, cylindrical, spherical and round metrics: `(1, r)`, `(1, r, 1)`,
`(1, ρ, ρ sin θ)`, `(1, sin θ)`, as the diagonal square roots of the metric matrices. -/
theorem scale_factors (r ρ θ : ℝ) :
    !![(1 : ℝ), 0; 0, r] * !![(1 : ℝ), 0; 0, r] = !![1, 0; 0, r ^ 2] ∧
    !![(1 : ℝ), 0, 0; 0, r, 0; 0, 0, 1] * !![(1 : ℝ), 0, 0; 0, r, 0; 0, 0, 1] = !![1, 0, 0; 0, r ^ 2, 0; 0, 0, 1] ∧
    !![(1 : ℝ), 0, 0; 0, ρ, 0; 0, 0, ρ * Real.sin θ] * !![(1 : ℝ), 0, 0; 0, ρ, 0; 0, 0, ρ * Real.sin θ] =
      !![1, 0, 0; 0, ρ ^ 2, 0; 0, 0, ρ ^ 2 * Real.sin θ ^ 2] ∧
    !![(1 : ℝ), 0; 0, Real.sin θ] * !![(1 : ℝ), 0; 0, Real.sin θ] = !![1, 0; 0, Real.sin θ ^ 2] := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-! ## prp-minkowski-metric -/

/-- `det η = −1 = 1 · (−1)³`: the product of the diagonal entries. -/
theorem eta_det_product : (1 : ℝ) * (-1) ^ 3 = -1 := by
  VERIPHY[de-poly]

/-! ## exm-minkowski-det-sign -/

/-- The sign of a product of non-zero numbers is `(−1)^q` with `q` the number of negative factors:
`∏ dᵢ = (∏ |dᵢ|) · (−1)^q`. -/
theorem prod_sign {n : ℕ} (d : Fin n → ℝ) :
    ∏ i, d i = (∏ i, |d i|) * (-1) ^ (Finset.univ.filter fun i => d i < 0).card := by
  have h : ∀ i, d i = |d i| * (if d i < 0 then -1 else 1) := by
    intro i; split_ifs with hi
    · rw [abs_of_neg hi]; ring
    · rw [abs_of_nonneg (not_lt.mp hi)]; ring
  conv_lhs => rw [Finset.prod_congr rfl fun i _ => h i]
  rw [Finset.prod_mul_distrib, Finset.prod_ite, Finset.prod_const, Finset.prod_const_one, mul_one]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.prod_sign, dg-me-prod-sign-test]

/-- A negative determinant stays negative in every chart: `det (Jᵀ g J) = (det J)² det g < 0` for `det J ≠ 0`. -/
theorem det_congr_neg {n : ℕ} (J g : Matrix (Fin n) (Fin n) ℝ) (hJ : J.det ≠ 0) (hg : g.det < 0) :
    (Jᵀ * g * J).det < 0 := by
  rw [det_congr]
  have := pow_pos (abs_pos.mpr hJ) 2; rw [sq_abs] at this
  nlinarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.det_congr_neg, dg-me-det-congr-neg-test]

/-! ## exm-minkowski-boost -/

/-- `Λᵀ η Λ = η` with `det η ≠ 0` gives `(det Λ)² = 1`, so `det Λ = ±1`. -/
theorem lorentz_det {n : ℕ} (Λ η : Matrix (Fin n) (Fin n) ℝ) (h : Λᵀ * η * Λ = η) (hη : η.det ≠ 0) :
    Λ.det = 1 ∨ Λ.det = -1 := by
  have h2 : Λ.det ^ 2 = 1 := by
    have := det_congr Λ η
    rw [h] at this
    have h3 : (Λ.det ^ 2 - 1) * η.det = 0 := by linear_combination -this
    rcases mul_eq_zero.mp h3 with h4 | h4
    · linarith
    · exact absurd h4 hη
  have h3 : (Λ.det - 1) * (Λ.det + 1) = 0 := by linear_combination h2
  rcases mul_eq_zero.mp h3 with h4 | h4
  · left; linarith
  · right; linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.lorentz_det, dg-me-lorentz-det-test]

/-- The boost in the `(t, x)` plane has `det Λ = cosh²ξ − sinh²ξ = 1`. -/
theorem boost_det (ξ : ℝ) : (!![Real.cosh ξ, Real.sinh ξ; Real.sinh ξ, Real.cosh ξ]).det = 1 := by
  VERIPHY[dg-me-matrix]

/-- The boost preserves the two-dimensional Minkowski metric: `Λᵀ η Λ = η`. -/
theorem boost_preserves_eta2 (ξ : ℝ) :
    (!![Real.cosh ξ, Real.sinh ξ; Real.sinh ξ, Real.cosh ξ])ᵀ * !![(1 : ℝ), 0; 0, -1] *
      !![Real.cosh ξ, Real.sinh ξ; Real.sinh ξ, Real.cosh ξ] = !![1, 0; 0, -1] := by
  VERIPHY[dg-me-matrix]

/-- The transformed metric has the same determinant `−1`. -/
theorem eta2_det : (!![(1 : ℝ), 0; 0, -1]).det = -1 := by
  VERIPHY[dg-me-matrix]

/-! ## exm-minkowski-five -/

/-- `det diag(1, −1, −1, −1, −R²) = R²`. -/
theorem kk_circle_det (R : ℝ) :
    (!![(1 : ℝ), 0, 0, 0, 0; 0, -1, 0, 0, 0; 0, 0, -1, 0, 0; 0, 0, 0, -1, 0; 0, 0, 0, 0, -R ^ 2]).det = R ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- `(−1)³ · (−R²) = R²`: four negative directions give a positive determinant. -/
theorem kk_circle_det_product (R : ℝ) : (-1 : ℝ) ^ 3 * (-R ^ 2) = R ^ 2 := by
  VERIPHY[de-poly]

/-- `√|det g| = √|R²| = R` for `R ≥ 0`: the volume element `R d⁴x dφ`. -/
theorem kk_circle_volume_element (R : ℝ) (h : 0 ≤ R) : Real.sqrt |R ^ 2| = R := by
  VERIPHY[dg-me-trig]

/-- `−det g = −R² < 0`: `√(−g)` would be imaginary in odd Lorentzian dimension. -/
theorem kk_circle_neg_det (R : ℝ) (hR : R ≠ 0) : -R ^ 2 < 0 := by
  have := pow_pos (abs_pos.mpr hR) 2; rw [sq_abs] at this; linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.kk_circle_neg_det, dg-me-kk-neg-det-test]

/-! ## exr-minkowski-det-sage -/

/-- The four-dimensional boost preserves `η`: `Λᵀ η Λ = η`. -/
theorem boost_preserves_eta4 (ξ : ℝ) :
    (!![Real.cosh ξ, Real.sinh ξ, 0, 0; Real.sinh ξ, Real.cosh ξ, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1])ᵀ *
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *
      !![Real.cosh ξ, Real.sinh ξ, 0, 0; Real.sinh ξ, Real.cosh ξ, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] =
      !![(1 : ℝ), 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] := by
  VERIPHY[dg-me-matrix]

/-- `det Λ = 1` for the four-dimensional boost. -/
theorem boost4_det (ξ : ℝ) :
    (!![Real.cosh ξ, Real.sinh ξ, 0, 0; Real.sinh ξ, Real.cosh ξ, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1]).det = 1 := by
  VERIPHY[dg-me-matrix]

/-! ## exr-minkowski-lightcone -/

/-- `dt² − dx² = (dt − dx)(dt + dx) = du dv`. -/
theorem lightcone_factor (t x : ℝ) : t ^ 2 - x ^ 2 = (t - x) * (t + x) := by
  VERIPHY[de-poly]

/-- The Jacobian `∂(t, x)/∂(u, v) = ½ [[1, 1], [−1, 1]]` has determinant `½`, and the inverse Jacobian
`∂(u, v)/∂(t, x) = [[1, −1], [1, 1]]` has determinant `2`. -/
theorem lightcone_jacobian_dets :
    (!![(1 / 2 : ℝ), 1 / 2; -1 / 2, 1 / 2]).det = 1 / 2 ∧ (!![(1 : ℝ), -1; 1, 1]).det = 2 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- In light-cone coordinates the `(t, x)` block becomes `Jᵀ diag(1, −1) J = [[0, ½], [½, 0]]`. -/
theorem lightcone_block (u v : ℝ) :
    (!![(1 / 2 : ℝ), 1 / 2; -1 / 2, 1 / 2])ᵀ * !![(1 : ℝ), 0; 0, -1] * !![(1 / 2 : ℝ), 1 / 2; -1 / 2, 1 / 2] =
      !![0, 1 / 2; 1 / 2, 0] := by
  VERIPHY[dg-me-matrix]

/-- The light-cone matrix has `det g = −¼ = (½)² · (−1)` and inverse with blocks `[[0, 2], [2, 0]]` and `−1₂`. -/
theorem lightcone_det_inverse :
    (!![(0 : ℝ), 1 / 2, 0, 0; 1 / 2, 0, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]).det = -1 / 4 ∧
      !![(0 : ℝ), 1 / 2, 0, 0; 1 / 2, 0, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] *
        !![(0 : ℝ), 2, 0, 0; 2, 0, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1] = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- The coordinate vectors `∂_u` and `∂_v` are null. -/
theorem lightcone_null :
    ![(1 : ℝ), 0] ⬝ᵥ (!![(0 : ℝ), 1 / 2; 1 / 2, 0] *ᵥ ![1, 0]) = 0 ∧
      ![(0 : ℝ), 1] ⬝ᵥ (!![(0 : ℝ), 1 / 2; 1 / 2, 0] *ᵥ ![0, 1]) = 0 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-! ## exr-minkowski-det-sign -/

/-- Lorentzian signature `(1, n − 1)`: `sign det g = (−1)^(n−1)`, negative for even `n`, positive for odd `n`. -/
theorem lorentzian_det_sign (n : ℕ) (hn : 0 < n) : (-1 : ℝ) ^ (n - 1) = if Even n then -1 else 1 := by
  split_ifs with h
  · exact (Nat.Even.sub_odd hn h odd_one).neg_one_pow
  · exact (Nat.Odd.sub_odd (Nat.not_even_iff_odd.mp h) odd_one).neg_one_pow
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.lorentzian_det_sign, dg-me-lorentzian-sign-test]

/-- The two cases: `n = 4` gives `det η = −1`, `n = 5` gives `R² > 0`. -/
theorem lorentzian_sign_cases : (-1 : ℝ) ^ 3 = -1 ∧ (-1 : ℝ) ^ 4 = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]

/-! ## exm-sphere-metric-latitudes -/

/-- On the equator `θ = π/2` the round metric is the identity and the parallel has length `2π`. -/
theorem sphere_equator_values :
    Real.sin (Real.pi / 2) ^ 2 = 1 ∧ 2 * Real.pi * Real.sin (Real.pi / 2) = 2 * Real.pi := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-trig]
  · VERIPHY[dg-me-trig]

/-- At `θ = π/6`: `sin²θ = ¼`, the inverse component is `4`, `|∂_φ| = ½` and the parallel has length `π`. -/
theorem sphere_pi_six_values :
    Real.sin (Real.pi / 6) ^ 2 = 1 / 4 ∧ 1 / Real.sin (Real.pi / 6) ^ 2 = 4 ∧
      Real.sqrt (Real.sin (Real.pi / 6) ^ 2) = 1 / 2 ∧ 2 * Real.pi * Real.sin (Real.pi / 6) = Real.pi := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · VERIPHY[dg-me-trig]
  · VERIPHY[dg-me-trig]
  · VERIPHY[dg-me-trig]
  · VERIPHY[dg-me-trig]

/-- The determinant `sin²θ` is continuous, tending to `sin²a` as `θ → a`: to zero at the poles. -/
theorem sphere_det_tendsto_pole (a : ℝ) : Tendsto (fun θ : ℝ => Real.sin θ ^ 2) (𝓝 a) (𝓝 (Real.sin a ^ 2)) :=
  (Real.continuous_sin.pow 2).tendsto a
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.sphere_det_tendsto_pole, dg-me-sphere-det-pole-test]

/-! ## exm-sphere-metric-radius -/

/-- `det (R² diag(1, sin²θ)) = R⁴ sin²θ`. -/
theorem sphere_radius_det (R θ : ℝ) : (!![R ^ 2, 0; 0, R ^ 2 * Real.sin θ ^ 2]).det = R ^ 4 * Real.sin θ ^ 2 := by
  VERIPHY[dg-me-matrix]

/-- The inverse is `R⁻² diag(1, 1/sin²θ)`. -/
theorem sphere_radius_inverse (R θ : ℝ) (hR : R ≠ 0) (h : Real.sin θ ≠ 0) :
    !![R ^ 2, 0; 0, R ^ 2 * Real.sin θ ^ 2] * ((1 / R ^ 2) • !![(1 : ℝ), 0; 0, 1 / Real.sin θ ^ 2]) = 1 := by
  VERIPHY[dg-me-matrix]

/-- Multiplying an `n`-dimensional metric by `λ` multiplies its determinant by `λⁿ`. -/
theorem det_smul_metric {n : ℕ} (l : ℝ) (g : Matrix (Fin n) (Fin n) ℝ) : (l • g).det = l ^ n * g.det := by
  rw [Matrix.det_smul, Fintype.card_fin]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.det_smul_metric, dg-me-det-smul-test]

/-- Multiplying a metric by `λ ≠ 0` multiplies its inverse by `λ⁻¹`: `(λ g)(λ⁻¹ g⁻¹) = 1`. -/
theorem inverse_smul_metric {n : ℕ} (l : ℝ) (hl : l ≠ 0) (g : Matrix (Fin n) (Fin n) ℝ) (hg : g.det ≠ 0) :
    (l • g) * (l⁻¹ • g⁻¹) = 1 := by
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, mul_inv_cancel₀ hl, one_smul,
    Matrix.mul_nonsing_inv g (isUnit_iff_ne_zero.mpr hg)]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.inverse_smul_metric, dg-me-inverse-smul-test]

/-- `√(det g) = R² sin θ` for the sphere of radius `R` (`sin θ ≥ 0` on the chart). -/
theorem sphere_radius_volume_element (R θ : ℝ) (h : 0 ≤ Real.sin θ) :
    Real.sqrt (R ^ 4 * Real.sin θ ^ 2) = R ^ 2 * Real.sin θ := by
  VERIPHY[dg-me-trig]

/-- Its area: `∫₀^π R² sin θ dθ = 2R²`, times `2π` from `φ`, gives `4πR²`. -/
theorem sphere_radius_area (R : ℝ) :
    (∫ θ in (0 : ℝ)..Real.pi, R ^ 2 * Real.sin θ) = 2 * R ^ 2 ∧ 2 * R ^ 2 * (2 * Real.pi) = 4 * Real.pi * R ^ 2 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-integral]
  · VERIPHY[de-poly]

/-! ## exm-sphere-metric-stereographic -/

/-- `det (4 δ_ij / (1 + u² + v²)²) = 16 / (1 + u² + v²)⁴`. -/
theorem stereo_det (u v : ℝ) :
    (!![4 / (1 + u ^ 2 + v ^ 2) ^ 2, 0; 0, 4 / (1 + u ^ 2 + v ^ 2) ^ 2]).det = 16 / (1 + u ^ 2 + v ^ 2) ^ 4 := by
  have hD : 1 + u ^ 2 + v ^ 2 ≠ 0 := by positivity
  rw [← sub_eq_zero]
  VERIPHY[dg-me-matrix]

/-- The inverse is `¼ (1 + u² + v²)² δ^ij`, finite at every point of the chart. -/
theorem stereo_inverse (u v : ℝ) :
    !![4 / (1 + u ^ 2 + v ^ 2) ^ 2, 0; 0, 4 / (1 + u ^ 2 + v ^ 2) ^ 2] *
      !![(1 + u ^ 2 + v ^ 2) ^ 2 / 4, 0; 0, (1 + u ^ 2 + v ^ 2) ^ 2 / 4] = 1 := by
  have hD : 1 + u ^ 2 + v ^ 2 ≠ 0 := by positivity
  rw [← sub_eq_zero]
  VERIPHY[dg-me-matrix]

/-- At `u = v = 0`, the pole the spherical chart misses, the matrix is `diag(4, 4)` with determinant `16`. -/
theorem stereo_at_pole :
    !![4 / (1 + (0 : ℝ) ^ 2 + (0 : ℝ) ^ 2) ^ 2, 0; 0, 4 / (1 + (0 : ℝ) ^ 2 + (0 : ℝ) ^ 2) ^ 2] = !![4, 0; 0, 4] ∧
      (!![(4 : ℝ), 0; 0, 4]).det = 16 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- The stereographic components are finite and non-zero everywhere: `0 < 4/(1 + u² + v²)²`. -/
theorem stereo_component_pos (u v : ℝ) : 0 < 4 / (1 + u ^ 2 + v ^ 2) ^ 2 := by positivity
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.stereo_component_pos, dg-me-stereo-component-test]

/-! ## exr-sphere-metric-stereographic -/

/-- The image of the inverse stereographic projection lies on the unit sphere: `Σ_k E^k(u, v)² = 1`. -/
theorem stereoEmbed_on_sphere (x : Fin 2 → ℝ) : ∑ k, stereoEmbed k x ^ 2 = 1 := by
  have hD : 1 + x 0 ^ 2 + x 1 ^ 2 ≠ 0 := by positivity
  simp only [stereoEmbed, Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero, Matrix.cons_val_succ,
    Matrix.cons_val_one, Matrix.head_cons, add_zero]
  rw [← sub_eq_zero]
  VERIPHY[de-poly]

/-! ## exr-sphere-metric-height-chart -/

/-- `z = cos θ` gives `dz = −sin θ dθ`. -/
theorem height_chart_dz (θ : ℝ) : deriv (fun θ : ℝ => Real.cos θ) θ = -Real.sin θ := by
  VERIPHY[de-derivative]

/-- `sin²θ = 1 − z²` with `z = cos θ`. -/
theorem height_chart_sin_sq (θ : ℝ) : Real.sin θ ^ 2 = 1 - Real.cos θ ^ 2 := by
  VERIPHY[dg-me-trig]

/-- In the height chart `ds² = dz²/(1 − z²) + (1 − z²) dφ²` has `det g = 1` and inverse `diag(1 − z², 1/(1 − z²))`. -/
theorem height_chart_det_inverse (z : ℝ) (hz : 1 - z ^ 2 ≠ 0) :
    (!![1 / (1 - z ^ 2), 0; 0, 1 - z ^ 2]).det = 1 ∧
      !![1 / (1 - z ^ 2), 0; 0, 1 - z ^ 2] * !![1 - z ^ 2, 0; 0, 1 / (1 - z ^ 2)] = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- Pulling the height-chart metric back through `z = cos θ` (Jacobian `diag(−sin θ, 1)`) returns the round
metric `diag(1, sin²θ)`. -/
theorem height_chart_pullback (θ : ℝ) (h : Real.sin θ ≠ 0) :
    (!![-Real.sin θ, 0; 0, 1])ᵀ * !![1 / (1 - Real.cos θ ^ 2), 0; 0, 1 - Real.cos θ ^ 2] * !![-Real.sin θ, 0; 0, 1] =
      !![1, 0; 0, Real.sin θ ^ 2] := by
  rw [← Real.sin_sq]
  VERIPHY[dg-me-matrix]

/-- The volume element is `dz dφ`: the zone `z₁ ≤ z ≤ z₂` has area `∫_{z₁}^{z₂} 2π dz = 2π(z₂ − z₁)` (Archimedes),
`4π` for the whole sphere. -/
theorem zone_area (z₁ z₂ : ℝ) :
    (∫ z in z₁..z₂, 2 * Real.pi) = 2 * Real.pi * (z₂ - z₁) ∧ (∫ z in (-1 : ℝ)..1, 2 * Real.pi) = 4 * Real.pi := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-integral]
  · VERIPHY[dg-me-integral]

/-- The chart is valid for `−1 < z < 1`, where `1 − z² > 0`. -/
theorem height_chart_domain (z : ℝ) (hz : -1 < z ∧ z < 1) : 0 < 1 - z ^ 2 := by nlinarith [hz.1, hz.2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.height_chart_domain, dg-me-height-domain-test]

/-! ## exr-surface-of-revolution -/

/-- A surface of revolution `(f cos φ, f sin φ, h)`: with `a = f'`, `c = h'`, the Jacobian gives
`Jᵀ J = diag(f'² + h'², f²)`. -/
theorem revolution_metric (f a c φ : ℝ) :
    (!![a * Real.cos φ, -(f * Real.sin φ); a * Real.sin φ, f * Real.cos φ; c, 0])ᵀ *
      !![a * Real.cos φ, -(f * Real.sin φ); a * Real.sin φ, f * Real.cos φ; c, 0] =
      !![a ^ 2 + c ^ 2, 0; 0, f ^ 2] := by
  VERIPHY[dg-me-matrix]

/-- `det g = f² (f'² + h'²)` and the inverse is `diag(1/(f'² + h'²), 1/f²)`. -/
theorem revolution_det_inverse (f a c : ℝ) (hf : f ≠ 0) (hac : a ^ 2 + c ^ 2 ≠ 0) :
    (!![a ^ 2 + c ^ 2, 0; 0, f ^ 2]).det = f ^ 2 * (a ^ 2 + c ^ 2) ∧
      !![a ^ 2 + c ^ 2, 0; 0, f ^ 2] * !![1 / (a ^ 2 + c ^ 2), 0; 0, 1 / f ^ 2] = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-me-matrix]
  · VERIPHY[dg-me-matrix]

/-- The sphere: `f = sin t`, `h = cos t` have `f' = cos t`, `h' = −sin t`. -/
theorem revolution_sphere_derivs (t : ℝ) :
    deriv (fun t : ℝ => Real.sin t) t = Real.cos t ∧ deriv (fun t : ℝ => Real.cos t) t = -Real.sin t := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-derivative]
  · VERIPHY[de-derivative]

/-- The sphere's profile has `f'² + h'² = cos²t + sin²t = 1`, so `ds² = dt² + sin²t dφ²`. -/
theorem revolution_sphere (t : ℝ) : Real.cos t ^ 2 + (-Real.sin t) ^ 2 = 1 := by
  VERIPHY[dg-me-trig]

/-- The cylinder `f = R`, `h = t`: `f'² + h'² = 0² + 1² = 1`; the cone `f = h = t`: `1² + 1² = 2`. -/
theorem revolution_cylinder_cone : (0 : ℝ) ^ 2 + 1 ^ 2 = 1 ∧ (1 : ℝ) ^ 2 + 1 ^ 2 = 2 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]

/-- The torus `f = R + r cos t`, `h = r sin t`: `f'² + h'² = r² sin²t + r² cos²t = r²`. -/
theorem revolution_torus (r t : ℝ) : (-(r * Real.sin t)) ^ 2 + (r * Real.cos t) ^ 2 = r ^ 2 := by
  VERIPHY[dg-me-trig]

/-- Parametrised by arc length (`f'² + h'² = 1`) every surface of revolution has `ds² = dt² + f² dφ²`. -/
theorem revolution_arclength (f a c : ℝ) (h : a ^ 2 + c ^ 2 = 1) :
    !![a ^ 2 + c ^ 2, 0; 0, f ^ 2] = !![1, 0; 0, f ^ 2] := by
  rw [h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Metric.revolution_arclength, dg-me-revolution-arclength-test]

end KaluzaKlein.DifferentialGeometry.Metric

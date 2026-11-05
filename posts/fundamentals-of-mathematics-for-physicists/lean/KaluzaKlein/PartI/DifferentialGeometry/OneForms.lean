import Mathlib
import KaluzaKlein.Geometry.VectorFields
import KaluzaKlein.Geometry.Forms
import KaluzaKlein.Geometry.MetricExamples
import KaluzaKlein.Geometry.Killing

/-!
# Differential geometry, section "Cotangent vectors and one-forms"

The claims of the section "Cotangent vectors and one-forms" of `tutorial/part-i-09-differential-geometry.qmd`
(design D28): cotangent vectors, coordinate differentials and their transformation law, the differential of a
function, one-forms and exactness, the differential of the height function of the sphere.

Modelling (D28, as `KaluzaKlein.Geometry.*` does it): geometry is in coordinates. A covector at a point is its row of
components `ω : Fin n → ℝ`, a tangent vector its column `X : Fin n → ℝ`, and the pairing `ω(X) = ωᵢ Xⁱ` is the dot
product `ω ⬝ᵥ X`; the coordinate differential `dxⁱ` is `Pi.single i 1` and the coordinate vector `∂ⱼ` is
`Pi.single j 1`. A change of chart is its Jacobian matrix `J = ∂x'/∂x`: vectors transform as `J *ᵥ X`, covectors as
`ω ᵥ* J⁻¹`. The differential of a function `f : (Fin n → ℝ) → ℝ` has the components `pd i f x` of the
`CoordinateGeometry` asset (`pd i f x = deriv (fun t => f (Function.update x i t)) (x i)`); the height of the sphere is
`heightZ x = cos (x 0)` in the chart `x = (θ, φ)`, the rotation field `∂_φ` is `dphi = (0, 1)` and `∂_θ` is
`dtheta = (1, 0)`. Rational identities of the stereographic chart are stated with the inverse `w = 1/(u² + v²)` as a
variable with its defining equation. Global statements about the circle (the one-form `dφ` is not exact) are stated on
the parametrisation `t ∈ [0, 2π]`: a function on the circle is a `2π`-periodic function of `t`.
-/

open CoordinateGeometry Matrix KaluzaKlein.Geometry

namespace KaluzaKlein.DifferentialGeometry.OneForms

/-! ## Partial derivatives in a two-dimensional chart (infrastructure) -/

/-- `∂₀ f` at `x` is the derivative of `t ↦ f (t, x 1)` at `x 0`. -/
theorem pd_zero_two (f : (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) : pd 0 f x = deriv (fun t => f ![t, x 1]) (x 0) := by
  unfold CoordinateGeometry.pd
  congr 1; funext t; congr 1; funext i; fin_cases i <;> simp

/-- `∂₁ f` at `x` is the derivative of `t ↦ f (x 0, t)` at `x 1`. -/
theorem pd_one_two (f : (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) : pd 1 f x = deriv (fun t => f ![x 0, t]) (x 1) := by
  unfold CoordinateGeometry.pd
  congr 1; funext t; congr 1; funext i; fin_cases i <;> simp

/-! ## def-cotangent-vector -/

/-- The cotangent space, the dual of the tangent space `ℝⁿ`, has dimension `n`. -/
theorem cotangent_dim (n : ℕ) : Module.finrank ℝ (Module.Dual ℝ (Fin n → ℝ)) = n := by
  rw [Subspace.dual_finrank_eq, Module.finrank_fin_fun]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.cotangent_dim, dg-vf2-cotangent-dim-test]

/-! ## exm-cotangent-evaluation -/

/-- The covector with values `3` on `∂_θ` and `−1` on `∂_φ` sends `X = 0.4 ∂_θ + 0.5 ∂_φ` to `1.2 − 0.5 = 0.7`. -/
theorem covector_on_X : ![(3 : ℝ), -1] ⬝ᵥ ![2 / 5, 1 / 2] = 7 / 10 := by
  VERIPHY[dg-vf2-matrix]

/-- A covector is determined by its values on a basis: two covectors agreeing on every `∂ᵢ` are equal. -/
theorem covector_determined (n : ℕ) (ω ω' : Fin n → ℝ) (h : ∀ i, ω ⬝ᵥ Pi.single i 1 = ω' ⬝ᵥ Pi.single i 1) :
    ω = ω' := by
  ext i; simpa using h i
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.covector_determined, dg-vf2-covector-determined-test]

/-! ## exm-cotangent-rate -/

/-- The rate of change of the height along `X`, `∂ᵢZ Xⁱ`, is linear in `X`: a cotangent vector at each point. -/
theorem dZ_pairing_linear (x : Fin 2 → ℝ) (a b : ℝ) (X Y : Fin 2 → ℝ) :
    (∑ i, pd i heightZ x * (a • X + b • Y) i) =
      a * (∑ i, pd i heightZ x * X i) + b * (∑ i, pd i heightZ x * Y i) := by
  simp only [Fin.sum_univ_two, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  VERIPHY[de-poly]

/-! ## exm-cotangent-rn -/

/-- In `ℝⁿ` the dot product with a fixed vector `a`, `X ↦ a · X`, is linear: a covector. -/
theorem dot_linear (n : ℕ) (a X Y : Fin n → ℝ) (c : ℝ) : a ⬝ᵥ (c • X + Y) = c * (a ⬝ᵥ X) + a ⬝ᵥ Y := by
  simp [dotProduct_add, dotProduct_smul]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dot_linear, dg-vf2-dot-linear-test]

/-- Every covector on `ℝⁿ` is a row acting on columns: `ω(X) = ωᵢ Xⁱ` for the row `ωᵢ = ω(eᵢ)`. -/
theorem covector_is_dot (n : ℕ) (ω : Module.Dual ℝ (Fin n → ℝ)) : ∃ a : Fin n → ℝ, ∀ X, ω X = a ⬝ᵥ X := by
  refine ⟨fun i => ω (Pi.single i 1), fun X => ?_⟩
  rw [LinearMap.pi_apply_eq_sum_univ ω X]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [smul_eq_mul, mul_comm]
  congr 2
  ext j; simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.covector_is_dot, dg-vf2-covector-is-dot-test]

/-! ## exr-of-covector-evaluate -/

/-- `ω = (2, 5)` takes the values `2 − 5 = −3`, `6 + 5 = 11` and `0.8 + 2.5 = 3.3` on the three vectors. -/
theorem omega_values :
    ![(2 : ℝ), 5] ⬝ᵥ ![1, -1] = -3 ∧ ![(2 : ℝ), 5] ⬝ᵥ ![3, 1] = 11 ∧ ![(2 : ℝ), 5] ⬝ᵥ ![2 / 5, 1 / 2] = 33 / 10 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]

/-- `ω = (2, 5)` vanishes on `5 ∂_θ − 2 ∂_φ` and on all its multiples. -/
theorem omega_kernel (t : ℝ) : ![(2 : ℝ), 5] ⬝ᵥ (t • ![5, -2]) = 0 := by
  VERIPHY[dg-vf2-matrix]

/-- The kernel of a non-zero covector on a two-dimensional space is one-dimensional. -/
theorem kernel_dim_two (ω : Module.Dual ℝ (Fin 2 → ℝ)) (h : ω ≠ 0) : Module.finrank ℝ (LinearMap.ker ω) = 1 := by
  have h1 := LinearMap.finrank_range_add_finrank_ker ω
  have hr : LinearMap.range ω = ⊤ := by
    rw [LinearMap.range_eq_top]
    obtain ⟨v, hv⟩ : ∃ v, ω v ≠ 0 := by
      by_contra hc
      push_neg at hc
      exact h (LinearMap.ext hc)
    intro c
    refine ⟨(c / ω v) • v, ?_⟩
    rw [map_smul, smul_eq_mul, div_mul_cancel₀ _ hv]
  rw [hr, finrank_top, Module.finrank_self, Module.finrank_fin_fun] at h1
  omega
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.kernel_dim_two, dg-vf2-kernel-dim-test]

/-! ## exr-of-dual-frame

The orthonormal frame `e₁ = ∂_θ = (1, 0)`, `e₂ = sin⁻¹θ ∂_φ = (0, w)` with `w = 1/sin θ`, and the coframe
`e¹ = dθ = (1, 0)`, `e² = sin θ dφ = (0, s)` with `s w = 1`. -/

/-- `e¹ = dθ` takes the values `1, 0` on `e₁, e₂`, and `e² = sin θ dφ` takes the value `0` on `e₁`. -/
theorem dual_frame_values (w s : ℝ) :
    ![(1 : ℝ), 0] ⬝ᵥ ![1, 0] = 1 ∧ ![(1 : ℝ), 0] ⬝ᵥ ![0, w] = 0 ∧ ![(0 : ℝ), s] ⬝ᵥ ![1, 0] = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]

/-- `e²(e₂) = sin θ · (1/sin θ) = 1`. -/
theorem dual_frame_e2 (w s : ℝ) (hw : s * w = 1) : ![(0 : ℝ), s] ⬝ᵥ ![0, w] = 1 := by
  simp only [dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  VERIPHY[de-poly]

/-- The conditions `e²(e₁) = a = 0` and `e²(e₂) = b/sin θ = 1` give `a = 0` and `b = sin θ`: the coordinate
differential is rescaled by the inverse of the factor that rescaled the vector. -/
theorem dual_frame_coefficients (a b s w : ℝ) (hw : s * w = 1) (ha : a * 1 + b * 0 = 0) (hb : a * 0 + b * w = 1) :
    a = 0 ∧ b = s := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]

/-! ## exr-of-pairing-invariant -/

/-- If `f^b = N^b_c e^c` is dual to `f_b = M^a_b e_a`, then `N M = 1` forces `N = M⁻¹`. -/
theorem dual_of_change (n : ℕ) (M N : Matrix (Fin n) (Fin n) ℝ) (h : N * M = 1) : N = M⁻¹ :=
  Matrix.inv_eq_left_inv h ▸ rfl
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dual_of_change, dg-vf2-dual-of-change-test]

/-- The pairing is basis independent: with `ω'_b = M^a_b ω_a` and `X'^b = (M⁻¹)^b_a X^a`, `ω'_b X'^b = ω_a X^a`. -/
theorem pairing_invariant (n : ℕ) (M : Matrix (Fin n) (Fin n) ℝ) (hM : IsUnit M.det) (ω X : Fin n → ℝ) :
    (ω ᵥ* M) ⬝ᵥ (M⁻¹ *ᵥ X) = ω ⬝ᵥ X := by
  rw [← Matrix.dotProduct_mulVec, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hM, Matrix.one_mulVec]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.pairing_invariant, dg-vf2-pairing-invariant-test]

/-- A covector vanishing on a basis is zero. -/
theorem vanish_on_basis (n : ℕ) (ω : Fin n → ℝ) (h : ∀ i, ω ⬝ᵥ Pi.single i 1 = 0) : ω = 0 := by
  ext i; simpa using h i
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.vanish_on_basis, dg-vf2-vanish-on-basis-test]

/-! ## def-coordinate-differentials -/

/-- `dxⁱ(∂ⱼ) = δⁱⱼ`: the coordinate differentials are the dual basis of the coordinate basis. -/
theorem dx_on_partial (n : ℕ) (i j : Fin n) :
    (Pi.single i 1 : Fin n → ℝ) ⬝ᵥ Pi.single j 1 = if i = j then 1 else 0 := by
  simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dx_on_partial, dg-vf2-dx-on-partial-test]

/-- The components of a covector are its values on the coordinate vectors: `ωᵢ = ω(∂ᵢ)`. -/
theorem component_formula (n : ℕ) (ω : Fin n → ℝ) (i : Fin n) : ω ⬝ᵥ Pi.single i 1 = ω i := by simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.component_formula, dg-vf2-component-formula-test]

/-- The covector law `ω'_j = (∂xⁱ/∂x'^j) ωᵢ`, that is `ω' = ω J⁻¹` for the Jacobian `J = ∂x'/∂x`, makes `ωᵢ Xⁱ` chart
independent against the vector law `X' = J X`. -/
theorem covector_law_invariant (n : ℕ) (J : Matrix (Fin n) (Fin n) ℝ) (hJ : IsUnit J.det) (ω X : Fin n → ℝ) :
    (ω ᵥ* J⁻¹) ⬝ᵥ (J *ᵥ X) = ω ⬝ᵥ X := by
  rw [← Matrix.dotProduct_mulVec, Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ hJ, Matrix.one_mulVec]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.covector_law_invariant, dg-vf2-covector-law-test]

/-! ## exm-differentials-sphere -/

/-- `dθ(X) = 0.4` and `dφ(X) = 0.5` for `X = 0.4 ∂_θ + 0.5 ∂_φ`. -/
theorem dtheta_dphi_on_X : ![(1 : ℝ), 0] ⬝ᵥ ![2 / 5, 1 / 2] = 2 / 5 ∧ ![(0 : ℝ), 1] ⬝ᵥ ![2 / 5, 1 / 2] = 1 / 2 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]

/-- The covector with values `3, −1` on the coordinate basis is `3 dθ − dφ`. -/
theorem covector_combination : (3 : ℝ) • ![(1 : ℝ), 0] - ![0, 1] = ![3, -1] := by
  VERIPHY[dg-vf2-matrix]

/-! ## exm-differentials-transformation, exr-covector-transformation

The components of `dφ` in the stereographic chart, `(∂φ/∂u, ∂φ/∂v) = (−v, u)/(u² + v²)`, with `φ = atan2(v, u)`
(`= arctan (v/u)` on the half plane `u > 0`), and their pairing with `∂_φ = −v ∂_u + u ∂_v`. -/

/-- `∂φ/∂u = −v/(u² + v²)` and `∂φ/∂v = u/(u² + v²)` for `φ = arctan (v/u)`, `u ≠ 0`. -/
theorem dphi_stereo_components (x : Fin 2 → ℝ) (hu : x 0 ≠ 0) :
    pd 0 (fun y => Real.arctan (y 1 / y 0)) x = -x 1 / (x 0 ^ 2 + x 1 ^ 2) ∧
      pd 1 (fun y => Real.arctan (y 1 / y 0)) x = x 0 / (x 0 ^ 2 + x 1 ^ 2) := by
  constructor
  · rw [pd_zero_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd : HasDerivAt (fun t => x 1 / t) (-(x 1) / (x 0) ^ 2) (x 0) := by
      have := (hasDerivAt_id (x 0)).inv hu
      convert this.const_mul (x 1) using 1 <;> simp [div_eq_mul_inv]; ring
    rw [hd.arctan.deriv]
    field_simp
    ring
  · rw [pd_one_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd : HasDerivAt (fun t => t / x 0) (1 / x 0) (x 1) := by
      simpa using (hasDerivAt_id (x 1)).div_const (x 0)
    rw [hd.arctan.deriv]
    field_simp
    ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dphi_stereo_components, dg-vf2-dphi-stereo-test]

/-- `dφ(∂_φ) = (−v)(−v)/(u² + v²) + u · u/(u² + v²) = 1` in the stereographic chart (`w = 1/(u² + v²)`). -/
theorem dphi_stereo_pairing (u v w : ℝ) (hw : (u ^ 2 + v ^ 2) * w = 1) : ![-v * w, u * w] ⬝ᵥ ![-v, u] = 1 := by
  simp only [dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  VERIPHY[de-poly]

/-- `dφ(∂_φ) = 1` in the spherical chart. -/
theorem dphi_on_dphi : ![(0 : ℝ), 1] ⬝ᵥ ![0, 1] = 1 := by
  VERIPHY[dg-vf2-matrix]

/-! ## exm-differentials-polar

`dr = (x dx + y dy)/r` and `dφ = (−y dx + x dy)/r²` in Cartesian components, with `r² = x² + y²`, `w = 1/r`;
`∂_r = (x, y)/r` and the rotation field `∂_φ = (−y, x)`. -/

/-- `dr` and `dφ` are the rows of the inverse polar Jacobian: `dr(∂_r) = 1`, `dr(∂_φ) = 0`, `dφ(∂_r) = 0`,
`dφ(∂_φ) = 1`. -/
theorem polar_differentials (x y r w : ℝ) (hr : x ^ 2 + y ^ 2 = r ^ 2) (hw : r * w = 1) :
    ![x * w, y * w] ⬝ᵥ ![x * w, y * w] = 1 ∧ ![x * w, y * w] ⬝ᵥ ![-y, x] = 0 ∧
      ![-y * w ^ 2, x * w ^ 2] ⬝ᵥ ![x * w, y * w] = 0 ∧ ![-y * w ^ 2, x * w ^ 2] ⬝ᵥ ![-y, x] = 1 := by
  simp only [dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  refine ⟨?_, ?_, ?_, ?_⟩
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]
  · VERIPHY[de-poly]

/-! ## exr-of-coframe-stereographic -/

/-- `dθ = −2(u du + v dv)/((1 + ρ²)ρ)` pairs with `∂_φ = −v ∂_u + u ∂_v` to `0` (its components are a multiple `a`
of `(u, v)`). -/
theorem dtheta_stereo_on_dphi (u v a : ℝ) : ![-2 * u * a, -2 * v * a] ⬝ᵥ ![-v, u] = 0 := by
  VERIPHY[dg-vf2-matrix]

/-- `dθ/dρ = −2/(1 + ρ²)` for `θ = 2 arctan (1/ρ)`, `ρ > 0`. -/
theorem dtheta_drho (ρ : ℝ) (h : 0 < ρ) : deriv (fun t => 2 * Real.arctan (1 / t)) ρ = -2 / (1 + ρ ^ 2) := by
  have hd : HasDerivAt (fun t => 2 * Real.arctan (1 / t)) (2 * (1 / (1 + (1 / ρ) ^ 2) * (-1 / ρ ^ 2))) ρ := by
    have h1 : HasDerivAt (fun t : ℝ => 1 / t) (-1 / ρ ^ 2) ρ := by
      have := (hasDerivAt_id ρ).inv h.ne'
      simpa [one_div] using this
    exact (h1.arctan).const_mul 2
  rw [hd.deriv]
  field_simp
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dtheta_drho, dg-vf2-dtheta-drho-test]

/-- The chain rule `∂θ/∂u = (dθ/dρ)(∂ρ/∂u)` with `∂ρ/∂u = u/ρ` gives `−2u/((1 + ρ²)ρ)`. -/
theorem dtheta_stereo_component (ρ u : ℝ) : -2 / (1 + ρ ^ 2) * (u / ρ) = -2 * u / ((1 + ρ ^ 2) * ρ) := by
  VERIPHY[de-poly]

/-! ## exr-of-dual-transform -/

/-- `dx'^j(∂_k) = ∂x'^j/∂x^k`: the row `j` of the Jacobian paired with `∂_k` is the entry `J j k`. -/
theorem jacobian_row_on_basis (n : ℕ) (J : Matrix (Fin n) (Fin n) ℝ) (j k : Fin n) :
    (fun i => J j i) ⬝ᵥ Pi.single k 1 = J j k := by simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.jacobian_row_on_basis, dg-vf2-jacobian-row-test]

/-- Substituting `dxⁱ = (∂xⁱ/∂x'^j) dx'^j` into `ω = ωᵢ dxⁱ` gives the coefficients `ω' = ω J⁻¹`, and transforming
back with `J` returns `ω`: the two laws are inverse to each other. -/
theorem covector_law_inverse (n : ℕ) (J : Matrix (Fin n) (Fin n) ℝ) (hJ : IsUnit J.det) (ω : Fin n → ℝ) :
    (ω ᵥ* J⁻¹) ᵥ* J = ω := by
  rw [Matrix.vecMul_vecMul, Matrix.nonsing_inv_mul _ hJ, Matrix.vecMul_one]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.covector_law_inverse, dg-vf2-covector-law-inverse-test]

/-! ## def-differential -/

/-- In a chart `df = ∂ᵢf dxⁱ`: the component of `df` on `∂ⱼ` is `∂ⱼ f`. -/
theorem differential_components (n : ℕ) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) (j : Fin n) :
    (fun i => pd i f x) ⬝ᵥ Pi.single j 1 = pd j f x := by simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.differential_components, dg-vf2-differential-components-test]

/-- The value of `df` on the velocity of a curve is the rate of change of `f` along the curve:
`d/dt f(γ(t)) = ∂ᵢf(γ(t)) γ̇ⁱ(t)` for `f` differentiable at `γ(t)`. -/
theorem velocity_rate {n : ℕ} (f : (Fin n → ℝ) → ℝ) (γ : ℝ → (Fin n → ℝ)) (v : Fin n → ℝ) (t : ℝ)
    (L : (Fin n → ℝ) →L[ℝ] ℝ) (hf : HasFDerivAt f L (γ t)) (hγ : ∀ i, HasDerivAt (fun s => γ s i) (v i) t) :
    deriv (fun s => f (γ s)) t = ∑ i, pd i f (γ t) * v i := by
  have hγ' : HasDerivAt γ v t := hasDerivAt_pi.mpr hγ
  have hc : HasDerivAt (fun s => f (γ s)) (L v) t := hf.comp_hasDerivAt t hγ'
  rw [hc.deriv]
  have hpd : ∀ i, pd i f (γ t) = L (Pi.single i 1) := by
    intro i
    unfold CoordinateGeometry.pd
    have h2 : HasDerivAt (Function.update (γ t) i) (Pi.single i 1) (γ t i) := hasDerivAt_update (γ t) i (γ t i)
    have h3 : HasFDerivAt f L (Function.update (γ t) i (γ t i)) := by rwa [Function.update_eq_self]
    exact (h3.comp_hasDerivAt (γ t i) h2).deriv
  simp only [hpd]
  have := LinearMap.pi_apply_eq_sum_univ (L : (Fin n → ℝ) →ₗ[ℝ] ℝ) v
  simp only [ContinuousLinearMap.coe_coe] at this
  rw [this]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [smul_eq_mul, mul_comm]
  congr 2
  ext j; simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.velocity_rate, dg-vf2-velocity-rate-test]

/-! ## exm-height-differential, prp-dZ -/

/-- `dZ = d(cos θ) = −sin θ dθ`: the components of `dZ` are `(−sin θ, 0)`. -/
theorem dZ_components (x : Fin 2 → ℝ) : pd 0 heightZ x = -Real.sin (x 0) ∧ pd 1 heightZ x = 0 := by
  refine ⟨?_, ?_⟩
  · rw [pd_zero_two]; simp only [heightZ, Matrix.cons_val_zero]
    VERIPHY[de-partials]
  · rw [pd_one_two]; simp only [heightZ, Matrix.cons_val_zero]
    VERIPHY[de-partials]

/-- `dZ(∂_φ) = 0`: moving along a parallel does not change the height. -/
theorem dZ_on_dphi (x : Fin 2 → ℝ) : (∑ i, pd i heightZ x * dphi i x) = 0 := by
  simp only [Fin.sum_univ_two, dphi, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, pd_zero_two,
    pd_one_two, heightZ]
  VERIPHY[de-partials]

/-- `dZ(∂_θ) = −sin θ`. -/
theorem dZ_on_dtheta (x : Fin 2 → ℝ) : (∑ i, pd i heightZ x * dtheta i x) = -Real.sin (x 0) := by
  simp only [Fin.sum_univ_two, dtheta, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, pd_zero_two,
    pd_one_two, heightZ]
  VERIPHY[de-partials]

/-! ## exm-differential-coordinate -/

/-- The coordinate differentials are the differentials of the coordinate functions: `∂ᵢ xʲ = δⁱⱼ`. -/
theorem coordinate_differential (n : ℕ) (i j : Fin n) (x : Fin n → ℝ) :
    pd i (fun y => y j) x = if i = j then 1 else 0 := by
  unfold CoordinateGeometry.pd
  by_cases h : i = j
  · subst h; simp
  · simp [Function.update_apply, Ne.symm h, h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.coordinate_differential, dg-vf2-coordinate-differential-test]

/-- `dφ` is well defined although `φ` jumps by a constant between angle charts: `d(φ + c) = dφ`, with component `1`
in every angle chart. -/
theorem angle_shift_differential (c t : ℝ) : deriv (fun s => s + c) t = 1 := by
  VERIPHY[de-partials]

/-! ## exm-differential-stereographic, exr-of-dz-stereographic

`Z = (u² + v² − 1)/(u² + v² + 1)` in the stereographic chart. -/

/-- `∂Z/∂u = 4u/(1 + u² + v²)²`. -/
theorem dZ_stereo_u (u v : ℝ) (h : u ^ 2 + v ^ 2 + 1 ≠ 0) :
    deriv (fun t => (t ^ 2 + v ^ 2 - 1) / (t ^ 2 + v ^ 2 + 1)) u = 4 * u / (1 + u ^ 2 + v ^ 2) ^ 2 := by
  VERIPHY[de-partials]

/-- `∂Z/∂v = 4v/(1 + u² + v²)²`. -/
theorem dZ_stereo_v (u v : ℝ) (h : u ^ 2 + v ^ 2 + 1 ≠ 0) :
    deriv (fun t => (u ^ 2 + t ^ 2 - 1) / (u ^ 2 + t ^ 2 + 1)) v = 4 * v / (1 + u ^ 2 + v ^ 2) ^ 2 := by
  VERIPHY[de-partials]

/-- `dZ = 4(u du + v dv)/(1 + ρ²)²` vanishes on the rotation field `−v ∂_u + u ∂_v`. -/
theorem dZ_stereo_on_rotation (u v a : ℝ) : ![4 * u * a, 4 * v * a] ⬝ᵥ ![-v, u] = 0 := by
  VERIPHY[dg-vf2-matrix]

/-- `dZ/dρ = 4ρ/(1 + ρ²)²` for `Z = (ρ² − 1)/(ρ² + 1)`. -/
theorem dZ_drho (ρ : ℝ) (h : ρ ^ 2 + 1 ≠ 0) : deriv (fun t => (t ^ 2 - 1) / (t ^ 2 + 1)) ρ = 4 * ρ / (1 + ρ ^ 2) ^ 2 := by
  VERIPHY[de-partials]

/-- `ρ dρ = u du + v dv`: `ρ ∂ρ/∂u = u` for `ρ = √(u² + v²)`. -/
theorem rho_drho (u v : ℝ) (h : 0 < u ^ 2 + v ^ 2) :
    Real.sqrt (u ^ 2 + v ^ 2) * deriv (fun t => Real.sqrt (t ^ 2 + v ^ 2)) u = u := by
  VERIPHY[de-partials]

/-- `dρ/dθ = −1/(2 sin²(θ/2))` for `ρ = cot(θ/2)`. -/
theorem drho_dtheta (θ : ℝ) (hs : Real.sin (θ / 2) ≠ 0) :
    deriv (fun t => Real.cos (t / 2) / Real.sin (t / 2)) θ = -1 / (2 * Real.sin (θ / 2) ^ 2) := by
  VERIPHY[de-partials]

/-- `1 + ρ² = 1/sin²(θ/2)` for `ρ = cot(θ/2) = c w`, `w = 1/s`, `c² + s² = 1`. -/
theorem one_add_rho_sq (c s w : ℝ) (h : c ^ 2 + s ^ 2 = 1) (hw : s * w = 1) : 1 + (c * w) ^ 2 = w ^ 2 := by
  VERIPHY[de-poly]

/-- `4ρ (dρ/dθ) = −2 cot(θ/2) sin²(θ/2) · (1 + ρ²)²`, that is `4ρ dρ/(1 + ρ²)² = −2 sin(θ/2) cos(θ/2) dθ`. -/
theorem dZ_chain_product (c s w : ℝ) (hw : s * w = 1) :
    4 * (c * w) * (-1 / (2 * s ^ 2)) = -2 * s * c * (w ^ 2) ^ 2 := by
  rw [div_eq_mul_inv, ← sq_eq_sq_iff_abs_eq_abs] at *
  VERIPHY[de-poly]

/-- `2 sin(θ/2) cos(θ/2) = sin θ`: the half-angle form of the differential is `−sin θ dθ`. -/
theorem half_angle (θ : ℝ) : Real.sin θ = 2 * Real.sin (θ / 2) * Real.cos (θ / 2) := by
  conv_lhs => rw [show θ = 2 * (θ / 2) by ring]
  VERIPHY[dg-vf2-trig]

/-! ## exr-differential-height -/

/-- `dZ(X) = −0.4 sin θ` for `X = 0.4 ∂_θ + 0.5 ∂_φ`. -/
theorem dZ_on_X (x : Fin 2 → ℝ) : ∑ i, pd i heightZ x * ![(2 / 5 : ℝ), 1 / 2] i = -(2 / 5) * Real.sin (x 0) := by
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, pd_zero_two, pd_one_two,
    heightZ]
  VERIPHY[de-partials]

/-! ## exr-of-dr-polar

`r = √(x² + y²)` on the punctured plane. -/

/-- `∂r/∂x = x/√(x² + y²)`. -/
theorem dr_cartesian_x (x y : ℝ) (h : 0 < x ^ 2 + y ^ 2) :
    deriv (fun t => Real.sqrt (t ^ 2 + y ^ 2)) x = x / Real.sqrt (x ^ 2 + y ^ 2) := by
  VERIPHY[de-partials]

/-- `∂r/∂y = y/√(x² + y²)`. -/
theorem dr_cartesian_y (x y : ℝ) (h : 0 < x ^ 2 + y ^ 2) :
    deriv (fun t => Real.sqrt (x ^ 2 + t ^ 2)) y = y / Real.sqrt (x ^ 2 + y ^ 2) := by
  VERIPHY[de-partials]

/-- In polar coordinates `dr` has components `(1, 0)`: `dr(∂_r) = 1` and `dr(∂_φ) = 0`. -/
theorem dr_polar : ![(1 : ℝ), 0] ⬝ᵥ ![1, 0] = 1 ∧ ![(1 : ℝ), 0] ⬝ᵥ ![0, 1] = 0 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]

/-- `d(r²) = 2r dr`: the derivative of `r²` with respect to `r`. -/
theorem d_r_sq (r : ℝ) : deriv (fun t => t ^ 2) r = 2 * r := by
  VERIPHY[de-partials]

/-- `d(r²) = 2r dr` in Cartesian components: `∂(x² + y²)/∂x = 2x = 2r · (x/r)` (`w = 1/r`). -/
theorem d_r_sq_cartesian (x y r w : ℝ) (hw : r * w = 1) :
    deriv (fun t => t ^ 2 + y ^ 2) x = 2 * x ∧ 2 * r * (x * w) = 2 * x := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-partials]
  · VERIPHY[de-poly]

/-! ## exr-of-leibniz-chain -/

/-- The product rule `d(fg) = f dg + g df`, componentwise. -/
theorem leibniz (n : ℕ) (f g : (Fin n → ℝ) → ℝ) (i : Fin n) (x : Fin n → ℝ)
    (hf : DifferentiableAt ℝ (fun t => f (Function.update x i t)) (x i))
    (hg : DifferentiableAt ℝ (fun t => g (Function.update x i t)) (x i)) :
    pd i (fun y => f y * g y) x = f x * pd i g x + g x * pd i f x := by
  unfold CoordinateGeometry.pd
  rw [deriv_fun_mul hf hg]
  simp [Function.update_eq_self]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.leibniz, dg-vf2-leibniz-test]

/-- The chain rule `d(h ∘ f) = h'(f) df`, componentwise. -/
theorem chain (n : ℕ) (f : (Fin n → ℝ) → ℝ) (h : ℝ → ℝ) (i : Fin n) (x : Fin n → ℝ)
    (hf : DifferentiableAt ℝ (fun t => f (Function.update x i t)) (x i)) (hh : DifferentiableAt ℝ h (f x)) :
    pd i (fun y => h (f y)) x = deriv h (f x) * pd i f x := by
  unfold CoordinateGeometry.pd
  have : (fun t => h (f (Function.update x i t))) = h ∘ (fun t => f (Function.update x i t)) := rfl
  rw [this, deriv_comp _ (by simpa [Function.update_eq_self] using hh) hf]
  simp [Function.update_eq_self]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.chain, dg-vf2-chain-test]

/-- `d(Z²) = 2Z dZ = −2 cos θ sin θ dθ`: the components of `d(cos²θ)` are `(−2 cos θ sin θ, 0)`. -/
theorem dZ_sq_components (x : Fin 2 → ℝ) :
    pd 0 (fun y => Real.cos (y 0) ^ 2) x = -2 * Real.cos (x 0) * Real.sin (x 0) ∧
      pd 1 (fun y => Real.cos (y 0) ^ 2) x = 0 := by
  refine ⟨?_, ?_⟩
  · rw [pd_zero_two]; simp only [Matrix.cons_val_zero]
    VERIPHY[de-partials]
  · rw [pd_one_two]; simp only [Matrix.cons_val_zero]
    VERIPHY[de-partials]

/-! ## def-one-form, exm-one-form-circle, exr-dphi-closed, exr-of-torus-forms

The circle is parametrised by `t ∈ [0, 2π]`; a function on the circle is a `2π`-periodic function `F`, and the
one-form `c dφ` has the constant component `c`. -/

/-- The integral of `dφ` around the circle is `∫₀^{2π} 1 dφ = 2π`. -/
theorem circle_integral_dphi : ∫ t in (0 : ℝ)..(2 * Real.pi), (1 : ℝ) = 2 * Real.pi := by
  VERIPHY[de-integral]

/-- The integral of an exact form `dF` along a curve is `F(end) − F(start)` (the fundamental theorem of calculus). -/
theorem curve_integral (F F' : ℝ → ℝ) (a b : ℝ) (hF : ∀ t, HasDerivAt F (F' t) t)
    (hint : IntervalIntegrable F' MeasureTheory.volume a b) : ∫ t in a..b, F' t = F b - F a :=
  intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hF t) hint
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.curve_integral, dg-vf2-curve-integral-test]

/-- The integral of an exact form around a closed curve vanishes: `∫₀^{2π} (F ∘ γ)' dt = F(γ(2π)) − F(γ(0)) = 0`
for a `2π`-periodic `F`. -/
theorem periodic_integral_zero (F F' : ℝ → ℝ) (hF : ∀ t, HasDerivAt F (F' t) t)
    (hint : IntervalIntegrable F' MeasureTheory.volume 0 (2 * Real.pi)) (hp : F (2 * Real.pi) = F 0) :
    ∫ t in (0 : ℝ)..(2 * Real.pi), F' t = 0 := by
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hF t) hint, hp, sub_self]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.periodic_integral_zero, dg-vf2-periodic-integral-test]

/-- `c dφ` with `c ≠ 0` (in particular `dφ`) is not the differential of a function on the circle: no `2π`-periodic `F`
has `F' = c`, since its integral around the circle would be `2πc ≠ 0` instead of `0`. -/
theorem not_exact_const (c : ℝ) (hc : c ≠ 0) :
    ¬ ∃ F : ℝ → ℝ, (∀ t, HasDerivAt F c t) ∧ F (2 * Real.pi) = F 0 := by
  rintro ⟨F, hF, hp⟩
  have := periodic_integral_zero F (fun _ => c) hF (by simp) hp
  simp [Real.pi_ne_zero, hc] at this
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.not_exact_const, dg-vf2-not-exact-test]

/-- `c dφ` is exact on each chart domain: `F = c φ` has `dF = c dφ` there. -/
theorem exact_on_chart (c : ℝ) : ∃ F : ℝ → ℝ, ∀ t, HasDerivAt F c t :=
  ⟨fun t => c * t, fun t => by simpa using (hasDerivAt_id t).const_mul c⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.exact_on_chart, dg-vf2-exact-on-chart-test]

/-- On the torus, `da` and `db` integrate to `2π` around their own circle and to `0` around the other
(`db` has component `0` along the `a`-circle). -/
theorem torus_integrals :
    (∫ t in (0 : ℝ)..(2 * Real.pi), (1 : ℝ)) = 2 * Real.pi ∧ (∫ t in (0 : ℝ)..(2 * Real.pi), (0 : ℝ)) = 0 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[de-integral]
  · VERIPHY[de-integral]

/-- If `λ da + μ db` were exact, both integrals `2πλ`, `2πμ` would vanish, forcing `λ = μ = 0`. -/
theorem torus_exact_zero (l m : ℝ) (h1 : 2 * Real.pi * l = 0) (h2 : 2 * Real.pi * m = 0) : l = 0 ∧ m = 0 := by
  constructor
  · simpa [Real.pi_ne_zero] using h1
  · simpa [Real.pi_ne_zero] using h2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.torus_exact_zero, dg-vf2-torus-exact-test]

/-! ## exm-one-form-sphere -/

/-- `cos θ dφ` cannot be extended to the pole: in the stereographic chart its pairing with the rotation field
`(−v, u)`, which vanishes at the pole `u = v = 0`, is `Z = (ρ² − 1)/(ρ² + 1) → −1`, while a covector field continuous
at the pole pairs with the vanishing field to `0` there (stated for every non-zero multiple `c cos θ dφ`). -/
theorem hopf_not_extendable (c : ℝ) (hc : c ≠ 0) : ¬ ∃ η : (Fin 2 → ℝ) → (Fin 2 → ℝ), ContinuousAt η 0 ∧
    ∀ p : Fin 2 → ℝ, p ≠ 0 → η p ⬝ᵥ ![-p 1, p 0] = c * ((p 0 ^ 2 + p 1 ^ 2 - 1) / (p 0 ^ 2 + p 1 ^ 2 + 1)) := by
  rintro ⟨η, hη, h⟩
  have hη0 : ContinuousAt (fun p => η p 0) 0 := (continuous_apply 0).continuousAt.comp hη
  have hη1 : ContinuousAt (fun p => η p 1) 0 := (continuous_apply 1).continuousAt.comp hη
  have hg : ContinuousAt (fun p : Fin 2 → ℝ => η p 0 * (-p 1) + η p 1 * p 0) 0 :=
    (hη0.mul (by fun_prop)).add (hη1.mul (by fun_prop))
  have h1 : Filter.Tendsto (fun p : Fin 2 → ℝ => η p ⬝ᵥ ![-p 1, p 0]) (nhds 0) (nhds 0) := by
    have : (fun p : Fin 2 → ℝ => η p ⬝ᵥ ![-p 1, p 0]) = fun p => η p 0 * (-p 1) + η p 1 * p 0 := by
      funext p; simp [dotProduct, Fin.sum_univ_two]
    rw [this]
    simpa using hg.tendsto
  have h2 : Filter.Tendsto (fun p : Fin 2 → ℝ => c * ((p 0 ^ 2 + p 1 ^ 2 - 1) / (p 0 ^ 2 + p 1 ^ 2 + 1))) (nhds 0)
      (nhds (-c)) := by
    have hc' : ContinuousAt (fun p : Fin 2 → ℝ => c * ((p 0 ^ 2 + p 1 ^ 2 - 1) / (p 0 ^ 2 + p 1 ^ 2 + 1))) 0 := by
      have : ∀ p : Fin 2 → ℝ, p 0 ^ 2 + p 1 ^ 2 + 1 ≠ 0 := fun p => by positivity
      fun_prop (disch := exact this _)
    simpa using hc'.tendsto
  have heq : (fun p : Fin 2 → ℝ => η p ⬝ᵥ ![-p 1, p 0]) =ᶠ[nhdsWithin 0 {0}ᶜ]
      (fun p : Fin 2 → ℝ => c * ((p 0 ^ 2 + p 1 ^ 2 - 1) / (p 0 ^ 2 + p 1 ^ 2 + 1))) :=
    eventually_nhdsWithin_of_forall fun p hp => h p hp
  have := tendsto_nhds_unique ((h1.mono_left nhdsWithin_le_nhds).congr' heq) (h2.mono_left nhdsWithin_le_nhds)
  exact hc (by linarith)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.hopf_not_extendable, dg-vf2-hopf-not-extendable-test]

/-! ## exm-one-form-gauge -/

/-- Gauge invariance of the field strength in a two-dimensional chart: `F₀₁ = ∂₀A₁ − ∂₁A₀` is unchanged when the
exact one-form `dλ` is added to `A`, because the second partial derivatives of `λ` commute (stated as a property of
the smooth functions `A₀, A₁, λ`; the test evaluates it on explicit potentials). -/
def GaugeInvariance (A₀ A₁ lam : (Fin 2 → ℝ) → ℝ) : Prop :=
  ∀ x : Fin 2 → ℝ, pd 0 (fun y => A₁ y + pd 1 lam y) x - pd 1 (fun y => A₀ y + pd 0 lam y) x =
    pd 0 A₁ x - pd 1 A₀ x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.GaugeInvariance, dg-vf2-gauge-invariance-test]

/-! ## exr-of-rotation-form-polar

`ω = −y dx + x dy` on the plane, `γ(t) = (cos t, sin t)`. -/

/-- `ω = r² dφ` in polar coordinates: the components `(−y, x) = (−r sin φ, r cos φ)` transform with the polar
Jacobian to `(0, r²)`. -/
theorem rotation_form_polar (r φ : ℝ) :
    ![-(r * Real.sin φ), r * Real.cos φ] ᵥ* !![Real.cos φ, -r * Real.sin φ; Real.sin φ, r * Real.cos φ] = ![0, r ^ 2] := by
  VERIPHY[dg-vf2-matrix]

/-- The pullback of `ω` along the unit circle is `dt`: `ω(γ̇) = (−sin t)(cos t)' + (cos t)(sin t)' = 1`. -/
theorem rotation_form_pullback (t : ℝ) :
    -Real.sin t * deriv (fun s => Real.cos s) t + Real.cos t * deriv (fun s => Real.sin s) t = 1 := by
  VERIPHY[de-partials]

/-- The rotation one-form `−y dx + x dy`, as a function of an index list of length one. -/
noncomputable def rotForm : (Fin 1 → Fin 2) → (Fin 2 → ℝ) → ℝ := fun I x => (![-x 1, x 0] : Fin 2 → ℝ) (I 0)

/-- `dω = 2 dx ∧ dy`: the single component of `d(−y dx + x dy)` is `2`. -/
theorem rotation_form_extDeriv (x : Fin 2 → ℝ) : extDeriv rotForm ![0, 1] x = 2 := by
  VERIPHY[dg-vf2-forms]

/-! ## exm-of-dz-point -/

/-- At `θ = π/3`, `dZ = −(√3/2) dθ`. -/
theorem dZ_at_pi_third : -Real.sin (Real.pi / 3) = -(Real.sqrt 3 / 2) := by
  VERIPHY[dg-vf2-trig-values]

/-- `dZ(X) = −0.4 · √3/2 = −0.2√3` and `dZ(∂_θ + 2 ∂_φ) = −√3/2` at `θ = π/3`. -/
theorem dZ_at_point_values :
    ![-(Real.sqrt 3 / 2), (0 : ℝ)] ⬝ᵥ ![2 / 5, 1 / 2] = -(1 / 5) * Real.sqrt 3 ∧
      ![-(Real.sqrt 3 / 2), (0 : ℝ)] ⬝ᵥ ![1, 2] = -(Real.sqrt 3 / 2) := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-vf2-matrix]
  · VERIPHY[dg-vf2-matrix]

/-- `−0.2√3 ≈ −0.346`: `−0.347 < −√3/5 < −0.346`. -/
theorem dZ_at_point_approx : -0.347 < -(1 / 5) * Real.sqrt 3 ∧ -(1 / 5) * Real.sqrt 3 < -0.346 := by
  have h1 : (1.73 : ℝ) < Real.sqrt 3 := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h2 : Real.sqrt 3 < 1.735 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  constructor <;> nlinarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.dZ_at_point_approx, dg-vf2-dz-approx-test]

/-! ## exm-of-dz-lean -/

/-- `dZ(X) = ωᵢ Xⁱ = −Xᶿ sin θ` for a general vector `X`. -/
theorem dZ_on_vector (x X : Fin 2 → ℝ) : ∑ i, pd i heightZ x * X i = -(X 0) * Real.sin (x 0) := by
  simp only [Fin.sum_univ_two, pd_zero_two, pd_one_two, heightZ, Matrix.cons_val_zero]
  VERIPHY[de-partials]

/-- `dZ(cos θ ∂_θ − ∂_φ) = −sin θ cos θ`. -/
theorem dZ_on_mixed (x : Fin 2 → ℝ) :
    ∑ i, pd i heightZ x * ![Real.cos (x 0), (-1 : ℝ)] i = -(Real.sin (x 0) * Real.cos (x 0)) := by
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, pd_zero_two, pd_one_two,
    heightZ]
  VERIPHY[de-partials]

/-! ## exm-of-dz-gradient -/

/-- The round metric is `diag(1, sin²θ)`. -/
theorem sphere_metric_matrix (x : Fin 2 → ℝ) : metricAt sphere x = !![(1 : ℝ), 0; 0, Real.sin (x 0) ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.sphere_metric_matrix, dg-vf2-sphere-metric-test]

/-- Raising the index of `dZ = (−sin θ, 0)` with `g⁻¹ = diag(1, 1/sin²θ)` gives `(−sin θ, 0)`: the gradient field
`−sin θ ∂_θ`. -/
theorem raise_dZ (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (invMetric sphere x) *ᵥ ![-Real.sin (x 0), 0] = ![-Real.sin (x 0), 0] := by
  rw [sphere_invMetric x h]
  VERIPHY[dg-vf2-matrix]

/-! ## exr-of-ambient-differentials -/

/-- `dX = cos θ cos φ dθ − sin θ sin φ dφ` and `dY = cos θ sin φ dθ + sin θ cos φ dφ`. -/
theorem dX_dY_components (x : Fin 2 → ℝ) :
    pd 0 (sphereEmbed 0) x = Real.cos (x 0) * Real.cos (x 1) ∧ pd 1 (sphereEmbed 0) x = -(Real.sin (x 0) * Real.sin (x 1)) ∧
      pd 0 (sphereEmbed 1) x = Real.cos (x 0) * Real.sin (x 1) ∧ pd 1 (sphereEmbed 1) x = Real.sin (x 0) * Real.cos (x 1) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [pd_zero_two]; simp only [sphereEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    VERIPHY[de-partials]
  · rw [pd_one_two]; simp only [sphereEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    VERIPHY[de-partials]
  · rw [pd_zero_two]; simp only [sphereEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    VERIPHY[de-partials]
  · rw [pd_one_two]; simp only [sphereEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    VERIPHY[de-partials]

/-- The rotation fields `K₁, K₂, K₃` of the sphere (`killingX`, `killingY`, `killingZ`). -/
noncomputable def rotationField : Fin 3 → Fin 2 → (Fin 2 → ℝ) → ℝ := ![killingX, killingY, killingZ]

/-- The table `dX_b(K_a) = K_a(X_b) = −ε_abc X_c` for the three rotation fields and the three ambient coordinates
(for instance `dY(K₁) = −Z` and `dZ(K₁) = Y`). -/
theorem ambient_differentials_on_rotations (a b : Fin 3) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∑ i, rotationField a i x * pd i (sphereEmbed b) x = -∑ c, (KaluzaKlein.eps a b c : ℝ) * sphereEmbed c x := by
  have hX := dX_dY_components x
  have hZ := dZ_components x
  have hZ' : pd 0 (sphereEmbed 2) x = -Real.sin (x 0) ∧ pd 1 (sphereEmbed 2) x = 0 := by
    simpa [heightZ, sphereEmbed] using hZ
  fin_cases a <;> fin_cases b <;>
    simp [rotationField, killingX, killingY, killingZ, KaluzaKlein.eps, Fin.sum_univ_succ, hX.1, hX.2.1, hX.2.2.1,
      hX.2.2.2, hZ'.1, hZ'.2, sphereEmbed] <;>
    field_simp <;> ring_nf <;> simp [Real.sin_sq] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.ambient_differentials_on_rotations, dg-vf2-ambient-rotations-test]

/-! ## exr-of-height-ftc -/

/-- `∫_γ dZ = Z(γ(b)) − Z(γ(a))` for a curve `γ` on the sphere: the integrand `dZ(γ̇) = d/dt Z(γ(t))`. -/
theorem height_ftc (γ : ℝ → (Fin 2 → ℝ)) (v : ℝ → (Fin 2 → ℝ)) (a b : ℝ)
    (hγ : ∀ t i, HasDerivAt (fun s => γ s i) (v t i) t)
    (hint : IntervalIntegrable (fun t => ∑ i, pd i heightZ (γ t) * v t i) MeasureTheory.volume a b) :
    ∫ t in a..b, (∑ i, pd i heightZ (γ t) * v t i) = heightZ (γ b) - heightZ (γ a) := by
  have hL : ∀ t, HasFDerivAt heightZ (-Real.sin (γ t 0) • ContinuousLinearMap.proj 0) (γ t) := by
    intro t
    have h0 : HasFDerivAt (fun y : Fin 2 → ℝ => y 0) (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) (γ t) :=
      hasFDerivAt_apply 0 (γ t)
    have := (Real.hasDerivAt_cos (γ t 0)).comp_hasFDerivAt (γ t) h0
    simpa [heightZ] using this
  have hd : ∀ t, HasDerivAt (fun s => heightZ (γ s)) (∑ i, pd i heightZ (γ t) * v t i) t := by
    intro t
    have hγ' : HasDerivAt γ (v t) t := hasDerivAt_pi.mpr (hγ t)
    have hc := (hL t).comp_hasDerivAt t hγ'
    convert hc using 1
    rw [← velocity_rate heightZ γ (v t) t _ (hL t) (hγ t)]
    exact hc.deriv.symm
  exact intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hd t) hint
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.OneForms.height_ftc, dg-vf2-height-ftc-test]

/-- Along the meridian from the equator (`θ = π/2`, `Z = 0`) to the north pole (`θ = 0`, `Z = 1`),
`∫ dZ(γ̇) dt = ∫_{π/2}^{0} (−sin t) dt = 1`. -/
theorem meridian_integral : ∫ t in (Real.pi / 2)..(0 : ℝ), -Real.sin t = 1 := by
  VERIPHY[de-integral]

/-- Around a parallel `dZ(∂_φ) = 0` and the integral is `0`. -/
theorem parallel_integral : ∫ t in (0 : ℝ)..(2 * Real.pi), (0 : ℝ) = 0 := by
  VERIPHY[de-integral]

end KaluzaKlein.DifferentialGeometry.OneForms

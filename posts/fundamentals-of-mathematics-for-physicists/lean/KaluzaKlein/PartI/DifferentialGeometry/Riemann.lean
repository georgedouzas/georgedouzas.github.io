import Mathlib
import KaluzaKlein.PartI.DifferentialGeometry.RiemannLemmas

/-!
# Differential geometry, part "The Riemann tensor"

The claims of the section "The Riemann tensor" of `tutorial/part-i-09-differential-geometry.qmd`, one `/-! ## -/`
header per block. The Riemann tensor is the `CoordinateGeometry` asset's `riemann` (the formula
`R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ` applied to `christoffel`), the derived
quantities (`lowered`, `sectional`, `gauss`, …) are those of `RiemannLemmas`, and every surface of the section is a
metric `E(u) du² + G(u) dv²` (`diag2`), whose curvature `RiemannLemmas` computes once: the round sphere
(`KaluzaKlein.Geometry.sphere`, `E = 1`, `G = sin²θ`), the plane in polar coordinates (`KaluzaKlein.Geometry.polar`),
the torus `torus R r` (`E = r²`, `G = (R + r cos a)²`), the cone `cone α`, the cylinder `cylinder R0`, a surface of
revolution `revolution f` (`G = f²`), the hyperbolic plane `hyperbolic A` (coordinates `(y, x)`, so that `E = G =
A²/y²` depend on the first coordinate) and the sphere of radius `R0` as the scaled metric `scaled (R0²) sphere`.
The stereographic chart of the sphere (`stereo`) depends on both coordinates and is computed by hand, from the
closed forms of its Christoffel symbols, which hold at every point.

Claims that quantify over all connections (the commutator identity, the curvature with torsion) or over all surface
metrics `E dx² + 2F dx dy + G dy²` are stated as `def … : Prop` on the explicit coordinate model and tested; claims
about connections that are not Levi-Civita use `riemannOfConst`, the curvature of a connection with constant
coefficients, where the counterexamples of the chapter live.
-/

namespace KaluzaKlein.DifferentialGeometry.Riemann

open CoordinateGeometry Matrix Filter Topology
open KaluzaKlein.Geometry (sphere polar su2 covDeriv)

/-! ## Derivative facts for the surfaces of the section -/

/-- `deriv G` has derivative `G₂` at `u` when `G` has derivative `G'` near `u` and `G'` has derivative `G₂`. -/
theorem hasDerivAt_deriv_of_eventually {G G' : ℝ → ℝ} {u G₂ : ℝ} (h : ∀ᶠ y in 𝓝 u, HasDerivAt G (G' y) y)
    (h2 : HasDerivAt G' G₂ u) : HasDerivAt (deriv G) G₂ u :=
  h2.congr_of_eventuallyEq (h.mono fun _ hy => hy.deriv)

theorem hasDerivAt_sin_sq (t : ℝ) : HasDerivAt (fun t => Real.sin t ^ 2) (2 * Real.sin t * Real.cos t) t := by
  have := (Real.hasDerivAt_sin t).pow 2
  convert this using 1; push_cast; ring

theorem hasDerivAt_deriv_sin_sq (t : ℝ) :
    HasDerivAt (deriv fun t => Real.sin t ^ 2) (2 * Real.cos t ^ 2 - 2 * Real.sin t ^ 2) t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall hasDerivAt_sin_sq) ?_
  have := ((Real.hasDerivAt_sin t).mul (Real.hasDerivAt_cos t)).const_mul 2
  convert this using 1
  · funext s; ring
  · ring

theorem hasDerivAt_sq (t : ℝ) : HasDerivAt (fun t : ℝ => t ^ 2) (2 * t) t := by
  have := (hasDerivAt_id t).pow 2
  convert this using 1; push_cast; ring

theorem hasDerivAt_deriv_sq (t : ℝ) : HasDerivAt (deriv fun t : ℝ => t ^ 2) 2 t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall hasDerivAt_sq) ?_
  have := (hasDerivAt_id t).const_mul 2
  convert this using 1; ring

theorem hasDerivAt_torusG (R r t : ℝ) :
    HasDerivAt (fun a => (R + r * Real.cos a) ^ 2) (-(2 * r * Real.sin t * (R + r * Real.cos t))) t := by
  have := ((hasDerivAt_const t R).add ((Real.hasDerivAt_cos t).const_mul r)).pow 2
  convert this using 1; push_cast; ring

theorem hasDerivAt_deriv_torusG (R r t : ℝ) :
    HasDerivAt (deriv fun a => (R + r * Real.cos a) ^ 2)
      (-(2 * r * Real.cos t * (R + r * Real.cos t)) + 2 * r ^ 2 * Real.sin t ^ 2) t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall (hasDerivAt_torusG R r)) ?_
  have := (((Real.hasDerivAt_sin t).mul ((hasDerivAt_const t R).add ((Real.hasDerivAt_cos t).const_mul r))).const_mul
    (2 * r)).neg
  convert this using 1
  · funext s; ring
  · ring

theorem hasDerivAt_coneG (α t : ℝ) : HasDerivAt (fun ρ => ρ ^ 2 * Real.sin α ^ 2) (2 * t * Real.sin α ^ 2) t := by
  have := (hasDerivAt_sq t).mul_const (Real.sin α ^ 2)
  convert this using 1

theorem hasDerivAt_deriv_coneG (α t : ℝ) :
    HasDerivAt (deriv fun ρ => ρ ^ 2 * Real.sin α ^ 2) (2 * Real.sin α ^ 2) t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall (hasDerivAt_coneG α)) ?_
  have := ((hasDerivAt_id t).const_mul 2).mul_const (Real.sin α ^ 2)
  convert this using 1; ring

theorem hasDerivAt_deriv_const (c t : ℝ) : HasDerivAt (deriv fun _ : ℝ => c) 0 t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall fun y => hasDerivAt_const y c) ?_
  exact hasDerivAt_const t 0

/-- `G = f²` for a surface of revolution with `f` differentiable. -/
theorem hasDerivAt_revG {f : ℝ → ℝ} (hf : Differentiable ℝ f) (t : ℝ) :
    HasDerivAt (fun u => f u ^ 2) (2 * f t * deriv f t) t := by
  have := (hf t).hasDerivAt.pow 2
  convert this using 1; push_cast; ring

theorem hasDerivAt_deriv_revG {f : ℝ → ℝ} {t f₂ : ℝ} (hf : Differentiable ℝ f) (hf2 : HasDerivAt (deriv f) f₂ t) :
    HasDerivAt (deriv fun u => f u ^ 2) (2 * deriv f t ^ 2 + 2 * f t * f₂) t := by
  refine hasDerivAt_deriv_of_eventually (Filter.Eventually.of_forall (hasDerivAt_revG hf)) ?_
  have := (((hf t).hasDerivAt.mul hf2).const_mul 2)
  convert this using 1
  · funext s; ring
  · ring

/-- `E = G = A²/y²` of the hyperbolic plane, away from `y = 0`. -/
theorem hasDerivAt_hypG (A t : ℝ) (ht : t ≠ 0) : HasDerivAt (fun y => A ^ 2 / y ^ 2) (-(2 * A ^ 2) / t ^ 3) t := by
  have := (hasDerivAt_const t (A ^ 2)).div (hasDerivAt_sq t) (pow_ne_zero 2 ht)
  convert this using 1; field_simp; ring

theorem hasDerivAt_deriv_hypG (A t : ℝ) (ht : t ≠ 0) :
    HasDerivAt (deriv fun y => A ^ 2 / y ^ 2) (6 * A ^ 2 / t ^ 4) t := by
  refine hasDerivAt_deriv_of_eventually ((eventually_ne_nhds ht).mono fun y hy => hasDerivAt_hypG A y hy) ?_
  have h3 : HasDerivAt (fun y : ℝ => y ^ 3) (3 * t ^ 2) t := by
    have := (hasDerivAt_id t).pow 3
    convert this using 1; push_cast; ring
  have := (hasDerivAt_const t (-(2 * A ^ 2))).div h3 (pow_ne_zero 3 ht)
  convert this using 1; field_simp; ring

/-! ## The surfaces of the section as `diag2` metrics -/

/-- The round sphere is `diag2 1 sin²`. -/
theorem sphere_eq_diag2 : sphere = diag2 (fun _ => 1) (fun t => Real.sin t ^ 2) := rfl

/-- The polar metric of the plane is `diag2 1 r²`. -/
theorem polar_eq_diag2 : polar = diag2 (fun _ => 1) (fun t => t ^ 2) := rfl

/-- `K = 1` on the unit sphere, from the general formula for `E du² + G dv²` with `E = 1`, `G = sin²θ`. -/
theorem sphere_gauss_aux (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : gauss sphere x = 1 := by
  rw [sphere_eq_diag2, diag2_gauss (by simp) (by simp [h]) (hasDerivAt_const _ _) (hasDerivAt_sin_sq (x 0))
    (hasDerivAt_deriv_sin_sq (x 0))]
  field_simp; ring

/-- The torus of radii `R` (centre circle) and `r` (tube): `r² da² + (R + r cos a)² db²`. -/
noncomputable def torus (R r : ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  diag2 (fun _ => r ^ 2) (fun a => (R + r * Real.cos a) ^ 2)

/-- The cone of half-angle `α`: `dρ² + ρ² sin²α dφ²`. -/
noncomputable def cone (α : ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  diag2 (fun _ => 1) (fun ρ => ρ ^ 2 * Real.sin α ^ 2)

/-- The cylinder of radius `R0`: `dz² + R0² dφ²`. -/
noncomputable def cylinder (R0 : ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  diag2 (fun _ => 1) (fun _ => R0 ^ 2)

/-- A surface of revolution with arc-length profile: `du² + f(u)² dφ²`. -/
noncomputable def revolution (f : ℝ → ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  diag2 (fun _ => 1) (fun u => f u ^ 2)

/-- The hyperbolic plane `A² (dx² + dy²)/y²`, in the coordinates `(y, x)` (`x 0 = y`, `x 1 = x`). -/
noncomputable def hyperbolic (A : ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  diag2 (fun y => A ^ 2 / y ^ 2) (fun y => A ^ 2 / y ^ 2)

/-- The sphere of radius `R0`: `R0²` times the round metric. -/
noncomputable def sphereR (R0 : ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ := scaled (R0 ^ 2) sphere

/-- The flat torus `dx² + dy²`, a constant metric. -/
noncomputable def flatTorus : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ := fun i j _ => (1 : Matrix (Fin 2) (Fin 2) ℝ) i j

/-- The stereographic metric of the unit sphere, `4 (du² + dv²)/(1 + u² + v²)²`. -/
noncomputable def stereo : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![![fun x => 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2, fun _ => 0], ![fun _ => 0, fun x => 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2]]

/-- Cylindrical coordinates of Euclidean space: `dρ² + ρ² dφ² + dz²`. -/
noncomputable def cylindrical : Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ :=
  ![![fun _ => 1, fun _ => 0, fun _ => 0], ![fun _ => 0, fun x => x 0 ^ 2, fun _ => 0], ![fun _ => 0, fun _ => 0, fun _ => 1]]

/-- Spherical coordinates of Euclidean space: `dr² + r² dθ² + r² sin²θ dφ²`. -/
noncomputable def spherical : Fin 3 → Fin 3 → (Fin 3 → ℝ) → ℝ :=
  ![![fun _ => 1, fun _ => 0, fun _ => 0], ![fun _ => 0, fun x => x 0 ^ 2, fun _ => 0],
    ![fun _ => 0, fun _ => 0, fun x => x 0 ^ 2 * Real.sin (x 1) ^ 2]]

/-- A general surface metric `E dx² + 2F dx dy + G dy²` with arbitrary component functions. -/
noncomputable def surface (E F G : (Fin 2 → ℝ) → ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ := ![![E, F], ![F, G]]

/-! ## def-riemann -/

/-- The second covariant derivative of a vector field in coordinates,
`∇_μ ∇_ν X^ρ = ∂_μ(∇_ν X^ρ) + Γ^ρ_μλ ∇_ν X^λ − Γ^λ_μν ∇_λ X^ρ` (the covariant derivative of the `(1,1)` tensor
`∇_ν X^ρ`). -/
noncomputable def covDeriv2 {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ)
    (μ ν ρ : Fin n) (x : Fin n → ℝ) : ℝ :=
  pd μ (fun y => covDeriv g X ν ρ y) x + ∑ l, christoffel g ρ μ l x * covDeriv g X ν l x -
    ∑ l, christoffel g l μ ν x * covDeriv g X l ρ x

/-- The commutator `[∇_μ, ∇_ν] X^ρ = ∇_μ ∇_ν X^ρ − ∇_ν ∇_μ X^ρ`. -/
noncomputable def covCommutator {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ)
    (ρ μ ν : Fin n) (x : Fin n → ℝ) : ℝ :=
  covDeriv2 g X μ ν ρ x - covDeriv2 g X ν μ ρ x

/-- The commutator of covariant derivatives on a vector field is the Riemann tensor, `[∇_μ, ∇_ν] X^ρ = R^ρ_σμν X^σ`,
stated on the unit sphere for every twice differentiable vector field `X` (the model of the general identity, which
holds for every Levi-Civita connection). -/
def sphere_commutator_eq_riemann : Prop :=
  ∀ (X : Fin 2 → (Fin 2 → ℝ) → ℝ) (ρ μ ν : Fin 2) (x : Fin 2 → ℝ), Real.sin (x 0) ≠ 0 →
    (∀ k, ContDiff ℝ 2 (X k)) → covCommutator sphere X ρ μ ν x = ∑ σ, riemann sphere ρ σ μ ν x * X σ x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_commutator_eq_riemann, dg-cc2-commutator-test]

/-! ## exm-riemann-plane -/

/-- `Γ^r_φφ = −r` in polar coordinates. -/
theorem polar_christoffel_r_phiphi (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 0 1 1 x = -x 0 := by
  VERIPHY[dg-cc2-polar-christoffel]

/-- `Γ^φ_rφ = 1/r` in polar coordinates. -/
theorem polar_christoffel_phi_rphi (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 1 0 1 x = 1 / x 0 := by
  VERIPHY[dg-cc2-polar-christoffel]

/-- The plane is flat in polar coordinates: `R^r_φrφ = 0`. -/
theorem polar_riemann_r_phirphi (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : riemann polar 0 1 0 1 x = 0 := by
  rw [polar_eq_diag2, diag2_riemann_0_1_0_1 (by simp) (by simpa using h) (hasDerivAt_const _ _) (hasDerivAt_sq _)
    (hasDerivAt_deriv_sq _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.polar_riemann_r_phirphi, dg-cc2-polar-riemann-test]

/-- The Gauss curvature of the plane in polar coordinates is `0`. -/
theorem polar_gauss (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : gauss polar x = 0 := by
  rw [polar_eq_diag2, diag2_gauss (by simp) (by simpa using h) (hasDerivAt_const _ _) (hasDerivAt_sq _)
    (hasDerivAt_deriv_sq _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.polar_gauss, dg-cc2-polar-gauss-test]

/-- Every component of the Riemann tensor of the plane vanishes in polar coordinates. -/
theorem polar_riemann_zero (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (ρ σ μ ν : Fin 2) : riemann polar ρ σ μ ν x = 0 := by
  have hg := polar_gauss x h
  rw [polar_eq_diag2] at hg ⊢
  rw [diag2_riemann_all (by simp) (by simpa using h) (hasDerivAt_const _ _) (hasDerivAt_sq _) (hasDerivAt_deriv_sq _), hg]
  fin_cases ρ <;> fin_cases σ <;> fin_cases μ <;> fin_cases ν <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.polar_riemann_zero, dg-cc2-polar-riemann-zero-test]

/-- The arithmetic of the example: `−1 − 0 + 0 − (−r)(1/r) = 0` for `r ≠ 0`. -/
theorem polar_riemann_arith (r : ℝ) (h : r ≠ 0) : -1 - 0 + 0 - (-r) * (1 / r) = 0 := by
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.polar_riemann_arith, dg-cc2-polar-arith-test]

/-! ## exm-riemann-sphere -/

/-- `∂_θ(−sin θ cos θ) = −(cos²θ − sin²θ)`. -/
theorem deriv_neg_sin_cos (θ : ℝ) :
    deriv (fun θ => -(Real.sin θ * Real.cos θ)) θ = -(Real.cos θ ^ 2 - Real.sin θ ^ 2) := by
  VERIPHY[dg-cc2-deriv]

/-- `∂_θ(cos θ / sin θ) = −1/sin²θ`. -/
theorem deriv_cot (θ : ℝ) (h : Real.sin θ ≠ 0) :
    deriv (fun θ => Real.cos θ / Real.sin θ) θ = -1 / Real.sin θ ^ 2 := by
  VERIPHY[dg-cc2-deriv]

/-- The trigonometric step of `R^θ_φθφ`: `−(cos²θ − sin²θ) + sin θ cos θ · cos θ / sin θ = sin²θ`. -/
theorem sphere_riemann_arith (θ : ℝ) (h : Real.sin θ ≠ 0) :
    -(Real.cos θ ^ 2 - Real.sin θ ^ 2) + Real.sin θ * Real.cos θ * (Real.cos θ / Real.sin θ) = Real.sin θ ^ 2 := by
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_riemann_arith, dg-cc2-sphere-arith-test]

/-- The trigonometric step of `R^φ_θθφ`: `(−sin²θ − cos²θ)/sin²θ + cos²θ/sin²θ = −1`. -/
theorem sphere_riemann_arith2 (θ : ℝ) (h : Real.sin θ ≠ 0) :
    (-Real.sin θ ^ 2 - Real.cos θ ^ 2) / Real.sin θ ^ 2 + Real.cos θ ^ 2 / Real.sin θ ^ 2 = -1 := by
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_riemann_arith2, dg-cc2-sphere-arith2-test]

/-! ## exm-riemann-torus -/

/-- `R^a_bab = cos a (R + r cos a) / r` on the torus. -/
theorem torus_riemann_a_bab (R r : ℝ) (x : Fin 2 → ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos (x 0) ≠ 0) :
    riemann (torus R r) 0 1 0 1 x = Real.cos (x 0) * (R + r * Real.cos (x 0)) / r := by
  unfold torus
  rw [diag2_riemann_0_1_0_1 (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _) (hasDerivAt_torusG R r _)
    (hasDerivAt_deriv_torusG R r _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_riemann_a_bab, dg-cc2-torus-riemann-test]

/-- The Gauss curvature of the torus, `K = cos a / (r (R + r cos a))`. -/
theorem torus_gauss (R r : ℝ) (x : Fin 2 → ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos (x 0) ≠ 0) :
    gauss (torus R r) x = Real.cos (x 0) / (r * (R + r * Real.cos (x 0))) := by
  unfold torus
  rw [diag2_gauss (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _) (hasDerivAt_torusG R r _)
    (hasDerivAt_deriv_torusG R r _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_gauss, dg-cc2-torus-gauss-test]

/-- For `0 < r < R` the component `R^a_bab = c (R + r c)/r`, `c = cos a`, has the sign of `cos a`: positive on the
outer half of the torus, negative on the inner half. -/
theorem torus_curvature_sign (R r c : ℝ) (hr : 0 < r) (hR : r < R) (hc : |c| ≤ 1) :
    (0 < c → 0 < c * (R + r * c) / r) ∧ (c < 0 → c * (R + r * c) / r < 0) := by
  have h1 : 0 < R + r * c := by
    have := abs_le.mp hc
    nlinarith
  constructor
  · intro h; positivity
  · intro h; exact div_neg_of_neg_of_pos (mul_neg_of_neg_of_pos h h1) hr
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_curvature_sign, dg-cc2-torus-sign-test]

/-! ## exr-riemann-torus-sage -/

/-- `R^b_aab = −r cos a / (R + r cos a)` on the torus. -/
theorem torus_riemann_b_aab (R r : ℝ) (x : Fin 2 → ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos (x 0) ≠ 0) :
    riemann (torus R r) 1 0 0 1 x = -(r * Real.cos (x 0)) / (R + r * Real.cos (x 0)) := by
  unfold torus
  rw [diag2_riemann_1_0_0_1 (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _) (hasDerivAt_torusG R r _)
    (hasDerivAt_deriv_torusG R r _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_riemann_b_aab, dg-cc2-torus-riemann-b-test]

/-- The components `R^a_aab` and `R^b_bab` of the torus vanish: only four components are non-zero. -/
theorem torus_riemann_diag_zero (R r : ℝ) (x : Fin 2 → ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos (x 0) ≠ 0) :
    riemann (torus R r) 0 0 0 1 x = 0 ∧ riemann (torus R r) 1 1 0 1 x = 0 := by
  unfold torus
  exact ⟨diag2_riemann_0_0_0_1 (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _) (hasDerivAt_torusG R r _)
    (hasDerivAt_deriv_torusG R r _), diag2_riemann_1_1_0_1 (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _)
    (hasDerivAt_torusG R r _) (hasDerivAt_deriv_torusG R r _)⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_riemann_diag_zero, dg-cc2-torus-riemann-diag-test]

/-- `R^b_aab = −g_aa R^a_bab / g_bb`: the closed forms satisfy `−r² · (c (R + r c)/r) / (R + r c)² = −r c/(R + r c)`. -/
theorem torus_lowered_relation (R r c : ℝ) (hr : r ≠ 0) (hR : R + r * c ≠ 0) :
    -(r ^ 2) * (c * (R + r * c) / r) / (R + r * c) ^ 2 = -(r * c) / (R + r * c) := by
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_lowered_relation, dg-cc2-torus-relation-test]

/-- Both components vanish on the top and bottom circles `a = ±π/2`, where `cos a = 0`. -/
theorem torus_riemann_top (R r b : ℝ) (hr : r ≠ 0) (hR : R ≠ 0) :
    riemann (torus R r) 0 1 0 1 ![Real.pi / 2, b] = 0 ∧ riemann (torus R r) 1 0 0 1 ![Real.pi / 2, b] = 0 ∧
      riemann (torus R r) 0 1 0 1 ![-(Real.pi / 2), b] = 0 ∧ riemann (torus R r) 1 0 0 1 ![-(Real.pi / 2), b] = 0 := by
  have h1 : R + r * Real.cos (![Real.pi / 2, b] 0) ≠ 0 := by simp [hR]
  have h2 : R + r * Real.cos (![-(Real.pi / 2), b] 0) ≠ 0 := by simp [hR]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [torus_riemann_a_bab R r _ hr h1]; simp
  · rw [torus_riemann_b_aab R r _ hr h1]; simp
  · rw [torus_riemann_a_bab R r _ hr h2]; simp
  · rw [torus_riemann_b_aab R r _ hr h2]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_riemann_top, dg-cc2-torus-top-test]

/-! ## exr-riemann-flat-chart -/

/-- A metric with constant components has vanishing Christoffel symbols. -/
theorem christoffel_of_const {n : ℕ} (c : Matrix (Fin n) (Fin n) ℝ) (k i j : Fin n) (x : Fin n → ℝ) :
    christoffel (fun i j _ => c i j) k i j x = 0 :=
  christoffel_const c k i j x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.christoffel_of_const, dg-cc2-const-christoffel-test]

/-- A metric with constant components is flat: its Riemann tensor vanishes. -/
theorem riemann_of_const {n : ℕ} (c : Matrix (Fin n) (Fin n) ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) :
    riemann (fun i j _ => c i j) ρ σ μ ν x = 0 :=
  riemann_const c ρ σ μ ν x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.riemann_of_const, dg-cc2-const-riemann-test]

/-- The flat torus `dx² + dy²` is flat. -/
theorem flatTorus_riemann (ρ σ μ ν : Fin 2) (x : Fin 2 → ℝ) : riemann flatTorus ρ σ μ ν x = 0 :=
  riemann_const _ ρ σ μ ν x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.flatTorus_riemann, dg-cc2-flattorus-riemann-test]

/-- Euclidean space is flat in cylindrical and in spherical coordinates: every Riemann component vanishes on the
chart domains (`ρ ≠ 0`; `r ≠ 0`, `sin θ ≠ 0`). -/
def euclidean_charts_flat : Prop :=
  (∀ (ρ σ μ ν : Fin 3) (x : Fin 3 → ℝ), x 0 ≠ 0 → riemann cylindrical ρ σ μ ν x = 0) ∧
    (∀ (ρ σ μ ν : Fin 3) (x : Fin 3 → ℝ), x 0 ≠ 0 → Real.sin (x 1) ≠ 0 → riemann spherical ρ σ μ ν x = 0)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.euclidean_charts_flat, dg-cc2-euclidean-flat-test]

/-! ## exr-riemann-cone -/

/-- `Γ^ρ_φφ = −ρ sin²α` on the cone. -/
theorem cone_christoffel_rho_phiphi (α : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hα : Real.sin α ≠ 0) :
    christoffel (cone α) 0 1 1 x = -(x 0 * Real.sin α ^ 2) := by
  unfold cone
  rw [diag2_christoffel _ _ x (by simp) (by simp [h, hα])]
  simp [(hasDerivAt_coneG α (x 0)).deriv]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_christoffel_rho_phiphi, dg-cc2-cone-christoffel-test]

/-- `Γ^φ_ρφ = 1/ρ` on the cone. -/
theorem cone_christoffel_phi_rhophi (α : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hα : Real.sin α ≠ 0) :
    christoffel (cone α) 1 0 1 x = 1 / x 0 := by
  unfold cone
  rw [diag2_christoffel _ _ x (by simp) (by simp [h, hα])]
  simp [(hasDerivAt_coneG α (x 0)).deriv]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_christoffel_phi_rhophi, dg-cc2-cone-christoffel2-test]

/-- The cone is flat away from the apex: `R^ρ_φρφ = 0`. -/
theorem cone_riemann (α : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hα : Real.sin α ≠ 0) : riemann (cone α) 0 1 0 1 x = 0 := by
  unfold cone
  rw [diag2_riemann_0_1_0_1 (by simp) (by simp [h, hα]) (hasDerivAt_const _ _) (hasDerivAt_coneG α _)
    (hasDerivAt_deriv_coneG α _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_riemann, dg-cc2-cone-riemann-test]

/-- The arithmetic of the solution: `−sin²α + ρ sin²α · (1/ρ) = 0` for `ρ ≠ 0`. -/
theorem cone_riemann_arith (α ρ : ℝ) (h : ρ ≠ 0) : -Real.sin α ^ 2 + ρ * Real.sin α ^ 2 * (1 / ρ) = 0 := by
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_riemann_arith, dg-cc2-cone-arith-test]

/-! ## def-riemann-lowered -/

/-- The lowered tensor of a surface `E du² + G dv²` is antisymmetric in its last pair, antisymmetric in its first
pair, symmetric under exchange of the pairs, and satisfies the first Bianchi identity. -/
theorem diag2_lowered_symmetries {E G : ℝ → ℝ} {E₁ G₁ G₂ : ℝ} {x : Fin 2 → ℝ} (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0)
    (hE1 : HasDerivAt E E₁ (x 0)) (hG1 : HasDerivAt G G₁ (x 0)) (hG2 : HasDerivAt (deriv G) G₂ (x 0))
    (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ ν μ x = -lowered (diag2 E G) ρ σ μ ν x ∧
      lowered (diag2 E G) σ ρ μ ν x = -lowered (diag2 E G) ρ σ μ ν x ∧
      lowered (diag2 E G) μ ν ρ σ x = lowered (diag2 E G) ρ σ μ ν x ∧
      lowered (diag2 E G) ρ σ μ ν x + lowered (diag2 E G) ρ μ ν σ x + lowered (diag2 E G) ρ ν σ μ x = 0 :=
  ⟨(diag2_lowered_antisymm_last hE hG hE1 hG1 hG2 ρ σ μ ν).symm ▸ by ring_nf,
    by rw [diag2_lowered_antisymm_first hE hG hE1 hG1 hG2 ρ σ μ ν]; ring,
    (diag2_lowered_pair_symm hE hG hE1 hG1 hG2 ρ σ μ ν).symm, diag2_bianchi hE hG hE1 hG1 hG2 ρ σ μ ν⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.diag2_lowered_symmetries, dg-cc2-lowered-symmetries-test]

/-- The lowered tensor of `SU(2)` has the four symmetries for all `81` components (on the chart domain). -/
def su2_lowered_symmetries : Prop :=
  ∀ (ρ σ μ ν : Fin 3) (x : Fin 3 → ℝ), Real.sin (x 0) ≠ 0 →
    lowered su2 ρ σ ν μ x = -lowered su2 ρ σ μ ν x ∧ lowered su2 σ ρ μ ν x = -lowered su2 ρ σ μ ν x ∧
      lowered su2 μ ν ρ σ x = lowered su2 ρ σ μ ν x ∧
      lowered su2 ρ σ μ ν x + lowered su2 ρ μ ν σ x + lowered su2 ρ ν σ μ x = 0
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_lowered_symmetries, dg-cc2-su2-symmetries-test]

/-- The number of independent components of the curvature in `n` dimensions, `n²(n² − 1)/12`. -/
def independentComponents (n : ℕ) : ℕ := n ^ 2 * (n ^ 2 - 1) / 12

/-- `n²(n² − 1)/12` is `1, 6, 20` in dimensions `2, 3, 4` (and `0` below). -/
theorem independentComponents_values (n : Fin 5) : independentComponents n = ![0, 0, 1, 6, 20] n := by
  fin_cases n <;> decide
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.independentComponents_values, dg-cc2-independent-test]

/-! ## exm-lowered-sphere -/

/-- The lowered tensor of the unit sphere: `R_θφθφ = sin²θ`, `R_φθθφ = −sin²θ`, `R_θφφθ = −sin²θ`,
`R_φθφθ = sin²θ`. -/
theorem sphere_lowered_components (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered sphere 0 1 0 1 x = Real.sin (x 0) ^ 2 ∧ lowered sphere 1 0 0 1 x = -Real.sin (x 0) ^ 2 ∧
      lowered sphere 0 1 1 0 x = -Real.sin (x 0) ^ 2 ∧ lowered sphere 1 0 1 0 x = Real.sin (x 0) ^ 2 := by
  have hg := KaluzaKlein.DifferentialGeometry.Riemann.sphere_gauss_aux x h
  rw [sphere_eq_diag2] at hg ⊢
  have hf := diag2_lowered_form (by simp) (by simp [h]) (hasDerivAt_const _ _) (hasDerivAt_sin_sq (x 0))
    (hasDerivAt_deriv_sin_sq (x 0))
  simp only [hf, hg, diag2]
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_lowered_components, dg-cc2-sphere-lowered-test]

/-- `g_φφ R^φ_θθφ = sin²θ · (−1) = −sin²θ`, consistent with `R_φθθφ`. -/
theorem sphere_lowered_1_0_0_1_direct (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    sphere 1 1 x * riemann sphere 1 0 0 1 x = -Real.sin (x 0) ^ 2 := by
  rw [KaluzaKlein.Geometry.sphere_riemann_1_0_0_1 x h]; simp [sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_lowered_1_0_0_1_direct, dg-cc2-sphere-lowered-direct-test]

/-! ## exm-lowered-antisymmetry -/

/-- A torsion-free connection that is not metric-compatible with `δ`: `Γ^0_01 = Γ^0_10 = Γ^1_01 = Γ^1_10 = c`, all
other coefficients zero. -/
def nonMetricConn (c : ℝ) : Fin 2 → Fin 2 → Fin 2 → ℝ :=
  ![![![0, c], ![c, 0]], ![![0, c], ![c, 0]]]

/-- For the (torsion-free, non-metric) connection `nonMetricConn c`, the lowered component `R_0001 = R^0_001 = c²`
is not zero: antisymmetry in the first pair needs metric compatibility. -/
theorem nonMetric_not_antisymm (c : ℝ) : riemannOfConst (nonMetricConn c) 0 0 0 1 = c ^ 2 := by
  simp [riemannOfConst, nonMetricConn, Fin.sum_univ_two]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.nonMetric_not_antisymm, dg-cc2-nonmetric-test]

/-- A metric-compatible (with `δ`) connection with torsion: `Γ^0_01 = Γ^0_11 = c`, `Γ^1_00 = Γ^1_10 = −c`. -/
def torsionConn (c : ℝ) : Fin 2 → Fin 2 → Fin 2 → ℝ :=
  ![![![0, c], ![0, c]], ![![-c, 0], ![-c, 0]]]

/-- For the (metric-compatible, torsionful) connection `torsionConn c`, `R_0110 = c² ≠ R_1001 = 0`: the pair
symmetry needs torsion freedom. -/
theorem torsion_not_pair_symm (c : ℝ) :
    riemannOfConst (torsionConn c) 0 1 1 0 = c ^ 2 ∧ riemannOfConst (torsionConn c) 1 0 0 1 = 0 := by
  constructor <;> simp [riemannOfConst, torsionConn, Fin.sum_univ_two] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torsion_not_pair_symm, dg-cc2-torsion-test]

/-! ## exm-lowered-counting -/

/-- The component counts in dimension `n ≤ 4`: `n⁴` components of `R^ρ_σμν`, `n²(n² − 1)/12` independent ones,
`n(n + 1)/2` in the symmetric Ricci tensor, and the rest in the Weyl tensor (`10` in four dimensions, `0` in
three). -/
theorem component_counts (n : Fin 5) :
    (n : ℕ) ^ 4 = ![0, 1, 16, 81, 256] n ∧ independentComponents n = ![0, 0, 1, 6, 20] n ∧
      (n : ℕ) * (n + 1) / 2 = ![0, 1, 3, 6, 10] n ∧
      (n = 4 → independentComponents n - (n : ℕ) * (n + 1) / 2 = 10) ∧
      (n = 3 → independentComponents n = (n : ℕ) * (n + 1) / 2) := by
  fin_cases n <;> decide
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.component_counts, dg-cc2-counts-test]

/-! ## exr-riemann-symmetries -/

/-- On the sphere `R_θφθφ = 1 · sin²θ` and `R_φθθφ = sin²θ · (−1)` are opposite (antisymmetry in the first pair), and
`R_θφφθ = −R_θφθφ` equals `R_φθθφ` (pair exchange). -/
theorem sphere_symmetries_check (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered sphere 0 1 0 1 x = 1 * Real.sin (x 0) ^ 2 ∧ lowered sphere 1 0 0 1 x = Real.sin (x 0) ^ 2 * (-1) ∧
      lowered sphere 1 0 0 1 x = -lowered sphere 0 1 0 1 x ∧ lowered sphere 0 1 1 0 x = lowered sphere 1 0 0 1 x := by
  obtain ⟨h1, h2, h3, _⟩ := sphere_lowered_components x h
  refine ⟨by rw [h1]; ring, by rw [h2]; ring, by rw [h1, h2], by rw [h2, h3]⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_symmetries_check, dg-cc2-sphere-symmetries-test]

/-! ## exr-counting-components -/

/-- With `m = n(n − 1)/2` pairs, a symmetric `m × m` matrix has `m(m + 1)/2` entries and the Bianchi identity removes
`C(n, 4) = n(n−1)(n−2)(n−3)/24` of them: the count is `n²(n² − 1)/12`. -/
theorem count_formula (n m : ℝ) (hm : m = n * (n - 1) / 2) :
    m * (m + 1) / 2 - n * (n - 1) * (n - 2) * (n - 3) / 24 = n ^ 2 * (n ^ 2 - 1) / 12 := by
  VERIPHY[dg-cc2-count]

/-- The counts in dimensions `2, 3, 4`: `m = 1, 3, 6` pairs, `1, 6, 21` entries, `0, 0, 1` Bianchi conditions, so
`1, 6, 20` components, and `16 · 15 / 12 = 20`. -/
theorem count_table (n : Fin 5) :
    (n : ℕ) * (n - 1) / 2 = ![0, 0, 1, 3, 6] n ∧
      ((n : ℕ) * (n - 1) / 2) * ((n : ℕ) * (n - 1) / 2 + 1) / 2 = ![0, 0, 1, 6, 21] n ∧
      Nat.choose n 4 = ![0, 0, 0, 0, 1] n ∧
      ((n : ℕ) * (n - 1) / 2) * ((n : ℕ) * (n - 1) / 2 + 1) / 2 - Nat.choose n 4 = ![0, 0, 1, 6, 20] n ∧
      (n = 4 → 16 * 15 / 12 = (20 : ℕ)) := by
  fin_cases n <;> decide
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.count_table, dg-cc2-count-table-test]

/-! ## exr-riemann-bianchi-sage -/

/-- `R_θφθφ = 1/16` on `SU(2)`. -/
theorem su2_lowered_0_1_0_1 (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : lowered su2 0 1 0 1 x = 1 / 16 := by
  unfold lowered
  rw [Fin.sum_univ_three, KaluzaKlein.Geometry.su2_riemann_0_1_0_1 x h]
  simp [su2]
  norm_num
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_lowered_0_1_0_1, dg-cc2-su2-lowered-test]

/-- `g_θθ g_φφ − g_θφ² = 1/16` on `SU(2)`: the value for sectional curvature one. -/
theorem su2_gram_0_1 (x : Fin 3 → ℝ) : su2 0 0 x * su2 1 1 x - su2 0 1 x ^ 2 = 1 / 16 := by
  simp [su2]; norm_num
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_gram_0_1, dg-cc2-su2-gram-test]

/-! ## def-sectional-curvature -/

/-- The bivector `X ∧ Y` in components, `X^ρ Y^σ − X^σ Y^ρ`. -/
def biv {n : ℕ} (X Y : Fin n → ℝ) (ρ σ : Fin n) : ℝ := X ρ * Y σ - X σ * Y ρ

/-- A sum against an antisymmetric array only sees the bivector: `∑ T_ρσ X^ρ Y^σ = ½ ∑ T_ρσ (X^ρ Y^σ − X^σ Y^ρ)`. -/
theorem antisym_sum {n : ℕ} (T : Fin n → Fin n → ℝ) (hT : ∀ ρ σ, T σ ρ = -T ρ σ) (X Y : Fin n → ℝ) :
    ∑ ρ, ∑ σ, T ρ σ * (X ρ * Y σ) = (1 / 2) * ∑ ρ, ∑ σ, T ρ σ * biv X Y ρ σ := by
  have h2 : ∑ ρ, ∑ σ, T ρ σ * (X σ * Y ρ) = -∑ ρ, ∑ σ, T ρ σ * (X ρ * Y σ) := by
    rw [Finset.sum_comm, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun σ _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun ρ _ => ?_
    rw [hT ρ σ]; ring
  have h3 : ∑ ρ, ∑ σ, T ρ σ * biv X Y ρ σ = ∑ ρ, ∑ σ, T ρ σ * (X ρ * Y σ) - ∑ ρ, ∑ σ, T ρ σ * (X σ * Y ρ) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun ρ _ => ?_
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun σ _ => ?_
    unfold biv; ring
  rw [h3, h2]; ring

/-- The quadratic form of an array antisymmetric in each pair is a quarter of its contraction with `X ∧ Y` twice. -/
theorem quadForm_eq_biv {n : ℕ} (T : Fin n → Fin n → Fin n → Fin n → ℝ) (h1 : ∀ ρ σ μ ν, T σ ρ μ ν = -T ρ σ μ ν)
    (h2 : ∀ ρ σ μ ν, T ρ σ ν μ = -T ρ σ μ ν) (X Y : Fin n → ℝ) :
    quadForm T X Y = (1 / 4) * ∑ ρ, ∑ σ, ∑ μ, ∑ ν, T ρ σ μ ν * biv X Y ρ σ * biv X Y μ ν := by
  unfold quadForm
  have hin : ∀ ρ σ, ∑ μ, ∑ ν, T ρ σ μ ν * X ρ * Y σ * X μ * Y ν =
      X ρ * Y σ * ((1 / 2) * ∑ μ, ∑ ν, T ρ σ μ ν * biv X Y μ ν) := by
    intro ρ σ
    rw [← antisym_sum (fun μ ν => T ρ σ μ ν) (fun μ ν => h2 ρ σ μ ν) X Y, Finset.mul_sum]
    refine Finset.sum_congr rfl fun μ _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun ν _ => ?_
    ring
  simp only [hin]
  have hout := antisym_sum (fun ρ σ => (1 / 2) * ∑ μ, ∑ ν, T ρ σ μ ν * biv X Y μ ν)
    (fun ρ σ => by simp only [h1 ρ σ, neg_mul, Finset.sum_neg_distrib, mul_neg]) X Y
  simp only [mul_comm (X _ * Y _)] at hout ⊢
  rw [hout, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ρ _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun σ _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun μ _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ν _ => ?_
  ring

/-- `(aX + bY) ∧ (cX + dY) = (ad − bc) X ∧ Y`. -/
theorem biv_change {n : ℕ} (X Y : Fin n → ℝ) (a b c d : ℝ) (ρ σ : Fin n) :
    biv (fun i => a * X i + b * Y i) (fun i => c * X i + d * Y i) ρ σ = (a * d - b * c) * biv X Y ρ σ := by
  unfold biv; ring

/-- The numerator of the sectional curvature is multiplied by `(ad − bc)²` under `X' = aX + bY`, `Y' = cX + dY`, for
every array antisymmetric in each pair (in every dimension). -/
theorem quadForm_change {n : ℕ} (T : Fin n → Fin n → Fin n → Fin n → ℝ) (h1 : ∀ ρ σ μ ν, T σ ρ μ ν = -T ρ σ μ ν)
    (h2 : ∀ ρ σ μ ν, T ρ σ ν μ = -T ρ σ μ ν) (X Y : Fin n → ℝ) (a b c d : ℝ) :
    quadForm T (fun i => a * X i + b * Y i) (fun i => c * X i + d * Y i) = (a * d - b * c) ^ 2 * quadForm T X Y := by
  rw [quadForm_eq_biv T h1 h2, quadForm_eq_biv T h1 h2]
  simp only [biv_change, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ρ _ => Finset.sum_congr rfl fun σ _ => Finset.sum_congr rfl fun μ _ =>
    Finset.sum_congr rfl fun ν _ => ?_
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.quadForm_change, dg-cc2-quadform-change-test]

/-- The Gram determinant is `det (M G Mᵀ) = (det M)² det G` for the `2 × 2` matrix `M` of the change of basis. -/
theorem gram_change (a b c d gXX gXY gYY : ℝ) :
    Matrix.det (!![a, b; c, d] * !![gXX, gXY; gXY, gYY] * (!![a, b; c, d])ᵀ) =
      (Matrix.det !![a, b; c, d]) ^ 2 * Matrix.det !![gXX, gXY; gXY, gYY] := by
  VERIPHY[dg-cc2-gram]

/-- The sectional curvature does not depend on the basis of the plane: `K(aX + bY, cX + dY) = K(X, Y)` when
`ad − bc ≠ 0`, for every metric whose lowered tensor is antisymmetric in each pair at `x`. -/
theorem sectional_basis_independent {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x : Fin n → ℝ)
    (hg : ∀ i j, g i j x = g j i x) (h1 : ∀ ρ σ μ ν, lowered g σ ρ μ ν x = -lowered g ρ σ μ ν x)
    (h2 : ∀ ρ σ μ ν, lowered g ρ σ ν μ x = -lowered g ρ σ μ ν x) (X Y : Fin n → ℝ) (a b c d : ℝ)
    (hdet : a * d - b * c ≠ 0) :
    sectional g (fun i => a * X i + b * Y i) (fun i => c * X i + d * Y i) x = sectional g X Y x := by
  unfold sectional
  rw [quadForm_change _ h1 h2]
  have hgram : gramDet g (fun i => a * X i + b * Y i) (fun i => c * X i + d * Y i) x = (a * d - b * c) ^ 2 * gramDet g X Y x := by
    unfold gramDet inner
    have hs : ∀ U V : Fin n → ℝ, ∑ i, ∑ j, g i j x * U i * V j = ∑ i, ∑ j, g i j x * V i * U j := by
      intro U V
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      rw [hg j i]; ring
    have e1 : ∀ U V : Fin n → ℝ, ∑ i, ∑ j, g i j x * (a * U i + b * V i) * (c * U j + d * V j) =
        a * c * ∑ i, ∑ j, g i j x * U i * U j + a * d * ∑ i, ∑ j, g i j x * U i * V j +
          b * c * ∑ i, ∑ j, g i j x * V i * U j + b * d * ∑ i, ∑ j, g i j x * V i * V j := by
      intro U V
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      ring
    have e2 : ∀ U V : Fin n → ℝ, ∑ i, ∑ j, g i j x * (a * U i + b * V i) * (a * U j + b * V j) =
        a * a * ∑ i, ∑ j, g i j x * U i * U j + a * b * ∑ i, ∑ j, g i j x * U i * V j +
          b * a * ∑ i, ∑ j, g i j x * V i * U j + b * b * ∑ i, ∑ j, g i j x * V i * V j := by
      intro U V
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      ring
    have e3 : ∀ U V : Fin n → ℝ, ∑ i, ∑ j, g i j x * (c * U i + d * V i) * (c * U j + d * V j) =
        c * c * ∑ i, ∑ j, g i j x * U i * U j + c * d * ∑ i, ∑ j, g i j x * U i * V j +
          d * c * ∑ i, ∑ j, g i j x * V i * U j + d * d * ∑ i, ∑ j, g i j x * V i * V j := by
      intro U V
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      ring
    rw [e1, e2, e3, hs Y X]
    ring
  rw [hgram, pow_two, pow_two, mul_assoc, mul_assoc, mul_div_mul_left _ _ hdet, mul_div_mul_left _ _ hdet]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sectional_basis_independent, dg-cc2-sectional-basis-test]

/-- The sectional curvature of a surface `E du² + G dv²` in the coordinate plane is its Gauss curvature: in two
dimensions the single sectional curvature is the Gauss curvature, which determines the whole lowered tensor
(`diag2_lowered_form`). -/
theorem diag2_sectional_eq_gauss {E G : ℝ → ℝ} {E₁ G₁ G₂ : ℝ} {x : Fin 2 → ℝ} (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0)
    (hE1 : HasDerivAt E E₁ (x 0)) (hG1 : HasDerivAt G G₁ (x 0)) (hG2 : HasDerivAt (deriv G) G₂ (x 0)) :
    sectional (diag2 E G) ![1, 0] ![0, 1] x = gauss (diag2 E G) x :=
  diag2_sectional hE hG hE1 hG1 hG2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.diag2_sectional_eq_gauss, dg-cc2-sectional-gauss-test]

/-! ## exm-sectional-sphere -/

/-- `K(∂_θ, ∂_φ) = R_θφθφ / (g_θθ g_φφ) = sin²θ / sin²θ = 1` on the unit sphere. -/
theorem sphere_sectional (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : sectional sphere ![1, 0] ![0, 1] x = 1 := by
  rw [sphere_eq_diag2, diag2_sectional (by simp) (by simp [h]) (hasDerivAt_const _ _) (hasDerivAt_sin_sq (x 0))
    (hasDerivAt_deriv_sin_sq (x 0)), ← sphere_eq_diag2]
  exact sphere_gauss_aux x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_sectional, dg-cc2-sphere-sectional-test]

/-- The sphere of radius `R0` has sectional curvature `1/R0²`. -/
theorem sphereR_sectional (R0 : ℝ) (hR : R0 ≠ 0) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    sectional (sphereR R0) ![1, 0] ![0, 1] x = 1 / R0 ^ 2 := by
  unfold sphereR
  rw [scaled_sectional _ _ _ (pow_ne_zero 2 hR), sphere_sectional x h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphereR_sectional, dg-cc2-sphereR-sectional-test]

/-! ## exm-sectional-torus -/

/-- `K(∂_a, ∂_b) = R_abab/(g_aa g_bb) = cos a / (r (R + r cos a))` on the torus. -/
theorem torus_sectional (R r : ℝ) (x : Fin 2 → ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos (x 0) ≠ 0) :
    sectional (torus R r) ![1, 0] ![0, 1] x = Real.cos (x 0) / (r * (R + r * Real.cos (x 0))) := by
  rw [← torus_gauss R r x hr hR]
  unfold torus
  exact diag2_sectional (by simpa using hr) (by simpa using hR) (hasDerivAt_const _ _) (hasDerivAt_torusG R r _)
    (hasDerivAt_deriv_torusG R r _)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_sectional, dg-cc2-torus-sectional-test]

/-- The Gauss curvature of the torus, `c / (r (R + r c))` with `c = cos a`, increases with `c` on `[−1, 1]` for
`0 < r < R`: it is largest on the outer equator (`c = 1`) and most negative on the inner one (`c = −1`). -/
theorem torus_gauss_mono (R r c₁ c₂ : ℝ) (hr : 0 < r) (hR : r < R) (h1 : -1 ≤ c₁) (h2 : c₂ ≤ 1) (hc : c₁ ≤ c₂) :
    c₁ / (r * (R + r * c₁)) ≤ c₂ / (r * (R + r * c₂)) := by
  have p1 : 0 < R + r * c₁ := by nlinarith
  have p2 : 0 < R + r * c₂ := by nlinarith
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_pos hr p1, mul_pos hr p2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_gauss_mono, dg-cc2-torus-mono-test]

/-! ## exm-sectional-su2 -/

/-- Every sectional curvature of the bi-invariant metric of `SU(2)` equals `1`: for every plane `X, Y` at every point
of the chart (where the Gram determinant is non-zero). -/
def su2_sectional_one : Prop :=
  ∀ (X Y : Fin 3 → ℝ) (x : Fin 3 → ℝ), Real.sin (x 0) ≠ 0 → gramDet su2 X Y x ≠ 0 → sectional su2 X Y x = 1
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_sectional_one, dg-cc2-su2-sectional-test]

/-- `Γ^φ_θψ = −1/(2 sin θ)` of `SU(2)` is unbounded near `θ = 0`: `|Γ^φ_θψ| ≥ 1/(2θ)` for `0 < θ < π`. -/
theorem su2_christoffel_unbounded (θ : ℝ) (h0 : 0 < θ) (hπ : θ < Real.pi) :
    1 / (2 * θ) ≤ |(-1 / (2 * Real.sin θ))| := by
  have hs : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi h0 hπ
  have hle : Real.sin θ ≤ θ := Real.sin_le h0.le
  rw [abs_div, abs_neg, abs_one, abs_of_pos (by positivity : 0 < 2 * Real.sin θ)]
  exact one_div_le_one_div_of_le (by positivity) (by linarith)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_christoffel_unbounded, dg-cc2-su2-unbounded-test]

/-- The chart's Christoffel symbol `Γ^φ_θψ = −1/(2 sin θ)` (on the chart domain). -/
theorem su2_christoffel_phi_thetapsi (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel su2 1 0 2 x = -1 / (2 * Real.sin (x 0)) := by
  rw [KaluzaKlein.Geometry.su2_christoffel_1_0_2 x h]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_christoffel_phi_thetapsi, dg-cc2-su2-christoffel-test]

/-! ## exr-sphere-radius -/

/-- Scaling a metric by `λ ≠ 0` leaves the Christoffel symbols unchanged. -/
theorem scaled_christoffel_eq {n : ℕ} (c : ℝ) (hc : c ≠ 0) (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (k i j : Fin n)
    (x : Fin n → ℝ) : christoffel (scaled c g) k i j x = christoffel g k i j x :=
  scaled_christoffel c g x hc k i j
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.scaled_christoffel_eq, dg-cc2-scaled-christoffel-test]

/-- Scaling a metric by `λ ≠ 0` leaves the mixed Riemann components unchanged. -/
theorem scaled_riemann_eq {n : ℕ} (c : ℝ) (hc : c ≠ 0) (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n)
    (x : Fin n → ℝ) : riemann (scaled c g) ρ σ μ ν x = riemann g ρ σ μ ν x :=
  scaled_riemann c g x hc ρ σ μ ν
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.scaled_riemann_eq, dg-cc2-scaled-riemann-test]

/-- The lowered tensor scales by `λ` and the Gram determinant by `λ²`. -/
theorem scaled_lowered_gram {n : ℕ} (c : ℝ) (hc : c ≠ 0) (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n)
    (X Y : Fin n → ℝ) (x : Fin n → ℝ) :
    lowered (scaled c g) ρ σ μ ν x = c * lowered g ρ σ μ ν x ∧ gramDet (scaled c g) X Y x = c ^ 2 * gramDet g X Y x :=
  ⟨scaled_lowered c g x hc ρ σ μ ν, scaled_gramDet c g x X Y⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.scaled_lowered_gram, dg-cc2-scaled-lowered-test]

/-- The Christoffel symbols and mixed Riemann components of the sphere of radius `R0` are those of the unit sphere
(`Γ^θ_φφ = −sin θ cos θ`, `R^θ_φθφ = sin²θ`, `R^φ_θθφ = −1`), and its Ricci tensor is unchanged. -/
theorem sphereR_unchanged (R0 : ℝ) (hR : R0 ≠ 0) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel (sphereR R0) 0 1 1 x = -(Real.sin (x 0) * Real.cos (x 0)) ∧
      riemann (sphereR R0) 0 1 0 1 x = Real.sin (x 0) ^ 2 ∧ riemann (sphereR R0) 1 0 0 1 x = -1 := by
  unfold sphereR
  rw [scaled_christoffel _ _ _ (pow_ne_zero 2 hR), scaled_riemann _ _ _ (pow_ne_zero 2 hR),
    scaled_riemann _ _ _ (pow_ne_zero 2 hR)]
  exact ⟨KaluzaKlein.Geometry.sphere_christoffel_0_1_1 x h, KaluzaKlein.Geometry.sphere_riemann_0_1_0_1 x h,
    KaluzaKlein.Geometry.sphere_riemann_1_0_0_1 x h⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphereR_unchanged, dg-cc2-sphereR-unchanged-test]

/-! ## exr-sectional-hyperbolic-sage -/

/-- The Christoffel symbols of the hyperbolic plane `A² (dx² + dy²)/y²` (coordinates `(y, x)`): `Γ^x_xy = −1/y`,
`Γ^y_xx = 1/y`, `Γ^y_yy = −1/y`, and the three others vanish. -/
theorem hyperbolic_christoffel (A : ℝ) (hA : A ≠ 0) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    christoffel (hyperbolic A) 1 1 0 x = -1 / x 0 ∧ christoffel (hyperbolic A) 0 1 1 x = 1 / x 0 ∧
      christoffel (hyperbolic A) 0 0 0 x = -1 / x 0 ∧ christoffel (hyperbolic A) 0 0 1 x = 0 ∧
      christoffel (hyperbolic A) 1 0 0 x = 0 ∧ christoffel (hyperbolic A) 1 1 1 x = 0 := by
  unfold hyperbolic
  have hq : A ^ 2 / x 0 ^ 2 ≠ 0 := by positivity
  simp only [diag2_christoffel _ _ x hq hq, (hasDerivAt_hypG A (x 0) h).deriv]
  simp
  refine ⟨?_, ?_, ?_⟩ <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.hyperbolic_christoffel, dg-cc2-hyperbolic-christoffel-test]

/-- The hyperbolic plane has constant Gauss curvature `K = −1/A²`. -/
theorem hyperbolic_gauss (A : ℝ) (hA : A ≠ 0) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : gauss (hyperbolic A) x = -1 / A ^ 2 := by
  unfold hyperbolic
  have hq : A ^ 2 / x 0 ^ 2 ≠ 0 := by positivity
  rw [diag2_gauss hq hq (hasDerivAt_hypG A _ h) (hasDerivAt_hypG A _ h) (hasDerivAt_deriv_hypG A _ h)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.hyperbolic_gauss, dg-cc2-hyperbolic-gauss-test]

/-- The hyperbolic plane has `R_ij = K g_ij` with `K = −1/A²` and scalar curvature `R = 2K = −2/A²`. -/
theorem hyperbolic_ricci_scalar (A : ℝ) (hA : A ≠ 0) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    (∀ i j, ricci (hyperbolic A) i j x = -1 / A ^ 2 * hyperbolic A i j x) ∧
      scalarCurvature (hyperbolic A) x = -2 / A ^ 2 := by
  have hg := hyperbolic_gauss A hA x h
  unfold hyperbolic at hg ⊢
  have hq : A ^ 2 / x 0 ^ 2 ≠ 0 := by positivity
  refine ⟨fun i j => ?_, ?_⟩
  · rw [diag2_ricci hq hq (hasDerivAt_hypG A _ h) (hasDerivAt_hypG A _ h) (hasDerivAt_deriv_hypG A _ h), hg]
  · rw [diag2_scalarCurvature hq hq (hasDerivAt_hypG A _ h) (hasDerivAt_hypG A _ h) (hasDerivAt_deriv_hypG A _ h), hg]
    ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.hyperbolic_ricci_scalar, dg-cc2-hyperbolic-ricci-test]

/-- The three model geometries: for `A = 1` the hyperbolic plane has `K = −1`, the unit sphere `K = 1`, the plane
`K = 0`. -/
theorem model_geometries (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hs : Real.sin (x 0) ≠ 0) :
    gauss (hyperbolic 1) x = -1 ∧ gauss sphere x = 1 ∧ gauss polar x = 0 :=
  ⟨by rw [hyperbolic_gauss 1 one_ne_zero x h]; norm_num, sphere_gauss_aux x hs, polar_gauss x h⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.model_geometries, dg-cc2-model-geometries-test]

/-! ## exr-sectional-basis-independence -/

/-- With `X' = aX + bY`, `Y' = cX + dY`, the numerator `Q(X', Y') = (ad − bc)² Q(X, Y)`, the denominator is the Gram
determinant `det(M G Mᵀ) = (det M)² det G`, and `K(X', Y') = K(X, Y)` when `ad − bc ≠ 0` (for a surface
`E du² + G dv²`, whose lowered tensor has the symmetries). -/
theorem diag2_sectional_basis_independent {E G : ℝ → ℝ} {E₁ G₁ G₂ : ℝ} {x : Fin 2 → ℝ} (hE : E (x 0) ≠ 0)
    (hG : G (x 0) ≠ 0) (hE1 : HasDerivAt E E₁ (x 0)) (hG1 : HasDerivAt G G₁ (x 0)) (hG2 : HasDerivAt (deriv G) G₂ (x 0))
    (X Y : Fin 2 → ℝ) (a b c d : ℝ) (hdet : a * d - b * c ≠ 0) :
    sectional (diag2 E G) (fun i => a * X i + b * Y i) (fun i => c * X i + d * Y i) x = sectional (diag2 E G) X Y x :=
  sectional_basis_independent (diag2 E G) x (fun i j => by rw [diag2_symm])
    (fun ρ σ μ ν => by rw [diag2_lowered_antisymm_first hE hG hE1 hG1 hG2 ρ σ μ ν]; ring)
    (fun ρ σ μ ν => by rw [diag2_lowered_antisymm_last hE hG hE1 hG1 hG2 ρ σ μ ν]; ring) X Y a b c d hdet
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.diag2_sectional_basis_independent, dg-cc2-diag2-basis-test]

/-! ## def-gauss-curvature -/

/-- In two dimensions the sectional curvature of the (single) tangent plane is `R_1212 / det g`, for every surface
metric: the Gauss curvature. -/
theorem sectional_eq_gauss (g : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) (hg : g 0 1 x = g 1 0 x)
    (h1 : ∀ ρ σ μ ν, lowered g σ ρ μ ν x = -lowered g ρ σ μ ν x) (h2 : ∀ ρ σ μ ν, lowered g ρ σ ν μ x = -lowered g ρ σ μ ν x) :
    sectional g ![1, 0] ![0, 1] x = gauss g x := by
  unfold sectional gauss quadForm gramDet inner
  have z1 : lowered g 0 0 0 1 x = 0 := by have := h1 0 0 0 1; linarith
  have z2 : lowered g 1 1 0 1 x = 0 := by have := h1 1 1 0 1; linarith
  have z3 : lowered g 0 1 0 0 x = 0 := by have := h2 0 1 0 0; linarith
  have z4 : lowered g 0 1 1 1 x = 0 := by have := h2 0 1 1 1; linarith
  simp [Fin.sum_univ_two, metricAt, Matrix.det_fin_two, z1, z2, z3, z4, hg]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sectional_eq_gauss, dg-cc2-sectional-eq-gauss-test]

/-- Theorema egregium on a surface of revolution: the Gauss curvature `−f''/f` of `du² + f² dφ²` is determined by
the induced metric alone (it is `gauss` of the metric), for every twice differentiable profile `f ≠ 0`. -/
theorem revolution_gauss (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x : Fin 2 → ℝ) (f₂ : ℝ)
    (hf2 : HasDerivAt (deriv f) f₂ (x 0)) (hfx : f (x 0) ≠ 0) : gauss (revolution f) x = -f₂ / f (x 0) := by
  unfold revolution
  rw [diag2_gauss (by simp) (by simp [hfx]) (hasDerivAt_const _ _) (hasDerivAt_revG hf _) (hasDerivAt_deriv_revG hf hf2)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.revolution_gauss, dg-cc2-revolution-gauss-test]

/-! ## exm-gauss-sphere -/

/-- `K = 1` on the unit sphere, `K = 1/R0²` on the sphere of radius `R0`. -/
theorem sphere_gauss_values (R0 : ℝ) (hR : R0 ≠ 0) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    gauss sphere x = 1 ∧ gauss (sphereR R0) x = 1 / R0 ^ 2 := by
  refine ⟨sphere_gauss_aux x h, ?_⟩
  unfold sphereR
  rw [scaled_gauss _ _ _ (pow_ne_zero 2 hR), sphere_gauss_aux x h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_gauss_values, dg-cc2-sphere-gauss-values-test]

/-- The product of the principal curvatures: `(1/R0)(1/R0) = 1/R0²` for the sphere and `(1/R0) · 0 = 0` for the
cylinder. -/
theorem principal_products (R0 : ℝ) (hR : R0 ≠ 0) : 1 / R0 * (1 / R0) = 1 / R0 ^ 2 ∧ 1 / R0 * 0 = 0 :=
  ⟨by field_simp; ring, by ring⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.principal_products, dg-cc2-principal-test]

/-- The cylinder `dz² + R0² dφ²` is flat: `K = 0`. -/
theorem cylinder_gauss (R0 : ℝ) (hR : R0 ≠ 0) (x : Fin 2 → ℝ) : gauss (cylinder R0) x = 0 := by
  unfold cylinder
  rw [diag2_gauss (by simp) (by simp [hR]) (hasDerivAt_const _ _) (hasDerivAt_const _ _) (hasDerivAt_deriv_const _ _)]
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cylinder_gauss, dg-cc2-cylinder-gauss-test]

/-! ## exm-gauss-torus -/

/-- The Gauss curvature of the torus takes both signs: `K = c/(r(R + rc))` is positive for `c = cos a > 0` and
negative for `c < 0` (`0 < r < R`). -/
theorem torus_gauss_signs (R r c : ℝ) (hr : 0 < r) (hR : r < R) (hc : |c| ≤ 1) :
    (0 < c → 0 < c / (r * (R + r * c))) ∧ (c < 0 → c / (r * (R + r * c)) < 0) := by
  have h1 : 0 < R + r * c := by have := abs_le.mp hc; nlinarith
  exact ⟨fun h => by positivity, fun h => div_neg_of_neg_of_pos h (by positivity)⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_gauss_signs, dg-cc2-torus-gauss-signs-test]

/-- `K √(det g) = cos a` on the torus: the curvature density is independent of `R` and `r`. -/
theorem torus_K_dA (R r a : ℝ) (hr : 0 < r) (hR : 0 < R + r * Real.cos a) :
    Real.cos a / (r * (R + r * Real.cos a)) * Real.sqrt (r ^ 2 * (R + r * Real.cos a) ^ 2) = Real.cos a := by
  rw [show r ^ 2 * (R + r * Real.cos a) ^ 2 = (r * (R + r * Real.cos a)) ^ 2 by ring,
    Real.sqrt_sq (by positivity)]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_K_dA, dg-cc2-torus-kda-test]

/-- `∫₀^{2π} cos a da = 0`: the total curvature of the torus vanishes. -/
theorem integral_cos_period : ∫ a in (0 : ℝ)..(2 * Real.pi), Real.cos a = 0 := by
  VERIPHY[dg-cc2-integral]

/-! ## exm-gauss-holonomy -/

/-- The area of the cap `θ < θ₀` of the unit sphere, `∫₀^{θ₀} 2π sin θ dθ = 2π(1 − cos θ₀)`: the holonomy angle
`2π(1 − cos θ₀)` is the enclosed area times `K = 1`. -/
theorem cap_area (θ₀ : ℝ) : ∫ θ in (0 : ℝ)..θ₀, 2 * Real.pi * Real.sin θ = 2 * Real.pi * (1 - Real.cos θ₀) := by
  rw [intervalIntegral.integral_const_mul, integral_sin]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cap_area, dg-cc2-cap-area-test]

/-- `K A = A` for `K = 1`: the holonomy `2π(1 − cos θ₀)` equals `K` times the cap area. -/
theorem holonomy_eq_K_area (θ₀ : ℝ) : 1 * (2 * Real.pi * (1 - Real.cos θ₀)) = 2 * Real.pi * (1 - Real.cos θ₀) := by
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.holonomy_eq_K_area, dg-cc2-holonomy-area-test]

/-! ## exr-gauss-cylinder-sage -/

/-- The embedding of the cylinder of radius `R0`, `(R0 cos φ, R0 sin φ, z)` with `x = (z, φ)`. -/
noncomputable def cylinderEmbed (R0 : ℝ) : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => R0 * Real.cos (x 1), fun x => R0 * Real.sin (x 1), fun x => x 0]

/-- The embedding of the cone of half-angle `α`, `(ρ sin α cos φ, ρ sin α sin φ, ρ cos α)` with `x = (ρ, φ)`. -/
noncomputable def coneEmbed (α : ℝ) : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => x 0 * Real.sin α * Real.cos (x 1), fun x => x 0 * Real.sin α * Real.sin (x 1), fun x => x 0 * Real.cos α]

/-- The Euclidean metric pulled back to the cylinder is `dz² + R0² dφ²`. -/
theorem cylinder_gram (R0 : ℝ) (i j : Fin 2) (x : Fin 2 → ℝ) :
    KaluzaKlein.Geometry.gram (cylinderEmbed R0) i j x = cylinder R0 i j x := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [KaluzaKlein.Geometry.gram, pd, cylinderEmbed, cylinder, diag2, Fin.sum_univ_succ,
      Function.update_apply, deriv_const_mul, Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | ((try ring_nf); (try simp only [Real.cos_sq']); ring1)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cylinder_gram, dg-cc2-cylinder-gram-test]

/-- The Euclidean metric pulled back to the cone is `dρ² + ρ² sin²α dφ²`. -/
theorem cone_gram (α : ℝ) (i j : Fin 2) (x : Fin 2 → ℝ) :
    KaluzaKlein.Geometry.gram (coneEmbed α) i j x = cone α i j x := by
  fin_cases i <;> fin_cases j <;>
    simp (disch := fun_prop) [KaluzaKlein.Geometry.gram, pd, coneEmbed, cone, diag2, Fin.sum_univ_succ,
      Function.update_apply, deriv_const_mul, deriv_mul_const, Real.deriv_sin, Real.deriv_cos]
  all_goals first | ring1 | ((try ring_nf); (try simp only [Real.cos_sq']); ring1)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_gram, dg-cc2-cone-gram-test]

/-- The cylinder has vanishing Christoffel symbols (constant components). -/
theorem cylinder_christoffel (R0 : ℝ) (hR : R0 ≠ 0) (k i j : Fin 2) (x : Fin 2 → ℝ) :
    christoffel (cylinder R0) k i j x = 0 := by
  unfold cylinder
  rw [diag2_christoffel _ _ x (by simp) (by simp [hR])]
  fin_cases k <;> fin_cases i <;> fin_cases j <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cylinder_christoffel, dg-cc2-cylinder-christoffel-test]

/-- The cone has `K = 0`: flat, like the cylinder. -/
theorem cone_gauss (α : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hα : Real.sin α ≠ 0) : gauss (cone α) x = 0 := by
  unfold cone
  rw [diag2_gauss (by simp) (by simp [h, hα]) (hasDerivAt_const _ _) (hasDerivAt_coneG α _) (hasDerivAt_deriv_coneG α _)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_gauss, dg-cc2-cone-gauss-test]

/-! ## exr-gauss-negative-sage -/

/-- `d/du e^{−u} = −e^{−u}`. -/
theorem deriv_exp_neg (u : ℝ) : deriv (fun u => Real.exp (-u)) u = -Real.exp (-u) := by
  VERIPHY[dg-cc2-deriv]

/-- `d/du (−e^{−u}) = e^{−u}`. -/
theorem deriv_neg_exp_neg (u : ℝ) : deriv (fun u => -Real.exp (-u)) u = Real.exp (-u) := by
  VERIPHY[dg-cc2-deriv]

/-- The pseudosphere `f = e^{−u}` has `K = −f''/f = −1`. -/
theorem pseudosphere_gauss (x : Fin 2 → ℝ) : gauss (revolution fun u => Real.exp (-u)) x = -1 := by
  have hf : Differentiable ℝ fun u : ℝ => Real.exp (-u) := by fun_prop
  have hd : deriv (fun u : ℝ => Real.exp (-u)) = fun u => -Real.exp (-u) := funext deriv_exp_neg
  have hf2 : HasDerivAt (deriv fun u : ℝ => Real.exp (-u)) (Real.exp (-(x 0))) (x 0) := by
    rw [hd]
    have := ((hasDerivAt_neg (x 0)).exp).neg
    convert this using 1; ring
  rw [revolution_gauss _ hf x _ hf2 (Real.exp_ne_zero _)]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.pseudosphere_gauss, dg-cc2-pseudosphere-test]

/-- The surface `f = cosh u` has `K = −cosh u / cosh u = −1`. -/
theorem cosh_surface_gauss (x : Fin 2 → ℝ) : gauss (revolution Real.cosh) x = -1 := by
  have hf2 : HasDerivAt (deriv Real.cosh) (Real.cosh (x 0)) (x 0) := by
    rw [Real.deriv_cosh]; exact Real.hasDerivAt_sinh _
  rw [revolution_gauss _ Real.differentiable_cosh x _ hf2 (Real.cosh_pos _).ne']
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cosh_surface_gauss, dg-cc2-cosh-surface-test]

/-- The cone `f = u sin α` has `K = 0`. -/
theorem cone_profile_gauss (α : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (hα : Real.sin α ≠ 0) :
    gauss (revolution fun u => u * Real.sin α) x = 0 := by
  have hf : Differentiable ℝ fun u : ℝ => u * Real.sin α := by fun_prop
  have hf2 : HasDerivAt (deriv fun u : ℝ => u * Real.sin α) 0 (x 0) := by
    have hd : deriv (fun u : ℝ => u * Real.sin α) = fun _ => Real.sin α :=
      funext fun u => ((hasDerivAt_id u).mul_const (Real.sin α)).deriv.trans (by simp)
    rw [hd]; exact hasDerivAt_const _ _
  rw [revolution_gauss _ hf x _ hf2 (mul_ne_zero h hα)]
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.cone_profile_gauss, dg-cc2-cone-profile-test]

/-- The sphere `f = sin u` has `K = −(−sin u)/sin u = 1`. -/
theorem sin_profile_gauss (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : gauss (revolution Real.sin) x = 1 := by
  have hf2 : HasDerivAt (deriv Real.sin) (-Real.sin (x 0)) (x 0) := by
    rw [Real.deriv_sin]; exact Real.hasDerivAt_cos _
  rw [revolution_gauss _ Real.differentiable_sin x _ hf2 h]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sin_profile_gauss, dg-cc2-sin-profile-test]

/-! ## exr-gauss-orthogonal-formula -/

/-- The Christoffel symbols of `du² + f² dφ²`: `Γ^u_φφ = −f f'` and `Γ^φ_uφ = f'/f`. -/
theorem revolution_christoffel (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x : Fin 2 → ℝ) (hfx : f (x 0) ≠ 0) :
    christoffel (revolution f) 0 1 1 x = -(f (x 0) * deriv f (x 0)) ∧
      christoffel (revolution f) 1 0 1 x = deriv f (x 0) / f (x 0) := by
  unfold revolution
  simp only [diag2_christoffel _ _ x (by simp) (by simp [hfx]), (hasDerivAt_revG hf (x 0)).deriv]
  simp
  constructor <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.revolution_christoffel, dg-cc2-revolution-christoffel-test]

/-- `R^u_φuφ = ∂_u Γ^u_φφ − Γ^u_φφ Γ^φ_uφ = −(f'² + f f'') + f'² = −f f''`. -/
theorem revolution_riemann (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x : Fin 2 → ℝ) (f₂ : ℝ)
    (hf2 : HasDerivAt (deriv f) f₂ (x 0)) (hfx : f (x 0) ≠ 0) :
    riemann (revolution f) 0 1 0 1 x = -(f (x 0) * f₂) := by
  unfold revolution
  rw [diag2_riemann_0_1_0_1 (by simp) (by simp [hfx]) (hasDerivAt_const _ _) (hasDerivAt_revG hf _)
    (hasDerivAt_deriv_revG hf hf2)]
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.revolution_riemann, dg-cc2-revolution-riemann-test]

/-- `K = g_uu R^u_φuφ / f² = −f''/f`: the formula of the exercise as the quotient `1 · (−f f'') / f²`. -/
theorem revolution_K_formula (f f₂ : ℝ) (hf : f ≠ 0) : 1 * (-(f * f₂)) / f ^ 2 = -f₂ / f := by
  field_simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.revolution_K_formula, dg-cc2-revolution-k-test]

/-- The torus profile `f = R + r cos(u/r)`: `f' = −sin(u/r)`. -/
theorem deriv_torus_profile (R r u : ℝ) (hr : r ≠ 0) :
    deriv (fun u => R + r * Real.cos (u / r)) u = -Real.sin (u / r) := by
  VERIPHY[dg-cc2-deriv]

/-- The torus profile: `f'' = −cos(u/r)/r`. -/
theorem deriv2_torus_profile (r u : ℝ) (hr : r ≠ 0) : deriv (fun u => -Real.sin (u / r)) u = -Real.cos (u / r) / r := by
  VERIPHY[dg-cc2-deriv]

/-- With `a = u/r`, `K = −f''/f = cos a / (r (R + r cos a))` for the torus profile. -/
theorem torus_profile_K (R r a : ℝ) (hr : r ≠ 0) (hR : R + r * Real.cos a ≠ 0) :
    -(-Real.cos a / r) / (R + r * Real.cos a) = Real.cos a / (r * (R + r * Real.cos a)) := by
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.torus_profile_K, dg-cc2-torus-profile-k-test]

/-! ## prp-riemann-antisymm -/

/-- `R^ρ_σνμ = −R^ρ_σμν` for every metric, directly from the formula. -/
theorem riemann_antisymm_last {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) :
    riemann g ρ σ ν μ x = -riemann g ρ σ μ ν x :=
  KaluzaKlein.Geometry.riemann_antisymm g ρ σ μ ν x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.riemann_antisymm_last, dg-cc2-antisymm-test]

/-- `R^ρ_σμμ = 0` for every metric. -/
theorem riemann_diag_zero {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ : Fin n) (x : Fin n → ℝ) :
    riemann g ρ σ μ μ x = 0 :=
  KaluzaKlein.Geometry.riemann_diag g ρ σ μ x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.riemann_diag_zero, dg-cc2-diag-test]

/-! ## exm-antisymm-count -/

/-- Antisymmetry in `μν` leaves `n² · n(n − 1)/2` components: `4` on a surface, `27` in three dimensions (and `96` in
four); the remaining symmetries reduce these to `1` and `6`. -/
theorem antisymm_counts (n : Fin 5) :
    (n : ℕ) ^ 4 = ![0, 1, 16, 81, 256] n ∧ (n : ℕ) ^ 2 * ((n : ℕ) * (n - 1) / 2) = ![0, 0, 4, 27, 96] n ∧
      independentComponents n = ![0, 0, 1, 6, 20] n := by
  fin_cases n <;> decide
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.antisymm_counts, dg-cc2-antisymm-counts-test]

/-! ## exm-antisymm-sphere -/

/-- From `R^θ_φθφ = sin²θ` and `R^φ_θθφ = −1`: `R^θ_φφθ = −sin²θ`, `R^φ_θφθ = 1`, `R^ρ_σθθ = R^ρ_σφφ = 0`, and
`R_θθ = R^φ_θφθ = 1`. -/
theorem sphere_antisymm_consequences (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann sphere 0 1 1 0 x = -Real.sin (x 0) ^ 2 ∧ riemann sphere 1 0 1 0 x = 1 ∧
      (∀ ρ σ, riemann sphere ρ σ 0 0 x = 0 ∧ riemann sphere ρ σ 1 1 x = 0) ∧
      ricci sphere 0 0 x = riemann sphere 1 0 1 0 x ∧ ricci sphere 0 0 x = 1 := by
  refine ⟨?_, ?_, fun ρ σ => ⟨KaluzaKlein.Geometry.riemann_diag _ _ _ _ _, KaluzaKlein.Geometry.riemann_diag _ _ _ _ _⟩, ?_,
    KaluzaKlein.Geometry.sphere_ricci_0_0 x h⟩
  · rw [KaluzaKlein.Geometry.riemann_antisymm, KaluzaKlein.Geometry.sphere_riemann_0_1_0_1 x h]
  · rw [KaluzaKlein.Geometry.riemann_antisymm, KaluzaKlein.Geometry.sphere_riemann_1_0_0_1 x h]; norm_num
  · unfold ricci
    rw [Fin.sum_univ_two, KaluzaKlein.Geometry.riemann_diag]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_antisymm_consequences, dg-cc2-sphere-antisymm-test]

/-! ## exm-antisymm-commutator -/

/-- The commutator changes sign when its entries are exchanged: `[∇_ν, ∇_μ] X^ρ = −[∇_μ, ∇_ν] X^ρ`. -/
theorem covCommutator_antisymm {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ)
    (ρ μ ν : Fin n) (x : Fin n → ℝ) : covCommutator g X ρ ν μ x = -covCommutator g X ρ μ ν x := by
  unfold covCommutator; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.covCommutator_antisymm, dg-cc2-commutator-antisymm-test]

/-- The antisymmetry in `μν` holds for every affine connection, with or without torsion or metric: for a connection
with constant coefficients, `R^ρ_σνμ = −R^ρ_σμν`. -/
theorem riemannOfConst_antisymm {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (ρ σ μ ν : Fin n) :
    riemannOfConst Γ ρ σ ν μ = -riemannOfConst Γ ρ σ μ ν := by
  unfold riemannOfConst
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun l _ => by ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.riemannOfConst_antisymm, dg-cc2-const-antisymm-test]

/-! ## exr-riemann-antisymm-sage -/

/-- For every surface metric `E dx² + 2F dx dy + G dy²` with arbitrary (twice differentiable) component functions and
non-degenerate at `x`, the lowered tensor is antisymmetric in its first pair (metric compatibility), besides the
antisymmetry `R^ρ_σνμ = −R^ρ_σμν` and `R^ρ_σμμ = 0`, which hold for every metric (`riemann_antisymm_last`,
`riemann_diag_zero`). -/
def surface_lowered_antisymm_first : Prop :=
  ∀ (E F G : (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ), ContDiff ℝ 2 E → ContDiff ℝ 2 F → ContDiff ℝ 2 G →
    E x * G x - F x ^ 2 ≠ 0 → ∀ ρ σ μ ν, lowered (surface E F G) σ ρ μ ν x = -lowered (surface E F G) ρ σ μ ν x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.surface_lowered_antisymm_first, dg-cc2-surface-antisymm-test]

/-- `K = −f''/f` for `dx² + f(x)² dy²` (the formula of the surface of revolution). -/
theorem orthogonal_chart_gauss (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x : Fin 2 → ℝ) (f₂ : ℝ)
    (hf2 : HasDerivAt (deriv f) f₂ (x 0)) (hfx : f (x 0) ≠ 0) : gauss (revolution f) x = -f₂ / f (x 0) :=
  revolution_gauss f hf x f₂ hf2 hfx
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.orthogonal_chart_gauss, dg-cc2-orthogonal-gauss-test]

/-! ## exr-antisymm-proof -/

/-- Exchanging `μ` and `ν` exchanges the two terms of each difference in the Riemann formula, hence changes the sign;
for `μ = ν` the component equals its own negative and vanishes. -/
theorem riemann_antisymm_from_formula {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) :
    riemann g ρ σ ν μ x = -riemann g ρ σ μ ν x ∧ riemann g ρ σ μ μ x = 0 :=
  ⟨KaluzaKlein.Geometry.riemann_antisymm g ρ σ μ ν x, KaluzaKlein.Geometry.riemann_diag g ρ σ μ x⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.riemann_antisymm_from_formula, dg-cc2-antisymm-formula-test]

/-! ## exr-antisymm-torsion -/

/-- The covariant derivative of a vector field for a connection with constant coefficients `Γ` (not necessarily
symmetric), `∇_i X^k = ∂_i X^k + Γ^k_ij X^j`. -/
noncomputable def covDerivConst {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ) (i k : Fin n)
    (x : Fin n → ℝ) : ℝ :=
  pd i (X k) x + ∑ j, Γ k i j * X j x

/-- The second covariant derivative for a constant connection, `∇_μ ∇_ν X^ρ = ∂_μ(∇_ν X^ρ) + Γ^ρ_μλ ∇_ν X^λ −
Γ^λ_μν ∇_λ X^ρ`. -/
noncomputable def covDeriv2Const {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ) (μ ν ρ : Fin n)
    (x : Fin n → ℝ) : ℝ :=
  pd μ (fun y => covDerivConst Γ X ν ρ y) x + ∑ l, Γ ρ μ l * covDerivConst Γ X ν l x -
    ∑ l, Γ l μ ν * covDerivConst Γ X l ρ x

/-- The torsion `T^λ_μν = Γ^λ_μν − Γ^λ_νμ` of a connection. -/
def torsionOf {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (l μ ν : Fin n) : ℝ := Γ l μ ν - Γ l ν μ

/-- With torsion, `[∇_μ, ∇_ν] X^ρ = R^ρ_σμν X^σ − T^λ_μν ∇_λ X^ρ`, for every constant connection on `ℝ²` and every
twice differentiable vector field (the model of the general identity). -/
def commutator_with_torsion : Prop :=
  ∀ (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (X : Fin 2 → (Fin 2 → ℝ) → ℝ) (ρ μ ν : Fin 2) (x : Fin 2 → ℝ),
    (∀ k, ContDiff ℝ 2 (X k)) →
    covDeriv2Const Γ X μ ν ρ x - covDeriv2Const Γ X ν μ ρ x =
      ∑ σ, riemannOfConst Γ ρ σ μ ν * X σ x - ∑ l, torsionOf Γ l μ ν * covDerivConst Γ X l ρ x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.commutator_with_torsion, dg-cc2-torsion-commutator-test]

/-! ## prp-sphere-riemann -/

/-- The four components of the round sphere with `(μ, ν) = (θ, φ)`: `R^θ_φθφ = sin²θ`, `R^φ_θθφ = −1`,
`R^θ_θθφ = 0`, `R^φ_φθφ = 0`. -/
theorem sphere_riemann_components (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    riemann sphere 0 1 0 1 x = Real.sin (x 0) ^ 2 ∧ riemann sphere 1 0 0 1 x = -1 ∧ riemann sphere 0 0 0 1 x = 0 ∧
      riemann sphere 1 1 0 1 x = 0 :=
  ⟨KaluzaKlein.Geometry.sphere_riemann_0_1_0_1 x h, KaluzaKlein.Geometry.sphere_riemann_1_0_0_1 x h,
    KaluzaKlein.Geometry.sphere_riemann_0_0_0_1 x h, KaluzaKlein.Geometry.sphere_riemann_1_1_0_1 x h⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_riemann_components, dg-cc2-sphere-riemann-test]

/-! ## exm-sphere-riemann-radius -/

/-- On the sphere of radius `R0`: `R_θφθφ = R0² sin²θ`, `det g = R0⁴ sin²θ`, `K = 1/R0²`. -/
theorem sphereR_lowered_det_gauss (R0 : ℝ) (hR : R0 ≠ 0) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered (sphereR R0) 0 1 0 1 x = R0 ^ 2 * Real.sin (x 0) ^ 2 ∧
      (metricAt (sphereR R0) x).det = R0 ^ 4 * Real.sin (x 0) ^ 2 ∧ gauss (sphereR R0) x = 1 / R0 ^ 2 := by
  refine ⟨?_, ?_, (sphere_gauss_values R0 hR x h).2⟩
  · unfold sphereR
    rw [scaled_lowered _ _ _ (pow_ne_zero 2 hR), (sphere_lowered_components x h).1]
  · unfold sphereR
    rw [scaled_metricAt, Matrix.det_smul, Fintype.card_fin, KaluzaKlein.Geometry.sphere_det]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphereR_lowered_det_gauss, dg-cc2-sphereR-lowered-test]

/-! ## exm-sphere-riemann-constant -/

/-- The lowered tensor of the unit sphere is `g_ρμ g_σν − g_ρν g_σμ` (constant curvature `K = 1`). -/
theorem sphere_lowered_form (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) (ρ σ μ ν : Fin 2) :
    lowered sphere ρ σ μ ν x = sphere ρ μ x * sphere σ ν x - sphere ρ ν x * sphere σ μ x := by
  have hg := sphere_gauss_aux x h
  rw [sphere_eq_diag2] at hg ⊢
  rw [diag2_lowered_form (by simp) (by simp [h]) (hasDerivAt_const _ _) (hasDerivAt_sin_sq (x 0))
    (hasDerivAt_deriv_sin_sq (x 0)), hg, one_mul]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_lowered_form, dg-cc2-sphere-lowered-form-test]

/-- For `(θφθφ)` the right side is `1 · sin²θ − 0 = sin²θ`. -/
theorem sphere_form_check (x : Fin 2 → ℝ) : sphere 0 0 x * sphere 1 1 x - sphere 0 1 x * sphere 1 0 x = Real.sin (x 0) ^ 2 := by
  simp [sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_form_check, dg-cc2-sphere-form-check-test]

/-- An array of the constant-curvature form `T = K (g_ρμ g_σν − g_ρν g_σμ)` has every sectional curvature equal to
`K`: its quadratic form is `K` times the Gram determinant. -/
theorem quadForm_const_curv {n : ℕ} (g : Fin n → Fin n → ℝ) (K : ℝ) (T : Fin n → Fin n → Fin n → Fin n → ℝ)
    (hT : ∀ ρ σ μ ν, T ρ σ μ ν = K * (g ρ μ * g σ ν - g ρ ν * g σ μ)) (X Y : Fin n → ℝ) :
    quadForm T X Y = K * ((∑ i, ∑ j, g i j * X i * X j) * (∑ i, ∑ j, g i j * Y i * Y j) -
      (∑ i, ∑ j, g i j * X i * Y j) ^ 2) := by
  unfold quadForm
  simp only [hT]
  rw [pow_two, Finset.sum_mul_sum, Finset.sum_mul_sum]
  simp only [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun ρ _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun σ _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun μ _ => ?_
  refine Finset.sum_congr rfl fun ν _ => ?_
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.quadForm_const_curv, dg-cc2-quadform-const-test]

/-- The unit `S³ = SU(2)` is the case `K = 1`, `n = 3`: its lowered tensor is `g_ρμ g_σν − g_ρν g_σμ`. -/
def su2_lowered_form : Prop :=
  ∀ (ρ σ μ ν : Fin 3) (x : Fin 3 → ℝ), Real.sin (x 0) ≠ 0 →
    lowered su2 ρ σ μ ν x = su2 ρ μ x * su2 σ ν x - su2 ρ ν x * su2 σ μ x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.su2_lowered_form, dg-cc2-su2-lowered-form-test]

/-! ## exm-sphere-riemann-lean -/

/-- The symbols that make `R^θ_θθφ` and `R^φ_φθφ` vanish term by term: `Γ^θ_θθ = Γ^θ_θφ = 0`, `Γ^θ_θθ = Γ^φ_θθ = 0`
(both `λ`), `Γ^φ_φφ = 0`. -/
theorem sphere_vanishing_symbols (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 0 0 0 x = 0 ∧ christoffel sphere 0 0 1 x = 0 ∧ christoffel sphere 1 0 0 x = 0 ∧
      christoffel sphere 1 1 1 x = 0 :=
  ⟨KaluzaKlein.Geometry.sphere_christoffel_0_0_0 x h, KaluzaKlein.Geometry.sphere_christoffel_0_0_1 x h,
    KaluzaKlein.Geometry.sphere_christoffel_1_0_0 x h, KaluzaKlein.Geometry.sphere_christoffel_1_1_1 x h⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_vanishing_symbols, dg-cc2-sphere-symbols-test]

/-- `∂_φ cot θ = 0`: a function of `θ` alone has no `φ`-derivative. -/
theorem pd_phi_cot (x : Fin 2 → ℝ) : pd 1 (fun y => Real.cos (y 0) / Real.sin (y 0)) x = 0 :=
  pd_other_comp (fun t => Real.cos t / Real.sin t) 1 (by decide) x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.pd_phi_cot, dg-cc2-pd-phi-cot-test]

/-- The two zeros are forced by symmetry: `R_θθθφ = 0` and `R_φφθφ = sin²θ R^φ_φθφ = 0` by antisymmetry in the first
pair. -/
theorem sphere_zero_by_symmetry (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered sphere 0 0 0 1 x = 0 ∧ lowered sphere 1 1 0 1 x = 0 ∧
      lowered sphere 1 1 0 1 x = Real.sin (x 0) ^ 2 * riemann sphere 1 1 0 1 x := by
  refine ⟨?_, ?_, ?_⟩
  · rw [sphere_lowered_form x h]; ring
  · rw [sphere_lowered_form x h]; ring
  · unfold lowered; rw [Fin.sum_univ_two]; simp [sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_zero_by_symmetry, dg-cc2-sphere-zero-symmetry-test]

/-! ## exr-sphere-riemann-sage -/

/-- All sixteen components of the unit sphere: `R^θ_φθφ = sin²θ`, `R^θ_φφθ = −sin²θ`, `R^φ_θθφ = −1`, `R^φ_θφθ = 1`,
and the twelve others vanish. -/
theorem sphere_riemann_all (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) (ρ σ μ ν : Fin 2) :
    riemann sphere ρ σ μ ν x =
      (![![![![0, 0], ![0, 0]], ![![0, Real.sin (x 0) ^ 2], ![-Real.sin (x 0) ^ 2, 0]]],
         ![![![0, -1], ![1, 0]], ![![0, 0], ![0, 0]]]] : Fin 2 → Fin 2 → Fin 2 → Fin 2 → ℝ) ρ σ μ ν := by
  have hg := sphere_gauss_aux x h
  rw [sphere_eq_diag2] at hg ⊢
  rw [diag2_riemann_all (by simp) (by simp [h]) (hasDerivAt_const _ _) (hasDerivAt_sin_sq (x 0))
    (hasDerivAt_deriv_sin_sq (x 0)), hg]
  fin_cases ρ <;> fin_cases σ <;> fin_cases μ <;> fin_cases ν <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_riemann_all, dg-cc2-sphere-riemann-all-test]

/-- `R_θφθφ = sin²θ = det g` on the unit sphere, so `K = 1`. -/
theorem sphere_lowered_eq_det (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered sphere 0 1 0 1 x = (metricAt sphere x).det ∧ gauss sphere x = 1 :=
  ⟨by rw [(sphere_lowered_components x h).1, KaluzaKlein.Geometry.sphere_det], sphere_gauss_aux x h⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_lowered_eq_det, dg-cc2-sphere-lowered-det-test]

/-! ## exr-sphere-riemann-stereographic-sage -/

/-- `q = 1 + u² + v² ≥ 1`: the stereographic metric is regular everywhere. -/
theorem stereo_q_pos (x : Fin 2 → ℝ) : 0 < 1 + x 0 ^ 2 + x 1 ^ 2 := by positivity

theorem stereo_metricAt (x : Fin 2 → ℝ) :
    metricAt stereo x = !![4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2, 0; 0, 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, stereo]

theorem stereo_invMetric (x : Fin 2 → ℝ) :
    invMetric stereo x = !![(1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 / 4, 0; 0, (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 / 4] := by
  have hq := (stereo_q_pos x).ne'
  unfold invMetric
  refine Matrix.inv_eq_right_inv ?_
  rw [stereo_metricAt]
  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp

/-- The Christoffel symbols of the stereographic metric, with `q = 1 + u² + v²`: `Γ^u_uu = −2u/q`, `Γ^u_uv = −2v/q`,
`Γ^u_vv = 2u/q`, `Γ^v_uu = 2v/q`, `Γ^v_uv = −2u/q`, `Γ^v_vv = −2v/q`. -/
theorem stereo_christoffel (k i j : Fin 2) (y : Fin 2 → ℝ) :
    christoffel stereo k i j y =
      (![![![fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2)],
            ![fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => 2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2)]],
          ![![fun y => 2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2)],
            ![fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2)]]] :
        Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) k i j y := by
  have hq := (stereo_q_pos y).ne'
  rw [christoffel, stereo_invMetric]
  fin_cases k <;> fin_cases i <;> fin_cases j <;>
    simp (disch := first | positivity | fun_prop (disch := positivity)) [stereo, Fin.sum_univ_succ, pd,
      Function.update_apply, deriv_fun_div, deriv_const, deriv_fun_pow, deriv_fun_add] <;>
    field_simp <;> ring

/-- The Christoffel symbols as functions (the closed forms hold everywhere). -/
theorem stereo_christoffel_fun (k i j : Fin 2) :
    christoffel stereo k i j =
      (![![![fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2)],
            ![fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => 2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2)]],
          ![![fun y => 2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2)],
            ![fun y => -2 * y 0 / (1 + y 0 ^ 2 + y 1 ^ 2), fun y => -2 * y 1 / (1 + y 0 ^ 2 + y 1 ^ 2)]]] :
        Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) k i j :=
  funext fun y => stereo_christoffel k i j y

/-- The four Riemann components of the stereographic metric with `(μ, ν) = (u, v)`: `R^u_vuv = 4/q²`,
`R^v_uuv = −4/q²`, `R^u_uuv = 0`, `R^v_vuv = 0`. -/
theorem stereo_riemann (x : Fin 2 → ℝ) :
    riemann stereo 0 1 0 1 x = 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 ∧ riemann stereo 1 0 0 1 x = -4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 ∧
      riemann stereo 0 0 0 1 x = 0 ∧ riemann stereo 1 1 0 1 x = 0 := by
  have hq := (stereo_q_pos x).ne'
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
  · unfold riemann
    simp only [stereo_christoffel_fun, stereo_christoffel]
    simp (disch := first | positivity | fun_prop (disch := positivity)) [Fin.sum_univ_succ, pd, Function.update_apply,
      deriv_fun_div, deriv_const, deriv_fun_pow, deriv_fun_add, deriv_const_mul]
    field_simp
    ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.stereo_riemann, dg-cc2-stereo-riemann-test]

/-- `R_uvuv = 16/q⁴ = det g`, so `K = 1` everywhere on the stereographic chart, the pole included, where both are
`16`. -/
theorem stereo_lowered_gauss (x : Fin 2 → ℝ) :
    lowered stereo 0 1 0 1 x = 16 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 4 ∧ (metricAt stereo x).det = 16 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 4 ∧
      gauss stereo x = 1 ∧ lowered stereo 0 1 0 1 ![0, 0] = 16 ∧ (metricAt stereo ![0, 0]).det = 16 := by
  have hq := (stereo_q_pos x).ne'
  have hl : ∀ y : Fin 2 → ℝ, lowered stereo 0 1 0 1 y = 16 / (1 + y 0 ^ 2 + y 1 ^ 2) ^ 4 := by
    intro y
    have hqy := (stereo_q_pos y).ne'
    unfold lowered
    rw [Fin.sum_univ_two, (stereo_riemann y).1, (stereo_riemann y).2.2.2]
    simp [stereo]
    field_simp; ring
  have hd : ∀ y : Fin 2 → ℝ, (metricAt stereo y).det = 16 / (1 + y 0 ^ 2 + y 1 ^ 2) ^ 4 := by
    intro y
    have hqy := (stereo_q_pos y).ne'
    rw [stereo_metricAt, Matrix.det_fin_two_of]
    field_simp; ring
  refine ⟨hl x, hd x, ?_, ?_, ?_⟩
  · unfold gauss; rw [hl, hd]; field_simp
  · rw [hl]; simp
  · rw [hd]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.stereo_lowered_gauss, dg-cc2-stereo-gauss-test]

/-! ## exr-sphere-riemann-form -/

/-- The Ricci contraction of the constant-curvature form is `(n − 1) K g_σν`, and its trace `n(n − 1) K`: `2` for the
unit `S²` and `6` for the unit `S³`. -/
theorem const_curv_ricci_scalar {n : ℕ} (g ginv : Matrix (Fin n) (Fin n) ℝ) (h : ginv * g = 1) (K : ℝ)
    (T : Fin n → Fin n → Fin n → Fin n → ℝ)
    (hT : ∀ ρ σ μ ν, T ρ σ μ ν = K * ((if ρ = μ then 1 else 0) * g σ ν - (if ρ = ν then 1 else 0) * g σ μ)) :
    (∀ σ ν, ricciOf T σ ν = ((n : ℝ) - 1) * K * g σ ν) ∧
      traceOf ginv (fun σ ν => ricciOf T σ ν) = (n : ℝ) * ((n : ℝ) - 1) * K := by
  refine ⟨ricciOf_const_curv g K T hT, ?_⟩
  simp only [ricciOf_const_curv g K T hT]
  exact traceOf_const_curv g ginv h K
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.const_curv_ricci_scalar, dg-cc2-const-curv-ricci-test]

/-- `n(n − 1) K` for `K = 1` is `2` in two and `6` in three dimensions. -/
theorem n_n_minus_one (n : Fin 5) : (n : ℕ) * (n - 1) = ![0, 0, 2, 6, 12] n := by
  fin_cases n <;> decide
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.n_n_minus_one, dg-cc2-n-n-minus-one-test]

/-- Both sides of the constant-curvature form are antisymmetric in each pair on the sphere, so only `(θφθφ)` and its
images need checking: the right side is `g_θθ g_φφ = sin²θ`, matching `R_θφθφ`. -/
theorem sphere_form_single_check (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lowered sphere 0 1 0 1 x = sphere 0 0 x * sphere 1 1 x ∧ sphere 0 0 x * sphere 1 1 x = Real.sin (x 0) ^ 2 := by
  refine ⟨?_, by simp [sphere]⟩
  rw [(sphere_lowered_components x h).1]; simp [sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.Riemann.sphere_form_single_check, dg-cc2-sphere-form-single-test]

end KaluzaKlein.DifferentialGeometry.Riemann

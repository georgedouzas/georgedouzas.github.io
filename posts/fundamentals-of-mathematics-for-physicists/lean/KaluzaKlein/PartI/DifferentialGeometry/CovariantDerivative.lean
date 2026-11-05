import Mathlib
import KaluzaKlein.Geometry.Polar
import KaluzaKlein.Geometry.Christoffel
import KaluzaKlein.Geometry.Curvature
import KaluzaKlein.Geometry.VectorFields

/-!
# Differential geometry, section "The covariant derivative"

The claims of the section "The covariant derivative" of `tutorial/part-i-09-differential-geometry.qmd`: connections
and their transformation law, covariant derivatives of vector fields, one-forms and tensors, torsion and metric
compatibility, and the covariantly constant fields of the plane.

Modelling (design D28). Geometry is in coordinates: a metric is a component family on `Fin n → ℝ` (the
`CoordinateGeometry` asset), a vector field a family `Fin n → (Fin n → ℝ) → ℝ`, and the covariant derivative
`∇_i X^k = ∂_i X^k + Γ^k_{ij} X^j` is `KaluzaKlein.Geometry.covDeriv`, with the asset's Christoffel symbols. A claim
that holds for every connection, every chart or every tensor (the transformation law of a connection, "the covariant
derivative is a tensor", the Leibniz rule, torsion is a tensor, inner products of parallel fields are constant) is
stated in dimension two in jet variables: the components `Γ`, the Jacobians `J = ∂x'/∂x` and `K = ∂x/∂x'`, the second
derivatives `H = ∂²x/∂x'∂x'` and `M = ∂²x'/∂x∂x`, the components of the fields and their partial derivatives are real
variables, the chain rule and `K J = 1` are hypotheses, and the claim is the polynomial identity between them, closed
by a certificate (`ideal_membership`). Explicit fields on the polar plane and the sphere are checked component by
component (`coordinate_geometry`), and derivatives of explicit functions by the `derivative` recipe.

Shared definitions of the chapter part (one-forms, Hessians, first-kind symbols, torsion and curvature of constant
connections, derivatives along curves) live here, since the later modules import this one.
-/

open CoordinateGeometry KaluzaKlein.Geometry

namespace KaluzaKlein.DifferentialGeometry.CovariantDerivative

/-! ## Shared definitions -/

/-- The Euclidean metric of the plane, `δ_{ij}`, as a constant component family. -/
noncomputable def euclid : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ := fun i j _ => (1 : Matrix (Fin 2) (Fin 2) ℝ) i j

/-- The covariant derivative of a one-form, `∇_i ω_j = ∂_i ω_j − Γ^k_{ij} ω_k`. -/
noncomputable def covDerivForm {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ω : Fin n → (Fin n → ℝ) → ℝ)
    (i j : Fin n) (x : Fin n → ℝ) : ℝ :=
  pd i (ω j) x - ∑ k, christoffel g k i j x * ω k x

/-- The Hessian of a function, `∇_i ∇_j f = ∂_i ∂_j f − Γ^k_{ij} ∂_k f`. -/
noncomputable def hessian {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (f : (Fin n → ℝ) → ℝ) (i j : Fin n)
    (x : Fin n → ℝ) : ℝ :=
  pd i (fun y => pd j f y) x - ∑ k, christoffel g k i j x * pd k f x

/-- The Christoffel symbols of the first kind, `Γ_{kij} = g_{km} Γ^m_{ij}`. -/
noncomputable def firstKind {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (k i j : Fin n) (x : Fin n → ℝ) : ℝ :=
  ∑ m, g k m x * christoffel g m i j x

/-- The torsion `T^k_{ij} = Γ^k_{ij} − Γ^k_{ji}` of a connection with constant coefficients. -/
def torsionOf {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (k i j : Fin n) : ℝ := Γ k i j - Γ k j i

/-- The Riemann tensor `R^ρ_{σμν} = Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}` of a connection with constant
coefficients (the derivative terms vanish). -/
def riemannConst {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (ρ σ μ ν : Fin n) : ℝ :=
  ∑ l, (Γ ρ μ l * Γ l ν σ - Γ ρ ν l * Γ l μ σ)

/-- The covariant derivative `DV^k/dt = V̇^k + Γ^k_{ij} ẋ^i V^j` of a field `V` along a curve `c`. -/
noncomputable def transportDeriv {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (c V : ℝ → Fin n → ℝ) (k : Fin n)
    (t : ℝ) : ℝ :=
  deriv (fun s => V s k) t + ∑ i, ∑ j, christoffel g k i j (c t) * deriv (fun s => c s i) t * V t j

/-- The geodesic residual `ẍ^k + Γ^k_{ij} ẋ^i ẋ^j` of a curve `c`. -/
noncomputable def geodesicResidual {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (c : ℝ → Fin n → ℝ) (k : Fin n)
    (t : ℝ) : ℝ :=
  deriv (fun s => deriv (fun u => c u k) s) t +
    ∑ i, ∑ j, christoffel g k i j (c t) * deriv (fun s => c s i) t * deriv (fun s => c s j) t

/-- The transformation law of a connection in dimension two (jet variables): `Γ'^k_{ij} = J^k_c K^a_i K^b_j Γ^c_{ab}
+ J^k_c H^c_{ij}` with `J = ∂x'/∂x`, `K = ∂x/∂x'`, `H = ∂²x/∂x'∂x'`. -/
def connLaw (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ) (H : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (k i j : Fin 2) : ℝ :=
  ∑ c, ∑ a, ∑ b, J k c * K a i * K b j * Γ c a b + ∑ c, J k c * H c i j

/-- The transformation law of a `(1, 2)` tensor in dimension two: `D'^k_{ij} = J^k_c K^a_i K^b_j D^c_{ab}`. -/
def tensorLaw (D : Fin 2 → Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ) (k i j : Fin 2) : ℝ :=
  ∑ c, ∑ a, ∑ b, J k c * K a i * K b j * D c a b

/-- The Kronecker delta of dimension two as an explicit array. -/
def δ2 : Fin 2 → Fin 2 → ℝ := ![![1, 0], ![0, 1]]

/-- The field `∂_y` in polar components, `(sin φ, cos φ / r)`. -/
noncomputable def dyPolar : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => Real.sin (x 1), fun x => Real.cos (x 1) / x 0]

/-- The constant field `a ∂_x + b ∂_y` in polar components. -/
noncomputable def constField (a b : ℝ) : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => a * Real.cos (x 1) + b * Real.sin (x 1), fun x => (-(a * Real.sin (x 1)) + b * Real.cos (x 1)) / x 0]

/-- The radial field `r ∂_r = x ∂_x + y ∂_y` in polar components, `(r, 0)`. -/
noncomputable def radialField : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => x 0, fun _ => 0]

/-- The one-form `dφ` on the sphere (components `(0, 1)`), lowered: `X_j = g_{jk} X^k = (0, sin²θ)` for `X = ∂_φ`. -/
noncomputable def dphiLowered : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun x => Real.sin (x 0) ^ 2]

/-- The one-form `dx = cos φ dr − r sin φ dφ` in polar components. -/
noncomputable def dxForm : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => Real.cos (x 1), fun x => -(x 0 * Real.sin (x 1))]

/-- The connection with the single non-zero coefficient `Γ^1_{12} = 1` on `ℝ²` (indices `0, 1`). -/
noncomputable def Γex : Fin 2 → Fin 2 → Fin 2 → ℝ := ![![![0, 1], ![0, 0]], ![![0, 0], ![0, 0]]]

/-- Its symmetric part, `Γ^1_{12} = Γ^1_{21} = 1/2`. -/
noncomputable def Γsym : Fin 2 → Fin 2 → Fin 2 → ℝ := ![![![0, 1 / 2], ![1 / 2, 0]], ![![0, 0], ![0, 0]]]

/-! ## Helper lemmas: the Christoffel symbols with the lower indices reversed -/

theorem polar_symm (i j : Fin 2) : polar i j = polar j i := by
  ext y; fin_cases i <;> fin_cases j <;> rfl

theorem polar_christoffel_0_1_0 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 0 1 0 x = 0 := by
  rw [KaluzaKlein.christoffel_symm polar polar_symm]; exact polar_christoffel_0_0_1 x h

theorem polar_christoffel_1_1_0 (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : christoffel polar 1 1 0 x = 1 / x 0 := by
  rw [KaluzaKlein.christoffel_symm polar polar_symm]; exact polar_christoffel_1_0_1 x h

theorem sphere_christoffel_0_1_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) : christoffel sphere 0 1 0 x = 0 := by
  rw [KaluzaKlein.christoffel_symm sphere sphere_symm]; exact sphere_christoffel_0_0_1 x h

theorem sphere_christoffel_1_1_0 (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    christoffel sphere 1 1 0 x = Real.cos (x 0) / Real.sin (x 0) := by
  rw [KaluzaKlein.christoffel_symm sphere sphere_symm]; exact sphere_christoffel_1_0_1 x h

/-! ## def-connection, exm-connection-trivial, exr-connection-transformation -/

/-- The trivial connection `Γ = 0` of Cartesian coordinates acquires `Γ^r_{φφ} = −r` in polar coordinates from the
inhomogeneous term alone: `(∂r/∂x^c) ∂²x^c/∂φ² = (x/r)(−x) + (y/r)(−y) = −r` for `x = r cos φ`, `y = r sin φ`. -/
theorem inhomogeneous_polar (r φ : ℝ) (hr : r ≠ 0) :
    r * Real.cos φ / r * deriv (fun s => deriv (fun u => r * Real.cos u) s) φ +
      r * Real.sin φ / r * deriv (fun s => deriv (fun u => r * Real.sin u) s) φ = -r := by
  VERIPHY[dg-cc1-partials]

/-- `∂²_φ (x, y) = −(x, y)`: the second `φ`-derivatives of `r cos φ`, `r sin φ`. -/
theorem second_deriv_polar (r φ : ℝ) :
    deriv (fun s => deriv (fun u => r * Real.cos u) s) φ = -(r * Real.cos φ) ∧
      deriv (fun s => deriv (fun u => r * Real.sin u) s) φ = -(r * Real.sin φ) := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-partials]
  · VERIPHY[dg-cc1-partials]

/-- `∂r/∂x = x / r` for `r = √(x² + y²)`. -/
theorem grad_radius (x y : ℝ) (h : 0 < x ^ 2 + y ^ 2) :
    deriv (fun s => Real.sqrt (s ^ 2 + y ^ 2)) x = x / Real.sqrt (x ^ 2 + y ^ 2) := by
  VERIPHY[dg-cc1-partials]

/-- `∂_r (r cos φ, r sin φ) = (cos φ, sin φ)`, the first step of the mixed derivative `∂_r ∂_φ (x, y)`. -/
theorem dr_polar_embedding (r φ : ℝ) :
    deriv (fun u => u * Real.cos φ) r = Real.cos φ ∧ deriv (fun u => u * Real.sin φ) r = Real.sin φ := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-partials]
  · VERIPHY[dg-cc1-partials]

/-- The inhomogeneous term gives `Γ^φ_{rφ} = 1/r` as well: `(∂φ/∂x^c) ∂_r ∂_φ x^c = (−y/r²)(−sin φ) + (x/r²) cos φ
= 1/r`. -/
theorem inhomogeneous_polar_rphi (r φ : ℝ) (hr : r ≠ 0) :
    -(r * Real.sin φ) / r ^ 2 * deriv (fun s => Real.cos s) φ +
      r * Real.cos φ / r ^ 2 * deriv (fun s => Real.sin s) φ = 1 / r := by
  VERIPHY[dg-cc1-partials]

/-- The constant field `∂_x`, polar components `(cos φ, −sin φ / r)`, is covariantly constant for the polar
connection: every component `∇_i X^k` vanishes (`r ≠ 0`). -/
theorem dx_parallel (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : ∀ i k, covDeriv polar dxPolar i k x = 0 := by
  VERIPHY[dg-cc1-geo]

/-! ## exm-connection-gauge -/

/-- The gauge covariant derivative `D_μ φ = ∂_μ φ − i q A_μ φ` transforms as `φ` does: with `φ' = E φ`
(`E = e^{iqλ}`), `∂_μ φ' = E (∂_μ φ + i q ∂_μ λ φ)` and `A'_μ = A_μ + ∂_μ λ`, one has `D'_μ φ' = E D_μ φ`. -/
theorem gauge_covariant (E φ dφ A dλ q : ℂ) :
    E * (dφ + Complex.I * q * dλ * φ) - Complex.I * q * (A + dλ) * (E * φ) = E * (dφ - Complex.I * q * A * φ) := by
  VERIPHY[dg-cc1-poly]

/-! ## exr-connection-difference -/

/-- The difference of two connections transforms homogeneously, as a `(1, 2)` tensor: the inhomogeneous terms
cancel (dimension two, jet variables). -/
theorem diff_tensor (Γ Γt : Fin 2 → Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ)
    (H : Fin 2 → Fin 2 → Fin 2 → ℝ) (k i j : Fin 2) :
    connLaw Γ J K H k i j - connLaw Γt J K H k i j = tensorLaw (fun c a b => Γ c a b - Γt c a b) J K k i j := by
  simp only [connLaw, tensorLaw, Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- A connection plus a `(1, 2)` tensor field is a connection: the sum transforms with the same inhomogeneous
term, so the space of connections is an affine space over the `(1, 2)` tensor fields. -/
theorem conn_add_tensor (Γ D : Fin 2 → Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ)
    (H : Fin 2 → Fin 2 → Fin 2 → ℝ) (k i j : Fin 2) :
    connLaw (fun c a b => Γ c a b + D c a b) J K H k i j = connLaw Γ J K H k i j + tensorLaw D J K k i j := by
  simp only [connLaw, tensorLaw, Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- The difference tensor `D^r_{φφ} = −r`, `D^φ_{rφ} = D^φ_{φr} = 1/r` between the Levi-Civita and the zero
connection of the polar chart is not zero in Cartesian components either: `D^x_{xx} = J^x_c K^a_x K^b_x D^c_{ab}
= sin²φ cos φ / r`. -/
theorem diff_cartesian_xx (r φ : ℝ) (hr : r ≠ 0) :
    Real.cos φ * (-(Real.sin φ) / r) * (-(Real.sin φ) / r) * (-r) +
        -(r * Real.sin φ) * (Real.cos φ * (-(Real.sin φ) / r) + -(Real.sin φ) / r * Real.cos φ) * (1 / r) =
      Real.sin φ ^ 2 * Real.cos φ / r := by
  VERIPHY[dg-cc1-poly]

/-! ## def-covariant-derivative-vector, exr-covariant-tensorial -/

/-- The covariant derivative of a vector field is a `(1, 1)` tensor (dimension two, jet variables): with
`J = ∂x'/∂x`, `K = ∂x/∂x'`, `K J = 1`, `M = ∂²x'/∂x∂x`, `H = ∂²x/∂x'∂x'` and the identity `J^k_c H^c_{jm} =
−K^c_j K^d_m M^k_{cd}`, the primed derivative `∂'_j X'^k + Γ'^k_{jm} X'^m` (with `∂'_j X'^k` from the
differentiated vector law and `Γ'` from the connection law) equals `K^i_j J^k_l (∂_i X^l + Γ^l_{ia} X^a)`. -/
theorem covDeriv_tensor (X : Fin 2 → ℝ) (dX : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (J K : Fin 2 → Fin 2 → ℝ) (M H : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (hKJ00 : K 0 0 * J 0 0 + K 0 1 * J 1 0 = 1) (hKJ01 : K 0 0 * J 0 1 + K 0 1 * J 1 1 = 0)
    (hKJ10 : K 1 0 * J 0 0 + K 1 1 * J 1 0 = 0) (hKJ11 : K 1 0 * J 0 1 + K 1 1 * J 1 1 = 1)
    (hH : ∀ k j m, ∑ c, J k c * H c j m = -∑ c, ∑ d, K c j * K d m * M k c d) :
    ∀ j k : Fin 2,
      (∑ i, ∑ l, K i j * J k l * dX i l + ∑ i, ∑ l, K i j * M k i l * X l) +
          ∑ m, connLaw Γ J K H k j m * (∑ l, J m l * X l) =
        ∑ i, ∑ l, K i j * J k l * (dX i l + ∑ a, Γ l i a * X a) := by
  have h00 := hH 0 0 0
  have h01 := hH 0 0 1
  have h10 := hH 0 1 0
  have h11 := hH 0 1 1
  have h20 := hH 1 0 0
  have h21 := hH 1 0 1
  have h30 := hH 1 1 0
  have h31 := hH 1 1 1
  simp only [Fin.sum_univ_two] at h00 h01 h10 h11 h20 h21 h30 h31
  simp only [Fin.forall_fin_two, connLaw, Fin.sum_univ_two]
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]

/-- The identity `J^k_c H^c_{jm} = −K^c_j K^d_m M^k_{cd}` follows from differentiating `J^k_c K^c_j = δ^k_j` along
`x'^m` (chain rule: `∂'_m J^k_c = K^d_m M^k_{cd}`). -/
theorem second_derivative_identity (J K : Fin 2 → Fin 2 → ℝ) (M H : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (hD : ∀ k j m, ∑ c, (∑ d, K d m * M k c d) * K c j + ∑ c, J k c * H c j m = 0) (k j m : Fin 2) :
    ∑ c, J k c * H c j m = -∑ c, ∑ d, K c j * K d m * M k c d := by
  have h := hD k j m
  simp only [Fin.sum_univ_two] at h ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-- The covariant derivative along a vector, `∇_Y X = Y^i ∇_i X`, is a vector field: contracting the tensor law of
`∇X` with the vector law of `Y` and using `K J = 1` gives the vector law. -/
theorem contraction_tensor (Y : Fin 2 → ℝ) (N : Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ)
    (hKJ00 : K 0 0 * J 0 0 + K 0 1 * J 1 0 = 1) (hKJ01 : K 0 0 * J 0 1 + K 0 1 * J 1 1 = 0)
    (hKJ10 : K 1 0 * J 0 0 + K 1 1 * J 1 0 = 0) (hKJ11 : K 1 0 * J 0 1 + K 1 1 * J 1 1 = 1) (k : Fin 2) :
    ∑ j, (∑ i, J j i * Y i) * (∑ i, ∑ l, K i j * J k l * N i l) = ∑ l, J k l * (∑ i, Y i * N i l) := by
  simp only [Fin.sum_univ_two]
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-! ## exm-covariant-constant, exr-covariant-polar -/

/-- `∂_φ X^r = −sin φ` is cancelled by `Γ^r_{φφ} X^φ = (−r)(−sin φ / r)`. -/
theorem dx_cancellation (r φ : ℝ) (hr : r ≠ 0) : -Real.sin φ + -r * (-(Real.sin φ / r)) = 0 := by
  VERIPHY[dg-cc1-poly]

/-- `∂_φ X^φ = −cos φ / r` is cancelled by `Γ^φ_{φr} X^r = (1/r) cos φ`. -/
theorem dx_cancellation_phi (r φ : ℝ) (hr : r ≠ 0) : -(Real.cos φ / r) + 1 / r * Real.cos φ = 0 := by
  VERIPHY[dg-cc1-poly]

/-! ## exm-covariant-rotation -/

/-- The rotation field `∂_φ = (0, 1)` on the sphere: `∇_θ X^φ = cot θ`, `∇_φ X^θ = −sin θ cos θ`, the other two
components vanish (`∇_i X^k` is the entry `i, k`). -/
theorem dphi_covDeriv (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ i k, covDeriv sphere dphi i k x =
      ![![0, Real.cos (x 0) / Real.sin (x 0)], ![-(Real.sin (x 0) * Real.cos (x 0)), 0]] i k := by
  VERIPHY[dg-cc1-geo]

/-- For the lowered rotation field `X_j = (0, sin²θ)`, `∇_i X_j + ∇_j X_i = 0`: `∂_φ` is a Killing field. -/
theorem dphi_lowered_killing (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ i j, covDerivForm sphere dphiLowered i j x + covDerivForm sphere dphiLowered j i x = 0 := by
  VERIPHY[dg-cc1-geo]

/-! ## exm-covariant-trivial -/

/-- With `Γ = 0` (a constant metric) the covariant derivative is the partial derivative, `∇_i X^k = ∂_i X^k`. -/
theorem covDeriv_const_metric (c : Matrix (Fin 2) (Fin 2) ℝ) (X : Fin 2 → (Fin 2 → ℝ) → ℝ) (i k : Fin 2)
    (x : Fin 2 → ℝ) : covDeriv (fun i j _ => c i j) X i k x = pd i (X k) x := by
  simp [covDeriv, christoffel_const]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.covDeriv_const_metric, dg-cc1-covderiv-const-test]

/-! ## exr-covariant-radial-field -/

/-- For the radial field `V = r ∂_r = (r, 0)`, `∇_i V^k = δ^k_i`. -/
theorem radial_covDeriv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i k, covDeriv polar radialField i k x = ![![1, 0], ![0, 1]] i k := by
  VERIPHY[dg-cc1-geo]

/-- For the rotation field `W = ∂_φ = (0, 1)` of the plane, `∇_r W^φ = 1/r` and `∇_φ W^r = −r`, the other components
vanish. -/
theorem rotation_covDeriv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i k, covDeriv polar dphi i k x = ![![0, 1 / x 0], ![-(x 0), 0]] i k := by
  VERIPHY[dg-cc1-geo]

/-- The polar-to-Cartesian Jacobian `∂(x, y)/∂(r, φ)` sends `r ∂_r = (r, 0)` to `(x, y) = (r cos φ, r sin φ)` and
`∂_φ = (0, 1)` to `(−y, x) = (−r sin φ, r cos φ)`. -/
theorem fields_cartesian (r φ : ℝ) :
    !![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ] *ᵥ ![r, 0] = ![r * Real.cos φ, r * Real.sin φ] ∧
      !![Real.cos φ, -(r * Real.sin φ); Real.sin φ, r * Real.cos φ] *ᵥ ![0, 1] =
        ![-(r * Real.sin φ), r * Real.cos φ] := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-matrix]
  · VERIPHY[dg-cc1-matrix]

/-- In Cartesian coordinates `∂_i x^k = δ^k_i` and the rotation field `(−y, x)` has the constant antisymmetric
derivative `∂_i W^k = (0, 1; −1, 0)` (entry `i, k`). -/
theorem cartesian_derivatives (x : Fin 2 → ℝ) :
    (∀ i k, pd i (fun y => y k) x = δ2 i k) ∧
      ∀ i k, pd i (![fun y => -(y 1), fun y => y 0] k) x = ![![0, 1], ![-1, 0]] i k := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-geo]

/-! ## exr-covariant-sphere-dtheta -/

/-- For the meridian field `∂_θ = (1, 0)` on the sphere the only non-zero component is `∇_φ X^φ = cot θ`; in
particular `∇_θ X = 0`. -/
theorem dtheta_covDeriv (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ i k, covDeriv sphere dtheta i k x = ![![0, 0], ![0, Real.cos (x 0) / Real.sin (x 0)]] i k := by
  VERIPHY[dg-cc1-geo]

/-! ## def-covariant-derivative-tensor, exr-covariant-leibniz -/

/-- With the sign rule for one-forms, the connection terms cancel in the derivative of the contraction:
`(∇_i ω_j) X^j + ω_j ∇_i X^j = (∂_i ω_j) X^j + ω_j ∂_i X^j = ∂_i (ω_j X^j)`. -/
theorem form_sign_rule_contraction (ω X : Fin 2 → ℝ) (dω dX : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (i : Fin 2) :
    ∑ j, ((dω i j - ∑ k, Γ k i j * ω k) * X j + ω j * (dX i j + ∑ k, Γ j i k * X k)) =
      ∑ j, (dω i j * X j + ω j * dX i j) := by
  simp only [Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- The Leibniz rule on a product of a vector and a one-form: the `(1, 1)` formula (one `+Γ` for the upper index,
one `−Γ` for the lower) applied to `X^k ω_j` equals `(∇_i X^k) ω_j + X^k ∇_i ω_j`. -/
theorem leibniz_vector_form (ω X : Fin 2 → ℝ) (dω dX : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (i k j : Fin 2) :
    (dX i k * ω j + X k * dω i j) + ∑ l, Γ k i l * (X l * ω j) - ∑ l, Γ l i j * (X k * ω l) =
      (dX i k + ∑ l, Γ k i l * X l) * ω j + X k * (dω i j - ∑ l, Γ l i j * ω l) := by
  simp only [Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- The covariant derivative commutes with contractions: contracting the `(1, 1)` derivative of `X^k ω_j` gives the
partial derivative of the function `X^k ω_k`. -/
theorem contraction_commutes (ω X : Fin 2 → ℝ) (dω dX : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (i : Fin 2) :
    ∑ k, ((dX i k * ω k + X k * dω i k) + ∑ l, Γ k i l * (X l * ω k) - ∑ l, Γ l i k * (X k * ω l)) =
      ∑ k, (dX i k * ω k + X k * dω i k) := by
  simp only [Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- The sign rule follows from the Leibniz rule: if `∇_i ω_j` (unknown `N_{ij}`) satisfies `N_{ij} X^j + ω_k (∂_i X^k
+ Γ^k_{ij} X^j) = (∂_i ω_j) X^j + ω_j ∂_i X^j` for every `X`, then `N_{ij} = ∂_i ω_j − Γ^k_{ij} ω_k`. -/
theorem sign_rule_of_leibniz (N dω : Fin 2 → Fin 2 → ℝ) (ω : Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (i : Fin 2)
    (h : ∀ X : Fin 2 → ℝ, ∑ j, N i j * X j + ∑ k, ω k * (∑ j, Γ k i j * X j) = ∑ j, dω i j * X j) :
    ∀ m, N i m = dω i m - ∑ k, Γ k i m * ω k := by
  have h0 := h ![1, 0]
  have h1 := h ![0, 1]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_one, mul_zero,
    add_zero, zero_add] at h0 h1
  simp only [Fin.forall_fin_two, Fin.sum_univ_two]
  refine ⟨?_, ?_⟩
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]

/-! ## exm-covariant-function -/

/-- The Hessian `∂_i ∂_j f − Γ^k_{ij} ∂_k f` is symmetric when the connection is torsion-free (second partial
derivatives commute). -/
theorem hessian_symm (ddf : Fin 2 → Fin 2 → ℝ) (df : Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (hdd : ∀ i j, ddf i j = ddf j i) (hΓ : ∀ k i j, Γ k i j = Γ k j i) (i j : Fin 2) :
    ddf i j - ∑ k, Γ k i j * df k = ddf j i - ∑ k, Γ k j i * df k := by
  have h := hdd i j
  have h0 := hΓ 0 i j
  have h1 := hΓ 1 i j
  simp only [Fin.sum_univ_two]
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-- The covariant derivative of a one-form is a `(0, 2)` tensor (dimension two, jet variables): with `ω'_m = K^l_m
ω_l`, `∂'_j ω'_m = K^i_j K^l_m ∂_i ω_l + H^l_{mj} ω_l`, the connection law for `Γ'`, `K J = 1` and `H` symmetric,
`∂'_j ω'_m − Γ'^k_{jm} ω'_k = K^i_j K^l_m (∂_i ω_l − Γ^a_{il} ω_a)`. -/
theorem covDerivForm_tensor (ω : Fin 2 → ℝ) (dω : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (J K : Fin 2 → Fin 2 → ℝ) (H : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (hKJ00 : K 0 0 * J 0 0 + K 0 1 * J 1 0 = 1) (hKJ01 : K 0 0 * J 0 1 + K 0 1 * J 1 1 = 0)
    (hKJ10 : K 1 0 * J 0 0 + K 1 1 * J 1 0 = 0) (hKJ11 : K 1 0 * J 0 1 + K 1 1 * J 1 1 = 1)
    (hH : ∀ c i j, H c i j = H c j i) :
    ∀ j m : Fin 2,
      (∑ i, ∑ l, K i j * K l m * dω i l + ∑ l, H l m j * ω l) - ∑ k, connLaw Γ J K H k j m * (∑ l, K l k * ω l) =
        ∑ i, ∑ l, K i j * K l m * (dω i l - ∑ a, Γ a i l * ω a) := by
  have h001 := hH 0 0 1
  have h101 := hH 1 0 1
  simp only [Fin.forall_fin_two, connLaw, Fin.sum_univ_two]
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]
  · rw [← sub_eq_zero]
    VERIPHY[dg-cc1-poly]

/-! ## exm-covariant-dphi-form -/

/-- The one-form `dφ = (0, 1)` on the sphere: `∇_θ (dφ)_φ = ∇_φ (dφ)_θ = −cot θ`, the other components vanish. -/
theorem dphi_form_covDeriv (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ i j, covDerivForm sphere dphi i j x =
      ![![0, -(Real.cos (x 0) / Real.sin (x 0))], ![-(Real.cos (x 0) / Real.sin (x 0)), 0]] i j := by
  VERIPHY[dg-cc1-geo]

/-! ## exr-covariant-hessian-sphere -/

/-- The Hessian of the height `f = cos θ` on the unit sphere is `−f g_{ij}`: `∇_θ∇_θ f = −cos θ`,
`∇_φ∇_φ f = −cos θ sin²θ`, the mixed components vanish (so it is symmetric). -/
theorem hessian_height (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ i j, hessian sphere heightZ i j x = -(Real.cos (x 0)) * sphere i j x := by
  VERIPHY[dg-cc1-geo]

/-- The trace `g^{ij} ∇_i ∇_j f` of the Hessian of `cos θ` is `−2 cos θ`: the height is an eigenfunction of the
Laplacian with eigenvalue `−2`. -/
theorem laplacian_height (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∑ i, ∑ j, invMetric sphere x i j * hessian sphere heightZ i j x = -2 * Real.cos (x 0) := by
  VERIPHY[dg-cc1-geo]

/-- The eigenvalue `−2` is `−ℓ(ℓ + 1)` for `ℓ = 1`. -/
theorem eigenvalue_one : (-2 : ℝ) = -(1 * (1 + 1)) := by
  VERIPHY[dg-cc1-poly]

/-! ## exr-covariant-polar-forms -/

/-- `∇_i (dr)_j = −Γ^r_{ij}`: the only non-zero component is `∇_φ (dr)_φ = r`. -/
theorem dr_form_covDeriv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i j, covDerivForm polar dtheta i j x = ![![0, 0], ![0, x 0]] i j := by
  VERIPHY[dg-cc1-geo]

/-- `∇_i (dφ)_j = −Γ^φ_{ij}`: non-zero for `∇_r (dφ)_φ = ∇_φ (dφ)_r = −1/r`. -/
theorem dphi_polar_form_covDeriv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i j, covDerivForm polar dphi i j x = ![![0, -(1 / x 0)], ![-(1 / x 0), 0]] i j := by
  VERIPHY[dg-cc1-geo]

/-- The one-form `dx = cos φ dr − r sin φ dφ` has vanishing covariant derivative. -/
theorem dx_form_parallel (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : ∀ i j, covDerivForm polar dxForm i j x = 0 := by
  VERIPHY[dg-cc1-geo]

/-! ## def-torsion, exr-torsion-tensor -/

/-- Torsion is a tensor: the inhomogeneous terms of the connection law are symmetric in the lower indices (second
derivatives commute, `H^c_{ij} = H^c_{ji}`) and cancel in `Γ'^k_{ij} − Γ'^k_{ji}`, leaving the `(1, 2)` tensor law. -/
theorem torsion_tensor (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (J K : Fin 2 → Fin 2 → ℝ)
    (H : Fin 2 → Fin 2 → Fin 2 → ℝ) (hH : ∀ c i j, H c i j = H c j i) (k i j : Fin 2) :
    connLaw Γ J K H k i j - connLaw Γ J K H k j i = tensorLaw (fun c a b => Γ c a b - Γ c b a) J K k i j := by
  have h0 := hH 0 i j
  have h1 := hH 1 i j
  simp only [connLaw, tensorLaw, Fin.sum_univ_two]
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-! ## exm-torsion-nonzero, exr-torsion-sage -/

/-- The connection `Γ^1_{12} = 1` (all other coefficients zero) has torsion `T^1_{12} = 1`, `T^1_{21} = −1`. -/
theorem torsionOf_example : torsionOf Γex 0 0 1 = 1 ∧ torsionOf Γex 0 1 0 = -1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-geo]

/-- The connection `Γ^1_{12} = 1` is flat: its constant coefficients have no derivatives and the quadratic terms of
the Riemann tensor cancel. -/
theorem riemannConst_example : ∀ ρ σ μ ν, riemannConst Γex ρ σ μ ν = 0 := by
  VERIPHY[dg-cc1-geo]

/-- The symmetric part `Γ^1_{12} = Γ^1_{21} = 1/2` is torsion-free. -/
theorem Γsym_torsion_free : ∀ k i j, torsionOf Γsym k i j = 0 := by
  VERIPHY[dg-cc1-geo]

/-- The symmetric part is curved: `R^1_{212} = −Γ^1_{21} Γ^1_{12} = −1/4`. -/
theorem Γsym_curved : riemannConst Γsym 0 1 0 1 = -1 / 4 := by
  VERIPHY[dg-cc1-geo]

/-! ## exm-torsion-lie-group -/

/-- The connection declaring a left-invariant frame covariantly constant (`Γ = 0` in the frame) has torsion
`T^a_{bc} = ∇_b e_c − ∇_c e_b − [e_b, e_c] = −f^a_{bc}`, the structure constants. -/
theorem frame_connection_torsion (f : Fin 3 → Fin 3 → Fin 3 → ℝ) (a b c : Fin 3) :
    (0 : ℝ) - 0 - f a b c = -(f a b c) := by
  VERIPHY[dg-cc1-poly]

/-- The Levi-Civita connection of the bi-invariant metric of `SU(2)` is curved: the scalar curvature of the unit
`S³` is `6`. -/
theorem su2_curved (x : Fin 3 → ℝ) (h : Real.sin (x 0) ≠ 0) : scalarCurvature su2 x = 6 :=
  su2_scalarCurvature x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.su2_curved, dg-cc1-su2-curved-test]

/-! ## exr-torsion-commutator -/

/-- `T(X, Y) = ∇_X Y − ∇_Y X − [X, Y]` has components `(Γ^k_{ij} − Γ^k_{ji}) X^i Y^j`: the derivative terms
cancel. -/
theorem torsion_commutator (X Y : Fin 2 → ℝ) (dX dY : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (k : Fin 2) :
    (∑ i, X i * dY i k + ∑ i, ∑ j, Γ k i j * X i * Y j) - (∑ i, Y i * dX i k + ∑ i, ∑ j, Γ k i j * Y i * X j) -
        (∑ i, X i * dY i k - ∑ i, Y i * dX i k) =
      ∑ i, ∑ j, (Γ k i j - Γ k j i) * X i * Y j := by
  simp only [Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- No derivatives of `X`, `Y` remain in `T(X, Y)`, so it is linear over functions: `T(f X, Y) = f T(X, Y)`. -/
theorem torsion_tensorial_f (X Y : Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (f : ℝ) (k : Fin 2) :
    ∑ i, ∑ j, (Γ k i j - Γ k j i) * (f * X i) * Y j = f * ∑ i, ∑ j, (Γ k i j - Γ k j i) * X i * Y j := by
  simp only [Fin.sum_univ_two]
  VERIPHY[dg-cc1-poly]

/-- For a torsion-free connection `∇_X Y − ∇_Y X = [X, Y]`. -/
theorem bracket_of_torsion_free (X Y : Fin 2 → ℝ) (dX dY : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (hΓ : ∀ k i j, Γ k i j = Γ k j i) (k : Fin 2) :
    (∑ i, X i * dY i k + ∑ i, ∑ j, Γ k i j * X i * Y j) - (∑ i, Y i * dX i k + ∑ i, ∑ j, Γ k i j * Y i * X j) =
      ∑ i, X i * dY i k - ∑ i, Y i * dX i k := by
  have h01 := hΓ k 0 1
  simp only [Fin.sum_univ_two]
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-! ## exm-compatible-flat, exm-compatible-sphere, exm-compatible-not, exr-compatible-polar-check -/

/-- The trivial connection is compatible with the Euclidean metric: both sides of `∂_k g_{ij} = Γ^l_{ki} g_{lj} +
Γ^l_{kj} g_{il}` vanish. -/
theorem euclid_compatible (x : Fin 2 → ℝ) :
    ∀ k i j, pd k (euclid i j) x =
      ∑ l, christoffel euclid l k i x * euclid l j x + ∑ l, christoffel euclid l k j x * euclid i l x := by
  VERIPHY[dg-cc1-geo]

/-- The polar connection is compatible with the polar metric: `∂_k g_{ij} = Γ^l_{ki} g_{lj} + Γ^l_{kj} g_{il}` for
every component (`∇g = 0`). -/
theorem polar_compatible (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ k i j, pd k (polar i j) x =
      ∑ l, christoffel polar l k i x * polar l j x + ∑ l, christoffel polar l k j x * polar i l x := by
  VERIPHY[dg-cc1-geo]

/-- The `(r, φ, φ)` component of compatibility in polar coordinates, `∂_r (r²) = 2 Γ^φ_{rφ} r²`, is `2r = 2 (1/r) r²`. -/
theorem polar_compat_component (r : ℝ) (hr : r ≠ 0) : 2 * (1 / r) * r ^ 2 = 2 * r := by
  VERIPHY[dg-cc1-poly]

/-- The sphere's Levi-Civita connection is compatible with the round metric: every component of `∇g` vanishes. -/
theorem sphere_compatible (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    ∀ k i j, pd k (sphere i j) x =
      ∑ l, christoffel sphere l k i x * sphere l j x + ∑ l, christoffel sphere l k j x * sphere i l x := by
  VERIPHY[dg-cc1-geo]

/-- Solving `∂_θ sin²θ = 2 Γ^φ_{θφ} sin²θ` gives `Γ^φ_{θφ} = 2 sin θ cos θ / (2 sin²θ) = cot θ`. -/
theorem cot_from_compatibility (s c : ℝ) (hs : s ≠ 0) : 2 * s * c / (2 * s ^ 2) = c / s := by
  VERIPHY[dg-cc1-poly]

/-- For the zero connection of the polar chart `∇g` has the single non-zero component `∇_r g_{φφ} = ∂_r (r²) = 2r`
(entry `k, i, j`). -/
theorem polar_zero_connection_nabla_g (x : Fin 2 → ℝ) :
    ∀ k i j, pd k (polar i j) x = ![![![0, 0], ![0, 2 * x 0]], ![![0, 0], ![0, 0]]] k i j := by
  VERIPHY[dg-cc1-geo]

/-- `∂_r g_{φφ} = 2r ≠ 0` for `r ≠ 0`: the zero connection of the polar chart is not compatible with the Euclidean
metric. -/
theorem polar_zero_connection_nonzero (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : pd 0 (polar 1 1) x ≠ 0 := by
  have := polar_zero_connection_nabla_g x 0 1 1
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons] at this
  rw [this]
  exact mul_ne_zero two_ne_zero h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.polar_zero_connection_nonzero, dg-cc1-zero-connection-test]

/-- The zero connection is torsion-free and flat. -/
theorem zero_connection_torsion_flat :
    (∀ k i j : Fin 2, torsionOf (fun _ _ _ => (0 : ℝ)) k i j = 0) ∧
      ∀ ρ σ μ ν : Fin 2, riemannConst (fun _ _ _ => (0 : ℝ)) ρ σ μ ν = 0 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-geo]

/-! ## exr-compatible-raising -/

/-- The Kronecker delta is covariantly constant: `∇_k δ^i_l = 0 + Γ^i_{km} δ^m_l − Γ^m_{kl} δ^i_m = 0`. -/
theorem delta_covariant_const (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (k : Fin 2) :
    ∀ i l : Fin 2, (0 : ℝ) + ∑ m, Γ i k m * δ2 m l - ∑ m, Γ m k l * δ2 i m = 0 := by
  simp only [Fin.forall_fin_two, Fin.sum_univ_two, δ2, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]

/-- A metric-compatible connection has `∇_k g^{ij} = 0`: from `g^{ij} g_{jl} = δ^i_l` (both orders), its derivative
`(∂_k g^{ij}) g_{jl} + g^{ij} ∂_k g_{jl} = 0`, and compatibility, `∂_k g^{ij} + Γ^i_{km} g^{mj} + Γ^j_{km} g^{im} =
0`. -/
theorem inverse_metric_covariant_const (g ginv : Fin 2 → Fin 2 → ℝ) (dg dginv : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (k : Fin 2)
    (hinv00 : ginv 0 0 * g 0 0 + ginv 0 1 * g 1 0 = 1) (hinv01 : ginv 0 0 * g 0 1 + ginv 0 1 * g 1 1 = 0)
    (hinv10 : ginv 1 0 * g 0 0 + ginv 1 1 * g 1 0 = 0) (hinv11 : ginv 1 0 * g 0 1 + ginv 1 1 * g 1 1 = 1)
    (hvni00 : g 0 0 * ginv 0 0 + g 0 1 * ginv 1 0 = 1) (hvni01 : g 0 0 * ginv 0 1 + g 0 1 * ginv 1 1 = 0)
    (hvni10 : g 1 0 * ginv 0 0 + g 1 1 * ginv 1 0 = 0) (hvni11 : g 1 0 * ginv 0 1 + g 1 1 * ginv 1 1 = 1)
    (hL : ∀ i l, ∑ j, dginv k i j * g j l + ∑ j, ginv i j * dg k j l = 0)
    (hc : ∀ i j, dg k i j = ∑ l, Γ l k i * g l j + ∑ l, Γ l k j * g i l) :
    ∀ i j : Fin 2, dginv k i j + ∑ m, Γ i k m * ginv m j + ∑ m, Γ j k m * ginv i m = 0 := by
  have hL00 := hL 0 0
  have hL01 := hL 0 1
  have hL10 := hL 1 0
  have hL11 := hL 1 1
  have hc00 := hc 0 0
  have hc01 := hc 0 1
  have hc10 := hc 1 0
  have hc11 := hc 1 1
  simp only [Fin.sum_univ_two] at hL00 hL01 hL10 hL11 hc00 hc01 hc10 hc11
  simp only [Fin.forall_fin_two, Fin.sum_univ_two]
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]
  · VERIPHY[dg-cc1-poly]

/-- Lowering commutes with a compatible covariant derivative: `∇_k (g_{ij} X^j) = (∂_k g_{ij}) X^j + g_{ij} ∂_k X^j
− Γ^l_{ki} g_{lj} X^j = g_{ij} (∂_k X^j + Γ^j_{kl} X^l)`. -/
theorem lower_commutes (g : Fin 2 → Fin 2 → ℝ) (dg : Fin 2 → Fin 2 → Fin 2 → ℝ) (X : Fin 2 → ℝ)
    (dX : Fin 2 → Fin 2 → ℝ) (Γ : Fin 2 → Fin 2 → Fin 2 → ℝ) (k i : Fin 2)
    (hc : ∀ i j, dg k i j = ∑ l, Γ l k i * g l j + ∑ l, Γ l k j * g i l) :
    ∑ j, (dg k i j * X j + g i j * dX k j) - ∑ l, Γ l k i * (∑ j, g l j * X j) =
      ∑ j, g i j * (dX k j + ∑ l, Γ j k l * X l) := by
  have hc0 := hc i 0
  have hc1 := hc i 1
  simp only [Fin.sum_univ_two] at hc0 hc1 ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-! ## exr-compatible-inner-product, def-metric-compatible, def-parallel-field -/

/-- Along a curve with velocity `ẋ` and parallel fields `V`, `W` (`V̇^k = −Γ^k_{ij} ẋ^i V^j`, likewise `W`),
`d/dt (g_{kl} V^k W^l) = ẋ^i V^j W^l (∂_i g_{jl} − Γ^k_{ij} g_{kl} − Γ^k_{il} g_{jk}) = ẋ^i V^j W^l ∇_i g_{jl}`. -/
theorem inner_product_derivative (g : Fin 2 → Fin 2 → ℝ) (dg Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (v V W dV dW : Fin 2 → ℝ)
    (hV : ∀ k, dV k = -∑ i, ∑ j, Γ k i j * v i * V j) (hW : ∀ k, dW k = -∑ i, ∑ j, Γ k i j * v i * W j) :
    ∑ k, ∑ l, ((∑ i, v i * dg i k l) * V k * W l + g k l * (dV k * W l + V k * dW l)) =
      ∑ i, ∑ j, ∑ l, v i * V j * W l * (dg i j l - ∑ k, Γ k i j * g k l - ∑ k, Γ k i l * g j k) := by
  have hV0 := hV 0
  have hV1 := hV 1
  have hW0 := hW 0
  have hW1 := hW 1
  simp only [Fin.sum_univ_two] at hV0 hV1 hW0 hW1 ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-cc1-poly]

/-- For a metric-compatible connection the inner product of two parallel fields is constant along the curve. -/
theorem inner_product_constant_of_compatible (g : Fin 2 → Fin 2 → ℝ) (dg Γ : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (v V W dV dW : Fin 2 → ℝ)
    (hV : ∀ k, dV k = -∑ i, ∑ j, Γ k i j * v i * V j) (hW : ∀ k, dW k = -∑ i, ∑ j, Γ k i j * v i * W j)
    (hc : ∀ k i j, dg k i j = ∑ l, Γ l k i * g l j + ∑ l, Γ l k j * g i l) :
    ∑ k, ∑ l, ((∑ i, v i * dg i k l) * V k * W l + g k l * (dV k * W l + V k * dW l)) = 0 := by
  have hV0 := hV 0
  have hV1 := hV 1
  have hW0 := hW 0
  have hW1 := hW 1
  have hc000 := hc 0 0 0
  have hc001 := hc 0 0 1
  have hc010 := hc 0 1 0
  have hc011 := hc 0 1 1
  have hc100 := hc 1 0 0
  have hc101 := hc 1 0 1
  have hc110 := hc 1 1 0
  have hc111 := hc 1 1 1
  simp only [Fin.sum_univ_two] at hV0 hV1 hW0 hW1 hc000 hc001 hc010 hc011 hc100 hc101 hc110 hc111 ⊢
  VERIPHY[dg-cc1-poly]

/-- Conversely, if `ẋ^i V^j W^l N_{ijl}` vanishes for all `ẋ`, `V`, `W` then every component `N_{ijl}` (here
`∇_i g_{jl}`) vanishes: compatibility along the curve for all parallel fields forces `∇g = 0`. -/
theorem compatible_of_inner_product_constant (N : Fin 2 → Fin 2 → Fin 2 → ℝ)
    (h : ∀ v V W : Fin 2 → ℝ, ∑ i, ∑ j, ∑ l, v i * V j * W l * N i j l = 0) : ∀ i j l, N i j l = 0 := by
  intro i j l
  have := h (Pi.single i 1) (Pi.single j 1) (Pi.single l 1)
  simpa [Pi.single_apply, Finset.sum_ite_eq', Finset.sum_ite_eq] using this
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.compatible_of_inner_product_constant, dg-cc1-compatible-converse-test]

/-! ## exm-dx-parallel-dy, exm-dx-parallel-radial, exm-dx-parallel-circle, exr-dx-parallel-dy-sage -/

/-- The field `∂_y = sin φ ∂_r + (cos φ / r) ∂_φ` is covariantly constant. -/
theorem dy_parallel (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : ∀ i k, covDeriv polar dyPolar i k x = 0 := by
  VERIPHY[dg-cc1-geo]

/-- Every constant field `a ∂_x + b ∂_y` is covariantly constant in polar coordinates. -/
theorem const_field_parallel (a b : ℝ) (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i k, covDeriv polar (constField a b) i k x = 0 := by
  VERIPHY[dg-cc1-geo]

/-- For the radial unit field `∂_r = (1, 0)`, `∇_i X^k = Γ^k_{ir}`: the only non-zero component is `∇_φ X^φ = 1/r`
(`∇ ∂_r = (1/r) ∂_φ ⊗ dφ`). -/
theorem dr_covDeriv (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    ∀ i k, covDeriv polar dtheta i k x = ![![0, 0], ![0, 1 / x 0]] i k := by
  VERIPHY[dg-cc1-geo]

/-- `|∂_φ|² = g_{φφ} = r²`, so the turning `(1/r) |∂_φ|` of the radial vector is `1` per radian. -/
theorem dr_turning (x : Fin 2 → ℝ) (h : x 0 ≠ 0) : polar 1 1 x = x 0 ^ 2 ∧ 1 / x 0 * x 0 = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-poly]

/-- On the unit circle `r = 1` the polar symbols are `Γ^r_{φφ} = −1` and `Γ^φ_{φr} = 1`. -/
theorem polar_christoffel_at_one (t : ℝ) :
    christoffel polar 0 1 1 ![1, t] = -1 ∧ christoffel polar 1 1 0 ![1, t] = 1 := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-geo]

/-- `X^r = cos t`, `X^φ = −sin t` solve the transport equations `Ẋ^r = X^φ`, `Ẋ^φ = −X^r` along the unit circle. -/
theorem circle_transport_solution (t : ℝ) :
    deriv (fun s => Real.cos s) t = -Real.sin t ∧ deriv (fun s => -Real.sin s) t = -Real.cos t := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-partials]
  · VERIPHY[dg-cc1-partials]

/-- At the point `(1, t)` of the unit circle the polar components of `∂_x` are `(cos t, −sin t)`. -/
theorem dxPolar_on_circle (t : ℝ) : dxPolar 0 ![1, t] = Real.cos t ∧ dxPolar 1 ![1, t] = -Real.sin t := by
  refine ⟨?_, ?_⟩
  · VERIPHY[dg-cc1-geo]
  · VERIPHY[dg-cc1-geo]

/-! ## exr-dx-parallel-general -/

/-- A function on the plane whose derivative vanishes everywhere is constant: the Cartesian components of a
covariantly constant field (`∂_i X^k = 0` in the Cartesian chart) are constants. -/
theorem const_of_fderiv_zero (f : (Fin 2 → ℝ) → ℝ) (hd : Differentiable ℝ f) (hf : ∀ x, fderiv ℝ f x = 0)
    (x y : Fin 2 → ℝ) : f x = f y :=
  is_const_of_fderiv_eq_zero hd hf x y
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.const_of_fderiv_zero, dg-cc1-const-field-test]

/-- On the sphere, `∇_φ X^θ = 0` and `∇_φ X^φ = 0` (`∂_φ X^θ = sin θ cos θ X^φ`, `∂_φ X^φ = −cot θ X^θ`) give
`∂²_φ X^θ = −cos²θ X^θ`. -/
theorem sphere_parallel_ode (s c k X0 X1 dX0 dX1 ddX0 : ℝ) (hk : k * s = c) (h1 : dX0 = s * c * X1)
    (h2 : dX1 = -k * X0) (h3 : ddX0 = s * c * dX1) : ddX0 = -c ^ 2 * X0 := by
  VERIPHY[dg-cc1-poly]

/-- A component oscillating in `φ` with frequency `ω ∈ (0, 1)` (`cos (ω φ)`) is not `2π`-periodic: `cos (2πω) ≠ 1`,
so a covariantly constant field on the sphere must vanish. -/
theorem periodicity_obstruction (ω : ℝ) (h0 : 0 < ω) (h1 : ω < 1) : Real.cos (2 * Real.pi * ω) ≠ 1 := by
  intro h
  rw [Real.cos_eq_one_iff] at h
  obtain ⟨n, hn⟩ := h
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  have hω : (n : ℝ) = ω := by
    have : (n : ℝ) * (2 * Real.pi) = ω * (2 * Real.pi) := by linarith
    exact mul_right_cancel₀ hpi.ne' this
  have hn0 : (0 : ℝ) < n := by rw [hω]; exact h0
  have hn1 : (n : ℝ) < 1 := by rw [hω]; exact h1
  have hz0 : (0 : ℤ) < n := by exact_mod_cast hn0
  have hz1 : n < (1 : ℤ) := by exact_mod_cast hn1
  omega
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.CovariantDerivative.periodicity_obstruction, dg-cc1-periodicity-test]

end KaluzaKlein.DifferentialGeometry.CovariantDerivative

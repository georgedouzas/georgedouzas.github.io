import Mathlib
import KaluzaKlein.Geometry.Curvature
import KaluzaKlein.Geometry.MetricExamples
import KaluzaKlein.Geometry.Polar

/-!
# Differential geometry, part "The Riemann tensor": general lemmas

Machinery shared by the modules `Riemann`, `Ricci` and `ConnectionsExercises` (sections "The Riemann tensor",
"Ricci tensor and scalar curvature" and the Exercises of `## Connections and curvature` of
`tutorial/part-i-09-differential-geometry.qmd`). Geometry is in coordinates (design D28): a metric is a family of
component functions on `Fin n → ℝ` and the curvature is the `CoordinateGeometry` asset's `riemann`, `ricci`,
`scalarCurvature`, built from `christoffel` exactly as in the chapter's formulas.

* Derived quantities: the lowered tensor `R_ρσμν = g_ρλ R^λ_σμν` (`lowered`), the quadratic form
  `R(X, Y, X, Y)` of an array (`quadForm`), the Gram determinant `g(X,X) g(Y,Y) − g(X,Y)²` (`gramDet`), the
  sectional curvature (`sectional`), the Gauss curvature `R_1212 / det g` of a surface (`gauss`), the Einstein
  tensor (`einstein`), the curvature `Γ Γ − Γ Γ` of a constant connection (`riemannOfConst`), and the scaled metric
  `λ g` (`scaled`).
* The surface metrics of the chapter are all of the form `E(u) du² + G(u) dv²` (`diag2 E G`): the plane in polar
  coordinates, the sphere, the torus, the cone, the cylinder, surfaces of revolution, the spheroid and the
  hyperbolic plane (in the order `(y, x)` of its coordinates). Their Christoffel symbols, Riemann components,
  lowered tensor, Gauss curvature, Ricci tensor, scalar curvature and Einstein tensor are proved once, for every
  `E` and `G` differentiable as the formulas require, at every point where `E` and `G` do not vanish; the derivative
  of a Christoffel symbol is computed through its closed form, which holds on a neighbourhood of the point
  (`pd_congr_nhds`, the pattern of `KaluzaKlein.Geometry.Curvature`).
* Scaling: `Γ`, `R^ρ_σμν` and `R_ij` of `λ g` are those of `g`; `R` and the sectional curvature divide by `λ`; the
  lowered tensor multiplies by `λ`.
* Abstract contractions: the Ricci tensor, scalar curvature and Einstein tensor of an array of the constant-curvature
  form `K (δ^ρ_μ g_σν − δ^ρ_ν g_σμ)`, and the trace algebra of the Einstein tensor.
-/

namespace KaluzaKlein.DifferentialGeometry.Riemann

open CoordinateGeometry Matrix Filter Topology

/-! ## Derived curvature quantities -/

/-- The Riemann tensor with all indices down, `R_ρσμν = g_ρλ R^λ_σμν`. -/
noncomputable def lowered {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) : ℝ :=
  ∑ l, g ρ l x * riemann g l σ μ ν x

/-- The quadratic form `T(X, Y, X, Y) = T_ρσμν X^ρ Y^σ X^μ Y^ν` of a four-index array. -/
def quadForm {n : ℕ} (T : Fin n → Fin n → Fin n → Fin n → ℝ) (X Y : Fin n → ℝ) : ℝ :=
  ∑ ρ, ∑ σ, ∑ μ, ∑ ν, T ρ σ μ ν * X ρ * Y σ * X μ * Y ν

/-- The inner product `g(X, Y) = g_ij X^i Y^j` of two tangent vectors at `x`. -/
def inner {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X Y : Fin n → ℝ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, g i j x * X i * Y j

/-- The Gram determinant `g(X,X) g(Y,Y) − g(X,Y)²`, the squared area of the parallelogram spanned by `X, Y`. -/
def gramDet {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X Y : Fin n → ℝ) (x : Fin n → ℝ) : ℝ :=
  inner g X X x * inner g Y Y x - inner g X Y x ^ 2

/-- The sectional curvature of the plane spanned by `X, Y`:
`K(X, Y) = R_ρσμν X^ρ Y^σ X^μ Y^ν / (g(X,X) g(Y,Y) − g(X,Y)²)`. -/
noncomputable def sectional {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X Y : Fin n → ℝ) (x : Fin n → ℝ) : ℝ :=
  quadForm (fun ρ σ μ ν => lowered g ρ σ μ ν x) X Y / gramDet g X Y x

/-- The Gauss curvature of a surface, `K = R_1212 / det g`. -/
noncomputable def gauss (g : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) : ℝ :=
  lowered g 0 1 0 1 x / (metricAt g x).det

/-- The Einstein tensor `G_ij = R_ij − ½ R g_ij`. -/
noncomputable def einstein {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (i j : Fin n) (x : Fin n → ℝ) : ℝ :=
  ricci g i j x - (1 / 2) * scalarCurvature g x * g i j x

/-- The curvature `R^ρ_σμν = Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ` of a connection with constant coefficients (the
derivative terms of the Riemann formula vanish). -/
def riemannOfConst {n : ℕ} (Γ : Fin n → Fin n → Fin n → ℝ) (ρ σ μ ν : Fin n) : ℝ :=
  ∑ l, (Γ ρ μ l * Γ l ν σ - Γ ρ ν l * Γ l μ σ)

/-- The metric `c g`, all components multiplied by the constant `c`. -/
def scaled {n : ℕ} (c : ℝ) (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) : Fin n → Fin n → (Fin n → ℝ) → ℝ :=
  fun i j x => c * g i j x

/-- The Ricci contraction `T^ρ_σρν` of an array. -/
def ricciOf {n : ℕ} (T : Fin n → Fin n → Fin n → Fin n → ℝ) (σ ν : Fin n) : ℝ := ∑ ρ, T ρ σ ρ ν

/-- The trace `g^σν T_σν` of a two-index array with an inverse metric matrix. -/
def traceOf {n : ℕ} (ginv : Matrix (Fin n) (Fin n) ℝ) (T : Fin n → Fin n → ℝ) : ℝ := ∑ σ, ∑ ν, ginv σ ν * T σ ν

/-! ## Partial derivatives -/

/-- Two functions that agree near `x` have the same partial derivatives at `x`. -/
theorem pd_congr_nhds {n : ℕ} {f g : (Fin n → ℝ) → ℝ} (i : Fin n) (x : Fin n → ℝ) (h : ∀ᶠ y in 𝓝 x, f y = g y) :
    pd i f x = pd i g x := by
  unfold pd
  apply Filter.EventuallyEq.deriv_eq
  have hc : ContinuousAt (fun t : ℝ => Function.update x i t) (x i) :=
    (continuous_const.update i continuous_id).continuousAt
  have hx : Function.update x i (x i) = x := Function.update_eq_self i x
  rw [← hx] at h
  exact hc.eventually h

/-- The partial derivative along the first coordinate of a function of the first coordinate. -/
theorem pd_zero_comp {n : ℕ} (E : ℝ → ℝ) (x : Fin (n + 1) → ℝ) : pd 0 (fun y => E (y 0)) x = deriv E (x 0) := by
  simp [pd]

/-- A function of the first coordinate has vanishing partial derivative along any other coordinate. -/
theorem pd_other_comp {n : ℕ} (E : ℝ → ℝ) (i : Fin (n + 1)) (hi : i ≠ 0) (x : Fin (n + 1) → ℝ) :
    pd i (fun y => E (y 0)) x = 0 := by
  simp [pd, Ne.symm hi]

/-! ## Surface metrics `E(u) du² + G(u) dv²` -/

/-- A diagonal surface metric whose components depend on the first coordinate only: `E(u) du² + G(u) dv²`. -/
noncomputable def diag2 (E G : ℝ → ℝ) : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![![fun x => E (x 0), fun _ => 0], ![fun _ => 0, fun x => G (x 0)]]

section diag2

variable (E G : ℝ → ℝ) (x : Fin 2 → ℝ)

theorem diag2_symm (i j : Fin 2) : diag2 E G i j = diag2 E G j i := by
  fin_cases i <;> fin_cases j <;> rfl

/-- The matrix of `diag2 E G` at `x`. -/
theorem diag2_metricAt : metricAt (diag2 E G) x = !![E (x 0), 0; 0, G (x 0)] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [metricAt, diag2]

/-- `det g = E G`. -/
theorem diag2_det : (metricAt (diag2 E G) x).det = E (x 0) * G (x 0) := by
  rw [diag2_metricAt, Matrix.det_fin_two_of]; ring

/-- The inverse metric `diag(1/E, 1/G)`, where `E` and `G` do not vanish. -/
theorem diag2_invMetric (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0) :
    invMetric (diag2 E G) x = !![1 / E (x 0), 0; 0, 1 / G (x 0)] := by
  unfold invMetric
  refine Matrix.inv_eq_right_inv ?_
  rw [diag2_metricAt]
  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ] <;> field_simp

/-- The six Christoffel symbols of `E(u) du² + G(u) dv²`: `Γ^u_uu = E'/(2E)`, `Γ^u_vv = −G'/(2E)`,
`Γ^v_uv = Γ^v_vu = G'/(2G)`, the others zero. -/
theorem diag2_christoffel (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0) (k i j : Fin 2) :
    christoffel (diag2 E G) k i j x =
      (![![![fun y => deriv E (y 0) / (2 * E (y 0)), fun _ => 0], ![fun _ => 0, fun y => -(deriv G (y 0)) / (2 * E (y 0))]],
         ![![fun _ => 0, fun y => deriv G (y 0) / (2 * G (y 0))], ![fun y => deriv G (y 0) / (2 * G (y 0)), fun _ => 0]]] :
        Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) k i j x := by
  rw [christoffel, diag2_invMetric E G x hE hG]
  fin_cases k <;> fin_cases i <;> fin_cases j <;>
    simp [diag2, Fin.sum_univ_succ, pd_zero_comp, pd_other_comp, pd_const] <;>
    field_simp

/-- The Christoffel symbols as functions on the chart domain (the point where they are evaluated carries the
hypotheses), for the derivative lemma. -/
theorem diag2_christoffel_fun (k i j : Fin 2) (y : Fin 2 → ℝ) (hE : E (y 0) ≠ 0) (hG : G (y 0) ≠ 0) :
    christoffel (diag2 E G) k i j y =
      (![![![fun y => deriv E (y 0) / (2 * E (y 0)), fun _ => 0], ![fun _ => 0, fun y => -(deriv G (y 0)) / (2 * E (y 0))]],
         ![![fun _ => 0, fun y => deriv G (y 0) / (2 * G (y 0))], ![fun y => deriv G (y 0) / (2 * G (y 0)), fun _ => 0]]] :
        Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) k i j y :=
  diag2_christoffel E G y hE hG k i j

variable {E G x}

/-- Where `E` and `G` are continuous and non-zero, the Christoffel symbols agree with their closed forms near `x`. -/
theorem diag2_christoffel_eventually (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0) (hEc : ContinuousAt E (x 0))
    (hGc : ContinuousAt G (x 0)) (k i j : Fin 2) :
    ∀ᶠ y in 𝓝 x, christoffel (diag2 E G) k i j y =
      (![![![fun y => deriv E (y 0) / (2 * E (y 0)), fun _ => 0], ![fun _ => 0, fun y => -(deriv G (y 0)) / (2 * E (y 0))]],
         ![![fun _ => 0, fun y => deriv G (y 0) / (2 * G (y 0))], ![fun y => deriv G (y 0) / (2 * G (y 0)), fun _ => 0]]] :
        Fin 2 → Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) k i j y := by
  have h0 : ContinuousAt (fun y : Fin 2 → ℝ => y 0) x := (continuous_apply 0).continuousAt
  have hE' : ∀ᶠ y in 𝓝 x, E (y 0) ≠ 0 :=
    (ContinuousAt.comp (f := fun y : Fin 2 → ℝ => y 0) (x := x) hEc h0).eventually_ne hE
  have hG' : ∀ᶠ y in 𝓝 x, G (y 0) ≠ 0 :=
    (ContinuousAt.comp (f := fun y : Fin 2 → ℝ => y 0) (x := x) hGc h0).eventually_ne hG
  filter_upwards [hE', hG'] with y hy1 hy2 using diag2_christoffel_fun E G k i j y hy1 hy2

variable (E G x)

/-- `∂_v` of a Christoffel symbol vanishes: the symbols depend on `u` only. -/
theorem diag2_pd_one_christoffel (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0) (k i j : Fin 2) :
    pd 1 (christoffel (diag2 E G) k i j) x = 0 := by
  have h : ∀ t, christoffel (diag2 E G) k i j (Function.update x 1 t) = christoffel (diag2 E G) k i j x := by
    intro t
    rw [diag2_christoffel_fun E G k i j _ (by simpa using hE) (by simpa using hG), diag2_christoffel E G x hE hG]
    fin_cases k <;> fin_cases i <;> fin_cases j <;> simp
  unfold pd
  simp only [h]
  exact deriv_const _ _

end diag2

section diag2curv

variable {E G : ℝ → ℝ} {E₁ G₁ G₂ : ℝ} {x : Fin 2 → ℝ}
variable (hE : E (x 0) ≠ 0) (hG : G (x 0) ≠ 0) (hE1 : HasDerivAt E E₁ (x 0)) (hG1 : HasDerivAt G G₁ (x 0))
  (hG2 : HasDerivAt (deriv G) G₂ (x 0))
include hE hG hE1 hG1 hG2

/-- `∂_u Γ^u_vv = ∂_u (−G'/(2E)) = −(G'' · 2E − G' · 2E') / (2E)²`. -/
theorem diag2_pd_zero_christoffel_0_1_1 :
    pd 0 (christoffel (diag2 E G) 0 1 1) x = -(G₂ * (2 * E (x 0)) - G₁ * (2 * E₁)) / (2 * E (x 0)) ^ 2 := by
  rw [pd_congr_nhds 0 x (diag2_christoffel_eventually hE hG hE1.continuousAt hG1.continuousAt 0 1 1)]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
  rw [pd_zero_comp (fun u => -(deriv G u) / (2 * E u)) x]
  have h2E : 2 * E (x 0) ≠ 0 := by positivity
  have hd : HasDerivAt (fun u => -(deriv G u) / (2 * E u))
      ((-G₂ * (2 * E (x 0)) - -(deriv G (x 0)) * (2 * E₁)) / (2 * E (x 0)) ^ 2) (x 0) :=
    hG2.neg.div (hE1.const_mul 2) h2E
  rw [hd.deriv, hG1.deriv]; ring

/-- `∂_u Γ^v_vu = ∂_u (G'/(2G)) = (G'' · 2G − G' · 2G') / (2G)²`. -/
theorem diag2_pd_zero_christoffel_1_1_0 :
    pd 0 (christoffel (diag2 E G) 1 1 0) x = (G₂ * (2 * G (x 0)) - G₁ * (2 * G₁)) / (2 * G (x 0)) ^ 2 := by
  rw [pd_congr_nhds 0 x (diag2_christoffel_eventually hE hG hE1.continuousAt hG1.continuousAt 1 1 0)]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
  rw [pd_zero_comp (fun u => deriv G u / (2 * G u)) x]
  have h2G : 2 * G (x 0) ≠ 0 := by positivity
  have hd : HasDerivAt (fun u => deriv G u / (2 * G u))
      ((G₂ * (2 * G (x 0)) - deriv G (x 0) * (2 * G₁)) / (2 * G (x 0)) ^ 2) (x 0) :=
    hG2.div (hG1.const_mul 2) h2G
  rw [hd.deriv, hG1.deriv]

omit hG2 in
/-- `∂_u Γ^u_vu = 0`. -/
theorem diag2_pd_zero_christoffel_0_1_0 : pd 0 (christoffel (diag2 E G) 0 1 0) x = 0 := by
  rw [pd_congr_nhds 0 x (diag2_christoffel_eventually hE hG hE1.continuousAt hG1.continuousAt 0 1 0)]
  simp [pd]

omit hG2 in
/-- `∂_u Γ^v_vv = 0`. -/
theorem diag2_pd_zero_christoffel_1_1_1 : pd 0 (christoffel (diag2 E G) 1 1 1) x = 0 := by
  rw [pd_congr_nhds 0 x (diag2_christoffel_eventually hE hG hE1.continuousAt hG1.continuousAt 1 1 1)]
  simp [pd]

/-- `R^u_vuv = −G''/(2E) + E'G'/(4E²) + G'²/(4EG)`. -/
theorem diag2_riemann_0_1_0_1 :
    riemann (diag2 E G) 0 1 0 1 x =
      -G₂ / (2 * E (x 0)) + E₁ * G₁ / (4 * E (x 0) ^ 2) + G₁ ^ 2 / (4 * E (x 0) * G (x 0)) := by
  unfold riemann
  rw [diag2_pd_zero_christoffel_0_1_1 hE hG hE1 hG1 hG2, diag2_pd_one_christoffel E G x hE hG]
  simp only [Fin.sum_univ_two, diag2_christoffel E G x hE hG]
  simp
  rw [hE1.deriv, hG1.deriv]
  field_simp
  ring

/-- `R^v_uuv = (G'' · 2G − 2G'²)/(4G²) + G'²/(4G²) − E'G'/(4EG)`. -/
theorem diag2_riemann_1_0_0_1 :
    riemann (diag2 E G) 1 0 0 1 x =
      G₂ / (2 * G (x 0)) - G₁ ^ 2 / (4 * G (x 0) ^ 2) - E₁ * G₁ / (4 * E (x 0) * G (x 0)) := by
  unfold riemann
  rw [diag2_pd_zero_christoffel_1_1_0 hE hG hE1 hG1 hG2, diag2_pd_one_christoffel E G x hE hG]
  simp only [Fin.sum_univ_two, diag2_christoffel E G x hE hG]
  simp
  rw [hE1.deriv, hG1.deriv]
  field_simp
  ring

/-- `R^u_uuv = 0`. -/
theorem diag2_riemann_0_0_0_1 : riemann (diag2 E G) 0 0 0 1 x = 0 := by
  unfold riemann
  rw [diag2_pd_zero_christoffel_0_1_0 hE hG hE1 hG1, diag2_pd_one_christoffel E G x hE hG]
  simp only [Fin.sum_univ_two, diag2_christoffel E G x hE hG]
  simp

/-- `R^v_vuv = 0`. -/
theorem diag2_riemann_1_1_0_1 : riemann (diag2 E G) 1 1 0 1 x = 0 := by
  unfold riemann
  rw [diag2_pd_zero_christoffel_1_1_1 hE hG hE1 hG1, diag2_pd_one_christoffel E G x hE hG]
  simp only [Fin.sum_univ_two, diag2_christoffel E G x hE hG]
  simp

/-- The Gauss curvature of `E du² + G dv²`: `K = R^u_vuv / G`. -/
theorem diag2_gauss :
    gauss (diag2 E G) x =
      (-G₂ / (2 * E (x 0)) + E₁ * G₁ / (4 * E (x 0) ^ 2) + G₁ ^ 2 / (4 * E (x 0) * G (x 0))) / G (x 0) := by
  unfold gauss lowered
  rw [diag2_det, Fin.sum_univ_two, diag2_riemann_0_1_0_1 hE hG hE1 hG1 hG2, diag2_riemann_1_1_0_1 hE hG hE1 hG1 hG2]
  simp [diag2]
  field_simp

/-- `R^u_vuv = K G`. -/
theorem diag2_riemann_0_1_0_1_gauss : riemann (diag2 E G) 0 1 0 1 x = gauss (diag2 E G) x * G (x 0) := by
  rw [diag2_gauss hE hG hE1 hG1 hG2, diag2_riemann_0_1_0_1 hE hG hE1 hG1 hG2]; field_simp

/-- `R^v_uuv = −K E`: the identity behind the antisymmetry of the lowered tensor in its first pair. -/
theorem diag2_riemann_1_0_0_1_gauss : riemann (diag2 E G) 1 0 0 1 x = -(gauss (diag2 E G) x * E (x 0)) := by
  rw [diag2_gauss hE hG hE1 hG1 hG2, diag2_riemann_1_0_0_1 hE hG hE1 hG1 hG2]; field_simp; ring

/-- Every component of the Riemann tensor of a surface `E du² + G dv²`, in terms of the Gauss curvature:
`R^u_vuv = K G`, `R^v_uuv = −K E`, their mirror images with the opposite sign, and zero otherwise. -/
theorem diag2_riemann_all (ρ σ μ ν : Fin 2) :
    riemann (diag2 E G) ρ σ μ ν x =
      (![![![![0, 0], ![0, 0]], ![![0, gauss (diag2 E G) x * G (x 0)], ![-(gauss (diag2 E G) x * G (x 0)), 0]]],
         ![![![0, -(gauss (diag2 E G) x * E (x 0))], ![gauss (diag2 E G) x * E (x 0), 0]], ![![0, 0], ![0, 0]]]] :
        Fin 2 → Fin 2 → Fin 2 → Fin 2 → ℝ) ρ σ μ ν := by
  have h1 := diag2_riemann_0_1_0_1_gauss hE hG hE1 hG1 hG2
  have h2 := diag2_riemann_1_0_0_1_gauss hE hG hE1 hG1 hG2
  have h3 := diag2_riemann_0_0_0_1 hE hG hE1 hG1 hG2
  have h4 := diag2_riemann_1_1_0_1 hE hG hE1 hG1 hG2
  fin_cases ρ <;> fin_cases σ <;> fin_cases μ <;> fin_cases ν <;>
    simp [h1, h2, h3, h4, KaluzaKlein.Geometry.riemann_diag,
      KaluzaKlein.Geometry.riemann_antisymm (diag2 E G) 0 1 0 1 x, KaluzaKlein.Geometry.riemann_antisymm (diag2 E G) 1 0 0 1 x,
      KaluzaKlein.Geometry.riemann_antisymm (diag2 E G) 0 0 0 1 x, KaluzaKlein.Geometry.riemann_antisymm (diag2 E G) 1 1 0 1 x]

/-- The lowered tensor of a surface: `R_ρσμν = K (g_ρμ g_σν − g_ρν g_σμ)`, the constant-curvature form with the
Gauss curvature as (pointwise) factor. -/
theorem diag2_lowered_form (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ μ ν x =
      gauss (diag2 E G) x * (diag2 E G ρ μ x * diag2 E G σ ν x - diag2 E G ρ ν x * diag2 E G σ μ x) := by
  unfold lowered
  rw [Fin.sum_univ_two]
  have h := diag2_riemann_all hE hG hE1 hG1 hG2
  simp only [h]
  fin_cases ρ <;> fin_cases σ <;> fin_cases μ <;> fin_cases ν <;>
    simp [diag2] <;> ring

/-- `R_uvuv = K E G = K det g`. -/
theorem diag2_lowered_0_1_0_1 : lowered (diag2 E G) 0 1 0 1 x = gauss (diag2 E G) x * (E (x 0) * G (x 0)) := by
  rw [diag2_lowered_form hE hG hE1 hG1 hG2]; simp [diag2]

/-- The lowered tensor is antisymmetric in its first pair. -/
theorem diag2_lowered_antisymm_first (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ μ ν x = -lowered (diag2 E G) σ ρ μ ν x := by
  rw [diag2_lowered_form hE hG hE1 hG1 hG2, diag2_lowered_form hE hG hE1 hG1 hG2]; ring

/-- The lowered tensor is antisymmetric in its last pair. -/
theorem diag2_lowered_antisymm_last (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ μ ν x = -lowered (diag2 E G) ρ σ ν μ x := by
  rw [diag2_lowered_form hE hG hE1 hG1 hG2, diag2_lowered_form hE hG hE1 hG1 hG2]; ring

/-- The lowered tensor is symmetric under exchange of the pairs. -/
theorem diag2_lowered_pair_symm (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ μ ν x = lowered (diag2 E G) μ ν ρ σ x := by
  rw [diag2_lowered_form hE hG hE1 hG1 hG2, diag2_lowered_form hE hG hE1 hG1 hG2]
  rw [diag2_symm E G ρ μ, diag2_symm E G σ ν, diag2_symm E G ρ ν, diag2_symm E G σ μ]; ring

/-- The first Bianchi identity `R_ρσμν + R_ρμνσ + R_ρνσμ = 0`. -/
theorem diag2_bianchi (ρ σ μ ν : Fin 2) :
    lowered (diag2 E G) ρ σ μ ν x + lowered (diag2 E G) ρ μ ν σ x + lowered (diag2 E G) ρ ν σ μ x = 0 := by
  rw [diag2_lowered_form hE hG hE1 hG1 hG2, diag2_lowered_form hE hG hE1 hG1 hG2,
    diag2_lowered_form hE hG hE1 hG1 hG2, diag2_symm E G ν σ, diag2_symm E G μ σ, diag2_symm E G ν μ]; ring

/-- The Ricci tensor of a surface is the Gauss curvature times the metric, `R_ij = K g_ij`. -/
theorem diag2_ricci (i j : Fin 2) : ricci (diag2 E G) i j x = gauss (diag2 E G) x * diag2 E G i j x := by
  unfold ricci
  rw [Fin.sum_univ_two]
  have h := diag2_riemann_all hE hG hE1 hG1 hG2
  simp only [h]
  fin_cases i <;> fin_cases j <;> simp [diag2]

/-- The Ricci tensor of a surface is symmetric. -/
theorem diag2_ricci_symm (i j : Fin 2) : ricci (diag2 E G) i j x = ricci (diag2 E G) j i x := by
  rw [diag2_ricci hE hG hE1 hG1 hG2, diag2_ricci hE hG hE1 hG1 hG2, diag2_symm]

/-- The scalar curvature of a surface is twice the Gauss curvature, `R = 2K`. -/
theorem diag2_scalarCurvature : scalarCurvature (diag2 E G) x = 2 * gauss (diag2 E G) x := by
  unfold scalarCurvature
  simp only [Fin.sum_univ_two, diag2_ricci hE hG hE1 hG1 hG2, diag2_invMetric E G x hE hG]
  simp [diag2]
  field_simp
  ring

/-- The Einstein tensor of every surface vanishes. -/
theorem diag2_einstein (i j : Fin 2) : einstein (diag2 E G) i j x = 0 := by
  unfold einstein
  rw [diag2_ricci hE hG hE1 hG1 hG2, diag2_scalarCurvature hE hG hE1 hG1 hG2]; ring

/-- The sectional curvature of the coordinate plane is the Gauss curvature. -/
theorem diag2_sectional : sectional (diag2 E G) ![1, 0] ![0, 1] x = gauss (diag2 E G) x := by
  unfold sectional quadForm gramDet inner
  simp only [Fin.sum_univ_two]
  have h := diag2_lowered_form hE hG hE1 hG1 hG2
  simp only [h]
  simp [diag2]
  field_simp

end diag2curv

/-! ## Scaling the metric by a constant -/

section scaling

variable {n : ℕ} (c : ℝ) (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x : Fin n → ℝ)

theorem scaled_metricAt : metricAt (scaled c g) x = c • metricAt g x := by
  ext i j; simp [metricAt, scaled]

/-- The inverse of `c g` is `c⁻¹ g⁻¹` (both sides vanish where `g` is degenerate). -/
theorem scaled_invMetric (hc : c ≠ 0) : invMetric (scaled c g) x = c⁻¹ • invMetric g x := by
  unfold invMetric
  rw [scaled_metricAt]
  by_cases h : IsUnit (metricAt g x).det
  · refine Matrix.inv_eq_left_inv ?_
    rw [Matrix.smul_mul, Matrix.mul_smul, Matrix.nonsing_inv_mul _ h, smul_smul, inv_mul_cancel₀ hc, one_smul]
  · rw [Matrix.nonsing_inv_apply_not_isUnit _ h, Matrix.nonsing_inv_apply_not_isUnit, smul_zero]
    rw [Matrix.det_smul, isUnit_iff_ne_zero] at *
    intro hu; exact h (by simpa [hc] using hu)

theorem scaled_pd (i l j : Fin n) : pd i (scaled c g l j) x = c * pd i (g l j) x := by
  unfold pd scaled
  rw [deriv_const_mul_field']

/-- The Christoffel symbols of `c g` are those of `g`. -/
theorem scaled_christoffel (hc : c ≠ 0) (k i j : Fin n) : christoffel (scaled c g) k i j x = christoffel g k i j x := by
  unfold christoffel
  rw [scaled_invMetric c g x hc]
  simp only [scaled_pd, Matrix.smul_apply, smul_eq_mul]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  field_simp

theorem scaled_christoffel_fun (hc : c ≠ 0) (k i j : Fin n) : christoffel (scaled c g) k i j = christoffel g k i j :=
  funext fun y => scaled_christoffel c g y hc k i j

/-- The Riemann tensor `R^ρ_σμν` of `c g` is that of `g`. -/
theorem scaled_riemann (hc : c ≠ 0) (ρ σ μ ν : Fin n) : riemann (scaled c g) ρ σ μ ν x = riemann g ρ σ μ ν x := by
  unfold riemann
  simp only [scaled_christoffel_fun c g hc, scaled_christoffel c g x hc]

/-- The Ricci tensor of `c g` is that of `g`. -/
theorem scaled_ricci (hc : c ≠ 0) (σ ν : Fin n) : ricci (scaled c g) σ ν x = ricci g σ ν x := by
  unfold ricci; simp only [scaled_riemann c g x hc]

/-- The scalar curvature of `c g` is `R / c`. -/
theorem scaled_scalarCurvature (hc : c ≠ 0) : scalarCurvature (scaled c g) x = scalarCurvature g x / c := by
  unfold scalarCurvature
  rw [scaled_invMetric c g x hc]
  simp only [scaled_ricci c g x hc, Matrix.smul_apply, smul_eq_mul, div_eq_inv_mul, Finset.mul_sum, mul_assoc]

/-- The lowered tensor of `c g` is `c` times that of `g`. -/
theorem scaled_lowered (hc : c ≠ 0) (ρ σ μ ν : Fin n) : lowered (scaled c g) ρ σ μ ν x = c * lowered g ρ σ μ ν x := by
  unfold lowered
  simp only [scaled_riemann c g x hc]
  unfold scaled
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun l _ => ?_
  ring

/-- The Gram determinant of `c g` is `c²` times that of `g`. -/
theorem scaled_gramDet (X Y : Fin n → ℝ) : gramDet (scaled c g) X Y x = c ^ 2 * gramDet g X Y x := by
  unfold gramDet inner scaled
  simp only [Finset.mul_sum, mul_assoc]
  ring

/-- `quadForm` of `c T` is `c` times `quadForm` of `T`. -/
theorem quadForm_smul (T : Fin n → Fin n → Fin n → Fin n → ℝ) (X Y : Fin n → ℝ) :
    quadForm (fun ρ σ μ ν => c * T ρ σ μ ν) X Y = c * quadForm T X Y := by
  unfold quadForm
  simp only [Finset.mul_sum, mul_assoc]

/-- The sectional curvature of `c g` is `K / c`. -/
theorem scaled_sectional (hc : c ≠ 0) (X Y : Fin n → ℝ) : sectional (scaled c g) X Y x = sectional g X Y x / c := by
  unfold sectional
  simp only [scaled_lowered c g x hc, scaled_gramDet, quadForm_smul]
  rw [pow_two, mul_assoc, mul_div_mul_left _ _ hc, div_div, mul_comm]

/-- The Einstein tensor of `c g` is that of `g`. -/
theorem scaled_einstein (hc : c ≠ 0) (i j : Fin n) : einstein (scaled c g) i j x = einstein g i j x := by
  unfold einstein scaled
  rw [scaled_ricci c g x hc, scaled_scalarCurvature c g x hc]
  field_simp

end scaling

/-- The Gauss curvature of `c g` is `K / c` (a surface). -/
theorem scaled_gauss (c : ℝ) (g : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) (hc : c ≠ 0) :
    gauss (scaled c g) x = gauss g x / c := by
  unfold gauss
  rw [scaled_lowered c g x hc, scaled_metricAt, Matrix.det_smul, Fintype.card_fin]
  field_simp
  ring

/-! ## Contractions of a constant-curvature array -/

section algebra

variable {n : ℕ}

/-- The Ricci contraction of `K (δ^ρ_μ g_σν − δ^ρ_ν g_σμ)` is `(n − 1) K g_σν`. -/
theorem ricciOf_const_curv (g : Matrix (Fin n) (Fin n) ℝ) (K : ℝ) (T : Fin n → Fin n → Fin n → Fin n → ℝ)
    (hT : ∀ ρ σ μ ν, T ρ σ μ ν = K * ((if ρ = μ then 1 else 0) * g σ ν - (if ρ = ν then 1 else 0) * g σ μ))
    (σ ν : Fin n) : ricciOf T σ ν = ((n : ℝ) - 1) * K * g σ ν := by
  unfold ricciOf
  simp only [hT, if_true, one_mul, mul_sub, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  rw [Finset.sum_mul_boole]
  simp
  ring

/-- `g^σν g_σν = n`: the trace of `g⁻¹ g = 1`. -/
theorem traceOf_metric (g ginv : Matrix (Fin n) (Fin n) ℝ) (h : ginv * g = 1) :
    traceOf ginv (fun σ ν => g σ ν) = (n : ℝ) := by
  unfold traceOf
  have := congrArg Matrix.trace h
  rw [Matrix.trace_one, Fintype.card_fin] at this
  rw [← this, Matrix.trace]
  refine Finset.sum_congr rfl fun σ _ => ?_
  simp only [Matrix.diag, Matrix.mul_apply]
  refine Finset.sum_congr rfl fun ν _ => ?_
  ring

/-- The trace is linear: `g^σν (a T_σν + b S_σν) = a g^σν T_σν + b g^σν S_σν`. -/
theorem traceOf_linear (ginv : Matrix (Fin n) (Fin n) ℝ) (a b : ℝ) (T S : Fin n → Fin n → ℝ) :
    traceOf ginv (fun σ ν => a * T σ ν + b * S σ ν) = a * traceOf ginv T + b * traceOf ginv S := by
  unfold traceOf
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun σ _ => Finset.sum_congr rfl fun ν _ => by ring

/-- The trace of a multiple of the metric: `g^ij (λ g_ij) = n λ`. -/
theorem traceOf_smul_metric (g ginv : Matrix (Fin n) (Fin n) ℝ) (h : ginv * g = 1) (lam : ℝ) :
    traceOf ginv (fun σ ν => lam * g σ ν) = (n : ℝ) * lam := by
  have := traceOf_linear ginv lam 0 (fun σ ν => g σ ν) (fun _ _ => 0)
  simp only [zero_mul, add_zero] at this
  rw [this, traceOf_metric g ginv h]; ring

/-- The trace of `(n − 1) K g` with the inverse metric is `n (n − 1) K`. -/
theorem traceOf_const_curv (g ginv : Matrix (Fin n) (Fin n) ℝ) (h : ginv * g = 1) (K : ℝ) :
    traceOf ginv (fun σ ν => ((n : ℝ) - 1) * K * g σ ν) = (n : ℝ) * ((n : ℝ) - 1) * K := by
  rw [traceOf_smul_metric g ginv h]; ring

/-- The trace of the Einstein tensor `G_ij = R_ij − ½ R g_ij` is `(1 − n/2) R`, where `R = g^ij R_ij`. -/
theorem traceOf_einstein (g ginv : Matrix (Fin n) (Fin n) ℝ) (h : ginv * g = 1) (Ric : Fin n → Fin n → ℝ) :
    traceOf ginv (fun i j => Ric i j - (1 / 2) * traceOf ginv Ric * g i j) = (1 - (n : ℝ) / 2) * traceOf ginv Ric := by
  have := traceOf_linear ginv 1 (-((1 / 2) * traceOf ginv Ric)) Ric (fun i j => g i j)
  simp only [one_mul, neg_mul, ← sub_eq_add_neg] at this
  rw [this, traceOf_metric g ginv h]; ring

end algebra

end KaluzaKlein.DifferentialGeometry.Riemann

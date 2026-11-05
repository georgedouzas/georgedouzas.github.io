import Mathlib
import KaluzaKlein.Geometry.VectorFields
import KaluzaKlein.Geometry.MetricExamples
import KaluzaKlein.Geometry.LieGroups
import KaluzaKlein.Geometry.Killing
import KaluzaKlein.PartI.DifferentialGeometry.TangentVectors

/-!
# Differential geometry, section "Vector fields"

The claims of the section "Vector fields" of `tutorial/part-i-09-differential-geometry.qmd`
(`## Vector fields, forms and tensors`), in coordinates (design D28):

* a vector field is its family of component functions `X : Fin n → (Fin n → ℝ) → ℝ`, acting on a function by
  `fieldAct X f x = Σ_i X^i(x) ∂_i f(x)`; the Lie bracket is `KaluzaKlein.lieBracket`, the chart formula
  `[X, Y]^k = X^l ∂_l Y^k − Y^l ∂_l X^k`;
* the fields of the sphere are those of `KaluzaKlein.Geometry` (`dphi`, `gradZ`, `dtheta`, `sinDphi`, the rotation
  generators `killingX/Y/Z`), the fields of the plane `rotPlane`, `radPlane` of `TangentVectors`;
* an integral curve of `X` is a curve `γ : ℝ → Fin n → ℝ` with `γ̇^i(t) = X^i(γ(t))` (`IsIntegralCurve`); the flows
  of the plane's fields are explicit maps;
* a frame is a family of `n` fields; the orthonormal frame of the sphere is `(∂_θ, ∂_φ / sin θ)`.

Facts that need the symmetry of second derivatives (the bracket as a commutator of derivations, the Jacobi identity)
and the hairy ball theorem are stated as propositions about explicit models and tested at instances.
-/

open CoordinateGeometry KaluzaKlein.Geometry Real KaluzaKlein.DifferentialGeometry.TangentVectors

namespace KaluzaKlein.DifferentialGeometry.VectorFields

/-- A vector field acting on a function: `X(f)(x) = Σ_i X^i(x) ∂_i f(x)`. -/
noncomputable def fieldAct {n : ℕ} (X : Fin n → (Fin n → ℝ) → ℝ) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, X i x * pd i f x

/-- The coordinate field `∂_i`, components `δ^k_i`. -/
def coordField {n : ℕ} (i : Fin n) : Fin n → (Fin n → ℝ) → ℝ := fun k _ => if k = i then 1 else 0

/-- The second vector of the orthonormal frame of the sphere, `e₂ = ∂_φ / sin θ`, components `(0, 1/sin θ)`. -/
noncomputable def e2Sphere : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun x => 1 / Real.sin (x 0)]

/-- The second vector of the orthonormal polar frame, `r⁻¹ ∂_φ`, components `(0, 1/r)`. -/
noncomputable def e2Polar : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun _ => 0, fun x => 1 / x 0]

/-- `γ` is an integral curve of `X`: `γ̇^i(t) = X^i(γ(t))` for every `t`. -/
def IsIntegralCurve {n : ℕ} (X : Fin n → (Fin n → ℝ) → ℝ) (γ : ℝ → Fin n → ℝ) : Prop :=
  ∀ t i, deriv (fun s => γ s i) t = X i (γ t)

/-- The rotation flow of the plane: `φ_t(x, y) = (x cos t − y sin t, x sin t + y cos t)`. -/
noncomputable def rotFlow (t : ℝ) (p : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![p 0 * Real.cos t - p 1 * Real.sin t, p 0 * Real.sin t + p 1 * Real.cos t]

/-- The dilation flow of the plane: `φ_t(p) = e^t p`. -/
noncomputable def radFlow (t : ℝ) (p : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => Real.exp t * p i

/-- The flow of `∂_φ` on the sphere by the angle `s`: `(θ, φ) ↦ (θ, φ + s)`. -/
def rotSphereFlow (s : ℝ) (x : Fin 2 → ℝ) : Fin 2 → ℝ := ![x 0, x 1 + s]

/-- The flow of the gradient field `−sin θ ∂_θ`: `θ ↦ 2 arctan(tan(θ/2) e^{−t})`, `φ` fixed. -/
noncomputable def gradFlow (t : ℝ) (x : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![2 * Real.arctan (Real.tan (x 0 / 2) * Real.exp (-t)), x 1]

/-! ## def-vector-field -/

/-- A vector field acts linearly on functions, pointwise. -/
theorem fieldAct_linear {n : ℕ} (X : Fin n → (Fin n → ℝ) → ℝ) (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f)
    (hg : Differentiable ℝ g) (c : ℝ) (x : Fin n → ℝ) :
    fieldAct X (f + g) x = fieldAct X f x + fieldAct X g x ∧ fieldAct X (fun y => c * f y) x = c * fieldAct X f x :=
  act_linear (fun i => X i x) f g hf hg c x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.fieldAct_linear, dg-vf1-fieldact-linear-test]

/-- A vector field obeys the Leibniz rule pointwise: `X(fg) = X(f) g + f X(g)`. -/
theorem fieldAct_mul {n : ℕ} (X : Fin n → (Fin n → ℝ) → ℝ) (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f)
    (hg : Differentiable ℝ g) (x : Fin n → ℝ) :
    fieldAct X (f * g) x = fieldAct X f x * g x + f x * fieldAct X g x :=
  act_mul (fun i => X i x) f g hf hg x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.fieldAct_mul, dg-vf1-fieldact-mul-test]

/-! ## exm-vector-field-rotation -/

/-- `∂_φ` has ambient length `sin θ`: `|∂_φ E|² = sin² θ`. -/
theorem dphi_ambient_length (x : Fin 2 → ℝ) :
    (∑ k, (∑ i, pd i (sphereEmbed k) x * dphi i x) ^ 2) = Real.sin (x 0) ^ 2 := by
  have h := fun k => (sphere_jacobian k x).2
  simp only [Fin.sum_univ_two, dphi, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_zero, mul_one,
    zero_add, h, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons]
  nlinarith [Real.sin_sq_add_cos_sq (x 1)]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.dphi_ambient_length, dg-vf1-dphi-ambient-length-test]

/-- At the poles `θ = 0, π` the ambient vector of `∂_φ` vanishes: the field extends to the poles by zero. -/
theorem dphi_vanishes_poles (φ : ℝ) (k : Fin 3) :
    (∑ i, pd i (sphereEmbed k) ![0, φ] * dphi i ![0, φ]) = 0 ∧
      (∑ i, pd i (sphereEmbed k) ![Real.pi, φ] * dphi i ![Real.pi, φ]) = 0 := by
  have h1 := (sphere_jacobian k ![0, φ]).2
  have h2 := (sphere_jacobian k ![Real.pi, φ]).2
  constructor
  · simp only [Fin.sum_univ_two, dphi, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_zero, mul_one,
      zero_add, h1]
    fin_cases k <;> simp
  · simp only [Fin.sum_univ_two, dphi, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_zero, mul_one,
      zero_add, h2]
    fin_cases k <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.dphi_vanishes_poles, dg-vf1-dphi-vanishes-poles-test]

/-- The flow of `∂_φ`, `φ ↦ φ + t`, is the rotation about the `Z`-axis by `t`: `R(t) E(θ, φ) = E(θ, φ + t)`. -/
theorem dphi_flow_rotation (t θ φ : ℝ) :
    rotation t *ᵥ ![Real.sin θ * Real.cos φ, Real.sin θ * Real.sin φ, Real.cos θ] =
      ![Real.sin θ * Real.cos (φ + t), Real.sin θ * Real.sin (φ + t), Real.cos θ] := by
  rw [Real.cos_add, Real.sin_add]
  VERIPHY[dg-vf1-matrix]

/-! ## exm-vector-field-gradient -/

/-- `−sin θ ∂_θ` points towards the north pole: its `θ`-component is negative for `0 < θ < π`. -/
theorem gradZ_points_north (x : Fin 2 → ℝ) (h0 : 0 < x 0) (hπ : x 0 < Real.pi) : gradZ 0 x < 0 := by
  simp only [gradZ, Matrix.cons_val_zero, neg_lt_zero]
  exact Real.sin_pos_of_pos_of_lt_pi h0 hπ
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_points_north, dg-vf1-gradz-points-north-test]

/-- `−sin θ ∂_θ` vanishes at both poles: `−sin 0 = 0` and `−sin π = 0`. -/
theorem gradZ_vanishes_poles (φ : ℝ) : gradZ 0 ![0, φ] = 0 ∧ gradZ 0 ![Real.pi, φ] = 0 := by
  simp [gradZ]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_vanishes_poles, dg-vf1-gradz-vanishes-poles-test]

/-- The hairy ball theorem for a continuous tangent field `v` on `S² ⊂ ℝ³`: `v` has a zero on the sphere. Stated as
a proposition about the explicit field (mathlib has no proof of the theorem); tested on explicit fields. -/
def HairyBall (v : (Fin 3 → ℝ) → Fin 3 → ℝ) : Prop :=
  Continuous v → (∀ p : Fin 3 → ℝ, ∑ k, p k ^ 2 = 1 → ∑ k, v p k * p k = 0) →
    ∃ p : Fin 3 → ℝ, ∑ k, p k ^ 2 = 1 ∧ v p = 0
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.HairyBall, dg-vf1-hairy-ball-test]

/-- Unlike on the sphere, `∂_φ` on the circle never vanishes: its ambient vector `(−sin φ, cos φ)` has length `1`. -/
theorem circle_field_nonvanishing (φ : ℝ) : (-Real.sin φ) ^ 2 + Real.cos φ ^ 2 ≠ 0 := by
  rw [neg_sq, Real.sin_sq_add_cos_sq]; exact one_ne_zero
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.circle_field_nonvanishing, dg-vf1-circle-field-nonvanishing-test]

/-- On the torus `∂_b` has length `R + r cos a > 0` when `0 < r < R`: a nowhere-vanishing field. -/
theorem torus_field_nonvanishing (R r a : ℝ) (hr : 0 < r) (hR : r < R) : 0 < R + r * Real.cos a := by
  have hc := Real.neg_one_le_cos a
  generalize Real.cos a = c at hc ⊢
  VERIPHY[dg-vf1-farkas]

/-! ## exm-vector-field-rn -/

/-- The radial field `x ∂_x + y ∂_y` vanishes at the origin. -/
theorem radPlane_origin : (fun k => radPlane k ![0, 0]) = 0 := by
  funext k; fin_cases k <;> simp [radPlane]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.radPlane_origin, dg-vf1-radplane-origin-test]

/-- The constant field `∂_x` has components `(1, 0)` at every point: its partial derivatives vanish. -/
theorem constant_field (i k : Fin 2) (x : Fin 2 → ℝ) : pd i (coordField 0 k) x = 0 := by
  simp [coordField, CoordinateGeometry.pd_const]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.constant_field, dg-vf1-constant-field-test]

/-! ## exr-hairy-ball -/

/-- The gradient field `−sin θ ∂_θ` also has ambient length `sin θ`. -/
theorem gradZ_ambient_length (x : Fin 2 → ℝ) :
    (∑ k, (∑ i, pd i (sphereEmbed k) x * gradZ i x) ^ 2) = Real.sin (x 0) ^ 2 := by
  have h := fun k => (sphere_jacobian k x).1
  simp only [Fin.sum_univ_two, gradZ, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_zero,
    add_zero, h, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons]
  have h1 := Real.sin_sq_add_cos_sq (x 1)
  have h2 := Real.sin_sq_add_cos_sq (x 0)
  nlinarith [h1, h2, sq_nonneg (Real.sin (x 0)), sq_nonneg (Real.cos (x 0))]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_ambient_length, dg-vf1-gradz-ambient-length-test]

/-- A combination `a ∂_φ + b sin θ ∂_θ` with constant `a, b` still vanishes at both poles (ambient vector zero). -/
theorem combination_vanishes_poles (a b φ : ℝ) (k : Fin 3) :
    (∑ i, pd i (sphereEmbed k) ![0, φ] * (a * dphi i ![0, φ] + b * Real.sin 0 * dtheta i ![0, φ])) = 0 ∧
      (∑ i, pd i (sphereEmbed k) ![Real.pi, φ] * (a * dphi i ![Real.pi, φ] + b * Real.sin Real.pi * dtheta i ![Real.pi, φ])) =
        0 := by
  have h1 := sphere_jacobian k ![0, φ]
  have h2 := sphere_jacobian k ![Real.pi, φ]
  constructor
  · simp only [Fin.sum_univ_two, dphi, dtheta, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, h1.1, h1.2]
    fin_cases k <;> simp
  · simp only [Fin.sum_univ_two, dphi, dtheta, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, h2.1, h2.2]
    fin_cases k <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.combination_vanishes_poles, dg-vf1-combination-vanishes-poles-test]

/-! ## exr-vf-rotation-fields -/

/-- The ambient vector of `K₁` is `e₁ × p = (0, −Z, Y)` with `p = E(θ, φ)`, away from the poles. -/
theorem killingX_ambient (k : Fin 3) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (∑ i, pd i (sphereEmbed k) x * killingX i x) =
      (Pi.single 0 1 ⨯₃ fun l => sphereEmbed l x) k := by
  have hj := sphere_jacobian k x
  simp only [Fin.sum_univ_two, killingX, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, hj.1, hj.2]
  fin_cases k <;> simp [crossProduct, sphereEmbed] <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.killingX_ambient, dg-vf1-killingx-ambient-test]

/-- The ambient vector of `K₂` is `e₂ × p = (Z, 0, −X)`. -/
theorem killingY_ambient (k : Fin 3) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (∑ i, pd i (sphereEmbed k) x * killingY i x) =
      (Pi.single 1 1 ⨯₃ fun l => sphereEmbed l x) k := by
  have hj := sphere_jacobian k x
  simp only [Fin.sum_univ_two, killingY, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, hj.1, hj.2]
  fin_cases k <;> simp [crossProduct, sphereEmbed] <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.killingY_ambient, dg-vf1-killingy-ambient-test]

/-- The ambient vector of `K₃ = ∂_φ` is `e₃ × p = (−Y, X, 0)`. -/
theorem killingZ_ambient (k : Fin 3) (x : Fin 2 → ℝ) :
    (∑ i, pd i (sphereEmbed k) x * killingZ i x) =
      (Pi.single 2 1 ⨯₃ fun l => sphereEmbed l x) k := by
  have hj := sphere_jacobian k x
  simp only [Fin.sum_univ_two, killingZ, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, hj.1, hj.2]
  fin_cases k <;> simp [crossProduct, sphereEmbed]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.killingZ_ambient, dg-vf1-killingz-ambient-test]

/-- `e_a × p` is orthogonal to `p`, hence tangent to the sphere. -/
theorem cross_tangent (e p : Fin 3 → ℝ) : (e ⨯₃ p) ⬝ᵥ p = 0 := by
  rw [dotProduct_comm]; exact dot_cross_self p e
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.cross_tangent, dg-vf1-cross-tangent-test]

/-- The squared length of `e × p` is `1 − (p · e)²` for unit `e` and `p`: it vanishes exactly at `p = ±e`. -/
theorem cross_length (e p : Fin 3 → ℝ) (he : e ⬝ᵥ e = 1) (hp : p ⬝ᵥ p = 1) :
    (e ⨯₃ p) ⬝ᵥ (e ⨯₃ p) = 1 - (p ⬝ᵥ e) ^ 2 := by
  rw [cross_dot_cross, he, hp, dotProduct_comm e p]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.cross_length, dg-vf1-cross-length-test]

/-! ## exr-vf-leibniz-field -/

/-- For the rotation field `X = −y ∂_x + x ∂_y`: `X(x² + y²) = 0`, `X(x) = −y`, `X(x³ + xy²) = −(x² + y²) y`. -/
theorem rotPlane_values (x : Fin 2 → ℝ) :
    fieldAct rotPlane (fun x => x 0 ^ 2 + x 1 ^ 2) x = 0 ∧ fieldAct rotPlane (fun x => x 0) x = -(x 1) ∧
      fieldAct rotPlane (fun x => x 0 ^ 3 + x 0 * x 1 ^ 2) x = -(x 0 ^ 2 + x 1 ^ 2) * x 1 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp (disch := fun_prop) [fieldAct, rotPlane, pd, Function.update_apply, Fin.sum_univ_succ] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.rotPlane_values, dg-vf1-rotplane-values-test]

/-- `X(fg) = f X(g)` since `X(f) = 0`: `−(x² + y²) y = (x² + y²)(−y)`. -/
theorem rotPlane_leibniz (x y : ℝ) : -(x ^ 2 + y ^ 2) * y = 0 * x + (x ^ 2 + y ^ 2) * (-y) := by
  VERIPHY[dg-vf1-poly]

/-! ## def-lie-bracket -/

/-- The bracket is antisymmetric: `[Y, X] = −[X, Y]`. -/
theorem lieBracket_antisymm {n : ℕ} (X Y : Fin n → (Fin n → ℝ) → ℝ) (k : Fin n) (x : Fin n → ℝ) :
    lieBracket Y X k x = -lieBracket X Y k x := by
  simp only [lieBracket, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun l _ => by ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.lieBracket_antisymm, dg-vf1-liebracket-antisymm-test]

/-- The chart formula agrees with the commutator of derivations: `[X, Y](f) = X(Y(f)) − Y(X(f))` (the second
derivatives cancel by the symmetry of mixed partials). -/
def BracketIsCommutator {n : ℕ} (X Y : Fin n → (Fin n → ℝ) → ℝ) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : Prop :=
  fieldAct X (fun y => fieldAct Y f y) x - fieldAct Y (fun y => fieldAct X f y) x = fieldAct (lieBracket X Y) f x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.BracketIsCommutator, dg-vf1-bracket-is-commutator-test]

/-- The Jacobi identity `[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0` for the chart formula. -/
def JacobiHolds {n : ℕ} (X Y Z : Fin n → (Fin n → ℝ) → ℝ) (k : Fin n) (x : Fin n → ℝ) : Prop :=
  lieBracket X (lieBracket Y Z) k x + lieBracket Y (lieBracket Z X) k x + lieBracket Z (lieBracket X Y) k x = 0
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.JacobiHolds, dg-vf1-jacobi-holds-test]

/-! ## exm-bracket-commuting -/

/-- `[∂_θ, ∂_φ] = 0`: coordinate fields commute. -/
theorem bracket_dtheta_dphi (k : Fin 2) (x : Fin 2 → ℝ) : lieBracket dtheta dphi k x = 0 := by
  fin_cases k <;> simp [lieBracket, dtheta, dphi, CoordinateGeometry.pd_const]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_dtheta_dphi, dg-vf1-bracket-dtheta-dphi-test]

/-- `[∂_φ, −sin θ ∂_θ] = 0`. -/
theorem bracket_rotation_gradient (k : Fin 2) (x : Fin 2 → ℝ) : lieBracket dphi gradZ k x = 0 :=
  bracket_dphi_gradZ k x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_rotation_gradient, dg-vf1-bracket-rotation-gradient-test]

/-! ## exm-bracket-nonzero -/

/-- `[∂_θ, sin θ ∂_φ] = cos θ ∂_φ`: component `0` is `0`, component `1` is `cos θ`. -/
theorem bracket_dtheta_sin_dphi (x : Fin 2 → ℝ) :
    lieBracket dtheta sinDphi 0 x = 0 ∧ lieBracket dtheta sinDphi 1 x = Real.cos (x 0) :=
  bracket_dtheta_sinDphi x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_dtheta_sin_dphi, dg-vf1-bracket-dtheta-sin-dphi-test]

/-! ## exm-bracket-rotations -/

/-- `[K₁, K₂] = −K₃`. -/
theorem bracket_K1_K2 (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingX killingY k x = -killingZ k x := bracket_killingX_killingY k x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_K1_K2, dg-vf1-bracket-k1-k2-test]

/-- `[K₂, K₃] = −K₁`. -/
theorem bracket_K2_K3 (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingY killingZ k x = -killingX k x := bracket_killingY_killingZ k x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_K2_K3, dg-vf1-bracket-k2-k3-test]

/-- `[K₃, K₁] = −K₂`. -/
theorem bracket_K3_K1 (k : Fin 2) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket killingZ killingX k x = -killingY k x := bracket_killingZ_killingX k x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_K3_K1, dg-vf1-bracket-k3-k1-test]

/-! ## exr-vf-function-rescaling -/

/-- `[fX, gY] = fg [X, Y] + f X(g) Y − g Y(f) X` for differentiable `f, g` and fields. -/
theorem bracket_rescaled {n : ℕ} (f g : (Fin n → ℝ) → ℝ) (X Y : Fin n → (Fin n → ℝ) → ℝ)
    (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) (hX : ∀ k, Differentiable ℝ (X k))
    (hY : ∀ k, Differentiable ℝ (Y k)) (k : Fin n) (x : Fin n → ℝ) :
    lieBracket (fun k y => f y * X k y) (fun k y => g y * Y k y) k x =
      f x * g x * lieBracket X Y k x + f x * fieldAct X g x * Y k x - g x * fieldAct Y f x * X k x := by
  have e1 : ∀ l, pd l (fun y => g y * Y k y) x = pd l g x * Y k x + g x * pd l (Y k) x := fun l =>
    pd_mul g (Y k) hg (hY k) l x
  have e2 : ∀ l, pd l (fun y => f y * X k y) x = pd l f x * X k x + f x * pd l (X k) x := fun l =>
    pd_mul f (X k) hf (hX k) l x
  simp only [lieBracket, fieldAct, e1, e2, Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun l _ => by ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_rescaled, dg-vf1-bracket-rescaled-test]

/-! ## exr-vf-so3-brackets -/

/-- The Jacobi identity for the three rotation fields, away from the poles. -/
def So3Jacobi (k : Fin 2) (x : Fin 2 → ℝ) : Prop :=
  Real.sin (x 0) ≠ 0 → JacobiHolds killingX killingY killingZ k x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.So3Jacobi, dg-vf1-so3-jacobi-test]

/-! ## exm-integral-parallel -/

/-- The parallels `t ↦ (θ₀, φ₀ + t)` are the integral curves of `∂_φ`: `θ̇ = 0`, `φ̇ = 1`. -/
theorem parallel_integral_curve (θ₀ φ₀ : ℝ) : IsIntegralCurve dphi (fun t => ![θ₀, φ₀ + t]) := by
  intro t i
  fin_cases i <;> simp [dphi]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.parallel_integral_curve, dg-vf1-parallel-integral-curve-test]

/-! ## exm-integral-gradient -/

/-- `θ(t) = 2 arctan(e^{−t})` solves `θ̇ = −sin θ`. -/
theorem gradient_curve_solves (t : ℝ) :
    deriv (fun t => 2 * Real.arctan (Real.exp (-t))) t = -Real.sin (2 * Real.arctan (Real.exp (-t))) := by
  have hd : HasDerivAt (fun t => 2 * Real.arctan (Real.exp (-t)))
      (2 * (1 / (1 + Real.exp (-t) ^ 2) * (Real.exp (-t) * -1))) t := by
    have h1 : HasDerivAt (fun t : ℝ => Real.exp (-t)) (Real.exp (-t) * -1) t := by
      have := (hasDerivAt_neg t).exp
      simpa using this
    exact (h1.arctan).const_mul 2
  rw [hd.deriv, Real.sin_two_mul, Real.sin_arctan, Real.cos_arctan]
  have hpos : 0 < 1 + Real.exp (-t) ^ 2 := by positivity
  have hs : Real.sqrt (1 + Real.exp (-t) ^ 2) ^ 2 = 1 + Real.exp (-t) ^ 2 := Real.sq_sqrt hpos.le
  have hsp : Real.sqrt (1 + Real.exp (-t) ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hpos).ne'
  field_simp
  linear_combination (2 * Real.exp (-t)) * hs
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradient_curve_solves, dg-vf1-gradient-curve-solves-test]

/-- The curve approaches the north pole: `2 arctan(e^{−t}) → 0` as `t → ∞`. -/
theorem gradient_curve_limit :
    Filter.Tendsto (fun t => 2 * Real.arctan (Real.exp (-t))) Filter.atTop (nhds 0) := by
  have h : Filter.Tendsto (fun t : ℝ => Real.arctan (Real.exp (-t))) Filter.atTop (nhds (Real.arctan 0)) :=
    (Real.continuous_arctan.tendsto 0).comp Real.tendsto_exp_neg_atTop_nhds_zero
  rw [Real.arctan_zero] at h
  simpa using h.const_mul 2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradient_curve_limit, dg-vf1-gradient-curve-limit-test]

/-- The pole is never reached: `0 < 2 arctan(e^{−t})` for every `t`. -/
theorem gradient_curve_positive (t : ℝ) : 0 < 2 * Real.arctan (Real.exp (-t)) := by
  have := Real.arctan_pos.mpr (Real.exp_pos (-t))
  linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradient_curve_positive, dg-vf1-gradient-curve-positive-test]

/-- The pole is a zero of the field, a fixed point of the flow: `gradZ` vanishes at `θ = 0`. -/
theorem gradZ_fixed_point (φ : ℝ) (k : Fin 2) : gradZ k ![0, φ] = 0 := by
  fin_cases k <;> simp [gradZ]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_fixed_point, dg-vf1-gradz-fixed-point-test]

/-! ## exm-integral-plane -/

/-- The integral curves of `∂_x` are the horizontal lines `t ↦ (x₀ + t, y₀)`. -/
theorem horizontal_integral_curve (x₀ y₀ : ℝ) : IsIntegralCurve (coordField 0) (fun t => ![x₀ + t, y₀]) := by
  intro t i
  fin_cases i <;> simp [coordField]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.horizontal_integral_curve, dg-vf1-horizontal-integral-curve-test]

/-- The rotation flow solves `ẋ = −y`, `ẏ = x`: circles about the origin at angular speed `1`. -/
theorem rotFlow_ode (x y t : ℝ) :
    deriv (fun t => x * Real.cos t - y * Real.sin t) t = -(x * Real.sin t + y * Real.cos t) ∧
      deriv (fun t => x * Real.sin t + y * Real.cos t) t = x * Real.cos t - y * Real.sin t := by
  constructor
  · VERIPHY[dg-vf1-deriv]
  · VERIPHY[dg-vf1-deriv]

/-- The rays `t ↦ e^t p` solve `ẋ = x`: the integral curves of the radial field. -/
theorem radFlow_ode (p t : ℝ) : deriv (fun t => Real.exp t * p) t = Real.exp t * p := by
  VERIPHY[dg-vf1-deriv]

/-! ## exr-integral-curves -/

/-- `∫ dθ / sin θ = log tan(θ/2)`: the derivative of `log tan(θ/2)` is `1/sin θ` for `0 < θ < π`. -/
theorem log_tan_half_deriv (θ : ℝ) (h0 : 0 < θ) (hπ : θ < Real.pi) :
    deriv (fun θ => Real.log (Real.tan (θ / 2))) θ = 1 / Real.sin θ := by
  have hc : 0 < Real.cos (θ / 2) := Real.cos_pos_of_mem_Ioo ⟨by linarith, by linarith⟩
  have hs : 0 < Real.sin (θ / 2) := Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)
  have ht : 0 < Real.tan (θ / 2) := by rw [Real.tan_eq_sin_div_cos]; positivity
  have h1 : HasDerivAt (fun θ => Real.tan (θ / 2)) (1 / Real.cos (θ / 2) ^ 2 * (1 / 2)) θ :=
    (Real.hasDerivAt_tan hc.ne').comp θ ((hasDerivAt_id θ).div_const 2)
  rw [(h1.log ht.ne').deriv, Real.tan_eq_sin_div_cos]
  have hsin : Real.sin θ = 2 * Real.sin (θ / 2) * Real.cos (θ / 2) := by
    rw [← Real.sin_two_mul]; ring_nf
  rw [hsin]
  field_simp
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.log_tan_half_deriv, dg-vf1-log-tan-half-deriv-test]

/-- `tan(θ(t)/2) = e^{−t}` for `θ(t) = 2 arctan(e^{−t})`. -/
theorem gradient_curve_tan (t : ℝ) : Real.tan (2 * Real.arctan (Real.exp (-t)) / 2) = Real.exp (-t) := by
  VERIPHY[dg-vf1-values]

/-- The curve starts on the equator: `θ(0) = 2 arctan(e⁰) = 2 arctan 1 = π/2`. -/
theorem gradient_curve_start : 2 * Real.arctan (Real.exp (-0)) = Real.pi / 2 := by
  VERIPHY[dg-vf1-values]

/-! ## exr-vf-flows-plane -/

/-- The dilation flow is a one-parameter group: `e^t e^s = e^{t + s}`. -/
theorem radFlow_group (t s : ℝ) : Real.exp t * Real.exp s = Real.exp (t + s) := by
  VERIPHY[dg-vf1-exp]

/-- `φ₀ = id` for the dilation flow: `e⁰ p = p`. -/
theorem radFlow_zero (p : ℝ) : Real.exp 0 * p = p := by
  VERIPHY[dg-vf1-values]

/-- From `ẋ = −y`, `ẏ = x`: `ẍ = −x` along the rotation flow. -/
theorem rotFlow_second_order (x y t : ℝ) :
    deriv (fun t => -(x * Real.sin t + y * Real.cos t)) t = -(x * Real.cos t - y * Real.sin t) := by
  VERIPHY[dg-vf1-deriv]

/-- Rotations compose by adding angles: `R(t) R(s) = R(t + s)` (the addition formulas). -/
theorem rot2_group (t s : ℝ) : rot2 t * rot2 s = rot2 (t + s) := by
  unfold rot2
  rw [Real.cos_add, Real.sin_add]
  VERIPHY[dg-vf1-matrix]

/-- `R(0) = 1`: the rotation flow starts at the identity. -/
theorem rot2_zero : rot2 0 = 1 := by
  unfold rot2
  rw [Real.cos_zero, Real.sin_zero]
  VERIPHY[dg-vf1-matrix]

/-! ## exr-vf-integral-k1 -/

/-- The curve `θ(t) = arccos(√3/2 sin t)`, `φ(t) = arctan(√3 cos t)` starts at `(π/2, π/3)`. -/
theorem k1_curve_start :
    Real.arccos (Real.sqrt 3 / 2 * Real.sin 0) = Real.pi / 2 ∧ Real.arctan (Real.sqrt 3 * Real.cos 0) = Real.pi / 3 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- The curve solves the equations of `K₁`: `θ̇ = −sin φ` and `φ̇ = −cot θ cos φ`; it is the rotation about the
`X`-axis, `E(θ(t), φ(t)) = (1/2, (√3/2) cos t, (√3/2) sin t)`. -/
def K1IntegralCurve (t : ℝ) : Prop :=
  deriv (fun t => Real.arccos (Real.sqrt 3 / 2 * Real.sin t)) t = -Real.sin (Real.arctan (Real.sqrt 3 * Real.cos t)) ∧
    deriv (fun t => Real.arctan (Real.sqrt 3 * Real.cos t)) t =
      -(Real.cos (Real.arccos (Real.sqrt 3 / 2 * Real.sin t)) / Real.sin (Real.arccos (Real.sqrt 3 / 2 * Real.sin t))) *
        Real.cos (Real.arctan (Real.sqrt 3 * Real.cos t)) ∧
    (fun k => sphereEmbed k ![Real.arccos (Real.sqrt 3 / 2 * Real.sin t), Real.arctan (Real.sqrt 3 * Real.cos t)]) =
      ![1 / 2, Real.sqrt 3 / 2 * Real.cos t, Real.sqrt 3 / 2 * Real.sin t]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.K1IntegralCurve, dg-vf1-k1-integral-curve-test]

/-! ## def-frame -/

/-- The coordinate basis is a frame: at every point the `n` coordinate fields are a basis of `T_pM`. -/
theorem coordinate_frame (n : ℕ) (x : Fin n → ℝ) :
    LinearIndependent ℝ (fun i : Fin n => fun k => coordField i k x) := by
  have : (fun i : Fin n => fun k => coordField i k x) = fun i => (Pi.single i (1 : ℝ) : Fin n → ℝ) := by
    funext i k; simp [coordField, Pi.single_apply]
  rw [this]
  exact (Pi.basisFun ℝ (Fin n)).linearIndependent
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.coordinate_frame, dg-vf1-coordinate-frame-test]

/-! ## exm-frame-orthonormal -/

/-- The frame `e₁ = ∂_θ`, `e₂ = ∂_φ / sin θ` is orthonormal for the round metric (the ambient Gram matrix is the
identity), away from the poles. -/
theorem sphere_frame_orthonormal (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (∑ i, ∑ j, dtheta i x * sphere i j x * dtheta j x) = 1 ∧
      (∑ i, ∑ j, e2Sphere i x * sphere i j x * e2Sphere j x) = 1 ∧
      (∑ i, ∑ j, dtheta i x * sphere i j x * e2Sphere j x) = 0 := by
  simp [Fin.sum_univ_two, dtheta, e2Sphere, sphere]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.sphere_frame_orthonormal, dg-vf1-sphere-frame-orthonormal-test]

/-- `[e₁, e₂] = −cos θ / sin² θ ∂_φ = −cot θ e₂ ≠ 0` (for `cos θ ≠ 0`): the frame is not a coordinate basis. -/
theorem sphere_frame_bracket (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    lieBracket dtheta e2Sphere 0 x = 0 ∧
      lieBracket dtheta e2Sphere 1 x = -(Real.cos (x 0) / Real.sin (x 0)) * e2Sphere 1 x := by
  have hd : HasDerivAt (fun t => 1 / Real.sin t) (-(Real.cos (x 0)) / Real.sin (x 0) ^ 2) (x 0) := by
    have := ((Real.hasDerivAt_sin (x 0)).inv h)
    simpa [one_div] using this
  constructor
  · simp [lieBracket, dtheta, e2Sphere, CoordinateGeometry.pd_const]
  · simp only [lieBracket, Fin.sum_univ_two, dtheta, e2Sphere, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, one_mul, zero_mul, sub_zero, mul_zero, add_zero, pd, Function.update_apply, Fin.isValue,
      Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ↓reduceIte, Fin.zero_eq_one_iff, OfNat.one_ne_ofNat, zero_sub]
    rw [hd.deriv]
    have : deriv (fun _ : ℝ => (1 : ℝ)) (x 1) = 0 := deriv_const _ _
    simp only [this, mul_zero, sub_zero]
    field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.sphere_frame_bracket, dg-vf1-sphere-frame-bracket-test]

/-! ## exm-frame-polar -/

/-- `∂_r`, `r⁻¹ ∂_φ` is an orthonormal frame for the polar metric on the punctured plane. -/
theorem polar_frame_orthonormal (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    (∑ i, ∑ j, dtheta i x * polar i j x * dtheta j x) = 1 ∧
      (∑ i, ∑ j, e2Polar i x * polar i j x * e2Polar j x) = 1 ∧
      (∑ i, ∑ j, dtheta i x * polar i j x * e2Polar j x) = 0 := by
  simp [Fin.sum_univ_two, dtheta, e2Polar, polar]
  field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.polar_frame_orthonormal, dg-vf1-polar-frame-orthonormal-test]

/-- The matrix relating the polar frame to the Cartesian frame is the rotation by `φ`: the Cartesian components of
`∂_r` and `r⁻¹ ∂_φ` are the columns of `rot2 φ`. -/
theorem polar_frame_rotation (x : Fin 2 → ℝ) (h : x 0 ≠ 0) (k a : Fin 2) :
    (∑ i, pd i (polarToCart k) x * (![dtheta, e2Polar] : Fin 2 → Fin 2 → (Fin 2 → ℝ) → ℝ) a i x) = rot2 (x 1) k a := by
  have hj := polar_jacobian k x
  fin_cases a <;> fin_cases k <;>
    simp [Fin.sum_univ_two, hj.1, hj.2, dtheta, e2Polar, rot2] <;> field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.polar_frame_rotation, dg-vf1-polar-frame-rotation-test]

/-! ## exm-frame-group -/

/-- The generators `T₀, T₁, T₂` are linearly independent over `ℝ` (a lemma: the coefficients are read off the entries). -/
theorem T_independent : LinearIndependent ℝ ![T 0, T 1, T 2] := by
  rw [Fintype.linearIndependent_iff]
  intro g hg a
  have h := su2_coordinates (g 0) (g 1) (g 2)
  have e : g 0 • T 0 + g 1 • T 1 + g 2 • T 2 = 0 := by
    simpa [Fin.sum_univ_three] using hg
  rw [e] at h
  simp at h
  fin_cases a <;> simp <;> linarith [h.1, h.2.1, h.2.2]

/-- Left translation by an invertible `g` carries the basis of `T_e SU(2)` to a frame: the fields `g T_a` are
independent at every `g`, so the group carries a frame defined everywhere (and nowhere-vanishing fields). -/
theorem left_invariant_frame (g : Matrix (Fin 2) (Fin 2) ℂ) (hg : IsUnit g) :
    LinearIndependent ℝ (fun a : Fin 3 => g * (![T 0, T 1, T 2] : Fin 3 → Matrix (Fin 2) (Fin 2) ℂ) a) := by
  have hinj : Function.Injective (fun M : Matrix (Fin 2) (Fin 2) ℂ => g * M) := by
    intro M N hMN
    have := congrArg (fun P => hg.unit⁻¹.val * P) hMN
    simpa [← Matrix.mul_assoc, Units.inv_mul] using this
  have hlin : IsLinearMap ℝ (fun M : Matrix (Fin 2) (Fin 2) ℂ => g * M) :=
    ⟨fun M N => Matrix.mul_add g M N, fun c M => by rw [Matrix.mul_smul]⟩
  exact T_independent.map' (IsLinearMap.mk' _ hlin) (LinearMap.ker_eq_bot.mpr hinj)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.left_invariant_frame, dg-vf1-left-invariant-frame-test]

/-! ## exr-vf-frame-components -/

/-- `∂_φ = sin θ e₂`: the component `1 · sin θ / sin θ = 1` when `sin θ ≠ 0`. -/
theorem dphi_in_frame (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) (k : Fin 2) :
    dphi k x = Real.sin (x 0) * e2Sphere k x := by
  fin_cases k <;> simp [dphi, e2Sphere] ; field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.dphi_in_frame, dg-vf1-dphi-in-frame-test]

/-- `X = ∂_θ + 2 ∂_φ = e₁ + 2 sin θ e₂`: frame components `(1, 2 sin θ)`. -/
theorem X_frame_components (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) (k : Fin 2) :
    (![1, 2] : Fin 2 → ℝ) k = 1 * dtheta k x + 2 * Real.sin (x 0) * e2Sphere k x := by
  fin_cases k <;> simp [dtheta, e2Sphere] ; field_simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.X_frame_components, dg-vf1-x-frame-components-test]

/-- The squared length of `X = ∂_θ + 2 ∂_φ` in the coordinate basis is `1 + 4 sin² θ`. -/
theorem X_length_coordinates (x : Fin 2 → ℝ) :
    (∑ i, ∑ j, (![1, 2] : Fin 2 → ℝ) i * sphere i j x * (![1, 2] : Fin 2 → ℝ) j) = 1 + 4 * Real.sin (x 0) ^ 2 := by
  simp [Fin.sum_univ_two, sphere]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.X_length_coordinates, dg-vf1-x-length-coordinates-test]

/-- In the orthonormal frame the same number is `1² + (2 sin θ)² = 1 + 4 sin² θ`. -/
theorem X_length_frame (s : ℝ) : 1 ^ 2 + (2 * s) ^ 2 = 1 + 4 * s ^ 2 := by
  VERIPHY[dg-vf1-poly]

/-! ## exr-vf-frame-noncoordinate -/

/-- The fields of a coordinate frame commute: `[∂_a, ∂_b] = 0`. -/
theorem coordinate_fields_commute {n : ℕ} (a b k : Fin n) (x : Fin n → ℝ) :
    lieBracket (coordField a) (coordField b) k x = 0 := by
  simp [lieBracket, coordField, CoordinateGeometry.pd_const]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.coordinate_fields_commute, dg-vf1-coordinate-fields-commute-test]

/-- `[∂_r, r⁻¹ ∂_φ] = −r⁻² ∂_φ`, non-zero: the orthonormal polar frame is not a coordinate frame. -/
theorem polar_frame_bracket (x : Fin 2 → ℝ) (h : x 0 ≠ 0) :
    lieBracket dtheta e2Polar 0 x = 0 ∧ lieBracket dtheta e2Polar 1 x = -(1 / x 0 ^ 2) := by
  have hd : HasDerivAt (fun t : ℝ => 1 / t) (-(1 / x 0 ^ 2)) (x 0) := by
    have := hasDerivAt_inv h
    simpa [one_div] using this
  constructor
  · simp [lieBracket, dtheta, e2Polar, CoordinateGeometry.pd_const]
  · simp only [lieBracket, Fin.sum_univ_two, dtheta, e2Polar, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, one_mul, zero_mul, sub_zero, mul_zero, add_zero, pd, Function.update_apply, Fin.isValue,
      Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ↓reduceIte, Fin.zero_eq_one_iff, OfNat.one_ne_ofNat, zero_sub]
    rw [hd.deriv]
    have : deriv (fun _ : ℝ => (1 : ℝ)) (x 1) = 0 := deriv_const _ _
    simp [this]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.polar_frame_bracket, dg-vf1-polar-frame-bracket-test]

/-- `−r⁻² ∂_φ = −r⁻¹ (r⁻¹ ∂_φ)`. -/
theorem polar_frame_bracket_form (r : ℝ) (h : r ≠ 0) : -(1 / r ^ 2) = -(1 / r) * (1 / r) := by
  field_simp

/-! ## exm-vf-brackets-point -/

/-- The bracket `cos θ ∂_φ` at a few latitudes: `cos(π/3) = 1/2`, `cos(π/2) = 0`, `cos(2π/3) = −1/2`. -/
theorem bracket_values :
    Real.cos (Real.pi / 3) = 1 / 2 ∧ Real.cos (Real.pi / 2) = 0 ∧ Real.cos (2 * Real.pi / 3) = -1 / 2 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- The ambient length of `cos θ ∂_φ` is `|cos θ| sin θ`: its square is `cos² θ sin² θ`. -/
theorem bracket_ambient_length (x : Fin 2 → ℝ) :
    (∑ k, (∑ i, pd i (sphereEmbed k) x * (Real.cos (x 0) * dphi i x)) ^ 2) = Real.cos (x 0) ^ 2 * Real.sin (x 0) ^ 2 := by
  have h := fun k => (sphere_jacobian k x).2
  simp only [Fin.sum_univ_two, dphi, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, mul_zero, mul_one,
    zero_add, h, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons]
  nlinarith [Real.sin_sq_add_cos_sq (x 1), sq_nonneg (Real.cos (x 0) * Real.sin (x 0))]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_ambient_length, dg-vf1-bracket-ambient-length-test]

/-- `√(cos² θ sin² θ) = |cos θ| sin θ` for `0 ≤ sin θ`. -/
theorem bracket_length_sqrt (c s : ℝ) (hs : 0 ≤ s) : Real.sqrt (c ^ 2 * s ^ 2) = |c| * s := by
  rw [show c ^ 2 * s ^ 2 = (|c| * s) ^ 2 by rw [mul_pow, sq_abs], Real.sqrt_sq (by positivity)]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_length_sqrt, dg-vf1-bracket-length-sqrt-test]

/-- On the equator `sin θ` is stationary: `d/dθ sin θ = cos(π/2) = 0`. -/
theorem sin_stationary_equator : deriv Real.sin (Real.pi / 2) = 0 := by
  VERIPHY[dg-vf1-deriv]

/-! ## exm-vf-brackets-lean -/

/-- The components of `−sin θ ∂_θ` do not depend on `φ`: `∂_φ(−sin θ) = 0` and `∂_φ 0 = 0`. -/
theorem gradZ_phi_independent (x : Fin 2 → ℝ) : pd 1 (gradZ 0) x = 0 ∧ pd 1 (gradZ 1) x = 0 := by
  constructor <;> simp [gradZ, pd, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_phi_independent, dg-vf1-gradz-phi-independent-test]

/-- `∂_θ (0, sin θ) = (0, cos θ)`: the derivative of the non-constant field along the constant one. -/
theorem sinDphi_theta_derivative (x : Fin 2 → ℝ) :
    pd 0 (sinDphi 0) x = 0 ∧ pd 0 (sinDphi 1) x = Real.cos (x 0) := by
  constructor <;> simp [sinDphi, pd, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.sinDphi_theta_derivative, dg-vf1-sindphi-theta-derivative-test]

/-! ## exm-vf-brackets-symmetry -/

/-- The height is invariant under the rotation about the `Z`-axis: `Z(θ, φ + t) = Z(θ, φ)`. -/
theorem height_rotation_invariant (t : ℝ) (x : Fin 2 → ℝ) : heightZ (rotSphereFlow t x) = heightZ x := by
  simp [heightZ, rotSphereFlow]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.height_rotation_invariant, dg-vf1-height-rotation-invariant-test]

/-- The gradient field is invariant under the rotation: its components at `(θ, φ + t)` equal those at `(θ, φ)`. -/
theorem gradZ_rotation_invariant (t : ℝ) (x : Fin 2 → ℝ) (k : Fin 2) : gradZ k (rotSphereFlow t x) = gradZ k x := by
  fin_cases k <;> simp [gradZ, rotSphereFlow]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.gradZ_rotation_invariant, dg-vf1-gradz-rotation-invariant-test]

/-- The flows of `∂_φ` and of the gradient field commute: the first moves `φ`, the second `θ`. -/
theorem flows_commute (s t : ℝ) (x : Fin 2 → ℝ) : gradFlow t (rotSphereFlow s x) = rotSphereFlow s (gradFlow t x) := by
  funext k; fin_cases k <;> simp [gradFlow, rotSphereFlow]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.flows_commute, dg-vf1-flows-commute-test]

/-! ## exr-vf-brackets-general -/

/-- `[∂_φ, f(θ) ∂_θ] = 0` for any differentiable `f`. -/
theorem bracket_dphi_f_dtheta (f : ℝ → ℝ) (hf : Differentiable ℝ f) (k : Fin 2) (x : Fin 2 → ℝ) :
    lieBracket dphi ![fun x => f (x 0), fun _ => 0] k x = 0 := by
  fin_cases k <;> simp [lieBracket, dphi, pd, Function.update_apply, Fin.sum_univ_two]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_dphi_f_dtheta, dg-vf1-bracket-dphi-f-dtheta-test]

/-- `[∂_θ, f(θ) ∂_φ] = f'(θ) ∂_φ` for any differentiable `f`. -/
theorem bracket_dtheta_f_dphi (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x : Fin 2 → ℝ) :
    lieBracket dtheta ![fun _ => 0, fun x => f (x 0)] 0 x = 0 ∧
      lieBracket dtheta ![fun _ => 0, fun x => f (x 0)] 1 x = deriv f (x 0) := by
  constructor <;> simp [lieBracket, dtheta, pd, Function.update_apply, Fin.sum_univ_two]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_dtheta_f_dphi, dg-vf1-bracket-dtheta-f-dphi-test]

/-! ## exr-vf-brackets-stereographic -/

/-- In the stereographic chart `[(−v, u), (u, v)] = 0`: the rotation and radial fields of the plane commute. -/
theorem bracket_rotPlane_radPlane (k : Fin 2) (x : Fin 2 → ℝ) : lieBracket rotPlane radPlane k x = 0 := by
  fin_cases k <;> simp [lieBracket, rotPlane, radPlane, pd, Function.update_apply, Fin.sum_univ_two]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.bracket_rotPlane_radPlane, dg-vf1-bracket-rotplane-radplane-test]

/-! ## exr-vf-brackets-frame -/

/-- `[∂_θ, h ∂_φ] = h' ∂_φ = (h'/h) e₂` with `e₂ = h ∂_φ`, for `h ≠ 0`: `h' = (h'/h) h`. -/
theorem rescaled_frame_bracket (h hd : ℝ) (hh : h ≠ 0) : hd = hd / h * h := by
  field_simp

/-- For `h = 1/sin θ`: `h' = −cos θ / sin² θ`. -/
theorem inv_sin_deriv (θ : ℝ) (h : Real.sin θ ≠ 0) :
    deriv (fun θ => 1 / Real.sin θ) θ = -Real.cos θ / Real.sin θ ^ 2 := by
  VERIPHY[dg-vf1-deriv]

/-- `h'/h = (−cos θ / sin² θ)/(1/sin θ) = −cot θ`. -/
theorem inv_sin_ratio (s c : ℝ) (hs : s ≠ 0) : -c / s ^ 2 / (1 / s) = -(c / s) := by
  field_simp
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- The fields commute iff `h' = 0`; a differentiable `h` with `h' = 0` everywhere is constant. -/
theorem commuting_rescaling_constant (h : ℝ → ℝ) (hh : Differentiable ℝ h) (hd : ∀ t, deriv h t = 0) (a b : ℝ) :
    h a = h b := by
  wlog hab : a ≤ b generalizing a b
  · exact (this b a (le_of_not_ge hab)).symm
  have key := constant_of_derivWithin_zero (f := h) (a := a) (b := b) (hh.differentiableOn) fun t ht => by
    rw [(hh t).derivWithin (uniqueDiffWithinAt_Icc (lt_of_le_of_lt ht.1 ht.2) |>.mono Set.Ico_subset_Icc_self)]
    exact hd t
  exact (key b ⟨hab, le_rfl⟩).symm
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.VectorFields.commuting_rescaling_constant, dg-vf1-commuting-rescaling-constant-test]

end KaluzaKlein.DifferentialGeometry.VectorFields

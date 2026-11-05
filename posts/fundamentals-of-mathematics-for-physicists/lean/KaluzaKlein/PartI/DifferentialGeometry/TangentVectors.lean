import Mathlib
import KaluzaKlein.Geometry.VectorFields
import KaluzaKlein.Geometry.MetricExamples
import KaluzaKlein.Geometry.LieGroups
import KaluzaKlein.Geometry.Charts

/-!
# Differential geometry, section "Tangent vectors"

The claims of the section "Tangent vectors" of `tutorial/part-i-09-differential-geometry.qmd`
(`## Vector fields, forms and tensors`), in coordinates (design D28):

* a chart point is `x : Fin n → ℝ`; a tangent vector at `x` is its component vector `X : Fin n → ℝ`, acting on a
  function `f : (Fin n → ℝ) → ℝ` by `act X f x = Σ_i X^i ∂_i f(x)`, with the asset's partial derivative `pd`;
* the coordinate basis vector `∂_i` is `Pi.single i 1`, the coordinate function `x^j` is `coord j`;
* a curve is `γ : ℝ → Fin n → ℝ`, its velocity the vector of derivatives of its coordinates;
* the running surfaces are explicit embeddings `Fin 3 → (Fin 2 → ℝ) → ℝ` (the spherical chart `sphereEmbed` of
  `KaluzaKlein.Geometry`, the stereographic chart `stereoEmbed`, the torus `torusEmbed`), the plane's polar chart
  is `polarToCart`, and `T_e SU(2)` is modelled by the generators `T a` of `KaluzaKlein.Conventions`.

General statements (linearity, the Leibniz rule, the chain rule, the transformation law) are theorems about
differentiable functions, tested at explicit instances; explicit values are certified.
-/

open CoordinateGeometry KaluzaKlein.Geometry Real

namespace KaluzaKlein.DifferentialGeometry.TangentVectors

/-- A tangent vector with components `X` at the chart point `x`, acting on `f`: `X(f) = Σ_i X^i ∂_i f(x)`. -/
noncomputable def act {n : ℕ} (X : Fin n → ℝ) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : ℝ := ∑ i, X i * pd i f x

/-- The coordinate function `x^j`. -/
def coord {n : ℕ} (j : Fin n) : (Fin n → ℝ) → ℝ := fun x => x j

/-- The polar chart of the plane, `(r, φ) ↦ (r cos φ, r sin φ)`. -/
noncomputable def polarToCart : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => x 0 * Real.cos (x 1), fun x => x 0 * Real.sin (x 1)]

/-- The inverse of the north stereographic chart, `(u, v) ↦ (2u, 2v, u² + v² − 1)/(1 + u² + v²)`. -/
noncomputable def stereoEmbed : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => 2 * x 0 / (1 + x 0 ^ 2 + x 1 ^ 2), fun x => 2 * x 1 / (1 + x 0 ^ 2 + x 1 ^ 2),
    fun x => (x 0 ^ 2 + x 1 ^ 2 - 1) / (1 + x 0 ^ 2 + x 1 ^ 2)]

/-- The torus of radii `R > r`, `(a, b) ↦ ((R + r cos a) cos b, (R + r cos a) sin b, r sin a)`. -/
noncomputable def torusEmbed (R r : ℝ) : Fin 3 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => (R + r * Real.cos (x 0)) * Real.cos (x 1), fun x => (R + r * Real.cos (x 0)) * Real.sin (x 1),
    fun x => r * Real.sin (x 0)]

/-- The south stereographic coordinate `u' = sin θ cos φ / (1 + cos θ)`. -/
noncomputable def stereoSU (x : Fin 2 → ℝ) : ℝ := Real.sin (x 0) * Real.cos (x 1) / (1 + Real.cos (x 0))

/-- The south stereographic coordinate `v' = sin θ sin φ / (1 + cos θ)`. -/
noncomputable def stereoSV (x : Fin 2 → ℝ) : ℝ := Real.sin (x 0) * Real.sin (x 1) / (1 + Real.cos (x 0))

/-- The rotation field of the plane `−y ∂_x + x ∂_y`, components `(−y, x)`. -/
def rotPlane : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => -(x 1), fun x => x 0]

/-- The radial field of the plane `x ∂_x + y ∂_y`, components `(x, y)`. -/
def radPlane : Fin 2 → (Fin 2 → ℝ) → ℝ := ![fun x => x 0, fun x => x 1]

/-- The rotation of the plane by the angle `t`. -/
noncomputable def rot2 (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos t, -Real.sin t; Real.sin t, Real.cos t]

/-- The generator of `SO(2)`, the velocity of `rot2` at `t = 0`. -/
def J2 : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The stereographic components of `K₁`: `(uv, (1 − u² + v²)/2)`. -/
noncomputable def k1Stereo : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => x 0 * x 1, fun x => (1 - x 0 ^ 2 + x 1 ^ 2) / 2]

/-- The stereographic components of `K₂`: `(−(1 + u² − v²)/2, −uv)`. -/
noncomputable def k2Stereo : Fin 2 → (Fin 2 → ℝ) → ℝ :=
  ![fun x => -(1 + x 0 ^ 2 - x 1 ^ 2) / 2, fun x => -(x 0 * x 1)]

/-! ### Partial derivatives of differentiable functions (lemmas) -/

/-- The derivative of `t ↦ f (update x i t)` at `x i` is the Fréchet derivative applied to `e_i`. -/
theorem hasDerivAt_update_comp {n : ℕ} (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (i : Fin n)
    (x : Fin n → ℝ) :
    HasDerivAt (fun t => f (Function.update x i t)) (fderiv ℝ f x (Pi.single i 1)) (x i) := by
  have h1 : HasDerivAt (Function.update x i) (Pi.single i (1 : ℝ)) (x i) := hasDerivAt_update x i (x i)
  have h2 : HasFDerivAt f (fderiv ℝ f x) (Function.update x i (x i)) := by
    rw [Function.update_eq_self]; exact (hf x).hasFDerivAt
  exact h2.comp_hasDerivAt (x i) h1

/-- `∂_i f(x) = Df(x) e_i` for a differentiable `f`. -/
theorem pd_eq_fderiv {n : ℕ} (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (i : Fin n) (x : Fin n → ℝ) :
    pd i f x = fderiv ℝ f x (Pi.single i 1) :=
  (hasDerivAt_update_comp f hf i x).deriv

theorem pd_add {n : ℕ} (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) (i : Fin n)
    (x : Fin n → ℝ) : pd i (f + g) x = pd i f x + pd i g x := by
  have hF := hasDerivAt_update_comp f hf i x
  have hG := hasDerivAt_update_comp g hg i x
  have e := (hF.add hG).deriv
  unfold pd
  rw [hF.deriv, hG.deriv, ← e]
  rfl

theorem pd_const_mul {n : ℕ} (c : ℝ) (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (i : Fin n)
    (x : Fin n → ℝ) : pd i (fun y => c * f y) x = c * pd i f x := by
  have hF := hasDerivAt_update_comp f hf i x
  have e := (hF.const_mul c).deriv
  unfold pd
  rw [hF.deriv, ← e]

/-- The Leibniz rule for a partial derivative: `∂_i (fg) = (∂_i f) g + f (∂_i g)`. -/
theorem pd_mul {n : ℕ} (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) (i : Fin n)
    (x : Fin n → ℝ) : pd i (f * g) x = pd i f x * g x + f x * pd i g x := by
  have hF := hasDerivAt_update_comp f hf i x
  have hG := hasDerivAt_update_comp g hg i x
  have e := (hF.mul hG).deriv
  simp only [Function.update_eq_self] at e
  unfold pd
  rw [hF.deriv, hG.deriv, ← e]
  rfl

/-- `∂_i x^j = δ_ij`. -/
theorem pd_coord {n : ℕ} (i j : Fin n) (x : Fin n → ℝ) : pd i (coord j) x = if i = j then 1 else 0 := by
  unfold pd coord
  by_cases h : i = j
  · subst h; simp
  · simp [Function.update_apply, Ne.symm h, h]

/-! ## def-tangent-vector -/

/-- A tangent vector is linear on functions: `X(f + g) = X(f) + X(g)` and `X(c f) = c X(f)`. -/
theorem act_linear {n : ℕ} (X : Fin n → ℝ) (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f)
    (hg : Differentiable ℝ g) (c : ℝ) (x : Fin n → ℝ) :
    act X (f + g) x = act X f x + act X g x ∧ act X (fun y => c * f y) x = c * act X f x := by
  constructor
  · unfold act
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by rw [pd_add f g hf hg]; ring
  · unfold act
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by rw [pd_const_mul c f hf]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_linear, dg-vf1-act-linear-test]

/-- The Leibniz rule: `X(fg) = X(f) g(p) + f(p) X(g)`. -/
theorem act_mul {n : ℕ} (X : Fin n → ℝ) (f g : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f)
    (hg : Differentiable ℝ g) (x : Fin n → ℝ) : act X (f * g) x = act X f x * g x + f x * act X g x := by
  unfold act
  rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by rw [pd_mul f g hf hg]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_mul, dg-vf1-act-mul-test]

/-! ## exm-tangent-partial -/

/-- `∂_θ` and `∂_φ` obey the Leibniz rule, since partial derivatives do (`pd_mul` on the sphere's chart). -/
theorem partial_leibniz (f g : (Fin 2 → ℝ) → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) (i : Fin 2)
    (x : Fin 2 → ℝ) : pd i (f * g) x = pd i f x * g x + f x * pd i g x :=
  pd_mul f g hf hg i x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.partial_leibniz, dg-vf1-partial-leibniz-test]

/-- The combination `a ∂_θ + b ∂_φ` acts by `a ∂_θ f + b ∂_φ f`. -/
theorem act_combination (a b : ℝ) (f : (Fin 2 → ℝ) → ℝ) (x : Fin 2 → ℝ) :
    act ![a, b] f x = a * pd 0 f x + b * pd 1 f x := by
  simp [act, Fin.sum_univ_succ]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_combination, dg-vf1-act-combination-test]

/-! ## exm-tangent-not -/

/-- Evaluation `f ↦ f(p)` is not a derivation: there are `f, g` with `(fg)(p) ≠ f(p) g(p) + f(p) g(p)`. -/
theorem eval_not_derivation (x : Fin 2 → ℝ) :
    ∃ f g : (Fin 2 → ℝ) → ℝ, (f * g) x ≠ f x * g x + f x * g x :=
  ⟨fun _ => 1, fun _ => 1, by norm_num⟩
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.eval_not_derivation, dg-vf1-eval-not-derivation-test]

/-- The second derivative fails the Leibniz rule by the cross term: `D(φψ) − D(φ)ψ − φD(ψ) = 2 φ' ψ'`. -/
theorem second_deriv_defect (φ ψ : ℝ → ℝ) (hφ : Differentiable ℝ φ) (hψ : Differentiable ℝ ψ)
    (hφ' : Differentiable ℝ (deriv φ)) (hψ' : Differentiable ℝ (deriv ψ)) (t : ℝ) :
    deriv (deriv (φ * ψ)) t - deriv (deriv φ) t * ψ t - φ t * deriv (deriv ψ) t = 2 * deriv φ t * deriv ψ t := by
  have h1 : deriv (φ * ψ) = deriv φ * ψ + φ * deriv ψ := by
    funext s
    exact ((hφ s).hasDerivAt.mul (hψ s).hasDerivAt).deriv
  rw [h1]
  have h2 : deriv (deriv φ * ψ + φ * deriv ψ) t =
      deriv (deriv φ) t * ψ t + deriv φ t * deriv ψ t + (deriv φ t * deriv ψ t + φ t * deriv (deriv ψ) t) :=
    (((hφ' t).hasDerivAt.mul (hψ t).hasDerivAt).add ((hφ t).hasDerivAt.mul (hψ' t).hasDerivAt)).deriv
  rw [h2]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.second_deriv_defect, dg-vf1-second-deriv-defect-test]

/-! ## exm-tangent-rn -/

/-- The directional derivative along the arrow `X` is `X^i ∂_i f(p)`: `d/dt f(p + tX)|₀ = act X f p`. -/
theorem directional_derivative {n : ℕ} (X p : Fin n → ℝ) (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) :
    deriv (fun t : ℝ => f (p + t • X)) 0 = act X f p := by
  have h1 : HasDerivAt (fun t : ℝ => p + t • X) X 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const X).const_add p
  have h2 := ((hf (p + (0 : ℝ) • X)).hasFDerivAt.comp_hasDerivAt 0 h1).deriv
  simp only [Function.comp_def, zero_smul, add_zero] at h2
  rw [h2]
  unfold act
  simp_rw [pd_eq_fderiv f hf]
  conv_lhs => rw [pi_eq_sum_univ X]
  rw [map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_smul, smul_eq_mul]
  congr 2
  ext j; simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.directional_derivative, dg-vf1-directional-derivative-test]

/-! ## exr-tv-leibniz-check -/

/-- `sin φ` on the sphere. -/
noncomputable def sinPhi : (Fin 2 → ℝ) → ℝ := fun x => Real.sin (x 1)

/-- For `X = 2 ∂_θ − ∂_φ`: `X(cos θ) = −2 sin θ`, `X(sin φ) = −cos φ`,
`X(cos θ sin φ) = −2 sin θ sin φ − cos θ cos φ`. -/
theorem leibniz_check (x : Fin 2 → ℝ) :
    act ![2, -1] heightZ x = -2 * Real.sin (x 0) ∧ act ![2, -1] sinPhi x = -Real.cos (x 1) ∧
      act ![2, -1] (heightZ * sinPhi) x = -2 * Real.sin (x 0) * Real.sin (x 1) - Real.cos (x 0) * Real.cos (x 1) := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp (disch := fun_prop) [act, pd, heightZ, sinPhi, Function.update_apply, Fin.sum_univ_succ] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.leibniz_check, dg-vf1-leibniz-check-test]

/-- `X(f) = −2 sin(π/3) = −√3` at `p`. -/
theorem leibniz_check_f : -2 * Real.sin (Real.pi / 3) = -Real.sqrt 3 := by
  VERIPHY[dg-vf1-values]

/-- `X(g) = −cos(π/4) = −√2/2` at `p`. -/
theorem leibniz_check_g : -Real.cos (Real.pi / 4) = -(Real.sqrt 2 / 2) := by
  VERIPHY[dg-vf1-values]

/-- `X(fg) = −2 sin(π/3) sin(π/4) − cos(π/3) cos(π/4) = −√6/2 − √2/4` at `p`. -/
theorem leibniz_check_fg :
    -2 * Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 3) * Real.cos (Real.pi / 4) =
      -(Real.sqrt 6 / 2) - Real.sqrt 2 / 4 := by
  VERIPHY[dg-vf1-values]

/-- The Leibniz rule holds numerically: `−√6/2 − √2/4 = (−√3)(√2/2) + (1/2)(−√2/2)`. -/
theorem leibniz_check_sum :
    -(Real.sqrt 6 / 2) - Real.sqrt 2 / 4 = -Real.sqrt 3 * (Real.sqrt 2 / 2) + 1 / 2 * (-(Real.sqrt 2 / 2)) := by
  VERIPHY[dg-vf1-values]

/-! ## exr-tv-constants -/

/-- A tangent vector kills constants: `X(c) = 0`. -/
theorem act_const {n : ℕ} (X : Fin n → ℝ) (c : ℝ) (x : Fin n → ℝ) : act X (fun _ => c) x = 0 := by
  simp [act, CoordinateGeometry.pd_const]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_const, dg-vf1-act-const-test]

/-- The components are the values on the coordinate functions: `X(x^j) = X^j`. -/
theorem act_coord {n : ℕ} (X : Fin n → ℝ) (j : Fin n) (x : Fin n → ℝ) : act X (coord j) x = X j := by
  simp [act, pd_coord]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_coord, dg-vf1-act-coord-test]

/-- Two tangent vectors that agree on the `n` coordinate functions are equal. -/
theorem act_coord_ext {n : ℕ} (X Y : Fin n → ℝ) (x : Fin n → ℝ)
    (h : ∀ j, act X (coord j) x = act Y (coord j) x) : X = Y :=
  funext fun j => by simpa [act_coord] using h j
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_coord_ext, dg-vf1-act-coord-ext-test]

/-! ## exr-tv-derivations -/

/-- `X(θ) = a` and `X(φ) = b` for `X = a ∂_θ + b ∂_φ`. -/
theorem act_coords (a b : ℝ) (x : Fin 2 → ℝ) : act ![a, b] (coord 0) x = a ∧ act ![a, b] (coord 1) x = b := by
  simp [act_coord]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_coords, dg-vf1-act-coords-test]

/-! ## def-velocity -/

/-- The chain rule: `d/dt f(γ(t)) = ẋ^i ∂_i f(γ(t))`, so the velocity has components `ẋ^i`. -/
theorem velocity_chain_rule {n : ℕ} (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (γ : ℝ → Fin n → ℝ)
    (hγ : ∀ i, Differentiable ℝ (fun t => γ t i)) (t : ℝ) :
    deriv (fun t => f (γ t)) t = ∑ i, deriv (fun t => γ t i) t * pd i f (γ t) := by
  have hγd : HasDerivAt γ (fun i => deriv (fun t => γ t i) t) t := by
    have : DifferentiableAt ℝ γ t := differentiableAt_pi.mpr fun i => (hγ i) t
    rw [← deriv_pi fun i => (hγ i) t]
    exact this.hasDerivAt
  have h := ((hf (γ t)).hasFDerivAt.comp_hasDerivAt t hγd).deriv
  simp only [Function.comp_def] at h
  rw [h]
  simp_rw [pd_eq_fderiv f hf]
  conv_lhs => rw [pi_eq_sum_univ (fun i => deriv (fun t => γ t i) t)]
  rw [map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_smul, smul_eq_mul]
  congr 2
  ext j; simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.velocity_chain_rule, dg-vf1-velocity-chain-rule-test]

/-- Every tangent vector is a velocity: the curve `x^i(t) = x^i(p) + t X^i` has velocity `X` at `t = 0`. -/
theorem line_velocity {n : ℕ} (x X : Fin n → ℝ) (i : Fin n) : deriv (fun t : ℝ => (x + t • X) i) 0 = X i := by
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.line_velocity, dg-vf1-line-velocity-test]

/-! ## exm-velocity-parallel -/

/-- The parallel `t ↦ (θ₀, φ₀ + t)` has velocity `(0, 1)`. -/
theorem parallel_velocity (θ₀ φ₀ t : ℝ) (i : Fin 2) :
    deriv (fun s => (![θ₀, φ₀ + s] : Fin 2 → ℝ) i) t = (![0, 1] : Fin 2 → ℝ) i := by
  fin_cases i <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.parallel_velocity, dg-vf1-parallel-velocity-test]

/-- The meridian `t ↦ (t, φ₀)` has velocity `(1, 0)`. -/
theorem meridian_velocity (φ₀ t : ℝ) (i : Fin 2) :
    deriv (fun s => (![s, φ₀] : Fin 2 → ℝ) i) t = (![1, 0] : Fin 2 → ℝ) i := by
  fin_cases i <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.meridian_velocity, dg-vf1-meridian-velocity-test]

/-! ## exm-velocity-general -/

/-- The slanted curve `t ↦ (π/3 + t, π/4 + 2t)` has velocity `(1, 2)`. -/
theorem slanted_velocity (t : ℝ) (i : Fin 2) :
    deriv (fun s => (![Real.pi / 3 + s, Real.pi / 4 + 2 * s] : Fin 2 → ℝ) i) t = (![1, 2] : Fin 2 → ℝ) i := by
  fin_cases i <;> simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.slanted_velocity, dg-vf1-slanted-velocity-test]

/-- Along the slanted curve the height `Z = cos θ` changes at the rate `−sin(π/3)` at `t = 0`. -/
theorem slanted_height_rate : deriv (fun t : ℝ => Real.cos (Real.pi / 3 + t)) 0 = -Real.sin (Real.pi / 3) := by
  VERIPHY[dg-vf1-deriv]

/-- `−sin(π/3) = −√3/2`. -/
theorem neg_sin_pi_div_three : -Real.sin (Real.pi / 3) = -(Real.sqrt 3 / 2) := by
  VERIPHY[dg-vf1-values]

/-! ## exm-velocity-ambient -/

/-- The spherical embedding is differentiable in each component. -/
theorem sphereEmbed_differentiable (k : Fin 3) : Differentiable ℝ (sphereEmbed k) := by
  fin_cases k <;> simp [sphereEmbed] <;> fun_prop

/-- The ambient velocity of the slanted curve is `∂_θ E + 2 ∂_φ E`: the chain rule applied to the embedding. -/
theorem ambient_velocity_slanted (k : Fin 3) :
    deriv (fun t : ℝ => sphereEmbed k ![Real.pi / 3 + t, Real.pi / 4 + 2 * t]) 0 =
      act ![1, 2] (sphereEmbed k) ![Real.pi / 3, Real.pi / 4] := by
  rw [velocity_chain_rule (sphereEmbed k) (sphereEmbed_differentiable k)
    (fun t => ![Real.pi / 3 + t, Real.pi / 4 + 2 * t]) (fun i => by fin_cases i <;> simp <;> fun_prop) 0]
  unfold act
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [slanted_velocity 0 i]
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.ambient_velocity_slanted, dg-vf1-ambient-velocity-slanted-test]

/-! ## exr-curve-velocity -/

/-- The velocity `∂_θ + 2 ∂_φ` applied to the height gives `−sin θ` (the proposition, restated). -/
theorem velocity_height (x : Fin 2 → ℝ) : act ![1, 2] heightZ x = -Real.sin (x 0) := by
  have h := dZ_components x
  simp [act, Fin.sum_univ_succ, h.1, h.2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.velocity_height, dg-vf1-velocity-height-test]

/-! ## exr-tv-great-circle -/

/-- The great circle `E(t) = (cos t, sin t/√2, sin t/√2)`. -/
noncomputable def greatCircle (t : ℝ) : Fin 3 → ℝ := ![Real.cos t, Real.sin t / Real.sqrt 2, Real.sin t / Real.sqrt 2]

/-- `E(t)` lies on the unit sphere: `cos² t + sin² t/2 + sin² t/2 = 1`. -/
theorem greatCircle_on_sphere (t : ℝ) : ∑ k, greatCircle t k ^ 2 = 1 := by
  rw [Fin.sum_univ_three]
  simp only [greatCircle, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
    Matrix.tail_cons, div_pow, Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)]
  have h := Real.sin_sq_add_cos_sq t
  generalize Real.sin t = s at h ⊢
  generalize Real.cos t = c at h ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- At `t = π/4` the colatitude is `θ = arccos(sin(π/4)/√2) = π/3`. -/
theorem greatCircle_theta : Real.arccos (Real.sin (Real.pi / 4) / Real.sqrt 2) = Real.pi / 3 := by
  VERIPHY[dg-vf1-values]

/-- At `t = π/4` the longitude is `φ = arctan(tan(π/4)/√2) = arctan(1/√2)`. -/
theorem greatCircle_phi : Real.tan (Real.pi / 4) / Real.sqrt 2 = 1 / Real.sqrt 2 := by
  VERIPHY[dg-vf1-values]

/-- `θ̇ = d/dt arccos(sin t/√2) = −(cos t/√2) / √(1 − sin² t/2)`. -/
theorem greatCircle_theta_dot (t : ℝ) :
    deriv (fun t => Real.arccos (Real.sin t / Real.sqrt 2)) t =
      -(Real.cos t / Real.sqrt 2) / Real.sqrt (1 - (Real.sin t / Real.sqrt 2) ^ 2) := by
  have hs : (Real.sin t / Real.sqrt 2) ^ 2 ≤ 1 / 2 := by
    rw [div_pow, Real.sq_sqrt (by norm_num)]
    have := Real.sin_sq_le_one t
    linarith
  have h1 : Real.sin t / Real.sqrt 2 ≠ -1 := by intro h; rw [h] at hs; norm_num at hs
  have h2 : Real.sin t / Real.sqrt 2 ≠ 1 := by intro h; rw [h] at hs; norm_num at hs
  have hd : HasDerivAt (fun t => Real.sin t / Real.sqrt 2) (Real.cos t / Real.sqrt 2) t :=
    (Real.hasDerivAt_sin t).div_const _
  have := (Real.hasDerivAt_arccos h1 h2).comp t hd
  rw [this.deriv]
  simp only [Function.comp_def]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.greatCircle_theta_dot, dg-vf1-greatcircle-theta-dot-test]

/-- `φ̇ = d/dt arctan(tan t/√2) = (1/(√2 cos² t)) / (1 + tan² t/2)` where `cos t ≠ 0`. -/
theorem greatCircle_phi_dot (t : ℝ) (h : Real.cos t ≠ 0) :
    deriv (fun t => Real.arctan (Real.tan t / Real.sqrt 2)) t =
      1 / Real.cos t ^ 2 / Real.sqrt 2 / (1 + (Real.tan t / Real.sqrt 2) ^ 2) := by
  have hd : HasDerivAt (fun t => Real.tan t / Real.sqrt 2) (1 / Real.cos t ^ 2 / Real.sqrt 2) t :=
    (Real.hasDerivAt_tan h).div_const _
  rw [hd.arctan.deriv]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.greatCircle_phi_dot, dg-vf1-greatcircle-phi-dot-test]

/-- At `t = π/4`: `θ̇ = −(cos(π/4)/√2)/√(1 − sin²(π/4)/2) = −1/√3`. -/
theorem greatCircle_theta_dot_value :
    -(Real.cos (Real.pi / 4) / Real.sqrt 2) / Real.sqrt (1 - (Real.sin (Real.pi / 4) / Real.sqrt 2) ^ 2) =
      -1 / Real.sqrt 3 := by
  VERIPHY[dg-vf1-values]

/-- At `t = π/4`: `φ̇ = (1/(√2 cos²(π/4))) / (1 + tan²(π/4)/2) = 2√2/3`. -/
theorem greatCircle_phi_dot_value :
    1 / Real.cos (Real.pi / 4) ^ 2 / Real.sqrt 2 / (1 + (Real.tan (Real.pi / 4) / Real.sqrt 2) ^ 2) =
      2 * Real.sqrt 2 / 3 := by
  VERIPHY[dg-vf1-values]

/-- The velocity on the height: `−sin θ θ̇ = (√3/2)(1/√3) = 1/2` at the point. -/
theorem greatCircle_height_rate : -Real.sin (Real.pi / 3) * (-1 / Real.sqrt 3) = 1 / 2 := by
  VERIPHY[dg-vf1-values]

/-- Along the curve, `d/dt Z(E(t)) = d/dt (sin t/√2) = cos t/√2`. -/
theorem greatCircle_dZ (t : ℝ) : deriv (fun t => Real.sin t / Real.sqrt 2) t = Real.cos t / Real.sqrt 2 := by
  VERIPHY[dg-vf1-deriv]

/-- At `t = π/4`, `cos(π/4)/√2 = 1/2`, the same number. -/
theorem greatCircle_dZ_value : Real.cos (Real.pi / 4) / Real.sqrt 2 = 1 / 2 := by
  VERIPHY[dg-vf1-values]

/-- The ambient velocity `θ̇ ∂_θ E + φ̇ ∂_φ E` at the point equals `E'(π/4)`, componentwise. -/
def GreatCircleAmbientCheck (k : Fin 3) : Prop :=
  act ![-1 / Real.sqrt 3, 2 * Real.sqrt 2 / 3] (sphereEmbed k) ![Real.pi / 3, Real.arctan (1 / Real.sqrt 2)] =
    deriv (fun t => greatCircle t k) (Real.pi / 4)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.GreatCircleAmbientCheck, dg-vf1-greatcircle-ambient-test]

/-! ## exr-tv-reparametrisation -/

/-- Reparametrisation rescales the velocity: `d/ds f(γ(τ(s))) = τ'(s) · (d/dt f(γ(t)))|_{t = τ(s)}`. -/
theorem reparametrisation {n : ℕ} (f : (Fin n → ℝ) → ℝ) (γ : ℝ → Fin n → ℝ) (τ : ℝ → ℝ)
    (hfγ : Differentiable ℝ (fun t => f (γ t))) (hτ : Differentiable ℝ τ) (s : ℝ) :
    deriv (fun s => f (γ (τ s))) s = deriv τ s * deriv (fun t => f (γ t)) (τ s) := by
  have h := ((hfγ (τ s)).hasDerivAt.comp s (hτ s).hasDerivAt).deriv
  simp only [Function.comp_def] at h
  rw [h]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.reparametrisation, dg-vf1-reparametrisation-test]

/-! ## def-coordinate-basis -/

/-- The coordinate basis vector `∂_i = e_i` sends `f` to its partial derivative `∂_i f`. -/
theorem act_single {n : ℕ} (i : Fin n) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) :
    act (Pi.single i 1) f x = pd i f x := by
  simp [act, Pi.single_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_single, dg-vf1-act-single-test]

/-- The coordinate basis vectors `∂_1, …, ∂_n` are linearly independent. -/
theorem coord_independent (n : ℕ) : LinearIndependent ℝ (fun i : Fin n => (Pi.single i (1 : ℝ) : Fin n → ℝ)) :=
  (Pi.basisFun ℝ (Fin n)).linearIndependent
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.coord_independent, dg-vf1-coord-independent-test]

/-- Every tangent vector is a combination of the coordinate basis: `X = Σ_i X^i ∂_i`. -/
theorem coord_span {n : ℕ} (X : Fin n → ℝ) : X = ∑ i, X i • (Pi.single i (1 : ℝ) : Fin n → ℝ) := by
  ext j; simp [Pi.single_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.coord_span, dg-vf1-coord-span-test]

/-! ## exm-basis-sphere -/

/-- `∂_θ E = (cos θ cos φ, cos θ sin φ, −sin θ)` and `∂_φ E = (−sin θ sin φ, sin θ cos φ, 0)`: the Jacobian columns. -/
theorem sphere_jacobian (k : Fin 3) (x : Fin 2 → ℝ) :
    pd 0 (sphereEmbed k) x =
        (![Real.cos (x 0) * Real.cos (x 1), Real.cos (x 0) * Real.sin (x 1), -Real.sin (x 0)] : Fin 3 → ℝ) k ∧
      pd 1 (sphereEmbed k) x =
        (![-(Real.sin (x 0) * Real.sin (x 1)), Real.sin (x 0) * Real.cos (x 1), 0] : Fin 3 → ℝ) k := by
  fin_cases k <;> constructor <;> simp (disch := fun_prop) [pd, sphereEmbed, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.sphere_jacobian, dg-vf1-sphere-jacobian-test]

/-- `∂_φ` has ambient length `sin θ`: `|∂_φ E|² = sin² θ` (the Gram matrix of the spherical chart). -/
theorem dphi_length (x : Fin 2 → ℝ) : gram sphereEmbed 1 1 x = Real.sin (x 0) ^ 2 := by
  rw [sphere_gram]; simp [sphere]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_length, dg-vf1-dphi-length-test]

/-! ## exm-basis-polar -/

/-- The polar Jacobian: `∂_r = (cos φ, sin φ)` and `∂_φ = (−r sin φ, r cos φ)` as Cartesian vectors. -/
theorem polar_jacobian (k : Fin 2) (x : Fin 2 → ℝ) :
    pd 0 (polarToCart k) x = (![Real.cos (x 1), Real.sin (x 1)] : Fin 2 → ℝ) k ∧
      pd 1 (polarToCart k) x = (![-(x 0 * Real.sin (x 1)), x 0 * Real.cos (x 1)] : Fin 2 → ℝ) k := by
  fin_cases k <;> constructor <;> simp (disch := fun_prop) [pd, polarToCart, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.polar_jacobian, dg-vf1-polar-jacobian-test]

/-- `∂_r` has unit length and `∂_φ` has length `r`: `|∂_r|² = 1`, `|∂_φ|² = r²`, in Cartesian components. -/
theorem polar_lengths (x : Fin 2 → ℝ) :
    gram polarToCart 0 0 x = 1 ∧ gram polarToCart 1 1 x = x 0 ^ 2 ∧ gram polarToCart 0 1 x = 0 := by
  have h0 := polar_jacobian 0 x
  have h1 := polar_jacobian 1 x
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons] at h0 h1
  simp only [gram, Fin.sum_univ_two, h0.1, h0.2, h1.1, h1.2]
  refine ⟨?_, ?_, ?_⟩
  · nlinarith [Real.sin_sq_add_cos_sq (x 1)]
  · nlinarith [Real.sin_sq_add_cos_sq (x 1)]
  · ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.polar_lengths, dg-vf1-polar-lengths-test]

/-- In Cartesian coordinates the basis is the same everywhere: `∂_i x^k = δ_ik` at every point. -/
theorem cartesian_basis (i k : Fin 2) (x : Fin 2 → ℝ) : pd i (coord k) x = if i = k then 1 else 0 :=
  pd_coord i k x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.cartesian_basis, dg-vf1-cartesian-basis-test]

/-! ## exm-basis-stereographic -/

/-- The stereographic basis: `∂_u E ⊥ ∂_v E` and `|∂_u E|² = |∂_v E|² = 4/(1 + u² + v²)²`. -/
theorem stereo_gram (x : Fin 2 → ℝ) :
    gram stereoEmbed 0 0 x = 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 ∧
      gram stereoEmbed 1 1 x = 4 / (1 + x 0 ^ 2 + x 1 ^ 2) ^ 2 ∧ gram stereoEmbed 0 1 x = 0 := by
  have hd : 1 + x 0 ^ 2 + x 1 ^ 2 ≠ 0 := by positivity
  have hd' : ∀ t : ℝ, 1 + t ^ 2 + x 1 ^ 2 ≠ 0 := fun t => by positivity
  have hd'' : ∀ t : ℝ, 1 + x 0 ^ 2 + t ^ 2 ≠ 0 := fun t => by positivity
  simp only [gram, Fin.sum_univ_three, stereoEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons, pd, Function.update_apply, Fin.zero_eta, Fin.isValue, Fin.one_eq_zero_iff,
    OfNat.ofNat_ne_one, ↓reduceIte, Fin.zero_eq_one_iff, OfNat.one_ne_ofNat]
  simp (disch := first | assumption | apply hd' | apply hd'' | fun_prop) only [deriv_div, deriv_add, deriv_const,
    deriv_pow'', deriv_id'', deriv_const_mul, deriv_sub, zero_add, add_zero, mul_one, one_mul, Nat.cast_ofNat,
    pow_one, mul_zero, sub_zero]
  refine ⟨?_, ?_, ?_⟩ <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.stereo_gram, dg-vf1-stereo-gram-test]

/-- On the equator `u² + v² = 1` the basis vectors have unit length: `4/(1 + u² + v²)² = 1`. -/
theorem stereo_equator_length (u v : ℝ) (h : u ^ 2 + v ^ 2 = 1) : 4 / (1 + u ^ 2 + v ^ 2) ^ 2 = 1 := by
  rw [div_eq_iff (by positivity)]
  VERIPHY[dg-vf1-poly]

/-- Far from the origin the basis vectors shrink: `2/(1 + ρ²) < 1` when `ρ² > 1`. -/
theorem stereo_shrink (ρ2 : ℝ) (h : 1 < ρ2) : 2 / (1 + ρ2) < 1 := by
  rw [div_lt_one (by linarith)]
  VERIPHY[dg-vf1-farkas]

/-! ## exr-tv-stereo-basis -/

/-- `E(u, v)` lies on the unit sphere: `(2u)² + (2v)² + (u² + v² − 1)² = (1 + u² + v²)²`. -/
theorem stereoEmbed_on_sphere (x : Fin 2 → ℝ) : ∑ k, stereoEmbed k x ^ 2 = 1 := by
  have hd : 1 + x 0 ^ 2 + x 1 ^ 2 ≠ 0 := by positivity
  rw [Fin.sum_univ_three]
  simp only [stereoEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two,
    Matrix.tail_cons, div_pow]
  rw [div_add_div_same, div_add_div_same, div_eq_one_iff_eq (pow_ne_zero 2 hd)]
  VERIPHY[dg-vf1-poly]

/-- `∂_u E` and `∂_v E` are tangent to the sphere: orthogonal to the position vector `E`. -/
theorem stereo_tangent (i : Fin 2) (x : Fin 2 → ℝ) : ∑ k, stereoEmbed k x * pd i (stereoEmbed k) x = 0 := by
  have hd : 1 + x 0 ^ 2 + x 1 ^ 2 ≠ 0 := by positivity
  have hd' : ∀ t : ℝ, 1 + t ^ 2 + x 1 ^ 2 ≠ 0 := fun t => by positivity
  have hd'' : ∀ t : ℝ, 1 + x 0 ^ 2 + t ^ 2 ≠ 0 := fun t => by positivity
  fin_cases i <;>
  · simp only [Fin.sum_univ_three, stereoEmbed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_two, Matrix.tail_cons, pd, Function.update_apply, Fin.zero_eta, Fin.isValue, Fin.mk_one,
      Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ↓reduceIte, Fin.zero_eq_one_iff, OfNat.one_ne_ofNat]
    simp (disch := first | assumption | apply hd' | apply hd'' | fun_prop) only [deriv_div, deriv_add, deriv_const,
      deriv_pow'', deriv_id'', deriv_const_mul, deriv_sub, zero_add, add_zero, mul_one, one_mul, Nat.cast_ofNat,
      pow_one, mul_zero, sub_zero]
    field_simp
    ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.stereo_tangent, dg-vf1-stereo-tangent-test]

/-- The common length `2/(1 + u² + v²)` is `2` at the south pole, `1` on the equator and `1/5` at `(3, 0)`. -/
theorem stereo_length_values :
    2 / (1 + 0 ^ 2 + 0 ^ 2 : ℝ) = 2 ∧ 2 / (1 + 1 ^ 2 + 0 ^ 2 : ℝ) = 1 ∧ 2 / (1 + 3 ^ 2 + 0 ^ 2 : ℝ) = 1 / 5 := by
  refine ⟨?_, ?_, ?_⟩
  · VERIPHY[dg-vf1-poly]
  · VERIPHY[dg-vf1-poly]
  · VERIPHY[dg-vf1-poly]

/-! ## exr-tv-torus-basis -/

/-- `∂_a E = (−r sin a cos b, −r sin a sin b, r cos a)` and `∂_b E = (−(R + r cos a) sin b, (R + r cos a) cos b, 0)`. -/
theorem torus_jacobian (R r : ℝ) (k : Fin 3) (x : Fin 2 → ℝ) :
    pd 0 (torusEmbed R r k) x =
        (![-(r * Real.sin (x 0) * Real.cos (x 1)), -(r * Real.sin (x 0) * Real.sin (x 1)), r * Real.cos (x 0)] :
          Fin 3 → ℝ) k ∧
      pd 1 (torusEmbed R r k) x =
        (![-((R + r * Real.cos (x 0)) * Real.sin (x 1)), (R + r * Real.cos (x 0)) * Real.cos (x 1), 0] :
          Fin 3 → ℝ) k := by
  fin_cases k <;> constructor <;> simp (disch := fun_prop) [pd, torusEmbed, Function.update_apply] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.torus_jacobian, dg-vf1-torus-jacobian-test]

/-- `∂_a E · ∂_b E = r sin a (R + r cos a)(cos b sin b − sin b cos b) = 0`. -/
theorem torus_orthogonal (R r sa ca sb cb : ℝ) :
    -(r * sa * cb) * -((R + r * ca) * sb) + -(r * sa * sb) * ((R + r * ca) * cb) + r * ca * 0 = 0 := by
  VERIPHY[dg-vf1-poly]

/-- `|∂_a E|² = r²`, using `sin² + cos² = 1` for both angles. -/
theorem torus_length_a (r a b : ℝ) :
    (-(r * Real.sin a * Real.cos b)) ^ 2 + (-(r * Real.sin a * Real.sin b)) ^ 2 + (r * Real.cos a) ^ 2 = r ^ 2 := by
  have ha := Real.sin_sq_add_cos_sq a
  have hb := Real.sin_sq_add_cos_sq b
  generalize Real.sin a = sa at ha ⊢
  generalize Real.cos a = ca at ha ⊢
  generalize Real.sin b = sb at hb ⊢
  generalize Real.cos b = cb at hb ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- `|∂_b E|² = (R + r cos a)²`. -/
theorem torus_length_b (R r a b : ℝ) :
    (-((R + r * Real.cos a) * Real.sin b)) ^ 2 + ((R + r * Real.cos a) * Real.cos b) ^ 2 + (0 : ℝ) ^ 2 =
      (R + r * Real.cos a) ^ 2 := by
  have hb := Real.sin_sq_add_cos_sq b
  generalize Real.sin b = sb at hb ⊢
  generalize Real.cos b = cb at hb ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- `∂_b` is longest on the outer equator: `R + r cos a ≤ R + r` (radius `R + r` at `a = 0`). -/
theorem torus_outer (R r a : ℝ) (hr : 0 < r) : R + r * Real.cos a ≤ R + r := by
  have hc := Real.cos_le_one a
  generalize Real.cos a = c at hc ⊢
  VERIPHY[dg-vf1-farkas]

/-- `∂_b` is shortest on the inner equator: `R − r ≤ R + r cos a` (radius `R − r` at `a = π`). -/
theorem torus_inner (R r a : ℝ) (hr : 0 < r) : R - r ≤ R + r * Real.cos a := by
  have hc := Real.neg_one_le_cos a
  generalize Real.cos a = c at hc ⊢
  VERIPHY[dg-vf1-farkas]

/-! ## exr-tv-basis-independence -/

/-- If `a^i ∂_i = 0` on the coordinate functions then every `a^i = 0`. -/
theorem independence_of_coords {n : ℕ} (a : Fin n → ℝ) (x : Fin n → ℝ) (h : ∀ j, act a (coord j) x = 0) :
    a = 0 :=
  funext fun j => by simpa [act_coord] using h j
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.independence_of_coords, dg-vf1-independence-of-coords-test]

/-- The tangent space `T_pM ≅ ℝⁿ` has dimension exactly `n`. -/
theorem tangent_dimension (n : ℕ) : Module.finrank ℝ (Fin n → ℝ) = n := by
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.tangent_dimension, dg-vf1-tangent-dimension-test]

/-! ## def-components -/

/-- The chain rule for a change of chart `x' = φ(x)`: `∂_i (f ∘ φ) = (∂_i φ^j) ∂'_j f`. -/
theorem pd_comp {n : ℕ} (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (φ : (Fin n → ℝ) → Fin n → ℝ)
    (hφ : ∀ j, Differentiable ℝ (fun x => φ x j)) (i : Fin n) (x : Fin n → ℝ) :
    pd i (fun x => f (φ x)) x = ∑ j, pd i (fun x => φ x j) x * pd j f (φ x) := by
  unfold pd
  have hγ : ∀ j, Differentiable ℝ (fun t => φ (Function.update x i t) j) := fun j =>
    (hφ j).comp (by fun_prop)
  have hγd : HasDerivAt (fun t => φ (Function.update x i t))
      (fun j => deriv (fun t => φ (Function.update x i t) j) (x i)) (x i) := by
    have : DifferentiableAt ℝ (fun t => φ (Function.update x i t)) (x i) :=
      differentiableAt_pi.mpr fun j => (hγ j) (x i)
    rw [← deriv_pi fun j => (hγ j) (x i)]
    exact this.hasDerivAt
  have h := ((hf (φ (Function.update x i (x i)))).hasFDerivAt.comp_hasDerivAt (x i) hγd).deriv
  simp only [Function.comp_def, Function.update_eq_self] at h
  rw [h]
  simp_rw [← pd, pd_eq_fderiv f hf]
  conv_lhs => rw [pi_eq_sum_univ (fun j => deriv (fun t => φ (Function.update x i t) j) (x i))]
  rw [map_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [map_smul, smul_eq_mul]
  congr 2
  ext l; simp [Pi.single_apply, eq_comm]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.pd_comp, dg-vf1-pd-comp-test]

/-- The transformation law: the components `X'^j = (∂x'^j/∂x^i) X^i` represent the same vector,
`X(f ∘ φ)(x) = X'(f)(φ(x))`. -/
theorem act_comp {n : ℕ} (X : Fin n → ℝ) (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f)
    (φ : (Fin n → ℝ) → Fin n → ℝ) (hφ : ∀ j, Differentiable ℝ (fun x => φ x j)) (x : Fin n → ℝ) :
    act X (fun x => f (φ x)) x = act (fun j => ∑ i, pd i (fun x => φ x j) x * X i) f (φ x) := by
  unfold act
  simp_rw [pd_comp f hf φ hφ, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_comp, dg-vf1-act-comp-test]

/-! ## exm-components-sphere -/

/-- `∂_φ`, components `(0, 1)`, has stereographic components `(∂u/∂φ, ∂v/∂φ) = (−v, u)`. -/
theorem dphi_stereo_components (x : Fin 2 → ℝ) :
    (∑ i, pd i stereoU x * dphi i x) = -stereoV x ∧ (∑ i, pd i stereoV x * dphi i x) = stereoU x := by
  simp [Fin.sum_univ_succ, dphi, dphi_u, dphi_v]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_stereo_components, dg-vf1-dphi-stereo-components-test]

/-! ## exm-components-circle -/

/-- The transition map `φ' = φ + c` of the circle has Jacobian `1`. -/
theorem circle_jacobian (c φ : ℝ) : deriv (fun φ => φ + c) φ = 1 := by
  VERIPHY[dg-vf1-deriv]

/-! ## exm-components-polar -/

/-- `∂_r = cos φ ∂_x + sin φ ∂_y` and `∂_φ = −r sin φ ∂_x + r cos φ ∂_y`: the Jacobian columns are the
Cartesian components of the polar basis vectors. -/
theorem polar_basis_cartesian (x : Fin 2 → ℝ) :
    (fun k => pd 0 (polarToCart k) x) = ![Real.cos (x 1), Real.sin (x 1)] ∧
      (fun k => pd 1 (polarToCart k) x) = ![-(x 0 * Real.sin (x 1)), x 0 * Real.cos (x 1)] := by
  constructor <;> funext k <;> fin_cases k <;> simp (disch := fun_prop) [pd, polarToCart, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.polar_basis_cartesian, dg-vf1-polar-basis-cartesian-test]

/-! ## exr-dphi-stereographic -/

/-- `X^u = ∂u/∂φ = −sin θ sin φ/(1 − cos θ) = −v`. -/
theorem dphi_u_component (x : Fin 2 → ℝ) : pd 1 stereoU x = -stereoV x := dphi_u x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_u_component, dg-vf1-dphi-u-component-test]

/-- `X^v = ∂v/∂φ = u`. -/
theorem dphi_v_component (x : Fin 2 → ℝ) : pd 1 stereoV x = stereoU x := dphi_v x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_v_component, dg-vf1-dphi-v-component-test]

/-! ## exr-components-polar -/

/-- `∂_x = cos φ ∂_r − (sin φ/r) ∂_φ`: the polar Jacobian sends `(cos φ, −sin φ/r)` to `(1, 0)`, for `r ≠ 0`. -/
theorem dx_polar_components (r φ : ℝ) (hr : r ≠ 0) :
    Real.cos φ * Real.cos φ + -(r * Real.sin φ) * (-(Real.sin φ) / r) = 1 ∧
      Real.sin φ * Real.cos φ + r * Real.cos φ * (-(Real.sin φ) / r) = 0 := by
  have h := Real.sin_sq_add_cos_sq φ
  generalize Real.sin φ = s at h ⊢
  generalize Real.cos φ = c at h ⊢
  constructor
  · field_simp
    rw [← sub_eq_zero]
    VERIPHY[dg-vf1-poly]
  · field_simp
    rw [← sub_eq_zero]
    VERIPHY[dg-vf1-poly]

/-! ## exr-tv-gradient-stereographic -/

/-- `∂u/∂θ = −cos φ/(1 − cos θ)` where `cos θ ≠ 1`. -/
theorem dtheta_u (x : Fin 2 → ℝ) (h : Real.cos (x 0) ≠ 1) :
    pd 0 stereoU x = -Real.cos (x 1) / (1 - Real.cos (x 0)) := by
  have hne : 1 - Real.cos (x 0) ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hs := Real.sin_sq_add_cos_sq (x 0)
  have hd : HasDerivAt (fun t => Real.sin t * Real.cos (x 1) / (1 - Real.cos t))
      ((Real.cos (x 0) * Real.cos (x 1) * (1 - Real.cos (x 0)) -
        Real.sin (x 0) * Real.cos (x 1) * Real.sin (x 0)) / (1 - Real.cos (x 0)) ^ 2) (x 0) := by
    have h1 : HasDerivAt (fun t => Real.sin t * Real.cos (x 1)) (Real.cos (x 0) * Real.cos (x 1)) (x 0) :=
      (Real.hasDerivAt_sin (x 0)).mul_const _
    have h2 : HasDerivAt (fun t => 1 - Real.cos t) (Real.sin (x 0)) (x 0) := by
      simpa using (Real.hasDerivAt_cos (x 0)).const_sub 1
    exact h1.div h2 hne
  unfold pd stereoU
  simp only [Function.update_apply, Fin.isValue, Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ↓reduceIte,
    Fin.zero_eq_one_iff, OfNat.one_ne_ofNat]
  rw [hd.deriv, div_eq_div_iff (pow_ne_zero 2 hne) hne]
  linear_combination (-(Real.cos (x 1)) * (1 - Real.cos (x 0))) * hs

/-- `∂v/∂θ = −sin φ/(1 − cos θ)` where `cos θ ≠ 1`. -/
theorem dtheta_v (x : Fin 2 → ℝ) (h : Real.cos (x 0) ≠ 1) :
    pd 0 stereoV x = -Real.sin (x 1) / (1 - Real.cos (x 0)) := by
  have hne : 1 - Real.cos (x 0) ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hs := Real.sin_sq_add_cos_sq (x 0)
  have hd : HasDerivAt (fun t => Real.sin t * Real.sin (x 1) / (1 - Real.cos t))
      ((Real.cos (x 0) * Real.sin (x 1) * (1 - Real.cos (x 0)) -
        Real.sin (x 0) * Real.sin (x 1) * Real.sin (x 0)) / (1 - Real.cos (x 0)) ^ 2) (x 0) := by
    have h1 : HasDerivAt (fun t => Real.sin t * Real.sin (x 1)) (Real.cos (x 0) * Real.sin (x 1)) (x 0) :=
      (Real.hasDerivAt_sin (x 0)).mul_const _
    have h2 : HasDerivAt (fun t => 1 - Real.cos t) (Real.sin (x 0)) (x 0) := by
      simpa using (Real.hasDerivAt_cos (x 0)).const_sub 1
    exact h1.div h2 hne
  unfold pd stereoV
  simp only [Function.update_apply, Fin.isValue, Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ↓reduceIte,
    Fin.zero_eq_one_iff, OfNat.one_ne_ofNat]
  rw [hd.deriv, div_eq_div_iff (pow_ne_zero 2 hne) hne]
  linear_combination (-(Real.sin (x 1)) * (1 - Real.cos (x 0))) * hs

/-- The gradient field `−sin θ ∂_θ` has stereographic components `(u, v)`: `X^u = −sin θ ∂u/∂θ = u` and
`X^v = v`, the radial field of the chart plane. -/
theorem gradZ_stereo_components (x : Fin 2 → ℝ) (h : Real.cos (x 0) ≠ 1) :
    (∑ i, pd i stereoU x * gradZ i x) = stereoU x ∧ (∑ i, pd i stereoV x * gradZ i x) = stereoV x := by
  have hne : 1 - Real.cos (x 0) ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  simp only [Fin.sum_univ_two, gradZ, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, dtheta_u x h,
    dtheta_v x h, mul_zero, add_zero, stereoU, stereoV]
  constructor <;> field_simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.gradZ_stereo_components, dg-vf1-gradz-stereo-components-test]

/-! ## def-tangent-space -/

/-- `T_pM` with basis `∂_1, …, ∂_n` has dimension `n`: the standard basis of `ℝⁿ` has `n` vectors. -/
theorem tangent_space_basis (n : ℕ) : Fintype.card (Fin n) = n ∧ Module.finrank ℝ (Fin n → ℝ) = n := by
  simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.tangent_space_basis, dg-vf1-tangent-space-basis-test]

/-! ## exm-tangent-space-sphere -/

/-- `∂_θ E` and `∂_φ E` are perpendicular to the position vector `p = E(θ, φ)`. -/
theorem sphere_tangent_perp (i : Fin 2) (x : Fin 2 → ℝ) :
    ∑ k, sphereEmbed k x * pd i (sphereEmbed k) x = 0 := by
  have h0 := fun k => (sphere_jacobian k x).1
  have h1 := fun k => (sphere_jacobian k x).2
  fin_cases i
  · simp only [Fin.zero_eta, Fin.isValue, h0, Fin.sum_univ_three, sphereEmbed, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]
    ring
  · simp only [Fin.mk_one, Fin.isValue, h1, Fin.sum_univ_three, sphereEmbed, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]
    ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.sphere_tangent_perp, dg-vf1-sphere-tangent-perp-test]

/-- `T_pS²` has dimension `2`: the Gram determinant of `∂_θ E, ∂_φ E` is `sin² θ`, nonzero away from the poles. -/
theorem sphere_tangent_dim (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (Matrix.of fun i j => gram sphereEmbed i j x).det ≠ 0 := by
  rw [sphere_gram_det]; exact pow_ne_zero 2 h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.sphere_tangent_dim, dg-vf1-sphere-tangent-dim-test]

/-! ## exm-tangent-space-group -/

/-- The generators `T_a` are traceless and anti-Hermitian. -/
theorem T_traceless_antiHermitian (a : Fin 3) : (T a).trace = 0 ∧ (T a).conjTranspose = -T a := by
  fin_cases a <;> constructor <;>
    simp [T, pauli, Matrix.trace, Fin.sum_univ_succ, Matrix.conjTranspose, Matrix.map] <;>
    ext i j <;> fin_cases i <;> fin_cases j <;> simp [Complex.ext_iff]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.T_traceless_antiHermitian, dg-vf1-t-traceless-test]

/-- Every traceless anti-Hermitian `2 × 2` matrix is a real combination of `T₀, T₁, T₂`. -/
theorem su2_tangent_span (A : Matrix (Fin 2) (Fin 2) ℂ) (hH : A.conjTranspose = -A) (hT : A.trace = 0) :
    ∃ a b c : ℝ, A = a • T 0 + b • T 1 + c • T 2 := by
  have h00 := congrFun (congrFun hH 0) 0
  have h01 := congrFun (congrFun hH 0) 1
  have h11 := congrFun (congrFun hH 1) 1
  simp only [Matrix.conjTranspose_apply, Matrix.neg_apply, Complex.ext_iff, Complex.conj_re, Complex.conj_im,
    Complex.neg_re, Complex.neg_im] at h00 h01 h11
  simp only [Matrix.trace, Matrix.diag, Fin.sum_univ_two, Complex.ext_iff, Complex.add_re, Complex.add_im,
    Complex.zero_re, Complex.zero_im] at hT
  refine ⟨-2 * (A 1 0).im, -2 * (A 1 0).re, -2 * (A 0 0).im, ?_⟩
  ext i j
  fin_cases i <;> fin_cases j <;> simp [T, pauli, Complex.ext_iff] <;> constructor <;> nlinarith [h00, h01, h11, hT]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.su2_tangent_span, dg-vf1-su2-tangent-span-test]

/-- The coefficients are read off the entries, so `T₀, T₁, T₂` are independent and `T_e SU(2)` is three-dimensional:
`(a T₀ + b T₁ + c T₂)₀₀ = −ic/2` and `(a T₀ + b T₁ + c T₂)₀₁ = −(ia + b)/2`. -/
theorem su2_coordinates (a b c : ℝ) :
    ((a • T 0 + b • T 1 + c • T 2) 0 0).im = -c / 2 ∧ ((a • T 0 + b • T 1 + c • T 2) 0 1).im = -a / 2 ∧
      ((a • T 0 + b • T 1 + c • T 2) 0 1).re = -b / 2 := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [T, pauli] <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.su2_coordinates, dg-vf1-su2-coordinates-test]

/-! ## exm-tangent-space-rn -/

/-- In `ℝⁿ`, `∂_i` is the standard basis vector `e_i` and `X` acts as the derivative along the arrow: `X(f) = Df(p) X`. -/
theorem act_eq_fderiv {n : ℕ} (X : Fin n → ℝ) (f : (Fin n → ℝ) → ℝ) (hf : Differentiable ℝ f) (p : Fin n → ℝ) :
    act X f p = fderiv ℝ f p X := by
  rw [← directional_derivative X p f hf]
  have h1 : HasDerivAt (fun t : ℝ => p + t • X) X 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const X).const_add p
  have h2 := ((hf (p + (0 : ℝ) • X)).hasFDerivAt.comp_hasDerivAt 0 h1).deriv
  simp only [Function.comp_def, zero_smul, add_zero] at h2
  exact h2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_eq_fderiv, dg-vf1-act-eq-fderiv-test]

/-! ## exr-tv-basis-change-tangent -/

/-- At `(x, y) = (1, 1)`: `r = √2`. -/
theorem point_radius : Real.sqrt (1 ^ 2 + 1 ^ 2) = Real.sqrt 2 := by
  VERIPHY[dg-vf1-values]

/-- At `(1, 1)`: `∂_r = (cos(π/4), sin(π/4)) = (1/√2)(1, 1)`. -/
theorem dr_at_point : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 ∧ Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- At `(1, 1)`: `∂_φ = (−r sin φ, r cos φ) = (−1, 1)`. -/
theorem dphi_at_point :
    -(Real.sqrt 2 * Real.sin (Real.pi / 4)) = -1 ∧ Real.sqrt 2 * Real.cos (Real.pi / 4) = 1 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- The polar components of `X = 3 ∂_x + ∂_y` at `(1, 1)`: `X^r = (x X^x + y X^y)/r = 4/√2 = 2√2` and
`X^φ = (−y X^x + x X^y)/r² = −1`. -/
theorem X_polar_components :
    (1 * 3 + 1 * 1) / Real.sqrt 2 = 2 * Real.sqrt 2 ∧ (-(1 * 3) + 1 * 1) / (Real.sqrt 2) ^ 2 = -1 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- Check: `2√2 · (1/√2)(1, 1) − (−1, 1) = (3, 1)`. -/
theorem X_polar_check :
    2 * Real.sqrt 2 * (1 / Real.sqrt 2) * 1 - (-1) = 3 ∧ 2 * Real.sqrt 2 * (1 / Real.sqrt 2) * 1 - 1 = 1 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-! ## exr-tv-so-tangent -/

/-- Differentiating `R(t)ᵀ R(t) = 1` at `t = 0` with `R(0) = 1` and `Ṙ(0) = A` gives `Aᵀ + A = 0`. -/
theorem orthogonal_velocity_antisymmetric {n : ℕ} (R : ℝ → Matrix (Fin n) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (hR : ∀ i j, HasDerivAt (fun t => R t i j) (A i j) 0) (h0 : R 0 = 1)
    (horth : ∀ t, (R t).transpose * R t = 1) : A.transpose + A = 0 := by
  ext i j
  have hc : ∀ t, ∑ k, R t k i * R t k j = (1 : Matrix (Fin n) (Fin n) ℝ) i j := fun t => by
    rw [← horth t, Matrix.mul_apply]; simp [Matrix.transpose_apply]
  have hd : HasDerivAt (fun t => ∑ k, R t k i * R t k j) (∑ k, (A k i * R 0 k j + R 0 k i * A k j)) 0 :=
    HasDerivAt.sum fun k _ => (hR k i).mul (hR k j)
  have hconst : HasDerivAt (fun t => ∑ k, R t k i * R t k j) 0 0 := by
    have : (fun t => ∑ k, R t k i * R t k j) = fun _ => (1 : Matrix (Fin n) (Fin n) ℝ) i j := funext hc
    rw [this]; exact hasDerivAt_const _ _
  have := hd.unique hconst
  rw [h0] at this
  simp only [Matrix.one_apply, mul_ite, mul_one, mul_zero, ite_mul, one_mul, zero_mul, Finset.sum_add_distrib,
    Finset.sum_ite_eq', Finset.sum_ite_eq, Finset.mem_univ, if_true] at this
  simp only [Matrix.add_apply, Matrix.transpose_apply, Matrix.zero_apply]
  linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.orthogonal_velocity_antisymmetric, dg-vf1-orthogonal-velocity-test]

/-- `T₁SO(2)` is one-dimensional: an antisymmetric `2 × 2` matrix is a multiple of `J = [[0, −1], [1, 0]]`. -/
theorem antisymmetric_two (A : Matrix (Fin 2) (Fin 2) ℝ) (h : A.transpose = -A) : ∃ c : ℝ, A = c • J2 := by
  have h00 := congrFun (congrFun h 0) 0
  have h01 := congrFun (congrFun h 0) 1
  have h11 := congrFun (congrFun h 1) 1
  simp only [Matrix.transpose_apply, Matrix.neg_apply] at h00 h01 h11
  refine ⟨A 1 0, ?_⟩
  ext i j; fin_cases i <;> fin_cases j <;> simp [J2] <;> linarith
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.antisymmetric_two, dg-vf1-antisymmetric-two-test]

/-- `J` is the velocity at `t = 0` of the rotation by the angle `t`. -/
theorem rot2_velocity (i j : Fin 2) : deriv (fun t => rot2 t i j) 0 = J2 i j := by
  fin_cases i <;> fin_cases j <;> simp [rot2, J2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.rot2_velocity, dg-vf1-rot2-velocity-test]

/-- `dim SO(3) = 3`: an antisymmetric `3 × 3` matrix has three free entries, `A = A₂₁ L₀ + A₀₂ L₁ + A₁₀ L₂`
with the generators `L_a` (`!![0,0,0; 0,0,-1; 0,1,0]`, `!![0,0,1; 0,0,0; -1,0,0]`, `!![0,-1,0; 1,0,0; 0,0,0]`). -/
theorem antisymmetric_three (A : Matrix (Fin 3) (Fin 3) ℝ) (h : A.transpose = -A) :
    A = A 2 1 • !![(0 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] + A 0 2 • !![(0 : ℝ), 0, 1; 0, 0, 0; -1, 0, 0] +
      A 1 0 • !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 0] := by
  have e : ∀ i j, A j i = -A i j := fun i j => by
    have := congrFun (congrFun h i) j
    simpa [Matrix.transpose_apply, Matrix.neg_apply] using this
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> linarith [e 0 0, e 1 1, e 2 2, e 0 1, e 0 2, e 1 2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.antisymmetric_three, dg-vf1-antisymmetric-three-test]

/-! ## exr-tv-no-addition -/

/-- In Cartesian components `∂_φ = (−y, x)` is `∂_y` at `p = (1, 0)` and `−∂_y` at `q = (−1, 0)`, so the Cartesian
"sum" is `0`. -/
theorem rotPlane_at_points :
    (fun k => rotPlane k ![1, 0]) = ![0, 1] ∧ (fun k => rotPlane k ![-1, 0]) = ![0, -1] ∧
      (fun k => rotPlane k ![1, 0] + rotPlane k ![-1, 0]) = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> funext k <;> fin_cases k <;> simp [rotPlane]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.rotPlane_at_points, dg-vf1-rotplane-at-points-test]

/-- The polar "sum" `(0, 2) = 2 ∂_φ` has squared length `r² · 2² = 4` at `r = 1`. -/
theorem polar_sum_length (φ : ℝ) : polar 1 1 ![1, φ] * 2 ^ 2 = 4 := by
  simp [polar]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.polar_sum_length, dg-vf1-polar-sum-length-test]

/-- `∂_φ = −y ∂_x + x ∂_y`: the pushforward of the polar `∂_φ` is the rotation field of the plane. -/
theorem dphi_is_rotPlane (k : Fin 2) (x : Fin 2 → ℝ) :
    (∑ i, pd i (polarToCart k) x * (![0, 1] : Fin 2 → ℝ) i) = rotPlane k (fun l => polarToCart l x) := by
  have h := polar_jacobian k x
  fin_cases k <;> simp [Fin.sum_univ_two, h.2, rotPlane, polarToCart]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_is_rotPlane, dg-vf1-dphi-is-rotplane-test]

/-! ## exm-tv-height-points -/

/-- At `θ = π/3`, `X(Z) = −sin(π/3) = −√3/2` (the same declaration as above, `neg_sin_pi_div_three`). On the equator,
`X(Z) = −sin(π/2) = −1`. -/
theorem height_rate_equator : -Real.sin (Real.pi / 2) = -1 := by
  VERIPHY[dg-vf1-values]

/-- Near the poles the rate tends to `0`: `−sin 0 = 0` and `−sin π = 0`. -/
theorem height_rate_poles : -Real.sin 0 = 0 ∧ -Real.sin Real.pi = 0 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- `−1` is the largest possible rate of descent for a vector with `θ`-component `1`: `−1 ≤ −sin θ`. -/
theorem height_rate_bound (θ : ℝ) : -1 ≤ -Real.sin θ := by
  have h := Real.sin_le_one θ
  generalize Real.sin θ = s at h ⊢
  VERIPHY[dg-vf1-farkas]

/-- The `∂_φ` component never contributes: `∂_φ Z = 0`, since `Z` is constant on parallels. -/
theorem dphi_height (x : Fin 2 → ℝ) : pd 1 heightZ x = 0 := (dZ_components x).2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_height, dg-vf1-dphi-height-test]

/-! ## exm-tv-height-lean -/

/-- For any `X = X^θ ∂_θ + X^φ ∂_φ`, `X(Z) = X^θ ∂_θ Z + X^φ ∂_φ Z = −X^θ sin θ`. -/
theorem act_height (X : Fin 2 → ℝ) (x : Fin 2 → ℝ) : act X heightZ x = -(X 0) * Real.sin (x 0) := by
  have h := dZ_components x
  simp [act, Fin.sum_univ_succ, h.1, h.2]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height, dg-vf1-act-height-test]

/-- For the components `(3, −5)`, `X(Z) = −3 sin θ`. -/
theorem act_height_three (x : Fin 2 → ℝ) : act ![3, -5] heightZ x = -3 * Real.sin (x 0) := by
  rw [act_height]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height_three, dg-vf1-act-height-three-test]

/-- A multiple of `∂_φ` gives `X(Z) = 0`. -/
theorem act_height_dphi (c : ℝ) (x : Fin 2 → ℝ) : act ![0, c] heightZ x = 0 := by
  rw [act_height]; simp
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height_dphi, dg-vf1-act-height-dphi-test]

/-! ## exm-tv-height-dz -/

/-- `X(Z) = −a sin θ` is linear in `(a, b)` with coefficients `(−sin θ, 0)`, the components of `dZ`. -/
theorem act_height_linear (a b : ℝ) (x : Fin 2 → ℝ) :
    act ![a, b] heightZ x = -Real.sin (x 0) * a + 0 * b := by
  rw [act_height]; simp; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height_linear, dg-vf1-act-height-linear-test]

/-- The components of `dZ` are `(∂_θ Z, ∂_φ Z) = (−sin θ, 0)`. -/
theorem dZ_comps (x : Fin 2 → ℝ) : pd 0 heightZ x = -Real.sin (x 0) ∧ pd 1 heightZ x = 0 := dZ_components x
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dZ_comps, dg-vf1-dz-comps-test]

/-- Raising the index of `dZ = (−sin θ, 0)` with the round metric gives the gradient field `−sin θ ∂_θ`. -/
theorem gradient_of_height (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    (invMetric sphere x).mulVec ![-Real.sin (x 0), 0] = ![-Real.sin (x 0), 0] := raise_dZ x h
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.gradient_of_height, dg-vf1-gradient-of-height-test]

/-! ## exr-tv-height-general -/

/-- `X(Z²) = 2 Z X(Z) = −2 a sin θ cos θ`. -/
theorem act_height_sq (a b : ℝ) (x : Fin 2 → ℝ) :
    act ![a, b] (fun x => heightZ x ^ 2) x = -2 * a * Real.sin (x 0) * Real.cos (x 0) := by
  simp (disch := fun_prop) [act, pd, heightZ, Function.update_apply, Fin.sum_univ_succ]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height_sq, dg-vf1-act-height-sq-test]

/-- Away from the poles, `X(Z) = 0` exactly when `a = 0`: the vectors tangent to the parallel. -/
theorem height_preserving (a b : ℝ) (x : Fin 2 → ℝ) (h : Real.sin (x 0) ≠ 0) :
    act ![a, b] heightZ x = 0 ↔ a = 0 := by
  rw [act_height]; simp [h]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.height_preserving, dg-vf1-height-preserving-test]

/-! ## exr-tv-height-ambient -/

/-- `X = ∂_θ + 2 ∂_φ` on the three ambient coordinates gives the components of the ambient vector:
`(cos θ cos φ − 2 sin θ sin φ, cos θ sin φ + 2 sin θ cos φ, −sin θ)`. -/
theorem act_ambient_coords (k : Fin 3) (x : Fin 2 → ℝ) :
    act ![1, 2] (sphereEmbed k) x =
      (![Real.cos (x 0) * Real.cos (x 1) - 2 * (Real.sin (x 0) * Real.sin (x 1)),
        Real.cos (x 0) * Real.sin (x 1) + 2 * (Real.sin (x 0) * Real.cos (x 1)), -Real.sin (x 0)] : Fin 3 → ℝ) k := by
  have h := sphere_jacobian k x
  rw [act_combination, h.1, h.2]
  fin_cases k <;> simp <;> ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_ambient_coords, dg-vf1-act-ambient-coords-test]

/-- At `(π/3, π/4)` the first value is `cos(π/3) cos(π/4) − 2 sin(π/3) sin(π/4) = −√3√2/2 + √2/4`. -/
theorem act_ambient_value :
    Real.cos (Real.pi / 3) * Real.cos (Real.pi / 4) - 2 * (Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4)) =
      -(Real.sqrt 3 * Real.sqrt 2) / 2 + Real.sqrt 2 / 4 := by
  VERIPHY[dg-vf1-values]

/-! ## exr-tv-height-function -/

/-- The chain rule in the language of tangent vectors: `X(h(Z)) = h'(Z) X(Z) = −h'(cos θ) sin θ` for
`X = ∂_θ + 2 ∂_φ`. -/
theorem act_height_function (h : ℝ → ℝ) (hh : Differentiable ℝ h) (x : Fin 2 → ℝ) :
    act ![1, 2] (fun x => h (heightZ x)) x = -(deriv h (Real.cos (x 0))) * Real.sin (x 0) := by
  rw [act_combination]
  have h0 : pd 0 (fun x => h (heightZ x)) x = deriv h (Real.cos (x 0)) * -Real.sin (x 0) := by
    unfold pd heightZ
    simp only [Function.update_self]
    exact ((hh (Real.cos (x 0))).hasDerivAt.comp (x 0) (Real.hasDerivAt_cos (x 0))).deriv
  have h1 : pd 1 (fun x => h (heightZ x)) x = 0 := by
    unfold pd heightZ
    simp [Function.update_apply]
  rw [h0, h1]; ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.act_height_function, dg-vf1-act-height-function-test]

/-- For `h(Z) = Z³` at `θ = π/3`: `−3 cos²(π/3) sin(π/3) = −3√3/8`. -/
theorem act_height_cube_value :
    -(3 * Real.cos (Real.pi / 3) ^ 2) * Real.sin (Real.pi / 3) = -(3 * Real.sqrt 3 / 8) := by
  VERIPHY[dg-vf1-values]

/-! ## exm-tv-dphi-point -/

/-- At `(θ, φ) = (π/2, 0)`: `u = 1`, `v = 0`. -/
theorem stereo_point_one :
    Real.sin (Real.pi / 2) * Real.cos 0 / (1 - Real.cos (Real.pi / 2)) = 1 ∧
      Real.sin (Real.pi / 2) * Real.sin 0 / (1 - Real.cos (Real.pi / 2)) = 0 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- At `(θ, φ) = (π/2, π/2)`: `u = 0`, `v = 1`. -/
theorem stereo_point_two :
    Real.sin (Real.pi / 2) * Real.cos (Real.pi / 2) / (1 - Real.cos (Real.pi / 2)) = 0 ∧
      Real.sin (Real.pi / 2) * Real.sin (Real.pi / 2) / (1 - Real.cos (Real.pi / 2)) = 1 := by
  constructor
  · VERIPHY[dg-vf1-values]
  · VERIPHY[dg-vf1-values]

/-- The vector `(−v, u)` is tangent to the circle through `(u, v)`: `(−v) u + u v = 0`. -/
theorem rotation_tangent_circle (u v : ℝ) : -v * u + u * v = 0 := by
  VERIPHY[dg-vf1-poly]

/-! ## exm-tv-dphi-lean -/

/-- `∂_φ (u² + v²) = 2u(−v) + 2v u = 0`: the rotation field preserves the distance from the chart origin. -/
theorem dphi_radius (x : Fin 2 → ℝ) : pd 1 (fun x => stereoU x ^ 2 + stereoV x ^ 2) x = 0 := by
  have hU : Differentiable ℝ (fun t => stereoU (Function.update x 1 t)) := by
    unfold stereoU; simp [Function.update_apply]; fun_prop
  have hV : Differentiable ℝ (fun t => stereoV (Function.update x 1 t)) := by
    unfold stereoV; simp [Function.update_apply]; fun_prop
  have h1 := dphi_u x
  have h2 := dphi_v x
  unfold pd at h1 h2 ⊢
  have := ((hU (x 1)).hasDerivAt.pow 2).add ((hV (x 1)).hasDerivAt.pow 2)
  rw [this.deriv, h1, h2]
  simp only [Function.update_eq_self]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.dphi_radius, dg-vf1-dphi-radius-test]

/-- `u² + v² = sin² θ/(1 − cos θ)² = (1 + cos θ)/(1 − cos θ)`, where `cos θ ≠ 1`. -/
theorem stereo_radius_sq (θ : ℝ) (h : Real.cos θ ≠ 1) :
    Real.sin θ ^ 2 / (1 - Real.cos θ) ^ 2 = (1 + Real.cos θ) / (1 - Real.cos θ) := by
  have hne : 1 - Real.cos θ ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hs := Real.sin_sq_add_cos_sq θ
  rw [div_eq_div_iff (pow_ne_zero 2 hne) hne]
  generalize Real.sin θ = s at hs ⊢
  generalize Real.cos θ = c at hs ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- `(1 + cos θ)/(1 − cos θ) = cot²(θ/2)`: the distance from the chart origin is `cot(θ/2)`. -/
theorem stereo_radius_cot (θ : ℝ) (h : Real.sin (θ / 2) ≠ 0) :
    (1 + Real.cos θ) / (1 - Real.cos θ) = (Real.cos (θ / 2) / Real.sin (θ / 2)) ^ 2 := by
  have h1 : Real.cos θ = 2 * Real.cos (θ / 2) ^ 2 - 1 := by
    rw [← Real.cos_two_mul]; ring_nf
  have h2 := Real.sin_sq_add_cos_sq (θ / 2)
  have hne : 1 - Real.cos θ ≠ 0 := by
    rw [h1]; intro h0
    have : Real.sin (θ / 2) ^ 2 = 0 := by nlinarith
    exact h (pow_eq_zero_iff (two_ne_zero)).mp this
  rw [div_pow, div_eq_div_iff hne (pow_ne_zero 2 h), h1]
  generalize Real.sin (θ / 2) = s at h2 ⊢
  generalize Real.cos (θ / 2) = c at h2 ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-! ## exm-tv-dphi-rotation -/

/-- Along the rotation flow of the chart plane the radius is preserved:
`(u cos t − v sin t)² + (u sin t + v cos t)² = u² + v²`. -/
theorem rotation_preserves_radius (u v t : ℝ) :
    (u * Real.cos t - v * Real.sin t) ^ 2 + (u * Real.sin t + v * Real.cos t) ^ 2 = u ^ 2 + v ^ 2 := by
  have hs := Real.sin_sq_add_cos_sq t
  generalize Real.sin t = s at hs ⊢
  generalize Real.cos t = c at hs ⊢
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- `K₁` and `K₂` have quadratic stereographic components: at the chart origin they are `(0, 1/2)` and `(−1/2, 0)`,
not zero, while `K₃ = (−v, u)` vanishes there. -/
theorem k_stereo_origin :
    (fun k => k1Stereo k ![0, 0]) = ![0, 1 / 2] ∧ (fun k => k2Stereo k ![0, 0]) = ![-1 / 2, 0] ∧
      (fun k => rotPlane k ![0, 0]) = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> funext k <;> fin_cases k <;> simp [k1Stereo, k2Stereo, rotPlane] <;> norm_num
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.k_stereo_origin, dg-vf1-k-stereo-origin-test]

/-! ## exr-tv-dtheta-stereographic -/

/-- `ρ = cot(θ/2)` has `ρ' = −1/(2 sin²(θ/2))`. -/
theorem cot_half_deriv (θ : ℝ) (h : Real.sin (θ / 2) ≠ 0) :
    deriv (fun θ => Real.cos (θ / 2) / Real.sin (θ / 2)) θ = -1 / (2 * Real.sin (θ / 2) ^ 2) := by
  have h1 : HasDerivAt (fun θ => Real.cos (θ / 2)) (-Real.sin (θ / 2) * (1 / 2)) θ :=
    (Real.hasDerivAt_cos (θ / 2)).comp θ ((hasDerivAt_id θ).div_const 2)
  have h2 : HasDerivAt (fun θ => Real.sin (θ / 2)) (Real.cos (θ / 2) * (1 / 2)) θ :=
    (Real.hasDerivAt_sin (θ / 2)).comp θ ((hasDerivAt_id θ).div_const 2)
  rw [(h1.div h2 h).deriv, div_eq_div_iff (pow_ne_zero 2 h) (by positivity)]
  linear_combination (-(Real.sin (θ / 2) ^ 2)) * Real.sin_sq_add_cos_sq (θ / 2)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.cot_half_deriv, dg-vf1-cot-half-deriv-test]

/-- `−1/(2 sin²(θ/2)) = −(1 + ρ²)/2` with `ρ = cot(θ/2)`. -/
theorem cot_half_deriv_form (s c : ℝ) (hs : s ≠ 0) (h : s ^ 2 + c ^ 2 = 1) :
    -1 / (2 * s ^ 2) = -(1 + (c / s) ^ 2) / 2 := by
  field_simp
  rw [← sub_eq_zero]
  VERIPHY[dg-vf1-poly]

/-- `∂_θ = ρ'(cos φ ∂_u + sin φ ∂_v) = −((1 + ρ²)/(2ρ))(u ∂_u + v ∂_v)`: with `u = ρ cos φ`,
`ρ' cos φ = −((1 + ρ²)/(2ρ)) u` for `ρ ≠ 0`. -/
theorem dtheta_stereo_form (ρ φ : ℝ) (hρ : ρ ≠ 0) :
    -(1 + ρ ^ 2) / 2 * Real.cos φ = -((1 + ρ ^ 2) / (2 * ρ)) * (ρ * Real.cos φ) ∧
      -(1 + ρ ^ 2) / 2 * Real.sin φ = -((1 + ρ ^ 2) / (2 * ρ)) * (ρ * Real.sin φ) := by
  constructor
  · field_simp
    rw [← sub_eq_zero]
    VERIPHY[dg-vf1-poly]
  · field_simp
    rw [← sub_eq_zero]
    VERIPHY[dg-vf1-poly]

/-- On the equator `ρ = 1`, at `(u, v) = (1, 0)`, `∂_θ = −((1 + 1)/2)(1, 0) = (−1, 0) = −∂_u`. -/
theorem dtheta_equator : -((1 + 1 ^ 2) / (2 * 1) : ℝ) * 1 = -1 ∧ -((1 + 1 ^ 2) / (2 * 1) : ℝ) * 0 = 0 := by
  constructor
  · VERIPHY[dg-vf1-poly]
  · VERIPHY[dg-vf1-poly]

/-! ## exr-tv-rotation-stereographic -/

/-- `K₁ = −sin φ ∂_θ − cot θ cos φ ∂_φ` has stereographic components `(uv, (1 − u² + v²)/2)`, by the
transformation law, away from the poles. -/
def K1StereoLaw (x : Fin 2 → ℝ) : Prop :=
  Real.sin (x 0) ≠ 0 → Real.cos (x 0) ≠ 1 →
    (∑ i, pd i stereoU x * killingX i x) = stereoU x * stereoV x ∧
      (∑ i, pd i stereoV x * killingX i x) = (1 - stereoU x ^ 2 + stereoV x ^ 2) / 2
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.K1StereoLaw, dg-vf1-k1-stereo-law-test]

/-- `K₂ = cos φ ∂_θ − cot θ sin φ ∂_φ` has stereographic components `(−(1 + u² − v²)/2, −uv)`. -/
def K2StereoLaw (x : Fin 2 → ℝ) : Prop :=
  Real.sin (x 0) ≠ 0 → Real.cos (x 0) ≠ 1 →
    (∑ i, pd i stereoU x * killingY i x) = -(1 + stereoU x ^ 2 - stereoV x ^ 2) / 2 ∧
      (∑ i, pd i stereoV x * killingY i x) = -(stereoU x * stereoV x)
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.K2StereoLaw, dg-vf1-k2-stereo-law-test]

/-! ## exr-tv-south-chart -/

/-- In the south chart, `∂u'/∂φ = −v'`. -/
theorem south_dphi_u (x : Fin 2 → ℝ) : pd 1 stereoSU x = -stereoSV x := by
  simp (disch := fun_prop) [pd, stereoSU, stereoSV, Function.update_apply, deriv_div_const, deriv_const_mul,
    Real.deriv_cos]
  ring
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.south_dphi_u, dg-vf1-south-dphi-u-test]

/-- In the south chart, `∂v'/∂φ = u'`. -/
theorem south_dphi_v (x : Fin 2 → ℝ) : pd 1 stereoSV x = stereoSU x := by
  simp (disch := fun_prop) [pd, stereoSU, stereoSV, Function.update_apply, deriv_div_const, deriv_const_mul,
    Real.deriv_sin]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.south_dphi_v, dg-vf1-south-dphi-v-test]

/-- Any chart of the form `(ρ(θ) cos φ, ρ(θ) sin φ)` gives `∂_φ = −v ∂_u + u ∂_v`, because `φ` enters only
through `(cos φ, sin φ)`. -/
theorem polar_form_dphi (ρ : ℝ → ℝ) (x : Fin 2 → ℝ) :
    pd 1 (fun x => ρ (x 0) * Real.cos (x 1)) x = -(ρ (x 0) * Real.sin (x 1)) ∧
      pd 1 (fun x => ρ (x 0) * Real.sin (x 1)) x = ρ (x 0) * Real.cos (x 1) := by
  constructor <;> simp (disch := fun_prop) [pd, Function.update_apply]
VERIPHY_TEST[KaluzaKlein.DifferentialGeometry.TangentVectors.polar_form_dphi, dg-vf1-polar-form-dphi-test]

end KaluzaKlein.DifferentialGeometry.TangentVectors

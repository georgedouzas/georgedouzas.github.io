/-
Differential geometry in coordinates, with mathlib only.

A recipe asset: a project copies this file to `.veriphy/lean/VeriphyAssets/CoordinateGeometry.lean` and declares the
`VeriphyAssets` library (see the user guide). Everything is stated on coordinates `Fin n → ℝ` through mathlib's
`deriv` and `Matrix` API, so the definitions can also be proposed upstream as they are.

* `pd i f x`: the partial derivative of `f` along the `i`-th coordinate at `x`.
* A metric is a family `g : Fin n → Fin n → (Fin n → ℝ) → ℝ` of component functions; `metricAt g x` is its matrix at
  `x` and `invMetric g x` the inverse (mathlib's `Matrix.inv`, so `invMetric g x * metricAt g x = 1` wherever the
  determinant is nonzero).
* `christoffel g k i j x` is `Γᵏᵢⱼ`; `riemann g ρ σ μ ν x` is `Rᵖ_σμν`; `ricci g σ ν x` and `scalarCurvature g x`.
* `lieMetric g X i j x` is the Lie derivative `(L_X g)ᵢⱼ`; `X` is a Killing vector when it vanishes.
* Differential `k`-forms are component functions `ω : (Fin k → Fin n) → (Fin n → ℝ) → ℝ`; `extDeriv`, `wedge` and
  `hodge` are the exterior derivative, the wedge product and the Hodge star for a metric.
-/
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.GroupTheory.Perm.Sign
import Mathlib.Logic.Equiv.Fin.Basic

namespace CoordinateGeometry

open Matrix

variable {n : ℕ}

/-- The partial derivative of `f` along the `i`-th coordinate at `x`. -/
noncomputable def pd (i : Fin n) (f : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : ℝ :=
  deriv (fun t => f (Function.update x i t)) (x i)

/-- A metric's matrix at a point. -/
def metricAt (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => g i j x

/-- The inverse metric at a point. -/
noncomputable def invMetric (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (metricAt g x)⁻¹

/-- The Christoffel symbol of the second kind, `Γᵏᵢⱼ = ½ gᵏˡ (∂ᵢ gₗⱼ + ∂ⱼ gₗᵢ - ∂ₗ gᵢⱼ)`. -/
noncomputable def christoffel (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (k i j : Fin n) (x : Fin n → ℝ) : ℝ :=
  (1 / 2) * ∑ l, invMetric g x k l * (pd i (g l j) x + pd j (g l i) x - pd l (g i j) x)

/-- The Riemann tensor `Rᵖ_σμν = ∂_μ Γᵖ_νσ - ∂_ν Γᵖ_μσ + Γᵖ_μλ Γᵏ_νσ - Γᵖ_νλ Γᵏ_μσ` (`λ` summed). -/
noncomputable def riemann (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) : ℝ :=
  pd μ (christoffel g ρ ν σ) x - pd ν (christoffel g ρ μ σ) x +
    ∑ l, (christoffel g ρ μ l x * christoffel g l ν σ x - christoffel g ρ ν l x * christoffel g l μ σ x)

/-- The Ricci tensor `R_σν = Rᵖ_σρν`. -/
noncomputable def ricci (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (σ ν : Fin n) (x : Fin n → ℝ) : ℝ :=
  ∑ ρ, riemann g ρ σ ρ ν x

/-- The scalar curvature `R = gᵖᵛ R_σν`. -/
noncomputable def scalarCurvature (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : ℝ :=
  ∑ σ, ∑ ν, invMetric g x σ ν * ricci g σ ν x

/-- The Lie derivative of the metric along a vector field, `(L_X g)ᵢⱼ = Xᵏ ∂ₖ gᵢⱼ + gₖⱼ ∂ᵢ Xᵏ + gᵢₖ ∂ⱼ Xᵏ`. -/
noncomputable def lieMetric (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ) (i j : Fin n)
    (x : Fin n → ℝ) : ℝ :=
  ∑ k, (X k x * pd k (g i j) x + g k j x * pd i (X k) x + g i k x * pd j (X k) x)

/-- `X` is a Killing vector field of `g`: its Lie derivative vanishes everywhere. -/
def IsKilling (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (X : Fin n → (Fin n → ℝ) → ℝ) : Prop :=
  ∀ i j x, lieMetric g X i j x = 0

/-! ## Differential forms in coordinates -/

/-- The exterior derivative of a `k`-form: `(dω)_{i₀ … iₖ} = Σₐ (-1)ᵃ ∂_{iₐ} ω_{i₀ … îₐ … iₖ}`. -/
noncomputable def extDeriv {k : ℕ} (ω : (Fin k → Fin n) → (Fin n → ℝ) → ℝ) :
    (Fin (k + 1) → Fin n) → (Fin n → ℝ) → ℝ :=
  fun I x => ∑ a : Fin (k + 1), (-1 : ℝ) ^ (a : ℕ) * pd (I a) (ω (Fin.removeNth a I)) x

/-- The wedge product, `(α ∧ β)_I = (1 / (p! q!)) Σ_σ sign σ · α_{σ I…} β_{σ I…}`. -/
noncomputable def wedge {p q : ℕ} (α : (Fin p → Fin n) → (Fin n → ℝ) → ℝ) (β : (Fin q → Fin n) → (Fin n → ℝ) → ℝ) :
    (Fin (p + q) → Fin n) → (Fin n → ℝ) → ℝ :=
  fun I x => (1 / ((p.factorial * q.factorial : ℕ) : ℝ)) * ∑ σ : Equiv.Perm (Fin (p + q)),
    (Equiv.Perm.sign σ : ℝ) * α (fun a => I (σ (Fin.castAdd q a))) x * β (fun b => I (σ (Fin.natAdd p b))) x

/-- The Levi-Civita symbol of an index list: the sign of the permutation it is, `0` when an index repeats. -/
noncomputable def levi (I : Fin n → Fin n) : ℝ :=
  if h : Function.Bijective I then (Equiv.Perm.sign (Equiv.ofBijective I h) : ℝ) else 0

/-- The Hodge star of a `k`-form for a metric, on an `n = k + m` dimensional space:
`(⋆ω)_J = (√|det g| / k!) Σ_I ω^I ε_{I J}`, with `ω^I` raised by the inverse metric. -/
noncomputable def hodge {k m : ℕ} (g : Fin (k + m) → Fin (k + m) → (Fin (k + m) → ℝ) → ℝ)
    (ω : (Fin k → Fin (k + m)) → (Fin (k + m) → ℝ) → ℝ) :
    (Fin m → Fin (k + m)) → (Fin (k + m) → ℝ) → ℝ :=
  fun J x => Real.sqrt |(metricAt g x).det| / (k.factorial : ℝ) *
    ∑ I : Fin k → Fin (k + m), ∑ L : Fin k → Fin (k + m),
      (∏ a, invMetric g x (I a) (L a)) * ω L x * levi (Fin.append I J ∘ finCongr (by omega))

/-! ## Sanity checks -/

theorem pd_const (i : Fin n) (c : ℝ) (x : Fin n → ℝ) : pd i (fun _ => c) x = 0 := by
  simp [pd]

/-- A constant metric (flat coordinates) has vanishing Christoffel symbols. -/
theorem christoffel_const (c : Matrix (Fin n) (Fin n) ℝ) (k i j : Fin n) (x : Fin n → ℝ) :
    christoffel (fun i j _ => c i j) k i j x = 0 := by
  simp [christoffel, pd_const]

/-- A constant metric is flat: its Riemann tensor vanishes. -/
theorem riemann_const (c : Matrix (Fin n) (Fin n) ℝ) (ρ σ μ ν : Fin n) (x : Fin n → ℝ) :
    riemann (fun i j _ => c i j) ρ σ μ ν x = 0 := by
  have h : ∀ k i j, christoffel (fun i j _ => c i j) k i j = fun _ => 0 :=
    fun k i j => funext fun y => christoffel_const c k i j y
  simp [riemann, h, pd_const]

/-- The exterior derivative of a constant form vanishes. -/
theorem extDeriv_const {k : ℕ} (c : (Fin k → Fin n) → ℝ) (I : Fin (k + 1) → Fin n) (x : Fin n → ℝ) :
    extDeriv (fun J _ => c J) I x = 0 := by
  simp [extDeriv, pd_const]

end CoordinateGeometry

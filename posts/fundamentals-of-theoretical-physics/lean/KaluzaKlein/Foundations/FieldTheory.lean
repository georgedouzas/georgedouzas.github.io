import KaluzaKlein.Conventions

/-!
# Field theory in jet variables

Identities of classical field theory that hold at every point are polynomial identities in the **jet variables**:
the fields and their derivatives at the point, treated as independent symbols. We verify them in that form.

* `noether_divergence`: for a complex scalar `φ = u + i v` the `U(1)` current `j^μ = i (φ̄ ∂^μ φ − φ ∂^μ φ̄)` has
  divergence `i (φ̄ □φ − φ □φ̄)`, which vanishes on the Klein-Gordon equation `(□ + m²) φ = 0`, that is `□φ = −m² φ`
  with `□ = η^{μν} ∂_μ ∂_ν` in the signature `(+, −, −, −)` (the `η` below is the diagonal of the inverse metric).
* `abelian_F_invariant`: `F_{μν} = ∂_μ A_ν − ∂_ν A_μ` is unchanged by `A → A + ∂ε` (second derivatives commute).
* `su2_field_strength`: the `SU(2)` field strength `F^a_{μν} = ∂_μ A^a_ν − ∂_ν A^a_μ + g ε_abc A^b_μ A^c_ν`, and its value on
  a constant configuration with `A¹₁ = c₁`, `A²₂ = c₂`: `F³₁₂ = g c₁ c₂`, all other components zero.
* `su2_bianchi`: the Bianchi identity `D_ρ F_{μν} + D_μ F_{νρ} + D_ν F_{ρμ} = 0` for every `SU(2)` gauge field, as a
  polynomial identity in the jets `A`, `∂A`, `∂∂A` (with symmetric second derivatives), by the Jacobi identity of `ε`.
-/

namespace KaluzaKlein.FieldTheory

/-! ## The Noether current of a complex scalar -/

/-- Real and imaginary parts of `φ`, its gradient and its d'Alembertian are independent jet variables; the
divergence of `j^μ = i(φ̄ ∂^μ φ − φ ∂^μ φ̄) = 2 (v ∂^μ u − u ∂^μ v)`, written below up to its overall sign as
`2 (u ∂^μ v − v ∂^μ u)`, is `2 (u □v − v □u)`: the gradient terms cancel. Here `du dv : Fin 4 → ℝ` are the gradients,
`ddu ddv : Fin 4 → Fin 4 → ℝ` the Hessians and `η` the diagonal of the inverse metric, `diag(1, −1, −1, −1)`. -/
theorem noether_divergence (u v : ℝ) (du dv : Fin 4 → ℝ) (ddu ddv : Fin 4 → Fin 4 → ℝ) (η : Fin 4 → ℝ) :
    ∑ μ, η μ * (du μ * dv μ + u * ddv μ μ - dv μ * du μ - v * ddu μ μ) =
      u * ∑ μ, η μ * ddv μ μ - v * ∑ μ, η μ * ddu μ μ := by
  simp only [Finset.mul_sum, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun μ _ => by ring

/-- On the Klein-Gordon equation `□u = −m² u`, `□v = −m² v` the divergence vanishes. -/
theorem noether_conserved (u v m : ℝ) (boxu boxv : ℝ) (hu : boxu = -m ^ 2 * u) (hv : boxv = -m ^ 2 * v) :
    u * boxv - v * boxu = 0 := by
  subst hu hv; ring

/-! ## Abelian gauge invariance of the field strength -/

/-- `F_{μν} = ∂_μ A_ν − ∂_ν A_μ` is invariant under `A_ν → A_ν + ∂_ν ε` when `∂_μ ∂_ν ε = ∂_ν ∂_μ ε`. -/
theorem abelian_F_invariant (dA : Fin 4 → Fin 4 → ℝ) (ddε : Fin 4 → Fin 4 → ℝ) (hsymm : ∀ μ ν, ddε μ ν = ddε ν μ)
    (μ ν : Fin 4) : (dA μ ν + ddε μ ν) - (dA ν μ + ddε ν μ) = dA μ ν - dA ν μ := by
  rw [hsymm μ ν]; ring

/-! ## The `SU(2)` field strength -/

/-- The jets of an `SU(2)` gauge field at a point: `A a μ = A^a_μ`, `dA ρ a μ = ∂_ρ A^a_μ`, `ddA σ ρ a μ = ∂_σ ∂_ρ A^a_μ`. -/
structure Jets where
  A : Fin 3 → Fin 4 → ℝ
  dA : Fin 4 → Fin 3 → Fin 4 → ℝ
  ddA : Fin 4 → Fin 4 → Fin 3 → Fin 4 → ℝ
  symm : ∀ σ ρ a μ, ddA σ ρ a μ = ddA ρ σ a μ

/-- `F^a_{μν} = ∂_μ A^a_ν − ∂_ν A^a_μ + g ε_abc A^b_μ A^c_ν`. -/
def F (g : ℝ) (J : Jets) (a : Fin 3) (μ ν : Fin 4) : ℝ :=
  J.dA μ a ν - J.dA ν a μ + g * ∑ b, ∑ c, (eps a b c : ℝ) * J.A b μ * J.A c ν

/-- `∂_ρ F^a_{μν}`, differentiating the definition with the product rule. -/
def dF (g : ℝ) (J : Jets) (ρ : Fin 4) (a : Fin 3) (μ ν : Fin 4) : ℝ :=
  J.ddA ρ μ a ν - J.ddA ρ ν a μ + g * ∑ b, ∑ c, (eps a b c : ℝ) * (J.dA ρ b μ * J.A c ν + J.A b μ * J.dA ρ c ν)

/-- The covariant derivative `D_ρ F^a_{μν} = ∂_ρ F^a_{μν} + g ε_abc A^b_ρ F^c_{μν}`. -/
def DF (g : ℝ) (J : Jets) (ρ : Fin 4) (a : Fin 3) (μ ν : Fin 4) : ℝ :=
  dF g J ρ a μ ν + g * ∑ b, ∑ c, (eps a b c : ℝ) * J.A b ρ * F g J c μ ν

/-- The constant configuration `A¹₁ = c₁`, `A²₂ = c₂`, all other components zero. -/
def constantJets (c₁ c₂ : ℝ) : Jets where
  A := fun a μ => if a = 0 ∧ μ = 1 then c₁ else if a = 1 ∧ μ = 2 then c₂ else 0
  dA := fun _ _ _ => 0
  ddA := fun _ _ _ _ => 0
  symm := fun _ _ _ _ => rfl

/-- On the constant configuration only the commutator term survives: `F³₁₂ = g c₁ c₂ = −F³₂₁`. -/
theorem su2_field_strength (g c₁ c₂ : ℝ) :
    F g (constantJets c₁ c₂) 2 1 2 = g * c₁ * c₂ ∧ F g (constantJets c₁ c₂) 2 2 1 = -(g * c₁ * c₂) ∧
      F g (constantJets c₁ c₂) 0 1 2 = 0 ∧ F g (constantJets c₁ c₂) 1 1 2 = 0 ∧ F g (constantJets c₁ c₂) 2 0 3 = 0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> simp [F, constantJets, eps, Fin.sum_univ_succ] <;> ring

/-- The Bianchi identity `D_ρ F_{μν} + D_μ F_{νρ} + D_ν F_{ρμ} = 0`, for every gauge field and every coupling. -/
theorem su2_bianchi (g : ℝ) (J : Jets) (a : Fin 3) (ρ μ ν : Fin 4) :
    DF g J ρ a μ ν + DF g J μ a ν ρ + DF g J ν a ρ μ = 0 := by
  -- the sums run over the Lie-algebra indices only: split on `a`, keep `ρ μ ν` symbolic, and cancel the second
  -- derivatives with their symmetry; the `A A A` terms cancel by the Jacobi identity of `ε`, a polynomial identity
  fin_cases a
  · simp only [DF, dF, F, eps, Fin.sum_univ_succ, Fin.sum_univ_zero]
    simp
    linear_combination J.symm ρ μ 0 ν + J.symm μ ν 0 ρ + J.symm ν ρ 0 μ
  · simp only [DF, dF, F, eps, Fin.sum_univ_succ, Fin.sum_univ_zero]
    simp
    linear_combination J.symm ρ μ 1 ν + J.symm μ ν 1 ρ + J.symm ν ρ 1 μ
  · simp only [DF, dF, F, eps, Fin.sum_univ_succ, Fin.sum_univ_zero]
    simp
    linear_combination J.symm ρ μ 2 ν + J.symm μ ν 2 ρ + J.symm ν ρ 2 μ

end KaluzaKlein.FieldTheory

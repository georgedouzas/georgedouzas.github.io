import Mathlib

/-!
# The abelian Kaluza-Klein reduction as polynomial identities in jets

Design D3 of the tutorial: the ansatz `ds² = g_{μν}(x) dx^μ dx^ν - φ(x)² (dy + A_μ(x) dx^μ)²` (signature `(+,-,-,-,-)`:
the circle direction is spacelike, so the internal block carries a minus sign) has arbitrary functions,
so the reduction is verified as an identity in the *jets* at a point: the values and first and second partial
derivatives of `g`, `A`, `φ`, together with symbols `ginv` for the inverse metric whose derivative is the formal
`-ginv ∂g ginv`. The five-dimensional Christoffel symbols, Ricci tensor and curvature scalar are the standard
formulas evaluated on these symbols (index type `Fin 4 ⊕ Unit`, the summand `Unit` being the circle direction `y`;
the cylinder condition sets every `y`-derivative to zero). Everything is a polynomial identity: the inverse
metric symbols never need the relation `g ginv = 1`, only `φ ≠ 0` is used to clear `φ⁻¹`. Every identity depends on
`φ` only through `φ²`, so the statements are those of the mostly-plus literature with `φ² ↦ -φ²`.
The oracle `sage/kk_jets.sage` computes the same objects.
-/

set_option maxHeartbeats 0

noncomputable section

namespace KaluzaKlein.AbelianKK

/-- Jets at a point: the metric `g`, its inverse `ginv`, first and second derivatives `dg ρ μ ν = ∂_ρ g_{μν}`,
`ddg ρ σ μ ν = ∂_ρ ∂_σ g_{μν}`, the vector `A` with `dA ρ μ = ∂_ρ A_μ`, `ddA ρ σ μ`, and the scalar `φ`, `dφ`, `ddφ`.
The stored components are unconstrained; the symmetrised accessors below make the symmetries hold identically. -/
structure Jets where
  g : Fin 4 → Fin 4 → ℝ
  ginv : Fin 4 → Fin 4 → ℝ
  dg : Fin 4 → Fin 4 → Fin 4 → ℝ
  ddg : Fin 4 → Fin 4 → Fin 4 → Fin 4 → ℝ
  A : Fin 4 → ℝ
  dA : Fin 4 → Fin 4 → ℝ
  ddA : Fin 4 → Fin 4 → Fin 4 → ℝ
  φ : ℝ
  dφ : Fin 4 → ℝ
  ddφ : Fin 4 → Fin 4 → ℝ

namespace Jets

variable (J : Jets)

/-! ### Symmetrised jets -/

/-- `g_{μν}`, symmetrised. -/
def gS (μ ν : Fin 4) : ℝ := (J.g μ ν + J.g ν μ) / 2
/-- `g^{μν}`, symmetrised. -/
def ginvS (μ ν : Fin 4) : ℝ := (J.ginv μ ν + J.ginv ν μ) / 2
/-- `∂_ρ g_{μν}`, symmetric in `μ ν`. -/
def dgS (ρ μ ν : Fin 4) : ℝ := (J.dg ρ μ ν + J.dg ρ ν μ) / 2
/-- `∂_ρ ∂_σ g_{μν}`, symmetric in `ρ σ` and in `μ ν`. -/
def ddgS (ρ σ μ ν : Fin 4) : ℝ :=
  (J.ddg ρ σ μ ν + J.ddg σ ρ μ ν + J.ddg ρ σ ν μ + J.ddg σ ρ ν μ) / 4
/-- `∂_ρ ∂_σ A_μ`, symmetric in `ρ σ`. -/
def ddAS (ρ σ μ : Fin 4) : ℝ := (J.ddA ρ σ μ + J.ddA σ ρ μ) / 2
/-- `∂_ρ ∂_σ φ`, symmetric. -/
def ddφS (ρ σ : Fin 4) : ℝ := (J.ddφ ρ σ + J.ddφ σ ρ) / 2
/-- `∂_ρ g^{μν} = -g^{μα} ∂_ρ g_{αβ} g^{βν}`, the formal derivative of the inverse metric. -/
def dginv (ρ μ ν : Fin 4) : ℝ := -∑ a, ∑ b, J.ginvS μ a * J.dgS ρ a b * J.ginvS b ν

/-! ### Four-dimensional geometry -/

/-- `Γ^k_{ij}` of the four-dimensional metric. -/
def Γ4 (k i j : Fin 4) : ℝ := (1 / 2) * ∑ l, J.ginvS k l * (J.dgS i l j + J.dgS j l i - J.dgS l i j)
/-- `∂_ρ Γ^k_{ij}` by the product rule. -/
def dΓ4 (ρ k i j : Fin 4) : ℝ :=
  (1 / 2) * ∑ l, (J.dginv ρ k l * (J.dgS i l j + J.dgS j l i - J.dgS l i j)
    + J.ginvS k l * (J.ddgS ρ i l j + J.ddgS ρ j l i - J.ddgS ρ l i j))
/-- `R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{kl} Γ^l_{ij} - Γ^k_{jl} Γ^l_{ik}`. -/
def Ric4 (i j : Fin 4) : ℝ :=
  ∑ k, J.dΓ4 k k i j - ∑ k, J.dΓ4 j k i k + ∑ k, ∑ l, J.Γ4 k k l * J.Γ4 l i j - ∑ k, ∑ l, J.Γ4 k j l * J.Γ4 l i k
/-- The four-dimensional curvature scalar. -/
def R4 : ℝ := ∑ i, ∑ j, J.ginvS i j * J.Ric4 i j
/-- `F_{μν} = ∂_μ A_ν - ∂_ν A_μ`. -/
def F (μ ν : Fin 4) : ℝ := J.dA μ ν - J.dA ν μ
/-- `F_μ{}^k = g^{kl} F_{μl}`. -/
def Fup (μ k : Fin 4) : ℝ := ∑ l, J.ginvS k l * J.F μ l
/-- `F_{μν} F^{μν}`. -/
def F2 : ℝ := ∑ μ, ∑ ν, ∑ k, ∑ l, J.ginvS μ k * J.ginvS ν l * J.F μ ν * J.F k l
/-- `F_μ{}^k F_{νk}`. -/
def FF (μ ν : Fin 4) : ℝ := ∑ k, ∑ l, J.ginvS k l * J.F μ k * J.F ν l
/-- `∂_ρ F_{μν}`. -/
def dF (ρ μ ν : Fin 4) : ℝ := J.ddAS ρ μ ν - J.ddAS ρ ν μ
/-- `∇^μ F_{μν}`. -/
def divF (ν : Fin 4) : ℝ :=
  ∑ μ, ∑ k, J.ginvS μ k * (J.dF μ k ν - ∑ l, (J.Γ4 l μ k * J.F l ν + J.Γ4 l μ ν * J.F k l))
/-- `∂^k φ`. -/
def dφup (k : Fin 4) : ℝ := ∑ l, J.ginvS k l * J.dφ l
/-- `A^k`. -/
def Aup (k : Fin 4) : ℝ := ∑ l, J.ginvS k l * J.A l
/-- `∇_μ ∂_ν φ`. -/
def nabla2φ (μ ν : Fin 4) : ℝ := J.ddφS μ ν - ∑ k, J.Γ4 k μ ν * J.dφ k
/-- `□φ = g^{μν} ∇_μ ∂_ν φ`. -/
def boxφ : ℝ := ∑ μ, ∑ ν, J.ginvS μ ν * J.nabla2φ μ ν

/-! ### The five-dimensional ansatz -/

/-- Five-dimensional indices: `inl μ` for spacetime, `inr ()` for the circle coordinate `y`. -/
abbrev I5 := Fin 4 ⊕ Unit

/-- The ansatz metric `G_{μν} = g_{μν} - φ² A_μ A_ν`, `G_{μy} = -φ² A_μ`, `G_{yy} = -φ²`. -/
def G5 : I5 → I5 → ℝ
  | Sum.inl μ, Sum.inl ν => J.gS μ ν - J.φ ^ 2 * J.A μ * J.A ν
  | Sum.inl μ, Sum.inr _ => -(J.φ ^ 2 * J.A μ)
  | Sum.inr _, Sum.inl ν => -(J.φ ^ 2 * J.A ν)
  | Sum.inr _, Sum.inr _ => -(J.φ ^ 2)
/-- Its inverse `G^{μν} = g^{μν}`, `G^{μy} = -A^μ`, `G^{yy} = -φ⁻² + A_μ A^μ`. -/
def Ginv5 : I5 → I5 → ℝ
  | Sum.inl μ, Sum.inl ν => J.ginvS μ ν
  | Sum.inl μ, Sum.inr _ => -J.Aup μ
  | Sum.inr _, Sum.inl ν => -J.Aup ν
  | Sum.inr _, Sum.inr _ => -(J.φ⁻¹ ^ 2) + ∑ k, ∑ l, J.ginvS k l * J.A k * J.A l
/-- `∂_ρ G_{ij}` (zero for `ρ = y`, the cylinder condition), by the product rule. -/
def dG5 : I5 → I5 → I5 → ℝ
  | Sum.inr _, _, _ => 0
  | Sum.inl ρ, Sum.inl μ, Sum.inl ν =>
      J.dgS ρ μ ν - 2 * J.φ * J.dφ ρ * J.A μ * J.A ν - J.φ ^ 2 * (J.dA ρ μ * J.A ν + J.A μ * J.dA ρ ν)
  | Sum.inl ρ, Sum.inl μ, Sum.inr _ => -(2 * J.φ * J.dφ ρ * J.A μ + J.φ ^ 2 * J.dA ρ μ)
  | Sum.inl ρ, Sum.inr _, Sum.inl ν => -(2 * J.φ * J.dφ ρ * J.A ν + J.φ ^ 2 * J.dA ρ ν)
  | Sum.inl ρ, Sum.inr _, Sum.inr _ => -(2 * J.φ * J.dφ ρ)
/-- `∂_ρ ∂_σ G_{ij}` by the product rule. -/
def ddG5 : I5 → I5 → I5 → I5 → ℝ
  | Sum.inr _, _, _, _ => 0
  | Sum.inl _, Sum.inr _, _, _ => 0
  | Sum.inl ρ, Sum.inl σ, Sum.inl μ, Sum.inl ν =>
      J.ddgS ρ σ μ ν - 2 * (J.dφ ρ * J.dφ σ + J.φ * J.ddφS ρ σ) * J.A μ * J.A ν
      - 2 * J.φ * J.dφ ρ * (J.dA σ μ * J.A ν + J.A μ * J.dA σ ν)
      - 2 * J.φ * J.dφ σ * (J.dA ρ μ * J.A ν + J.A μ * J.dA ρ ν)
      - J.φ ^ 2 * (J.ddAS ρ σ μ * J.A ν + J.dA ρ μ * J.dA σ ν + J.dA σ μ * J.dA ρ ν + J.A μ * J.ddAS ρ σ ν)
  | Sum.inl ρ, Sum.inl σ, Sum.inl μ, Sum.inr _ =>
      -(2 * (J.dφ ρ * J.dφ σ + J.φ * J.ddφS ρ σ) * J.A μ + 2 * J.φ * J.dφ ρ * J.dA σ μ
      + 2 * J.φ * J.dφ σ * J.dA ρ μ + J.φ ^ 2 * J.ddAS ρ σ μ)
  | Sum.inl ρ, Sum.inl σ, Sum.inr _, Sum.inl ν =>
      -(2 * (J.dφ ρ * J.dφ σ + J.φ * J.ddφS ρ σ) * J.A ν + 2 * J.φ * J.dφ ρ * J.dA σ ν
      + 2 * J.φ * J.dφ σ * J.dA ρ ν + J.φ ^ 2 * J.ddAS ρ σ ν)
  | Sum.inl ρ, Sum.inl σ, Sum.inr _, Sum.inr _ => -(2 * (J.dφ ρ * J.dφ σ + J.φ * J.ddφS ρ σ))
/-- `∂_ρ G^{ij}`, the formal derivative of `Ginv5`. -/
def dGinv5 : I5 → I5 → I5 → ℝ
  | Sum.inr _, _, _ => 0
  | Sum.inl ρ, Sum.inl μ, Sum.inl ν => J.dginv ρ μ ν
  | Sum.inl ρ, Sum.inl μ, Sum.inr _ => -∑ k, (J.dginv ρ μ k * J.A k + J.ginvS μ k * J.dA ρ k)
  | Sum.inl ρ, Sum.inr _, Sum.inl ν => -∑ k, (J.dginv ρ ν k * J.A k + J.ginvS ν k * J.dA ρ k)
  | Sum.inl ρ, Sum.inr _, Sum.inr _ =>
      2 * J.φ⁻¹ ^ 3 * J.dφ ρ
      + ∑ k, ∑ l, (J.dginv ρ k l * J.A k * J.A l + J.ginvS k l * (J.dA ρ k * J.A l + J.A k * J.dA ρ l))
/-- The five-dimensional Christoffel symbols. -/
def Γ5 (k i j : I5) : ℝ := (1 / 2) * ∑ l, J.Ginv5 k l * (J.dG5 i l j + J.dG5 j l i - J.dG5 l i j)
/-- `∂_ρ Γ^k_{ij}` in five dimensions, by the product rule. -/
def dΓ5 (ρ k i j : I5) : ℝ :=
  (1 / 2) * ∑ l, (J.dGinv5 ρ k l * (J.dG5 i l j + J.dG5 j l i - J.dG5 l i j)
    + J.Ginv5 k l * (J.ddG5 ρ i l j + J.ddG5 ρ j l i - J.ddG5 ρ l i j))
/-- The five-dimensional Ricci tensor, standard formula. -/
def Ric5 (i j : I5) : ℝ :=
  ∑ k, J.dΓ5 k k i j - ∑ k, J.dΓ5 j k i k + ∑ k, ∑ l, J.Γ5 k k l * J.Γ5 l i j - ∑ k, ∑ l, J.Γ5 k j l * J.Γ5 l i k
/-- The five-dimensional curvature scalar. -/
def R5 : ℝ := ∑ i, ∑ j, J.Ginv5 i j * J.Ric5 i j

/-! ### The Christoffel symbols of the ansatz (D3 step 2) -/

open Sum

/-- `Γ^λ_{μν} = Γ4^λ_{μν} - ½φ²(A_μ F_ν{}^λ + A_ν F_μ{}^λ) + φ A_μ A_ν ∂^λφ`. -/
theorem Γ5_lll (k μ ν : Fin 4) :
    J.Γ5 (inl k) (inl μ) (inl ν) =
      J.Γ4 k μ ν - J.φ ^ 2 / 2 * (J.A μ * J.Fup ν k + J.A ν * J.Fup μ k) + J.φ * J.A μ * J.A ν * J.dφup k := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, Γ4, Fup, F, dφup, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  ring

/-- `Γ^λ_{μy} = -½φ² F_μ{}^λ + φ A_μ ∂^λφ`. -/
theorem Γ5_lly (k μ : Fin 4) :
    J.Γ5 (inl k) (inl μ) (inr ()) = -(J.φ ^ 2 / 2 * J.Fup μ k) + J.φ * J.A μ * J.dφup k := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, Fup, F, dφup, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  ring

/-- `Γ^λ_{yy} = φ ∂^λφ`. -/
theorem Γ5_lyy (k : Fin 4) : J.Γ5 (inl k) (inr ()) (inr ()) = J.φ * J.dφup k := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, dφup, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  ring

/-- `Γ^y_{μν} = φ⁻¹(A_μ ∂_νφ + A_ν ∂_μφ) + ½(∂_μ A_ν + ∂_ν A_μ) - A_λ Γ^λ_{μν}`. -/
theorem Γ5_yll (μ ν : Fin 4) (hφ : J.φ ≠ 0) :
    J.Γ5 (inr ()) (inl μ) (inl ν) =
      J.φ⁻¹ * (J.A μ * J.dφ ν + J.A ν * J.dφ μ) + (J.dA μ ν + J.dA ν μ) / 2
        - ∑ k, J.A k * J.Γ5 (inl k) (inl μ) (inl ν) := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  field_simp
  ring

/-- `Γ^y_{μy} = φ⁻¹ ∂_μφ - A_λ Γ^λ_{μy}`. -/
theorem Γ5_yly (μ : Fin 4) (hφ : J.φ ≠ 0) :
    J.Γ5 (inr ()) (inl μ) (inr ()) = J.φ⁻¹ * J.dφ μ - ∑ k, J.A k * J.Γ5 (inl k) (inl μ) (inr ()) := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  field_simp
  ring

/-- `Γ^y_{yy} = -A_λ Γ^λ_{yy}`. -/
theorem Γ5_yyy : J.Γ5 (inr ()) (inr ()) (inr ()) = -∑ k, J.A k * J.Γ5 (inl k) (inr ()) (inr ()) := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Γ5, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5, dG5, Aup]
  ring

/-! ### The Ricci tensor of the ansatz (D3 step 3) -/

/-- `R_{yy} = φ □φ + ¼φ⁴ F_{μν}F^{μν}`. -/
theorem Ric5_yy (hφ : J.φ ≠ 0) : J.Ric5 (inr ()) (inr ()) = J.φ * J.boxφ + J.φ ^ 4 / 4 * J.F2 := by
  simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Ric5, dΓ5, Γ5, boxφ, nabla2φ, Γ4, F2, F, Fintype.sum_sum_type, Fintype.sum_unique,
    Fin.sum_univ_four, Ginv5, dG5, ddG5, dGinv5, dginv, Aup]
  field_simp
  ring

/-- `R_{μy} = A_μ R_{yy} + ½φ² ∇^k F_{kμ} - (3/2) φ ∂_kφ F_μ{}^k`. -/
theorem Ric5_ly (μ : Fin 4) (hφ : J.φ ≠ 0) :
    J.Ric5 (inl μ) (inr ()) =
      J.A μ * J.Ric5 (inr ()) (inr ()) + J.φ ^ 2 / 2 * J.divF μ - 3 * J.φ / 2 * ∑ k, J.dφ k * J.Fup μ k := by
  fin_cases μ <;>
  · simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Ric5, dΓ5, Γ5, divF, dF, Γ4, Fup, F, Fintype.sum_sum_type, Fintype.sum_unique,
      Fin.sum_univ_four, Ginv5, dG5, ddG5, dGinv5, dginv, Aup]
    field_simp
    ring

/-- `R_{μν} = R4_{μν} + ½φ² F_μ{}^k F_{νk} - φ⁻¹ ∇_μ∂_νφ + A_μ R_{νy} + A_ν R_{μy} - A_μ A_ν R_{yy}`. -/
theorem Ric5_ll (μ ν : Fin 4) (hφ : J.φ ≠ 0) :
    J.Ric5 (inl μ) (inl ν) =
      J.Ric4 μ ν + J.φ ^ 2 / 2 * J.FF μ ν - J.φ⁻¹ * J.nabla2φ μ ν
        + J.A μ * J.Ric5 (inl ν) (inr ()) + J.A ν * J.Ric5 (inl μ) (inr ())
        - J.A μ * J.A ν * J.Ric5 (inr ()) (inr ()) := by
  fin_cases μ <;> fin_cases ν <;>
  · simp only [gS, ginvS, dgS, ddgS, ddAS, ddφS, Ric5, dΓ5, Γ5, Ric4, dΓ4, Γ4, FF, nabla2φ, F, Fintype.sum_sum_type, Fintype.sum_unique,
      Fin.sum_univ_four, Ginv5, dG5, ddG5, dGinv5, dginv, Aup]
    field_simp
    ring

/-! ### The curvature scalar (D3 step 4) -/

/-- `R_5 = R_4 + ¼φ² F_{μν}F^{μν} - 2φ⁻¹ □φ` (`□ = g^{μν}∇_μ∂_ν` with the mostly-minus `g`). -/
theorem R5_eq (hφ : J.φ ≠ 0) : J.R5 = J.R4 + J.φ ^ 2 / 4 * J.F2 - 2 * J.φ⁻¹ * J.boxφ := by
  have h0 := J.Ric5_yy hφ
  have h1 := fun μ => J.Ric5_ly μ hφ
  have h2 := fun μ ν => J.Ric5_ll μ ν hφ
  simp only [R5, R4, boxφ, F2, Fintype.sum_sum_type, Fintype.sum_unique, Fin.sum_univ_four, Ginv5] at *
  simp only [h0, h1, h2, gS, ginvS, dgS, ddgS, ddAS, ddφS, Aup, FF, nabla2φ, Fup, F, Fin.sum_univ_four] at *
  field_simp
  ring

end Jets
end KaluzaKlein.AbelianKK

end

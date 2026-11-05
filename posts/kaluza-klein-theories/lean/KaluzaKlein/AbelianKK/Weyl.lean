import KaluzaKlein.AbelianKK.Jets

/-!
# The Weyl rescaling of the curvature scalar (D3 step 6)

For `g̃ = Ω² g` in four dimensions, `R̃ = Ω⁻² R - 6 Ω⁻³ □Ω` with `□ = g^{μν}∇_μ∂_ν`. The formula holds for every
signature (under `g ↦ -g` both `R` and `□` change sign), so the mostly-minus convention of the tutorial needs no
change here; the metric components are arbitrary non-zero functions. As a jet identity this one *does* need the relation
`g ginv = 1` (the oracle `sage/kk_weyl.sage` shows it fails with `ginv` free), so, following the fallback of
design D3, the Lean statement takes the four-dimensional metric diagonal, `g = diag(g_0, …, g_3)` with arbitrary
functions, where the inverse is `1/g_μ` and the identity becomes a rational-function identity. The jets of `Ω` are
stored in the `φ` slot of `Jets`, so `□Ω` is `Jets.boxφ`.
-/

set_option maxHeartbeats 0

noncomputable section

namespace KaluzaKlein.AbelianKK.Weyl

open KaluzaKlein.AbelianKK

/-- Jets of a diagonal metric `g_{μμ} = gd μ` with the scalar `Ω` in the `φ` slot (and no vector). -/
def diagJets (gd : Fin 4 → ℝ) (dgd : Fin 4 → Fin 4 → ℝ) (ddgd : Fin 4 → Fin 4 → Fin 4 → ℝ)
    (W : ℝ) (dW : Fin 4 → ℝ) (ddW : Fin 4 → Fin 4 → ℝ) : Jets where
  g := fun μ ν => if μ = ν then gd μ else 0
  ginv := fun μ ν => if μ = ν then (gd μ)⁻¹ else 0
  dg := fun ρ μ ν => if μ = ν then dgd ρ μ else 0
  ddg := fun ρ σ μ ν => if μ = ν then ddgd ρ σ μ else 0
  A := 0
  dA := 0
  ddA := 0
  φ := W
  dφ := dW
  ddφ := ddW

/-- Jets of the rescaled diagonal metric `g̃_{μμ} = Ω² gd μ`, with `∂(Ω² g) = 2Ω ∂Ω g + Ω² ∂g` and the product rule
for the second derivatives. -/
def weylJets (gd : Fin 4 → ℝ) (dgd : Fin 4 → Fin 4 → ℝ) (ddgd : Fin 4 → Fin 4 → Fin 4 → ℝ)
    (W : ℝ) (dW : Fin 4 → ℝ) (ddW : Fin 4 → Fin 4 → ℝ) : Jets where
  g := fun μ ν => if μ = ν then W ^ 2 * gd μ else 0
  ginv := fun μ ν => if μ = ν then W⁻¹ ^ 2 * (gd μ)⁻¹ else 0
  dg := fun ρ μ ν => if μ = ν then 2 * W * dW ρ * gd μ + W ^ 2 * dgd ρ μ else 0
  ddg := fun ρ σ μ ν => if μ = ν then
      2 * (dW ρ * dW σ + W * ddW ρ σ) * gd μ + 2 * W * dW ρ * dgd σ μ + 2 * W * dW σ * dgd ρ μ
        + W ^ 2 * ddgd ρ σ μ
    else 0
  A := 0
  dA := 0
  ddA := 0
  φ := W
  dφ := dW
  ddφ := ddW

/-- `R[Ω² g] = Ω⁻² R[g] - 6 Ω⁻³ □Ω` for a diagonal four-dimensional metric with arbitrary functions. -/
theorem weyl_scalar (gd : Fin 4 → ℝ) (dgd : Fin 4 → Fin 4 → ℝ) (ddgd : Fin 4 → Fin 4 → Fin 4 → ℝ)
    (W : ℝ) (dW : Fin 4 → ℝ) (ddW : Fin 4 → Fin 4 → ℝ) (hg : ∀ μ, gd μ ≠ 0) (hW : W ≠ 0) :
    (weylJets gd dgd ddgd W dW ddW).R4 =
      W⁻¹ ^ 2 * (diagJets gd dgd ddgd W dW ddW).R4 - 6 * W⁻¹ ^ 3 * (diagJets gd dgd ddgd W dW ddW).boxφ := by
  have h0 := hg 0; have h1 := hg 1; have h2 := hg 2; have h3 := hg 3
  simp (config := { decide := true }) only [weylJets, diagJets, Jets.R4, Jets.Ric4, Jets.dΓ4, Jets.Γ4, Jets.dginv,
    Jets.boxφ, Jets.nabla2φ, Jets.gS, Jets.ginvS, Jets.dgS, Jets.ddgS, Jets.ddφS, Fin.sum_univ_four,
    Fin.isValue, Fin.reduceEq, if_true, if_false, ite_true, ite_false]
  field_simp
  ring

end KaluzaKlein.AbelianKK.Weyl

end

import Mathlib
import KaluzaKlein.Foundations.Fields

/-!
# Supergravity: gamma-matrix and Levi-Civita identities

Three groups of facts used by Part II and Part VI: the charge-conjugation matrix of the explicit Dirac matrices
and the antisymmetrised product `γ^{μνρ}`, which make the Rarita-Schwinger operator invariant under
`δψ_μ = ∂_μ ε`; the four-dimensional Levi-Civita contractions; and the Freund-Rubin values of the Ricci tensor
on `AdS_4 × S^7` that follow from them (Duff-Nilsson-Pope, arXiv:2502.07710, eqs. 7-11, converted from their
mostly-plus metric to our `η = diag(1, −1, −1, −1)`: the Levi-Civita contractions depend only on `det η = −1`, while
the stress tensor of the four-form and the Ricci values change sign).
-/

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false

namespace KaluzaKlein.Supergravity

open Matrix Complex

/-! ### Charge conjugation and the Rarita-Schwinger operator -/

/-- The charge-conjugation matrix `C = γ^0 γ^2` (up to a phase) for the Dirac matrices of `Fields.gamma`. -/
noncomputable def C : Matrix (Fin 4) (Fin 4) ℂ := Fields.gamma 0 * Fields.gamma 2

set_option maxHeartbeats 2000000 in
/-- `C γ^μ = -(γ^μ)ᵀ C`, the defining property of the charge-conjugation matrix (Tanii eq. 2.1). -/
theorem C_gamma (μ : Fin 4) : C * Fields.gamma μ = -((Fields.gamma μ)ᵀ * C) := by
  fin_cases μ <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [C, Fields.gamma, Matrix.mul_apply, Fin.sum_univ_four, Matrix.transpose_apply] <;>
    norm_num [Complex.ext_iff]

/-- `Cᵀ = -C`: the charge-conjugation matrix is antisymmetric, so Majorana spinors exist in four dimensions. -/
theorem C_antisymm : Cᵀ = -C := by
  ext i j; fin_cases i <;> fin_cases j <;>
    simp [C, Fields.gamma, Matrix.mul_apply, Fin.sum_univ_four, Matrix.transpose_apply] <;>
    norm_num [Complex.ext_iff]

/-- The antisymmetrisation `(1/3!)(A B D ± permutations)` of three matrices. -/
noncomputable def antisym3 (A B D : Matrix (Fin 4) (Fin 4) ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  (1 / 6 : ℂ) • (A * B * D - A * D * B - B * A * D + B * D * A + D * A * B - D * B * A)

/-- Swapping the last two arguments changes the sign. -/
theorem antisym3_swap (A B D : Matrix (Fin 4) (Fin 4) ℂ) : antisym3 A D B = -antisym3 A B D := by
  unfold antisym3
  have h : A * D * B - A * B * D - D * A * B + D * B * A + B * A * D - B * D * A =
      -(A * B * D - A * D * B - B * A * D + B * D * A + D * A * B - D * B * A) := by abel
  rw [h, smul_neg]

/-- The antisymmetrised product `γ^{μνρ} = (1/3!)(γ^μ γ^ν γ^ρ ± permutations)` (Tanii eq. 2.3). -/
noncomputable def gamma3 (μ ν ρ : Fin 4) : Matrix (Fin 4) (Fin 4) ℂ :=
  antisym3 (Fields.gamma μ) (Fields.gamma ν) (Fields.gamma ρ)

/-- `γ^{μνρ}` is antisymmetric in its last two indices. -/
theorem gamma3_antisymm (μ ν ρ : Fin 4) : gamma3 μ ρ ν = -gamma3 μ ν ρ :=
  antisym3_swap _ _ _

/-- A symmetric tensor contracted with an antisymmetric one vanishes: for `A ν ρ = -A ρ ν` the double sum is zero. -/
theorem sum_antisymm_eq_zero {M : Type*} [AddCommGroup M] [Module ℂ M] (A : Fin 4 → Fin 4 → M)
    (hA : ∀ ν ρ, A ρ ν = -A ν ρ) : (∑ ν : Fin 4, ∑ ρ : Fin 4, A ν ρ) = 0 := by
  have h : (∑ ν : Fin 4, ∑ ρ : Fin 4, A ν ρ) = -(∑ ν : Fin 4, ∑ ρ : Fin 4, A ν ρ) := by
    conv_lhs => rw [Finset.sum_comm]
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun y _ => hA x y
  have h2 : (2 : ℂ) • (∑ ν : Fin 4, ∑ ρ : Fin 4, A ν ρ) = 0 := by
    rw [two_smul]; exact add_eq_zero_iff_eq_neg.mpr h
  rcases smul_eq_zero.mp h2 with h3 | h3
  · exact absurd h3 two_ne_zero
  · exact h3

/-- Gauge invariance of the Rarita-Schwinger operator: for `ψ_ρ = ∂_ρ ε` the operator `γ^{μνρ} ∂_ν ψ_ρ` acts on
the symmetric second derivatives `s ν ρ = ∂_ν ∂_ρ ε` and vanishes, `Σ_{νρ} γ^{μνρ} s_{νρ} = 0`. -/
theorem rarita_schwinger_gauge (μ : Fin 4) (s : Fin 4 → Fin 4 → (Fin 4 → ℂ)) (hs : ∀ ν ρ, s ν ρ = s ρ ν) :
    (∑ ν : Fin 4, ∑ ρ : Fin 4, (gamma3 μ ν ρ).mulVec (s ν ρ)) = 0 := by
  apply sum_antisymm_eq_zero
  intro ν ρ
  rw [gamma3_antisymm, hs ν ρ, Matrix.neg_mulVec]

/-! ### Levi-Civita contractions in four dimensions -/

/-- Sign of an integer. -/
def sgn (n : ℤ) : ℤ := if 0 < n then 1 else if n < 0 then -1 else 0

/-- The Levi-Civita symbol on `Fin 4` with `ε_{0123} = 1`. -/
def levi (a b c d : Fin 4) : ℤ :=
  sgn (b.val - a.val) * sgn (c.val - a.val) * sgn (d.val - a.val) *
  sgn (c.val - b.val) * sgn (d.val - b.val) * sgn (d.val - c.val)

/-- The Lorentzian metric `diag(1, -1, -1, -1)` as integers. -/
def etaZ (μ ν : Fin 4) : ℤ := if μ = ν then (if μ = 0 then 1 else -1) else 0

theorem levi_0123 : levi 0 1 2 3 = 1 := by decide

theorem levi_antisymm_01 (a b c d : Fin 4) : levi b a c d = -levi a b c d := by
  revert a b c d; decide

/-- `ε_{μαβγ} ε_ν{}^{αβγ} = -6 η_{μν}` in Lorentzian signature (indices raised with `η`); the value depends only on
`det η = -1`, so it is the same in both signature conventions. -/
theorem levi_contraction (μ ν : Fin 4) :
    (∑ a : Fin 4, ∑ b : Fin 4, ∑ c : Fin 4,
      levi μ a b c * levi ν a b c * etaZ a a * etaZ b b * etaZ c c) = -6 * etaZ μ ν := by
  fin_cases μ <;> fin_cases ν <;> decide

/-- `ε_{αβγδ} ε^{αβγδ} = -24` in Lorentzian signature. -/
theorem levi_square :
    (∑ a : Fin 4, ∑ b : Fin 4, ∑ c : Fin 4, ∑ d : Fin 4,
      levi a b c d * levi a b c d * etaZ a a * etaZ b b * etaZ c c * etaZ d d) = -24 := by
  decide

/-! ### The Freund-Rubin solution -/

/-- Freund-Rubin ansatz `F_{μνρσ} = 3m ε_{μνρσ}`: the contraction `F_{μPQR} F_ν{}^{PQR} = -54 m² η_{μν}`. -/
theorem fr_FF (m : ℝ) (μ ν : Fin 4) :
    (3 * m) ^ 2 * ((∑ a : Fin 4, ∑ b : Fin 4, ∑ c : Fin 4,
      levi μ a b c * levi ν a b c * etaZ a a * etaZ b b * etaZ c c : ℤ) : ℝ) = -54 * m ^ 2 * (etaZ μ ν : ℝ) := by
  rw [levi_contraction]; push_cast; ring

/-- `F_{PQRS} F^{PQRS} = -216 m²` on the ansatz. -/
theorem fr_F2 (m : ℝ) :
    (3 * m) ^ 2 * ((∑ a : Fin 4, ∑ b : Fin 4, ∑ c : Fin 4, ∑ d : Fin 4,
      levi a b c d * levi a b c d * etaZ a a * etaZ b b * etaZ c c * etaZ d d : ℤ) : ℝ) = -216 * m ^ 2 := by
  rw [levi_square]; push_cast; ring

/-- The eleven-dimensional scalar curvature on the ansatz: the field equations of `L = -¼ e R - (1/48) e F²` in
the mostly-minus metric are `G_MN = T_MN` with `T_MN = -(1/3)(F_MPQR F_N^PQR - (1/8) g_MN F²)` (the opposite sign
to Duff-Nilsson-Pope eq. 7, who use the mostly-plus metric); the trace gives `R = -(2/9) T` with
`T = -(1/3)(F² - (11/8) F²) = F²/8`, hence `R = 6 m²`. -/
theorem fr_scalar_curvature (m : ℝ) :
    -(2 / 9 : ℝ) * (-(1 / 3) * ((-216 * m ^ 2) - (11 / 8) * (-216 * m ^ 2))) = 6 * m ^ 2 := by ring

/-- Spacetime components: `R_μν = T_μν + ½ R g_μν = 12 m² g_μν`, anti-de Sitter in the mostly-minus metric
(`-12 m² g_μν` in the mostly-plus metric of DNP eq. 10). -/
theorem fr_ricci_spacetime (m g : ℝ) :
    -(1 / 3 : ℝ) * ((-54 * m ^ 2 * g) - (1 / 8) * g * (-216 * m ^ 2)) + (1 / 2) * g * (6 * m ^ 2)
      = 12 * m ^ 2 * g := by ring

/-- Internal components: `F` has no internal legs, so `R_mn = T_mn + ½ R g_mn = -6 m² g_mn`; with the negative
definite internal metric `g_mn = -γ_mn` this is `R_mn = 6 m² γ_mn`, the seven-sphere (DNP eq. 11). -/
theorem fr_ricci_internal (m g : ℝ) :
    -(1 / 3 : ℝ) * (0 - (1 / 8) * g * (-216 * m ^ 2)) + (1 / 2) * g * (6 * m ^ 2) = -6 * m ^ 2 * g := by ring

end KaluzaKlein.Supergravity

import Mathlib
import KaluzaKlein.Foundations.Fields

/-!
# Supersymmetry: two-component spinor identities and the chiral multiplet

The σ-matrix identities of Martin's primer (hep-ph/9709356, eqs. 2.21-2.25) on explicit `2 × 2` matrices, the
relation of the four-component gamma matrices of `Fields` to the `σ^μ`, and the algebraic facts behind the
Wess-Zumino model: completing the square for the auxiliary field, the supersymmetric vacua of a cubic
superpotential, and the absence of a supersymmetric vacuum in the O'Raifeartaigh model. Mostly-minus signature
`η = diag(1, −1, −1, −1)`: Martin's identities, written for the mostly-plus metric, appear with the opposite sign of
`η`.
-/

set_option linter.unusedSimpArgs false

namespace KaluzaKlein.Supersymmetry

open Matrix Complex

/-- The Pauli matrices. -/
noncomputable def pauli : Fin 3 → Matrix (Fin 2) (Fin 2) ℂ
  | 0 => !![0, 1; 1, 0]
  | 1 => !![0, -I; I, 0]
  | 2 => !![1, 0; 0, -1]

/-- `σ^μ = (1, σ_1, σ_2, σ_3)`, with index structure `(σ^μ)_{α α̇}`. -/
noncomputable def sigma : Fin 4 → Matrix (Fin 2) (Fin 2) ℂ
  | 0 => 1
  | 1 => pauli 0
  | 2 => pauli 1
  | 3 => pauli 2

/-- `σ̄^μ = (1, -σ_1, -σ_2, -σ_3)`, with index structure `(σ̄^μ)^{α̇ α}`. -/
noncomputable def sigmaBar : Fin 4 → Matrix (Fin 2) (Fin 2) ℂ
  | 0 => 1
  | 1 => -pauli 0
  | 2 => -pauli 1
  | 3 => -pauli 2

/-- The mostly-minus metric `η = diag(1, -1, -1, -1)` as complex numbers. -/
def eta (μ ν : Fin 4) : ℂ := if μ = ν then (if μ = 0 then 1 else -1) else 0

/-- `σ^μ σ̄^ν + σ^ν σ̄^μ = 2 η^{μν}` (Martin eq. (2.24) reads `-2 η^{μν}` in the mostly-plus metric). -/
theorem sigma_clifford (μ ν : Fin 4) :
    sigma μ * sigmaBar ν + sigma ν * sigmaBar μ = (2 * eta μ ν) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  fin_cases μ <;> fin_cases ν <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [sigma, sigmaBar, pauli, eta] <;> norm_num [Complex.ext_iff]

/-- `tr(σ^μ σ̄^ν) = 2 η^{μν}`. -/
theorem sigma_trace (μ ν : Fin 4) : Matrix.trace (sigma μ * sigmaBar ν) = 2 * eta μ ν := by
  fin_cases μ <;> fin_cases ν <;>
    simp [sigma, sigmaBar, pauli, eta, Matrix.trace, Fin.sum_univ_two] <;> norm_num [Complex.ext_iff]

/-- `(σ^μ)_{α α̇} (σ̄_μ)^{β̇ β} = 2 δ_α^β δ_α̇^β̇`, the index `μ` lowered with `η` (Martin eq. (2.21) reads `-2` in
the mostly-plus metric). -/
theorem sigma_fierz (a b c d : Fin 2) :
    (∑ μ : Fin 4, eta μ μ * (sigma μ a b * sigmaBar μ c d)) =
      2 * (if a = d then 1 else 0) * (if b = c then 1 else 0) := by
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d <;>
    simp [Fin.sum_univ_four, sigma, sigmaBar, pauli, eta] <;> norm_num [Complex.ext_iff]

/-- The Dirac matrices of `Fields.gamma` in `2 × 2` blocks: `γ^μ = [[0, σ^μ], [σ̄^μ, 0]]`. -/
theorem gamma_block (μ : Fin 4) (i j : Fin 2) :
    Fields.gamma μ (Fin.castAdd 2 i) (Fin.natAdd 2 j) = sigma μ i j ∧
    Fields.gamma μ (Fin.natAdd 2 i) (Fin.castAdd 2 j) = sigmaBar μ i j ∧
    Fields.gamma μ (Fin.castAdd 2 i) (Fin.castAdd 2 j) = 0 ∧
    Fields.gamma μ (Fin.natAdd 2 i) (Fin.natAdd 2 j) = 0 := by
  fin_cases μ <;> fin_cases i <;> fin_cases j <;>
    simp [Fields.gamma, sigma, sigmaBar, pauli, Fin.castAdd, Fin.natAdd]

/-- Completing the square for the auxiliary field: `F* F + w F + w̄ F̄ = |F + w̄|² - |w|²`, so the Lagrangian is
maximised at `F = -w̄` and the eliminated auxiliary field leaves the potential `-|w|²` with `w = ∂W/∂φ`. -/
theorem auxiliary_complete_square (w F : ℂ) :
    (normSq F : ℂ) + w * F + (starRingEnd ℂ) w * (starRingEnd ℂ) F =
      (normSq (F + (starRingEnd ℂ) w) : ℂ) - normSq w := by
  simp only [normSq_apply, Complex.ext_iff, Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, Complex.conj_re, Complex.conj_im]
  constructor <;> ring

/-- The F-term potential of `W = ½ m φ² + ⅙ y φ³`, `V = |m φ + ½ y φ²|²`, vanishes exactly at the two
supersymmetric vacua `φ = 0` and `φ = -2m/y`. -/
theorem wz_vacua (m y φ : ℂ) (hy : y ≠ 0) :
    normSq (m * φ + y / 2 * φ ^ 2) = 0 ↔ φ = 0 ∨ φ = -2 * m / y := by
  rw [normSq_eq_zero]
  constructor
  · intro h
    have h' : φ * (m + y / 2 * φ) = 0 := by rw [← h]; ring
    rcases mul_eq_zero.mp h' with h1 | h2
    · exact Or.inl h1
    · right
      rw [eq_div_iff hy]
      linear_combination 2 * h2
  · rintro (h | h)
    · simp [h]
    · rw [h]; field_simp; ring

/-- The F-term potential `Σ |W_i|²` is non-negative. -/
theorem f_term_nonneg {n : ℕ} (W : Fin n → ℂ) : 0 ≤ ∑ i, normSq (W i) :=
  Finset.sum_nonneg fun i _ => normSq_nonneg (W i)

/-- The O'Raifeartaigh model `W = -k Φ₁ + m Φ₂ Φ₃ + ½ y Φ₁ Φ₃²` (Martin eq. 7.3.1): the equations
`F_1 = F_2 = F_3 = 0`, that is `k - ½ y φ₃² = 0`, `m φ₃ = 0`, `m φ₂ + y φ₁ φ₃ = 0`, have no common solution when
`k ≠ 0` and `m ≠ 0`, so supersymmetry is spontaneously broken. -/
theorem oraifeartaigh_no_susy_vacuum (k m y : ℂ) (hk : k ≠ 0) (hm : m ≠ 0) :
    ¬ ∃ φ₁ φ₂ φ₃ : ℂ, k - y / 2 * φ₃ ^ 2 = 0 ∧ m * φ₃ = 0 ∧ m * φ₂ + y * φ₁ * φ₃ = 0 := by
  rintro ⟨φ₁, φ₂, φ₃, h1, h2, -⟩
  rcases mul_eq_zero.mp h2 with h | h
  · exact hm h
  · subst h
    simp at h1
    exact hk h1

end KaluzaKlein.Supersymmetry

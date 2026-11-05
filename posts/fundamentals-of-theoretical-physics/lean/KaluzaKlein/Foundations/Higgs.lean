import KaluzaKlein.Geometry.LieGroups

/-!
# Spontaneous symmetry breaking and the Higgs mechanism, in polynomial form

* `abelian_higgs_min`: `V(r) = (λ/4)(r² − μ²)²` is non-negative and vanishes exactly on the circle `r = ±μ`.
* `abelian_higgs_expansion`: around the vacuum `r = μ + h`, `V = λμ² h² + λμ h³ + (λ/4) h⁴`: the Higgs mass is
  `m_h² = 2λμ²` (coefficient `½ m² h²`).
* `abelian_gauge_mass`: the kinetic term `|D_μ φ|²` of a scalar of charge `e` evaluated on the constant vacuum
  `φ = μ` is `e² μ² A_μ A^μ`, i.e. a mass term `½ m_A² A²` with `m_A² = 2 e² μ²`.
* `su2_higgs_mass_matrix`: for the `SU(2)` doublet vacuum `φ₀ = (0, v)`, the Hermitian form
  `Re ((T_a φ₀)† (T_b φ₀)) = (v²/4) δ_ab`: all three gauge bosons get the same mass² `g² v² / 2`
  (with `|D φ|² ⊃ g² A^a A^b (T_a φ₀)†(T_b φ₀)`).
-/

namespace KaluzaKlein.Higgs

/-- The Mexican-hat potential is non-negative and vanishes on the vacuum circle `r² = μ²`. -/
theorem abelian_higgs_min (lam μ r : ℝ) (hlam : 0 < lam) :
    0 ≤ lam / 4 * (r ^ 2 - μ ^ 2) ^ 2 ∧ (lam / 4 * (r ^ 2 - μ ^ 2) ^ 2 = 0 ↔ r ^ 2 = μ ^ 2) := by
  refine ⟨by positivity, ?_⟩
  constructor
  · intro h
    have : (r ^ 2 - μ ^ 2) ^ 2 = 0 := by
      have hl : lam / 4 ≠ 0 := by positivity
      exact (mul_eq_zero.mp h).resolve_left hl
    exact sub_eq_zero.mp (pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this)
  · intro h; rw [h]; ring

/-- Expanding around the vacuum, `V(μ + h) = λμ² h² + λμ h³ + (λ/4) h⁴`. -/
theorem abelian_higgs_expansion (lam μ h : ℝ) :
    lam / 4 * ((μ + h) ^ 2 - μ ^ 2) ^ 2 = lam * μ ^ 2 * h ^ 2 + lam * μ * h ^ 3 + lam / 4 * h ^ 4 := by
  ring

/-- `|D_μ φ|²` with `D_μ = ∂_μ − i e A_μ` on the constant vacuum `φ = μ` (real): the derivative term vanishes and
`|−i e A_μ μ|² = e² μ² A_μ A^μ` (written with the components and the inverse metric `η`). -/
theorem abelian_gauge_mass (e μ : ℝ) (A η : Fin 4 → ℝ) :
    ∑ ν, η ν * ((e * A ν * μ) * (e * A ν * μ)) = e ^ 2 * μ ^ 2 * ∑ ν, η ν * A ν * A ν := by
  rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun ν _ => by ring

/-- For the doublet vacuum `φ₀ = (0, v)` the vector `T_a φ₀` has components `(T_a)_{i1} v`, so
`(T_a φ₀)† (T_b φ₀) = Σ_i conj((T_a)_{i1} v) (T_b)_{i1} v`; its real part is `(v²/4) δ_ab`: the `SU(2)` gauge-boson
mass matrix is proportional to the identity. -/
theorem su2_higgs_mass_matrix (v : ℝ) (a b : Fin 3) :
    (∑ i, star ((T a) i 1 * (v : ℂ)) * ((T b) i 1 * (v : ℂ))).re = if a = b then v ^ 2 / 4 else 0 := by
  fin_cases a <;> fin_cases b <;>
    simp [T, pauli, Fin.sum_univ_succ, Complex.ext_iff] <;> ring_nf <;> norm_num

end KaluzaKlein.Higgs

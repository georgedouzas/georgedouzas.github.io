import KaluzaKlein.Geometry.Coset

/-!
# Representations: `so(4) ≅ su(2) ⊕ su(2)`, the roots of `su(3)` and a branching rule

* `J_a = ½(ε_abc M_bc + M_a3)` and `K_a = ½(ε_abc M_bc − M_a3)` (with `M_ij` the rotation generators of `ℝ⁴`) each
  satisfy the `so(3)` relations and commute with each other: `so(4) ≅ su(2) ⊕ su(2)`.
* The roots of `su(3)` are the pairs of `(ad λ₃, ad λ₈)`-eigenvalues on `λ₁ ± iλ₂`, `λ₄ ± iλ₅`, `λ₆ ± iλ₇`
  (`Coset.lean`); the branching `8 → 3₀ ⊕ 1₀ ⊕ 2_{√3} ⊕ 2_{−√3}` under `su(2) × u(1)` follows from `[λ₈, λ_a] = 0`
  for `a = 1, 2, 3` and the `λ₈`-charges `±√3` of the two doublets.
-/

namespace KaluzaKlein.Representations

open Matrix

/-- The combinations `J_a = −½(ε_abc M_bc + M_a3)` of the `so(4)` generators, written out. -/
def J : Fin 3 → Matrix (Fin 4) (Fin 4) ℚ :=
  ![!![0, 0, 0, -1/2; 0, 0, -1/2, 0; 0, 1/2, 0, 0; 1/2, 0, 0, 0],
    !![0, 0, 1/2, 0; 0, 0, 0, -1/2; -1/2, 0, 0, 0; 0, 1/2, 0, 0],
    !![0, -1/2, 0, 0; 1/2, 0, 0, 0; 0, 0, 0, -1/2; 0, 0, 1/2, 0]]

/-- The combinations `K_a = −½(ε_abc M_bc − M_a3)`. -/
def K : Fin 3 → Matrix (Fin 4) (Fin 4) ℚ :=
  ![!![0, 0, 0, 1/2; 0, 0, -1/2, 0; 0, 1/2, 0, 0; -1/2, 0, 0, 0],
    !![0, 0, 1/2, 0; 0, 0, 0, 1/2; -1/2, 0, 0, 0; 0, -1/2, 0, 0],
    !![0, -1/2, 0, 0; 1/2, 0, 0, 0; 0, 0, 0, 1/2; 0, 0, -1/2, 0]]

/-- `J_a = −½(ε_abc M_bc + M_a3)` and `K_a = −½(ε_abc M_bc − M_a3)` in terms of the rotation generators. -/
theorem JK_of_M4 : J 0 = -(1/2 : ℚ) • (M4 1 2 + M4 0 3) ∧ K 0 = -(1/2 : ℚ) • (M4 1 2 - M4 0 3) := by
  refine ⟨?_, ?_⟩ <;> (ext i j; fin_cases i <;> fin_cases j <;> simp [J, K, M4] <;> norm_num)

/-- `J` and `K` each close into `so(3)`: `[J₀, J₁] = J₂` (and cyclically), likewise for `K`. -/
theorem so4_splits_brackets :
    ⁅J 0, J 1⁆ = J 2 ∧ ⁅J 1, J 2⁆ = J 0 ∧ ⁅J 2, J 0⁆ = J 1 ∧ ⁅K 0, K 1⁆ = K 2 ∧ ⁅K 1, K 2⁆ = K 0 ∧ ⁅K 2, K 0⁆ = K 1 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [J, K, Matrix.mul_apply, Fin.sum_univ_succ] <;> norm_num)

set_option maxHeartbeats 1600000 in
/-- `J` and `K` commute: `[J_a, K_b] = 0` for all `a, b`, so `so(4) ≅ su(2) ⊕ su(2)`. -/
theorem so4_splits_commute (a b : Fin 3) : ⁅J a, K b⁆ = 0 := by
  fin_cases a <;> fin_cases b <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [J, K, Matrix.mul_apply, Fin.sum_univ_succ] <;> norm_num)

/-- The `su(2)` subalgebra `λ₁, λ₂, λ₃` commutes with `λ₈`: the triplet and the singlet have zero `u(1)` charge. -/
theorem su3_branching_neutral : ⁅gellMann 7, gellMann 0⁆ = 0 ∧ ⁅gellMann 7, gellMann 1⁆ = 0 ∧ ⁅gellMann 7, gellMann 2⁆ = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [gellMann, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff])

/-- `λ₁, λ₂, λ₃` close into `su(2)` inside `su(3)`: `[λ₁, λ₂] = 2i λ₃` (and cyclically). -/
theorem su3_su2_subalgebra :
    ⁅gellMann 0, gellMann 1⁆ = (2 * Complex.I) • gellMann 2 ∧ ⁅gellMann 1, gellMann 2⁆ = (2 * Complex.I) • gellMann 0 ∧
      ⁅gellMann 2, gellMann 0⁆ = (2 * Complex.I) • gellMann 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [gellMann, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff] <;> norm_num)

end KaluzaKlein.Representations

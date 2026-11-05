import Mathlib
import TutorialAssets.CoordinateGeometry

/-!
# Conventions shared by every part

* Coordinates are `Fin n → ℝ`; a metric is a family of component functions (the `CoordinateGeometry` asset), with
  `christoffel`, `riemann` (`Rᵖ_σμν = ∂_μ Γᵖ_νσ − ∂_ν Γᵖ_μσ + Γᵖ_μλ Γᵏ_νσ − Γᵖ_νλ Γᵏ_μσ`), `ricci` (`R_σν = Rᵖ_σρν`)
  and `scalarCurvature` from that asset. With these signs the round sphere has positive curvature.
* `eps a b c` is the Levi-Civita symbol on `Fin 3`, the structure constants of `so(3) ≅ su(2)`.
* `L a` are the real `so(3)` generators `(L_a)_{bc} = −ε_{abc}`, with `⁅L a, L b⁆ = Σ_c ε_{abc} L c`.
* `lieBracket X Y` is the Lie bracket of coordinate vector fields, `[X, Y]ᵏ = Xˡ ∂_l Yᵏ − Yˡ ∂_l Xᵏ`.
-/

namespace KaluzaKlein

open CoordinateGeometry Matrix

/-- The Levi-Civita symbol `ε_{abc}` on `Fin 3`: `ε_{012} = 1`, totally antisymmetric. -/
def eps : Fin 3 → Fin 3 → Fin 3 → ℚ :=
  ![![![0, 0, 0], ![0, 0, 1], ![0, -1, 0]],
    ![![0, 0, -1], ![0, 0, 0], ![1, 0, 0]],
    ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]]

/-- `L₀`: the generator of rotations in the `(1, 2)`-plane, `(L₀)_{bc} = −ε_{0bc}`. -/
def L0 : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

/-- `L₁`: the generator of rotations in the `(2, 0)`-plane. -/
def L1 : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 1; 0, 0, 0; -1, 0, 0]

/-- `L₂`: the generator of rotations in the `(0, 1)`-plane. -/
def L2 : Matrix (Fin 3) (Fin 3) ℚ := !![0, -1, 0; 1, 0, 0; 0, 0, 0]

/-- The `so(3)` generators `(L_a)_{bc} = −ε_{abc}`, indexed by `a`. -/
def L : Fin 3 → Matrix (Fin 3) (Fin 3) ℚ := ![L0, L1, L2]

/-- The cyclic bracket relations of `so(3)`: `⁅L₀, L₁⁆ = L₂`, `⁅L₁, L₂⁆ = L₀`, `⁅L₂, L₀⁆ = L₁`. -/
theorem so3_brackets : ⁅L0, L1⁆ = L2 ∧ ⁅L1, L2⁆ = L0 ∧ ⁅L2, L0⁆ = L1 := by
  have hb_0_1 : ⁅L0, L1⁆ = L2 := by
    unfold L0 L1 L2
    rw [Ring.lie_def]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  have hb_1_2 : ⁅L1, L2⁆ = L0 := by
    unfold L1 L2 L0
    rw [Ring.lie_def]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  have hb_2_0 : ⁅L2, L0⁆ = L1 := by
    unfold L2 L0 L1
    rw [Ring.lie_def]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  refine ⟨?_, ?_, ?_⟩ <;> assumption
  -- certificate: so3_brackets.sage

/-- The structure constants of `so(3)` are `ε_{abc}`: `⁅L_a, L_b⁆ = Σ_c ε_{abc} L_c`. -/
theorem so3_structure (a b : Fin 3) : ⁅L a, L b⁆ = ∑ c, eps a b c • L c := by
  fin_cases a <;> fin_cases b <;> simp [eps, L, Fin.sum_univ_succ] <;>
    (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
      norm_num [L0, L1, L2, Matrix.mul_apply, Fin.sum_univ_succ])

/-- The Lie bracket of coordinate vector fields, `[X, Y]ᵏ = Xˡ ∂_l Yᵏ − Yˡ ∂_l Xᵏ`. -/
noncomputable def lieBracket {n : ℕ} (X Y : Fin n → (Fin n → ℝ) → ℝ) : Fin n → (Fin n → ℝ) → ℝ :=
  fun k x => ∑ l, (X l x * pd l (Y k) x - Y l x * pd l (X k) x)

/-- For a symmetric metric the Christoffel symbols are symmetric in the lower indices. -/
theorem christoffel_symm {n : ℕ} (g : Fin n → Fin n → (Fin n → ℝ) → ℝ) (hg : ∀ i j, g i j = g j i)
    (k i j : Fin n) (x : Fin n → ℝ) :
    christoffel g k i j x = christoffel g k j i x := by
  unfold christoffel
  congr 1
  exact Finset.sum_congr rfl fun l _ => by rw [hg i j]; ring

end KaluzaKlein

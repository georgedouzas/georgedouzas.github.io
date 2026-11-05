import KaluzaKlein.Conventions

/-!
# The coset `S² = SO(3)/SO(2)`: the reductive decomposition of `so(3)`

The isotropy algebra of the north pole is `𝔯 = span(L₂)` (rotations about the `Z`-axis) and its complement
`𝔨 = span(L₀, L₁)` is the tangent space of the sphere at the pole. The decomposition is reductive, `[𝔯, 𝔨] ⊆ 𝔨`,
and symmetric, `[𝔨, 𝔨] ⊆ 𝔯`; `ad L₂` rotates `𝔨`, so the complex combinations `L₀ ± i L₁` carry `U(1)` charges
`∓ i`, i.e. `𝔨 ⊗ ℂ` is the charge `±1` representation of the isotropy group.
-/

namespace KaluzaKlein

open Matrix

/-- `[L₂, L₀] = L₁` and `[L₂, L₁] = −L₀`: the isotropy generator maps `𝔨` to itself (reductive). -/
theorem coset_reductive : ⁅L2, L0⁆ = L1 ∧ ⁅L2, L1⁆ = -L0 := by
  refine ⟨?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    norm_num [L0, L1, L2, Matrix.mul_apply, Fin.sum_univ_succ])

/-- `[L₀, L₁] = L₂ ∈ 𝔯`: the coset is symmetric. -/
theorem coset_symmetric : ⁅L0, L1⁆ = L2 := by
  rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;> norm_num [L0, L1, L2, Matrix.mul_apply, Fin.sum_univ_succ]

/-- The complexified generators: `L` over `ℂ`. -/
noncomputable def LC (a : Fin 3) : Matrix (Fin 3) (Fin 3) ℂ := (L a).map (algebraMap ℚ ℂ)

/-- `E₊ = L₀ + i L₁` has charge `−i` under `ad L₂`: `[L₂, E₊] = −i E₊`. -/
theorem coset_charge_plus : ⁅LC 2, LC 0 + Complex.I • LC 1⁆ = -Complex.I • (LC 0 + Complex.I • LC 1) := by
  rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [LC, L, L0, L1, L2, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.map_apply, Complex.ext_iff]

/-- `E₋ = L₀ − i L₁` has charge `+i` under `ad L₂`: `[L₂, E₋] = i E₋`. -/
theorem coset_charge_minus : ⁅LC 2, LC 0 - Complex.I • LC 1⁆ = Complex.I • (LC 0 - Complex.I • LC 1) := by
  rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [LC, L, L0, L1, L2, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.map_apply, Complex.ext_iff]

end KaluzaKlein

namespace KaluzaKlein

/-! ## `S³ = SO(4)/SO(3)`: a symmetric coset with explicit generators

`so(4)` is spanned by the six rotation generators `M_{ij}` of `ℝ⁴`, `(M_{ij})_{kl} = δ_{ik}δ_{jl} − δ_{il}δ_{jk}`. The
isotropy algebra of the point `(0,0,0,1)` is `𝔯 = span(M₀₁, M₀₂, M₁₂) ≅ so(3)` and `𝔨 = span(M₀₃, M₁₃, M₂₃)` is the
tangent space of `S³`; `[𝔯, 𝔨] ⊆ 𝔨` and `[𝔨, 𝔨] ⊆ 𝔯`. -/

/-- The rotation generator `M_{ij}` of `ℝ⁴`. -/
def M4 (i j : Fin 4) : Matrix (Fin 4) (Fin 4) ℚ :=
  Matrix.of fun k l => (if k = i ∧ l = j then 1 else 0) - (if k = j ∧ l = i then 1 else 0)

/-- `[𝔨, 𝔨] ⊆ 𝔯`: `[M₀₃, M₁₃] = −M₀₁`, `[M₀₃, M₂₃] = −M₀₂`, `[M₁₃, M₂₃] = −M₁₂` (`S³` is a symmetric coset). -/
theorem so4_coset_symmetric :
    ⁅M4 0 3, M4 1 3⁆ = -M4 0 1 ∧ ⁅M4 0 3, M4 2 3⁆ = -M4 0 2 ∧ ⁅M4 1 3, M4 2 3⁆ = -M4 1 2 := by
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext k l; fin_cases k <;> fin_cases l <;>
    simp [M4, Matrix.mul_apply, Fin.sum_univ_succ])

/-- `[𝔯, 𝔨] ⊆ 𝔨`: `[M₀₁, M₀₃] = −M₁₃`, `[M₀₁, M₁₃] = M₀₃`, `[M₀₁, M₂₃] = 0` (the isotropy generator `M₀₁` rotates `𝔨`). -/
theorem so4_coset_reductive :
    ⁅M4 0 1, M4 0 3⁆ = -M4 1 3 ∧ ⁅M4 0 1, M4 1 3⁆ = M4 0 3 ∧ ⁅M4 0 1, M4 2 3⁆ = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext k l; fin_cases k <;> fin_cases l <;>
    simp [M4, Matrix.mul_apply, Fin.sum_univ_succ])

/-! ## `SU(3)/U(1)×U(1)`: a non-symmetric coset

With the Gell-Mann matrices `λ_A`, the isotropy algebra is `𝔯 = span(λ₃, λ₈)` and `𝔨 = span(λ₁, λ₂, λ₄, λ₅, λ₆, λ₇)`.
The complex combinations `λ₁ + iλ₂`, `λ₄ + iλ₅`, `λ₆ + iλ₇` are eigenvectors of `ad λ₃` and `ad λ₈` (the isotropy
charges), and `[λ₁, λ₄] = i λ₇ ∈ 𝔨`: the coset is reductive but not symmetric. -/

/-- The Gell-Mann matrices `λ₁ … λ₈` (indexed `0 … 7`). -/
noncomputable def gellMann : Fin 8 → Matrix (Fin 3) (Fin 3) ℂ :=
  ![!![0, 1, 0; 1, 0, 0; 0, 0, 0], !![0, -Complex.I, 0; Complex.I, 0, 0; 0, 0, 0], !![1, 0, 0; 0, -1, 0; 0, 0, 0],
    !![0, 0, 1; 0, 0, 0; 1, 0, 0], !![0, 0, -Complex.I; 0, 0, 0; Complex.I, 0, 0], !![0, 0, 0; 0, 0, 1; 0, 1, 0],
    !![0, 0, 0; 0, 0, -Complex.I; 0, Complex.I, 0],
    (1 / Real.sqrt 3 : ℂ) • !![1, 0, 0; 0, 1, 0; 0, 0, -2]]

/-- The isotropy charges under `λ₃`: `[λ₃, λ₁ + iλ₂] = 2(λ₁ + iλ₂)`, `[λ₃, λ₄ + iλ₅] = (λ₄ + iλ₅)`,
`[λ₃, λ₆ + iλ₇] = −(λ₆ + iλ₇)`. -/
theorem su3_coset_charges_lambda3 :
    ⁅gellMann 2, gellMann 0 + Complex.I • gellMann 1⁆ = (2 : ℂ) • (gellMann 0 + Complex.I • gellMann 1) ∧
      ⁅gellMann 2, gellMann 3 + Complex.I • gellMann 4⁆ = gellMann 3 + Complex.I • gellMann 4 ∧
      ⁅gellMann 2, gellMann 5 + Complex.I • gellMann 6⁆ = -(gellMann 5 + Complex.I • gellMann 6) := by
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [gellMann, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff] <;> ring_nf <;> simp [Complex.I_sq] <;> ring)

/-- The isotropy charges under `λ₈`: `[λ₈, λ₁ + iλ₂] = 0`, `[λ₈, λ₄ + iλ₅] = √3 (λ₄ + iλ₅)`,
`[λ₈, λ₆ + iλ₇] = √3 (λ₆ + iλ₇)`. -/
theorem su3_coset_charges_lambda8 :
    ⁅gellMann 7, gellMann 0 + Complex.I • gellMann 1⁆ = 0 ∧
      ⁅gellMann 7, gellMann 3 + Complex.I • gellMann 4⁆ = (Real.sqrt 3 : ℂ) • (gellMann 3 + Complex.I • gellMann 4) ∧
      ⁅gellMann 7, gellMann 5 + Complex.I • gellMann 6⁆ = (Real.sqrt 3 : ℂ) • (gellMann 5 + Complex.I • gellMann 6) := by
  have h3 : (Real.sqrt 3 : ℂ) * (1 / Real.sqrt 3 : ℂ) = 1 := by
    rw [mul_one_div, div_self]; exact_mod_cast (Real.sqrt_pos.mpr (by norm_num : (0:ℝ) < 3)).ne'
  refine ⟨?_, ?_, ?_⟩ <;> (rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [gellMann, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff] <;> ring_nf <;>
    simp [Complex.I_sq] <;> (try linear_combination h3) <;> (try linear_combination (-1 : ℂ) * h3) <;>
    (try linear_combination Complex.I * h3) <;> (try linear_combination (-Complex.I) * h3))

/-- `[λ₁, λ₄] = i λ₇ ∈ 𝔨`: `[𝔨, 𝔨] ⊄ 𝔯`, so `SU(3)/U(1)×U(1)` is not symmetric. -/
theorem su3_coset_not_symmetric : ⁅gellMann 0, gellMann 3⁆ = Complex.I • gellMann 6 := by
  rw [Ring.lie_def]; ext i j; fin_cases i <;> fin_cases j <;>
    simp [gellMann, Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff]

end KaluzaKlein

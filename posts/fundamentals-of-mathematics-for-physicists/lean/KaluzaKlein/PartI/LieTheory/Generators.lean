import KaluzaKlein.Geometry.LieGroups
import KaluzaKlein.Groups.Representations
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaces
import VeriphyCheck

/-!
# Lie groups and Lie algebras: the explicit matrices of the chapter

Literal-bodied definitions shared by every section module of `tutorial/part-i-11-lie-groups-algebras.qmd`: the
`su(2)` generators `T₁, T₂, T₃ = −(i/2) σ_a` as `T0, T1, T2` over `ℂ`, the `sl(2, ℝ)` basis `H, E, F`, the Heisenberg
matrices `E₁₂, E₂₃, E₁₃` and group elements `h(x, y, z)`, the affine group `(a, b)` and its generators `X, Y`, the
rotation generators `L_a` over `ℝ`, the rotations about the first and third axes, the boosts `B(w)`, the
exponentials of the `su(2)` generators and the Euler-angle element, and the Gell-Mann matrices `λ₁ … λ₇` and `√3 λ₈`.
The literal bodies are what Veriphy's translation reads; the lemmas `T0_eq` … link them to the definitions of
`KaluzaKlein.Geometry.LieGroups` and `KaluzaKlein.Geometry.Coset` used elsewhere in the book.
-/

namespace KaluzaKlein.LieTheory.Generators

open Matrix

/-! ## su(2): Pauli matrices and generators -/

/-- `σ₁`. -/
noncomputable def s1 : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]
/-- `σ₂`. -/
noncomputable def s2 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]
/-- `σ₃`. -/
noncomputable def s3 : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]
/-- `T₁ = −(i/2) σ₁`. -/
noncomputable def T0 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -(Complex.I / 2); -(Complex.I / 2), 0]
/-- `T₂ = −(i/2) σ₂`. -/
noncomputable def T1 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -(1 / 2); 1 / 2, 0]
/-- `T₃ = −(i/2) σ₃`. -/
noncomputable def T2 : Matrix (Fin 2) (Fin 2) ℂ := !![-(Complex.I / 2), 0; 0, Complex.I / 2]

/-- The literal generators are the `T a` of the geometry chapter. -/
theorem T0_eq : T0 = T 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [T0, T, pauli]
theorem T1_eq : T1 = T 1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [T1, T, pauli, Complex.ext_iff]
theorem T2_eq : T2 = T 2 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [T2, T, pauli]

/-- The three generators as a family. -/
noncomputable def Tv : Fin 3 → Matrix (Fin 2) (Fin 2) ℂ := ![T0, T1, T2]

/-- The quaternion units `𝐢 = 2T₁, 𝐣 = 2T₂, 𝐤 = 2T₃`. -/
noncomputable def qi : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; -Complex.I, 0]
noncomputable def qj : Matrix (Fin 2) (Fin 2) ℂ := !![0, -1; 1, 0]
noncomputable def qk : Matrix (Fin 2) (Fin 2) ℂ := !![-Complex.I, 0; 0, Complex.I]

/-- Hermitian spin matrices `S_a = i T_a = ½ σ_a`. -/
noncomputable def S1 : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1 / 2; 1 / 2, 0]
noncomputable def S2 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -(Complex.I / 2); Complex.I / 2, 0]
noncomputable def S3 : Matrix (Fin 2) (Fin 2) ℂ := !![1 / 2, 0; 0, -(1 / 2)]

/-- A general element `X = x₁ T₁ + x₂ T₂ + x₃ T₃` of `su(2)`, written out. -/
noncomputable def su2El (x1 x2 x3 : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![-(Complex.I * x3 / 2), -(Complex.I * x1 / 2) - x2 / 2; -(Complex.I * x1 / 2) + x2 / 2, Complex.I * x3 / 2]

/-! ## so(3) over ℝ -/

/-- `L₁` over `ℝ`. -/
def L0R : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]
/-- `L₂` over `ℝ`. -/
def L1R : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 0, 0; -1, 0, 0]
/-- `n · L` for `n ∈ ℝ³`. -/
def nL (n1 n2 n3 : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![0, -n3, n2; n3, 0, -n1; -n2, n1, 0]
/-- `i L₃`, the Hermitian generator `J₃` of the vector representation (over `ℂ`). -/
noncomputable def J3v : Matrix (Fin 3) (Fin 3) ℂ := !![0, -Complex.I, 0; Complex.I, 0, 0; 0, 0, 0]
/-- `J₊ = i L₁ − L₂` on `ℂ³`. -/
noncomputable def Jpv : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, -1; 0, 0, -Complex.I; 1, Complex.I, 0]
/-- `J₋ = i L₁ + L₂` on `ℂ³`. -/
noncomputable def Jmv : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 1; 0, 0, -Complex.I; -1, Complex.I, 0]
/-- `J₁ = i L₁`, `J₂ = i L₂` on `ℂ³`. -/
noncomputable def J1v : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 0; 0, 0, -Complex.I; 0, Complex.I, 0]
noncomputable def J2v : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, Complex.I; 0, 0, 0; -Complex.I, 0, 0]

/-- The rotation by `t` about the first axis. -/
noncomputable def rotX (t : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1, 0, 0; 0, Real.cos t, -Real.sin t; 0, Real.sin t, Real.cos t]
/-- The rotation by `t` about the second axis. -/
noncomputable def rotY (t : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos t, 0, Real.sin t; 0, 1, 0; -Real.sin t, 0, Real.cos t]
/-- The plane rotation by `t`. -/
noncomputable def rot2 (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos t, -Real.sin t; Real.sin t, Real.cos t]

/-! ## sl(2, ℝ), the Heisenberg and affine groups -/

/-- `H = diag(1, −1)`. -/
def H : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, -1]
/-- `E = E₁₂`. -/
def E : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 0, 0]
/-- `F = E₂₁`. -/
def F : Matrix (Fin 2) (Fin 2) ℚ := !![0, 0; 1, 0]
/-- `H, E, F` over `ℝ`. -/
def HR : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
def ER : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 0, 0]
def FR : Matrix (Fin 2) (Fin 2) ℝ := !![0, 0; 1, 0]

/-- The Heisenberg basis `X = E₁₂`, `Y = E₂₃`, `Z = E₁₃`. -/
def E12 : Matrix (Fin 3) (Fin 3) ℚ := !![0, 1, 0; 0, 0, 0; 0, 0, 0]
def E23 : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 0; 0, 0, 1; 0, 0, 0]
def E13 : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 1; 0, 0, 0; 0, 0, 0]
/-- The Heisenberg group element `h(x, y, z)`. -/
def heis (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, x, z; 0, 1, y; 0, 0, 1]
/-- A general element `N = x E₁₂ + y E₂₃ + z E₁₃` of the Heisenberg algebra. -/
def heisAlg (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![0, x, z; 0, 0, y; 0, 0, 0]

/-- The affine generators `X = E₁₁` (dilations) and `Y = E₁₂` (translations). -/
def affX : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, 0]
def affY : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 0, 0]
/-- The affine group element `x ↦ a x + b`. -/
def aff (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; 0, 1]
/-- `x X + y Y` over `ℝ`. -/
def affAlg (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; 0, 0]

/-- The boost `B(w)` of the Minkowski plane and the metric `η = diag(1, −1)`. -/
noncomputable def boost (w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cosh w, Real.sinh w; Real.sinh w, Real.cosh w]
def eta2 : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
/-- The symplectic form `J` of the plane. -/
def Jsym : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

/-! ## Exponentials of the su(2) generators and the Euler-angle element -/

/-- `e^{φ T₃} = diag(e^{−iφ/2}, e^{iφ/2})`. -/
noncomputable def expT3 (φ : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.exp (-(Complex.I * φ / 2)), 0; 0, Complex.exp (Complex.I * φ / 2)]
/-- `e^{θ T₁} = cos(θ/2) 1 − i sin(θ/2) σ₁`. -/
noncomputable def expT1 (θ : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.cos (θ / 2), -(Complex.I * Complex.sin (θ / 2)); -(Complex.I * Complex.sin (θ / 2)), Complex.cos (θ / 2)]
/-- `e^{t T₂} = cos(t/2) 1 − i sin(t/2) σ₂`. -/
noncomputable def expT2 (t : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.cos (t / 2), -Complex.sin (t / 2); Complex.sin (t / 2), Complex.cos (t / 2)]
/-- The Euler-angle element `g(θ, φ, ψ) = e^{φ T₃} e^{θ T₁} e^{ψ T₃}`, written out. -/
noncomputable def euler (θ φ ψ : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.cos (θ / 2) * Complex.exp (-(Complex.I * (φ + ψ) / 2)),
      -(Complex.I * Complex.sin (θ / 2) * Complex.exp (-(Complex.I * (φ - ψ) / 2)));
    -(Complex.I * Complex.sin (θ / 2) * Complex.exp (Complex.I * (φ - ψ) / 2)),
      Complex.cos (θ / 2) * Complex.exp (Complex.I * (φ + ψ) / 2)]
/-- A general element `a₀ 1 + 2 (a₁ T₁ + a₂ T₂ + a₃ T₃)` of `SU(2)`. -/
noncomputable def su2Q (a0 a1 a2 a3 : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![a0 - Complex.I * a3, -(Complex.I * a1) - a2; -(Complex.I * a1) + a2, a0 + Complex.I * a3]

/-! ## Gell-Mann matrices (literal) -/

noncomputable def gm1 : Matrix (Fin 3) (Fin 3) ℂ := !![0, 1, 0; 1, 0, 0; 0, 0, 0]
noncomputable def gm2 : Matrix (Fin 3) (Fin 3) ℂ := !![0, -Complex.I, 0; Complex.I, 0, 0; 0, 0, 0]
noncomputable def gm3 : Matrix (Fin 3) (Fin 3) ℂ := !![1, 0, 0; 0, -1, 0; 0, 0, 0]
noncomputable def gm4 : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 1; 0, 0, 0; 1, 0, 0]
noncomputable def gm5 : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, -Complex.I; 0, 0, 0; Complex.I, 0, 0]
noncomputable def gm6 : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 0; 0, 0, 1; 0, 1, 0]
noncomputable def gm7 : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 0; 0, 0, -Complex.I; 0, Complex.I, 0]
/-- `√3 λ₈ = diag(1, 1, −2)`. -/
noncomputable def gm8s : Matrix (Fin 3) (Fin 3) ℂ := !![1, 0, 0; 0, 1, 0; 0, 0, -2]

/-- The literal Gell-Mann matrices are those of the coset chapter; `λ₈ = (1/√3) gm8s`. -/
theorem gm_eq : gm1 = gellMann 0 ∧ gm2 = gellMann 1 ∧ gm3 = gellMann 2 ∧ gm4 = gellMann 3 ∧ gm5 = gellMann 4 ∧
    gm6 = gellMann 5 ∧ gm7 = gellMann 6 ∧ gellMann 7 = (1 / Real.sqrt 3 : ℂ) • gm8s := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> (ext i j; fin_cases i <;> fin_cases j <;>
    simp [gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8s, gellMann])

/-- The matrix units `E₁₂, E₁₃, E₂₃` of `3 × 3` matrices over `ℂ`. -/
noncomputable def E12c : Matrix (Fin 3) (Fin 3) ℂ := !![0, 1, 0; 0, 0, 0; 0, 0, 0]
noncomputable def E13c : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 1; 0, 0, 0; 0, 0, 0]
noncomputable def E23c : Matrix (Fin 3) (Fin 3) ℂ := !![0, 0, 0; 0, 0, 1; 0, 0, 0]

/-! ## Half-angle variables: `c = cos(t/2)`, `s = sin(t/2)` as polynomial atoms -/

/-- `e^{t T₃}` with `c = cos(t/2)`, `s = sin(t/2)`: `diag(c − i s, c + i s)`. -/
noncomputable def expT3cs (c s : ℂ) : Matrix (Fin 2) (Fin 2) ℂ := !![c - Complex.I * s, 0; 0, c + Complex.I * s]
/-- `e^{t T₁}` with `c = cos(t/2)`, `s = sin(t/2)`: `c 1 − i s σ₁`. -/
noncomputable def expT1cs (c s : ℂ) : Matrix (Fin 2) (Fin 2) ℂ := !![c, -(Complex.I * s); -(Complex.I * s), c]
/-- `e^{t T₂}` with `c = cos(t/2)`, `s = sin(t/2)`: `c 1 − i s σ₂`. -/
noncomputable def expT2cs (c s : ℂ) : Matrix (Fin 2) (Fin 2) ℂ := !![c, -s; s, c]
/-- The rotation about the third axis with `c = cos t`, `s = sin t`. -/
def rotcs (c s : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![c, -s, 0; s, c, 0; 0, 0, 1]
/-- The rotation about the first axis with `c = cos t`, `s = sin t`. -/
def rotXcs (c s : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, c, -s; 0, s, c]
/-- The plane rotation with `c = cos t`, `s = sin t`. -/
def rot2cs (c s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![c, -s; s, c]
/-- The boost with `ch = cosh w`, `sh = sinh w`. -/
def boostcs (ch sh : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![ch, sh; sh, ch]

/-- `expT3 t = expT3cs (cos (t/2)) (sin (t/2))`, by Euler's formula. -/
theorem expT3_eq_cs (t : ℂ) : expT3 t = expT3cs (Complex.cos (t / 2)) (Complex.sin (t / 2)) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [expT3, expT3cs]
  · rw [show -(Complex.I * t / 2) = -(t / 2) * Complex.I by ring, Complex.exp_mul_I, Complex.cos_neg, Complex.sin_neg]; ring
  · rw [show Complex.I * t / 2 = t / 2 * Complex.I by ring, Complex.exp_mul_I]; ring

/-- `expT1 t = expT1cs (cos (t/2)) (sin (t/2))`. -/
theorem expT1_eq_cs (t : ℂ) : expT1 t = expT1cs (Complex.cos (t / 2)) (Complex.sin (t / 2)) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [expT1, expT1cs]

/-- `rotation t = rotcs (cos t) (sin t)`. -/
theorem rotation_eq_cs (t : ℝ) : rotation t = rotcs (Real.cos t) (Real.sin t) := rfl
theorem rotX_eq_cs (t : ℝ) : rotX t = rotXcs (Real.cos t) (Real.sin t) := rfl
theorem rot2_eq_cs (t : ℝ) : rot2 t = rot2cs (Real.cos t) (Real.sin t) := rfl
theorem boost_eq_cs (w : ℝ) : boost w = boostcs (Real.cosh w) (Real.sinh w) := rfl

end KaluzaKlein.LieTheory.Generators

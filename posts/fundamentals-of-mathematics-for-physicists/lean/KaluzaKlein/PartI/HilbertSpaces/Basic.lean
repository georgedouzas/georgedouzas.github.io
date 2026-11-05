import Mathlib

/-!
# Hilbert spaces and operator theory, shared definitions

The explicit objects that run through `tutorial/part-i-07-hilbert-spaces-operators.qmd`: the Pauli matrices, the
rotation matrix, the matrices of the figures (`!![2, 1; 0, 1]`, `!![2, 1; 1, 2]`, the tridiagonal `3 × 3` real and
complex matrices), and the model of the inner product of `ℂⁿ` as a row-times-column product. Every section module
of the chapter imports this one.
-/

namespace KaluzaKlein.HilbertSpaces.Basic

noncomputable section

open Matrix

/-- The Pauli matrix `σ₁`. -/
def σ1 : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]

/-- The Pauli matrix `σ₂`. -/
def σ2 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]

/-- The Pauli matrix `σ₃`. -/
def σ3 : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]

/-- The three Pauli matrices as a family indexed by `Fin 3`. -/
def pauli : Fin 3 → Matrix (Fin 2) (Fin 2) ℂ := ![σ1, σ2, σ3]

/-- The identity and the three Pauli matrices, the Pauli basis of the `2 × 2` matrices. -/
def pauliFour : Fin 4 → Matrix (Fin 2) (Fin 2) ℂ := ![1, σ1, σ2, σ3]

/-- The rotation of the plane by the angle `θ`. -/
def rot (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

/-- The matrix `A = !![2, 1; 0, 1]` of the operator-norm figure. -/
def Afig : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 0, 1]

/-- The symmetric matrix `!![2, 1; 1, 2]` of the spectral figure. -/
def Asym : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 1, 2]

/-- The real tridiagonal matrix `!![2, 1, 0; 1, 2, 1; 0, 1, 2]`. -/
def Atri : Matrix (Fin 3) (Fin 3) ℝ := !![2, 1, 0; 1, 2, 1; 0, 1, 2]

/-- The complex Hermitian tridiagonal matrix `!![2, i, 0; -i, 2, i; 0, -i, 2]`. -/
def Mtri : Matrix (Fin 3) (Fin 3) ℂ := !![2, Complex.I, 0; -Complex.I, 2, Complex.I; 0, -Complex.I, 2]

/-- The complex matrix `!![1, i; 0, 2]` of the adjoint example. -/
def Aadj : Matrix (Fin 2) (Fin 2) ℂ := !![1, Complex.I; 0, 2]

/-- The complex matrix `!![1, i, 0; 2, 0, 1 - i; 0, 3i, 1]` of the adjoint exercise. -/
def A3c : Matrix (Fin 3) (Fin 3) ℂ := !![1, Complex.I, 0; 2, 0, 1 - Complex.I; 0, 3 * Complex.I, 1]

/-- The projection `!![1/2, 1/2; 1/2, 1/2]` onto the diagonal of the plane. -/
def Pdiag : Matrix (Fin 2) (Fin 2) ℝ := !![1 / 2, 1 / 2; 1 / 2, 1 / 2]

/-- The projection `(1 + σ₂)/2` onto the line of `(1, i)/√2`. -/
def Pplus : Matrix (Fin 2) (Fin 2) ℂ := !![1 / 2, -Complex.I / 2; Complex.I / 2, 1 / 2]

/-- The projection `(1 - σ₂)/2` onto the line of `(1, -i)/√2`. -/
def Pminus : Matrix (Fin 2) (Fin 2) ℂ := !![1 / 2, Complex.I / 2; -Complex.I / 2, 1 / 2]

/-- The rotation generator `J = !![0, -1; 1, 0]`. -/
def J : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The truncated shift `S_N`: `1` on the first subdiagonal. -/
def shiftN (N : ℕ) : Matrix (Fin N) (Fin N) ℚ := Matrix.of fun i j => if (i : ℕ) = j + 1 then 1 else 0

/-- The Hermitian matrix `M = !![1, 2 - i; 2 + i, 3]` of the Pauli-basis exercise. -/
def Mex : Matrix (Fin 2) (Fin 2) ℂ := !![1, 2 - Complex.I; 2 + Complex.I, 3]

/-- The projection onto the first coordinate of `ℂ²`. -/
def P1 : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, 0]

/-- The Fourier matrix `F_{jk} = e^{2π i jk / 4} / 2` of size `4`. -/
def F4 : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1 / 2, 1 / 2, 1 / 2, 1 / 2; 1 / 2, Complex.I / 2, -1 / 2, -Complex.I / 2;
     1 / 2, -1 / 2, 1 / 2, -1 / 2; 1 / 2, -Complex.I / 2, -1 / 2, Complex.I / 2]

end
end KaluzaKlein.HilbertSpaces.Basic

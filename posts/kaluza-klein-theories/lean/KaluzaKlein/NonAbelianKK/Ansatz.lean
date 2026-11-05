import Mathlib

/-!
# The non-abelian Kaluza-Klein ansatz: block form and inverse (D5, task 5.1)

For an internal space with round (positive-definite) metric `γ` and Killing vectors `K_a`, the ansatz is
`ds² = g_{μν} dx^μ dx^ν - γ_{mn} (dy^m + K^m_a A^a_μ dx^μ)(dy^n + K^n_b A^b_ν dx^ν)`, signature `(+,-,…,-)`: the
internal block is `h = -γ`. At a point it is the block matrix `[[g + Bᵀ h B, Bᵀ h], [h B, h]]` with `B = K A`
(internal × spacetime), and its inverse is `[[g⁻¹, -g⁻¹ Bᵀ], [-B g⁻¹, h⁻¹ + B g⁻¹ Bᵀ]]`. The identity is proved
for abstract matrices of any sizes and any signs, so it covers the circle (`B = A`, `h = -φ²`), the sphere
(`h = -diag(1, sin²θ)`) and every other internal space at once.
-/

namespace KaluzaKlein.NonAbelianKK

open Matrix

variable {μ m : Type*} [Fintype μ] [Fintype m] [DecidableEq μ] [DecidableEq m]

/-- The ansatz metric at a point: `g` on spacetime, `h` on the internal space, `B = K A` the mixing. -/
def ansatz (g : Matrix μ μ ℝ) (h : Matrix m m ℝ) (B : Matrix m μ ℝ) : Matrix (μ ⊕ m) (μ ⊕ m) ℝ :=
  Matrix.fromBlocks (g + Bᵀ * h * B) (Bᵀ * h) (h * B) h

/-- The candidate inverse, built from inverses `ginv`, `hinv` of the blocks. -/
def ansatzInv (ginv : Matrix μ μ ℝ) (hinv : Matrix m m ℝ) (B : Matrix m μ ℝ) : Matrix (μ ⊕ m) (μ ⊕ m) ℝ :=
  Matrix.fromBlocks ginv (-(ginv * Bᵀ)) (-(B * ginv)) (hinv + B * ginv * Bᵀ)

/-- `G G⁻¹ = 1` whenever `g ginv = 1` and `h hinv = 1`: the block inverse of the non-abelian ansatz. -/
theorem ansatz_mul_inv (g ginv : Matrix μ μ ℝ) (h hinv : Matrix m m ℝ) (B : Matrix m μ ℝ)
    (hg : g * ginv = 1) (hh : h * hinv = 1) :
    ansatz g h B * ansatzInv ginv hinv B = 1 := by
  have hgB : g * ginv * Bᵀ = Bᵀ := by rw [hg, Matrix.one_mul]
  have hBh : Bᵀ * h * hinv = Bᵀ := by rw [Matrix.mul_assoc, hh, Matrix.mul_one]
  unfold ansatz ansatzInv
  rw [Matrix.fromBlocks_multiply, ← Matrix.fromBlocks_one]
  congr 1 <;>
  · simp only [Matrix.mul_add, Matrix.add_mul, Matrix.neg_mul, Matrix.mul_neg, ← Matrix.mul_assoc, hg, hgB, hBh,
      hh, Matrix.one_mul, Matrix.mul_one]
    abel

/-- Also `G⁻¹ G = 1`, given `ginv g = 1` and `hinv h = 1`. -/
theorem inv_mul_ansatz (g ginv : Matrix μ μ ℝ) (h hinv : Matrix m m ℝ) (B : Matrix m μ ℝ)
    (hg : ginv * g = 1) (hh : hinv * h = 1) :
    ansatzInv ginv hinv B * ansatz g h B = 1 := by
  have hBg : B * ginv * g = B := by rw [Matrix.mul_assoc, hg, Matrix.mul_one]
  have hhB : hinv * h * B = B := by rw [hh, Matrix.one_mul]
  unfold ansatz ansatzInv
  rw [Matrix.fromBlocks_multiply, ← Matrix.fromBlocks_one]
  congr 1 <;>
  · simp only [Matrix.mul_add, Matrix.add_mul, Matrix.neg_mul, Matrix.mul_neg, ← Matrix.mul_assoc, hg, hBg, hhB,
      hh, Matrix.one_mul, Matrix.mul_one]
    abel

/-- The circle as a special case: one internal direction, `h = -φ²` (so `hinv = -φ⁻²`), `B = A`. The block entries
reproduce the inverse of Part III, `G^{yy} = -φ⁻² + A g⁻¹ Aᵀ`. -/
theorem ansatzInv_circle (ginv : Matrix μ μ ℝ) (φ : ℝ) (A : Matrix Unit μ ℝ) (hφ : φ ≠ 0) :
    ansatzInv ginv (Matrix.of fun _ _ => -(φ ^ 2)⁻¹) A (Sum.inr ()) (Sum.inr ()) =
      -(φ ^ 2)⁻¹ + (A * ginv * Aᵀ) () () := by
  simp [ansatzInv, Matrix.fromBlocks]

end KaluzaKlein.NonAbelianKK

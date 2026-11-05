import Mathlib

/-!
# The ansatz metric as a matrix: volume factor and the gauge transformation (D3 steps 5-6, task 4.6)

The Kaluza-Klein metric at a point is the block matrix `[[g - φ² A Aᵀ, -φ² A], [-φ² Aᵀ, -φ²]]` on `Fin 4 ⊕ Unit`
(signature `(+,-,-,-,-)`: the circle direction is spacelike). Its determinant is `-φ² det g`, positive for a
Lorentzian `g`, so the volume element is `√|G| = φ √(-g)`, and the coordinate change `y ↦ y + ε(x)` acts on it as a congruence by the Jacobian
and leaves it of the same form with `A ↦ A - ∂ε`: the abelian gauge transformation is a coordinate transformation
along the circle.
-/

namespace KaluzaKlein.AbelianKK.Gauge

open Matrix

abbrev I5 := Fin 4 ⊕ Unit

/-- The ansatz metric at a point, as a `5 × 5` matrix. -/
def ansatz (g : Matrix (Fin 4) (Fin 4) ℝ) (A : Fin 4 → ℝ) (φ : ℝ) : Matrix I5 I5 ℝ :=
  Matrix.fromBlocks (g - φ ^ 2 • Matrix.vecMulVec A A) (Matrix.of fun μ _ => -(φ ^ 2 * A μ))
    (Matrix.of fun _ ν => -(φ ^ 2 * A ν)) (Matrix.of fun _ _ => -(φ ^ 2))

/-- `det G = -φ² det g`: the volume factor `√|G| = φ √(-g)` (`det g < 0` for a Lorentzian `g`, so `det G > 0`). -/
theorem det_ansatz (g : Matrix (Fin 4) (Fin 4) ℝ) (A : Fin 4 → ℝ) (φ : ℝ) (hφ : φ ≠ 0) :
    (ansatz g A φ).det = -(φ ^ 2) * g.det := by
  have hφ2 : φ ^ 2 ≠ 0 := pow_ne_zero 2 hφ
  -- the lower-right block is the `1 × 1` matrix `-φ²`, invertible
  have hinv1 : Matrix.of (fun (_ : Unit) (_ : Unit) => (-(φ ^ 2))⁻¹) * Matrix.of (fun (_ : Unit) (_ : Unit) => -(φ ^ 2)) = 1 := by
    ext i j; simp [Matrix.mul_apply, hφ2]
  have hinv2 : Matrix.of (fun (_ : Unit) (_ : Unit) => -(φ ^ 2)) * Matrix.of (fun (_ : Unit) (_ : Unit) => (-(φ ^ 2))⁻¹) = 1 := by
    ext i j; simp [Matrix.mul_apply, hφ2]
  letI : Invertible (Matrix.of fun (_ : Unit) (_ : Unit) => -(φ ^ 2)) := ⟨_, hinv1, hinv2⟩
  unfold ansatz
  rw [Matrix.det_fromBlocks₂₂]
  have hD : (Matrix.of fun (_ : Unit) (_ : Unit) => -(φ ^ 2)).det = -(φ ^ 2) := by simp [Matrix.det_unique]
  have hS : g - φ ^ 2 • Matrix.vecMulVec A A
      - Matrix.of (fun μ (_ : Unit) => -(φ ^ 2 * A μ)) * ⅟(Matrix.of fun (_ : Unit) (_ : Unit) => -(φ ^ 2))
        * Matrix.of (fun (_ : Unit) ν => -(φ ^ 2 * A ν)) = g := by
    have hinv : ⅟(Matrix.of fun (_ : Unit) (_ : Unit) => -(φ ^ 2)) = Matrix.of fun (_ : Unit) (_ : Unit) => (-(φ ^ 2))⁻¹ := rfl
    rw [hinv]
    ext μ ν
    simp [Matrix.mul_apply, Matrix.vecMulVec, Matrix.add_apply, Matrix.sub_apply]
    field_simp
    ring
  rw [hD, hS]

/-- The Jacobian `∂x/∂x'` of `y' = y + ε(x)`, `x' = x`: the identity on spacetime, `∂y/∂x'^μ = -∂_μ ε`, `∂y/∂y' = 1`. -/
def jacobian (dε : Fin 4 → ℝ) : Matrix I5 I5 ℝ :=
  Matrix.fromBlocks 1 0 (Matrix.of fun _ ν => -dε ν) 1

/-- The coordinate change `y ↦ y + ε(x)` maps the ansatz with `A` to the ansatz with `A - ∂ε`:
`Jᵀ G J = G'`. The abelian gauge transformation `δA_μ = -∂_μ ε` is a coordinate transformation of the circle. -/
theorem gauge_shift (g : Matrix (Fin 4) (Fin 4) ℝ) (A dε : Fin 4 → ℝ) (φ : ℝ) :
    (jacobian dε)ᵀ * ansatz g A φ * jacobian dε = ansatz g (fun μ => A μ - dε μ) φ := by
  ext i j
  rcases i with μ | _ <;> rcases j with ν | _ <;>
    simp [ansatz, jacobian, Matrix.mul_apply, Matrix.fromBlocks, Fintype.sum_sum_type, Matrix.vecMulVec,
      Matrix.one_apply] <;> ring

end KaluzaKlein.AbelianKK.Gauge

import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra and calculus facts for the primer chapters

Small statements displayed in the vector-space, dual-space, tensor-product and calculus primers: dimensions, a
kernel vector, dual-basis evaluations, values of the Minkowski form, the dimension of a tensor product, the
Jacobian determinant of the polar map and two derivatives.
-/

namespace KaluzaKlein.Linear

open Matrix

/-- `ℝ³` has dimension three. -/
theorem finrank_R3 : Module.finrank ℝ (Fin 3 → ℝ) = 3 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-- The matrix of the exercise on kernels and images. -/
def A : Matrix (Fin 2) (Fin 3) ℚ := !![1, 2, 3; 2, 4, 6]

/-- `(1, 1, -1)` lies in the kernel of `A`. -/
theorem A_kernel_vector : A.mulVec ![1, 1, -1] = 0 := by
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, KaluzaKlein.Linear.A,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The two rows of `A` are proportional: the second is twice the first. -/
theorem A_rows_proportional : A 1 = (2 : ℚ) • A 0 := by
  show ![(2 : ℚ), 4, 6] = (2 : ℚ) • ![(1 : ℚ), 2, 3]
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The second dual-basis vector of the standard basis of `ℝ³` reads off the second coordinate. -/
theorem dual_basis_eval : (Pi.basisFun ℝ (Fin 3)).coord 1 ![5, 7, 9] = 7 := by simp

/-- The dual basis evaluates to the Kronecker delta on the basis. -/
theorem dual_basis_delta (i j : Fin 3) : (Pi.basisFun ℝ (Fin 3)).coord i (Pi.basisFun ℝ (Fin 3) j) = if i = j then 1 else 0 := by
  fin_cases i <;> fin_cases j <;> simp

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (0) (0) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (0) (1) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (0) (2) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (1) (0) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (1) (1) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (1) (2) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (2) (0) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (2) (1) := by
  simp [Pi.single_apply]

example : veriphy_instance% KaluzaKlein.Linear.dual_basis_delta (2) (2) := by
  simp [Pi.single_apply]

attribute [veriphy_tested kernel_test 9 0] KaluzaKlein.Linear.dual_basis_delta
-- test: la-dual-basis-delta-test.sage

/-- The dual basis of the standard basis of `ℝⁿ` reads off coordinates: `e^i(x) = x^i`. -/
theorem coord_basisFun (n : ℕ) (i : Fin n) (x : Fin n → ℝ) : (Pi.basisFun ℝ (Fin n)).coord i x = x i := by
  simp

example : veriphy_instance% KaluzaKlein.Linear.coord_basisFun (1) (0) (![5]) := by
  simp

example : veriphy_instance% KaluzaKlein.Linear.coord_basisFun (3) (1) (![5, 7, 9]) := by
  simp

example : veriphy_instance% KaluzaKlein.Linear.coord_basisFun (4) (3) (![1, -2, 1 / 2, 3]) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Linear.coord_basisFun
-- test: la-coord-basisfun-test.sage

/-- The Minkowski metric on `ℝ⁴` with signature `(+,-,-,-)`. -/
def eta : Matrix (Fin 4) (Fin 4) ℝ := Matrix.diagonal ![1, -1, -1, -1]

/-- `η(v, v) = 0` for the null vector `v = (1, 1, 0, 0)`. -/
theorem eta_null : ![(1 : ℝ), 1, 0, 0] ⬝ᵥ eta.mulVec ![1, 1, 0, 0] = 0 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, KaluzaKlein.Linear.eta,
        Matrix.diagonal_apply, Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- `η(w, w) = 1` for the timelike vector `w = (1, 0, 0, 0)`. -/
theorem eta_timelike : ![(1 : ℝ), 0, 0, 0] ⬝ᵥ eta.mulVec ![1, 0, 0, 0] = 1 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, KaluzaKlein.Linear.eta,
        Matrix.diagonal_apply, Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- `η` is symmetric. -/
theorem eta_symm : eta.transpose = eta := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, KaluzaKlein.Linear.eta,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- `η` is non-degenerate: its determinant is `-1`. -/
theorem eta_det : eta.det = -1 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, KaluzaKlein.Linear.eta,
        Matrix.diagonal_apply, Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- `ℝ³ ⊗ ℝ³` has dimension `9`. -/
theorem finrank_tensor : Module.finrank ℝ (TensorProduct ℝ (Fin 3 → ℝ) (Fin 3 → ℝ)) = 9 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-- The Jacobian determinant of the polar map `(r, φ) ↦ (r cos φ, r sin φ)` is `r`. -/
theorem polar_jacobian_det (r φ : ℝ) :
    Matrix.det !![Real.cos φ, -r * Real.sin φ; Real.sin φ, r * Real.cos φ] = r := by
  simp only [Matrix.det_fin_two_of]
  linear_combination r * Real.sin_sq_add_cos_sq φ

/-- The partial derivative of `x² y + sin y` in `x` at `(x, y)` is `2 x y`. -/
theorem partial_x (x y : ℝ) : deriv (fun x : ℝ => x ^ 2 * y + Real.sin y) x = 2 * x * y := by
  simp

/-- The chain rule on `sin (t²)`. -/
theorem chain_rule_example (t : ℝ) : deriv (fun t : ℝ => Real.sin (t ^ 2)) t = Real.cos (t ^ 2) * (2 * t) := by
  have h : HasDerivAt (fun t : ℝ => Real.sin (t ^ 2)) (Real.cos (t ^ 2) * (2 * t)) t := by
    have := (hasDerivAt_pow 2 t).sin
    simpa using this
  exact h.deriv

end KaluzaKlein.Linear

/-! ## Linear algebra group chapter: Veriphy-certified statements

Each result below has its computational core proved by a Veriphy certificate (recipe named in the docstring); the
committed proof is the Sage-free replacement that `sage_run` emitted, regenerable from `.veriphy/certificates/`. -/

namespace KaluzaKlein.Linear

/-- The vectors of the span example: `v₁ = (1, 2, 0)`, `v₂ = (0, 1, 1)`, `v₃ = (1, 3, 1)`. -/
def v1 : Fin 3 → ℚ := ![1, 2, 0]
def v2 : Fin 3 → ℚ := ![0, 1, 1]
def v3 : Fin 3 → ℚ := ![1, 3, 1]

/-- `v₃` is a linear combination of `v₁` and `v₂`: coefficients exist solving the three component equations
(Veriphy `witness` recipe finds `a = b = 1`). -/
theorem v3_combination_coefficients :
    ∃ a b : ℚ, a * 1 + b * 0 = 1 ∧ a * 2 + b * 1 = 3 ∧ a * 0 + b * 1 = 1 := by
  exact ⟨1, 1, by norm_num⟩
  -- certificate: witness.sage

/-- `v₃` lies in the span of `v₁` and `v₂`. -/
theorem v3_mem_span : v3 ∈ Submodule.span ℚ {v1, v2} := by
  rw [Submodule.mem_span_pair]
  obtain ⟨a, b, h1, h2, h3⟩ := v3_combination_coefficients
  refine ⟨a, b, ?_⟩
  ext i; fin_cases i <;> simp [v1, v2, v3] <;> linarith

/-- The basis of the plane used throughout: `f₁ = (2, 1)`, `f₂ = (1, 2)`. -/
def f1 : Fin 2 → ℚ := ![2, 1]
def f2 : Fin 2 → ℚ := ![1, 2]

/-- A vanishing combination of `f₁, f₂` has both coefficients zero (Veriphy `ideal_membership` recipe: each
coefficient is a combination of the two component equations). -/
theorem f_independent_coefficients (a b : ℚ) (h1 : a * 2 + b * 1 = 0) (h2 : a * 1 + b * 2 = 0) :
    a = 0 ∧ b = 0 := by
  refine ⟨?_, ?_⟩
  · linear_combination (2 / 3) * h1 + (-1 / 3) * h2
    -- certificate: ideal_membership.sage
  · linear_combination (-1 / 3) * h1 + (2 / 3) * h2
    -- certificate: ideal_membership.sage

/-- `f₁, f₂` are linearly independent. -/
theorem f_linearIndependent : LinearIndependent ℚ ![f1, f2] := by
  show LinearIndependent ℚ ![![(2 : ℚ), 1], ![1, 2]]
  veriphy_certified
    (exact
        Veriphy.Linear.linearIndependent_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-independence.sage

/-- The coordinates of `(5, 4)` in the basis `f`: a solution of `2a + b = 5`, `a + 2b = 4` exists (Veriphy
`witness` recipe finds `a = 2`, `b = 1`). -/
theorem coordinates_f_exist : ∃ a b : ℚ, 2 * a + b = 5 ∧ a + 2 * b = 4 := by
  exact ⟨2, 1, by norm_num⟩
  -- certificate: witness.sage

/-- `(5, 4) = 2 f₁ + 1 f₂`. -/
theorem coordinates_f : (2 : ℚ) • f1 + (1 : ℚ) • f2 = ![5, 4] := by
  show (2 : ℚ) • ![(2 : ℚ), 1] + (1 : ℚ) • ![(1 : ℚ), 2] = ![5, 4]
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The `3 × 3` real matrices form a space of dimension `9`. -/
theorem finrank_matrix3 : Module.finrank ℝ (Matrix (Fin 3) (Fin 3) ℝ) = 9 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-- The dual basis of `f`, as the rows of the inverse of the matrix with columns `f₁, f₂`: it multiplies that
matrix to the identity (Veriphy `matrix_from_goal` recipe). -/
theorem dual_f_mul : !![(2 : ℚ) / 3, -1 / 3; -1 / 3, 2 / 3] * !![2, 1; 1, 2] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num
  -- certificate: matrix_from_goal.sage

/-- The dual basis of `f` reads off the coordinates `(2, 1)` of `(5, 4)`. -/
theorem dual_f_coordinates : (!![(2 : ℚ) / 3, -1 / 3; -1 / 3, 2 / 3]).mulVec ![5, 4] = ![2, 1] := by
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The matrix of the derivative on polynomials of degree at most two, in the bases `1, x, x²` and `1, x`. -/
def D : Matrix (Fin 2) (Fin 3) ℚ := !![0, 1, 0; 0, 0, 2]

/-- The constants are in the kernel of the derivative. -/
theorem D_kernel_const : D.mulVec ![1, 0, 0] = 0 := by
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, KaluzaKlein.Linear.D,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The degenerate form of the signature examples. -/
def S : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 2, 4]

/-- Its determinant vanishes (Veriphy `matrix_from_goal` recipe, closed `det` formula). -/
theorem S_det : Matrix.det !![(1 : ℚ), 2; 2, 4] = 0 := by
  rw [Matrix.det_fin_two_of]
  ring
  -- certificate: matrix_from_goal.sage

/-- `(2, -1)` pairs to zero with everything: the form is degenerate. -/
theorem S_kernel : S.mulVec ![2, -1] = 0 := by
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, KaluzaKlein.Linear.S,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- Sylvester: the congruence `Pᵀ S P` with `P = (1, -2; 0, 1)` diagonalises the degenerate form to `diag(1, 0)`
(Veriphy `matrix_from_goal` recipe). -/
theorem S_sylvester :
    !![(1 : ℚ), 0; -2, 1] * !![1, 2; 2, 4] * !![1, -2; 0, 1] = !![1, 0; 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num
  -- certificate: matrix_from_goal.sage

/-- The quadratic form of the degenerate matrix is a square, hence non-negative (Veriphy `sos` recipe). -/
theorem S_quadratic_nonneg (x y : ℝ) : 0 ≤ x ^ 2 + 4 * x * y + 4 * y ^ 2 := by
  linear_combination sq_nonneg (x + 2 * y)
  -- certificate: sos.sage

/-- Lowering the index of `(2, 1, 0, 3)` with `η` gives `(2, -1, 0, -3)`. -/
theorem eta_lower : eta.mulVec ![2, 1, 0, 3] = ![2, -1, 0, -3] := by
  veriphy_certified
    ((  ext i
        fin_cases i <;>
          simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, KaluzaKlein.Linear.eta,
            Matrix.diagonal_apply, Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- `(2, 1, 0, 3)` is spacelike: `η(v, v) = -6`. -/
theorem eta_spacelike : ![(2 : ℝ), 1, 0, 3] ⬝ᵥ eta.mulVec ![2, 1, 0, 3] = -6 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, KaluzaKlein.Linear.eta,
        Matrix.diagonal_apply, Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- The components of the elementary tensor `u ⊗ v` for `u = (1, 2, 3)`, `v = (1, 0, -1)` form the outer product. -/
theorem outer_product : Matrix.vecMulVec ![(1 : ℚ), 2, 3] ![1, 0, -1] = !![1, 0, -1; 2, 0, -2; 3, 0, -3] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ, Matrix.diagonal_apply,
            Matrix.vecMulVec_apply]
        all_goals ring1))
  -- certificate: la-matrix-named.sage

/-- The contraction of the covector `a = (1, 0, -1)` with `u = (1, 2, 3)`. -/
theorem contraction_a_u : ![(1 : ℚ), 0, -1] ⬝ᵥ ![1, 2, 3] = -2 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, Matrix.diagonal_apply,
        Matrix.vecMulVec_apply]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- The trace of the type `(1, 1)` tensor of the contraction example is `9` (Veriphy `matrix_from_goal` recipe,
closed `trace` formula). -/
theorem trace_T : Matrix.trace !![(1 : ℚ), 2, 0; 0, 3, 1; 0, 0, 5] = 9 := by
  rw [Matrix.trace_fin_three_of]
  ring
  -- certificate: matrix_from_goal.sage

/-- The trace does not depend on the basis: `tr (P⁻¹ T P) = tr T` for any invertible `P`. -/
theorem trace_basis_independent {n : ℕ} (T P Q : Matrix (Fin n) (Fin n) ℚ) (hPQ : Q * P = 1) :
    Matrix.trace (Q * T * P) = Matrix.trace T := by
  rw [Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc, mul_eq_one_comm.mp hPQ, Matrix.mul_one]

example : veriphy_instance% KaluzaKlein.Linear.trace_basis_independent (0) (0) (0) (0) ((by ext i j; exact i.elim0)) := by
  simp [Matrix.trace]

example : veriphy_instance% KaluzaKlein.Linear.trace_basis_independent (2) (!![1, 2; 3, 4]) (!![2, 1; 1, 1]) (!![1, -1; -1, 2]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.Linear.trace_basis_independent (3) (!![1, 2, 0; 0, 3, 1; 0, 0, 5]) (!![1, 0, 0; 1, 1, 0; 0, 0, 1]) (!![1, 0, 0; -1, 1, 0; 0, 0, 1]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) := by
  norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.Linear.trace_basis_independent
-- test: la-trace-basis-independent-test.sage

/-- `V* ⊗ V` has dimension `9` for `V = ℝ³`. -/
theorem finrank_dual_tensor :
    Module.finrank ℝ (TensorProduct ℝ (Module.Dual ℝ (Fin 3 → ℝ)) (Fin 3 → ℝ)) = 9 := by
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

end KaluzaKlein.Linear

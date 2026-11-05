import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Elementary tensors"

The claims of the section "Elementary tensors" of `tutorial/part-i-02-linear-algebra.qmd`: `v ⊗ w` with components
`v^i w^a` (the outer product), elementary tensors as matrices of rank at most one, sums of elementary tensors, product
metrics, and the evaluation of `u ⊗ v` on two covectors.
-/

namespace KaluzaKlein.LinearAlgebra.ElementaryTensors

open Matrix Module TensorProduct

/-! ## def-elementary-tensor -/

/-- `v ⊗ w` is bilinear in `(v, w)` (mathlib's `TensorProduct.mk`), and its components are the outer product
`v^i w^a`. -/
theorem outer_components {m n : ℕ} (v : Fin m → ℝ) (w : Fin n → ℝ) (i : Fin m) (a : Fin n) :
    Matrix.vecMulVec v w i a = v i * w a := rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.outer_components (1) (1) (![0]) (![5]) (0) (0) := by
  simp [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.outer_components (3) (3) (![1, 2, 3]) (![1, 0, -1]) (2) (0) := by
  simp [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.outer_components (2) (3) (![1 / 2, -1]) (![2, 0, 4]) (1) (2) := by
  simp [Matrix.vecMulVec_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.outer_components
-- test: la-outer-components-test.sage

theorem tmul_bilinear {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W] (v v' : V) (w : W)
    (t : ℝ) : (v + t • v') ⊗ₜ[ℝ] w = v ⊗ₜ w + t • (v' ⊗ₜ w) := by
  rw [add_tmul, smul_tmul']

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.tmul_bilinear ((Fin 2 → ℝ)) ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (0) (0) (0) (0) := by
  rw [add_tmul, smul_tmul']

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.tmul_bilinear ((Fin 2 → ℝ)) ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (![1, 2]) (![0, 1]) (![3, -1]) (2) := by
  rw [add_tmul, smul_tmul']

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.tmul_bilinear (ℝ) ((Fin 3 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (1) (-2) (![1, 0, 1]) (1 / 2) := by
  rw [add_tmul, smul_tmul']

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.tmul_bilinear
-- test: la-tmul-bilinear-test.sage

/-! ## exm-elementary-outer, exm-elementary-matrix, exr-rank-one -/

/-- An elementary tensor of `ℝ³ ⊗ ℝ³` has rank at most one as a matrix. -/
theorem rank_elementary_le (u v : Fin 3 → ℚ) : (Matrix.vecMulVec u v).rank ≤ 1 :=
  Matrix.rank_vecMulVec_le u v

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.rank_elementary_le (0) (0) := by
  exact Veriphy.Linear.rank_le_of_factor _ !![0; 0; 0] !![0, 0, 0] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.rank_elementary_le (![1, 2, 3]) (![1, 0, -1]) := by
  exact Veriphy.Linear.rank_le_of_factor _ !![1; 2; 3] !![1, 0, -1] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.rank_elementary_le (![2, -1, 0]) (![0, 0, 5]) := by
  exact Veriphy.Linear.rank_le_of_factor _ !![2; -1; 0] !![0, 0, 5] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.rank_elementary_le
-- test: la-rank-elementary-le-test.sage

/-- The outer product `(1, 2, 3) ⊗ (1, 0, -1)` has rank exactly one. -/
theorem rank_outer : (!![(1 : ℚ), 0, -1; 2, 0, -2; 3, 0, -3]).rank = 1 := by
  veriphy_certified
    ( apply le_antisymm
      ·
        exact
          Veriphy.Linear.rank_le_of_factor _ !![1; 2; 3] !![1, 0, -1]
            (by
              ext i j <;> fin_cases i <;> fin_cases j <;>
                  simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
                norm_num)
      ·
        exact
          Veriphy.Linear.le_rank_of_minor _ ![0] ![0]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
  -- certificate: la-rank.sage

/-- The identity is `e₁ ⊗ e₁ + e₂ ⊗ e₂ + e₃ ⊗ e₃`. -/
theorem identity_sum_elementary :
    !![(1 : ℚ); 0; 0] * !![(1 : ℚ), 0, 0] + !![(0 : ℚ); 1; 0] * !![(0 : ℚ), 1, 0] + !![(0 : ℚ); 0; 1] * !![(0 : ℚ), 0, 1] =
      (1 : Matrix (Fin 3) (Fin 3) ℚ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- The identity has rank three, so it is not elementary. -/
theorem identity_not_elementary : ¬ ∃ u v : Fin 3 → ℚ, Matrix.vecMulVec u v = 1 := by
  rintro ⟨u, v, h⟩
  have h1 := Matrix.rank_vecMulVec_le u v
  have h3 : Matrix.rank (!![1, 0, 0; 0, 1, 0; 0, 0, 1] : Matrix (Fin 3) (Fin 3) ℚ) = 3 := by
    veriphy_certified
      ( apply le_antisymm
        · exact Matrix.rank_le_height _
        ·
          exact
            Veriphy.Linear.le_rank_of_minor _ ![0, 1, 2] ![0, 1, 2]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
    -- certificate: la-rank.sage
  rw [h, show (1 : Matrix (Fin 3) (Fin 3) ℚ) = !![1, 0, 0; 0, 1, 0; 0, 0, 1] by
    ext i j; fin_cases i <;> fin_cases j <;> rfl, h3] at h1
  norm_num at h1

/-- `diag(1, 1, 0) = e₁ ⊗ e₁ + e₂ ⊗ e₂` has rank two: a sum of two elementary tensors that is not elementary. -/
theorem rank_diag_110 : (Matrix.diagonal ![(1 : ℚ), 1, 0]).rank = 2 := by
  veriphy_certified
    ( apply le_antisymm
      ·
        exact
          Veriphy.Linear.rank_le_of_factor _ !![1, 0; 0, 1; 0, 0] !![1, 0, 0; 0, 1, 0]
            (by
              ext i j <;> fin_cases i <;> fin_cases j <;>
                  simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
                norm_num)
      ·
        exact
          Veriphy.Linear.le_rank_of_minor _ ![0, 1] ![0, 1]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
  -- certificate: la-rank.sage

theorem diag_110_not_elementary : ¬ ∃ u v : Fin 3 → ℚ, Matrix.vecMulVec u v = Matrix.diagonal ![1, 1, 0] := by
  rintro ⟨u, v, h⟩
  have h1 := Matrix.rank_vecMulVec_le u v
  have h3 : Matrix.rank (Matrix.diagonal ![(1 : ℚ), 1, 0]) = 2 := by
    veriphy_certified
      ( apply le_antisymm
        ·
          exact
            Veriphy.Linear.rank_le_of_factor _ !![1, 0; 0, 1; 0, 0] !![1, 0, 0; 0, 1, 0]
              (by
                ext i j <;> fin_cases i <;> fin_cases j <;>
                    simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.diagonal] <;>
                  norm_num)
        ·
          exact
            Veriphy.Linear.le_rank_of_minor _ ![0, 1] ![0, 1]
              (by
                simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                    Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                  norm_num))
    -- certificate: la-rank.sage
  rw [h, h3] at h1
  norm_num at h1

/-! ## exm-elementary-metric -/

/-- A metric (any matrix of components) is the sum `g_{ij} e^i ⊗ e^j` of elementary tensors of basis covectors. -/
theorem metric_sum_elementary {n : ℕ} (g : Matrix (Fin n) (Fin n) ℝ) :
    ∑ i, ∑ j, g i j • Matrix.vecMulVec (Pi.single i 1) (Pi.single j 1) = g := by
  ext a b
  simp [Matrix.sum_apply, Matrix.vecMulVec, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.metric_sum_elementary (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.metric_sum_elementary (2) (!![1, 2; 2, 4]) := by
  ext a b
  fin_cases a <;> fin_cases b <;> simp [Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.metric_sum_elementary (3) (!![1, 0, 0; 0, -1, 0; 0, 0, 2]) := by
  ext a b
  fin_cases a <;> fin_cases b <;> simp [Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.metric_sum_elementary
-- test: la-metric-sum-elementary-test.sage

/-! ## exr-elementary-evaluate -/

/-- `(u ⊗ v)(a, b) = a_i b_j u^i v^j = 2`, and `a(u) b(v) = (-2)(-1) = 2`. -/
theorem elementary_evaluate :
    !![(1 : ℚ), 0, -1] * !![(1 : ℚ), 0, -1; 2, 0, -2; 3, 0, -3] * !![(0 : ℚ); 1; 1] = !![(2 : ℚ)] ∧
      (!![(1 : ℚ), 0, -1] * !![(1 : ℚ); 2; 3]) * (!![(0 : ℚ), 1, 1] * !![(1 : ℚ); 0; -1]) = !![(2 : ℚ)] := by
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-elementary-evaluate.sage
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
    -- certificate: la-elementary-evaluate.sage

/-! ## exr-elementary-sum -/

/-- `u ⊗ v + λ u ⊗ z = u ⊗ (v + λ z)`. -/
theorem elementary_sum_left {m n : ℕ} (u : Fin m → ℝ) (v z : Fin n → ℝ) (c : ℝ) :
    Matrix.vecMulVec u v + Matrix.vecMulVec (c • u) z = Matrix.vecMulVec u (v + c • z) := by
  ext i j
  simp [Matrix.vecMulVec]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_left (1) (1) (![0]) (![0]) (![0]) (0) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_left (2) (3) (![1, 2]) (![1, 0, -1]) (![0, 1, 1]) (2) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_left (3) (2) (![1, -1, 0]) (![1 / 2, 3]) (![2, 2]) (-1) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_left
-- test: la-elementary-sum-left-test.sage

/-- `u ⊗ v + w ⊗ μ v = (u + μ w) ⊗ v`. -/
theorem elementary_sum_right {m n : ℕ} (u w : Fin m → ℝ) (v : Fin n → ℝ) (c : ℝ) :
    Matrix.vecMulVec u v + Matrix.vecMulVec w (c • v) = Matrix.vecMulVec (u + c • w) v := by
  ext i j
  simp [Matrix.vecMulVec]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_right (1) (1) (![0]) (![0]) (![0]) (0) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_right (2) (3) (![1, 2]) (![0, 1]) (![1, 0, -1]) (2) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_right (3) (2) (![1, -1, 0]) (![2, 0, 1]) (![1 / 2, 3]) (-1) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.vecMulVec_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.ElementaryTensors.elementary_sum_right
-- test: la-elementary-sum-right-test.sage

end KaluzaKlein.LinearAlgebra.ElementaryTensors

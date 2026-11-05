import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Multilinear maps"

The claims of the section "Multilinear maps" of `tutorial/part-i-02-linear-algebra.qmd`: the determinant as an
alternating multilinear map, products of functionals, the scale invariance of sectional curvature, the triple product,
the components of a trilinear map on `ℝ³`, and alternating maps.
-/

namespace KaluzaKlein.LinearAlgebra.MultilinearMaps

open Matrix Module

/-! ## def-multilinear-map, exm-multilinear-det -/

/-- The determinant is linear in each row (here: the first row of a `3 × 3` matrix). -/
theorem det_linear_first_row (a b c a' b' c' t d e f g h i : ℝ) :
    Matrix.det !![a + t * a', b + t * b', c + t * c'; d, e, f; g, h, i] =
      Matrix.det !![a, b, c; d, e, f; g, h, i] + t * Matrix.det !![a', b', c'; d, e, f; g, h, i] := by
  norm_num [Matrix.det_fin_three] <;> ring1
  -- certificate: la-matrix.sage

/-- The determinant is alternating: swapping two rows changes its sign. -/
theorem det_swap_rows (a b c d e f g h i : ℝ) :
    Matrix.det !![d, e, f; a, b, c; g, h, i] = -Matrix.det !![a, b, c; d, e, f; g, h, i] := by
  norm_num [Matrix.det_fin_three] <;> ring1
  -- certificate: la-matrix.sage

/-- The determinant as mathlib's alternating multilinear map of the rows. -/
theorem det_is_alternating {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    Matrix.detRowAlternating (fun i => A i) = A.det := rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.det_is_alternating (0) (0) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.det_is_alternating (2) (!![1, 2; 3, 4]) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.det_is_alternating (3) (!![2, 1, 0; 1, 2, 1; 0, 1, 2]) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.det_is_alternating
-- test: la-det-is-alternating-test.sage

/-! ## exm-multilinear-product -/

/-- `(u, v, w) ↦ α(u) β(v) γ(w)` is linear in each argument (shown for the first). -/
theorem product_functionals_linear {n : ℕ} (α β γ u u' v w : Fin n → ℝ) (t : ℝ) :
    (α ⬝ᵥ (u + t • u')) * (β ⬝ᵥ v) * (γ ⬝ᵥ w) =
      (α ⬝ᵥ u) * (β ⬝ᵥ v) * (γ ⬝ᵥ w) + t * ((α ⬝ᵥ u') * (β ⬝ᵥ v) * (γ ⬝ᵥ w)) := by
  rw [dotProduct_add, dotProduct_smul, smul_eq_mul]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.product_functionals_linear (0) (![]) (![]) (![]) (![]) (![]) (![]) (![]) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.product_functionals_linear (2) (![1, 2]) (![0, 1]) (![3, -1]) (![1, 1]) (![2, 0]) (![1, -1]) (![0, 5]) (2) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.product_functionals_linear (3) (![1, 0, -1]) (![2, 1, 0]) (![0, 0, 1]) (![1, 2, 3]) (![0, 1, -1]) (![2, 0, 1]) (![1, 1, 1]) (-1 / 2) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.product_functionals_linear
-- test: la-product-functionals-test.sage

/-! ## exm-multilinear-riemann -/

/-- Sectional curvature `K(v, w) = R(v, w, w, v) / (|v|²|w|² - (v·w)²)` is not bilinear: rescaling `v` by `t ≠ 0`
multiplies numerator and denominator by `t²` and leaves `K` unchanged. -/
theorem sectional_scale_invariant (r q t : ℝ) (ht : t ≠ 0) : (t ^ 2 * r) / (t ^ 2 * q) = r / q := by
  rw [mul_div_mul_left _ _ (pow_ne_zero 2 ht)]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.sectional_scale_invariant (0) (0) (1) (one_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.sectional_scale_invariant (1) (2) (3) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.sectional_scale_invariant (-1 / 2) (5) (-2) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.sectional_scale_invariant
-- test: la-sectional-scale-test.sage

/-! ## exr-multilinear-check -/

/-- The triple product is the determinant: `u · (v × w) = det(u | v | w)`. -/
theorem triple_product_det (u v w : Fin 3 → ℝ) : u ⬝ᵥ (v ⨯₃ w) = Matrix.det ![u, v, w] :=
  (triple_product_eq_det u v w)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.triple_product_det (0) (0) (0) := by
  rw [show Matrix.det ![(0 : Fin 3 → ℝ), 0, 0] = Matrix.det (Matrix.of ![(0 : Fin 3 → ℝ), 0, 0]) from rfl,
    Matrix.det_fin_three]
  simp [cross_apply, dotProduct, Fin.sum_univ_succ]
  <;> ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.triple_product_det (![1, 0, 0]) (![0, 1, 0]) (![0, 0, 1]) := by
  rw [show Matrix.det ![(![1, 0, 0] : Fin 3 → ℝ), ![0, 1, 0], ![0, 0, 1]] = Matrix.det (Matrix.of ![(![1, 0, 0] : Fin 3 → ℝ), ![0, 1, 0], ![0, 0, 1]]) from rfl,
    Matrix.det_fin_three]
  simp [cross_apply, dotProduct, Fin.sum_univ_succ]
  <;> ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.triple_product_det (![1, 2, 3]) (![0, 1, -1]) (![2, 0, 1]) := by
  rw [show Matrix.det ![(![1, 2, 3] : Fin 3 → ℝ), ![0, 1, -1], ![2, 0, 1]] = Matrix.det (Matrix.of ![(![1, 2, 3] : Fin 3 → ℝ), ![0, 1, -1], ![2, 0, 1]]) from rfl,
    Matrix.det_fin_three]
  simp [cross_apply, dotProduct, Fin.sum_univ_succ]
  <;> ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.triple_product_det
-- test: la-triple-product-test.sage

/-- `|u| (v · w)` is not linear in `u`: already `u ↦ |u|` is not linear (`|-u| = |u|`). -/
theorem norm_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = ‖x‖ := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (-1 : _) (1 : _)
      rw [hf, hf] at h
      norm_num at h)
  -- certificate: la-nonlinear.sage

/-- `(u · v)(v · w)` is quadratic in `v`: replacing `v` by `2v` multiplies it by `4`. -/
theorem quadratic_in_v {n : ℕ} (u v w : Fin n → ℝ) :
    (u ⬝ᵥ ((2 : ℝ) • v)) * (((2 : ℝ) • v) ⬝ᵥ w) = 4 * ((u ⬝ᵥ v) * (v ⬝ᵥ w)) := by
  rw [dotProduct_smul, smul_dotProduct, smul_eq_mul, smul_eq_mul]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.quadratic_in_v (0) (![]) (![]) (![]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.quadratic_in_v (2) (![1, 2]) (![3, -1]) (![0, 1]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.quadratic_in_v (3) (![1, 0, -1]) (![2, 1, 1 / 2]) (![1, 1, 1]) := by
  norm_num [dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.quadratic_in_v
-- test: la-quadratic-in-v-test.sage

/-! ## exr-trilinear-components -/

/-- The trilinear map `T(u, v, w) = u¹ v² w³ - u² v¹ w³`. -/
def T (u v w : Fin 3 → ℚ) : ℚ := u 0 * v 1 * w 2 - u 1 * v 0 * w 2

/-- Its components: `T₁₂₃ = 1`, `T₂₁₃ = -1`, all others vanish. -/
theorem T_components (i j k : Fin 3) :
    T (Pi.single i 1) (Pi.single j 1) (Pi.single k 1) =
      if (i, j, k) = (0, 1, 2) then 1 else if (i, j, k) = (1, 0, 2) then -1 else 0 := by
  fin_cases i <;> fin_cases j <;> fin_cases k <;> simp [T]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (0) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (0) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (0) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (1) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (1) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (1) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (2) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (2) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (0) (2) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (0) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (0) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (0) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (1) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (1) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (1) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (2) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (2) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (1) (2) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (0) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (0) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (0) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (1) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (1) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (1) (2) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (2) (0) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (2) (1) := by
  simp [T, Pi.single_apply]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components (2) (2) (2) := by
  simp [T, Pi.single_apply]

attribute [veriphy_tested kernel_test 27 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components
-- test: la-t-components-test.sage

/-- A trilinear map on `ℝ³` has `3³ = 27` components. -/
theorem trilinear_component_count : Fintype.card (Fin 3 × Fin 3 × Fin 3) = 27 := by simp

/-- A trilinear map in dimensions `a`, `b`, `c` has `a b c` components. -/
theorem card_fin_prod3 (a b c : ℕ) : Fintype.card (Fin a × Fin b × Fin c) = a * b * c := by
  simp [mul_assoc]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (0) (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (4) (2) (3) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (0) (0) (1) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (0) (0) (2) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (0) (1) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 (0) (1) (1) := by
  simp

attribute [veriphy_tested kernel_test 6 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3
-- test: la-card-fin-prod3-test.sage

/-- `T(u, v, w) = 1` for `u = (1, 2, 3)`, `v = (0, 1, -1)`, `w = (2, 0, 1)`. -/
theorem T_value : T ![1, 2, 3] ![0, 1, -1] ![2, 0, 1] = 1 := by
  unfold T
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply, Matrix.diagonal_apply,
        Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
      all_goals ring1)
  -- certificate: la-matrix-named.sage

/-- Linearity in the first slot: `T(2u + v, v, w) = 2 T(u, v, w) + T(v, v, w)`. -/
theorem T_linear (u v w : Fin 3 → ℚ) : T (2 • u + v) v w = 2 * T u v w + T v v w := by
  simp [T]
  ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_linear (0) (0) (0) := by
  norm_num [T]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_linear (![1, 2, 3]) (![0, 1, -1]) (![2, 0, 1]) := by
  norm_num [T]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.T_linear (![1 / 2, -1, 0]) (![3, 0, 1]) (![1, 1, 1]) := by
  norm_num [T]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.T_linear
-- test: la-t-linear-test.sage

/-! ## exr-multilinear-alternating -/

/-- An alternating map vanishes when two arguments coincide. -/
theorem alternating_repeated {ι V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]
    (f : V [⋀^ι]→ₗ[ℝ] W) (v : ι → V) (i j : ι) (h : v i = v j) (hij : i ≠ j) : f v = 0 :=
  f.map_eq_zero_of_eq v h hij

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.alternating_repeated ((Fin 2)) ((Fin 2 → ℝ)) (ℝ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (0) (![![1, 2], ![1, 2]]) (0) (1) (rfl) ((by decide)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.alternating_repeated ((Fin 2)) ((Fin 2 → ℝ)) (ℝ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Matrix.detRowAlternating) (![![1, 2], ![1, 2]]) (0) (1) (rfl) ((by decide)) := by
  show Matrix.det !![(1 : ℝ), 2; 1, 2] = 0
  norm_num [Matrix.det_fin_two_of]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.alternating_repeated ((Fin 3)) ((Fin 3 → ℝ)) (ℝ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Matrix.detRowAlternating) (![![1, 2, 3], ![0, 1, 4], ![1, 2, 3]]) (0) (2) (rfl) ((by decide)) := by
  show Matrix.det !![(1 : ℝ), 2, 3; 0, 1, 4; 1, 2, 3] = 0
  norm_num [Matrix.det_fin_three]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.alternating_repeated
-- test: la-alternating-repeated-test.sage

/-- The algebraic core of the converse: `B(v + w, v + w) = 0`, `B(v, v) = B(w, w) = 0` force `B(v, w) = -B(w, v)`. -/
theorem antisym_core (bvv bvw bwv bww s : ℝ) (hs : s = bvv + bvw + bwv + bww) (h0 : s = 0) (h1 : bvv = 0)
    (h2 : bww = 0) : bvw = -bwv := by
  linear_combination (-1) * hs + h0 + (-1) * h1 + (-1) * h2
  -- certificate: ideal_membership.sage

/-- A bilinear form vanishing on the diagonal is antisymmetric. -/
theorem antisym_of_diag_zero {V : Type*} [AddCommGroup V] [Module ℝ V] (B : LinearMap.BilinForm ℝ V)
    (h : ∀ v, B v v = 0) (v w : V) : B v w = -B w v :=
  antisym_core (B v v) (B v w) (B w v) (B w w) (B (v + w) (v + w)) (by simp [map_add]; ring) (h _) (h _) (h _)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.antisym_of_diag_zero ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (0) ((fun _ => rfl)) (![1, 2]) (![3, -1]) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.antisym_of_diag_zero ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (Matrix.toBilin' !![0, 1; -1, 0]) ((by intro v; simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]; ring)) (![1, 2]) (![3, -1]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.MultilinearMaps.antisym_of_diag_zero ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (Matrix.toBilin' !![0, 3; -3, 0]) ((by intro v; simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]; ring)) (![1 / 2, 0]) (![2, 5]) := by
  norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.MultilinearMaps.antisym_of_diag_zero
-- test: la-antisym-of-diag-zero-test.sage

/-- A matrix with two equal rows is singular. -/
theorem det_repeated_rows (a b c g h i : ℝ) : Matrix.det !![a, b, c; a, b, c; g, h, i] = 0 := by
  norm_num [Matrix.det_fin_three] <;> ring1
  -- certificate: la-matrix.sage

end KaluzaKlein.LinearAlgebra.MultilinearMaps

import Mathlib
import Veriphy.Kit.Linear
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Vector spaces"

The claims of the section "Vector spaces" of `tutorial/part-i-02-linear-algebra.qmd` (design D28): the definition of
a vector space and of a subspace (mathlib's `Module` and `Submodule`), the subspace tests of the examples and
exercises, and the dimensions of the symmetric and antisymmetric `n × n` matrices, `n (n + 1) / 2` and
`n (n - 1) / 2`, which later sections reuse. Computational cores are Veriphy certificates (`matrix_from_goal`,
`ideal_membership`, `sum_induction`); the scripts are under `.veriphy/certificates/`.
-/

namespace KaluzaKlein.LinearAlgebra.VectorSpaces

open Matrix Module

/-! ## def-vector-space -/

/-- A vector space over a field `K` is a module over `K`; a subspace is a `Submodule K V`. -/
abbrev VectorSpace (K V : Type*) [Field K] [AddCommGroup V] := Module K V

/-- `ℝⁿ` is a real vector space: the four scalar-multiplication rules of the definition. -/
theorem rn_axioms (n : ℕ) (a b : ℝ) (v w : Fin n → ℝ) :
    a • (v + w) = a • v + a • w ∧ (a + b) • v = a • v + b • v ∧ (a * b) • v = a • b • v ∧ (1 : ℝ) • v = v :=
  ⟨smul_add a v w, add_smul a b v, mul_smul a b v, one_smul ℝ v⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.rn_axioms (0) (0) (0) (![]) (![]) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> funext i <;> fin_cases i <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.rn_axioms (2) (2) (-1) (![1, 2]) (![3, -1]) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> funext i <;> fin_cases i <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.rn_axioms (3) (1 / 2) (3) (![1, 0, -2]) (![0, 5, 1]) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> funext i <;> fin_cases i <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaces.rn_axioms
-- test: la-rn-axioms-test.sage

/-! ## exm-vs-rn -/

/-- The plane `x³ = 0` of `ℝ³`, the kernel of the third coordinate, is a subspace. -/
def planeX3 : Submodule ℝ (Fin 3 → ℝ) := LinearMap.ker (LinearMap.proj 2)

theorem mem_planeX3 (v : Fin 3 → ℝ) : v ∈ planeX3 ↔ v 2 = 0 := by simp [planeX3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.mem_planeX3 (0) := by
  simp [planeX3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.mem_planeX3 (![1, 2, 0]) := by
  simp [planeX3]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.mem_planeX3 (![1, 2, 3]) := by
  simp [planeX3]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaces.mem_planeX3
-- test: la-mem-planex3-test.sage

/-- The sum of the unit vectors `e₁` and `e₂` has squared length `2` (rows times their transpose). -/
theorem unit_sum_sq_length :
    (!![(1 : ℝ), 0, 0] + !![(0 : ℝ), 1, 0]) * (!![(1 : ℝ), 0, 0] + !![(0 : ℝ), 1, 0])ᵀ = !![(2 : ℝ)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-- The unit sphere `v · v = 1` is not a subspace (it misses `0`). -/
theorem sphere_not_subspace :
    ¬ ∃ S : Submodule ℝ (Fin 3 → ℝ), (S : Set (Fin 3 → ℝ)) = {v | v ⬝ᵥ v = 1} := by
  rintro ⟨S, hS⟩
  have h0 : (0 : Fin 3 → ℝ) ∈ (S : Set (Fin 3 → ℝ)) := S.zero_mem
  rw [hS] at h0
  change (0 : Fin 3 → ℝ) ⬝ᵥ 0 = 1 at h0
  rw [dotProduct_zero] at h0
  clear hS
  veriphy_certified
    ( have h1 : (1 : ℝ) = 0 := by linear_combination (-1) * h0
      exact one_ne_zero h1)
  -- certificate: la-infeasible.sage

/-! ## Symmetric and antisymmetric matrices (exm-vs-matrices, exr-vs-symmetric-matrices, and the dimension counts of
later sections) -/

/-- The `n × n` real matrices have dimension `n²`. -/
theorem finrank_matrix_sq (n : ℕ) : finrank ℝ (Matrix (Fin n) (Fin n) ℝ) = n ^ 2 := by
  simp [Module.finrank_matrix, sq]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq (0) := by
  norm_num [Module.finrank_matrix]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq (1) := by
  norm_num [Module.finrank_matrix]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq (2) := by
  norm_num [Module.finrank_matrix]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq (3) := by
  norm_num [Module.finrank_matrix]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq
-- test: la-finrank-matrix-sq-test.sage

/-- The symmetric `n × n` real matrices, `Aᵀ = A`. -/
def symMatrices (n : ℕ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℝ) where
  carrier := {A | Aᵀ = A}
  add_mem' := by
    intro A B hA hB
    simp only [Set.mem_ofPred_eq] at *
    rw [Matrix.transpose_add, hA, hB]
  zero_mem' := by simp
  smul_mem' := by
    intro c A hA
    simp only [Set.mem_ofPred_eq] at *
    rw [Matrix.transpose_smul, hA]

/-- The antisymmetric `n × n` real matrices, `Aᵀ = -A` (the Lie algebra `𝔰𝔬(n)`). -/
def antisymMatrices (n : ℕ) : Submodule ℝ (Matrix (Fin n) (Fin n) ℝ) where
  carrier := {A | Aᵀ = -A}
  add_mem' := by
    intro A B hA hB
    simp only [Set.mem_ofPred_eq] at *
    rw [Matrix.transpose_add, hA, hB, neg_add]
  zero_mem' := by simp
  smul_mem' := by
    intro c A hA
    simp only [Set.mem_ofPred_eq] at *
    rw [Matrix.transpose_smul, hA, smul_neg]

/-- The antisymmetric matrices are closed under linear combinations: they form a subspace. -/
theorem antisym_add_smul (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) (hA : Aᵀ = -A) (hB : Bᵀ = -B) :
    (A + c • B)ᵀ = -(A + c • B) := by
  rw [Matrix.transpose_add, Matrix.transpose_smul, hA, hB, smul_neg, neg_add]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.antisym_add_smul (0) (0) (0) (0) ((by ext i j; exact i.elim0)) ((by ext i j; exact i.elim0)) := by
  ext i j
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.antisym_add_smul (2) (!![0, 1; -1, 0]) (!![0, -3; 3, 0]) (2) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.antisym_add_smul (3) (!![0, -5, 2; 5, 0, -7; -2, 7, 0]) (!![0, 0, 0; 0, 0, -1; 0, 1, 0]) (-1 / 2) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaces.antisym_add_smul
-- test: la-antisym-add-smul-test.sage

/-- Index pairs on or above the diagonal. -/
abbrev UpperPairs (n : ℕ) := {p : Fin n × Fin n // p.1 ≤ p.2}

/-- Index pairs strictly above the diagonal. -/
abbrev StrictPairs (n : ℕ) := {p : Fin n × Fin n // p.1 < p.2}

/-- A symmetric matrix is determined by its entries on and above the diagonal. -/
noncomputable def symEquiv (n : ℕ) : symMatrices n ≃ₗ[ℝ] (UpperPairs n → ℝ) where
  toFun A p := (A : Matrix (Fin n) (Fin n) ℝ) p.1.1 p.1.2
  map_add' A B := by ext p; simp
  map_smul' c A := by ext p; simp
  invFun f := ⟨Matrix.of fun i j => if h : i ≤ j then f ⟨(i, j), h⟩ else f ⟨(j, i), (not_le.mp h).le⟩, by
    show Matrix.transpose _ = _
    ext i j
    simp only [Matrix.transpose_apply, Matrix.of_apply]
    rcases lt_trichotomy i j with h | rfl | h
    · simp [h.le, not_le.mpr h]
    · simp
    · simp [h.le, not_le.mpr h]⟩
  left_inv A := by
    obtain ⟨A, hA⟩ := A
    apply Subtype.ext
    ext i j
    simp only [Matrix.of_apply]
    split_ifs with h
    · rfl
    · exact (congrFun (congrFun hA i) j)
  right_inv f := by
    ext p
    simp [p.2]

/-- An antisymmetric matrix is determined by its entries strictly above the diagonal. -/
noncomputable def antisymEquiv (n : ℕ) : antisymMatrices n ≃ₗ[ℝ] (StrictPairs n → ℝ) where
  toFun A p := (A : Matrix (Fin n) (Fin n) ℝ) p.1.1 p.1.2
  map_add' A B := by ext p; simp
  map_smul' c A := by ext p; simp
  invFun f := ⟨Matrix.of fun i j =>
      if h : i < j then f ⟨(i, j), h⟩ else if h' : j < i then -f ⟨(j, i), h'⟩ else 0, by
    show Matrix.transpose _ = _
    ext i j
    simp only [Matrix.transpose_apply, Matrix.of_apply, Matrix.neg_apply]
    rcases lt_trichotomy i j with h | rfl | h
    · simp [h, not_lt.mpr h.le]
    · simp
    · simp [h, not_lt.mpr h.le]⟩
  left_inv A := by
    obtain ⟨A, hA⟩ := A
    apply Subtype.ext
    ext i j
    have hij := congrFun (congrFun hA j) i
    simp only [Matrix.transpose_apply, Matrix.neg_apply] at hij
    simp only [Matrix.of_apply]
    split_ifs with h h'
    · rfl
    · rw [hij]
    · have : i = j := le_antisymm (not_lt.mp h') (not_lt.mp h)
      subst this
      linarith
  right_inv f := by
    ext p
    simp [p.2]

/-- Gauss: `∑_{k < n} k = n (n - 1) / 2`, over `ℚ`. -/
theorem gauss_sum_id (n : ℕ) : ∑ k ∈ Finset.range n, (k : ℚ) = (n : ℚ) * ((n : ℚ) - 1) / 2 := by
  induction n with
    | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
    | succ n ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1
  -- certificate: la-gauss-sum.sage

/-- `∑_{k < n} (k + 1) = n (n + 1) / 2`, over `ℚ`. -/
theorem gauss_sum_succ (n : ℕ) : ∑ k ∈ Finset.range n, ((k : ℚ) + 1) = (n : ℚ) * ((n : ℚ) + 1) / 2 := by
  induction n with
    | zero => rw [Finset.sum_range_zero]; (try push_cast); ring1
    | succ n ih => rw [Finset.sum_range_succ, ih]; (try push_cast); ring1
  -- certificate: la-gauss-sum.sage

theorem card_strictPairs (n : ℕ) : Fintype.card (StrictPairs n) = ∑ k ∈ Finset.range n, k := by
  rw [Fintype.card_subtype, Finset.card_filter, Fintype.sum_prod_type_right,
    ← Fin.sum_univ_eq_sum_range (fun k => k) n]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.card_filter, ← Fin.card_Iio j]
  congr 1
  ext i
  simp

theorem card_upperPairs (n : ℕ) : Fintype.card (UpperPairs n) = ∑ k ∈ Finset.range n, (k + 1) := by
  rw [Fintype.card_subtype, Finset.card_filter, Fintype.sum_prod_type_right,
    ← Fin.sum_univ_eq_sum_range (fun k => k + 1) n]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.card_filter, ← Fin.card_Iic j]
  congr 1
  ext i
  simp

/-- `dim 𝔰𝔬(n) = n (n - 1) / 2`, in the form `2 dim = n (n - 1)`. -/
theorem two_mul_finrank_antisym (n : ℕ) : 2 * finrank ℝ (antisymMatrices n) = n * (n - 1) := by
  rw [(antisymEquiv n).finrank_eq, Module.finrank_fintype_fun_eq_card, card_strictPairs]
  have h := gauss_sum_id n
  rcases n with _ | m
  · simp
  · have : ((2 * ∑ k ∈ Finset.range (m + 1), k : ℕ) : ℚ) = (((m + 1) * (m + 1 - 1) : ℕ) : ℚ) := by
      push_cast [h]
      ring
    exact_mod_cast this

/-- The symmetric matrices: `2 dim = n (n + 1)`. -/
theorem two_mul_finrank_sym (n : ℕ) : 2 * finrank ℝ (symMatrices n) = n * (n + 1) := by
  rw [(symEquiv n).finrank_eq, Module.finrank_fintype_fun_eq_card, card_upperPairs]
  have h := gauss_sum_succ n
  have : ((2 * ∑ k ∈ Finset.range n, (k + 1) : ℕ) : ℚ) = ((n * (n + 1) : ℕ) : ℚ) := by
    push_cast [h]
    ring
  exact_mod_cast this

/-- `dim 𝔰𝔬(n) = n (n - 1) / 2`. -/
theorem finrank_antisym (n : ℕ) : finrank ℝ (antisymMatrices n) = n * (n - 1) / 2 := by
  rw [← two_mul_finrank_antisym n]
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym (0) := by
  rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym (1) := by
  rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym (2) := by
  rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym (3) := by
  rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym (4) := by
  rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym
-- test: la-finrank-antisym-test.sage

/-- The symmetric `n × n` matrices have dimension `n (n + 1) / 2`. -/
theorem finrank_sym (n : ℕ) : finrank ℝ (symMatrices n) = n * (n + 1) / 2 := by
  rw [← two_mul_finrank_sym n]
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym (0) := by
  rw [(symEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym (1) := by
  rw [(symEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym (2) := by
  rw [(symEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym (3) := by
  rw [(symEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym (4) := by
  rw [(symEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]
  decide

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym
-- test: la-finrank-sym-test.sage

/-- The two counts add up to `n²` (the core: `a + b = n²` from `2a = n(n+1)`, `2b = n(n-1)`, over `ℚ`). -/
theorem sym_antisym_count (a b n : ℚ) (ha : 2 * a = n * (n + 1)) (hb : 2 * b = n * (n - 1)) : a + b = n ^ 2 := by
  linear_combination (1 / 2) * ha + (1 / 2) * hb
  -- certificate: ideal_membership.sage

/-- `dim Sym(n) + dim 𝔰𝔬(n) = n²`. -/
theorem finrank_sym_add_antisym (n : ℕ) :
    finrank ℝ (symMatrices n) + finrank ℝ (antisymMatrices n) = n ^ 2 := by
  have ha := two_mul_finrank_sym n
  have hb := two_mul_finrank_antisym n
  rcases n with _ | m
  · rw [show (0 : ℕ) ^ 2 = 0 from rfl]
    omega
  · have hb2 : 2 * finrank ℝ (antisymMatrices (m + 1)) = (m + 1) * m := by simpa using hb
    have ha' : 2 * (finrank ℝ (symMatrices (m + 1)) : ℚ) = ((m : ℚ) + 1) * (((m : ℚ) + 1) + 1) := by
      exact_mod_cast ha
    have hb' : 2 * (finrank ℝ (antisymMatrices (m + 1)) : ℚ) = ((m : ℚ) + 1) * (((m : ℚ) + 1) - 1) := by
      have : (2 * finrank ℝ (antisymMatrices (m + 1)) : ℚ) = ((m : ℚ) + 1) * m := by exact_mod_cast hb2
      rw [this]
      ring
    have key := sym_antisym_count _ _ _ ha' hb'
    exact_mod_cast key

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym (0) := by
  rw [(symEquiv _).finrank_eq, (antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card,
    Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym (1) := by
  rw [(symEquiv _).finrank_eq, (antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card,
    Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym (2) := by
  rw [(symEquiv _).finrank_eq, (antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card,
    Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym (3) := by
  rw [(symEquiv _).finrank_eq, (antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card,
    Module.finrank_fintype_fun_eq_card]
  decide

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym (4) := by
  rw [(symEquiv _).finrank_eq, (antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card,
    Module.finrank_fintype_fun_eq_card]
  decide

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_sym_add_antisym
-- test: la-finrank-sym-add-antisym-test.sage

theorem finrank_sym3 : finrank ℝ (symMatrices 3) = 6 := by rw [finrank_sym]
theorem finrank_antisym3 : finrank ℝ (antisymMatrices 3) = 3 := by rw [finrank_antisym]
theorem finrank_antisym4 : finrank ℝ (antisymMatrices 4) = 6 := by rw [finrank_antisym]
theorem finrank_sym4 : finrank ℝ (symMatrices 4) = 10 := by rw [finrank_sym]

/-! ## Rotations do not form a subspace (exm-vs-matrices, exr-vs-symmetric-matrices) -/

/-- The rotation by `90°` about the third axis is orthogonal. -/
theorem R1_orthogonal :
    !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1]ᵀ * !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-vs-symmetric-matrices.sage

theorem R1_det : Matrix.det !![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] = 1 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-vs-symmetric-matrices.sage

/-- The rotation by `90°` about the first axis is orthogonal. -/
theorem R2_orthogonal :
    !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0]ᵀ * !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-vs-symmetric-matrices.sage

theorem R2_det : Matrix.det !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0] = 1 := by
  norm_num [Matrix.det_fin_three]
  -- certificate: la-vs-symmetric-matrices.sage

/-- `(R₁ + R₂)ᵀ (R₁ + R₂)`, computed. -/
theorem rot_sum_gram :
    (!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] + !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0])ᵀ *
      (!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] + !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0]) =
      !![(2 : ℝ), -1, -1; -1, 2, 1; -1, 1, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-vs-symmetric-matrices.sage

/-- The sum of the two rotations is not a rotation: `(R₁ + R₂)ᵀ (R₁ + R₂) ≠ 1`. -/
theorem rot_sum_not_orthogonal :
    (!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] + !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0])ᵀ *
      (!![(0 : ℝ), -1, 0; 1, 0, 0; 0, 0, 1] + !![(1 : ℝ), 0, 0; 0, 0, -1; 0, 1, 0]) ≠ 1 := by
  rw [rot_sum_gram]
  intro h
  have h00 := congrFun (congrFun h 0) 0
  rw [Matrix.one_apply_eq] at h00
  change (2 : ℝ) = 1 at h00
  clear h
  veriphy_certified
    ( have h1 : (1 : ℝ) = 0 := by linear_combination h00
      exact one_ne_zero h1)
  -- certificate: la-infeasible.sage

/-- The symmetric matrices `A`, `B` of the exercise, and their sum, are symmetric. -/
theorem sym_example_sum :
    (!![(1 : ℝ), 2, 0; 2, 3, 1; 0, 1, 4] + !![(0 : ℝ), 1, 1; 1, 0, 2; 1, 2, 5])ᵀ =
      !![(1 : ℝ), 2, 0; 2, 3, 1; 0, 1, 4] + !![(0 : ℝ), 1, 1; 1, 0, 2; 1, 2, 5] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num
  -- certificate: la-vs-symmetric-matrices.sage

/-! ## exm-vs-functions -/

/-- The polynomials of degree at most `2` form a three-dimensional subspace. -/
theorem finrank_poly_deg_le_two : finrank ℝ (Polynomial.degreeLT ℝ 3) = 3 := by
  rw [(Polynomial.degreeLTEquiv ℝ 3).finrank_eq]
  veriphy_certified
    (simp [Module.finrank_tensorProduct, Module.finrank_matrix, Module.finrank_fin_fun, Subspace.dual_finrank_eq,
        Module.finrank_fintype_fun_eq_card, Module.finrank_pi_fintype, Fintype.card_fun, Fintype.card_prod,
        Complex.finrank_real_complex, Module.finrank_self])
  -- certificate: la-dim-product.sage

/-- The real functions on an infinite set (here `ℕ`) form an infinite-dimensional space. -/
theorem not_finiteDimensional_functions (X : Type) [Infinite X] : ¬ FiniteDimensional ℝ (X → ℝ) := by
  intro h
  classical
  have hli : LinearIndependent ℝ (fun i : X => (Pi.single i (1 : ℝ) : X → ℝ)) :=
    Pi.linearIndependent_single_one X ℝ
  have := hli.finite_of_isNoetherian
  exact not_finite X

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.not_finiteDimensional_functions (ℕ) (inferInstance) := by
  intro h
  classical
  have := (Pi.linearIndependent_single_one ℕ ℝ).finite_of_isNoetherian
  exact not_finite ℕ

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.not_finiteDimensional_functions (ℤ) (inferInstance) := by
  intro h
  classical
  have := (Pi.linearIndependent_single_one ℤ ℝ).finite_of_isNoetherian
  exact not_finite ℤ

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.not_finiteDimensional_functions (ℚ) (inferInstance) := by
  intro h
  classical
  have := (Pi.linearIndependent_single_one ℚ ℝ).finite_of_isNoetherian
  exact not_finite ℚ

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaces.not_finiteDimensional_functions
-- test: la-functions-infinite-test.sage

theorem functions_infinite_dimensional : ¬ FiniteDimensional ℝ (ℕ → ℝ) := not_finiteDimensional_functions ℕ

/-! ## exr-vs-subspace-check -/

/-- The plane `x + y + z = 0` is closed under addition. -/
theorem plane0_add (a b c a' b' c' : ℝ) (h : a + b + c = 0) (h' : a' + b' + c' = 0) :
    (a + a') + (b + b') + (c + c') = 0 := by
  linear_combination h + h'
  -- certificate: ideal_membership.sage

/-- The plane `x + y + z = 0` is closed under scaling. -/
theorem plane0_smul (t a b c : ℝ) (h : a + b + c = 0) : t * a + t * b + t * c = 0 := by
  linear_combination (t) * h
  -- certificate: ideal_membership.sage

/-- The plane `x + y + z = 0` as a subspace of `ℝ³`. -/
def planeSum0 : Submodule ℝ (Fin 3 → ℝ) where
  carrier := {v | v 0 + v 1 + v 2 = 0}
  add_mem' := fun {v w} hv hw => by simpa using plane0_add _ _ _ _ _ _ hv hw
  zero_mem' := by simp
  smul_mem' := fun t v hv => by simpa using plane0_smul t _ _ _ hv

/-- The plane `x + y + z = 1` misses `0`. -/
theorem plane1_zero_not_mem : ¬ ((0 : Fin 3 → ℝ) ∈ {v : Fin 3 → ℝ | v 0 + v 1 + v 2 = 1}) := by
  intro h
  change (0 : ℝ) + 0 + 0 = 1 at h
  veriphy_certified
    ( have h1 : (1 : ℝ) = 0 := by linear_combination (-1) * h
      exact one_ne_zero h1)
  -- certificate: la-infeasible.sage

/-- Two points of the plane `x + y + z = 1` add to a point with coordinate sum `2`. -/
theorem plane1_add (a b c a' b' c' : ℝ) (h : a + b + c = 1) (h' : a' + b' + c' = 1) :
    (a + a') + (b + b') + (c + c') = 2 := by
  linear_combination h + h'
  -- certificate: ideal_membership.sage

/-- `(1, 0, 0) + (0, 1, 0) = (1, 1, 0)`. -/
theorem axes_sum : !![(1 : ℝ), 0, 0] + !![(0 : ℝ), 1, 0] = !![(1 : ℝ), 1, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num
  -- certificate: matrix_from_goal.sage

/-- `(1, 1, 0)` lies on neither the first nor the second coordinate axis. -/
theorem axes_union_not_closed :
    (![1, 1, 0] : Fin 3 → ℝ) ∉ ({v | v 1 = 0 ∧ v 2 = 0} ∪ {v | v 0 = 0 ∧ v 2 = 0} : Set (Fin 3 → ℝ)) := by
  rintro (⟨h, -⟩ | ⟨h, -⟩)
  · change (1 : ℝ) = 0 at h
    veriphy_certified
      ( have h1 : (1 : ℝ) = 0 := by linear_combination h
        exact one_ne_zero h1)
    -- certificate: la-infeasible.sage
  · change (1 : ℝ) = 0 at h
    veriphy_certified
      ( have h1 : (1 : ℝ) = 0 := by linear_combination h
        exact one_ne_zero h1)
    -- certificate: la-infeasible.sage

/-! ## exr-vs-polynomials-axioms -/

/-- `x² + (-x² + x) = x`: two polynomials of degree exactly `2` add to one of degree `1`. -/
theorem deg_two_sum : ((Polynomial.X ^ 2 : Polynomial ℝ) + (-Polynomial.X ^ 2 + Polynomial.X)) = Polynomial.X := by
  ring

theorem deg_two_sum_natDegree : ((Polynomial.X ^ 2 : Polynomial ℝ) + (-Polynomial.X ^ 2 + Polynomial.X)).natDegree = 1 := by
  have h : ((Polynomial.X ^ 2 : Polynomial ℝ) + (-Polynomial.X ^ 2 + Polynomial.X)) = Polynomial.X := by
    veriphy_certified (linear_combination)
    -- certificate: la-polynomial.sage
  rw [h, Polynomial.natDegree_X]

/-- The zero polynomial does not have degree `2`. -/
theorem zero_not_deg_two : (0 : Polynomial ℝ).degree ≠ 2 := by simp

/-- The zero polynomial has no degree `n` (its degree is `⊥`). -/
theorem zero_degree_ne (n : ℕ) : (0 : Polynomial ℝ).degree ≠ n := by simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne (0) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne (1) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne (2) := by
  simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne (3) := by
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne
-- test: la-zero-degree-ne-test.sage

/-- The polynomials of degree at most `2` are closed under addition and scaling (a `Submodule`). -/
theorem deg_le_two_closed (p q : Polynomial ℝ) (t : ℝ) (hp : p ∈ Polynomial.degreeLT ℝ 3)
    (hq : q ∈ Polynomial.degreeLT ℝ 3) : p + q ∈ Polynomial.degreeLT ℝ 3 ∧ t • p ∈ Polynomial.degreeLT ℝ 3 :=
  ⟨Submodule.add_mem _ hp hq, Submodule.smul_mem _ t hp⟩

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.deg_le_two_closed (0) (0) (0) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) := by
  constructor
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.deg_le_two_closed (Polynomial.X ^ 2 + 1) (3 * Polynomial.X) (2) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) := by
  constructor
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp

example : veriphy_instance% KaluzaKlein.LinearAlgebra.VectorSpaces.deg_le_two_closed (Polynomial.X ^ 2) (-Polynomial.X ^ 2 + Polynomial.X) (-1 / 2) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) ((by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)) := by
  constructor
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp
  · rw [Polynomial.mem_degreeLT]
    refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))
    first | compute_degree! | simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.VectorSpaces.deg_le_two_closed
-- test: la-deg-le-two-closed-test.sage

end KaluzaKlein.LinearAlgebra.VectorSpaces

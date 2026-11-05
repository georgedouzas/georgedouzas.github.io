import Mathlib
import Veriphy.Kit.Linear
import KaluzaKlein.Foundations.Linear
import KaluzaKlein.PartI.LinearAlgebra.VectorSpaceResults
import Veriphy.Kit.Levels

/-!
# Linear algebra, section "Linear maps"

The claims of the section "Linear maps" of `tutorial/part-i-02-linear-algebra.qmd`: linear maps, kernels, images,
rank-nullity and matrices; the matrix `A = (1, 2, 3; 2, 4, 6)` (rank `1`, nullity `2`), rotations, the derivative on
quadratics (rank `2`, kernel the constants) and maps that are not linear.
-/

namespace KaluzaKlein.LinearAlgebra.LinearMaps

open Matrix Module

/-! ## Rank of a matrix with proportional rows (used here and in the dual-space results) -/

/-- A `2 × n` matrix whose first row is non-zero and whose second row is a multiple of the first has rank `1`. -/
theorem rank_of_proportional_rows {K : Type*} [Field K] {n : ℕ} (A : Matrix (Fin 2) (Fin n) K) (h0 : A 0 ≠ 0)
    (c : K) (h : A 1 = c • A 0) : A.rank = 1 := by
  rw [Matrix.rank_eq_finrank_span_row, Matrix.row_def]
  have : Submodule.span K (Set.range fun i => A i) = K ∙ A 0 := by
    apply le_antisymm
    · rw [Submodule.span_le]
      rintro x ⟨i, rfl⟩
      fin_cases i
      · exact Submodule.mem_span_singleton_self _
      · simp only [Fin.mk_one, SetLike.mem_coe]
        rw [h]
        exact Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _)
    · exact Submodule.span_mono (by rintro x rfl; exact ⟨0, rfl⟩)
  rw [this, finrank_span_singleton h0]

/-- Conversely, a `2 × n` matrix of rank `1` with non-zero first row has its second row a multiple of the first. -/
theorem proportional_rows_of_rank {K : Type*} [Field K] {n : ℕ} (A : Matrix (Fin 2) (Fin n) K) (h0 : A 0 ≠ 0)
    (hr : A.rank = 1) : ∃ c : K, A 1 = c • A 0 := by
  rw [Matrix.rank_eq_finrank_span_row, Matrix.row_def] at hr
  have hle : K ∙ A 0 ≤ Submodule.span K (Set.range fun i => A i) :=
    Submodule.span_mono (by rintro x rfl; exact ⟨0, rfl⟩)
  have heq : K ∙ A 0 = Submodule.span K (Set.range fun i => A i) :=
    Submodule.eq_of_le_of_finrank_eq hle (by rw [hr, finrank_span_singleton h0])
  have h1 : A 1 ∈ K ∙ A 0 := heq ▸ Submodule.subset_span ⟨1, rfl⟩
  obtain ⟨c, hc⟩ := Submodule.mem_span_singleton.mp h1
  exact ⟨c, hc.symm⟩

/-! ## def-linear-map -/

/-- Rank-nullity: `dim ker L + dim L(V) = dim V`. -/
theorem rank_nullity {K V W : Type*} [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W] [Module K W]
    [FiniteDimensional K V] (L : V →ₗ[K] W) :
    finrank K (LinearMap.ker L) + finrank K (LinearMap.range L) = finrank K V := by
  rw [add_comm]; exact LinearMap.finrank_range_add_finrank_ker L

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.rank_nullity (ℚ) ((Fin 1 → ℚ)) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0) := by
  have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0) ![1] (by simp)
  have hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0) = ⊤ :=
    LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 0 c, by simp⟩
  rw [hr, finrank_top]
  simp at hk ⊢
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.rank_nullity (ℚ) ((Fin 2 → ℚ)) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) := by
  have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) ![0, 1] (by simp)
  have hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) = ⊤ :=
    LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 1 c, by simp⟩
  rw [hr, finrank_top]
  simp at hk ⊢
  omega

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.rank_nullity (ℚ) ((Fin 3 → ℚ)) (ℚ) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) := by
  have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) ![1, 0, 0] (by simp)
  have hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) = ⊤ :=
    LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 0 c, by simp⟩
  rw [hr, finrank_top]
  simp at hk ⊢
  omega

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearMaps.rank_nullity
-- test: la-rank-nullity-test.sage

/-- In coordinates `L(v)^i = L^i_j v^j`: the map of a matrix is matrix-vector multiplication. -/
theorem matrix_map_apply {m n : ℕ} (L : Matrix (Fin m) (Fin n) ℝ) (v : Fin n → ℝ) (i : Fin m) :
    Matrix.toLin' L v i = ∑ j, L i j * v j := by
  simp [Matrix.mulVec, dotProduct]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.matrix_map_apply (1) (1) (!![0]) (![3]) (0) := by
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.matrix_map_apply (2) (3) (!![1, 2, 3; 4, 5, 6]) (![1, 0, -1]) (1) := by
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.matrix_map_apply (2) (2) (!![2, -1; 0, 0]) (![1 / 2, 1]) (0) := by
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearMaps.matrix_map_apply
-- test: la-matrix-map-apply-test.sage

/-! ## exm-linear-matrix, exr-kernel-image -/

/-- `A (1, 1, -1)ᵀ = 0`. -/
theorem A_mul_kernel : !![(1 : ℚ), 2, 3; 2, 4, 6] * !![(1 : ℚ); 1; -1] = !![(0 : ℚ); 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-kernel-image.sage

/-- The rank of `A` is `1`: its image is the line through `(1, 2)`. -/
theorem rank_A : KaluzaKlein.Linear.A.rank = 1 := by
  unfold KaluzaKlein.Linear.A
  veriphy_certified
    ( apply le_antisymm
      ·
        exact
          Veriphy.Linear.rank_le_of_factor _ !![1; 2] !![1, 2, 3]
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

/-- The nullity of `A` is `2`: its kernel is a plane. -/
theorem nullity_A : finrank ℚ (LinearMap.ker KaluzaKlein.Linear.A.mulVecLin) = 2 := by
  have hs : Submodule.span ℚ (Set.range ![![(-2 : ℚ), 1, 0], ![-3, 0, 1]]) =
      LinearMap.ker (Matrix.mulVecLin !![(1 : ℚ), 2, 3; 2, 4, 6]) := by
    veriphy_certified
      ( apply le_antisymm
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i <;>
            (simp only [SetLike.mem_coe, LinearMap.mem_ker, Matrix.mulVecLin_apply];
              ext j <;> fin_cases j <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> norm_num)
        · intro x hx
          rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx
          have h0 := congrFun hx 0
          have h1 := congrFun hx 1
          simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0 h1
          refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![1 * x 1, (-1 / 3) * x 0 + (-2 / 3) * x 1], ?_⟩
          ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
    -- certificate: la-span-ker.sage
  rw [show KaluzaKlein.Linear.A = !![(1 : ℚ), 2, 3; 2, 4, 6] from rfl, ← hs]
  veriphy_certified
    (exact
        Veriphy.Linear.finrank_span_of_minor _ ![0, 1]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-dim-span.sage

/-- The image of `A` is the line spanned by `(1, 2)`. -/
theorem range_A : LinearMap.range KaluzaKlein.Linear.A.mulVecLin = ℚ ∙ ![1, 2] := by
  rw [Matrix.range_mulVecLin]
  have hc : (KaluzaKlein.Linear.A).col = ![![(1 : ℚ), 2], ![2, 4], ![3, 6]] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  rw [hc, ← Matrix.range_cons_empty (![(1 : ℚ), 2]) ![]]
  veriphy_certified
    ( apply le_antisymm
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![1], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![2], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![3], by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩
      · rw [Submodule.span_le]
        rintro _ ⟨i, rfl⟩
        fin_cases i
        ·
          exact
            (Submodule.mem_span_range_iff_exists_fun ℚ).2
              ⟨![1, 0, 0], by
                ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num⟩)
  -- certificate: la-span-eq.sage

/-- The kernel of `A` is the plane `x + 2y + 3z = 0`. -/
theorem mem_ker_A (x : Fin 3 → ℚ) :
    x ∈ LinearMap.ker KaluzaKlein.Linear.A.mulVecLin ↔ x 0 + 2 * x 1 + 3 * x 2 = 0 := by
  simp only [LinearMap.mem_ker, Matrix.mulVecLin_apply]
  constructor
  · intro h; simpa [KaluzaKlein.Linear.A, Matrix.mulVec, dotProduct, Fin.sum_univ_three] using congrFun h 0
  · intro h
    ext i
    fin_cases i <;> simp [KaluzaKlein.Linear.A, Matrix.mulVec, dotProduct, Fin.sum_univ_three] <;> linarith

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A (0) := by
  simp [KaluzaKlein.Linear.A, LinearMap.mem_ker, funext_iff, Fin.forall_fin_succ, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A (![1, 1, -1]) := by
  simp [KaluzaKlein.Linear.A, LinearMap.mem_ker, funext_iff, Fin.forall_fin_succ, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A (![-3, 0, 1]) := by
  simp [KaluzaKlein.Linear.A, LinearMap.mem_ker, funext_iff, Fin.forall_fin_succ, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  <;> norm_num

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A (![1, 0, 0]) := by
  simp [KaluzaKlein.Linear.A, LinearMap.mem_ker, funext_iff, Fin.forall_fin_succ, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  <;> norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A
-- test: la-mem-ker-a-test.sage

/-! ## exm-linear-rotation, exr-matrix-of-derivative -/

/-- An orthogonal matrix (`Rᵀ R = 1`) has kernel `{0}`. -/
theorem orthogonal_ker_trivial {n : ℕ} (R : Matrix (Fin n) (Fin n) ℝ) (hR : Rᵀ * R = 1) (v : Fin n → ℝ)
    (hv : R *ᵥ v = 0) : v = 0 := by
  have : Rᵀ *ᵥ (R *ᵥ v) = v := by rw [Matrix.mulVec_mulVec, hR, Matrix.one_mulVec]
  rw [← this, hv, Matrix.mulVec_zero]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.orthogonal_ker_trivial (0) (0) ((by ext i j; exact i.elim0)) (![]) ((by ext i; exact i.elim0)) := by
  ext i
  exact i.elim0

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.orthogonal_ker_trivial (2) (!![0, -1; 1, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) (0) ((by simp)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.orthogonal_ker_trivial (3) (!![1, 0, 0; 0, 0, -1; 0, 1, 0]) ((by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])) (0) ((by simp)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearMaps.orthogonal_ker_trivial
-- test: la-orthogonal-ker-test.sage

open Polynomial in
/-- The derivative sends `1 ↦ 0`, `x ↦ 1`, `x² ↦ 2x`: the columns of `D`. -/
theorem derivative_basis :
    derivative (X ^ 0 : ℝ[X]) = 0 ∧ derivative (X : ℝ[X]) = 1 ∧ derivative (X ^ 2 : ℝ[X]) = 2 * X := by
  refine ⟨?_, ?_, ?_⟩
  · veriphy_certified ((norm_num; done))
    -- certificate: la-polynomial.sage
  · veriphy_certified ((norm_num; done))
    -- certificate: la-polynomial.sage
  · veriphy_certified ((simp [Polynomial.derivative_X_pow]; norm_num [map_ofNat]; done))
    -- certificate: la-polynomial.sage

/-- The matrix of the derivative has rank `2` (its rows `(0, 1, 0)`, `(0, 0, 2)` are independent). -/
theorem rank_D : KaluzaKlein.Linear.D.rank = 2 := by
  unfold KaluzaKlein.Linear.D
  veriphy_certified
    ( apply le_antisymm
      · exact Matrix.rank_le_height _
      ·
        exact
          Veriphy.Linear.le_rank_of_minor _ ![0, 1] ![1, 2]
            (by
              simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                  Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
                norm_num))
  -- certificate: la-rank.sage

/-- The kernel of the derivative (the constants) has dimension `3 - 2 = 1`. -/
theorem nullity_D : finrank ℚ (LinearMap.ker KaluzaKlein.Linear.D.mulVecLin) = 1 := by
  have hs : Submodule.span ℚ (Set.range ![![(1 : ℚ), 0, 0]]) =
      LinearMap.ker (Matrix.mulVecLin !![(0 : ℚ), 1, 0; 0, 0, 2]) := by
    veriphy_certified
      ( apply le_antisymm
        · rw [Submodule.span_le]
          rintro _ ⟨i, rfl⟩
          fin_cases i <;>
            (simp only [SetLike.mem_coe, LinearMap.mem_ker, Matrix.mulVecLin_apply];
              ext j <;> fin_cases j <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> norm_num)
        · intro x hx
          rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx
          have h0 := congrFun hx 0
          have h1 := congrFun hx 1
          simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h0 h1
          refine (Submodule.mem_span_range_iff_exists_fun ℚ).2 ⟨![1 * x 0], ?_⟩
          ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> linarith)
    -- certificate: la-span-ker.sage
  rw [show KaluzaKlein.Linear.D = !![(0 : ℚ), 1, 0; 0, 0, 2] from rfl, ← hs]
  veriphy_certified
    (exact
        Veriphy.Linear.finrank_span_of_minor _ ![0]
          (by
            simp +decide [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
                Fin.succAbove_of_le_castSucc, Matrix.diagonal] <;>
              norm_num))
  -- certificate: la-dim-span.sage

/-- `D (1, 0, 0)ᵀ = 0`: the constants are in the kernel. -/
theorem D_mul_const : !![(0 : ℚ), 1, 0; 0, 0, 2] * !![(1 : ℚ); 0; 0] = !![(0 : ℚ); 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]
  -- certificate: la-matrix.sage

/-! ## exm-linear-not, exr-linear-check -/

/-- `x ↦ x + 1` is not linear (`L(0) ≠ 0`). -/
theorem translation_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = x + 1 := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := hf 0
      simp at h)
  -- certificate: la-nonlinear.sage

/-- `x ↦ x²` is not linear (`(2 · 1)² ≠ 2 · 1²`). -/
theorem square_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = x ^ 2 := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (2 : _) (1 : _)
      rw [hf, hf] at h
      norm_num at h)
  -- certificate: la-nonlinear.sage

/-- The circle parametrisation `φ ↦ (cos φ, sin φ)` is not linear (it sends `0` to `(1, 0)`). -/
theorem circle_not_linear : ¬ ∃ f : ℝ →ₗ[ℝ] (Fin 2 → ℝ), ∀ φ, f φ = ![Real.cos φ, Real.sin φ] := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := hf 0
      have hc := congrFun h 0
      simp at hc)
  -- certificate: la-nonlinear.sage

/-- A linear map is determined by its values on a basis. -/
theorem determined_by_basis {ι V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]
    (b : Basis ι ℝ V) (f g : V →ₗ[ℝ] W) (h : ∀ i, f (b i) = g (b i)) : f = g :=
  b.ext h

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.determined_by_basis ((Fin 2)) ((Fin 2 → ℝ)) ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℝ (Fin 2)) (0) (0) ((fun _ => rfl)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.determined_by_basis ((Fin 2)) ((Fin 2 → ℝ)) ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℝ (Fin 2)) (LinearMap.id) (LinearMap.id) ((fun _ => rfl)) := by
  rfl

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.determined_by_basis ((Fin 2)) ((Fin 2 → ℝ)) ((Fin 2 → ℝ)) (inferInstance) (inferInstance) (inferInstance) (inferInstance) (Pi.basisFun ℝ (Fin 2)) (Matrix.toLin' !![2, -1; 0, 0]) (Matrix.toLin' !![2, -1; 0, 0]) ((fun _ => rfl)) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearMaps.determined_by_basis
-- test: la-determined-by-basis-test.sage

/-- `L(x, y) = (x + y, xy)`: `L(2, 2) = (4, 4)` but `2 L(1, 1) = (4, 2)`. -/
theorem product_map_values :
    ![(2 : ℝ) + 2, (2 : ℝ) * 2] = ![4, 4] ∧ (2 : ℝ) • ![(1 : ℝ) + 1, (1 : ℝ) * 1] = ![4, 2] := by
  constructor
  · veriphy_certified
      ((  ext i
          fin_cases i <;>
            simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
              Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
          all_goals ring1))
    -- certificate: la-matrix-named.sage
  · veriphy_certified
      ((  ext i
          fin_cases i <;>
            simp [Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, Matrix.diagonal_apply,
              Matrix.vecMulVec_apply, Matrix.mulVec, Matrix.vecMul, dotProduct, Fin.prod_univ_succ]
          all_goals ring1))
    -- certificate: la-matrix-named.sage

theorem product_map_not_linear : ¬ ∃ f : (Fin 2 → ℝ) →ₗ[ℝ] (Fin 2 → ℝ), ∀ v, f v = ![v 0 + v 1, v 0 * v 1] := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (2 : _) (1 : _)
      rw [hf, hf] at h
      have hc := congrFun h 1
      norm_num at hc)
  -- certificate: la-nonlinear.sage

/-- `(x, y) ↦ (2x - y, 0)` is linear (as the matrix `(2, -1; 0, 0)`). -/
theorem linear_example (v : Fin 2 → ℝ) : Matrix.toLin' !![(2 : ℝ), -1; 0, 0] v = ![2 * v 0 - v 1, 0] := by
  ext i
  fin_cases i <;> simp [Matrix.mulVec, dotProduct] <;> ring

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.linear_example (0) := by
  ext i
  fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.linear_example (![1, 2]) := by
  ext i
  fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.linear_example (![-1 / 2, 3]) := by
  ext i
  fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.LinearAlgebra.LinearMaps.linear_example
-- test: la-linear-example-test.sage

/-- The integral over `[0, 1]` is additive on continuous functions. -/
theorem integral_additive (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g) :
    ∫ x in (0 : ℝ)..1, (f x + g x) = (∫ x in (0 : ℝ)..1, f x) + ∫ x in (0 : ℝ)..1, g x :=
  intervalIntegral.integral_add (hf.intervalIntegrable 0 1) (hg.intervalIntegrable 0 1)

example : veriphy_instance% KaluzaKlein.LinearAlgebra.LinearMaps.integral_additive ((fun _ => 0)) ((fun _ => 0)) (continuous_const) (continuous_const) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.LinearAlgebra.LinearMaps.integral_additive
-- test: la-integral-additive-test.sage

/-- `f ↦ f²` is not linear on functions (it fails homogeneity at the constant function `1`). -/
theorem square_functional_not_additive : ¬ ∃ F : (ℝ → ℝ) →ₗ[ℝ] (ℝ → ℝ), ∀ f, F f = f ^ 2 := by
  veriphy_certified
    ( rintro ⟨f, hf⟩
      have h := f.map_smul (2 : _) (1 : _)
      rw [hf, hf] at h
      have hc := congrFun h 0
      norm_num at hc)
  -- certificate: la-nonlinear.sage

end KaluzaKlein.LinearAlgebra.LinearMaps

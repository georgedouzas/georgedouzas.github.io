import KaluzaKlein.PartI.HilbertSpaces.Basic
import Veriphy.Kit.Levels

/-!
# Hilbert spaces and operator theory, section "Inner products"

The claims of the section "Inner products" of `tutorial/part-i-07-hilbert-spaces-operators.qmd`: the axioms and
consequences of an inner product (antilinearity in the first slot, Cauchy-Schwarz, the triangle inequality), the
inner product of `ℂⁿ` on `(1, i)` and `(i, 1)`, the trace form on the Pauli matrices, the integral inner product on
continuous functions and the orthogonality of the exponentials, and the polarisation and parallelogram identities.
-/

namespace KaluzaKlein.HilbertSpaces.InnerProducts

noncomputable section

open Matrix Basic ComplexConjugate

/-! ## def-inner-product -/

/-- Linearity in the second slot and conjugate symmetry make an inner product antilinear in its first slot:
`⟪α u, v⟫ = conj α ⟪u, v⟫`. -/
theorem inner_smul_left_conj (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (α : ℂ) (u v : E) :
    inner ℂ (α • u) v = conj α * inner ℂ u v := inner_smul_left u v α
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.inner_smul_left_conj
-- test: hs-ip-smul-left-test.sage

/-- The Cauchy-Schwarz inequality `|⟪u, v⟫| ≤ ‖u‖ ‖v‖`. -/
theorem cauchy_schwarz (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    ‖inner ℂ u v‖ ≤ ‖u‖ * ‖v‖ := norm_inner_le_norm u v
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.HilbertSpaces.InnerProducts.cauchy_schwarz
-- test: hs-ip-cauchy-schwarz-test.sage

/-- The triangle inequality `‖u + v‖ ≤ ‖u‖ + ‖v‖` of the norm `‖u‖ = √⟪u, u⟫`. -/
theorem triangle (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    ‖u + v‖ ≤ ‖u‖ + ‖v‖ := norm_add_le u v
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.triangle
-- test: hs-ip-triangle-test.sage

/-- The norm is the square root of `⟪u, u⟫`: `‖u‖² = ⟪u, u⟫`, a real number, positive for `u ≠ 0`. -/
theorem norm_sq_eq_inner_self (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u : E) :
    ‖u‖ ^ 2 = RCLike.re (inner ℂ u u) ∧ (u ≠ 0 → 0 < RCLike.re (inner ℂ u u)) := by
  refine ⟨(inner_self_eq_norm_sq u).symm, fun hu => ?_⟩
  rw [inner_self_eq_norm_sq]
  exact pow_pos (norm_pos_iff.mpr hu) 2
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.norm_sq_eq_inner_self
-- test: hs-ip-norm-sq-test.sage

/-- Conjugate symmetry `⟪v, u⟫ = conj ⟪u, v⟫`. -/
theorem inner_conj_symm_claim (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    inner ℂ v u = conj (inner ℂ u v) := (inner_conj_symm v u).symm
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.inner_conj_symm_claim
-- test: hs-ip-conj-symm-test.sage

/-- On a real vector space the same axioms give a symmetric form, positive on non-zero vectors. -/
theorem real_inner_symm_pos (F : Type) [NormedAddCommGroup F] [InnerProductSpace ℝ F] (u v : F) :
    inner ℝ u v = inner ℝ v u ∧ (u ≠ 0 → 0 < inner ℝ u u) := by
  refine ⟨real_inner_comm v u, fun hu => ?_⟩
  rw [real_inner_self_eq_norm_sq]
  exact pow_pos (norm_pos_iff.mpr hu) 2
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.real_inner_symm_pos
-- test: hs-ip-real-symm-test.sage

/-! ## exm-ip-cn -/

/-- The inner product of `ℂⁿ` is `⟪u, v⟫ = ∑ₖ conj uₖ vₖ`. -/
theorem euclidean_inner (n : ℕ) (u v : EuclideanSpace ℂ (Fin n)) :
    inner ℂ u v = ∑ k, conj (u.ofLp k) * v.ofLp k := by
  simp only [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Pi.star_apply, Complex.star_def]
  exact Finset.sum_congr rfl fun k _ => by ring
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_inner (2) (!₂[(1 : ℂ), Complex.I]) (!₂[(Complex.I : ℂ), 1]) := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_succ, mul_comm]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_inner (3) (!₂[(1 : ℂ), 2, Complex.I]) (!₂[(0 : ℂ), 1, 1]) := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_succ, mul_comm]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_inner (0) (!₂[]) (!₂[]) := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_succ, mul_comm]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_inner
-- test: hs-ip-euclidean-inner-test.sage

/-- `‖u‖² = ∑ₖ |uₖ|²` on `ℂⁿ`. -/
theorem euclidean_norm_sq (n : ℕ) (u : EuclideanSpace ℂ (Fin n)) : ‖u‖ ^ 2 = ∑ k, ‖u.ofLp k‖ ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun k _ => by positivity)]
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_norm_sq (2) (!₂[(1 : ℂ), Complex.I]) := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun k _ => by positivity)]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_norm_sq (3) (!₂[(3 : ℂ), 4, 0]) := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun k _ => by positivity)]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_norm_sq (2) (!₂[(0 : ℂ), 0]) := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun k _ => by positivity)]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.HilbertSpaces.InnerProducts.euclidean_norm_sq
-- test: hs-ip-euclidean-norm-test.sage

/-- `‖(1, i)‖² = conj 1 · 1 + conj i · i = 1 + 1 = 2`, the row of conjugates times the column. -/
theorem norm_sq_one_I : !![(1 : ℂ), -Complex.I] * !![1; Complex.I] = !![2] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3,
            KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.pauliFour,
            KaluzaKlein.HilbertSpaces.Basic.rot, KaluzaKlein.HilbertSpaces.Basic.Afig,
            KaluzaKlein.HilbertSpaces.Basic.Asym, KaluzaKlein.HilbertSpaces.Basic.Atri,
            KaluzaKlein.HilbertSpaces.Basic.Mtri, KaluzaKlein.HilbertSpaces.Basic.Aadj,
            KaluzaKlein.HilbertSpaces.Basic.A3c, KaluzaKlein.HilbertSpaces.Basic.Pdiag,
            KaluzaKlein.HilbertSpaces.Basic.Pplus, KaluzaKlein.HilbertSpaces.Basic.Pminus,
            KaluzaKlein.HilbertSpaces.Basic.J, KaluzaKlein.HilbertSpaces.Basic.P1, KaluzaKlein.HilbertSpaces.Basic.F4,
            KaluzaKlein.HilbertSpaces.Basic.Mex]
        all_goals ring1))
  -- certificate: hs-matrix.sage

/-- Without the conjugation, `∑ uₖ uₖ = 1 + i² = 0` for `u = (1, i)`. -/
theorem unconjugated_one_I : !![(1 : ℂ), Complex.I] * !![1; Complex.I] = !![0] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3,
            KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.pauliFour,
            KaluzaKlein.HilbertSpaces.Basic.rot, KaluzaKlein.HilbertSpaces.Basic.Afig,
            KaluzaKlein.HilbertSpaces.Basic.Asym, KaluzaKlein.HilbertSpaces.Basic.Atri,
            KaluzaKlein.HilbertSpaces.Basic.Mtri, KaluzaKlein.HilbertSpaces.Basic.Aadj,
            KaluzaKlein.HilbertSpaces.Basic.A3c, KaluzaKlein.HilbertSpaces.Basic.Pdiag,
            KaluzaKlein.HilbertSpaces.Basic.Pplus, KaluzaKlein.HilbertSpaces.Basic.Pminus,
            KaluzaKlein.HilbertSpaces.Basic.J, KaluzaKlein.HilbertSpaces.Basic.P1, KaluzaKlein.HilbertSpaces.Basic.F4,
            KaluzaKlein.HilbertSpaces.Basic.Mex]
        all_goals ring1))
  -- certificate: hs-matrix.sage

/-- `⟪(1, i), (i, 1)⟫ = conj 1 · i + conj i · 1 = i - i = 0`: the two vectors are orthogonal. -/
theorem inner_one_I_I_one : !![(1 : ℂ), -Complex.I] * !![Complex.I; 1] = !![0] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ,
            KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3,
            KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.pauliFour,
            KaluzaKlein.HilbertSpaces.Basic.rot, KaluzaKlein.HilbertSpaces.Basic.Afig,
            KaluzaKlein.HilbertSpaces.Basic.Asym, KaluzaKlein.HilbertSpaces.Basic.Atri,
            KaluzaKlein.HilbertSpaces.Basic.Mtri, KaluzaKlein.HilbertSpaces.Basic.Aadj,
            KaluzaKlein.HilbertSpaces.Basic.A3c, KaluzaKlein.HilbertSpaces.Basic.Pdiag,
            KaluzaKlein.HilbertSpaces.Basic.Pplus, KaluzaKlein.HilbertSpaces.Basic.Pminus,
            KaluzaKlein.HilbertSpaces.Basic.J, KaluzaKlein.HilbertSpaces.Basic.P1, KaluzaKlein.HilbertSpaces.Basic.F4,
            KaluzaKlein.HilbertSpaces.Basic.Mex]
        all_goals ring1))
  -- certificate: hs-matrix.sage

/-! ## exm-ip-matrices -/

/-- The trace form has `⟪A, A⟫ = tr (A† A) = ∑ⱼₖ |Aⱼₖ|²`. -/
theorem trace_conjTranspose_mul_self (n : ℕ) (A : Matrix (Fin n) (Fin n) ℂ) :
    Matrix.trace (Aᴴ * A) = ∑ j, ∑ k, ((‖A j k‖ ^ 2 : ℝ) : ℂ) := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.conjTranspose_apply, Complex.star_def]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
  rw [Complex.conj_mul']
  push_cast
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.trace_conjTranspose_mul_self
-- test: hs-ip-trace-form-test.sage

/-- `⟪σₐ, σ_b⟫ = tr (σₐ† σ_b) = 2 δₐ_b`. -/
theorem pauli_trace_inner (a b : Fin 3) :
    Matrix.trace ((pauli a)ᴴ * pauli b) = if a = b then 2 else 0 := by
  fin_cases a <;> fin_cases b <;>
    norm_num [pauli, σ1, σ2, σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (0) (0) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (0) (1) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (0) (2) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (1) (0) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (1) (1) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (1) (2) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (2) (0) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (2) (1) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner (2) (2) := by
  norm_num [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply]

attribute [veriphy_tested kernel_test 9 0] KaluzaKlein.HilbertSpaces.InnerProducts.pauli_trace_inner
-- test: hs-ip-pauli-inner-test.sage

/-- `⟪1, σₐ⟫ = tr σₐ = 0`. -/
theorem pauli_traceless (a : Fin 3) : Matrix.trace (pauli a) = 0 := by
  fin_cases a <;> simp [pauli, σ1, σ2, σ3, Matrix.trace, Fin.sum_univ_succ]
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_traceless (0) := by
  simp [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_traceless (1) := by
  simp [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Fin.sum_univ_succ]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.pauli_traceless (2) := by
  simp [KaluzaKlein.HilbertSpaces.Basic.pauli, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Fin.sum_univ_succ]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.HilbertSpaces.InnerProducts.pauli_traceless
-- test: hs-ip-pauli-traceless-test.sage

/-- Every `2 × 2` matrix is a combination of `1, σ₁, σ₂, σ₃`: the Pauli basis spans; the four summands are
`((a + d)/2) 1`, `((b + c)/2) σ₁`, `(i(b - c)/2) σ₂` and `((a - d)/2) σ₃`, written out entrywise. -/
theorem pauli_expansion (a b c d : ℂ) : !![a, b; c, d] =
    !![(a + d) / 2, 0; 0, (a + d) / 2] + !![0, (b + c) / 2; (b + c) / 2, 0] +
      !![0, (Complex.I * (b - c) / 2) * (-Complex.I); (Complex.I * (b - c) / 2) * Complex.I, 0] +
      !![(a - d) / 2, 0; 0, -((a - d) / 2)] := by
  VERIPHY[hs-matrix]

/-- A family of four `2 × 2` matrices that is orthogonal in the trace form, each of norm `√2`, is linearly
independent; the Pauli basis is one. -/
theorem linearIndependent_of_trace_orthogonal (v : Fin 4 → Matrix (Fin 2) (Fin 2) ℂ)
    (h : ∀ a b, Matrix.trace ((v a)ᴴ * v b) = if a = b then 2 else 0) : LinearIndependent ℂ v := by
  rw [Fintype.linearIndependent_iff]
  intro g hg a
  have := congrArg (fun M => Matrix.trace ((v a)ᴴ * M)) hg
  simp only [Matrix.mul_sum, Matrix.mul_smul, Matrix.trace_sum, Matrix.trace_smul, h, smul_eq_mul, mul_ite,
    mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true, Matrix.mul_zero, Matrix.trace_zero] at this
  simpa using this
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.linearIndependent_of_trace_orthogonal
-- test: hs-ip-pauli-independent-test.sage

/-! ## exm-ip-functions -/

/-- Positive definiteness of `⟪f, g⟫ = ∫₀^{2π} conj f g`: a continuous `f` that is non-zero somewhere inside the
interval has `∫ |f|² > 0`. -/
theorem integral_norm_sq_pos (f : ℝ → ℂ) (hf : Continuous f) (x₀ : ℝ) (hx₀ : x₀ ∈ Set.Ioo 0 (2 * Real.pi))
    (h : f x₀ ≠ 0) : 0 < ∫ x in (0 : ℝ)..(2 * Real.pi), ‖f x‖ ^ 2 := by
  have hcont : Continuous fun x => ‖f x‖ ^ 2 := by fun_prop
  rw [intervalIntegral.integral_pos_iff_support_of_nonneg_ae (Filter.Eventually.of_forall fun x => by positivity)
    (hcont.intervalIntegrable _ _)]
  refine ⟨by positivity, ?_⟩
  have hopen : IsOpen (Function.support (fun x => ‖f x‖ ^ 2) ∩ Set.Ioo 0 (2 * Real.pi)) := by
    refine IsOpen.inter ?_ isOpen_Ioo
    have : Function.support (fun x => ‖f x‖ ^ 2) = {x | f x ≠ 0} := by ext x; simp
    rw [this]; exact isOpen_ne_fun hf continuous_const
  have hne : (Function.support (fun x => ‖f x‖ ^ 2) ∩ Set.Ioo 0 (2 * Real.pi)).Nonempty :=
    ⟨x₀, by simp [h], hx₀⟩
  refine lt_of_lt_of_le (hopen.measure_pos MeasureTheory.volume hne) (MeasureTheory.measure_mono ?_)
  exact Set.inter_subset_inter_right _ Set.Ioo_subset_Ioc_self
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.integral_norm_sq_pos
-- test: hs-ip-positive-definite-test.sage

/-- `⟪e_m, e_n⟫ = ∫₀^{2π} e^{i(n - m)x} dx = 2π δ_mn` for `e_n(x) = e^{inx}`. -/
theorem exp_orthogonality (m n : ℤ) :
    ∫ x in (0 : ℝ)..(2 * Real.pi), conj (Complex.exp (Complex.I * m * x)) * Complex.exp (Complex.I * n * x) =
      if m = n then ((2 * Real.pi : ℝ) : ℂ) else 0 := by
  have h1 : ∀ x : ℝ, conj (Complex.exp (Complex.I * m * x)) * Complex.exp (Complex.I * n * x) =
      Complex.exp ((Complex.I * (n - m)) * x) := by
    intro x
    rw [← Complex.exp_conj, ← Complex.exp_add]
    congr 1
    simp only [map_mul, Complex.conj_I, map_intCast, Complex.conj_ofReal]
    ring
  simp_rw [h1]
  split_ifs with h
  · subst h
    simp
  · rw [integral_exp_mul_complex]
    · have : Complex.exp (Complex.I * (n - m) * (2 * (Real.pi : ℂ))) = 1 := by
        rw [show Complex.I * (n - m) * (2 * (Real.pi : ℂ)) = ((n - m : ℤ) : ℂ) * (2 * Real.pi * Complex.I) by
          push_cast; ring]
        exact Complex.exp_int_mul_two_pi_mul_I _
      push_cast
      simp [this]
    · refine mul_ne_zero Complex.I_ne_zero (sub_ne_zero.mpr ?_)
      exact_mod_cast Ne.symm h
attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.HilbertSpaces.InnerProducts.exp_orthogonality
-- test: hs-ip-exp-orthogonality-test.sage

/-! ## exr-ip-cauchy-schwarz -/

/-- `‖u + v‖² = ‖u‖² + 2 Re ⟪u, v⟫ + ‖v‖²`. -/
theorem norm_add_sq_claim (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    ‖u + v‖ ^ 2 = ‖u‖ ^ 2 + 2 * RCLike.re (inner ℂ u v) + ‖v‖ ^ 2 := norm_add_sq u v
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.norm_add_sq_claim
-- test: hs-ip-norm-add-sq-test.sage

/-- With `t = ⟪v, u⟫ / ‖v‖²`, `0 ≤ ‖u - t v‖² = ‖u‖² - |⟪u, v⟫|² / ‖v‖²`, which is Cauchy-Schwarz. -/
theorem norm_sub_proj_sq (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) (hv : v ≠ 0) :
    ‖u - (inner ℂ v u / ((‖v‖ ^ 2 : ℝ) : ℂ)) • v‖ ^ 2 = ‖u‖ ^ 2 - ‖inner ℂ u v‖ ^ 2 / ‖v‖ ^ 2 := by
  have hv' : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  have h1 : inner ℂ v u = conj (inner ℂ u v) := (inner_conj_symm v u).symm
  rw [norm_sub_sq (𝕜 := ℂ), inner_smul_right, norm_smul, h1]
  have h2 : RCLike.re (conj (inner ℂ u v) / ((‖v‖ ^ 2 : ℝ) : ℂ) * inner ℂ u v) = ‖inner ℂ u v‖ ^ 2 / ‖v‖ ^ 2 := by
    rw [div_mul_eq_mul_div, Complex.conj_mul', ← Complex.ofReal_pow, ← Complex.ofReal_div]
    exact Complex.ofReal_re _
  have h3 : ‖conj (inner ℂ u v) / ((‖v‖ ^ 2 : ℝ) : ℂ)‖ = ‖inner ℂ u v‖ / ‖v‖ ^ 2 := by
    rw [norm_div, Complex.norm_conj, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  rw [h2, h3]
  field_simp
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.HilbertSpaces.InnerProducts.norm_sub_proj_sq
-- test: hs-ip-cs-identity-test.sage

/-! ## exr-ip-polarisation -/

/-- The parallelogram law `‖u + v‖² + ‖u - v‖² = 2‖u‖² + 2‖v‖²`. -/
theorem parallelogram (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    ‖u + v‖ ^ 2 + ‖u - v‖ ^ 2 = 2 * ‖u‖ ^ 2 + 2 * ‖v‖ ^ 2 := by
  rw [parallelogram_law_with_norm ℂ]; ring
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.InnerProducts.parallelogram, hs-ip-parallelogram-test]

/-- The polarisation identity `⟪u, v⟫ = ¼ ∑ₖ i⁻ᵏ ‖u + iᵏ v‖²`, written out: `i⁰, i⁻¹, i⁻², i⁻³ = 1, -i, -1, i`. -/
theorem polarisation (E : Type) [NormedAddCommGroup E] [InnerProductSpace ℂ E] (u v : E) :
    inner ℂ u v = ((‖u + v‖ ^ 2 : ℂ) - Complex.I * ‖u + Complex.I • v‖ ^ 2 - ‖u - v‖ ^ 2 +
      Complex.I * ‖u - Complex.I • v‖ ^ 2) / 4 := by
  have hI : (RCLike.I : ℂ) = Complex.I := rfl
  rw [inner_eq_sum_norm_sq_div_four, hI]
  push_cast
  ring
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.InnerProducts.polarisation, hs-ip-polarisation-test]

/-- `∑ₖ i⁻ᵏ = 0` over a full period of four (here `∑_{k < 4m} iᵏ = 0`, the same sum). -/
theorem sum_I_pow (m : ℕ) : ∑ k ∈ Finset.range (4 * m), Complex.I ^ k = 0 := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [show 4 * (m + 1) = 4 * m + 1 + 1 + 1 + 1 by ring]
    simp only [Finset.sum_range_succ, ih, zero_add]
    have : Complex.I ^ (4 * m) = 1 := by rw [pow_mul, Complex.I_pow_four, one_pow]
    simp only [pow_succ, this, one_mul]
    have h3 : Complex.I * Complex.I * Complex.I = -Complex.I := by rw [Complex.I_mul_I]; ring
    linear_combination Complex.I_mul_I + h3
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_I_pow (0) := by
  norm_num [Finset.sum_range_succ, pow_succ, Complex.I_sq] <;> ring_nf

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_I_pow (1) := by
  norm_num [Finset.sum_range_succ, pow_succ, Complex.I_sq] <;> ring_nf

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_I_pow (2) := by
  norm_num [Finset.sum_range_succ, pow_succ, Complex.I_sq] <;> ring_nf

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.HilbertSpaces.InnerProducts.sum_I_pow
-- test: hs-ip-sum-i-pow-test.sage

/-- `∑ₖ (-1)ᵏ = 0` over a full period of two: `∑_{k < 2m} (-1)ᵏ = 0`. -/
theorem sum_neg_one_pow (m : ℕ) : ∑ k ∈ Finset.range (2 * m), (-1 : ℂ) ^ k = 0 := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [show 2 * (m + 1) = 2 * m + 1 + 1 by ring]
    simp only [Finset.sum_range_succ, ih, zero_add, pow_succ, pow_mul, neg_one_sq, one_pow]
    ring
example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_neg_one_pow (0) := by
  norm_num [Finset.sum_range_succ, pow_succ]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_neg_one_pow (1) := by
  norm_num [Finset.sum_range_succ, pow_succ]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_neg_one_pow (2) := by
  norm_num [Finset.sum_range_succ, pow_succ]

example : veriphy_instance% KaluzaKlein.HilbertSpaces.InnerProducts.sum_neg_one_pow (3) := by
  norm_num [Finset.sum_range_succ, pow_succ]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.HilbertSpaces.InnerProducts.sum_neg_one_pow
-- test: hs-ip-sum-neg-one-pow-test.sage

/-! ## exr-ip-sage-matrices -/

/-- `M = 2 · 1 + 2 σ₁ + 1 σ₂ - 1 σ₃`. -/
theorem Mex_pauli : Mex = (2 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) + (2 : ℂ) • σ1 + (1 : ℂ) • σ2 + (-1 : ℂ) • σ3 := by
  VERIPHY[hs-matrix]

/-- The coefficients `c = ½ ⟪σ, M⟫ = ½ tr (σ M)` are `2, 2, 1, -1` for `1, σ₁, σ₂, σ₃`. -/
theorem Mex_coefficients (a : Fin 4) : Matrix.trace (pauliFour a * Mex) / 2 = ![2, 2, 1, -1] a := by
  fin_cases a <;> simp [pauliFour, σ1, σ2, σ3, Mex, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ] <;>
    linear_combination (-1 : ℂ) * Complex.I_sq
VERIPHY_TEST[KaluzaKlein.HilbertSpaces.InnerProducts.Mex_coefficients, hs-ip-mex-coefficients-test]

end
end KaluzaKlein.HilbertSpaces.InnerProducts

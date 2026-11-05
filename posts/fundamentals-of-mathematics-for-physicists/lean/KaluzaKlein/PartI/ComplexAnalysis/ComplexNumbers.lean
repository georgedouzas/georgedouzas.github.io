import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Complex numbers"

The claims of the section "Complex numbers" of `tutorial/part-i-04-complex-analysis.qmd`: the product rule, the
field structure, conjugation, `ℂ` as `ℝ²` with basis `1, i`, explicit products and quotients, square roots,
complex numbers as `2 × 2` real matrices, and the impossibility of an order on `ℂ`.
-/

namespace KaluzaKlein.ComplexAnalysis.ComplexNumbers

open Complex ComplexConjugate

/-- A complex number with non-zero real part is non-zero. -/
theorem ne_zero_of_re_ne_zero {z : ℂ} (h : z.re ≠ 0) : z ≠ 0 := fun hz => h (by simp [hz])

/-! ## def-complex-number -/

/-- The product rule `(x + iy)(u + iv) = (xu - yv) + i(xv + yu)`, from `i² = -1`. -/
theorem mul_formula (x y u v : ℝ) :
    ((x : ℂ) + y * I) * (u + v * I) = (x * u - y * v) + (x * v + y * u) * I := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination ((↑y) * (↑v)) * hI)
  -- certificate: ideal_membership.sage
/-- `ℂ` is a field: every `z ≠ 0` has an inverse. -/
theorem mul_inv_of_ne_zero (z : ℂ) (hz : z ≠ 0) : ∃ w : ℂ, z * w = 1 := ⟨z⁻¹, mul_inv_cancel₀ hz⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.mul_inv_of_ne_zero (1) (one_ne_zero) := by
  exact ⟨1, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.mul_inv_of_ne_zero (Complex.I) (Complex.I_ne_zero) := by
  exact ⟨-Complex.I, by linear_combination -Complex.I_sq⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.mul_inv_of_ne_zero ((2 + Complex.I)) ((by intro h; have := congrArg Complex.re h; norm_num at this)) := by
  exact ⟨(2 - Complex.I) / 5, by linear_combination -(1 / 5 : ℂ) * Complex.I_sq⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.mul_inv_of_ne_zero
-- test: ca-num-mul-inv-test.sage

/-- Conjugation respects sums and products. -/
theorem conj_add_mul (z w : ℂ) : conj (z + w) = conj z + conj w ∧ conj (z * w) = conj z * conj w :=
  ⟨map_add _ _ _, map_mul _ _ _⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.conj_add_mul (0) (0) := by
  constructor <;> simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.conj_add_mul (1) (Complex.I) := by
  constructor <;> apply Complex.ext <;> simp

attribute [veriphy_tested oracle_test 2 1] KaluzaKlein.ComplexAnalysis.ComplexNumbers.conj_add_mul
-- test: ca-num-conj-test.sage

/-- As a real vector space `ℂ` is `ℝ²` with basis `1, i`: `x + iy` has coordinates `(x, y)`. -/
theorem coords_one_I (x y : ℝ) : ((x : ℂ) + y * I).re = x ∧ ((x : ℂ) + y * I).im = y := by
  simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.coords_one_I (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.coords_one_I (1) (2) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.coords_one_I ((-3 / 2)) ((5 / 7)) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.coords_one_I
-- test: ca-num-coords-test.sage


/-! ## exm-complex-arithmetic-basic -/

/-- `(2 + i)(3 - i) = 7 + i`. -/
theorem prod_example : (2 + I) * (3 - I) = 7 + I := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1) * hI)
  -- certificate: ideal_membership.sage
/-- `1/(2 + i) = (2 - i)/5`: multiply by the conjugate. -/
theorem quot_example : 1 / (2 + I) = (2 - I) / 5 := by
  have h : (2 + I) ≠ 0 := ne_zero_of_re_ne_zero (by simp)
  rw [div_eq_div_iff h (by norm_num)]
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage
/-- `(2 + i)(2 - i) = 5` is real. -/
theorem conj_denominator : (2 + I) * (2 - I) = 5 := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-1) * hI)
  -- certificate: ideal_membership.sage
/-! ## exm-complex-square-roots -/

/-- `(2i)² = -4` and `(-2i)² = -4`. -/
theorem sq_two_I : (2 * I) ^ 2 = -4 ∧ (-(2 * I)) ^ 2 = -4 := by
  have hI : I ^ 2 = -1 := I_sq
  constructor
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (4) * hI)
    -- certificate: ideal_membership.sage
  · rw [← sub_eq_zero]
    veriphy_certified (linear_combination (4) * hI)
    -- certificate: ideal_membership.sage

/-- Every non-zero complex number has exactly two square roots, `w` and `-w ≠ w`. -/
theorem two_square_roots (z : ℂ) (hz : z ≠ 0) :
    ∃ w : ℂ, w ^ 2 = z ∧ w ≠ -w ∧ ∀ u : ℂ, u ^ 2 = z → u = w ∨ u = -w := by
  obtain ⟨w, hw⟩ := IsAlgClosed.exists_pow_nat_eq z (by norm_num : 0 < 2)
  refine ⟨w, hw, ?_, fun u hu => ?_⟩
  · intro h
    have hw0 : w = 0 := by linear_combination h / 2
    exact hz (by rw [← hw, hw0]; ring)
  · have : (u - w) * (u + w) = 0 := by linear_combination hu - hw
    rcases mul_eq_zero.mp this with h | h
    · exact Or.inl (by linear_combination h)
    · exact Or.inr (by linear_combination h)
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.two_square_roots (1) (one_ne_zero) := by
  refine ⟨1, by norm_num, by norm_num, fun u hu => ?_⟩
  have h : (u - 1) * (u + 1) = 0 := by linear_combination hu
  rcases mul_eq_zero.mp h with h | h
  · exact Or.inl (by linear_combination h)
  · exact Or.inr (by linear_combination h)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.two_square_roots ((-4)) ((by norm_num)) := by
  refine ⟨2 * Complex.I, by linear_combination 4 * Complex.I_sq, fun h => by have := congrArg Complex.im h; norm_num at this, fun u hu => ?_⟩
  have h : (u - 2 * Complex.I) * (u + 2 * Complex.I) = 0 := by linear_combination hu - 4 * Complex.I_sq
  rcases mul_eq_zero.mp h with h | h
  · exact Or.inl (by linear_combination h)
  · exact Or.inr (by linear_combination h)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.two_square_roots ((2 * Complex.I)) ((by simp)) := by
  refine ⟨1 + Complex.I, by linear_combination Complex.I_sq, fun h => by have := congrArg Complex.re h; norm_num at this, fun u hu => ?_⟩
  have h : (u - (1 + Complex.I)) * (u + (1 + Complex.I)) = 0 := by linear_combination hu - Complex.I_sq
  rcases mul_eq_zero.mp h with h | h
  · exact Or.inl (by linear_combination h)
  · exact Or.inr (by linear_combination h)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.two_square_roots
-- test: ca-num-sqrt-roots-test.sage

/-- `x² + 1 = (x + i)(x - i)` over `ℂ`. -/
theorem sq_add_one_factor (x : ℂ) : x ^ 2 + 1 = (x + I) * (x - I) := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage
/-- Over `ℝ`, `x² + 1` has no zero (it is positive), so it does not factor into real linear factors. -/
theorem sq_add_one_pos (x : ℝ) : 0 < x ^ 2 + 1 := by
  veriphy_certified (linear_combination sq_nonneg (x))
  -- certificate: sos.sage

/-! ## exm-complex-matrices -/

/-- Sums go to sums. -/
theorem toMat_add (x y u v : ℝ) :
    !![x, -y; y, x] + !![u, -v; v, u] = !![x + u, -(y + v); y + v, x + u] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1)
  -- certificate: la-matrix.sage

/-- Products go to products: the matrix of `(x + iy)(u + iv)`. -/
theorem toMat_mul (x y u v : ℝ) :
    !![x, -y; y, x] * !![u, -v; v, u] = !![x * u - y * v, -(x * v + y * u); x * v + y * u, x * u - y * v] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1)
  -- certificate: la-matrix.sage

/-- `i` becomes the rotation by `90°`, whose square is `-1`. -/
theorem toMat_I_sq : !![(0 : ℝ), -1; 1, 0] * !![(0 : ℝ), -1; 1, 0] = !![(-1 : ℝ), 0; 0, -1] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])
  -- certificate: la-matrix.sage

/-- The determinant is `x² + y² = |z|²`. -/
theorem toMat_det (x y : ℝ) : Matrix.det !![x, -y; y, x] = x ^ 2 + y ^ 2 := by
  veriphy_certified (norm_num [Matrix.det_fin_two_of] <;> ring1)
  -- certificate: la-matrix.sage

/-- The transpose represents `z̄ = x - iy`. -/
theorem toMat_transpose (x y : ℝ) : Matrix.transpose !![x, -y; y, x] = !![x, -(-y); -y, x] := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1)
  -- certificate: la-matrix.sage

/-- `x² + y² = |x + iy|²`. -/
theorem sq_norm_eq (x y : ℝ) : ‖(x : ℂ) + y * I‖ ^ 2 = x ^ 2 + y ^ 2 := by
  rw [norm_add_mul_I, Real.sq_sqrt (by positivity)]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.sq_norm_eq (0) (0) := by
  rw [← Complex.normSq_eq_norm_sq]
  norm_num [Complex.normSq_apply]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.sq_norm_eq (3) (4) := by
  rw [← Complex.normSq_eq_norm_sq]
  norm_num [Complex.normSq_apply]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.sq_norm_eq ((1 / 2)) ((-2)) := by
  rw [← Complex.normSq_eq_norm_sq]
  norm_num [Complex.normSq_apply]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.sq_norm_eq
-- test: ca-num-sq-norm-test.sage

/-! ## exr-complex-arithmetic -/

/-- `(1 + i)/(1 - i) = i`. -/
theorem quot_one_add_I : (1 + I) / (1 - I) = I := by
  have h : (1 - I) ≠ 0 := ne_zero_of_re_ne_zero (by simp)
  rw [div_eq_iff h]
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage
/-- `(1 + i)² = 2i`. -/
theorem one_add_I_sq : (1 + I) ^ 2 = 2 * I := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage
/-- `(1 + i)⁸ = 16`. -/
theorem one_add_I_pow_eight : (1 + I) ^ 8 = 16 := by
  have hI : I ^ 2 = -1 := I_sq
  rw [show (1 + I) ^ 8 = ((1 + I) ^ 2) ^ 4 by ring, one_add_I_sq]
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (16 * I ^ 2 - 16) * hI)
  -- certificate: ideal_membership.sage
/-- `conj((2 + i)(3 - i)) = conj(2 + i) conj(3 - i)`: both are `7 - i`. -/
theorem conj_prod_example : conj (7 + I) = (2 - I) * (3 + I) := by
  simp only [map_add, conj_I, map_ofNat]
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hI)
  -- certificate: ideal_membership.sage
/-! ## exr-complex-conjugate-identities -/

/-- `conj (x + iy) = x - iy`. -/
theorem conj_coords (x y : ℝ) : conj ((x : ℂ) + y * I) = x - y * I := by
  rw [map_add, map_mul, conj_ofReal, conj_ofReal, conj_I]
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-- `Re z = (z + z̄)/2`. -/
theorem re_eq (x y : ℝ) : ((x : ℂ) + y * I + (x - y * I)) / 2 = x := by
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-- `Im z = (z - z̄)/(2i)`. -/
theorem im_eq (x y : ℝ) : ((x : ℂ) + y * I - (x - y * I)) / (2 * I) = y := by
  rw [div_eq_iff (mul_ne_zero two_ne_zero I_ne_zero)]
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-- `z z̄ = x² + y²`. -/
theorem mul_conj_coords (x y : ℝ) : ((x : ℂ) + y * I) * (x - y * I) = x ^ 2 + y ^ 2 := by
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (-(↑y) ^ 2) * hI)
  -- certificate: ideal_membership.sage
/-- `z` is real exactly when `z = z̄`. -/
theorem real_iff_conj (z : ℂ) : conj z = z ↔ z.im = 0 := conj_eq_iff_im
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj (0) := by
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj (3) := by
  simp [Complex.ext_iff]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj ((2 + Complex.I)) := by
  simp [Complex.ext_iff]
  all_goals norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj (Complex.I) := by
  simp [Complex.ext_iff]
  all_goals norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj
-- test: ca-num-real-conj-test.sage

/-! ## exr-complex-field-axioms -/

/-- `z⁻¹ = z̄/(z z̄)` for `z ≠ 0`: `z · (z̄ · (z z̄)⁻¹) = 1`, with `z z̄ = |z|²` real and positive. -/
theorem inv_eq_conj_div (z : ℂ) (hz : z ≠ 0) : z * (conj z * (((normSq z)⁻¹ : ℝ) : ℂ)) = 1 := by
  rw [← mul_assoc, Complex.mul_conj, ← Complex.ofReal_mul, mul_inv_cancel₀ (Complex.normSq_pos.mpr hz).ne',
    Complex.ofReal_one]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.inv_eq_conj_div (1) (one_ne_zero) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexNumbers.inv_eq_conj_div
-- test: ca-num-inv-conj-test.sage

/-- `z z̄ = x² + y² > 0` for `z ≠ 0`. -/
theorem normSq_pos_of_ne_zero (z : ℂ) (hz : z ≠ 0) : 0 < z.re ^ 2 + z.im ^ 2 := by
  have := Complex.normSq_pos.mpr hz
  rw [Complex.normSq_apply] at this
  nlinarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.normSq_pos_of_ne_zero (1) (one_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.normSq_pos_of_ne_zero (Complex.I) (Complex.I_ne_zero) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.normSq_pos_of_ne_zero ((2 + Complex.I)) ((by intro h; have := congrArg Complex.re h; norm_num at this)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.normSq_pos_of_ne_zero
-- test: ca-num-normsq-pos-test.sage

/-- `ℂ` admits no order compatible with its arithmetic: there is no set `P` of "positive" numbers closed under sums
and products, not containing `0`, with every non-zero `a` or its negative in `P` (`i²` and `1²` would both be
positive, and so would their sum `0`). -/
theorem no_positive_cone (P : Set ℂ) :
    ¬ ((∀ a ∈ P, ∀ b ∈ P, a + b ∈ P) ∧ (∀ a ∈ P, ∀ b ∈ P, a * b ∈ P) ∧
      (∀ a : ℂ, a ≠ 0 → a ∈ P ∨ -a ∈ P) ∧ (0 : ℂ) ∉ P) := by
  rintro ⟨hadd, hmul, htri, h0⟩
  have sq_mem : ∀ a : ℂ, a ≠ 0 → a * a ∈ P := fun a ha => by
    rcases htri a ha with h | h
    · exact hmul a h a h
    · simpa using hmul _ h _ h
  have h1 : (1 : ℂ) ∈ P := by simpa using sq_mem 1 one_ne_zero
  have hm1 : (-1 : ℂ) ∈ P := by simpa using sq_mem I I_ne_zero
  exact h0 (by simpa using hadd _ h1 _ hm1)
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.no_positive_cone (∅) := by
  rintro ⟨-, -, htri, -⟩
  simpa using htri 1 one_ne_zero

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.no_positive_cone (Set.univ) := by
  rintro ⟨-, -, -, h0⟩
  exact h0 (Set.mem_univ _)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexNumbers.no_positive_cone ({z : ℂ | 0 < z.re}) := by
  rintro ⟨-, -, htri, -⟩
  rcases htri Complex.I Complex.I_ne_zero with h | h <;> simp at h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexNumbers.no_positive_cone
-- test: ca-num-no-order-test.sage

end KaluzaKlein.ComplexAnalysis.ComplexNumbers

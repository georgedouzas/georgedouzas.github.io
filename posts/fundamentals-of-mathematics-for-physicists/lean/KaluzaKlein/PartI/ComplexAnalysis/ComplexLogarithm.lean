import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The complex logarithm"

The claims of the section "The complex logarithm" of `tutorial/part-i-04-complex-analysis.qmd`: the set of logarithms
`log z = Log z + 2πik`, the principal logarithm `Log z = ln|z| + i Arg z`, principal powers, explicit logarithms, the
failure and the criterion of additivity, `i^i`, cube roots of `-8` and the two values of `z^{1/2}`.
-/

namespace KaluzaKlein.ComplexAnalysis.ComplexLogarithm

open Complex Real

/-! ## def-complex-logarithm -/

/-- The logarithms of `z ≠ 0` (the solutions of `e^w = z`) are `Log z + 2πik`. -/
theorem logarithms_iff (z : ℂ) (hz : z ≠ 0) (w : ℂ) :
    Complex.exp w = z ↔ ∃ k : ℤ, w = Complex.log z + 2 * π * I * k := by
  conv_lhs => rw [← Complex.exp_log hz]
  rw [Complex.exp_eq_exp_iff_exists_int]
  exact exists_congr fun k => by constructor <;> intro h <;> rw [h] <;> ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logarithms_iff (1) (one_ne_zero) (0) := by
  exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logarithms_iff ((-1)) ((by norm_num)) ((Real.pi * Complex.I)) := by
  exact ⟨fun _ => ⟨0, by simp [Complex.log_neg_one]⟩, fun _ => Complex.exp_pi_mul_I⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logarithms_iff (1) (one_ne_zero) ((2 * Real.pi * Complex.I)) := by
  exact ⟨fun _ => ⟨1, by simp⟩, fun _ => Complex.exp_two_pi_mul_I⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logarithms_iff
-- test: ca-num-logs-test.sage

/-- The principal logarithm is `Log z = ln|z| + i Arg z`, with imaginary part in `(-π, π]`. -/
theorem log_eq (z : ℂ) :
    Complex.log z = Real.log ‖z‖ + arg z * I ∧ -π < (Complex.log z).im ∧ (Complex.log z).im ≤ π := by
  refine ⟨?_, neg_pi_lt_log_im z, log_im_le_pi z⟩
  apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq (0) := by
  refine ⟨?_, ?_, ?_⟩
  · apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]
  · exact Complex.neg_pi_lt_log_im _
  · exact Complex.log_im_le_pi _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq (1) := by
  refine ⟨?_, ?_, ?_⟩
  · apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]
  · exact Complex.neg_pi_lt_log_im _
  · exact Complex.log_im_le_pi _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq (Complex.I) := by
  refine ⟨?_, ?_, ?_⟩
  · apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]
  · exact Complex.neg_pi_lt_log_im _
  · exact Complex.log_im_le_pi _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq ((-1)) := by
  refine ⟨?_, ?_, ?_⟩
  · apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]
  · exact Complex.neg_pi_lt_log_im _
  · exact Complex.log_im_le_pi _

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq
-- test: ca-num-log-eq-test.sage

/-- The principal power `z^a = e^{a Log z}` for `z ≠ 0`. -/
theorem cpow_eq (z a : ℂ) (hz : z ≠ 0) : z ^ a = Complex.exp (a * Complex.log z) := by
  rw [Complex.cpow_def_of_ne_zero hz, mul_comm]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.cpow_eq (1) (0) (one_ne_zero) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.cpow_eq
-- test: ca-num-cpow-test.sage

/-- For integer `a = n` all values `e^{n(Log z + 2πik)}` agree, with `zⁿ`. -/
theorem int_power_values (z : ℂ) (hz : z ≠ 0) (n k : ℤ) :
    Complex.exp (n * (Complex.log z + 2 * π * I * k)) = z ^ n := by
  rw [mul_add, Complex.exp_add, show (n : ℂ) * (2 * π * I * k) = ((n * k : ℤ) : ℂ) * (2 * π * I) by push_cast; ring,
    Complex.exp_int_mul_two_pi_mul_I, mul_one, Complex.exp_int_mul, Complex.exp_log hz]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.int_power_values (1) (one_ne_zero) (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.int_power_values
-- test: ca-num-int-power-test.sage

/-- For `a = 1/n` every value `e^{(Log z + 2πik)/n}` is an `n`-th root of `z`. -/
theorem nth_root_values (z : ℂ) (hz : z ≠ 0) (n : ℕ) (hn : n ≠ 0) (k : ℤ) :
    Complex.exp (((1 / n : ℝ) : ℂ) * (Complex.log z + 2 * π * I * k)) ^ n = z := by
  have hn' : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  rw [← Complex.exp_nat_mul, ← mul_assoc, show (n : ℂ) * ((1 / n : ℝ) : ℂ) = 1 by push_cast; field_simp, one_mul,
    Complex.exp_add, show 2 * π * I * (k : ℂ) = k * (2 * π * I) by ring, Complex.exp_int_mul_two_pi_mul_I, mul_one,
    Complex.exp_log hz]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values (1) (one_ne_zero) (1) (one_ne_zero) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values
-- test: ca-num-nth-root-test.sage

/-- The values for `k = 0, …, n - 1` are pairwise distinct, so there are exactly `n` of them. -/
theorem nth_root_values_distinct (z : ℂ) (n : ℕ) (hn : n ≠ 0) (j k : ℕ) (hj : j < n) (hk : k < n)
    (h : Complex.exp (((1 / n : ℝ) : ℂ) * (Complex.log z + 2 * π * I * j)) =
      Complex.exp (((1 / n : ℝ) : ℂ) * (Complex.log z + 2 * π * I * k))) :
    j = k := by
  obtain ⟨m, hm⟩ := Complex.exp_eq_exp_iff_exists_int.mp h
  have hn' : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  have hπ : (2 * π * I : ℂ) ≠ 0 := by simp [Real.pi_ne_zero, I_ne_zero]
  have e : Complex.log z + 2 * π * I * j = (Complex.log z + 2 * π * I * k) + m * n * (2 * π * I) := by
    have hinv : (n : ℂ) * ((1 / n : ℝ) : ℂ) = 1 := by push_cast; field_simp
    have := congrArg (fun x => (n : ℂ) * x) hm
    simp only [mul_add, ← mul_assoc, hinv, one_mul] at this
    linear_combination this
  have h2 : (j : ℂ) = k + m * n := mul_left_cancel₀ hπ (by linear_combination e)
  have h3 : (j : ℤ) = k + m * n := by exact_mod_cast h2
  have hj' : (j : ℤ) < n := by exact_mod_cast hj
  have hk' : (k : ℤ) < n := by exact_mod_cast hk
  have hj0 : (0 : ℤ) ≤ j := Nat.cast_nonneg j
  have hk0 : (0 : ℤ) ≤ k := Nat.cast_nonneg k
  have hm0 : m = 0 := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with h | h
    · nlinarith
    · nlinarith
  rw [hm0] at h3
  simp at h3
  exact_mod_cast h3
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values_distinct (1) (1) (one_ne_zero) (0) (0) (one_pos) (one_pos) (rfl) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values_distinct (Complex.I) (3) ((by norm_num)) (2) (2) ((by norm_num)) ((by norm_num)) (rfl) := by
  rfl

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values_distinct ((-8)) (3) ((by norm_num)) (1) (1) ((by norm_num)) ((by norm_num)) (rfl) := by
  rfl

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values_distinct
-- test: ca-num-nth-root-distinct-test.sage

/-! ## exm-log-values -/

/-- `Log(-1) = iπ` and `Log(i) = iπ/2`. -/
theorem log_neg_one_I : Complex.log (-1) = π * I ∧ Complex.log I = π / 2 * I := by
  refine ⟨?_, ?_⟩
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- `Log(-x) = ln x + iπ` for `x > 0`; at `x = e`, `Log(-e) = 1 + iπ`. -/
theorem log_neg_pos (x : ℝ) (hx : 0 < x) : Complex.log (-(x : ℂ)) = Real.log x + π * I := by
  apply Complex.ext
  · rw [Complex.log_re]; simp [abs_of_pos hx]
  · rw [Complex.log_im]
    have : (-(x : ℂ)) = ((-x : ℝ) : ℂ) := by push_cast; ring
    rw [this, arg_ofReal_of_neg (by linarith)]; simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_neg_pos (1) (one_pos) := by
  simp [Complex.log_neg_one]

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_neg_pos
-- test: ca-num-log-neg-test.sage

/-- The logarithms of `-1` are `iπ(2k + 1)`. -/
theorem logs_neg_one (w : ℂ) : Complex.exp w = -1 ↔ ∃ k : ℤ, w = π * I * (2 * k + 1) := by
  rw [logarithms_iff (-1) (by norm_num), Complex.log_neg_one]
  exact exists_congr fun k => by constructor <;> intro h <;> rw [h] <;> ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one (0) := by
  constructor
  · intro h
    norm_num at h
  · rintro ⟨k, hk⟩
    exfalso
    have := congrArg Complex.im hk
    simp at this
    have h3 : ((2 * k + 1 : ℤ) : ℝ) = 0 := by push_cast; linarith
    have h4 : (2 * k + 1 : ℤ) = 0 := by exact_mod_cast h3
    omega

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one ((Real.pi * Complex.I)) := by
  exact ⟨fun _ => ⟨0, by simp⟩, fun _ => Complex.exp_pi_mul_I⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one ((-(Real.pi * Complex.I))) := by
  constructor
  · intro _
    exact ⟨-1, by push_cast; ring⟩
  · intro _
    rw [Complex.exp_neg, Complex.exp_pi_mul_I]
    norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one ((3 * Real.pi * Complex.I)) := by
  constructor
  · intro _
    exact ⟨1, by push_cast; ring⟩
  · intro _
    rw [show (3 * Real.pi * Complex.I : ℂ) = Real.pi * Complex.I + (1 : ℤ) * (2 * Real.pi * Complex.I) by push_cast; ring, Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, Complex.exp_pi_mul_I]
    ring

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one
-- test: ca-num-logs-neg-one-test.sage

/-! ## exm-log-product -/

/-- `Log(-1) + Log(-1) = 2πi`, but `Log((-1)(-1)) = 0`. -/
theorem log_sum_failure :
    Complex.log (-1) + Complex.log (-1) = 2 * π * I ∧ Complex.log ((-1) * (-1)) = 0 := by
  rw [show ((-1 : ℂ) * (-1)) = 1 by norm_num]
  refine ⟨?_, ?_⟩
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- The set identity: `Log z + Log w` is a logarithm of `zw`. -/
theorem log_add_is_log (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
    Complex.exp (Complex.log z + Complex.log w) = z * w := by
  rw [Complex.exp_add, Complex.exp_log hz, Complex.exp_log hw]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_add_is_log (1) (1) (one_ne_zero) (one_ne_zero) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_add_is_log
-- test: ca-num-log-add-test.sage

/-- The principal values satisfy additivity up to `2πik`. -/
theorem log_mul_mod (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
    ∃ k : ℤ, Complex.log (z * w) = Complex.log z + Complex.log w + 2 * π * I * k := by
  have := (logarithms_iff (z * w) (mul_ne_zero hz hw) (Complex.log z + Complex.log w)).mp
    (log_add_is_log z w hz hw)
  obtain ⟨k, hk⟩ := this
  exact ⟨-k, by rw [hk]; push_cast; ring⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_mod (1) (1) (one_ne_zero) (one_ne_zero) := by
  exact ⟨0, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_mod ((-1)) ((-1)) ((by norm_num)) ((by norm_num)) := by
  exact ⟨-1, by rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.log_one, Complex.log_neg_one]; push_cast; ring⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_mod (Complex.I) (Complex.I) (Complex.I_ne_zero) (Complex.I_ne_zero) := by
  exact ⟨0, by rw [Complex.I_mul_I, Complex.log_neg_one, Complex.log_I]; push_cast; ring⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_mod
-- test: ca-num-log-mul-mod-test.sage

/-! ## exm-log-i-to-the-i -/

/-- All values of `i^i = e^{i log i}` are real: `e^{i(iπ/2 + 2πik)} = e^{-π/2 - 2πk}`. -/
theorem I_pow_I_values (k : ℤ) :
    Complex.exp (I * (π / 2 * I + 2 * π * I * k)) = Real.exp (-(π / 2) - 2 * π * k) := by
  push_cast
  congr 1
  linear_combination (↑π / 2 + 2 * ↑π * (k : ℂ)) * I_sq
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.I_pow_I_values (0) := by
  push_cast
  congr 1
  linear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (0)) * Complex.I_sq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.I_pow_I_values (1) := by
  push_cast
  congr 1
  linear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (1)) * Complex.I_sq

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.I_pow_I_values ((-2)) := by
  push_cast
  congr 1
  linear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (-2)) * Complex.I_sq

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.I_pow_I_values
-- test: ca-num-i-pow-i-test.sage

/-- The principal value `i^i = e^{-π/2}`. -/
theorem I_cpow_I : I ^ I = Real.exp (-(π / 2)) := by
  rw [Complex.cpow_def_of_ne_zero I_ne_zero, Complex.log_I]
  push_cast
  congr 1
  have hI : I ^ 2 = -1 := I_sq
  generalize ((π : ℝ) : ℂ) / 2 = p
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (p) * hI)
  -- certificate: ideal_membership.sage

/-- `e^{-π/2} ≈ 0.2079`: `0.2078 < e^{-π/2} < 0.2079`. -/
theorem exp_neg_pi_div_two_approx : 0.2078 < Real.exp (-(π / 2)) ∧ Real.exp (-(π / 2)) < 0.2079 := by
  have h1 : 0.45593 < Real.exp (-0.7854) := by
    veriphy_certified
      ( have h := Real.exp_bound (x := (-3927 / 5000 : ℝ)) (by norm_num [abs_le]) (n := 9) (by norm_num)
        norm_num [Finset.sum_range_succ, Nat.factorial, abs_le] at h
        linarith [h.1])
    -- certificate: ca-num-exp-bound.sage
  have h2 : Real.exp (-0.785395) < 0.45595 := by
    veriphy_certified
      ( have h := Real.exp_bound (x := (-157079 / 200000 : ℝ)) (by norm_num [abs_le]) (n := 8) (by norm_num)
        norm_num [Finset.sum_range_succ, Nat.factorial, abs_le] at h
        linarith [h.2])
    -- certificate: ca-num-exp-bound.sage
  have e1 : Real.exp (-1.5708) = Real.exp (-0.7854) ^ 2 := by rw [← Real.exp_nat_mul]; norm_num
  have e2 : Real.exp (-1.57079) = Real.exp (-0.785395) ^ 2 := by rw [← Real.exp_nat_mul]; norm_num
  have p1 : 0.2078 < Real.exp (-1.5708) := by rw [e1]; nlinarith [Real.exp_pos (-0.7854)]
  have p2 : Real.exp (-1.57079) < 0.2079 := by rw [e2]; nlinarith [Real.exp_pos (-0.785395)]
  constructor
  · exact lt_of_lt_of_le p1 (Real.exp_le_exp.mpr (by linarith [Real.pi_lt_d4]))
  · exact lt_of_le_of_lt (Real.exp_le_exp.mpr (by linarith [Real.pi_gt_d6])) p2

/-! ## exr-log-values -/

/-- `Log(1 + i) = ½ ln 2 + iπ/4`. -/
theorem log_one_add_I : Complex.log (1 + I) = Real.log 2 / 2 + π / 4 * I := by
  apply Complex.ext
  · have hn : ‖(1 : ℂ) + I‖ = √2 := by
      rw [show (1 : ℂ) + I = ((1 : ℝ) : ℂ) + ((1 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
      veriphy_certified (norm_num)
      -- certificate: ca-num-sqrt.sage
    rw [Complex.log_re, hn, Real.log_sqrt (by norm_num)]
    simp [Complex.log_re]
  · rw [Complex.log_im]
    have h2 : ((√2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
    have hh : ((√2 / 2 : ℝ) : ℂ) * 2 = ((√2 : ℝ) : ℂ) := by push_cast; ring
    have key : (1 : ℂ) + I = ((√2 : ℝ) : ℂ) * (Complex.cos ↑(π / 4) + Complex.sin ↑(π / 4) * I) := by
      rw [← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four]
      generalize ((√2 / 2 : ℝ) : ℂ) = h at hh ⊢
      generalize ((√2 : ℝ) : ℂ) = s at h2 hh ⊢
      rw [← sub_eq_zero]
      veriphy_certified (linear_combination (-1 / 2 * I - 1 / 2) * h2 + (-1 / 2 * I * s - 1 / 2 * s) * hh)
      -- certificate: ideal_membership.sage
    rw [key, arg_mul_cos_add_sin_mul_I (by positivity) ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩]
    simp [Complex.log_im]
/-- The principal value of `(-1)^{1/3}` is `e^{iπ/3} = ½ + i√3/2`, not `-1`. -/
theorem neg_one_cpow_third (n : ℕ) (hn : n ≠ 0) :
    (-1 : ℂ) ^ (((1 / n : ℝ) : ℂ)) = Complex.exp (((π / n : ℝ) : ℂ) * I) := by
  rw [Complex.cpow_def_of_ne_zero (by norm_num), Complex.log_neg_one]
  congr 1
  push_cast
  ring
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.neg_one_cpow_third
-- test: ca-num-neg-one-root-test.sage
/-- The three cube roots of `-8`: `z³ = -8` exactly for `z ∈ {1 + i√3, -2, 1 - i√3}`. -/
theorem cube_roots_neg_eight (z : ℂ) :
    z ^ 3 = -8 ↔ z = 1 + √3 * I ∨ z = -2 ∨ z = 1 - √3 * I := by
  have hI : I ^ 2 = -1 := I_sq
  have h3 : ((√3 : ℝ) : ℂ) ^ 2 = 3 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have hfac : z ^ 3 + 8 = (z - (1 + √3 * I)) * (z + 2) * (z - (1 - √3 * I)) := by
    veriphy_certified (linear_combination (z * (↑√3) ^ 2 + 2 * (↑√3) ^ 2) * hI + (-z - 2) * h3)
    -- certificate: ideal_membership.sage
  constructor
  · intro h
    have h0 : (z - (1 + √3 * I)) * (z + 2) * (z - (1 - √3 * I)) = 0 := by rw [← hfac, h]; ring
    rcases mul_eq_zero.mp h0 with h0 | h3'
    · rcases mul_eq_zero.mp h0 with h1 | h2
      · exact Or.inl (by linear_combination h1)
      · exact Or.inr (Or.inl (by linear_combination h2))
    · exact Or.inr (Or.inr (by linear_combination h3'))
  · intro h
    have : z ^ 3 + 8 = 0 := by rw [hfac]; rcases h with rfl | rfl | rfl <;> ring
    linear_combination this

/-- The principal value of `(-8)^{1/3}` is the first, `1 + i√3`. -/
theorem neg_eight_cpow_third (a : ℝ) (ha : 0 < a) :
    (-((a : ℂ) ^ 3)) ^ (((1 / 3 : ℝ) : ℂ)) = a * Complex.exp (((π / 3 : ℝ) : ℂ) * I) := by
  have hlog : Complex.log (-((a : ℂ) ^ 3)) = Real.log (a ^ 3) + π * I := by
    rw [show (-((a : ℂ) ^ 3)) = -((a ^ 3 : ℝ) : ℂ) by push_cast; ring]
    apply Complex.ext
    · rw [Complex.log_re]; simp [abs_of_pos (pow_pos ha 3)]
    · rw [Complex.log_im, show (-((a ^ 3 : ℝ) : ℂ)) = ((-(a ^ 3) : ℝ) : ℂ) by push_cast; ring,
        arg_ofReal_of_neg (by nlinarith [pow_pos ha 3])]
      simp
  have h0 : (-((a : ℂ) ^ 3)) ≠ 0 := by
    have : (a : ℂ) ≠ 0 := by exact_mod_cast ha.ne'
    exact neg_ne_zero.mpr (pow_ne_zero 3 this)
  rw [Complex.cpow_def_of_ne_zero h0, hlog, Real.log_pow,
    show (((3 : ℕ) * Real.log a : ℝ) + π * I : ℂ) * ((1 / 3 : ℝ) : ℂ) = ((Real.log a : ℝ) : ℂ) + ((π / 3 : ℝ) : ℂ) * I by
      push_cast; ring,
    Complex.exp_add, ← Complex.ofReal_exp, Real.exp_log ha]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.neg_eight_cpow_third
-- test: ca-num-cube-root-neg-test.sage
/-! ## exr-log-product-failure -/

/-- `Log(zw) = Log z + Log w` exactly when `Arg z + Arg w ∈ (-π, π]`. -/
theorem log_mul_iff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
    Complex.log (z * w) = Complex.log z + Complex.log w ↔ arg z + arg w ∈ Set.Ioc (-π) π :=
  log_mul_eq_add_log_iff hz hw
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_iff (1) (1) (one_ne_zero) (one_ne_zero) := by
  simp [Real.pi_pos, Real.pi_pos.le]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_iff ((-1)) ((-1)) ((by norm_num)) ((by norm_num)) := by
  rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.log_one, Complex.log_neg_one, Complex.arg_neg_one]
  constructor
  · intro h
    have := congrArg Complex.im h
    simp at this
    linarith [Real.pi_pos]
  · intro h
    simp at h
    linarith [h.2, Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_iff (1) ((-1)) (one_ne_zero) ((by norm_num)) := by
  simp [Complex.arg_neg_one, Real.pi_pos]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_iff
-- test: ca-num-log-mul-iff-test.sage

/-- On the right half-plane the principal logarithm is additive. -/
theorem log_mul_right_half (z w : ℂ) (hz : 0 < z.re) (hw : 0 < w.re) :
    Complex.log (z * w) = Complex.log z + Complex.log w := by
  have hz0 : z ≠ 0 := fun h => by simp [h] at hz
  have hw0 : w ≠ 0 := fun h => by simp [h] at hw
  rw [log_mul_iff z w hz0 hw0]
  have h1 := abs_arg_lt_pi_div_two_iff.mpr (Or.inl hz)
  have h2 := abs_arg_lt_pi_div_two_iff.mpr (Or.inl hw)
  constructor
  · linarith [neg_abs_le (arg z), neg_abs_le (arg w)]
  · linarith [le_abs_self (arg z), le_abs_self (arg w)]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_right_half (1) (1) ((by norm_num)) ((by norm_num)) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_right_half
-- test: ca-num-log-mul-right-test.sage

/-! ## exr-log-power-branches -/

/-- The values `e^{½(Log z + 2πik)}` of `z^{1/2}` are `±e^{½ Log z}` according to the parity of `k`. -/
theorem sqrt_values (z : ℂ) (k : ℤ) :
    Complex.exp ((Complex.log z + 2 * π * I * k) / 2) = (-1) ^ k * Complex.exp (Complex.log z / 2) := by
  rw [add_div, Complex.exp_add, mul_comm, show 2 * π * I * (k : ℂ) / 2 = k * (π * I) by ring,
    Complex.exp_int_mul, Complex.exp_pi_mul_I]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_values (1) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_values
-- test: ca-num-sqrt-values-test.sage

/-- The principal square root maps the slit plane into the right half-plane. -/
theorem sqrt_right_half (z : ℂ) (hz : z ∈ slitPlane) : 0 < (z ^ (1 / 2 : ℂ)).re := by
  have hz0 : z ≠ 0 := slitPlane_ne_zero hz
  rw [Complex.cpow_def_of_ne_zero hz0, Complex.exp_re]
  apply mul_pos (Real.exp_pos _)
  apply Real.cos_pos_of_mem_Ioo
  have h1 := neg_pi_lt_arg z
  have h2 : arg z < π := by
    rcases (mem_slitPlane_iff.mp hz) with h | h
    · exact arg_lt_pi_iff.mpr (Or.inl h.le)
    · exact arg_lt_pi_iff.mpr (Or.inr h)
  have him : (Complex.log z * (1 / 2)).im = arg z / 2 := by simp [Complex.log_im]; ring
  rw [him]
  constructor <;> linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_right_half (1) ((by simp [Complex.mem_slitPlane_iff])) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_right_half
-- test: ca-num-sqrt-right-test.sage

/-- Near the cut: a square root `w` of `-1 + εi` in the closed upper half-plane is within `|ε|` of `i`. -/
theorem sqrt_near_I (w : ℂ) (ε : ℝ) (hw : w ^ 2 = -1 + ε * I) (him : 0 ≤ w.im) : ‖w - I‖ ≤ |ε| := by
  have hprod : (w - I) * (w + I) = ε * I := by linear_combination hw - I_sq
  have hge : 1 ≤ ‖w + I‖ := by
    have := Complex.im_le_norm (w + I)
    simp at this
    linarith
  have h := congrArg norm hprod
  rw [norm_mul, norm_mul, Complex.norm_I, mul_one, Complex.norm_real, Real.norm_eq_abs] at h
  nlinarith [norm_nonneg (w - I)]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_near_I (Complex.I) (0) ((by simp [Complex.I_sq])) ((by simp)) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_near_I
-- test: ca-num-sqrt-near-i-test.sage

/-- The principal square root is a square root: `(z^{1/2})² = z`. -/
theorem sqrt_sq_eq (z : ℂ) : (z ^ (1 / 2 : ℂ)) ^ 2 = z := by
  rw [one_div]; exact_mod_cast Complex.cpow_ofNat_inv_pow z 2
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_sq_eq (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_sq_eq (1) := by
  simp

attribute [veriphy_tested oracle_test 2 1] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_sq_eq
-- test: ca-num-sqrt-sq-test.sage

/-- The principal square root of a point above the negative axis lies above the real axis, of a point below it
below: across the cut the root jumps from near `i` to near `-i`. -/
theorem sqrt_im_sign (z : ℂ) :
    (0 < z.im → 0 < (z ^ (1 / 2 : ℂ)).im) ∧ (z.im < 0 → (z ^ (1 / 2 : ℂ)).im < 0) := by
  have key : ∀ z : ℂ, z ≠ 0 → (z ^ (1 / 2 : ℂ)).im = Real.exp (Real.log ‖z‖ / 2) * Real.sin (arg z / 2) := by
    intro z hz
    rw [Complex.cpow_def_of_ne_zero hz, Complex.exp_im]
    congr 2 <;> simp [Complex.log_re, Complex.log_im] <;> ring
  constructor
  · intro h
    have hz : z ≠ 0 := fun h0 => by simp [h0] at h
    rw [key z hz]
    apply mul_pos (Real.exp_pos _)
    have h1 : 0 < arg z := by
      rcases (arg_nonneg_iff.mpr h.le).lt_or_eq with h' | h'
      · exact h'
      · exact absurd (arg_eq_zero_iff.mp h'.symm).2 h.ne'
    exact Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith [arg_le_pi z, Real.pi_pos])
  · intro h
    have hz : z ≠ 0 := fun h0 => by simp [h0] at h
    rw [key z hz]
    apply mul_neg_of_pos_of_neg (Real.exp_pos _)
    have h1 : arg z < 0 := arg_neg_iff.mpr h
    exact Real.sin_neg_of_neg_of_neg_pi_lt (by linarith) (by linarith [neg_pi_lt_arg z])
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_im_sign (0) := by
  constructor <;> intro h <;> simp at h

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_im_sign
-- test: ca-num-sqrt-im-test.sage

end KaluzaKlein.ComplexAnalysis.ComplexLogarithm

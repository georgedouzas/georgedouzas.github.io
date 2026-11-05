import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The complex exponential"

The claims of the section "The complex exponential" of `tutorial/part-i-04-complex-analysis.qmd`: the exponential
series and Euler's formula, the functional equation, `|e^{x+iy}| = e^x`, periodicity, the trigonometric and hyperbolic
functions, Euler's identity, the roots of unity, `cos(iy) = cosh y`, explicit values, the addition formulas and the
image of the exponential.
-/

namespace KaluzaKlein.ComplexAnalysis.ComplexExponential

open Complex Real

/-! ## def-complex-exponential -/

/-- `e^z = ∑ zⁿ/n!` for every `z`. -/
theorem exp_series (z : ℂ) : HasSum (fun n : ℕ => z ^ n / n.factorial) (Complex.exp z) := by
  rw [Complex.exp_eq_exp_ℂ]; exact NormedSpace.expSeries_div_hasSum_exp z
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_series
-- test: ca-num-exp-series-test.sage

/-- The exponential series converges absolutely. -/
theorem exp_series_abs (z : ℂ) : Summable (fun n : ℕ => ‖z ^ n / n.factorial‖) := by
  have := Real.summable_pow_div_factorial ‖z‖
  refine this.congr fun n => ?_
  rw [norm_div, norm_pow, Complex.norm_natCast]
attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_series_abs
-- test: ca-num-exp-series-abs-test.sage

/-- Euler's formula `e^{x + iy} = e^x (cos y + i sin y)`. -/
theorem euler_formula (x y : ℝ) :
    Complex.exp (x + y * I) = Real.exp x * (Complex.cos y + Complex.sin y * I) := by
  rw [Complex.exp_add_mul_I, Complex.ofReal_exp]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.euler_formula (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.euler_formula
-- test: ca-num-euler-test.sage

/-- The functional equation `e^{z + w} = e^z e^w`. -/
theorem exp_add_claim (z w : ℂ) : Complex.exp (z + w) = Complex.exp z * Complex.exp w := Complex.exp_add z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_add_claim (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_add_claim
-- test: ca-num-exp-add-test.sage

/-- The exponential never vanishes. -/
theorem exp_ne_zero_claim (z : ℂ) : Complex.exp z ≠ 0 := Complex.exp_ne_zero z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_ne_zero_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_ne_zero_claim
-- test: ca-num-exp-ne-zero-test.sage

/-- `|e^{x + iy}| = e^x`. -/
theorem norm_exp_claim (x y : ℝ) : ‖Complex.exp (x + y * I)‖ = Real.exp x := by
  rw [Complex.norm_exp]; simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.norm_exp_claim (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.norm_exp_claim
-- test: ca-num-norm-exp-test.sage

/-- `y` is an argument of `e^{x + iy}`: `y = Arg e^{x+iy} + 2πk`. -/
theorem arg_exp_mem (x y : ℝ) : ∃ k : ℤ, y = arg (Complex.exp (x + y * I)) + 2 * π * k := by
  set E := Complex.exp (x + y * I) with hE
  have key : Complex.exp (arg E * I) = Complex.exp (y * I) := by
    have h1 := norm_mul_exp_arg_mul_I E
    have hEn : ‖E‖ = Real.exp x := norm_exp_claim x y
    have hE2 : E = ((Real.exp x : ℝ) : ℂ) * Complex.exp (y * I) := by
      rw [hE, Complex.exp_add, Complex.ofReal_exp]
    rw [hEn] at h1
    conv_rhs at h1 => rw [hE2]
    exact mul_left_cancel₀ (a := ((Real.exp x : ℝ) : ℂ)) (by exact_mod_cast (Real.exp_pos x).ne') h1
  obtain ⟨n, hn⟩ := Complex.exp_eq_exp_iff_exists_int.mp key.symm
  refine ⟨n, ?_⟩
  have := congrArg Complex.im hn
  simp at this
  linarith
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.arg_exp_mem (0) (0) := by
  exact ⟨0, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.arg_exp_mem (0) ((Real.pi / 2)) := by
  refine ⟨0, ?_⟩
  push_cast
  simp [Complex.exp_pi_div_two_mul_I]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.arg_exp_mem (0) (Real.pi) := by
  refine ⟨0, ?_⟩
  push_cast
  simp [Complex.exp_pi_mul_I]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexExponential.arg_exp_mem
-- test: ca-num-arg-exp-test.sage

/-- The exponential is periodic with period `2πi`. -/
theorem exp_periodic_claim (z : ℂ) : Complex.exp (z + 2 * π * I) = Complex.exp z := Complex.exp_periodic z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_periodic_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_periodic_claim
-- test: ca-num-exp-periodic-test.sage

/-- The polar form `z = r e^{iθ}` with `r = |z|`, `θ = Arg z`. -/
theorem polar_exp (z : ℂ) : (‖z‖ : ℂ) * Complex.exp (arg z * I) = z := norm_mul_exp_arg_mul_I z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.polar_exp (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.polar_exp (1) := by
  simp

attribute [veriphy_tested oracle_test 2 1] KaluzaKlein.ComplexAnalysis.ComplexExponential.polar_exp
-- test: ca-num-polar-exp-test.sage

/-- `cos z = (e^{iz} + e^{-iz})/2`. -/
theorem cos_def_claim (z : ℂ) : Complex.cos z = (Complex.exp (z * I) + Complex.exp (-z * I)) / 2 := by
  rw [← Complex.two_cos]; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_def_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_def_claim
-- test: ca-num-cos-def-test.sage

/-- `sin z = (e^{iz} - e^{-iz})/(2i)`, that is `2i sin z = e^{iz} - e^{-iz}`. -/
theorem sin_def_claim (z : ℂ) : 2 * I * Complex.sin z = Complex.exp (z * I) - Complex.exp (-z * I) := by
  have h := Complex.two_sin z
  have hI : I ^ 2 = -1 := I_sq
  linear_combination I * h + (Complex.exp (-z * I) - Complex.exp (z * I)) * hI
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.sin_def_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.sin_def_claim
-- test: ca-num-sin-def-test.sage

/-- `cosh z = (e^z + e^{-z})/2`. -/
theorem cosh_def_claim (z : ℂ) : Complex.cosh z = (Complex.exp z + Complex.exp (-z)) / 2 := rfl
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cosh_def_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.cosh_def_claim
-- test: ca-num-cosh-def-test.sage

/-- `sinh z = (e^z - e^{-z})/2`. -/
theorem sinh_def_claim (z : ℂ) : Complex.sinh z = (Complex.exp z - Complex.exp (-z)) / 2 := rfl
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.sinh_def_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 3] KaluzaKlein.ComplexAnalysis.ComplexExponential.sinh_def_claim
-- test: ca-num-sinh-def-test.sage

/-! ## exm-exp-euler-identity -/

/-- Euler's identity `e^{iπ} = -1`, and `e^{iπ/2} = i`, `e^{2πi} = 1`. -/
theorem euler_identity :
    Complex.exp (π * I) = -1 ∧ Complex.exp (π / 2 * I) = I ∧ Complex.exp (2 * π * I) = 1 := by
  refine ⟨?_, ?_, ?_⟩
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
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- `cos π = -1` and `sin π = 0`, so `cos π + i sin π = -1`. -/
theorem cos_sin_pi : Real.cos π = -1 ∧ Real.sin π = 0 := by
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

/-! ## exm-exp-roots-of-unity -/

/-- `ω_k = e^{2πik/n}` satisfies `ω_kⁿ = 1`. -/
theorem root_of_unity_pow (n k : ℕ) (hn : n ≠ 0) : Complex.exp (((2 * π * k / n : ℝ) : ℂ) * I) ^ n = 1 := by
  have hn' : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  rw [← Complex.exp_nat_mul, Complex.exp_eq_one_iff]
  refine ⟨k, ?_⟩
  push_cast
  field_simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_pow (1) (0) (one_ne_zero) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_pow
-- test: ca-num-root-pow-test.sage

/-- The solutions of `zⁿ = 1` are the `ω_k`, `k = 0, …, n - 1`. -/
theorem roots_of_unity_iff (n : ℕ) (hn : n ≠ 0) (z : ℂ) :
    z ^ n = 1 ↔ ∃ k < n, Complex.exp (2 * π * I * (k / n)) = z := by
  haveI : NeZero n := ⟨hn⟩
  constructor
  · intro h
    have hz0 : z ≠ 0 := by rintro rfl; simp [zero_pow hn] at h
    have hmem : Units.mk0 z hz0 ∈ rootsOfUnity n ℂ := by
      rw [_root_.mem_rootsOfUnity, Units.ext_iff, Units.val_pow_eq_pow_val, Units.val_mk0, Units.val_one]
      exact h
    simpa using (Complex.mem_rootsOfUnity n (Units.mk0 z hz0)).mp hmem
  · rintro ⟨k, -, rfl⟩
    rw [← Complex.exp_nat_mul, Complex.exp_eq_one_iff]
    exact ⟨k, by field_simp; norm_cast⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.roots_of_unity_iff (1) (one_ne_zero) (1) := by
  constructor
  · intro _
    exact ⟨0, by norm_num, by simp⟩
  · intro _
    simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.roots_of_unity_iff (2) (two_ne_zero) ((-1)) := by
  constructor
  · intro _
    refine ⟨1, by norm_num, ?_⟩
    push_cast
    rw [show 2 * (Real.pi : ℂ) * Complex.I * (1 / 2) = Real.pi * Complex.I by ring]
    exact Complex.exp_pi_mul_I
  · intro _
    norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.roots_of_unity_iff (3) ((by norm_num)) (1) := by
  constructor
  · intro _
    exact ⟨0, by norm_num, by simp⟩
  · intro _
    simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexExponential.roots_of_unity_iff
-- test: ca-num-roots-iff-test.sage

/-- The roots of unity lie on the unit circle. -/
theorem root_of_unity_norm (n k : ℕ) : ‖Complex.exp (2 * π * I * k / n)‖ = 1 := by
  rw [Complex.norm_exp]
  simp
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_norm (0) (0) := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_norm (6) (1) := by
  rw [Complex.norm_exp]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_norm (4) (3) := by
  rw [Complex.norm_exp]
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_norm
-- test: ca-num-root-norm-one-test.sage

/-- They are the powers of one of them: `ω₁ᵏ = ω_k`. -/
theorem root_of_unity_powers (n k : ℕ) :
    Complex.exp (((2 * π / n : ℝ) : ℂ) * I) ^ k = Complex.exp (((2 * π * k / n : ℝ) : ℂ) * I) := by
  rw [← Complex.exp_nat_mul]; congr 1; push_cast; ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_powers (1) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_powers
-- test: ca-num-root-powers-test.sage

/-! ## exm-exp-trig-complex -/

/-- `cos(iy) = cosh y` and `sin(iy) = i sinh y`. -/
theorem cos_sin_imaginary (y : ℂ) :
    Complex.cos (y * I) = Complex.cosh y ∧ Complex.sin (y * I) = Complex.sinh y * I :=
  ⟨Complex.cos_mul_I y, Complex.sin_mul_I y⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_sin_imaginary (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_sin_imaginary
-- test: ca-num-cos-imag-test.sage

/-- `cosh y = (e^{-y} + e^y)/2` on the real line. -/
theorem cosh_real (y : ℝ) : Real.cosh y = (Real.exp (-y) + Real.exp y) / 2 := by
  have h := Real.cosh_eq y
  generalize Real.cosh y = c at h ⊢
  generalize Real.exp y = a at h ⊢
  generalize Real.exp (-y) = b at h ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination h)
  -- certificate: ideal_membership.sage

/-- `|cos z|` is not bounded by `1`: on the imaginary axis it exceeds any bound. -/
theorem cos_unbounded (M : ℝ) : ∃ y : ℝ, M < ‖Complex.cos (y * I)‖ := by
  obtain ⟨y, hy⟩ : ∃ y : ℝ, M < Real.cosh y := by
    refine ⟨2 * |M| + 2, ?_⟩
    have := Real.add_one_le_exp (2 * |M| + 2)
    rw [Real.cosh_eq]
    have := Real.exp_pos (-(2 * |M| + 2))
    have := le_abs_self M
    linarith
  refine ⟨y, ?_⟩
  rw [Complex.cos_mul_I, ← Complex.ofReal_cosh, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos (Real.cosh_pos y)]
  exact hy
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_unbounded (0) := by
  exact ⟨0, by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_unbounded ((-5)) := by
  exact ⟨0, by norm_num⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_unbounded (1) := by
  refine ⟨1, ?_⟩
  rw [Complex.cos_mul_I, ← Complex.ofReal_cosh, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos (Real.cosh_pos 1)]
  exact Real.one_lt_cosh.mpr one_ne_zero

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_unbounded
-- test: ca-num-cos-unbounded-test.sage

/-! ## exr-exp-values -/

/-- `e^{1 + iπ} = -e`. -/
theorem exp_one_add_pi_I : Complex.exp (1 + π * I) = -Complex.exp 1 := by
  veriphy_certified
    ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
        Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
        Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
      all_goals ring)
  -- certificate: ca-num-rewrite.sage

/-- `e^{iπ/2} = i`. -/
theorem exp_pi_div_two_I : Complex.exp (π / 2 * I) = I := by
  veriphy_certified
    ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
        Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
        Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
      all_goals ring)
  -- certificate: ca-num-rewrite.sage

/-- `e^{ln 2 + iπ/3} = 1 + i√3`. -/
theorem exp_log_two_add : Complex.exp (Real.log 2 + π / 3 * I) = 1 + √3 * I := by
  rw [show (Real.log 2 : ℂ) + π / 3 * I = ↑(Real.log 2) + ((π / 3 : ℝ) : ℂ) * I by push_cast; ring,
    Complex.exp_add, ← Complex.ofReal_exp, Real.exp_log (by norm_num), Complex.exp_mul_I,
    ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_three, Real.sin_pi_div_three]
  have h1 : ((1 / 2 : ℝ) : ℂ) * 2 = 1 := by push_cast; ring
  have h3 : ((√3 / 2 : ℝ) : ℂ) * 2 = ((√3 : ℝ) : ℂ) := by push_cast; ring
  have hc : ((2 : ℝ) : ℂ) = 2 := by push_cast; rfl
  generalize ((1 / 2 : ℝ) : ℂ) = a at h1 ⊢
  generalize ((√3 / 2 : ℝ) : ℂ) = b at h3 ⊢
  generalize ((√3 : ℝ) : ℂ) = t at h3 ⊢
  generalize ((2 : ℝ) : ℂ) = c at hc ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (1 / 2 * c) * h1 + (1 / 2 * c * I) * h3 + (1 / 2 * I * t + 1 / 2) * hc)
  -- certificate: ideal_membership.sage
/-- `e^{3πi/2} = -i`. -/
theorem exp_three_pi_div_two : Complex.exp (3 * π / 2 * I) = -I := by
  rw [show (3 * π / 2 * I : ℂ) = π * I + π / 2 * I by ring]
  veriphy_certified
    ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
        Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
        Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
      all_goals ring)
  -- certificate: ca-num-rewrite.sage

/-- `e^{z + 2πi} = e^z` at `z = 1 + i`. -/
theorem exp_periodic_example : Complex.exp (1 + I + 2 * π * I) = Complex.exp (1 + I) := by
  veriphy_certified
    ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
        Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
        Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
      all_goals ring)
  -- certificate: ca-num-rewrite.sage

/-! ## exr-exp-trig -/

/-- `cos(z + w) = cos z cos w - sin z sin w` for complex `z, w`. -/
theorem cos_add_claim (z w : ℂ) : Complex.cos (z + w) = Complex.cos z * Complex.cos w - Complex.sin z * Complex.sin w :=
  Complex.cos_add z w
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_add_claim (0) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_add_claim
-- test: ca-num-cos-add-test.sage

/-- `cos² z + sin² z = 1` for complex `z`. -/
theorem cos_sq_add_sin_sq_claim (z : ℂ) : Complex.cos z ^ 2 + Complex.sin z ^ 2 = 1 := Complex.cos_sq_add_sin_sq z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_sq_add_sin_sq_claim (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_sq_add_sin_sq_claim
-- test: ca-num-pythagoras-test.sage

/-- The computation of the solution, with `a = e^{iz}`, `b = e^{iw}`:
`¼(a + a⁻¹)(b + b⁻¹) + ¼(a - a⁻¹)(b - b⁻¹) = ½(ab + (ab)⁻¹)`. -/
theorem addition_core (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
    (a + a⁻¹) * (b + b⁻¹) / 4 + (a - a⁻¹) * (b - b⁻¹) / 4 = (a * b + (a * b)⁻¹) / 2 := by
  field_simp
  veriphy_certified (ring1)
  -- certificate: ca-num-ring.sage

/-! ## exr-exp-surjective -/

/-- `e^z e^{-z} = 1`, so `e^z ≠ 0`. -/
theorem exp_mul_exp_neg (z : ℂ) : Complex.exp z * Complex.exp (-z) = 1 := by
  rw [← Complex.exp_add, add_neg_cancel, Complex.exp_zero]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_mul_exp_neg (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_mul_exp_neg
-- test: ca-num-exp-inv-test.sage

/-- Every `w ≠ 0` is `e^z` for `z = ln|w| + iθ`, `θ = Arg w`, and for every `z + 2πik`. -/
theorem exp_surjective_claim (w : ℂ) (hw : w ≠ 0) (k : ℤ) :
    Complex.exp (Real.log ‖w‖ + arg w * I + 2 * π * I * k) = w := by
  rw [Complex.exp_add, mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I, mul_one,
    Complex.exp_add, ← Complex.ofReal_exp, Real.exp_log (norm_pos_iff.mpr hw), norm_mul_exp_arg_mul_I]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_surjective_claim (1) (one_ne_zero) (0) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_surjective_claim
-- test: ca-num-exp-onto-test.sage

/-- The solutions of `e^z = -1` are `z = iπ + 2πik`. -/
theorem exp_eq_neg_one_iff (z : ℂ) : Complex.exp z = -1 ↔ ∃ k : ℤ, z = π * I + 2 * π * I * k := by
  rw [← Complex.exp_pi_mul_I, Complex.exp_eq_exp_iff_exists_int]
  exact exists_congr fun k => by constructor <;> intro h <;> rw [h] <;> ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_neg_one_iff (0) := by
  constructor
  · intro h
    norm_num at h
  · rintro ⟨k, hk⟩
    exfalso
    have := congrArg Complex.im hk
    simp at this
    have h2 : (1 + 2 * (k : ℝ)) * Real.pi = 0 := by linarith
    rcases mul_eq_zero.mp h2 with h | h
    · have h3 : (1 + 2 * k : ℤ) = 0 := by exact_mod_cast h
      omega
    · exact Real.pi_ne_zero h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_neg_one_iff ((Real.pi * Complex.I)) := by
  exact ⟨fun _ => ⟨0, by simp⟩, fun _ => Complex.exp_pi_mul_I⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_neg_one_iff ((3 * Real.pi * Complex.I)) := by
  constructor
  · intro _
    exact ⟨1, by push_cast; ring⟩
  · intro _
    rw [show (3 : ℂ) * Real.pi * Complex.I = Real.pi * Complex.I + 2 * Real.pi * Complex.I by ring,
      Complex.exp_add, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I]
    ring

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_neg_one_iff
-- test: ca-num-exp-neg-one-test.sage

/-- The solutions of `e^z = 1 + i` are `z = ½ ln 2 + i(π/4 + 2πk)`. -/
theorem exp_eq_one_add_I_iff (z : ℂ) :
    Complex.exp z = 1 + I ↔ ∃ k : ℤ, z = Real.log 2 / 2 + (π / 4 + 2 * π * k) * I := by
  have h0 : Complex.exp (Real.log 2 / 2 + π / 4 * I) = 1 + I := by
    rw [Complex.exp_add, ← Complex.ofReal_ofNat, ← Complex.ofReal_div, ← Complex.ofReal_exp,
      Complex.exp_mul_I, ← Complex.ofReal_ofNat 4, ← Complex.ofReal_div, ← Complex.ofReal_cos,
      ← Complex.ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four,
      show Real.log 2 / 2 = Real.log (√2) by rw [Real.log_sqrt (by norm_num)],
      Real.exp_log (by positivity)]
    have h2 : ((√2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
    have hh : ((√2 / 2 : ℝ) : ℂ) * 2 = ((√2 : ℝ) : ℂ) := by push_cast; ring
    generalize ((√2 / 2 : ℝ) : ℂ) = h at hh ⊢
    generalize ((√2 : ℝ) : ℂ) = s at h2 hh ⊢
    rw [← sub_eq_zero]
    veriphy_certified (linear_combination (1 / 2 * I + 1 / 2) * h2 + (1 / 2 * s * I + 1 / 2 * s) * hh)
    -- certificate: ideal_membership.sage
  rw [← h0, Complex.exp_eq_exp_iff_exists_int]
  exact exists_congr fun k => by constructor <;> intro h <;> rw [h] <;> push_cast <;> ring
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_one_add_I_iff (0) := by
  constructor
  · intro h
    have := congrArg Complex.im h
    simp at this
  · rintro ⟨k, hk⟩
    exfalso
    have := congrArg Complex.im hk
    simp [Complex.log_im, Complex.ofNat_arg] at this
    have hπ : (Real.pi : ℝ) * (1 + 8 * k) = 0 := by first | linear_combination (-4) * this | linear_combination 4 * this
    rcases mul_eq_zero.mp hπ with h | h
    · exact Real.pi_ne_zero h
    · have h8 : ((1 + 8 * k : ℤ) : ℝ) = 0 := by push_cast; linarith
      have h9 : (1 + 8 * k : ℤ) = 0 := by exact_mod_cast h8
      omega

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_one_add_I_iff (1) := by
  constructor
  · intro h
    have := congrArg Complex.im h
    simp [Complex.exp_im] at this
  · rintro ⟨k, hk⟩
    exfalso
    have := congrArg Complex.im hk
    simp [Complex.log_im, Complex.ofNat_arg] at this
    have hπ : (Real.pi : ℝ) * (1 + 8 * k) = 0 := by first | linear_combination (-4) * this | linear_combination 4 * this
    rcases mul_eq_zero.mp hπ with h | h
    · exact Real.pi_ne_zero h
    · have h8 : ((1 + 8 * k : ℤ) : ℝ) = 0 := by push_cast; linarith
      have h9 : (1 + 8 * k : ℤ) = 0 := by exact_mod_cast h8
      omega

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_one_add_I_iff (2) := by
  constructor
  · intro h
    have := congrArg Complex.im h
    simp [Complex.exp_im] at this
  · rintro ⟨k, hk⟩
    exfalso
    have := congrArg Complex.im hk
    simp [Complex.log_im, Complex.ofNat_arg] at this
    have hπ : (Real.pi : ℝ) * (1 + 8 * k) = 0 := by first | linear_combination (-4) * this | linear_combination 4 * this
    rcases mul_eq_zero.mp hπ with h | h
    · exact Real.pi_ne_zero h
    · have h8 : ((1 + 8 * k : ℤ) : ℝ) = 0 := by push_cast; linarith
      have h9 : (1 + 8 * k : ℤ) = 0 := by exact_mod_cast h8
      omega

attribute [veriphy_tested oracle_test 3 1] KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_one_add_I_iff
-- test: ca-num-exp-one-add-I-test.sage

end KaluzaKlein.ComplexAnalysis.ComplexExponential

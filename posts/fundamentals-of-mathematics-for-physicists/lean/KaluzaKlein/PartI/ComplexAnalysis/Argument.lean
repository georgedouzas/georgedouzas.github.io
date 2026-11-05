import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "Argument"

The claims of the section "Argument" of `tutorial/part-i-04-complex-analysis.qmd`: the polar form, the set of
arguments modulo `2π`, the principal argument in `(-π, π]`, additivity modulo `2π` and its failure for principal
values, De Moivre's formula, explicit arguments and the roots of `z⁴ = -16`.
-/

namespace KaluzaKlein.ComplexAnalysis.Argument

open Complex Real Filter Topology

/-! ## def-argument -/

/-- The polar form `z = |z|(cos Arg z + i sin Arg z)`. -/
theorem polar_form (z : ℂ) : (‖z‖ : ℂ) * (Complex.cos (arg z) + Complex.sin (arg z) * I) = z :=
  norm_mul_cos_add_sin_mul_I z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.polar_form (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.polar_form (1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.polar_form (Complex.I) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.polar_form ((-1)) := by
  simp

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.ComplexAnalysis.Argument.polar_form
-- test: ca-num-polar-test.sage

/-- The arguments of `z ≠ 0` are exactly the `Arg z + 2πk`. -/
theorem arguments_iff (z : ℂ) (hz : z ≠ 0) (θ : ℝ) :
    (‖z‖ : ℂ) * (Complex.cos θ + Complex.sin θ * I) = z ↔ ∃ k : ℤ, θ = arg z + 2 * π * k := by
  have hn : (‖z‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr hz
  have hz' := norm_mul_exp_arg_mul_I z
  rw [← Complex.exp_mul_I]
  constructor
  · intro h
    have h2 : Complex.exp (θ * I) = Complex.exp (arg z * I) := by
      apply mul_left_cancel₀ hn; rw [h, hz']
    obtain ⟨n, hn2⟩ := Complex.exp_eq_exp_iff_exists_int.mp h2
    refine ⟨n, ?_⟩
    have := congrArg Complex.im hn2
    simp at this
    linarith
  · rintro ⟨k, hk⟩
    rw [hk]
    push_cast
    rw [show ((arg z : ℂ) + 2 * π * k) * I = arg z * I + k * (2 * π * I) by ring, Complex.exp_add,
      Complex.exp_int_mul_two_pi_mul_I, mul_one, hz']
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arguments_iff (1) (one_ne_zero) (0) := by
  exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arguments_iff (1) (one_ne_zero) ((2 * Real.pi)) := by
  exact ⟨fun _ => ⟨1, by simp⟩, fun _ => by push_cast; simp⟩

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arguments_iff (Complex.I) (Complex.I_ne_zero) ((Real.pi / 2)) := by
  exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by push_cast; simp⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arguments_iff
-- test: ca-num-arguments-test.sage

/-- The principal argument lies in `(-π, π]`. -/
theorem arg_mem_Ioc_claim (z : ℂ) : -π < arg z ∧ arg z ≤ π := ⟨neg_pi_lt_arg z, arg_le_pi z⟩
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mem_Ioc_claim (0) := by
  simp only [Complex.arg_zero]
  constructor <;> linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mem_Ioc_claim (Complex.I) := by
  simp
  constructor <;> linarith [Real.pi_pos]

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mem_Ioc_claim ((-1)) := by
  simp [Real.pi_pos]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arg_mem_Ioc_claim
-- test: ca-num-arg-range-test.sage

/-- Multiplication adds arguments modulo `2π`. -/
theorem arg_mul_mod (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
    (arg (z * w) : Real.Angle) = arg z + arg w := arg_mul_coe_angle hz hw
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mul_mod (1) (1) (one_ne_zero) (one_ne_zero) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mul_mod (Complex.I) (Complex.I) (Complex.I_ne_zero) (Complex.I_ne_zero) := by
  rw [Complex.I_mul_I, Complex.arg_neg_one, Complex.arg_I, ← Real.Angle.coe_add]
  congr 1
  ring

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_mul_mod ((-1)) ((-1)) ((by norm_num)) ((by norm_num)) := by
  rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.arg_one, Complex.arg_neg_one, ← Real.Angle.coe_add,
    show Real.pi + Real.pi = 2 * Real.pi by ring, Real.Angle.coe_two_pi]
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arg_mul_mod
-- test: ca-num-arg-mul-test.sage

/-! ## exm-argument-values -/

/-- `Arg 1 = 0`, `Arg i = π/2`, `Arg(-1) = π`, `Arg(-i) = -π/2`. -/
theorem arg_values : arg 1 = 0 ∧ arg I = π / 2 ∧ arg (-1) = π ∧ arg (-I) = -(π / 2) := by
  refine ⟨?_, ?_, ?_, ?_⟩
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
  · veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- On the right half-plane `Arg z = arctan(Im z / Re z)`; for instance `Arg(2 + i) = arctan(1/2)`. -/
theorem arg_eq_arctan (z : ℂ) (h : 0 < z.re) : arg z = Real.arctan (z.im / z.re) := by
  have habs : |arg z| < π / 2 := abs_arg_lt_pi_div_two_iff.mpr (Or.inl h)
  rw [← tan_arg, Real.arctan_tan (by linarith [neg_abs_le (arg z)]) (by linarith [le_abs_self (arg z)])]
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_eq_arctan (1) ((by norm_num)) := by
  simp

attribute [veriphy_tested oracle_test 1 2] KaluzaKlein.ComplexAnalysis.Argument.arg_eq_arctan
-- test: ca-num-arg-arctan-test.sage

/-- Just below the negative real axis the principal argument tends to `-π`. -/
theorem arg_tendsto_below (z : ℂ) (hre : z.re < 0) (him : z.im = 0) :
    Tendsto arg (𝓝[{w : ℂ | w.im < 0}] z) (𝓝 (-π)) :=
  tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_below ((-1)) ((by norm_num)) ((by simp)) := by
  exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_below ((-4)) ((by norm_num)) ((by simp)) := by
  exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_below ((-1 / 2)) ((by norm_num)) ((by simp)) := by
  exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_below
-- test: ca-num-arg-below-test.sage

/-- Just above (and on) the negative real axis the principal argument tends to `π`. -/
theorem arg_tendsto_above (z : ℂ) (hre : z.re < 0) (him : z.im = 0) :
    Tendsto arg (𝓝[{w : ℂ | 0 ≤ w.im}] z) (𝓝 π) := by
  have h := continuousWithinAt_arg_of_re_neg_of_im_zero hre him
  rwa [ContinuousWithinAt, arg_eq_pi_iff.mpr ⟨hre, him⟩] at h
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_above ((-1)) ((by norm_num)) ((by simp)) := by
  have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-1 : ℂ)) (by norm_num) (by simp)
  rwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_above ((-4)) ((by norm_num)) ((by simp)) := by
  have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-4 : ℂ)) (by norm_num) (by simp)
  rwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_above ((-1 / 2)) ((by norm_num)) ((by simp)) := by
  have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-1 / 2 : ℂ)) (by norm_num) (by simp)
  rwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_above
-- test: ca-num-arg-above-test.sage

/-! ## exm-argument-de-moivre -/

/-- De Moivre's formula `(cos θ + i sin θ)ⁿ = cos nθ + i sin nθ`. -/
theorem de_moivre (n : ℕ) (θ : ℝ) :
    (Complex.cos θ + Complex.sin θ * I) ^ n = Complex.cos (n * θ) + Complex.sin (n * θ) * I :=
  cos_add_sin_mul_I_pow n θ
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.de_moivre (0) (0) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.de_moivre (3) (0) := by
  simp

attribute [veriphy_tested oracle_test 2 2] KaluzaKlein.ComplexAnalysis.Argument.de_moivre
-- test: ca-num-de-moivre-test.sage

/-- The case `n = 3`, real part: `cos 3θ = cos³θ - 3 cos θ sin²θ`. -/
theorem cos_three (θ : ℝ) : Real.cos (3 * θ) = Real.cos θ ^ 3 - 3 * Real.cos θ * Real.sin θ ^ 2 := by
  have hc := Real.cos_three_mul θ
  have hp := Real.sin_sq_add_cos_sq θ
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hc + (3 * (Real.cos θ)) * hp)
  -- certificate: ideal_membership.sage

/-- The case `n = 3`, imaginary part: `sin 3θ = 3 cos²θ sin θ - sin³θ`. -/
theorem sin_three (θ : ℝ) : Real.sin (3 * θ) = 3 * Real.cos θ ^ 2 * Real.sin θ - Real.sin θ ^ 3 := by
  have hs := Real.sin_three_mul θ
  have hp := Real.sin_sq_add_cos_sq θ
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hs + (-3 * (Real.sin θ)) * hp)
  -- certificate: ideal_membership.sage

/-! ## exm-argument-product-failure -/

/-- `Arg(-1) + Arg(-1) = 2π`, while `Arg((-1)(-1)) = Arg 1 = 0`: the two differ by `2π ≡ 0` (the set-valued
statement holds, the single-valued one is off by `2π`). -/
theorem arg_sum_neg_one : arg (-1) + arg (-1) = 2 * π ∧ arg ((-1) * (-1)) = 0 ∧
    arg (-1) + arg (-1) - arg ((-1) * (-1)) = 2 * π := by
  rw [show ((-1 : ℂ) * (-1)) = 1 by norm_num]
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

/-! ## exr-argument-values -/

/-- `|-1 - i| = √2` and `Arg(-1 - i) = -3π/4`. -/
theorem arg_neg_one_neg_I : ‖(-1 : ℂ) - I‖ = √2 ∧ arg (-1 - I) = -(3 * π / 4) := by
  constructor
  · rw [show (-1 : ℂ) - I = ((-1 : ℝ) : ℂ) + ((-1 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified (norm_num)
    -- certificate: ca-num-sqrt.sage
  · have hc : Real.cos (-(3 * π / 4)) = -(√2 / 2) := by
      rw [Real.cos_neg, show 3 * π / 4 = π - π / 4 by ring, Real.cos_pi_sub, Real.cos_pi_div_four]
    have hs : Real.sin (-(3 * π / 4)) = -(√2 / 2) := by
      rw [Real.sin_neg, show 3 * π / 4 = π - π / 4 by ring, Real.sin_pi_sub, Real.sin_pi_div_four]
    have h2 : ((√2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
    have hh : ((√2 / 2 : ℝ) : ℂ) * 2 = ((√2 : ℝ) : ℂ) := by push_cast; ring
    have key : (-1 : ℂ) - I =
        ((√2 : ℝ) : ℂ) * (Complex.cos ↑(-(3 * π / 4)) + Complex.sin ↑(-(3 * π / 4)) * I) := by
      rw [← Complex.ofReal_cos, ← Complex.ofReal_sin, hc, hs, Complex.ofReal_neg]
      clear hc hs
      generalize ((√2 / 2 : ℝ) : ℂ) = h at hh ⊢
      generalize ((√2 : ℝ) : ℂ) = s at h2 hh ⊢
      rw [← sub_eq_zero]
      veriphy_certified (linear_combination (1 / 2 * I + 1 / 2) * h2 + (1 / 2 * I * s + 1 / 2 * s) * hh)
      -- certificate: ideal_membership.sage
    rw [key, arg_mul_cos_add_sin_mul_I (by positivity) ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩]

/-- `|1 - i√3| = 2` and `Arg(1 - i√3) = -π/3`. -/
theorem arg_one_sub_sqrt_three : ‖(1 : ℂ) - √3 * I‖ = 2 ∧ arg (1 - √3 * I) = -(π / 3) := by
  constructor
  · rw [show (1 : ℂ) - √3 * I = ((1 : ℝ) : ℂ) + ((-√3 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I,
      neg_sq, Real.sq_sqrt (by norm_num)]
    veriphy_certified
      ( conv_rhs => rw [show (2 : ℝ) = √((2) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
        norm_num)
    -- certificate: ca-num-sqrt.sage
  · have h1 : ((1 / 2 : ℝ) : ℂ) * 2 = 1 := by push_cast; ring
    have h3 : ((√3 / 2 : ℝ) : ℂ) * 2 = ((√3 : ℝ) : ℂ) := by push_cast; ring
    have hc : ((2 : ℝ) : ℂ) = 2 := by push_cast; rfl
    have key : (1 : ℂ) - √3 * I = ((2 : ℝ) : ℂ) * (Complex.cos ↑(-(π / 3)) + Complex.sin ↑(-(π / 3)) * I) := by
      rw [← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_neg, Real.sin_neg, Real.cos_pi_div_three,
        Real.sin_pi_div_three, Complex.ofReal_neg]
      generalize ((1 / 2 : ℝ) : ℂ) = a at h1 ⊢
      generalize ((√3 / 2 : ℝ) : ℂ) = b at h3 ⊢
      generalize ((√3 : ℝ) : ℂ) = t at h3 ⊢
      generalize ((2 : ℝ) : ℂ) = c at hc ⊢
      rw [← sub_eq_zero]
      veriphy_certified (linear_combination (-1 / 2 * c) * h1 + (I) * h3 + (I * b - 1 / 2) * hc)
      -- certificate: ideal_membership.sage
    rw [key, arg_mul_cos_add_sin_mul_I (by norm_num) ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩]

/-- `|-2i| = 2` and `Arg(-2i) = -π/2`. -/
theorem arg_neg_two_I : ‖-(2 * I)‖ = 2 ∧ arg (-(2 * I)) = -(π / 2) := by
  constructor
  · rw [show -(2 * I) = ((0 : ℝ) : ℂ) + ((-2 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified
      ( conv_rhs => rw [show (2 : ℝ) = √((2) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
        norm_num)
    -- certificate: ca-num-sqrt.sage
  · rw [show -(2 * I) = ((2 : ℝ) : ℂ) * (-I) by push_cast; ring, arg_real_mul _ (by norm_num)]
    veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-- `(1 + i)¹⁰ = 32i`, of modulus `32` and argument `π/2`. -/
theorem one_add_I_pow_ten : (1 + I) ^ 10 = 32 * I ∧ ‖32 * I‖ = 32 ∧ arg (32 * I) = π / 2 := by
  refine ⟨?_, ?_, ?_⟩
  · have hI : I ^ 2 = -1 := I_sq
    rw [← sub_eq_zero]
    veriphy_certified
      (linear_combination
          (I ^ 8 + 10 * I ^ 7 + 44 * I ^ 6 + 110 * I ^ 5 + 166 * I ^ 4 + 142 * I ^ 3 + 44 * I ^ 2 - 22 * I + 1) * hI)
    -- certificate: ideal_membership.sage
  · rw [show (32 : ℂ) * I = ((0 : ℝ) : ℂ) + ((32 : ℝ) : ℂ) * I by push_cast; ring, norm_add_mul_I]
    veriphy_certified
      ( conv_rhs => rw [show (32 : ℝ) = √((32) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
        norm_num)
    -- certificate: ca-num-sqrt.sage
  · rw [show (32 : ℂ) * I = ((32 : ℝ) : ℂ) * I by push_cast; ring, arg_real_mul _ (by norm_num)]
    veriphy_certified
      ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
          Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
          Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
        all_goals ring)
    -- certificate: ca-num-rewrite.sage

/-! ## exr-argument-multiplicative -/

/-- The addition formula: `r(cos θ + i sin θ) s(cos φ + i sin φ) = rs(cos(θ + φ) + i sin(θ + φ))`. -/
theorem polar_mul (r s θ φ : ℝ) :
    ((r : ℂ) * (Complex.cos θ + Complex.sin θ * I)) * ((s : ℂ) * (Complex.cos φ + Complex.sin φ * I)) =
      (r * s : ℂ) * (Complex.cos (θ + φ) + Complex.sin (θ + φ) * I) := by
  rw [Complex.cos_add, Complex.sin_add]
  have hI : I ^ 2 = -1 := I_sq
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination ((↑r) * (Complex.sin ↑θ) * (↑s) * (Complex.sin ↑φ)) * hI)
  -- certificate: ideal_membership.sage

/-- `arg(1/z) = -arg z` modulo `2π`. -/
theorem arg_inv_mod (z : ℂ) : (arg z⁻¹ : Real.Angle) = -arg z := arg_inv_coe_angle z
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_inv_mod (1) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_inv_mod (Complex.I) := by
  rw [Complex.inv_I, Complex.arg_neg_I, Complex.arg_I]
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.arg_inv_mod ((-1)) := by
  rw [inv_neg, inv_one, Complex.arg_neg_one]
  exact Real.Angle.neg_coe_pi.symm

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.arg_inv_mod
-- test: ca-num-arg-inv-test.sage

/-- A failure for principal values: `Arg((-1)/(-i)) = Arg(-i) = -π/2`, but `Arg(-1) - Arg(-i) = 3π/2`. -/
theorem arg_div_failure : arg ((-1) / (-I)) = -(π / 2) ∧ arg (-1) - arg (-I) = 3 * π / 2 := by
  constructor
  · rw [neg_div_neg_eq, one_div, Complex.inv_I]
    veriphy_certified
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

/-- The first example of the solution: `Arg((-1)/i) = Arg(i) = π/2 = Arg(-1) - Arg(i)`. -/
theorem arg_div_example : arg ((-1) / I) = π / 2 ∧ arg (-1) - arg I = π / 2 := by
  constructor
  · rw [neg_div, one_div, Complex.inv_I, neg_neg]
    veriphy_certified
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

/-! ## exr-argument-roots-polynomial -/

/-- `-16 = 16 e^{iπ}`. -/
theorem neg_sixteen_polar : (-16 : ℂ) = 16 * Complex.exp (π * I) := by
  veriphy_certified
    ( simp only [Complex.arg_one, Complex.arg_I, Complex.arg_neg_one, Complex.arg_neg_I, Complex.log_neg_one,
        Complex.log_I, Complex.log_neg_I, Complex.log_one, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I,
        Complex.exp_pi_div_two_mul_I, Complex.exp_zero, Complex.exp_add, Real.cos_pi, Real.sin_pi, neg_neg]
      all_goals ring)
  -- certificate: ca-num-rewrite.sage

/-- The four solutions of `z⁴ = -16` are `√2(±1 ± i)`. -/
theorem roots_neg_sixteen (z : ℂ) :
    z ^ 4 = -16 ↔ z = √2 * (1 + I) ∨ z = √2 * (-1 + I) ∨ z = √2 * (-1 - I) ∨ z = √2 * (1 - I) := by
  have hI : I ^ 2 = -1 := I_sq
  have h2 : ((√2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
  have hfac : z ^ 4 + 16 = (z - √2 * (1 + I)) * (z - √2 * (-1 + I)) * (z - √2 * (-1 - I)) *
      (z - √2 * (1 - I)) := by
    veriphy_certified
      (linear_combination (2 * z ^ 2 * (↑√2) ^ 2 - (↑√2) ^ 4 * I ^ 2 + 3 * (↑√2) ^ 4) * hI + (-4 * (↑√2) ^ 2 - 8) * h2)
    -- certificate: ideal_membership.sage
  constructor
  · intro h
    have h0 : (z - √2 * (1 + I)) * (z - √2 * (-1 + I)) * (z - √2 * (-1 - I)) * (z - √2 * (1 - I)) = 0 := by
      rw [← hfac, h]; ring
    rcases mul_eq_zero.mp h0 with h0 | h4
    · rcases mul_eq_zero.mp h0 with h0 | h3
      · rcases mul_eq_zero.mp h0 with h1 | h2'
        · exact Or.inl (by linear_combination h1)
        · exact Or.inr (Or.inl (by linear_combination h2'))
      · exact Or.inr (Or.inr (Or.inl (by linear_combination h3)))
    · exact Or.inr (Or.inr (Or.inr (by linear_combination h4)))
  · intro h
    have : z ^ 4 + 16 = 0 := by
      rw [hfac]; rcases h with rfl | rfl | rfl | rfl <;> ring
    linear_combination this

/-- A number whose `n`-th power has modulus `rⁿ` has modulus `r`: the four roots of `z⁴ = -16` (`|z⁴| = 16 = 2⁴`)
lie on the circle of radius `2`. -/
theorem roots_neg_sixteen_norm (z : ℂ) (n : ℕ) (r : ℝ) (hn : n ≠ 0) (hr : 0 ≤ r) (h : ‖z ^ n‖ = r ^ n) :
    ‖z‖ = r := by
  rw [norm_pow] at h
  exact (pow_left_inj₀ (norm_nonneg z) hr hn).mp h
example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.roots_neg_sixteen_norm (1) (1) (1) (one_ne_zero) (zero_le_one) ((by simp)) := by
  simp

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.roots_neg_sixteen_norm (2) (4) (2) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.Argument.roots_neg_sixteen_norm (Complex.I) (2) (1) ((by norm_num)) (zero_le_one) ((by simp)) := by
  simp

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.Argument.roots_neg_sixteen_norm
-- test: ca-num-root-norm-test.sage

end KaluzaKlein.ComplexAnalysis.Argument

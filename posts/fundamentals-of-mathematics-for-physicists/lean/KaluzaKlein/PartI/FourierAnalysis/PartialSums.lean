import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.FourierCoefficients

/-! # Fourier analysis: partial sums (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.PartialSums

open Real Filter Topology

/-- The Dirichlet kernel in real form, `D_N(t) = 1 + 2 Σ_{n=1}^N cos n t = Σ_{|n| ≤ N} e^{int}`. -/
noncomputable def dirichlet (N : ℕ) (t : ℝ) : ℝ := 1 + 2 * ∑ n ∈ Finset.range N, Real.cos ((n + 1) * t)

/-- The partial sum `S_N` of the square wave, `Σ_{n ≤ N} b_n sin n x` with `b_n = 2 (1 - (-1)^n) / (π n)`. -/
noncomputable def squarePartial (N : ℕ) (x : ℝ) : ℝ :=
  ∑ n ∈ Finset.range N, 2 * (1 - (-1) ^ (n + 1)) / (π * (n + 1)) * Real.sin ((n + 1) * x)

/-! ## def-fourier-partial-sum, exr-fs-dirichlet-closed-form -/

/-- The closed form of the Dirichlet kernel: `D_N(t) sin (t/2) = sin ((N + 1/2) t)`. -/
theorem dirichlet_mul_sin (N : ℕ) (t : ℝ) : dirichlet N t * Real.sin (t / 2) = Real.sin ((N + 1 / 2) * t) := by
  induction N with
  | zero =>
    simp only [dirichlet, Finset.range_zero, Finset.sum_empty, mul_zero, add_zero, one_mul, Nat.cast_zero,
      zero_add]
    congr 1; ring
  | succ N ih =>
    simp only [dirichlet] at ih ⊢
    rw [Finset.sum_range_succ, mul_add 2, ← add_assoc, add_mul, ih]
    have h1 : Real.sin ((((N + 1 : ℕ) : ℝ) + 1 / 2) * t) =
        Real.sin ((N + 1) * t) * Real.cos (t / 2) + Real.cos ((N + 1) * t) * Real.sin (t / 2) := by
      rw [← Real.sin_add]; congr 1; push_cast; ring
    have h2 : Real.sin (((N : ℝ) + 1 / 2) * t) =
        Real.sin ((N + 1) * t) * Real.cos (t / 2) - Real.cos ((N + 1) * t) * Real.sin (t / 2) := by
      rw [← Real.sin_sub]; congr 1; ring
    rw [h1, h2]; ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_mul_sin
-- test: fa-dirichlet-mul-sin-test.sage

/-- `D_N(0) = 2N + 1`. -/
theorem dirichlet_zero (N : ℕ) : dirichlet N 0 = 2 * N + 1 := by
  simp [dirichlet]; ring

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_zero (0) := by
  norm_num [KaluzaKlein.FourierAnalysis.PartialSums.dirichlet]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_zero (4) := by
  norm_num [KaluzaKlein.FourierAnalysis.PartialSums.dirichlet]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_zero (1) := by
  norm_num [KaluzaKlein.FourierAnalysis.PartialSums.dirichlet]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_zero
-- test: fa-dirichlet-zero-test.sage

/-- `D_N` is even. -/
theorem dirichlet_even (N : ℕ) (t : ℝ) : dirichlet N (-t) = dirichlet N t := by
  simp [dirichlet, mul_neg, Real.cos_neg]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_even
-- test: fa-dirichlet-even-test.sage

theorem integral_cos_nat_succ (n : ℕ) : ∫ t in (-π)..π, Real.cos ((n + 1) * t) = 0 := by
  have hc : ((n : ℝ) + 1) ≠ 0 := by positivity
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) hc, integral_cos]
  have : Real.sin (((n : ℝ) + 1) * π) = 0 := by exact_mod_cast Real.sin_nat_mul_pi (n + 1)
  rw [mul_neg, Real.sin_neg, this]; simp

/-- `(1/2π) ∫_{-π}^{π} D_N = 1`. -/
theorem dirichlet_integral (N : ℕ) : ∫ t in (-π)..π, dirichlet N t = 2 * π := by
  simp only [dirichlet]
  rw [intervalIntegral.integral_add intervalIntegrable_const (by apply Continuous.intervalIntegrable; fun_prop),
    intervalIntegral.integral_const_mul,
    intervalIntegral.integral_finset_sum (fun n _ => by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [integral_cos_nat_succ, Finset.sum_const_zero, mul_zero, add_zero, intervalIntegral.integral_const,
    smul_eq_mul, mul_one]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_integral
-- test: fa-dirichlet-integral-test.sage

/-- The complex form `Σ_{|n| ≤ N} e^{int}` equals the real form `D_N(t)`. -/
theorem dirichlet_complex (N : ℕ) (t : ℝ) :
    ∑ n ∈ Finset.Icc (-(N : ℤ)) N, Complex.exp (n * t * Complex.I) = (dirichlet N t : ℂ) := by
  induction N with
  | zero => simp [dirichlet]
  | succ N ih =>
    have hs : Finset.Icc (-((N + 1 : ℕ) : ℤ)) ((N + 1 : ℕ) : ℤ) =
        insert (-((N : ℤ) + 1)) (insert ((N : ℤ) + 1) (Finset.Icc (-(N : ℤ)) N)) := by
      ext k; simp only [Finset.mem_Icc, Finset.mem_insert]; push_cast; omega
    rw [hs, Finset.sum_insert (by simp; omega), Finset.sum_insert (by simp), ih]
    simp only [dirichlet, Finset.sum_range_succ]
    have e1 : Complex.exp (((-((N : ℤ) + 1) : ℤ) : ℂ) * t * Complex.I) =
        (Real.cos ((N + 1) * t) : ℂ) - (Real.sin ((N + 1) * t) : ℂ) * Complex.I := by
      rw [show (((-((N : ℤ) + 1) : ℤ) : ℂ) * t * Complex.I) = (((-((N + 1) * t)) : ℝ) : ℂ) * Complex.I by
        push_cast; ring, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_neg, Real.sin_neg]
      push_cast; ring
    have e2 : Complex.exp ((((N : ℤ) + 1 : ℤ) : ℂ) * t * Complex.I) =
        (Real.cos ((N + 1) * t) : ℂ) + (Real.sin ((N + 1) * t) : ℂ) * Complex.I := by
      rw [show ((((N : ℤ) + 1 : ℤ) : ℂ) * t * Complex.I) = ((((N + 1) * t)) : ℝ) * Complex.I by
        push_cast; ring, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
    rw [e1, e2]; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_complex
-- test: fa-dirichlet-complex-test.sage

/-! ## exm-fs-partial-square -/

/-- `S_5 = (4/π) (sin x + sin 3x / 3 + sin 5x / 5)`. -/
theorem square_partial_five (x : ℝ) :
    squarePartial 5 x = 4 / π * (Real.sin x + Real.sin (3 * x) / 3 + Real.sin (5 * x) / 5) := by
  have hπ : π ≠ 0 := Real.pi_ne_zero
  simp only [squarePartial, Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty]
  norm_num
  field_simp
  try ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.square_partial_five
-- test: fa-square-partial-five-test.sage

/-! ## exm-fs-dirichlet-kernel -/

theorem dirichlet_three_integral :
    ∫ t in (-π)..π, (1 + 2 * Real.cos t + 2 * Real.cos (2 * t) + 2 * Real.cos (3 * t)) = 2 * π := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-π : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (2 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - (2 * ((Real.sin x_veriphy) ^ 3) / 3) +
                      (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy)) +
                    x_veriphy +
                  (2 * (Real.sin x_veriphy)) :
                ℝ))
            ((fun t => 1 + 2 * cos t + 2 * cos (2 * t) + 2 * cos (3 * t)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        have hmc3 :
          Real.cos (3 * x_veriphy) =
            ((Real.cos x_veriphy) ^ 3 - (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2)) : ℝ) :=
          by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc2, hms2]; ring
        have hms3 :
          Real.sin (3 * x_veriphy) =
            (3 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - ((Real.sin x_veriphy) ^ 3) : ℝ) :=
          by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc2, hms2]; ring
        refine
          (((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                    (2 : ℝ)).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (-(2 / 3) : ℝ))).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                  ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                              (2 : ℝ))).add
                        (hasDerivAt_id' (x_veriphy : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (2 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - (2 * ((Real.sin x_veriphy) ^ 3) / 3) +
                  (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy)) +
                x_veriphy +
              (2 * (Real.sin x_veriphy)) :
            ℝ))
          (f' := fun t => 1 + 2 * cos t + 2 * cos (2 * t) + 2 * cos (3 * t)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      have he2 : Real.cos (((-π) : ℝ)) = -1 := by
        rw [show (((-π) : ℝ) : ℝ) = -(((1 : ℕ) : ℝ) * Real.pi) by push_cast; ring, Real.cos_neg, Real.cos_nat_mul_pi]; norm_num
      have he3 : Real.sin (((-π) : ℝ)) = 0 := by
        rw [show (((-π) : ℝ) : ℝ) = -(((1 : ℕ) : ℝ) * Real.pi) by push_cast; ring, Real.sin_neg, Real.sin_nat_mul_pi]; simp
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: fa-trig-integral.sage

/-- `D_3(t) = sin (7t/2) / sin (t/2)` vanishes at `t = 2πk/7`. -/
theorem dirichlet_three_zeros (k : ℤ) : Real.sin ((3 + 1 / 2) * (2 * π * k / 7)) = 0 := by
  rw [show (3 + 1 / 2) * (2 * π * k / 7) = k * π by ring]
  exact Real.sin_int_mul_pi k

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.dirichlet_three_zeros
-- test: fa-dirichlet-three-zeros-test.sage

/-! ## exm-fs-gibbs-constant -/

/-- `Σ_{k < M} cos (2k+1)x · 2 sin x = sin 2Mx`: the derivative `S_N' = 2 sin (N+1)x / (π sin x)`. -/
theorem odd_cos_sum (M : ℕ) (x : ℝ) :
    (∑ k ∈ Finset.range M, Real.cos ((2 * k + 1) * x)) * (2 * Real.sin x) = Real.sin (2 * M * x) := by
  induction M with
  | zero => simp
  | succ M ih =>
    rw [Finset.sum_range_succ, add_mul, ih]
    have h1 : Real.sin (2 * ((M + 1 : ℕ) : ℝ) * x) =
        Real.sin ((2 * M + 1) * x) * Real.cos x + Real.cos ((2 * M + 1) * x) * Real.sin x := by
      rw [← Real.sin_add]; congr 1; push_cast; ring
    have h2 : Real.sin (2 * (M : ℝ) * x) =
        Real.sin ((2 * M + 1) * x) * Real.cos x - Real.cos ((2 * M + 1) * x) * Real.sin x := by
      rw [← Real.sin_sub]; congr 1; ring
    rw [h1, h2]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.odd_cos_sum
-- test: fa-odd-cos-sum-test.sage

/-- `Si(π) = ∫₀^π sin t / t dt`. -/
noncomputable def siPi : ℝ := ∫ t in (0 : ℝ)..π, Real.sin t / t

/-- The Gibbs phenomenon for the square wave, as stated (tested, not proved: no recipe evaluates `Si(π)`): the
value `S_{2M-1}(π/2M)` at the first maximum exceeds the Gibbs constant `(2/π) Si(π) ∈ (1.1789, 1.1790)` by less
than `1/M²`. -/
def GibbsOvershoot : Prop :=
  ∀ M : ℕ, 1 ≤ M → 1.1789 < 2 / π * siPi ∧ 2 / π * siPi < 1.179 ∧
    0 < squarePartial (2 * M) (π / (2 * M)) - 2 / π * siPi ∧
    squarePartial (2 * M) (π / (2 * M)) - 2 / π * siPi < 1 / M ^ 2

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PartialSums.GibbsOvershoot
-- test: fa-gibbs-overshoot-test.sage

/-! ## exr-fs-fejer -/

/-- `Σ_{n ≤ N} sin ((n + 1/2) t) · sin (t/2) = sin² ((N+1) t / 2)`. -/
theorem fejer_sum (N : ℕ) (t : ℝ) :
    (∑ n ∈ Finset.range (N + 1), Real.sin ((n + 1 / 2) * t)) * Real.sin (t / 2) =
      Real.sin ((N + 1) * t / 2) ^ 2 := by
  induction N with
  | zero =>
    simp only [zero_add, Finset.range_one, Finset.sum_singleton, Nat.cast_zero]
    ring_nf
  | succ N ih =>
    rw [Finset.sum_range_succ, add_mul, ih]
    set A := (N + 1) * t / 2
    set B := t / 2
    have e1 : Real.sin ((((N + 1 : ℕ) : ℝ) + 1 / 2) * t) = Real.sin (2 * A + B) := by congr 1; push_cast; ring
    have e2 : Real.sin ((((N + 1 : ℕ) : ℝ) + 1) * t / 2) = Real.sin (A + B) := by congr 1; push_cast; ring
    rw [e1, e2, Real.sin_add, Real.sin_add, Real.sin_two_mul, Real.cos_two_mul]
    linear_combination (Real.sin B) ^ 2 * Real.sin_sq_add_cos_sq A - (Real.sin A) ^ 2 * Real.sin_sq_add_cos_sq B

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.fejer_sum
-- test: fa-fejer-sum-test.sage

/-- The Fejér kernel `Σ_{n ≤ N} D_n` is non-negative. -/
theorem fejer_nonneg (N : ℕ) (t : ℝ) : 0 ≤ ∑ n ∈ Finset.range (N + 1), dirichlet n t := by
  by_cases hs : Real.sin (t / 2) = 0
  · obtain ⟨k, hk⟩ := Real.sin_eq_zero_iff.1 hs
    have hcos : ∀ m : ℕ, Real.cos ((m + 1) * t) = 1 := fun m => by
      rw [show ((m : ℝ) + 1) * t = (((m + 1 : ℕ) : ℤ) * k : ℤ) * (2 * π) by push_cast; rw [show t = 2 * (k * π) by linarith]; ring]
      exact Real.cos_int_mul_two_pi _
    apply Finset.sum_nonneg
    intro n _
    simp only [dirichlet, hcos, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
    positivity
  · have key : (∑ n ∈ Finset.range (N + 1), dirichlet n t) * Real.sin (t / 2) ^ 2 =
        Real.sin ((N + 1) * t / 2) ^ 2 := by
      rw [← fejer_sum, Finset.sum_mul, Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro n _
      rw [sq, ← mul_assoc, dirichlet_mul_sin]
    have hpos : 0 < Real.sin (t / 2) ^ 2 := by positivity
    by_contra hneg
    push_neg at hneg
    have : (∑ n ∈ Finset.range (N + 1), dirichlet n t) * Real.sin (t / 2) ^ 2 < 0 :=
      mul_neg_of_neg_of_pos hneg hpos
    rw [key] at this
    linarith [sq_nonneg (Real.sin ((N + 1) * t / 2))]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PartialSums.fejer_nonneg
-- test: fa-fejer-nonneg-test.sage

/-- `(1/2π) ∫ F_N = 1`, with `F_N = (1/(N+1)) Σ_{n ≤ N} D_n`. -/
theorem fejer_integral (N : ℕ) :
    ∫ t in (-π)..π, (1 / (N + 1 : ℝ)) * ∑ n ∈ Finset.range (N + 1), dirichlet n t = 2 * π := by
  rw [intervalIntegral.integral_const_mul,
    intervalIntegral.integral_finset_sum (fun n _ => by
      apply Continuous.intervalIntegrable; unfold dirichlet; fun_prop)]
  simp only [dirichlet_integral, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp
  push_cast
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.fejer_integral
-- test: fa-fejer-integral-test.sage

/-- A non-negative kernel of mass `2π` averages without overshoot: `|(1/2π) ∫ F(x - y) f(y) dy| ≤ sup |f|`. -/
theorem positive_kernel_bound (F f : ℝ → ℝ) (hF : Continuous F) (hf : Continuous f) (hpos : ∀ t, 0 ≤ F t)
    (M : ℝ) (hM : ∀ y, |f y| ≤ M) (x : ℝ) (hmass : ∫ y in (-π)..π, F (x - y) = 2 * π) :
    |1 / (2 * π) * ∫ y in (-π)..π, F (x - y) * f y| ≤ M := by
  have hπ : 0 < 2 * π := by positivity
  have h1 : |∫ y in (-π)..π, F (x - y) * f y| ≤ ∫ y in (-π)..π, F (x - y) * M := by
    apply intervalIntegral.abs_integral_le_integral_abs (by linarith [Real.pi_pos]) |>.trans
    apply intervalIntegral.integral_mono_on (by linarith [Real.pi_pos])
    · apply Continuous.intervalIntegrable; fun_prop
    · apply Continuous.intervalIntegrable; fun_prop
    · intro y _
      rw [abs_mul, abs_of_nonneg (hpos _)]
      exact mul_le_mul_of_nonneg_left (hM y) (hpos _)
  rw [intervalIntegral.integral_mul_const, hmass] at h1
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 1 / (2 * π))]
  calc 1 / (2 * π) * |∫ y in (-π)..π, F (x - y) * f y| ≤ 1 / (2 * π) * (2 * π * M) := by gcongr
    _ = M := by field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PartialSums.positive_kernel_bound
-- test: fa-positive-kernel-bound-test.sage

end KaluzaKlein.FourierAnalysis.PartialSums

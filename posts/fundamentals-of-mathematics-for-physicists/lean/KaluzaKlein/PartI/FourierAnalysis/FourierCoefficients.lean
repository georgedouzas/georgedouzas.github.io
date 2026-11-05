import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.PeriodicFunctions

/-! # Fourier analysis: fourier coefficients (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.FourierCoefficients

open Real

/-! ## def-fourier-coefficient -/

/-- The orthogonality of the modes: `∫₀^{2π} e^{inx} e^{-imx} dx = 2π δ_{nm}`. -/
theorem mode_orthogonality (n m : ℤ) :
    ∫ x in (0 : ℝ)..2 * π, Complex.exp (n * x * Complex.I) * Complex.exp (-(m * x * Complex.I)) =
      if n = m then 2 * π else 0 := by
  have hfun : (fun x : ℝ => Complex.exp (n * x * Complex.I) * Complex.exp (-(m * x * Complex.I))) =
      fun x : ℝ => Complex.exp (((n - m : ℤ) * Complex.I) * x) := by
    funext x; rw [← Complex.exp_add]; push_cast; ring_nf
  rw [hfun]
  split_ifs with h
  · subst h; simp
  · have hc : ((n - m : ℤ) : ℂ) * Complex.I ≠ 0 := by
      apply mul_ne_zero _ Complex.I_ne_zero
      exact_mod_cast sub_ne_zero.2 h
    have : Complex.exp (((n - m : ℤ) : ℂ) * Complex.I * ((2 * π : ℝ) : ℂ)) = 1 := by
      rw [show ((n - m : ℤ) : ℂ) * Complex.I * ((2 * π : ℝ) : ℂ) = ((n - m : ℤ) : ℂ) * (2 * π * Complex.I) by
        push_cast; ring]
      exact Complex.exp_int_mul_two_pi_mul_I _
    rw [integral_exp_mul_complex hc, this]
    simp

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.FourierAnalysis.FourierCoefficients.mode_orthogonality
-- test: fa-mode-orthogonality-test.sage

/-- For a real `f`, `2π c_n = π a_n - i π b_n`: the complex coefficient in terms of the real ones,
`c_n = (a_n - i b_n)/2`. -/
theorem coeff_complex_real (f : ℝ → ℝ) (hf : Continuous f) (n : ℝ) :
    ∫ x in (0 : ℝ)..2 * π, (f x : ℂ) * Complex.exp (-(n * x * Complex.I)) =
      ((∫ x in (0 : ℝ)..2 * π, f x * Real.cos (n * x) : ℝ) : ℂ) -
        ((∫ x in (0 : ℝ)..2 * π, f x * Real.sin (n * x) : ℝ) : ℂ) * Complex.I := by
  have hpt : ∀ x : ℝ, (f x : ℂ) * Complex.exp (-(n * x * Complex.I)) =
      ((f x * Real.cos (n * x) : ℝ) : ℂ) - ((f x * Real.sin (n * x) : ℝ) : ℂ) * Complex.I := by
    intro x
    rw [show -((n : ℂ) * x * Complex.I) = ((-(n * x) : ℝ) : ℂ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I]
    push_cast
    simp only [Complex.cos_neg, Complex.sin_neg]
    ring
  simp_rw [hpt]
  have hc : Continuous fun x => ((f x * Real.cos (n * x) : ℝ) : ℂ) := by fun_prop
  have hs : Continuous fun x => ((f x * Real.sin (n * x) : ℝ) : ℂ) * Complex.I := by fun_prop
  rw [intervalIntegral.integral_sub (hc.intervalIntegrable _ _) (hs.intervalIntegrable _ _),
    intervalIntegral.integral_mul_const, intervalIntegral.integral_ofReal, intervalIntegral.integral_ofReal]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.coeff_complex_real
-- test: fa-coeff-complex-real-test.sage

/-! ## exm-fs-coefficients-square -/

/-- `∫₀^π sin n x dx = (1 - cos n π) / n`. -/
theorem integral_sin_nat_mul (n : ℕ) (hn : n ≠ 0) :
    ∫ x in (0 : ℝ)..π, Real.sin (n * x) = (1 - Real.cos (n * π)) / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hn', integral_sin, smul_eq_mul]
  simp only [mul_zero, Real.cos_zero]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.integral_sin_nat_mul
-- test: fa-integral-sin-nat-mul-test.sage

/-- The square wave's sine coefficient: `b_n = (2/π) ∫₀^π sin n x dx = 2 (1 - (-1)^n) / (π n)`. -/
theorem square_wave_bn (n : ℕ) (hn : n ≠ 0) :
    2 / π * ∫ x in (0 : ℝ)..π, Real.sin (n * x) = 2 * (1 - (-1) ^ n) / (π * n) := by
  rw [integral_sin_nat_mul n hn, Real.cos_nat_mul_pi]
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.FourierCoefficients.square_wave_bn
-- test: fa-square-wave-bn-test.sage

/-- `b_3 = (2/π) · 2/3 = 4/(3π)`. -/
theorem square_wave_int3 : ∫ x in (0 : ℝ)..π, Real.sin (3 * x) = 2 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos (3 * x_veriphy)) / 3 : ℝ)) ((fun x => sin (3 * x)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).cos).const_mul (-(1 / 3) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos (3 * x_veriphy)) / 3 : ℝ))
          (f' := fun x => sin (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((3 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((3 * (π) : ℝ)) = -1 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-integral.sage

theorem square_wave_b2 : ∫ x in (0 : ℝ)..π, Real.sin (2 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ)) ((fun x => sin (2 * x)) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos (2 * x_veriphy)) / 2 : ℝ))
          (f' := fun x => sin (2 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos ((2 * (0) : ℝ)) = 1 := by simp
      have he1 : Real.cos ((2 * (π) : ℝ)) = 1 := by
        rw [show ((2 * (π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-integral.sage

/-! ## exm-fs-coefficients-sawtooth -/

/-- The sawtooth's coefficient: `(1/π) ∫_{-π}^{π} x sin n x dx = 2 (-1)^{n+1} / n`. -/
theorem sawtooth_bn (n : ℕ) (hn : n ≠ 0) :
    1 / π * ∫ x in (-π)..π, x * Real.sin (n * x) = 2 * (-1) ^ (n + 1) / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hF : ∀ x ∈ Set.uIcc (-π) π, HasDerivAt
      (fun x => Real.sin (n * x) / n ^ 2 - x * Real.cos (n * x) / n) (x * Real.sin (n * x)) x := by
    intro x _
    have e1 := (((hasDerivAt_id' x).const_mul (n : ℝ)).sin).div_const ((n : ℝ) ^ 2)
    have e2 := ((hasDerivAt_id' x).mul (((hasDerivAt_id' x).const_mul (n : ℝ)).cos)).div_const (n : ℝ)
    convert e1.sub e2 using 1
    all_goals fa_close1
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_neg, Real.sin_neg, Real.cos_neg, Real.sin_nat_mul_pi, Real.cos_nat_mul_pi]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.FourierCoefficients.sawtooth_bn
-- test: fa-sawtooth-bn-test.sage

theorem sawtooth_b2 : ∫ x in (-π)..π, x * Real.sin (2 * x) = -π := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-π : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-x_veriphy * ((Real.cos x_veriphy) ^ 2) / 2 + (x_veriphy * ((Real.sin x_veriphy) ^ 2) / 2) +
                  ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) :
                ℝ))
            ((fun x => x * sin (2 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).mul (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2)).const_mul
                            (-(1 / 2) : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).mul (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                          (1 / 2 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-x_veriphy * ((Real.cos x_veriphy) ^ 2) / 2 + (x_veriphy * ((Real.sin x_veriphy) ^ 2) / 2) +
              ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) :
            ℝ))
          (f' := fun x => x * sin (2 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
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

/-! ## exm-fs-coefficients-smooth, exr-fs-coefficients-abs -/

/-- The coefficient of `|x|`: `(2/π) ∫₀^π x cos n x dx = 2 (cos n π - 1) / (π n²)`. -/
theorem abs_an (n : ℕ) (hn : n ≠ 0) :
    2 / π * ∫ x in (0 : ℝ)..π, x * Real.cos (n * x) = 2 * (Real.cos (n * π) - 1) / (π * n ^ 2) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hF : ∀ x ∈ Set.uIcc 0 π, HasDerivAt
      (fun x => x * Real.sin (n * x) / n + Real.cos (n * x) / n ^ 2) (x * Real.cos (n * x)) x := by
    intro x _
    have e1 := ((hasDerivAt_id' x).mul (((hasDerivAt_id' x).const_mul (n : ℝ)).sin)).div_const (n : ℝ)
    have e2 := (((hasDerivAt_id' x).const_mul (n : ℝ)).cos).div_const ((n : ℝ) ^ 2)
    convert e1.add e2 using 1
    all_goals fa_close1
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, Real.sin_zero, Real.cos_zero, Real.sin_nat_mul_pi]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.abs_an
-- test: fa-abs-an-test.sage

/-- `a_0 = (1/π) ∫_{-π}^{π} |x| dx = π`, from the two halves `∫_{-π}^0 (-x) = ∫_0^π x = π²/2`. -/
theorem abs_a0 : 1 / π * ∫ x in (-π)..π, |x| = π := by
  have hr : ∫ x in (0 : ℝ)..π, x = π ^ 2 / 2 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
            HasDerivAt (fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) ((fun x => x) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 2) / 2 : ℝ)) (f' :=
            fun x => x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals (norm_num; done))
    -- certificate: fa-integral.sage
  have hl : ∫ x in (-π)..0, -x = π ^ 2 / 2 := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (-π : ℝ) (0 : ℝ),
            HasDerivAt (fun x_veriphy => (-(x_veriphy ^ 2) / 2 : ℝ)) ((fun x => -x) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · ring1
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(x_veriphy ^ 2) / 2 : ℝ)) (f' :=
            fun x => -x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        all_goals ring1)
    -- certificate: fa-integral.sage
  have h1 : ∫ x in (-π)..π, |x| = π ^ 2 := by
    rw [← intervalIntegral.integral_add_adjacent_intervals (b := 0)
      (by apply Continuous.intervalIntegrable; fun_prop) (by apply Continuous.intervalIntegrable; fun_prop)]
    have l : ∫ x in (-π)..0, |x| = ∫ x in (-π)..0, -x := by
      apply intervalIntegral.integral_congr
      intro x hx
      rw [Set.uIcc_of_le (by linarith [Real.pi_pos])] at hx
      exact abs_of_nonpos hx.2
    have r : ∫ x in (0 : ℝ)..π, |x| = ∫ x in (0 : ℝ)..π, x := by
      apply intervalIntegral.integral_congr
      intro x hx
      rw [Set.uIcc_of_le (by linarith [Real.pi_pos])] at hx
      exact abs_of_nonneg hx.1
    rw [l, r, hl, hr]
    ring
  rw [h1]; field_simp

/-- `a_1 = (2/π) · (-2) = -4/π`. -/
theorem abs_int1 : ∫ x in (0 : ℝ)..π, x * Real.cos x = -2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (x_veriphy * (Real.sin x_veriphy) + (Real.cos x_veriphy) : ℝ))
            ((fun x => x * cos x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).cos))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (x_veriphy * (Real.sin x_veriphy) + (Real.cos x_veriphy) : ℝ)) (f' := fun x => x * cos x) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      have he2 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he3 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-integral.sage

theorem abs_a3 : ∫ x in (0 : ℝ)..π, x * Real.cos (3 * x) = -2 / 9 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy => (x_veriphy * (Real.sin (3 * x_veriphy)) / 3 + ((Real.cos (3 * x_veriphy)) / 9) : ℝ))
            ((fun x => x * cos (3 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                            (((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).sin)).const_mul
                        (1 / 3 : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 : ℝ)).cos).const_mul
                      (1 / 9 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (x_veriphy * (Real.sin (3 * x_veriphy)) / 3 + ((Real.cos (3 * x_veriphy)) / 9) : ℝ)) (f' := fun x =>
          x * cos (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((3 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((3 * (π) : ℝ)) = 0 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 3
      have he2 : Real.cos ((3 * (0) : ℝ)) = 1 := by simp
      have he3 : Real.cos ((3 * (π) : ℝ)) = -1 := by
        rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-integral.sage

theorem cos_cube_coeff_four : ∫ x in (0 : ℝ)..(2 * π), Real.cos x ^ 3 * Real.cos (4 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-8 * ((Real.sin x_veriphy) ^ 7) / 7 + (16 * ((Real.sin x_veriphy) ^ 5) / 5) -
                    (3 * ((Real.sin x_veriphy) ^ 3)) +
                  (Real.sin x_veriphy) :
                ℝ))
            ((fun x => cos x ^ 3 * cos (4 * x)) x_veriphy) x_veriphy :=
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
        have hmc4 :
          Real.cos (4 * x_veriphy) =
            ((Real.cos x_veriphy) ^ 4 - (6 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2)) +
                ((Real.sin x_veriphy) ^ 4) :
              ℝ) :=
          by rw [show (4 : ℝ) * x_veriphy = 3 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc3, hms3]; ring
        have hms4 :
          Real.sin (4 * x_veriphy) =
            (4 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) -
                (4 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) :
              ℝ) :=
          by rw [show (4 : ℝ) * x_veriphy = 3 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc3, hms3]; ring
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 7).const_mul (-(8 / 7) : ℝ)).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (16 / 5 : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (-3 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).sin))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3, hmc4, hms4]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 5) + (7 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 2)) -
                        (8 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 4)) -
                      ((Real.cos x_veriphy) ^ 3) +
                    (8 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2)) -
                  (Real.cos x_veriphy) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-8 * ((Real.sin x_veriphy) ^ 7) / 7 + (16 * ((Real.sin x_veriphy) ^ 5) / 5) -
                (3 * ((Real.sin x_veriphy) ^ 3)) +
              (Real.sin x_veriphy) :
            ℝ))
          (f' := fun x => cos x ^ 3 * cos (4 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-! ## exr-fs-coefficients-complex-real -/

/-- An even function has vanishing sine coefficients. -/
theorem even_bn_zero (f : ℝ → ℝ) (hf : ∀ x, f (-x) = f x) (n : ℝ) :
    ∫ x in (-π)..π, f x * Real.sin (n * x) = 0 := by
  have h := intervalIntegral.integral_comp_neg (a := -π) (b := π) (fun x => f x * Real.sin (n * x))
  simp only [neg_neg, hf, mul_neg, Real.sin_neg, intervalIntegral.integral_neg] at h
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.even_bn_zero
-- test: fa-even-bn-zero-test.sage

/-- An odd function has vanishing cosine coefficients. -/
theorem odd_an_zero (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (n : ℝ) :
    ∫ x in (-π)..π, f x * Real.cos (n * x) = 0 := by
  have h := intervalIntegral.integral_comp_neg (a := -π) (b := π) (fun x => f x * Real.cos (n * x))
  simp only [neg_neg, hf, mul_neg, Real.cos_neg, neg_mul, intervalIntegral.integral_neg] at h
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.odd_an_zero
-- test: fa-odd-an-zero-test.sage

/-- For real `f`, `c_{-n} = conj c_n`. -/
theorem real_coeff_conj (f : ℝ → ℝ) (n : ℝ) :
    (starRingEnd ℂ) (∫ x in (0 : ℝ)..2 * π, (f x : ℂ) * Complex.exp (-(n * x * Complex.I))) =
      ∫ x in (0 : ℝ)..2 * π, (f x : ℂ) * Complex.exp (-((-n : ℝ) * x * Complex.I)) := by
  rw [intervalIntegral.integral_of_le (by positivity), intervalIntegral.integral_of_le (by positivity), ← integral_conj]
  congr 1
  funext x
  rw [map_mul, Complex.conj_ofReal, ← Complex.exp_conj]
  congr 2
  first | (simp [Complex.conj_ofReal]; done) | (simp [Complex.conj_ofReal]; ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.real_coeff_conj
-- test: fa-real-coeff-conj-test.sage

/-! ## exr-fs-coefficients-decay -/

/-- Integration by parts for the coefficients: `∫₀^{2π} f' e^{-inx} = [f e^{-inx}]₀^{2π} + i n ∫₀^{2π} f e^{-inx}`;
for a periodic `f` the boundary term vanishes and `c_n(f') = i n c_n(f)`. -/
theorem coeff_derivative (f f' : ℝ → ℂ) (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : Continuous f')
    (hper : f (2 * π) = f 0) (n : ℤ) :
    ∫ x in (0 : ℝ)..2 * π, f' x * Complex.exp (-(n * x * Complex.I)) =
      n * Complex.I * ∫ x in (0 : ℝ)..2 * π, f x * Complex.exp (-(n * x * Complex.I)) := by
  have he : ∀ x : ℝ, HasDerivAt (fun x : ℝ => Complex.exp (-(n * x * Complex.I)))
      (-(n * Complex.I) * Complex.exp (-(n * x * Complex.I))) x := by
    intro x
    have := ((hasDerivAt_id (x : ℂ)).const_mul (-(n * Complex.I))).cexp.comp_ofReal
    convert this using 1
    all_goals fa_close1
  have hcont : Continuous f := continuous_iff_continuousAt.2 fun x => (hf x).continuousAt
  have e2 : Complex.exp (-(n * ((2 * π : ℝ) : ℂ) * Complex.I)) = 1 := by
    rw [show -((n : ℂ) * ((2 * π : ℝ) : ℂ) * Complex.I) = ((-n : ℤ) : ℂ) * (2 * π * Complex.I) by push_cast; ring]
    exact Complex.exp_int_mul_two_pi_mul_I _
  have key : ∫ x in (0 : ℝ)..2 * π, f x * (-(n * Complex.I) * Complex.exp (-(n * x * Complex.I))) =
      -(n * Complex.I) * ∫ x in (0 : ℝ)..2 * π, f x * Complex.exp (-(n * x * Complex.I)) := by
    rw [← intervalIntegral.integral_const_mul]; congr 1; funext x; ring
  have parts := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := 0) (b := 2 * π)
    (fun x _ => hf x) (fun x _ => he x) (hf'.intervalIntegrable _ _)
    (by apply Continuous.intervalIntegrable; fun_prop)
  rw [key] at parts
  simp only [Complex.ofReal_zero, mul_zero, zero_mul, neg_zero, Complex.exp_zero, mul_one, hper, e2] at parts
  linear_combination parts

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.coeff_derivative
-- test: fa-coeff-derivative-test.sage

/-- The sawtooth's boundary term: `[x e^{-inx}]_{-π}^{π} = 2π (-1)^n`, giving `c_n = i (-1)^n / n`. -/
theorem sawtooth_boundary (n : ℤ) :
    (π : ℂ) * Complex.exp (-(n * π * Complex.I)) - (-π : ℝ) * Complex.exp (-(n * (-π : ℝ) * Complex.I)) =
      2 * π * (-1) ^ n := by
  have h1 : Complex.exp (-(n * π * Complex.I)) = (-1) ^ n := by
    rw [show -((n : ℂ) * π * Complex.I) = (-n : ℤ) * (π * Complex.I) by push_cast; ring,
      Complex.exp_int_mul, Complex.exp_pi_mul_I, zpow_neg, ← inv_zpow, inv_neg, inv_one]
  have h2 : Complex.exp (-(n * ((-π : ℝ) : ℂ) * Complex.I)) = (-1) ^ n := by
    rw [show -((n : ℂ) * ((-π : ℝ) : ℂ) * Complex.I) = (n : ℤ) * (π * Complex.I) by push_cast; ring,
      Complex.exp_int_mul, Complex.exp_pi_mul_I]
  rw [h1, h2]; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.FourierCoefficients.sawtooth_boundary
-- test: fa-sawtooth-boundary-test.sage

end KaluzaKlein.FourierAnalysis.FourierCoefficients

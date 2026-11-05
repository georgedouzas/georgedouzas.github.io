import Mathlib
import Veriphy.Kit.Levels

/-- Closes the goals left by `convert` on a composed `HasDerivAt`: function equality (pointwise) and the derivative. -/
macro "fa_close1" : tactic => `(tactic| first
  | rfl
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; done)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; ring1)
  | (funext y; simp only [Function.comp_apply, id_eq, Pi.mul_apply, Pi.add_apply, Pi.sub_apply, Pi.div_apply, Pi.neg_apply, Pi.pow_apply]; field_simp; done)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; ring1)
  | (simp only [Function.comp_apply, id_eq, Pi.mul_apply, mul_one, one_mul]; field_simp; done)
  | ring1
  | (field_simp; done)
  | (field_simp; ring1)
  | (norm_num; done)
  | (norm_num; ring1)
  | (norm_num; field_simp; ring1)
  | (simp; done)
  | (simp; ring1)
  | (ring_nf; done)
  | (field_simp; ring_nf; done))

/-! # Fourier analysis: periodic functions (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.PeriodicFunctions

open Real

/-! ## def-periodic-function -/

/-- A function with period `T` has every period `k T`, `k ∈ ℕ`. -/
theorem periodic_nat_mul (f : ℝ → ℝ) (T : ℝ) (hf : ∀ x, f (x + T) = f x) (k : ℕ) (x : ℝ) :
    f (x + k * T) = f x := by
  induction k generalizing x with
  | zero => simp
  | succ k ih =>
    rw [show x + ((k + 1 : ℕ) : ℝ) * T = (x + k * T) + T by push_cast; ring, hf, ih]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.periodic_nat_mul
-- test: fa-periodic-nat-mul-test.sage

/-- The rescaling `x ↦ T x / 2π` turns a function of period `T` into one of period `2π`. -/
theorem rescale_two_pi (f : ℝ → ℝ) (T : ℝ) (hf : ∀ x, f (x + T) = f x) (x : ℝ) :
    f (T * (x + 2 * π) / (2 * π)) = f (T * x / (2 * π)) := by
  rw [show T * (x + 2 * π) / (2 * π) = T * x / (2 * π) + T by field_simp, hf]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.rescale_two_pi
-- test: fa-rescale-two-pi-test.sage

/-- The complex and real forms of a trigonometric polynomial: `c e^{it} + c̄ e^{-it} = a cos t + b sin t` for
`c = (a - i b) / 2`. -/
theorem trig_pair (a b t : ℝ) :
    ((a : ℂ) - b * Complex.I) / 2 * Complex.exp (t * Complex.I) +
      ((a : ℂ) + b * Complex.I) / 2 * Complex.exp ((-t : ℝ) * Complex.I) =
      ((a * Real.cos t + b * Real.sin t : ℝ) : ℂ) := by
  rw [Complex.exp_mul_I, Complex.exp_mul_I]
  push_cast
  simp only [Complex.cos_neg, Complex.sin_neg]
  linear_combination (-((b : ℂ) * Complex.sin (t : ℂ))) * Complex.I_sq

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.trig_pair
-- test: fa-trig-pair-test.sage

/-! ## exm-fs-periodic-basic -/

theorem cos_int_mul_periodic (n : ℤ) (x : ℝ) : Real.cos (n * (x + 2 * π)) = Real.cos (n * x) := by
  rw [mul_add, show (n : ℝ) * (2 * π) = (n : ℝ) * (2 * π) from rfl, Real.cos_add_int_mul_two_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.cos_int_mul_periodic
-- test: fa-cos-int-mul-periodic-test.sage

theorem sin_int_mul_periodic (n : ℤ) (x : ℝ) : Real.sin (n * (x + 2 * π)) = Real.sin (n * x) := by
  rw [mul_add, Real.sin_add_int_mul_two_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_int_mul_periodic
-- test: fa-sin-int-mul-periodic-test.sage

/-- `2π / n` is a period of `sin (n x)`. -/
theorem sin_short_period (n : ℝ) (hn : n ≠ 0) (x : ℝ) :
    Real.sin (n * (x + 2 * π / n)) = Real.sin (n * x) := by
  rw [show n * (x + 2 * π / n) = n * x + 2 * π by field_simp, Real.sin_add_two_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_short_period
-- test: fa-sin-short-period-test.sage

/-- No `T ∈ (0, 2π)` is a period of `sin`: `2π` is the smallest period. -/
theorem sin_no_smaller_period (T : ℝ) (h0 : 0 < T) (h1 : T < 2 * π) :
    Real.sin (π / 2 + T) ≠ Real.sin (π / 2) := by
  rw [Real.sin_pi_div_two, Real.sin_add, Real.sin_pi_div_two, Real.cos_pi_div_two]
  intro h
  have h2 : Real.cos T = 1 := by linarith
  have := (Real.cos_eq_one_iff_of_lt_of_lt (by linarith [Real.pi_pos]) h1).1 h2
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_no_smaller_period
-- test: fa-sin-no-smaller-period-test.sage

/-- Euler's formula for the harmonics: `e^{i n x} = cos n x + i sin n x`. -/
theorem euler_harmonic (n x : ℝ) :
    Complex.exp (((n * x : ℝ) : ℂ) * Complex.I) =
      ((Real.cos (n * x) : ℝ) : ℂ) + ((Real.sin (n * x) : ℝ) : ℂ) * Complex.I := by
  rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.euler_harmonic
-- test: fa-euler-harmonic-test.sage

/-! ## exm-fs-periodic-extension -/

/-- The square wave: the periodic extension of `sgn x` from `(-π, π]`. -/
noncomputable def squareWave (x : ℝ) : ℝ := Real.sign (toIocMod Real.two_pi_pos (-π) x)

/-- The sawtooth: the periodic extension of `x` from `(-π, π]`. -/
noncomputable def sawtooth (x : ℝ) : ℝ := toIocMod Real.two_pi_pos (-π) x

/-- The periodic extension of `|x|` from `(-π, π]`. -/
noncomputable def absWave (x : ℝ) : ℝ := |toIocMod Real.two_pi_pos (-π) x|

theorem squareWave_periodic (x : ℝ) : squareWave (x + 2 * π) = squareWave x := by
  simp only [squareWave, toIocMod_add_right]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PeriodicFunctions.squareWave_periodic
-- test: fa-squarewave-periodic-test.sage

theorem sawtooth_periodic (x : ℝ) : sawtooth (x + 2 * π) = sawtooth x := by
  simp only [sawtooth, toIocMod_add_right]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sawtooth_periodic
-- test: fa-sawtooth-periodic-test.sage

theorem squareWave_of_pos (x : ℝ) (h0 : 0 < x) (h1 : x < π) : squareWave x = 1 := by
  rw [squareWave, (toIocMod_eq_self _).2 ⟨by linarith, by linarith⟩]
  exact Real.sign_of_pos h0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.squareWave_of_pos
-- test: fa-squarewave-of-pos-test.sage

theorem squareWave_of_neg (x : ℝ) (h0 : -π < x) (h1 : x < 0) : squareWave x = -1 := by
  rw [squareWave, (toIocMod_eq_self _).2 ⟨by linarith, by linarith [Real.pi_pos]⟩]
  exact Real.sign_of_neg h1

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.squareWave_of_neg
-- test: fa-squarewave-of-neg-test.sage

/-- Left of `π` the sawtooth is `x`, so `f(π⁻) = π`. -/
theorem sawtooth_left (t : ℝ) (h0 : 0 < t) (h1 : t < 2 * π) : sawtooth (π - t) = π - t := by
  rw [sawtooth, (toIocMod_eq_self _).2 ⟨by linarith, by linarith⟩]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sawtooth_left
-- test: fa-sawtooth-left-test.sage

/-- Right of `π` the sawtooth is `x - 2π`, so `f(π⁺) = -π`: the jump is `-2π`. -/
theorem sawtooth_right (t : ℝ) (h0 : 0 < t) (h1 : t ≤ 2 * π) : sawtooth (π + t) = t - π := by
  rw [sawtooth, show π + t = (t - π) + 2 * π by ring, toIocMod_add_right,
    (toIocMod_eq_self _).2 ⟨by linarith, by linarith⟩]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sawtooth_right
-- test: fa-sawtooth-right-test.sage

/-- The extension of `|x|` takes the values `|π - t|` on both sides of `π`: it is continuous there. -/
theorem absWave_around_pi (t : ℝ) (h0 : 0 < t) (h1 : t < 2 * π) :
    absWave (π - t) = |π - t| ∧ absWave (π + t) = |π - t| := by
  constructor
  · rw [absWave, (toIocMod_eq_self _).2 ⟨by linarith, by linarith⟩]
  · rw [absWave, show π + t = (t - π) + 2 * π by ring, toIocMod_add_right,
      (toIocMod_eq_self _).2 ⟨by linarith, by linarith⟩, abs_sub_comm]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.absWave_around_pi
-- test: fa-abswave-around-pi-test.sage

/-! ## exm-fs-trig-polynomial -/

theorem cos_cube (x : ℝ) : Real.cos x ^ 3 = 3 / 4 * Real.cos x + 1 / 4 * Real.cos (3 * x) := by
  veriphy_certified
    ((  rw [Real.cos_three_mul]
        all_goals ring1))
  -- certificate: fa-trig.sage

theorem cos_cube_coeff_one : ∫ x in (0 : ℝ)..(2 * π), Real.cos x ^ 3 * Real.cos x = 3 * π / 4 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 8 -
                      ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 8) +
                    ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                  (3 * x_veriphy / 8) :
                ℝ))
            ((fun x => cos x ^ 3 * cos x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                (1 / 8 : ℝ)).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (-(1 / 8) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / 2 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-7 * ((Real.cos x_veriphy) ^ 2) / 8 + (((Real.sin x_veriphy) ^ 2) / 8) - (3 / 8) : ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 8 -
                  ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 8) +
                ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
              (3 * x_veriphy / 8) :
            ℝ))
          (f' := fun x => cos x ^ 3 * cos x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: fa-trig-integral.sage

theorem cos_cube_coeff_two : ∫ x in (0 : ℝ)..(2 * π), Real.cos x ^ 3 * Real.cos (2 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (2 * ((Real.sin x_veriphy) ^ 5) / 5 - ((Real.sin x_veriphy) ^ 3) + (Real.sin x_veriphy) : ℝ))
            ((fun x => cos x ^ 3 * cos (2 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (2 / 5 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (-1 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).sin))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · simp only [hmc2, hms2]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 3) + (2 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2)) -
                  (Real.cos x_veriphy) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (2 * ((Real.sin x_veriphy) ^ 5) / 5 - ((Real.sin x_veriphy) ^ 3) + (Real.sin x_veriphy) : ℝ)) (f' := fun x =>
          cos x ^ 3 * cos (2 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem cos_cube_coeff_three : ∫ x in (0 : ℝ)..(2 * π), Real.cos x ^ 3 * Real.cos (3 * x) = π / 4 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-2 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 3 +
                        (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 8) -
                      (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 8) +
                    ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                  (x_veriphy / 8) :
                ℝ))
            ((fun x => cos x ^ 3 * cos (3 * x)) x_veriphy) x_veriphy :=
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
          (((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                    (-(2 / 3) : ℝ)).add
                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                      ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                  (3 / 8 : ℝ))).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (-(3 / 8) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / 2 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 8 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) + (2 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2)) -
                      (5 * ((Real.cos x_veriphy) ^ 2) / 8) +
                    (3 * ((Real.sin x_veriphy) ^ 2) / 8) -
                  (1 / 8) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-2 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 3 +
                    (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 8) -
                  (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 8) +
                ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
              (x_veriphy / 8) :
            ℝ))
          (f' := fun x => cos x ^ 3 * cos (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: fa-trig-integral.sage

/-! ## exr-fs-period-sum -/

theorem sin_sum_periodic (x : ℝ) :
    Real.sin (x + 2 * π) + Real.sin (2 * (x + 2 * π)) = Real.sin x + Real.sin (2 * x) := by
  rw [Real.sin_add_two_pi, mul_add, show 2 * (2 * π) = (2 : ℕ) * (2 * π) by norm_num,
    Real.sin_add_nat_mul_two_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_sum_periodic
-- test: fa-sin-sum-periodic-test.sage

theorem sin_half_sum_periodic (x : ℝ) :
    Real.sin (x + 4 * π) + Real.sin ((x + 4 * π) / 2) = Real.sin x + Real.sin (x / 2) := by
  rw [show x + 4 * π = x + (2 : ℕ) * (2 * π) by push_cast; ring, Real.sin_add_nat_mul_two_pi,
    show (x + (2 : ℕ) * (2 * π)) / 2 = x / 2 + 2 * π by push_cast; ring, Real.sin_add_two_pi]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_half_sum_periodic
-- test: fa-sin-half-sum-periodic-test.sage

/-- A period `T` of `sin x + sin (x/2)` has `sin (T/2) = 0` and `cos (T/2) = 1`, so `T ∈ 4πℤ`. -/
theorem sin_half_sum_period (T : ℝ)
    (hT : ∀ x, Real.sin (x + T) + Real.sin ((x + T) / 2) = Real.sin x + Real.sin (x / 2)) :
    Real.sin (T / 2) = 0 ∧ Real.cos (T / 2) = 1 := by
  have h0 := hT 0
  have hπ := hT π
  simp only [zero_add, zero_div, Real.sin_zero, add_zero, Real.sin_pi, Real.sin_pi_div_two] at h0 hπ
  have e1 : Real.sin T = 2 * Real.sin (T / 2) * Real.cos (T / 2) := by
    rw [← Real.sin_two_mul]; ring_nf
  have e2 : Real.sin (π + T) = -Real.sin T := by rw [add_comm, Real.sin_add_pi]
  have e3 : Real.sin ((π + T) / 2) = Real.cos (T / 2) := by
    rw [show (π + T) / 2 = T / 2 + π / 2 by ring, Real.sin_add_pi_div_two]
  rw [e2, e3, e1] at hπ
  rw [e1] at h0
  have hsc := Real.sin_sq_add_cos_sq (T / 2)
  have hs : Real.sin (T / 2) = 0 := by
    by_contra hne
    have hc : 2 * Real.cos (T / 2) + 1 = 0 := by
      have : Real.sin (T / 2) * (2 * Real.cos (T / 2) + 1) = 0 := by linarith
      rcases mul_eq_zero.1 this with h | h
      · exact absurd h hne
      · exact h
    nlinarith
  refine ⟨hs, ?_⟩
  rw [hs] at hπ
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_half_sum_period
-- test: fa-sin-half-sum-period-test.sage

/-- `sin x + sin (√2 x)` has no period `T ≠ 0`. -/
theorem sin_sqrt_two_not_periodic (T : ℝ)
    (hT : ∀ x, Real.sin (x + T) + Real.sin (Real.sqrt 2 * (x + T)) = Real.sin x + Real.sin (Real.sqrt 2 * x)) :
    T = 0 := by
  set a := Real.sqrt 2 with ha
  have ha2 : a ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  -- h = g(· + T) - g vanishes, so do its first and second derivatives
  let h1 : ℝ → ℝ := fun x => Real.cos (x + T) + a * Real.cos (a * (x + T)) - Real.cos x - a * Real.cos (a * x)
  let h2 : ℝ → ℝ := fun x =>
    -Real.sin (x + T) - a ^ 2 * Real.sin (a * (x + T)) + Real.sin x + a ^ 2 * Real.sin (a * x)
  have d0 : ∀ x, HasDerivAt (fun x => Real.sin (x + T) + Real.sin (a * (x + T)) - Real.sin x - Real.sin (a * x))
      (h1 x) x := by
    intro x
    have e1 := ((Real.hasDerivAt_sin (x + T)).comp x ((hasDerivAt_id x).add_const T))
    have e2 := ((Real.hasDerivAt_sin (a * (x + T))).comp x
      (((hasDerivAt_id x).add_const T).const_mul a))
    have e3 := Real.hasDerivAt_sin x
    have e4 := ((Real.hasDerivAt_sin (a * x)).comp x ((hasDerivAt_id x).const_mul a))
    convert ((e1.add e2).sub e3).sub e4 using 1 <;> (try simp only [h1]) <;> fa_close1
  have d1 : ∀ x, HasDerivAt h1 (h2 x) x := by
    intro x
    have e1 := ((Real.hasDerivAt_cos (x + T)).comp x ((hasDerivAt_id x).add_const T))
    have e2 := ((Real.hasDerivAt_cos (a * (x + T))).comp x
      (((hasDerivAt_id x).add_const T).const_mul a)).const_mul a
    have e3 := Real.hasDerivAt_cos x
    have e4 := ((Real.hasDerivAt_cos (a * x)).comp x ((hasDerivAt_id x).const_mul a)).const_mul a
    convert ((e1.add e2).sub e3).sub e4 using 1 <;> (try simp only [h2]) <;> fa_close1
  have z0 : (fun x => Real.sin (x + T) + Real.sin (a * (x + T)) - Real.sin x - Real.sin (a * x)) =
      fun _ => (0 : ℝ) := by
    funext x; have := hT x; linarith
  have z1 : ∀ x, h1 x = 0 := fun x => by
    have := d0 x; rw [z0] at this; exact this.unique (hasDerivAt_const x 0)
  have z1f : h1 = fun _ => (0 : ℝ) := funext z1
  have z2 : ∀ x, h2 x = 0 := fun x => by
    have := d1 x; rw [z1f] at this; exact this.unique (hasDerivAt_const x 0)
  -- h'' + h = (1 - a²) (sin (a (x + T)) - sin (a x)), so both pieces are periodic
  have pa : ∀ x, Real.sin (a * (x + T)) = Real.sin (a * x) := fun x => by
    have e0 := hT x
    have e2 := z2 x
    simp only [h2] at e2
    rw [ha2] at e2
    linarith
  have p1 : ∀ x, Real.sin (x + T) = Real.sin x := fun x => by
    have := hT x; rw [pa x] at this; linarith
  have c1 : Real.cos T = 1 := by
    have := p1 (π / 2); rw [Real.sin_add, Real.sin_pi_div_two, Real.cos_pi_div_two] at this; linarith
  have apos : 0 < a := Real.sqrt_pos.2 (by norm_num)
  have c2 : Real.cos (a * T) = 1 := by
    have := pa (π / 2 / a)
    rw [show a * (π / 2 / a + T) = π / 2 + a * T by field_simp, show a * (π / 2 / a) = π / 2 by field_simp,
      Real.sin_add, Real.sin_pi_div_two, Real.cos_pi_div_two] at this
    linarith
  obtain ⟨m, hm⟩ := (Real.cos_eq_one_iff T).1 c1
  obtain ⟨n, hn⟩ := (Real.cos_eq_one_iff (a * T)).1 c2
  by_contra hT0
  have hm0 : (m : ℝ) ≠ 0 := by
    intro h; apply hT0; rw [← hm, h, zero_mul]
  have hirr : Irrational a := by rw [ha]; exact irrational_sqrt_two
  apply hirr.ne_rat (n / m)
  have hπ : (0 : ℝ) < 2 * π := by positivity
  rw [← hm] at hn
  push_cast
  field_simp
  nlinarith [hn, hπ]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_sqrt_two_not_periodic
-- test: fa-sin-sqrt-two-not-periodic-test.sage

/-! ## exr-fs-cos-power -/

theorem cos_four (x : ℝ) :
    Real.cos x ^ 4 = 3 / 8 + 1 / 2 * Real.cos (2 * x) + 1 / 8 * Real.cos (4 * x) := by
  veriphy_certified
    ( try (rw [show ((2 * x : ℝ) : ℝ) = 2 * ((x : ℝ)) by ring])
      try (rw [show ((4 * x : ℝ) : ℝ) = 4 * ((x : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((x : ℝ))) = ((Real.cos x) ^ 2 - ((Real.sin x) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((x : ℝ)) = ((x : ℝ)) + ((x : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((x : ℝ))) = (2 * (Real.cos x) * (Real.sin x) : ℝ) := by
        rw [show (2 : ℝ) * ((x : ℝ)) = ((x : ℝ)) + ((x : ℝ)) by ring, Real.sin_add]; ring
      have hmc3 : Real.cos (3 * ((x : ℝ))) = ((Real.cos x) ^ 3 - (3 * (Real.cos x) * ((Real.sin x) ^ 2)) : ℝ) := by
        rw [show (3 : ℝ) * ((x : ℝ)) = 2 * ((x : ℝ)) + ((x : ℝ)) by ring, Real.cos_add, hmc2, hms2]; ring
      have hms3 : Real.sin (3 * ((x : ℝ))) = (3 * ((Real.cos x) ^ 2) * (Real.sin x) - ((Real.sin x) ^ 3) : ℝ) := by
        rw [show (3 : ℝ) * ((x : ℝ)) = 2 * ((x : ℝ)) + ((x : ℝ)) by ring, Real.sin_add, hmc2, hms2]; ring
      have hmc4 :
        Real.cos (4 * ((x : ℝ))) =
          ((Real.cos x) ^ 4 - (6 * ((Real.cos x) ^ 2) * ((Real.sin x) ^ 2)) + ((Real.sin x) ^ 4) : ℝ) :=
        by rw [show (4 : ℝ) * ((x : ℝ)) = 3 * ((x : ℝ)) + ((x : ℝ)) by ring, Real.cos_add, hmc3, hms3]; ring
      have hms4 :
        Real.sin (4 * ((x : ℝ))) =
          (4 * ((Real.cos x) ^ 3) * (Real.sin x) - (4 * (Real.cos x) * ((Real.sin x) ^ 3)) : ℝ) :=
        by rw [show (4 : ℝ) * ((x : ℝ)) = 3 * ((x : ℝ)) + ((x : ℝ)) by ring, Real.sin_add, hmc3, hms3]; ring
      simp only [hmc2, hms2, hmc3, hms3, hmc4, hms4]
      linear_combination
        ((7 * ((Real.cos x) ^ 2) / 8 - (((Real.sin x) ^ 2) / 8) + (3 / 8) : ℝ)) * Real.sin_sq_add_cos_sq ((x : ℝ)))
  -- certificate: fa-trig-poly.sage

theorem sin_cube (x : ℝ) : Real.sin x ^ 3 = 3 / 4 * Real.sin x - 1 / 4 * Real.sin (3 * x) := by
  veriphy_certified
    ((  rw [Real.sin_three_mul]
        all_goals ring1))
  -- certificate: fa-trig.sage

/-! ## exr-fs-periodic-integral -/

/-- The integral over a period does not depend on where the period starts. -/
theorem periodic_integral_shift (f : ℝ → ℝ) (hf : Function.Periodic f (2 * π)) (a : ℝ) :
    ∫ x in a..a + 2 * π, f x = ∫ x in (0 : ℝ)..2 * π, f x := by
  have := hf.intervalIntegral_add_eq a 0
  simpa using this

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.periodic_integral_shift
-- test: fa-periodic-integral-shift-test.sage

/-- `F(t) = ∫₀^{2π} f(x) g(x - t) dx` is `2π`-periodic in `t`. -/
theorem shifted_product_periodic (f g : ℝ → ℝ) (hg : Function.Periodic g (2 * π)) (t : ℝ) :
    ∫ x in (0 : ℝ)..2 * π, f x * g (x - (t + 2 * π)) = ∫ x in (0 : ℝ)..2 * π, f x * g (x - t) := by
  congr 1
  funext x
  rw [show x - (t + 2 * π) = (x - t) - 2 * π by ring, hg.sub_eq]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeriodicFunctions.shifted_product_periodic
-- test: fa-shifted-product-periodic-test.sage

end KaluzaKlein.FourierAnalysis.PeriodicFunctions

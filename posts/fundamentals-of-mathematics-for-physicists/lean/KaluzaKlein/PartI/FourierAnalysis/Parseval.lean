import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Convergence

/-! # Fourier analysis: parseval (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.Parseval

open Real Filter Topology

/-! ## thm-parseval -/

/-- Parseval's identity on an interval (mathlib): `Σ |c_n|² = (1/(b - a)) ∫_a^b |f|²`. -/
theorem parseval (a b : ℝ) (hab : a < b) (f : ℝ → ℂ)
    (hL2 : MeasureTheory.MemLp f 2 (MeasureTheory.volume.restrict (Set.Ioc a b))) :
    ∑' i : ℤ, ‖fourierCoeffOn hab f i‖ ^ 2 = (b - a)⁻¹ • ∫ x in a..b, ‖f x‖ ^ 2 :=
  tsum_sq_fourierCoeffOn hab hL2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.parseval
-- test: fa-parseval-test.sage

/-! ## exm-fs-parseval-basel, exr-fs-zeta4 -/

theorem x_sq_integral : ∫ x in (-π)..π, x ^ 2 = 2 * π ^ 3 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-π : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) ((fun x => x ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 3) / 3 : ℝ)) (f' :=
          fun x => x ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: fa-integral.sage

theorem x_four_integral : ∫ x in (-π)..π, x ^ 4 = 2 * π ^ 5 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (-π : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) ((fun x => x ^ 4) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (1 / 5 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => ((x_veriphy ^ 5) / 5 : ℝ)) (f' :=
          fun x => x ^ 4) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: fa-integral.sage

/-- Euler's sum: `Σ a / n² = a π² / 6`. -/
theorem basel (a : ℝ) : HasSum (fun n : ℕ => a / (n : ℝ) ^ 2) (a * π ^ 2 / 6) := by
  convert hasSum_zeta_two.mul_left a using 1
  · funext n; ring
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.basel
-- test: fa-basel-test.sage

/-- `Σ a / n⁴ = a π⁴ / 90`. -/
theorem zeta_four (a : ℝ) : HasSum (fun n : ℕ => a / (n : ℝ) ^ 4) (a * π ^ 4 / 90) := by
  convert hasSum_zeta_four.mul_left a using 1
  · funext n; ring
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.zeta_four
-- test: fa-zeta-four-test.sage

/-! ## exm-fs-parseval-abs -/

/-- `Σ_{k ≥ 0} a / (2k+1)⁴ = a π⁴ / 96`. -/
theorem odd_fourth_sum (a : ℝ) : HasSum (fun k : ℕ => a / (2 * k + 1) ^ 4) (a * π ^ 4 / 96) := by
  set f : ℕ → ℝ := fun n => 1 / (n : ℝ) ^ 4 with hf
  have h : HasSum f (π ^ 4 / 90) := hasSum_zeta_four
  have he : HasSum (fun k : ℕ => f (2 * k)) (π ^ 4 / 1440) := by
    have h4 := h.mul_left (1 / 16)
    have hfun : (fun k : ℕ => f (2 * k)) = fun k : ℕ => 1 / 16 * f k := by
      funext k
      simp only [hf]
      push_cast
      rw [mul_pow, one_div, one_div, one_div, mul_inv]
      norm_num
    rw [hfun]
    convert h4 using 1
    ring
  have hi : Function.Injective (fun k : ℕ => 2 * k + 1) := fun x y hxy => by
    simp only at hxy; omega
  have hs : Summable (fun k : ℕ => f (2 * k + 1)) := h.summable.comp_injective hi
  have hsum := HasSum.even_add_odd (f := f) he hs.hasSum
  have hu := h.unique hsum
  have hodd : HasSum (fun k : ℕ => f (2 * k + 1)) (π ^ 4 / 96) := by
    convert hs.hasSum using 1
    linarith
  convert hodd.mul_left a using 1
  · funext k
    simp only [hf]
    push_cast
    ring
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.odd_fourth_sum
-- test: fa-odd-fourth-sum-test.sage

/-- The even terms contribute a quarter: `Σ a / (2k)² = a π² / 24`. -/
theorem even_sq_sum (a : ℝ) : HasSum (fun k : ℕ => a / (2 * k : ℝ) ^ 2) (a * π ^ 2 / 24) := by
  convert hasSum_zeta_two.mul_left (a / 4) using 1
  · funext k; ring
  · ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.even_sq_sum
-- test: fa-even-sq-sum-test.sage

/-! ## exm-fs-bessel-inequality -/

theorem cos_six_integral : ∫ x in (0 : ℝ)..(2 * π), Real.cos x ^ 6 = 5 * π / 8 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 +
                        (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 16) -
                      (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 16) +
                    ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                  (5 * x_veriphy / 16) :
                ℝ))
            ((fun x => cos x ^ 6) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                        (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                    (-(1 / 6) : ℝ)).add
                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                      ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                  (3 / 16 : ℝ))).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (-(3 / 16) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / 2 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 / 16 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) + (((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) -
                      (13 * ((Real.cos x_veriphy) ^ 2) / 16) +
                    (3 * ((Real.sin x_veriphy) ^ 2) / 16) -
                  (5 / 16) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 +
                    (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 16) -
                  (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 16) +
                ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
              (5 * x_veriphy / 16) :
            ℝ))
          (f' := fun x => cos x ^ 6) hF (by apply Continuous.intervalIntegrable; fun_prop)]
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

/-- The mean-square error of `S₁ f = (3/4) cos x` for `f = cos³ x`: `(1/2π) ∫ |f - S₁ f|² = 1/32`. -/
theorem cube_s1_error : ∫ x in (0 : ℝ)..(2 * π), (Real.cos x ^ 3 - 3 / 4 * Real.cos x) ^ 2 = π / 16 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 +
                    ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 32) +
                  (x_veriphy / 32) :
                ℝ))
            ((fun x => (cos x ^ 3 - 3 / 4 * cos x) ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                            (-(1 / 6) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / 32 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 32 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) + (((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) +
                    (((Real.cos x_veriphy) ^ 2) / 2) -
                  (1 / 32) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 +
                ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 32) +
              (x_veriphy / 32) :
            ℝ))
          (f' := fun x => (cos x ^ 3 - 3 / 4 * cos x) ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
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

/-- Any other `t cos x` has error `1/32 + (t - 3/4)² / 2` (times `2π`). -/
theorem cube_t_error (t : ℝ) :
    ∫ x in (0 : ℝ)..(2 * π), (Real.cos x ^ 3 - t * Real.cos x) ^ 2 = π / 16 + π * (t - 3 / 4) ^ 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 -
                                    (t * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 4) +
                                  (t * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 4) +
                                ((t ^ 2) * (Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                              (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 16) -
                            (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 16) +
                          ((t ^ 2) * x_veriphy / 2) -
                        (t * (Real.cos x_veriphy) * (Real.sin x_veriphy)) -
                      (3 * t * x_veriphy / 4) +
                    ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                  (5 * x_veriphy / 16) :
                ℝ))
            ((fun x => (cos x ^ 3 - t * cos x) ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                                                (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                  3)).const_mul
                                                            (-(1 / 6) : ℝ)).add
                                                        (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                                              ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                                          (-t / 4 : ℝ))).add
                                                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                                      (t / 4 : ℝ))).add
                                                ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                                      ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                                  ((t ^ 2) / 2 : ℝ))).add
                                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                                  ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                              (3 / 16 : ℝ))).add
                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                          (-(3 / 16) : ℝ))).add
                                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul ((t ^ 2) / 2 : ℝ))).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                      ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                  (-t : ℝ))).add
                            ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (-3 * t / 4 : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / 2 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (5 / 16 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) + (((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) +
                              (7 * t * ((Real.cos x_veriphy) ^ 2) / 4) -
                            (t * ((Real.sin x_veriphy) ^ 2) / 4) -
                          ((t ^ 2) / 2) -
                        (13 * ((Real.cos x_veriphy) ^ 2) / 16) +
                      (3 * ((Real.sin x_veriphy) ^ 2) / 16) +
                    (3 * t / 4) -
                  (5 / 16) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / 6 -
                                (t * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 4) +
                              (t * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 4) +
                            ((t ^ 2) * (Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
                          (3 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 16) -
                        (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 16) +
                      ((t ^ 2) * x_veriphy / 2) -
                    (t * (Real.cos x_veriphy) * (Real.sin x_veriphy)) -
                  (3 * t * x_veriphy / 4) +
                ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2) +
              (5 * x_veriphy / 16) :
            ℝ))
          (f' := fun x => (cos x ^ 3 - t * cos x) ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
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

/-! ## exr-fs-best-approximation -/

/-- The mean-square error of a finite combination of orthonormal modes:
`‖f - Σ tᵢ eᵢ‖² = ‖f‖² - Σ cᵢ² + Σ (cᵢ - tᵢ)²` with `cᵢ = ⟪eᵢ, f⟫`. -/
theorem best_approx {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (e : ι → E) (he : Orthonormal ℝ e) (f : E) (t : ι → ℝ) :
    ‖f - ∑ i ∈ s, t i • e i‖ ^ 2 =
      ‖f‖ ^ 2 - ∑ i ∈ s, (inner ℝ (e i) f) ^ 2 + ∑ i ∈ s, (inner ℝ (e i) f - t i) ^ 2 := by
  rw [norm_sub_sq_real, inner_sum]
  have hn : ‖∑ i ∈ s, t i • e i‖ ^ 2 = ∑ i ∈ s, t i ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    simp_rw [inner_sum, inner_smul_left, inner_smul_right]
    simp only [RCLike.conj_to_real]
    have : ∀ i ∈ s, ∑ j ∈ s, t i * (t j * inner ℝ (e i) (e j)) = t i ^ 2 := by
      intro i hi
      rw [Finset.sum_eq_single i]
      · rw [real_inner_self_eq_norm_sq, he.1 i]; ring
      · intro j _ hji; rw [he.2 hji.symm]; ring
      · intro h; exact absurd hi h
    rw [Finset.sum_congr rfl this]
  rw [hn]
  simp_rw [inner_smul_right, real_inner_comm f]
  have expand : ∑ i ∈ s, (inner ℝ f (e i) - t i) ^ 2 =
      ∑ i ∈ s, (inner ℝ f (e i)) ^ 2 - 2 * ∑ i ∈ s, t i * inner ℝ f (e i) + ∑ i ∈ s, t i ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl; intro i _; ring
  rw [expand]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.best_approx
-- test: fa-best-approx-test.sage

/-- Bessel's inequality: `Σ ⟪eᵢ, f⟫² ≤ ‖f‖²`. -/
theorem bessel {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (e : ι → E) (he : Orthonormal ℝ e) (f : E) :
    ∑ i ∈ s, (inner ℝ (e i) f) ^ 2 ≤ ‖f‖ ^ 2 := by
  have h := best_approx s e he f (fun i => inner ℝ (e i) f)
  simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Finset.sum_const_zero,
    add_zero] at h
  nlinarith [sq_nonneg ‖f - ∑ i ∈ s, inner ℝ (e i) f • e i‖]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.bessel
-- test: fa-bessel-test.sage

/-! ## exr-fs-zeta4 -/

/-- The cosine coefficients of `x²`: `(1/π) ∫_{-π}^{π} x² cos n x dx = 4 (-1)^n / n²`. -/
theorem x_sq_an (n : ℕ) (hn : n ≠ 0) :
    1 / π * ∫ x in (-π)..π, x ^ 2 * Real.cos (n * x) = 4 * (-1) ^ n / n ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hF : ∀ x ∈ Set.uIcc (-π) π, HasDerivAt
      (fun x => x ^ 2 * Real.sin (n * x) / n + 2 * x * Real.cos (n * x) / n ^ 2 - 2 * Real.sin (n * x) / n ^ 3)
      (x ^ 2 * Real.cos (n * x)) x := by
    intro x _
    have hs : HasDerivAt (fun x => Real.sin (n * x)) (Real.cos (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).sin
    have hc : HasDerivAt (fun x => Real.cos (n * x)) (-Real.sin (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).cos
    have e1 := (((hasDerivAt_pow 2 x).mul hs).div_const (n : ℝ))
    have e2 := ((((hasDerivAt_id x).const_mul 2).mul hc).div_const ((n : ℝ) ^ 2))
    have e3 := ((hs.const_mul 2).div_const ((n : ℝ) ^ 3))
    convert (e1.add e2).sub e3 using 1
    all_goals fa_close1b
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_neg, Real.sin_neg, Real.cos_neg, Real.sin_nat_mul_pi, Real.cos_nat_mul_pi]
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.Parseval.x_sq_an
-- test: fa-x-sq-an-test.sage

end KaluzaKlein.FourierAnalysis.Parseval

import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.SphereLaplacian

/-! # Fourier analysis: spherical harmonics (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.SphericalHarmonics

open Real MeasureTheory Filter Topology

/-! ## def-spherical-harmonic, exm-sph-low-harmonics, exr-sph-orthonormality, exr-sph-l3-check -/

/-- `∫₀^π Θ_{00} Θ_{00} sin θ dθ = 2`. -/
theorem theta_int_00_0 : ∫ t in (0 : ℝ)..π, (Real.sin t) = 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) ((fun t => sin t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; ring1
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-(Real.cos x_veriphy) : ℝ)) (f' :=
          fun t => sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{00} Θ_{10} sin θ dθ = 0`. -/
theorem theta_int_01_0 : ∫ t in (0 : ℝ)..π, (Real.sin t * Real.cos t) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 2) / 2 : ℝ)) ((fun t => sin t * cos t) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).const_mul (-(1 / 2) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-((Real.cos x_veriphy) ^ 2) / 2 : ℝ))
          (f' := fun t => sin t * cos t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{00} Θ_{20} sin θ dθ = 0`. -/
theorem theta_int_02_0 : ∫ t in (0 : ℝ)..π, (3 / 2 * Real.sin t * Real.cos t ^ 2 - 1 / 2 * Real.sin t) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 3) / 2 + ((Real.cos x_veriphy) / 2) : ℝ))
            ((fun t => 3 / 2 * sin t * cos t ^ 2 - 1 / 2 * sin t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (-(1 / 2) : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) / 2 + ((Real.cos x_veriphy) / 2) : ℝ)) (f' := fun t =>
          3 / 2 * sin t * cos t ^ 2 - 1 / 2 * sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{10} Θ_{10} sin θ dθ = 2/3`. -/
theorem theta_int_11_0 : ∫ t in (0 : ℝ)..π, (Real.sin t * Real.cos t ^ 2) = 2/3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 3) / 3 : ℝ)) ((fun t => sin t * cos t ^ 2) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (-(1 / 3) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-((Real.cos x_veriphy) ^ 3) / 3 : ℝ))
          (f' := fun t => sin t * cos t ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{10} Θ_{20} sin θ dθ = 0`. -/
theorem theta_int_12_0 : ∫ t in (0 : ℝ)..π, (3 / 2 * Real.sin t * Real.cos t ^ 3 - 1 / 2 * Real.sin t * Real.cos t) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-3 * ((Real.cos x_veriphy) ^ 4) / 8 + (((Real.cos x_veriphy) ^ 2) / 4) : ℝ))
            ((fun t => 3 / 2 * sin t * cos t ^ 3 - 1 / 2 * sin t * cos t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).const_mul (-(3 / 8) : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).const_mul (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-3 * ((Real.cos x_veriphy) ^ 4) / 8 + (((Real.cos x_veriphy) ^ 2) / 4) : ℝ)) (f' := fun t =>
          3 / 2 * sin t * cos t ^ 3 - 1 / 2 * sin t * cos t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{11} Θ_{11} sin θ dθ = 4/3`. -/
theorem theta_int_11_1 : ∫ t in (0 : ℝ)..π, (Real.sin t ^ 3) = 4/3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (((Real.cos x_veriphy) ^ 3) / 3 - (Real.cos x_veriphy) : ℝ))
            ((fun t => sin t ^ 3) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (1 / 3 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination ((-(Real.sin x_veriphy) : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 3) / 3 - (Real.cos x_veriphy) : ℝ)) (f' := fun t => sin t ^ 3) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{11} Θ_{21} sin θ dθ = 0`. -/
theorem theta_int_12_1 : ∫ t in (0 : ℝ)..π, (3 * Real.sin t ^ 3 * Real.cos t) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (3 * ((Real.sin x_veriphy) ^ 4) / 4 : ℝ))
            ((fun t => 3 * sin t ^ 3 * cos t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 4).const_mul (3 / 4 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (3 * ((Real.sin x_veriphy) ^ 4) / 4 : ℝ)) (f' := fun t => 3 * sin t ^ 3 * cos t) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{20} Θ_{20} sin θ dθ = 2/5`. -/
theorem theta_int_22_0 : ∫ t in (0 : ℝ)..π, (9 / 4 * Real.sin t * Real.cos t ^ 4 - 3 / 2 * Real.sin t * Real.cos t ^ 2 + 1 / 4 * Real.sin t) = 2/5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-9 * ((Real.cos x_veriphy) ^ 5) / 20 + (((Real.cos x_veriphy) ^ 3) / 2) - ((Real.cos x_veriphy) / 4) :
                ℝ))
            ((fun t => 9 / 4 * sin t * cos t ^ 4 - 3 / 2 * sin t * cos t ^ 2 + 1 / 4 * sin t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(9 / 20) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (1 / 2 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(1 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-9 * ((Real.cos x_veriphy) ^ 5) / 20 + (((Real.cos x_veriphy) ^ 3) / 2) - ((Real.cos x_veriphy) / 4) : ℝ))
          (f' := fun t => 9 / 4 * sin t * cos t ^ 4 - 3 / 2 * sin t * cos t ^ 2 + 1 / 4 * sin t) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{21} Θ_{21} sin θ dθ = 12/5`. -/
theorem theta_int_22_1 : ∫ t in (0 : ℝ)..π, (9 * Real.sin t ^ 3 * Real.cos t ^ 2) = 12/5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (9 * ((Real.cos x_veriphy) ^ 5) / 5 - (3 * ((Real.cos x_veriphy) ^ 3)) : ℝ))
            ((fun t => 9 * sin t ^ 3 * cos t ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (9 / 5 : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (-3 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-9 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (9 * ((Real.cos x_veriphy) ^ 5) / 5 - (3 * ((Real.cos x_veriphy) ^ 3)) : ℝ)) (f' := fun t =>
          9 * sin t ^ 3 * cos t ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{22} Θ_{22} sin θ dθ = 48/5`. -/
theorem theta_int_22_2 : ∫ t in (0 : ℝ)..π, (9 * Real.sin t ^ 5) = 48/5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-9 * ((Real.cos x_veriphy) ^ 5) / 5 + (6 * ((Real.cos x_veriphy) ^ 3)) - (9 * (Real.cos x_veriphy)) : ℝ))
            ((fun t => 9 * sin t ^ 5) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(9 / 5) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (6 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-9 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((9 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - (9 * ((Real.sin x_veriphy) ^ 3)) -
                  (9 * (Real.sin x_veriphy)) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-9 * ((Real.cos x_veriphy) ^ 5) / 5 + (6 * ((Real.cos x_veriphy) ^ 3)) - (9 * (Real.cos x_veriphy)) : ℝ))
          (f' := fun t => 9 * sin t ^ 5) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{30} Θ_{30} sin θ dθ = 2/7`. -/
theorem theta_int_33_0 : ∫ t in (0 : ℝ)..π, (25 / 4 * Real.sin t * Real.cos t ^ 6 - 15 / 2 * Real.sin t * Real.cos t ^ 4 + 9 / 4 * Real.sin t * Real.cos t ^ 2) = 2/7 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-25 * ((Real.cos x_veriphy) ^ 7) / 28 + (3 * ((Real.cos x_veriphy) ^ 5) / 2) -
                  (3 * ((Real.cos x_veriphy) ^ 3) / 4) :
                ℝ))
            ((fun t => 25 / 4 * sin t * cos t ^ 6 - 15 / 2 * sin t * cos t ^ 4 + 9 / 4 * sin t * cos t ^ 2) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 7).const_mul (-(25 / 28) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (3 / 2 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (-(3 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-25 * ((Real.cos x_veriphy) ^ 7) / 28 + (3 * ((Real.cos x_veriphy) ^ 5) / 2) -
              (3 * ((Real.cos x_veriphy) ^ 3) / 4) :
            ℝ))
          (f' := fun t => 25 / 4 * sin t * cos t ^ 6 - 15 / 2 * sin t * cos t ^ 4 + 9 / 4 * sin t * cos t ^ 2) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{31} Θ_{31} sin θ dθ = 24/7`. -/
theorem theta_int_33_1 : ∫ t in (0 : ℝ)..π, (225 / 4 * Real.sin t ^ 3 * Real.cos t ^ 4 - 45 / 2 * Real.sin t ^ 3 * Real.cos t ^ 2 + 9 / 4 * Real.sin t ^ 3) = 24/7 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (225 * ((Real.cos x_veriphy) ^ 7) / 28 - (63 * ((Real.cos x_veriphy) ^ 5) / 4) +
                    (33 * ((Real.cos x_veriphy) ^ 3) / 4) -
                  (9 * (Real.cos x_veriphy) / 4) :
                ℝ))
            ((fun t => 225 / 4 * sin t ^ 3 * cos t ^ 4 - 45 / 2 * sin t ^ 3 * cos t ^ 2 + 9 / 4 * sin t ^ 3) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 7).const_mul (225 / 28 : ℝ)).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(63 / 4) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (33 / 4 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(9 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-225 * ((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / 4 +
                    (45 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / 2) -
                  (9 * (Real.sin x_veriphy) / 4) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (225 * ((Real.cos x_veriphy) ^ 7) / 28 - (63 * ((Real.cos x_veriphy) ^ 5) / 4) +
                (33 * ((Real.cos x_veriphy) ^ 3) / 4) -
              (9 * (Real.cos x_veriphy) / 4) :
            ℝ))
          (f' := fun t => 225 / 4 * sin t ^ 3 * cos t ^ 4 - 45 / 2 * sin t ^ 3 * cos t ^ 2 + 9 / 4 * sin t ^ 3) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{32} Θ_{32} sin θ dθ = 240/7`. -/
theorem theta_int_33_2 : ∫ t in (0 : ℝ)..π, (225 * Real.sin t ^ 5 * Real.cos t ^ 2) = 240/7 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-225 * ((Real.cos x_veriphy) ^ 7) / 7 + (90 * ((Real.cos x_veriphy) ^ 5)) -
                  (75 * ((Real.cos x_veriphy) ^ 3)) :
                ℝ))
            ((fun t => 225 * sin t ^ 5 * cos t ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 7).const_mul (-(225 / 7) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (90 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (-75 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((225 * ((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) -
                    (225 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3)) -
                  (225 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy)) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-225 * ((Real.cos x_veriphy) ^ 7) / 7 + (90 * ((Real.cos x_veriphy) ^ 5)) -
              (75 * ((Real.cos x_veriphy) ^ 3)) :
            ℝ))
          (f' := fun t => 225 * sin t ^ 5 * cos t ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `∫₀^π Θ_{33} Θ_{33} sin θ dθ = 1440/7`. -/
theorem theta_int_33_3 : ∫ t in (0 : ℝ)..π, (225 * Real.sin t ^ 7) = 1440/7 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (225 * ((Real.cos x_veriphy) ^ 7) / 7 - (135 * ((Real.cos x_veriphy) ^ 5)) +
                    (225 * ((Real.cos x_veriphy) ^ 3)) -
                  (225 * (Real.cos x_veriphy)) :
                ℝ))
            ((fun t => 225 * sin t ^ 7) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 7).const_mul (225 / 7 : ℝ)).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-135 : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (225 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-225 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-225 * ((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) +
                          (225 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3)) -
                        (225 * ((Real.sin x_veriphy) ^ 5)) +
                      (450 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy)) -
                    (225 * ((Real.sin x_veriphy) ^ 3)) -
                  (225 * (Real.sin x_veriphy)) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (225 * ((Real.cos x_veriphy) ^ 7) / 7 - (135 * ((Real.cos x_veriphy) ^ 5)) +
                (225 * ((Real.cos x_veriphy) ^ 3)) -
              (225 * (Real.cos x_veriphy)) :
            ℝ))
          (f' := fun t => 225 * sin t ^ 7) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- The normalisation: `N_{lm}² · 2π · ∫ Θ_{lm}² sin θ = 1` with `N_{lm}² = (2l+1)(l-m)!/(4π (l+m)!)`, from the
Legendre integral `∫ Θ_{lm}² sin θ = (2/(2l+1)) (l+m)!/(l-m)!`. -/
theorem normalization (l m : ℕ) (hm : m ≤ l) :
    (2 * l + 1 : ℝ) / (4 * π) * ((l - m).factorial / (l + m).factorial) * (2 * π) *
      (2 / (2 * l + 1) * ((l + m).factorial / (l - m).factorial)) = 1 := by
  have h1 : (0 : ℝ) < (l - m).factorial := by exact_mod_cast Nat.factorial_pos _
  have h2 : (0 : ℝ) < (l + m).factorial := by exact_mod_cast Nat.factorial_pos _
  have hπ : π ≠ 0 := Real.pi_ne_zero
  field_simp
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization (0) (0) (le_refl 0) := by
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization (2) (1) ((by norm_num)) := by
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization (3) (3) (le_refl 3) := by
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  norm_num [Nat.factorial]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization (3) (0) ((by norm_num)) := by
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  norm_num [Nat.factorial]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization
-- test: fa-normalization-test.sage

/-- The Condon-Shortley phase for `l = 1`: `Y_{1,-1} = (-1) conj(Y_{11})` with `Y_{1,±1} = ∓√(3/8π) sin θ e^{±iφ}`. -/
theorem condon_shortley (θ φ : ℝ) :
    (Real.sqrt (3 / (8 * π)) * Real.sin θ : ℂ) * Complex.exp (-(φ : ℂ) * Complex.I) =
      -(starRingEnd ℂ) (-(Real.sqrt (3 / (8 * π)) * Real.sin θ : ℂ) * Complex.exp ((φ : ℂ) * Complex.I)) := by
  simp only [map_mul, map_neg, Complex.conj_ofReal, ← Complex.exp_conj, Complex.conj_I]
  ring_nf

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphericalHarmonics.condon_shortley
-- test: fa-condon-shortley-test.sage

/-! ## exm-sph-expansion, exr-sph-expansion -/

/-- `cos² θ = 1/3 + (2/3) P₂(cos θ)` with `P₂(u) = (3u² - 1)/2`. -/
theorem cos_sq_legendre (θ : ℝ) : Real.cos θ ^ 2 = 1 / 3 + 2 / 3 * (3 / 2 * Real.cos θ ^ 2 - 1 / 2) := by
  veriphy_certified (ring1)
  -- certificate: fa-identity.sage

/-- Parseval for `cos² θ`: `∫ |cos² θ|² dΩ = 2π ∫₀^π cos⁴ θ sin θ dθ`, with `θ`-integral `2/5`. -/
theorem cos_four_theta : ∫ t in (0 : ℝ)..π, Real.cos t ^ 4 * Real.sin t = 2 / 5 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 5) / 5 : ℝ)) ((fun t => cos t ^ 4 * sin t) x_veriphy)
            x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(1 / 5) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (-((Real.cos x_veriphy) ^ 5) / 5 : ℝ))
          (f' := fun t => cos t ^ 4 * sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `|f₀₀|² + |f₂₀|² = 4π/9 + 16π/45 = 4π/5 = 2π · (2/5) = ∫ |cos² θ|² dΩ`. -/
theorem cos_sq_parseval : 4 * π / 9 + 16 * π / 45 = 2 * π * (2 / 5) := by
  veriphy_certified (ring1)
  -- certificate: fa-identity.sage

/-- `sin² θ cos 2φ = √(8π/15) (Y₂₂ + Y₂,₋₂)` with `Y₂,±₂ = √(15/32π) sin² θ e^{±2iφ}`. -/
theorem sin_sq_cos_two_phi (θ φ : ℝ) :
    ((Real.sin θ ^ 2 * Real.cos (2 * φ) : ℝ) : ℂ) =
      (Real.sqrt (8 * π / 15) : ℂ) * ((Real.sqrt (15 / (32 * π)) * Real.sin θ ^ 2 : ℂ) * Complex.exp (2 * φ * Complex.I) +
        (Real.sqrt (15 / (32 * π)) * Real.sin θ ^ 2 : ℂ) * Complex.exp (-(2 * φ) * Complex.I)) := by
  have hs : Real.sqrt (8 * π / 15) * Real.sqrt (15 / (32 * π)) = 1 / 2 := by
    rw [← Real.sqrt_mul (by positivity), show 8 * π / 15 * (15 / (32 * π)) = (1 / 2) ^ 2 by first | (field_simp; done) | (field_simp; ring),
      Real.sqrt_sq (by norm_num)]
  have he : Complex.exp (2 * φ * Complex.I) + Complex.exp (-(2 * φ) * Complex.I) = 2 * Real.cos (2 * φ) := by
    rw [Complex.ofReal_cos, Complex.cos]; push_cast; ring_nf
  have hs' : (Real.sqrt (8 * π / 15) : ℂ) * (Real.sqrt (15 / (32 * π)) : ℂ) = 1 / 2 := by
    rw [← Complex.ofReal_mul, hs]; norm_num
  rw [show (Real.sqrt (8 * π / 15) : ℂ) * ((Real.sqrt (15 / (32 * π)) * Real.sin θ ^ 2 : ℂ) * Complex.exp (2 * φ * Complex.I) +
      (Real.sqrt (15 / (32 * π)) * Real.sin θ ^ 2 : ℂ) * Complex.exp (-(2 * φ) * Complex.I)) =
      ((Real.sqrt (8 * π / 15) : ℂ) * (Real.sqrt (15 / (32 * π)) : ℂ)) * (Real.sin θ ^ 2 : ℂ) *
        (Complex.exp (2 * φ * Complex.I) + Complex.exp (-(2 * φ) * Complex.I)) by push_cast; ring, hs', he]
  push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphericalHarmonics.sin_sq_cos_two_phi
-- test: fa-sin-sq-cos-two-phi-test.sage

/-! ## exm-sph-addition-theorem -/

/-- The addition theorem for `l = 1`: `Σ_m Y_{1m}(n) conj(Y_{1m}(n')) = (3/4π) cos γ`. -/
theorem addition_l1 (θ φ θ' φ' : ℝ) :
    ((3 / (4 * π) : ℝ) : ℂ) * (Real.cos θ * Real.cos θ') +
      ((3 / (8 * π) : ℝ) : ℂ) * (Real.sin θ * Real.sin θ') *
        (Complex.exp ((φ - φ') * Complex.I) + Complex.exp (-((φ - φ') * Complex.I))) =
      ((3 / (4 * π) * (Real.cos θ * Real.cos θ' + Real.sin θ * Real.sin θ' * Real.cos (φ - φ')) : ℝ) : ℂ) := by
  have he : Complex.exp ((φ - φ') * Complex.I) + Complex.exp (-((φ - φ') * Complex.I)) = 2 * Real.cos (φ - φ') := by
    rw [Complex.ofReal_cos, Complex.cos]; push_cast; ring_nf
  rw [he]; push_cast; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphericalHarmonics.addition_l1
-- test: fa-addition-l1-test.sage

/-- On the diagonal `θ = θ'`, `φ = φ'`: `Σ_m |Y_{1m}|² = 3/(4π)`. -/
theorem addition_l1_diag (θ : ℝ) :
    3 / (4 * π) * (Real.cos θ * Real.cos θ + Real.sin θ * Real.sin θ * Real.cos 0) = 3 / (4 * π) := by
  rw [Real.cos_zero, mul_one, ← sq, ← sq, Real.cos_sq_add_sin_sq, mul_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SphericalHarmonics.addition_l1_diag
-- test: fa-addition-l1-diag-test.sage

/-! ## exr-sph-parseval -/

theorem sin_cube_theta : ∫ t in (0 : ℝ)..π, Real.sin t ^ 3 = 4 / 3 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (((Real.cos x_veriphy) ^ 3) / 3 - (Real.cos x_veriphy) : ℝ))
            ((fun t => sin t ^ 3) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (1 / 3 : ℝ)).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination ((-(Real.sin x_veriphy) : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 3) / 3 - (Real.cos x_veriphy) : ℝ)) (f' := fun t => sin t ^ 3) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem sin_fifth_theta : ∫ t in (0 : ℝ)..π, Real.sin t ^ 5 = 16 / 15 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 5) / 5 + (2 * ((Real.cos x_veriphy) ^ 3) / 3) - (Real.cos x_veriphy) : ℝ))
            ((fun t => sin t ^ 5) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).const_mul (-(1 / 5) : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (2 / 3 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - ((Real.sin x_veriphy) ^ 3) - (Real.sin x_veriphy) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 5) / 5 + (2 * ((Real.cos x_veriphy) ^ 3) / 3) - (Real.cos x_veriphy) : ℝ)) (f' :=
          fun t => sin t ^ 5) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem cos_sq_phi : ∫ p in (0 : ℝ)..(2 * π), Real.cos p ^ 2 = π := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt (fun x_veriphy => ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2 + (x_veriphy / 2) : ℝ))
            ((fun p => cos p ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                        (1 / 2 : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination ((-(1 / 2) : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((Real.cos x_veriphy) * (Real.sin x_veriphy) / 2 + (x_veriphy / 2) : ℝ)) (f' := fun p => cos p ^ 2) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem cos_two_sq_phi : ∫ p in (0 : ℝ)..(2 * π), Real.cos (2 * p) ^ 2 = π := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 2 -
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 2) +
                  (x_veriphy / 2) :
                ℝ))
            ((fun p => cos (2 * p) ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                            (1 / 2 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                          (-(1 / 2) : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 2) / 2 - (((Real.sin x_veriphy) ^ 2) / 2) - (1 / 2) : ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / 2 -
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / 2) +
              (x_veriphy / 2) :
            ℝ))
          (f' := fun p => cos (2 * p) ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem cos_cos_two_phi : ∫ p in (0 : ℝ)..(2 * π), Real.cos p * Real.cos (2 * p) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / 2 - (((Real.sin x_veriphy) ^ 3) / 6) +
                  ((Real.sin x_veriphy) / 2) :
                ℝ))
            ((fun p => cos p * cos (2 * p)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                            (1 / 2 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (-(1 / 6) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          linear_combination ((-(Real.cos x_veriphy) / 2 : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / 2 - (((Real.sin x_veriphy) ^ 3) / 6) +
              ((Real.sin x_veriphy) / 2) :
            ℝ))
          (f' := fun p => cos p * cos (2 * p)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem cos_phi_int : ∫ p in (0 : ℝ)..(2 * π), Real.cos p = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt (fun x_veriphy => (Real.sin x_veriphy : ℝ)) ((fun p => cos p) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).sin)).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (Real.sin x_veriphy : ℝ)) (f' :=
          fun p => cos p) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Parseval on the sphere for `cos θ + sin θ cos φ + sin² θ cos 2φ`: `4π/3 + 4π/3 + 16π/15 = Σ |f_{lm}|²` with
`|f₁₀|² = 4π/3`, `|f₁,±₁|² = 2π/3`, `|f₂,±₂|² = 8π/15`. -/
theorem sphere_parseval :
    2 * π * (2 / 3) + π * (4 / 3) + π * (16 / 15) = 4 * π / 3 + 2 * (2 * π / 3) + 2 * (8 * π / 15) := by
  veriphy_certified (ring1)
  -- certificate: fa-identity.sage

end KaluzaKlein.FourierAnalysis.SphericalHarmonics

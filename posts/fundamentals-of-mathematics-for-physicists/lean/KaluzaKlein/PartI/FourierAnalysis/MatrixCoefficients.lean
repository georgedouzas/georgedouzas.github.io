import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.SphericalHarmonics

/-! # Fourier analysis: matrix coefficients (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.MatrixCoefficients

open Real MeasureTheory Filter Topology Matrix

/-! ## def-matrix-coefficient, exr-sph-haar -/

theorem sin_theta_int : ∫ t in (0 : ℝ)..π, Real.sin t = 2 := by
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

/-- The Haar measure `sin θ dθ dφ dψ / 16π²` has total mass `1`: `2 · 2π · 4π = 16π²`. -/
theorem haar_mass : 2 * (2 * π) * (4 * π) / (16 * π ^ 2) = 1 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: fa-identity.sage

/-- The round metric on `S³` in Euler angles, `(1/4)(dθ² + dφ² + dψ² + 2 cos θ dφ dψ)`, has determinant
`sin² θ / 64`: the volume form is `(1/8) sin θ dθ dφ dψ`. -/
theorem s3_metric_det (θ : ℝ) :
    Matrix.det !![(1 : ℝ) / 4, 0, 0; 0, 1 / 4, Real.cos θ / 4; 0, Real.cos θ / 4, 1 / 4] = Real.sin θ ^ 2 / 64 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals linear_combination ((-(1 / 64) : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)))
  -- certificate: fa-matrix.sage

/-- The characters as Dirichlet kernels: `χ_{1/2} sin(θ/2) = sin θ`, `χ₁ sin(θ/2) = sin(3θ/2)`,
`χ_{3/2} sin(θ/2) = sin 2θ` (with `t = θ/2`). -/
theorem character_half (t : ℝ) : 2 * Real.cos t * Real.sin t = Real.sin (2 * t) := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      simp only [hmc2, hms2]
      all_goals ring1)
  -- certificate: fa-trig-poly.sage

theorem character_one (t : ℝ) : (1 + 2 * Real.cos (2 * t)) * Real.sin t = Real.sin (3 * t) := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      try (rw [show ((3 * t : ℝ) : ℝ) = 3 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      have hmc3 : Real.cos (3 * ((t : ℝ))) = ((Real.cos t) ^ 3 - (3 * (Real.cos t) * ((Real.sin t) ^ 2)) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add, hmc2, hms2]; ring
      have hms3 : Real.sin (3 * ((t : ℝ))) = (3 * ((Real.cos t) ^ 2) * (Real.sin t) - ((Real.sin t) ^ 3) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add, hmc2, hms2]; ring
      simp only [hmc2, hms2, hmc3, hms3]
      linear_combination ((-(Real.sin t) : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

theorem character_three_halves (t : ℝ) : (2 * Real.cos t + 2 * Real.cos (3 * t)) * Real.sin t = Real.sin (4 * t) := by
  veriphy_certified
    ( try (rw [show ((3 * t : ℝ) : ℝ) = 3 * ((t : ℝ)) by ring])
      try (rw [show ((4 * t : ℝ) : ℝ) = 4 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      have hmc3 : Real.cos (3 * ((t : ℝ))) = ((Real.cos t) ^ 3 - (3 * (Real.cos t) * ((Real.sin t) ^ 2)) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add, hmc2, hms2]; ring
      have hms3 : Real.sin (3 * ((t : ℝ))) = (3 * ((Real.cos t) ^ 2) * (Real.sin t) - ((Real.sin t) ^ 3) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add, hmc2, hms2]; ring
      have hmc4 :
        Real.cos (4 * ((t : ℝ))) =
          ((Real.cos t) ^ 4 - (6 * ((Real.cos t) ^ 2) * ((Real.sin t) ^ 2)) + ((Real.sin t) ^ 4) : ℝ) :=
        by rw [show (4 : ℝ) * ((t : ℝ)) = 3 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add, hmc3, hms3]; ring
      have hms4 :
        Real.sin (4 * ((t : ℝ))) =
          (4 * ((Real.cos t) ^ 3) * (Real.sin t) - (4 * (Real.cos t) * ((Real.sin t) ^ 3)) : ℝ) :=
        by rw [show (4 : ℝ) * ((t : ℝ)) = 3 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add, hmc3, hms3]; ring
      simp only [hmc2, hms2, hmc3, hms3, hmc4, hms4]
      linear_combination ((-2 * (Real.cos t) * (Real.sin t) : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

/-! ## exm-sph-u1-coefficients, exr-sph-schur -/

/-- The phases over the doubled range: `∫₀^{4π} e^{inψ/2} dψ = 4π δ_{n0}`: the `ψ` and `φ` integrations kill every
term whose phases do not match, including the mixed spin-`1/2` × spin-`1` terms (odd `n`). -/
theorem half_phase_orthogonality (n : ℤ) :
    ∫ ψ in (0 : ℝ)..(4 * π), Complex.exp (n * ψ / 2 * Complex.I) = if n = 0 then 4 * π else 0 := by
  split_ifs with h
  · subst h; simp
  · have hc : (n / 2 * Complex.I : ℂ) ≠ 0 := by
      apply mul_ne_zero (div_ne_zero (by exact_mod_cast h) two_ne_zero) Complex.I_ne_zero
    have := integral_exp_mul_complex (a := 0) (b := 4 * π) hc
    rw [show (fun ψ : ℝ => Complex.exp (n * ψ / 2 * Complex.I)) = fun ψ : ℝ => Complex.exp (n / 2 * Complex.I * ψ) by
      funext ψ; congr 1; ring, this]
    have h1 : Complex.exp (n / 2 * Complex.I * ((4 * π : ℝ) : ℂ)) = 1 := by
      rw [show (n : ℂ) / 2 * Complex.I * ((4 * π : ℝ) : ℂ) = n * (2 * π * Complex.I) by push_cast; ring]
      exact Complex.exp_int_mul_two_pi_mul_I n
    rw [h1]; simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.MatrixCoefficients.half_phase_orthogonality
-- test: fa-half-phase-orthogonality-test.sage

/-! ## exm-sph-su2-half -/

/-- `D^{1/2}` has determinant `|a|² + |b|² = cos²(θ/2) + sin²(θ/2) = 1` (with `t = θ/2` and unimodular phases
`u = e^{-i(φ+ψ)/2}`, `v = e^{i(φ-ψ)/2}`). -/
theorem d_half_det (t φ ψ : ℝ) :
    (Complex.exp (-((φ + ψ) / 2) * Complex.I) * Real.cos t) * (Complex.exp (((φ + ψ) / 2) * Complex.I) * Real.cos t) -
      (-(Complex.exp (-((φ - ψ) / 2) * Complex.I)) * Real.sin t) * (Complex.exp (((φ - ψ) / 2) * Complex.I) * Real.sin t) = 1 := by
  have e1 : Complex.exp (-((φ + ψ) / 2) * Complex.I) * Complex.exp (((φ + ψ) / 2) * Complex.I) = 1 := by
    rw [← Complex.exp_add]; ring_nf; simp
  have e2 : Complex.exp (-((φ - ψ) / 2) * Complex.I) * Complex.exp (((φ - ψ) / 2) * Complex.I) = 1 := by
    rw [← Complex.exp_add]; ring_nf; simp
  have hc : ((Real.cos t : ℂ)) ^ 2 + ((Real.sin t : ℂ)) ^ 2 = 1 := by exact_mod_cast Real.cos_sq_add_sin_sq t
  linear_combination (Real.cos t : ℂ) ^ 2 * e1 + (Real.sin t : ℂ) ^ 2 * e2 + hc

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.MatrixCoefficients.d_half_det
-- test: fa-d-half-det-test.sage

/-- `ψ ↦ ψ + 2π` flips the sign of `D^{1/2}`: `e^{-i(ψ + 2π)/2} = -e^{-iψ/2}`. -/
theorem d_half_sign (ψ : ℝ) : Complex.exp (-((ψ + 2 * π) / 2) * Complex.I) = -Complex.exp (-(ψ / 2) * Complex.I) := by
  rw [show -(((ψ : ℂ) + 2 * π) / 2) * Complex.I = -(ψ / 2) * Complex.I + -(π * Complex.I) by ring, Complex.exp_add,
    show -(↑π * Complex.I) = ((-1 : ℤ) : ℂ) * (π * Complex.I) by push_cast; ring, Complex.exp_int_mul, Complex.exp_pi_mul_I]
  norm_num

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.MatrixCoefficients.d_half_sign
-- test: fa-d-half-sign-test.sage

/-- The Schur integral `∫ |D^{1/2}_{11}|² dg = (8π²/16π²) ∫₀^π cos²(θ/2) sin θ dθ = 1/2`: the `θ`-integral with
`cos²(θ/2) = (1 + cos θ)/2`. -/
theorem schur_half_theta : ∫ t in (0 : ℝ)..π, (1 + Real.cos t) / 2 * Real.sin t = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 2) / 4 - ((Real.cos x_veriphy) / 2) - (1 / 4) : ℝ))
            ((fun t => (1 + cos t) / 2 * sin t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).const_mul (-(1 / 4) : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).add
                    (hasDerivAt_const (x_veriphy : ℝ) (-(1 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 2) / 4 - ((Real.cos x_veriphy) / 2) - (1 / 4) : ℝ)) (f' := fun t =>
          (1 + cos t) / 2 * sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

theorem schur_half_theta_alt : ∫ t in (0 : ℝ)..π, (1 - Real.cos t) / 2 * Real.sin t = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (((Real.cos x_veriphy) ^ 2) / 4 - ((Real.cos x_veriphy) / 2) + (1 / 4) : ℝ))
            ((fun t => (1 - cos t) / 2 * sin t) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).const_mul (1 / 4 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(1 / 2) : ℝ))).add
                    (hasDerivAt_const (x_veriphy : ℝ) (1 / 4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 2) / 4 - ((Real.cos x_veriphy) / 2) + (1 / 4) : ℝ)) (f' := fun t =>
          (1 - cos t) / 2 * sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- The half-angle step `cos²(θ/2) = (1 + cos θ)/2` (with `t = θ/2`). -/
theorem half_angle (t : ℝ) : Real.cos t ^ 2 = (1 + Real.cos (2 * t)) / 2 := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      simp only [hmc2, hms2]
      linear_combination ((1 / 2 : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

/-- `8π²/16π² · 1 = 1/2 = 1/d` for `d = 2`. -/
theorem schur_half_value : 2 * π * (4 * π) / (16 * π ^ 2) * 1 = 1 / 2 := by
  veriphy_certified ((field_simp; ring1))
  -- certificate: fa-identity.sage

/-! ## exm-sph-su2-spin-one, exm-sph-sage-wigner, exr-sph-wigner-exp -/

/-- The Wigner matrix `d¹(θ)` in the weight basis `m = 1, 0, -1`. -/
noncomputable def d1 (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
     Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
     (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2]

/-- `d¹` is orthogonal: `d¹ (d¹)ᵀ = 1`. -/
theorem d1_orthogonal (θ : ℝ) :
    !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2] *
    (!![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2])ᵀ = 1 := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals
          first
          | ring1
          | (norm_num; done)
          |
            linear_combination
              ((1 / 2 : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                ((((Real.sin θ) ^ 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -(((1 / 2 : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                  ((((Real.sin θ) ^ 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          |
            linear_combination
              ((-(1 / 2) : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                ((-((Real.sin θ) ^ 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -(((-(1 / 2) : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                  ((-((Real.sin θ) ^ 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          |
            linear_combination
              ((1 : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                ((((Real.sin θ) ^ 2) / 2 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -(((1 : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
                  ((((Real.sin θ) ^ 2) / 2 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))))
  -- certificate: fa-matrix.sage

/-- `det d¹ = 1`. -/
theorem d1_det (θ : ℝ) :
    Matrix.det !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2] = 1 := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals
        linear_combination
          ((1 : ℝ)) * (Real.cos_sq_add_sin_sq (θ : ℝ)) +
            ((((Real.sin θ) ^ 2) / 2 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
  -- certificate: fa-matrix.sage

/-- `d¹(0) = 1`. -/
theorem d1_zero :
    !![(1 + Real.cos 0) / 2, -(Real.sqrt 2 / 2) * Real.sin 0, (1 - Real.cos 0) / 2;
       Real.sqrt 2 / 2 * Real.sin 0, Real.cos 0, -(Real.sqrt 2 / 2) * Real.sin 0;
       (1 - Real.cos 0) / 2, Real.sqrt 2 / 2 * Real.sin 0, (1 + Real.cos 0) / 2] = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;>
          simp [Matrix.one_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi, Real.sin_pi, Real.cos_two_pi,
            Real.sin_two_pi] <;>
        norm_num)
  -- certificate: fa-matrix-values.sage

/-- The differential equation `d' = -iJ_y d`: the entrywise derivative of `d¹` equals `A d¹` with
`A = -iJ_y = (1/√2) [[0, -1, 0], [1, 0, -1], [0, 1, 0]]`. -/
theorem d1_ode (θ : ℝ) :
    !![-Real.sin θ / 2, -(Real.sqrt 2 / 2) * Real.cos θ, Real.sin θ / 2;
       Real.sqrt 2 / 2 * Real.cos θ, -Real.sin θ, -(Real.sqrt 2 / 2) * Real.cos θ;
       Real.sin θ / 2, Real.sqrt 2 / 2 * Real.cos θ, -Real.sin θ / 2] =
    !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] *
    !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals
          first
          | ring1
          | (norm_num; done)
          | linear_combination (((Real.sin θ) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          | linear_combination -((((Real.sin θ) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          | linear_combination ((-(Real.sin θ) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          | linear_combination -(((-(Real.sin θ) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          | linear_combination (((Real.sin θ) / 2 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          | linear_combination -((((Real.sin θ) / 2 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))))
  -- certificate: fa-matrix.sage

/-- The left side of `d1_ode` is the entrywise derivative of `d¹`. -/
theorem d1_deriv (θ : ℝ) (i j : Fin 3) :
    HasDerivAt (fun θ => d1 θ i j)
      (!![-Real.sin θ / 2, -(Real.sqrt 2 / 2) * Real.cos θ, Real.sin θ / 2;
         Real.sqrt 2 / 2 * Real.cos θ, -Real.sin θ, -(Real.sqrt 2 / 2) * Real.cos θ;
         Real.sin θ / 2, Real.sqrt 2 / 2 * Real.cos θ, -Real.sin θ / 2] i j) θ := by
  have hc := Real.hasDerivAt_cos θ
  have hs := Real.hasDerivAt_sin θ
  fin_cases i <;> fin_cases j <;> simp only [d1, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.cons_val_two, Matrix.empty_val', Matrix.cons_val_fin_one, Matrix.head_cons,
    Matrix.head_fin_const, Matrix.tail_cons, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Fin.isValue] <;>
  first
  | exact hc
  | (convert ((hc.const_add 1).div_const 2) using 1 <;> first | (funext x; ring1) | ring1)
  | (convert ((hc.const_sub 1).div_const 2) using 1 <;> first | (funext x; ring1) | ring1)
  | (convert (hs.const_mul (-(Real.sqrt 2 / 2))) using 1 <;> first | (funext x; ring1) | ring1)
  | (convert (hs.const_mul (Real.sqrt 2 / 2)) using 1 <;> first | (funext x; ring1) | ring1)
  | (convert hc using 1 <;> first | (funext x; ring1) | ring1)
  | (convert (hs.neg.div_const 2) using 1 <;> first | (funext x; ring1) | ring1)
  | (convert (hs.div_const 2) using 1 <;> first | (funext x; ring1) | ring1)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.MatrixCoefficients.d1_deriv
-- test: fa-d1-deriv-test.sage

/-- The spin-`1` commutation relations in the weight basis: `[J_z, J_+] = J_+`, `[J_z, J_-] = -J_-`,
`[J_+, J_-] = 2 J_z`. -/
theorem spin_one_commutators :
    !![(1 : ℝ), 0, 0; 0, 0, 0; 0, 0, -1] * !![0, Real.sqrt 2, 0; 0, 0, Real.sqrt 2; 0, 0, 0] -
      !![0, Real.sqrt 2, 0; 0, 0, Real.sqrt 2; 0, 0, 0] * !![(1 : ℝ), 0, 0; 0, 0, 0; 0, 0, -1] =
      !![0, Real.sqrt 2, 0; 0, 0, Real.sqrt 2; 0, 0, 0] ∧
    !![0, Real.sqrt 2, 0; 0, 0, Real.sqrt 2; 0, 0, 0] * !![0, 0, 0; Real.sqrt 2, 0, 0; 0, Real.sqrt 2, 0] -
      !![0, 0, 0; Real.sqrt 2, 0, 0; 0, Real.sqrt 2, 0] * !![0, Real.sqrt 2, 0; 0, 0, Real.sqrt 2; 0, 0, 0] =
      2 • !![(1 : ℝ), 0, 0; 0, 0, 0; 0, 0, -1] := by
  constructor
  · veriphy_certified
      ((  ext i j
          fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
          all_goals ring1))
    -- certificate: fa-matrix.sage
  · veriphy_certified
      ((  ext i j
          fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
          all_goals ring1))
    -- certificate: fa-matrix.sage

/-- The spin-`1/2` relations: `[J_z, J_+] = J_+`, `[J_+, J_-] = 2 J_z`. -/
theorem spin_half_commutators :
    !![(1 : ℝ) / 2, 0; 0, -1 / 2] * !![0, 1; 0, 0] - !![0, 1; 0, 0] * !![(1 : ℝ) / 2, 0; 0, -1 / 2] = !![0, 1; 0, 0] ∧
    !![(0 : ℝ), 1; 0, 0] * !![0, 0; 1, 0] - !![0, 0; 1, 0] * !![(0 : ℝ), 1; 0, 0] = 2 • !![(1 : ℝ) / 2, 0; 0, -1 / 2] := by
  constructor
  · veriphy_certified
      ((  ext i j
          fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
          all_goals ring1))
    -- certificate: fa-matrix.sage
  · veriphy_certified
      ((  ext i j
          fin_cases i <;> fin_cases j <;>
            simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
          all_goals ring1))
    -- certificate: fa-matrix.sage

/-- `(-iJ_y)² = -J_y²` for spin `1/2` is `-1/4`: `J_y² = 1/4`, so the series gives `cos(θ/2) - 2i sin(θ/2) J_y`. -/
theorem spin_half_jy_sq : !![(0 : ℝ), -1 / 2; 1 / 2, 0] * !![(0 : ℝ), -1 / 2; 1 / 2, 0] = !![-1 / 4, 0; 0, -1 / 4] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals ring1))
  -- certificate: fa-matrix.sage

/-- For spin `1`, `(-iJ_y)³ = -(-iJ_y)`, i.e. `J_y³ = J_y`: the exponential is `1 - i J_y sin θ - J_y²(1 - cos θ)`. -/
theorem spin_one_jy_cube :
    !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] *
    !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] *
    !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] =
    -!![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals
          first
          | ring1
          | (norm_num; done)
          | linear_combination (((Real.sqrt 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          | linear_combination -((((Real.sqrt 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          | linear_combination ((-(Real.sqrt 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          | linear_combination -(((-(Real.sqrt 2) / 4 : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))))
  -- certificate: fa-matrix.sage

/-- The closed form `d¹ = 1 + sin θ A + (1 - cos θ) A²` with `A = -iJ_y`. -/
theorem d1_closed_form (θ : ℝ) :
    (1 : Matrix (Fin 3) (Fin 3) ℝ) + Real.sin θ • !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] +
      (1 - Real.cos θ) • (!![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0] *
        !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0]) =
    !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals
          first
          | ring1
          | (norm_num; done)
          |
            linear_combination
              (((Real.cos θ) / 4 - (1 / 4) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -((((Real.cos θ) / 4 - (1 / 4) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          |
            linear_combination
              ((-(Real.cos θ) / 4 + (1 / 4) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -(((-(Real.cos θ) / 4 + (1 / 4) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))
          |
            linear_combination
              (((Real.cos θ) / 2 - (1 / 2) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity))
          |
            linear_combination
              -((((Real.cos θ) / 2 - (1 / 2) : ℝ)) * (Real.sq_sqrt (show (0 : ℝ) ≤ (2 : ℝ) by positivity)))))
  -- certificate: fa-matrix.sage

/-- The half-angle structure: `d^{1/2}(2π) = -1` (the rotation by `θ/2 = π`) ... -/
theorem double_cover_half :
    !![Real.cos π, -Real.sin π; Real.sin π, Real.cos π] = -(1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;>
          simp [Matrix.one_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi, Real.sin_pi, Real.cos_two_pi,
            Real.sin_two_pi] <;>
        norm_num)
  -- certificate: fa-matrix-values.sage

/-- ... while `d¹(2π) = 1`. -/
theorem double_cover_one :
    !![(1 + Real.cos (2 * π)) / 2, -(Real.sqrt 2 / 2) * Real.sin (2 * π), (1 - Real.cos (2 * π)) / 2;
       Real.sqrt 2 / 2 * Real.sin (2 * π), Real.cos (2 * π), -(Real.sqrt 2 / 2) * Real.sin (2 * π);
       (1 - Real.cos (2 * π)) / 2, Real.sqrt 2 / 2 * Real.sin (2 * π), (1 + Real.cos (2 * π)) / 2] =
      (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  veriphy_certified
    ( ext i j
      fin_cases i <;> fin_cases j <;>
          simp [Matrix.one_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi, Real.sin_pi, Real.cos_two_pi,
            Real.sin_two_pi] <;>
        norm_num)
  -- certificate: fa-matrix-values.sage

/-- The exponential `exp(θ A) = d¹(θ)`, as stated (tested, not proved: the rotation exponential is a known Veriphy
gap, design D30). -/
def WignerExponential : Prop :=
  ∀ θ : ℝ, NormedSpace.exp (θ • !![0, -(Real.sqrt 2 / 2), 0; Real.sqrt 2 / 2, 0, -(Real.sqrt 2 / 2); 0, Real.sqrt 2 / 2, 0]) =
    !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.MatrixCoefficients.WignerExponential
-- test: fa-wigner-exponential-test.sage

/-- The middle column of `D¹` is `√(4π/3) conj(Y_{1m})`: `√(4π/3) √(3/8π) = 1/√2 = √2/2`. -/
theorem middle_column_constant (c : ℝ) (hc : 0 < c) :
    Real.sqrt (4 * π * c / 3) * Real.sqrt (3 / (8 * π * c)) = Real.sqrt 2 / 2 := by
  rw [← Real.sqrt_mul (by positivity), show 4 * π * c / 3 * (3 / (8 * π * c)) = 1 / 2 by first | (field_simp; done) | (field_simp; ring),
    Real.sqrt_div' _ (by norm_num), Real.sqrt_one]
  rw [div_eq_div_iff (Real.sqrt_pos.2 (by norm_num)).ne' (by norm_num)]
  rw [one_mul, ← Real.sqrt_sq (show (0 : ℝ) ≤ 2 by norm_num)]
  rw [Real.sqrt_sq (by norm_num), Real.mul_self_sqrt (by norm_num)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.MatrixCoefficients.middle_column_constant
-- test: fa-middle-column-constant-test.sage

/-- The characters: `χ₁ = 1 + 2 cos θ = tr d¹(θ)`. -/
theorem character_trace (θ : ℝ) :
    Matrix.trace !![(1 + Real.cos θ) / 2, -(Real.sqrt 2 / 2) * Real.sin θ, (1 - Real.cos θ) / 2;
       Real.sqrt 2 / 2 * Real.sin θ, Real.cos θ, -(Real.sqrt 2 / 2) * Real.sin θ;
       (1 - Real.cos θ) / 2, Real.sqrt 2 / 2 * Real.sin θ, (1 + Real.cos θ) / 2] = 1 + 2 * Real.cos θ := by
  veriphy_certified
    ( simp +decide [Matrix.det_succ_row_zero, Matrix.trace, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt,
        Fin.succAbove_of_le_castSucc, Matrix.mul_apply, Matrix.transpose_apply]
      all_goals ring1)
  -- certificate: fa-matrix.sage

/-- `χ_{1/2}² = χ₀ + χ₁`: `(2 cos t)² = 1 + (1 + 2 cos 2t)` with `t = θ/2`, i.e. `1/2 ⊗ 1/2 = 0 ⊕ 1`. -/
theorem character_square (t : ℝ) : (2 * Real.cos t) ^ 2 = 1 + (1 + 2 * Real.cos (2 * t)) := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      simp only [hmc2, hms2]
      linear_combination ((2 : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

/-! ## exr-sph-characters -/

/-- The Weyl measure `(1/π) sin²(θ/2) dθ` is `(2/π) sin² t dt` in `t = θ/2 ∈ [0, π]`, of total mass `1`. -/
theorem weyl_mass : ∫ t in (0 : ℝ)..π, 2 / π * Real.sin t ^ 2 = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy => (-(Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi + (x_veriphy / Real.pi) : ℝ))
            ((fun t => 2 / π * sin t ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                        (-1 / Real.pi : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination ((-1 / Real.pi : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-(Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi + (x_veriphy / Real.pi) : ℝ)) (f' := fun t =>
          2 / π * sin t ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- `χ_{1/2} χ₁ = χ_{1/2} + χ_{3/2}`: `1/2 ⊗ 1 = 1/2 ⊕ 3/2`. -/
theorem character_product (t : ℝ) :
    2 * Real.cos t * (1 + 2 * Real.cos (2 * t)) = 2 * Real.cos t + (2 * Real.cos t + 2 * Real.cos (3 * t)) := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      try (rw [show ((3 * t : ℝ) : ℝ) = 3 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      have hmc3 : Real.cos (3 * ((t : ℝ))) = ((Real.cos t) ^ 3 - (3 * (Real.cos t) * ((Real.sin t) ^ 2)) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add, hmc2, hms2]; ring
      have hms3 : Real.sin (3 * ((t : ℝ))) = (3 * ((Real.cos t) ^ 2) * (Real.sin t) - ((Real.sin t) ^ 3) : ℝ) := by
        rw [show (3 : ℝ) * ((t : ℝ)) = 2 * ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add, hmc2, hms2]; ring
      simp only [hmc2, hms2, hmc3, hms3]
      linear_combination ((2 * (Real.cos t) : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

/-- Weyl orthonormality: `∫ χ₀ χ₀ dμ_W = 1` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_0_0 : ∫ t in (0 : ℝ)..π, 1 * 1 * (2 / π * Real.sin t ^ 2) = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy => (-(Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi + (x_veriphy / Real.pi) : ℝ))
            ((fun t => 1 * 1 * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                        (-1 / Real.pi : ℝ)).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination ((-1 / Real.pi : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-(Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi + (x_veriphy / Real.pi) : ℝ)) (f' := fun t =>
          1 * 1 * (2 / π * sin t ^ 2)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ₀ χ_{1/2} dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_0_half : ∫ t in (0 : ℝ)..π, 1 * 2 * Real.cos t * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt (fun x_veriphy => (4 * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi) : ℝ))
            ((fun t => 1 * 2 * cos t * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (4 / (3 * Real.pi) : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (4 * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi) : ℝ)) (f' := fun t => 1 * 2 * cos t * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ₀ χ₁ dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_0_one : ∫ t in (0 : ℝ)..π, 1 * (1 + 2 * Real.cos (2 * t)) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / Real.pi +
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
                  ((Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
                ℝ))
            ((fun t => 1 * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                            (-1 / Real.pi : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                          (1 / Real.pi ^ 1 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 2) / Real.pi + (3 * ((Real.sin x_veriphy) ^ 2) / Real.pi) : ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / Real.pi +
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
              ((Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
            ℝ))
          (f' := fun t => 1 * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ₀ χ_{3/2} dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_0_threehalves : ∫ t in (0 : ℝ)..π, 1 * (2 * Real.cos t + 2 * Real.cos (3 * t)) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-16 * ((Real.sin x_veriphy) ^ 5) / (5 * Real.pi) + (8 * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) : ℝ))
            ((fun t => 1 * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
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
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (-16 / (5 * Real.pi) : ℝ)).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul
                      (8 / (3 * Real.pi) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-4 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / Real.pi : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-16 * ((Real.sin x_veriphy) ^ 5) / (5 * Real.pi) + (8 * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) : ℝ))
          (f' := fun t => 1 * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he1 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ_{1/2} χ_{1/2} dμ_W = 1` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_half_half : ∫ t in (0 : ℝ)..π, 2 * Real.cos t * 2 * Real.cos t * (2 / π * Real.sin t ^ 2) = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / Real.pi +
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
                  (x_veriphy / Real.pi) :
                ℝ))
            ((fun t => 2 * cos t * 2 * cos t * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                            (-1 / Real.pi : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                          (1 / Real.pi ^ 1 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          linear_combination
            ((-((Real.cos x_veriphy) ^ 2) / Real.pi - (((Real.sin x_veriphy) ^ 2) / Real.pi) - (1 / Real.pi ^ 1) : ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / Real.pi +
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
              (x_veriphy / Real.pi) :
            ℝ))
          (f' := fun t => 2 * cos t * 2 * cos t * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ_{1/2} χ₁ dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_half_one : ∫ t in (0 : ℝ)..π, 2 * Real.cos t * (1 + 2 * Real.cos (2 * t)) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / Real.pi +
                        (2 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                      (((Real.sin x_veriphy) ^ 5) / (5 * Real.pi)) +
                    (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / Real.pi) +
                  (((Real.sin x_veriphy) ^ 3) / Real.pi) :
                ℝ))
            ((fun t => 2 * cos t * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          (((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                    (-1 / Real.pi : ℝ)).add
                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                  (2 / Real.pi : ℝ))).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (-1 / (5 * Real.pi) : ℝ))).add
                        (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                              ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / Real.pi ^ 1 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul
                      (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 3) / Real.pi +
                  (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / Real.pi) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / Real.pi +
                    (2 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                  (((Real.sin x_veriphy) ^ 5) / (5 * Real.pi)) +
                (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / Real.pi) +
              (((Real.sin x_veriphy) ^ 3) / Real.pi) :
            ℝ))
          (f' := fun t => 2 * cos t * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ_{1/2} χ_{3/2} dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_half_threehalves : ∫ t in (0 : ℝ)..π, 2 * Real.cos t * (2 * Real.cos t + 2 * Real.cos (3 * t)) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / Real.pi +
                      (10 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) -
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                  ((Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
                ℝ))
            ((fun t => 2 * cos t * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
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
          ((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                (-1 / Real.pi : ℝ)).add
                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (10 / (3 * Real.pi) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5)).const_mul
                          (-1 / Real.pi : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) / Real.pi +
                        (8 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) +
                      (((Real.sin x_veriphy) ^ 4) / Real.pi) -
                    (((Real.cos x_veriphy) ^ 2) / Real.pi) +
                  (((Real.sin x_veriphy) ^ 2) / Real.pi) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / Real.pi +
                  (10 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) -
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
              ((Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
            ℝ))
          (f' := fun t => 2 * cos t * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ₁ χ₁ dμ_W = 1` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_one_one : ∫ t in (0 : ℝ)..π, (1 + 2 * Real.cos (2 * t)) * (1 + 2 * Real.cos (2 * t)) * (2 / π * Real.sin t ^ 2) = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / Real.pi +
                      (10 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) -
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                  (x_veriphy / Real.pi) :
                ℝ))
            ((fun t => (1 + 2 * cos (2 * t)) * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          ((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                (-1 / Real.pi : ℝ)).add
                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (10 / (3 * Real.pi) : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5)).const_mul
                          (-1 / Real.pi : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 4) / Real.pi +
                          (8 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                        (7 * ((Real.sin x_veriphy) ^ 4) / Real.pi) -
                      (((Real.cos x_veriphy) ^ 2) / Real.pi) +
                    (((Real.sin x_veriphy) ^ 2) / Real.pi) -
                  (1 / Real.pi ^ 1) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / Real.pi +
                  (10 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) -
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
              (x_veriphy / Real.pi) :
            ℝ))
          (f' := fun t => (1 + 2 * cos (2 * t)) * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ₁ χ_{3/2} dμ_W = 0` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_one_threehalves : ∫ t in (0 : ℝ)..π, (1 + 2 * Real.cos (2 * t)) * (2 * Real.cos t + 2 * Real.cos (3 * t)) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 6) * (Real.sin x_veriphy) / Real.pi +
                                  (5 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                                (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                              (((Real.sin x_veriphy) ^ 7) / (7 * Real.pi)) +
                            (((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / Real.pi) -
                          (2 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                        (3 * ((Real.sin x_veriphy) ^ 5) / Real.pi) -
                      (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / Real.pi) +
                    (3 * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
                  ((Real.sin x_veriphy) / Real.pi) :
                ℝ))
            ((fun t => (1 + 2 * cos (2 * t)) * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) x_veriphy)
            x_veriphy :=
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
          ((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 6).mul
                                                            ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                                        (-1 / Real.pi : ℝ)).add
                                                    (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).mul
                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                                      (5 / Real.pi : ℝ))).add
                                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5)).const_mul
                                                  (-3 / Real.pi : ℝ))).add
                                            ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 7).const_mul
                                              (1 / (7 * Real.pi) : ℝ))).add
                                        (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).mul
                                              ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                          (1 / Real.pi ^ 1 : ℝ))).add
                                    (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                      (-2 / Real.pi : ℝ))).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (-3 / Real.pi : ℝ))).add
                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                  ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                              (-1 / Real.pi : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (3 / Real.pi : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 5) / Real.pi +
                        (14 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                      (17 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 4) / Real.pi) -
                    (8 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                  ((Real.cos x_veriphy) / Real.pi) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 6) * (Real.sin x_veriphy) / Real.pi +
                              (5 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                            (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                          (((Real.sin x_veriphy) ^ 7) / (7 * Real.pi)) +
                        (((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / Real.pi) -
                      (2 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                    (3 * ((Real.sin x_veriphy) ^ 5) / Real.pi) -
                  (((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / Real.pi) +
                (3 * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
              ((Real.sin x_veriphy) / Real.pi) :
            ℝ))
          (f' := fun t => (1 + 2 * cos (2 * t)) * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

/-- Weyl orthonormality: `∫ χ_{3/2} χ_{3/2} dμ_W = 1` (`θ = 2t`, `dμ_W = (2/π) sin² t dt`). -/
theorem weyl_threehalves_threehalves : ∫ t in (0 : ℝ)..π, (2 * Real.cos t + 2 * Real.cos (3 * t)) * (2 * Real.cos t + 2 * Real.cos (3 * t)) * (2 / π * Real.sin t ^ 2) = 1 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-((Real.cos x_veriphy) ^ 7) * (Real.sin x_veriphy) / Real.pi +
                        (7 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                      (7 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                    ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 7) / Real.pi) +
                  (x_veriphy / Real.pi) :
                ℝ))
            ((fun t => (2 * cos t + 2 * cos (3 * t)) * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) x_veriphy)
            x_veriphy :=
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
          (((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 7).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                    (-1 / Real.pi : ℝ)).add
                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).mul
                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                  (7 / Real.pi : ℝ))).add
                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5)).const_mul
                              (-7 / Real.pi : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 7)).const_mul
                          (1 / Real.pi ^ 1 : ℝ))).add
                    ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (1 / Real.pi ^ 1 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 6) / Real.pi +
                                  (21 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                                (43 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 4) / Real.pi) -
                              (((Real.sin x_veriphy) ^ 6) / Real.pi) -
                            (((Real.cos x_veriphy) ^ 4) / Real.pi) +
                          (6 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                        (((Real.sin x_veriphy) ^ 4) / Real.pi) -
                      (((Real.cos x_veriphy) ^ 2) / Real.pi) -
                    (((Real.sin x_veriphy) ^ 2) / Real.pi) -
                  (1 / Real.pi ^ 1) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-((Real.cos x_veriphy) ^ 7) * (Real.sin x_veriphy) / Real.pi +
                    (7 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 3) / Real.pi) -
                  (7 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 5) / Real.pi) +
                ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 7) / Real.pi) +
              (x_veriphy / Real.pi) :
            ℝ))
          (f' := fun t => (2 * cos t + 2 * cos (3 * t)) * (2 * cos t + 2 * cos (3 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

end KaluzaKlein.FourierAnalysis.MatrixCoefficients

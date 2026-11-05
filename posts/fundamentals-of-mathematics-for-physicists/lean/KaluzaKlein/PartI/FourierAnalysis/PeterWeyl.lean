import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.MatrixCoefficients

/-! # Fourier analysis: peter weyl (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.PeterWeyl

open Real MeasureTheory Filter Topology Matrix

/-! ## thm-peter-weyl, exm-sph-pw-u1 -/

/-- Peter-Weyl for `U(1)` (mathlib): the characters `e^{inφ}` form a Hilbert basis of `L²` of the circle. -/
theorem pw_circle (T : ℝ) [hT : Fact (0 < T)] : ⇑(@fourierBasis T hT) = fourierLp 2 := coe_fourierBasis

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeterWeyl.pw_circle
-- test: fa-pw-circle-test.sage

/-- The translation `f(φ) ↦ f(φ - α)` acts on the `n`-th mode by the phase `e^{-inα}`. -/
theorem translation_phase (n α φ : ℝ) :
    Complex.exp (n * (φ - α) * Complex.I) = Complex.exp (-(n * α) * Complex.I) * Complex.exp (n * φ * Complex.I) := by
  rw [← Complex.exp_add]; congr 1; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeterWeyl.translation_phase
-- test: fa-translation-phase-test.sage

/-! ## exm-sph-pw-su2, exm-sph-coset-s3, exr-sph-coset-degeneracies -/

/-- The Casimir and the round `S³`: `-4j(j+1) = -n(n+2)` for `n = 2j`, with multiplicity `(2j+1)² = (n+1)²`. -/
theorem s3_spectrum (n : ℕ) : 4 * ((n : ℚ) / 2) * ((n : ℚ) / 2 + 1) = n * (n + 2) ∧ (2 * ((n : ℚ) / 2) + 1) ^ 2 = (n + 1) ^ 2 := by
  constructor <;> ring

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.s3_spectrum (0) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.s3_spectrum (4) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.s3_spectrum (1) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.PeterWeyl.s3_spectrum
-- test: fa-s3-spectrum-test.sage

/-- `cos θ = (χ₁ - χ₀)/2`. -/
theorem cos_in_characters (θ : ℝ) : Real.cos θ = ((1 + 2 * Real.cos θ) - 1) / 2 := by ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeterWeyl.cos_in_characters
-- test: fa-cos-in-characters-test.sage

/-! ## exm-sph-pw-finite, exr-sph-pw-dimension -/

/-- The character table of `S₃` (classes of sizes `1, 3, 2`): `X W Xᵀ = 6 I` with `W = diag(1, 3, 2)`, the
orthonormality of the characters for `(1/|G|) Σ_g`. -/
theorem s3_character_orthogonality :
    !![(1 : ℚ), 1, 1; 1, -1, 1; 2, 0, -1] * !![(1 : ℚ), 0, 0; 0, 3, 0; 0, 0, 2] * (!![(1 : ℚ), 1, 1; 1, -1, 1; 2, 0, -1])ᵀ =
      !![(6 : ℚ), 0, 0; 0, 6, 0; 0, 0, 6] := by
  veriphy_certified
    ((  ext i j
        fin_cases i <;> fin_cases j <;>
          simp [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, Fin.sum_univ_succ]
        all_goals ring1))
  -- certificate: fa-matrix.sage

/-- `Σ d_r² = |G|` for `S₃`: `1 + 1 + 4 = 6`. -/
theorem s3_dimension_sum : (1 : ℚ) ^ 2 + 1 ^ 2 + 2 ^ 2 = 6 := by
  veriphy_certified (ring1)
  -- certificate: fa-identity.sage

/-- `|S_n| = n!`, so `|S₃| = 6`. -/
theorem card_perm (n : ℕ) : Fintype.card (Equiv.Perm (Fin n)) = n.factorial := by
  rw [Fintype.card_perm, Fintype.card_fin]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.card_perm (0) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.card_perm (1) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.card_perm (2) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.PeterWeyl.card_perm (3) := by
  decide

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.FourierAnalysis.PeterWeyl.card_perm
-- test: fa-card-perm-test.sage

/-- For `ℤ_n` the `n` characters are orthogonal: `Σ_k ζ^{(j-j')k} = n δ` with `ζ = e^{2πi/n}`, and `Σ d² = n`. -/
theorem cyclic_characters (n : ℕ) (hn : 0 < n) (a : ℤ) :
    ∑ k ∈ Finset.range n, Complex.exp (((2 * π * a * k / n : ℝ) : ℂ) * Complex.I) =
      if (n : ℤ) ∣ a then (n : ℂ) else 0 := by
  have hn' : (n : ℂ) ≠ 0 := by exact_mod_cast hn.ne'
  have h2 : (2 * π * Complex.I : ℂ) ≠ 0 := by simp [Real.pi_ne_zero, Complex.I_ne_zero]
  set z := Complex.exp (((2 * π * a / n : ℝ) : ℂ) * Complex.I) with hz
  have hk : ∀ k : ℕ, Complex.exp (((2 * π * a * k / n : ℝ) : ℂ) * Complex.I) = z ^ k := by
    intro k; rw [hz, ← Complex.exp_nat_mul]; congr 1; push_cast; ring
  simp_rw [hk]
  have hzn : z ^ n = 1 := by
    rw [hz, ← Complex.exp_nat_mul,
      show (n : ℂ) * (((2 * π * a / n : ℝ) : ℂ) * Complex.I) = a * (2 * π * Complex.I) by
        push_cast; field_simp]
    exact Complex.exp_int_mul_two_pi_mul_I a
  split_ifs with h
  · obtain ⟨m, rfl⟩ := h
    have : z = 1 := by
      rw [hz, show (((2 * π * (((n : ℤ) * m : ℤ) : ℝ) / n : ℝ)) : ℂ) * Complex.I = m * (2 * π * Complex.I) by
        push_cast; field_simp]
      exact Complex.exp_int_mul_two_pi_mul_I m
    simp [this]
  · have hz1 : z ≠ 1 := by
      intro h1
      rw [hz, Complex.exp_eq_one_iff] at h1
      obtain ⟨m, hm⟩ := h1
      apply h
      refine ⟨m, ?_⟩
      have : (a : ℂ) = n * m := by
        push_cast at hm
        rw [div_mul_eq_mul_div, div_eq_iff hn'] at hm
        apply mul_right_cancel₀ h2
        linear_combination hm
      exact_mod_cast this
    rw [geom_sum_eq hz1, hzn, sub_self, zero_div]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.FourierAnalysis.PeterWeyl.cyclic_characters
-- test: fa-cyclic-characters-test.sage

/-! ## exr-sph-pw-weyl -/

/-- The conjugacy class of angle `θ` is the 2-sphere `a₀ = cos(θ/2)` of radius `sin(θ/2)`; slicing the uniform measure
of `S³` (volume `2π²`) gives the density `4π sin²(θ/2) · (1/2)/(2π²) = (1/π) sin²(θ/2)`. -/
theorem weyl_density (θ : ℝ) :
    4 * π * Real.sin (θ / 2) ^ 2 * (1 / 2) / (2 * π ^ 2) = 1 / π * Real.sin (θ / 2) ^ 2 ∧
      1 - Real.cos (θ / 2) ^ 2 = Real.sin (θ / 2) ^ 2 := by
  constructor
  · have : π ≠ 0 := Real.pi_ne_zero
    field_simp; ring
  · rw [← Real.sin_sq_add_cos_sq (θ / 2)]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeterWeyl.weyl_density
-- test: fa-weyl-density-test.sage

/-! ## exr-sph-class-function -/

/-- `cos²(θ/2) = (χ₀ + χ₁)/4 = χ_{1/2}²/4` (with `t = θ/2`). -/
theorem cos_sq_half_characters (t : ℝ) : Real.cos t ^ 2 = (1 + (1 + 2 * Real.cos (2 * t))) / 4 := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
      have hmc2 : Real.cos (2 * ((t : ℝ))) = ((Real.cos t) ^ 2 - ((Real.sin t) ^ 2) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.cos_add]; ring
      have hms2 : Real.sin (2 * ((t : ℝ))) = (2 * (Real.cos t) * (Real.sin t) : ℝ) := by
        rw [show (2 : ℝ) * ((t : ℝ)) = ((t : ℝ)) + ((t : ℝ)) by ring, Real.sin_add]; ring
      simp only [hmc2, hms2]
      linear_combination ((1 / 2 : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

/-- `χ_j sin²(θ/2) = (cos jθ - cos(j+1)θ)/2` for integer `j` (here `j = 1`, `t = θ/2`). -/
theorem character_weight (t : ℝ) :
    (1 + 2 * Real.cos (2 * t)) * Real.sin t ^ 2 = (Real.cos (2 * t) - Real.cos (4 * t)) / 2 := by
  veriphy_certified
    ( try (rw [show ((2 * t : ℝ) : ℝ) = 2 * ((t : ℝ)) by ring])
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
      linear_combination
        ((((Real.cos t) ^ 2) / 2 - (3 * ((Real.sin t) ^ 2) / 2) : ℝ)) * Real.sin_sq_add_cos_sq ((t : ℝ)))
  -- certificate: fa-trig-poly.sage

set_option maxHeartbeats 1000000 in
/-- The coefficient of `θ(2π - θ)` on `χ₁` is `-2·3/(1·4) = -3/2`. -/
theorem class_coeff_one :
    ∫ t in (0 : ℝ)..π, 2 * t * (2 * π - 2 * t) * (1 + 2 * Real.cos (2 * t)) * (2 / π * Real.sin t ^ 2) = -3 / 2 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) +
                                                    (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) *
                                                        (Real.sin x_veriphy) /
                                                      Real.pi) +
                                                  (4 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) -
                                                (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) *
                                                    ((Real.sin x_veriphy) ^ 3) /
                                                  Real.pi) +
                                              (x_veriphy * ((Real.cos x_veriphy) ^ 4) / (2 * Real.pi)) -
                                            (((Real.cos x_veriphy) ^ 4) / 4) -
                                          (3 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) /
                                            Real.pi) +
                                        (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) +
                                      (x_veriphy * ((Real.sin x_veriphy) ^ 4) / (2 * Real.pi)) -
                                    (((Real.sin x_veriphy) ^ 4) / 4) +
                                  (4 * x_veriphy * (Real.cos x_veriphy) * (Real.sin x_veriphy)) -
                                (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) -
                              (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / (2 * Real.pi)) +
                            ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / (2 * Real.pi)) -
                          (2 * x_veriphy * ((Real.cos x_veriphy) ^ 2) / Real.pi) +
                        ((Real.cos x_veriphy) ^ 2) +
                      (2 * x_veriphy * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                    ((Real.sin x_veriphy) ^ 2) +
                  (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
                ℝ))
            ((fun t => 2 * t * (2 * π - 2 * t) * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        refine
          ((((((((((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                                                    (((hasDerivAt_id'
                                                                                                            (x_veriphy :
                                                                                                              ℝ)).cos).pow
                                                                                                      3)).mul
                                                                                                ((hasDerivAt_id'
                                                                                                    (x_veriphy :
                                                                                                      ℝ)).sin)).const_mul
                                                                                            (-4 : ℝ)).add
                                                                                        (((((hasDerivAt_id'
                                                                                                          (x_veriphy :
                                                                                                            ℝ)).pow
                                                                                                      2).mul
                                                                                                  (((hasDerivAt_id'
                                                                                                          (x_veriphy :
                                                                                                            ℝ)).cos).pow
                                                                                                    3)).mul
                                                                                              ((hasDerivAt_id'
                                                                                                  (x_veriphy :
                                                                                                    ℝ)).sin)).const_mul
                                                                                          (4 / Real.pi : ℝ))).add
                                                                                    ((((hasDerivAt_id'
                                                                                                  (x_veriphy : ℝ)).mul
                                                                                              ((hasDerivAt_id'
                                                                                                  (x_veriphy :
                                                                                                    ℝ)).cos)).mul
                                                                                          (((hasDerivAt_id'
                                                                                                  (x_veriphy :
                                                                                                    ℝ)).sin).pow
                                                                                            3)).const_mul
                                                                                      (4 : ℝ))).add
                                                                                (((((hasDerivAt_id' (x_veriphy : ℝ)).pow
                                                                                              2).mul
                                                                                          ((hasDerivAt_id'
                                                                                              (x_veriphy : ℝ)).cos)).mul
                                                                                      (((hasDerivAt_id'
                                                                                              (x_veriphy : ℝ)).sin).pow
                                                                                        3)).const_mul
                                                                                  (-4 / Real.pi : ℝ))).add
                                                                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                                  (((hasDerivAt_id'
                                                                                          (x_veriphy : ℝ)).cos).pow
                                                                                    4)).const_mul
                                                                              (1 / (2 * Real.pi) : ℝ))).add
                                                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow
                                                                              4).const_mul
                                                                          (-(1 / 4) : ℝ))).add
                                                                    ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                              (((hasDerivAt_id'
                                                                                      (x_veriphy : ℝ)).cos).pow
                                                                                2)).mul
                                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                            2)).const_mul
                                                                      (-3 / Real.pi : ℝ))).add
                                                                (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                        2)).const_mul
                                                                  (3 / 2 : ℝ))).add
                                                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                    4)).const_mul
                                                              (1 / (2 * Real.pi) : ℝ))).add
                                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 4).const_mul
                                                          (-(1 / 4) : ℝ))).add
                                                    ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                              ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).mul
                                                          ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                                      (4 : ℝ))).add
                                                (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                                          ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).mul
                                                      ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                                  (-4 / Real.pi : ℝ))).add
                                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                                  ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                              (-1 / (2 * Real.pi) : ℝ))).add
                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                          (1 / (2 * Real.pi) : ℝ))).add
                                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                          (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2)).const_mul
                                      (-2 / Real.pi : ℝ))).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                              (2 / Real.pi : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2).const_mul (-1 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (2 / Real.pi : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2]
          have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
          linear_combination (norm := (field_simp; try ring1))
            ((-4 * x_veriphy * ((Real.cos x_veriphy) ^ 2) +
                      (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 2) / Real.pi) +
                    (12 * x_veriphy * ((Real.sin x_veriphy) ^ 2)) -
                  (12 * (x_veriphy ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) +
                                                (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) *
                                                    (Real.sin x_veriphy) /
                                                  Real.pi) +
                                              (4 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) -
                                            (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) /
                                              Real.pi) +
                                          (x_veriphy * ((Real.cos x_veriphy) ^ 4) / (2 * Real.pi)) -
                                        (((Real.cos x_veriphy) ^ 4) / 4) -
                                      (3 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) /
                                        Real.pi) +
                                    (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) +
                                  (x_veriphy * ((Real.sin x_veriphy) ^ 4) / (2 * Real.pi)) -
                                (((Real.sin x_veriphy) ^ 4) / 4) +
                              (4 * x_veriphy * (Real.cos x_veriphy) * (Real.sin x_veriphy)) -
                            (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) -
                          (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / (2 * Real.pi)) +
                        ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / (2 * Real.pi)) -
                      (2 * x_veriphy * ((Real.cos x_veriphy) ^ 2) / Real.pi) +
                    ((Real.cos x_veriphy) ^ 2) +
                  (2 * x_veriphy * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                ((Real.sin x_veriphy) ^ 2) +
              (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) / Real.pi) :
            ℝ))
          (f' := fun t => 2 * t * (2 * π - 2 * t) * (1 + 2 * cos (2 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 :=
        by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
        norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 :=
        by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]
        exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (field_simp; ring1))
  -- certificate: fa-trig-integral.sage

set_option maxHeartbeats 1000000 in
/-- The coefficient on `χ₂` is `-2·5/(4·9) = -5/18`. -/
theorem class_coeff_two :
    ∫ t in (0 : ℝ)..π, 2 * t * (2 * π - 2 * t) * (1 + 2 * Real.cos (2 * t) + 2 * Real.cos (4 * t)) * (2 / π * Real.sin t ^ 2) =
      -5 / 18 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) +
                                                                        (4 * (x_veriphy ^ 2) *
                                                                              ((Real.cos x_veriphy) ^ 5) *
                                                                            (Real.sin x_veriphy) /
                                                                          Real.pi) +
                                                                      (40 * x_veriphy * ((Real.cos x_veriphy) ^ 3) *
                                                                          ((Real.sin x_veriphy) ^ 3) /
                                                                        3) -
                                                                    (40 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) *
                                                                        ((Real.sin x_veriphy) ^ 3) /
                                                                      (3 * Real.pi)) -
                                                                  (4 * x_veriphy * (Real.cos x_veriphy) *
                                                                    ((Real.sin x_veriphy) ^ 5)) +
                                                                (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) *
                                                                    ((Real.sin x_veriphy) ^ 5) /
                                                                  Real.pi) +
                                                              (2 * x_veriphy * ((Real.cos x_veriphy) ^ 6) /
                                                                (9 * Real.pi)) -
                                                            (((Real.cos x_veriphy) ^ 6) / 9) -
                                                          (10 * x_veriphy * ((Real.cos x_veriphy) ^ 4) *
                                                              ((Real.sin x_veriphy) ^ 2) /
                                                            (3 * Real.pi)) +
                                                        (5 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 2) /
                                                          3) +
                                                      (10 * x_veriphy * ((Real.cos x_veriphy) ^ 2) *
                                                          ((Real.sin x_veriphy) ^ 4) /
                                                        (3 * Real.pi)) -
                                                    (5 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 4) / 3) -
                                                  (2 * x_veriphy * ((Real.sin x_veriphy) ^ 6) / (9 * Real.pi)) +
                                                (((Real.sin x_veriphy) ^ 6) / 9) +
                                              (4 * x_veriphy * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy)) -
                                            (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) /
                                              Real.pi) -
                                          (2 * ((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / (9 * Real.pi)) -
                                        (4 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) +
                                      (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) /
                                        Real.pi) +
                                    (20 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (27 * Real.pi)) -
                                  (2 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / (9 * Real.pi)) -
                                (x_veriphy * ((Real.cos x_veriphy) ^ 4) / (2 * Real.pi)) +
                              (((Real.cos x_veriphy) ^ 4) / 4) +
                            (3 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                          (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) -
                        (x_veriphy * ((Real.sin x_veriphy) ^ 4) / (2 * Real.pi)) +
                      (((Real.sin x_veriphy) ^ 4) / 4) +
                    (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / (2 * Real.pi)) -
                  ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / (2 * Real.pi)) :
                ℝ))
            ((fun t => 2 * t * (2 * π - 2 * t) * (1 + 2 * cos (2 * t) + 2 * cos (4 * t)) * (2 / π * sin t ^ 2))
              x_veriphy)
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
          ((((((((((((((((((((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                                                                                            (((hasDerivAt_id'
                                                                                                                                                    (x_veriphy :
                                                                                                                                                      ℝ)).cos).pow
                                                                                                                                              5)).mul
                                                                                                                                        ((hasDerivAt_id'
                                                                                                                                            (x_veriphy :
                                                                                                                                              ℝ)).sin)).const_mul
                                                                                                                                    (-4 :
                                                                                                                                      ℝ)).add
                                                                                                                                (((((hasDerivAt_id'
                                                                                                                                                  (x_veriphy :
                                                                                                                                                    ℝ)).pow
                                                                                                                                              2).mul
                                                                                                                                          (((hasDerivAt_id'
                                                                                                                                                  (x_veriphy :
                                                                                                                                                    ℝ)).cos).pow
                                                                                                                                            5)).mul
                                                                                                                                      ((hasDerivAt_id'
                                                                                                                                          (x_veriphy :
                                                                                                                                            ℝ)).sin)).const_mul
                                                                                                                                  (4 /
                                                                                                                                      Real.pi :
                                                                                                                                    ℝ))).add
                                                                                                                            ((((hasDerivAt_id'
                                                                                                                                          (x_veriphy :
                                                                                                                                            ℝ)).mul
                                                                                                                                      (((hasDerivAt_id'
                                                                                                                                              (x_veriphy :
                                                                                                                                                ℝ)).cos).pow
                                                                                                                                        3)).mul
                                                                                                                                  (((hasDerivAt_id'
                                                                                                                                          (x_veriphy :
                                                                                                                                            ℝ)).sin).pow
                                                                                                                                    3)).const_mul
                                                                                                                              (40 /
                                                                                                                                  3 :
                                                                                                                                ℝ))).add
                                                                                                                        (((((hasDerivAt_id'
                                                                                                                                          (x_veriphy :
                                                                                                                                            ℝ)).pow
                                                                                                                                      2).mul
                                                                                                                                  (((hasDerivAt_id'
                                                                                                                                          (x_veriphy :
                                                                                                                                            ℝ)).cos).pow
                                                                                                                                    3)).mul
                                                                                                                              (((hasDerivAt_id'
                                                                                                                                      (x_veriphy :
                                                                                                                                        ℝ)).sin).pow
                                                                                                                                3)).const_mul
                                                                                                                          (-40 /
                                                                                                                              (3 *
                                                                                                                                Real.pi) :
                                                                                                                            ℝ))).add
                                                                                                                    ((((hasDerivAt_id'
                                                                                                                                  (x_veriphy :
                                                                                                                                    ℝ)).mul
                                                                                                                              ((hasDerivAt_id'
                                                                                                                                  (x_veriphy :
                                                                                                                                    ℝ)).cos)).mul
                                                                                                                          (((hasDerivAt_id'
                                                                                                                                  (x_veriphy :
                                                                                                                                    ℝ)).sin).pow
                                                                                                                            5)).const_mul
                                                                                                                      (-4 :
                                                                                                                        ℝ))).add
                                                                                                                (((((hasDerivAt_id'
                                                                                                                                  (x_veriphy :
                                                                                                                                    ℝ)).pow
                                                                                                                              2).mul
                                                                                                                          ((hasDerivAt_id'
                                                                                                                              (x_veriphy :
                                                                                                                                ℝ)).cos)).mul
                                                                                                                      (((hasDerivAt_id'
                                                                                                                              (x_veriphy :
                                                                                                                                ℝ)).sin).pow
                                                                                                                        5)).const_mul
                                                                                                                  (4 /
                                                                                                                      Real.pi :
                                                                                                                    ℝ))).add
                                                                                                            (((hasDerivAt_id'
                                                                                                                      (x_veriphy :
                                                                                                                        ℝ)).mul
                                                                                                                  (((hasDerivAt_id'
                                                                                                                          (x_veriphy :
                                                                                                                            ℝ)).cos).pow
                                                                                                                    6)).const_mul
                                                                                                              (2 /
                                                                                                                  (9 *
                                                                                                                    Real.pi) :
                                                                                                                ℝ))).add
                                                                                                        ((((hasDerivAt_id'
                                                                                                                    (x_veriphy :
                                                                                                                      ℝ)).cos).pow
                                                                                                              6).const_mul
                                                                                                          (-(1 / 9) :
                                                                                                            ℝ))).add
                                                                                                    ((((hasDerivAt_id'
                                                                                                                  (x_veriphy :
                                                                                                                    ℝ)).mul
                                                                                                              (((hasDerivAt_id'
                                                                                                                      (x_veriphy :
                                                                                                                        ℝ)).cos).pow
                                                                                                                4)).mul
                                                                                                          (((hasDerivAt_id'
                                                                                                                  (x_veriphy :
                                                                                                                    ℝ)).sin).pow
                                                                                                            2)).const_mul
                                                                                                      (-10 /
                                                                                                          (3 *
                                                                                                            Real.pi) :
                                                                                                        ℝ))).add
                                                                                                (((((hasDerivAt_id'
                                                                                                                (x_veriphy :
                                                                                                                  ℝ)).cos).pow
                                                                                                          4).mul
                                                                                                      (((hasDerivAt_id'
                                                                                                              (x_veriphy :
                                                                                                                ℝ)).sin).pow
                                                                                                        2)).const_mul
                                                                                                  (5 / 3 : ℝ))).add
                                                                                            ((((hasDerivAt_id'
                                                                                                          (x_veriphy :
                                                                                                            ℝ)).mul
                                                                                                      (((hasDerivAt_id'
                                                                                                              (x_veriphy :
                                                                                                                ℝ)).cos).pow
                                                                                                        2)).mul
                                                                                                  (((hasDerivAt_id'
                                                                                                          (x_veriphy :
                                                                                                            ℝ)).sin).pow
                                                                                                    4)).const_mul
                                                                                              (10 / (3 * Real.pi) :
                                                                                                ℝ))).add
                                                                                        (((((hasDerivAt_id'
                                                                                                        (x_veriphy :
                                                                                                          ℝ)).cos).pow
                                                                                                  2).mul
                                                                                              (((hasDerivAt_id'
                                                                                                      (x_veriphy :
                                                                                                        ℝ)).sin).pow
                                                                                                4)).const_mul
                                                                                          (-(5 / 3) : ℝ))).add
                                                                                    (((hasDerivAt_id'
                                                                                              (x_veriphy : ℝ)).mul
                                                                                          (((hasDerivAt_id'
                                                                                                  (x_veriphy :
                                                                                                    ℝ)).sin).pow
                                                                                            6)).const_mul
                                                                                      (-2 / (9 * Real.pi) : ℝ))).add
                                                                                ((((hasDerivAt_id'
                                                                                            (x_veriphy : ℝ)).sin).pow
                                                                                      6).const_mul
                                                                                  (1 / 9 : ℝ))).add
                                                                            ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                                      (((hasDerivAt_id'
                                                                                              (x_veriphy : ℝ)).cos).pow
                                                                                        3)).mul
                                                                                  ((hasDerivAt_id'
                                                                                      (x_veriphy : ℝ)).sin)).const_mul
                                                                              (4 : ℝ))).add
                                                                        (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                                                                  (((hasDerivAt_id'
                                                                                          (x_veriphy : ℝ)).cos).pow
                                                                                    3)).mul
                                                                              ((hasDerivAt_id'
                                                                                  (x_veriphy : ℝ)).sin)).const_mul
                                                                          (-4 / Real.pi : ℝ))).add
                                                                    (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 5).mul
                                                                          ((hasDerivAt_id'
                                                                              (x_veriphy : ℝ)).sin)).const_mul
                                                                      (-2 / (9 * Real.pi) : ℝ))).add
                                                                ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                          ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).mul
                                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                        3)).const_mul
                                                                  (-4 : ℝ))).add
                                                            (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                                                      ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).mul
                                                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                    3)).const_mul
                                                              (4 / Real.pi : ℝ))).add
                                                        (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                                                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                                                          (20 / (27 * Real.pi) : ℝ))).add
                                                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5)).const_mul
                                                      (-2 / (9 * Real.pi) : ℝ))).add
                                                (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4)).const_mul
                                                  (-1 / (2 * Real.pi) : ℝ))).add
                                            ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).const_mul (1 / 4 : ℝ))).add
                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                  (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2)).mul
                                              (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                                          (3 / Real.pi : ℝ))).add
                                    (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                                      (-(3 / 2) : ℝ))).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 4)).const_mul
                                  (-1 / (2 * Real.pi) : ℝ))).add
                            ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 4).const_mul (1 / 4 : ℝ))).add
                        (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).mul
                              ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                          (1 / (2 * Real.pi) : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                      (-1 / (2 * Real.pi) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3, hmc4, hms4]
          have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
          linear_combination (norm := (field_simp; try ring1))
            ((-4 * x_veriphy * ((Real.cos x_veriphy) ^ 4) +
                              (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 4) / Real.pi) +
                            (48 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2)) -
                          (48 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                        (12 * x_veriphy * ((Real.sin x_veriphy) ^ 4)) +
                      (12 * (x_veriphy ^ 2) * ((Real.sin x_veriphy) ^ 4) / Real.pi) +
                    (8 * x_veriphy * ((Real.sin x_veriphy) ^ 2)) -
                  (8 * (x_veriphy ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) +
                                                                    (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 5) *
                                                                        (Real.sin x_veriphy) /
                                                                      Real.pi) +
                                                                  (40 * x_veriphy * ((Real.cos x_veriphy) ^ 3) *
                                                                      ((Real.sin x_veriphy) ^ 3) /
                                                                    3) -
                                                                (40 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) *
                                                                    ((Real.sin x_veriphy) ^ 3) /
                                                                  (3 * Real.pi)) -
                                                              (4 * x_veriphy * (Real.cos x_veriphy) *
                                                                ((Real.sin x_veriphy) ^ 5)) +
                                                            (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) *
                                                                ((Real.sin x_veriphy) ^ 5) /
                                                              Real.pi) +
                                                          (2 * x_veriphy * ((Real.cos x_veriphy) ^ 6) / (9 * Real.pi)) -
                                                        (((Real.cos x_veriphy) ^ 6) / 9) -
                                                      (10 * x_veriphy * ((Real.cos x_veriphy) ^ 4) *
                                                          ((Real.sin x_veriphy) ^ 2) /
                                                        (3 * Real.pi)) +
                                                    (5 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 2) / 3) +
                                                  (10 * x_veriphy * ((Real.cos x_veriphy) ^ 2) *
                                                      ((Real.sin x_veriphy) ^ 4) /
                                                    (3 * Real.pi)) -
                                                (5 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 4) / 3) -
                                              (2 * x_veriphy * ((Real.sin x_veriphy) ^ 6) / (9 * Real.pi)) +
                                            (((Real.sin x_veriphy) ^ 6) / 9) +
                                          (4 * x_veriphy * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy)) -
                                        (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) /
                                          Real.pi) -
                                      (2 * ((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) / (9 * Real.pi)) -
                                    (4 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) +
                                  (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / Real.pi) +
                                (20 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3) / (27 * Real.pi)) -
                              (2 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5) / (9 * Real.pi)) -
                            (x_veriphy * ((Real.cos x_veriphy) ^ 4) / (2 * Real.pi)) +
                          (((Real.cos x_veriphy) ^ 4) / 4) +
                        (3 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / Real.pi) -
                      (3 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) / 2) -
                    (x_veriphy * ((Real.sin x_veriphy) ^ 4) / (2 * Real.pi)) +
                  (((Real.sin x_veriphy) ^ 4) / 4) +
                (((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) / (2 * Real.pi)) -
              ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3) / (2 * Real.pi)) :
            ℝ))
          (f' := fun t => 2 * t * (2 * π - 2 * t) * (1 + 2 * cos (2 * t) + 2 * cos (4 * t)) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (field_simp; ring1))
  -- certificate: fa-trig-integral.sage

/-- The coefficient on `χ_{1/2}` vanishes (half-integer spins do not occur). -/
theorem class_coeff_half :
    ∫ t in (0 : ℝ)..π, 2 * t * (2 * π - 2 * t) * (2 * Real.cos t) * (2 / π * Real.sin t ^ 2) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) +
                                            (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) /
                                              Real.pi) +
                                          (4 * x_veriphy * ((Real.sin x_veriphy) ^ 3) / 3) -
                                        (4 * (x_veriphy ^ 2) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) +
                                      (8 * x_veriphy * ((Real.cos x_veriphy) ^ 3) / (9 * Real.pi)) -
                                    (4 * ((Real.cos x_veriphy) ^ 3) / 9) -
                                  (8 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / (3 * Real.pi)) +
                                (4 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / 3) +
                              (4 * x_veriphy * (Real.sin x_veriphy)) -
                            (4 * (x_veriphy ^ 2) * (Real.sin x_veriphy) / Real.pi) -
                          (8 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / (9 * Real.pi)) +
                        (8 * ((Real.sin x_veriphy) ^ 3) / (27 * Real.pi)) -
                      (8 * x_veriphy * (Real.cos x_veriphy) / Real.pi) +
                    (8 * (Real.sin x_veriphy) / Real.pi) +
                  (4 * (Real.cos x_veriphy)) :
                ℝ))
            ((fun t => 2 * t * (2 * π - 2 * t) * (2 * cos t) * (2 / π * sin t ^ 2)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2)).mul
                                                                                ((hasDerivAt_id'
                                                                                    (x_veriphy : ℝ)).sin)).const_mul
                                                                            (-4 : ℝ)).add
                                                                        (((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                                                                  (((hasDerivAt_id'
                                                                                          (x_veriphy : ℝ)).cos).pow
                                                                                    2)).mul
                                                                              ((hasDerivAt_id'
                                                                                  (x_veriphy : ℝ)).sin)).const_mul
                                                                          (4 / Real.pi : ℝ))).add
                                                                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                            3)).const_mul
                                                                      (4 / 3 : ℝ))).add
                                                                ((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow
                                                                        3)).const_mul
                                                                  (-4 / (3 * Real.pi) : ℝ))).add
                                                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                                  (((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow
                                                                    3)).const_mul
                                                              (8 / (9 * Real.pi) : ℝ))).add
                                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul
                                                          (-(4 / 9) : ℝ))).add
                                                    ((((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                              ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).mul
                                                          (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                                                      (-8 / (3 * Real.pi) : ℝ))).add
                                                ((((hasDerivAt_id' (x_veriphy : ℝ)).cos).mul
                                                      (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 2)).const_mul
                                                  (4 / 3 : ℝ))).add
                                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul
                                                  ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                              (4 : ℝ))).add
                                        ((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).mul
                                              ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                          (-4 / Real.pi : ℝ))).add
                                    (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                          ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                      (-8 / (9 * Real.pi) : ℝ))).add
                                ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3).const_mul (8 / (27 * Real.pi) : ℝ))).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).const_mul
                              (-8 / Real.pi : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (8 / Real.pi : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (4 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
          linear_combination (norm := (field_simp; try ring1))
            ((-4 * x_veriphy * (Real.cos x_veriphy) + (4 * (x_veriphy ^ 2) * (Real.cos x_veriphy) / Real.pi) : ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-4 * x_veriphy * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) +
                                        (4 * (x_veriphy ^ 2) * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) /
                                          Real.pi) +
                                      (4 * x_veriphy * ((Real.sin x_veriphy) ^ 3) / 3) -
                                    (4 * (x_veriphy ^ 2) * ((Real.sin x_veriphy) ^ 3) / (3 * Real.pi)) +
                                  (8 * x_veriphy * ((Real.cos x_veriphy) ^ 3) / (9 * Real.pi)) -
                                (4 * ((Real.cos x_veriphy) ^ 3) / 9) -
                              (8 * x_veriphy * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / (3 * Real.pi)) +
                            (4 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / 3) +
                          (4 * x_veriphy * (Real.sin x_veriphy)) -
                        (4 * (x_veriphy ^ 2) * (Real.sin x_veriphy) / Real.pi) -
                      (8 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) / (9 * Real.pi)) +
                    (8 * ((Real.sin x_veriphy) ^ 3) / (27 * Real.pi)) -
                  (8 * x_veriphy * (Real.cos x_veriphy) / Real.pi) +
                (8 * (Real.sin x_veriphy) / Real.pi) +
              (4 * (Real.cos x_veriphy)) :
            ℝ))
          (f' := fun t => 2 * t * (2 * π - 2 * t) * (2 * cos t) * (2 / π * sin t ^ 2)) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals (field_simp; ring1))
  -- certificate: fa-trig-integral.sage

/-- The cosine coefficients `∫₀^{2π} θ(2π - θ) cos nθ dθ = -4π/n²` behind the general coefficient `-2(2j+1)/(j²(j+1)²)`. -/
theorem quadratic_cos_coeff (n : ℕ) (hn : n ≠ 0) :
    ∫ θ in (0 : ℝ)..(2 * π), θ * (2 * π - θ) * Real.cos (n * θ) = -4 * π / n ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hF : ∀ x ∈ Set.uIcc 0 (2 * π), HasDerivAt
      (fun x => x * (2 * π - x) * Real.sin (n * x) / n + (2 * π - 2 * x) * Real.cos (n * x) / n ^ 2 + 2 * Real.sin (n * x) / n ^ 3)
      (x * (2 * π - x) * Real.cos (n * x)) x := by
    intro x _
    have hs : HasDerivAt (fun x => Real.sin (n * x)) (Real.cos (n * x) * n) x := by
      simpa using ((hasDerivAt_id x).const_mul (n : ℝ)).sin
    have hc : HasDerivAt (fun x => Real.cos (n * x)) (-Real.sin (n * x) * n) x := by
      simpa using ((hasDerivAt_id x).const_mul (n : ℝ)).cos
    have e1 := (((hasDerivAt_id' x).mul ((hasDerivAt_id' x).const_sub (2 * π))).mul hs).div_const (n : ℝ)
    have e2 := ((((hasDerivAt_id' x).const_mul 2).const_sub (2 * π)).mul hc).div_const ((n : ℝ) ^ 2)
    have e3 := (hs.const_mul 2).div_const ((n : ℝ) ^ 3)
    convert (e1.add e2).add e3 using 1
    all_goals fa_close4
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  have h1 : Real.sin (n * (2 * π)) = 0 := by
    rw [show (n : ℝ) * (2 * π) = ((2 * n : ℕ) : ℝ) * π by push_cast; ring]; exact Real.sin_nat_mul_pi _
  have h2 : Real.cos (n * (2 * π)) = 1 := Real.cos_nat_mul_two_pi n
  simp only [mul_zero, Real.sin_zero, Real.cos_zero, h1, h2]
  field_simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.PeterWeyl.quadratic_cos_coeff
-- test: fa-quadratic-cos-coeff-test.sage

end KaluzaKlein.FourierAnalysis.PeterWeyl

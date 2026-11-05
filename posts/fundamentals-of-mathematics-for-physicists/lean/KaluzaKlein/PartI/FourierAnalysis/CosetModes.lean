import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.PeterWeyl

/-! # Fourier analysis: coset modes (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.CosetModes

open Real MeasureTheory Filter Topology

/-! ## def-coset-mode-expansion, exr-sph-coset-singlets, exm-sph-coset-s2 -/

/-- The spin-`j` representation (`2j = k`) restricted to `U(1)` contains the trivial weight `0` exactly when `j` is an
integer: some weight `j - i`, `0 ≤ i ≤ 2j`, vanishes iff `k` is even. -/
theorem zero_weight (k : ℕ) : (∃ i ∈ Finset.range (k + 1), (k : ℤ) - 2 * i = 0) ↔ Even k := by
  constructor
  · rintro ⟨i, -, hi⟩
    exact ⟨i, by omega⟩
  · rintro ⟨i, rfl⟩
    exact ⟨i, Finset.mem_range.2 (by omega), by push_cast; ring⟩

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.zero_weight (0) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.zero_weight (4) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.zero_weight (1) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.CosetModes.zero_weight
-- test: fa-zero-weight-test.sage

/-- The antipodal map `(θ, φ) ↦ (π - θ, φ + π)` multiplies the `l = 2` harmonics by `(-1)² = 1` and the `l = 1` ones by
`-1` (here `Θ₂₁ cos φ` and `Θ₁₁ cos φ`): only even `l` live on `ℝP²`. -/
theorem antipodal_parity (θ φ : ℝ) :
    3 * Real.cos (π - θ) * Real.sin (π - θ) * Real.cos (φ + π) = 3 * Real.cos θ * Real.sin θ * Real.cos φ ∧
      Real.sin (π - θ) * Real.cos (φ + π) = -(Real.sin θ * Real.cos φ) ∧ Real.cos (π - θ) = -Real.cos θ := by
  rw [Real.cos_pi_sub, Real.sin_pi_sub, Real.cos_add_pi]
  refine ⟨by ring, by ring, rfl⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.CosetModes.antipodal_parity
-- test: fa-antipodal-parity-test.sage

/-- The `ℝP²` spectrum: `-l(l+1) = 0, -6, -20` with multiplicities `2l + 1 = 1, 5, 9` for `l = 0, 2, 4`. -/
theorem rp2_spectrum (l : ℕ) (hl : Even l) : Even (l * (l + 1)) ∧ 2 * l + 1 ≥ 1 := by
  exact ⟨Nat.even_mul.2 (Or.inl hl), by omega⟩

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.rp2_spectrum (0) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.rp2_spectrum (2) ((by decide)) := by
  decide

example : veriphy_instance% KaluzaKlein.FourierAnalysis.CosetModes.rp2_spectrum (4) ((by decide)) := by
  decide

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.CosetModes.rp2_spectrum
-- test: fa-rp2-spectrum-test.sage

/-! ## exm-sph-coset-torus -/

/-- On the flat torus, `e^{i(n₁φ₁ + n₂φ₂)}` has eigenvalue `-(n₁²/R₁² + n₂²/R₂²)` (second derivatives in the arc
lengths `R₁φ₁`, `R₂φ₂`). -/
theorem torus_mode (n₁ n₂ R₁ R₂ x y : ℝ) (h1 : R₁ ≠ 0) (h2 : R₂ ≠ 0) :
    deriv (deriv fun x : ℝ => Real.cos (n₁ * x / R₁)) x + deriv (deriv fun y : ℝ => Real.cos (n₂ * y / R₂)) y =
      -(n₁ ^ 2 / R₁ ^ 2) * Real.cos (n₁ * x / R₁) - n₂ ^ 2 / R₂ ^ 2 * Real.cos (n₂ * y / R₂) := by
  have hd : ∀ (n R : ℝ), R ≠ 0 → ∀ x, deriv (deriv fun x : ℝ => Real.cos (n * x / R)) x = -(n ^ 2 / R ^ 2) * Real.cos (n * x / R) := by
    intro n R hR x
    have h1 : deriv (fun x : ℝ => Real.cos (n * x / R)) = fun x => -(n / R) * Real.sin (n * x / R) := by
      funext x
      exact ((((hasDerivAt_id' x).const_mul n).div_const R).cos).deriv.trans (by ring)
    rw [h1]
    exact (((((hasDerivAt_id' x).const_mul n).div_const R).sin).const_mul (-(n / R))).deriv.trans (by ring)
  rw [hd n₁ R₁ h1, hd n₂ R₂ h2]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.CosetModes.torus_mode
-- test: fa-torus-mode-test.sage

/-! ## exm-sph-coset-s3 -/

/-- A half-integer spin mode is odd under `ψ ↦ ψ + 2π`: `e^{-in(ψ+2π)/2} = (-1)^n e^{-inψ/2}`. -/
theorem half_spin_odd (n : ℤ) (ψ : ℝ) :
    Complex.exp (-(n * (ψ + 2 * π) / 2) * Complex.I) = (-1) ^ n * Complex.exp (-(n * ψ / 2) * Complex.I) := by
  rw [show -((n : ℂ) * ((ψ : ℂ) + 2 * π) / 2) * Complex.I = -(n * ψ / 2) * Complex.I + ((-n : ℤ) : ℂ) * (π * Complex.I) by
    push_cast; ring, Complex.exp_add, Complex.exp_int_mul, Complex.exp_pi_mul_I, mul_comm]
  congr 1
  rw [zpow_neg, ← inv_zpow, inv_neg, inv_one]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.CosetModes.half_spin_odd
-- test: fa-half-spin-odd-test.sage

/-! ## exr-sph-coset-modes -/

/-- `∫ |D¹₀₀|² dΩ = 2π ∫₀^π cos² θ sin θ dθ = 4π/3` and `∫ |D¹±₁,₀|² dΩ = 2π ∫₀^π (sin² θ / 2) sin θ dθ = 4π/3`. -/
theorem wigner_column_norms :
    ∫ t in (0 : ℝ)..π, Real.cos t ^ 2 * Real.sin t = 2 / 3 ∧ ∫ t in (0 : ℝ)..π, Real.sin t ^ 2 / 2 * Real.sin t = 2 / 3 := by
  constructor
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
            HasDerivAt (fun x_veriphy => (-((Real.cos x_veriphy) ^ 3) / 3 : ℝ)) ((fun t => cos t ^ 2 * sin t) x_veriphy)
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
            (f' := fun t => cos t ^ 2 * sin t) hF (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
        have he1 : Real.cos (((π) : ℝ)) = -1 := by
          rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
        simp only [he0, he1]
        all_goals (norm_num; done))
    -- certificate: fa-trig-integral.sage
  · veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
            HasDerivAt (fun x_veriphy => (((Real.cos x_veriphy) ^ 3) / 6 - ((Real.cos x_veriphy) / 2) : ℝ))
              ((fun t => sin t ^ 2 / 2 * sin t) x_veriphy) x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          refine
            (((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 3).const_mul (1 / 6 : ℝ)).add
                      (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul (-(1 / 2) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · (try simp only [Pi.pow_apply, Pi.mul_apply])
            linear_combination ((-(Real.sin x_veriphy) / 2 : ℝ)) * Real.sin_sq_add_cos_sq x_veriphy
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (((Real.cos x_veriphy) ^ 3) / 6 - ((Real.cos x_veriphy) / 2) : ℝ)) (f' := fun t => sin t ^ 2 / 2 * sin t) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
        have he1 : Real.cos (((π) : ℝ)) = -1 := by
          rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
        simp only [he0, he1]
        all_goals (norm_num; done))
    -- certificate: fa-trig-integral.sage

end KaluzaKlein.FourierAnalysis.CosetModes

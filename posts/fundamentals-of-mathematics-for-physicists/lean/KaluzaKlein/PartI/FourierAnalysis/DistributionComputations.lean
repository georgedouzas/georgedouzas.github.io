import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.DistributionTransform

/-! # Fourier analysis: distribution computations (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.DistributionComputations

open Real MeasureTheory Filter Topology

/-! ## prp-distribution-identities, exm-dist-sage-limit -/

/-- The limits `e^{-ε²/4} → 1` and `1 + ε²/2 → 1` as `ε → 0`. -/
theorem gauss_limits (c : ℝ) : Tendsto (fun ε : ℝ => c * Real.exp (-ε ^ 2 / 4)) (𝓝 0) (𝓝 c) ∧
    Tendsto (fun ε : ℝ => c * (1 + ε ^ 2 / 2)) (𝓝 0) (𝓝 c) := by
  constructor
  · have : Continuous fun ε : ℝ => c * Real.exp (-ε ^ 2 / 4) := by fun_prop
    simpa using this.tendsto 0
  · have : Continuous fun ε : ℝ => c * (1 + ε ^ 2 / 2) := by fun_prop
    simpa using this.tendsto 0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.gauss_limits
-- test: fa-gauss-limits-test.sage

/-- The Fourier kernel: `(1/2π) ∫_{-L}^{L} e^{ikx} dk = sin(Lx)/(πx)` (real part; the sine part vanishes). -/
theorem fourier_kernel (L x : ℝ) (hx : x ≠ 0) :
    1 / (2 * π) * ∫ k in (-L)..L, Real.cos (x * k) = Real.sin (L * x) / (π * x) := by
  rw [intervalIntegral.integral_comp_mul_left (fun t => Real.cos t) hx, integral_cos, smul_eq_mul, mul_neg, Real.sin_neg]
  have : π ≠ 0 := Real.pi_ne_zero
  rw [mul_comm x L]; field_simp; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.fourier_kernel
-- test: fa-fourier-kernel-test.sage

/-! ## exm-dist-sage-pv -/

/-- `[arctan(x/ε)]_{-∞}^{∞} = π`. -/
theorem arctan_limits (ε : ℝ) (hε : 0 < ε) :
    Tendsto (fun R : ℝ => Real.arctan (R / ε) - Real.arctan (-R / ε)) atTop (𝓝 π) := by
  have h1 := (Real.tendsto_arctan_atTop.mono_right nhdsWithin_le_nhds).comp (tendsto_id.atTop_div_const hε)
  have h2 := (Real.tendsto_arctan_atBot.mono_right nhdsWithin_le_nhds).comp
    ((tendsto_neg_atTop_atBot).atBot_div_const hε)
  have := h1.sub h2
  rw [show π / 2 - -(π / 2) = π by ring] at this
  exact this.congr (fun R => by simp [Function.comp, neg_div])

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.arctan_limits
-- test: fa-arctan-limits-test.sage

/-! ## exr-dist-delta-prime -/

/-- The dipole and derivative sequences against `x, x³, sin x`, as stated (tested, not proved): `∫ g_ε' x³ = -3ε²/2`,
`∫ d_ε x³ = -3ε²/2 - ε²`, and both give `-1` on `x` and `-e^{-ε²/4}`-type values on `sin x`. -/
def DipoleMoments : Prop :=
  ∀ ε : ℝ, 0 < ε →
    let g : ℝ → ℝ := fun x => Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π)
    let d : ℝ → ℝ := fun x => (g (x + ε) - g (x - ε)) / (2 * ε)
    (∫ x, deriv g x * x ^ 3) = -3 / 2 * ε ^ 2 ∧ (∫ x, d x * x ^ 3) = -3 / 2 * ε ^ 2 - ε ^ 2 ∧
    (∫ x, d x * x) = -1 ∧ (∫ x, deriv g x * x) = -1

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.DipoleMoments
-- test: fa-dipole-moments-test.sage

/-! ## exr-dist-heat-kernel-delta -/

/-- `∫ K_t cos x = e^{-κt} → 1`, `∫ K_t (x² + x + 1) = 1 + 2κt → 1` and `∫ K_t x² = 2κt`: the heat kernel is a delta
sequence whose second moment grows linearly in `t` (with `K_t = g_ε`, `ε = 2√(κt)`). -/
theorem heat_kernel_moments (κ t : ℝ) (hκ : 0 < κ) (ht : 0 < t) :
    (∫ x : ℝ, Real.exp (-x ^ 2 / (2 * Real.sqrt (κ * t)) ^ 2) / ((2 * Real.sqrt (κ * t)) * Real.sqrt π) * Real.cos (1 * x)) =
      Real.exp (-(κ * t)) ∧
    (∫ x : ℝ, Real.exp (-x ^ 2 / (2 * Real.sqrt (κ * t)) ^ 2) / ((2 * Real.sqrt (κ * t)) * Real.sqrt π) * (x ^ 2 + 1)) =
      1 + 2 * κ * t := by
  have hs : 0 < 2 * Real.sqrt (κ * t) := by positivity
  have hsq : (2 * Real.sqrt (κ * t)) ^ 2 = 4 * (κ * t) := by rw [mul_pow, Real.sq_sqrt (by positivity)]; norm_num
  constructor
  · rw [KaluzaKlein.FourierAnalysis.Distributions.gauss_cos _ 1 hs, hsq]; congr 1; ring
  · rw [KaluzaKlein.FourierAnalysis.Distributions.gauss_poly _ hs, hsq]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.heat_kernel_moments
-- test: fa-heat-kernel-moments-test.sage

/-! ## exr-dist-laplacian-delta -/

/-- `1/r` is harmonic away from the origin: its radial Laplacian `r⁻² (r² (1/r)')'` vanishes, since `r² (1/r)' = -1`. -/
theorem inv_r_radial (r : ℝ) (hr : r ≠ 0) : r ^ 2 * deriv (fun r : ℝ => 1 / r) r = -1 := by
  have := (hasDerivAt_inv hr)
  rw [show (fun r : ℝ => 1 / r) = fun r => r⁻¹ by funext; ring, this.deriv]
  field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.inv_r_radial
-- test: fa-inv-r-radial-test.sage

/-- The Yukawa transform `4π/(k² + m²)` tends to `4π/k²` as `m → 0`: the transform of `1/r`. -/
theorem yukawa_to_coulomb (k : ℝ) (hk : k ≠ 0) :
    Tendsto (fun m : ℝ => 4 * π / (k ^ 2 + m ^ 2)) (𝓝 0) (𝓝 (4 * π / k ^ 2)) := by
  have hc : ContinuousAt (fun m : ℝ => 4 * π / (k ^ 2 + m ^ 2)) 0 := by
    apply ContinuousAt.div continuousAt_const (by fun_prop); simp; positivity
  simpa using hc.tendsto

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.yukawa_to_coulomb
-- test: fa-yukawa-to-coulomb-test.sage

/-- `∇²(1/r) = -4πδ³`: the sphere integral `-∮ (ε ∂_r φ + φ) dΩ → -4π φ(0)`, here for the radial
`φ = c e^{-r²}`, on whose spheres the integrand is constant. -/
theorem coulomb_delta (c : ℝ) :
    Tendsto (fun ε : ℝ => -(4 * π) * (ε * (-2 * ε * c * Real.exp (-ε ^ 2)) + c * Real.exp (-ε ^ 2))) (𝓝[>] 0)
      (𝓝 (-4 * π * c)) := by
  have : Continuous fun ε : ℝ => -(4 * π) * (ε * (-2 * ε * c * Real.exp (-ε ^ 2)) + c * Real.exp (-ε ^ 2)) := by fun_prop
  have h := (this.tendsto 0).mono_left (nhdsWithin_le_nhds (s := Set.Ioi 0))
  simpa using h

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionComputations.coulomb_delta
-- test: fa-coulomb-delta-test.sage

end KaluzaKlein.FourierAnalysis.DistributionComputations

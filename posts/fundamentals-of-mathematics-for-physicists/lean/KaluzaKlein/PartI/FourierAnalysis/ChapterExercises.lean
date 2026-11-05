import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.HarmonicComputations

/-! # Fourier analysis: chapter exercises (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.ChapterExercises

open Real MeasureTheory Filter Topology

/-! ## exr-fa-heat-circle -/

/-- Each mode `c e^{-n²t}` solves `u' = -n² u`. -/
theorem heat_mode (c n t : ℝ) : deriv (fun t : ℝ => c * Real.exp (-n ^ 2 * t)) t = -n ^ 2 * (c * Real.exp (-n ^ 2 * t)) := by
  veriphy_certified
    ( have hd : HasDerivAt (fun t => c * rexp (-n ^ 2 * t)) ((-(n ^ 2) * c * (Real.exp (-(n ^ 2) * t)) : ℝ)) (t) :=
        by
        convert ((((hasDerivAt_id' (t : ℝ)).const_mul (-(n ^ 2) : ℝ)).exp).const_mul (c : ℝ)) using 1
        all_goals ring1
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv2.sage

/-- The theta kernel equals the periodised heat kernel, `(1/2π) Σ e^{-n²t} e^{inx} = Σ_k e^{-(x - 2πk)²/4t}/√(4πt)`, as
stated (tested, not proved: Poisson summation with a shift). -/
def ThetaIdentity : Prop :=
  ∀ t x : ℝ, 0 < t →
    1 / (2 * π) * ∑' n : ℤ, Real.exp (-(n : ℝ) ^ 2 * t) * Real.cos (n * x) =
      ∑' k : ℤ, Real.exp (-(x - 2 * π * k) ^ 2 / (4 * t)) / Real.sqrt (4 * π * t)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ChapterExercises.ThetaIdentity
-- test: fa-theta-identity-test.sage

/-! ## exr-fa-multipole -/

/-- `|x - x'|² = r² + r'² - 2 r r' cos γ` for `x = r n`, `x' = r' n'`, `cos γ = n · n'`. -/
theorem law_of_cosines (r r' : ℝ) (n n' : EuclideanSpace ℝ (Fin 3)) (hn : ‖n‖ = 1) (hn' : ‖n'‖ = 1) :
    ‖r • n - r' • n'‖ ^ 2 = r ^ 2 + r' ^ 2 - 2 * r * r' * inner ℝ n n' := by
  rw [norm_sub_sq_real, norm_smul, norm_smul, hn, hn', inner_smul_left, inner_smul_right]
  simp [Real.norm_eq_abs, sq_abs]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ChapterExercises.law_of_cosines
-- test: fa-law-of-cosines-test.sage

/-- The generating function `(1 - 2ut + t²)^{-1/2} = Σ P_l(u) tˡ` to order `t³`: the Taylor coefficients are
`1, u, (3u² - 1)/2, (5u³ - 3u)/2`, as stated (tested, not proved). -/
def LegendreGenerating : Prop :=
  ∀ u : ℝ, |u| ≤ 1 →
    iteratedDeriv 2 (fun t : ℝ => (1 - 2 * u * t + t ^ 2) ^ (-(1 : ℝ) / 2)) 0 / 2 = (3 * u ^ 2 - 1) / 2 ∧
    iteratedDeriv 3 (fun t : ℝ => (1 - 2 * u * t + t ^ 2) ^ (-(1 : ℝ) / 2)) 0 / 6 = (5 * u ^ 3 - 3 * u) / 2

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ChapterExercises.LegendreGenerating
-- test: fa-legendre-generating-test.sage

/-! ## exr-fa-series-to-transform -/

/-- The tail of the Gaussian: `∫_L^∞ e^{-x²/2} ≤ e^{-L²/2}` for `L ≥ 1`, so `2L c_n^{(L)}` differs from `f̂(πn/L)` by
an exponentially small amount. -/
theorem gaussian_tail (L : ℝ) (hL : 1 ≤ L) : ∫ x in Set.Ioi L, Real.exp (-x ^ 2 / 2) ≤ Real.exp (-L ^ 2 / 2) := by
  have hderiv : ∀ x ∈ Set.Ici L, HasDerivAt (fun x => -Real.exp (-x ^ 2 / 2)) (x * Real.exp (-x ^ 2 / 2)) x := by
    intro x _
    have := ((hasDerivAt_pow 2 x).neg.div_const 2).exp.neg
    convert this using 1; simp; ring
  have hint : IntegrableOn (fun x => x * Real.exp (-x ^ 2 / 2)) (Set.Ioi L) := by
    have := (integrable_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)).integrableOn (s := Set.Ioi L)
    refine this.congr_fun (fun x _ => ?_) measurableSet_Ioi
    simp only; ring_nf
  have hlim : Tendsto (fun x => -Real.exp (-x ^ 2 / 2)) atTop (𝓝 0) := by
    have : Tendsto (fun x : ℝ => -x ^ 2 / 2) atTop atBot := by
      have := (tendsto_pow_atTop (two_ne_zero)).atTop_div_const (show (0 : ℝ) < 2 by norm_num)
      simpa [neg_div] using this
    simpa using (Real.tendsto_exp_atBot.comp this).neg
  have hval := integral_Ioi_of_hasDerivAt_of_tendsto' hderiv hint hlim
  have hle : ∫ x in Set.Ioi L, Real.exp (-x ^ 2 / 2) ≤ ∫ x in Set.Ioi L, x * Real.exp (-x ^ 2 / 2) := by
    apply setIntegral_mono_on _ hint measurableSet_Ioi
    · intro x hx
      have : 1 ≤ x := le_trans hL (le_of_lt hx)
      nlinarith [Real.exp_pos (-x ^ 2 / 2)]
    · refine Integrable.mono' (hint) (by fun_prop) (ae_restrict_of_forall_mem measurableSet_Ioi fun x hx => ?_)
      have : 1 ≤ x := le_trans hL (le_of_lt hx)
      rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
      nlinarith [Real.exp_pos (-x ^ 2 / 2)]
  rw [hval] at hle
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ChapterExercises.gaussian_tail
-- test: fa-gaussian-tail-test.sage

/-- The interval's Dirichlet kernel becomes the Fourier kernel: `(1/2L) sin(Kx)/sin(πx/2L) → sin(Kx)/(πx)`, as stated
(tested, not proved). -/
def IntervalKernelLimit : Prop :=
  ∀ K x : ℝ, x ≠ 0 → Tendsto (fun L : ℝ => 1 / (2 * L) * (Real.sin (K * x) / Real.sin (π * x / (2 * L)))) atTop
    (𝓝 (Real.sin (K * x) / (π * x)))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.ChapterExercises.IntervalKernelLimit
-- test: fa-interval-kernel-limit-test.sage

end KaluzaKlein.FourierAnalysis.ChapterExercises

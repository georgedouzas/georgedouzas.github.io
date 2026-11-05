import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The Gamma function"

The claims of the section "The Gamma function" of `tutorial/part-i-04-complex-analysis.qmd`: the Euler integral
(convergent exactly for `Re z > 0`), the recursion and `Γ(n + 1) = n!`, holomorphy off the poles, the residues
`(-1)ⁿ/n!` at `-n`, the absence of zeros and the entire function `1/Γ`, the reflection formula, the half-integer
values `Γ(n + 1/2) = (2n)!/(4ⁿ n!) √π`, the Gaussian moments and the Laplace transform of `t^{a-1}`.
-/

namespace KaluzaKlein.ComplexAnalysis.GammaFunction

open Complex Filter Topology Set MeasureTheory
open scoped Nat Real

/-! ## def-gamma-function, exr-gamma-convergence -/

/-- For `Re z > 0` the Euler integral `∫₀^∞ t^{z-1} e^{-t} dt` converges absolutely and equals `Γ(z)`. -/
theorem gamma_integral (z : ℂ) (hz : 0 < z.re) :
    IntegrableOn (fun t : ℝ => (t : ℂ) ^ (z - 1) * Complex.exp (-t)) (Ioi 0) ∧
      Complex.Gamma z = ∫ t in Ioi (0 : ℝ), (t : ℂ) ^ (z - 1) * Complex.exp (-t) := by
  have hfun : (fun t : ℝ => (t : ℂ) ^ (z - 1) * Complex.exp (-t)) =
      fun t : ℝ => ((Real.exp (-t) : ℝ) : ℂ) * (t : ℂ) ^ (z - 1) := by
    funext t
    rw [Complex.ofReal_exp, Complex.ofReal_neg, mul_comm]
  rw [hfun]
  exact ⟨Complex.GammaIntegral_convergent hz, Complex.Gamma_eq_integral hz⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_integral
-- test: ca-spec-gamma-int-test.sage

/-- `|t^{z-1}| = t^{Re z - 1}` for `t > 0`. -/
theorem norm_cpow_real (t : ℝ) (ht : 0 < t) (z : ℂ) : ‖(t : ℂ) ^ (z - 1)‖ = t ^ (z.re - 1) := by
  rw [Complex.norm_cpow_eq_rpow_re_of_pos ht]
  simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.norm_cpow_real
-- test: ca-spec-norm-cpow-test.sage

/-- The Euler integral converges absolutely exactly when `Re z > 0` (near `0` it behaves like `t^{Re z - 1}`). -/
theorem gamma_integral_integrable_iff (z : ℂ) :
    IntegrableOn (fun t : ℝ => (t : ℂ) ^ (z - 1) * Complex.exp (-t)) (Ioi 0) ↔ 0 < z.re := by
  refine ⟨fun h => ?_, fun hz => (gamma_integral z hz).1⟩
  by_contra hz
  push_neg at hz
  have hn : IntegrableOn (fun t : ℝ => ‖(t : ℂ) ^ (z - 1) * Complex.exp (-t)‖ * Real.exp 1) (Ioo 0 1) :=
    (h.mono_set Ioo_subset_Ioi_self).norm.mul_const _
  have h1 : IntegrableOn (fun t : ℝ => t ^ (z.re - 1)) (Ioo 0 1) := by
    refine Integrable.mono' hn ((measurable_id.pow_const _).aestronglyMeasurable) ?_
    refine ae_restrict_of_forall_mem measurableSet_Ioo fun t ht => ?_
    have ht0 : 0 < t := ht.1
    have hp : 0 ≤ t ^ (z.re - 1) := Real.rpow_nonneg ht0.le _
    rw [norm_mul, norm_cpow_real t ht0, Complex.norm_exp, Real.norm_eq_abs, abs_of_nonneg hp]
    have hre : (-(t : ℂ)).re = -t := by simp
    rw [hre]
    have he : 1 ≤ Real.exp (-t) * Real.exp 1 := by
      rw [← Real.exp_add]
      exact Real.one_le_exp (by linarith [ht.2])
    calc t ^ (z.re - 1) = t ^ (z.re - 1) * 1 := (mul_one _).symm
      _ ≤ t ^ (z.re - 1) * (Real.exp (-t) * Real.exp 1) := mul_le_mul_of_nonneg_left he hp
      _ = t ^ (z.re - 1) * Real.exp (-t) * Real.exp 1 := by ring
  have := (intervalIntegral.integrableOn_Ioo_rpow_iff one_pos).mp h1
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_integral_integrable_iff
-- test: ca-spec-gamma-conv-test.sage

/-- `Γ` is holomorphic at every `z ∉ {0, -1, -2, …}` (in particular on `Re z > 0`). -/
theorem gamma_differentiableAt (z : ℂ) (hz : ∀ m : ℕ, z ≠ -m) : DifferentiableAt ℂ Complex.Gamma z :=
  Complex.differentiableAt_Gamma z hz

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_differentiableAt (1) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  exact Complex.differentiableAt_Gamma _ (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_differentiableAt (1 / 2) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  exact Complex.differentiableAt_Gamma _ (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_differentiableAt (2 + Complex.I) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  exact Complex.differentiableAt_Gamma _ (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_differentiableAt
-- test: ca-spec-gamma-hol-test.sage

/-- The recursion `Γ(z + 1) = z Γ(z)`. -/
theorem gamma_add_one (z : ℂ) (hz : z ≠ 0) : Complex.Gamma (z + 1) = z * Complex.Gamma z :=
  Complex.Gamma_add_one z hz

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_add_one
-- test: ca-spec-gamma-rec-test.sage

/-- `Γ(n + 1) = n!`. -/
theorem gamma_nat (n : ℕ) : Complex.Gamma (n + 1) = n ! :=
  Complex.Gamma_nat_eq_factorial n

attribute [veriphy_tested oracle_test 0 11] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_nat
-- test: ca-spec-gamma-fact-test.sage

/-- `Γ(z) = Γ(z + n + 1)/(z(z + 1)⋯(z + n))` (the division form). -/
theorem gamma_eq_div_prod' (z : ℂ) (hz : ∀ m : ℕ, z ≠ -m) (n : ℕ) :
    Complex.Gamma z = Complex.Gamma (z + n + 1) / ∏ k ∈ Finset.range (n + 1), (z + k) := by
  have hz' : ∀ m : ℕ, z + m ≠ 0 := fun m h => hz m (by linear_combination h)
  induction n with
  | zero =>
    have h0 : z ≠ 0 := by simpa using hz' 0
    rw [Finset.prod_range_one]
    simp only [Nat.cast_zero, add_zero]
    rw [Complex.Gamma_add_one z h0, mul_div_cancel_left₀ _ h0]
  | succ n ih =>
    have hn : z + ((n : ℂ) + 1) ≠ 0 := by
      have := hz' (n + 1)
      push_cast at this
      exact this
    have hΓ : Complex.Gamma (z + ((n : ℂ) + 1) + 1) = (z + ((n : ℂ) + 1)) * Complex.Gamma (z + n + 1) := by
      rw [Complex.Gamma_add_one _ hn, add_assoc z (n : ℂ) 1]
    have hp : ∏ k ∈ Finset.range (n + 1), (z + k) ≠ 0 := Finset.prod_ne_zero_iff.mpr fun k _ => hz' k
    rw [Finset.prod_range_succ]
    push_cast
    rw [hΓ, ih, div_eq_div_iff hp (mul_ne_zero hp hn)]
    ring

/-- Running the recursion `n + 1` times: `Γ(z) z(z + 1)⋯(z + n) = Γ(z + n + 1)`, i.e.
`Γ(z) = Γ(z + n + 1)/(z(z + 1)⋯(z + n))`. -/
theorem gamma_eq_div_prod (z : ℂ) (hz : ∀ m : ℕ, z ≠ -m) (n : ℕ) :
    Complex.Gamma z * ∏ k ∈ Finset.range (n + 1), (z + k) = Complex.Gamma (z + n + 1) := by
  have hp : ∏ k ∈ Finset.range (n + 1), (z + k) ≠ 0 :=
    Finset.prod_ne_zero_iff.mpr fun k _ => fun h => hz k (by linear_combination h)
  rw [gamma_eq_div_prod' z hz n, div_mul_cancel₀ _ hp]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_eq_div_prod
-- test: ca-spec-gamma-prod-test.sage

/-- `Γ` has a simple pole at `-n` with residue `(-1)ⁿ/n!`: `(z + n) Γ(z) → (-1)ⁿ/n!`. -/
theorem gamma_residue (n : ℕ) :
    Tendsto (fun z => (z + n) * Complex.Gamma z) (𝓝[≠] (-(n : ℂ))) (𝓝 ((-1) ^ n / n !)) := by
  induction n with
  | zero => simpa using Complex.tendsto_self_mul_Gamma_nhds_zero
  | succ n ih =>
    have hne : (-((n + 1 : ℕ) : ℂ)) ≠ 0 := by
      push_cast
      exact neg_ne_zero.mpr (by exact_mod_cast Nat.succ_ne_zero n)
    have hφ : Tendsto (fun z : ℂ => z + 1) (𝓝[≠] (-((n + 1 : ℕ) : ℂ))) (𝓝[≠] (-(n : ℂ))) := by
      refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
      · have h' : -((n + 1 : ℕ) : ℂ) + 1 = -(n : ℂ) := by push_cast <;> ring
        have : Tendsto (fun z : ℂ => z + 1) (𝓝 (-((n + 1 : ℕ) : ℂ))) (𝓝 (-((n + 1 : ℕ) : ℂ) + 1)) :=
          (continuous_id.add continuous_const).tendsto _
        rw [h'] at this
        exact this.mono_left nhdsWithin_le_nhds
      · filter_upwards [self_mem_nhdsWithin] with z hz
        simp only [mem_compl_iff, mem_singleton_iff] at hz ⊢
        intro h
        apply hz
        push_cast
        linear_combination h
    have hinv : Tendsto (fun z : ℂ => z⁻¹) (𝓝[≠] (-((n + 1 : ℕ) : ℂ))) (𝓝 ((-((n + 1 : ℕ) : ℂ))⁻¹)) :=
      ((continuousAt_inv₀ hne).tendsto).mono_left nhdsWithin_le_nhds
    have hlim := (ih.comp hφ).mul hinv
    have hf : (n ! : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
    have hn1 : (n : ℂ) + 1 ≠ 0 := by exact_mod_cast Nat.succ_ne_zero n
    have hval : (-1 : ℂ) ^ n / n ! * (-((n + 1 : ℕ) : ℂ))⁻¹ = (-1) ^ (n + 1) / ((n + 1)! : ℕ) := by
      rw [Nat.factorial_succ]
      push_cast
      field_simp <;> ring
    rw [hval] at hlim
    refine hlim.congr' ?_
    have hne0 : ∀ᶠ z in 𝓝[≠] (-((n + 1 : ℕ) : ℂ)), z ≠ 0 :=
      (eventually_ne_nhds hne).filter_mono nhdsWithin_le_nhds
    filter_upwards [hne0] with z hz
    simp only [Function.comp]
    rw [Complex.Gamma_add_one z hz]
    push_cast
    field_simp <;> ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_residue
-- test: ca-spec-gamma-res-test.sage

/-- `Γ` has no zeros. -/
theorem gamma_ne_zero (z : ℂ) (hz : ∀ m : ℕ, z ≠ -m) : Complex.Gamma z ≠ 0 :=
  Complex.Gamma_ne_zero hz

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_ne_zero
-- test: ca-spec-gamma-nonzero-test.sage

/-- `1/Γ` is entire. -/
theorem inv_gamma_differentiableAt (z : ℂ) : DifferentiableAt ℂ (fun s => (Complex.Gamma s)⁻¹) z :=
  Complex.differentiable_one_div_Gamma z

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.inv_gamma_differentiableAt (0) := by
  exact Complex.differentiable_one_div_Gamma _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.inv_gamma_differentiableAt (-1) := by
  exact Complex.differentiable_one_div_Gamma _

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.GammaFunction.inv_gamma_differentiableAt (1 / 2) := by
  exact Complex.differentiable_one_div_Gamma _

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.GammaFunction.inv_gamma_differentiableAt
-- test: ca-spec-inv-gamma-test.sage

/-- The reflection formula `Γ(z) Γ(1 - z) = π / sin πz`, i.e. `Γ(z) Γ(1 - z) sin πz = π` for `z ∉ ℤ`. -/
theorem gamma_reflection (z : ℂ) (hz : ∀ n : ℤ, z ≠ n) :
    Complex.Gamma z * Complex.Gamma (1 - z) * Complex.sin (π * z) = π := by
  have hπ : (π : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
  have hs : Complex.sin (π * z) ≠ 0 := by
    rw [Ne, Complex.sin_eq_zero_iff]
    rintro ⟨k, hk⟩
    exact hz k (mul_left_cancel₀ hπ (hk.trans (mul_comm _ _)))
  rw [Complex.Gamma_mul_Gamma_one_sub, div_mul_cancel₀ _ hs]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_reflection
-- test: ca-spec-reflection-test.sage

/-- `Γ(1/2)² = π`, from the reflection formula at `z = 1/2`. -/
theorem gamma_half_sq : Real.Gamma (1 / 2) ^ 2 = π := by
  have h := Real.Gamma_mul_Gamma_one_sub (1 / 2)
  rw [show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num, show π * (1 / 2) = π / 2 by ring, Real.sin_pi_div_two,
    div_one] at h
  generalize Real.Gamma (1 / 2) = G at h ⊢
  generalize π = p at h ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination h)
  -- certificate: ideal_membership.sage

/-- The half-integer values `Γ(n + 1/2) = (2n)!/(4ⁿ n!) √π` (`n = 0`: `Γ(1/2) = √π`). -/
theorem gamma_half_integer (n : ℕ) : Real.Gamma (n + 1 / 2) = (2 * n)! / (4 ^ n * n !) * √π := by
  induction n with
  | zero =>
    rw [Nat.cast_zero, zero_add, Real.Gamma_one_half_eq]
    simp
  | succ n ih =>
    have h := Real.Gamma_add_one (s := (n : ℝ) + 1 / 2) (by positivity)
    rw [show ((n + 1 : ℕ) : ℝ) + 1 / 2 = ((n : ℝ) + 1 / 2) + 1 by push_cast <;> ring, h, ih,
      show 2 * (n + 1) = (2 * n + 1) + 1 by ring, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ]
    push_cast
    field_simp <;> ring

attribute [veriphy_tested oracle_test 0 7] KaluzaKlein.ComplexAnalysis.GammaFunction.gamma_half_integer
-- test: ca-spec-half-int-test.sage

/-- `Γ(-3/2) = (4/3) √π`, running the recursion down from `Γ(1/2) = √π`. -/
theorem gamma_neg_three_halves : Real.Gamma (-3 / 2) = 4 / 3 * √π := by
  have h1 : Real.Gamma (-3 / 2 + 1) = (-3 / 2) * Real.Gamma (-3 / 2) := Real.Gamma_add_one (by norm_num)
  have h2 : Real.Gamma (-1 / 2 + 1) = (-1 / 2) * Real.Gamma (-1 / 2) := Real.Gamma_add_one (by norm_num)
  have h3 : Real.Gamma (1 / 2) = √π := Real.Gamma_one_half_eq
  rw [show (-3 / 2 : ℝ) + 1 = -1 / 2 by norm_num] at h1
  rw [show (-1 / 2 : ℝ) + 1 = 1 / 2 by norm_num] at h2
  generalize Real.Gamma (-3 / 2) = A at h1 ⊢
  generalize Real.Gamma (-1 / 2) = B at h1 h2
  generalize Real.Gamma (1 / 2) = C at h2 h3
  generalize √π = r at h3 ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination (2 / 3) * h1 + (-4 / 3) * h2 + (4 / 3) * h3)
  -- certificate: ideal_membership.sage

/-- `Γ(1/3) Γ(2/3) = π / sin(π/3) = 2π/√3`, by the reflection formula. -/
theorem gamma_third_product : Real.Gamma (1 / 3) * Real.Gamma (2 / 3) = 2 * π / √3 := by
  have h := Real.Gamma_mul_Gamma_one_sub (1 / 3)
  have hs : Real.sin (π * (1 / 3)) = √3 / 2 := by
    rw [show π * (1 / 3) = π / 3 by ring]
    exact Real.sin_pi_div_three
  have h3 : √3 ≠ 0 := by positivity
  rw [hs, show (1 : ℝ) - 1 / 3 = 2 / 3 by norm_num] at h
  rw [eq_div_iff h3]
  field_simp at h
  veriphy_certified (linear_combination h)
  -- certificate: ideal_membership.sage

/-! ## exm-gamma-values, exm-gamma-gaussian-moments -/

/-- The half-line Gaussian `∫₀^∞ e^{-b u²} du = √(π/b)/2` (with `b = 1`: `Γ(1/2) = 2∫₀^∞ e^{-u²} du = √π`). -/
theorem gaussian_half_line (b : ℝ) (hb : 0 < b) : ∫ x in Ioi (0 : ℝ), Real.exp (-b * x ^ 2) = √(π / b) / 2 := by
  exact integral_gaussian_Ioi b

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.GammaFunction.gaussian_half_line
-- test: ca-spec-gauss-half-test.sage

/-- The Gaussian moments `∫₀^∞ t^{2k} e^{-t²} dt = Γ(k + 1/2)/2`. -/
theorem gaussian_moment (k : ℕ) :
    ∫ t in Ioi (0 : ℝ), t ^ (2 * k) * Real.exp (-t ^ 2) = Real.Gamma (k + 1 / 2) / 2 := by
  have h := _root_.integral_rpow_mul_exp_neg_rpow (p := 2) (q := ((2 * k : ℕ) : ℝ)) two_pos (by linarith [Nat.cast_nonneg (α := ℝ) (2 * k)])
  simp_rw [Real.rpow_natCast, Real.rpow_two] at h
  rw [h, show (((2 * k : ℕ) : ℝ) + 1) / 2 = (k : ℝ) + 1 / 2 by push_cast <;> ring]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.gaussian_moment
-- test: ca-spec-gauss-moment-test.sage

/-- `∫₀^∞ t^{a-1} e^{-λt} dt = Γ(a)/λ^a` for `a, λ > 0` (`a = λ = 1`: `Γ(1) = ∫₀^∞ e^{-t} dt = 1`). -/
theorem laplace_gamma (a r : ℝ) (ha : 0 < a) (hr : 0 < r) :
    ∫ t in Ioi (0 : ℝ), t ^ (a - 1) * Real.exp (-(r * t)) = Real.Gamma a / r ^ a := by
  rw [Real.integral_rpow_mul_exp_neg_mul_Ioi ha hr, one_div, Real.inv_rpow hr.le]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.GammaFunction.laplace_gamma
-- test: ca-spec-laplace-test.sage

-- The continuation of `∫₀^∞ t^{a-1} e^{-λt} dt = Γ(a)/λ^a` to complex `λ` with `Re λ > 0` is not formalised: mathlib
-- has the formula for real `λ > 0` only (`Real.integral_rpow_mul_exp_neg_mul_Ioi`, `Complex.integral_cpow_mul_exp_neg_mul_Ioi`).

end KaluzaKlein.ComplexAnalysis.GammaFunction

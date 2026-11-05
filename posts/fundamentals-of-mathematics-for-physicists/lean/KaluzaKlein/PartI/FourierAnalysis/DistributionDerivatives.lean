import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Distributions

/-! # Fourier analysis: distribution derivatives (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.DistributionDerivatives

open Real MeasureTheory Filter Topology

/-! ## def-distributional-derivative -/

/-- For a `C¹` function the distributional derivative is the classical one: `∫ f φ' = -∫ f' φ` when `φ` vanishes
at the ends. -/
theorem classical_derivative (f f' φ φ' : ℝ → ℝ) (a b : ℝ) (hf : ∀ x, HasDerivAt f (f' x) x)
    (hφ : ∀ x, HasDerivAt φ (φ' x) x) (hf' : Continuous f') (hφ' : Continuous φ') (ha : φ a = 0) (hb : φ b = 0) :
    ∫ x in a..b, f x * φ' x = -∫ x in a..b, f' x * φ x := by
  have := intervalIntegral.integral_mul_deriv_eq_deriv_mul (fun x _ => hf x) (fun x _ => hφ x)
    (hf'.intervalIntegrable a b) (hφ'.intervalIntegrable a b)
  rw [this, ha, hb]; simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.classical_derivative
-- test: fa-classical-derivative-test.sage

/-- The Leibniz rule `(ψT)' = ψ'T + ψT'` for a linear functional `T`, with `T'(φ) = -T(φ')` and
`(ψT)(φ) = T(ψφ)`. -/
theorem leibniz_rule (T : (ℝ → ℝ) →ₗ[ℝ] ℝ) (ψ φ : ℝ → ℝ) (hψ : Differentiable ℝ ψ) (hφ : Differentiable ℝ φ) :
    -T (ψ * deriv φ) = T (deriv ψ * φ) + -T (deriv (ψ * φ)) := by
  have h : deriv (ψ * φ) = deriv ψ * φ + ψ * deriv φ := by
    funext x; simp only [Pi.mul_apply, Pi.add_apply]; exact deriv_mul (hψ x) (hφ x)
  rw [h, map_add]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.leibniz_rule
-- test: fa-leibniz-rule-test.sage

/-! ## exm-dist-step-derivative, prp-distribution-identities -/

/-- `θ' = δ`: `-∫₀^∞ φ' = φ(0)` for `φ` decaying at infinity (here also `φ = (1 + x) e^{-x²}`). -/
theorem step_derivative (φ φ' : ℝ → ℝ) (hd : ∀ x, HasDerivAt φ (φ' x) x) (hint : IntegrableOn φ' (Set.Ioi 0))
    (hlim : Tendsto φ atTop (𝓝 0)) : -∫ x in Set.Ioi (0 : ℝ), φ' x = φ 0 := by
  rw [integral_Ioi_of_hasDerivAt_of_tendsto (hd 0).continuousAt.continuousWithinAt (fun x _ => hd x) hint hlim]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.step_derivative
-- test: fa-step-derivative-test.sage

/-- The jump formula: for `f = g` left of `a` and `h` right of `a`, `-∫ f φ' = ∫ {f'} φ + (h(a) - g(a)) φ(a)`. -/
theorem jump_formula (g g' h h' φ φ' : ℝ → ℝ) (a b c : ℝ)
    (hg : ∀ x, HasDerivAt g (g' x) x) (hh : ∀ x, HasDerivAt h (h' x) x) (hφ : ∀ x, HasDerivAt φ (φ' x) x)
    (hg' : Continuous g') (hh' : Continuous h') (hφ' : Continuous φ') (hb : φ b = 0) (hc : φ c = 0) :
    -((∫ x in b..a, g x * φ' x) + ∫ x in a..c, h x * φ' x) =
      (∫ x in b..a, g' x * φ x) + (∫ x in a..c, h' x * φ x) + (h a - g a) * φ a := by
  rw [intervalIntegral.integral_mul_deriv_eq_deriv_mul (fun x _ => hg x) (fun x _ => hφ x)
      (hg'.intervalIntegrable b a) (hφ'.intervalIntegrable b a),
    intervalIntegral.integral_mul_deriv_eq_deriv_mul (fun x _ => hh x) (fun x _ => hφ x)
      (hh'.intervalIntegrable a c) (hφ'.intervalIntegrable a c), hb, hc]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.jump_formula
-- test: fa-jump-formula-test.sage

/-! ## exm-dist-abs-derivative -/

/-- `|x|'' = 2δ`: `∫_{-c}^{c} |x| φ'' = 2 φ(0)` when `φ` and `φ'` vanish at `±c`. -/
theorem abs_second_derivative (φ φ' φ'' : ℝ → ℝ) (c : ℝ) (hc : 0 < c) (hφ : ∀ x, HasDerivAt φ (φ' x) x)
    (hφ' : ∀ x, HasDerivAt φ' (φ'' x) x) (hcont' : Continuous φ') (hcont'' : Continuous φ'')
    (h1 : φ c = 0) (h2 : φ (-c) = 0) (h3 : φ' c = 0) (h4 : φ' (-c) = 0) :
    ∫ x in (-c)..c, |x| * φ'' x = 2 * φ 0 := by
  have hl : ∫ x in (-c)..0, |x| * φ'' x = ∫ x in (-c)..0, -x * φ'' x := by
    apply intervalIntegral.integral_congr; intro x hx
    rw [Set.uIcc_of_le (by linarith)] at hx
    simp [abs_of_nonpos hx.2]
  have hr : ∫ x in (0 : ℝ)..c, |x| * φ'' x = ∫ x in (0 : ℝ)..c, x * φ'' x := by
    apply intervalIntegral.integral_congr; intro x hx
    rw [Set.uIcc_of_le hc.le] at hx
    simp [abs_of_nonneg hx.1]
  rw [← intervalIntegral.integral_add_adjacent_intervals (b := 0) (by apply Continuous.intervalIntegrable; fun_prop)
    (by apply Continuous.intervalIntegrable; fun_prop), hl, hr]
  have pl := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := -c) (b := 0) (u := fun x => -x) (u' := fun _ => -1)
    (fun x _ => (hasDerivAt_id x).neg) (fun x _ => hφ' x) (continuous_const.intervalIntegrable _ _) (hcont''.intervalIntegrable _ _)
  have pr := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := 0) (b := c) (u := fun x => x) (u' := fun _ => 1)
    (fun x _ => hasDerivAt_id x) (fun x _ => hφ' x) (continuous_const.intervalIntegrable _ _) (hcont''.intervalIntegrable _ _)
  rw [pl, pr]
  have il := intervalIntegral.integral_eq_sub_of_hasDerivAt (a := -c) (b := 0) (fun x _ => hφ x) (hcont'.intervalIntegrable _ _)
  have ir := intervalIntegral.integral_eq_sub_of_hasDerivAt (a := 0) (b := c) (fun x _ => hφ x) (hcont'.intervalIntegrable _ _)
  simp only [neg_mul, one_mul, intervalIntegral.integral_neg, neg_neg] at *
  rw [il, ir, h1, h2, h3, h4]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.abs_second_derivative
-- test: fa-abs-second-derivative-test.sage

/-! ## exm-dist-delta-derivative, exr-dist-product-rule -/

/-- `x δ' = -δ`: `-(xφ)'(0) = -φ(0)`. -/
theorem x_delta_prime (φ : ℝ → ℝ) (hφ : DifferentiableAt ℝ φ 0) : -(deriv (fun x => x * φ x) 0) = -φ 0 := by
  rw [show deriv (fun x => x * φ x) 0 = 1 * φ 0 + 0 * deriv φ 0 from ((hasDerivAt_id' 0).mul hφ.hasDerivAt).deriv]
  simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.x_delta_prime
-- test: fa-x-delta-prime-test.sage

/-- `ψ δ' = ψ(0) δ' - ψ'(0) δ`: `-(ψφ)'(0) = ψ(0)(-φ'(0)) - ψ'(0) φ(0)`. -/
theorem psi_delta_prime (ψ φ : ℝ → ℝ) (hψ : DifferentiableAt ℝ ψ 0) (hφ : DifferentiableAt ℝ φ 0) :
    -(deriv (fun x => ψ x * φ x) 0) = ψ 0 * -(deriv φ 0) - deriv ψ 0 * φ 0 := by
  rw [deriv_fun_mul hψ hφ]; ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.psi_delta_prime
-- test: fa-psi-delta-prime-test.sage

/-- `x² δ'' = 2δ`: `(x² φ)''(0) = 2 φ(0)`. -/
theorem x_sq_delta_second (φ φ' φ'' : ℝ → ℝ) (hφ : ∀ x, HasDerivAt φ (φ' x) x) (hφ' : ∀ x, HasDerivAt φ' (φ'' x) x) :
    deriv (deriv fun x => x ^ 2 * φ x) 0 = 2 * φ 0 := by
  have h1 : deriv (fun x => x ^ 2 * φ x) = fun x => 2 * x * φ x + x ^ 2 * φ' x := by
    funext x
    exact ((hasDerivAt_pow 2 x).mul (hφ x)).deriv.trans (by fa_close3)
  rw [h1]
  exact ((((hasDerivAt_id' 0).const_mul 2).mul (hφ 0)).add ((hasDerivAt_pow 2 (0 : ℝ)).mul (hφ' 0))).deriv.trans
    (by fa_close3)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.x_sq_delta_second
-- test: fa-x-sq-delta-second-test.sage

/-- `x δ'' = -2δ'`: `(xφ)''(0) = 2 φ'(0)`. -/
theorem x_delta_second (φ φ' φ'' : ℝ → ℝ) (hφ : ∀ x, HasDerivAt φ (φ' x) x) (hφ' : ∀ x, HasDerivAt φ' (φ'' x) x) :
    deriv (deriv fun x => x * φ x) 0 = 2 * φ' 0 := by
  have h1 : deriv (fun x => x * φ x) = fun x => φ x + x * φ' x := by
    funext x
    exact ((hasDerivAt_id' x).mul (hφ x)).deriv.trans (by fa_close3)
  rw [h1]
  exact ((hφ 0).add ((hasDerivAt_id' 0).mul (hφ' 0))).deriv.trans (by fa_close3)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.x_delta_second
-- test: fa-x-delta-second-test.sage

/-- The derivatives of the Gaussian sequence against test functions, as stated (tested, not proved: improper
integrals with a parameter): `∫ g_ε' x = -1`, `∫ g_ε' x² = 0`, `∫ g_ε' eˣ = -e^{ε²/4}`, `∫ g_ε'' x² = 2`,
`∫ g_ε'' cos x = -e^{-ε²/4}`, `∫ g_ε' sin x = -e^{-ε²/4}`. -/
def GaussianDerivativeMoments : Prop :=
  ∀ ε : ℝ, 0 < ε →
    let g : ℝ → ℝ := fun x => Real.exp (-x ^ 2 / ε ^ 2) / (ε * Real.sqrt π)
    (∫ x, deriv g x * x) = -1 ∧ (∫ x, deriv g x * x ^ 2) = 0 ∧
    (∫ x, deriv g x * Real.exp x) = -Real.exp (ε ^ 2 / 4) ∧ (∫ x, deriv (deriv g) x * x ^ 2) = 2 ∧
    (∫ x, deriv (deriv g) x * Real.cos x) = -Real.exp (-ε ^ 2 / 4) ∧ (∫ x, deriv g x * Real.sin x) = -Real.exp (-ε ^ 2 / 4)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.GaussianDerivativeMoments
-- test: fa-gaussian-derivative-moments-test.sage

/-- `x^n δ^{(n)} = (-1)^n n! δ`: `(x^n φ)^{(n)}(0) = n! φ(0)`, as stated (tested, not proved). -/
def PowerDeltaDerivative : Prop :=
  ∀ n : ℕ, ∀ φ : ℝ → ℝ, ContDiff ℝ n φ → iteratedDeriv n (fun x => x ^ n * φ x) 0 = n.factorial * φ 0

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.PowerDeltaDerivative
-- test: fa-power-delta-derivative-test.sage

/-! ## exr-dist-greens-ode -/

/-- Right of `0`, `u = -e^{-x}/2` has slope `1/2` at `0`. -/
theorem green_slope_right : deriv (fun x : ℝ => -1 / 2 * Real.exp (-x)) 0 = 1 / 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -1 / 2 * rexp (-x)) ((1 / 2 : ℝ)) (0) :=
        by
        convert ((((hasDerivAt_id' (0 : ℝ)).const_mul (-1 : ℝ)).exp).const_mul (-(1 / 2) : ℝ)) using 1
        all_goals ((norm_num; done))
      exact hd.deriv.trans (by rfl))
  -- certificate: fa-deriv.sage

/-- Left of `0`, `u = -eˣ/2` has slope `-1/2` at `0`: the jump of `u'` is `1`, the delta. -/
theorem green_slope_left : deriv (fun x : ℝ => -1 / 2 * Real.exp x) 0 = -1 / 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => -1 / 2 * rexp x) ((-(1 / 2) : ℝ)) (0) :=
        by
        convert (((hasDerivAt_id' (0 : ℝ)).exp).const_mul (-(1 / 2) : ℝ)) using 1
        all_goals ((norm_num; done))
      exact hd.deriv.trans (by ring1))
  -- certificate: fa-deriv.sage

/-- Away from `0`, `u = -e^{∓x}/2` solves `u'' - u = 0`. -/
theorem green_homogeneous (s x : ℝ) (hs : s = 1 ∨ s = -1) :
    deriv (deriv fun x : ℝ => -1 / 2 * Real.exp (s * x)) x - -1 / 2 * Real.exp (s * x) = 0 := by
  have h1 : deriv (fun x : ℝ => -1 / 2 * Real.exp (s * x)) = fun x => -1 / 2 * (s * Real.exp (s * x)) := by
    funext y
    exact ((((hasDerivAt_id' y).const_mul s).exp).const_mul (-1 / 2 : ℝ)).deriv.trans (by ring)
  rw [h1]
  have h2 : deriv (fun x : ℝ => -1 / 2 * (s * Real.exp (s * x))) x = -1 / 2 * (s * (Real.exp (s * x) * (s * 1))) :=
    ((((hasDerivAt_id' x).const_mul s).exp).const_mul s).const_mul (-1 / 2 : ℝ) |>.deriv
  rw [h2]
  rcases hs with rfl | rfl <;> ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.green_homogeneous
-- test: fa-green-homogeneous-test.sage

/-- `∫ u (φ'' - φ) = φ(0)` for `u = -e^{-|x|}/2` and `φ = e^{-x²}`, as stated (tested, not proved). -/
def GreenGaussianCheck : Prop :=
  ∀ c : ℝ, c * ∫ x : ℝ, (-1 / 2 * Real.exp (-|x|)) * ((4 * x ^ 2 - 2) * Real.exp (-x ^ 2) - Real.exp (-x ^ 2)) = c * 1

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.DistributionDerivatives.GreenGaussianCheck
-- test: fa-green-gaussian-check-test.sage

end KaluzaKlein.FourierAnalysis.DistributionDerivatives

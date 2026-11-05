import Mathlib
import Veriphy.Kit.Levels

/-!
# Complex analysis, section "The Beta function"

The claims of the section "The Beta function" of `tutorial/part-i-04-complex-analysis.qmd`: the Beta integral
(`Complex.betaIntegral`), its convergence, symmetry and the relation `B = ΓΓ/Γ`, its meromorphic continuation, the
values `B(2, 3) = 1/12`, `B(m, n) = (m-1)!(n-1)!/(m+n-1)!`, `B(1/2, 1/2) = π`, `B(3, 4) = 1/60`, the Wallis integral
`∫₀^{π/2} sin⁴ = 3π/16`, the volume of the unit ball, and the substitutions behind the trigonometric and rational
forms. The Wallis-Beta relation `∫₀^{π/2} sinⁿθ dθ = B((n + 1)/2, 1/2)/2` is proved for every `n` (both sides obey
`W_{n+2} = (n + 1)/(n + 2) W_n`); the trigonometric form is proved at `a = b = 1/2` and the rational form at `a = 1`
(the general forms, substitutions in improper integrals with real powers, are not formalised).
-/

namespace KaluzaKlein.ComplexAnalysis.BetaFunction

open Complex Filter Topology Set MeasureTheory
open scoped Nat Real

/-! ## def-beta-function -/

/-- The Beta integral converges for `Re a, Re b > 0`. -/
theorem beta_convergent (a b : ℂ) (ha : 0 < a.re) (hb : 0 < b.re) :
    IntervalIntegrable (fun x : ℝ => (x : ℂ) ^ (a - 1) * (1 - (x : ℂ)) ^ (b - 1)) volume 0 1 :=
  Complex.betaIntegral_convergent ha hb

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_convergent (1) (1) ((by norm_num)) ((by norm_num)) := by
  exact Complex.betaIntegral_convergent (by norm_num) (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_convergent (1 / 2) (1 / 2) ((by norm_num)) ((by norm_num)) := by
  exact Complex.betaIntegral_convergent (by norm_num) (by norm_num)

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_convergent (2) (3) ((by norm_num)) ((by norm_num)) := by
  exact Complex.betaIntegral_convergent (by norm_num) (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_convergent
-- test: ca-spec-beta-conv-test.sage

/-- `B(a, b) = B(b, a)`. -/
theorem beta_symm (a b : ℂ) : betaIntegral a b = betaIntegral b a :=
  Complex.betaIntegral_symm b a

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_symm
-- test: ca-spec-beta-symm-test.sage

/-- `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` for `Re a, Re b > 0` (the division form). -/
theorem beta_eq_gamma' (a b : ℂ) (ha : 0 < a.re) (hb : 0 < b.re) :
    betaIntegral a b = Complex.Gamma a * Complex.Gamma b / Complex.Gamma (a + b) :=
  Complex.betaIntegral_eq_Gamma_mul_div a b ha hb

/-- `B(a, b) = Γ(a)Γ(b)/Γ(a + b)`, i.e. `B(a, b) Γ(a + b) = Γ(a)Γ(b)`, for `Re a, Re b > 0`. -/
theorem beta_eq_gamma (a b : ℂ) (ha : 0 < a.re) (hb : 0 < b.re) :
    betaIntegral a b * Complex.Gamma (a + b) = Complex.Gamma a * Complex.Gamma b := by
  rw [Complex.Gamma_mul_Gamma_eq_betaIntegral ha hb, mul_comm]

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_eq_gamma
-- test: ca-spec-beta-gamma-test.sage

/-- The meromorphic continuation `Γ(a)Γ(b)/Γ(a + b)` is holomorphic in `a` away from the poles of `Γ(a)`
(`1/Γ` being entire). -/
theorem beta_continuation_differentiableAt (a b : ℂ) (ha : ∀ m : ℕ, a ≠ -m) :
    DifferentiableAt ℂ (fun s => Complex.Gamma s * Complex.Gamma b * (Complex.Gamma (s + b))⁻¹) a := by
  have h1 : DifferentiableAt ℂ (fun s => Complex.Gamma s * Complex.Gamma b) a :=
    (Complex.differentiableAt_Gamma a ha).mul_const _
  have h2 : DifferentiableAt ℂ (fun s => (Complex.Gamma (s + b))⁻¹) a :=
    DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s => s + b) a
      (Complex.differentiable_one_div_Gamma (a + b)) (differentiableAt_id.add_const b)
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_continuation_differentiableAt (1) (1) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  have h1 : DifferentiableAt ℂ (fun s => Complex.Gamma s * Complex.Gamma (1 : ℂ)) (1 : ℂ) := (Complex.differentiableAt_Gamma (1 : ℂ) (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])).mul_const _
  have h2 : DifferentiableAt ℂ (fun s => (Complex.Gamma (s + (1 : ℂ)))⁻¹) (1 : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s => s + (1 : ℂ)) (1 : ℂ) (Complex.differentiable_one_div_Gamma ((1 : ℂ) + (1 : ℂ))) (differentiableAt_id.add_const (1 : ℂ))
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_continuation_differentiableAt (1 / 2) (2) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  have h1 : DifferentiableAt ℂ (fun s => Complex.Gamma s * Complex.Gamma (2 : ℂ)) (1 / 2 : ℂ) := (Complex.differentiableAt_Gamma (1 / 2 : ℂ) (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])).mul_const _
  have h2 : DifferentiableAt ℂ (fun s => (Complex.Gamma (s + (2 : ℂ)))⁻¹) (1 / 2 : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s => s + (2 : ℂ)) (1 / 2 : ℂ) (Complex.differentiable_one_div_Gamma ((1 / 2 : ℂ) + (2 : ℂ))) (differentiableAt_id.add_const (2 : ℂ))
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.beta_continuation_differentiableAt (2 + Complex.I) (-1 / 2) ((fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])) := by
  have h1 : DifferentiableAt ℂ (fun s => Complex.Gamma s * Complex.Gamma (-1 / 2 : ℂ)) (2 + Complex.I : ℂ) := (Complex.differentiableAt_Gamma (2 + Complex.I : ℂ) (fun m h => by have h' := congrArg Complex.re h; simp at h' <;> linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)])).mul_const _
  have h2 : DifferentiableAt ℂ (fun s => (Complex.Gamma (s + (-1 / 2 : ℂ)))⁻¹) (2 + Complex.I : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s => s + (-1 / 2 : ℂ)) (2 + Complex.I : ℂ) (Complex.differentiable_one_div_Gamma ((2 + Complex.I : ℂ) + (-1 / 2 : ℂ))) (differentiableAt_id.add_const (-1 / 2 : ℂ))
  exact h1.mul h2

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_continuation_differentiableAt
-- test: ca-spec-beta-mero-test.sage

/-- The rational form `B(a, b) = ∫₀^∞ x^{a-1}(1 + x)^{-a-b} dx` at `a = 1`: `B(1, b) = ∫₀^∞ (1 + x)^{-1-b} dx = 1/b`
(`b > 0`). (The general form is not formalised.) -/
theorem beta_one_rational (b : ℝ) (hb : 0 < b) :
    (betaIntegral 1 b).re = ∫ x in Ioi (0 : ℝ), x ^ ((1 : ℝ) - 1) * (1 + x) ^ (-(1 : ℝ) - b) := by
  have hb' : b ≠ 0 := hb.ne'
  have hΓ : Real.Gamma b ≠ 0 := (Real.Gamma_pos_of_pos hb).ne'
  have hL : (betaIntegral 1 b).re = 1 / b := by
    rw [beta_eq_gamma' _ _ (by norm_num) (by simpa using hb), ← Complex.ofReal_one, ← Complex.ofReal_add,
      Complex.Gamma_ofReal, Complex.Gamma_ofReal, Complex.Gamma_ofReal, ← Complex.ofReal_mul,
      ← Complex.ofReal_div, Complex.ofReal_re, Real.Gamma_one, add_comm, Real.Gamma_add_one hb', one_mul,
      mul_comm b, div_mul_eq_div_div, div_self hΓ]
  have e : ∀ x : ℝ, x ^ ((1 : ℝ) - 1) * (1 + x) ^ (-(1 : ℝ) - b) = (1 + x) ^ (-b - 1) := fun x => by
    rw [sub_self, Real.rpow_zero, one_mul]
    congr 1
    ring
  have hd : ∀ x : ℝ, 1 + x ≠ 0 → HasDerivAt (fun x => -(1 + x) ^ (-b) / b) ((1 + x) ^ (-b - 1)) x := by
    intro x hx
    have h := (((hasDerivAt_id' x).const_add 1).rpow_const (p := -b) (Or.inl hx)).neg.div_const b
    convert h using 1
    first | (field_simp; done) | (field_simp; ring)
  have hR : ∫ x in Ioi (0 : ℝ), (1 + x) ^ (-b - 1) = 0 - (-(1 + 0) ^ (-b) / b) := by
    refine integral_Ioi_of_hasDerivAt_of_nonneg (g := fun x : ℝ => -(1 + x) ^ (-b) / b)
      (hd 0 (by norm_num)).continuousAt.continuousWithinAt
      (fun x hx => hd x (by have := Set.mem_Ioi.mp hx; intro h; linarith))
      (fun x hx => Real.rpow_nonneg (by have := Set.mem_Ioi.mp hx; linarith) _) ?_
    have h1 : Tendsto (fun x : ℝ => 1 + x) atTop atTop :=
      tendsto_atTop_mono (fun x => (le_add_of_nonneg_left zero_le_one : x ≤ 1 + x)) tendsto_id
    have ht : Tendsto (fun x : ℝ => -(1 + x) ^ (-b) / b) atTop (𝓝 (-0 / b)) :=
      ((tendsto_rpow_neg_atTop hb).comp h1).neg.div_const b
    rwa [neg_zero, zero_div] at ht
  simp_rw [e]
  rw [hL, hR]
  first | (rw [add_zero, Real.one_rpow, neg_div, sub_neg_eq_add, zero_add]) | (simp; ring) | norm_num

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_one_rational
-- test: ca-spec-beta-rat-test.sage

/-! ## exm-beta-integers -/

/-- `B(2, 3) = ∫₀¹ t(1 - t)² dt = 1/2 - 2/3 + 1/4 = 1/12`. -/
theorem beta_two_three_integral : ∫ x in (0 : ℝ)..1, x * (1 - x) ^ 2 = 1 / 12 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt (fun x_veriphy => ((x_veriphy ^ 4) / 4 - (2 * (x_veriphy ^ 3) / 3) + ((x_veriphy ^ 2) / 2) : ℝ))
            ((fun x => x * (1 - x) ^ 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          (((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (1 / 4 : ℝ)).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (-(2 / 3) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          ((x_veriphy ^ 4) / 4 - (2 * (x_veriphy ^ 3) / 3) + ((x_veriphy ^ 2) / 2) : ℝ)) (f' := fun x =>
          x * (1 - x) ^ 2) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ca-spec-integral.sage

/-- `Γ(2)Γ(3)/Γ(5) = 1 · 2/24 = 1/12`. -/
theorem gamma_ratio_two_three : Real.Gamma 2 * Real.Gamma 3 / Real.Gamma 5 = 1 / 12 := by
  have h2 : Real.Gamma 2 = 1 := by
    rw [show (2 : ℝ) = (1 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; norm_num [Nat.factorial]
  have h3 : Real.Gamma 3 = 2 := by
    rw [show (3 : ℝ) = (2 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; norm_num [Nat.factorial]
  have h5 : Real.Gamma 5 = 24 := by
    rw [show (5 : ℝ) = (4 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; norm_num [Nat.factorial]
  rw [div_eq_iff (by rw [h5]; norm_num)]
  veriphy_certified (linear_combination (2) * h2 + ((Real.Gamma 2)) * h3 + (-1 / 12) * h5)
  -- certificate: ideal_membership.sage

/-- `B(m + 1, n + 1) = m! n!/(m + n + 1)!`, i.e. `B(m + 1, n + 1) (m + n + 1)! = m! n!`. -/
theorem beta_nat (m n : ℕ) : betaIntegral (m + 1) (n + 1) * (m + n + 1)! = (m ! * n ! : ℂ) := by
  have hf : (((m + n + 1)! : ℕ) : ℂ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero _
  rw [← eq_div_iff hf]
  rw [beta_eq_gamma' _ _ (by rw [Complex.add_re, Complex.natCast_re, Complex.one_re]; positivity)
    (by rw [Complex.add_re, Complex.natCast_re, Complex.one_re]; positivity)]
  rw [show (m : ℂ) + 1 + (n + 1) = ((m + n + 1 : ℕ) : ℂ) + 1 by push_cast <;> ring]
  rw [Complex.Gamma_nat_eq_factorial, Complex.Gamma_nat_eq_factorial, Complex.Gamma_nat_eq_factorial]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_nat
-- test: ca-spec-beta-nat-test.sage

/-! ## exm-beta-half -/

/-- `B(1/2, 1/2) = Γ(1/2)²/Γ(1) = π`. -/
theorem beta_half_half : betaIntegral (1 / 2) (1 / 2) = π := by
  rw [beta_eq_gamma' _ _ (by norm_num) (by norm_num), show (1 / 2 : ℂ) + 1 / 2 = 1 by norm_num, Complex.Gamma_one,
    div_one, Complex.Gamma_one_half_eq]
  have hc : ((π : ℂ) ^ (1 / 2 : ℂ)) ^ 2 = π := by
    rw [show (π : ℂ) ^ (1 / 2 : ℂ) = ((√π : ℝ) : ℂ) by
      rw [Real.sqrt_eq_rpow, Complex.ofReal_cpow Real.pi_pos.le]; push_cast <;> ring_nf]
    rw [← Complex.ofReal_pow, Real.sq_sqrt Real.pi_pos.le]
  generalize (π : ℂ) ^ (1 / 2 : ℂ) = c at hc ⊢
  generalize (π : ℂ) = p at hc ⊢
  rw [← sub_eq_zero]
  veriphy_certified (linear_combination hc)
  -- certificate: ideal_membership.sage

/-- The trigonometric form `B(a, b) = 2∫₀^{π/2} sin^{2a-1}θ cos^{2b-1}θ dθ` at `a = b = 1/2`: `B(1/2, 1/2) = 2∫₀^{π/2} dθ
= π`. (The general form, a substitution in an improper integral with real powers, is not formalised.) -/
theorem beta_half_half_trig :
    (betaIntegral (1 / 2) (1 / 2)).re =
      2 * ∫ θ in (0 : ℝ)..(π / 2), Real.sin θ ^ (2 * (1 / 2 : ℝ) - 1) * Real.cos θ ^ (2 * (1 / 2 : ℝ) - 1) := by
  have h : ∀ θ : ℝ, Real.sin θ ^ (2 * (1 / 2 : ℝ) - 1) * Real.cos θ ^ (2 * (1 / 2 : ℝ) - 1) = 1 := fun θ => by
    norm_num
  simp_rw [h]
  rw [beta_half_half, Complex.ofReal_re, intervalIntegral.integral_const, smul_eq_mul, mul_one, sub_zero]
  generalize π = p
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- The substitution `t = sin²θ` turns `∫₀¹ dt/√(t(1 - t))` into `2∫₀^{π/2} dθ = π`. -/
theorem two_quarter_turn : ∫ θ in (0 : ℝ)..(π / 2), (2 : ℝ) = π := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π / 2 : ℝ),
          HasDerivAt (fun x_veriphy => (2 * x_veriphy : ℝ)) ((fun θ => 2) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; rfl
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy => (2 * x_veriphy : ℝ)) (f' := fun θ => 2)
          hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals ring1)
  -- certificate: ca-spec-integral.sage

/-- The Wallis integrals in Gamma functions: `∫₀^{π/2} sinⁿθ dθ = Γ((n+1)/2) Γ(1/2) / Γ((n+1)/2 + 1/2) / 2`; both sides
satisfy `W_{n+2} = (n + 1)/(n + 2) W_n` (`integral_sin_pow`, `Real.Gamma_add_one`) with `W₀ = π/2`, `W₁ = 1`. -/
theorem wallis_beta_real (n : ℕ) :
    ∫ θ in (0 : ℝ)..(π / 2), Real.sin θ ^ n =
      Real.Gamma ((n + 1) / 2) * Real.Gamma (1 / 2) / Real.Gamma ((n + 1) / 2 + 1 / 2) / 2 := by
  induction n using Nat.twoStepInduction with
  | zero =>
    simp only [pow_zero, intervalIntegral.integral_const, smul_eq_mul, mul_one, sub_zero, Nat.cast_zero, zero_add]
    rw [show (1 / 2 : ℝ) + 1 / 2 = 1 by norm_num, Real.Gamma_one, div_one, Real.Gamma_one_half_eq,
      Real.mul_self_sqrt Real.pi_pos.le]
  | one =>
    simp only [pow_one, integral_sin, Real.cos_zero, Real.cos_pi_div_two, sub_zero, Nat.cast_one]
    rw [show ((1 : ℝ) + 1) / 2 = 1 by norm_num, show (1 : ℝ) + 1 / 2 = 1 / 2 + 1 by norm_num,
      Real.Gamma_add_one (s := (1 / 2 : ℝ)) (by norm_num), Real.Gamma_one, Real.Gamma_one_half_eq, one_mul]
    have hs : √π ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
    first | (field_simp; done) | (field_simp; ring)
  | more n ih _ =>
    have hx : ((n : ℝ) + 1) / 2 ≠ 0 := by positivity
    have hy : ((n : ℝ) + 1) / 2 + 1 / 2 ≠ 0 := by positivity
    have hn2 : ((n : ℝ) + 2) ≠ 0 := by positivity
    have hΓ : Real.Gamma (((n : ℝ) + 1) / 2 + 1 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by positivity)).ne'
    have g1 : Real.Gamma ((((n + 2 : ℕ) : ℝ) + 1) / 2) = ((n : ℝ) + 1) / 2 * Real.Gamma (((n : ℝ) + 1) / 2) := by
      rw [← Real.Gamma_add_one hx]
      congr 1
      push_cast
      ring
    have g2 : Real.Gamma ((((n + 2 : ℕ) : ℝ) + 1) / 2 + 1 / 2) =
        (((n : ℝ) + 1) / 2 + 1 / 2) * Real.Gamma (((n : ℝ) + 1) / 2 + 1 / 2) := by
      rw [← Real.Gamma_add_one hy]
      congr 1
      push_cast
      ring
    rw [integral_sin_pow, Real.sin_zero, Real.cos_pi_div_two, zero_pow (Nat.succ_ne_zero n), zero_mul, mul_zero,
      sub_zero, zero_div, zero_add, ih, g1, g2]
    first | (field_simp; done) | (field_simp; ring)

/-- The Wallis integrals as Beta values: `∫₀^{π/2} sinⁿθ dθ = B((n + 1)/2, 1/2)/2` for every `n`. -/
theorem wallis_beta (n : ℕ) :
    ∫ θ in (0 : ℝ)..(π / 2), Real.sin θ ^ n = (betaIntegral (((n : ℂ) + 1) / 2) (1 / 2)).re / 2 := by
  have h1 : (((n : ℂ) + 1) / 2) = (((n + 1) / 2 : ℝ) : ℂ) := by norm_num
  have h2 : (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) := by norm_num
  rw [beta_eq_gamma' _ _ (by rw [h1, Complex.ofReal_re]; positivity) (by norm_num), h1, h2, ← Complex.ofReal_add,
    Complex.Gamma_ofReal, Complex.Gamma_ofReal, Complex.Gamma_ofReal, ← Complex.ofReal_mul, ← Complex.ofReal_div,
    Complex.ofReal_re, wallis_beta_real]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.wallis_beta
-- test: ca-spec-wallis-test.sage

/-! ## exm-beta-ball-volume -/

/-- The Gaussian over `ℝⁿ`: `∫ e^{-|x|²} dx = π^{n/2}`. -/
theorem gaussian_euclidean (n : ℕ) :
    ∫ x : EuclideanSpace ℝ (Fin n), Real.exp (-‖x‖ ^ 2) = π ^ ((n : ℝ) / 2) := by
  have h := GaussianFourier.integral_rexp_neg_mul_sq_norm (V := EuclideanSpace ℝ (Fin n)) (b := 1) one_pos
  simpa using h

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.gaussian_euclidean
-- test: ca-spec-gauss-rn-test.sage

/-- The radial integral `∫₀^∞ r^{n-1} e^{-r²} dr = Γ(n/2)/2`, `n ≥ 1`. -/
theorem radial_gaussian (n : ℕ) (hn : 1 ≤ n) :
    ∫ r in Ioi (0 : ℝ), r ^ (n - 1) * Real.exp (-r ^ 2) = Real.Gamma ((n : ℝ) / 2) / 2 := by
  have h := _root_.integral_rpow_mul_exp_neg_rpow (p := 2) (q := ((n - 1 : ℕ) : ℝ)) two_pos (by linarith [Nat.cast_nonneg (α := ℝ) (n - 1)])
  simp_rw [Real.rpow_natCast, Real.rpow_two] at h
  rw [h, show (((n - 1 : ℕ) : ℝ) + 1) / 2 = (n : ℝ) / 2 by rw [Nat.cast_sub hn]; push_cast <;> ring]
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.radial_gaussian
-- test: ca-spec-radial-test.sage

/-- `V_n = S_{n-1}/n`: `2π^{n/2}/Γ(n/2)/n = π^{n/2}/Γ(n/2 + 1)`. -/
theorem ball_volume_from_sphere (n : ℕ) (hn : 0 < n) :
    2 * π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2) / n = π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2 + 1) := by
  have hpos : 0 < (n : ℝ) / 2 := by positivity
  have h := Real.Gamma_add_one (s := (n : ℝ) / 2) hpos.ne'
  have hg : Real.Gamma ((n : ℝ) / 2) ≠ 0 := (Real.Gamma_pos_of_pos hpos).ne'
  have hn' : (n : ℝ) ≠ 0 := by positivity
  rw [h, div_div, div_eq_div_iff (mul_ne_zero hg hn') (mul_ne_zero hpos.ne' hg)]
  generalize π ^ ((n : ℝ) / 2) = P
  generalize Real.Gamma ((n : ℝ) / 2) = G
  generalize (n : ℝ) = x
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- The volume of the unit ball in `ℝⁿ`: `V_n = π^{n/2}/Γ(n/2 + 1)` (`V₂ = π`, `V₃ = 4π/3`). -/
theorem unit_ball_volume (n : ℕ) (hn : 0 < n) :
    (volume (Metric.ball (0 : EuclideanSpace ℝ (Fin n)) 1)).toReal =
      π ^ ((n : ℝ) / 2) / Real.Gamma ((n : ℝ) / 2 + 1) := by
  haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  rw [EuclideanSpace.volume_ball, Fintype.card_fin, ENNReal.ofReal_one, one_pow, one_mul,
    ENNReal.toReal_ofReal (by positivity [Real.Gamma_pos_of_pos (show (0 : ℝ) < n / 2 + 1 by positivity)])]
  congr 1
  rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul Real.pi_pos.le]
  ring_nf

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.unit_ball_volume
-- test: ca-spec-ball-vol-test.sage

/-- `V_n = π^{n/2}/Γ(n/2 + 1)` makes sense for every complex `n`: it is an entire function of `n`. -/
theorem ball_volume_entire (s : ℂ) :
    DifferentiableAt ℂ (fun s : ℂ => (π : ℂ) ^ (s / 2) * (Complex.Gamma (s / 2 + 1))⁻¹) s := by
  have h1 : DifferentiableAt ℂ (fun s : ℂ => (π : ℂ) ^ (s / 2)) s :=
    (differentiableAt_id.div_const (2 : ℂ)).const_cpow (Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero))
  have h2 : DifferentiableAt ℂ (fun s : ℂ => (Complex.Gamma (s / 2 + 1))⁻¹) s :=
    DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s : ℂ => s / 2 + 1) s
      (Complex.differentiable_one_div_Gamma (s / 2 + 1)) ((differentiableAt_id.div_const (2 : ℂ)).add_const 1)
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.ball_volume_entire (0) := by
  have h1 : DifferentiableAt ℂ (fun s : ℂ => (Real.pi : ℂ) ^ (s / 2)) (0 : ℂ) := (differentiableAt_id.div_const (2 : ℂ)).const_cpow (Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero))
  have h2 : DifferentiableAt ℂ (fun s : ℂ => (Complex.Gamma (s / 2 + 1))⁻¹) (0 : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s : ℂ => s / 2 + 1) (0 : ℂ) (Complex.differentiable_one_div_Gamma ((0 : ℂ) / 2 + 1)) ((differentiableAt_id.div_const (2 : ℂ)).add_const 1)
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.ball_volume_entire (2) := by
  have h1 : DifferentiableAt ℂ (fun s : ℂ => (Real.pi : ℂ) ^ (s / 2)) (2 : ℂ) := (differentiableAt_id.div_const (2 : ℂ)).const_cpow (Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero))
  have h2 : DifferentiableAt ℂ (fun s : ℂ => (Complex.Gamma (s / 2 + 1))⁻¹) (2 : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s : ℂ => s / 2 + 1) (2 : ℂ) (Complex.differentiable_one_div_Gamma ((2 : ℂ) / 2 + 1)) ((differentiableAt_id.div_const (2 : ℂ)).add_const 1)
  exact h1.mul h2

example : veriphy_instance% KaluzaKlein.ComplexAnalysis.BetaFunction.ball_volume_entire (3 + Complex.I) := by
  have h1 : DifferentiableAt ℂ (fun s : ℂ => (Real.pi : ℂ) ^ (s / 2)) (3 + Complex.I : ℂ) := (differentiableAt_id.div_const (2 : ℂ)).const_cpow (Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero))
  have h2 : DifferentiableAt ℂ (fun s : ℂ => (Complex.Gamma (s / 2 + 1))⁻¹) (3 + Complex.I : ℂ) := DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s : ℂ => s / 2 + 1) (3 + Complex.I : ℂ) (Complex.differentiable_one_div_Gamma ((3 + Complex.I : ℂ) / 2 + 1)) ((differentiableAt_id.div_const (2 : ℂ)).add_const 1)
  exact h1.mul h2

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.ComplexAnalysis.BetaFunction.ball_volume_entire
-- test: ca-spec-ball-entire-test.sage

/-! ## exr-beta-integral -/

/-- `B(3, 4) = ∫₀¹ t²(1 - t)³ dt = 1/60`. -/
theorem beta_three_four_integral : ∫ x in (0 : ℝ)..1, x ^ 2 * (1 - x) ^ 3 = 1 / 60 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (1 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-(x_veriphy ^ 6) / 6 + (3 * (x_veriphy ^ 5) / 5) - (3 * (x_veriphy ^ 4) / 4) + ((x_veriphy ^ 3) / 3) :
                ℝ))
            ((fun x => x ^ 2 * (1 - x) ^ 3) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).pow 6).const_mul (-(1 / 6) : ℝ)).add
                            (((hasDerivAt_id' (x_veriphy : ℝ)).pow 5).const_mul (3 / 5 : ℝ))).add
                        (((hasDerivAt_id' (x_veriphy : ℝ)).pow 4).const_mul (-(3 / 4) : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).pow 3).const_mul (1 / 3 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-(x_veriphy ^ 6) / 6 + (3 * (x_veriphy ^ 5) / 5) - (3 * (x_veriphy ^ 4) / 4) + ((x_veriphy ^ 3) / 3) : ℝ))
          (f' := fun x => x ^ 2 * (1 - x) ^ 3) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      all_goals (norm_num; done))
  -- certificate: ca-spec-integral.sage

/-- `Γ(3)Γ(4)/Γ(7) = 2! 3!/6! = 12/720 = 1/60`. -/
theorem gamma_ratio_three_four : Real.Gamma 3 * Real.Gamma 4 / Real.Gamma 7 = 1 / 60 := by
  have h3 : Real.Gamma 3 = 2 := by
    rw [show (3 : ℝ) = (2 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; simp [Nat.factorial]
  have h4 : Real.Gamma 4 = 6 := by
    rw [show (4 : ℝ) = (3 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; norm_num [Nat.factorial]
  have h7 : Real.Gamma 7 = 720 := by
    rw [show (7 : ℝ) = (6 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; norm_num [Nat.factorial]
  rw [div_eq_iff (by rw [h7]; norm_num)]
  veriphy_certified (linear_combination (6) * h3 + ((Real.Gamma 3)) * h4 + (-1 / 60) * h7)
  -- certificate: ideal_membership.sage

/-- Power reduction: `sin⁴θ = 3/8 - cos 2θ/2 + cos 4θ/8`. -/
theorem sin_pow_four_reduction (θ : ℝ) :
    Real.sin θ ^ 4 = 3 / 8 - Real.cos (2 * θ) / 2 + Real.cos (4 * θ) / 8 := by
  have h2 : Real.cos (2 * θ) = 2 * Real.cos θ ^ 2 - 1 := Real.cos_two_mul θ
  have h4 : Real.cos (4 * θ) = 2 * Real.cos (2 * θ) ^ 2 - 1 := by
    rw [show 4 * θ = 2 * (2 * θ) by ring, Real.cos_two_mul]
  have hp : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := Real.sin_sq_add_cos_sq θ
  generalize Real.sin θ = s at hp ⊢
  generalize Real.cos θ = c at h2 hp
  generalize Real.cos (2 * θ) = C2 at h2 h4 ⊢
  generalize Real.cos (4 * θ) = C4 at h4 ⊢
  rw [← sub_eq_zero]
  veriphy_certified
    (linear_combination (s ^ 2 - 1 / 2 * C2 + 1 / 2) * hp + (1 / 2 * s ^ 2 - 1 / 4 * C2 + 1 / 4) * h2 + (-1 / 8) * h4)
  -- certificate: ideal_membership.sage

/-- The power-reduced integral `∫₀^{π/2} (3/8 - cos 2θ/2 + cos 4θ/8) dθ = 3π/16`. -/
theorem wallis_four_reduced :
    ∫ θ in (0 : ℝ)..(π / 2), (3 / 8 - Real.cos (2 * θ) / 2 + Real.cos (4 * θ) / 8) = 3 * π / 16 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π / 2 : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (3 * x_veriphy / 8 + ((Real.sin (4 * x_veriphy)) / 32) - ((Real.sin (2 * x_veriphy)) / 4) : ℝ))
            ((fun θ => 3 / 8 - Real.cos (2 * θ) / 2 + Real.cos (4 * θ) / 8) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (3 / 8 : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).sin).const_mul (1 / 32 : ℝ))).add
                    ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                      (-(1 / 4) : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · ring1
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (3 * x_veriphy / 8 + ((Real.sin (4 * x_veriphy)) / 32) - ((Real.sin (2 * x_veriphy)) / 4) : ℝ)) (f' :=
          fun θ => 3 / 8 - Real.cos (2 * θ) / 2 + Real.cos (4 * θ) / 8) hF
          (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.sin ((4 * (0) : ℝ)) = 0 := by simp
      have he1 : Real.sin ((4 * (π / 2) : ℝ)) = 0 := by
        rw [show ((4 * (π / 2) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      have he2 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
      have he3 : Real.sin ((2 * (π / 2) : ℝ)) = 0 := by
        rw [show ((2 * (π / 2) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: ca-spec-integral.sage

/-- The Wallis integral `∫₀^{π/2} sin⁴θ dθ = 3π/16`. -/
theorem wallis_four : ∫ θ in (0 : ℝ)..(π / 2), Real.sin θ ^ 4 = 3 * π / 16 := by
  simp_rw [sin_pow_four_reduction]
  exact wallis_four_reduced

/-- `B(5/2, 1/2)/2 = Γ(5/2)Γ(1/2)/Γ(3)/2 = (3π/4)/2/2 = 3π/16`. -/
theorem half_beta_five_halves : Real.Gamma (5 / 2) * Real.Gamma (1 / 2) / Real.Gamma 3 / 2 = 3 * π / 16 := by
  have h1 : Real.Gamma (3 / 2 + 1) = 3 / 2 * Real.Gamma (3 / 2) := Real.Gamma_add_one (by norm_num)
  have h2 : Real.Gamma (1 / 2 + 1) = 1 / 2 * Real.Gamma (1 / 2) := Real.Gamma_add_one (by norm_num)
  have h3 : Real.Gamma (1 / 2) = √π := Real.Gamma_one_half_eq
  have h4 : Real.Gamma 3 = 2 := by
    rw [show (3 : ℝ) = (2 : ℕ) + 1 by norm_num, Real.Gamma_nat_eq_factorial]; simp [Nat.factorial]
  have h5 : √π ^ 2 = π := Real.sq_sqrt Real.pi_pos.le
  rw [show (3 / 2 : ℝ) + 1 = 5 / 2 by norm_num] at h1
  rw [show (1 / 2 : ℝ) + 1 = 3 / 2 by norm_num] at h2
  rw [div_div, div_eq_iff (by rw [h4]; norm_num)]
  veriphy_certified
    (linear_combination
        ((√π)) * h1 + (3 / 2 * (√π)) * h2 + ((Real.Gamma (5 / 2)) + 3 / 4 * (√π)) * h3 + (-3 / 8 * π) * h4 +
          (3 / 4) * h5)
  -- certificate: ideal_membership.sage

/-! ## exr-beta-trig -/

/-- `t = sin²θ`: `dt = 2 sinθ cosθ dθ`. -/
theorem deriv_sin_sq (θ : ℝ) : deriv (fun θ => Real.sin θ ^ 2) θ = 2 * Real.sin θ * Real.cos θ := by
  veriphy_certified
    ( have hd : HasDerivAt (fun θ => Real.sin θ ^ 2) ((2 * (Real.cos θ) * (Real.sin θ) : ℝ)) (θ) :=
        by
        convert (((hasDerivAt_id' (θ : ℝ)).sin).pow 2) using 1
        all_goals (ring1)
      exact hd.deriv.trans (by ring1))
  -- certificate: ca-spec-derivative.sage

/-- `t = sin²θ`: `1 - t = cos²θ`. -/
theorem one_sub_sin_sq (θ : ℝ) : 1 - Real.sin θ ^ 2 = Real.cos θ ^ 2 := by
  veriphy_certified (rw [Real.cos_sq'])
  -- certificate: ca-spec-trig.sage

/-- `t = x/(x + 1)`: `dt = dx/(x + 1)²`. -/
theorem deriv_rational_subst (x : ℝ) (hx : x + 1 ≠ 0) : deriv (fun x => x / (x + 1)) x = 1 / (x + 1) ^ 2 := by
  veriphy_certified
    ( have hd : HasDerivAt (fun x => x / (x + 1)) ((1 / (x + 1) ^ 1 - (x / ((x + 1) ^ 2)) : ℝ)) (x) :=
        by
        convert
          ((hasDerivAt_id' (x : ℝ)).div ((hasDerivAt_id' (x : ℝ)).add (hasDerivAt_const (x : ℝ) (1 : ℝ))) hx) using 1
        all_goals (simp only [Pi.add_apply]; (field_simp; ring1))
      exact hd.deriv.trans (by (field_simp; ring1)))
  -- certificate: ca-spec-derivative.sage

/-- `t = x/(1 + x)`: `1 - t = 1/(1 + x)`. -/
theorem one_sub_rational_subst (x : ℝ) (hx : 1 + x ≠ 0) : 1 - x / (1 + x) = 1 / (1 + x) := by
  field_simp
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- The exponent count `-(a - 1) - (b - 1) - 2 = -a - b`. -/
theorem rational_exponent (a b : ℝ) : -(a - 1) - (b - 1) - 2 = -a - b := by
  rw [← sub_eq_zero]
  veriphy_certified (ring1)
  -- certificate: ca-spec-ring.sage

/-- The integrand after `t = x/(1 + x)`: `t^{a-1}(1 - t)^{b-1} dt/dx = x^{a-1}(1 + x)^{-a-b}` for `x > 0`. -/
theorem rational_integrand (a b x : ℝ) (hx : 0 < x) :
    (x / (1 + x)) ^ (a - 1) * (1 / (1 + x)) ^ (b - 1) * (1 / (1 + x) ^ 2) = x ^ (a - 1) * (1 + x) ^ (-a - b) := by
  have h1 : 0 < 1 + x := by linarith
  have key : (1 + x) ^ (-a - b) * ((1 + x) ^ (a - 1) * (1 + x) ^ (b - 1) * (1 + x) ^ 2) = 1 := by
    rw [← Real.rpow_natCast (1 + x) 2, ← Real.rpow_add h1, ← Real.rpow_add h1, ← Real.rpow_add h1]
    convert Real.rpow_zero (1 + x) using 2
    push_cast
    ring
  rw [Real.div_rpow hx.le h1.le, Real.div_rpow zero_le_one h1.le, Real.one_rpow, eq_one_div_of_mul_eq_one_left key]
  generalize (1 + x) ^ (a - 1) = P
  generalize (1 + x) ^ (b - 1) = Q
  generalize (1 + x) ^ 2 = R
  ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.rational_integrand
-- test: ca-spec-rat-integrand-test.sage

/-! ## exr-beta-gamma-derivation -/

/-- `Γ(a)Γ(b) = Γ(a + b) B(a, b)` for `Re a, Re b > 0`. -/
theorem gamma_mul_gamma (a b : ℂ) (ha : 0 < a.re) (hb : 0 < b.re) :
    Complex.Gamma a * Complex.Gamma b = Complex.Gamma (a + b) * betaIntegral a b :=
  Complex.Gamma_mul_Gamma_eq_betaIntegral ha hb

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.ComplexAnalysis.BetaFunction.gamma_mul_gamma
-- test: ca-spec-gamma-beta-test.sage

/-- `Γ(a)Γ(b)` as a double integral over the quadrant. -/
theorem gamma_mul_gamma_double (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    Real.Gamma a * Real.Gamma b = ∫ p in Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ),
      (p.1 ^ (a - 1) * Real.exp (-p.1)) * (p.2 ^ (b - 1) * Real.exp (-p.2)) := by
  rw [Real.Gamma_eq_integral ha, Real.Gamma_eq_integral hb, Measure.volume_eq_prod,
    setIntegral_prod_mul (fun u => u ^ (a - 1) * Real.exp (-u)) (fun v => v ^ (b - 1) * Real.exp (-v))]
  simp only [mul_comm (Real.exp _)]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.BetaFunction.gamma_mul_gamma_double
-- test: ca-spec-gamma-double-test.sage

/-- The Jacobian of `u = st`, `v = s(1 - t)`: `det ∂(u, v)/∂(t, s) = -s`, of absolute value `s`. -/
theorem beta_jacobian (s t : ℝ) : Matrix.det !![s, t; -s, 1 - t] = s := by
  veriphy_certified
    ( rw [Matrix.det_fin_two_of]
      ring)
  -- certificate: matrix_from_goal.sage

/-- The integrand in the variables `s, t`: `u^{a-1}v^{b-1}e^{-u-v} · s = s^{a+b-1}t^{a-1}(1 - t)^{b-1}e^{-s}`. -/
theorem beta_change_integrand (a b s t : ℝ) (hs : 0 < s) (ht : 0 < t) (ht1 : t < 1) :
    (s * t) ^ (a - 1) * (s * (1 - t)) ^ (b - 1) * Real.exp (-(s * t) - s * (1 - t)) * s =
      s ^ (a + b - 1) * t ^ (a - 1) * (1 - t) ^ (b - 1) * Real.exp (-s) := by
  have h1 : 0 < 1 - t := by linarith
  rw [Real.mul_rpow hs.le ht.le, Real.mul_rpow hs.le h1.le, show -(s * t) - s * (1 - t) = -s by ring]
  have : s ^ (a + b - 1) = s ^ (a - 1) * s ^ (b - 1) * s := by
    rw [show a + b - 1 = (a - 1) + (b - 1) + 1 by ring, Real.rpow_add hs, Real.rpow_add hs, Real.rpow_one]
  rw [this]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.ComplexAnalysis.BetaFunction.beta_change_integrand
-- test: ca-spec-beta-change-test.sage

end KaluzaKlein.ComplexAnalysis.BetaFunction

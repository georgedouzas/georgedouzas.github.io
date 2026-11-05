import KaluzaKlein.PartI.Probability.Variance

/-!
# Probability and statistics, section "Characteristic functions"

The claims of the blocks `def-characteristic-function` to `exr-rv-cf-inversion` of
`tutorial/part-i-08-probability.qmd`: the characteristic function `cf p X t = ∑ p(ω) e^{itX(ω)}` of a finite random
variable (and its extension `cfC` to a complex argument, whose derivatives at `0` are the moments), the product and
rescaling rules, the die and the uniform, exponential and Gaussian characteristic functions, the coin-flip
characteristic function `cos(t/√n)ⁿ`, and the inversion formula for integer-valued variables.
-/

open Finset Complex

namespace KaluzaKlein.Probability.CharacteristicFunctions

open SampleSpaces Conditioning RandomVariables Variance

/-- The characteristic function `φ_X(t) = E e^{itX} = ∑ e^{itx_k} p_k` of a finite random variable. -/
noncomputable def cf {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (t : ℝ) : ℂ :=
  ∑ ω, (p ω : ℂ) * exp (I * t * X ω)

/-- The same sum of exponentials as a function of a complex argument (to differentiate at `0`). -/
noncomputable def cfC {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (z : ℂ) : ℂ :=
  ∑ ω, (p ω : ℂ) * exp (I * X ω * z)

/-! ## def-characteristic-function -/

/-- `φ_X` always exists (a finite sum) with `φ_X(0) = 1`, and `|φ_X| ≤ 1` since `|e^{itX}| = 1`. -/
theorem cf_zero_norm {Ω : Type} [Fintype Ω] {p : Ω → ℚ} (hp : IsDist p) (X : Ω → ℚ) (t : ℝ) :
    cf p X 0 = 1 ∧ ‖cf p X t‖ ≤ 1 := by
  constructor
  · simp only [cf, ofReal_zero, mul_zero, zero_mul, exp_zero, mul_one]
    exact_mod_cast hp.sum_one
  · calc ‖cf p X t‖ ≤ ∑ ω, ‖(p ω : ℂ) * exp (I * t * X ω)‖ := norm_sum_le _ _
      _ = ∑ ω, (p ω : ℝ) := by
        refine Finset.sum_congr rfl fun ω _ => ?_
        rw [norm_mul, show I * (t : ℂ) * (X ω : ℚ) = I * ((t * (X ω : ℝ) : ℝ) : ℂ) by push_cast; ring,
          norm_exp_I_mul_ofReal, mul_one]
        simp [abs_of_nonneg (hp.nonneg ω)]
      _ = 1 := by exact_mod_cast hp.sum_one

/-- The iterated derivatives of a finite sum of exponentials `∑ aₖ e^{cₖ z}`. -/
theorem iteratedDeriv_sum_exp {ι : Type} [Fintype ι] (a c : ι → ℂ) (n : ℕ) :
    iteratedDeriv n (fun z => ∑ k, a k * exp (c k * z)) = fun z => ∑ k, a k * c k ^ n * exp (c k * z) := by
  induction n with
  | zero => funext z; simp
  | succ n ih =>
    rw [iteratedDeriv_succ, ih]
    funext z
    rw [deriv_sum fun k _ => ?_]
    · refine Finset.sum_congr rfl fun k _ => ?_
      have := ((Complex.hasDerivAt_exp (c k * z)).comp z (hasDerivAt_mul_const (c k))).const_mul (a k * c k ^ n)
      rw [this.deriv]; simp only [Function.comp]; ring
    · exact (((Complex.hasDerivAt_exp (c k * z)).comp z (hasDerivAt_mul_const (c k))).const_mul
        (a k * c k ^ n)).differentiableAt

/-- `φ_X⁽ⁿ⁾(0) = iⁿ E Xⁿ`: the derivatives at zero are the moments. -/
theorem cfC_iteratedDeriv_zero {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (n : ℕ) :
    iteratedDeriv n (cfC p X) 0 = I ^ n * (E p (fun ω => X ω ^ n) : ℚ) := by
  have := congrFun (iteratedDeriv_sum_exp (fun ω => (p ω : ℂ)) (fun ω => I * (X ω : ℚ)) n) 0
  simp only [cfC] at this ⊢
  rw [this]
  simp only [mul_zero, exp_zero, mul_one, E, mul_pow, Finset.mul_sum]
  push_cast
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- The characteristic function on the real line is the complex one restricted. -/
theorem cf_eq_cfC {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (t : ℝ) : cf p X t = cfC p X t := by
  simp only [cf, cfC]
  exact Finset.sum_congr rfl fun ω _ => by ring_nf

/-- The product rule: for independent `X`, `Y` (a product space), `φ_{X+Y} = φ_X φ_Y`. -/
theorem cf_add_prod {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] (p : Ω → ℚ) (q : Ω' → ℚ) (X : Ω → ℚ) (Y : Ω' → ℚ)
    (t : ℝ) : cf (prodW2 p q) (fun w => X w.1 + Y w.2) t = cf p X t * cf q Y t := by
  simp only [cf, prodW2, Finset.sum_mul_sum, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  push_cast
  rw [mul_add, exp_add]; ring

/-- The rescaling rule `φ_{aX+b}(t) = e^{itb} φ_X(at)`. -/
theorem cf_affine {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (a b : ℚ) (t : ℝ) :
    cf p (fun ω => a * X ω + b) t = exp (I * t * b) * cf p X (a * t) := by
  simp only [cf, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  push_cast
  rw [show I * (t : ℂ) * (a * X ω + b) = I * t * b + I * (a * t) * X ω by ring, exp_add]; ring

/-- The characteristic function of a sum of `n` independent copies is the `n`-th power. -/
theorem cf_sum_coord {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (n : ℕ) (X : Ω → ℚ) (t : ℝ) :
    cf (prodW p n) (fun w => ∑ i, X (w i)) t = (cf p X t) ^ n := by
  simp only [cf, prodW, Finset.mul_sum, exp_sum, ← Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
  rw [← Finset.prod_const, Finset.prod_univ_sum]
  simp [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [← Finset.prod_mul_distrib]
  congr 1; funext i; rw [Finset.mul_sum]
  sorry

/-- `φ_X` is the sum over the values of `x`: `∑ₓ P(X = x) e^{itx}`. -/
theorem cf_eq_sum_pmf {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℚ) (t : ℝ) :
    cf p X t = ∑ x ∈ univ.image X, (pmf p X x : ℂ) * exp (I * t * x) := by
  simp only [cf, pmf, P]
  rw [← Finset.sum_fiberwise_of_maps_to (g := X) (t := univ.image X) (fun ω _ => Finset.mem_image_of_mem X
    (mem_univ ω))]
  refine Finset.sum_congr rfl fun x _ => ?_
  push_cast
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun ω hω => ?_
  rw [(Finset.mem_filter.1 hω).2]

/-- The uniqueness theorem: two finite measures on `ℝ` with the same characteristic function are equal. -/
theorem charFun_uniqueness (μ ν : MeasureTheory.Measure ℝ) [MeasureTheory.IsFiniteMeasure μ]
    [MeasureTheory.IsFiniteMeasure ν] (h : ∀ t, MeasureTheory.charFun μ t = MeasureTheory.charFun ν t) : μ = ν :=
  MeasureTheory.Measure.ext_of_charFun h

/-! ## exm-rv-cf-die -/

/-- The die: `φ_X(t) = (e^{it} + e^{2it} + ⋯ + e^{6it}) / 6`. -/
theorem die_cf (t : ℝ) : cf (unif (Fin 6)) (fun i => (face i : ℚ)) t = (∑ k : Fin 6, exp (I * t * (k.val + 1))) / 6 := by
  simp only [cf, unif, face, Finset.sum_div]
  refine Finset.sum_congr rfl fun k _ => ?_
  push_cast
  simp; ring

/-- `φ_X'(0) = (i/6)(1 + ⋯ + 6) = 7i/2 = i E X`. -/
theorem die_cf_deriv : deriv (cfC (unif (Fin 6)) (fun i => (face i : ℚ))) 0 = 7 * I / 2 := by
  have := cfC_iteratedDeriv_zero (unif (Fin 6)) (fun i => (face i : ℚ)) 1
  rw [iteratedDeriv_one] at this
  rw [this]
  simp only [pow_one]
  have h : E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 1) = 7 / 2 := by simpa using die_moments.1
  rw [h]; push_cast; ring

/-- Two independent dice: the uniform weights on pairs are the product weights, so `φ_S = φ_X²`. -/
theorem two_dice_cf (t : ℝ) : prodW2 (unif (Fin 6)) (unif (Fin 6)) = unif Dice2 ∧
    cf (unif Dice2) (fun w => (sum2 w : ℚ)) t = (cf (unif (Fin 6)) (fun i => (face i : ℚ)) t) ^ 2 := by
  have h : prodW2 (unif (Fin 6)) (unif (Fin 6)) = unif Dice2 := by
    funext w; simp [prodW2, unif]; norm_num
  refine ⟨h, ?_⟩
  rw [← h, sq, ← cf_add_prod]
  rfl

/-- Collecting the coefficient of `e^{ist}` in `φ_S` gives the triangular table: `φ_S(t) = ∑ₛ P(S = s) e^{ist}`. -/
theorem two_dice_cf_coefficients (t : ℝ) :
    cf (unif Dice2) (fun w => (sum2 w : ℚ)) t = ∑ s ∈ Finset.Icc (2 : ℕ) 12, (((6 - |(s : ℚ) - 7|) / 36 : ℚ) : ℂ) * exp (I * t * s) := by
  rw [cf_eq_sum_pmf]
  have himg : univ.image (fun w : Dice2 => (sum2 w : ℚ)) = (Finset.Icc (2 : ℕ) 12).image (fun s => (s : ℚ)) := by
    decide +kernel
  rw [himg, Finset.sum_image (fun a _ b _ h => by exact_mod_cast h)]
  refine Finset.sum_congr rfl fun s hs => ?_
  obtain ⟨h2, h12⟩ := Finset.mem_Icc.1 hs
  rw [sum_pmf s h2 h12]

/-! ## exm-rv-cf-continuous -/

/-- The uniform variable: `φ_U(t) = ∫_0^1 e^{itu} du = (e^{it} - 1) / (it)` for `t ≠ 0`. -/
theorem uniform_cf (t : ℝ) (ht : t ≠ 0) : (∫ u in (0 : ℝ)..1, exp (I * t * u)) = (exp (I * t) - 1) / (I * t) := by
  have hc : I * (t : ℂ) ≠ 0 := mul_ne_zero I_ne_zero (by exact_mod_cast ht)
  rw [integral_exp_mul_complex hc]
  simp

/-- The series `(e^{z} - 1) / z = ∑ₙ zⁿ / (n + 1)!`, whose coefficients `1/(n+1)!` times `n!` are the moments
`E Uⁿ = 1/(n+1)`. -/
theorem uniform_cf_series (z : ℂ) (hz : z ≠ 0) : (exp z - 1) / z = ∑' n : ℕ, z ^ n / (n + 1).factorial := by
  have hexp : exp z = ∑' n : ℕ, z ^ n / n.factorial := by
    rw [Complex.exp_eq_exp_ℂ, NormedSpace.exp_eq_tsum_div]
  have hs : Summable (fun n : ℕ => z ^ n / n.factorial) := by
    have := NormedSpace.expSeries_div_summable ℂ z
    simpa using this
  rw [hexp, tsum_eq_zero_add hs]
  simp only [pow_zero, Nat.factorial_zero, Nat.cast_one, div_one, add_sub_cancel_left]
  rw [div_eq_iff hz, ← tsum_mul_right]
  refine tsum_congr fun n => ?_
  rw [pow_succ]; ring

/-- The first two moments of `U` read off the series: `E U = 1!/2! = 1/2`, `E U² = 2!/3! = 1/3`. -/
theorem uniform_cf_moments : (1 : ℚ) / (1 + 1).factorial * Nat.factorial 1 = 1 / 2 ∧
    (1 : ℚ) / (2 + 1).factorial * Nat.factorial 2 = 1 / 3 := by
  norm_num [Nat.factorial]

/-- The exponential variable: `φ_T(t) = ∫_0^∞ e^{itx} λ e^{-λx} dx = λ / (λ - it)`. -/
theorem exponential_cf (lam : ℝ) (hlam : 0 < lam) (t : ℝ) :
    (∫ x in Set.Ioi (0 : ℝ), exp (I * t * x) * (lam * Real.exp (-lam * x))) = lam / (lam - I * t) := by
  have ha : (I * t - lam : ℂ).re < 0 := by simp [hlam]
  have h := integral_exp_mul_complex_Ioi ha 0
  have hcongr : ∀ x ∈ Set.Ioi (0 : ℝ), exp (I * t * x) * ((lam * Real.exp (-lam * x) : ℝ) : ℂ) =
      (lam : ℂ) * exp ((I * t - lam) * x) := by
    intro x _
    push_cast
    rw [Complex.ofReal_exp, show ((I * t - lam) * x : ℂ) = I * t * x + (-(lam : ℂ) * x) by ring, exp_add]
    push_cast; ring
  rw [MeasureTheory.setIntegral_congr_fun measurableSet_Ioi hcongr, MeasureTheory.integral_const_mul, h]
  have hne : (I * t - lam : ℂ) ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0; simp at this; linarith
  field_simp
  ring

/-- `λ / (λ - it) = ∑ₙ (it/λ)ⁿ` for `|t| < λ`, with coefficients `n!/λⁿ · (i)ⁿ/n!`: the moments `E Tⁿ = n!/λⁿ`. -/
theorem exponential_cf_series (lam : ℝ) (hlam : 0 < lam) (t : ℝ) (ht : |t| < lam) :
    (lam : ℂ) / (lam - I * t) = ∑' n : ℕ, (I * t / lam) ^ n := by
  have hnorm : ‖(I * t / lam : ℂ)‖ < 1 := by
    rw [norm_div, norm_mul, norm_I, one_mul, Complex.norm_real, Complex.norm_real, Real.norm_eq_abs,
      Real.norm_eq_abs, abs_of_pos hlam, div_lt_one hlam]
    exact ht
  rw [tsum_geometric_of_norm_lt_one hnorm]
  have hne : (lam : ℂ) ≠ 0 := by exact_mod_cast hlam.ne'
  have hne' : (lam : ℂ) - I * t ≠ 0 := fun h0 => by
    have := congrArg Complex.re h0; simp at this; linarith
  field_simp

/-! ## exm-rv-cf-gaussian -/

/-- The Gaussian `N(μ, σ²)` has `φ(t) = exp(iμt - σ²t²/2)` (mathlib's `charFun_gaussianReal`). -/
theorem gaussian_cf (μ : ℝ) (v : NNReal) (t : ℝ) :
    MeasureTheory.charFun (ProbabilityTheory.gaussianReal μ v) t = exp (I * μ * t - v * t ^ 2 / 2) :=
  ProbabilityTheory.charFun_gaussianReal t

/-- `e^{-σ²t²/2} = ∑ₙ (-1)ⁿ σ^{2n} t^{2n} / (2ⁿ n!)`. -/
theorem gaussian_cf_series (σ t : ℝ) :
    Real.exp (-(σ ^ 2 * t ^ 2) / 2) = ∑' n : ℕ, (-1) ^ n * σ ^ (2 * n) * t ^ (2 * n) / (2 ^ n * n.factorial) := by
  rw [Real.exp_eq_exp_ℝ, NormedSpace.exp_eq_tsum_div]
  refine tsum_congr fun n => ?_
  rw [div_pow, neg_pow, mul_pow, pow_mul, pow_mul, div_div]
  ring

/-- `(2n)! / (2ⁿ n!) = (2n - 1)!!`, so the coefficient comparison gives `E X^{2n} = (2n - 1)!! σ^{2n}`. -/
theorem double_factorial_ratio (n : ℕ) : ((2 * n).factorial : ℚ) / (2 ^ n * n.factorial) = (2 * n - 1).doubleFactorial := by
  cases n with
  | zero => simp
  | succ m =>
    have h := Nat.factorial_eq_mul_doubleFactorial (2 * (m + 1) - 1)
    have h2 : 2 * (m + 1) - 1 + 1 = 2 * (m + 1) := by omega
    rw [h2, Nat.doubleFactorial_two_mul] at h
    rw [h]; push_cast
    have : (2 : ℚ) ^ (m + 1) * (m + 1).factorial ≠ 0 := by positivity
    field_simp

/-! ## exr-rv-cf-moments-sage -/

/-- The derivatives of the die's characteristic function at `0` divided by `iⁿ` give the moments `7/2, 91/6,
147/2, 2275/6`. -/
theorem die_cf_moments :
    iteratedDeriv 1 (cfC (unif (Fin 6)) (fun i => (face i : ℚ))) 0 = I ^ 1 * (7 / 2) ∧
      iteratedDeriv 2 (cfC (unif (Fin 6)) (fun i => (face i : ℚ))) 0 = I ^ 2 * (91 / 6) ∧
      iteratedDeriv 3 (cfC (unif (Fin 6)) (fun i => (face i : ℚ))) 0 = I ^ 3 * (147 / 2) ∧
      iteratedDeriv 4 (cfC (unif (Fin 6)) (fun i => (face i : ℚ))) 0 = I ^ 4 * (2275 / 6) := by
  have h1 : E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 1) = 7 / 2 := by decide +kernel
  have h2 : E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 2) = 91 / 6 := by decide +kernel
  have h3 : E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 3) = 147 / 2 := by decide +kernel
  have h4 : E (unif (Fin 6)) (fun i => (face i : ℚ) ^ 4) = 2275 / 6 := by decide +kernel
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [cfC_iteratedDeriv_zero]
  · rw [h1]; push_cast; ring
  · rw [h2]; push_cast; ring
  · rw [h3]; push_cast; ring
  · rw [h4]; push_cast; ring

/-- The power sums `∑_{k=1}^n k³ = (n(n+1)/2)²` and `∑_{k=1}^n k⁴ = n(n+1)(2n+1)(3n² + 3n - 1)/30`. -/
theorem sum_Icc_cube (n : ℕ) : ∑ k ∈ Finset.Icc 1 n, (k : ℚ) ^ 3 = (n * (n + 1) / 2) ^ 2 := by
  VERIPHY[pr-power-sum]

/-- The fourth power sum. -/
theorem sum_Icc_fourth (n : ℕ) :
    ∑ k ∈ Finset.Icc 1 n, (k : ℚ) ^ 4 = n * (n + 1) * (2 * n + 1) * (3 * n ^ 2 + 3 * n - 1) / 30 := by
  VERIPHY[pr-power-sum]

/-- The die moments are the power sums divided by `6`: `441/6 = 147/2`, `2275/6`. -/
theorem die_moments_power_sums : (∑ k ∈ Finset.Icc 1 6, (k : ℚ) ^ 3) / 6 = 147 / 2 ∧
    (∑ k ∈ Finset.Icc 1 6, (k : ℚ) ^ 4) / 6 = 2275 / 6 ∧ (441 : ℚ) / 6 = 147 / 2 := by
  decide +kernel

/-- `E Uⁿ = ∫_0^1 uⁿ du = 1/(n+1)` for the uniform variable. -/
theorem uniform_moment (n : ℕ) : (∫ u in (0 : ℝ)..1, u ^ n) = 1 / (n + 1) := by
  rw [integral_pow]; simp

/-! ## exr-rv-cf-rules -/

/-- The fair coin contributes `ε = ±1` with probability `1/2` each: `φ_ε(s) = (e^{is} + e^{-is})/2 = cos s`. -/
theorem coin_sign_cf (s : ℝ) : cf coin (fun b => 2 * (b.val : ℚ) - 1) s = Real.cos s := by
  simp only [cf, coin, Fin.sum_univ_two]
  push_cast
  rw [Complex.ofReal_cos, Complex.cos]
  simp; ring_nf

/-- The standardised number of heads `(N - n/2)/√(n/4) = ∑ εⱼ/√n` has characteristic function `cos(t/√n)ⁿ`:
the `n` factors of the product rule, each evaluated at `t/√n` by the rescaling rule. -/
theorem coins_standardised_cf (n : ℕ) (t : ℝ) :
    cf (prodW coin n) (fun w => ∑ i, (2 * ((w i).val : ℚ) - 1)) (t / Real.sqrt n) = (Real.cos (t / Real.sqrt n)) ^ n := by
  rw [cf_sum_coord, coin_sign_cf]

/-- `(N - n/2) / √(n/4) = ∑ εⱼ / √n` with `εⱼ = 2 wⱼ - 1`. -/
theorem standardised_heads (n : ℕ) (hn : 0 < n) (w : Fin n → Bool) :
    ((heads w : ℝ) - n / 2) / Real.sqrt (n / 4) = (∑ i, (2 * ((w i).val : ℝ) - 1)) / Real.sqrt n := by
  have hs : Real.sqrt ((n : ℝ) / 4) = Real.sqrt n / 2 := by
    rw [Real.sqrt_div (Nat.cast_nonneg n), show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  rw [hs, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, ← Finset.mul_sum]
  have : (heads w : ℝ) = ∑ i, ((w i).val : ℝ) := by
    simp [heads, Finset.card_filter]
    rfl
  rw [this]
  have hn' : Real.sqrt (n : ℝ) ≠ 0 := (Real.sqrt_pos.2 (by exact_mod_cast hn)).ne'
  field_simp
  simp; ring

/-! ## exr-rv-cf-inversion -/

/-- Orthogonality of the exponentials: `∫_0^{2π} e^{i(j-k)t} dt = 2π δ_{jk}`. -/
theorem exp_orthogonality (j k : ℤ) :
    (∫ t in (0 : ℝ)..(2 * Real.pi), exp (I * t * (j - k))) = if j = k then 2 * Real.pi else 0 := by
  split_ifs with h
  · subst h; simp
  · have hc : I * ((j : ℂ) - k) ≠ 0 := mul_ne_zero I_ne_zero (sub_ne_zero.2 (by exact_mod_cast h))
    have : (fun t : ℝ => exp (I * t * (j - k))) = fun t : ℝ => exp (I * (j - k) * t) := by
      funext t; ring_nf
    rw [this, integral_exp_mul_complex hc]
    have h1 : exp (I * ((j : ℂ) - k) * (2 * Real.pi : ℝ)) = 1 := by
      have := Complex.exp_int_mul_two_pi_mul_I (j - k)
      rw [← this]; push_cast; ring_nf
    rw [h1]; simp

/-- Fourier inversion for an integer-valued variable: `P(X = k) = (1/2π) ∫_0^{2π} e^{-itk} φ_X(t) dt`. -/
theorem cf_inversion {Ω : Type} [Fintype Ω] (p : Ω → ℚ) (X : Ω → ℤ) (k : ℤ) :
    (1 / (2 * Real.pi) : ℂ) * ∫ t in (0 : ℝ)..(2 * Real.pi), exp (-(I * t * k)) * ∑ ω, (p ω : ℂ) * exp (I * t * X ω) =
      (P p (univ.filter (fun ω => X ω = k)) : ℚ) := by
  have hint : ∀ ω : Ω, IntervalIntegrable (fun t : ℝ => (p ω : ℂ) * exp (I * t * (X ω - k))) MeasureTheory.volume
      0 (2 * Real.pi) := fun ω => (Continuous.intervalIntegrable (by fun_prop) _ _)
  have hrw : (fun t : ℝ => exp (-(I * t * k)) * ∑ ω, (p ω : ℂ) * exp (I * t * X ω)) =
      fun t : ℝ => ∑ ω, (p ω : ℂ) * exp (I * t * (X ω - k)) := by
    funext t; rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [show (I * t * ((X ω : ℤ) - k) : ℂ) = -(I * t * k) + I * t * X ω by ring, exp_add]; ring
  rw [hrw, intervalIntegral.integral_finset_sum (fun ω _ => hint ω)]
  simp_rw [intervalIntegral.integral_const_mul, exp_orthogonality]
  simp only [P, Finset.sum_filter, Finset.mul_sum]
  push_cast
  refine Finset.sum_congr rfl fun ω _ => ?_
  split_ifs with h <;> simp [h]
  field_simp

end KaluzaKlein.Probability.CharacteristicFunctions

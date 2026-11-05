import Mathlib
import KaluzaKlein.PartI.RealAnalysis.Integral
import Veriphy.Kit.Levels

/-!
# Real analysis, section "Exercises"

The claims of the three closing exercises of `tutorial/part-i-03-real-analysis.qmd`: Stirling's formula from the
comparison of `log n!` with `∫ log x`, the exponential function built from its power series (Cauchy product), and
Wallis's product from the integrals of powers of the sine.
-/

namespace KaluzaKlein.RealAnalysis.Exercises

open Filter Topology Real Set MeasureTheory intervalIntegral
open scoped Nat

/-! ## exr-ra-stirling -/

/-- `log n! = ∑_{k ≤ n} log k`. -/
theorem log_factorial_sum (n : ℕ) : Real.log (n ! : ℝ) = ∑ i ∈ Finset.range n, Real.log ((i : ℝ) + 1) := by
  rw [Nat.factorial_eq_prod_range_add_one]
  push_cast
  rw [Real.log_prod]
  intro i _
  positivity

/-- `∫₁ⁿ log x dx ≤ log n! ≤ ∫₁^(n+1) log x dx`, with `∫ log x dx = x log x - x`. -/
theorem log_factorial_bounds (n : ℕ) (hn : 1 ≤ n) :
    (n : ℝ) * Real.log n - n + 1 ≤ Real.log (n ! : ℝ) ∧
      Real.log (n ! : ℝ) ≤ ((n : ℝ) + 1) * Real.log ((n : ℝ) + 1) - ((n : ℝ) + 1) + 1 := by
  have hmono : ∀ a b : ℕ, 1 ≤ a → MonotoneOn Real.log (Icc (a : ℝ) b) := fun a b ha =>
    Real.strictMonoOn_log.monotoneOn.mono (fun x hx => lt_of_lt_of_le (by exact_mod_cast ha) hx.1)
  have hsum : Real.log (n ! : ℝ) = ∑ i ∈ Finset.Ico 1 (n + 1), Real.log (i : ℝ) := by
    rw [log_factorial_sum, Finset.sum_Ico_eq_sum_range]
    simp only [add_tsub_cancel_right]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    push_cast; ring_nf
  constructor
  · have h := (hmono 1 n le_rfl).integral_le_sum_Ico hn
    rw [integral_log] at h
    simp only [Nat.cast_one, Real.log_one, mul_zero, sub_zero] at h
    rw [log_factorial_sum]
    have hsub : ∑ i ∈ Finset.Ico 1 n, Real.log (((i + 1 : ℕ)) : ℝ) ≤ ∑ i ∈ Finset.range n, Real.log ((i : ℝ) + 1) := by
      have e : ∀ i : ℕ, Real.log (((i + 1 : ℕ)) : ℝ) = Real.log ((i : ℝ) + 1) := fun i => by push_cast; ring_nf
      simp_rw [e]
      apply Finset.sum_le_sum_of_subset_of_nonneg
      · intro i hi
        simp only [Finset.mem_Ico, Finset.mem_range] at hi ⊢
        omega
      · intro i _ _
        exact Real.log_nonneg (by linarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)])
    linarith
  · have h := (hmono 1 (n + 1) le_rfl).sum_le_integral_Ico (by omega)
    rw [integral_log] at h
    push_cast at h
    simp only [Real.log_one, mul_zero, sub_zero] at h
    rw [hsum]
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Exercises.log_factorial_bounds
-- test: ra-log-factorial-bounds-test.sage

/-- Hence `log n! = n log n - n + O(log n)`: `|log n! - (n log n - n)| ≤ log(n + 1) + 1`. -/
theorem log_factorial_asymptotic (n : ℕ) (hn : 1 ≤ n) :
    |Real.log (n ! : ℝ) - ((n : ℝ) * Real.log n - n)| ≤ Real.log ((n : ℝ) + 1) + 1 := by
  obtain ⟨h1, h2⟩ := log_factorial_bounds n hn
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hlog : Real.log ((n : ℝ) + 1) - Real.log n ≤ 1 / n := by
    rw [← Real.log_div (by positivity) hn0.ne']
    have := Real.log_le_sub_one_of_pos (show 0 < ((n : ℝ) + 1) / n by positivity)
    rw [show ((n : ℝ) + 1) / n - 1 = 1 / n by rw [div_sub_one hn0.ne', add_sub_cancel_left]] at this
    exact this
  have hlog0 : 0 ≤ Real.log ((n : ℝ) + 1) := Real.log_nonneg (by linarith)
  rw [abs_le]
  constructor
  · linarith
  · have : (n : ℝ) * (Real.log ((n : ℝ) + 1) - Real.log n) ≤ 1 := by
      calc (n : ℝ) * (Real.log ((n : ℝ) + 1) - Real.log n) ≤ n * (1 / n) := by gcongr
        _ = 1 := by field_simp
    nlinarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Exercises.log_factorial_asymptotic
-- test: ra-log-factorial-asymptotic-test.sage

/-- Stirling's formula with its constant: `r_n = n!/(√(2πn)(n/e)ⁿ)` satisfies `1 ≤ r_n`, decreases, and tends to
`1`. -/
theorem stirling_ratio (n : ℕ) (hn : 1 ≤ n) :
    1 ≤ (n ! : ℝ) / (√(2 * π * n) * (n / Real.exp 1) ^ n) ∧
      (((n + 1)! : ℝ) / (√(2 * π * ((n : ℝ) + 1)) * (((n : ℝ) + 1) / Real.exp 1) ^ (n + 1))) ≤
        (n ! : ℝ) / (√(2 * π * n) * (n / Real.exp 1) ^ n) := by
  have hpi : 0 < √π := Real.sqrt_pos.2 Real.pi_pos
  have hr : ∀ m : ℕ, (m ! : ℝ) / (√(2 * π * m) * (m / Real.exp 1) ^ m) = Stirling.stirlingSeq m / √π := by
    intro m
    unfold Stirling.stirlingSeq
    rw [show 2 * π * (m : ℝ) = π * (2 * m) by ring, Real.sqrt_mul Real.pi_pos.le]
    field_simp
  have hn' : n ≠ 0 := by omega
  constructor
  · rw [hr, le_div_iff₀ hpi, one_mul]
    exact Stirling.sqrt_pi_le_stirlingSeq hn'
  · have h1 := hr (n + 1)
    push_cast at h1
    rw [h1, hr]
    apply div_le_div_of_nonneg_right _ hpi.le
    obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
    exact Stirling.stirlingSeq'_antitone (Nat.le_succ k)

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Exercises.stirling_ratio
-- test: ra-stirling-ratio-test.sage

/-! ## exr-ra-exp-functional -/

/-- The term `xⁱ/i! · yⁿ⁻ⁱ/(n-i)!` of the Cauchy product's inner sum. -/
noncomputable def cauchyTerm (x y : ℝ) (n i : ℕ) : ℝ := x ^ i / (i ! : ℝ) * (y ^ (n - i) / ((n - i)! : ℝ))

/-- The inner sum of the Cauchy product: `∑_{i+j=n} xⁱ/i! · yʲ/j! = (x + y)ⁿ/n!` (the binomial theorem). -/
theorem cauchy_product_term (x y : ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range (n + 1), cauchyTerm x y n i = (x + y) ^ n / (n ! : ℝ) := by
  simp only [cauchyTerm]
  rw [add_pow, Finset.sum_div]
  refine Finset.sum_congr rfl (fun i hi => ?_)
  have hi' : i ≤ n := Nat.lt_succ_iff.1 (Finset.mem_range.1 hi)
  rw [Nat.cast_choose ℝ hi']
  have h1 : (i ! : ℝ) ≠ 0 := by positivity
  have h2 : ((n - i)! : ℝ) ≠ 0 := by positivity
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Exercises.cauchy_product_term (1) (2) (0) := by
  simp only [cauchyTerm, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Exercises.cauchy_product_term (1) (2) (3) := by
  simp only [cauchyTerm, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Exercises.cauchy_product_term (1 / 2) (-3) (4) := by
  simp only [cauchyTerm, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Exercises.cauchy_product_term
-- test: ra-cauchy-product-term-test.sage

/-- `E(x) = ∑ xᵏ/k!` converges absolutely for every `x`, satisfies `E(x + y) = E(x) E(y)` and `E' = E`. -/
theorem exp_from_series (x y : ℝ) :
    Summable (fun k : ℕ => |x ^ k / (k ! : ℝ)|) ∧
      (∑' k : ℕ, (x + y) ^ k / (k ! : ℝ)) = (∑' k : ℕ, x ^ k / (k ! : ℝ)) * (∑' k : ℕ, y ^ k / (k ! : ℝ)) ∧
      HasDerivAt (fun z => ∑' k : ℕ, z ^ k / (k ! : ℝ)) (∑' k : ℕ, x ^ k / (k ! : ℝ)) x := by
  have hS : ∀ z : ℝ, HasSum (fun k : ℕ => z ^ k / (k ! : ℝ)) (Real.exp z) := fun z => by
    rw [Real.exp_eq_exp_ℝ]
    exact NormedSpace.expSeries_div_hasSum_exp z
  have hE : ∀ z : ℝ, (∑' k : ℕ, z ^ k / (k ! : ℝ)) = Real.exp z := fun z => (hS z).tsum_eq
  refine ⟨?_, ?_, ?_⟩
  · have := (hS |x|).summable
    refine this.congr (fun k => ?_)
    rw [abs_div, abs_pow, abs_of_pos (by positivity : (0 : ℝ) < k !)]
  · rw [hE, hE, hE, Real.exp_add]
  · simp_rw [hE]
    exact Real.hasDerivAt_exp x

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Exercises.exp_from_series
-- test: ra-exp-from-series-test.sage

/-! ## exr-ra-wallis -/

/-- `I_n = ∫₀^(π/2) sinⁿ x dx` decreases: `I_(2n+1) ≤ I_(2n) ≤ I_(2n-1)`. -/
theorem wallis_monotone (n : ℕ) (hn : 1 ≤ n) :
    ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n + 1) ≤ ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n) ∧
      ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n) ≤ ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n - 1) := by
  have hpi : (0 : ℝ) ≤ π / 2 := by positivity
  have hsin : ∀ x ∈ Icc (0 : ℝ) (π / 2), 0 ≤ Real.sin x ∧ Real.sin x ≤ 1 := fun x hx =>
    ⟨Real.sin_nonneg_of_nonneg_of_le_pi hx.1 (by linarith [hx.2, Real.pi_pos]), Real.sin_le_one x⟩
  have hc : ∀ m : ℕ, IntervalIntegrable (fun x => Real.sin x ^ m) volume 0 (π / 2) := fun m =>
    (by fun_prop : Continuous fun x => Real.sin x ^ m).intervalIntegrable _ _
  constructor
  · exact intervalIntegral.integral_mono_on hpi (hc _) (hc _) (fun x hx =>
      pow_le_pow_of_le_one (hsin x hx).1 (hsin x hx).2 (by omega))
  · exact intervalIntegral.integral_mono_on hpi (hc _) (hc _) (fun x hx =>
      pow_le_pow_of_le_one (hsin x hx).1 (hsin x hx).2 (by omega))

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Exercises.wallis_monotone
-- test: ra-wallis-monotone-test.sage

/-- `I_(2n) = ((2n-1)!!/(2n)!!) π/2` and `I_(2n+1) = (2n)!!/(2n+1)!!`. -/
theorem wallis_values (n : ℕ) :
    ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n) = ((2 * n - 1 : ℕ)‼ : ℝ) / ((2 * n : ℕ)‼ : ℝ) * (π / 2) ∧
      ∫ x in (0 : ℝ)..(π / 2), Real.sin x ^ (2 * n + 1) = ((2 * n : ℕ)‼ : ℝ) / ((2 * n + 1 : ℕ)‼ : ℝ) := by
  have hodd : ∀ m : ℕ, (2 * m + 1)‼ = (2 * m + 1) * (2 * m - 1)‼ := by
    intro m
    cases m with
    | zero => rfl
    | succ k =>
      rw [show 2 * (k + 1) + 1 = 2 * k + 1 + 2 by ring, Nat.doubleFactorial_add_two,
        show 2 * (k + 1) - 1 = 2 * k + 1 by omega]
  induction n with
  | zero =>
    constructor
    · simp
    · simp [integral_sin]
  | succ n ih =>
    obtain ⟨h0, h1⟩ := ih
    have r0 := Integral.sin_pow_reduction (2 * n)
    have r1 := Integral.sin_pow_reduction (2 * n + 1)
    rw [show 2 * (n + 1) = 2 * n + 2 by ring, show 2 * n + 2 + 1 = 2 * n + 1 + 2 by ring]
    rw [show 2 * n + 2 - 1 = 2 * n + 1 by omega]
    constructor
    · rw [r0, h0, Nat.doubleFactorial_add_two, hodd n]
      push_cast
      have : ((2 * n : ℕ)‼ : ℝ) ≠ 0 := by positivity
      field_simp <;> ring
    · rw [show 2 * n + 1 + 2 = 2 * n + 1 + 2 from rfl, r1, h1, Nat.doubleFactorial_add_two, Nat.doubleFactorial_add_two]
      push_cast
      have : ((2 * n + 1 : ℕ)‼ : ℝ) ≠ 0 := by positivity
      field_simp
      ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Exercises.wallis_values
-- test: ra-wallis-values-test.sage

/-- Wallis's product: the partial products `W_k = ∏_{i<k} (2i+2)²/((2i+1)(2i+3))` satisfy
`(2k+1)/(2k+2) · π/2 ≤ W_k ≤ π/2`, so `π/2 = ∏ 4k²/(4k² - 1)`. -/
theorem wallis_product (k : ℕ) :
    ((2 : ℝ) * k + 1) / (2 * k + 2) * (π / 2) ≤ Real.Wallis.W k ∧ Real.Wallis.W k ≤ π / 2 :=
  ⟨Real.Wallis.le_W k, Real.Wallis.W_le k⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Exercises.wallis_product
-- test: ra-wallis-product-test.sage

end KaluzaKlein.RealAnalysis.Exercises
